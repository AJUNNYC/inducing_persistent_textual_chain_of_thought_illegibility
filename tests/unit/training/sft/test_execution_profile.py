"""CPU execution audits; these fixtures do not certify GPU throughput."""
import copy
import random
from types import SimpleNamespace

import pytest

from training.sft.execution_profile import SFTExecutionProfile, resolve_execution_profile
from training.sft.individual_fit.sft_fit import (
    SFTInputError, SFTRuntimeBlocked, capture_rng, one_optimizer_update,
    restore_rng, train_optimizer_updates, _prepare_completion_input_cache,
    prepare_cuda_input_cache,
)


BASELINE = SFTExecutionProfile()
EXACT_CANDIDATE = SFTExecutionProfile('eager', True, True)


def fixture_runtime(count=17, invalid_at_call=None):
    import numpy as np
    import torch
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    class DropoutModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(16, 7)
            self.dropout = torch.nn.Dropout(.05)
            self.output = torch.nn.Linear(7, 16)
            self.calls = 0
        def forward(self, input_ids, attention_mask):
            self.calls += 1
            logits = self.output(self.dropout(self.embedding(input_ids)))
            if self.calls == invalid_at_call:
                logits = logits * float('nan')
            return SimpleNamespace(logits=logits)
    model = DropoutModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 - step / 10)
    examples = []
    for index in range(count):
        ids = [(index + offset) % 16 for offset in range(4 + index % 3)]
        examples.append({'input_ids': ids, 'labels': [-100, -100] + ids[2:]})
    return model, optimizer, scheduler, examples


def snapshot(model, optimizer, scheduler):
    return copy.deepcopy({
        'weights': model.state_dict(),
        'gradients': {name: parameter.grad for name, parameter in model.named_parameters()},
        'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
        'rng': capture_rng(),
    })


def assert_bitwise_equal(left, right):
    import numpy as np
    import torch
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_bitwise_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for item_left, item_right in zip(left, right, strict=True):
            assert_bitwise_equal(item_left, item_right)
    else:
        assert left == right


def test_execution_profile_is_explicit_and_baseline_default_is_unchanged():
    assert resolve_execution_profile(None) == BASELINE
    assert BASELINE.to_dict() == {
        'attention_implementation': 'eager', 'validated_loss_counts': False,
        'defer_finite_checks': False, 'cache_collated_inputs': False,
        'gradient_checkpointing': True, 'loss_implementation': 'spatial_ce',
    }
    for values in [('flash_attention_2', False, False), ('eager', 1, False), ('eager', False, 'true')]:
        with pytest.raises(ValueError):
            SFTExecutionProfile(*values)
    with pytest.raises(ValueError, match='explicit validated'):
        resolve_execution_profile(EXACT_CANDIDATE.to_dict())


def test_exact_candidate_one_update_preserves_losses_gradients_states_and_rng_bitwise():
    outputs = []
    for profile in (BASELINE, EXACT_CANDIDATE):
        model, optimizer, scheduler, examples = fixture_runtime(16)
        losses = one_optimizer_update(model, optimizer, scheduler, examples, 0,
                                      execution_profile=profile)
        assert len(losses) == 16 and all(not loss.requires_grad for loss in losses)
        outputs.append({'losses': losses, 'state': snapshot(model, optimizer, scheduler)})
    assert_bitwise_equal(*outputs)


def test_exact_candidate_full_exposure_and_partial_final_update_are_bitwise_equal():
    outputs = []
    for profile in (BASELINE, EXACT_CANDIDATE):
        model, optimizer, scheduler, examples = fixture_runtime(17)
        saved = []
        progress = train_optimizer_updates(model, optimizer, scheduler, examples, 0,
                                           execution_profile=profile, save=saved.append)
        assert progress['completed_steps'] == 3 and progress['example_presentations'] == 34
        assert len(saved) == 1 and saved[0]['completed_steps'] == 3
        assert progress.pop('execution_profile') == profile.to_dict()
        assert saved[0].pop('execution_profile') == profile.to_dict()
        outputs.append({'state': snapshot(model, optimizer, scheduler),
                        'progress': progress, 'saved': saved})
    assert_bitwise_equal(*outputs)


def test_exact_candidate_profiler_eliminates_31_scalar_extractions_per_full_update():
    import torch
    counts = []
    for profile in (BASELINE, EXACT_CANDIDATE):
        model, optimizer, scheduler, examples = fixture_runtime(16)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as audit:
            one_optimizer_update(model, optimizer, scheduler, examples, 0,
                                 execution_profile=profile)
        events = {event.key: event.count for event in audit.key_averages()}
        assert events['aten::cross_entropy_loss'] == 16
        counts.append(events.get('aten::_local_scalar_dense', 0))
    # Sixteen count checks plus fifteen of the sixteen finite-loss checks.
    assert counts[0] - counts[1] == 31


@pytest.mark.parametrize('profile', [BASELINE, EXACT_CANDIDATE])
def test_nonfinite_loss_blocks_before_clipping_optimizer_or_scheduler_step(monkeypatch, profile):
    import torch
    model, optimizer, scheduler, examples = fixture_runtime(16, invalid_at_call=3)
    initial = snapshot(model, optimizer, scheduler)
    calls = []
    monkeypatch.setattr(optimizer, 'step', lambda: calls.append('optimizer'))
    # Patch the class: LambdaLR.state_dict includes extra instance attributes.
    monkeypatch.setattr(type(scheduler), 'step', lambda self: calls.append('scheduler'))
    monkeypatch.setattr(torch.nn.utils, 'clip_grad_norm_', lambda *args, **kwargs: calls.append('clip'))
    with pytest.raises(SFTRuntimeBlocked, match='partial update requires rollback'):
        one_optimizer_update(model, optimizer, scheduler, examples, 0, execution_profile=profile)
    assert calls == []
    assert_bitwise_equal(initial['weights'], model.state_dict())
    assert_bitwise_equal(initial['optimizer'], optimizer.state_dict())
    assert_bitwise_equal(initial['scheduler'], scheduler.state_dict())
    assert model.calls == (16 if profile.defer_finite_checks else 3)


def test_resume_rejects_profile_change_and_legacy_baseline_progress_resumes_exactly():
    model, optimizer, scheduler, examples = fixture_runtime(17)
    train_optimizer_updates(model, optimizer, scheduler, examples, 0)
    uninterrupted = snapshot(model, optimizer, scheduler)

    model, optimizer, scheduler, examples = fixture_runtime(17)
    paused = train_optimizer_updates(model, optimizer, scheduler, examples, 0,
                                     should_pause=lambda: True)
    recovered = snapshot(model, optimizer, scheduler)
    assert paused['completed_steps'] == 1
    with pytest.raises(SFTInputError, match='execution profile differs'):
        train_optimizer_updates(model, optimizer, scheduler, examples, 0, progress=paused,
                                execution_profile=EXACT_CANDIDATE)

    # Older checkpoints did not have the optional execution-profile field.
    legacy = {key: value for key, value in paused.items() if key != 'execution_profile'}
    with pytest.raises(SFTInputError, match='execution profile differs'):
        train_optimizer_updates(model, optimizer, scheduler, examples, 0, progress=legacy,
                                execution_profile=EXACT_CANDIDATE)
    model, optimizer, scheduler, examples = fixture_runtime(17)
    model.load_state_dict(recovered['weights'])
    optimizer.load_state_dict(recovered['optimizer'])
    scheduler.load_state_dict(recovered['scheduler'])
    restore_rng(recovered['rng'])
    final = train_optimizer_updates(model, optimizer, scheduler, examples, 0, progress=legacy)
    assert final['status'] == 'complete' and final['completed_steps'] == 3
    assert_bitwise_equal(uninterrupted, snapshot(model, optimizer, scheduler))


@pytest.mark.parametrize('loss_checks', [False, True])
def test_cached_inputs_preserve_dropout_losses_gradients_optimizer_and_rng_bitwise(loss_checks):
    outputs = []
    for cached in (False, True):
        model, optimizer, scheduler, examples = fixture_runtime(16)
        profile = SFTExecutionProfile(validated_loss_counts=loss_checks,
            defer_finite_checks=loss_checks, cache_collated_inputs=cached)
        prepared = None
        if cached:
            before = capture_rng()
            prepared = _prepare_completion_input_cache(examples, 0, 'cpu', validate_counts=loss_checks)
            assert_bitwise_equal(before, capture_rng())
            for example, entry in zip(examples, prepared, strict=True):
                tensors = entry.batch_for(example, 0, 'cpu')
                assert tensors['input_ids'].tolist() == [example['input_ids']]
                assert tensors['labels'].tolist() == [example['labels']]
                expected = 3 * 8 * len(example['input_ids']) + (8 if loss_checks else 0)
                assert entry.resident_bytes == expected
        losses = one_optimizer_update(model, optimizer, scheduler, examples, 0,
                                      execution_profile=profile, prepared_batches=prepared)
        outputs.append({'losses': losses, 'state': snapshot(model, optimizer, scheduler)})
    assert_bitwise_equal(*outputs)


def test_input_cache_rejects_changed_source_tensor_aliases_counts_and_mapping_writes():
    import torch
    example = {'example_id': 'immutable', 'input_ids': [1, 2, 3], 'labels': [-100, 2, 3]}
    cache = _prepare_completion_input_cache([example], 0, 'cpu', validate_counts=True)[0]
    with pytest.raises(TypeError):
        cache.tensors['labels'] = torch.tensor([[0, 0, 0]])
    for modified in ({**example, 'input_ids': [1, 2, 4]},
                     {**example, 'labels': [-100, 2, 4]},
                     {**example, 'example_id': 'another'}):
        with pytest.raises(SFTInputError, match='frozen input'):
            cache.batch_for(modified, 0, 'cpu')
    with pytest.raises(SFTInputError, match='frozen input'):
        cache.batch_for(example, 1, 'cpu')
    cache.tensors['attention_mask'].view(-1)[-1] = 0
    with pytest.raises(SFTInputError, match='mutated'):
        cache.batch_for(example, 0, 'cpu')
    cache = _prepare_completion_input_cache([example], 0, 'cpu', validate_counts=True)[0]
    cache.tensors['validated_completion_counts'].counts.fill_(0)
    with pytest.raises(SFTInputError, match='counts were mutated'):
        cache.batch_for(example, 0, 'cpu')
    with pytest.raises(SFTRuntimeBlocked, match='requires a CUDA'):
        prepare_cuda_input_cache([example], 0, 'cpu')


def test_cache_requires_explicit_profile_matching_rows_and_count_policy_before_backward():
    model, optimizer, scheduler, examples = fixture_runtime(16)
    prepared = _prepare_completion_input_cache(examples, 0, 'cpu')
    profile = SFTExecutionProfile(cache_collated_inputs=True)
    initial = snapshot(model, optimizer, scheduler)
    for batches in (None, prepared[:-1], tuple(reversed(prepared))):
        with pytest.raises(SFTInputError):
            one_optimizer_update(model, optimizer, scheduler, examples, 0,
                                 execution_profile=profile, prepared_batches=batches)
    with pytest.raises(SFTInputError, match='explicit cache execution profile'):
        one_optimizer_update(model, optimizer, scheduler, examples, 0,
                             prepared_batches=prepared)
    with pytest.raises(SFTInputError, match='counts differ'):
        one_optimizer_update(model, optimizer, scheduler, examples, 0,
            execution_profile=SFTExecutionProfile(cache_collated_inputs=True, validated_loss_counts=True),
            prepared_batches=prepared)
    assert model.calls == 0
    assert_bitwise_equal(initial, snapshot(model, optimizer, scheduler))


def test_cached_full_queue_and_partial_update_match_baseline_without_recollation(monkeypatch):
    from training.sft.individual_fit import sft_fit
    model, optimizer, scheduler, examples = fixture_runtime(17)
    baseline_progress = train_optimizer_updates(model, optimizer, scheduler, examples, 0)
    baseline = snapshot(model, optimizer, scheduler)
    model, optimizer, scheduler, examples = fixture_runtime(17)
    prepared = _prepare_completion_input_cache(examples, 0, 'cpu')
    def unexpected_collation(*args, **kwargs):
        raise AssertionError('Cached training unexpectedly rebuilt collated inputs')
    monkeypatch.setattr(sft_fit, 'collate_completion', unexpected_collation)
    cached_progress = train_optimizer_updates(model, optimizer, scheduler, examples, 0,
        execution_profile=SFTExecutionProfile(cache_collated_inputs=True), prepared_batches=prepared)
    baseline_progress.pop('execution_profile')
    cached_progress.pop('execution_profile')
    assert baseline_progress == cached_progress
    assert_bitwise_equal(baseline, snapshot(model, optimizer, scheduler))


def test_earlier_three_field_profile_recovers_baseline_defaults():
    assert SFTExecutionProfile.from_dict({'attention_implementation': 'eager',
        'validated_loss_counts': False, 'defer_finite_checks': False}) == BASELINE
    with pytest.raises(ValueError, match='Unknown'):
        SFTExecutionProfile.from_dict({'unrecorded_execution_policy': True})


def test_disabling_nonreentrant_checkpointing_after_identical_peft_setup_preserves_cpu_update():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from training.sft.individual_fit.sft_fit import TARGET_MODULES
    outputs = []
    for enabled in (True, False):
        torch.manual_seed(42)
        base = LlamaForCausalLM(LlamaConfig(vocab_size=16, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=32, attn_implementation='eager'))
        base.config.use_cache = False
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant': False})
        model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=.05,
            target_modules=list(TARGET_MODULES), bias='none', task_type='CAUSAL_LM'))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        if not enabled:
            model.gradient_checkpointing_disable()
        assert model.is_gradient_checkpointing is enabled
        model.train()
        optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad],
                                     lr=2e-4, weight_decay=0.)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 - step / 10)
        examples = [{'input_ids': [1, 2, 3, 4, 5], 'labels': [-100, -100, 3, 4, 5]}] * 4
        losses = one_optimizer_update(model, optimizer, scheduler, examples, 0,
            execution_profile=SFTExecutionProfile(gradient_checkpointing=enabled))
        outputs.append({'losses': losses, 'state': snapshot(model, optimizer, scheduler)})
    assert_bitwise_equal(*outputs)
