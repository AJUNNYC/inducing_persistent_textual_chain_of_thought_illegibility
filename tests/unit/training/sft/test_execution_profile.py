"""CPU execution audits; these fixtures do not certify GPU throughput."""
import copy
import random
from types import SimpleNamespace

import pytest

from training.sft.execution_profile import SFTExecutionProfile, resolve_execution_profile
from training.sft.individual_fit.sft_fit import (
    SFTInputError, SFTRuntimeBlocked, capture_rng, one_optimizer_update,
    restore_rng, train_optimizer_updates,
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
        'defer_finite_checks': False,
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
