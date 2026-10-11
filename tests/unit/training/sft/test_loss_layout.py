"""Actual CPU loss/gradient audits; CUDA candidate approval is separate."""
import copy
from types import SimpleNamespace

import pytest

from training.sft.execution_profile import SFTExecutionProfile
from training.sft.individual_fit.sft_fit import (
    SFTInputError, capture_rng, collate_completion, completion_loss,
    one_optimizer_update, profile_completion_loss, validation_loss, verify_mask_gradients,
)


@pytest.mark.parametrize('dtype_name', ['float32', 'bfloat16'])
@pytest.mark.parametrize('bound_counts', [False, True])
def test_flat_ce_keeps_causal_masks_full_fp32_loss_and_equal_example_weighting(dtype_name, bound_counts):
    import torch
    import torch.nn.functional as functional
    torch.manual_seed(713)
    examples = [
        {'input_ids': [1, 2, 3, 4, 5, 6, 7], 'labels': [1, -100, -100, 4, 5, 6, 7]},
        {'input_ids': [8, 9, 10, 11, 12], 'labels': [-100, -100, -100, -100, 12]},
        {'input_ids': [13, 14, 15, 16], 'labels': [-100, -100, 15, 16]},
    ]
    batch = collate_completion(examples, 0)
    labels = batch['labels']
    # Unequal target counts/padding and a non-contiguous logits tensor.
    values = torch.randn(3, 19, 7, dtype=getattr(torch, dtype_name)).transpose(1, 2)
    flat_logits, oracle_logits, spatial_logits = [values.detach().requires_grad_() for _ in range(3)]
    state_before = torch.get_rng_state().clone()
    flat = profile_completion_loss(flat_logits, labels,
        execution_profile=SFTExecutionProfile(loss_implementation='flat_ce'),
        validated_counts=(batch['validated_completion_counts'] if bound_counts else None))
    per_example = []
    for row in range(3):
        targets = labels[row, 1:]
        eligible = targets != -100
        per_example.append(functional.cross_entropy(
            oracle_logits[row, :-1].float()[eligible], targets[eligible]))
    oracle = torch.stack(per_example).mean()
    spatial = completion_loss(spatial_logits, labels)
    assert flat.dtype == torch.float32
    torch.testing.assert_close(flat, oracle, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(flat, spatial, atol=1e-6, rtol=1e-6)
    flat.backward(); oracle.backward(); spatial.backward()
    torch.testing.assert_close(flat_logits.grad, oracle_logits.grad, atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(flat_logits.grad, spatial_logits.grad, atol=1e-7, rtol=1e-5)
    assert not torch.count_nonzero(flat_logits.grad[:, -1])
    for row in range(3):
        ignored_positions = labels[row, 1:] == -100
        assert not torch.count_nonzero(flat_logits.grad[row, :-1][ignored_positions])
    assert torch.equal(state_before, torch.get_rng_state())


def test_spatial_profile_wrapper_is_bitwise_identical_and_flat_rejects_invalid_counts():
    import torch
    torch.manual_seed(811)
    values = torch.randn(1, 4, 7)
    direct_logits, wrapped_logits = [values.detach().requires_grad_() for _ in range(2)]
    labels = torch.tensor([[-100, -100, 2, 3]])
    direct = completion_loss(direct_logits, labels)
    wrapped = profile_completion_loss(wrapped_logits, labels)
    assert torch.equal(direct, wrapped)
    direct.backward(); wrapped.backward()
    assert torch.equal(direct_logits.grad, wrapped_logits.grad)
    flat_profile = SFTExecutionProfile(loss_implementation='flat_ce')
    with pytest.raises(SFTInputError, match='no supervised'):
        profile_completion_loss(values, torch.tensor([[1, -100, -100, -100]]),
                                execution_profile=flat_profile)
    batch = collate_completion([{'input_ids': [1, 2, 3, 4], 'labels': [-100, -100, 2, 3]}], 0)
    bound = batch['validated_completion_counts']
    batch['labels'].view(-1)[-1] = -100
    with pytest.raises(SFTInputError, match='unchanged collated labels'):
        profile_completion_loss(values, batch['labels'], execution_profile=flat_profile,
                                validated_counts=bound)


def fixture_model():
    import torch
    torch.manual_seed(42)
    class DropoutModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(19, 7)
            self.dropout = torch.nn.Dropout(.05)
            self.output = torch.nn.Linear(7, 19)
        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(logits=self.output(self.dropout(self.embedding(input_ids))))
    model = DropoutModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 - step / 10)
    examples = []
    for index in range(16):
        ids = [(index + offset) % 19 for offset in range(4 + index % 3)]
        examples.append({'input_ids': ids, 'labels': [-100, -100] + ids[2:],
                        'prompt_token_count': 2, 'supervised_token_count': len(ids) - 2})
    return model, optimizer, scheduler, examples


def assert_states_close(left, right):
    import torch
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=1e-7, rtol=1e-5)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_states_close(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            assert_states_close(first, second)
    else:
        assert left == right


def test_flat_ce_preserves_real_dropout_optimizer_update_and_rng_within_cpu_tolerance():
    outputs = []
    rngs = []
    for implementation in ('spatial_ce', 'flat_ce'):
        model, optimizer, scheduler, examples = fixture_model()
        profile = SFTExecutionProfile(gradient_checkpointing=False, loss_implementation=implementation)
        losses = one_optimizer_update(model, optimizer, scheduler, examples, 0, execution_profile=profile)
        outputs.append(copy.deepcopy({'losses': losses, 'weights': model.state_dict(),
            'gradients': {name: value.grad for name, value in model.named_parameters()},
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}))
        rngs.append(capture_rng())
    assert_states_close(*outputs)
    import torch
    import numpy as np
    assert rngs[0]['python'] == rngs[1]['python']
    assert rngs[0]['numpy'][0] == rngs[1]['numpy'][0]
    assert np.array_equal(rngs[0]['numpy'][1], rngs[1]['numpy'][1])
    assert rngs[0]['numpy'][2:] == rngs[1]['numpy'][2:]
    assert torch.equal(rngs[0]['torch_cpu'], rngs[1]['torch_cpu'])


def test_gradient_probe_and_validation_use_the_selected_loss_without_updating_model(monkeypatch):
    import torch
    from training.sft.individual_fit import sft_fit
    original = sft_fit.profile_completion_loss
    calls = []
    def observed(logits, labels, **kwargs):
        calls.append(kwargs['execution_profile'].loss_implementation)
        return original(logits, labels, **kwargs)
    monkeypatch.setattr(sft_fit, 'profile_completion_loss', observed)
    model, _, _, examples = fixture_model()
    before = copy.deepcopy(model.state_dict())
    rng_before = torch.get_rng_state().clone()
    profile = SFTExecutionProfile(gradient_checkpointing=False, loss_implementation='flat_ce')
    evidence = verify_mask_gradients(model, examples[0], 0, require_lora=False, execution_profile=profile)
    assert evidence['status'] == 'passed'
    value = validation_loss(model, examples[:3], 0, execution_profile=profile)
    assert value > 0 and calls == ['flat_ce'] * 4
    assert torch.equal(rng_before, torch.get_rng_state())
    assert all(torch.equal(before[name], values) for name, values in model.state_dict().items())


def test_loss_profile_defaults_normalize_prior_checkpoint_metadata_and_reject_unknown_backend():
    older = {'attention_implementation': 'eager', 'validated_loss_counts': False,
             'defer_finite_checks': False, 'cache_collated_inputs': False,
             'gradient_checkpointing': False}
    assert SFTExecutionProfile.from_dict(older).loss_implementation == 'spatial_ce'
    with pytest.raises(ValueError, match='completion-loss candidates'):
        SFTExecutionProfile(loss_implementation='unregistered_loss')


def test_fused_dispatch_bridge_preserves_causal_rows_and_example_weighting(monkeypatch):
    """Declared Torch stub checks routing; this does not exercise Triton math."""
    import torch
    import torch.nn.functional as functional
    from training.sft import fused_loss
    calls = []
    def controlled_kernel(logits, targets, *, ignore_index=-100):
        assert logits.dtype == torch.float32 and logits.ndim == 2
        assert targets.dtype == torch.long and targets.ndim == 1
        assert logits.shape[0] == targets.numel() and ignore_index == -100
        calls.append((logits.shape, targets.tolist()))
        return functional.cross_entropy(logits, targets, ignore_index=ignore_index, reduction='none')
    monkeypatch.setattr(fused_loss, 'fused_token_cross_entropy', controlled_kernel)
    torch.manual_seed(329)
    values = torch.randn(2, 5, 11, dtype=torch.bfloat16)
    labels = torch.tensor([[-100, -100, 2, 3, 4], [-100, -100, -100, -100, 5]])
    fused_logits, direct_logits = [values.detach().requires_grad_() for _ in range(2)]
    fused = profile_completion_loss(fused_logits, labels,
        execution_profile=SFTExecutionProfile(loss_implementation='fused_ce'))
    direct = completion_loss(direct_logits, labels)
    torch.testing.assert_close(fused, direct, atol=1e-6, rtol=1e-6)
    fused.backward(); direct.backward()
    torch.testing.assert_close(fused_logits.grad, direct_logits.grad, atol=1e-7, rtol=1e-5)
    assert calls == [(torch.Size([8, 11]), [-100, 2, 3, 4, -100, -100, -100, 5])]


def test_fused_execution_failure_propagates_without_spatial_or_flat_fallback(monkeypatch):
    import torch
    from training.sft import fused_loss
    def unavailable(*args, **kwargs):
        raise RuntimeError('Declared missing CUDA kernel fixture')
    monkeypatch.setattr(fused_loss, 'fused_token_cross_entropy', unavailable)
    with pytest.raises(RuntimeError, match='missing CUDA kernel'):
        profile_completion_loss(torch.randn(1, 3, 7), torch.tensor([[-100, 2, 3]]),
            execution_profile=SFTExecutionProfile(loss_implementation='fused_ce'))


def install_native_spatial_forward_stub(monkeypatch):
    """CPU routing fixture; native Triton backward approval remains separate."""
    import torch
    import torch.nn.functional as functional
    from training.sft import spatial_loss
    calls = []
    def native_forward(logits, targets, *, ignore_index=-100):
        assert logits.dtype == torch.float32 and logits.ndim == 3
        assert targets.dtype == torch.long and targets.ndim == 2
        assert targets.shape == (logits.shape[0], logits.shape[2])
        assert ignore_index == -100
        calls.append((logits.detach().clone(), targets.detach().clone()))
        return functional.nll_loss(torch.log_softmax(logits, dim=1, dtype=logits.dtype),
                                   targets, ignore_index=ignore_index, reduction='none')
    monkeypatch.setattr(spatial_loss, 'spatial_token_cross_entropy', native_forward)
    return calls


@pytest.mark.parametrize('dtype_name', ['float32', 'bfloat16'])
@pytest.mark.parametrize('bound_counts', [False, True])
def test_spatial_backward_dispatch_retains_original_forward_layout_masks_and_weighting(monkeypatch, dtype_name, bound_counts):
    import torch
    calls = install_native_spatial_forward_stub(monkeypatch)
    torch.manual_seed(782)
    examples = [
        {'input_ids': [1, 2, 3, 4, 5, 6], 'labels': [1, -100, 3, 4, 5, 6]},
        {'input_ids': [7, 8, 9, 10], 'labels': [-100, -100, -100, 10]},
        {'input_ids': [2, 3, 4, 5, 6], 'labels': [-100, -100, 4, 5, 6]},
    ]
    batch = collate_completion(examples, 0)
    values = torch.randn(3, 13, 6, dtype=getattr(torch, dtype_name)).transpose(1, 2)
    reference_logits, candidate_logits = [values.detach().requires_grad_() for _ in range(2)]
    before = torch.get_rng_state().clone()
    reference = completion_loss(reference_logits, batch['labels'])
    candidate = profile_completion_loss(candidate_logits, batch['labels'],
        execution_profile=SFTExecutionProfile(loss_implementation='spatial_fused_backward'),
        validated_counts=(batch['validated_completion_counts'] if bound_counts else None))
    assert torch.equal(candidate, reference)
    candidate.backward(); reference.backward()
    assert torch.equal(candidate_logits.grad, reference_logits.grad)
    assert not torch.count_nonzero(candidate_logits.grad[:, -1])
    for row in range(3):
        assert not torch.count_nonzero(candidate_logits.grad[row, :-1][batch['labels'][row, 1:] == -100])
    assert len(calls) == 1 and calls[0][0].shape == (3, 13, 5)
    assert torch.equal(calls[0][0], values[:, :-1].float().transpose(1, 2))
    assert torch.equal(calls[0][1], batch['labels'][:, 1:])
    assert torch.equal(before, torch.get_rng_state())


def test_spatial_backward_update_probe_and_validation_keep_native_forward_rng_and_receipt(monkeypatch, tmp_path):
    import hashlib
    import torch
    from training.sft import spatial_loss
    from training.sft.individual_fit import sft_fit
    calls = install_native_spatial_forward_stub(monkeypatch)
    helper_file = tmp_path / 'actually_imported_spatial_loss.py'
    helper_file.write_bytes(b'declared loaded helper fixture\n')
    monkeypatch.setattr(spatial_loss, '__file__', str(helper_file))
    expected_hash = hashlib.sha256(helper_file.read_bytes()).hexdigest()
    candidate_profile = SFTExecutionProfile(gradient_checkpointing=False,
                                            loss_implementation='spatial_fused_backward')
    assert sft_fit._loaded_loss_helper_hash(SFTExecutionProfile()) == {}
    assert sft_fit._loaded_loss_helper_hash(candidate_profile) == {'loss_helper_sha256': expected_hash}
    outputs, rngs = [], []
    for profile in (SFTExecutionProfile(gradient_checkpointing=False), candidate_profile):
        model, optimizer, scheduler, examples = fixture_model()
        losses = one_optimizer_update(model, optimizer, scheduler, examples, 0, execution_profile=profile)
        outputs.append(copy.deepcopy({'losses': losses, 'weights': model.state_dict(),
            'gradients': {name: value.grad for name, value in model.named_parameters()},
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}))
        rngs.append(capture_rng())
    def equal_states(left, right):
        if isinstance(left, torch.Tensor):
            assert torch.equal(left, right)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                equal_states(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            assert len(left) == len(right)
            for first, second in zip(left, right, strict=True):
                equal_states(first, second)
        else:
            assert left == right
    equal_states(*outputs)
    assert torch.equal(rngs[0]['torch_cpu'], rngs[1]['torch_cpu'])
    model, _, _, examples = fixture_model()
    before, rng_before = copy.deepcopy(model.state_dict()), torch.get_rng_state().clone()
    probe = verify_mask_gradients(model, examples[0], 0, require_lora=False,
                                  execution_profile=candidate_profile)
    assert probe['status'] == 'passed' and probe['loss_helper_sha256'] == expected_hash
    candidate_validation = validation_loss(model, examples[:3], 0, execution_profile=candidate_profile)
    baseline_validation = validation_loss(model, examples[:3], 0)
    assert candidate_validation == baseline_validation
    assert torch.equal(rng_before, torch.get_rng_state())
    equal_states(before, model.state_dict())
    assert len(calls) == 20  # 16 training microbatches, one probe and three validation examples.


def test_spatial_backward_failure_propagates_and_count_guards_precede_kernel(monkeypatch):
    import torch
    from training.sft import spatial_loss
    profile = SFTExecutionProfile(loss_implementation='spatial_fused_backward')
    def unavailable(*args, **kwargs):
        raise RuntimeError('Declared missing spatial CUDA kernel fixture')
    monkeypatch.setattr(spatial_loss, 'spatial_token_cross_entropy', unavailable)
    values = torch.randn(1, 3, 7)
    with pytest.raises(RuntimeError, match='missing spatial CUDA kernel'):
        profile_completion_loss(values, torch.tensor([[-100, 2, 3]]), execution_profile=profile)
    with pytest.raises(SFTInputError, match='no supervised'):
        profile_completion_loss(values, torch.tensor([[-100, -100, -100]]), execution_profile=profile)
    batch = collate_completion([{'input_ids': [1, 2, 3], 'labels': [-100, 2, 3]}], 0)
    batch['labels'].view(-1)[-1] = -100
    with pytest.raises(SFTInputError, match='unchanged collated labels'):
        profile_completion_loss(values, batch['labels'], execution_profile=profile,
                                validated_counts=batch['validated_completion_counts'])
