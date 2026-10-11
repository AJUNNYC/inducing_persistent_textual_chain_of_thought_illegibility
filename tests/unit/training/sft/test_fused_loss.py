"""CPU contract checks; only the separate native screen certifies Triton math."""

from contextlib import nullcontext

import pytest
import torch
import torch.nn.functional as functional

from training.sft import fused_loss


@pytest.mark.parametrize("shape,target_shape", [((3, 5, 1), (3,)), ((3, 5), (2,)), ((3, 5), (3, 1)), ((0, 5), (0,)), ((3, 0), (3,))])
def test_shape_contract_rejects_before_optional_runtime_import(shape, target_shape):
    with pytest.raises(ValueError):
        fused_loss.fused_token_cross_entropy(torch.empty(shape), torch.zeros(target_shape, dtype=torch.long))


@pytest.mark.parametrize("logit_dtype,target_dtype", [(torch.float16, torch.long), (torch.bfloat16, torch.long), (torch.float64, torch.long), (torch.float32, torch.int32), (torch.float32, torch.float32)])
def test_precision_contract_cannot_silently_cast(logit_dtype, target_dtype):
    with pytest.raises(ValueError, match="FP32 logits and int64"):
        fused_loss.fused_token_cross_entropy(torch.empty(3, 5, dtype=logit_dtype), torch.zeros(3, dtype=target_dtype))


def test_cpu_fallback_is_forbidden():
    with pytest.raises(fused_loss.FusedLossUnsupported, match="CPU fallback is forbidden"):
        fused_loss.fused_token_cross_entropy(torch.zeros(3, 5), torch.zeros(3, dtype=torch.long))


@pytest.mark.parametrize("ignore_index", [0, -1, -100.0, False])
def test_ignore_index_is_fixed(ignore_index):
    with pytest.raises(ValueError, match="ignore_index=-100"):
        fused_loss.fused_token_cross_entropy(torch.zeros(3, 5), torch.zeros(3, dtype=torch.long), ignore_index=ignore_index)


class _ReferenceKernel:
    """CPU backend stands in for kernels to exercise the actual autograd bridge.

    It deliberately uses PyTorch's oracle instead of asserting anything about
    Triton arithmetic, hardware support or performance on this CPU host.
    """
    def __init__(self, operation):
        self.operation = operation

    def __getitem__(self, launch_grid):
        return self.operation


@pytest.fixture
def reference_backend(monkeypatch):
    def forward(logits, targets, losses, gradients, vocabulary, ignore_index, **options):
        with torch.enable_grad():
            reference = logits.detach().clone().requires_grad_()
            values = functional.cross_entropy(reference, targets, reduction="none", ignore_index=ignore_index)
            derivatives, = torch.autograd.grad(values.sum(), reference)
        losses.copy_(values.detach())
        gradients.copy_(derivatives)

    def backward(gradients, row_scales, output, vocabulary, **options):
        output.copy_(gradients * row_scales.unsqueeze(-1))

    monkeypatch.setattr(fused_loss, "_load_kernels", lambda: (_ReferenceKernel(forward), _ReferenceKernel(backward)))
    monkeypatch.setattr(fused_loss, "_validate_inputs", lambda logits, targets, ignore_index: None)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())


def test_unreduced_bridge_preserves_mask_weighting_inputs_rng_and_repeated_backward(reference_backend):
    logits = torch.tensor([[0.3, -0.2, 0.1], [0.7, 0.4, -0.9], [-0.1, 0.2, 0.8]], requires_grad=True)
    targets = torch.tensor([-100, 1, 2])
    initial_logits = logits.detach().clone()
    initial_targets = targets.clone()
    rng = torch.get_rng_state().clone()
    values = fused_loss.fused_token_cross_entropy(logits, targets)
    reference = functional.cross_entropy(logits, targets, reduction="none", ignore_index=-100)
    assert torch.equal(values, reference)
    assert values.shape == targets.shape
    weights = torch.tensor([9.0, 0.125, 0.75])
    actual, = torch.autograd.grad((values * weights).sum(), logits, retain_graph=True)
    expected, = torch.autograd.grad((reference * weights).sum(), logits)
    assert torch.equal(actual, expected)
    repeated, = torch.autograd.grad((values * weights).sum(), logits)
    assert torch.equal(repeated, actual)
    assert torch.equal(actual[0], torch.zeros(3))
    assert torch.equal(logits.detach(), initial_logits)
    assert torch.equal(targets, initial_targets)
    assert torch.equal(torch.get_rng_state(), rng)


def test_unequal_example_weights_are_owned_by_caller(reference_backend):
    # Completion lengths1 and2 must receive equal example weight, rather than
    # a stock three-token mean. The bridge returns exactly the needed token CE.
    logits = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.7, 0.0]], requires_grad=True)
    targets = torch.tensor([1, 0, 0])
    losses = fused_loss.fused_token_cross_entropy(logits, targets)
    objective = (losses[0] + losses[1:].sum() / 2) / 2
    reference = functional.cross_entropy(logits, targets, reduction="none")
    expected_objective = (reference[0] + reference[1:].sum() / 2) / 2
    actual, = torch.autograd.grad(objective, logits, retain_graph=True)
    expected, = torch.autograd.grad(expected_objective, logits)
    assert torch.equal(objective, expected_objective)
    assert torch.equal(actual, expected)
    assert not torch.isclose(objective, reference.mean())


def test_masked_nonfinite_logits_remain_rejected_through_gradients(reference_backend):
    logits = torch.tensor([[float("nan"), 0.0], [0.7, 0.0]], requires_grad=True)
    targets = torch.tensor([-100, 0])
    losses = fused_loss.fused_token_cross_entropy(logits, targets)
    assert losses[0] == 0
    gradient, = torch.autograd.grad(losses.sum(), logits)
    assert not torch.isfinite(gradient[0]).all()


def test_noncontiguous_inputs_keep_correct_gradient_mapping(reference_backend):
    original = torch.tensor([[0.1, 0.5], [-0.2, 0.1], [0.7, -0.4]], requires_grad=True)
    logits = original.t()
    targets = torch.tensor([1, 2])
    loss = fused_loss.fused_token_cross_entropy(logits, targets).sum()
    expected = functional.cross_entropy(logits, targets, reduction="sum")
    actual_gradient, = torch.autograd.grad(loss, original)
    expected_gradient, = torch.autograd.grad(expected, original)
    assert torch.equal(loss, expected)
    assert torch.equal(actual_gradient, expected_gradient)
