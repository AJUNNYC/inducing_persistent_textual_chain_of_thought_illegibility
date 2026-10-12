"""CPU bridge tests; native Triton agreement and performance remain separate."""

from contextlib import nullcontext
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as functional

from training.sft import spatial_loss


@pytest.mark.parametrize("shape,target_shape", [((2, 5), (2,)), ((2, 5, 3), (2, 2)), ((2, 5, 3), (2, 3, 1)), ((0, 5, 3), (0, 3)), ((2, 0, 3), (2, 3)), ((2, 5, 0), (2, 0))])
def test_shape_contract_rejects_before_optional_runtime(shape, target_shape):
    with pytest.raises(ValueError):
        spatial_loss.spatial_token_cross_entropy(torch.empty(shape), torch.zeros(target_shape, dtype=torch.long))


@pytest.mark.parametrize("logit_dtype,target_dtype", [(torch.float16, torch.long), (torch.bfloat16, torch.long), (torch.float64, torch.long), (torch.float32, torch.int32), (torch.float32, torch.float32)])
def test_precision_is_not_silently_changed(logit_dtype, target_dtype):
    with pytest.raises(ValueError, match="FP32 logits and int64"):
        spatial_loss.spatial_token_cross_entropy(torch.empty(2, 5, 3, dtype=logit_dtype), torch.zeros(2, 3, dtype=target_dtype))


def test_cpu_fallback_is_forbidden():
    with pytest.raises(spatial_loss.SpatialLossUnsupported, match="CPU fallback is forbidden"):
        spatial_loss.spatial_token_cross_entropy(torch.zeros(2, 5, 3), torch.zeros(2, 3, dtype=torch.long))


@pytest.mark.parametrize("capability", [(8, 0), (12, 0)])
def test_native_candidate_guard_accepts_only_declared_architectures(monkeypatch, capability):
    """A guard pass permits measurement, not native numerical certification."""
    device = torch.device("cuda:0")
    logits = MagicMock(ndim=3, shape=(2, 5, 3), dtype=torch.float32, device=device)
    targets = MagicMock(ndim=2, shape=(2, 3), dtype=torch.long, device=device, requires_grad=False)
    targets.__lt__.return_value = False
    targets.__ge__.return_value = False
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda value: capability)
    monkeypatch.setattr(torch, "any", lambda value: False)
    spatial_loss._validate_inputs(logits, targets, -100)


@pytest.mark.parametrize("capability", [(7, 5), (8, 6), (8, 9), (9, 0), (10, 0), (12, 1)])
def test_unmeasured_architectures_fail_before_loading_kernel(monkeypatch, capability):
    device = torch.device("cuda:0")
    logits = MagicMock(ndim=3, shape=(2, 5, 3), dtype=torch.float32, device=device)
    targets = MagicMock(ndim=2, shape=(2, 3), dtype=torch.long, device=device, requires_grad=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda value: capability)
    monkeypatch.setattr(spatial_loss, "_load_backward_kernel", lambda: pytest.fail("unsupported architecture loaded kernel"))
    with pytest.raises(spatial_loss.SpatialLossUnsupported, match="SM80.*SM120"):
        spatial_loss.spatial_token_cross_entropy(logits, targets)


@pytest.mark.parametrize("ignore_index", [0, -1, -100.0, False])
def test_ignore_index_is_fixed(ignore_index):
    with pytest.raises(ValueError, match="ignore_index=-100"):
        spatial_loss.spatial_token_cross_entropy(torch.zeros(2, 5, 3), torch.zeros(2, 3, dtype=torch.long), ignore_index=ignore_index)


class _ReferenceKernel:
    """Exercise the production autograd bridge with the original expression.

    This CPU stand-in is not evidence about Triton's libdevice, native CUDA
    contraction, resources or execution speed.
    """
    def __getitem__(self, launch_grid):
        def run(log_probabilities, targets, row_scales, output, times, elements, vocabulary, ignore_index, **options):
            classes = torch.arange(vocabulary).view(1, -1, 1)
            summed_gradient = torch.where(targets == ignore_index, 0.0, -row_scales).unsqueeze(1)
            target_gradient = torch.where(classes == targets.unsqueeze(1), summed_gradient, 0.0)
            output.copy_(target_gradient - torch.exp(log_probabilities) * summed_gradient)
        return run


@pytest.fixture
def reference_backend(monkeypatch):
    monkeypatch.setattr(spatial_loss, "_load_backward_kernel", lambda: _ReferenceKernel())
    monkeypatch.setattr(spatial_loss, "_validate_inputs", lambda logits, targets, ignore_index: None)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())


def test_native_forward_ops_layout_mask_inputs_and_rng_are_preserved(reference_backend, monkeypatch):
    logits = torch.tensor([[[0.2, -0.4, 0.7], [0.9, 0.2, -0.1]]], requires_grad=True)
    targets = torch.tensor([[-100, 1, 0]])
    saved_logits, saved_targets = logits.detach().clone(), targets.clone()
    rng = torch.get_rng_state().clone()
    original_softmax, original_nll = torch.log_softmax, functional.nll_loss
    calls = []
    def log_softmax(value, **kwargs):
        calls.append(("log_softmax", value.shape, kwargs.copy()))
        return original_softmax(value, **kwargs)
    def nll(value, labels, **kwargs):
        calls.append(("nll_loss", value.shape, kwargs.copy()))
        return original_nll(value, labels, **kwargs)
    monkeypatch.setattr(torch, "log_softmax", log_softmax)
    monkeypatch.setattr(functional, "nll_loss", nll)
    actual = spatial_loss.spatial_token_cross_entropy(logits, targets)
    expected = functional.cross_entropy(logits, targets, ignore_index=-100, reduction="none")
    assert torch.equal(actual, expected)
    assert calls == [("log_softmax", logits.shape, {"dim": 1, "dtype": torch.float32}),
                     ("nll_loss", logits.shape, {"ignore_index": -100, "reduction": "none"})]
    assert torch.equal(logits.detach(), saved_logits)
    assert torch.equal(targets, saved_targets)
    assert torch.equal(torch.get_rng_state(), rng)


def test_unequal_examples_and_upstream_weights_remain_unchanged(reference_backend):
    logits = torch.tensor([[[0.2, -0.4, 0.7], [0.9, 0.2, -0.1]], [[0.1, 0.8, -0.3], [0.2, -0.7, 0.5]]], requires_grad=True)
    targets = torch.tensor([[-100, 1, 0], [-100, -100, 1]])
    counts = (targets != -100).sum(dim=1)
    losses = spatial_loss.spatial_token_cross_entropy(logits, targets)
    expected = functional.cross_entropy(logits, targets, ignore_index=-100, reduction="none")
    objective = (losses.sum(dim=1) / counts).mean()
    expected_objective = (expected.sum(dim=1) / counts).mean()
    actual_gradient, = torch.autograd.grad(objective, logits, retain_graph=True)
    expected_gradient, = torch.autograd.grad(expected_objective, logits)
    repeated_gradient, = torch.autograd.grad(objective, logits)
    assert torch.equal(objective, expected_objective)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-6, atol=1e-7)
    assert torch.equal(actual_gradient, repeated_gradient)
    assert not torch.isclose(objective, expected.sum() / counts.sum())


def test_noncontiguous_input_and_nonuniform_row_scales_have_correct_gradient_mapping(reference_backend):
    original = torch.tensor([[[0.1, 0.5, -0.1], [-0.2, 0.1, 0.2]]], requires_grad=True)
    logits = original.transpose(1, 2)
    targets = torch.tensor([[1, 2]])
    weights = torch.tensor([[0.125, 0.7]])
    losses = spatial_loss.spatial_token_cross_entropy(logits, targets)
    expected = functional.cross_entropy(logits, targets, reduction="none")
    actual_gradient, = torch.autograd.grad((losses * weights).sum(), original)
    expected_gradient, = torch.autograd.grad((expected * weights).sum(), original)
    assert torch.equal(losses, expected)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_ignored_nonfinite_logits_are_not_hidden_in_backward(reference_backend, bad_value):
    logits = torch.tensor([[[bad_value, 0.0], [0.0, 0.7]]], requires_grad=True)
    targets = torch.tensor([[-100, 1]])
    losses = spatial_loss.spatial_token_cross_entropy(logits, targets)
    expected = functional.cross_entropy(logits, targets, reduction="none")
    actual_gradient, = torch.autograd.grad(losses.sum(), logits)
    expected_gradient, = torch.autograd.grad(expected.sum(), logits)
    assert torch.equal(losses, expected)
    assert torch.equal(torch.isfinite(actual_gradient), torch.isfinite(expected_gradient))
    assert not torch.isfinite(actual_gradient[:, :, 0]).all() or bad_value == -float("inf")


def test_saved_target_mutation_rejects_corrupted_backward(reference_backend):
    logits = torch.zeros(1, 3, 2, requires_grad=True)
    targets = torch.tensor([[1, 2]])
    losses = spatial_loss.spatial_token_cross_entropy(logits, targets)
    targets[0, 0] = 2
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        losses.sum().backward()
