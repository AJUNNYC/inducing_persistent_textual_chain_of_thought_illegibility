"""CPU route/bridge audits; these tests do not certify native CUDA arithmetic."""

import ast
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch
import torch.nn.functional as functional

from training.sft import scheduled_spatial_loss as scheduled


@pytest.mark.parametrize("length,route", [(1, "original_spatial"), (64, "original_spatial"), (65, "serial_token_blocks"), (3190, "serial_token_blocks")])
def test_declared_shape_router(length, route):
    assert scheduled.forward_route(length) == route


@pytest.mark.parametrize("length", [0, -1, 1.0, True])
def test_router_rejects_invalid_time(length):
    with pytest.raises(ValueError, match="positive integer"):
        scheduled.forward_route(length)


@pytest.mark.parametrize("shape,target_shape", [((2, 5), (2,)), ((2, 5, 3), (2, 2)), ((2, 5, 3), (2, 3, 1)), ((0, 5, 3), (0, 3)), ((2, 0, 3), (2, 3)), ((2, 5, 0), (2, 0))])
def test_shared_input_shape_contract(shape, target_shape):
    with pytest.raises(ValueError):
        scheduled.scheduled_spatial_token_cross_entropy(torch.empty(shape), torch.zeros(target_shape, dtype=torch.long))


@pytest.mark.parametrize("logit_dtype,target_dtype", [(torch.float16, torch.long), (torch.bfloat16, torch.long), (torch.float64, torch.long), (torch.float32, torch.int32)])
def test_dtype_contract_is_not_silently_cast(logit_dtype, target_dtype):
    with pytest.raises(ValueError, match="FP32 logits and int64"):
        scheduled.scheduled_spatial_token_cross_entropy(torch.empty(2, 5, 3, dtype=logit_dtype), torch.zeros(2, 3, dtype=target_dtype))


def test_cpu_fallback_is_forbidden():
    with pytest.raises(scheduled.SpatialLossUnsupported, match="CPU fallback is forbidden"):
        scheduled.scheduled_spatial_token_cross_entropy(torch.zeros(2, 5, 3), torch.zeros(2, 3, dtype=torch.long))


class _ReferenceKernel:
    def __init__(self, operation, name, launches):
        self.operation, self.name, self.launches = operation, name, launches

    def __getitem__(self, grid):
        def run(*args, **kwargs):
            self.launches.append((self.name, grid, kwargs.copy()))
            self.operation(*args, **kwargs)
        return run


@pytest.fixture
def reference_backend(monkeypatch):
    """Use CPU math to exercise the real autograd bridge and declared routes.

    CPU torch.exp/log are not claims about CUDA libdevice or its compiler. The
    actual native paired screen independently measures those implementation
    differences and every original strict numerical bound.
    """
    launches = []
    def forward(logits, targets, losses, maxima, log_sums, times, vocabulary, ignore_index, **options):
        maximum = torch.full_like(losses, torch.finfo(torch.float32).min)
        for index in range(vocabulary):
            maximum = torch.where(maximum < logits[:, index, :], logits[:, index, :], maximum)
        normalization = torch.zeros_like(losses)
        for index in range(vocabulary):
            normalization = normalization + torch.exp(logits[:, index, :] - maximum)
        log_sum = torch.log(normalization)
        target_values = logits.gather(1, targets.clamp_min(0).unsqueeze(1)).squeeze(1)
        losses.copy_(torch.where(targets == ignore_index, 0.0, -(target_values - maximum - log_sum)))
        maxima.copy_(maximum)
        log_sums.copy_(log_sum)

    def derivative(log_probabilities, targets, row_scales, output, vocabulary, ignore_index):
        classes = torch.arange(vocabulary).view(1, -1, 1)
        summed_gradient = torch.where(targets == ignore_index, 0.0, -row_scales).unsqueeze(1)
        target_gradient = torch.where(classes == targets.unsqueeze(1), summed_gradient, 0.0)
        output.copy_(target_gradient - torch.exp(log_probabilities) * summed_gradient)

    def original_backward(log_probabilities, targets, row_scales, output, times, elements, vocabulary, ignore_index, **options):
        derivative(log_probabilities, targets, row_scales, output, vocabulary, ignore_index)

    def scheduled_backward(logits, targets, maxima, log_sums, row_scales, output, times, elements, vocabulary, ignore_index, **options):
        log_probabilities = logits - maxima.unsqueeze(1) - log_sums.unsqueeze(1)
        derivative(log_probabilities, targets, row_scales, output, vocabulary, ignore_index)

    forward_kernel = _ReferenceKernel(forward, "serial_forward", launches)
    scheduled_backward_kernel = _ReferenceKernel(scheduled_backward, "reconstructed_backward", launches)
    original_backward_kernel = _ReferenceKernel(original_backward, "original_backward", launches)
    monkeypatch.setattr(scheduled, "_load_scheduled_kernels", lambda: (forward_kernel, scheduled_backward_kernel))
    monkeypatch.setattr(scheduled, "_load_backward_kernel", lambda: original_backward_kernel)
    monkeypatch.setattr(scheduled, "_validate_inputs", lambda logits, targets, ignore_index: None)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    return launches


@pytest.mark.parametrize("times", [1, 64, 65, 129])
def test_shape_routes_masks_weights_gradient_inputs_and_rng(reference_backend, times):
    logits = torch.linspace(-1.7, 1.3, 2 * 5 * times).reshape(2, 5, times).requires_grad_()
    targets = torch.arange(2 * times).reshape(2, times) % 5
    targets[0, :times // 2] = -100
    targets[1, :times // 3] = -100
    original_logits, original_targets = logits.detach().clone(), targets.clone()
    rng = torch.get_rng_state().clone()
    losses = scheduled.scheduled_spatial_token_cross_entropy(logits, targets)
    expected = functional.cross_entropy(logits, targets, ignore_index=-100, reduction="none")
    counts = (targets != -100).sum(dim=1)
    objective = (losses.sum(dim=1) / counts).mean()
    expected_objective = (expected.sum(dim=1) / counts).mean()
    actual_gradient, = torch.autograd.grad(objective, logits, retain_graph=True)
    expected_gradient, = torch.autograd.grad(expected_objective, logits)
    repeated_gradient, = torch.autograd.grad(objective, logits)
    torch.testing.assert_close(losses, expected, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-6, atol=1e-7)
    assert torch.equal(actual_gradient, repeated_gradient)
    assert torch.equal(logits.detach(), original_logits)
    assert torch.equal(targets, original_targets)
    assert torch.equal(torch.get_rng_state(), rng)
    names = [name for name, _, _ in reference_backend]
    if times <= 64:
        assert names == ["original_backward", "original_backward"]
        assert torch.equal(losses, expected)
    else:
        assert names == ["serial_forward", "reconstructed_backward", "reconstructed_backward"]
        name, grid, options = reference_backend[0]
        assert grid == (2, (times + 63) // 64)
        assert options == {"block_size": 64, "num_warps": 2, "enable_fp_fusion": False}


def test_short_route_uses_original_forward_operations(reference_backend, monkeypatch):
    original_softmax, original_nll = torch.log_softmax, functional.nll_loss
    calls = []
    def softmax(value, **options):
        calls.append(("log_softmax", options))
        return original_softmax(value, **options)
    def nll(value, targets, **options):
        calls.append(("nll_loss", options))
        return original_nll(value, targets, **options)
    monkeypatch.setattr(torch, "log_softmax", softmax)
    monkeypatch.setattr(functional, "nll_loss", nll)
    scheduled.scheduled_spatial_token_cross_entropy(torch.zeros(1, 5, 64), torch.zeros(1, 64, dtype=torch.long))
    assert calls == [("log_softmax", {"dim": 1, "dtype": torch.float32}),
                     ("nll_loss", {"ignore_index": -100, "reduction": "none"})]


def test_noncontiguous_inputs_and_nonuniform_row_scales(reference_backend):
    original = torch.linspace(-0.5, 0.7, 65 * 5).reshape(1, 65, 5).requires_grad_()
    logits = original.transpose(1, 2)
    targets = torch.arange(65).reshape(1, 65) % 5
    weights = torch.linspace(0.01, 0.04, 65).reshape(1, 65)
    losses = scheduled.scheduled_spatial_token_cross_entropy(logits, targets)
    expected = functional.cross_entropy(logits, targets, reduction="none")
    actual_gradient, = torch.autograd.grad((losses * weights).sum(), original)
    expected_gradient, = torch.autograd.grad((expected * weights).sum(), original)
    torch.testing.assert_close(losses, expected, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_ignored_nonfinite_rows_preserve_gradient_rejection(reference_backend, bad_value):
    logits = torch.zeros(1, 5, 65, requires_grad=True)
    with torch.no_grad():
        logits[0, 0, 0] = bad_value
    targets = torch.zeros(1, 65, dtype=torch.long)
    targets[0, 0] = -100
    losses = scheduled.scheduled_spatial_token_cross_entropy(logits, targets)
    expected = functional.cross_entropy(logits, targets, reduction="none")
    actual_gradient, = torch.autograd.grad(losses.sum(), logits)
    expected_gradient, = torch.autograd.grad(expected.sum(), logits)
    torch.testing.assert_close(losses, expected, rtol=1e-6, atol=1e-7)
    assert torch.equal(torch.isfinite(actual_gradient), torch.isfinite(expected_gradient))


def test_kernel_failure_does_not_fallback(reference_backend, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("native kernel failed")
    monkeypatch.setattr(scheduled, "_load_scheduled_kernels", lambda: (_ReferenceKernel(fail, "failed", []), None))
    with pytest.raises(RuntimeError, match="native kernel failed"):
        scheduled.scheduled_spatial_token_cross_entropy(torch.zeros(1, 5, 65), torch.zeros(1, 65, dtype=torch.long))


def test_kernel_source_retains_serial_class_recurrence_and_native_transcendentals():
    source = Path(scheduled.__file__).read_text()
    tree = ast.parse(source)
    forward = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "forward" and len(node.args.args) > 4)
    loops = [node for node in forward.body if isinstance(node, ast.For)]
    assert len(loops) == 2
    assert all(ast.unparse(node.iter) == "range(0, n_columns)" for node in loops)
    assert next(arg for arg in forward.args.args if arg.arg == "n_columns").annotation is None
    assert "maximum = tl.where(maximum < values, values, maximum)" in ast.unparse(forward)
    assert "normalization = normalization + libdevice.exp(values - maximum)" in ast.unparse(forward)
    assert "log_sum = libdevice.log(normalization)" in ast.unparse(forward)
    assert "tl.sum" not in ast.unparse(forward) and "tl.max" not in ast.unparse(forward)
