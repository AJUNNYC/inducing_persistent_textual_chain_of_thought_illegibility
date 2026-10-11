"""CPU causal/layout bridge checks; native bandwidth and arithmetic are separate."""

import ast
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch
import torch.nn.functional as functional

from training.sft import causal_gradient_loss as causal
from training.sft import scheduled_spatial_loss as scheduled


@pytest.mark.parametrize("shape,target_shape", [((2, 5), (2,)), ((2, 4, 5), (2, 4)), ((2, 4, 5), (2, 2)), ((2, 4, 5), (2, 3, 1)), ((2, 1, 5), (2, 0)), ((0, 4, 5), (0, 3)), ((2, 4, 0), (2, 3))])
def test_full_logit_and_shifted_target_shape_contract(shape, target_shape):
    with pytest.raises(ValueError):
        causal.causal_gradient_token_cross_entropy(torch.empty(shape), torch.zeros(target_shape, dtype=torch.long))


@pytest.mark.parametrize("logit_dtype,target_dtype", [(torch.float16, torch.long), (torch.bfloat16, torch.long), (torch.float64, torch.long), (torch.float32, torch.int32)])
def test_dtype_contract_does_not_silently_change_precision(logit_dtype, target_dtype):
    with pytest.raises(ValueError, match="FP32 logits and int64"):
        causal.causal_gradient_token_cross_entropy(torch.empty(2, 4, 5, dtype=logit_dtype), torch.zeros(2, 3, dtype=target_dtype))


def test_cpu_fallback_is_forbidden():
    with pytest.raises(causal.SpatialLossUnsupported, match="CPU fallback is forbidden"):
        causal.causal_gradient_token_cross_entropy(torch.zeros(2, 4, 5), torch.zeros(2, 3, dtype=torch.long))


class _ReferenceKernel:
    def __init__(self, operation, name, launches):
        self.operation, self.name, self.launches = operation, name, launches

    def __getitem__(self, grid):
        def run(*args, **options):
            self.launches.append((self.name, grid, options.copy()))
            self.operation(*args, **options)
        return run


@pytest.fixture
def reference_backend(monkeypatch):
    """Exercise the real delegation and full-gradient allocation on a CPU.

    Tensor math stands in for native kernels; it makes no claim about CUDA
    libdevice, coalescing, allocation savings or measured execution speed.
    """
    launches = []
    def serial_forward(logits, targets, losses, maxima, log_sums, times, vocabulary, ignore_index, **options):
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

    def write_full_gradient(values, targets, maxima, log_sums, row_scales, output, times, full_times, vocabulary, ignore_index, reconstruct, **options):
        assert output.shape == (values.shape[0], times + 1, vocabulary)
        assert output.is_contiguous()
        log_probabilities = values - maxima.unsqueeze(1) - log_sums.unsqueeze(1) if reconstruct else values
        classes = torch.arange(vocabulary).view(1, -1, 1)
        summed_gradient = torch.where(targets == ignore_index, 0.0, -row_scales).unsqueeze(1)
        target_gradient = torch.where(classes == targets.unsqueeze(1), summed_gradient, 0.0)
        derivative = target_gradient - torch.exp(log_probabilities) * summed_gradient
        output[:, :-1, :].copy_(derivative.transpose(1, 2))

    def zero_tail(output, full_times, vocabulary, **options):
        output[:, -1, :].zero_()

    serial_kernel = _ReferenceKernel(serial_forward, "accepted_serial_forward", launches)
    full_gradient_kernel = _ReferenceKernel(write_full_gradient, "full_gradient", launches)
    zero_kernel = _ReferenceKernel(zero_tail, "zero_tail", launches)
    monkeypatch.setattr(scheduled, "_load_scheduled_kernels", lambda: (serial_kernel, None))
    monkeypatch.setattr(scheduled, "_load_backward_kernel", lambda: None)
    monkeypatch.setattr(causal, "_load_gradient_kernels", lambda: (full_gradient_kernel, zero_kernel))
    monkeypatch.setattr(causal, "_validate_inputs", lambda logits, targets, ignore_index: None)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    return launches


def _graph_node_names(node):
    names, pending, visited = [], [node], set()
    while pending:
        item = pending.pop()
        if item is None or item in visited:
            continue
        visited.add(item)
        names.append(type(item).__name__)
        pending.extend(next_item for next_item, _ in item.next_functions)
    return names


@pytest.mark.parametrize("full_times", [2, 65, 66, 130])
def test_causal_shift_masks_weights_full_gradient_strides_and_rng(reference_backend, full_times):
    logits = torch.linspace(-1.5, 0.9, 2 * full_times * 5).reshape(2, full_times, 5).requires_grad_()
    targets = torch.arange(2 * (full_times - 1)).reshape(2, full_times - 1) % 5
    targets[0, :(full_times - 1) // 2] = -100
    targets[1, :(full_times - 1) // 3] = -100
    original_logits, original_targets = logits.detach().clone(), targets.clone()
    rng = torch.get_rng_state().clone()
    losses = causal.causal_gradient_token_cross_entropy(logits, targets)
    expected = functional.cross_entropy(logits[:, :-1, :].transpose(1, 2), targets,
                                        ignore_index=-100, reduction="none")
    counts = (targets != -100).sum(dim=1)
    objective = (losses.sum(dim=1) / counts).mean()
    expected_objective = (expected.sum(dim=1) / counts).mean()
    actual_gradient, = torch.autograd.grad(objective, logits, retain_graph=True)
    expected_gradient, = torch.autograd.grad(expected_objective, logits)
    repeated_gradient, = torch.autograd.grad(objective, logits)
    torch.testing.assert_close(losses, expected, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-6, atol=1e-7)
    assert actual_gradient.shape == logits.shape
    assert actual_gradient.is_contiguous()
    assert actual_gradient.stride() == (full_times * 5, 5, 1)
    assert torch.equal(actual_gradient[:, -1, :], torch.zeros(2, 5))
    assert torch.equal(actual_gradient, repeated_gradient)
    assert torch.equal(logits.detach(), original_logits)
    assert torch.equal(targets, original_targets)
    assert torch.equal(torch.get_rng_state(), rng)
    assert not any("SliceBackward" in name or "TransposeBackward" in name
                   for name in _graph_node_names(losses.grad_fn))
    launches = [name for name, _, _ in reference_backend]
    expected_launches = ["full_gradient", "zero_tail", "full_gradient", "zero_tail"]
    if full_times - 1 > 64:
        expected_launches.insert(0, "accepted_serial_forward")
    assert launches == expected_launches


def test_noncontiguous_full_input_has_correct_gradient_mapping(reference_backend):
    original = torch.linspace(-0.5, 0.7, 66 * 5).reshape(1, 5, 66).requires_grad_()
    logits = original.transpose(1, 2)
    targets = torch.arange(65).reshape(1, 65) % 5
    weights = torch.linspace(0.01, 0.04, 65).reshape(1, 65)
    losses = causal.causal_gradient_token_cross_entropy(logits, targets)
    expected = functional.cross_entropy(logits[:, :-1, :].transpose(1, 2), targets, reduction="none")
    direct_gradient, = torch.autograd.grad((losses * weights).sum(), logits, retain_graph=True)
    actual_gradient, = torch.autograd.grad((losses * weights).sum(), original)
    expected_gradient, = torch.autograd.grad((expected * weights).sum(), original)
    assert direct_gradient.is_contiguous()
    assert direct_gradient.stride() == (66 * 5, 5, 1)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_unused_final_logits_never_contaminate_loss_or_tail_gradient(reference_backend, bad_value):
    logits = torch.zeros(1, 66, 5, requires_grad=True)
    with torch.no_grad():
        logits[:, -1, :] = bad_value
    targets = torch.zeros(1, 65, dtype=torch.long)
    losses = causal.causal_gradient_token_cross_entropy(logits, targets)
    gradient, = torch.autograd.grad(losses.sum(), logits)
    assert torch.isfinite(losses).all()
    assert torch.isfinite(gradient).all()
    assert torch.equal(gradient[:, -1, :], torch.zeros(1, 5))


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_masked_nonfinite_causal_rows_keep_original_rejection(reference_backend, bad_value):
    logits = torch.zeros(1, 66, 5, requires_grad=True)
    with torch.no_grad():
        logits[0, 0, 0] = bad_value
    targets = torch.zeros(1, 65, dtype=torch.long)
    targets[0, 0] = -100
    losses = causal.causal_gradient_token_cross_entropy(logits, targets)
    expected = functional.cross_entropy(logits[:, :-1, :].transpose(1, 2), targets,
                                        ignore_index=-100, reduction="none")
    actual_gradient, = torch.autograd.grad(losses.sum(), logits)
    expected_gradient, = torch.autograd.grad(expected.sum(), logits)
    torch.testing.assert_close(losses, expected, rtol=1e-6, atol=1e-7)
    assert torch.equal(torch.isfinite(actual_gradient), torch.isfinite(expected_gradient))
    assert torch.equal(actual_gradient[:, -1, :], torch.zeros(1, 5))


def test_frozen_fp32_head_receives_original_full_gradient_shape(reference_backend):
    hidden = torch.linspace(-0.9, 0.5, 2 * 66 * 3).reshape(2, 66, 3).requires_grad_()
    weight = torch.linspace(-0.3, 0.6, 5 * 3).reshape(5, 3)
    targets = torch.arange(2 * 65).reshape(2, 65) % 5
    logits = functional.linear(hidden, weight)
    actual = causal.causal_gradient_token_cross_entropy(logits, targets).mean()
    expected = functional.cross_entropy(logits[:, :-1, :].transpose(1, 2), targets)
    actual_gradient, = torch.autograd.grad(actual, hidden, retain_graph=True)
    expected_gradient, = torch.autograd.grad(expected, hidden)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-6, atol=1e-7)
    assert torch.equal(actual_gradient[:, -1, :], torch.zeros(2, 3))
    assert weight.dtype == torch.float32 and not weight.requires_grad


def test_native_backward_error_does_not_select_another_kernel(reference_backend, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("native gradient kernel failed")
    monkeypatch.setattr(causal, "_load_gradient_kernels", lambda: (_ReferenceKernel(fail, "failed", []), None))
    logits = torch.zeros(1, 66, 5, requires_grad=True)
    targets = torch.zeros(1, 65, dtype=torch.long)
    losses = causal.causal_gradient_token_cross_entropy(logits, targets)
    with pytest.raises(RuntimeError, match="native gradient kernel failed"):
        losses.sum().backward()


def test_source_retains_accepted_delegation_direct_full_layout_and_zero_tail():
    tree = ast.parse(Path(causal.__file__).read_text())
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    forward = next(node for node in functions if node.name == "forward" and node.args.args[0].arg == "ctx")
    assert "shifted_logits = logits[:, :-1, :].float()" in ast.unparse(forward)
    assert "_scheduled_autograd_function().forward(ctx, shifted_logits.transpose(1, 2), targets, ignore_index)" in ast.unparse(forward)
    backward = next(node for node in functions if node.name == "backward" and node.args.args[0].arg == "values")
    assert "destination_offset = (batch * full_times + times) * n_columns + classes" in ast.unparse(backward)
    assert "derivative = target_gradient - libdevice.exp(log_probability) * summed_gradient" in ast.unparse(backward)
    tail = next(node for node in functions if node.name == "zero_last_row")
    assert "tl.store(output + offsets, 0.0, classes < n_columns)" in ast.unparse(tail)
