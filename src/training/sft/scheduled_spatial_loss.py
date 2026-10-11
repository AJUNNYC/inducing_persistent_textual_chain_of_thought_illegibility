"""Explicit token scheduling candidate for the original FP32 spatial CE.

Pinned PyTorch 2.9.0 ``SpatialSoftMax_getBlockSize`` uses one thread per token
when time >64. Its ``cunn_SpatialSoftMaxForward`` then accumulates vocabulary
classes serially in FP32. This candidate retains that exact class order while
scheduling 64 tokens per CTA. It stores target losses and row normalization
metadata rather than materializing every log-probability. Backward reconstructs
``input - max_input - logsum`` before the accepted pointwise derivative.

For time <=64 the original spatial forward remains explicit: PyTorch may use
multiple threads for class reduction there. This is a declared shape route,
not a fallback after a kernel error. No exception selects another backend.

Authoritative reference: PyTorch v2.9.0
https://github.com/pytorch/pytorch/blob/v2.9.0/aten/src/ATen/native/cuda/SoftMax.cu
Its BSD-3-Clause copyright, conditions and disclaimer are retained in the
adjacent ``spatial_loss.py``, whose original forward/backward contract is reused.
FP32 CUDA libdevice exp/log retain the original transcendental functions;
actual CUDA losses, gradients, memory and speed must pass the unchanged screen.
"""

from functools import lru_cache
from importlib import metadata
from typing import Any

from .spatial_loss import SpatialLossUnsupported, _load_backward_kernel, _validate_inputs


TOKEN_BLOCK_SIZE = 64
ORIGINAL_FORWARD_MAX_TIME = 64


def forward_route(n_times: int) -> str:
    """Expose the declared shape rule for evidence and focused CPU checks."""
    if type(n_times) is not int or n_times < 1:
        raise ValueError("Spatial CE time must be a positive integer")
    return "original_spatial" if n_times <= ORIGINAL_FORWARD_MAX_TIME else "serial_token_blocks"


@lru_cache(maxsize=1)
def _load_scheduled_kernels() -> tuple[Any, Any]:
    global triton, tl, libdevice
    try:
        import triton
        import triton.language as tl
        from triton.language.extra.cuda import libdevice
    except ImportError as error:
        raise SpatialLossUnsupported("Scheduled spatial CE requires pinned Triton 3.5.0") from error
    if metadata.version("triton") != "3.5.0":
        raise SpatialLossUnsupported("Scheduled spatial CE requires exactly Triton 3.5.0")

    @triton.jit(do_not_specialize=["n_times", "n_columns"])
    def forward(
        logits, targets, losses, maxima, log_sums,
        n_times, n_columns, ignore_index: tl.constexpr, block_size: tl.constexpr,
    ):
        batch = tl.program_id(0).to(tl.int64)
        times = tl.program_id(1) * block_size + tl.arange(0, block_size)
        present = times < n_times
        row_offset = batch * n_columns * n_times + times
        token_offset = batch * n_times + times
        # Match at::numeric_limits<float>::lowest(), not negative infinity.
        maximum = tl.full((block_size,), -3.4028234663852886e38, tl.float32)
        # Runtime loop bounds preserve serial class order without giant unroll.
        for class_index in range(0, n_columns):
            values = tl.load(logits + row_offset + class_index.to(tl.int64) * n_times,
                             present, other=-3.4028234663852886e38)
            maximum = tl.where(maximum < values, values, maximum)
        normalization = tl.full((block_size,), 0.0, tl.float32)
        for class_index in range(0, n_columns):
            values = tl.load(logits + row_offset + class_index.to(tl.int64) * n_times,
                             present, other=-3.4028234663852886e38)
            normalization = normalization + libdevice.exp(values - maximum)
        log_sum = libdevice.log(normalization)
        target = tl.load(targets + token_offset, present, other=ignore_index)
        target_value = tl.load(logits + row_offset + target * n_times,
                               present & (target != ignore_index), other=0.0)
        loss = tl.where(target == ignore_index, 0.0, -(target_value - maximum - log_sum))
        tl.store(losses + token_offset, loss, present)
        tl.store(maxima + token_offset, maximum, present)
        tl.store(log_sums + token_offset, log_sum, present)

    @triton.jit(do_not_specialize=["n_times", "n_elements"])
    def backward(
        logits, targets, maxima, log_sums, row_scales, output,
        n_times, n_elements, n_columns: tl.constexpr,
        ignore_index: tl.constexpr, block_size: tl.constexpr,
    ):
        offsets = (tl.program_id(0).to(tl.int64) * block_size
                   + tl.arange(0, block_size))
        present = offsets < n_elements
        times = offsets % n_times
        class_index = (offsets // n_times) % n_columns
        batch = offsets // (n_times * n_columns)
        token_offset = batch * n_times + times
        values = tl.load(logits + offsets, present, other=0.0)
        maximum = tl.load(maxima + token_offset, present, other=0.0)
        log_sum = tl.load(log_sums + token_offset, present, other=0.0)
        # Preserve both original FP32 subtractions before native expf.
        log_probability = values - maximum - log_sum
        target = tl.load(targets + token_offset, present, other=ignore_index)
        row_scale = tl.load(row_scales + token_offset, present, other=0.0)
        summed_gradient = tl.where(target == ignore_index, 0.0, -row_scale)
        target_gradient = tl.where(class_index == target, summed_gradient, 0.0)
        derivative = target_gradient - libdevice.exp(log_probability) * summed_gradient
        # Ignored nonfinite rows retain 0-exp(NaN)*0 and the original rejection.
        tl.store(output + offsets, derivative, present)

    return forward, backward


@lru_cache(maxsize=1)
def _autograd_function() -> Any:
    import torch
    import torch.nn.functional as functional

    class ScheduledSpatialCrossEntropy(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, logits: Any, targets: Any, ignore_index: int) -> Any:
            ctx.route = forward_route(logits.shape[2])
            ctx.ignore_index = ignore_index
            if ctx.route == "original_spatial":
                _load_backward_kernel()
                log_probabilities = torch.log_softmax(logits, dim=1, dtype=logits.dtype)
                losses = functional.nll_loss(log_probabilities, targets,
                                             ignore_index=ignore_index, reduction="none")
                ctx.save_for_backward(log_probabilities.contiguous(), targets.contiguous())
                return losses
            kernel, _ = _load_scheduled_kernels()
            raw_logits, targets = logits.contiguous(), targets.contiguous()
            row_shape = (logits.shape[0], logits.shape[2])
            losses = torch.empty(row_shape, dtype=torch.float32, device=logits.device)
            maxima = torch.empty_like(losses)
            log_sums = torch.empty_like(losses)
            with torch.cuda.device(logits.device):
                kernel[(logits.shape[0], (logits.shape[2] + TOKEN_BLOCK_SIZE - 1) // TOKEN_BLOCK_SIZE)](
                    raw_logits, targets, losses, maxima, log_sums,
                    logits.shape[2], logits.shape[1], ignore_index,
                    block_size=TOKEN_BLOCK_SIZE, num_warps=2, enable_fp_fusion=False,
                )
            ctx.save_for_backward(raw_logits, targets, maxima, log_sums)
            return losses

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx: Any, row_scales: Any) -> tuple[Any, None, None]:
            saved = ctx.saved_tensors
            values, targets = saved[:2]
            output = torch.empty_like(values)
            row_scales = row_scales.contiguous()
            with torch.cuda.device(values.device):
                if ctx.route == "original_spatial":
                    kernel = _load_backward_kernel()
                    kernel[((values.numel() + 1023) // 1024,)](
                        values, targets, row_scales, output,
                        values.shape[2], values.numel(), values.shape[1], ctx.ignore_index,
                        block_size=1024, num_warps=4, enable_fp_fusion=True,
                    )
                else:
                    _, kernel = _load_scheduled_kernels()
                    maxima, log_sums = saved[2:]
                    kernel[((values.numel() + 1023) // 1024,)](
                        values, targets, maxima, log_sums, row_scales, output,
                        values.shape[2], values.numel(), values.shape[1], ctx.ignore_index,
                        block_size=1024, num_warps=4, enable_fp_fusion=True,
                    )
            return output, None, None

    return ScheduledSpatialCrossEntropy


def scheduled_spatial_token_cross_entropy(logits: Any, targets: Any, *, ignore_index: int = -100) -> Any:
    """Return unreduced FP32 spatial CE with a declared exact-order shape route.

    Labels, original causal layout and per-example loss reduction are unchanged.
    CUDA/Triton failures propagate; they never silently select original forward.
    """
    _validate_inputs(logits, targets, ignore_index)
    return _autograd_function().apply(logits, targets, ignore_index)
