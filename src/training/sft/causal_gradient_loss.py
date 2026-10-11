"""Final copy-reduction candidate; retain the accepted FP32 CE forward.

The helper accepts full [B,T,V] logits and already causally shifted [B,T-1]
integer targets. Its custom autograd forward performs the original one-token
logit shift and delegates the accepted scheduled spatial forward unchanged.
Backward writes the complete contiguous [B,T,V] gradient directly, with its
unused last prediction row set to zero. It does not change the output-head
GEMM, vocabulary, precision, masks, token weighting or randomness.

Pinned PyTorch 2.9.0 ``aten/src/ATen/native/TensorShape.cpp::slice_backward``
allocates a full zeros tensor and copies the sliced gradient into it. Moving
that exact shift into this custom forward removes that separate allocation
and scatter. A 32x32 tiled backward reads contiguous spatial [B,V,T-1] data
and writes contiguous token-major output. Triton 3.5's Coalesce pass selects
memory layouts from pointer axis information; native performance still needs
measurement. The accepted coalesced spatial forward copy is retained.

https://github.com/pytorch/pytorch/blob/v2.9.0/aten/src/ATen/native/TensorShape.cpp
https://github.com/triton-lang/triton/blob/release/3.5.x/lib/Dialect/TritonGPU/Transforms/Coalesce.cpp
PyTorch's BSD-3-Clause reference notice is retained in spatial_loss.py. The
accepted forward and original CUDA derivative are reused, not redefined.
"""

from functools import lru_cache
from importlib import metadata
from typing import Any

from .scheduled_spatial_loss import _autograd_function as _scheduled_autograd_function
from .spatial_loss import SpatialLossUnsupported, _validate_inputs as _validate_spatial_inputs


CLASS_TILE_SIZE = 32
TIME_TILE_SIZE = 32


@lru_cache(maxsize=1)
def _load_gradient_kernels() -> tuple[Any, Any]:
    global triton, tl, libdevice
    try:
        import triton
        import triton.language as tl
        from triton.language.extra.cuda import libdevice
    except ImportError as error:
        raise SpatialLossUnsupported("Causal gradient CE requires pinned Triton 3.5.0") from error
    if metadata.version("triton") != "3.5.0":
        raise SpatialLossUnsupported("Causal gradient CE requires exactly Triton 3.5.0")

    @triton.jit(do_not_specialize=["n_times", "full_times"])
    def backward(
        values, targets, maxima, log_sums, row_scales, output,
        n_times, full_times, n_columns: tl.constexpr,
        ignore_index: tl.constexpr, reconstruct: tl.constexpr,
        class_tile: tl.constexpr, time_tile: tl.constexpr,
    ):
        batch = tl.program_id(0).to(tl.int64)
        classes = (tl.program_id(1) * class_tile + tl.arange(0, class_tile))[:, None]
        times = (tl.program_id(2) * time_tile + tl.arange(0, time_tile))[None, :]
        present = (classes < n_columns) & (times < n_times)
        source_offset = (batch * n_columns + classes) * n_times + times
        token_offset = batch * n_times + times
        log_probability = tl.load(values + source_offset, present, other=0.0)
        if reconstruct:
            maximum = tl.load(maxima + token_offset, times < n_times, other=0.0)
            log_sum = tl.load(log_sums + token_offset, times < n_times, other=0.0)
            log_probability = log_probability - maximum - log_sum
        target = tl.load(targets + token_offset, times < n_times, other=ignore_index)
        row_scale = tl.load(row_scales + token_offset, times < n_times, other=0.0)
        summed_gradient = tl.where(target == ignore_index, 0.0, -row_scale)
        target_gradient = tl.where(classes == target, summed_gradient, 0.0)
        derivative = target_gradient - libdevice.exp(log_probability) * summed_gradient
        # These pointers expose contiguous time reads and vocabulary writes to
        # Triton's memory-op coalescing pass. FP32 arithmetic is unchanged.
        destination_offset = (batch * full_times + times) * n_columns + classes
        tl.store(output + destination_offset, derivative, present)

    @triton.jit(do_not_specialize=["full_times"])
    def zero_last_row(
        output, full_times, n_columns: tl.constexpr, block_size: tl.constexpr,
    ):
        batch = tl.program_id(0).to(tl.int64)
        classes = tl.program_id(1) * block_size + tl.arange(0, block_size)
        offsets = (batch * full_times + full_times - 1) * n_columns + classes
        tl.store(output + offsets, 0.0, classes < n_columns)

    return backward, zero_last_row


@lru_cache(maxsize=1)
def _autograd_function() -> Any:
    import torch

    class CausalGradientCrossEntropy(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, logits: Any, targets: Any, ignore_index: int) -> Any:
            ctx.full_shape = tuple(logits.shape)
            shifted_logits = logits[:, :-1, :].float()
            # Direct delegation retains the accepted original/serial forward
            # routes and saved tensors. Inside a custom forward this view has
            # no separate SliceBackward/TransposeBackward graph edge.
            return _scheduled_autograd_function().forward(
                ctx, shifted_logits.transpose(1, 2), targets, ignore_index)

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx: Any, row_scales: Any) -> tuple[Any, None, None]:
            saved = ctx.saved_tensors
            values, targets = saved[:2]
            row_scales = row_scales.contiguous()
            output = torch.empty(ctx.full_shape, dtype=torch.float32, device=values.device)
            reconstruct = ctx.route == "serial_token_blocks"
            # Unused pointers in the declared short route are never loaded:
            # its saved values already contain original log-probabilities.
            maxima, log_sums = saved[2:] if reconstruct else (row_scales, row_scales)
            kernel, zero_last_row = _load_gradient_kernels()
            with torch.cuda.device(values.device):
                kernel[(values.shape[0],
                        (values.shape[1] + CLASS_TILE_SIZE - 1) // CLASS_TILE_SIZE,
                        (values.shape[2] + TIME_TILE_SIZE - 1) // TIME_TILE_SIZE)](
                    values, targets, maxima, log_sums, row_scales, output,
                    values.shape[2], ctx.full_shape[1], values.shape[1],
                    ctx.ignore_index, reconstruct,
                    class_tile=CLASS_TILE_SIZE, time_tile=TIME_TILE_SIZE,
                    num_warps=4, enable_fp_fusion=True,
                )
                zero_last_row[(values.shape[0], (values.shape[1] + 1023) // 1024)](
                    output, ctx.full_shape[1], values.shape[1],
                    block_size=1024, num_warps=4, enable_fp_fusion=False,
                )
            return output, None, None

    return CausalGradientCrossEntropy


def _validate_inputs(logits: Any, targets: Any, ignore_index: int) -> None:
    import torch

    if (logits.ndim != 3 or targets.ndim != 2 or logits.shape[1] < 2
            or targets.shape != (logits.shape[0], logits.shape[1] - 1)):
        raise ValueError("Causal gradient CE requires full logits [batch,time,vocabulary] and shifted targets [batch,time-1]")
    # Validation creates no extra differentiable slice edge or tensor copy.
    with torch.no_grad():
        _validate_spatial_inputs(logits[:, :-1, :].transpose(1, 2), targets, ignore_index)


def causal_gradient_token_cross_entropy(logits: Any, targets: Any, *, ignore_index: int = -100) -> Any:
    """Return [B,T-1] CE and a full contiguous first-order causal gradient.

    Targets already carry the caller's original one-token shift. Only logits
    shift internally; the unused final logit row receives exactly zero gradient.
    Original per-example loss reduction remains the caller's responsibility.
    """
    _validate_inputs(logits, targets, ignore_index)
    return _autograd_function().apply(logits, targets, ignore_index)
