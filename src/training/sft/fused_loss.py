"""Opt-in FP32 token CE candidate using the already pinned Triton runtime.

This module changes only the numerical CE implementation. It does not change
causal shifting, masks, per-example weighting, model precision, or randomness.
The caller retains those operations and the existing nonfinite-update checks.
Production use requires an actual paired GPU comparison; importing this module
does not approve it or import Triton on a CPU host.

Algorithmic precedent: stable online log-sum-exp and CE-gradient materialization
in Liger-Kernel v0.4.0, ``src/liger_kernel/ops/cross_entropy.py``:
https://github.com/linkedin/Liger-Kernel/blob/v0.4.0/src/liger_kernel/ops/cross_entropy.py
The independently written implementation below retains FP32 logits, separate
gradient storage and the caller's unreduced token-loss reduction. It does not
vendor Liger or introduce a dependency. Online normalization changes operation
order and may change floating-point results; equality is not assumed.
"""

# Reference-project notice (Liger-Kernel v0.4.0, BSD-2-Clause):
# Copyright 2024 LinkedIn Corporation. All Rights Reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from functools import lru_cache
from importlib import metadata
from typing import Any


class FusedLossUnsupported(RuntimeError):
    """The explicitly selected kernel cannot run under its measured contract."""


SUPPORTED_COMPUTE_CAPABILITIES = {(12, 0)}
PINNED_TRITON_VERSION = "3.5.0"


@lru_cache(maxsize=1)
def _load_kernels() -> tuple[Any, Any]:
    # Triton's source compiler resolves language symbols in function globals.
    # Keep the optional imports lazy, but make that binding explicit.
    global triton, tl
    try:
        import triton
        import triton.language as tl
    except ImportError as error:
        raise FusedLossUnsupported("FP32 fused CE requires pinned Triton 3.5.0") from error
    if metadata.version("triton") != PINNED_TRITON_VERSION:
        raise FusedLossUnsupported("FP32 fused CE requires exactly Triton 3.5.0")

    @triton.jit
    def token_ce_forward(
        logits, targets, losses, gradients,
        n_columns: tl.constexpr, ignore_index: tl.constexpr, block_size: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        row_offset = row * n_columns
        target = tl.load(targets + row)
        offsets = tl.arange(0, block_size)
        maximum = tl.full((), float("-inf"), tl.float32)
        exponential_sum = tl.full((), 0.0, tl.float32)
        has_nonfinite = tl.full((), False, tl.int1)
        for start in range(0, n_columns, block_size):
            columns = start + offsets
            values = tl.load(logits + row_offset + columns, columns < n_columns,
                             other=float("-inf"))
            invalid = (columns < n_columns) & ((values != values) | (tl.abs(values) == float("inf")))
            has_nonfinite = has_nonfinite | (tl.sum(invalid.to(tl.int32), 0) > 0)
            next_maximum = tl.maximum(maximum, tl.max(values, 0))
            exponential_sum = (exponential_sum * tl.exp(maximum - next_maximum)
                               + tl.sum(tl.exp(values - next_maximum), 0))
            maximum = next_maximum

        if target == ignore_index:
            tl.store(losses + row, 0.0)
        else:
            target_value = tl.load(logits + row_offset + target)
            tl.store(losses + row, -(target_value - maximum - tl.log(exponential_sum)))

        for start in range(0, n_columns, block_size):
            columns = start + offsets
            if target == ignore_index:
                # PyTorch's ignored-row log-softmax backward still rejects
                # nonfinite logits through nonfinite gradients. Preserve that
                # update rejection rather than hiding malformed masked rows.
                derivative = tl.full((block_size,), 0.0, tl.float32)
                derivative = tl.where(has_nonfinite, float("nan"), derivative)
            else:
                values = tl.load(logits + row_offset + columns, columns < n_columns,
                                 other=float("-inf"))
                derivative = tl.exp(values - maximum) / exponential_sum
                derivative = derivative - (columns == target).to(tl.float32)
            tl.store(gradients + row_offset + columns, derivative, columns < n_columns)

    @triton.jit
    def token_ce_backward(
        gradients, row_scales, output,
        n_columns: tl.constexpr, block_size: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        columns = tl.program_id(1) * block_size + tl.arange(0, block_size)
        offsets = row * n_columns + columns
        derivative = tl.load(gradients + offsets, columns < n_columns, other=0.0)
        scale = tl.load(row_scales + row)
        tl.store(output + offsets, derivative * scale, columns < n_columns)

    return token_ce_forward, token_ce_backward


@lru_cache(maxsize=1)
def _autograd_function() -> Any:
    import torch

    class TokenCrossEntropy(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, logits: Any, targets: Any, ignore_index: int) -> Any:
            forward, _ = _load_kernels()
            losses = torch.empty(logits.shape[0], dtype=torch.float32, device=logits.device)
            gradients = torch.empty_like(logits)
            with torch.cuda.device(logits.device):
                forward[(logits.shape[0],)](
                    logits, targets, losses, gradients, logits.shape[1], ignore_index,
                    block_size=32768, num_warps=32, enable_fp_fusion=False,
                )
            ctx.save_for_backward(gradients)
            return losses

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx: Any, row_scales: Any) -> tuple[Any, None, None]:
            _, backward = _load_kernels()
            gradients, = ctx.saved_tensors
            output = torch.empty_like(gradients)
            row_scales = row_scales.contiguous()
            with torch.cuda.device(gradients.device):
                backward[(gradients.shape[0], (gradients.shape[1] + 8191) // 8192)](
                    gradients, row_scales, output, gradients.shape[1],
                    block_size=8192, num_warps=8, enable_fp_fusion=False,
                )
            return output, None, None

    return TokenCrossEntropy


def _validate_inputs(logits: Any, targets: Any, ignore_index: int) -> None:
    import torch

    if logits.ndim != 2 or targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError("Fused token CE requires logits [rows,vocabulary] and targets [rows]")
    if not logits.shape[0] or not logits.shape[1]:
        raise ValueError("Fused token CE requires nonempty rows and vocabulary")
    if logits.dtype != torch.float32 or targets.dtype != torch.long:
        raise ValueError("Fused token CE retains FP32 logits and int64 targets")
    if targets.requires_grad or type(ignore_index) is not int or ignore_index != -100:
        raise ValueError("Fused token CE retains integer targets and ignore_index=-100")
    if logits.device != targets.device:
        raise ValueError("Fused token CE inputs must share one CUDA device")
    if logits.device.type != "cuda":
        raise FusedLossUnsupported("FP32 fused CE is an explicit CUDA candidate; CPU fallback is forbidden")
    if torch.cuda.get_device_capability(logits.device) not in SUPPORTED_COMPUTE_CAPABILITIES:
        raise FusedLossUnsupported("FP32 fused CE is scoped to the RTX PRO 6000 SM120 comparison")
    if torch.any((targets != ignore_index) & ((targets < 0) | (targets >= logits.shape[1]))):
        raise ValueError("Fused token CE target lies outside the unchanged vocabulary")


def fused_token_cross_entropy(logits: Any, targets: Any, *, ignore_index: int = -100) -> Any:
    """Return unreduced FP32 token CE; never mutate logits or consume RNG.

    This first-order training candidate fails closed on unmeasured architecture,
    dtype, or dependency versions. The caller owns causal alignment and original
    per-example averaging. No vocabulary classes or context tokens are omitted.
    """
    _validate_inputs(logits, targets, ignore_index)
    return _autograd_function().apply(logits.contiguous(), targets.contiguous(), ignore_index)
