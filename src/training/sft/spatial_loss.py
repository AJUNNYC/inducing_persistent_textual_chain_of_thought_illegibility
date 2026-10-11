"""Original FP32 spatial CE forward with an explicit tiled-backward candidate.

For integer labels without class weights or smoothing, pinned PyTorch 2.9.0
``aten/src/ATen/native/LossNLL.cpp`` defines cross-entropy as log-softmax followed
by NLL. This candidate calls those original operations on the same [B,V,T]
layout; it does not replace the forward vocabulary accumulation. Backward uses
the original ``LogSoftMaxBackwardEpilogue`` expression from
``aten/src/ATen/native/cuda/SoftMax.cu``. The NLL upstream gradient has only one
nonzero class, so its vocabulary sum is known without a dense zero-filled
intermediate or serial vocabulary loop.

References (BSD-3-Clause; notice below):
https://github.com/pytorch/pytorch/blob/v2.9.0/aten/src/ATen/native/LossNLL.cpp
https://github.com/pytorch/pytorch/blob/v2.9.0/aten/src/ATen/native/cuda/SoftMax.cu
Triton 3.5 CUDA libdevice.exp dispatches FP32 to __nv_expf, rather than the
approximate exp2-based tl.exp. Floating-point contraction remains enabled as
in the original CUDA expression. Actual gradient parity and speed still need
the fixed native comparison; no equality or production approval is assumed.
"""

# Reference-project notice (PyTorch v2.9.0, BSD-3-Clause):
# From PyTorch:
# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
# From Caffe2: Copyright (c) 2016-present, Facebook Inc. All rights reserved.
# All contributions by Facebook: Copyright (c) 2016 Facebook Inc.
# All contributions by Google: Copyright (c) 2015 Google Inc. All rights reserved.
# All contributions by Yangqing Jia: Copyright (c) 2015 Yangqing Jia. All rights reserved.
# All contributions by Kakao Brain: Copyright 2019-2020 Kakao Brain.
# All contributions by Cruise LLC: Copyright (c) 2022 Cruise LLC. All rights reserved.
# All contributions by Tri Dao: Copyright (c) 2024 Tri Dao. All rights reserved.
# All contributions by Arm: Copyright (c) 2021, 2023-2024 Arm Limited and/or its affiliates.
# All contributions from Caffe: Copyright(c) 2013, 2014, 2015, the respective contributors.
# All rights reserved.
# All other contributions: Copyright(c) 2015, 2016 the respective contributors.
# All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories
#    America and IDIAP Research Institute nor the names of its contributors may
#    be used to endorse or promote products derived from this software without
#    specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
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


class SpatialLossUnsupported(RuntimeError):
    """The selected backward candidate cannot silently use another backend."""


@lru_cache(maxsize=1)
def _load_backward_kernel() -> Any:
    global triton, tl, libdevice
    try:
        import triton
        import triton.language as tl
        from triton.language.extra.cuda import libdevice
    except ImportError as error:
        raise SpatialLossUnsupported("Spatial CE backward requires pinned Triton 3.5.0") from error
    if metadata.version("triton") != "3.5.0":
        raise SpatialLossUnsupported("Spatial CE backward requires exactly Triton 3.5.0")

    @triton.jit(do_not_specialize=["n_times", "n_elements"])
    def backward(
        log_probabilities, targets, row_scales, output,
        n_times, n_elements, n_columns: tl.constexpr,
        ignore_index: tl.constexpr, block_size: tl.constexpr,
    ):
        offsets = (tl.program_id(0).to(tl.int64) * block_size
                   + tl.arange(0, block_size))
        present = offsets < n_elements
        time = offsets % n_times
        class_index = (offsets // n_times) % n_columns
        batch = offsets // (n_times * n_columns)
        token = batch * n_times + time
        target = tl.load(targets + token, present, other=ignore_index)
        row_scale = tl.load(row_scales + token, present, other=0.0)
        # Original unweighted NLL backward has one -grad_loss entry, and its
        # vocabulary sum is exactly that entry (or +0 for ignored targets).
        summed_gradient = tl.where(target == ignore_index, 0.0, -row_scale)
        target_gradient = tl.where(class_index == target, summed_gradient, 0.0)
        log_probability = tl.load(log_probabilities + offsets, present, other=0.0)
        derivative = target_gradient - libdevice.exp(log_probability) * summed_gradient
        # Preserve 0 - exp(NaN) * 0 on ignored nonfinite rows: clipping must
        # reject the same malformed update, rather than hiding its gradients.
        tl.store(output + offsets, derivative, present)

    return backward


@lru_cache(maxsize=1)
def _autograd_function() -> Any:
    import torch
    import torch.nn.functional as functional

    class SpatialCrossEntropy(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, logits: Any, targets: Any, ignore_index: int) -> Any:
            _load_backward_kernel()
            log_probabilities = torch.log_softmax(logits, dim=1, dtype=logits.dtype)
            losses = functional.nll_loss(log_probabilities, targets,
                                         ignore_index=ignore_index, reduction="none")
            ctx.save_for_backward(log_probabilities.contiguous(), targets.contiguous())
            ctx.ignore_index = ignore_index
            return losses

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx: Any, row_scales: Any) -> tuple[Any, None, None]:
            kernel = _load_backward_kernel()
            log_probabilities, targets = ctx.saved_tensors
            output = torch.empty_like(log_probabilities)
            row_scales = row_scales.contiguous()
            with torch.cuda.device(log_probabilities.device):
                kernel[((log_probabilities.numel() + 1023) // 1024,)](
                    log_probabilities, targets, row_scales, output,
                    log_probabilities.shape[2], log_probabilities.numel(),
                    log_probabilities.shape[1], ctx.ignore_index,
                    block_size=1024, num_warps=4, enable_fp_fusion=True,
                )
            return output, None, None

    return SpatialCrossEntropy


def _validate_inputs(logits: Any, targets: Any, ignore_index: int) -> None:
    import torch

    if (logits.ndim != 3 or targets.ndim != 2
            or targets.shape != (logits.shape[0], logits.shape[2])):
        raise ValueError("Spatial CE requires logits [batch,vocabulary,time] and targets [batch,time]")
    if not all(logits.shape):
        raise ValueError("Spatial CE requires nonempty batch, vocabulary and time")
    if logits.dtype != torch.float32 or targets.dtype != torch.long:
        raise ValueError("Spatial CE retains FP32 logits and int64 targets")
    if targets.requires_grad or type(ignore_index) is not int or ignore_index != -100:
        raise ValueError("Spatial CE retains integer targets and ignore_index=-100")
    if logits.device != targets.device:
        raise ValueError("Spatial CE inputs must share one CUDA device")
    if logits.device.type != "cuda":
        raise SpatialLossUnsupported("Spatial CE backward is an explicit CUDA candidate; CPU fallback is forbidden")
    if torch.cuda.get_device_capability(logits.device) != (12, 0):
        raise SpatialLossUnsupported("Spatial CE backward is scoped to the RTX PRO 6000 SM120 comparison")
    if torch.any((targets != ignore_index) & ((targets < 0) | (targets >= logits.shape[1]))):
        raise ValueError("Spatial CE target lies outside the unchanged vocabulary")


def spatial_token_cross_entropy(logits: Any, targets: Any, *, ignore_index: int = -100) -> Any:
    """Return original unreduced spatial CE with opt-in tiled first backward.

    Inputs, dtype, layout and forward operation order remain unchanged. Targets
    are integer class labels; class weights and smoothing are deliberately not
    supported. The caller retains its original per-example loss reduction.
    """
    _validate_inputs(logits, targets, ignore_index)
    return _autograd_function().apply(logits, targets, ignore_index)
