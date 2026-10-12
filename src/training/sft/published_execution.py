"""Published-method execution for the declared efficient one-epoch SFT recipe.

This is a new scientific recipe, not an approved migration of historical NF4
checkpoints. It combines ordinary BF16 LoRA and SDPA with upstream unfiltered
Cut Cross-Entropy. No vocabulary logits are materialized during supervised
training. The completion mask and per-example loss reduction are unchanged;
the old explicit FP32 classifier arithmetic is deliberately not claimed equal.

CUDA dependencies remain lazy so that repository tools can inspect the policy
without importing or allocating a training runtime.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import random
from typing import Any, Mapping


CCE_COMMIT = "3de376c106a1916bc5e1b619f9c77c87a461ee1c"
CCE_REPOSITORY = "https://github.com/apple/ml-cross-entropy.git"
PUBLISHED_BUDGET = "efficient_one_epoch_v1"
SUPPLEMENTAL_LOCK_SHA256 = "0260a293515f8aca6252e56685ae4df92153ae7d2c145255c86369b22d6304c9"
PUBLISHED_EXECUTION_POLICY = {
    "policy": "published_bf16_lora_full_gradient_cce_v2",
    "training_budget": PUBLISHED_BUDGET,
    "requested_physical_microbatch": 2,
    "requested_gradient_accumulation": 8,
    "fallback_physical_microbatch": 1,
    "fallback_gradient_accumulation": 16,
    "effective_batch_size": 16,
    "physical_batch_capacity_policy": "duplicate_longest_example_non_updating_backward",
    "physical_batch_selection_order": [
        {"physical_microbatch": 2, "gradient_checkpointing": False},
        {"physical_microbatch": 1, "gradient_checkpointing": False},
        {"physical_microbatch": 2, "gradient_checkpointing": True},
        {"physical_microbatch": 1, "gradient_checkpointing": True},
    ],
    "physical_batch_adaptation": "startup_only_frozen_for_fit_and_resume",
    "base_weights_dtype": "bfloat16",
    "base_quantization": "none",
    "adapter_parameters_dtype": "float32",
    "classifier_weights_dtype": "bfloat16",
    "classifier_dot_accumulator_dtype": "float32",
    "classifier_logits_rounding_dtype": "bfloat16",
    "loss_dtype": "float32",
    "optimizer_state_dtype": "float32",
    "attention_implementation": "sdpa",
    "sdpa_backend_policy": "prefer_flash_then_efficient_allow_math_fallback",
    "sdpa_actual_backend": "not_instrumented",
    "gradient_checkpointing_policy": "off_unless_full_single_example_cuda_oom",
    "gradient_checkpointing_use_reentrant": False,
    "gradient_checkpointing_preserve_rng_state": True,
    "loss_implementation": "upstream_cce_exact_full_gradient",
    "loss_reduction": "mean_supervised_tokens_per_example_then_mean_examples",
    "cce_repository": CCE_REPOSITORY,
    "cce_commit": CCE_COMMIT,
    "supplemental_dependency_lock": "requirements_sft_published.lock",
    "supplemental_dependency_lock_sha256": SUPPLEMENTAL_LOCK_SHA256,
    "cce_options": {
        "bias": None, "ignore_index": -100, "softcap": None,
        "reduction": "none", "shift": 1, "filter_eps": None,
        "accum_e_fp32": True, "accum_c_fp32": True,
        "filter_e_grad": False, "filter_c_grad": False, "impl": "cce_exact",
    },
    "cce_autotune": False,
    "approximate_gradient_filtering": False,
    "global_tf32": False,
    "historical_fp32_head_equivalence": False,
    "native_speedup_measured": False,
}


def _blocked(message: str) -> Exception:
    from training.sft.individual_fit.sft_fit import SFTRuntimeBlocked
    return SFTRuntimeBlocked(message)


def _verify_cce_installation() -> tuple[Any, dict[str, Any]]:
    """Require the exact upstream commit and its full-gradient API surface."""
    if os.environ.get("CCE_AUTOTUNE", "0") != "0":
        raise _blocked("The finalized CCE policy requires CCE_AUTOTUNE=0")
    os.environ.setdefault("CCE_AUTOTUNE", "0")
    try:
        distribution = importlib.metadata.distribution("cut-cross-entropy")
        direct_url_text = distribution.read_text("direct_url.json")
        direct_url = json.loads(direct_url_text or "null")
        from cut_cross_entropy import linear_cross_entropy
        from cut_cross_entropy.cce_utils import LinearCrossEntropyImpl
    except (ImportError, importlib.metadata.PackageNotFoundError, ValueError) as error:
        raise _blocked("Install Cut Cross-Entropy from the pinned upstream Git revision") from error
    if (not isinstance(direct_url, dict)
            or direct_url.get("url") != CCE_REPOSITORY
            or direct_url.get("vcs_info", {}).get("vcs") != "git"
            or direct_url.get("vcs_info", {}).get("commit_id") != CCE_COMMIT):
        raise _blocked("Installed Cut Cross-Entropy does not prove the declared upstream commit")
    required = {"e", "c", "targets", *PUBLISHED_EXECUTION_POLICY["cce_options"]}
    if (not required <= set(inspect.signature(linear_cross_entropy).parameters)
            or not hasattr(LinearCrossEntropyImpl, "CCE_EXACT")):
        raise _blocked("Installed CCE lacks the required exact full-gradient API")
    return linear_cross_entropy, {
        "cce_distribution_version": distribution.version,
        "cce_installed_commit": direct_url["vcs_info"]["commit_id"],
        "cce_direct_api_signature_verified": True,
    }


def build_runtime(spec: Any, device: str, *, cache_root: str | Path | None = None,
                  train_count: int = 10000) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    """Load pinned BF16 base weights and attach the unchanged FP32 LoRA adapters."""
    if spec.training_budget != PUBLISHED_BUDGET or train_count != 10000:
        raise _blocked("Published execution requires the declared 10,000-example one-epoch budget")
    recipe = spec.recipe
    if (recipe.microbatch_size, recipe.gradient_accumulation_steps,
            recipe.effective_batch_size) != (2, 8, 16):
        raise _blocked("Published execution requires requested physical batch 2 x 8, effective batch 16")
    if (recipe.optimizer_steps(train_count), recipe.warmup_steps(train_count)) != (625, 19):
        raise _blocked("Published execution schedule differs from the declared 625 updates/19 warmup")
    try:
        import numpy as np
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
    except ImportError as error:
        raise _blocked("Install the pinned published-method CUDA training dependencies") from error
    selected_device = torch.device(device)
    if (selected_device.type != "cuda" or not torch.cuda.is_available()
            or not torch.cuda.is_bf16_supported()
            or torch.cuda.get_device_capability(selected_device)[0] < 8):
        raise _blocked("Published BF16/CCE execution requires an Ampere-or-newer CUDA GPU")
    packages = {name: importlib.metadata.version(name)
                for name in ("torch", "triton", "transformers", "peft", "numpy")}
    if (packages["torch"].split("+")[0] != "2.9.0"
            or packages["triton"] != "3.5.0"
            or packages["transformers"] != "4.57.3"
            or packages["peft"] != "0.18.0"):
        raise _blocked("Published execution requires Torch 2.9.0/Triton 3.5.0/Transformers 4.57.3/PEFT 0.18.0")
    cce_function, cce_evidence = _verify_cce_installation()
    lock_path = Path(__file__).resolve().parents[3] / "requirements_sft_published.lock"
    if (not lock_path.is_file()
            or hashlib.sha256(lock_path.read_bytes()).hexdigest() != SUPPLEMENTAL_LOCK_SHA256):
        raise _blocked("Published execution requires the unchanged supplemental dependency lock")
    random.seed(recipe.seed)
    np.random.seed(recipe.seed)
    torch.manual_seed(recipe.seed)
    torch.cuda.manual_seed_all(recipe.seed)
    # Only the declared BF16 tensor-core arithmetic is enabled. In particular,
    # no global FP32->TF32 approximation is introduced by this execution path.
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from operations.assets import _cache_root, model_load_kwargs
    verification_receipts = _cache_root(cache_root) / "private_verification_receipts"
    tokenizer = AutoTokenizer.from_pretrained(
        spec.model_name, revision=spec.tokenizer_revision, use_fast=True, trust_remote_code=False,
        **model_load_kwargs(spec.model_name, spec.tokenizer_revision, cache_root,
                            tokenizer_only=True, verification_receipts=verification_receipts))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_name, revision=spec.model_revision, trust_remote_code=False,
        **model_load_kwargs(spec.model_name, spec.model_revision, cache_root,
                            verification_receipts=verification_receipts),
        torch_dtype=torch.bfloat16, device_map={"": device}, attn_implementation="sdpa")
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model = get_peft_model(model, LoraConfig(
        r=recipe.lora_rank, lora_alpha=recipe.lora_alpha, lora_dropout=recipe.lora_dropout,
        target_modules=list(recipe.target_modules), bias="none", task_type="CAUSAL_LM"),
        autocast_adapter_dtype=True)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if "lora_" not in name:
                raise _blocked("Published execution found a trainable non-LoRA parameter")
            parameter.data = parameter.data.to(dtype=torch.float32)
        elif parameter.is_floating_point() and parameter.dtype != torch.bfloat16:
            raise _blocked("Published execution found frozen weights outside BF16")
    classifier = model.get_output_embeddings()
    if (classifier.weight.dtype != torch.bfloat16 or classifier.weight.requires_grad
            or getattr(classifier, "bias", None) is not None):
        raise _blocked("Published CCE requires the frozen bias-free BF16 classifier")
    model.gradient_checkpointing_disable()
    model.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=recipe.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=recipe.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=19, num_training_steps=625)
    runtime = {
        "published_execution_policy": json.loads(json.dumps(PUBLISHED_EXECUTION_POLICY)),
        "execution_profile": None,
        "asset_integrity_policy": "full_sha256_then_private_manifest_and_file_identity_receipts",
        "asset_verification_receipt_root": str(verification_receipts),
        "device": str(selected_device), "gpu_name": torch.cuda.get_device_name(selected_device),
        "vram_bytes": torch.cuda.get_device_properties(selected_device).total_memory,
        "cuda_version": torch.version.cuda,
        "compute_capability": list(torch.cuda.get_device_capability(selected_device)),
        "attention_implementation": "sdpa",
        "sdpa_backend_policy": PUBLISHED_EXECUTION_POLICY["sdpa_backend_policy"],
        "sdpa_actual_backend": "not_instrumented",
        "gradient_checkpointing": False, "use_reentrant": False,
        "requested_physical_microbatch": 2,
        "requested_gradient_accumulation": 8,
        "physical_microbatch": None, "gradient_accumulation": None,
        "effective_batch_size": 16,
        "physical_batch_selection_status": "PENDING_CAPACITY_SETUP",
        "loss_implementation": "upstream_cce_exact_full_gradient",
        "optimizer_betas": [0.9, 0.999], "optimizer_epsilon": 1e-8,
        "optimizer_state_dtype": "float32", "warmup_steps": 19,
        "training_budget": spec.training_budget, "total_optimizer_steps": 625,
        "packages": packages, **cce_evidence,
        "loss_helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "supplemental_dependency_lock_sha256": SUPPLEMENTAL_LOCK_SHA256,
    }
    model._sft_published_execution = True
    model._sft_published_cce = cce_function
    model._sft_published_runtime = runtime
    return model, tokenizer, optimizer, scheduler, runtime


def _published_forward(model: Any, batch: Mapping[str, Any], *, retain_hidden_grad: bool) -> tuple[Any, Any]:
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    if not getattr(model, "_sft_published_execution", False):
        raise _blocked("Published loss requires a runtime built under the declared published policy")
    labels = batch["labels"]
    if labels.ndim != 2 or labels.shape[1] < 2 or labels.dtype != torch.long:
        raise _blocked("Published loss requires full two-dimensional int64 completion labels")
    counts = (labels[:, 1:] != -100).sum(dim=1)
    if torch.any(counts == 0):
        raise _blocked("An example has no supervised shifted completion target")
    base = model.get_base_model()
    # Accessing the decoder directly retains every attached PEFT projection,
    # but avoids QwenForCausalLM's materialized B x T x vocabulary projection.
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16), sdpa_kernel(
            [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH],
            set_priority=True):
        hidden = base.model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            use_cache=False, return_dict=True).last_hidden_state
    classifier = model.get_output_embeddings()
    if hidden.dtype != torch.bfloat16 or classifier.weight.dtype != torch.bfloat16:
        raise _blocked("Published CCE requires BF16 decoder states and classifier weights")
    if retain_hidden_grad and hidden.requires_grad:
        hidden.retain_grad()
    token_losses = model._sft_published_cce(
        hidden, classifier.weight, labels, **PUBLISHED_EXECUTION_POLICY["cce_options"])
    if token_losses.shape != labels[:, 1:].shape or token_losses.dtype != torch.float32:
        raise _blocked("Upstream CCE returned an unexpected shifted loss layout or dtype")
    return (token_losses.sum(dim=1) / counts).mean(), hidden


def published_loss_with_hidden(model: Any, batch: Mapping[str, Any]) -> tuple[Any, Any]:
    """Return scalar FP32 loss and full retained decoder states for mask proof."""
    return _published_forward(model, batch, retain_hidden_grad=True)


def published_loss(model: Any, batch: Mapping[str, Any]) -> Any:
    """Supervised training/evaluation without materializing vocabulary logits."""
    return _published_forward(model, batch, retain_hidden_grad=False)[0]


def capacity_setup(model: Any, longest_example: Mapping[str, Any], pad_token_id: int,
                   *, optimizer: Any = None, fixed_microbatch_size: int | None = None,
                   fixed_gradient_checkpointing: bool | None = None) -> dict[str, Any]:
    """Select physical batching with a non-updating worst-pair capacity backward.

    This is an ordinary capacity/mask setup operation, not timing, a benchmark,
    or candidate-profile optimization. Two copies of the longest example are a
    conservative bound for a padded pair. Prefer 2 x 8, then 1 x 16 without
    checkpointing, then the same order with checkpointing. A resumed fit tests
    only its frozen physical batch and checkpointing mode. RNG is restored and
    gradients are cleared; no optimizer or scheduler step occurs and no examples
    or targets are cut.
    """
    import torch
    from training.sft.individual_fit.sft_fit import capture_rng, collate_completion, restore_rng
    if not getattr(model, "_sft_published_execution", False):
        raise _blocked("Capacity setup requires the declared published execution runtime")
    if any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad):
        raise _blocked("Capacity setup requires an optimizer-update boundary with cleared gradients")
    fixed = fixed_microbatch_size is not None or fixed_gradient_checkpointing is not None
    if fixed and (type(fixed_microbatch_size) is not int
                  or fixed_microbatch_size not in (1, 2)
                  or type(fixed_gradient_checkpointing) is not bool):
        raise _blocked("Resumed capacity setup requires both saved physical batch and checkpointing mode")
    before_rng, before_training = capture_rng(), model.training
    device = next(model.parameters()).device
    # Adam initializes two FP32 moments lazily at its first step. Reserve their
    # exact tensor storage during capacity setup so a successful pre-update
    # backward does not overlook the next step's persistent optimizer memory.
    optimizer_moment_bytes = 0
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        state = optimizer.state.get(parameter, {}) if optimizer is not None else {}
        for moment in ("exp_avg", "exp_avg_sq"):
            existing = state.get(moment)
            if existing is None:
                optimizer_moment_bytes += parameter.numel() * parameter.element_size()
            elif existing.shape != parameter.shape or existing.dtype != torch.float32:
                raise _blocked("Restored Adam moments differ from FP32 adapter shape or precision")

    def backward_fits(physical_microbatch: int) -> bool:
        try:
            optimizer_moment_reservation = torch.empty(
                optimizer_moment_bytes, device=device, dtype=torch.uint8)
            batch = collate_completion([longest_example] * physical_microbatch, pad_token_id, device)
            loss, hidden = published_loss_with_hidden(model, batch)
            if not torch.isfinite(loss):
                raise _blocked("Worst-pair completion loss is nonfinite")
            loss.backward()
            # The causal loss is shifted by one token; prompt inputs remain
            # context, but unsupervised final-decoder rows must contribute no
            # direct loss gradient in every selected physical-batch row.
            mask = torch.zeros_like(batch["labels"], dtype=torch.bool)
            mask[:, :-1] = batch["labels"][:, 1:] != -100
            if (hidden.grad is None or not torch.isfinite(hidden.grad).all()
                    or torch.count_nonzero(hidden.grad[~mask]).item() != 0):
                raise _blocked("Worst-pair capacity loss has invalid masked decoder-row gradients")
            nonzero_adapter_gradient = False
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                if "lora_" not in name:
                    raise _blocked("Worst-pair capacity found a trainable non-LoRA parameter")
                if parameter.grad is not None:
                    if not torch.isfinite(parameter.grad).all():
                        raise _blocked("Worst-pair capacity has a nonfinite LoRA gradient")
                    if parameter.grad.count_nonzero().item() > 0:
                        nonzero_adapter_gradient = True
            if not nonzero_adapter_gradient:
                raise _blocked("Worst-pair capacity produced no nonzero LoRA gradient")
            torch.cuda.synchronize(device)
            # Keep the reservation alive through every backward kernel.
            del optimizer_moment_reservation
            return True
        except torch.cuda.OutOfMemoryError:
            return False
        finally:
            model.zero_grad(set_to_none=True)

    try:
        model.train()
        modes = ([(fixed_microbatch_size, fixed_gradient_checkpointing)] if fixed else
                 [(2, False), (1, False), (2, True), (1, True)])
        attempts = []
        selected = None
        for physical_microbatch, checkpointing in modes:
            restore_rng(before_rng)
            if checkpointing:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={
                    "use_reentrant": False, "preserve_rng_state": True})
            else:
                model.gradient_checkpointing_disable()
            fits = backward_fits(physical_microbatch)
            attempts.append({
                "physical_microbatch": physical_microbatch,
                "gradient_accumulation": 16 // physical_microbatch,
                "gradient_checkpointing": checkpointing,
                "status": "passed" if fits else "cuda_out_of_memory",
            })
            if fits:
                selected = physical_microbatch
                break
            # Release a failed graph before the next complete capacity attempt.
            # Never switch physical batching during an optimizer update.
            gc.collect()
            torch.cuda.empty_cache()
        if selected is None:
            message = ("Saved physical batch/checkpointing mode does not fit on this GPU; "
                       "resume requires the same execution mode" if fixed else
                       "Full longest example does not fit; use a GPU with more VRAM")
            raise _blocked(message)
        result = {
            "status": "passed", "operation": "non_updating_worst_pair_backward",
            "longest_example_tokens": len(longest_example["input_ids"]),
            "capacity_bound": "duplicated_longest_example_with_padding",
            "physical_microbatch": selected,
            "gradient_accumulation": 16 // selected,
            "effective_batch_size": 16,
            "selection_frozen_for_resume": True,
            "resumed_saved_mode_only": fixed,
            "capacity_attempts": attempts,
            "gradient_checkpointing": bool(model.is_gradient_checkpointing),
            "use_reentrant": False, "preserve_rng_state": True,
            "optimizer_updates": 0, "rng_restored": True,
            "masked_decoder_rows": "zero_direct_loss_gradient",
            "nonzero_lora_gradient": True,
            "optimizer_moment_reservation_bytes": optimizer_moment_bytes,
            "sequence_truncation": False,
        }
        model._sft_published_runtime.update({
            "gradient_checkpointing": result["gradient_checkpointing"],
            "physical_microbatch": selected, "gradient_accumulation": 16 // selected,
            "physical_batch_selection_status": "FROZEN", "capacity_setup": result})
        return result
    finally:
        model.zero_grad(set_to_none=True)
        model.train(before_training)
        restore_rng(before_rng)
