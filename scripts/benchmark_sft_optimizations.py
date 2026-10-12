#!/usr/bin/env python3
"""Measure isolated SFT execution profiles without touching an official fit.

Each profile starts from the same pinned base and fresh seed, uses the exact
first whole optimizer batches of the frozen exposure order, and calls the
production optimizer-update helper. A small numerical comparison is a technical
screen, not evidence that full training outcomes are equivalent. Plan-only mode
does not import CUDA dependencies, load models, or approve any optimization.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from experiment_overview.experiment_overview import code_revision
from run_certified_sft import certified_tasks
from shared_procedures.configuration import configuration_hash, validate_configuration
from shared_procedures.identity import content_hash, file_hash
from shared_procedures.storage import read_json
from training.sft.individual_fit import sft_fit
from training.sft.sft_training import load_dataset_bundle

SCHEMA = "isolated_sft_execution_benchmark_v2"
PROFILE_SETTINGS = {
    "baseline": {"attention_implementation": "eager", "validated_loss_counts": False,
                 "defer_finite_checks": False},
    "deferred_checks": {"attention_implementation": "eager", "validated_loss_counts": True,
                        "defer_finite_checks": True},
    "sdpa": {"attention_implementation": "sdpa", "validated_loss_counts": False,
             "defer_finite_checks": False},
    "combined_sdpa": {"attention_implementation": "sdpa", "validated_loss_counts": True,
                      "defer_finite_checks": True},
    "cuda_input_cache": {"attention_implementation": "eager", "validated_loss_counts": False,
                         "defer_finite_checks": False, "cache_collated_inputs": True,
                         "gradient_checkpointing": True},
    "no_gradient_checkpointing": {"attention_implementation": "eager", "validated_loss_counts": False,
                                  "defer_finite_checks": False, "cache_collated_inputs": False,
                                  "gradient_checkpointing": False},
    "cached_no_gradient_checkpointing": {"attention_implementation": "eager", "validated_loss_counts": False,
                                         "defer_finite_checks": False, "cache_collated_inputs": True,
                                         "gradient_checkpointing": False},
    "flat_ce_no_gradient_checkpointing": {"attention_implementation": "eager", "validated_loss_counts": False,
                                          "defer_finite_checks": False, "cache_collated_inputs": False,
                                          "gradient_checkpointing": False, "loss_implementation": "flat_ce"},
    "fused_ce_no_gradient_checkpointing": {"attention_implementation": "eager", "validated_loss_counts": False,
                                           "defer_finite_checks": False, "cache_collated_inputs": False,
                                           "gradient_checkpointing": False, "loss_implementation": "fused_ce"},
    "spatial_scheduled_ce_checkpointed": {
        "attention_implementation": "eager", "validated_loss_counts": False,
        "defer_finite_checks": False, "cache_collated_inputs": False,
        "gradient_checkpointing": True, "loss_implementation": "spatial_scheduled_ce"},
    "spatial_scheduled_ce_checkpointed_sdpa": {
        "attention_implementation": "sdpa", "validated_loss_counts": False,
        "defer_finite_checks": False, "cache_collated_inputs": False,
        "gradient_checkpointing": True, "loss_implementation": "spatial_scheduled_ce"},
    "spatial_fused_backward_no_gradient_checkpointing": {
        "attention_implementation": "eager", "validated_loss_counts": False,
        "defer_finite_checks": False, "cache_collated_inputs": False,
        "gradient_checkpointing": False, "loss_implementation": "spatial_fused_backward"},
    "spatial_scheduled_ce_no_gradient_checkpointing": {
        "attention_implementation": "eager", "validated_loss_counts": False,
        "defer_finite_checks": False, "cache_collated_inputs": False,
        "gradient_checkpointing": False, "loss_implementation": "spatial_scheduled_ce"},
    "causal_contiguous_gradient_ce_no_gradient_checkpointing": {
        "attention_implementation": "eager", "validated_loss_counts": False,
        "defer_finite_checks": False, "cache_collated_inputs": False,
        "gradient_checkpointing": False, "loss_implementation": "causal_contiguous_gradient_ce"},
}
for settings in PROFILE_SETTINGS.values():
    settings.setdefault("cache_collated_inputs", False)
    settings.setdefault("gradient_checkpointing", True)
    settings.setdefault("loss_implementation", "spatial_ce")
LOSS_HELPER_FILES = {
    "spatial_fused_backward": "src/training/sft/spatial_loss.py",
    "spatial_scheduled_ce": "src/training/sft/scheduled_spatial_loss.py",
    "causal_contiguous_gradient_ce": "src/training/sft/causal_gradient_loss.py",
}


def write_json(path, value):
    """Private benchmark output; no production artifact hook or checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "certificate", "bundle", "worklist", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--model", choices=("7b", "14b"), default="7b")
    parser.add_argument("--family", choices=("rot13", "kazitext"), default="rot13")
    parser.add_argument("--level", type=int, choices=range(1, 11), default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--profiles", nargs="+", choices=tuple(PROFILE_SETTINGS),
                        default=["baseline", "cuda_input_cache", "no_gradient_checkpointing"])
    parser.add_argument("--reference-profile", choices=tuple(PROFILE_SETTINGS), default="baseline",
                        help="Declared paired reference, listed first in --profiles; baseline remains the legacy default.")
    parser.add_argument("--warmup-updates", type=int, default=1)
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--recovery-checkpoint", help="Read an immutable verified full checkpoint; never rewrite it.")
    parser.add_argument("--loss-atol", type=float, default=1e-5)
    parser.add_argument("--loss-rtol", type=float, default=1e-4)
    parser.add_argument("--gradient-relative-l2", type=float, default=1e-3)
    parser.add_argument("--update-relative-l2", type=float, default=1e-3)
    parser.add_argument("--gradient-cosine-floor", type=float, default=0.999)
    parser.add_argument("--minimum-speedup", type=float, default=1.05)
    parser.add_argument("--profile-timeout-seconds", type=int, default=3600)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--_profile", choices=tuple(PROFILE_SETTINGS), help=argparse.SUPPRESS)
    parser.add_argument("--_plan", help=argparse.SUPPRESS)
    parser.add_argument("--_diagnostics", help=argparse.SUPPRESS)
    parser.add_argument("--_pair", help=argparse.SUPPRESS)
    parser.add_argument("--_repetition", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if (args.warmup_updates < 1 or args.updates < 5 or args.repetitions < 2
            or (args.warmup_updates + args.updates) * 16 > 10000):
        parser.error("Use at least one warm-up, five measured updates and two paired repetitions within 10,000 records")
    bounds = (args.loss_atol, args.loss_rtol, args.gradient_relative_l2,
              args.update_relative_l2, args.gradient_cosine_floor, args.minimum_speedup)
    if (not all(math.isfinite(value) for value in bounds) or any(value < 0 for value in bounds[:4])
            or not 0 <= args.gradient_cosine_floor <= 1 or args.minimum_speedup < 1
            or args.profile_timeout_seconds < 1):
        parser.error("Numerical tolerances and timeout must be finite valid bounds")
    if len(args.profiles) < 2 or len(set(args.profiles)) != len(args.profiles) or args.profiles[0] != args.reference_profile:
        parser.error("Profiles must be unique and start with the declared reference profile")
    if (any(name in args.profiles[1:] for name in ("cached_no_gradient_checkpointing",
            "flat_ce_no_gradient_checkpointing", "fused_ce_no_gradient_checkpointing",
            "spatial_fused_backward_no_gradient_checkpointing"))
            and args.reference_profile != "no_gradient_checkpointing"):
        parser.error("Additional no-checkpoint candidates require the accepted no_gradient_checkpointing reference")
    if ("spatial_scheduled_ce_no_gradient_checkpointing" in args.profiles[1:]
            and args.reference_profile != "spatial_fused_backward_no_gradient_checkpointing"):
        parser.error("Scheduled spatial CE requires the accepted spatial_fused_backward_no_gradient_checkpointing reference")
    if ("causal_contiguous_gradient_ce_no_gradient_checkpointing" in args.profiles[1:]
            and args.reference_profile != "spatial_scheduled_ce_no_gradient_checkpointing"):
        parser.error("Contiguous causal gradient CE requires the accepted spatial_scheduled_ce_no_gradient_checkpointing reference")
    return args


def recovery_binding(path, task, config):
    """Verify original immutable bytes and recipe; do not deserialize in the plan."""
    from training.sft.runtime_migration import validate_identity_compatibility
    root = Path(path).resolve()
    manifest = read_json(root / "manifest.json")
    if (root.is_symlink() or manifest.get("status") != "complete"
            or manifest.get("checkpoint_hash") != content_hash({k: v for k, v in manifest.items() if k != "checkpoint_hash"})
            or set(manifest.get("files", {})) != {"full_state.pt"}
            or not {"adapter", "optimizer", "scheduler", "rng", "progress"}.issubset(manifest.get("state_keys", []))):
        raise ValueError("Original recovery checkpoint manifest is not a full immutable checkpoint")
    spec = sft_fit.SFTFitSpec.from_dict(task["spec"])
    target_identity = {**spec.identity(), "code_revision": config["runtime"]["implementation_revision"],
                       "dependency_hash": config["runtime"]["dependency_lock_hash"]}
    validate_identity_compatibility(manifest["identity"], target_identity)
    payload = root / "full_state.pt"
    inventory = manifest["files"]["full_state.pt"]
    if (payload.is_symlink() or not payload.is_file() or payload.stat().st_size != inventory["size_bytes"]
            or file_hash(payload) != inventory["sha256"]):
        raise ValueError("Original recovery checkpoint payload bytes changed")
    progress = manifest["progress"]
    position = progress["sampler_position"]
    if (type(position) is not int or not 0 <= position <= 20000
            or position != min(progress["completed_steps"] * 16, 20000)
            or progress.get("exposure_order_hash") != content_hash(sft_fit.exposure_order(10000))):
        raise ValueError("Original recovery state is not at the exact frozen optimizer boundary")
    return {"mode": "verified_full_checkpoint", "path": str(root),
            "checkpoint_hash": manifest["checkpoint_hash"], "original_identity": manifest["identity"],
            "progress": progress, "manifest_file_sha256": file_hash(root / "manifest.json"),
            "payload_sha256": inventory["sha256"], "files": manifest["files"],
            "full_state_deserialization_verified": False}


def source_tree_state():
    """Reject uncommitted executed source in real checkouts; describe archives."""
    repository = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=ROOT,
                                capture_output=True, text=True, check=False)
    if repository.returncode:
        return {"mode": "archive_certificate_source_bytes_and_explicit_commit", "errors": []}
    paths = ["src/", "scripts/benchmark_sft_optimizations.py", "scripts/run_certified_sft.py", "requirements.lock"]
    status = subprocess.run(["git", "status", "--porcelain=v1", "--untracked-files=all", "--", *paths],
                            cwd=ROOT, capture_output=True, text=True, check=False)
    if status.returncode:
        return {"mode": "git", "errors": ["Cannot establish clean committed benchmark source"]}
    dirty = status.stdout.splitlines()
    return {"mode": "git", "executed_source_clean": not dirty, "uncommitted_paths": dirty,
            "errors": ["Executed benchmark/training source has uncommitted changes"] if dirty else []}


def build_plan(args):
    config = validate_configuration(read_json(args.config), runtime=True)
    certificate, bundle, worklist = (read_json(getattr(args, name))
                                     for name in ("certificate", "bundle", "worklist"))
    source_state = source_tree_state()
    errors = list(source_state["errors"])
    source = code_revision()
    if config["runtime"]["implementation_revision"] != source:
        errors.append("Executed source differs from the frozen consumer; recertify before GPU work")
    task = None
    try:
        task = certified_tasks(config, certificate, bundle, worklist,
                               family=args.family, model=args.model, level=args.level)[0]
    except (ValueError, KeyError, TypeError) as error:
        errors.append(str(error))
    if task is None:
        selected = [row for row in worklist if (row.get("family"), row.get("model_id"), row.get("level"))
                    == (args.family, args.model, args.level)]
        if len(selected) != 1:
            raise ValueError("Benchmark requires one declared frozen SFT condition")
        task = selected[0]
    count = (args.warmup_updates + args.updates) * 16
    order = sft_fit.exposure_order(10000)
    initialization = {"mode": "fresh_pinned_base", "progress": {"completed_steps": 0, "sampler_position": 0}}
    if args.recovery_checkpoint:
        initialization = recovery_binding(args.recovery_checkpoint, task, config)
        from training.sft.execution_profile import SFTExecutionProfile
        restored_profile = SFTExecutionProfile.from_dict(initialization["original_identity"].get("execution_profile", {})).to_dict()
        if restored_profile != PROFILE_SETTINGS[args.reference_profile]:
            errors.append("Recovery checkpoint execution profile differs from the declared paired reference")
    position = initialization["progress"]["sampler_position"]
    if position + count > len(order):
        raise ValueError("Benchmark would exceed the remaining frozen training exposure")
    paths = ["scripts/benchmark_sft_optimizations.py", "scripts/run_certified_sft.py",
             "src/training/sft/individual_fit/sft_fit.py", "src/training/sft/execution_profile.py",
             *LOSS_HELPER_FILES.values(),
             "requirements.lock"]
    paths = sorted(set(paths) | {str(path.relative_to(ROOT)) for path in (ROOT / "src").rglob("*.py")})
    source_hashes = {path: file_hash(ROOT / path) for path in paths if (ROOT / path).is_file()}
    for name in args.profiles:
        helper = LOSS_HELPER_FILES.get(PROFILE_SETTINGS[name]["loss_implementation"])
        if helper and helper not in source_hashes:
            errors.append("Selected loss helper source is missing: " + helper)
    plan = {
        "schema_version": SCHEMA, "status": "PLANNED_PENDING_GPU",
        "source_code_commit": source, "configuration_hash": configuration_hash(config),
        "source_tree_validation": source_state,
        "input_file_hashes": {name: file_hash(getattr(args, name))
                              for name in ("config", "certificate", "bundle", "worklist")},
        "source_file_hashes": source_hashes,
        "handoff_certificate_hash": certificate.get("certificate_hash"),
        "consumer_bindings_valid": not errors, "binding_errors": errors,
        "condition": {"family": args.family, "model": args.model, "level": args.level,
                      "spec": task["spec"], "dataset": task["dataset"]},
        "profiles": {name: PROFILE_SETTINGS[name] for name in args.profiles},
        "reference_profile": args.reference_profile,
        "warmup_updates": args.warmup_updates, "measured_updates": args.updates,
        "paired_repetitions": args.repetitions,
        "comparison_design": "Fresh reference/candidate subprocesses per repetition; reference first then candidate first on alternate repetitions",
        "initialization": initialization,
        "recovery_checkpoint": ({"checkpoint_hash": initialization["checkpoint_hash"],
            "identity": initialization["original_identity"], "progress": initialization["progress"],
            "files": initialization["files"], "manifest_file_sha256": initialization["manifest_file_sha256"]}
            if args.recovery_checkpoint else None),
        "microbatches_per_update": 16, "recipe": sft_fit.SFTRecipe().to_dict(),
        "frozen_exposure_order_hash": content_hash(order),
        "selected_record_indices": order[position:position + count],
        "selected_order_hash": content_hash(order[position:position + count]),
        "device": args.device,
        "technical_tolerances": {"loss_atol": args.loss_atol, "loss_rtol": args.loss_rtol,
            "gradient_relative_l2": args.gradient_relative_l2,
            "update_relative_l2": args.update_relative_l2,
            "gradient_cosine_floor": args.gradient_cosine_floor,
            "minimum_speedup": args.minimum_speedup, "require_identical_rng_states": True,
            "require_exact_state_comparison": any(PROFILE_SETTINGS[name]["loss_implementation"] == "causal_contiguous_gradient_ce"
                                                   for name in args.profiles),
            "speedup_rule": "minimum speedup across corresponding logical update pairs in every repetition"},
        "profile_timeout_seconds": args.profile_timeout_seconds,
        "official_training_state_read": bool(args.recovery_checkpoint), "official_training_state_modified": False,
        "scientific_equivalence_established": False, "approved_for_production": False,
        "full_fit_time_extrapolation_supported": False,
        "timing_limitations": "A few fresh-state updates are a noisy technical screen. Repeat on the target GPU before extrapolating full-fit throughput.",
    }
    plan["plan_hash"] = content_hash(plan)
    return plan, config, task


def tensor_digest(tensors):
    """Exact named tensor identity, including BF16 bytes without float casting."""
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(content_hash([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.reshape(-1).view(__import__("torch").uint8).numpy().tobytes())
    return digest.hexdigest()


def rng_digest(state):
    digest = hashlib.sha256(repr(state["python"]).encode())
    algorithm, keys, position, has_gauss, cached_gauss = state["numpy"]
    digest.update(content_hash([algorithm, position, has_gauss, cached_gauss]).encode())
    digest.update(keys.tobytes())
    digest.update(state["torch_cpu"].cpu().numpy().tobytes())
    for value in state["torch_cuda"]:
        digest.update(value.cpu().numpy().tobytes())
    return digest.hexdigest()


def state_digest(state):
    """Canonical recursively typed optimizer/scheduler state, including tensors."""
    import torch
    import numpy as np
    digest = hashlib.sha256()
    def visit(value):
        if isinstance(value, torch.Tensor):
            digest.update(b"tensor" + tensor_digest({"value": value}).encode())
        elif isinstance(value, np.ndarray):
            digest.update(content_hash(["ndarray", str(value.dtype), list(value.shape)]).encode())
            digest.update(value.tobytes())
        elif isinstance(value, dict):
            digest.update(b"dict")
            for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
                visit(key)
                visit(value[key])
        elif isinstance(value, (list, tuple)):
            digest.update(content_hash([type(value).__name__, len(value)]).encode())
            for item in value:
                visit(item)
        else:
            digest.update(content_hash([type(value).__name__, value]).encode())
    visit(state)
    return digest.hexdigest()


def profile_run(args, plan, config, task):
    import torch
    from training.sft.execution_profile import SFTExecutionProfile
    if not plan["consumer_bindings_valid"]:
        raise ValueError("Current source/configuration/corpus certificate must match before GPU work")
    spending = config.get("spending", {})
    if not (spending.get("authorized") and spending.get("authorization_evidence")
            and spending.get("authorized_usd", 0) > 0):
        raise ValueError("GPU benchmarking requires existing spending authorization")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable; no profile has been approved")
    packages = {name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "peft", "bitsandbytes", "numpy")}
    locked = dict(re.findall(r"^([A-Za-z0-9_-]+)==([^\s]+)", (ROOT / "requirements.lock").read_text(), re.M))
    if any(version.split("+")[0] != locked[name] for name, version in packages.items()):
        raise RuntimeError("Installed training dependencies differ from the pinned lock")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    spec = sft_fit.SFTFitSpec.from_dict(task["spec"])
    train, validation = load_dataset_bundle(task["dataset"])
    data_evidence = sft_fit.validate_datasets(spec, train, validation)
    if any(type(record.get("transformed_token_count")) is not int
           or not 0 < record["transformed_token_count"] <= 4096 for record in [*train, *validation]):
        raise ValueError("Frozen corpus must declare positive, untruncated native token lengths")
    longest_record = max([*train, *validation], key=lambda record: record["transformed_token_count"])
    profile = SFTExecutionProfile(**PROFILE_SETTINGS[args._profile])
    helper_file = LOSS_HELPER_FILES.get(profile.loss_implementation)
    loss_helper_hash = plan["source_file_hashes"].get(helper_file) if helper_file else None
    if helper_file and not loss_helper_hash:
        raise RuntimeError("Selected loss helper bytes are missing from the frozen plan")
    model, tokenizer, optimizer, scheduler, runtime = sft_fit._runtime(
        spec, str(device), cache_root=config["runtime"].get("cache_root"), execution_profile=profile)
    if model.config._attn_implementation != profile.attention_implementation:
        raise RuntimeError("Runtime did not apply the explicitly requested attention implementation")
    if bool(model.is_gradient_checkpointing) != profile.gradient_checkpointing or model.config.use_cache is not False:
        raise RuntimeError("Actual activation checkpoint/cache policy differs from the explicit candidate")
    if runtime.get("loss_implementation", "spatial_ce") != profile.loss_implementation:
        raise RuntimeError("Runtime did not apply the explicitly requested completion-loss implementation")
    if helper_file and runtime.get("loss_helper_sha256") != loss_helper_hash:
        raise RuntimeError("Loaded loss helper differs from the frozen source bytes")
    full_tokenized_train = ([sft_fit.tokenize_completion(record, tokenizer) for record in train]
                            if profile.cache_collated_inputs else None)
    selected = ([full_tokenized_train[index] for index in plan["selected_record_indices"]]
                if full_tokenized_train else [sft_fit.tokenize_completion(train[index], tokenizer)
                                             for index in plan["selected_record_indices"]])
    restore_evidence = {"mode": "fresh_pinned_base", "status": "NOT_APPLICABLE"}
    if args.recovery_checkpoint:
        from shared_procedures.checkpointing import load_checkpoint, verify_checkpoint, _same_state
        from peft import get_peft_model_state_dict, set_peft_model_state_dict
        original_identity = plan["initialization"]["original_identity"]
        manifest = verify_checkpoint(args.recovery_checkpoint, original_identity)
        if manifest["checkpoint_hash"] != plan["initialization"]["checkpoint_hash"]:
            raise ValueError("Original recovery checkpoint changed after the frozen plan")
        restored = load_checkpoint(args.recovery_checkpoint, original_identity)
        set_peft_model_state_dict(model, restored["adapter"])
        optimizer.load_state_dict(restored["optimizer"])
        scheduler.load_state_dict(restored["scheduler"])
        sft_fit.restore_rng(restored["rng"])
        checks = {"adapter_verified": _same_state(get_peft_model_state_dict(model, save_embedding_layers=False), restored["adapter"]),
                  "optimizer_verified": _same_state(optimizer.state_dict(), restored["optimizer"]),
                  "scheduler_verified": _same_state(scheduler.state_dict(), restored["scheduler"]),
                  "rng_verified": _same_state(sft_fit.capture_rng(), restored["rng"]),
                  "progress_verified": restored["progress"] == plan["initialization"]["progress"]}
        if not all(checks.values()):
            raise RuntimeError("Original full checkpoint state did not restore exactly into the isolated runtime")
        restore_evidence = {"mode": "verified_full_checkpoint", "status": "PASSED", **checks,
            "checkpoint_hash": manifest["checkpoint_hash"], "original_identity": original_identity,
            "progress": restored["progress"], "official_checkpoint_modified": False}
        del restored
    initial = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()
               if parameter.requires_grad}
    if not initial or any("lora_" not in name for name in initial):
        raise RuntimeError("Benchmark must optimize only the frozen LoRA adapter")
    initial_rng = rng_digest(sft_fit.capture_rng())
    initial_optimizer_hash = state_digest(optimizer.state_dict())
    initial_scheduler_hash = state_digest(scheduler.state_dict())
    longest = sft_fit.tokenize_completion(longest_record, tokenizer)
    cache = None
    longest_cached = None
    cache_setup_seconds = 0.0
    cache_resident_bytes = 0
    full_cache = None
    if profile.cache_collated_inputs:
        cache_start = time.perf_counter()
        full_cache = sft_fit.prepare_cuda_input_cache(full_tokenized_train, tokenizer.pad_token_id, device,
                                                     validate_counts=profile.validated_loss_counts)
        cache = tuple(full_cache[index] for index in plan["selected_record_indices"])
        longest_cached = sft_fit.prepare_cuda_input_cache([longest], tokenizer.pad_token_id, device,
                                                        validate_counts=profile.validated_loss_counts)[0]
        torch.cuda.synchronize(device)
        cache_setup_seconds = time.perf_counter() - cache_start
        cache_resident_bytes = sum(item.resident_bytes for item in full_cache) + longest_cached.resident_bytes
    torch.cuda.reset_peak_memory_stats(device)
    longest_probe = sft_fit.verify_mask_gradients(model, longest, tokenizer.pad_token_id,
                                               prepared_batch=longest_cached, execution_profile=profile)
    if helper_file and longest_probe.get("loss_helper_sha256") != loss_helper_hash:
        raise RuntimeError("Longest-sequence probe did not execute the frozen loss helper")
    torch.cuda.synchronize(device)
    longest_probe = {**longest_probe, "example_id": longest["example_id"],
        "native_token_count": len(longest["input_ids"]), "optimizer_updates": 0,
        "loss_implementation": profile.loss_implementation,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
    if rng_digest(sft_fit.capture_rng()) != initial_rng:
        raise RuntimeError("Non-updating longest-sequence probe changed the initialization RNG state")
    model.train()
    losses, cuda_ms, wall_seconds = [], [], []
    # Retaining scalar detached losses does not synchronize individual microbatches.
    for update in range(args.warmup_updates + args.updates):
        group = selected[update * 16:(update + 1) * 16]
        torch.cuda.synchronize(device)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        if update == args.warmup_updates:
            torch.cuda.reset_peak_memory_stats(device)
        wall_start = time.perf_counter()
        start.record()
        values = sft_fit.one_optimizer_update(model, optimizer, scheduler, group, tokenizer.pad_token_id,
                                             execution_profile=profile,
                                             prepared_batches=cache[update * 16:(update + 1) * 16] if cache else None)
        end.record()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - wall_start
        losses.extend(value.detach() for value in values)
        if update >= args.warmup_updates:
            cuda_ms.append(start.elapsed_time(end))
            wall_seconds.append(elapsed)
    final = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()
             if parameter.requires_grad}
    gradients = {name: parameter.grad.detach().cpu().clone() for name, parameter in model.named_parameters()
                 if parameter.requires_grad and parameter.grad is not None}
    deltas = {name: final[name].float() - value.float() for name, value in initial.items()}
    finite = (all(torch.isfinite(value).all().item() for value in [*final.values(), *gradients.values(), *losses])
              and bool(gradients))
    if not finite:
        raise RuntimeError("Nonfinite loss/adapter/gradient rejects the measured profile")
    torch.save({"initial": initial, "gradients": gradients, "deltas": deltas}, args._diagnostics)
    repo = "models--" + spec.model_name.replace("/", "--")
    from operations.assets import _cache_root
    cache = _cache_root(config["runtime"].get("cache_root"))
    assets = {}
    for variant, revision in (("full", spec.model_revision), ("tokenizer", spec.tokenizer_revision)):
        path = cache / "hub" / repo / "verified" / revision / variant / "asset_manifest.json"
        if not path.is_file():
            raise RuntimeError("Pinned model/tokenizer integrity manifest missing")
        manifest = read_json(path)
        assets[variant] = {"manifest_file_sha256": file_hash(path), "manifest_hash": manifest["manifest_hash"],
                           "revision": manifest["revision"], "files": manifest["files"]}
    props = torch.cuda.get_device_properties(device)
    return {"status": "MEASURED", "profile": args._profile, "plan_hash": plan["plan_hash"],
        "comparison_candidate": args._pair, "repetition": args._repetition,
        "settings": PROFILE_SETTINGS[args._profile], "finite": finite, "data": data_evidence,
        "runtime": runtime, "hardware": {"gpu": props.name, "vram_bytes": props.total_memory,
            "compute_capability": [props.major, props.minor], "cuda": torch.version.cuda},
        "packages": packages, "longest_sequence_probe": longest_probe,
        "restore_evidence": restore_evidence,
        "recovery_checkpoint": ({**plan["recovery_checkpoint"], "full_state_restore_verified": True,
            **{key: value for key, value in restore_evidence.items() if key.endswith("_verified")},
            "initial_optimizer_state_hash": initial_optimizer_hash,
            "initial_scheduler_state_hash": initial_scheduler_hash, "initial_rng_hash": initial_rng,
            "initial_adapter_hash": tensor_digest(initial)} if args.recovery_checkpoint else None),
        "actual_execution_policy": {"gradient_checkpointing": bool(model.is_gradient_checkpointing),
            "cache_collated_inputs": profile.cache_collated_inputs, "attention_implementation": model.config._attn_implementation,
            "use_cache": model.config.use_cache, "checkpoint_rng_preserved": True,
            "loss_implementation": profile.loss_implementation,
            **({"loss_helper_sha256": runtime["loss_helper_sha256"]}
               if helper_file else {})},
        "cuda_input_cache": {"enabled": profile.cache_collated_inputs, "resident_bytes": cache_resident_bytes,
                             "setup_seconds": cache_setup_seconds, "cached_records": len(full_cache) if full_cache else 0},
        "applied_attention_implementation": model.config._attn_implementation,
        "attention_module_classes": sorted({type(module).__name__ for name, module in model.named_modules()
                                             if name.endswith("self_attn")}),
        "sdpa_backend_flags": {"flash_enabled": torch.backends.cuda.flash_sdp_enabled(),
            "memory_efficient_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math_enabled": torch.backends.cuda.math_sdp_enabled()},
        "assets": assets, "tokenized_records_hash": content_hash(selected),
        "sample_ids": [row["example_id"] for row in selected],
        "token_counts": [len(row["input_ids"]) for row in selected],
        "supervised_token_counts": [row["supervised_token_count"] for row in selected],
        "initial_adapter_hash": tensor_digest(initial), "final_adapter_hash": tensor_digest(final),
        "clipped_gradient_hash": tensor_digest(gradients), "adapter_delta_hash": tensor_digest(deltas),
        "initial_rng_hash": initial_rng, "final_rng_hash": rng_digest(sft_fit.capture_rng()),
        "initial_optimizer_state_hash": initial_optimizer_hash, "initial_scheduler_state_hash": initial_scheduler_hash,
        "final_optimizer_state_hash": state_digest(optimizer.state_dict()),
        "final_scheduler_state_hash": state_digest(scheduler.state_dict()),
        "losses": [value.item() for value in losses], "cuda_ms_per_update": cuda_ms,
        "wall_seconds_per_update": wall_seconds,
        "median_update_seconds": statistics.median(wall_seconds),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "approved_for_production": False, "scientific_equivalence_established": False}


def vector_comparison(reference, candidate):
    import torch
    if set(reference) != set(candidate):
        return {"compatible": False, "reason": "Tensor names differ"}
    squares, difference, dot, other = 0.0, 0.0, 0.0, 0.0
    maximum, equal, finite = 0.0, True, True
    for name, first in reference.items():
        second = candidate[name]
        if first.shape != second.shape:
            return {"compatible": False, "reason": "Tensor shapes differ"}
        left, right = first.double(), second.double()
        finite = finite and bool(torch.isfinite(left).all() and torch.isfinite(right).all())
        if not finite:
            return {"compatible": False, "reason": "Nonfinite comparison tensor"}
        delta = left - right
        squares += (left * left).sum().item()
        other += (right * right).sum().item()
        difference += (delta * delta).sum().item()
        dot += (left * right).sum().item()
        maximum = max(maximum, delta.abs().max().item())
        equal = equal and torch.equal(first, second)
    norm = math.sqrt(squares)
    relative = math.sqrt(difference) / norm if norm else (0.0 if not difference else None)
    cosine = dot / math.sqrt(squares * other) if squares and other else (1.0 if equal else 0.0)
    return {"compatible": True, "bitwise_equal": equal, "relative_l2": relative,
            "cosine": min(1.0, max(-1.0, cosine)), "maximum_absolute_difference": maximum}


def compare_profiles(baseline, candidate, first, second, bounds):
    invariant_names = ("plan_hash", "data", "hardware", "packages", "assets", "sample_ids", "token_counts",
                       "supervised_token_counts", "tokenized_records_hash", "initial_adapter_hash", "initial_rng_hash",
                       "initial_optimizer_state_hash", "initial_scheduler_state_hash", "recovery_checkpoint")
    invariant_failures = [name for name in invariant_names if baseline.get(name) != candidate.get(name)]
    if baseline.get("status") != "MEASURED" or candidate.get("status") != "MEASURED":
        return {"technical_gate": "FAILED", "reason": "Both profiles require actual GPU measurements",
                "approved_for_production": False}
    losses = baseline["losses"], candidate["losses"]
    loss_match = len(losses[0]) == len(losses[1]) and all(
        math.isfinite(left) and math.isfinite(right)
        and abs(left - right) <= bounds["loss_atol"] + bounds["loss_rtol"] * abs(left)
        for left, right in zip(*losses))
    gradient = vector_comparison(first["gradients"], second["gradients"])
    update = vector_comparison(first["deltas"], second["deltas"])
    rng_equal = baseline["final_rng_hash"] == candidate["final_rng_hash"]
    exact_state = (gradient.get("bitwise_equal") and update.get("bitwise_equal") and rng_equal
        and baseline.get("final_optimizer_state_hash") == candidate.get("final_optimizer_state_hash")
        and baseline.get("final_scheduler_state_hash") == candidate.get("final_scheduler_state_hash"))
    exact_hash_fields = ("initial_adapter_hash", "initial_rng_hash", "initial_optimizer_state_hash",
        "initial_scheduler_state_hash", "clipped_gradient_hash", "adapter_delta_hash", "final_adapter_hash",
        "final_optimizer_state_hash", "final_scheduler_state_hash", "final_rng_hash")
    complete_exact_state = (exact_state and losses[0] == losses[1]
        and all(isinstance(baseline.get(key), str) and re.fullmatch(r"[0-9a-f]{64}", baseline[key]) is not None
                and baseline.get(key) == candidate.get(key) for key in exact_hash_fields))
    require_exact = bounds.get("require_exact_state_comparison", False)
    policy_failures = []
    for receipt in (baseline, candidate):
        actual = dict(receipt.get("actual_execution_policy", {}))
        # Earlier v2 receipts used the original spatial CE and omitted this field.
        actual.setdefault("loss_implementation", "spatial_ce")
        settings = receipt.get("settings", {})
        helper_metadata_valid = True
        expected = {"gradient_checkpointing": settings.get("gradient_checkpointing", True),
                    "cache_collated_inputs": settings.get("cache_collated_inputs", False),
                    "attention_implementation": settings.get("attention_implementation", "eager"),
                    "use_cache": False, "checkpoint_rng_preserved": True,
                    "loss_implementation": settings.get("loss_implementation", "spatial_ce")}
        if expected["loss_implementation"] in LOSS_HELPER_FILES:
            helper_hash = receipt.get("runtime", {}).get("loss_helper_sha256")
            expected["loss_helper_sha256"] = helper_hash
            helper_metadata_valid = isinstance(helper_hash, str) and re.fullmatch(r"[0-9a-f]{64}", helper_hash) is not None
        if actual != expected or not helper_metadata_valid:
            policy_failures.append(receipt.get("profile", "unknown"))
    def within(value, threshold):
        return value.get("compatible") and value.get("relative_l2") is not None and value["relative_l2"] <= threshold
    numerical = (not invariant_failures and not policy_failures and loss_match and baseline["finite"] and candidate["finite"]
                 and within(gradient, bounds["gradient_relative_l2"])
                 and within(update, bounds["update_relative_l2"])
                 and gradient["cosine"] >= bounds["gradient_cosine_floor"]
                 and (rng_equal or not bounds["require_identical_rng_states"])
                 and (complete_exact_state or not require_exact))
    speedup = baseline["median_update_seconds"] / candidate["median_update_seconds"]
    first_times, second_times = baseline.get("wall_seconds_per_update", []), candidate.get("wall_seconds_per_update", [])
    valid_times = (bool(first_times) and len(first_times) == len(second_times)
                   and all(math.isfinite(value) and value > 0 for value in [*first_times, *second_times]))
    observed_minimum_speedup = min(first_times) / max(second_times) if valid_times else None
    paired_speedups = [first / second for first, second in zip(first_times, second_times)] if valid_times else []
    paired_minimum = min(paired_speedups) if paired_speedups else None
    speed_screen = valid_times and paired_minimum >= bounds["minimum_speedup"]
    return {"technical_gate": "PASSED" if numerical and speed_screen else "FAILED",
        "numerical_screen_passed": numerical, "invariant_failures": invariant_failures,
        "execution_policy_failures": policy_failures,
        "intended_execution_policy_changes": {key: {"reference": baseline["settings"].get(key),
            "candidate": candidate["settings"].get(key)} for key in PROFILE_SETTINGS["baseline"]
            if baseline["settings"].get(key) != candidate["settings"].get(key)},
        "losses_within_declared_tolerance": loss_match,
        "maximum_loss_absolute_difference": max((abs(a - b) for a, b in zip(*losses)), default=None),
        "gradients": gradient, "adapter_updates": update, "final_rng_states_identical": rng_equal,
        "final_optimizer_states_identical": baseline.get("final_optimizer_state_hash") == candidate.get("final_optimizer_state_hash"),
        "final_scheduler_states_identical": baseline.get("final_scheduler_state_hash") == candidate.get("final_scheduler_state_hash"),
        "losses_exactly_identical": losses[0] == losses[1],
        "final_adapters_identical": baseline.get("final_adapter_hash") == candidate.get("final_adapter_hash"),
        "exact_state_comparison_required": require_exact,
        "exact_state_comparison_passed": complete_exact_state if require_exact else exact_state,
        "measured_speedup": speedup, "observed_minimum_speedup": observed_minimum_speedup,
        "paired_update_speedups": paired_speedups, "minimum_paired_update_speedup": paired_minimum,
        "speedup_threshold_passed": speed_screen,
        "approved_for_production": False, "scientific_equivalence_established": False,
        "full_fit_time_extrapolation_supported": False,
        "limitation": "Short fresh-state timing samples are noisy; this screen cannot establish full-fit throughput, equal full training results, or exact checkpoint migration."}


def run_basename(pair, repetition, profile):
    return f"{pair}_repeat_{repetition:02d}_{profile}"


def aggregate_comparisons(values, reference):
    if not values or any(row.get("reason") for row in values):
        return {"technical_gate": "FAILED", "numerical_screen_passed": False,
                "reference_profile": reference, "repetitions": values,
                "reason": "A required paired measurement is unavailable", "approved_for_production": False}
    result = dict(values[0])
    for field in ("numerical_screen_passed", "losses_within_declared_tolerance", "final_rng_states_identical",
                  "speedup_threshold_passed", "final_optimizer_states_identical", "final_scheduler_states_identical",
                  "exact_state_comparison_passed", "losses_exactly_identical", "final_adapters_identical"):
        result[field] = all(row[field] for row in values)
    for field in ("invariant_failures", "execution_policy_failures"):
        result[field] = sorted({entry for row in values for entry in row[field]})
    for field in ("gradients", "adapter_updates"):
        vectors = [row[field] for row in values]
        result[field] = {"compatible": all(row.get("compatible") for row in vectors),
                         "bitwise_equal": all(row.get("bitwise_equal") for row in vectors),
                         "relative_l2": max((row.get("relative_l2") for row in vectors if row.get("relative_l2") is not None), default=None),
                         "cosine": min((row.get("cosine", -1) for row in vectors)),
                         "maximum_absolute_difference": max(row.get("maximum_absolute_difference", 0) for row in vectors)}
    result.update(reference_profile=reference, repetitions=values,
        technical_gate="PASSED" if all(row["technical_gate"] == "PASSED" for row in values) else "FAILED",
        minimum_paired_update_speedup=min(row["minimum_paired_update_speedup"] for row in values),
        observed_minimum_speedup=min(row["observed_minimum_speedup"] for row in values),
        maximum_loss_absolute_difference=max(row["maximum_loss_absolute_difference"] for row in values))
    return result


def main(argv=None):
    args = arguments(argv)
    plan, config, task = build_plan(args)
    output = Path(args.output).resolve()
    if any(part.startswith("checkpoint_") for part in output.parts) or {"training", "sft"}.issubset(set(output.parts)):
        raise ValueError("Benchmark output must be outside production fit/checkpoint namespaces")
    if args._profile:
        frozen = read_json(args._plan)
        if frozen != plan:
            raise ValueError("Frozen benchmark plan changed before profile execution")
        report_path = output / (run_basename(args._pair, args._repetition, args._profile) + ".json")
        try:
            report = profile_run(args, plan, config, task)
        except Exception as error:
            report = {"status": "BLOCKED", "profile": args._profile, "plan_hash": plan["plan_hash"],
                      "error_type": type(error).__name__, "detail": str(error), "approved_for_production": False}
            write_json(report_path, report)
            return 2
        write_json(report_path, report)
        return 0
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty benchmark output directory; never overwrite measurements")
    write_json(output / "plan.json", plan)
    if args.plan_only:
        print(json.dumps({"status": plan["status"], "plan_hash": plan["plan_hash"],
                          "consumer_bindings_valid": plan["consumer_bindings_valid"],
                          "gpu_work_started": False, "approved_for_production": False}))
        return 0
    if not plan["consumer_bindings_valid"]:
        raise ValueError("Recertify the current source and frozen consumer before GPU benchmarking")
    reports, comparisons, runs = {}, {}, {}
    original = list(argv if argv is not None else sys.argv[1:])
    with tempfile.TemporaryDirectory(prefix="private_sft_benchmark_", dir=output) as temporary:
        def launch(pair, repetition, name):
            basename = run_basename(pair, repetition, name)
            path = Path(temporary) / (basename + ".pt")
            command = [sys.executable, str(Path(__file__).resolve()), *original,
                       "--_profile", name, "--_plan", str(output / "plan.json"), "--_diagnostics", str(path),
                       "--_pair", pair, "--_repetition", str(repetition)]
            report_path = output / (basename + ".json")
            with (output / (basename + ".log")).open("w") as log:
                try:
                    result = subprocess.run(command, cwd=Path.cwd(), stdout=log, stderr=subprocess.STDOUT,
                                            timeout=args.profile_timeout_seconds, check=False)
                    if not report_path.is_file():
                        write_json(report_path, {"status": "BLOCKED", "profile": name,
                            "detail": "Profile exited without a receipt", "exit_code": result.returncode,
                            "approved_for_production": False})
                except subprocess.TimeoutExpired:
                    write_json(report_path, {"status": "BLOCKED", "profile": name,
                        "detail": "Bounded profile timeout exceeded", "approved_for_production": False})
            row = read_json(report_path)
            row["run_id"] = basename
            runs[basename] = row
            if name not in reports:
                reports[name] = {**row, "repetitions": []}
            reports[name]["repetitions"].append(row)
            if row["status"] != "MEASURED":
                reports[name]["status"] = "BLOCKED"
            print(json.dumps({"phase": "profile_complete", "profile": name, "repetition": repetition,
                              "comparison_candidate": pair, "status": row["status"]}), flush=True)
            return row, path
        for name in args.profiles[1:]:
            reference = args.reference_profile
            paired = []
            for repetition in range(1, args.repetitions + 1):
                order = (reference, name) if repetition % 2 else (name, reference)
                actual = {profile: launch(name, repetition, profile) for profile in order}
                reference_row, first_path = actual[reference]
                candidate_row, second_path = actual[name]
                if reference_row["status"] == candidate_row["status"] == "MEASURED":
                    import torch
                    first = torch.load(first_path, map_location="cpu", weights_only=True)
                    second = torch.load(second_path, map_location="cpu", weights_only=True)
                    comparison = compare_profiles(reference_row, candidate_row, first, second, plan["technical_tolerances"])
                    del first, second
                else:
                    comparison = {"technical_gate": "FAILED", "reason": "Actual measurements unavailable", "approved_for_production": False}
                comparison.update(repetition=repetition, reference_profile=reference,
                                  reference_run_id=reference_row["run_id"], candidate_run_id=candidate_row["run_id"])
                paired.append(comparison)
                first_path.unlink(missing_ok=True)
                second_path.unlink(missing_ok=True)
            comparisons[name] = aggregate_comparisons(paired, reference)
    receipt = {"schema_version": SCHEMA, "status": "MEASURED" if all(r["status"] == "MEASURED" for r in reports.values()) else "BLOCKED",
        "at": datetime.now(timezone.utc).isoformat(), "plan_hash": plan["plan_hash"],
        "source_code_commit": plan["source_code_commit"], "configuration_hash": plan["configuration_hash"],
        "plan": plan, "profiles": reports, "profile_runs": runs, "comparisons": comparisons,
        "official_training_state_read": bool(args.recovery_checkpoint), "official_training_state_modified": False,
        "approved_for_production": False, "scientific_equivalence_established": False}
    receipt["receipt_hash"] = content_hash(receipt)
    write_json(output / "report.json", receipt)
    print(json.dumps({"status": receipt["status"], "receipt_hash": receipt["receipt_hash"],
                      "technical_gates": {name: row["technical_gate"] for name, row in comparisons.items()},
                      "official_training_state_modified": False, "approved_for_production": False}))
    if receipt["status"] != "MEASURED":
        return 2  # no valid measurement
    return 0 if all(row["technical_gate"] == "PASSED" for row in comparisons.values()) else 3


if __name__ == "__main__":
    raise SystemExit(main())
