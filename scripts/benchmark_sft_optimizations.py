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

SCHEMA = "isolated_sft_execution_benchmark_v1"
PROFILE_SETTINGS = {
    "baseline": {"attention_implementation": "eager", "validated_loss_counts": False,
                 "defer_finite_checks": False},
    "deferred_checks": {"attention_implementation": "eager", "validated_loss_counts": True,
                        "defer_finite_checks": True},
    "sdpa": {"attention_implementation": "sdpa", "validated_loss_counts": False,
             "defer_finite_checks": False},
    "combined_sdpa": {"attention_implementation": "sdpa", "validated_loss_counts": True,
                      "defer_finite_checks": True},
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
                        default=["baseline", "deferred_checks", "sdpa"])
    parser.add_argument("--warmup-updates", type=int, default=1)
    parser.add_argument("--updates", type=int, default=2)
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
    args = parser.parse_args(argv)
    if (args.warmup_updates < 0 or args.updates < 1
            or (args.warmup_updates + args.updates) * 16 > 10000):
        parser.error("Benchmark must use 1..10,000 records in whole groups of sixteen")
    bounds = (args.loss_atol, args.loss_rtol, args.gradient_relative_l2,
              args.update_relative_l2, args.gradient_cosine_floor, args.minimum_speedup)
    if (not all(math.isfinite(value) for value in bounds) or any(value < 0 for value in bounds[:4])
            or not 0 <= args.gradient_cosine_floor <= 1 or args.minimum_speedup < 1
            or args.profile_timeout_seconds < 1):
        parser.error("Numerical tolerances and timeout must be finite valid bounds")
    if len(set(args.profiles)) != len(args.profiles) or args.profiles[0] != "baseline":
        parser.error("Profiles must be unique and start with baseline")
    return args


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
    paths = ["scripts/benchmark_sft_optimizations.py", "scripts/run_certified_sft.py",
             "src/training/sft/individual_fit/sft_fit.py", "src/training/sft/execution_profile.py",
             "requirements.lock"]
    paths = sorted(set(paths) | {str(path.relative_to(ROOT)) for path in (ROOT / "src").rglob("*.py")})
    plan = {
        "schema_version": SCHEMA, "status": "PLANNED_PENDING_GPU",
        "source_code_commit": source, "configuration_hash": configuration_hash(config),
        "source_tree_validation": source_state,
        "input_file_hashes": {name: file_hash(getattr(args, name))
                              for name in ("config", "certificate", "bundle", "worklist")},
        "source_file_hashes": {path: file_hash(ROOT / path) for path in paths if (ROOT / path).is_file()},
        "handoff_certificate_hash": certificate.get("certificate_hash"),
        "consumer_bindings_valid": not errors, "binding_errors": errors,
        "condition": {"family": args.family, "model": args.model, "level": args.level,
                      "spec": task["spec"], "dataset": task["dataset"]},
        "profiles": {name: PROFILE_SETTINGS[name] for name in args.profiles},
        "warmup_updates": args.warmup_updates, "measured_updates": args.updates,
        "microbatches_per_update": 16, "recipe": sft_fit.SFTRecipe().to_dict(),
        "frozen_exposure_order_hash": content_hash(order),
        "selected_record_indices": order[:count], "selected_order_hash": content_hash(order[:count]),
        "device": args.device,
        "technical_tolerances": {"loss_atol": args.loss_atol, "loss_rtol": args.loss_rtol,
            "gradient_relative_l2": args.gradient_relative_l2,
            "update_relative_l2": args.update_relative_l2,
            "gradient_cosine_floor": args.gradient_cosine_floor,
            "minimum_speedup": args.minimum_speedup, "require_identical_rng_states": True},
        "profile_timeout_seconds": args.profile_timeout_seconds,
        "official_training_state_read": False, "official_training_state_modified": False,
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
        digest.update(value.view(__import__("torch").uint8).numpy().tobytes())
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
    model, tokenizer, optimizer, scheduler, runtime = sft_fit._runtime(
        spec, str(device), cache_root=config["runtime"].get("cache_root"), execution_profile=profile)
    if model.config._attn_implementation != profile.attention_implementation:
        raise RuntimeError("Runtime did not apply the explicitly requested attention implementation")
    selected = [sft_fit.tokenize_completion(train[index], tokenizer)
                for index in plan["selected_record_indices"]]
    initial = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()
               if parameter.requires_grad}
    if not initial or any("lora_" not in name for name in initial):
        raise RuntimeError("Benchmark must optimize only the frozen LoRA adapter")
    initial_rng = rng_digest(sft_fit.capture_rng())
    longest = sft_fit.tokenize_completion(longest_record, tokenizer)
    torch.cuda.reset_peak_memory_stats(device)
    longest_probe = sft_fit.verify_mask_gradients(model, longest, tokenizer.pad_token_id)
    torch.cuda.synchronize(device)
    longest_probe = {**longest_probe, "example_id": longest["example_id"],
        "native_token_count": len(longest["input_ids"]), "optimizer_updates": 0,
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
                                             execution_profile=profile)
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
        "settings": PROFILE_SETTINGS[args._profile], "finite": finite, "data": data_evidence,
        "runtime": runtime, "hardware": {"gpu": props.name, "vram_bytes": props.total_memory,
            "compute_capability": [props.major, props.minor], "cuda": torch.version.cuda},
        "packages": packages, "longest_sequence_probe": longest_probe,
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
                       "supervised_token_counts", "tokenized_records_hash", "initial_adapter_hash", "initial_rng_hash")
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
    def within(value, threshold):
        return value.get("compatible") and value.get("relative_l2") is not None and value["relative_l2"] <= threshold
    numerical = (not invariant_failures and loss_match and baseline["finite"] and candidate["finite"]
                 and within(gradient, bounds["gradient_relative_l2"])
                 and within(update, bounds["update_relative_l2"])
                 and gradient["cosine"] >= bounds["gradient_cosine_floor"]
                 and (rng_equal or not bounds["require_identical_rng_states"]))
    speedup = baseline["median_update_seconds"] / candidate["median_update_seconds"]
    first_times, second_times = baseline.get("wall_seconds_per_update", []), candidate.get("wall_seconds_per_update", [])
    valid_times = (bool(first_times) and bool(second_times)
                   and all(math.isfinite(value) and value > 0 for value in [*first_times, *second_times]))
    observed_minimum_speedup = min(first_times) / max(second_times) if valid_times else None
    speed_screen = valid_times and observed_minimum_speedup >= bounds["minimum_speedup"]
    return {"technical_gate": "PASSED" if numerical and speed_screen else "FAILED",
        "numerical_screen_passed": numerical, "invariant_failures": invariant_failures,
        "losses_within_declared_tolerance": loss_match,
        "maximum_loss_absolute_difference": max((abs(a - b) for a, b in zip(*losses)), default=None),
        "gradients": gradient, "adapter_updates": update, "final_rng_states_identical": rng_equal,
        "measured_speedup": speedup, "observed_minimum_speedup": observed_minimum_speedup,
        "speedup_threshold_passed": speed_screen,
        "approved_for_production": False, "scientific_equivalence_established": False,
        "full_fit_time_extrapolation_supported": False,
        "limitation": "Short fresh-state timing samples are noisy; this screen cannot establish full-fit throughput, equal full training results, or exact checkpoint migration."}


def main(argv=None):
    args = arguments(argv)
    plan, config, task = build_plan(args)
    output = Path(args.output).resolve()
    if "checkpoint_" in str(output) or {"training", "sft"}.issubset(set(output.parts)):
        raise ValueError("Benchmark output must be outside production fit/checkpoint namespaces")
    if args._profile:
        frozen = read_json(args._plan)
        if frozen != plan:
            raise ValueError("Frozen benchmark plan changed before profile execution")
        report_path = output / (args._profile + ".json")
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
    reports, comparisons = {}, {}
    with tempfile.TemporaryDirectory(prefix="private_sft_benchmark_", dir=output) as temporary:
        for name in args.profiles:
            path = Path(temporary) / (name + ".pt")
            original = list(argv if argv is not None else sys.argv[1:])
            command = [sys.executable, str(Path(__file__).resolve()), *original,
                       "--_profile", name, "--_plan", str(output / "plan.json"), "--_diagnostics", str(path)]
            with (output / (name + ".log")).open("w") as log:
                try:
                    result = subprocess.run(command, cwd=Path.cwd(), stdout=log, stderr=subprocess.STDOUT,
                                            timeout=args.profile_timeout_seconds, check=False)
                    if not (output / (name + ".json")).is_file():
                        write_json(output / (name + ".json"), {"status": "BLOCKED", "profile": name,
                                   "detail": "Profile exited without a receipt", "exit_code": result.returncode,
                                   "approved_for_production": False})
                except subprocess.TimeoutExpired:
                    write_json(output / (name + ".json"), {"status": "BLOCKED", "profile": name,
                               "detail": "Bounded profile timeout exceeded", "approved_for_production": False})
            reports[name] = read_json(output / (name + ".json"))
            if name != "baseline" and reports["baseline"]["status"] == reports[name]["status"] == "MEASURED":
                import torch
                baseline_state = torch.load(Path(temporary) / "baseline.pt", map_location="cpu", weights_only=True)
                candidate_state = torch.load(path, map_location="cpu", weights_only=True)
                comparisons[name] = compare_profiles(reports["baseline"], reports[name], baseline_state,
                                                     candidate_state, plan["technical_tolerances"])
                del baseline_state, candidate_state
            elif name != "baseline":
                comparisons[name] = {"technical_gate": "FAILED", "reason": "Actual measurements unavailable",
                                     "approved_for_production": False}
    receipt = {"schema_version": SCHEMA, "status": "MEASURED" if all(r["status"] == "MEASURED" for r in reports.values()) else "BLOCKED",
        "at": datetime.now(timezone.utc).isoformat(), "plan_hash": plan["plan_hash"],
        "profiles": reports, "comparisons": comparisons,
        "official_training_state_read": False, "official_training_state_modified": False,
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
