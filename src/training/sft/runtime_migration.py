"""Explicit, evidence-bound continuation from an immutable full SFT checkpoint.

Changing source, hardware or execution kernels is a declared migration, not an
exact continuation under the parent's identity. Parent checkpoint bytes remain
untouched; adapter, optimizer, scheduler, sampler and RNG state are inherited.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from shared_procedures.identity import content_hash, file_hash
from .execution_profile import SFTExecutionProfile, resolve_execution_profile

ROOT = Path(__file__).resolve().parents[3]
IMMUTABLE_IDENTITY_FIELDS = (
    "family", "model_id", "level", "run_id", "model_name", "model_revision",
    "tokenizer_revision", "strength", "dataset_hash", "producer_configuration_hash",
    "recipe", "dependency_hash", "implementation_version",
)
MIGRATABLE_IDENTITY_FIELDS = {
    "code_revision", "configuration_hash", "handoff_certificate_hash",
    "execution_profile", "migration_hash", "runtime_migration_hash",
}
REQUIRED_SOURCE_FILES = (
    "scripts/benchmark_sft_optimizations.py", "scripts/run_certified_sft.py",
    "src/training/sft/individual_fit/sft_fit.py", "src/training/sft/execution_profile.py",
    "src/training/sft/runtime_migration.py", "requirements.lock",
)
LOSS_HELPER_FILES = {
    "fused_ce": "src/training/sft/fused_loss.py",
    "spatial_fused_backward": "src/training/sft/spatial_loss.py",
    "spatial_scheduled_ce": "src/training/sft/scheduled_spatial_loss.py",
}


class MigrationError(ValueError):
    """Continuation evidence or full-state compatibility failed closed."""


def _profile(value: Any) -> SFTExecutionProfile:
    if value is None:
        return SFTExecutionProfile()
    if isinstance(value, Mapping):
        return SFTExecutionProfile(**dict(value))
    return resolve_execution_profile(value)


def _verified_hash(record: Mapping[str, Any], field: str) -> str:
    if not isinstance(record, Mapping):
        raise MigrationError("Migration evidence must be a mapping")
    expected = content_hash({key: value for key, value in record.items() if key != field})
    if record.get(field) != expected:
        raise MigrationError(f"Migration evidence {field} integrity differs")
    return expected


def validate_identity_compatibility(original_identity: Mapping[str, Any],
                                    target_identity: Mapping[str, Any]) -> dict[str, Any]:
    """Compare frozen scientific fields; report, never conceal, renamed identities."""
    if not isinstance(original_identity, Mapping) or not isinstance(target_identity, Mapping):
        raise MigrationError("Original and target checkpoint identities are required")
    for name in IMMUTABLE_IDENTITY_FIELDS:
        if name not in original_identity or name not in target_identity:
            raise MigrationError(f"Missing immutable checkpoint identity: {name}")
        if content_hash(original_identity[name]) != content_hash(target_identity[name]):
            raise MigrationError(f"Immutable checkpoint identity changed: {name}")
    changed = []
    for name in set(original_identity) | set(target_identity):
        if (name in original_identity) != (name in target_identity) or (
                name in original_identity and content_hash(original_identity[name]) != content_hash(target_identity[name])):
            if name not in MIGRATABLE_IDENTITY_FIELDS:
                raise MigrationError(f"Undeclared checkpoint identity change: {name}")
            changed.append(name)
    return {"scientific_identity_unchanged": True, "changed_identity_fields": sorted(changed),
            "source_profile": _profile(original_identity.get("execution_profile")).to_dict(),
            "target_profile": _profile(target_identity.get("execution_profile")).to_dict()}


def _source_bindings(plan: Mapping[str, Any]) -> None:
    hashes = plan.get("source_file_hashes")
    required = set(REQUIRED_SOURCE_FILES)
    for settings in plan.get("profiles", {}).values():
        loss_implementation = _profile(settings).to_dict().get("loss_implementation")
        helper = LOSS_HELPER_FILES.get(loss_implementation)
        if helper is not None:
            required.add(helper)
        if loss_implementation == "spatial_scheduled_ce":
            required.add(LOSS_HELPER_FILES["spatial_fused_backward"])
    if not isinstance(hashes, Mapping) or required - hashes.keys():
        raise MigrationError("Benchmark lacks executed source-file bindings")
    for name, digest in hashes.items():
        relative = Path(name)
        path = ROOT / relative
        if (relative.is_absolute() or ".." in relative.parts or path.is_symlink()
                or not path.is_file() or file_hash(path) != digest):
            raise MigrationError("Benchmark executed source bytes differ: " + str(name))


def _loss_execution_binding(row: Mapping[str, Any], settings: Mapping[str, Any],
                            source_hashes: Mapping[str, Any] | None = None) -> None:
    """Old spatial-CE receipts need no new field; new kernels do."""
    expected = settings.get("loss_implementation", "spatial_ce")
    for field in ("runtime", "actual_execution_policy", "longest_sequence_probe"):
        evidence = row.get(field, {})
        if not isinstance(evidence, Mapping) or evidence.get("loss_implementation", "spatial_ce") != expected:
            raise MigrationError("Actual benchmark loss execution evidence differs")
        if expected in ("spatial_fused_backward", "spatial_scheduled_ce"):
            digest = (source_hashes or {}).get(LOSS_HELPER_FILES[expected])
            if not digest or evidence.get("loss_helper_sha256") != digest:
                raise MigrationError("Actual loaded spatial-loss helper differs from frozen source")


def validate_execution_benchmark(report: Mapping[str, Any], source: str,
                                 profile: SFTExecutionProfile | Mapping[str, Any], *,
                                 plan: Mapping[str, Any] | None = None,
                                 profile_name: str | None = None,
                                 recovery_report: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Require measured source/profile, numerical, RNG and frozen speed evidence.

    A passed technical screen permits a separately recorded migration. It does
    not certify identical eventual scientific results or full-fit throughput.
    """
    chosen = _profile(profile).to_dict()
    receipt_hash = _verified_hash(report, "receipt_hash")
    if report.get("schema_version") != "isolated_sft_execution_benchmark_v2" or report.get("status") != "MEASURED":
        raise MigrationError("Actual measured SFT benchmark evidence is required")
    plan = plan if plan is not None else report.get("plan")
    if not isinstance(plan, Mapping):
        raise MigrationError("Hash-verified benchmark plan is required for source/profile binding")
    plan_hash = _verified_hash(plan, "plan_hash")
    if (report.get("plan_hash") != plan_hash or plan.get("source_code_commit") != source
            or plan.get("consumer_bindings_valid") is not True or plan.get("binding_errors") != []
            or plan.get("source_tree_validation", {}).get("errors") != []):
        raise MigrationError("Benchmark consumer/source binding differs")
    _source_bindings(plan)
    names = [name for name, value in plan.get("profiles", {}).items() if _profile(value).to_dict() == chosen]
    if profile_name is None:
        if len(names) != 1:
            raise MigrationError("Benchmark does not uniquely identify the requested execution profile")
        profile_name = names[0]
    # Older v2 plans compared against the eager baseline implicitly. New plans
    # can name the already accepted production profile as their reference so a
    # candidate must demonstrate an incremental gain over the actual parent.
    reference_name = plan.get("reference_profile", "baseline")
    if not isinstance(reference_name, str) or reference_name not in plan.get("profiles", {}):
        raise MigrationError("Frozen benchmark reference profile is missing")
    reference_profile = _profile(plan["profiles"][reference_name]).to_dict()
    if profile_name not in names or profile_name == reference_name:
        raise MigrationError("A compared measured optimization candidate is required")
    if chosen["loss_implementation"] == "spatial_scheduled_ce" and (
            reference_profile["loss_implementation"] != "spatial_fused_backward"
            or reference_profile["gradient_checkpointing"] is not False
            or any(chosen[key] != value for key, value in reference_profile.items()
                   if key != "loss_implementation")):
        raise MigrationError("Scheduled spatial CE requires the unchanged accepted spatial-backward reference")
    measured = report.get("profiles", {})
    comparison = report.get("comparisons", {}).get(profile_name, {})
    if comparison.get("reference_profile", "baseline") != reference_name:
        raise MigrationError("Compared benchmark reference differs from the frozen plan")
    baseline, candidate = measured.get(reference_name, {}), measured.get(profile_name, {})
    if (baseline.get("status") != "MEASURED" or candidate.get("status") != "MEASURED"
            or baseline.get("finite") is not True or candidate.get("finite") is not True
            or baseline.get("plan_hash") != plan_hash or candidate.get("plan_hash") != plan_hash
            or _profile(candidate.get("settings")).to_dict() != chosen
            or _profile(candidate.get("runtime", {}).get("execution_profile")).to_dict() != chosen
            or reference_name not in plan.get("profiles", {})
            or _profile(baseline.get("settings")).to_dict() != reference_profile
            or _profile(baseline.get("runtime", {}).get("execution_profile")).to_dict() != reference_profile
            or baseline.get("profile") != reference_name or candidate.get("profile") != profile_name
            or baseline.get("hardware") != candidate.get("hardware")
            or baseline.get("packages") != candidate.get("packages")):
        raise MigrationError("Benchmark profile was not actually measured as requested")
    _loss_execution_binding(baseline, reference_profile, plan["source_file_hashes"])
    _loss_execution_binding(candidate, chosen, plan["source_file_hashes"])
    required_flags = ("numerical_screen_passed", "speedup_threshold_passed",
                      "final_rng_states_identical", "losses_within_declared_tolerance")
    if (comparison.get("technical_gate") != "PASSED" or comparison.get("invariant_failures") != []
            or any(comparison.get(flag) is not True for flag in required_flags)
            or baseline.get("initial_rng_hash") != candidate.get("initial_rng_hash")
            or baseline.get("final_rng_hash") != candidate.get("final_rng_hash")):
        raise MigrationError("Benchmark numerical, speed or RNG screen did not pass")
    tolerances = plan.get("technical_tolerances", {})
    limits = {"loss_atol": 1e-5, "loss_rtol": 1e-4,
              "gradient_relative_l2": 1e-3, "update_relative_l2": 1e-3}
    import math
    if (tolerances.get("require_identical_rng_states") is not True
            or any(type(tolerances.get(key)) not in (int, float) or not math.isfinite(tolerances[key])
                   or not 0 <= tolerances[key] <= bound for key, bound in limits.items())
            or tolerances.get("gradient_cosine_floor", -1) < 0.999
            or tolerances.get("minimum_speedup", 0) < 1.05):
        raise MigrationError("Benchmark acceptance bounds were weakened")
    speed_rule = plan.get("speedup_rule", tolerances.get("speedup_rule"))
    if (speed_rule != "minimum speedup across corresponding logical update pairs in every repetition"
            or plan.get("paired_repetitions", 0) < 2 or plan.get("measured_updates", 0) < 5
            or plan.get("warmup_updates", 0) < 1):
        raise MigrationError("Prospective paired speed measurement design is missing")
    for name, key in (("gradients", "gradient_relative_l2"), ("adapter_updates", "update_relative_l2")):
        row = comparison.get(name, {})
        if (row.get("compatible") is not True or type(row.get("relative_l2")) not in (int, float)
                or not math.isfinite(row["relative_l2"]) or not 0 <= row["relative_l2"] <= tolerances[key]):
            raise MigrationError("Benchmark compared numerical tensors exceed the frozen bounds")
    if (comparison["gradients"].get("cosine", -1) < tolerances["gradient_cosine_floor"]
            or comparison.get("minimum_paired_update_speedup", 0) < tolerances["minimum_speedup"]):
        raise MigrationError("Benchmark compared gradients or speed exceed the frozen bounds")
    repetitions = comparison.get("repetitions")
    if not isinstance(repetitions, list) or len(repetitions) != plan["paired_repetitions"]:
        raise MigrationError("Every paired repetition requires actual comparison evidence")
    runs = report.get("profile_runs", {})
    paired_speedups = []
    for repeat in repetitions:
        if (repeat.get("technical_gate") != "PASSED" or repeat.get("invariant_failures") != []
                or any(repeat.get(flag) is not True for flag in required_flags)
                or repeat.get("reference_profile", "baseline") != reference_name):
            raise MigrationError("A paired repetition failed its numerical, speed or RNG screen")
        reference_run, candidate_run = (runs.get(repeat.get(key), {})
                                        for key in ("reference_run_id", "candidate_run_id"))
        if (reference_run.get("status") != "MEASURED" or candidate_run.get("status") != "MEASURED"
                or reference_run.get("plan_hash") != plan_hash or candidate_run.get("plan_hash") != plan_hash
                or reference_run.get("profile") != reference_name or candidate_run.get("profile") != profile_name
                or _profile(reference_run.get("settings")).to_dict() != reference_profile
                or _profile(candidate_run.get("settings")).to_dict() != chosen
                or _profile(reference_run.get("runtime", {}).get("execution_profile")).to_dict() != reference_profile
                or _profile(candidate_run.get("runtime", {}).get("execution_profile")).to_dict() != chosen
                or reference_run.get("finite") is not True or candidate_run.get("finite") is not True
                or reference_run.get("sample_ids") != candidate_run.get("sample_ids")
                or reference_run.get("tokenized_records_hash") != candidate_run.get("tokenized_records_hash")
                or reference_run.get("initial_adapter_hash") != candidate_run.get("initial_adapter_hash")
                or reference_run.get("initial_rng_hash") != candidate_run.get("initial_rng_hash")
                or reference_run.get("final_rng_hash") != candidate_run.get("final_rng_hash")
                or reference_run.get("hardware") != candidate_run.get("hardware")
                or reference_run.get("packages") != candidate_run.get("packages")):
            raise MigrationError("Paired measurements changed execution inputs or hardware")
        _loss_execution_binding(reference_run, reference_profile, plan["source_file_hashes"])
        _loss_execution_binding(candidate_run, chosen, plan["source_file_hashes"])
        first, second = reference_run.get("wall_seconds_per_update"), candidate_run.get("wall_seconds_per_update")
        if (not isinstance(first, list) or not isinstance(second, list)
                or len(first) != plan["measured_updates"] or len(second) != len(first)
                or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in [*first, *second])):
            raise MigrationError("Paired update timing evidence is incomplete")
        actual_speedups = [left / right for left, right in zip(first, second)]
        if (repeat.get("paired_update_speedups") != actual_speedups
                or repeat.get("minimum_paired_update_speedup") != min(actual_speedups)
                or min(actual_speedups) < tolerances["minimum_speedup"]):
            raise MigrationError("Paired update speed evidence differs or missed the threshold")
        paired_speedups.extend(actual_speedups)
    if comparison.get("minimum_paired_update_speedup") != min(paired_speedups):
        raise MigrationError("Aggregate paired speed bound differs from its measurements")
    condition = plan.get("condition", {})
    if not all(key in condition for key in ("family", "model", "level", "spec")):
        raise MigrationError("Benchmark condition identity is missing")
    acceptance = {"accepted": True, "source_commit": source, "profile_name": profile_name,
                  "reference_profile": reference_name, "reference_execution_profile": reference_profile,
                  "execution_profile": chosen, "configuration_hash": plan.get("configuration_hash"),
                  "handoff_certificate_hash": plan.get("handoff_certificate_hash"),
                  "condition": {key: condition[key] for key in ("family", "model", "level")},
                  "benchmark_report_hash": receipt_hash, "benchmark_plan_hash": plan_hash,
                  "target_hardware": candidate.get("hardware"), "hardware": candidate.get("hardware"),
                  "packages": candidate.get("packages"), "recovery_validated": False,
                  "scientific_equivalence_established": False}
    recovery = validate_recovery_evidence(report, plan, recovery_report=recovery_report)
    if (recovery.get("recovery_validated") is True
            and _profile(recovery["original_checkpoint_identity"].get("execution_profile")).to_dict() != reference_profile):
        raise MigrationError("Measured reference is not the original checkpoint execution profile")
    acceptance.update(recovery)
    acceptance["acceptance_hash"] = content_hash(acceptance)
    return acceptance


def validate_recovery_evidence(report: Mapping[str, Any], plan: Mapping[str, Any], *,
                               recovery_report: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Read explicit checkpoint-backed measurements; fresh adapters are distinct."""
    binding = plan.get("recovery_checkpoint")
    if binding is None:
        if recovery_report is not None:
            raise MigrationError("Recovery evidence is not bound in the measured benchmark plan")
        return {"recovery_validated": False, "original_checkpoint_hash": None, "recovery_report_hash": None}
    if not isinstance(binding, Mapping) or not all(key in binding for key in ("checkpoint_hash", "identity", "progress", "files")):
        raise MigrationError("Benchmark recovery checkpoint binding is incomplete")
    profiles = report.get("profile_runs") or report.get("profiles", {})
    flags = ("full_state_restore_verified", "adapter_verified", "optimizer_verified", "scheduler_verified",
             "rng_verified", "progress_verified")
    if not profiles or any(not isinstance(row.get("recovery_checkpoint"), Mapping)
                           or any(row["recovery_checkpoint"].get(flag) is not True for flag in flags)
                           or any(row["recovery_checkpoint"].get(key) != binding[key]
                                  for key in ("checkpoint_hash", "identity", "progress", "files"))
                           for row in profiles.values()):
        raise MigrationError("Actual full-state restoration was not verified for every measured profile")
    if recovery_report is not None:
        if recovery_report.get("schema_version") == "isolated_sft_execution_benchmark_v2":
            recovery_hash = _verified_hash(recovery_report, "receipt_hash")
            if recovery_hash != report["receipt_hash"]:
                raise MigrationError("Recovery benchmark differs from the accepted measured continuation")
        else:
            recovery_hash = _verified_hash(recovery_report, "report_hash")
            if (recovery_report.get("status") != "PASSED"
                    or recovery_report.get("checkpoint_hash") != binding["checkpoint_hash"]
                    or recovery_report.get("full_state_restore_verified") is not True):
                raise MigrationError("Separate checkpoint recovery report did not pass")
    else:
        recovery_hash = report["receipt_hash"]
    return {"recovery_validated": True, "original_checkpoint_hash": binding["checkpoint_hash"],
            "original_checkpoint_identity": dict(binding["identity"]),
            "original_checkpoint_progress": dict(binding["progress"]),
            "original_checkpoint_files": dict(binding["files"]), "recovery_report_hash": recovery_hash}


def migrate_checkpoint(originaldir: str | Path, targetdir: str | Path,
                       oldidentity: Mapping[str, Any], newidentity: Mapping[str, Any],
                       profile: SFTExecutionProfile | Mapping[str, Any],
                       benchmark_report: Mapping[str, Any], *,
                       benchmark_plan: Mapping[str, Any] | None = None,
                       recovery_report: Mapping[str, Any] | None = None,
                       source_runtime: Mapping[str, Any] | None = None,
                       target_runtime: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Publish a new lineage checkpoint; its normal artifact hook is a Drive barrier."""
    from shared_procedures.checkpointing import load_checkpoint, save_checkpoint, verify_checkpoint, _same_state
    from shared_procedures.storage import atomic_write_json
    source_path, target_path = Path(originaldir).resolve(), Path(targetdir).resolve()
    if target_path.exists() or target_path == source_path or source_path in target_path.parents:
        raise MigrationError("Migration requires a new checkpoint outside the immutable parent namespace")
    compatibility = validate_identity_compatibility(oldidentity, newidentity)
    chosen = _profile(profile).to_dict()
    if compatibility["target_profile"] != chosen:
        raise MigrationError("Target checkpoint identity does not bind the chosen execution profile")
    accepted = validate_execution_benchmark(benchmark_report, newidentity["code_revision"], profile,
                                           plan=benchmark_plan, recovery_report=recovery_report)
    if (accepted["configuration_hash"] != newidentity["configuration_hash"]
            or accepted["handoff_certificate_hash"] != newidentity["handoff_certificate_hash"]
            or accepted["condition"] != {"family": newidentity["family"], "model": newidentity["model_id"], "level": newidentity["level"]}
            or accepted["recovery_validated"] is not True):
        raise MigrationError("Benchmark does not bind this target fit and verified parent recovery state")
    original_manifest = verify_checkpoint(source_path, dict(oldidentity))
    if (original_manifest["checkpoint_hash"] != accepted["original_checkpoint_hash"]
            or original_manifest["identity"] != accepted["original_checkpoint_identity"]
            or original_manifest["progress"] != accepted["original_checkpoint_progress"]
            or original_manifest["files"] != accepted["original_checkpoint_files"]):
        raise MigrationError("Recovered parent differs from the measured original checkpoint")
    original = load_checkpoint(source_path, dict(oldidentity))
    progress = original["progress"]
    if (type(progress.get("sampler_position")) is not int
            or progress["sampler_position"] != progress["completed_steps"] * 16
            or progress.get("example_presentations", progress["sampler_position"]) != progress["sampler_position"]):
        raise MigrationError("Parent recovery state is not a whole optimizer boundary")
    inherited = deepcopy(original)
    inherited["progress"] = {**inherited["progress"], "execution_profile": chosen}
    old_runtime = dict(source_runtime if source_runtime is not None else original.get("runtime", {}))
    new_runtime = dict(target_runtime if target_runtime is not None else accepted["target_hardware"] or {})
    lineage = {"schema_version": "sft_explicit_runtime_migration_v1", "status": "VERIFIED_PARENT_STATE",
               "parent_identity": dict(oldidentity), "target_identity": dict(newidentity),
               "parent_checkpoint_hash": original_manifest["checkpoint_hash"],
               "parent_checkpoint_files": original_manifest["files"],
               "parent_progress": dict(progress), "source_runtime": old_runtime, "target_runtime": new_runtime,
               "execution_profile": chosen, "compatibility": compatibility,
               "benchmark_acceptance": accepted, "full_training_state_inherited": True,
               "hardware_change_recorded": {"source": old_runtime, "target": new_runtime},
               "parent_modified": False, "scientific_equivalence_established": False}
    lineage["migration_hash"] = content_hash(lineage)
    inherited["runtime"] = {**new_runtime, "migration_hash": lineage["migration_hash"],
                            "parent_runtime": old_runtime, "execution_profile": chosen}
    for key in ("adapter", "optimizer", "scheduler", "rng"):
        if not _same_state(original[key], inherited[key]):
            raise MigrationError("Inherited full training state changed: " + key)
    unchanged_progress = {key: value for key, value in inherited["progress"].items() if key != "execution_profile"}
    if not _same_state({key: value for key, value in progress.items() if key != "execution_profile"}, unchanged_progress):
        raise MigrationError("Inherited sampler/exposure state changed")
    saved = save_checkpoint(target_path, dict(newidentity), inherited)
    loaded = load_checkpoint(target_path, dict(newidentity))
    for key in ("adapter", "optimizer", "scheduler", "rng", "progress"):
        if not _same_state(inherited[key], loaded[key]):
            raise MigrationError("Migrated full-state readback differs: " + key)
    if verify_checkpoint(source_path, dict(oldidentity)) != original_manifest:
        raise MigrationError("Immutable parent changed during migration")
    receipt = {**lineage, "status": "MIGRATED", "target_checkpoint_hash": saved["checkpoint_hash"],
               "target_checkpoint_path": str(target_path), "exact_state_readback_verified": True,
               "drive_barrier": "Required synchronous caller artifact hook before continuation"}
    receipt["receipt_hash"] = content_hash(receipt)
    atomic_write_json(target_path.parent / (target_path.name + "_migration.json"), receipt)
    return receipt
