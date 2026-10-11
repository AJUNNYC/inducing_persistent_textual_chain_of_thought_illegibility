"""Declared CPU fixtures exercise migration guards, never live GPU approval."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from shared_procedures.checkpointing import _same_state, load_checkpoint, save_checkpoint
from shared_procedures.identity import content_hash, file_hash
from training.sft.execution_profile import SFTExecutionProfile
from training.sft.individual_fit.sft_fit import SFTFitSpec, capture_rng, restore_rng
from training.sft.runtime_migration import (
    IMMUTABLE_IDENTITY_FIELDS, REQUIRED_SOURCE_FILES, ROOT, MigrationError,
    migrate_checkpoint, validate_execution_benchmark, validate_identity_compatibility,
)


def identities():
    old = {**SFTFitSpec("rot13", "7b", 1, "a" * 40, "a" * 40, "b" * 64, "c" * 64,
                       "d" * 64, "e" * 64).identity(),
           "code_revision": "f" * 40, "dependency_hash": "1" * 64}
    new = {**old, "code_revision": "2" * 40, "configuration_hash": "3" * 64,
           "handoff_certificate_hash": "4" * 64,
           "execution_profile": SFTExecutionProfile(cache_collated_inputs=True).to_dict()}
    return old, new


def rehash(report):
    plan = report["plan"]
    plan["plan_hash"] = content_hash({key: value for key, value in plan.items() if key != "plan_hash"})
    report["plan_hash"] = plan["plan_hash"]
    for row in [*report["profiles"].values(), *report["profile_runs"].values()]:
        row["plan_hash"] = plan["plan_hash"]
    report["receipt_hash"] = content_hash({key: value for key, value in report.items() if key != "receipt_hash"})
    return report


def measured_fixture(identity, manifest=None):
    profile = SFTExecutionProfile(cache_collated_inputs=True).to_dict()
    binding = None if manifest is None else {key: deepcopy(manifest[key])
                                            for key in ("checkpoint_hash", "identity", "progress", "files")}
    hardware = {"gpu": "DECLARED_CPU_TEST_GPU_FIXTURE", "vram_bytes": 96 * 2**30, "cuda": "fixture"}
    plan = {"schema_version": "isolated_sft_execution_benchmark_v2", "source_code_commit": identity["code_revision"],
            "consumer_bindings_valid": True, "binding_errors": [], "source_tree_validation": {"errors": []},
            "source_file_hashes": {name: file_hash(ROOT / name) for name in REQUIRED_SOURCE_FILES},
            "configuration_hash": identity["configuration_hash"],
            "handoff_certificate_hash": identity["handoff_certificate_hash"],
            "condition": {"family": "rot13", "model": "7b", "level": 1, "spec": identity},
            "profiles": {"baseline": SFTExecutionProfile().to_dict(), "cuda_cached": profile},
            "warmup_updates": 1, "measured_updates": 5, "paired_repetitions": 2,
            "technical_tolerances": {"loss_atol": 1e-5, "loss_rtol": 1e-4,
                "gradient_relative_l2": 1e-3, "update_relative_l2": 1e-3,
                "gradient_cosine_floor": 0.999, "minimum_speedup": 1.05,
                "require_identical_rng_states": True,
                "speedup_rule": "minimum speedup across corresponding logical update pairs in every repetition"},
            "recovery_checkpoint": binding}
    def row(name, repetition):
        settings = plan["profiles"][name]
        recovery = None if binding is None else {**deepcopy(binding), "full_state_restore_verified": True,
                    **{key: True for key in ("adapter_verified", "optimizer_verified", "scheduler_verified",
                                            "rng_verified", "progress_verified")}}
        return {"status": "MEASURED", "profile": name, "settings": settings,
                "runtime": {"execution_profile": settings}, "finite": True, "hardware": hardware,
                "packages": {"torch": "declared_cpu_fixture"}, "recovery_checkpoint": recovery,
                "sample_ids": ["fixture_" + str(i) for i in range(96)], "tokenized_records_hash": "5" * 64,
                "initial_adapter_hash": "6" * 64, "initial_rng_hash": "7" * 64, "final_rng_hash": "8" * 64,
                "wall_seconds_per_update": [10, 20, 30, 40, 50] if name == "baseline" else [8, 16, 24, 32, 40]}
    rows = {f"{name}_{repeat}": row(name, repeat) for repeat in (1, 2) for name in plan["profiles"]}
    vector = {"compatible": True, "relative_l2": 0.0, "cosine": 1.0, "bitwise_equal": True}
    comparison = {"technical_gate": "PASSED", "numerical_screen_passed": True,
                  "speedup_threshold_passed": True, "final_rng_states_identical": True,
                  "losses_within_declared_tolerance": True, "invariant_failures": [],
                  "gradients": vector, "adapter_updates": vector,
                  "paired_update_speedups": [1.25] * 5, "minimum_paired_update_speedup": 1.25,
                  "observed_minimum_speedup": 0.25, "reference_profile": "baseline"}
    repetitions = [{**deepcopy(comparison), "reference_run_id": f"baseline_{i}",
                    "candidate_run_id": f"cuda_cached_{i}"} for i in (1, 2)]
    report = {"schema_version": "isolated_sft_execution_benchmark_v2", "status": "MEASURED", "plan": plan,
              "profiles": {name: deepcopy(rows[name + "_1"]) for name in plan["profiles"]},
              "profile_runs": rows, "comparisons": {"cuda_cached": {**comparison, "repetitions": repetitions}}}
    return rehash(report)


@pytest.mark.parametrize("field", IMMUTABLE_IDENTITY_FIELDS)
def test_scientific_checkpoint_identity_cannot_change(field):
    old, new = identities()
    new[field] = {"changed": True}
    with pytest.raises(MigrationError, match="Immutable checkpoint identity changed"):
        validate_identity_compatibility(old, new)


def test_legacy_profile_defaults_and_explicit_identity_lineage():
    old, new = identities()
    old["execution_profile"] = {"attention_implementation": "eager", "validated_loss_counts": False,
                                "defer_finite_checks": False}
    compatible = validate_identity_compatibility(old, new)
    assert compatible["source_profile"] == SFTExecutionProfile().to_dict()
    assert compatible["target_profile"] == new["execution_profile"]
    assert compatible["scientific_identity_unchanged"] is True
    new["unspecified_override"] = True
    with pytest.raises(MigrationError, match="Undeclared"):
        validate_identity_compatibility(old, new)


def test_paired_update_screen_accepts_matching_work_despite_unpaired_length_range():
    _, new = identities()
    result = validate_execution_benchmark(measured_fixture(new), new["code_revision"], new["execution_profile"])
    assert result["accepted"] is True
    assert result["hardware"]["gpu"] == "DECLARED_CPU_TEST_GPU_FIXTURE"
    assert result["configuration_hash"] == new["configuration_hash"]
    assert result["condition"] == {"family": "rot13", "model": "7b", "level": 1}
    assert result["recovery_validated"] is False
    assert result["scientific_equivalence_established"] is False


@pytest.mark.parametrize("flag", ("numerical_screen_passed", "speedup_threshold_passed", "final_rng_states_identical"))
def test_failed_repetition_cannot_be_hidden_by_passed_aggregate(flag):
    _, new = identities(); report = measured_fixture(new)
    report["comparisons"]["cuda_cached"]["repetitions"][1][flag] = False
    rehash(report)
    with pytest.raises(MigrationError, match="paired repetition"):
        validate_execution_benchmark(report, new["code_revision"], new["execution_profile"])


def test_stale_source_weakened_bounds_and_lying_paired_speed_reject():
    _, new = identities()
    with pytest.raises(MigrationError, match="source binding"):
        validate_execution_benchmark(measured_fixture(new), "9" * 40, new["execution_profile"])
    report = measured_fixture(new); report["plan"]["technical_tolerances"]["minimum_speedup"] = 1.0; rehash(report)
    with pytest.raises(MigrationError, match="weakened"):
        validate_execution_benchmark(report, new["code_revision"], new["execution_profile"])
    report = measured_fixture(new); report["profile_runs"]["cuda_cached_2"]["wall_seconds_per_update"][0] = 11; rehash(report)
    with pytest.raises(MigrationError, match="Paired update speed"):
        validate_execution_benchmark(report, new["code_revision"], new["execution_profile"])


def test_actual_full_state_migration_preserves_next_optimizer_update_and_parent(tmp_path):
    torch.manual_seed(42)
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.2), torch.nn.Linear(2, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.99)
    def update(m, o, s):
        o.zero_grad(); m(torch.tensor([[1., 2.]])).square().mean().backward(); o.step(); s.step()
    update(model, optimizer, scheduler)
    old, new = identities()
    original = tmp_path / "original" / "checkpoint_000001"
    saved = save_checkpoint(original, old, {"adapter": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "rng": capture_rng(),
        "progress": {"completed_steps": 1, "sampler_position": 16, "example_presentations": 16,
                     "processed_tokens": 112, "total_optimizer_steps": 1250, "exposure_order_hash": "9" * 64},
        "runtime": {"gpu_name": "H100_DECLARED_FIXTURE"}})
    parent_hashes = {path.name: file_hash(path) for path in original.iterdir()}
    report = measured_fixture(new, saved)
    target = tmp_path / "migrated" / "checkpoint_000001"
    receipt = migrate_checkpoint(original, target, old, new, new["execution_profile"], report,
                                 recovery_report=report)
    inherited = load_checkpoint(target, new)
    unchanged = load_checkpoint(original, old)
    assert all(_same_state(inherited[key], unchanged[key]) for key in ("adapter", "optimizer", "scheduler", "rng"))
    assert inherited["progress"]["sampler_position"] == 16
    assert inherited["progress"]["execution_profile"] == new["execution_profile"]
    assert {path.name: file_hash(path) for path in original.iterdir()} == parent_hashes
    assert receipt["parent_checkpoint_hash"] == saved["checkpoint_hash"]
    assert receipt["exact_state_readback_verified"] is True
    restore_rng(unchanged["rng"]); update(model, optimizer, scheduler)
    resumed = deepcopy(model); resumed.load_state_dict(inherited["adapter"])
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.01)
    resumed_scheduler = torch.optim.lr_scheduler.StepLR(resumed_optimizer, 1, 0.99)
    resumed_optimizer.load_state_dict(inherited["optimizer"]); resumed_scheduler.load_state_dict(inherited["scheduler"])
    restore_rng(inherited["rng"]); update(resumed, resumed_optimizer, resumed_scheduler)
    assert _same_state(model.state_dict(), resumed.state_dict())
    assert _same_state(optimizer.state_dict(), resumed_optimizer.state_dict())
    assert _same_state(scheduler.state_dict(), resumed_scheduler.state_dict())


def test_fresh_state_screen_cannot_migrate_official_recovery(tmp_path):
    old, new = identities()
    target = tmp_path / "must_not_exist"
    with pytest.raises(MigrationError, match="verified parent recovery"):
        migrate_checkpoint(tmp_path / "not_read", target, old, new, new["execution_profile"], measured_fixture(new))
    assert not target.exists()
