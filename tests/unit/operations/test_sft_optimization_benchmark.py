"""Benchmark guard fixtures are not measurements of production model speed."""
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "benchmark_sft_optimizations.py"
SPEC = importlib.util.spec_from_file_location("sft_optimization_benchmark_guards", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def static_inputs(tmp_path, monkeypatch):
    commit = "a" * 40
    config = {"runtime": {"implementation_revision": commit}}
    task = {"family": "rot13", "model_id": "7b", "level": 1,
            "spec": {"dataset_hash": "b" * 64}, "dataset": {"train_path": "/unread/corpus"}}
    values = {"config": config, "certificate": {"certificate_hash": "c" * 64},
              "bundle": {"datasets": [task["dataset"]]}, "worklist": [task]}
    argv = []
    for name, value in values.items():
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(value))
        argv.extend(["--" + name, str(path)])
    argv.extend(["--output", str(tmp_path / "benchmark")])
    monkeypatch.setattr(MODULE, "validate_configuration", lambda value, runtime: value)
    monkeypatch.setattr(MODULE, "configuration_hash", MODULE.content_hash)
    monkeypatch.setattr(MODULE, "code_revision", lambda: commit)
    monkeypatch.setattr(MODULE, "source_tree_state", lambda: {"mode": "declared_test_fixture", "errors": []})
    def stale(*args, **kwargs):
        raise ValueError("SFT handoff configuration/source compatibility differs")
    monkeypatch.setattr(MODULE, "certified_tasks", stale)
    return argv


def test_plan_only_reports_stale_certificate_without_reading_corpus_or_starting_gpu(tmp_path, monkeypatch):
    argv = static_inputs(tmp_path, monkeypatch)
    def forbidden(*args, **kwargs):
        pytest.fail("Plan-only must not read production corpus or start a GPU profile")
    monkeypatch.setattr(MODULE, "profile_run", forbidden)
    monkeypatch.setattr(MODULE, "load_dataset_bundle", forbidden)
    assert MODULE.main([*argv, "--plan-only"]) == 0
    plan = json.loads((tmp_path / "benchmark" / "plan.json").read_text())
    assert plan["status"] == "PLANNED_PENDING_GPU"
    assert plan["consumer_bindings_valid"] is False
    assert "compatibility differs" in plan["binding_errors"][0]
    assert plan["approved_for_production"] is False
    assert plan["official_training_state_modified"] is False
    assert len(plan["selected_record_indices"]) == 48
    assert plan["selected_record_indices"] == MODULE.sft_fit.exposure_order(10000)[:48]


def test_gpu_path_cannot_ignore_stale_source_certificate(tmp_path, monkeypatch):
    argv = static_inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(MODULE.subprocess, "run", lambda *a, **k: pytest.fail("No profile may launch"))
    with pytest.raises(ValueError, match="Recertify"):
        MODULE.main(argv)


def test_matching_certificate_cannot_authorize_dirty_executed_source(tmp_path, monkeypatch):
    argv = static_inputs(tmp_path, monkeypatch)
    worklist = json.loads((tmp_path / "worklist.json").read_text())
    monkeypatch.setattr(MODULE, "certified_tasks", lambda *a, **k: worklist)
    monkeypatch.setattr(MODULE, "source_tree_state", lambda: {
        "mode": "git", "errors": ["Executed benchmark/training source has uncommitted changes"]})
    plan, _, _ = MODULE.build_plan(MODULE.arguments(argv))
    assert plan["consumer_bindings_valid"] is False
    assert "uncommitted" in plan["binding_errors"][0]


def measured_fixture():
    return {"status": "MEASURED", "plan_hash": "frozen", "finite": True,
            "tokenized_records_hash": "exact mask and token IDs", "data": {"dataset_hash": "frozen"},
            "initial_adapter_hash": "initial", "initial_rng_hash": "rng0", "final_rng_hash": "rng1",
            "losses": [1.0, 2.0], "median_update_seconds": 10.0, "wall_seconds_per_update": [10.0, 10.0]}


def declared_bounds():
    return {"loss_atol": 1e-5, "loss_rtol": 1e-4, "gradient_relative_l2": 1e-3,
            "update_relative_l2": 1e-3, "gradient_cosine_floor": 0.999,
            "minimum_speedup": 1.05, "require_identical_rng_states": True}


def state_fixture():
    torch = pytest.importorskip("torch")
    return {"gradients": {"lora_B": torch.tensor([0.1, -0.2])},
            "deltas": {"lora_A": torch.tensor([0.002, -0.003])}}


def test_exact_numerical_match_and_measured_gain_remain_a_technical_screen():
    baseline = measured_fixture()
    candidate = {**baseline, "median_update_seconds": 8.0, "wall_seconds_per_update": [8.0, 8.0]}
    state = state_fixture()
    result = MODULE.compare_profiles(baseline, candidate, state, state, declared_bounds())
    assert result["technical_gate"] == "PASSED"
    assert result["measured_speedup"] == 1.25
    assert result["gradients"]["bitwise_equal"] is True
    assert result["approved_for_production"] is False
    assert result["scientific_equivalence_established"] is False


def test_noisy_median_speedup_is_not_sufficient_for_a_pass():
    baseline = measured_fixture()
    candidate = {**baseline, "median_update_seconds": 9.0, "wall_seconds_per_update": [7.0, 11.0]}
    state = state_fixture()
    result = MODULE.compare_profiles(baseline, candidate, state, state, declared_bounds())
    assert result["measured_speedup"] > declared_bounds()["minimum_speedup"]
    assert result["observed_minimum_speedup"] < 1.0
    assert result["technical_gate"] == "FAILED"
    assert result["full_fit_time_extrapolation_supported"] is False


@pytest.mark.parametrize("field,value", [
    ("tokenized_records_hash", "different completion mask"),
    ("initial_adapter_hash", "different initialization"),
    ("final_rng_hash", "different dropout sequence"),
    ("losses", [1.1, 2.0]),
])
def test_speed_cannot_compensate_for_changed_data_rng_or_losses(field, value):
    baseline = measured_fixture()
    candidate = {**baseline, "median_update_seconds": 1.0, field: value}
    state = state_fixture()
    result = MODULE.compare_profiles(baseline, candidate, state, state, declared_bounds())
    assert result["technical_gate"] == "FAILED"
    assert result["numerical_screen_passed"] is False


def test_gradient_or_update_drift_rejects_fast_candidate():
    torch = pytest.importorskip("torch")
    baseline = measured_fixture()
    candidate = {**baseline, "median_update_seconds": 1.0}
    state = state_fixture()
    different = {"gradients": {"lora_B": torch.tensor([0.2, -0.2])}, "deltas": state["deltas"]}
    result = MODULE.compare_profiles(baseline, candidate, state, different, declared_bounds())
    assert result["technical_gate"] == "FAILED"
    assert result["gradients"]["relative_l2"] > declared_bounds()["gradient_relative_l2"]


def test_nonfinite_diagnostics_and_missing_measurements_cannot_pass():
    torch = pytest.importorskip("torch")
    vector = MODULE.vector_comparison({"a": torch.tensor([1.0])}, {"a": torch.tensor([float("nan")])})
    assert vector["compatible"] is False
    result = MODULE.compare_profiles(measured_fixture(), {"status": "BLOCKED"}, {}, {}, declared_bounds())
    assert result["technical_gate"] == "FAILED"


def test_unbounded_workload_or_infinite_tolerances_are_rejected():
    required = ["--" + name for name in ("config", "certificate", "bundle", "worklist", "output")]
    argv = [part for name in required for part in (name, "fixture")]
    with pytest.raises(SystemExit):
        MODULE.arguments([*argv, "--updates", "1000"])
    with pytest.raises(SystemExit):
        MODULE.arguments([*argv, "--gradient-relative-l2", "inf"])


def test_exact_tensor_hash_preserves_bf16_bits():
    torch = pytest.importorskip("torch")
    first = {"lora_A": torch.tensor([1.0, 2.0], dtype=torch.bfloat16)}
    same = {"lora_A": first["lora_A"].clone()}
    changed = {"lora_A": torch.tensor([1.0, 2.125], dtype=torch.bfloat16)}
    assert MODULE.tensor_digest(first) == MODULE.tensor_digest(same)
    assert MODULE.tensor_digest(first) != MODULE.tensor_digest(changed)
