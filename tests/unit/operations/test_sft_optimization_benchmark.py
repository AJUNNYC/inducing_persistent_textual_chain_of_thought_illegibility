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
    config = {"runtime": {"implementation_revision": commit, "dependency_lock_hash": "d" * 64}}
    task = {"family": "rot13", "model_id": "7b", "level": 1,
            "spec": MODULE.sft_fit.SFTFitSpec("rot13", "7b", 1, "d" * 40, "e" * 40,
                "f" * 64, "b" * 64, "a" * 64, "c" * 64).identity(),
            "dataset": {"train_path": "/unread/corpus"}}
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
    assert len(plan["selected_record_indices"]) == 96
    assert plan["selected_record_indices"] == MODULE.sft_fit.exposure_order(10000)[:96]
    assert plan["paired_repetitions"] == 2
    assert plan["reference_profile"] == "baseline"


def test_pilot_plan_binds_one_epoch_order_and_recipe_without_loading_gpu(tmp_path, monkeypatch):
    argv = static_inputs(tmp_path, monkeypatch)
    path = tmp_path / 'worklist.json'
    worklist = json.loads(path.read_text())
    spec = MODULE.sft_fit.SFTFitSpec.from_dict({**worklist[0]['spec'],
        'training_budget': 'one_epoch_pilot_v1'})
    worklist[0]['spec'] = spec.identity()
    path.write_text(json.dumps(worklist))
    monkeypatch.setattr(MODULE, 'certified_tasks', lambda *a, **k: worklist)
    monkeypatch.setattr(MODULE, 'load_dataset_bundle', lambda *a, **k: pytest.fail('Planning must not load a corpus'))
    plan, _, _ = MODULE.build_plan(MODULE.arguments(argv))
    order = MODULE.sft_fit.exposure_order(10000, epochs=1)
    assert len(order) == 10000
    assert plan['recipe']['epochs'] == 1
    assert plan['frozen_exposure_order_hash'] == MODULE.content_hash(order)
    assert plan['selected_record_indices'] == order[:96]
    assert plan['consumer_bindings_valid'] is True
    assert plan['approved_for_production'] is False


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


def test_scheduled_plan_requires_both_loaded_helper_sources_before_gpu_work(tmp_path, monkeypatch):
    argv = static_inputs(tmp_path, monkeypatch)
    worklist = json.loads((tmp_path / "worklist.json").read_text())
    monkeypatch.setattr(MODULE, "certified_tasks", lambda *a, **k: worklist)
    source_root = tmp_path / "declared_source_fixture"
    monkeypatch.setattr(MODULE, "ROOT", source_root)
    argv.extend(["--reference-profile", "spatial_fused_backward_no_gradient_checkpointing", "--profiles",
                 "spatial_fused_backward_no_gradient_checkpointing", "spatial_scheduled_ce_no_gradient_checkpointing"])
    plan, _, _ = MODULE.build_plan(MODULE.arguments(argv))
    assert plan["consumer_bindings_valid"] is False
    assert len([error for error in plan["binding_errors"] if "Selected loss helper source is missing" in error]) == 2
    for name, relative in MODULE.LOSS_HELPER_FILES.items():
        path = source_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Explicit non-executable source-binding fixture: " + name + "\n")
    plan, _, _ = MODULE.build_plan(MODULE.arguments(argv))
    assert plan["consumer_bindings_valid"] is True
    assert all(relative in plan["source_file_hashes"] for relative in MODULE.LOSS_HELPER_FILES.values())
    assert len({plan["source_file_hashes"][relative] for relative in MODULE.LOSS_HELPER_FILES.values()}) == len(MODULE.LOSS_HELPER_FILES)


def measured_fixture():
    return {"status": "MEASURED", "plan_hash": "frozen", "finite": True,
            "profile": "baseline", "settings": dict(MODULE.PROFILE_SETTINGS["baseline"]),
            "actual_execution_policy": {"gradient_checkpointing": True, "cache_collated_inputs": False,
                "attention_implementation": "eager", "use_cache": False, "checkpoint_rng_preserved": True},
            "tokenized_records_hash": "exact mask and token IDs", "data": {"dataset_hash": "frozen"},
            "initial_adapter_hash": "a" * 64, "initial_rng_hash": "0" * 64, "final_rng_hash": "1" * 64,
            "initial_optimizer_state_hash": "2" * 64, "initial_scheduler_state_hash": "3" * 64,
            "final_optimizer_state_hash": "4" * 64, "final_scheduler_state_hash": "5" * 64,
            "final_adapter_hash": "b" * 64, "clipped_gradient_hash": "c" * 64, "adapter_delta_hash": "d" * 64,
            "losses": [1.0, 2.0], "median_update_seconds": 10.0, "wall_seconds_per_update": [10.0] * 5}


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
    candidate = {**baseline, "median_update_seconds": 8.0, "wall_seconds_per_update": [8.0] * 5}
    state = state_fixture()
    result = MODULE.compare_profiles(baseline, candidate, state, state, declared_bounds())
    assert result["technical_gate"] == "PASSED"
    assert result["measured_speedup"] == 1.25
    assert result["gradients"]["bitwise_equal"] is True
    assert result["approved_for_production"] is False
    assert result["scientific_equivalence_established"] is False


def test_noisy_median_speedup_is_not_sufficient_for_a_pass():
    baseline = measured_fixture()
    candidate = {**baseline, "median_update_seconds": 9.0, "wall_seconds_per_update": [7.0, 8.0, 9.0, 10.0, 11.0]}
    state = state_fixture()
    result = MODULE.compare_profiles(baseline, candidate, state, state, declared_bounds())
    assert result["measured_speedup"] > declared_bounds()["minimum_speedup"]
    assert result["observed_minimum_speedup"] < 1.0
    assert result["technical_gate"] == "FAILED"
    assert result["full_fit_time_extrapolation_supported"] is False


@pytest.mark.parametrize("field,value", [("losses", [1.000001, 2.0]),
    ("final_adapter_hash", "e" * 64), ("clipped_gradient_hash", "e" * 64),
    ("adapter_delta_hash", "e" * 64), ("final_optimizer_state_hash", "e" * 64)])
def test_final_copy_gate_rejects_tolerated_losses_or_any_different_state_hash(field, value):
    reference = measured_fixture()
    candidate = {**reference, "median_update_seconds": 8.0, "wall_seconds_per_update": [8.0] * 5, field: value}
    state = state_fixture()
    legacy = MODULE.compare_profiles(reference, candidate, state, state, declared_bounds())
    assert legacy["losses_within_declared_tolerance"] is True
    assert legacy["numerical_screen_passed"] is True
    exact = MODULE.compare_profiles(reference, candidate, state, state,
                                    {**declared_bounds(), "require_exact_state_comparison": True})
    assert exact["technical_gate"] == "FAILED"
    assert exact["exact_state_comparison_required"] is True
    assert exact["exact_state_comparison_passed"] is False


def test_speed_gate_pairs_corresponding_inputs_instead_of_mixing_sequence_lengths():
    baseline = {**measured_fixture(), "wall_seconds_per_update": [8.0, 14.0, 10.0, 16.0, 12.0],
                "median_update_seconds": 12.0}
    candidate = {**baseline, "wall_seconds_per_update": [6.4, 11.2, 8.0, 12.8, 9.6],
                 "median_update_seconds": 9.6}
    state = state_fixture()
    result = MODULE.compare_profiles(baseline, candidate, state, state, declared_bounds())
    assert result["observed_minimum_speedup"] < 1.0
    assert result["minimum_paired_update_speedup"] == pytest.approx(1.25)
    assert result["technical_gate"] == "PASSED"


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


def test_loss_candidates_require_an_explicit_current_reference_first():
    argv = [part for name in ("config", "certificate", "bundle", "worklist", "output")
            for part in ("--" + name, "fixture")]
    for profiles in (["baseline", "flat_ce_no_gradient_checkpointing"],
                     ["baseline", "fused_ce_no_gradient_checkpointing"],
                     ["baseline", "spatial_fused_backward_no_gradient_checkpointing"],
                     ["baseline", "spatial_scheduled_ce_no_gradient_checkpointing"],
                     ["baseline", "causal_contiguous_gradient_ce_no_gradient_checkpointing"],
                     ["baseline", "cached_no_gradient_checkpointing"]):
        with pytest.raises(SystemExit):
            MODULE.arguments([*argv, "--profiles", *profiles])
    with pytest.raises(SystemExit):
        MODULE.arguments([*argv, "--reference-profile", "no_gradient_checkpointing",
                          "--profiles", "flat_ce_no_gradient_checkpointing", "no_gradient_checkpointing"])
    args = MODULE.arguments([*argv, "--reference-profile", "no_gradient_checkpointing", "--profiles",
                             "no_gradient_checkpointing", "flat_ce_no_gradient_checkpointing"])
    assert args.reference_profile == "no_gradient_checkpointing"
    assert args.updates == 5 and args.repetitions == 2 and args.warmup_updates == 1
    cached = MODULE.arguments([*argv, "--reference-profile", "no_gradient_checkpointing", "--profiles",
                               "no_gradient_checkpointing", "cached_no_gradient_checkpointing"])
    assert cached.profiles == ["no_gradient_checkpointing", "cached_no_gradient_checkpointing"]
    with pytest.raises(SystemExit):
        MODULE.arguments([*argv, "--reference-profile", "no_gradient_checkpointing", "--profiles",
                          "no_gradient_checkpointing", "spatial_scheduled_ce_no_gradient_checkpointing"])
    scheduled = MODULE.arguments([*argv, "--reference-profile", "spatial_fused_backward_no_gradient_checkpointing", "--profiles",
                                  "spatial_fused_backward_no_gradient_checkpointing", "spatial_scheduled_ce_no_gradient_checkpointing"])
    assert scheduled.reference_profile == "spatial_fused_backward_no_gradient_checkpointing"
    for wrong_reference in ("no_gradient_checkpointing", "spatial_fused_backward_no_gradient_checkpointing"):
        with pytest.raises(SystemExit):
            MODULE.arguments([*argv, "--reference-profile", wrong_reference, "--profiles", wrong_reference,
                              "causal_contiguous_gradient_ce_no_gradient_checkpointing"])
    causal = MODULE.arguments([*argv, "--reference-profile", "spatial_scheduled_ce_no_gradient_checkpointing", "--profiles",
                               "spatial_scheduled_ce_no_gradient_checkpointing", "causal_contiguous_gradient_ce_no_gradient_checkpointing"])
    assert causal.reference_profile == "spatial_scheduled_ce_no_gradient_checkpointing"


@pytest.mark.parametrize("candidate_name", ["flat_ce_no_gradient_checkpointing", "fused_ce_no_gradient_checkpointing",
    "spatial_fused_backward_no_gradient_checkpointing", "spatial_scheduled_ce_no_gradient_checkpointing",
    "causal_contiguous_gradient_ce_no_gradient_checkpointing"])
def test_loss_mode_is_an_intended_change_but_missing_actual_dispatch_evidence_fails(candidate_name):
    reference = measured_fixture()
    reference.update(profile="no_gradient_checkpointing", settings=dict(MODULE.PROFILE_SETTINGS["no_gradient_checkpointing"]))
    reference["actual_execution_policy"] = {**reference["actual_execution_policy"], "gradient_checkpointing": False}
    reference_name = {"spatial_scheduled_ce_no_gradient_checkpointing": "spatial_fused_backward_no_gradient_checkpointing",
                      "causal_contiguous_gradient_ce_no_gradient_checkpointing": "spatial_scheduled_ce_no_gradient_checkpointing"}.get(candidate_name)
    if reference_name:
        reference.update(profile=reference_name, settings=dict(MODULE.PROFILE_SETTINGS[reference_name]))
        reference["runtime"] = {"loss_helper_sha256": "e" * 64}
        reference["actual_execution_policy"].update(loss_implementation=reference["settings"]["loss_implementation"],
                                                   loss_helper_sha256="e" * 64)
    settings = dict(MODULE.PROFILE_SETTINGS[candidate_name])
    candidate = {**reference, "profile": candidate_name, "settings": settings,
        "actual_execution_policy": {**reference["actual_execution_policy"], "loss_implementation": settings["loss_implementation"]},
        "median_update_seconds": 8.0, "wall_seconds_per_update": [8.0] * 5}
    if settings["loss_implementation"] in MODULE.LOSS_HELPER_FILES:
        candidate["runtime"] = {"loss_helper_sha256": "f" * 64}
        candidate["actual_execution_policy"]["loss_helper_sha256"] = "f" * 64
    state = state_fixture()
    result = MODULE.compare_profiles(reference, candidate, state, state, declared_bounds())
    assert result["technical_gate"] == "PASSED"
    assert result["intended_execution_policy_changes"] == {
        "loss_implementation": {"reference": reference["settings"]["loss_implementation"],
                                 "candidate": settings["loss_implementation"]}}
    candidate["actual_execution_policy"].pop("loss_implementation")
    result = MODULE.compare_profiles(reference, candidate, state, state, declared_bounds())
    assert result["technical_gate"] == "FAILED"
    assert result["execution_policy_failures"] == [candidate_name]


@pytest.mark.parametrize("helper_hash", [None, "invalid", "e" * 64])
@pytest.mark.parametrize("candidate_name", ["spatial_fused_backward_no_gradient_checkpointing",
    "spatial_scheduled_ce_no_gradient_checkpointing", "causal_contiguous_gradient_ce_no_gradient_checkpointing"])
def test_spatial_backward_policy_cannot_omit_or_misrepresent_loaded_helper(helper_hash, candidate_name):
    reference = measured_fixture()
    reference.update(profile="no_gradient_checkpointing", settings=dict(MODULE.PROFILE_SETTINGS["no_gradient_checkpointing"]))
    reference["actual_execution_policy"] = {**reference["actual_execution_policy"], "gradient_checkpointing": False}
    name = candidate_name
    candidate = {**reference, "profile": name, "settings": dict(MODULE.PROFILE_SETTINGS[name]),
        "runtime": {"loss_helper_sha256": "f" * 64},
        "actual_execution_policy": {**reference["actual_execution_policy"],
            "loss_implementation": MODULE.PROFILE_SETTINGS[name]["loss_implementation"], "loss_helper_sha256": helper_hash},
        "median_update_seconds": 8.0, "wall_seconds_per_update": [8.0] * 5}
    state = state_fixture()
    result = MODULE.compare_profiles(reference, candidate, state, state, declared_bounds())
    assert result["technical_gate"] == "FAILED"
    assert result["execution_policy_failures"] == [name]


def test_exact_tensor_hash_preserves_bf16_bits():
    torch = pytest.importorskip("torch")
    first = {"lora_A": torch.tensor([1.0, 2.0], dtype=torch.bfloat16)}
    same = {"lora_A": first["lora_A"].clone()}
    changed = {"lora_A": torch.tensor([1.0, 2.125], dtype=torch.bfloat16)}
    assert MODULE.tensor_digest(first) == MODULE.tensor_digest(same)
    assert MODULE.tensor_digest(first) != MODULE.tensor_digest(changed)


def test_recursive_state_hash_supports_scalar_optimizer_tensors():
    torch = pytest.importorskip("torch")
    first = {"step": torch.tensor(214), "moments": torch.tensor([0.1, 0.2])}
    same = {"moments": first["moments"].clone(), "step": torch.tensor(214)}
    changed = {**same, "step": torch.tensor(215)}
    assert MODULE.state_digest(first) == MODULE.state_digest(same)
    assert MODULE.state_digest(first) != MODULE.state_digest(changed)


def test_recovery_plan_uses_original_sampler_and_rejects_changed_checkpoint_bytes(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from shared_procedures.checkpointing import save_checkpoint
    argv = static_inputs(tmp_path, monkeypatch)
    task = json.loads((tmp_path / "worklist.json").read_text())[0]
    config = json.loads((tmp_path / "config.json").read_text())
    original = {**task["spec"], "code_revision": "b" * 40, "dependency_hash": config["runtime"]["dependency_lock_hash"]}
    checkpoint = tmp_path / "original214"
    order = MODULE.sft_fit.exposure_order(10000)
    state = {"adapter": {}, "optimizer": {}, "scheduler": {}, "rng": MODULE.sft_fit.capture_rng(),
             "progress": {"completed_steps": 214, "sampler_position": 3424, "exposure_order_hash": MODULE.content_hash(order)}}
    save_checkpoint(checkpoint, original, state)
    plan, _, _ = MODULE.build_plan(MODULE.arguments([*argv, "--recovery-checkpoint", str(checkpoint)]))
    assert plan["selected_record_indices"] == order[3424:3520]
    assert plan["official_training_state_read"] is True
    assert plan["official_training_state_modified"] is False
    assert plan["recovery_checkpoint"]["identity"] == original
    current_reference = ["--reference-profile", "no_gradient_checkpointing", "--profiles",
                         "no_gradient_checkpointing", "flat_ce_no_gradient_checkpointing"]
    mismatched, _, _ = MODULE.build_plan(MODULE.arguments([*argv, *current_reference,
                                                         "--recovery-checkpoint", str(checkpoint)]))
    assert any("declared paired reference" in error for error in mismatched["binding_errors"])
    current_checkpoint = tmp_path / "current_no_checkpointing"
    current_identity = {**original, "execution_profile": dict(MODULE.PROFILE_SETTINGS["no_gradient_checkpointing"])}
    save_checkpoint(current_checkpoint, current_identity, state)
    matched, _, _ = MODULE.build_plan(MODULE.arguments([*argv, *current_reference,
                                                      "--recovery-checkpoint", str(current_checkpoint)]))
    assert matched["reference_profile"] == "no_gradient_checkpointing"
    assert not any("declared paired reference" in error for error in matched["binding_errors"])
    with (checkpoint / "full_state.pt").open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="payload bytes changed"):
        MODULE.recovery_binding(checkpoint, task, config)


def test_paired_launch_order_and_all_repetition_gates_are_recorded(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    argv = static_inputs(tmp_path, monkeypatch)
    worklist = json.loads((tmp_path / "worklist.json").read_text())
    monkeypatch.setattr(MODULE, "certified_tasks", lambda *a, **k: worklist)
    launched = []
    def fixture_process(command, **kwargs):
        args = MODULE.arguments(command[2:])
        launched.append(args._profile)
        row = measured_fixture()
        settings = MODULE.PROFILE_SETTINGS[args._profile]
        row.update(profile=args._profile, settings=dict(settings),
            actual_execution_policy={"gradient_checkpointing": settings["gradient_checkpointing"],
                "cache_collated_inputs": settings["cache_collated_inputs"],
                "attention_implementation": settings["attention_implementation"],
                "use_cache": False, "checkpoint_rng_preserved": True,
                "loss_implementation": settings["loss_implementation"]})
        if args._profile != "baseline":
            row.update(median_update_seconds=8.0, wall_seconds_per_update=[8.0] * 5)
        basename = MODULE.run_basename(args._pair, args._repetition, args._profile)
        MODULE.write_json(Path(args.output) / (basename + ".json"), row)
        torch.save(state_fixture(), args._diagnostics)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(MODULE.subprocess, "run", fixture_process)
    assert MODULE.main(argv) == 0
    assert launched == ["baseline", "cuda_input_cache", "cuda_input_cache", "baseline",
                        "baseline", "no_gradient_checkpointing", "no_gradient_checkpointing", "baseline"]
    report = json.loads((tmp_path / "benchmark" / "report.json").read_text())
    assert report["plan"]["plan_hash"] == report["plan_hash"]
    assert len(report["profile_runs"]) == 8
    for row in report["comparisons"].values():
        assert row["technical_gate"] == "PASSED"
        assert len(row["repetitions"]) == 2
        assert all(pair["reference_run_id"] in report["profile_runs"] for pair in row["repetitions"])


@pytest.mark.parametrize("reference,candidate_names", [
    ("no_gradient_checkpointing", ["flat_ce_no_gradient_checkpointing"]),
    ("no_gradient_checkpointing", ["flat_ce_no_gradient_checkpointing", "fused_ce_no_gradient_checkpointing"]),
    ("no_gradient_checkpointing", ["spatial_fused_backward_no_gradient_checkpointing"]),
    ("spatial_fused_backward_no_gradient_checkpointing", ["spatial_scheduled_ce_no_gradient_checkpointing"]),
    ("spatial_scheduled_ce_no_gradient_checkpointing", ["causal_contiguous_gradient_ce_no_gradient_checkpointing"])])
def test_loss_candidates_measure_incremental_gain_against_current_accepted_reference_only(tmp_path, monkeypatch, reference, candidate_names):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    argv = static_inputs(tmp_path, monkeypatch)
    worklist = json.loads((tmp_path / "worklist.json").read_text())
    monkeypatch.setattr(MODULE, "certified_tasks", lambda *a, **k: worklist)
    if reference in ("spatial_fused_backward_no_gradient_checkpointing", "spatial_scheduled_ce_no_gradient_checkpointing"):
        source_root = tmp_path / "declared_source_fixture"
        monkeypatch.setattr(MODULE, "ROOT", source_root)
        for name, relative in MODULE.LOSS_HELPER_FILES.items():
            path = source_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# Explicit non-executable source-binding fixture: " + name + "\n")
    argv.extend(["--reference-profile", reference, "--profiles", reference, *candidate_names])
    launched = []
    def fixture_process(command, **kwargs):
        args = MODULE.arguments(command[2:])
        assert args.reference_profile == reference
        launched.append(args._profile)
        settings = dict(MODULE.PROFILE_SETTINGS[args._profile])
        row = {**measured_fixture(), "profile": args._profile, "settings": settings,
            "actual_execution_policy": {"gradient_checkpointing": False, "cache_collated_inputs": False,
                "attention_implementation": "eager", "use_cache": False, "checkpoint_rng_preserved": True,
                "loss_implementation": settings["loss_implementation"]}}
        if settings["loss_implementation"] in MODULE.LOSS_HELPER_FILES:
            helper_hash = ("f" if args._profile != reference else "e") * 64
            row["runtime"] = {"loss_helper_sha256": helper_hash}
            row["actual_execution_policy"]["loss_helper_sha256"] = helper_hash
        if args._profile != reference:
            row.update(median_update_seconds=8.0, wall_seconds_per_update=[8.0] * 5)
        basename = MODULE.run_basename(args._pair, args._repetition, args._profile)
        MODULE.write_json(Path(args.output) / (basename + ".json"), row)
        torch.save(state_fixture(), args._diagnostics)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(MODULE.subprocess, "run", fixture_process)
    assert MODULE.main(argv) == 0
    assert launched == [name for candidate in candidate_names for name in (reference, candidate, candidate, reference)]
    assert "baseline" not in launched
    report = json.loads((tmp_path / "benchmark" / "report.json").read_text())
    assert report["plan"]["reference_profile"] == reference
    require_exact = "causal_contiguous_gradient_ce_no_gradient_checkpointing" in candidate_names
    assert report["plan"]["technical_tolerances"]["require_exact_state_comparison"] is require_exact
    assert len(report["profile_runs"]) == len(candidate_names) * 4
    for candidate in candidate_names:
        comparison = report["comparisons"][candidate]
        assert comparison["reference_profile"] == reference
        assert comparison["technical_gate"] == "PASSED"
        assert comparison["exact_state_comparison_required"] is require_exact
        assert comparison["minimum_paired_update_speedup"] == 1.25
        assert all(pair["reference_profile"] == reference for pair in comparison["repetitions"])
        assert comparison["intended_execution_policy_changes"] == {"loss_implementation": {
            "reference": MODULE.PROFILE_SETTINGS[reference]["loss_implementation"],
            "candidate": MODULE.PROFILE_SETTINGS[candidate]["loss_implementation"]}}
