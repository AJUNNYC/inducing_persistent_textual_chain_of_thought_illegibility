"""Coordinator fixtures prove launch guards, not GPU speed or scientific results."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "run_optimized_sft_queue.py"
SPEC = importlib.util.spec_from_file_location("optimized_sft_queue_guards", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def receipt(*, status="MEASURED", passed=True):
    row = {"status": status, "plan": {"source_code_commit": "source", "configuration_hash": "config",
        "handoff_certificate_hash": "certificate", "condition": {"family": "rot13", "model": "7b", "level": 1},
        "consumer_bindings_valid": True, "binding_errors": [],
        "profiles": {"sdpa": {"attention_implementation": "sdpa"}}},
        "comparisons": {"sdpa": {"technical_gate": "PASSED" if passed else "FAILED", "minimum_paired_update_speedup": 1.2}}}
    row["receipt_hash"] = MODULE.content_hash(row)
    return row


def task(level=1):
    return {"family": "rot13", "model_id": "7b", "level": level}


def selection(row, task_row=None):
    return MODULE.select_candidate(row, "config", "certificate", "source", task_row or task(), ["sdpa"])


def approval(**overrides):
    return {"accepted": True, "recovery_validated": True, "condition": {"family": "rot13", "model": "7b", "level": 1},
            "configuration_hash": "config", "handoff_certificate_hash": "certificate", **overrides}


def test_passed_receipt_still_runs_existing_strict_validator(monkeypatch):
    calls = []
    def validate(row, code, profile, **kwargs):
        calls.append((code, profile.attention_implementation, kwargs["profile_name"]))
        return approval()
    monkeypatch.setattr(MODULE, "validate_execution_benchmark", validate)
    assert selection(receipt()) == "sdpa"
    assert calls == [("source", "sdpa", "sdpa")]


def test_approval_cannot_expand_to_another_level(monkeypatch):
    monkeypatch.setattr(MODULE, "validate_execution_benchmark", lambda *a, **k: pytest.fail("Wrong scope must stop before activation"))
    with pytest.raises(ValueError, match="exact consumer condition"):
        selection(receipt(), task(2))


@pytest.mark.parametrize("field,value", [("source_code_commit", "other"), ("configuration_hash", "other"),
    ("handoff_certificate_hash", "other"), ("consumer_bindings_valid", False)])
def test_rehashed_wrong_bindings_cannot_launch(field, value):
    row = receipt()
    row["plan"][field] = value
    row["receipt_hash"] = MODULE.content_hash({k: v for k, v in row.items() if k != "receipt_hash"})
    with pytest.raises(ValueError, match="exact consumer condition"):
        selection(row)


def test_full_recovery_proof_is_mandatory_even_for_fresh_official_base(monkeypatch):
    monkeypatch.setattr(MODULE, "validate_execution_benchmark", lambda *a, **k: approval(recovery_validated=False))
    with pytest.raises(ValueError, match="full recovery evidence"):
        selection(receipt())


def test_failed_speed_or_numerical_candidate_is_never_activated(monkeypatch):
    monkeypatch.setattr(MODULE, "validate_execution_benchmark", lambda *a, **k: pytest.fail("Failed candidate must not activate"))
    assert selection(receipt(passed=False)) is None
    assert selection(receipt(status="BLOCKED")) is None


def test_passed_flag_cannot_bypass_rejecting_strict_validator(monkeypatch):
    def reject(*args, **kwargs):
        raise ValueError("Paired update speed missed the threshold")
    monkeypatch.setattr(MODULE, "validate_execution_benchmark", reject)
    with pytest.raises(ValueError, match="speed"):
        selection(receipt())


def test_receipt_tampering_stops_before_fallback():
    row = receipt()
    row["comparisons"]["sdpa"]["technical_gate"] = "FAILED"
    with pytest.raises(ValueError, match="integrity"):
        selection(row)


def test_fresh_launch_has_exact_condition_and_no_migration_arguments():
    args = SimpleNamespace(config="config", certificate="certificate", bundle="bundle", worklist="worklist",
        family="rot13", model="7b", deadline="9999-12-31T23:59:59.999999+00:00")
    command = MODULE.training_command(args, "scoped_preflight", 3, "measured_report", "sdpa")
    assert command[command.index("--level") + 1] == "3"
    assert command[command.index("--execution-benchmark") + 1] == "measured_report"
    assert command[command.index("--preflight") + 1] == "scoped_preflight"
    assert not any("migration" in value or value == "--recovery-benchmark" for value in command)
    fallback = MODULE.training_command(args, "baseline_preflight", 3)
    assert "--execution-profile" not in fallback


def test_last_publication_is_verified_drive_barrier():
    calls = []
    experiment = SimpleNamespace(publish_permanent=lambda path, metadata: calls.append((path, metadata)))
    MODULE.publish_files(experiment, [Path("payload"), Path("manifest")], {"kind": "isolated_parent"})
    assert calls[0][1]["defer_until_batch_end"] is True
    assert calls[1][1]["defer_until_batch_end"] is False


def test_drive_failure_stops_publication_batch():
    calls = []
    def fail(path, metadata):
        calls.append(path)
        raise RuntimeError("Drive unavailable")
    with pytest.raises(RuntimeError, match="Drive"):
        MODULE.publish_files(SimpleNamespace(publish_permanent=fail), [Path("payload"), Path("manifest")], {})
    assert calls == [Path("payload")]


def test_pause_propagates_sigterm_to_owned_subprocess_group(monkeypatch):
    calls = []
    monkeypatch.setattr(MODULE.os, "killpg", lambda pid, signum: calls.append((pid, signum)))
    MODULE.pause_child(SimpleNamespace(pid=123, poll=lambda: None))
    MODULE.pause_child(SimpleNamespace(pid=456, poll=lambda: 0))
    MODULE.pause_child(None)
    assert calls == [(123, MODULE.signal.SIGTERM)]


def test_completed_condition_cannot_skip_without_matching_durable_decision(tmp_path):
    output = tmp_path / "training" / "sft" / "rot13" / "7b" / "level_01"
    output.mkdir(parents=True)
    (output / "fit_result.json").write_text("{}")
    level_root = tmp_path / "optimization_queue" / "rot13" / "7b" / "level_01"
    level_root.mkdir(parents=True)
    decision = {"condition": {"family": "rot13", "model": "7b", "level": 2}}
    decision["selection_hash"] = MODULE.content_hash(decision)
    MODULE.atomic_write_json(level_root / "selection.json", decision)
    experiment = SimpleNamespace(artifacts=tmp_path, code="source", hash="config")
    with pytest.raises(ValueError, match="exact queue decision"):
        MODULE.completed_condition(experiment, task(), level_root)
