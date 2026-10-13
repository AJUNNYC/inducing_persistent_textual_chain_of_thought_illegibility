"""Campaign ordering and real-payload baseline barriers; no GPU or Drive calls."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from operations.artifacts import ArtifactService
from shared_procedures.identity import content_hash, file_hash
from shared_procedures.storage import atomic_write_json

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("campaign", ROOT / "scripts/run_14b_sft_campaign.py")
CAMPAIGN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAMPAIGN)


def seal(record, field):
    record[field] = content_hash(record)
    return record


def test_fixed_order_keeps_completed_rot13_ten_outside_new_queue():
    tasks = [{"family": f, "model_id": "14b", "level": l, "run_id": f+str(l)}
             for f in ("rot13", "kazitext") for l in range(1, 11)]
    jobs = CAMPAIGN.ordered_jobs(tasks)
    assert [j["run_id"] for j in jobs] == ["kazitext10"] + [
        f+str(l) for l in range(1, 10) for f in ("rot13", "kazitext")]
    assert len(jobs) == 19 and "rot1310" not in [j["run_id"] for j in jobs]
    with pytest.raises(ValueError, match="exactly twenty"):
        CAMPAIGN.ordered_jobs(tasks[:-1])


def test_job_calls_existing_fit_with_explicit_level_and_no_benchmark():
    args = SimpleNamespace(**{name: name for name in ("config", "certificate", "bundle", "worklist", "preflight")})
    command = CAMPAIGN.training_command(args, {"family": "kazitext", "level": 10}, CAMPAIGN.utc("2026-10-20T00:00:00Z"))
    assert command[1].endswith("scripts/run_certified_sft.py")
    assert command[command.index("--level")+1] == "10"
    assert command[command.index("--model")+1] == "14b"
    assert not any("benchmark" in arg or "migration" in arg for arg in command)


def baseline_fixture(tmp_path, monkeypatch):
    import evaluation.supplemental_gpqa as gpqa
    # Protocol freezing has its own tests. This fixture isolates the barrier's
    # actual file/inventory checks from model loading and native prompt fixtures.
    monkeypatch.setattr(gpqa, "validate_gpqa_protocol", lambda protocol: None)
    config = {"models": {"14b": {"revision": "a"*40, "tokenizer_revision": "b"*40}},
              "runtime": {"implementation_revision": "c"*40}}
    state = {"state_id": "base_14b", "family": "base", "model": "14b",
             "model_revision": "a"*40, "tokenizer_revision": "b"*40}
    items = [{"question_id": f"q{i:03d}", "item_hash": content_hash(i)} for i in range(198)]
    protocol = {"version": gpqa.BASELINE_VERSION, "states": [state],
                "panel": {"variant": "gpqa_diamond", "item_count": 198,
                          "items": items, "panel_hash": "d"*64}}
    seal(protocol, "protocol_hash")
    root = tmp_path / protocol["protocol_hash"]
    root.mkdir()
    protocol_path = tmp_path / "frozen_protocol.json"
    atomic_write_json(protocol_path, protocol)
    atomic_write_json(root / "protocol.json", protocol)
    paths, rows = [root / "protocol.json"], []
    for item in items:
        identity = {"question_id": item["question_id"], "state": state, "seed": 42,
                    "protocol_hash": protocol["protocol_hash"], "execution_mode": "production"}
        raw = seal({"status": "success", "simulation": False, "identity": identity,
                    "request_hash": content_hash(identity), "raw_output": "<answer>A</answer>",
                    "raw_hash": content_hash("<answer>A</answer>")}, "record_hash")
        raw_path = root / "raw/base_14b" / (raw["request_hash"] + ".json")
        atomic_write_json(raw_path, raw)
        row = seal({"protocol_hash": protocol["protocol_hash"], "state_id": "base_14b",
                    "question_id": item["question_id"], "raw_hash": raw["raw_hash"],
                    "simulation": False, "score": {"item_hash": item["item_hash"], "correct": True}}, "record_hash")
        score_path = root / "scores/base_14b" / (item["question_id"] + ".json")
        atomic_write_json(score_path, row)
        paths += [raw_path, score_path]
        rows.append(row)
    report = seal({"status": "complete", "simulation": False, "protocol_hash": protocol["protocol_hash"],
                   "panel_hash": protocol["panel"]["panel_hash"], "variant": "gpqa_diamond", "records": rows,
                   "accuracy_difference_trained_minus_base": None,
                   "summaries": {"base_14b": {"denominator": 198, "observed": 198,
                                               "correct_count": 198, "accuracy": 1}}}, "report_hash")
    report_path = root / "baseline.json"
    atomic_write_json(report_path, report)
    paths.append(report_path)
    service = ArtifactService(tmp_path, tmp_path / "registry.json")
    provenance = {"experiment_id": "gpqa", "branch": "supplemental_gpqa", "model": "14b", "run_id": "base",
                  "stage": "supplemental_capability", "producer": "run_supplemental_gpqa", "source_code_commit": "c"*40,
                  "configuration_hash": protocol["protocol_hash"], "dataset_revision": "e"*40,
                  "model_revision": "a"*40, "seeds": [42]}
    artifacts = {}
    for path in paths:
        record = service.register(path, provenance)
        record.update(sync_status="VERIFIED", drive_file_id="fixture-file", drive_folder_id="fixture-folder")
        artifacts[record["artifact_id"]] = record
    registry_path = root / "drive_manifest.json"
    atomic_write_json(registry_path, {"version": 2, "artifacts": artifacts})
    receipt = seal({"version": "supplemental_gpqa_completion_receipt_v1", "status": "COMPLETE",
                    "drive_status": "VERIFIED", "simulation": False, "protocol_hash": protocol["protocol_hash"],
                    "panel_hash": protocol["panel"]["panel_hash"], "model": "14b", "seed": 42,
                    "source_code_commit": "c"*40, "model_revision": "a"*40, "tokenizer_revision": "b"*40,
                    "report_path": str(report_path), "report_sha256": file_hash(report_path),
                    "report_hash": report["report_hash"], "registry_path": str(registry_path),
                    "registry_sha256": file_hash(registry_path), "summaries": report["summaries"]}, "receipt_hash")
    receipt_path = root / "completion_receipt.json"
    atomic_write_json(receipt_path, receipt)
    args = SimpleNamespace(gpqa_protocol=str(protocol_path), gpqa_output_root=str(tmp_path), gpqa_receipt=str(receipt_path))
    return args, config, root, receipt, artifacts


def test_baseline_barrier_checks_all_actual_payloads_and_drive_records(tmp_path, monkeypatch):
    args, config, _, receipt, _ = baseline_fixture(tmp_path, monkeypatch)
    assert CAMPAIGN.verify_gpqa_barrier(args, config) == receipt


@pytest.mark.parametrize("mutation", ["missing_raw", "missing_score", "wrong_revision", "simulated_receipt", "unverified_drive", "wrong_raw_state"])
def test_complete_flag_cannot_bypass_missing_or_wrong_payloads(tmp_path, monkeypatch, mutation):
    args, config, root, receipt, artifacts = baseline_fixture(tmp_path, monkeypatch)
    if mutation == "missing_raw":
        next((root / "raw/base_14b").glob("*.json")).unlink()
    elif mutation == "missing_score":
        next((root / "scores/base_14b").glob("*.json")).unlink()
    elif mutation == "wrong_revision":
        config["models"]["14b"]["revision"] = "f"*40
    elif mutation == "simulated_receipt":
        receipt.pop("receipt_hash")
        receipt["simulation"] = True
        atomic_write_json(args.gpqa_receipt, seal(receipt, "receipt_hash"))
    elif mutation == "wrong_raw_state":
        path = next((root / "raw/base_14b").glob("*.json"))
        raw = CAMPAIGN.read_json(path)
        raw.pop("record_hash")
        raw["identity"]["state"]["family"] = "rot13"
        atomic_write_json(path, seal(raw, "record_hash"))
    else:
        next(iter(artifacts.values()))["sync_status"] = "PENDING"
        atomic_write_json(root / "drive_manifest.json", {"version": 2, "artifacts": artifacts})
        receipt.pop("receipt_hash")
        receipt["registry_sha256"] = file_hash(root / "drive_manifest.json")
        atomic_write_json(args.gpqa_receipt, seal(receipt, "receipt_hash"))
    with pytest.raises((ValueError, FileNotFoundError)):
        CAMPAIGN.verify_gpqa_barrier(args, config)
