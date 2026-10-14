"""CPU campaign safety: exact conditions, baseline reuse, restart and barriers."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import pytest

from evaluation.supplemental_gpqa import (
    GPQAVLLMGenerator, SOURCE_REPOSITORY, SOURCE_REVISION, prepare_gpqa_panel, score_gpqa,
)
from operations.artifacts import ArtifactService
from shared_procedures.identity import content_hash, file_hash
from shared_procedures.storage import atomic_write_json, read_json
from training.sft.individual_fit.sft_fit import TARGET_MODULES

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("gpqa_campaign", ROOT / "scripts/run_gpqa_sft_campaign.py")
CAMPAIGN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAMPAIGN)


def seal(record, field):
    record.pop(field, None)
    record[field] = content_hash(record)
    return record


def inputs(tmp_path):
    rows = [{"Question": f"Campaign fixture {i}?", "Correct Answer": f"right {i}",
             "Incorrect Answer 1": f"wrong one {i}", "Incorrect Answer 2": f"wrong two {i}",
             "Incorrect Answer 3": f"wrong three {i}"} for i in range(198)]
    panel = prepare_gpqa_panel(rows, variant="gpqa_diamond", source={
        "repo_id": SOURCE_REPOSITORY, "revision": SOURCE_REVISION, "filename": "gpqa_diamond.csv",
        "csv_sha256": "c" * 64, "asset_manifest_hash": "d" * 64})
    panel_path = tmp_path / "panel.json"
    atomic_write_json(panel_path, panel)
    declared = []
    for level in range(1, 11):
        path = tmp_path / f"level_{level:02d}" / "final_adapter"
        path.mkdir(parents=True)
        config = {"r": 8, "lora_alpha": 16, "lora_dropout": .05, "target_modules": list(TARGET_MODULES),
                  "base_model_name_or_path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"}
        atomic_write_json(path / "adapter_config.json", config)
        (path / "adapter_model.safetensors").write_bytes(b"CPU integrity fixture, not model weights")
        identity = {"family": "rot13", "model_id": "14b", "level": level,
                    "model_name": config["base_model_name_or_path"],
                    "model_revision": CAMPAIGN.MODEL_REVISION, "tokenizer_revision": CAMPAIGN.MODEL_REVISION,
                    "configuration_hash": content_hash("old config" if level == 10 else "new config"),
                    "code_revision": "2" * 40 if level == 10 else "9" * 40,
                    "execution_mode": "production", "original_training_recipe": "unchanged"}
        manifest = seal({"identity": identity, "checkpoint_hash": content_hash({"level": level}),
                         "files": {p.name: file_hash(p) for p in path.iterdir()}}, "manifest_hash")
        atomic_write_json(path / "manifest.json", manifest)
        atomic_write_json(path.parent / "fit_result.json", {"status": "complete", "identity": identity,
                          "adapter_manifest_hash": manifest["manifest_hash"],
                          "checkpoint_hash": manifest["checkpoint_hash"]})
        declared.append({"level": level, "final_adapter": str(path)})
    manifest_path = tmp_path / "adapters.json"
    atomic_write_json(manifest_path, {"version": CAMPAIGN.MANIFEST_VERSION, "family": "rot13", "model": "14b",
                                    "adapters": list(reversed(declared))})
    return SimpleNamespace(panel=str(panel_path), adapters_manifest=str(manifest_path),
                           model_cache=str(tmp_path / "cache"), output_root=str(tmp_path / "output"),
                           generation_profile="long_32000", save_to_drive=False, validate_only=False)


def test_plan_declares_all_conditions_and_keeps_old_level_ten_identity(tmp_path):
    args = inputs(tmp_path)
    plan, protocols = CAMPAIGN.campaign_plan(args)
    assert [job["state_id"] for job in plan["order"]] == ["base_14b"] + [
        f"rot13_14b_level_{level:02d}" for level in range(1, 11)]
    assert len(protocols) == 11
    assert plan["generation"]["settings"]["max_new_tokens"] == 32000
    assert plan["generation"]["backend"]["max_model_len"] == 40960
    assert plan["generation"]["backend"]["max_num_seqs"] == 8
    old = plan["adapter_bindings"][-1]
    assert old["training_identity"]["code_revision"] == "2" * 40
    assert old["training_identity"]["configuration_hash"] != plan["adapter_bindings"][0]["training_identity"]["configuration_hash"]
    assert all(p["baseline_reuse"]["protocol_hash"] == protocols[0]["protocol_hash"] for p in protocols[1:])
    assert protocols[-1]["states"][1]["training_identity_hash"] == content_hash(old["training_identity"])
    assert plan["primary_suite_changed"] is False and plan["training_changed"] is False


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong_declared_level", "nonfinal", "tampered_weights", "wrong_pin", "kazitext"])
def test_rejects_invalid_complete_matrix_before_launch(tmp_path, monkeypatch, mutation):
    args = inputs(tmp_path)
    manifest = read_json(args.adapters_manifest)
    first = manifest["adapters"][0]
    path = Path(first["final_adapter"])
    if mutation == "missing":
        manifest["adapters"].pop()
    elif mutation == "duplicate":
        manifest["adapters"][1]["level"] = first["level"]
    elif mutation == "wrong_declared_level":
        first["final_adapter"], manifest["adapters"][1]["final_adapter"] = manifest["adapters"][1]["final_adapter"], first["final_adapter"]
    elif mutation == "nonfinal":
        first["final_adapter"] = str(path.parent)
    elif mutation == "tampered_weights":
        (path / "adapter_model.safetensors").write_bytes(b"changed")
    else:
        original = read_json(path / "manifest.json")
        if mutation == "wrong_pin":
            original["identity"]["model_revision"] = "f" * 40
        else:
            original["identity"]["family"] = "kazitext"
        atomic_write_json(path / "manifest.json", seal(original, "manifest_hash"))
        result = read_json(path.parent / "fit_result.json")
        result["identity"] = original["identity"]
        result["adapter_manifest_hash"] = original["manifest_hash"]
        atomic_write_json(path.parent / "fit_result.json", result)
    atomic_write_json(args.adapters_manifest, manifest)
    monkeypatch.setattr(CAMPAIGN.subprocess, "Popen", lambda *a, **k: pytest.fail("GPU child must not start"))
    with pytest.raises(ValueError):
        CAMPAIGN.campaign_plan(args)


def test_commands_use_published_cli_and_the_same_new_baseline(tmp_path):
    args = inputs(tmp_path)
    plan, _ = CAMPAIGN.campaign_plan(args)
    base_path = tmp_path / "base.json"
    command = CAMPAIGN.freeze_command(args, plan["order"][1], tmp_path / "level.json", base_path)
    assert command[:3] == [sys.executable, str(CAMPAIGN.CLI), "freeze"]
    assert command[command.index("--baseline-protocol") + 1] == str(base_path)
    assert command[command.index("--generation-profile") + 1] == "long_32000"
    assert command[command.index("--adapter") + 1].endswith("level_01/final_adapter")
    args.save_to_drive = True
    assert "--save-to-drive" in CAMPAIGN.run_command(args, base_path)


def child_fixture(monkeypatch, args, protocols, *, failure_state=None, signal_during_run=False):
    calls, current = [], {"protocol": None}
    by_state = {p["states"][-1]["state_id"]: p for p in protocols}
    class Child:
        def __init__(self, command, **kwargs):
            self.command, self.received = command, []
            calls.append(command)
            if command[2] in {"freeze", "freeze-base"}:
                output = Path(command[command.index("--output") + 1])
                current["protocol"] = by_state[output.stem]
                atomic_write_json(output, current["protocol"])
            else:
                current["protocol"] = read_json(command[command.index("--protocol") + 1])
        def poll(self):
            return None
        def send_signal(self, signum):
            self.received.append(signum)
        def wait(self):
            if self.command[2] == "run":
                if signal_during_run:
                    signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
                    assert self.received == [signal.SIGTERM]
                    return -signal.SIGTERM
                if current["protocol"]["states"][-1]["state_id"] == failure_state:
                    return 7
            return 0
    monkeypatch.setattr(CAMPAIGN.subprocess, "Popen", Child)
    def barrier(protocol, root, **kwargs):
        receipt = {"drive_status": "NOT_REQUESTED", "receipt_hash": content_hash(protocol["protocol_hash"])}
        atomic_write_json(Path(root) / protocol["protocol_hash"] / "completion_receipt.json", receipt)
        return receipt
    monkeypatch.setattr(CAMPAIGN, "completion_barrier", barrier)
    return calls


def test_restart_reuses_all_completed_payloads_without_generating_again(tmp_path, monkeypatch):
    args = inputs(tmp_path)
    plan, protocols = CAMPAIGN.campaign_plan(args)
    calls = child_fixture(monkeypatch, args, protocols)
    assert CAMPAIGN.execute_campaign(args, plan, protocols) == 0
    assert [c[2] for c in calls] == ["freeze-base", "run"] + [c for _ in range(10) for c in ("freeze", "run")]
    assert CAMPAIGN.execute_campaign(args, plan, protocols) == 0
    assert sum(c[2] == "run" for c in calls) == 11
    status = read_json(Path(args.output_root) / "campaigns" / plan["plan_hash"] / "status.json")
    assert status["status"] == "COMPLETE" and len(status["completed_state_ids"]) == 11
    assert status["status_storage"] == "LOCAL_ONLY" and "drive_status" not in status
    changed = deepcopy(plan)
    changed["save_to_drive"] = True
    seal(changed, "plan_hash")
    with pytest.raises(ValueError, match="different contents"):
        CAMPAIGN.execute_campaign(args, changed, protocols)


def test_failed_child_preserves_exit_code_and_prevents_next_level(tmp_path, monkeypatch):
    args = inputs(tmp_path)
    plan, protocols = CAMPAIGN.campaign_plan(args)
    calls = child_fixture(monkeypatch, args, protocols, failure_state="rot13_14b_level_02")
    assert CAMPAIGN.execute_campaign(args, plan, protocols) == 7
    assert sum(c[2] == "run" for c in calls) == 3
    status = read_json(Path(args.output_root) / "campaigns" / plan["plan_hash"] / "status.json")
    assert status["completed_state_ids"] == ["base_14b", "rot13_14b_level_01"]
    assert status["status"] == "FAILED" and status["child_exit_code"] == 7


def test_stop_is_forwarded_to_active_child_and_no_next_state_starts(tmp_path, monkeypatch):
    args = inputs(tmp_path)
    plan, protocols = CAMPAIGN.campaign_plan(args)
    calls = child_fixture(monkeypatch, args, protocols, signal_during_run=True)
    assert CAMPAIGN.execute_campaign(args, plan, protocols) == 143
    assert len(calls) == 2


def payload_fixture(args, protocol, *, drive):
    root = Path(args.output_root) / protocol["protocol_hash"]
    atomic_write_json(root / "protocol.json", protocol)
    generator = GPQAVLLMGenerator(root, protocol)
    paths, rows, summaries = [root / "protocol.json"], [], {}
    for state in protocol["states"]:
        reuse = protocol.get("baseline_reuse") if state["family"] == "base" else None
        raw_root = Path(args.output_root) / (reuse["protocol_hash"] if reuse else protocol["protocol_hash"])
        observed = []
        for item in protocol["panel"]["items"]:
            identity = generator.identity(state, item)
            text = "<think>Fixture.</think>Final answer: A"
            raw = seal({"status": "success", "simulation": False, "identity": identity,
                        "request_hash": content_hash(identity), "raw_output": text,
                        "raw_hash": content_hash(text), "finish_reason": "stop", "generated_tokens": 10}, "record_hash")
            raw_path = raw_root / "raw" / state["state_id"] / (raw["request_hash"] + ".json")
            atomic_write_json(raw_path, raw)
            row = seal({"protocol_hash": protocol["protocol_hash"], "state_id": state["state_id"],
                        "question_id": item["question_id"], "raw_hash": raw["raw_hash"],
                        "simulation": False, "score": score_gpqa(item, raw)}, "record_hash")
            score_path = root / "scores" / state["state_id"] / (item["question_id"] + ".json")
            atomic_write_json(score_path, row)
            paths += [raw_path, score_path]
            observed.append(row)
            rows.append(row)
        correct = sum(r["score"]["correct"] for r in observed)
        summaries[state["state_id"]] = {"denominator": 198, "observed": 198, "correct_count": correct, "accuracy": correct / 198}
    difference = None if len(protocol["states"]) == 1 else 0
    report = seal({"status": "complete", "simulation": False, "protocol_hash": protocol["protocol_hash"],
                   "panel_hash": protocol["panel"]["panel_hash"], "variant": "gpqa_diamond", "records": rows,
                   "summaries": summaries, "accuracy_difference_trained_minus_base": difference}, "report_hash")
    report_path = root / ("baseline.json" if len(protocol["states"]) == 1 else "comparison.json")
    atomic_write_json(report_path, report)
    paths.append(report_path)
    receipt = {"version": "supplemental_gpqa_completion_receipt_v1", "status": "COMPLETE",
               "drive_status": "VERIFIED" if drive else "NOT_REQUESTED", "simulation": False,
               "protocol_hash": protocol["protocol_hash"], "panel_hash": protocol["panel"]["panel_hash"],
               "model": "14b", "seed": 42, "source_code_commit": CAMPAIGN.code_revision(),
               "model_revision": CAMPAIGN.MODEL_REVISION, "tokenizer_revision": CAMPAIGN.MODEL_REVISION,
               "report_path": str(report_path), "report_sha256": file_hash(report_path),
               "report_hash": report["report_hash"], "summaries": summaries}
    if drive:
        service = ArtifactService(args.output_root, Path(args.output_root) / "test_registry.json")
        artifacts = {}
        for path in paths:
            raw = path.parent.parent.name == "raw"
            configuration_hash = protocol.get("baseline_reuse", {}).get("protocol_hash") if raw and "base_14b" == path.parent.name and protocol.get("baseline_reuse") else protocol["protocol_hash"]
            provenance = {"experiment_id": configuration_hash, "branch": "supplemental_gpqa", "model": "14b",
                          "run_id": "fixture", "stage": "supplemental_capability", "producer": "run_supplemental_gpqa",
                          "source_code_commit": CAMPAIGN.code_revision(), "configuration_hash": configuration_hash,
                          "dataset_revision": SOURCE_REVISION, "model_revision": CAMPAIGN.MODEL_REVISION, "seeds": [42]}
            record = service.register(path, provenance)
            record.update(sync_status="VERIFIED", drive_file_id="fixture-file", drive_folder_id="fixture-folder")
            artifacts[record["artifact_id"]] = record
        inventory = root / "drive_manifest.json"
        atomic_write_json(inventory, {"version": 2, "artifacts": artifacts})
        receipt.update(registry_path=str(inventory), registry_sha256=file_hash(inventory))
    receipt = seal(receipt, "receipt_hash")
    atomic_write_json(root / "completion_receipt.json", receipt)
    return root, receipt


def test_comparison_barrier_accepts_original_base_raw_and_separate_paired_scores(tmp_path):
    args = inputs(tmp_path)
    plan, protocols = CAMPAIGN.campaign_plan(args)
    root, receipt = payload_fixture(args, protocols[1], drive=True)
    assert CAMPAIGN.completion_barrier(protocols[1], args.output_root, save_to_drive=True,
                                       source_code_commit=plan["source_code_commit"]) == receipt


@pytest.mark.parametrize("mutation", ["missing_raw", "missing_score", "simulated", "raw_identity", "unverified_drive"])
def test_completion_flag_cannot_skip_actual_payload_barriers(tmp_path, mutation):
    args = inputs(tmp_path)
    plan, protocols = CAMPAIGN.campaign_plan(args)
    protocol = protocols[0]
    root, receipt = payload_fixture(args, protocol, drive=True)
    if mutation == "missing_raw":
        next((root / "raw/base_14b").glob("*.json")).unlink()
    elif mutation == "missing_score":
        next((root / "scores/base_14b").glob("*.json")).unlink()
    elif mutation == "simulated":
        receipt["simulation"] = True
        atomic_write_json(root / "completion_receipt.json", seal(receipt, "receipt_hash"))
    elif mutation == "raw_identity":
        path = next((root / "raw/base_14b").glob("*.json"))
        raw = read_json(path)
        raw["identity"]["settings"]["max_new_tokens"] = 3072
        atomic_write_json(path, seal(raw, "record_hash"))
    else:
        path = root / "drive_manifest.json"
        inventory = read_json(path)
        next(iter(inventory["artifacts"].values()))["sync_status"] = "PENDING"
        atomic_write_json(path, inventory)
        receipt["registry_sha256"] = file_hash(path)
        atomic_write_json(root / "completion_receipt.json", seal(receipt, "receipt_hash"))
    with pytest.raises((ValueError, FileNotFoundError)):
        CAMPAIGN.completion_barrier(protocol, args.output_root, save_to_drive=True,
                                    source_code_commit=plan["source_code_commit"])
