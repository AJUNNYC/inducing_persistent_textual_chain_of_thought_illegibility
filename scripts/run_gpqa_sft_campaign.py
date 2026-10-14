#!/usr/bin/env python3
"""Run full Diamond GPQA for untouched 14B and all ten final ROT13 adapters.

Each state runs through the published supplemental CLI in a separate process.
Only the matching long-budget untouched-base evidence is reused. Adapter
identities retain their original training source/configuration, including level
10; no checkpoint selection, training, KaziText, or primary-suite work occurs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import json
import re
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.supplemental_gpqa import (
    BASELINE_VERSION, GPQAVLLMGenerator, SOURCE_REVISION, baseline_state,
    comparison_states, freeze_gpqa_baseline, freeze_gpqa_protocol,
    generation_contract, publish_immutable, score_gpqa, validate_gpqa_panel,
    validate_gpqa_protocol,
)
from experiment_overview.experiment_overview import code_revision
from shared_procedures.identity import content_hash, file_hash
from shared_procedures.storage import atomic_write_json, file_lock, read_json

MODEL_REVISION = "1df8507178afcc1bef68cd8c393f61a886323761"
MANIFEST_VERSION = "gpqa_sft_adapters_manifest_v1"
PLAN_VERSION = "gpqa_rot13_14b_campaign_plan_v1"
CLI = ROOT / "scripts/run_supplemental_gpqa.py"


def sealed(record, field):
    if record.get(field) != content_hash({k: v for k, v in record.items() if k != field}):
        raise ValueError(f"GPQA campaign {field} integrity differs")


def seal(record, field):
    record[field] = content_hash(record)
    return record


def campaign_inputs(panel_path, adapters_manifest, model_cache):
    """Validate the entire preselected panel/adapter matrix before any GPU work."""
    panel = read_json(panel_path)
    validate_gpqa_panel(panel)
    if (panel["variant"] != "gpqa_diamond" or panel["item_count"] != 198
            or panel["source"]["revision"] != SOURCE_REVISION):
        raise ValueError("Campaign requires the pinned full Diamond198 panel")
    manifest_path = Path(adapters_manifest).resolve()
    manifest = read_json(manifest_path)
    adapters = manifest.get("adapters")
    if (manifest.get("version") != MANIFEST_VERSION or manifest.get("family") != "rot13"
            or manifest.get("model") != "14b" or not isinstance(adapters, list)
            or len(adapters) != 10 or any(not isinstance(a, dict) for a in adapters)):
        raise ValueError("Campaign manifest must declare exactly ten ROT13 14B final adapters")
    levels = [a.get("level") for a in adapters]
    if any(type(level) is not int for level in levels) or set(levels) != set(range(1, 11)):
        raise ValueError("Campaign requires exactly unique levels 1 through 10")
    bindings, paths = [], set()
    for declared in sorted(adapters, key=lambda a: a["level"]):
        raw_path = declared.get("final_adapter")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("Every campaign condition requires an explicit final_adapter path")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = manifest_path.parent / path
        if path.is_symlink():
            raise ValueError("Campaign final_adapter paths cannot be symbolic links")
        path = path.resolve()
        if path in paths:
            raise ValueError("Campaign conditions cannot reuse an adapter directory")
        paths.add(path)
        states = comparison_states(path, model_cache)
        trained = states[1]
        if (trained["family"] != "rot13" or trained["model"] != "14b"
                or trained["level"] != declared["level"]
                or any(s[k] != MODEL_REVISION for s in states
                       for k in ("model_revision", "tokenizer_revision"))):
            raise ValueError("Declared level/family/model or immutable model/tokenizer pins differ")
        adapter_manifest = read_json(path / "manifest.json")
        # Bind exact original identities; do not require all training sources or
        # configurations to match this new evaluation checkout or one another.
        bindings.append({"level": declared["level"], "final_adapter": str(path),
                         "states": states, "training_identity": adapter_manifest["identity"],
                         "adapter_manifest_hash": adapter_manifest["manifest_hash"],
                         "adapter_manifest_sha256": file_hash(path / "manifest.json"),
                         "fit_result_sha256": file_hash(path.parent / "fit_result.json"),
                         "checkpoint_hash": adapter_manifest["checkpoint_hash"]})
    return panel, bindings


def campaign_plan(args):
    if args.generation_profile != "long_32000":
        raise ValueError("This campaign is explicitly the long_32000 GPQA evaluation")
    panel, bindings = campaign_inputs(args.panel, args.adapters_manifest, args.model_cache)
    source = code_revision()
    if not re.fullmatch(r"[0-9a-f]{40}", source):
        raise ValueError("Campaign execution requires a pinned source code commit")
    contract = generation_contract("vllm", args.generation_profile)
    baseline = freeze_gpqa_baseline(panel, baseline_state("14b", args.model_cache,
                                     model_revision=MODEL_REVISION, tokenizer_revision=MODEL_REVISION),
                                    backend="vllm", generation_profile=args.generation_profile)
    protocols = [baseline] + [freeze_gpqa_protocol(panel, binding["states"], backend="vllm",
                              baseline_protocol=baseline, generation_profile=args.generation_profile)
                             for binding in bindings]
    jobs = []
    for protocol in protocols:
        state = protocol["states"][-1]
        jobs.append({"state_id": state["state_id"], "level": state.get("level"),
                     "protocol_hash": protocol["protocol_hash"],
                     "adapter_path": state.get("adapter_path")})
    plan = seal({"version": PLAN_VERSION, "family": "rot13", "model": "14b",
                 "variant": "gpqa_diamond", "questions_per_state": 198,
                 "state_count": 11, "order": jobs, "seed": 42,
                 "model_revision": MODEL_REVISION, "tokenizer_revision": MODEL_REVISION,
                 "generation_profile": args.generation_profile, "generation": contract,
                 "generation_contract_hash": content_hash(contract),
                 "source_code_commit": source,
                 "implementation_sha256": {"scripts/run_gpqa_sft_campaign.py": file_hash(__file__),
                                            "scripts/run_supplemental_gpqa.py": file_hash(CLI)},
                 "panel_path": str(Path(args.panel).resolve()), "panel_hash": panel["panel_hash"],
                 "panel_sha256": file_hash(args.panel),
                 "adapters_manifest_path": str(Path(args.adapters_manifest).resolve()),
                 "adapters_manifest_sha256": file_hash(args.adapters_manifest),
                 "adapter_bindings": bindings,
                 "model_cache": str(Path(args.model_cache).resolve()),
                 "output_root": str(Path(args.output_root).resolve()),
                 "save_to_drive": bool(args.save_to_drive),
                 "baseline_reuse": "matching_new_long_budget_baseline_raw_only",
                 "checkpoint_selection": "all_preselected_completed_final_adapters_before_evaluation",
                 "primary_suite_changed": False, "training_changed": False}, "plan_hash")
    return plan, protocols


def freeze_command(args, job, protocol_path, baseline_path):
    command = [sys.executable, str(CLI), "freeze-base" if job["level"] is None else "freeze",
               "--panel", str(Path(args.panel).resolve()), "--model-cache", str(Path(args.model_cache).resolve()),
               "--backend", "vllm", "--generation-profile", args.generation_profile,
               "--output", str(protocol_path)]
    if job["level"] is None:
        command += ["--model", "14b", "--model-revision", MODEL_REVISION,
                    "--tokenizer-revision", MODEL_REVISION]
    else:
        command += ["--adapter", job["adapter_path"], "--baseline-protocol", str(baseline_path)]
    return command


def run_command(args, protocol_path):
    command = [sys.executable, str(CLI), "run", "--protocol", str(protocol_path),
               "--output-root", str(Path(args.output_root).resolve())]
    return command + (["--save-to-drive"] if args.save_to_drive else [])


def completion_barrier(protocol, output_root, *, save_to_drive, source_code_commit):
    """Check real per-question bytes, score bindings and the child's Drive barrier.

    This is read-only and never loads a GPU model or contacts Drive. Synchronous
    upload/checksum validation belongs to the authoritative supplemental CLI.
    """
    validate_gpqa_protocol(protocol)
    root = Path(output_root).resolve() / protocol["protocol_hash"]
    receipt = read_json(root / "completion_receipt.json")
    sealed(receipt, "receipt_hash")
    report_path = root / ("baseline.json" if protocol["version"] == BASELINE_VERSION else "comparison.json")
    if (receipt.get("version") != "supplemental_gpqa_completion_receipt_v1"
            or receipt.get("status") != "COMPLETE" or receipt.get("simulation") is not False
            or receipt.get("drive_status") != ("VERIFIED" if save_to_drive else "NOT_REQUESTED")
            or receipt.get("protocol_hash") != protocol["protocol_hash"]
            or receipt.get("panel_hash") != protocol["panel"]["panel_hash"]
            or receipt.get("model") != "14b" or receipt.get("seed") != 42
            or receipt.get("source_code_commit") != source_code_commit
            or any(receipt.get(k) != MODEL_REVISION for k in ("model_revision", "tokenizer_revision"))
            or Path(receipt.get("report_path", "")).resolve() != report_path
            or receipt.get("report_sha256") != file_hash(report_path)
            or read_json(root / "protocol.json") != protocol):
        raise ValueError("GPQA child completion receipt differs from its frozen verified payloads")
    report = read_json(report_path)
    sealed(report, "report_hash")
    state_ids = {s["state_id"] for s in protocol["states"]}
    items = protocol["panel"]["items"]
    rows = report.get("records", [])
    if (report.get("status") != "complete" or report.get("simulation") is not False
            or report.get("protocol_hash") != protocol["protocol_hash"]
            or report.get("panel_hash") != protocol["panel"]["panel_hash"]
            or report.get("variant") != "gpqa_diamond"
            or report.get("report_hash") != receipt.get("report_hash")
            or len(rows) != 198 * len(state_ids)
            or set(report.get("summaries", {})) != state_ids
            or receipt.get("summaries") != report["summaries"]):
        raise ValueError("GPQA actual observation matrix is incomplete or different")
    observed = {(r["state_id"], r["question_id"]): r for r in rows}
    if len(observed) != len(rows):
        raise ValueError("GPQA observation matrix contains duplicate state/question records")
    paths = [(root / "protocol.json", protocol["protocol_hash"]), (report_path, protocol["protocol_hash"])]
    generator = GPQAVLLMGenerator(root, protocol)
    for state in protocol["states"]:
        reuse = protocol.get("baseline_reuse") if state["family"] == "base" else None
        raw_protocol = reuse["protocol_hash"] if reuse else protocol["protocol_hash"]
        raw_root = Path(output_root).resolve() / raw_protocol
        correct = 0
        for item in items:
            identity = generator.identity(state, item)
            raw_path = raw_root / "raw" / state["state_id"] / (content_hash(identity) + ".json")
            raw = read_json(raw_path)
            sealed(raw, "record_hash")
            if (raw.get("identity") != identity or raw.get("request_hash") != content_hash(identity)
                    or raw.get("status") != "success" or raw.get("simulation") is not False
                    or not isinstance(raw.get("raw_output"), str)
                    or raw.get("raw_hash") != content_hash(raw["raw_output"])
                    or raw.get("finish_reason") not in {"stop", "length"}
                    or type(raw.get("generated_tokens")) is not int
                    or not 0 <= raw["generated_tokens"] <= protocol["generation"]["settings"]["max_new_tokens"]):
                raise ValueError("GPQA raw output differs from its actual frozen request")
            row = observed.get((state["state_id"], item["question_id"]))
            if row is None:
                raise ValueError("GPQA actual observation matrix is missing a question")
            sealed(row, "record_hash")
            score_path = root / "scores" / state["state_id"] / (item["question_id"] + ".json")
            if (row.get("simulation") is not False or row.get("protocol_hash") != protocol["protocol_hash"]
                    or row.get("raw_hash") != raw["raw_hash"] or row.get("score") != score_gpqa(item, raw)
                    or read_json(score_path) != row):
                raise ValueError("GPQA score differs from its actual raw output/question")
            correct += row["score"]["correct"]
            paths += [(raw_path, raw_protocol), (score_path, protocol["protocol_hash"])]
        if report["summaries"][state["state_id"]] != {
                "denominator": 198, "observed": 198, "correct_count": correct, "accuracy": correct / 198}:
            raise ValueError("GPQA summary differs from its actual score records")
    expected_difference = (None if len(protocol["states"]) == 1 else
                           report["summaries"][protocol["states"][1]["state_id"]]["accuracy"] -
                           report["summaries"][protocol["states"][0]["state_id"]]["accuracy"])
    if report.get("accuracy_difference_trained_minus_base") != expected_difference:
        raise ValueError("GPQA paired difference differs from its actual summaries")
    if save_to_drive:
        from operations.artifacts import validate_registered_record
        inventory_path = root / "drive_manifest.json"
        if (Path(receipt.get("registry_path", "")).resolve() != inventory_path
                or receipt.get("registry_sha256") != file_hash(inventory_path)):
            raise ValueError("GPQA completion lacks its immutable verified Drive inventory")
        inventory = read_json(inventory_path)
        if inventory.get("version") != 2 or not isinstance(inventory.get("artifacts"), dict):
            raise ValueError("GPQA completion lacks its canonical Drive inventory")
        records = list(inventory["artifacts"].values())
        indexed = {}
        for record in records:
            indexed.setdefault((record.get("local_path"), record.get("sha256")), []).append(record)
        for path, configuration_hash in paths:
            matching = indexed.get((str(path), file_hash(path)), [])
            if (not matching or any(r.get("sync_status") != "VERIFIED" or not r.get("drive_file_id")
                    or r.get("configuration_hash") != configuration_hash
                    or r.get("source_code_commit") != source_code_commit for r in matching)):
                raise ValueError("GPQA payload lacks its actual verified Drive checksum record")
            for record in matching:
                validate_registered_record(record)
    return receipt


def execute_campaign(args, plan, protocols):
    output = Path(args.output_root).resolve()
    campaign = output / "campaigns" / plan["plan_hash"]
    campaign.mkdir(parents=True, exist_ok=True)
    completed, receipts, child, stop = [], [], None, {"requested": False, "signal": None}
    old_handlers = {}

    def status(value, **fields):
        # This mutable local status is deliberately not represented as a Drive
        # artifact. Only completed child payload barriers carry VERIFIED status.
        record = {"version": "gpqa_campaign_local_status_v1", "status": value,
                  "updated_at": datetime.now(timezone.utc).isoformat(),
                  "plan_hash": plan["plan_hash"], "completed_state_ids": list(completed),
                  "state_count": 11, "status_storage": "LOCAL_ONLY", **fields}
        atomic_write_json(campaign / "status.json", record)
        print(json.dumps(record), flush=True)

    def request_stop(signum, frame):
        stop.update(requested=True, signal=signum)
        if child is not None and child.poll() is None:
            child.send_signal(signum)

    def invoke(command, log_path):
        nonlocal child
        if stop["requested"]:
            return 128 + stop["signal"]
        with log_path.open("a") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if stop["requested"] and child.poll() is None:
                child.send_signal(stop["signal"])
            code = child.wait()
        child = None
        return code if code >= 0 else 128 - code

    for signum in (signal.SIGTERM, signal.SIGINT):
        old_handlers[signum] = signal.signal(signum, request_stop)
    try:
        with file_lock(output / "campaigns" / "writer.lock"):
            # One output root cannot silently become a different long-budget
            # campaign on restart, even if an older campaign finished.
            publish_immutable(output / "campaigns" / "plan.json", plan)
            publish_immutable(campaign / "plan.json", plan)
            baseline_path = campaign / "protocols" / "base_14b.json"
            for job, expected in zip(plan["order"], protocols):
                state_id = job["state_id"]
                protocol_path = campaign / "protocols" / (state_id + ".json")
                log_path = campaign / (state_id + ".log")
                status("FREEZING", current_state_id=state_id)
                code = invoke(freeze_command(args, job, protocol_path, baseline_path), log_path)
                if code:
                    status("STOPPED" if stop["requested"] else "FAILED", current_state_id=state_id,
                           child_exit_code=code, log_path=str(log_path))
                    return code
                if read_json(protocol_path) != expected:
                    raise ValueError("CLI-frozen protocol differs from the predeclared campaign plan")
                receipt_path = output / job["protocol_hash"] / "completion_receipt.json"
                if not receipt_path.exists():
                    status("RUNNING", current_state_id=state_id, log_path=str(log_path))
                    code = invoke(run_command(args, protocol_path), log_path)
                    if code:
                        status("STOPPED" if stop["requested"] else "FAILED", current_state_id=state_id,
                               child_exit_code=code, log_path=str(log_path))
                        return code
                receipt = completion_barrier(expected, output, save_to_drive=args.save_to_drive,
                                             source_code_commit=plan["source_code_commit"])
                completed.append(state_id)
                receipts.append({"state_id": state_id, "protocol_hash": job["protocol_hash"],
                                 "receipt_hash": receipt["receipt_hash"],
                                 "receipt_path": str(receipt_path),
                                 "receipt_sha256": file_hash(receipt_path),
                                 "payload_drive_status": receipt["drive_status"]})
                status("STATE_COMPLETE", current_state_id=state_id,
                       completed_payload_drive_status=receipt["drive_status"],
                       completion_receipt_hash=receipt["receipt_hash"])
                if stop["requested"]:
                    status("STOPPED", child_exit_code=128 + stop["signal"])
                    return 128 + stop["signal"]
            publish_immutable(campaign / "completion_receipt.json", seal({
                "version": "gpqa_campaign_completion_receipt_v1", "status": "COMPLETE",
                "plan_hash": plan["plan_hash"], "source_code_commit": plan["source_code_commit"],
                "completed_state_ids": completed, "state_count": 11,
                "questions_per_state": 198, "generation_profile": "long_32000",
                "model_revision": MODEL_REVISION, "tokenizer_revision": MODEL_REVISION,
                "child_receipts": receipts, "receipt_storage": "LOCAL_ONLY",
                "completed_payload_drive_status": "VERIFIED" if args.save_to_drive else "NOT_REQUESTED",
                "primary_suite_changed": False, "training_changed": False}, "receipt_hash"))
            status("COMPLETE", current_state_id=None,
                   completed_payload_drive_status="VERIFIED" if args.save_to_drive else "NOT_REQUESTED")
            return 0
    except Exception as error:
        status("FAILED", error=str(error))
        raise
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--adapters-manifest", required=True)
    parser.add_argument("--model-cache", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--generation-profile", choices=("long_32000",), default="long_32000")
    parser.add_argument("--save-to-drive", action="store_true")
    parser.add_argument("--validate-only", action="store_true", help="Validate all input bytes and freeze the local plan without GPU or network work")
    args = parser.parse_args(argv)
    plan, protocols = campaign_plan(args)
    if args.validate_only:
        with file_lock(Path(args.output_root).resolve() / "campaigns" / "writer.lock"):
            publish_immutable(Path(args.output_root).resolve() / "campaigns" / "plan.json", plan)
        print(json.dumps({"status": "READY", "plan_hash": plan["plan_hash"], "states": 11,
                          "questions_per_state": 198, "generation_profile": "long_32000",
                          "gpu_work_started": False, "network_requests_made": False}), flush=True)
        return 0
    return execute_campaign(args, plan, protocols)


if __name__ == "__main__":
    raise SystemExit(main())
