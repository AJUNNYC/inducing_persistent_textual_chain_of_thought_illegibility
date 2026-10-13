#!/usr/bin/env python3
"""Run the nineteen remaining published 14B fits after a verified GPQA base.

This coordinator never changes the per-fit scientific implementation. Each job
calls run_certified_sft.py, whose exact checkpoint recovery and synchronous Drive
barriers remain authoritative. The completed ROT13 level-10 fit keeps its old
identity; it is excluded from new jobs, never relabelled or used as a warm start.
The authorization is recorded; the coordinator does not infer a price or enforce
a rate-based cutoff. The explicit UTC deadline and user pause remain operative.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from experiment_overview.experiment_overview import Experiment
from run_certified_sft import certified_tasks
from shared_procedures.configuration import sft_training_budget, validate_configuration
from shared_procedures.identity import content_hash, file_hash
from shared_procedures.storage import atomic_write_json, file_lock, read_json
from training.sft.individual_fit.sft_fit import verify_adapter_manifest
from training.sft.published_execution import PUBLISHED_EXECUTION_POLICY


def utc(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.utcoffset() is None or result.utcoffset().total_seconds() != 0:
        raise ValueError("Campaign timestamps must explicitly use UTC")
    return result


def ordered_jobs(tasks):
    inventory = {(t["family"], t["model_id"], t["level"]): t for t in tasks}
    expected = {(f, "14b", l) for f in ("rot13", "kazitext") for l in range(1, 11)}
    if len(tasks) != 20 or set(inventory) != expected:
        raise ValueError("Campaign requires exactly twenty declared 14B conditions")
    order = [("kazitext", "14b", 10)] + [
        (f, "14b", l) for l in range(1, 10) for f in ("rot13", "kazitext")]
    return [inventory[key] for key in order]


def retained_fit(receipt, tasks, config):
    """Preserve old identity only after CPU-proved unchanged training bindings."""
    if receipt.get("receipt_hash") != content_hash({k: v for k, v in receipt.items() if k != "receipt_hash"}):
        raise ValueError("Retained-fit receipt integrity differs")
    result = receipt.get("fit_result", {})
    identity = result.get("identity", {})
    matching = next(t for t in tasks if t["run_id"] == "rot13_14b_level_10")["spec"]
    keys = ("family", "model_id", "level", "run_id", "model_name", "model_revision",
            "tokenizer_revision", "dataset_hash", "producer_configuration_hash", "recipe",
            "training_budget", "published_execution_policy", "implementation_version")
    if (receipt.get("status") != "RETAINED_VERIFIED"
            or receipt.get("consumer_commit") != config["runtime"]["implementation_revision"]
            or result.get("status") != "complete"
            or result.get("progress", {}).get("completed_steps") != 625
            or result.get("progress", {}).get("example_presentations") != 10000
            or identity.get("dependency_hash") != config["runtime"]["dependency_lock_hash"]
            or any(identity.get(k) != matching.get(k) for k in keys)
            or identity.get("published_execution_policy") != PUBLISHED_EXECUTION_POLICY
            or receipt.get("drive_status") != "VERIFIED"
            or not receipt.get("unchanged_training_files")):
        raise ValueError("Retained completion differs from the declared data/model/recipe")
    for name, digest in receipt["unchanged_training_files"].items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or file_hash(ROOT / path) != digest:
            raise ValueError("Retained training-source compatibility differs")
    return identity


def sealed(record, field):
    if record.get(field) != content_hash({k: v for k, v in record.items() if k != field}):
        raise ValueError("GPQA artifact integrity differs: " + field)


def verify_gpqa_barrier(args, config):
    """Require the actual pinned untouched-14B panel and all verified payloads."""
    from evaluation.supplemental_gpqa import BASELINE_VERSION, validate_gpqa_protocol
    from operations.artifacts import validate_registered_record

    protocol = read_json(args.gpqa_protocol)
    validate_gpqa_protocol(protocol)
    root = Path(args.gpqa_output_root).resolve() / protocol["protocol_hash"]
    state = protocol["states"][0]
    model = config["models"]["14b"]
    if (protocol.get("version") != BASELINE_VERSION
            or len(protocol["states"]) != 1 or state["state_id"] != "base_14b"
            or state.get("family") != "base" or state.get("adapter_path")
            or state["model_revision"] != model["revision"]
            or state["tokenizer_revision"] != model["tokenizer_revision"]
            or protocol["panel"]["variant"] != "gpqa_diamond"
            or protocol["panel"]["item_count"] != 198):
        raise ValueError("GPQA barrier requires the frozen untouched14B Diamond198 baseline")
    receipt = read_json(args.gpqa_receipt)
    sealed(receipt, "receipt_hash")
    report_path = root / "baseline.json"
    registry_path = root / "drive_manifest.json"
    if (receipt.get("version") != "supplemental_gpqa_completion_receipt_v1"
            or receipt.get("status") != "COMPLETE" or receipt.get("drive_status") != "VERIFIED"
            or receipt.get("simulation") is not False
            or receipt.get("protocol_hash") != protocol["protocol_hash"]
            or receipt.get("panel_hash") != protocol["panel"]["panel_hash"]
            or receipt.get("model") != "14b" or receipt.get("seed") != 42
            or receipt.get("source_code_commit") != config["runtime"]["implementation_revision"]
            or receipt.get("model_revision") != model["revision"]
            or receipt.get("tokenizer_revision") != model["tokenizer_revision"]
            or Path(receipt.get("report_path", "")).resolve() != report_path
            or Path(receipt.get("registry_path", "")).resolve() != registry_path
            or file_hash(report_path) != receipt.get("report_sha256")
            or file_hash(registry_path) != receipt.get("registry_sha256")):
        raise ValueError("GPQA completion receipt lacks matching verified Drive payloads")
    report = read_json(report_path)
    sealed(report, "report_hash")
    items = {i["question_id"]: i for i in protocol["panel"]["items"]}
    rows = report.get("records", [])
    observed = {r["question_id"]: r for r in rows}
    if (report.get("status") != "complete" or report.get("simulation") is not False
            or report.get("protocol_hash") != protocol["protocol_hash"]
            or report.get("panel_hash") != protocol["panel"]["panel_hash"]
            or report.get("report_hash") != receipt.get("report_hash")
            or report.get("variant") != "gpqa_diamond"
            or len(rows) != 198 or len(observed) != 198 or set(observed) != set(items)
            or report.get("accuracy_difference_trained_minus_base") is not None):
        raise ValueError("GPQA actual observation matrix is incomplete or is a different comparison")
    raw_paths = sorted((root / "raw/base_14b").glob("*.json"))
    raws = {}
    for path in raw_paths:
        raw = read_json(path)
        sealed(raw, "record_hash")
        identity = raw.get("identity", {})
        question = identity.get("question_id")
        if (question in raws or question not in items or raw.get("status") != "success"
                or raw.get("simulation") is not False
                or identity.get("protocol_hash") != protocol["protocol_hash"]
                or identity.get("state") != state or identity.get("seed") != 42
                or identity.get("execution_mode") != "production"
                or raw.get("request_hash") != content_hash(identity)
                or path.name != raw["request_hash"] + ".json"
                or not isinstance(raw.get("raw_output"), str)
                or raw.get("raw_hash") != content_hash(raw["raw_output"])):
            raise ValueError("GPQA raw output has wrong immutable request or simulation identity")
        raws[question] = (raw, path)
    if len(raws) != 198 or set(raws) != set(items):
        raise ValueError("GPQA requires all 198 actual raw outputs")
    paths = [root / "protocol.json", report_path, *raw_paths]
    if read_json(paths[0]) != protocol:
        raise ValueError("GPQA saved input protocol differs")
    for question, row in observed.items():
        sealed(row, "record_hash")
        path = root / "scores/base_14b" / (question + ".json")
        if (row.get("simulation") is not False or row.get("state_id") != "base_14b"
                or row.get("protocol_hash") != protocol["protocol_hash"]
                or row.get("raw_hash") != raws[question][0]["raw_hash"]
                or row.get("score", {}).get("item_hash") != items[question]["item_hash"]
                or read_json(path) != row):
            raise ValueError("GPQA score does not bind its real raw output and exact question")
        paths.append(path)
    summary = report.get("summaries", {}).get("base_14b", {})
    if (set(report.get("summaries", {})) != {"base_14b"}
            or summary.get("denominator") != 198 or summary.get("observed") != 198
            or summary.get("correct_count") != sum(r["score"]["correct"] for r in rows)
            or summary.get("accuracy") != summary["correct_count"] / 198
            or receipt.get("summaries") != report["summaries"]):
        raise ValueError("GPQA baseline summary differs from its 198 actual score records")
    registry = read_json(registry_path)
    if registry.get("version") != 2 or not isinstance(registry.get("artifacts"), dict):
        raise ValueError("GPQA Drive inventory is not the canonical artifact registry")
    records = list(registry["artifacts"].values())
    for path in paths:
        matching = [r for r in records if r.get("local_path") == str(path)
                    and r.get("sha256") == file_hash(path)]
        if (not matching or any(r.get("sync_status") != "VERIFIED" or not r.get("drive_file_id")
                or r.get("configuration_hash") != protocol["protocol_hash"]
                or r.get("source_code_commit") != config["runtime"]["implementation_revision"] for r in matching)):
            raise ValueError("GPQA payload lacks its verified Drive checksum record")
        for record in matching:
            validate_registered_record(record)
    return receipt


def training_command(args, job, deadline):
    command = [sys.executable, str(ROOT / "scripts/run_certified_sft.py")]
    for name in ("config", "certificate", "bundle", "worklist", "preflight"):
        command += ["--" + name, getattr(args, name)]
    return command + ["--family", job["family"], "--model", "14b", "--level", str(job["level"]),
                      "--deadline", deadline.isoformat()]


def verified_fit(root, job, config, certificate_hash, code):
    directory = root / "training/sft" / job["family"] / "14b" / f"level_{job['level']:02d}"
    result = read_json(directory / "fit_result.json")
    worker = read_json(root / "training/sft" / f"worker_status_{job['family']}_14b.json")
    identity = {**job["spec"], "code_revision": code,
                "dependency_hash": config["runtime"]["dependency_lock_hash"]}
    if (result.get("status") != "complete" or result.get("identity") != identity
            or result.get("progress", {}).get("completed_steps") != 625
            or result.get("progress", {}).get("example_presentations") != 10000
            or worker.get("status") != "COMPLETE" or worker.get("drive_status") != "VERIFIED"
            or worker.get("completed_run_ids") != [job["run_id"]]
            or worker.get("handoff_certificate_hash") != certificate_hash
            or worker.get("configuration_hash") != identity["configuration_hash"]
            or worker.get("source_code_commit") != code):
        raise ValueError("Per-fit completion or verified Drive barrier is missing")
    manifest = verify_adapter_manifest(directory / "final_adapter", identity)
    if (result.get("adapter_manifest_hash") != manifest["manifest_hash"]
            or result.get("checkpoint_hash") != manifest["checkpoint_hash"]):
        raise ValueError("Final adapter and completed-fit report differ")
    return result


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "certificate", "bundle", "worklist", "preflight", "retained-fit",
                 "gpqa-protocol", "gpqa-output-root", "gpqa-receipt", "deadline"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--wait-for-baseline", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    config = validate_configuration(read_json(args.config), runtime=True)
    if sft_training_budget(config) != "efficient_one_epoch_v1":
        raise ValueError("Campaign supports only the finalized published one-epoch recipe")
    if not (config["spending"].get("authorized") and config["spending"].get("authorization_evidence")):
        raise ValueError("Current explicit campaign spending authorization required")
    experiment = Experiment(config, config_path=args.config)
    if experiment.code != config["runtime"]["implementation_revision"]:
        raise ValueError("Executed source differs from the frozen campaign")
    certificate, bundle, worklist = (read_json(getattr(args, name)) for name in ("certificate", "bundle", "worklist"))
    tasks = [*certified_tasks(config, certificate, bundle, worklist, family="rot13", model="14b"),
             *certified_tasks(config, certificate, bundle, worklist, family="kazitext", model="14b")]
    jobs = ordered_jobs(tasks)
    retained_identity = retained_fit(read_json(args.retained_fit), tasks, config)
    if config["spending"].get("authorized_usd", 0) <= 0:
        raise ValueError("Positive authorized spending amount required")
    deadline = utc(args.deadline)
    if datetime.now(timezone.utc) >= deadline:
        raise ValueError("Campaign deadline already passed")
    output = experiment.artifacts / "training/sft/campaign"
    output.mkdir(parents=True, exist_ok=True)
    plan = {"source_code_commit": experiment.code, "configuration_hash": experiment.hash,
            "handoff_certificate_hash": certificate["certificate_hash"],
            "retained_fit_identity": retained_identity, "new_job_order": [j["run_id"] for j in jobs],
            "authorized_usd": config["spending"]["authorized_usd"],
            "rate_usd_per_hour": None, "rate_based_cutoff": False,
            "deadline": deadline.isoformat(), "billing_api_used": False, "pod_shutdown_automated": False}
    plan["plan_hash"] = content_hash(plan)
    if args.validate_only:
        print(json.dumps({**plan, "status": "READY_TO_WAIT_FOR_BASELINE", "gpu_work_started": False}), flush=True)
        return 0
    stop, child, completed = {"requested": False}, None, []

    def request_pause(_signum, _frame):
        stop["requested"] = True
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)

    signal.signal(signal.SIGTERM, request_pause)
    signal.signal(signal.SIGINT, request_pause)

    def status(state, **extra):
        report = {"status": state, "plan_hash": plan["plan_hash"], "at": datetime.now(timezone.utc).isoformat(),
                  "source_code_commit": experiment.code, "configuration_hash": experiment.hash,
                  "retained_run_ids": ["rot13_14b_level_10"], "completed_new_run_ids": completed[:],
                  "required_new_fits": len(jobs), "deadline": deadline.isoformat(), **extra}
        path = output / "status.json"
        atomic_write_json(path, report)
        experiment.publish_permanent(path, {"kind": "sft_campaign_status"})
        print(json.dumps(report), flush=True)

    with file_lock(output / ".campaign.lock", timeout=1):
        atomic_write_json(output / "plan.json", plan)
        experiment.publish_permanent(output / "plan.json", {"kind": "sft_campaign_plan"})
        try:
            status("WAITING_FOR_GPQA_BASELINE", drive_status="VERIFIED")
            while True:
                if stop["requested"] or datetime.now(timezone.utc) >= deadline:
                    status("PAUSED", reason="deadline_or_stop", drive_status="VERIFIED")
                    return 3
                if Path(args.gpqa_receipt).is_file():
                    verify_gpqa_barrier(args, config)
                    break
                if not args.wait_for_baseline:
                    raise ValueError("GPQA completion receipt is not available")
                time.sleep(5)
            status("TRAINING", gpqa_drive_status="VERIFIED")
            for job in jobs:
                if stop["requested"] or datetime.now(timezone.utc) >= deadline:
                    status("PAUSED", reason="deadline_or_stop", drive_status="VERIFIED")
                    return 3
                status("TRAINING", current_run_id=job["run_id"])
                with (output / (job["run_id"] + ".log")).open("a") as log:
                    child = subprocess.Popen(training_command(args, job, deadline), cwd=ROOT,
                                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    returncode = child.wait()
                child = None
                if returncode == 3:
                    status("PAUSED", current_run_id=job["run_id"], drive_status="VERIFIED")
                    return 3
                if returncode:
                    raise RuntimeError(f"Certified fit {job['run_id']} exited {returncode}; inspect its log")
                verified_fit(experiment.artifacts, job, config, certificate["certificate_hash"], experiment.code)
                completed.append(job["run_id"])
                status("FIT_COMPLETE", current_run_id=job["run_id"], drive_status="VERIFIED")
            status("COMPLETE", drive_status="VERIFIED")
            return 0
        except Exception as error:
            status("FAILED", error_type=type(error).__name__, detail=str(error), drive_status="REQUIRED")
            raise


if __name__ == "__main__":
    raise SystemExit(main())
