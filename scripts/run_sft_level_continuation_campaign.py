#!/usr/bin/env python3
"""Run an explicit shard of original 14B ROT13 level continuations serially.

One process owns one GPU queue; any number of workers may use disjoint --levels
subsets and separate output roots. Each child restores its own original parent,
publishes full checkpoints to Drive synchronously, and pauses after a whole
optimizer update. Existing one-epoch results and the level-10 pilot are inputs
only. Evaluation is a separate, later frozen campaign.
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

from experiment_overview.experiment_overview import code_revision
from operations.artifacts import validate_registered_record
from operations.drive import DriveStore, drive_service, file_hash as transport_hash
from restore_gpqa_sft_inputs import PROJECT_FOLDER_ID, immutable_json, safe_path
from restore_sft_level_continuation_inputs import validate_inputs_manifest
from shared_procedures.identity import content_hash, file_hash
from shared_procedures.storage import atomic_write_json, file_lock, read_json
from training.sft.level_continuation import ContinuationSpec, final_checkpoint, verified_completion

VERSION = "sft_level_continuation_campaign_v1"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def parse_levels(value):
    try:
        levels = [int(part) for part in value.split(",")]
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError("Levels must be comma-separated integers") from None
    if not levels or len(set(levels)) != len(levels) or any(not 1 <= item <= 10 for item in levels):
        raise argparse.ArgumentTypeError("Levels must be distinct integers from one to ten")
    return levels


def campaign_plan(manifest, entries, levels, *, source, additional_epochs, learning_rate):
    available = {entry["level"]: entry for entry in entries}
    if len(set(levels)) != len(levels) or not set(levels) <= available.keys():
        raise ValueError("Campaign levels must be distinct and present in the restored input manifest")
    jobs = []
    for level in levels:
        spec = ContinuationSpec(level, additional_epochs, learning_rate)
        entry = available[level]
        binding = {"level": level, "parent_checkpoint_hash": entry["parent_checkpoint_hash"],
                   "parent_identity_hash": entry["parent_identity_hash"], "policy": spec.policy(),
                   "source_code_commit": source, "inputs_manifest_hash": manifest["manifest_hash"]}
        jobs.append({**entry, "job_hash": content_hash(binding), "policy": spec.policy(),
                     "output_relative": f"jobs/level_{level:02d}_{content_hash(binding)[:12]}"})
    plan = {"version": VERSION, "source_code_commit": source,
            "inputs_manifest_hash": manifest["manifest_hash"],
            "levels": levels, "jobs": jobs,
            "worker_script_sha256": file_hash(ROOT / "scripts/continue_sft_level.py"),
            "training_module_sha256": file_hash(ROOT / "src/training/sft/level_continuation.py")}
    plan["plan_hash"] = content_hash(plan)
    return plan


def training_command(job, args, output):
    command = [args.python, "-u", str(ROOT / "scripts/continue_sft_level.py")]
    for name in ("parent_fit", "parent_checkpoint", "train", "validation"):
        command += ["--" + name.replace("_", "-"), job[name]]
    return command + ["--output", str(output), "--model-cache", args.model_cache,
                      "--level", str(job["level"]), "--additional-epochs", str(args.additional_epochs),
                      "--learning-rate", str(args.learning_rate), "--device", args.device]


def verified_job_completion(output, job, source, *, store):
    """A COMPLETE flag alone cannot skip a job: verify full state and live Drive."""
    receipt_path = safe_path(output, "completion_receipt.json")
    if not receipt_path.exists():
        return None
    identity = read_json(output / "run_identity.json")
    result = verified_completion(output, identity)
    receipt = read_json(receipt_path)
    payload_path = output / "drive_payload_manifest.json"
    payloads = read_json(payload_path)
    expected = {
        "version": "sft_level_continuation_completion_receipt_v2", "status": "COMPLETE",
        "drive_status": "VERIFIED", "simulation": False, "run_id": identity["run_id"],
        "configuration_hash": identity["configuration_hash"], "source_code_commit": source,
        "identity_hash": content_hash(identity), "parent_checkpoint_hash": job["parent_checkpoint_hash"],
        "continuation_policy": job["policy"], "fit_result_sha256": file_hash(output / "fit_result.json"),
        "checkpoint_hash": result["checkpoint_hash"], "adapter_manifest_hash": result["adapter_manifest_hash"],
        "progress": result["progress"], "payload_manifest_sha256": file_hash(payload_path),
    }
    if (receipt != {**expected, "receipt_hash": content_hash(expected)}
            or identity.get("code_revision") != source
            or identity.get("continuation_policy") != job["policy"]
            or identity.get("parent", {}).get("identity_hash") != job["parent_identity_hash"]
            or identity.get("parent", {}).get("checkpoint_hash") != job["parent_checkpoint_hash"]
            or payloads.get("version") != "sft_level_continuation_drive_payloads_v2"
            or payloads.get("configuration_hash") != identity["configuration_hash"]):
        raise ValueError("Continuation completion differs from this campaign's exact condition")
    registry_path = safe_path(output, "drive_registry.json")
    # A process can stop after writing its valid local receipt but before the
    # synchronous upload/export completes. Re-enter the child's idempotent
    # publication path; never skip or discard that valid scientific endpoint.
    if not registry_path.exists():
        return None
    records = read_json(registry_path).get("artifacts", {})
    if not isinstance(records, dict):
        raise ValueError("Continuation completion registry is malformed")
    if not records or any(record.get("sync_status") != "VERIFIED" for record in records.values()):
        return None
    required = [output / "run_identity.json", output / "fit_result.json", receipt_path, payload_path,
                *sorted(final_checkpoint(output, identity).glob("*")),
                *sorted(path for path in (output / "final_adapter").rglob("*") if path.is_file())]
    for path in required:
        safe_path(output, path.relative_to(output))
        matches = [record for record in records.values() if record.get("local_path") == str(path)]
        matches = [record for record in matches if record.get("sha256") == file_hash(path)]
        if not matches:
            return None
        if len(matches) != 1:
            raise ValueError("Continuation final payload lacks one exact verified Drive registry row")
        record = matches[0]
        validate_registered_record(record)
        if (record.get("configuration_hash") != identity["configuration_hash"]
                or record.get("source_code_commit") != source or record.get("size") != path.stat().st_size
                or record.get("md5") != transport_hash(path, "md5")
                or (path not in (receipt_path, payload_path)
                    and payloads.get("artifacts", {}).get(record["artifact_id"]) != record)):
            raise ValueError("Continuation verified Drive payload manifest differs from complete local bytes")
        store.verify_registered_artifact(record)
    return {"level": job["level"], "run_id": identity["run_id"], "output": str(output),
            "configuration_hash": identity["configuration_hash"],
            "checkpoint_hash": result["checkpoint_hash"], "adapter_manifest_hash": result["adapter_manifest_hash"],
            "completion_receipt_sha256": file_hash(receipt_path),
            "final_adapter": str(output / "final_adapter"), "drive_status": "VERIFIED"}


def unchanged_source(plan):
    if (code_revision() != plan["source_code_commit"]
            or file_hash(ROOT / "scripts/continue_sft_level.py") != plan["worker_script_sha256"]
            or file_hash(ROOT / "src/training/sft/level_continuation.py") != plan["training_module_sha256"]):
        raise RuntimeError("Continuation campaign source changed; preserving existing jobs")


def run_campaign(args, *, popen=subprocess.Popen, sleep=time.sleep, store=None):
    manifest = read_json(args.inputs_manifest)
    entries = validate_inputs_manifest(manifest)
    source = code_revision()
    plan = campaign_plan(manifest, entries, args.levels, source=source,
                         additional_epochs=args.additional_epochs, learning_rate=args.learning_rate)
    output = safe_path(args.output)
    model_cache = safe_path(args.model_cache)
    if not model_cache.is_dir():
        raise ValueError("Separately provisioned verified model/tokenizer cache is required")
    if any(output == Path(entry[key]) or output.is_relative_to(Path(entry[key]))
           for entry in entries for key in ("parent_fit", "parent_checkpoint")):
        raise ValueError("Campaign output cannot overwrite an original parent")
    if args.plan_only:
        return {"status": "PLANNED", "plan": plan, "gpu_work_started": False,
                "drive_requests_made": False}
    if not store:
        token = os.environ.get("GOOGLE_OAUTH_TOKEN_FILE")
        if not token or os.environ.get("GOOGLE_DRIVE_FOLDER_ID") != PROJECT_FOLDER_ID:
            raise ValueError("Private Google token file and canonical Drive folder are required")
        store = DriveStore(drive_service(token), PROJECT_FOLDER_ID, state_root=output / "state/drive")
        if store.create_project_folder() != PROJECT_FOLDER_ID:
            raise ValueError("Continuation campaign Drive project differs")
    stop = {"requested": False}
    child = {"process": None}

    def request_pause(_signum, _frame):
        stop["requested"] = True
        process = child["process"]
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGTERM)

    previous = {sig: signal.signal(sig, request_pause) for sig in (signal.SIGINT, signal.SIGTERM)}
    completed, locked = [], False
    try:
        with file_lock(output / "campaign.lock", timeout=1):
            locked = True
            immutable_json(output, output / "campaign_plan.json", plan)
            for job in plan["jobs"]:
                unchanged_source(plan)
                destination = safe_path(output, job["output_relative"])
                endpoint = verified_job_completion(destination, job, source, store=store)
                if endpoint is not None:
                    completed.append(endpoint)
                    continue
                if stop["requested"]:
                    break
                destination.mkdir(parents=True, exist_ok=True)
                state = {"version": VERSION, "status": "RUNNING", "plan_hash": plan["plan_hash"],
                         "active_level": job["level"], "completed": completed, "updated_at": utc_now()}
                atomic_write_json(output / "campaign_status.json", state)
                environment = dict(os.environ, SOURCE_CODE_COMMIT=source)
                with (destination / "worker.log").open("ab", buffering=0) as log:
                    child["process"] = popen(training_command(job, args, destination),
                        cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    # A signal can arrive between the preceding stop check and Popen.
                    if stop["requested"]:
                        child["process"].send_signal(signal.SIGTERM)
                    while child["process"].poll() is None:
                        sleep(1)
                    code = child["process"].returncode
                    child["process"] = None
                if code == 3:
                    # The child only returns this after its full pause state reaches Drive.
                    stop["requested"] = True
                    break
                if code != 0:
                    raise RuntimeError("Continuation worker failed; later levels were not started")
                unchanged_source(plan)
                endpoint = verified_job_completion(destination, job, source, store=store)
                if endpoint is None:
                    raise RuntimeError("Continuation worker exited without its verified full-state Drive receipt")
                completed.append(endpoint)
            status = "COMPLETE" if len(completed) == len(plan["jobs"]) else "PAUSED"
            result = {"version": VERSION, "status": status, "plan_hash": plan["plan_hash"],
                      "completed": completed, "active_level": None, "updated_at": utc_now()}
            atomic_write_json(output / "campaign_status.json", result)
            if status == "COMPLETE":
                endpoints = {"version": "sft_level_continuation_endpoints_v1", "status": "COMPLETE",
                             "drive_status": "VERIFIED", "plan_hash": plan["plan_hash"], "adapters": completed}
                endpoints["manifest_hash"] = content_hash(endpoints)
                immutable_json(output, output / "evaluation_endpoints.json", endpoints)
            return result
    except Exception as error:
        if locked:
            atomic_write_json(output / "campaign_status.json", {
                "version": VERSION, "status": "FAILED", "plan_hash": plan["plan_hash"],
                "completed": completed, "updated_at": utc_now(), "error_type": type(error).__name__})
        raise
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-manifest", required=True)
    parser.add_argument("--levels", type=parse_levels, required=True)
    parser.add_argument("--additional-epochs", type=int, choices=(1, 2), required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-cache", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--plan-only", action="store_true", help="Check local inputs and print plan without writes/GPU/Drive")
    result = run_campaign(parser.parse_args(argv))
    print(json.dumps(result), flush=True)
    return 3 if result["status"] == "PAUSED" else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"status": "FAILED", "error_type": type(error).__name__}), file=sys.stderr, flush=True)
        raise SystemExit(1) from None
