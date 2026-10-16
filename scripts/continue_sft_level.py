#!/usr/bin/env python3
"""Continue one original 14B ROT13 level with explicitly declared extra passes.

The original fit remains immutable. Each level requires its own verified full
adapter/AdamW/RNG parent. Checkpoints are synchronously copied to Drive, signals
pause after a whole optimizer update, and completion requires a sealed Drive
receipt. The separate fixed level-10 v1 pilot is never reused or relabelled.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_generation_collection.sft_data.sft_data_preparation import load_sft_records
from experiment_overview.experiment_overview import code_revision
from operations.artifacts import ArtifactService, validate_registered_record
from operations.drive import DriveStore, drive_service
from shared_procedures.identity import content_hash, file_hash
from shared_procedures.recovery_procedure import configure_worker_recovery
from shared_procedures.storage import artifact_hook, atomic_write_json, notify_artifact, read_json
from training.sft.level_continuation import (
    ContinuationSpec, final_checkpoint, prepare_inputs, run_continuation, verified_completion,
    validate_output_directory,
)


def verified_drive_records(service, output: Path, identity: dict) -> list[dict]:
    """Require matching checksummed registry rows for every final scientific payload."""
    paths = [output / "run_identity.json", output / "fit_result.json",
             *sorted(final_checkpoint(output, identity).glob("*")),
             *sorted(path for path in (output / "final_adapter").rglob("*") if path.is_file())]
    records = service.list_artifacts()
    for path in paths:
        matches = [record for record in records if record.get("local_path") == str(path)
                   and record.get("sha256") == file_hash(path)]
        if (not matches or any(record.get("sync_status") != "VERIFIED"
                or not record.get("drive_file_id") or not record.get("drive_folder_id")
                or record.get("configuration_hash") != identity["configuration_hash"]
                or record.get("source_code_commit") != identity["code_revision"] for record in matches)):
            raise RuntimeError("Final continuation payload lacks its verified Drive checksum record")
        for record in matches:
            validate_registered_record(record)
    if any(record.get("sync_status") != "VERIFIED" for record in records):
        raise RuntimeError("Continuation Drive barrier still contains unverified artifacts")
    return records


def publish_drive_completion(service, output: Path, identity: dict) -> dict:
    result = verified_completion(output, identity)
    if service.sync_artifacts()["status"] != "VERIFIED":
        raise RuntimeError("Continuation Drive barrier is incomplete")
    records = verified_drive_records(service, output, identity)
    snapshot_path = output / "drive_payload_manifest.json"
    snapshot = {"version": "sft_level_continuation_drive_payloads_v2",
                "configuration_hash": identity["configuration_hash"],
                "artifacts": {record["artifact_id"]: record for record in records
                    if Path(record["relative_path"]).name not in
                    ("drive_payload_manifest.json", "completion_receipt.json")}}
    if snapshot_path.exists():
        if read_json(snapshot_path) != snapshot:
            raise RuntimeError("Immutable continuation Drive payload manifest differs")
    else:
        atomic_write_json(snapshot_path, snapshot)
    notify_artifact(snapshot_path, {"kind": "sft_level_continuation_drive_payload_manifest"})
    receipt = {
        "version": "sft_level_continuation_completion_receipt_v2",
        "status": "COMPLETE", "drive_status": "VERIFIED", "simulation": False,
        "run_id": identity["run_id"], "configuration_hash": identity["configuration_hash"],
        "source_code_commit": identity["code_revision"], "identity_hash": content_hash(identity),
        "parent_checkpoint_hash": identity["parent"]["checkpoint_hash"],
        "continuation_policy": identity["continuation_policy"],
        "fit_result_sha256": file_hash(output / "fit_result.json"),
        "checkpoint_hash": result["checkpoint_hash"],
        "adapter_manifest_hash": result["adapter_manifest_hash"],
        "progress": result["progress"], "payload_manifest_sha256": file_hash(snapshot_path),
    }
    receipt["receipt_hash"] = content_hash(receipt)
    receipt_path = output / "completion_receipt.json"
    if receipt_path.exists():
        if read_json(receipt_path) != receipt:
            raise RuntimeError("Immutable continuation completion receipt differs")
    else:
        atomic_write_json(receipt_path, receipt)
    notify_artifact(receipt_path, {"kind": "sft_level_continuation_drive_completion"})
    if service.sync_artifacts()["status"] != "VERIFIED":
        raise RuntimeError("Continuation completion receipt is not verified in Drive")
    verified_drive_records(service, output, identity)
    service.export_manifest()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("parent-fit", "parent-checkpoint", "train", "validation", "output", "model-cache"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--level", type=int, choices=range(1, 11), required=True)
    parser.add_argument("--additional-epochs", type=int, choices=(1, 2), required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--validate-only", action="store_true",
                        help="Verify original full parent and unchanged data without GPU or Drive requests")
    args = parser.parse_args()
    continuation = ContinuationSpec(args.level, args.additional_epochs, args.learning_rate)
    output = validate_output_directory(args.output, args.parent_fit, args.parent_checkpoint)
    train, validation = load_sft_records(args.train), load_sft_records(args.validation)
    source = code_revision()
    if args.validate_only:
        identity, data, _train, _validation = prepare_inputs(args.parent_fit, args.parent_checkpoint,
            train, validation, source, continuation=continuation, cache_root=args.model_cache)
        print(json.dumps({"status": "READY", "identity": identity, "data": data,
                          "gpu_work_started": False, "drive_requests_made": False}), flush=True)
        return 0
    token, folder = os.environ.get("GOOGLE_OAUTH_TOKEN_FILE"), os.environ.get("GOOGLE_DRIVE_FOLDER_ID")
    if not token or not folder:
        raise RuntimeError("Private Drive token path and project folder ID are required")
    configure_worker_recovery(output, worker_id=f"sft_rot13_14b_level_{args.level:02d}_continuation_v2")
    service = ArtifactService(output, output / "drive_registry.json",
                              DriveStore(drive_service(token), folder), drive_recovery=True)
    def preserve(path, metadata):
        identity = read_json(output / "run_identity.json")
        provenance = {
            "experiment_id": identity["configuration_hash"], "branch": "rot13_level_continuation",
            "model": "14b", "run_id": identity["run_id"], "stage": "sft_level_continuation",
            "producer": "continue_sft_level", "source_code_commit": source,
            "configuration_hash": identity["configuration_hash"],
            "dataset_revision": identity["dataset_hash"], "model_revision": identity["model_revision"],
            "seeds": [42],
        }
        service.completion_hook(provenance)(path, metadata)
    stop = {"requested": False}
    def request_pause(_signum, _frame):
        stop["requested"] = True
    signal.signal(signal.SIGTERM, request_pause)
    signal.signal(signal.SIGINT, request_pause)
    with artifact_hook(preserve):
        result = run_continuation(args.parent_fit, args.parent_checkpoint, train, validation,
            output, source_revision=source, continuation=continuation, cache_root=args.model_cache,
            device=args.device, should_pause=lambda: stop["requested"])
        if result["status"] == "complete":
            publish_drive_completion(service, output, result["identity"])
        else:
            if service.sync_artifacts()["status"] != "VERIFIED":
                raise RuntimeError("Paused continuation full checkpoint has not reached Drive")
            service.export_manifest()
    print(json.dumps({"status": result["status"], "run_id": result["identity"]["run_id"],
                      "progress": result["progress"], "drive_status": "VERIFIED"}), flush=True)
    return 3 if result["status"] == "paused" else 0


if __name__ == "__main__":
    raise SystemExit(main())
