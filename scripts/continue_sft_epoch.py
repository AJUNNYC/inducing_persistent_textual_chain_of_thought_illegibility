#!/usr/bin/env python3
"""Continue completed14B ROT13 level10 for one separately identified extra epoch.

The parent625-update fit remains immutable. Actual adapter/AdamW/RNG checkpoint
state is required; an adapter-only export cannot substitute. Checkpoints are
synchronously verified in Drive. SIGTERM/SIGINT pauses at a whole optimizer
update. No epoch1 training, generation or GPQA evaluation is rerun.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment_overview.experiment_overview import code_revision
from operations.artifacts import ArtifactService
from operations.drive import DriveStore, drive_service
from shared_procedures.recovery_procedure import configure_worker_recovery
from shared_procedures.storage import artifact_hook
from training.sft.continuation import prepare_inputs, run_continuation
from data_generation_collection.sft_data.sft_data_preparation import load_sft_records
from shared_procedures.storage import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("parent-fit", "parent-checkpoint", "train", "validation", "output", "model-cache"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--validate-only", action="store_true",
                        help="Verify trusted full parent and unchanged dataset; no GPU or Drive requests")
    args = parser.parse_args()
    train, validation = load_sft_records(args.train), load_sft_records(args.validation)
    source = code_revision()
    if args.validate_only:
        identity, data, _train_tokens, _validation_tokens = prepare_inputs(
            args.parent_fit, args.parent_checkpoint, train, validation, source, cache_root=args.model_cache)
        print(json.dumps({"status": "READY", "identity": identity, "data": data,
                          "gpu_work_started": False, "drive_requests_made": False}), flush=True)
        return 0
    token, folder = os.environ.get("GOOGLE_OAUTH_TOKEN_FILE"), os.environ.get("GOOGLE_DRIVE_FOLDER_ID")
    if not token or not folder:
        raise RuntimeError("Private Drive token path and project folder ID are required")
    output = Path(args.output).resolve()
    configure_worker_recovery(output, worker_id="sft_rot13_14b_extra_epoch")
    service = ArtifactService(output, output / "drive_registry.json",
        DriveStore(drive_service(token), folder), drive_recovery=True)
    def preserve(path, metadata):
        identity = read_json(output / "run_identity.json")
        provenance = {"experiment_id": identity["configuration_hash"], "branch": "rot13_continuation",
                      "model": "14b", "run_id": identity["run_id"], "stage": "sft_continuation",
                      "producer": "continue_sft_epoch", "source_code_commit": source,
                      "configuration_hash": identity["configuration_hash"],
                      "dataset_revision": identity["dataset_hash"],
                      "model_revision": identity["model_revision"], "seeds": [42]}
        service.completion_hook(provenance)(path, metadata)
    stop = {"requested": False}
    def request_pause(_signum, _frame):
        stop["requested"] = True
    signal.signal(signal.SIGTERM, request_pause)
    signal.signal(signal.SIGINT, request_pause)
    with artifact_hook(preserve):
        result = run_continuation(args.parent_fit, args.parent_checkpoint, train, validation,
            output, source_revision=source, cache_root=args.model_cache, device=args.device,
            should_pause=lambda: stop["requested"])
        if service.sync_artifacts()["status"] != "VERIFIED":
            raise RuntimeError("Continuation Drive barrier is incomplete")
        service.export_manifest()
    print(json.dumps({"status": result["status"], "run_id": result["identity"]["run_id"],
                      "progress": result["progress"], "drive_status": "VERIFIED"}), flush=True)
    return 3 if result["status"] == "paused" else 0


if __name__ == "__main__":
    raise SystemExit(main())
