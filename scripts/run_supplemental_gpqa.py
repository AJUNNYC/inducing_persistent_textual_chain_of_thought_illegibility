#!/usr/bin/env python3
"""Prepare, freeze, and run the appended GPQA comparison without changing SFT.

The José et al. implementation maps GPQA to gpqa_diamond. This tool defaults to
that verified 198-question source variant, while exposing gpqa_main explicitly.
It retains this project's native multiple-choice protocol; it does not claim to
replicate José's question-only prompting or LLM correctness grading.
"""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.supplemental_gpqa import (
    SOURCE_REVISION, VARIANTS, comparison_states, evaluate_gpqa, freeze_gpqa_protocol,
    load_official_gpqa, prepare_gpqa_panel, publish_immutable, validate_gpqa_panel,
    validate_gpqa_protocol,
)
from experiment_overview.experiment_overview import code_revision
from shared_procedures.storage import artifact_hook, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Acquire the gated official CSV and freeze its full panel")
    prepare.add_argument("--variant", choices=tuple(VARIANTS), default="gpqa_diamond")
    prepare.add_argument("--revision", default=SOURCE_REVISION)
    prepare.add_argument("--model-cache", required=True)
    prepare.add_argument("--offline", action="store_true")
    prepare.add_argument("--output", required=True)
    freeze = commands.add_parser("freeze", help="Predeclare one final adapter paired with its untouched base")
    freeze.add_argument("--panel", required=True)
    freeze.add_argument("--adapter", required=True, help="Published final_adapter directory with completed-fit report alongside")
    freeze.add_argument("--model-cache", required=True)
    freeze.add_argument("--backend", choices=("native", "vllm"), default="native")
    freeze.add_argument("--output", required=True)
    run = commands.add_parser("run", help="Run/resume both frozen states on the identical supplemental questions")
    run.add_argument("--protocol", required=True)
    run.add_argument("--output-root", required=True)
    run.add_argument("--validate-only", action="store_true", help="Check local pins and adapter bytes without GPU or network work")
    run.add_argument("--save-to-drive", action="store_true",
                     help="Synchronously save immutable evaluation artifacts using private GOOGLE_OAUTH_TOKEN_FILE")
    args = parser.parse_args()

    if args.command == "prepare":
        rows, source = load_official_gpqa(args.variant, args.model_cache, revision=args.revision,
                                         offline=args.offline)
        panel = prepare_gpqa_panel(rows, variant=args.variant, source=source)
        publish_immutable(args.output, panel)
        print(json.dumps({"status": "PREPARED", "variant": panel["variant"], "questions": panel["item_count"],
                          "panel_hash": panel["panel_hash"], "source": source}), flush=True)
        return 0
    if args.command == "freeze":
        panel = read_json(args.panel)
        validate_gpqa_panel(panel)
        states = comparison_states(args.adapter, args.model_cache)
        protocol = freeze_gpqa_protocol(panel, states, backend=args.backend)
        publish_immutable(args.output, protocol)
        print(json.dumps({"status": "FROZEN", "protocol_hash": protocol["protocol_hash"],
                          "states": [s["state_id"] for s in states], "source_code_commit": code_revision(),
                          "training_configuration_hash": states[1]["configuration_hash"], "backend": args.backend}), flush=True)
        return 0
    protocol = read_json(args.protocol)
    validate_gpqa_protocol(protocol)
    if args.validate_only:
        print(json.dumps({"status": "READY", "scope": protocol["scope"],
                          "questions_per_state": protocol["panel"]["item_count"],
                          "states": [s["state_id"] for s in protocol["states"]],
                          "protocol_hash": protocol["protocol_hash"], "gpu_work_started": False,
                          "network_requests_made": False}), flush=True)
        return 0
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    hook = nullcontext()
    if args.save_to_drive:
        from operations.artifacts import ArtifactService
        from operations.drive import DriveStore, drive_service
        token, folder = os.environ.get("GOOGLE_OAUTH_TOKEN_FILE"), os.environ.get("GOOGLE_DRIVE_FOLDER_ID")
        if not token or not folder:
            raise RuntimeError("Private Drive token path and project folder ID are required for verified saving")
        service = ArtifactService(root, root / "drive_registry.json", DriveStore(drive_service(token), folder), drive_recovery=True)
        provenance = {"experiment_id": protocol["protocol_hash"], "branch": "supplemental_gpqa",
                      "model": protocol["states"][0]["model"], "run_id": protocol["protocol_hash"][:16],
                      "stage": "supplemental_capability", "producer": "run_supplemental_gpqa",
                      "source_code_commit": code_revision(), "configuration_hash": protocol["protocol_hash"],
                      "dataset_revision": protocol["panel"]["source"]["revision"],
                      "model_revision": protocol["states"][0]["model_revision"], "seeds": [42]}
        hook = artifact_hook(service.completion_hook(provenance))
    with hook:
        # Publish the input protocol in the same namespace before any generation.
        publish_immutable(root / protocol["protocol_hash"] / "protocol.json", protocol)
        report = evaluate_gpqa(protocol, root)
    print(json.dumps({"status": report["status"], "scope": report["scope"],
                      "variant": report["variant"], "summaries": report["summaries"],
                      "accuracy_difference_trained_minus_base": report["accuracy_difference_trained_minus_base"],
                      "report_hash": report["report_hash"], "drive_status": "VERIFIED" if args.save_to_drive else "NOT_REQUESTED"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
