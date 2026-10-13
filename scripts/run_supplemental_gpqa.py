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
    BASELINE_VERSION, SOURCE_REVISION, VARIANTS, baseline_state, comparison_states,
    evaluate_gpqa, freeze_gpqa_baseline, freeze_gpqa_protocol,
    load_official_gpqa, prepare_gpqa_panel, publish_immutable, validate_gpqa_panel,
    validate_gpqa_protocol,
)
from experiment_overview.experiment_overview import code_revision
from shared_procedures.identity import content_hash, file_hash
from shared_procedures.storage import artifact_hook, atomic_write_json, read_json


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
    freeze.add_argument("--baseline-protocol", help="Reuse matching base raw evidence under the same output root")
    base = commands.add_parser("freeze-base", help="Predeclare the untouched base without requiring a trained adapter")
    base.add_argument("--panel", required=True)
    base.add_argument("--model", choices=("7b", "14b"), required=True)
    pins = base.add_mutually_exclusive_group(required=True)
    pins.add_argument("--config", help="Read official model/tokenizer commits from the project configuration")
    pins.add_argument("--model-revision", help="Explicit immutable model commit; also requires --tokenizer-revision")
    base.add_argument("--tokenizer-revision")
    base.add_argument("--model-cache", required=True)
    base.add_argument("--backend", choices=("native", "vllm"), default="vllm")
    base.add_argument("--output", required=True)
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
        protocol = freeze_gpqa_protocol(panel, states, backend=args.backend,
                                       baseline_protocol=read_json(args.baseline_protocol) if args.baseline_protocol else None)
        publish_immutable(args.output, protocol)
        print(json.dumps({"status": "FROZEN", "protocol_hash": protocol["protocol_hash"],
                          "states": [s["state_id"] for s in states], "source_code_commit": code_revision(),
                          "training_configuration_hash": states[1]["configuration_hash"], "backend": args.backend}), flush=True)
        return 0
    if args.command == "freeze-base":
        if args.config:
            if args.tokenizer_revision:
                parser.error("--config cannot be mixed with explicit tokenizer pins")
            from shared_procedures.configuration import load_configuration
            from evaluation.shared_model_evaluation.item_evaluation.native_output.output_generation import MODEL_IDS
            model = load_configuration(args.config)["models"][args.model]
            if model["id"] != MODEL_IDS[args.model]:
                raise ValueError("GPQA baseline model differs from the official construction model")
            model_revision, tokenizer_revision = model["revision"], model["tokenizer_revision"]
        else:
            if not args.tokenizer_revision:
                parser.error("--model-revision requires --tokenizer-revision")
            model_revision, tokenizer_revision = args.model_revision, args.tokenizer_revision
        state = baseline_state(args.model, args.model_cache, model_revision=model_revision,
                               tokenizer_revision=tokenizer_revision)
        protocol = freeze_gpqa_baseline(read_json(args.panel), state, backend=args.backend)
        publish_immutable(args.output, protocol)
        print(json.dumps({"status": "FROZEN", "protocol_hash": protocol["protocol_hash"],
                          "states": [state["state_id"]], "source_code_commit": code_revision(),
                          "backend": args.backend, "trained_adapter_required": False}), flush=True)
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
    service = None
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
        if service is not None:
            # An immutable snapshot binds verified payloads even after the live
            # registry grows during later adapter comparisons.
            service.export_manifest()
            registry_path = root / protocol["protocol_hash"] / "drive_manifest.json"
            # A resumed successful run keeps its original inventory; the live
            # registry now also contains this inventory's own artifact record.
            publish_immutable(registry_path, read_json(registry_path) if registry_path.exists()
                              else read_json(service.registry))
    report_path = root / protocol["protocol_hash"] / ("baseline.json" if protocol["version"] == BASELINE_VERSION else "comparison.json")
    receipt = {"version": "supplemental_gpqa_completion_receipt_v1", "status": "COMPLETE",
               "drive_status": "VERIFIED" if service is not None else "NOT_REQUESTED",
               "protocol_hash": protocol["protocol_hash"], "panel_hash": protocol["panel"]["panel_hash"],
               "source_code_commit": code_revision(), "model": protocol["states"][0]["model"],
               "model_revision": protocol["states"][0]["model_revision"],
               "tokenizer_revision": protocol["states"][0]["tokenizer_revision"], "seed": 42,
               "simulation": report["simulation"], "summaries": report["summaries"],
               "report_path": str(report_path), "report_sha256": file_hash(report_path),
               "report_hash": report["report_hash"],
               "registry_path": str(registry_path) if service is not None else None,
               "registry_sha256": file_hash(registry_path) if service is not None else None}
    receipt["receipt_hash"] = content_hash(receipt)
    # This is the local orchestration barrier, written only after every payload
    # and the immutable Drive inventory have completed synchronous verification.
    atomic_write_json(root / protocol["protocol_hash"] / "completion_receipt.json", receipt)
    print(json.dumps({"status": report["status"], "scope": report["scope"],
                      "variant": report["variant"], "summaries": report["summaries"],
                      "accuracy_difference_trained_minus_base": report["accuracy_difference_trained_minus_base"],
                      "report_hash": report["report_hash"], "drive_status": "VERIFIED" if args.save_to_drive else "NOT_REQUESTED"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
