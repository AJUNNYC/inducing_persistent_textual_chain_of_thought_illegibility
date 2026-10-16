#!/usr/bin/env python3
"""Restore sealed original 14B ROT13 parents and unchanged data from Drive.

The private inventory supplies independently preserved complete-file SHA/size,
remote artifact identity and folder provenance. Only original 625-update parents
are accepted. The GPQA panel and the separate level-10 continuation pilot are
outside this restore allowlist. No GPU work or model-cache downloads occur.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from operations.drive import DriveStore, drive_service
from restore_gpqa_sft_inputs import (
    PROJECT_FOLDER_ID, MODEL_REVISION, ensure_directory, immutable_json,
    publish_existing_or_new, safe_path, verify_local, verify_remote,
)
from shared_procedures.checkpointing import verify_checkpoint
from shared_procedures.identity import content_hash
from shared_procedures.storage import read_json
from training.sft.individual_fit.sft_fit import verify_adapter_manifest

VERSION = "sft_level_continuation_inputs_v1"
RESTORED_VERSION = "sft_level_continuation_restored_inputs_v1"
DATA_ROOT = "data/vllm_sft_preparation/transformed"
_SHA = re.compile(r"[0-9a-f]{64}")
_MD5 = re.compile(r"[0-9a-f]{32}")
_PARENT = re.compile(r"training/sft/rot13/14b/level_(0[1-9]|10)/(?:fit_result\.json|"
                     r"final_adapter/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}|"
                     r"checkpoint_000625/(?:manifest\.json|full_state\.pt))")
_DATA = re.compile(DATA_ROOT + r"/rot13/14b/level_(0[1-9]|10)/(?:train\.jsonl|validation\.jsonl|manifest\.json)")


def sealed(value, field):
    if not isinstance(value, dict) or value.get(field) != content_hash({
            key: item for key, item in value.items() if key != field}):
        raise ValueError("Continuation input integrity seal differs")


def level_paths(level):
    parent = f"training/sft/rot13/14b/level_{level:02d}"
    data = f"{DATA_ROOT}/rot13/14b/level_{level:02d}"
    return {"parent_fit": parent, "parent_checkpoint": parent + "/checkpoint_000625",
            "train": data + "/train.jsonl", "validation": data + "/validation.jsonl"}


def validate_inventory(inventory):
    sealed(inventory, "inventory_hash")
    if (inventory.get("version") != VERSION or inventory.get("project_folder_id") != PROJECT_FOLDER_ID
            or inventory.get("family") != "rot13" or inventory.get("model") != "14b"
            or inventory.get("model_revision") != MODEL_REVISION):
        raise ValueError("Continuation input inventory identity differs")
    levels, artifacts = inventory.get("levels"), inventory.get("artifacts")
    if not isinstance(levels, list) or not levels or not isinstance(artifacts, list) or not artifacts:
        raise ValueError("Continuation inventory requires explicit levels and artifacts")
    selected = set()
    for entry in levels:
        level = entry.get("level")
        if type(level) is not int or not 1 <= level <= 10 or level in selected:
            raise ValueError("Continuation inventory levels must be distinct integers from one to ten")
        selected.add(level)
        if (any(entry.get(key) != path for key, path in level_paths(level).items())
                or any(not isinstance(entry.get(key), str) or not _SHA.fullmatch(entry[key])
                       for key in ("parent_checkpoint_hash", "parent_identity_hash"))):
            raise ValueError("Continuation inventory parent paths or immutable commitments differ")
    seen = {key: set() for key in ("relative_path", "artifact_id", "drive_file_id")}
    for record in artifacts:
        name = record.get("relative_path")
        match = _PARENT.fullmatch(name) or _DATA.fullmatch(name) if isinstance(name, str) else None
        if (not isinstance(name, str) or PurePosixPath(name).as_posix() != name or "\\" in name
                or (name != DATA_ROOT + "/manifest.json" and
                    (match is None or int(match.group(1)) not in selected))):
            raise ValueError("Continuation input path is outside the original-parent/data allowlist")
        if (any(not isinstance(record.get(key), str) or not _SHA.fullmatch(record[key])
                for key in ("artifact_id", "configuration_hash", "sha256"))
                or not isinstance(record.get("md5"), str) or not _MD5.fullmatch(record["md5"])
                or type(record.get("size")) is not int or record["size"] <= 0
                or any(not isinstance(record.get(key), str) or not record[key]
                       for key in ("drive_file_id", "drive_folder_id"))):
            raise ValueError("Continuation input lacks complete file and Drive commitments")
        for key, values in seen.items():
            if record[key] in values:
                raise ValueError("Continuation inventory contains duplicate artifact identities")
            values.add(record[key])
    required = {DATA_ROOT + "/manifest.json"}
    for entry in levels:
        paths = level_paths(entry["level"])
        required.update({paths["parent_fit"] + "/fit_result.json",
                         paths["parent_fit"] + "/final_adapter/manifest.json",
                         paths["parent_fit"] + "/final_adapter/adapter_config.json",
                         paths["parent_fit"] + "/final_adapter/adapter_model.safetensors",
                         paths["parent_checkpoint"] + "/manifest.json",
                         paths["parent_checkpoint"] + "/full_state.pt",
                         paths["train"], paths["validation"],
                         str(PurePosixPath(paths["train"]).parent / "manifest.json")})
    if not required <= seen["relative_path"]:
        raise ValueError("Continuation input inventory is missing required original-parent/data files")
    return artifacts


def validate_restored(root, inventory):
    """Hash every full file before a parent checkpoint can be deserialized later."""
    records = validate_inventory(inventory)
    by_path = {record["relative_path"]: record for record in records}
    for record in records:
        verify_local(safe_path(root, record["relative_path"]), record)
    entries = []
    for entry in inventory["levels"]:
        parent = root / entry["parent_fit"]
        result = read_json(parent / "fit_result.json")
        identity = result.get("identity", {})
        checkpoint = verify_checkpoint(root / entry["parent_checkpoint"], identity)
        adapter = verify_adapter_manifest(parent / "final_adapter", identity)
        progress = checkpoint["progress"]
        data_manifest = read_json(Path(root / entry["train"]).parent / "manifest.json")
        adapter_files = {str(p.relative_to(parent / "final_adapter"))
                         for p in (parent / "final_adapter").rglob("*") if p.is_file()}
        inventoried_adapter = {name.removeprefix(entry["parent_fit"] + "/final_adapter/")
                              for name in by_path if name.startswith(entry["parent_fit"] + "/final_adapter/")}
        if (result.get("status") != "complete" or identity.get("family") != "rot13"
                or identity.get("model_id") != "14b" or identity.get("level") != entry["level"]
                or identity.get("run_id") != f"rot13_14b_level_{entry['level']:02d}"
                or "parent" in identity or "continuation_policy" in identity
                or identity.get("model_revision") != MODEL_REVISION
                or identity.get("tokenizer_revision") != MODEL_REVISION
                or content_hash(identity) != entry["parent_identity_hash"]
                or checkpoint["checkpoint_hash"] != entry["parent_checkpoint_hash"]
                or result.get("checkpoint_hash") != checkpoint["checkpoint_hash"]
                or adapter.get("checkpoint_hash") != checkpoint["checkpoint_hash"]
                or result.get("adapter_manifest_hash") != adapter["manifest_hash"]
                or result.get("adapter_files") != adapter["files"]
                or adapter_files != {"manifest.json", *adapter["files"]}
                or inventoried_adapter != adapter_files
                or progress.get("completed_steps") != 625
                or progress.get("total_optimizer_steps") != 625
                or progress.get("example_presentations") != 10000
                or progress.get("sampler_position") != 10000
                or progress.get("physical_microbatch") != 1
                or progress.get("gradient_accumulation") != 16
                or progress.get("gradient_checkpointing") is not False
                or {k: v for k, v in result.get("progress", {}).items() if k != "status"} != progress
                or not {"adapter", "optimizer", "scheduler", "rng", "progress"} <= set(checkpoint.get("state_keys", []))
                or any(record["configuration_hash"] != identity.get("configuration_hash")
                       for name, record in by_path.items() if name.startswith(entry["parent_fit"] + "/"))
                or data_manifest.get("simulation") is not False
                or data_manifest.get("status") != "verified"
                or data_manifest.get("family") != "rot13" or data_manifest.get("model_id") != "14b"
                or data_manifest.get("level") != entry["level"]
                or data_manifest.get("train_count") != 10000
                or data_manifest.get("validation_count") != 100):
            raise ValueError("Restored continuation inputs do not bind the unchanged original one-epoch condition")
        payload = checkpoint["files"]["full_state.pt"]
        record = by_path[entry["parent_checkpoint"] + "/full_state.pt"]
        if payload != {"sha256": record["sha256"], "size_bytes": record["size"]}:
            raise ValueError("Original parent full-state payload differs from its independent commitment")
        entries.append({**entry, **{key: str(root / entry[key]) for key in level_paths(entry["level"])}})
    return entries


def restore_inventory(inventory, destination_root, *, store_factory=None, workers=1, validate_only=False):
    records = validate_inventory(inventory)
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Restore workers must be between one and four")
    root = safe_path(destination_root)
    for record in records:
        path = safe_path(root, record["relative_path"])
        if path.exists():
            verify_local(path, record)
    summary = {"version": "sft_level_continuation_restore_summary_v1",
               "inventory_hash": inventory["inventory_hash"], "artifact_count": len(records),
               "levels": [entry["level"] for entry in inventory["levels"]]}
    if not validate_only:
        if store_factory is None:
            raise ValueError("Continuation restore requires the restricted project Drive connection")
        ensure_directory(root, root)
        local, failed = threading.local(), threading.Event()

        def restore(record):
            if failed.is_set():
                raise RuntimeError("Continuation restore stopped after an integrity/transport failure")
            try:
                if not hasattr(local, "store"):
                    local.store = store_factory()
                    if (local.store.folder_id != PROJECT_FOLDER_ID
                            or local.store.create_project_folder() != PROJECT_FOLDER_ID):
                        raise ValueError("Continuation restore Drive project folder differs")
                store = local.store
                path = safe_path(root, record["relative_path"])
                verify_remote(store, record)
                existed = path.exists()
                if not existed:
                    ensure_directory(root, path.parent)
                    with tempfile.TemporaryDirectory(prefix=".continuation_restore_", dir=path.parent) as temporary:
                        staged = Path(temporary) / "payload"
                        restored = store.download_artifact(record["artifact_id"], staged,
                                                          config_hash=record["configuration_hash"])
                        if Path(restored) != staged:
                            raise ValueError("Drive download returned an unexpected destination")
                        verify_local(staged, record)
                        verify_remote(store, record)
                        publish_existing_or_new(root, path, staged, record)
                verify_local(safe_path(root, record["relative_path"]), record)
                verify_remote(store, record)
                return existed
            except Exception:
                failed.set()
                raise

        with ThreadPoolExecutor(max_workers=workers) as executor:
            reused = list(executor.map(restore, records))
        summary.update(downloaded=sum(not item for item in reused), reused=sum(reused))
    entries = validate_restored(root, inventory)
    summary["status"] = "VALIDATED" if validate_only else "VERIFIED"
    if not validate_only:
        manifest = {"version": RESTORED_VERSION, "status": "VERIFIED",
                    "project_folder_id": PROJECT_FOLDER_ID, "model_revision": MODEL_REVISION,
                    "inventory": inventory, "destination_root": str(root), "levels": entries}
        manifest["manifest_hash"] = content_hash(manifest)
        target = root / "sft_level_continuation_inputs_manifest.json"
        immutable_json(root, target, manifest)
        summary.update(inputs_manifest=str(target), manifest_hash=manifest["manifest_hash"])
    return summary


def validate_inputs_manifest(manifest):
    sealed(manifest, "manifest_hash")
    if (manifest.get("version") != RESTORED_VERSION or manifest.get("status") != "VERIFIED"
            or manifest.get("project_folder_id") != PROJECT_FOLDER_ID
            or manifest.get("model_revision") != MODEL_REVISION):
        raise ValueError("Restored continuation manifest identity differs")
    root = safe_path(manifest["destination_root"])
    entries = validate_restored(root, manifest["inventory"])
    if entries != manifest.get("levels"):
        raise ValueError("Restored continuation manifest paths or parent commitments differ")
    return entries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--destination-root", required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--validate-only", action="store_true", help="Check restored bytes without network or writes")
    args = parser.parse_args(argv)
    inventory = read_json(args.inventory)
    factory = None
    if not args.validate_only:
        token = os.environ.get("GOOGLE_OAUTH_TOKEN_FILE")
        folder = os.environ.get("GOOGLE_DRIVE_FOLDER_ID")
        if not token or folder != PROJECT_FOLDER_ID:
            raise ValueError("Private Google token file and canonical Drive folder are required")
        state = safe_path(args.destination_root, "state/drive_restore")
        factory = lambda: DriveStore(drive_service(token), folder, state_root=state)
    print(json.dumps(restore_inventory(inventory, args.destination_root, store_factory=factory,
                                      workers=args.workers, validate_only=args.validate_only)), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"status": "FAILED", "error_type": type(error).__name__}), file=sys.stderr, flush=True)
        raise SystemExit(1) from None
