#!/usr/bin/env python3
"""Restore only the preselected 14B ROT13 final adapters and frozen GPQA panel.

The sealed inventory is supplied separately from the private Drive credential.
Current remote identity, folder ancestry and checksums are verified both before
and after transfer. Existing different bytes and symbolic links are never used
or overwritten. This performs no training, checkpoint selection or GPU work.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tempfile
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.supplemental_gpqa import SOURCE_REVISION, comparison_states, validate_gpqa_panel
from operations.drive import DriveStore, drive_service
from shared_procedures.identity import canonical_json, content_hash, file_hash
from shared_procedures.storage import fsync_directory, read_json

VERSION = "gpqa_sft_restore_inventory_v1"
PROJECT_FOLDER_ID = "176a8Tm6l4WhfakY_0nCOo4ICeT7KPSmx"
MODEL_REVISION = "1df8507178afcc1bef68cd8c393f61a886323761"
PANEL_PATH = "evaluation/gpqa_diamond_panel.json"
_SHA = re.compile(r"[0-9a-f]{64}")
_MD5 = re.compile(r"[0-9a-f]{32}")
_ADAPTER = re.compile(r"training/sft/rot13/14b/level_(0[1-9]|10)/final_adapter/([A-Za-z0-9][A-Za-z0-9_.-]{0,127})")
_FIT = re.compile(r"training/sft/rot13/14b/level_(0[1-9]|10)/fit_result\.json")


def validate_inventory(inventory):
    """Reject other branches, intermediate/full states and duplicate identities."""
    if (not isinstance(inventory, dict) or inventory.get("version") != VERSION
            or inventory.get("project_folder_id") != PROJECT_FOLDER_ID
            or inventory.get("family") != "rot13" or inventory.get("model") != "14b"
            or inventory.get("generation_profile") != "long_32000"
            or inventory.get("inventory_hash") != content_hash({k: v for k, v in inventory.items()
                                                                 if k != "inventory_hash"})):
        raise ValueError("GPQA restore inventory identity or seal differs")
    records = inventory.get("artifacts")
    if not isinstance(records, list) or not records:
        raise ValueError("GPQA restore inventory lacks artifacts")
    seen = {key: set() for key in ("relative_path", "artifact_id", "drive_file_id")}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Invalid GPQA restore artifact")
        name = record.get("relative_path")
        if (not isinstance(name, str) or PurePosixPath(name).as_posix() != name
                or "\\" in name or not (name == PANEL_PATH or _FIT.fullmatch(name)
                                            or _ADAPTER.fullmatch(name))
                or any(term in name.lower() for term in ("optimizer", "checkpoint", "full_state", "fullstate"))):
            raise ValueError("GPQA restore path is outside the final-adapter/panel allowlist")
        for key in ("artifact_id", "configuration_hash", "sha256"):
            if not isinstance(record.get(key), str) or not _SHA.fullmatch(record[key]):
                raise ValueError("GPQA restore artifact lacks immutable integrity/provenance")
        if (not isinstance(record.get("md5"), str) or not _MD5.fullmatch(record["md5"])
                or type(record.get("size")) is not int or record["size"] < 0
                or any(not isinstance(record.get(key), str) or not record[key]
                       for key in ("drive_file_id", "drive_folder_id"))):
            raise ValueError("GPQA restore artifact lacks complete remote identity/checksums")
        for key, values in seen.items():
            if record[key] in values:
                raise ValueError("GPQA restore inventory contains duplicate paths or remote identities")
            values.add(record[key])
    required = {PANEL_PATH}
    for level in range(1, 11):
        fit = f"training/sft/rot13/14b/level_{level:02d}"
        required.update({f"{fit}/fit_result.json", f"{fit}/final_adapter/manifest.json",
                         f"{fit}/final_adapter/adapter_config.json", f"{fit}/final_adapter/adapter_model.safetensors"})
    if not required <= seen["relative_path"]:
        raise ValueError("GPQA restore requires all ten final adapters, matching fit results and panel")
    return records


def safe_path(root, relative=None):
    """Check every existing ancestor without resolving through symbolic links."""
    root = Path(os.path.abspath(Path(root).expanduser()))
    target = root if relative is None else root / relative
    if not target.is_relative_to(root):
        raise ValueError("GPQA restore destination escapes its root")
    for path in reversed([target, *target.parents]):
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ValueError("GPQA restore refuses symbolic links in destinations or ancestors")
        if path != target and not stat.S_ISDIR(mode):
            raise ValueError("GPQA restore destination ancestor is not a directory")
    return target


def ensure_directory(root, directory):
    safe_path(root, directory.relative_to(root))
    directory.mkdir(parents=True, exist_ok=True)
    safe_path(root, directory.relative_to(root))
    if not directory.is_dir():
        raise ValueError("GPQA restore directory is not a directory")


def verify_local(path, record):
    if (path.is_symlink() or not path.is_file() or path.stat().st_size != record["size"]
            or file_hash(path) != record["sha256"]):
        raise ValueError("GPQA restored complete-file integrity differs")
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != record["md5"]:
        raise ValueError("GPQA restored complete-file MD5 differs")


def verify_remote(store, record):
    # Each thread owns its store. Force a fresh ancestry walk rather than relying
    # on an earlier cache if the provider moved a folder during transfer.
    store._verified_folders = {store.folder_id}
    store.verify_registered_artifact(record)
    remote = store.resolve_artifact_id(record["artifact_id"])
    if remote is None or remote.get("id") != record["drive_file_id"]:
        raise ValueError("GPQA restore artifact lookup differs from its sealed inventory")


def publish_existing_or_new(root, path, staged, record):
    safe_path(root, path.relative_to(root))
    try:
        os.link(staged, path)  # Atomic creation; cannot replace an existing file.
    except FileExistsError:
        safe_path(root, path.relative_to(root))
        verify_local(path, record)
    fsync_directory(path.parent)


def immutable_json(root, path, value):
    path = safe_path(root, path.relative_to(root))
    ensure_directory(root, path.parent)
    payload = canonical_json(value) + b"\n"
    record = {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
              "md5": hashlib.md5(payload).hexdigest()}
    if path.exists():
        verify_local(path, record)
        return
    with tempfile.TemporaryDirectory(prefix=".gpqa_publish_", dir=path.parent) as temporary:
        staged = Path(temporary) / "payload"
        with staged.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        publish_existing_or_new(root, path, staged, record)


def validate_restored(root, records):
    by_path = {record["relative_path"]: record for record in records}
    for record in records:
        verify_local(safe_path(root, record["relative_path"]), record)
    panel = read_json(root / PANEL_PATH)
    validate_gpqa_panel(panel)
    if (panel["variant"] != "gpqa_diamond" or panel["item_count"] != 198
            or panel["source"]["revision"] != SOURCE_REVISION):
        raise ValueError("GPQA restore panel differs from the pinned full Diamond198 panel")
    adapters, bindings = [], []
    for level in range(1, 11):
        prefix = f"training/sft/rot13/14b/level_{level:02d}"
        adapter_relative = prefix + "/final_adapter"
        adapter = root / adapter_relative
        manifest = read_json(adapter / "manifest.json")
        files = manifest.get("files")
        if (not isinstance(files, dict) or any(not _ADAPTER.fullmatch(f"{adapter_relative}/{name}")
                                               or name == "manifest.json" for name in files)
                or {name for name in by_path if name.startswith(adapter_relative + "/")}
                != {f"{adapter_relative}/{name}" for name in ["manifest.json", *files]}):
            raise ValueError("GPQA restore adapter inventory differs from its original manifest")
        actual = {str(p.relative_to(adapter)) for p in adapter.iterdir()}
        if actual != {"manifest.json", *files}:
            raise ValueError("GPQA restore adapter contains uninventoried files")
        states = comparison_states(adapter, root / "model_cache")
        trained = states[1]
        if (trained["model"] != "14b" or trained["family"] != "rot13" or trained["level"] != level
                or any(state[key] != MODEL_REVISION for state in states
                       for key in ("model_revision", "tokenizer_revision"))
                or any(record["configuration_hash"] != trained["configuration_hash"] for name, record in by_path.items()
                       if name.startswith(prefix + "/"))):
            raise ValueError("GPQA restored original training/model/tokenizer identity differs")
        adapters.append({"level": level, "final_adapter": str(adapter)})
        bindings.append({"level": level, "training_identity_hash": content_hash(manifest["identity"]),
                         "configuration_hash": trained["configuration_hash"],
                         "adapter_manifest_hash": manifest["manifest_hash"],
                         "checkpoint_hash": manifest["checkpoint_hash"]})
    return panel, {"version": "gpqa_sft_adapters_manifest_v1", "family": "rot13", "model": "14b",
                   "adapters": adapters}, bindings


def restore_inventory(inventory, destination_root, *, store_factory=None, workers=1,
                      validate_only=False, receipt_path=None):
    records = validate_inventory(inventory)
    if type(workers) is not int or workers not in range(1, 5):
        raise ValueError("GPQA restore workers must be between one and four")
    root = safe_path(destination_root)
    if root.exists() and not root.is_dir():
        raise ValueError("GPQA restore root is not a directory")
    for record in records:
        path = safe_path(root, record["relative_path"])
        if path.exists():
            verify_local(path, record)  # Fail before networking or any download.
    summary = {"version": "gpqa_sft_restore_summary_v1", "status": "VALIDATED" if validate_only else "VERIFIED",
               "inventory_hash": inventory["inventory_hash"], "artifact_count": len(records),
               "adapters_manifest": str(root / "gpqa_sft_adapters_manifest.json"),
               "panel": str(root / PANEL_PATH)}
    if not validate_only:
        if store_factory is None:
            raise ValueError("GPQA restore requires the restricted project Drive connection")
        ensure_directory(root, root)
        local = threading.local()
        failed = threading.Event()

        def restore_record(record):
            if not hasattr(local, "store"):
                local.store = store_factory()
                if (local.store.folder_id != inventory["project_folder_id"]
                        or local.store.create_project_folder() != inventory["project_folder_id"]):
                    raise ValueError("GPQA restore Drive project folder differs")
            store = local.store
            path = safe_path(root, record["relative_path"])
            verify_remote(store, record)
            existed = path.exists()
            if existed:
                verify_local(path, record)
            else:
                ensure_directory(root, path.parent)
                with tempfile.TemporaryDirectory(prefix=".gpqa_restore_", dir=path.parent) as temporary:
                    staged = Path(temporary) / "payload"
                    restored = store.download_artifact(record["artifact_id"], staged,
                                                       config_hash=record["configuration_hash"])
                    if Path(restored) != staged:
                        raise ValueError("GPQA Drive download returned an unexpected destination")
                    verify_local(staged, record)
                    verify_remote(store, record)
                    publish_existing_or_new(root, path, staged, record)
            verify_remote(store, record)
            verify_local(safe_path(root, record["relative_path"]), record)
            return existed

        def restore(record):
            if failed.is_set():
                raise RuntimeError("GPQA restore stopped after an artifact integrity/transport failure")
            try:
                return restore_record(record)
            except Exception:
                failed.set()
                raise

        with ThreadPoolExecutor(max_workers=workers) as executor:
            reused = list(executor.map(restore, records))
        summary.update(downloaded=sum(not value for value in reused), reused=sum(reused))
    panel, adapters_manifest, bindings = validate_restored(root, records)
    summary["panel_hash"] = panel["panel_hash"]
    if not validate_only:
        receipt = {"version": "gpqa_sft_restore_receipt_v1", "status": "VERIFIED",
                   "inventory_hash": inventory["inventory_hash"], "project_folder_id": PROJECT_FOLDER_ID,
                   "panel_hash": panel["panel_hash"], "artifact_count": len(records),
                   "adapters_manifest_hash": content_hash(adapters_manifest), "adapter_bindings": bindings,
                   "verification": "current_remote_metadata_and_ancestry_before_after_complete_local_bytes"}
        receipt["receipt_hash"] = content_hash(receipt)
        receipt_path = (Path(os.path.abspath(Path(receipt_path).expanduser())) if receipt_path
                        else root / "gpqa_sft_restore_receipt.json")
        if not receipt_path.is_relative_to(root):
            raise ValueError("GPQA restore receipt must remain inside destination-root")
        immutable_json(root, root / "gpqa_sft_adapters_manifest.json", adapters_manifest)
        immutable_json(root, receipt_path, receipt)
        summary.update(receipt=str(receipt_path), receipt_hash=receipt["receipt_hash"])
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--destination-root", required=True)
    parser.add_argument("--receipt")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--validate-only", action="store_true", help="Verify restored local bytes without network or writes")
    args = parser.parse_args(argv)
    inventory = read_json(args.inventory)
    factory = None
    if not args.validate_only:
        token = os.environ.get("GOOGLE_OAUTH_TOKEN_FILE")
        folder = os.environ.get("GOOGLE_DRIVE_FOLDER_ID")
        if not token or folder != PROJECT_FOLDER_ID or folder != inventory.get("project_folder_id"):
            raise ValueError("GPQA restore requires private GOOGLE_OAUTH_TOKEN_FILE and the canonical GOOGLE_DRIVE_FOLDER_ID")
        state_root = safe_path(args.destination_root, "state/drive_restore")
        # Per-thread HTTP clients; one persistent shared retry ledger.
        factory = lambda: DriveStore(drive_service(token), folder, state_root=state_root)
    result = restore_inventory(inventory, args.destination_root, store_factory=factory,
                               workers=args.workers, validate_only=args.validate_only, receipt_path=args.receipt)
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        # Transport messages can contain credential request arguments. Keep the
        # production failure log deliberately bounded to the exception class.
        print(json.dumps({"status": "FAILED", "error_type": type(error).__name__}), file=sys.stderr, flush=True)
        raise SystemExit(1) from None
