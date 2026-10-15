#!/usr/bin/env python3
"""Supervise an unchanged frozen GPQA campaign through transient Drive failures.

This operations wrapper never changes protocols, scores, generation, source
identity, or native retry counters. It resumes the exact campaign command after
incrementally synchronizing pending/failed artifacts with the frozen source.
Linux process groups and advisory locks prevent duplicate supervised workers.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time

FATAL = b"RuntimeError: Permanent artifact synchronization incomplete; registry retains pending/failed records"
TRANSIENT_ERRORS = {"TimeoutError", "ConnectionError", "HttpError"}
WORKER_NAMES = {"run_gpqa_sft_campaign.py", "run_supplemental_gpqa.py"}


def private_json(path, record):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".status-")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(record, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def exclusive_lock(path):
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another campaign writer or supervisor is active") from None
        yield
    finally:
        os.close(fd)


def registry_state(output):
    database = Path(output) / "drive_registry.sqlite3"
    if not database.exists():
        return {"pending": 0, "failed": 0, "unknown": 0, "error_classes": []}
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True, timeout=30) as db:
        rows = db.execute("SELECT sync_status,record FROM artifacts WHERE sync_status!='VERIFIED'").fetchall()
    errors = {json.loads(record).get("sync_error", "UNKNOWN") for status, record in rows if status == "FAILED"}
    return {"pending": sum(s == "PENDING" for s, _ in rows),
            "failed": sum(s == "FAILED" for s, _ in rows),
            "unknown": sum(s not in {"PENDING", "FAILED"} for s, _ in rows),
            "error_classes": sorted(errors)}


def recoverable(state):
    # The completion hook's message alone also covers integrity failures.
    return (state["failed"] > 0 and not state["unknown"] and bool(state["error_classes"])
            and set(state["error_classes"]) <= TRANSIENT_ERRORS)


class FreshFailures:
    """Only inspect bytes appended after this attempt, including split lines."""
    def __init__(self, output):
        self.output = Path(output)
        self.cursors = {}
        for path in self.paths():
            stat = path.stat()
            self.cursors[path] = ((stat.st_dev, stat.st_ino), stat.st_size, b"")

    def paths(self):
        return self.output.glob("campaigns/*/*.log")

    def poll(self):
        for path in self.paths():
            try:
                with path.open("rb") as handle:
                    stat = os.fstat(handle.fileno())
                    identity = (stat.st_dev, stat.st_ino)
                    old_identity, offset, tail = self.cursors.get(path, (identity, 0, b""))
                    if old_identity != identity or stat.st_size < offset:
                        offset, tail = 0, b""
                    handle.seek(offset)
                    found = False
                    while chunk := handle.read(65536):
                        data = tail + chunk
                        found |= FATAL in data
                        tail = data[-len(FATAL):]
                    self.cursors[path] = (identity, handle.tell(), tail)
                    if found:
                        return True
            except FileNotFoundError:
                continue
        return False


def active_workers(output, proc_root=Path("/proc")):
    """Read only script/output identity; never expose process arguments."""
    matches = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            arguments = (entry / "cmdline").read_bytes().decode().split("\0")
            if not any(Path(arg).name in WORKER_NAMES for arg in arguments):
                continue
            for index, argument in enumerate(arguments):
                value = (arguments[index + 1] if argument == "--output-root"
                         else argument.removeprefix("--output-root=") if argument.startswith("--output-root=") else None)
                if value and Path(value).resolve() == Path(output).resolve():
                    matches.append(int(entry.name))
                    break
        except (OSError, UnicodeError, IndexError):
            continue
    return matches


def assert_idle(output):
    if active_workers(output):
        raise RuntimeError("An existing campaign worker uses this output root")
    campaigns = Path(output) / "campaigns"
    campaigns.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(campaigns / "writer.lock"):
        pass


def launch(command, *, source_root, environment, log):
    return subprocess.Popen(command, cwd=source_root, env=environment, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True)


def terminate_owned_group(child, grace_seconds):
    """Only terminate the dedicated group created by launch(), including engines."""
    if child.pid == os.getpgrp():
        raise RuntimeError("Refusing to signal the supervisor process group")
    try:
        if os.getpgid(child.pid) != child.pid:
            raise RuntimeError("Child does not own a dedicated process group")
    except ProcessLookupError:
        pass  # The leader may have exited while its engine remains in the group.
    for signum, deadline in ((signal.SIGTERM, time.monotonic() + grace_seconds),
                             (signal.SIGKILL, time.monotonic() + grace_seconds + 5)):
        try:
            os.killpg(child.pid, signum)
        except ProcessLookupError:
            break
        while time.monotonic() < deadline:
            child.poll()
            try:
                os.killpg(child.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(.1)
        else:
            continue
        break
    child.wait(timeout=5)


def campaign_command(args):
    return [str(args.python), str(args.source_root / "scripts/run_gpqa_sft_campaign.py"),
            "--panel", str(args.panel), "--adapters-manifest", str(args.adapters_manifest),
            "--model-cache", str(args.model_cache), "--output-root", str(args.output_root),
            "--generation-profile", "long_32000", "--save-to-drive"]


def source_signature(args, revision):
    paths = ("scripts/run_gpqa_sft_campaign.py", "scripts/run_supplemental_gpqa.py")
    hashes = {name: hashlib.sha256((args.source_root / name).read_bytes()).hexdigest() for name in paths}
    plan_path = args.output_root / "campaigns/plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if (plan.get("source_code_commit") != revision
                or plan.get("implementation_sha256") != hashes):
            raise RuntimeError("Frozen source differs from the existing campaign plan")
    return hashes


def repair(args):
    # This subprocess starts clean and imports only the frozen scientific tree.
    sys.path.insert(0, str(args.source_root / "src"))
    import operations.artifacts as artifacts
    import operations.drive as drive
    for module in (artifacts, drive):
        if not Path(module.__file__).resolve().is_relative_to(args.source_root / "src"):
            raise RuntimeError("Repair imported a different scientific source")
    token, folder = os.environ.get("GOOGLE_OAUTH_TOKEN_FILE"), os.environ.get("GOOGLE_DRIVE_FOLDER_ID")
    if not token or not folder:
        raise RuntimeError("Private Drive configuration is missing")
    service = artifacts.ArtifactService(args.output_root, args.output_root / "drive_registry.json",
        drive.DriveStore(drive.drive_service(token), folder), drive_recovery=True)
    result = service.sync_artifacts()  # Default incremental mode; native ledger retained.
    state = registry_state(args.output_root)
    if result.get("status") != "VERIFIED" or state["pending"] or state["failed"] or state["unknown"]:
        raise RuntimeError("Incremental Drive repair remains incomplete")
    return 0


def supervise(args):
    environment = os.environ.copy()
    revision = environment.get("SOURCE_CODE_COMMIT", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError("SOURCE_CODE_COMMIT must retain the frozen scientific revision")
    environment["PYTHONPATH"] = str(args.source_root / "src")
    command = campaign_command(args)
    original_source = source_signature(args, revision)
    args.monitor_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(args.monitor_root, 0o700)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "campaigns").mkdir(exist_ok=True)
    status_path = args.monitor_root / "status.json"
    stop = {"signal": None}
    child = None
    recoveries = 0

    def status(value, **fields):
        record = {"version": "gpqa_recovery_supervisor_v1", "status": value,
                  "updated_at": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
                  "source_code_commit": revision, "recoveries": recoveries,
                  "max_recoveries": args.max_recoveries, "status_storage": "LOCAL_ONLY", **fields}
        private_json(status_path, record)
        with (args.monitor_root / "events.jsonl").open("a") as events:
            events.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)

    previous = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, lambda number, frame: stop.update(signal=number))
    try:
        with exclusive_lock(args.output_root / "campaigns/recovery_supervisor.lock"):
            while True:
                if source_signature(args, revision) != original_source:
                    raise RuntimeError("Frozen campaign source changed during supervision")
                assert_idle(args.output_root)
                failures = FreshFailures(args.output_root)
                with (args.monitor_root / "campaign.log").open("a") as log:
                    child = launch(command, source_root=args.source_root, environment=environment, log=log)
                    status("RUNNING", campaign_pid=child.pid)
                    detected = False
                    while True:
                        detected |= failures.poll()
                        code = child.poll()
                        if stop["signal"] is not None:
                            terminate_owned_group(child, args.terminate_grace)
                            child = None
                            status("STOPPED", signal=stop["signal"])
                            return 128 + stop["signal"]
                        if detected or code is not None:
                            break
                        time.sleep(args.poll_seconds)
                    if code == 0:
                        child = None
                        status("COMPLETE")
                        return 0
                    state = registry_state(args.output_root)
                    if not detected or not recoverable(state):
                        terminate_owned_group(child, args.terminate_grace)
                        child = None
                        status("FAILED", reason="unrecognized_or_nontransient_failure", child_exit_code=code,
                               registry=state)
                        return 1
                    terminate_owned_group(child, args.terminate_grace)
                    child = None
                assert_idle(args.output_root)
                if recoveries >= args.max_recoveries:
                    status("FAILED", reason="recovery_limit", registry=state)
                    return 1
                recoveries += 1
                status("REPAIRING", registry=state)
                repair_command = [str(args.python), str(Path(__file__).resolve()),
                                  "--source-root", str(args.source_root), "--output-root", str(args.output_root),
                                  "--repair-only"]
                with (args.monitor_root / "repair.log").open("a") as log:
                    child = launch(repair_command, source_root=args.source_root, environment=environment, log=log)
                    deadline = time.monotonic() + args.repair_timeout
                    while child.poll() is None and not stop["signal"] and time.monotonic() < deadline:
                        time.sleep(min(args.poll_seconds, 1))
                    code = child.poll()
                    if code is None:
                        terminate_owned_group(child, args.terminate_grace)
                    child = None
                if stop["signal"]:
                    status("STOPPED", signal=stop["signal"])
                    return 128 + stop["signal"]
                if code != 0:
                    status("FAILED", reason="incremental_repair_failed_or_timed_out", child_exit_code=code)
                    return 1
                status("REPAIRED", registry=registry_state(args.output_root))
    except Exception as error:
        status("FAILED", reason="supervisor_error", error_class=type(error).__name__)
        return 1
    finally:
        if child is not None:
            terminate_owned_group(child, args.terminate_grace)
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def main(argv=None):
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    for name in ("panel", "adapters-manifest", "model-cache", "output-root", "monitor-root"):
        parser.add_argument("--" + name, type=Path, required=name == "output-root")
    parser.add_argument("--max-recoveries", type=int, default=12)
    parser.add_argument("--poll-seconds", type=float, default=15)
    parser.add_argument("--terminate-grace", type=float, default=10)
    parser.add_argument("--repair-timeout", type=float, default=600)
    parser.add_argument("--repair-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.source_root = args.source_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    if args.repair_only:
        try:
            return repair(args)
        except Exception as error:
            print(json.dumps({"status": "FAILED", "error_class": type(error).__name__}), flush=True)
            return 1
    if any(getattr(args, name) is None for name in ("panel", "adapters_manifest", "model_cache", "monitor_root")):
        parser.error("panel, adapters-manifest, model-cache and monitor-root are required")
    if args.max_recoveries < 0 or min(args.poll_seconds, args.terminate_grace, args.repair_timeout) <= 0:
        parser.error("Recovery count must be nonnegative and intervals must be positive")
    for name in ("panel", "adapters_manifest", "model_cache", "monitor_root"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    # Preserve the venv executable path: resolve() would discard its environment.
    args.python = args.python.expanduser().absolute()
    return supervise(args)


if __name__ == "__main__":
    raise SystemExit(main())
