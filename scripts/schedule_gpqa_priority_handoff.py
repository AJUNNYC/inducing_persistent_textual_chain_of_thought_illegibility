#!/usr/bin/env python3
"""Finish an active frozen GPQA state before switching operational schedulers.

Only the coordinator is paused. Its existing supplemental worker and engines
finish normally; the frozen completion barrier and absence of owned workers
are required before the old recovery supervisor is asked to shut down.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


def private_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".handoff-")
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


def read_json(path):
    return json.loads(Path(path).read_text())


def process(pid, proc_root=Path("/proc")):
    """Return the Linux identity without logging arguments or credentials."""
    try:
        root = proc_root / str(pid)
        fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
        arguments = (root / "cmdline").read_bytes().decode().split("\0")
        return {"pid": int(pid), "state": fields[0], "ppid": int(fields[1]),
                "pgrp": int(fields[2]), "starttime": int(fields[19]),
                "arguments": [a for a in arguments if a]}
    except (OSError, UnicodeError, ValueError, IndexError):
        return None


def option(arguments, name):
    for index, value in enumerate(arguments):
        if value == name:
            return arguments[index + 1] if index + 1 < len(arguments) else None
        if value.startswith(name + "="):
            return value[len(name) + 1:]
    return None


def same_process(identity, proc_root=Path("/proc")):
    current = process(identity["pid"], proc_root)
    return (current is not None and current["state"] != "Z"
            and current["starttime"] == identity["starttime"]
            and current["arguments"] == identity["arguments"]
            and current["pgrp"] == identity["pgrp"])


def script_identity(pid, script, output, *, source=None, monitor=None, proc_root=Path("/proc")):
    current = process(pid, proc_root)
    if not current or current["state"] == "Z":
        raise RuntimeError("The declared process is no longer active")
    arguments = current["arguments"]
    # Require the exact executed script, rather than a basename appearing in an
    # unrelated argument. The expected Python invocation has no interpreter flags.
    if len(arguments) < 2 or Path(arguments[1]).resolve() != Path(script).resolve():
        raise RuntimeError("Declared process script identity differs")
    for name, expected in (("--output-root", output), ("--source-root", source),
                           ("--monitor-root", monitor)):
        if expected is not None:
            actual = option(arguments, name)
            if actual is None or Path(actual).resolve() != Path(expected).resolve():
                raise RuntimeError("Declared process configuration identity differs")
    return current


def signal_identity(identity, signum):
    if not same_process(identity):
        raise RuntimeError("Process identity changed before signaling")
    os.kill(identity["pid"], signum)


def owned_group_workers(coordinator, proc_root=Path("/proc")):
    found = []
    for entry in proc_root.iterdir():
        if entry.name.isdigit() and int(entry.name) != coordinator["pid"]:
            current = process(int(entry.name), proc_root)
            if current and current["state"] != "Z" and current["pgrp"] == coordinator["pgrp"]:
                found.append(current["pid"])
    return found


def active_supplemental_child(coordinator, source, output, protocol_path, proc_root=Path("/proc")):
    """RUNNING status alone precedes Popen and cannot establish a live worker."""
    for pid in owned_group_workers(coordinator, proc_root):
        current = process(pid, proc_root)
        if not current or current["ppid"] != coordinator["pid"]:
            continue
        arguments = current["arguments"]
        if (len(arguments) > 2 and arguments[2] == "run"
                and Path(arguments[1]).resolve() == (source / "scripts/run_supplemental_gpqa.py").resolve()
                and option(arguments, "--output-root") is not None
                and Path(option(arguments, "--output-root")).resolve() == output.resolve()
                and option(arguments, "--protocol") is not None
                and Path(option(arguments, "--protocol")).resolve() == protocol_path.resolve()):
            return current
    return None


@contextmanager
def exclusive_lock(path):
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another handoff or campaign writer is active") from None
        yield
    finally:
        os.close(fd)


def lock_is_free(path):
    try:
        with exclusive_lock(path):
            return True
    except RuntimeError:
        return False


def load_frozen(args):
    plan = read_json(args.plan)
    if (plan.get("source_code_commit") != os.environ.get("SOURCE_CODE_COMMIT")
            or Path(plan.get("output_root", "")).resolve() != args.output_root
            or plan.get("save_to_drive") is not True):
        raise RuntimeError("Frozen plan source/output/Drive identity differs")
    for name, digest in plan.get("implementation_sha256", {}).items():
        if name not in {"scripts/run_gpqa_sft_campaign.py", "scripts/run_supplemental_gpqa.py"}:
            raise RuntimeError("Unexpected frozen implementation identity")
        if hashlib.sha256((args.source_root / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Frozen scientific source changed")
    spec = importlib.util.spec_from_file_location("_handoff_frozen_campaign",
        args.source_root / "scripts/run_gpqa_sft_campaign.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    inputs = argparse.Namespace(panel=plan["panel_path"],
        adapters_manifest=plan["adapters_manifest_path"], model_cache=plan["model_cache"],
        output_root=str(args.output_root), generation_profile="long_32000", save_to_drive=True)
    rebuilt, protocols = module.campaign_plan(inputs)
    if rebuilt != plan:
        raise RuntimeError("Existing plan differs from the frozen input validation")
    by_state = {job["state_id"]: protocol for job, protocol in zip(plan["order"], protocols)}
    if args.after_state not in by_state or args.priority_state not in by_state or args.after_state == args.priority_state:
        raise RuntimeError("Handoff states must be distinct existing frozen conditions")
    return module, plan, by_state[args.after_state]


def supervisor_command(args, plan, old_status):
    recoveries, maximum = old_status.get("recoveries"), old_status.get("max_recoveries")
    if (type(recoveries) is not int or type(maximum) is not int
            or not 0 <= recoveries <= maximum):
        raise RuntimeError("Existing recovery counters are invalid")
    return [str(args.python), str(args.supervisor_script),
        "--source-root", str(args.source_root), "--python", str(args.python),
        "--panel", plan["panel_path"], "--adapters-manifest", plan["adapters_manifest_path"],
        "--model-cache", plan["model_cache"], "--output-root", str(args.output_root),
        "--monitor-root", str(args.monitor_root), "--priority-state", args.priority_state,
        "--after-state", args.after_state, "--priority-runner", str(args.priority_runner),
        "--initial-recoveries", str(recoveries), "--max-recoveries", str(maximum)]


def handoff(args):
    args.status_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    stopped = {"signal": None}
    paused = None
    old_supervisor = None
    shutdown_started = False

    def status(value, **fields):
        record = {"version": "gpqa_priority_handoff_v1", "status": value,
            "updated_at": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
            "after_state": args.after_state, "priority_state": args.priority_state,
            "status_storage": "LOCAL_ONLY", **fields}
        private_json(args.status_path, record)
        with args.status_path.with_suffix(".events.jsonl").open("a") as events:
            events.write(json.dumps(record, sort_keys=True) + "\n")

    handlers = {number: signal.signal(number, lambda n, frame: stopped.update(signal=n))
                for number in (signal.SIGINT, signal.SIGTERM)}
    try:
        with exclusive_lock(args.status_path.with_suffix(".lock")):
            frozen, plan, protocol = load_frozen(args)
            campaign_path = args.output_root / "campaigns" / plan["plan_hash"] / "status.json"
            protocol_path = campaign_path.parent / "protocols" / (args.after_state + ".json")
            receipt_path = args.output_root / protocol["protocol_hash"] / "completion_receipt.json"
            old_status = read_json(args.monitor_root / "status.json")
            if args.supervisor_pid is not None and old_status.get("pid") != args.supervisor_pid:
                raise RuntimeError("Original supervisor differs from the explicitly selected PID")
            old_supervisor = script_identity(old_status["pid"], args.existing_supervisor_script, args.output_root,
                source=args.source_root, monitor=args.monitor_root)
            # Validate counters before pausing anything; they cannot be reset.
            supervisor_command(args, plan, old_status)
            status("WAITING_FOR_ACTIVE_STATE")
            while not stopped["signal"]:
                if not same_process(old_supervisor):
                    raise RuntimeError("Original recovery supervisor disappeared")
                old_status = read_json(args.monitor_root / "status.json")
                if old_status.get("pid") != old_supervisor["pid"]:
                    raise RuntimeError("Original recovery supervisor identity changed")
                if old_status.get("status") in {"REPAIRING", "REPAIRED"}:
                    time.sleep(args.poll_seconds)
                    continue
                if old_status.get("status") != "RUNNING":
                    raise RuntimeError("Original supervisor stopped before the handoff")
                if paused is not None and not same_process(paused):
                    paused = None  # A native recovery replaced its failed group.
                    status("WAITING_AFTER_NATIVE_RECOVERY", recoveries=old_status["recoveries"])
                campaign = read_json(campaign_path)
                if paused is None:
                    if campaign.get("current_state_id") != args.after_state or campaign.get("status") != "RUNNING":
                        # Completed earlier states are replayed during native recovery.
                        completed = campaign.get("completed_state_ids", [])
                        if args.after_state in completed:
                            raise RuntimeError("Target handoff boundary was already passed")
                        time.sleep(args.poll_seconds)
                        continue
                    paused = script_identity(old_status["campaign_pid"],
                        args.source_root / "scripts/run_gpqa_sft_campaign.py", args.output_root)
                    if paused["pgrp"] != paused["pid"]:
                        raise RuntimeError("Coordinator does not own the expected process group")
                    if not receipt_path.exists() and not active_supplemental_child(
                            paused, args.source_root, args.output_root, protocol_path):
                        # The coordinator writes RUNNING before invoking Popen.
                        # It must stay runnable until the actual child exists.
                        paused = None
                        time.sleep(args.poll_seconds)
                        continue
                    signal_identity(paused, signal.SIGSTOP)
                    # Detect the narrow status-read/signal race before proceeding.
                    campaign = read_json(campaign_path)
                    if campaign.get("current_state_id") != args.after_state:
                        raise RuntimeError("Coordinator advanced before it could be paused")
                    if not receipt_path.exists() and not active_supplemental_child(
                            paused, args.source_root, args.output_root, protocol_path):
                        signal_identity(paused, signal.SIGCONT)
                        paused = None
                        time.sleep(args.poll_seconds)
                        continue
                    status("COORDINATOR_PAUSED_WORKER_CONTINUES", coordinator_pid=paused["pid"],
                           coordinator_starttime=paused["starttime"], recoveries=old_status["recoveries"])
                if receipt_path.exists():
                    # Invalid receipts fail closed; no handoff on a counter alone.
                    receipt = frozen.completion_barrier(protocol, args.output_root,
                        save_to_drive=True, source_code_commit=plan["source_code_commit"])
                    if not owned_group_workers(paused):
                        if not same_process(paused):
                            raise RuntimeError("Paused coordinator changed at the completion boundary")
                        shutdown_started = True
                        status("VERIFIED_BOUNDARY_STOPPING_OLD_SUPERVISOR",
                               completion_receipt_hash=receipt["receipt_hash"])
                        signal_identity(old_supervisor, signal.SIGTERM)
                        deadline = time.monotonic() + args.shutdown_timeout
                        while time.monotonic() < deadline:
                            alive = same_process(old_supervisor) or same_process(paused)
                            children = owned_group_workers(paused)
                            locks_free = (lock_is_free(args.output_root / "campaigns/writer.lock")
                                and lock_is_free(args.output_root / "campaigns/recovery_supervisor.lock"))
                            if not alive and not children and locks_free:
                                break
                            time.sleep(min(args.poll_seconds, .2))
                        else:
                            raise RuntimeError("Old owned processes or campaign locks did not release")
                        if stopped["signal"]:
                            status("STOPPED_AFTER_VERIFIED_BOUNDARY", signal=stopped["signal"])
                            return 128 + stopped["signal"]
                        final_old = read_json(args.monitor_root / "status.json")
                        if final_old.get("pid") != old_supervisor["pid"] or final_old.get("status") != "STOPPED":
                            raise RuntimeError("Old supervisor did not record its controlled shutdown")
                        command = supervisor_command(args, plan, final_old)
                        environment = os.environ.copy()
                        environment["PYTHONPATH"] = str(args.source_root / "src")
                        with args.status_path.with_suffix(".launch.log").open("a") as log:
                            child = subprocess.Popen(command, cwd=args.source_root, env=environment,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                        status("NEW_SUPERVISOR_STARTED", supervisor_pid=child.pid,
                               initial_recoveries=final_old["recoveries"],
                               max_recoveries=final_old["max_recoveries"])
                        # Remain its parent: the Jupyter ancestor uses a read-only
                        # /proc counter under Linux's normal Yama restrictions.
                        # Reparenting the supervisor to PID 1 would break that
                        # observation route without affecting actual inference.
                        forwarded = False
                        while child.poll() is None:
                            if stopped["signal"] and not forwarded:
                                identity = script_identity(child.pid, args.supervisor_script,
                                    args.output_root, source=args.source_root, monitor=args.monitor_root)
                                signal_identity(identity, signal.SIGTERM)
                                forwarded = True
                                status("STOPPING_NEW_SUPERVISOR", supervisor_pid=child.pid,
                                       signal=stopped["signal"])
                            time.sleep(args.poll_seconds)
                        code = child.wait()
                        status("NEW_SUPERVISOR_COMPLETE" if code == 0 else
                               "STOPPED" if stopped["signal"] else "FAILED",
                               supervisor_pid=child.pid, supervisor_exit_code=code)
                        return 128 + stopped["signal"] if stopped["signal"] else (0 if code == 0 else 1)
                time.sleep(args.poll_seconds)
            status("STOPPED", signal=stopped["signal"])
            return 128 + stopped["signal"]
    except Exception as error:
        status("FAILED", error_class=type(error).__name__, reason=str(error))
        return 1
    finally:
        # Aborting the gate must not strand the original coordinator. Never wake
        # it once controlled supervisor shutdown has begun, or it could start L7.
        if (not shutdown_started and paused is not None and old_supervisor is not None
                and same_process(old_supervisor) and same_process(paused)):
            signal_identity(paused, signal.SIGCONT)
        for number, handler in handlers.items():
            signal.signal(number, handler)


def main(argv=None):
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "output-root", "monitor-root", "plan", "supervisor-script",
                 "priority-runner", "status-path"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--existing-supervisor-script", type=Path,
                        help="Original supervisor script identity; defaults to supervisor-script")
    parser.add_argument("--after-state", required=True)
    parser.add_argument("--priority-state", required=True)
    parser.add_argument("--supervisor-pid", type=int, help="Require this existing recovery supervisor PID")
    parser.add_argument("--poll-seconds", type=float, default=1)
    parser.add_argument("--shutdown-timeout", type=float, default=45)
    args = parser.parse_args(argv)
    if args.existing_supervisor_script is None:
        args.existing_supervisor_script = args.supervisor_script
    if args.poll_seconds <= 0 or args.shutdown_timeout <= 0:
        parser.error("Polling and shutdown intervals must be positive")
    for name in ("source_root", "output_root", "monitor_root", "plan", "supervisor_script",
                 "existing_supervisor_script", "priority_runner", "status_path"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.python = args.python.expanduser().absolute()  # Preserve the venv path.
    return handoff(args)


if __name__ == "__main__":
    raise SystemExit(main())
