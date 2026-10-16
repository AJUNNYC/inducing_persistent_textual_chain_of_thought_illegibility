#!/usr/bin/env python3
"""Run one preselected frozen GPQA state early, then resume its original campaign.

This operations-only coordinator preserves the original plan, protocols, raw
identities, scoring and Drive barriers. It records an independent execution-order
amendment; the unchanged scientific coordinator produces the final campaign
receipt after it finishes the remaining states.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from types import SimpleNamespace


IMPLEMENTATIONS = ("scripts/run_gpqa_sft_campaign.py", "scripts/run_supplemental_gpqa.py")
ORDER_VERSION = "gpqa_campaign_execution_order_v1"


def load_frozen(source_root, plan):
    """Authenticate both frozen entrypoints before importing scientific helpers."""
    source_root = Path(source_root).resolve()
    revision = plan.get("source_code_commit")
    if (not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision)
            or os.environ.get("SOURCE_CODE_COMMIT") != revision):
        raise ValueError("SOURCE_CODE_COMMIT must match the original frozen plan")
    hashes = {name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
              for name in IMPLEMENTATIONS}
    if hashes != plan.get("implementation_sha256"):
        raise ValueError("Frozen scientific entrypoints differ from the original plan")
    src = source_root / "src"
    # A fresh CLI process normally has no scientific imports. Refuse a mixed
    # source process instead of silently reusing modules from another checkout.
    for name, module in tuple(sys.modules.items()):
        if name.split(".")[0] in {"evaluation", "experiment_overview", "shared_procedures", "operations"}:
            path = getattr(module, "__file__", None)
            if path and not Path(path).resolve().is_relative_to(src):
                raise ValueError("Scientific modules were already imported from a different source")
    sys.path.insert(0, str(src))
    spec = importlib.util.spec_from_file_location("gpqa_original_frozen_campaign", source_root / IMPLEMENTATIONS[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def original_arguments(plan):
    return SimpleNamespace(panel=plan["panel_path"], adapters_manifest=plan["adapters_manifest_path"],
                           model_cache=plan["model_cache"], output_root=plan["output_root"],
                           generation_profile=plan["generation_profile"],
                           save_to_drive=plan["save_to_drive"])


def execution_order(frozen, plan, after_state, priority_state):
    """Seal a permutation without changing or resealing the scientific plan."""
    states = [job["state_id"] for job in plan["order"]]
    if len(states) != len(set(states)) or not states or states[0] != "base_14b":
        raise ValueError("Original campaign state order is invalid")
    if after_state not in states or priority_state not in states:
        raise ValueError("Priority and preceding states must belong to the original plan")
    after, priority = states.index(after_state), states.index(priority_state)
    if priority <= after or priority == 0:
        raise ValueError("Priority state must originally follow its required preceding state")
    order = states[:after + 1] + [priority_state] + [s for s in states[after + 1:] if s != priority_state]
    record = {"version": ORDER_VERSION, "plan_hash": plan["plan_hash"],
              "source_code_commit": plan["source_code_commit"],
              "after_state_id": after_state, "priority_state_id": priority_state,
              "state_ids": order, "scientific_protocols_changed": False, "training_changed": False}
    return frozen.seal(record, "order_hash")


def original_command(frozen, plan, source_root):
    command = [sys.executable, str(Path(source_root) / IMPLEMENTATIONS[0]),
               "--panel", plan["panel_path"], "--adapters-manifest", plan["adapters_manifest_path"],
               "--model-cache", plan["model_cache"], "--output-root", plan["output_root"],
               "--generation-profile", plan["generation_profile"]]
    return command + (["--save-to-drive"] if plan["save_to_drive"] else [])


def local_immutable(frozen, path, record):
    """Publish operations evidence locally, without claiming Drive verification."""
    with frozen.file_lock(str(path) + ".lock"):
        if path.exists() and frozen.read_json(path) != record:
            raise ValueError("Execution-order evidence already exists with different contents")
        frozen.atomic_write_json(path, record)


def run_priority(args, frozen, plan, protocols):
    output = Path(args.output_root).resolve()
    campaign = output / "campaigns" / plan["plan_hash"]
    campaign.mkdir(parents=True, exist_ok=True)
    science_args = original_arguments(plan)
    order = execution_order(frozen, plan, args.after_state, args.priority_state)
    by_state = {job["state_id"]: (job, protocol) for job, protocol in zip(plan["order"], protocols)}
    prefix = order["state_ids"][:order["state_ids"].index(args.priority_state)]
    completed, child, stopping = [], None, {"signal": None}
    handlers = {}

    def status(value, **fields):
        record = {"version": "gpqa_campaign_local_status_v1", "status": value,
                  "updated_at": datetime.now(timezone.utc).isoformat(), "plan_hash": plan["plan_hash"],
                  "execution_order_hash": order["order_hash"], "execution_order": order["state_ids"],
                  "completed_state_ids": list(completed), "state_count": len(plan["order"]),
                  "status_storage": "LOCAL_ONLY", **fields}
        frozen.atomic_write_json(campaign / "status.json", record)
        print(json.dumps(record), flush=True)

    def stop(signum, frame):
        stopping["signal"] = signum
        if child is not None and child.poll() is None:
            child.send_signal(signum)

    def invoke(command, log_path):
        nonlocal child
        if stopping["signal"]:
            return 128 + stopping["signal"]
        with log_path.open("a") as log:
            child = subprocess.Popen(command, cwd=args.source_root, stdout=log, stderr=subprocess.STDOUT)
            if stopping["signal"] and child.poll() is None:
                child.send_signal(stopping["signal"])
            code = child.wait()
        child = None
        return code if code >= 0 else 128 - code

    def barrier(state_id):
        job, protocol = by_state[state_id]
        path = campaign / "protocols" / (state_id + ".json")
        if frozen.read_json(path) != protocol:
            raise ValueError("Existing protocol differs from the original frozen plan")
        return frozen.completion_barrier(protocol, output, save_to_drive=plan["save_to_drive"],
                                         source_code_commit=plan["source_code_commit"])

    for number in (signal.SIGINT, signal.SIGTERM):
        handlers[number] = signal.signal(number, stop)
    try:
        with frozen.file_lock(output / "campaigns" / "writer.lock"):
            # Do not publish or replace either original immutable plan copy.
            for path in (Path(args.plan), output / "campaigns" / "plan.json", campaign / "plan.json"):
                if frozen.read_json(path) != plan:
                    raise ValueError("Original immutable campaign plan copy differs")
            if args.execution_order and frozen.read_json(args.execution_order) != order:
                raise ValueError("Requested execution order is not the exact sealed permutation")
            order_path = campaign / ("execution_order_" + order["order_hash"] + ".json")
            original_states = [job["state_id"] for job in plan["order"]]
            moved = original_states[original_states.index(args.after_state) + 1:
                                    original_states.index(args.priority_state)]
            if (not order_path.exists() and any(
                    (output / by_state[state][0]["protocol_hash"] / "completion_receipt.json").exists()
                    for state in moved)):
                raise ValueError("A state moved after the priority state already completed before this amendment")
            local_immutable(frozen, order_path, order)
            for state_id in prefix:
                if stopping["signal"]:
                    status("STOPPED", child_exit_code=128 + stopping["signal"])
                    return 128 + stopping["signal"]
                barrier(state_id)  # Read-only reuse requires the full frozen payload barrier.
                completed.append(state_id)
            job, expected = by_state[args.priority_state]
            protocol_path = campaign / "protocols" / (args.priority_state + ".json")
            receipt_path = output / job["protocol_hash"] / "completion_receipt.json"
            log_path = campaign / (args.priority_state + ".log")
            if not receipt_path.exists():
                status("FREEZING", current_state_id=args.priority_state)
                code = invoke(frozen.freeze_command(science_args, job, protocol_path,
                                                    campaign / "protocols/base_14b.json"), log_path)
                if code:
                    status("STOPPED" if stopping["signal"] else "FAILED", current_state_id=args.priority_state,
                           child_exit_code=code, log_path=str(log_path))
                    return code
                if frozen.read_json(protocol_path) != expected:
                    raise ValueError("CLI-frozen priority protocol differs from the original plan")
                status("RUNNING", current_state_id=args.priority_state, log_path=str(log_path))
                code = invoke(frozen.run_command(science_args, protocol_path), log_path)
                if code:
                    status("STOPPED" if stopping["signal"] else "FAILED", current_state_id=args.priority_state,
                           child_exit_code=code, log_path=str(log_path))
                    return code
            receipt = barrier(args.priority_state)
            completed.append(args.priority_state)
            command = original_command(frozen, plan, args.source_root)
            record = frozen.seal({"version": "gpqa_priority_execution_receipt_v1", "status": "COMPLETE",
                "plan_hash": plan["plan_hash"], "execution_order_hash": order["order_hash"],
                "after_state_id": args.after_state, "priority_state_id": args.priority_state,
                "priority_protocol_hash": job["protocol_hash"], "child_receipt_hash": receipt["receipt_hash"],
                "child_receipt_sha256": frozen.file_hash(receipt_path),
                "completed_payload_drive_status": receipt["drive_status"],
                "resume_command": command, "receipt_storage": "LOCAL_ONLY"}, "receipt_hash")
            local_immutable(frozen, campaign / ("priority_execution_" + order["order_hash"] + ".json"), record)
            status("PRIORITY_COMPLETE", current_state_id=args.priority_state,
                   completed_payload_drive_status=receipt["drive_status"])
            if stopping["signal"]:
                status("STOPPED", child_exit_code=128 + stopping["signal"])
                return 128 + stopping["signal"]
        # Release the original writer lock before replacing this coordinator.
        # The original CLI independently rechecks every cached payload and owns
        # its original final receipt; it will skip completed level 10 inference.
        os.chdir(args.source_root)
        os.environ["PYTHONPATH"] = str(Path(args.source_root) / "src")
    except Exception as error:
        status("FAILED", error=str(error))
        raise
    finally:
        for number, handler in handlers.items():
            signal.signal(number, handler)
    os.execv(command[0], command)
    raise RuntimeError("Original campaign exec unexpectedly returned")


def main(argv=None):
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "plan", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--priority-state", required=True)
    parser.add_argument("--after-state", required=True)
    parser.add_argument("--execution-order", type=Path)
    args = parser.parse_args(argv)
    for name in ("source_root", "plan", "output_root"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    plan = json.loads(args.plan.read_text())
    frozen = load_frozen(args.source_root, plan)
    frozen.sealed(plan, "plan_hash")
    if Path(plan["output_root"]).resolve() != args.output_root:
        raise ValueError("Output root differs from the original frozen campaign")
    reconstructed, protocols = frozen.campaign_plan(original_arguments(plan))
    if reconstructed != plan:
        raise ValueError("Current input bindings differ from the original immutable plan")
    return run_priority(args, frozen, plan, protocols)


if __name__ == "__main__":
    raise SystemExit(main())
