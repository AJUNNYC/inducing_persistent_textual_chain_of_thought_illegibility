"""Boundary handoff safety checks without GPU work or external services."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location("gpqa_priority_handoff",
    Path(__file__).resolve().parents[3] / "scripts/schedule_gpqa_priority_handoff.py")
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


def fake_process(root, pid, *, script="/frozen/scripts/run_gpqa_sft_campaign.py",
                 output="/results", starttime=100, group=None, state="S"):
    directory = root / str(pid)
    directory.mkdir(parents=True)
    fields = [state, "1", str(group or pid)] + ["0"] * 16 + [str(starttime)]
    (directory / "stat").write_text(f"{pid} (name with spaces) " + " ".join(fields))
    arguments = ["python", script, "--output-root", output]
    (directory / "cmdline").write_bytes(b"\0".join(a.encode() for a in arguments) + b"\0")
    return directory


def test_process_identity_requires_exact_script_output_and_original_starttime(tmp_path):
    directory = fake_process(tmp_path, 123)
    identity = GATE.script_identity(123, "/frozen/scripts/run_gpqa_sft_campaign.py",
                                   "/results", proc_root=tmp_path)
    assert identity["starttime"] == 100 and identity["pgrp"] == 123
    assert GATE.same_process(identity, tmp_path)
    with pytest.raises(RuntimeError, match="script identity"):
        GATE.script_identity(123, "/elsewhere/run_gpqa_sft_campaign.py", "/results", proc_root=tmp_path)
    with pytest.raises(RuntimeError, match="configuration identity"):
        GATE.script_identity(123, "/frozen/scripts/run_gpqa_sft_campaign.py", "/results_other", proc_root=tmp_path)
    fields = (directory / "stat").read_text().split()
    fields[-1] = "101"
    (directory / "stat").write_text(" ".join(fields))
    assert not GATE.same_process(identity, tmp_path)


def test_owned_group_waits_for_live_engine_but_ignores_zombies_and_other_groups(tmp_path):
    fake_process(tmp_path, 123)
    fake_process(tmp_path, 124, script="engine", group=123)
    fake_process(tmp_path, 125, script="engine", group=123, state="Z")
    fake_process(tmp_path, 126, script="unrelated", group=126)
    identity = GATE.process(123, tmp_path)
    assert GATE.owned_group_workers(identity, tmp_path) == [124]


def test_status_without_matching_direct_run_child_cannot_pause_coordinator(tmp_path):
    fake_process(tmp_path, 123)
    coordinator = GATE.process(123, tmp_path)
    source, output, protocol = Path("/frozen"), Path("/results"), Path("/results/protocols/level6.json")
    assert GATE.active_supplemental_child(coordinator, source, output, protocol, tmp_path) is None
    worker = fake_process(tmp_path, 124, group=123, script="/frozen/scripts/run_supplemental_gpqa.py")
    fields = (worker / "stat").read_text().rsplit(")", 1)[1].split()
    fields[1] = "123"
    (worker / "stat").write_text("124 (supplemental) " + " ".join(fields))
    arguments = ["python", "/frozen/scripts/run_supplemental_gpqa.py", "run", "--protocol", str(protocol),
                 "--output-root", str(output)]
    (worker / "cmdline").write_bytes(b"\0".join(a.encode() for a in arguments) + b"\0")
    assert GATE.active_supplemental_child(coordinator, source, output, protocol, tmp_path)["pid"] == 124
    arguments[2] = "freeze"
    (worker / "cmdline").write_bytes(b"\0".join(a.encode() for a in arguments) + b"\0")
    assert GATE.active_supplemental_child(coordinator, source, output, protocol, tmp_path) is None


def test_recovery_counts_and_exact_frozen_paths_survive_command_handoff(tmp_path):
    args = SimpleNamespace(python=Path("/venv/bin/python"), supervisor_script=Path("/ops/wrapper.py"),
        source_root=Path("/frozen"), output_root=Path("/results"), monitor_root=tmp_path,
        after_state="rot13_14b_level_06", priority_state="rot13_14b_level_10",
        priority_runner=Path("/ops/priority.py"))
    plan = {"panel_path": "/inputs/panel.json", "adapters_manifest_path": "/inputs/adapters.json",
            "model_cache": "/models"}
    command = GATE.supervisor_command(args, plan, {"recoveries": 2, "max_recoveries": 12})
    assert GATE.option(command, "--initial-recoveries") == "2"
    assert GATE.option(command, "--max-recoveries") == "12"
    assert GATE.option(command, "--panel") == "/inputs/panel.json"
    assert GATE.option(command, "--priority-state") == "rot13_14b_level_10"
    assert command[:2] == ["/venv/bin/python", "/ops/wrapper.py"]
    for counters in ({"recoveries": -1, "max_recoveries": 12},
                     {"recoveries": 13, "max_recoveries": 12},
                     {"recoveries": True, "max_recoveries": 12}):
        with pytest.raises(RuntimeError, match="counters"):
            GATE.supervisor_command(args, plan, counters)


def test_writer_lock_must_release_before_starting_new_supervisor(tmp_path):
    path = tmp_path / "writer.lock"
    assert GATE.lock_is_free(path)
    with GATE.exclusive_lock(path):
        assert not GATE.lock_is_free(path)
    assert GATE.lock_is_free(path)


def test_pausing_only_coordinator_lets_existing_child_finish(tmp_path):
    script = tmp_path / "run_gpqa_sft_campaign.py"
    script.write_text('''
import pathlib, subprocess, sys, time
root = pathlib.Path(sys.argv[sys.argv.index("--output-root") + 1])
child = subprocess.Popen([sys.executable, "-c", "import pathlib,sys,time; time.sleep(.3); pathlib.Path(sys.argv[1]).write_text('done')", str(root / "child_complete")])
(root / "child_ready").write_text(str(child.pid))
child.wait()
time.sleep(60)
''')
    child = subprocess.Popen([sys.executable, str(script), "--output-root", str(tmp_path)],
                             start_new_session=True)
    identity = None
    try:
        deadline = time.monotonic() + 5
        while not (tmp_path / "child_ready").exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert (tmp_path / "child_ready").exists()
        identity = GATE.script_identity(child.pid, script, tmp_path)
        GATE.signal_identity(identity, signal.SIGSTOP)
        while not (tmp_path / "child_complete").exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert (tmp_path / "child_complete").read_text() == "done"
        assert GATE.process(child.pid)["state"] == "T"
        assert GATE.same_process(identity)
        GATE.signal_identity(identity, signal.SIGCONT)
        child.terminate()
        assert child.wait(timeout=5) == -signal.SIGTERM
    finally:
        if child.poll() is None:
            if identity is not None and GATE.same_process(identity):
                GATE.signal_identity(identity, signal.SIGCONT)
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=5)


def test_failed_source_validation_writes_private_failure_without_signaling(tmp_path, monkeypatch):
    args = SimpleNamespace(status_path=tmp_path / "status.json", after_state="level6", priority_state="level10")
    monkeypatch.setattr(GATE, "load_frozen", lambda args: (_ for _ in ()).throw(RuntimeError("frozen mismatch")))
    monkeypatch.setattr(GATE, "signal_identity", lambda *a: pytest.fail("No process may be signaled"))
    assert GATE.handoff(args) == 1
    assert json.loads(args.status_path.read_text())["reason"] == "frozen mismatch"
    assert args.status_path.stat().st_mode & 0o777 == 0o600


def test_running_status_before_popen_does_not_signal_coordinator(tmp_path, monkeypatch):
    output, monitor = tmp_path / "output", tmp_path / "monitor"
    campaign = output / "campaigns" / "plan"
    campaign.mkdir(parents=True)
    monitor.mkdir()
    (monitor / "status.json").write_text(json.dumps({"pid": 100, "campaign_pid": 101,
        "status": "RUNNING", "recoveries": 2, "max_recoveries": 12}))
    (campaign / "status.json").write_text(json.dumps({"status": "RUNNING",
        "current_state_id": "level6", "completed_state_ids": ["base"]}))
    args = SimpleNamespace(status_path=tmp_path / "gate.json", after_state="level6",
        priority_state="level10", output_root=output, monitor_root=monitor,
        source_root=tmp_path / "source", existing_supervisor_script=Path("/old/wrapper.py"),
        supervisor_script=Path("/new/wrapper.py"), supervisor_pid=100,
        python=Path(sys.executable), priority_runner=Path("/new/priority.py"), poll_seconds=.01)
    plan = {"plan_hash": "plan", "panel_path": "/panel", "adapters_manifest_path": "/adapters",
            "model_cache": "/models"}
    monkeypatch.setattr(GATE, "load_frozen", lambda args: (object(), plan, {"protocol_hash": "level6hash"}))
    monkeypatch.setattr(GATE, "script_identity", lambda pid, *a, **kw:
        {"pid": pid, "pgrp": pid, "starttime": 1, "arguments": []})
    monkeypatch.setattr(GATE, "same_process", lambda identity: True)
    monkeypatch.setattr(GATE, "active_supplemental_child", lambda *a, **kw: None)
    monkeypatch.setattr(GATE, "signal_identity", lambda *a: pytest.fail("Coordinator must remain runnable"))
    monkeypatch.setattr(GATE.time, "sleep", lambda seconds: (_ for _ in ()).throw(RuntimeError("fixture end")))
    assert GATE.handoff(args) == 1
    assert json.loads(args.status_path.read_text())["reason"] == "fixture end"


def test_verified_boundary_handoff_preserves_counters_and_waits_as_parent(tmp_path, monkeypatch):
    output, monitor = tmp_path / "output", tmp_path / "monitor"
    campaign = output / "campaigns" / "plan"
    campaign.mkdir(parents=True)
    monitor.mkdir()
    old_status = {"pid": 100, "campaign_pid": 101, "status": "RUNNING",
                  "recoveries": 2, "max_recoveries": 12}
    (monitor / "status.json").write_text(json.dumps(old_status))
    (campaign / "status.json").write_text(json.dumps({"status": "RUNNING",
        "current_state_id": "level6", "completed_state_ids": ["base"]}))
    receipt = output / "level6hash/completion_receipt.json"
    receipt.parent.mkdir()
    receipt.write_text("{}")
    args = SimpleNamespace(status_path=tmp_path / "gate.json", after_state="level6",
        priority_state="level10", output_root=output, monitor_root=monitor,
        source_root=tmp_path / "source", existing_supervisor_script=Path("/old/wrapper.py"),
        supervisor_script=Path("/new/wrapper.py"), supervisor_pid=100,
        python=Path(sys.executable), priority_runner=Path("/new/priority.py"),
        poll_seconds=.01, shutdown_timeout=1)
    plan = {"plan_hash": "plan", "panel_path": "/panel", "adapters_manifest_path": "/adapters",
            "model_cache": "/models", "source_code_commit": "a" * 40}
    events, alive, sleeps = [], {100: True, 101: True}, []

    def barrier(*a, **kw):
        assert kw == {"save_to_drive": True, "source_code_commit": "a" * 40}
        events.append("verified_barrier")
        return {"receipt_hash": "verified"}

    def signal_owned(identity, signum):
        events.append((identity["pid"], signum))
        if identity["pid"] == 100 and signum == signal.SIGTERM:
            assert "verified_barrier" in events
            alive.update({100: False, 101: False})
            (monitor / "status.json").write_text(json.dumps({**old_status, "status": "STOPPED"}))

    class NewSupervisor:
        pid = 200
        polls = 0
        command = None

        def __init__(self, command, **kw):
            assert not any(alive.values())
            assert kw["start_new_session"] is True
            type(self).command = command

        def poll(self):
            type(self).polls += 1
            return 0 if self.polls >= 3 else None

        def wait(self):
            assert self.polls >= 3
            return 0

    monkeypatch.setattr(GATE, "load_frozen", lambda args:
        (SimpleNamespace(completion_barrier=barrier), plan, {"protocol_hash": "level6hash"}))
    monkeypatch.setattr(GATE, "script_identity", lambda pid, *a, **kw:
        {"pid": pid, "pgrp": pid, "starttime": 1, "arguments": []})
    monkeypatch.setattr(GATE, "same_process", lambda identity: alive[identity["pid"]])
    monkeypatch.setattr(GATE, "owned_group_workers", lambda coordinator: [])
    monkeypatch.setattr(GATE, "signal_identity", signal_owned)
    monkeypatch.setattr(GATE.subprocess, "Popen", NewSupervisor)
    monkeypatch.setattr(GATE.time, "sleep", lambda seconds: sleeps.append(seconds))
    assert GATE.handoff(args) == 0
    assert events == [(101, signal.SIGSTOP), "verified_barrier", (100, signal.SIGTERM)]
    assert GATE.option(NewSupervisor.command, "--initial-recoveries") == "2"
    assert NewSupervisor.polls == 3 and len(sleeps) == 2
    recorded = [json.loads(line)["status"] for line in args.status_path.with_suffix(".events.jsonl").read_text().splitlines()]
    assert recorded[-2:] == ["NEW_SUPERVISOR_STARTED", "NEW_SUPERVISOR_COMPLETE"]
