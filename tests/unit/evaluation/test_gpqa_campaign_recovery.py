"""Operations supervisor checks without model loading or external services."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location("gpqa_recovery", Path(__file__).resolve().parents[3]
                                            / "scripts/run_gpqa_campaign_with_recovery.py")
WRAPPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WRAPPER)


def test_only_fresh_appended_failures_trigger_including_split_lines(tmp_path):
    log = tmp_path / "campaigns/plan/state.log"
    log.parent.mkdir(parents=True)
    log.write_bytes(WRAPPER.FATAL + b"\n")
    failures = WRAPPER.FreshFailures(tmp_path)
    assert not failures.poll()
    with log.open("ab") as handle:
        handle.write(b"ValueError: protocol mismatch\n" + WRAPPER.FATAL[:31])
    assert not failures.poll()
    with log.open("ab") as handle:
        handle.write(WRAPPER.FATAL[31:] + b"\n")
    assert failures.poll()
    assert not failures.poll()


def test_new_and_replaced_logs_are_read(tmp_path):
    failures = WRAPPER.FreshFailures(tmp_path)
    log = tmp_path / "campaigns/plan/state.log"
    log.parent.mkdir(parents=True)
    log.write_bytes(WRAPPER.FATAL)
    assert failures.poll()
    replacement = log.with_suffix(".replacement")
    replacement.write_bytes(WRAPPER.FATAL)
    replacement.replace(log)
    assert failures.poll()


def test_integrity_unknown_and_pending_only_failures_are_not_retryable():
    state = dict(pending=0, failed=1, unknown=0, error_classes=["TimeoutError"])
    assert WRAPPER.recoverable(state)
    assert WRAPPER.recoverable({**state, "error_classes": ["HttpError"]})
    for error in ("RuntimeError", "ValueError", "RefreshError", "UNKNOWN"):
        assert not WRAPPER.recoverable({**state, "error_classes": [error]})
    assert not WRAPPER.recoverable({**state, "failed": 0, "pending": 1})


def test_lock_refuses_concurrent_supervisors(tmp_path):
    with WRAPPER.exclusive_lock(tmp_path / "lock"):
        with pytest.raises(RuntimeError, match="active"):
            with WRAPPER.exclusive_lock(tmp_path / "lock"):
                pytest.fail("A second supervisor acquired the live lock")


def test_worker_scan_matches_only_exact_output_and_script(tmp_path):
    output = tmp_path / "output"
    proc = tmp_path / "proc"
    for pid, command in ((111111, ["python", "/frozen/scripts/run_supplemental_gpqa.py", "--output-root", str(output)]),
                         (111112, ["python", "/frozen/scripts/run_gpqa_sft_campaign.py", "--output-root", str(output) + "_other"]),
                         (111113, ["python", "unrelated.py", "--output-root", str(output)])):
        directory = proc / str(pid)
        directory.mkdir(parents=True)
        (directory / "cmdline").write_bytes(b"\0".join(s.encode() for s in command))
    assert WRAPPER.active_workers(output, proc) == [111111]


def test_owned_process_group_cleanup_preserves_supervisor(tmp_path):
    with (tmp_path / "child.log").open("w") as log:
        child = WRAPPER.launch([sys.executable, "-c", "import time; time.sleep(60)"],
            source_root=tmp_path, environment=os.environ.copy(), log=log)
        assert os.getpgid(child.pid) == child.pid
        assert child.pid != os.getpgrp()
        WRAPPER.terminate_owned_group(child, .2)
        assert child.returncode == -signal.SIGTERM
        assert os.getpid() > 0


def fixture_args(tmp_path):
    source = tmp_path / "frozen"
    (source / "scripts").mkdir(parents=True)
    (source / "scripts/run_gpqa_sft_campaign.py").write_text("raise SystemExit(23)\n")
    (source / "scripts/run_supplemental_gpqa.py").write_text("# unchanged fixture\n")
    return SimpleNamespace(source_root=source, python=Path(sys.executable), panel=tmp_path / "panel.json",
        adapters_manifest=tmp_path / "adapters.json", model_cache=tmp_path / "models",
        output_root=tmp_path / "output", monitor_root=tmp_path / "monitor", max_recoveries=12,
        poll_seconds=.01, terminate_grace=.1, repair_timeout=1)


def test_unknown_child_failure_stops_without_repair_and_writes_private_status(tmp_path, monkeypatch):
    args = fixture_args(tmp_path)
    monkeypatch.setenv("SOURCE_CODE_COMMIT", "a" * 40)
    previous = os.umask(0o077)
    try:
        assert WRAPPER.supervise(args) == 1
    finally:
        os.umask(previous)
    status = json.loads((args.monitor_root / "status.json").read_text())
    assert status["reason"] == "unrecognized_or_nontransient_failure"
    assert status["child_exit_code"] == 23
    assert status["recoveries"] == 0
    assert (args.monitor_root / "status.json").stat().st_mode & 0o777 == 0o600
    assert not (args.monitor_root / "repair.log").exists()


def test_source_signature_rejects_mismatched_frozen_plan(tmp_path):
    args = fixture_args(tmp_path)
    original = WRAPPER.source_signature(args, "a" * 40)
    campaigns = args.output_root / "campaigns"
    campaigns.mkdir(parents=True)
    (campaigns / "plan.json").write_text(json.dumps({"source_code_commit": "a" * 40,
        "implementation_sha256": original}))
    assert WRAPPER.source_signature(args, "a" * 40) == original
    (args.source_root / "scripts/run_supplemental_gpqa.py").write_text("# changed\n")
    with pytest.raises(RuntimeError, match="Frozen source"):
        WRAPPER.source_signature(args, "a" * 40)


def test_actual_hung_campaign_repairs_with_frozen_modules_and_resumes_cached_root(tmp_path, monkeypatch):
    args = fixture_args(tmp_path)
    (args.source_root / "scripts/run_gpqa_sft_campaign.py").write_text('''
import json, os, pathlib, sqlite3, sys, time
root = pathlib.Path(sys.argv[sys.argv.index("--output-root") + 1])
root.mkdir(exist_ok=True)
invocations = root / "invocations.json"
calls = json.loads(invocations.read_text()) if invocations.exists() else []
calls.append({"argv": sys.argv, "source": os.environ["SOURCE_CODE_COMMIT"],
              "pythonpath": os.environ["PYTHONPATH"]})
invocations.write_text(json.dumps(calls))
db = sqlite3.connect(root / "drive_registry.sqlite3")
db.execute("CREATE TABLE IF NOT EXISTS artifacts(sync_status TEXT,record TEXT)")
if len(calls) == 1:
    db.execute("INSERT INTO artifacts VALUES (?,?)", ("FAILED", json.dumps({"sync_error":"TimeoutError"})))
    db.commit()
    (root / "cached_raw").write_bytes(b"original generated bytes")
    log = root / "campaigns/fixture/state.log"
    log.parent.mkdir(parents=True)
    log.write_text("RuntimeError: Permanent artifact synchronization incomplete; registry retains pending/failed records\\n")
    time.sleep(60)
else:
    assert db.execute("SELECT sync_status FROM artifacts").fetchone()[0] == "VERIFIED"
    assert (root / "cached_raw").read_bytes() == b"original generated bytes"
''')
    operations = args.source_root / "src/operations"
    operations.mkdir(parents=True)
    (operations / "__init__.py").write_text("")
    (operations / "drive.py").write_text('''
def drive_service(token):
    return object()
class DriveStore:
    def __init__(self, service, folder):
        pass
''')
    (operations / "artifacts.py").write_text('''
import sqlite3
class ArtifactService:
    def __init__(self, root, registry, drive, *, drive_recovery=False):
        assert drive_recovery is True
        self.root = root
    def sync_artifacts(self, *, verify_existing=False):
        assert verify_existing is False
        with sqlite3.connect(self.root / "drive_registry.sqlite3") as db:
            db.execute("UPDATE artifacts SET sync_status='VERIFIED',record='{}'")
        return {"status":"VERIFIED"}
''')
    monkeypatch.setenv("SOURCE_CODE_COMMIT", "a" * 40)
    monkeypatch.setenv("GOOGLE_OAUTH_TOKEN_FILE", "/private/fixture-token-path")
    monkeypatch.setenv("GOOGLE_DRIVE_FOLDER_ID", "fixture-folder")
    assert WRAPPER.supervise(args) == 0
    status = json.loads((args.monitor_root / "status.json").read_text())
    assert status["status"] == "COMPLETE"
    assert status["recoveries"] == 1
    calls = json.loads((args.output_root / "invocations.json").read_text())
    assert len(calls) == 2 and calls[0] == calls[1]
    assert calls[0]["source"] == "a" * 40
    assert calls[0]["pythonpath"] == str(args.source_root / "src")
    assert calls[0]["argv"][1:] == WRAPPER.campaign_command(args)[2:]
