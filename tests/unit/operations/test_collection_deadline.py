"""Deadline supervision exercises real disposable subprocesses, never a model."""
from datetime import datetime, timedelta, timezone
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading

import pytest
import yaml


SCRIPT = Path(__file__).resolve().parents[3] / 'scripts/run_collection_until.py'


@pytest.fixture
def wrapper():
    spec = importlib.util.spec_from_file_location('collection_deadline', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_actual_deadline_requests_native_pause_and_keeps_completed_work(wrapper, tmp_path):
    marker = tmp_path / 'data/clean_traces/STOP_REQUESTED'
    completed = tmp_path / 'completed.json'
    child = """import json, pathlib, sys, time
completed=pathlib.Path(sys.argv[1]); marker=pathlib.Path(sys.argv[2])
completed.write_text(json.dumps({'status':'completed-before-pause'}))
while not marker.exists(): time.sleep(0.01)
assert json.loads(completed.read_text())['status']=='completed-before-pause'
sys.exit(3)
"""
    with (tmp_path / 'native.log').open('w') as log:
        result = wrapper.supervise([sys.executable, '-c', child, str(completed), str(marker)],
            marker=marker, deadline=datetime.now(timezone.utc) + timedelta(seconds=0.3),
            environment=dict(os.environ), log=log, interrupted=threading.Event())
    assert result['status'] == 'PAUSED' and result['native_exit_code'] == 3
    assert result['stop_reason'] == 'deadline_reached' and result['pod_billing_stopped'] is False
    assert marker.read_bytes() == b'' and marker.stat().st_mode & 0o777 == 0o600
    assert json.loads(completed.read_text()) == {'status': 'completed-before-pause'}


def test_success_before_deadline_does_not_leave_a_stop_marker(wrapper, tmp_path):
    marker = tmp_path / 'clean_traces/STOP_REQUESTED'
    with (tmp_path / 'native.log').open('w') as log:
        result = wrapper.supervise([sys.executable, '-c', 'raise SystemExit(0)'], marker=marker,
            deadline=datetime.now(timezone.utc) + timedelta(seconds=5), environment=dict(os.environ),
            log=log, interrupted=threading.Event())
    assert result['status'] == 'COMPLETED' and result['stop_requested'] is False
    assert not marker.exists()


def test_failed_stop_request_aborts_only_native_child_and_sanitizes_error(wrapper, tmp_path, monkeypatch):
    def fail(marker):
        raise OSError('PUBLIC_FIXTURE_DETAIL_MUST_NOT_APPEAR')
    monkeypatch.setattr(wrapper, 'request_stop', fail)
    with (tmp_path / 'native.log').open('w') as log:
        result = wrapper.supervise([sys.executable, '-c', 'import time;time.sleep(30)'],
            marker=tmp_path/'STOP_REQUESTED', deadline=datetime.now(timezone.utc)+timedelta(seconds=0.2),
            environment=dict(os.environ), log=log, interrupted=threading.Event())
    assert result['status'] == 'STOP_REQUEST_FAILED' and result['native_child_aborted'] is True
    assert result['native_exit_code'] < 0 and result['stop_requested'] is False
    assert result['pod_billing_stopped'] is False and result['error_type'] == 'OSError'
    assert 'PUBLIC_FIXTURE_DETAIL' not in str(result)


def test_parent_sigterm_requests_pause_without_signaling_native_child(tmp_path):
    marker = tmp_path / 'data/clean_traces/STOP_REQUESTED'
    ready = tmp_path / 'ready'
    native = tmp_path / 'native.py'
    native.write_text("""import pathlib, signal, sys, time
signal.signal(signal.SIGTERM, lambda *args: sys.exit(99))
pathlib.Path(sys.argv[2]).write_text('ready')
while not pathlib.Path(sys.argv[1]).exists(): time.sleep(0.01)
sys.exit(3)
""")
    outer = tmp_path / 'supervisor.py'
    outer.write_text("""import importlib.util, datetime, json, os, pathlib, sys
spec=importlib.util.spec_from_file_location('wrapper',sys.argv[1]); wrapper=importlib.util.module_from_spec(spec); spec.loader.exec_module(wrapper)
with open(sys.argv[5],'w') as log, wrapper.graceful_signals() as interrupted:
 result=wrapper.supervise([sys.executable,sys.argv[2],sys.argv[3],sys.argv[4]],marker=pathlib.Path(sys.argv[3]),deadline=datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(seconds=10),environment=dict(os.environ),log=log,interrupted=interrupted)
print(json.dumps(result))
""")
    process = subprocess.Popen([sys.executable, str(outer), str(SCRIPT), str(native), str(marker),
                                str(ready), str(tmp_path / 'native.log')], stdout=subprocess.PIPE, text=True)
    try:
        import time
        limit = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < limit:
            time.sleep(0.01)
        assert ready.exists()
        process.send_signal(signal.SIGTERM)
        output, _ = process.communicate(timeout=5)
        result = json.loads(output.splitlines()[-1])
        assert result['status'] == 'PAUSED' and result['stop_reason'] == 'parent_interrupted'
        assert result['native_exit_code'] == 3 and marker.is_file()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_paths_match_native_config_base_and_environment_precedence(wrapper, tmp_path):
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({'runtime': {'data_root': 'data', 'artifact_root': 'artifacts',
                                                 'cache_root': 'cache'}}))
    marker, artifacts = wrapper.collection_paths(config, {'DATA_ROOT': str(tmp_path / 'override')})
    assert marker == tmp_path / 'override/clean_traces/STOP_REQUESTED'
    assert artifacts == tmp_path / 'artifacts'
    marker.parent.mkdir(parents=True)
    marker.touch()
    with pytest.raises(ValueError, match='existing STOP_REQUESTED'):
        wrapper.collection_paths(config, {'DATA_ROOT': str(tmp_path / 'override')})


def test_symlink_roots_stop_paths_and_repository_outputs_are_refused(wrapper, tmp_path):
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({'runtime': {'data_root': 'data', 'artifact_root': 'artifacts',
                                                 'cache_root': 'cache'}}))
    actual = tmp_path / 'actual'; actual.mkdir()
    link = tmp_path / 'linked'; link.symlink_to(actual, target_is_directory=True)
    with pytest.raises(ValueError, match='symbolic links'):
        wrapper.collection_paths(config, {'DATA_ROOT': str(link / 'data')})
    with pytest.raises(ValueError, match='outside the source repository'):
        wrapper.collection_paths(config, {'ARTIFACT_ROOT': str(wrapper.ROOT / 'artifacts')})
    data = tmp_path / 'data'; data.mkdir()
    (data / 'clean_traces').symlink_to(actual, target_is_directory=True)
    with pytest.raises(ValueError, match='symbolic links'):
        wrapper.collection_paths(config, {})
    with pytest.raises(ValueError, match='symbolic links'):
        wrapper.request_stop(data / 'clean_traces/STOP_REQUESTED')
    assert not (actual / 'STOP_REQUESTED').exists()


def test_past_deadline_and_old_marker_prevent_actual_subprocess_launch(wrapper, tmp_path):
    marker = tmp_path / 'STOP_REQUESTED'; launched = tmp_path / 'launched'
    command = [sys.executable, '-c', 'import pathlib,sys;pathlib.Path(sys.argv[1]).touch()', str(launched)]
    with (tmp_path / 'native.log').open('w') as log:
        with pytest.raises(ValueError, match='already passed'):
            wrapper.supervise(command, marker=marker, deadline=datetime.now(timezone.utc)-timedelta(seconds=1),
                environment=dict(os.environ), log=log, interrupted=threading.Event())
        marker.write_bytes(b'old-stop-request')
        with pytest.raises(ValueError, match='existing STOP_REQUESTED'):
            wrapper.supervise(command, marker=marker, deadline=datetime.now(timezone.utc)+timedelta(seconds=5),
                environment=dict(os.environ), log=log, interrupted=threading.Event())
    assert not launched.exists() and marker.read_bytes() == b'old-stop-request'


@pytest.mark.parametrize('value', ['2026-10-11T20:20:00', '2026-10-11T20:20:00+01:00', 'tomorrow'])
def test_non_utc_or_ambiguous_deadline_is_rejected(wrapper, value):
    with pytest.raises(ValueError):
        wrapper.utc_deadline(value)


def test_exact_utc_deadline_and_native_failure_are_reported_without_retry(wrapper, tmp_path):
    assert wrapper.utc_deadline('2026-10-11T20:20:00Z') == datetime(2026, 10, 11, 20, 20, tzinfo=timezone.utc)
    counter = tmp_path / 'attempts'
    child = 'import pathlib,sys;p=pathlib.Path(sys.argv[1]);p.write_text(p.read_text()+"x" if p.exists() else "x");sys.exit(2)'
    with (tmp_path / 'native.log').open('w') as log:
        result = wrapper.supervise([sys.executable, '-c', child, str(counter)], marker=tmp_path/'STOP_REQUESTED',
            deadline=datetime.now(timezone.utc)+timedelta(seconds=5), environment=dict(os.environ),
            log=log, interrupted=threading.Event())
    assert result['status'] == 'NATIVE_EXITED' and result['native_exit_code'] == 2
    assert counter.read_text() == 'x'
