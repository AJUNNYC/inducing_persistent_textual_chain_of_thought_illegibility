#!/usr/bin/env python3
"""Run native clean collection until a UTC deadline requests its existing pause.

Normal deadlines request a safe response boundary. An unusable pause path
aborts only the native child; this supervisor never stops the Pod or billing.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading


ROOT = Path(__file__).resolve().parents[1]


def utc_deadline(value: str) -> datetime:
    try:
        selected = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except (ValueError, TypeError):
        raise ValueError('Deadline must be an absolute UTC ISO8601 timestamp') from None
    if selected.utcoffset() is None or selected.utcoffset().total_seconds() != 0:
        raise ValueError('Deadline must include UTC Z or +00:00')
    return selected.astimezone(timezone.utc)


def _safe_root(value: str, base: Path) -> Path:
    selected = Path(value)
    selected = base / selected if not selected.is_absolute() else selected
    if '..' in selected.parts or any(path.is_symlink() for path in (selected, *selected.parents)):
        raise ValueError('Collection output roots must not traverse symbolic links or parent components')
    selected = selected.resolve()
    if selected.is_relative_to(ROOT):
        raise ValueError('Set collection output roots outside the source repository')
    return selected


def collection_paths(config_path: str | Path, environment: dict) -> tuple[Path, Path]:
    # This matches Experiment.__init__: relative roots use the configuration
    # file's directory, and these three environment variables take precedence.
    import yaml
    configuration = Path(config_path).resolve()
    try:
        document = yaml.safe_load(configuration.read_text())
    except yaml.YAMLError:
        raise ValueError('The native runtime configuration could not be parsed') from None
    runtime = document.get('runtime') if isinstance(document, dict) else None
    if not isinstance(runtime, dict):
        raise ValueError('A native runtime configuration is required')
    roots = {}
    for field, variable in (('data_root', 'DATA_ROOT'), ('artifact_root', 'ARTIFACT_ROOT'),
                            ('cache_root', 'MODEL_CACHE_ROOT')):
        value = environment.get(variable) or runtime.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError('The native data, artifact and cache roots must be configured')
        roots[field] = _safe_root(value, configuration.parent)
    marker = roots['data_root'] / 'clean_traces' / 'STOP_REQUESTED'
    if marker.exists() or marker.is_symlink():
        raise ValueError('An existing STOP_REQUESTED marker must be reviewed before a new run')
    if any(path.is_symlink() for path in marker.parents):
        raise ValueError('The collection stop path must not traverse symbolic links')
    return marker, roots['artifact_root']


def request_stop(marker: Path) -> None:
    """Atomically create the collector's empty stop marker without replacing it."""
    if any(path.is_symlink() for path in (marker, *marker.parents)):
        raise ValueError('The collection stop path must not traverse symbolic links')
    marker.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(marker.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


@contextmanager
def graceful_signals():
    interrupted = threading.Event()
    previous = {}
    def mark_interrupted(signum, frame):
        interrupted.set()
    for name in (signal.SIGINT, signal.SIGTERM):
        previous[name] = signal.signal(name, mark_interrupted)
    try:
        yield interrupted
    finally:
        for name, handler in previous.items():
            signal.signal(name, handler)


def supervise(command: list[str], *, marker: Path, deadline: datetime, environment: dict,
              log, interrupted: threading.Event) -> dict:
    if deadline <= datetime.now(timezone.utc):
        raise ValueError('Deadline has already passed; collection was not launched')
    if marker.exists() or marker.is_symlink():
        raise ValueError('An existing STOP_REQUESTED marker must be reviewed before a new run')
    if any(path.is_symlink() for path in marker.parents):
        raise ValueError('The collection stop path must not traverse symbolic links')
    process = subprocess.Popen(command, env=environment, cwd=ROOT, stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
    reason = None
    while process.poll() is None:
        remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
        if reason is None and (interrupted.is_set() or remaining <= 0):
            reason = 'parent_interrupted' if interrupted.is_set() else 'deadline_reached'
            try:
                request_stop(marker)
            except (ValueError, OSError) as error:
                # A failed supervisor must not leave its paid native child
                # running without any stop request. This exceptional abort
                # can discard the in-flight operation; prior Drive artifacts
                # remain the only completed results. Normal deadlines pause.
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                return {'status': 'STOP_REQUEST_FAILED', 'native_exit_code': process.returncode,
                        'stop_requested': False, 'stop_reason': reason,
                        'error_type': type(error).__name__, 'native_child_aborted': True,
                        'pod_billing_stopped': False}
            print(json.dumps({'status': 'STOP_REQUESTED', 'reason': reason,
                'native_process': 'waiting_for_safe_pause', 'pod_billing_stopped': False}), flush=True)
        try:
            process.wait(timeout=0.25 if reason else max(0.001, min(0.25, remaining)))
        except subprocess.TimeoutExpired:
            pass
    return {'status': {0: 'COMPLETED', 3: 'PAUSED'}.get(process.returncode, 'NATIVE_EXITED'),
            'native_exit_code': process.returncode, 'stop_requested': reason is not None,
            'stop_reason': reason, 'pod_billing_stopped': False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--deadline', required=True, help='Absolute UTC timestamp, e.g. 2026-10-11T20:20:00Z')
    args = parser.parse_args(argv)
    try:
        deadline = utc_deadline(args.deadline)
        if deadline <= datetime.now(timezone.utc):
            raise ValueError('Deadline has already passed; collection was not launched')
        environment = dict(os.environ)
        marker, artifacts = collection_paths(args.config, environment)
        directory = artifacts / 'operations' / 'collection_deadlines'
        if any(path.is_symlink() for path in (directory, *directory.parents)):
            raise ValueError('The collection log path must not traverse symbolic links')
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        log_path = directory / f'{stamp}_{os.getpid()}.log'
        environment['PYTHONPATH'] = str(ROOT / 'src') + (os.pathsep + environment['PYTHONPATH']
                                                     if environment.get('PYTHONPATH') else '')
        command = [sys.executable, '-m', 'experiment_overview.experiment_overview',
                   '--config', str(Path(args.config).resolve()), 'run', '--target', 'clean']
        # Keep credential-bearing environment values out of command summaries
        # and supervisor output. Native output stays in the configured scratch.
        with log_path.open('x') as log, graceful_signals() as interrupted:
            os.chmod(log_path, 0o600)
            result = supervise(command, marker=marker, deadline=deadline,
                               environment=environment, log=log, interrupted=interrupted)
        print(json.dumps({**result, 'native_log': str(log_path)}), flush=True)
        return result['native_exit_code'] if result['native_exit_code'] >= 0 else 2
    except (ValueError, OSError) as error:
        # OS/transport details can contain sensitive values; report the type.
        message = str(error) if isinstance(error, ValueError) else type(error).__name__
        print(json.dumps({'status': 'BLOCKED', 'reason': message, 'pod_billing_stopped': False}), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
