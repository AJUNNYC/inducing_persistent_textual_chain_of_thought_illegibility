#!/usr/bin/env python3
"""Run disjoint continuation/evaluation configs sequentially on one GPU.

Scientific work stays in the pinned continuation and GPQA CLIs. Successful exit
alone never advances the queue: training endpoints and evaluation Drive receipts
must also show completion. Restarting reuses the native idempotent job paths.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.chmod(0o600)
    temporary.replace(path)


def training_complete(config):
    endpoints = read(config['training_endpoints'])
    if (endpoints.get('status') != 'COMPLETE'
            or endpoints.get('drive_status') != 'VERIFIED'
            or [e['level'] for e in endpoints.get('adapters', [])] != [config['level']]):
        raise RuntimeError('Selected training endpoint is not Drive-verified')


def evaluation_complete(config):
    status = read(Path(config['output']) / 'followup_status.json')
    receipt = read(Path(config['output']) / 'handoff/completion_verified.json')
    if (status.get('phase') != 'COMPLETE_DRIVE_VERIFIED'
            or status.get('questions') != 198 or status.get('level') != config['level']
            or receipt.get('status') != 'COMPLETE' or receipt.get('drive_status') != 'VERIFIED'):
        raise RuntimeError('Evaluation has not passed its full Drive completion barrier')


def run(config_paths, root):
    root.mkdir(parents=True, exist_ok=True)
    configs = [read(p) for p in config_paths]
    if len({c['level'] for c in configs}) != len(configs) or len({c['output'] for c in configs}) != len(configs):
        raise ValueError('Queue requires distinct levels and evaluation outputs')
    stop = {'requested': False}
    def pause(_sig, _frame):
        stop['requested'] = True
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, pause)
    def stopped():
        return stop['requested'] or (root/'PAUSE_REQUESTED').exists()
    def status(phase, **extra):
        write(root/'queue_status.json', {'phase': phase, 'updated_unix': time.time(), **extra})
    def child(command, env, log_path):
        with log_path.open('ab') as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            sent = False
            while process.poll() is None:
                if stopped() and not sent:
                    # The training campaign saves a full checkpoint on SIGTERM;
                    # the evaluation supervisor stops its owned inference group.
                    process.send_signal(signal.SIGTERM)
                    sent = True
                time.sleep(2)
        if stopped() or process.returncode == 3:
            status('PAUSED'); return False
        if process.returncode != 0:
            raise RuntimeError(f'Published worker exited with code {process.returncode}')
        return True
    lock = (root/'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    completed = []
    try:
        for config_path, config in zip(config_paths, configs):
            level = config['level']
            if stopped():
                status('PAUSED', completed=completed); return 3
            env = {**os.environ, **config['training_environment']}
            status('TRAINING', active_level=level, completed=completed)
            if not child(config['training_command'], env, root/f'level_{level:02d}_training.log'):
                return 3
            training_complete(config)
            write(config['training_watcher_status'], {'phase': 'COMPLETE_DRIVE_VERIFIED',
                  'level': level, 'completed_updates': 625, 'updated_unix': time.time()})
            status('EVALUATING', active_level=level, completed=completed)
            command = [config['training_command'][0], config['dispatcher'], '--config', str(config_path)]
            if not child(command, env, root/f'level_{level:02d}_evaluation.log'):
                return 3
            evaluation_complete(config)
            completed.append(level)
        status('COMPLETE_DRIVE_VERIFIED', completed=completed)
        return 0
    except Exception as error:
        status('FAILED', completed=completed, error_type=type(error).__name__)
        raise


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    return run(args.configs, args.output)


if __name__ == '__main__':
    raise SystemExit(main())
