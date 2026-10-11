#!/usr/bin/env python3
"""Read-only monitoring of the two existing Jupyter-backed generation workers.

Connection credentials stay in protected files. Monitoring never restarts a
worker, changes experimental data, or treats raw generation as accepted data.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from shared_procedures.storage import atomic_write_json, append_jsonl, read_json
import requests


def inspect(model, credential_file, artifact_root, expected_commit):
    auth = read_json(credential_file)
    session = requests.Session()
    session.headers['Authorization'] = 'token ' + auth['token']
    api = auth['base_url'].rstrip('/') + '/api/contents/'
    def fetch(path):
        response = session.get(api + quote(path.lstrip('/')),
            params={'content':1,'format':'base64'},timeout=30)
        if response.status_code != 200:
            raise RuntimeError('HTTP_'+str(response.status_code))
        return base64.b64decode(response.json()['content']).decode()
    try:
        state = json.loads(fetch(artifact_root + '/progress.json'))
        if state['source_code_commit'] != expected_commit or state['model'] != model or state['backend'] != 'vllm':
            return {'model':model,'status':'IDENTITY_MISMATCH'}
        log = fetch('/workspace/bootstrap/vllm_generation_'+model+'_dc520cb.log')
        events = []
        for line in log.splitlines():
            if line.startswith('{'):
                try:
                    events.append(json.loads(line))
                except ValueError:
                    pass
        terminal = next((e for e in reversed(events) if e.get('event') in ('COMPLETE','PAUSED','FAILED')),None)
        age = (datetime.now(timezone.utc)-datetime.fromisoformat(state['updated_at'])).total_seconds()
        status = terminal['event'] if terminal else ('NO_PROGRESS_10_MINUTES' if age>600 else 'RUNNING')
        rate = (state['last_batch_generation_seconds']+state['last_batch_drive_seconds'])/state['batch_size']
        remaining = max(0,state['requested']-state['completed'])
        return {'model':model,'status':status,'completed':state['completed'],
            'drive_verified':state['drive_verified'],'requested':state['requested'],
            'remaining_minutes_estimate':round(remaining*rate/60,1) if state['completed']>state['batch_size'] else None,
            'last_batch_generation_seconds':round(state['last_batch_generation_seconds'],1),
            'last_batch_drive_seconds':round(state['last_batch_drive_seconds'],1),
            'last_progress_at':state['updated_at'],'source_code_commit':expected_commit}
    except Exception as error:
        # Never serialize transport exceptions that could disclose connection data.
        return {'model':model,'status':'MONITOR_CONNECTION_ERROR','error_type':type(error).__name__}
    finally:
        session.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--credentials-7b',default='/workspace/.secrets/runpod_jupyter.json')
    p.add_argument('--credentials-14b',default='/workspace/.secrets/runpod_jupyter_14b.json')
    p.add_argument('--artifact-root',default='/workspace/artifacts/collection_vllm_dc520cb')
    p.add_argument('--source-commit',default='dc520cb73bf43b59a9566404dd4a7aa8358f1cc3')
    p.add_argument('--output-root',default='/workspace/scratch/vllm_monitor')
    p.add_argument('--interval',type=int,default=30)
    p.add_argument('--once',action='store_true')
    a=p.parse_args()
    if a.interval<10:
        p.error('Use at least ten seconds between observations')
    output=Path(a.output_root);output.mkdir(parents=True,exist_ok=True)
    previous=None
    with ThreadPoolExecutor(max_workers=2) as pool:
        while True:
            futures=[pool.submit(inspect,m,c,a.artifact_root,a.source_commit)
                for m,c in [('7b',a.credentials_7b),('14b',a.credentials_14b)]]
            models=[f.result() for f in futures]
            snapshot={'observed_at':datetime.now(timezone.utc).isoformat(),'models':models,'scope':'initial_raw_generation_and_drive_replication'}
            atomic_write_json(output/'latest.json',snapshot)
            changed=json.dumps(models,sort_keys=True)!=previous
            if changed:
                append_jsonl(output/'history.jsonl',snapshot)
                print(json.dumps(snapshot),flush=True)
                previous=json.dumps(models,sort_keys=True)
            if a.once or all(m['status'] in ('COMPLETE','PAUSED','FAILED') for m in models):
                return
            time.sleep(a.interval)


if __name__=='__main__':
    main()
