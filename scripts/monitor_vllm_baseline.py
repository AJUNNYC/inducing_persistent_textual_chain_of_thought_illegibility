#!/usr/bin/env python3
"""Read-only progress monitor for sequential one-GPU baseline verification.

Writes local snapshots and terminal markers; never restarts work or sends chat,
email or desktop notifications. Transport errors never serialize credentials.
"""
import argparse
import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from urllib.parse import quote
import requests
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from shared_procedures.storage import atomic_write_json,append_jsonl,file_lock,read_json


def inspect(session,base,worker_path,expected_commit):
    def fetch(path,optional=False,decode_json=True):
        response=session.get(base+'/api/contents/'+quote(path.lstrip('/')),
            params={'content':1,'format':'base64'},timeout=30)
        if optional and response.status_code==404:return None
        if response.status_code!=200:raise RuntimeError('HTTP_'+str(response.status_code))
        content=base64.b64decode(response.json()['content']).decode()
        return json.loads(content) if decode_json else content
    worker=fetch(worker_path)
    if worker['source_commit']!=expected_commit:raise ValueError('Worker identity changed')
    root=worker['artifact_root']
    models=[]
    for model in ('7b','14b'):
        status=fetch(root+'/baseline_'+model+'_status.json',optional=True)
        if status is None:
            models.append({'model':model,'status':'QUEUED','completed':0,'total':3699})
            continue
        if status.get('source_commit',expected_commit)!=expected_commit or status['model']!=model:
            raise ValueError('Status identity changed')
        done=status.get('items',0) if status['status']=='COMPLETE' else (
            1233*(status.get('seed',42)-42)+status.get('completed_items_in_pass',0))
        timings=status.get('timings',[])
        # Exclude cold startup; no cross-model end-to-end ETA is inferred.
        warm=timings[1:][-3:]
        rate=sum(t['items'] for t in warm)/sum(t['generation_and_scoring_seconds']+t['drive_seconds'] for t in warm) if warm else None
        models.append({'model':model,'status':status['status'],'completed':done,'total':3699,
            'seed':status.get('seed'),'completed_in_pass':status.get('completed_items_in_pass'),
            'drive_replication':status.get('drive_replication'),
            'latest_progress_at':status.get('at'),'recent_items_per_second':round(rate,2) if rate else None})
    report=fetch(root+'/preflight/baseline_repeatability.json',optional=True)
    state='RUNNING'
    if report is not None:
        # Parent emits this event only after verifying the final report in Drive.
        events=[]
        for line in fetch(worker['log'],decode_json=False).splitlines():
            if line.startswith('{'):
                try:events.append(json.loads(line))
                except ValueError:pass
        final=next((e for e in reversed(events) if e.get('full_evaluation')=='pending_legibility_judgments'),None)
        verified=final is not None and final.get('repeatability_gate')==report.get('ready_for_training') and final.get('status')==report.get('status')
        state=('COMPLETE' if report.get('ready_for_training') is True else 'GATE_FAILED') if verified else 'FINALIZING'
    elif any(m['status']=='FAILED' for m in models):state='FAILED'
    elif any(m['status']=='PAUSED' for m in models):state='PAUSED'
    elif all(m['status']=='COMPLETE' for m in models):state='FINALIZING'
    return {'observed_at':datetime.now(timezone.utc).isoformat(),'status':state,'models':models,
        'completed':sum(m['completed'] for m in models),'total':7398,
        'repeatability_gate_passed':report.get('ready_for_training') if report else None,
        'judging':'deferred','scope':'capability_repeatability_only','source_commit':expected_commit}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--credentials',default='/workspace/.secrets/runpod_jupyter.json')
    parser.add_argument('--worker-metadata',default='/workspace/bootstrap/baseline_vllm_worker.json')
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--output-root',default='/workspace/scratch/vllm_baseline_monitor')
    parser.add_argument('--interval',type=int,default=30)
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--local-status',help='Cloud scorer status when GPU generation and CPU scoring are separated')
    args=parser.parse_args()
    if args.interval<10:parser.error('Minimum interval is ten seconds')
    output=Path(args.output_root);output.mkdir(parents=True,exist_ok=True)
    auth=read_json(args.credentials)
    session=requests.Session();session.headers['Authorization']='token '+auth['token']
    previous=None
    with file_lock(output/'monitor.lock'):
        while True:
            try:
                snapshot=read_json(args.local_status) if args.local_status else inspect(session,auth['base_url'].rstrip('/'),args.worker_metadata,args.source_commit)
            except Exception as error:
                snapshot={'observed_at':datetime.now(timezone.utc).isoformat(),'status':'MONITOR_CONNECTION_ERROR',
                    'error_type':type(error).__name__,'does_not_establish_worker_failure':True}
            atomic_write_json(output/'latest.json',snapshot)
            append_jsonl(output/'history.jsonl',snapshot)
            marker={k:v for k,v in snapshot.items() if k!='observed_at'}
            if marker!=previous:
                print(json.dumps(snapshot),flush=True);previous=marker
            if snapshot['status'] in ('COMPLETE','GATE_FAILED','FAILED','PAUSED'):
                atomic_write_json(output/'terminal.json',snapshot)
                return 0 if snapshot['status']=='COMPLETE' else 1
            if args.once:return 0
            time.sleep(args.interval)

if __name__=='__main__':raise SystemExit(main())
