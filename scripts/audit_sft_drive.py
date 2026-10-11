#!/usr/bin/env python3
"""Independently read back and validate the final SFT handoff from Google Drive.

No RunPod connection, GPU, generation, or transformation recomputation is used.
Only protected credential file paths, never credential values, enter arguments.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import requests
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from data_generation_collection.sft_data.sft_data_preparation import validate_accepted_pool, validate_sft_bundle
from shared_procedures.identity import content_hash, file_hash
from shared_procedures.storage import atomic_write_json

CONFIG_HASH='3757493e93d7f3ac31d9215bd493ae4f13cae0bed7c97e773f4ca4ef5be2ed54'
GENERATION_COMMIT='dc520cb73bf43b59a9566404dd4a7aa8358f1cc3'
PREFIX='data/vllm_sft_preparation/'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--credentials',required=True)
    p.add_argument('--index-id',required=True)
    p.add_argument('--report',required=True)
    args=p.parse_args()
    credentials=Credentials.from_authorized_user_file(args.credentials)
    if not credentials.valid:
        credentials.refresh(Request())
    headers={'Authorization':'Bearer '+credentials.token}
    base='https://www.googleapis.com/drive/v3/files/'
    response=requests.get(base+args.index_id,headers=headers,params={'alt':'media'},timeout=60)
    response.raise_for_status()
    index=response.json()
    entries=index['artifacts']
    if (index['status']!='VERIFIED' or index['conditions']!=40 or index['configuration_hash']!=CONFIG_HASH
            or len(entries)!=122 or len({e['relative_path'] for e in entries})!=122
            or len({e['drive_file_id'] for e in entries})!=122):
        raise ValueError('Unexpected final Drive index')
    for entry in entries:
        rel=Path(entry['relative_path'])
        if rel.is_absolute() or '..' in rel.parts or not str(rel).startswith(PREFIX):
            raise ValueError('Invalid indexed artifact path')
        if not (str(rel).startswith(PREFIX+'transformed/') or str(rel)==PREFIX+'accepted/accepted_pool.json'):
            raise ValueError('Artifact outside final dataset handoff')
    def download(entry):
        target=Path('/workspace')/entry['relative_path']
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and file_hash(target)!=entry['sha256']:
            raise ValueError('Existing local artifact differs; refusing overwrite')
        r=requests.get(base+entry['drive_file_id'],headers=headers,params={'alt':'media'},
                       stream=True,timeout=(30,60))
        r.raise_for_status()
        digest=hashlib.sha256();size=0
        fd,name=tempfile.mkstemp(prefix='.drive_readback_',dir=target.parent)
        temporary=Path(name)
        try:
            with os.fdopen(fd,'wb') as f:
                for chunk in r.iter_content(1024*1024):
                    f.write(chunk);digest.update(chunk);size+=len(chunk)
            if size!=entry['size'] or digest.hexdigest()!=entry['sha256']:
                raise ValueError('Drive artifact readback checksum/size mismatch')
            os.replace(temporary,target)
        finally:
            r.close()
            temporary.unlink(missing_ok=True)
        return {'path':entry['relative_path'],'sha256':digest.hexdigest(),'bytes':size}
    verified=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(download,e) for e in entries]
        for future in as_completed(futures):
            verified.append(future.result())
            if len(verified)%10==0 or len(verified)==122:
                print(json.dumps({'status':'READBACK','verified_files':len(verified),'total_files':122}),flush=True)
    root=Path('/workspace')/PREFIX
    pool=json.loads((root/'accepted/accepted_pool.json').read_text())
    bundle=json.loads((root/'transformed/manifest.json').read_text())
    for condition in bundle['datasets']:
        manifest=Path(condition['train_path']).parent/'manifest.json'
        if json.loads(manifest.read_text())!=condition:
            raise ValueError('Downloaded condition manifest differs from bundle')
    validate_accepted_pool(pool,CONFIG_HASH)
    revisions={model:set() for model in ('7b','14b')}
    for pair in pool['pairs']:
        for model,trace in pair['traces'].items():
            if (trace['producer_commit']!=GENERATION_COMMIT or trace['simulation'] is not False
                    or trace['execution_mode']!='production'
                    or trace['runtime']['generation_backend']!='vllm_seeded_batch_v1'):
                raise ValueError('Accepted trace has incompatible production provenance')
            revisions[model].add((trace['model_revision'],trace['tokenizer_revision']))
    if any(len(values)!=1 for values in revisions.values()):
        raise ValueError('Model/tokenizer revision changed within corpus')
    print(json.dumps({'status':'VALIDATING_RECORDS','records':404000}),flush=True)
    result=validate_sft_bundle(bundle,pool)
    if result['record_count']!=404000:
        raise ValueError('Final record count differs')
    report={'status':'PASSED','checked_at':datetime.now(timezone.utc).isoformat(),
            'source':'Google Drive full byte readback','index_file_id':args.index_id,
            'index_sha256':content_hash(response.content),'verified_files':len(verified),
            'downloaded_bytes':sum(e['bytes'] for e in verified),
            'accepted_pairs':len(pool['pairs']),'train_pairs':len(pool['train_ids']),
            'condition_manifests_checked':len(bundle['datasets']),
            'validation_pairs':len(pool['validation_ids']),'rejected_pairs':len(pool['rejected_ids']),
            'production_backend':'vllm_seeded_batch_v1','generation_commit':GENERATION_COMMIT,
            'pinned_revisions':{m:list(next(iter(values))) for m,values in revisions.items()},
            'validation':result,'artifacts':sorted(verified,key=lambda e:e['path']),
            'gpu_used':False,'pod_connections_used':False,
            'limits':'Integrity and frozen record invariants; no answer-quality grading or transform/tokenizer recomputation.'}
    atomic_write_json(args.report,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('artifacts','pinned_revisions')}),flush=True)


if __name__=='__main__':
    main()
