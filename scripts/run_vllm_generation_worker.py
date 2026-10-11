#!/usr/bin/env python3
"""Fresh or resumed single-model vLLM generation with a Drive barrier per batch.

Outputs are raw construction responses. Acceptance, pairing, transformations,
training and any required refill requests remain separate native stages.
"""
import argparse
import copy
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from data_generation_collection.sft_data.clean_trace_collection.candidate_batch.vllm_generator import VLLMCleanGenerator, BACKEND
from data_generation_collection.sft_data.clean_trace_collection.candidate_batch.clean_trace_batch import GENERATION_SETTINGS
from operations.artifacts import ArtifactService
from operations.drive import DriveStore, drive_service
from shared_procedures.configuration import load_configuration, configuration_hash
from shared_procedures.identity import content_hash
from shared_procedures.storage import atomic_write_json, read_json, file_lock


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config','candidates','data-root','artifact-root','cache-root','deadline','run-id'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--model', choices=['7b','14b'], required=True)
    p.add_argument('--batch-size', type=int, default=128)
    a = p.parse_args()
    deadline = datetime.fromisoformat(a.deadline.replace('Z','+00:00'))
    if deadline.utcoffset() is None or deadline.utcoffset().total_seconds() != 0 or deadline <= datetime.now(timezone.utc):
        raise ValueError('Future absolute UTC deadline required')
    base = load_configuration(a.config)
    if not base['spending']['authorized']:
        raise ValueError('Existing compute authorization required')
    selected = read_json(a.candidates)
    candidates = selected['candidates']
    if (selected['consumer_configuration_hash'] != configuration_hash(base)
            or selected['selected_candidates_hash'] != content_hash(candidates)
            or len({c['example_id'] for c in candidates}) != len(candidates)):
        raise ValueError('Previously audited candidate identity differs')
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    config = copy.deepcopy(base)
    config['runtime']['implementation_revision'] = commit
    # The native Torch lock is not the dependency lock for this separate engine.
    config['runtime'].pop('dependency_lock_hash', None)
    config['runtime']['generation_backend'] = BACKEND
    config['runtime']['environment_inventory_policy'] = 'per_model_run_manifest'
    identity = configuration_hash(config)
    packages = {d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata.get('Name')}
    data, artifacts = Path(a.data_root), Path(a.artifact_root)
    for root in (data, artifacts):
        if not root.is_absolute() or '..' in root.parts or not root.is_relative_to('/workspace') or root.is_relative_to(ROOT):
            raise ValueError('Output roots must be independent scratch paths below /workspace')
        root.mkdir(parents=True, exist_ok=True)
    raw_root = data/'clean_traces/raw/production'/a.model
    raw_root.mkdir(parents=True, exist_ok=True)
    manifest = {'run_id':a.run_id,'model':a.model,'source_code_commit':commit,
        'configuration_hash':identity,'original_input_configuration_hash':configuration_hash(base),
        'selected_candidates_hash':content_hash(candidates),'requested':len(candidates),
        'backend':BACKEND,'generation_settings':GENERATION_SETTINGS,
        'environment_packages':packages,'environment_inventory_hash':content_hash(packages),
        'batch_size':a.batch_size,'deadline':a.deadline,'dataset_stage':'raw_generation'}
    frozen = artifacts/('run_manifest_'+a.model+'.json')
    if frozen.exists() and read_json(frozen) != manifest:
        raise ValueError('Immutable run manifest differs; use the original source/runtime or a new run root')
    atomic_write_json(frozen, manifest)
    atomic_write_json(artifacts/'configuration.json', config)
    atomic_write_json(artifacts/'candidate_manifest.json', selected)
    interrupted = False
    def request_pause(*unused):
        nonlocal interrupted
        interrupted = True
    signal.signal(signal.SIGTERM, request_pause)
    signal.signal(signal.SIGINT, request_pause)
    def stop():
        return interrupted or (data/'STOP_REQUESTED').exists() or datetime.now(timezone.utc) >= deadline
    store = DriveStore(drive_service(os.environ['GOOGLE_OAUTH_TOKEN_FILE']),
        os.environ['GOOGLE_DRIVE_FOLDER_ID'],state_root=artifacts/'state/drive')
    store.create_project_folder()
    service = ArtifactService(Path('/workspace'), artifacts/'state/artifact_registry.json', store)
    provenance = {'experiment_id':identity,'configuration_hash':identity,'source_code_commit':commit,
        'dataset_revision':content_hash(config['datasets']),'model_revision':config['models'][a.model]['revision'],
        'model':a.model,'seeds':[42],'branch':'shared','run_id':a.run_id,
        'stage':'collect_clean','producer':'vllm_model_generation_worker'}
    generator = VLLMCleanGenerator(config['models'],cache_root=a.cache_root,batch_size=a.batch_size)
    progress = artifacts/'progress.json'
    with file_lock(data/('worker_'+a.model+'.lock')):
        for path in (frozen, artifacts/'configuration.json', artifacts/'candidate_manifest.json'):
            service.register(path,provenance)
        if service.sync_artifacts(workers=8)['status'] != 'VERIFIED':
            raise RuntimeError('Run identity and prior results must reach Drive before GPU work')
        completed = 0
        print(json.dumps({'event':'RUN_READY','model':a.model,'configuration_hash':identity,'batch_size':a.batch_size}),flush=True)
        for start in range(0,len(candidates),a.batch_size):
            if stop():
                break
            batch = candidates[start:start+a.batch_size]
            pending, paths = [], []
            for c in batch:
                path = raw_root/(content_hash(c['example_id'])+'.json')
                paths.append(path)
                if path.exists():
                    r = read_json(path)
                    if (r['candidate_hash'] != content_hash(c) or r['config_hash'] != identity
                            or r['raw_hash'] != content_hash(r['raw_generation'])
                            or r['runtime']['generation_backend'] != 'vllm_seeded_batch_v1'):
                        raise ValueError('Existing response differs from this vLLM run identity')
                else:
                    pending.append(c)
            begin = time.perf_counter()
            # Once a batch begins, finish and save it even if a pause arrives.
            # No next batch starts until every raw record has a verified Drive copy.
            for c, raw in zip(pending, generator.generate_many(a.model,pending)):
                record = {**raw,'example_id':c['example_id'],'model_id':a.model,
                    'candidate_hash':content_hash(c),'config_hash':identity,
                    'raw_hash':content_hash(raw['raw_generation']),'producer_commit':commit,
                    'producer_configuration_hash':identity,'environment_inventory_hash':content_hash(packages),
                    'run_id':a.run_id,'original_input_configuration_hash':configuration_hash(base)}
                atomic_write_json(raw_root/(content_hash(c['example_id'])+'.json'),record)
            generated_seconds = time.perf_counter()-begin
            for path in paths:
                service.register(path,provenance)
            sync_begin = time.perf_counter()
            if service.sync_artifacts(workers=8)['status'] != 'VERIFIED':
                raise RuntimeError('Whole batch Drive replication incomplete')
            completed += len(batch)
            state = {'model':a.model,'completed':completed,'requested':len(candidates),
                'drive_verified':completed,'configuration_hash':identity,'source_code_commit':commit,
                'run_id':a.run_id,'backend':'vllm','batch_size':a.batch_size,
                'last_batch_generation_seconds':generated_seconds,
                'last_batch_drive_seconds':time.perf_counter()-sync_begin,
                'updated_at':datetime.now(timezone.utc).isoformat()}
            atomic_write_json(progress,state)
            print(json.dumps({'event':'BATCH_DRIVE_VERIFIED',**state}),flush=True)
        print(json.dumps({'event':'COMPLETE' if completed==len(candidates) else 'PAUSED',
            'model':a.model,'completed':completed,'raw_generation_only':True}),flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'event':'FAILED','error_type':type(error).__name__}),flush=True)
        raise SystemExit(1)
