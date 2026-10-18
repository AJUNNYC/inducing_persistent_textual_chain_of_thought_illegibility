#!/usr/bin/env python3
"""Run frozen GPQA after a Drive-verified continuation, preserving cached answers.

This dispatcher uses separate training/evaluation Python environments and never
changes the frozen evaluator, its token allowance, scoring, or inference settings.
A PAUSE_REQUESTED file prevents automatic restart during a GPU handoff.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

FROZEN_REVISION = '5e6be1b0848e0e88725323591baddef37d69d8ae'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)); temporary.chmod(0o600)
    temporary.replace(path)


def training_ready(status):
    return status.get('phase') == 'COMPLETE_DRIVE_VERIFIED' and status.get('completed_updates') == 625


def check_records(records, baseline_hash):
    if len(records) != 198:
        raise ValueError('All 198 original baseline responses are required')
    names = set()
    for record in records:
        name = Path(record['local_path']).name
        if (f'/{baseline_hash}/raw/base_14b/' not in record['local_path']
                or not re.fullmatch(r'[0-9a-f]{64}\.json', name)
                or not re.fullmatch(r'[0-9a-f]{64}', record['sha256'])
                or record.get('sync_status') != 'VERIFIED' or name in names):
            raise ValueError('Baseline inventory identity or integrity differs')
        names.add(name)


def frozen_api(config):
    source = Path(config['eval_source']).resolve()
    for relative, expected in config['frozen_files'].items():
        if digest(source / relative) != expected:
            raise ValueError('Frozen evaluation implementation changed')
    sys.path.insert(0, str(source / 'src'))
    from evaluation import supplemental_gpqa as gpqa
    from operations.artifacts import ArtifactService
    from operations.drive import DriveStore, drive_service
    from shared_procedures import storage
    if not Path(gpqa.__file__).resolve().is_relative_to(source):
        raise ValueError('Wrong evaluator imported')
    spec = importlib.util.spec_from_file_location('frozen_gpqa_campaign', source/'scripts/run_gpqa_sft_campaign.py')
    campaign = importlib.util.module_from_spec(spec); spec.loader.exec_module(campaign)
    return gpqa, ArtifactService, DriveStore, drive_service, storage, campaign


def service_for(config, api):
    _, ArtifactService, DriveStore, drive_service, _, _ = api
    output = Path(config['output'])
    return ArtifactService(output, output/'drive_registry.json',
        DriveStore(drive_service(os.environ['GOOGLE_OAUTH_TOKEN_FILE']),
                   os.environ['GOOGLE_DRIVE_FOLDER_ID']), drive_recovery=True)


def provenance(protocol, revision=FROZEN_REVISION):
    key = protocol['protocol_hash']
    return dict(experiment_id=key, branch='supplemental_gpqa', model='14b', run_id=key[:16],
                stage='supplemental_capability', producer='run_supplemental_gpqa',
                source_code_commit=revision, configuration_hash=key,
                dataset_revision=protocol['panel']['source']['revision'],
                model_revision=protocol['states'][0]['model_revision'], seeds=[42])


def prepare(config):
    api = frozen_api(config); gpqa, _, _, drive_service, _, _ = api
    baseline = read(config['baseline_protocol']); gpqa.validate_gpqa_protocol(baseline)
    if (baseline['panel']['variant'] != 'gpqa_diamond' or baseline['panel']['item_count'] != 198
            or baseline['generation'] != gpqa.generation_contract('vllm', 'long_32000')):
        raise ValueError('Original full-budget Diamond protocol required')
    records = read(config['baseline_records']); check_records(records, baseline['protocol_hash'])
    output = Path(config['output']); output.mkdir(parents=True, exist_ok=True)
    local = threading.local()
    def fetch(record):
        from googleapiclient.http import MediaIoBaseDownload
        destination = output/baseline['protocol_hash']/'raw/base_14b'/Path(record['local_path']).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if digest(destination) != record['sha256']:raise ValueError('Existing baseline bytes differ')
            return
        if not hasattr(local, 'service'):
            local.service = drive_service(os.environ['GOOGLE_OAUTH_TOKEN_FILE'])
        temporary = destination.with_suffix('.download')
        with temporary.open('wb') as stream:
            download = MediaIoBaseDownload(stream, local.service.files().get_media(fileId=record['drive_file_id']), chunksize=1024*1024)
            done = False
            while not done:_, done = download.next_chunk(num_retries=2)
        if temporary.stat().st_size != record['size'] or digest(temporary) != record['sha256']:
            raise ValueError('Drive baseline checksum differs')
        temporary.chmod(0o600); temporary.replace(destination)
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(fetch, records))
    generator = gpqa.GPQAVLLMGenerator(output/baseline['protocol_hash'], baseline)
    expected = {gpqa.content_hash(generator.identity(baseline['states'][0], item))+'.json'
                for item in baseline['panel']['items']}
    if expected != {Path(r['local_path']).name for r in records}:raise ValueError('Baseline question coverage differs')
    service = service_for(config, api)
    for item in baseline['panel']['items']:
        raw = generator._long_raw(baseline['states'][0], item)
        path = output/baseline['protocol_hash']/'raw/base_14b'/(raw['request_hash']+'.json')
        service.register(path, {**provenance(baseline), 'kind':'evaluation_raw', 'state_id':'base_14b',
                               'request_hash':raw['request_hash'], 'raw_hash':raw['raw_hash'], 'simulation':False})
    if service.sync_artifacts(workers=4)['status'] != 'VERIFIED':raise RuntimeError('Baseline upload incomplete')
    service.export_manifest()
    write(output/'handoff/baseline_ready.json', {'status':'VERIFIED','questions':198,
          'baseline_protocol_hash':baseline['protocol_hash'],'new_baseline_inference':False})


def evaluate(config):
    api = frozen_api(config); gpqa, _, _, _, _, campaign = api
    output=Path(config['output']); baseline=read(config['baseline_protocol'])
    endpoint=read(config['training_endpoints'])
    if endpoint.get('status') != 'COMPLETE' or endpoint.get('drive_status') != 'VERIFIED':
        raise ValueError('Training endpoints are not verified')
    matches=[e for e in endpoint['adapters'] if e['level']==config['level']]
    if len(matches)!=1:raise ValueError('Exactly one selected adapter is required')
    adapter=Path(matches[0]['final_adapter'])
    states=gpqa.comparison_states(adapter,config['model_cache'])
    if states[1]['model']!='14b' or states[1]['family']!='rot13' or states[1]['level']!=config['level']:
        raise ValueError('Adapter condition differs')
    if states[1]['adapter_manifest_hash']!=matches[0]['adapter_manifest_hash']:
        raise ValueError('Adapter differs from verified training endpoint')
    protocol=gpqa.freeze_gpqa_protocol(baseline['panel'],states,backend='vllm',
                                       generation_profile='long_32000',baseline_protocol=baseline)
    protocol_path=output/'handoff/protocol.json';gpqa.publish_immutable(protocol_path,protocol)
    # Hardware observations are separate from the fixed scientific identity.
    segment=output/'handoff'/('hardware_'+str(time.time_ns())+'.json')
    observed=subprocess.run(['nvidia-smi','--query-gpu=name,uuid,memory.total,driver_version','--format=csv,noheader'],capture_output=True,text=True,check=True)
    write(segment,{'hardware':observed.stdout.strip(),'protocol_hash':protocol['protocol_hash'],
                   'evaluation_source':FROZEN_REVISION,'time_unix':time.time()})
    service=service_for(config,api)
    for path in (protocol_path, segment, Path(config['baseline_protocol']), Path(config['baseline_records']),
                 Path(config['config_path']), Path(__file__).resolve(), Path(config['frozen_archive'])):
        # External preparation files are copied into the output namespace for recovery.
        target=output/'handoff'/path.name
        if target!=path:
            if target.exists() and target.read_bytes()!=path.read_bytes():raise ValueError('Handoff bytes changed')
            if not target.exists():target.write_bytes(path.read_bytes())
        service.register(target,{**provenance(protocol,config['operations_commit']),'producer':'watch_gpqa_continuation','stage':'post_sft_gpqa_handoff'})
    if service.sync_artifacts()['status']!='VERIFIED':raise RuntimeError('Evaluation handoff upload incomplete')
    command=[sys.executable,str(Path(config['eval_source'])/'scripts/run_supplemental_gpqa.py'),
             'run','--protocol',str(protocol_path),'--output-root',str(output),'--save-to-drive']
    subprocess.run(command,check=True,cwd=config['eval_source'])
    receipt=campaign.completion_barrier(protocol,output,save_to_drive=True,source_code_commit=FROZEN_REVISION)
    # Publish the native completion receipt as well as raw answers, scores and report.
    path=output/protocol['protocol_hash']/'completion_receipt.json'
    service=service_for(config,api)
    service.register(path,{**provenance(protocol),'kind':'evaluation_completion_receipt'})
    if service.sync_artifacts()['status']!='VERIFIED':raise RuntimeError('Completion receipt upload incomplete')
    service.export_manifest()
    write(output/'handoff/completion_verified.json',receipt)


def supervise(config, config_path):
    output=Path(config['output']);output.mkdir(parents=True,exist_ok=True)
    lock=(output/'followup.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status_path=output/'followup_status.json'; pause=output/'PAUSE_REQUESTED'
    attempts=read(status_path).get('attempt',0) if status_path.exists() else 0
    stopping={'requested':False}
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stopping.update(requested=True))
    def status(phase,**extra):write(status_path,{'phase':phase,'attempt':attempts,'updated_unix':time.time(),**extra})
    def stopped():return stopping['requested'] or pause.exists()
    status('WAITING_FOR_VERIFIED_TRAINING_AND_EVAL_ENVIRONMENT')
    while not (training_ready(read(config['training_watcher_status'])) and Path(config['eval_ready_marker']).exists()):
        if stopped():status('PAUSED');return
        time.sleep(15)
    # Reuse the published training campaign's idempotent live-Drive completion gate.
    train_env={**os.environ,**config['training_environment']}
    with (output/'training_completion_gate.log').open('ab') as log:
        subprocess.run(config['training_command'],env=train_env,stdout=log,stderr=subprocess.STDOUT,check=True)
    eval_env={**os.environ,'SOURCE_CODE_COMMIT':FROZEN_REVISION,'PYTHONPATH':str(Path(config['eval_source'])/'src')}
    for phase in ('prepare','evaluate'):
        while attempts<3:
            if stopped():status('PAUSED');return
            status(phase.upper())
            command=[config['eval_python'],str(Path(__file__).resolve()),'--config',str(config_path),'--phase',phase]
            with (output/(phase+'.log')).open('ab') as log:
                child=subprocess.Popen(command,env=eval_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                while child.poll() is None:
                    if stopped():
                        os.killpg(child.pid,signal.SIGTERM)
                        try:child.wait(timeout=20)
                        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                        status('PAUSED');return
                    time.sleep(5)
            if child.returncode==0:break
            attempts+=1;status('RETRY_PENDING',failed_phase=phase,exit_code=child.returncode)
            time.sleep(60)
        else:status('FAILED_RETRY_LIMIT');return
    status('COMPLETE_DRIVE_VERIFIED',questions=198,level=config['level'])


def main():
    os.umask(0o077)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--phase',choices=('supervise','prepare','evaluate'),default='supervise')
    args=parser.parse_args();config=read(args.config)
    if args.phase=='supervise':supervise(config,args.config)
    elif args.phase=='prepare':prepare(config)
    else:evaluate(config)

if __name__=='__main__':main()
