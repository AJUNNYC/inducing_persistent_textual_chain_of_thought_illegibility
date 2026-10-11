#!/usr/bin/env python3
"""Score frozen Pod raw bundles in the existing validated cloud Docker sandbox."""
import argparse
from datetime import datetime,timezone
import json,os
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from coordinate_vllm_sft_preparation import Pod
from experiment_overview.experiment_overview import Experiment
from operations.workflow import Stage
from shared_procedures.configuration import validate_configuration
from shared_procedures.identity import content_hash,file_hash
from shared_procedures.storage import read_json,atomic_write_json,artifact_hook,file_lock
from evaluation.shared_model_evaluation.model_evaluation import evaluate_model
from evaluation.shared_model_evaluation.item_evaluation.native_output.vllm_generation import VLLMEvaluationGenerator,BACKEND
from evaluation.shared_model_evaluation.item_evaluation.native_output.output_generation import question_id
from data_generation_collection.evaluation_data.protocol.evaluation_protocol import validate_protocol
from data_generation_collection.evaluation_data.protocol.analysis_plan.baseline_repeatability import baseline_repeatability


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--protocol',required=True)
    p.add_argument('--remote-artifact-root',required=True)
    p.add_argument('--artifact-root',default='/workspace/artifacts/baseline_vllm_cloud_0941ce5')
    p.add_argument('--credentials',default='/workspace/.secrets/runpod_jupyter.json')
    p.add_argument('--adapter',default='/workspace/scratch/runpod_control.py')
    p.add_argument('--controller-python',default='/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python')
    a=p.parse_args()
    os.environ['ARTIFACT_ROOT']=a.artifact_root
    os.environ['GOOGLE_OAUTH_TOKEN_FILE']='/workspace/.secrets/google_token.json'
    os.environ['GOOGLE_DRIVE_FOLDER_ID']='176a8Tm6l4WhfakY_0nCOo4ICeT7KPSmx'
    experiment=Experiment(validate_configuration(read_json(a.config),runtime=True),config_path=a.config)
    protocol=read_json(a.protocol);validate_protocol(protocol,allow_draft=True)
    sandbox=experiment.sandbox()
    if sandbox.identity!=protocol['specifications']['sandbox']['identity'] or sandbox.available()['status']!='passed':
        raise RuntimeError('Cloud sandbox differs from frozen validated executor')
    pod=Pod(a.credentials,a.adapter,a.controller_python)
    def remote(path,optional=False):
        try:return json.loads(pod.read(path))
        except __import__('requests').HTTPError as error:
            if optional and error.response.status_code==404:return None
            raise
    service=experiment.artifact_service();current_seed=None;current_model=None
    status_path=experiment.artifacts/'coordinator_status.json'
    completed_passes=0
    def status(kind,count=0,**extras):
        value={'status':kind,'model':current_model,'seed':current_seed,'completed_in_pass':count,
            'completed':completed_passes*1233+count,'total':7398,'judging':'deferred',
            'observed_at':datetime.now(timezone.utc).isoformat(),'source_commit':experiment.code,**extras}
        atomic_write_json(status_path,value)
        print(json.dumps(value),flush=True)
    def publish(path,kind):
        provenance=experiment.provenance(metadata={'kind':kind})
        provenance.update(producer='cloud_baseline_capability_scorer',seeds=[current_seed] if current_seed else [42,43,44])
        if not service.list_artifacts(local_path=str(path.resolve()),sha256=file_hash(path)):
            service.register(path,provenance)
        if service.sync_artifacts(workers=8)['status']!='VERIFIED':raise RuntimeError('Scored output Drive barrier failed')
    output=experiment.artifacts/'evaluation';output.mkdir(parents=True,exist_ok=True)
    with file_lock(experiment.artifacts/'cloud_scoring.lock'):
        try:
            atomic_write_json(experiment.artifacts/'measurement_protocol.json',protocol)
            publish(experiment.artifacts/'measurement_protocol.json','measurement_manifest')
            for current_model in ('7b','14b'):
                experiment._current_stage=Stage('baseline_'+current_model,'baseline',(),model=current_model)
                for current_seed in (42,43,44):
                    status('WAITING_FOR_RAW')
                    state=None
                    for start in range(0,len(protocol['items']),BACKEND['batch_size']):
                        batch=protocol['items'][start:start+BACKEND['batch_size']];end=start+len(batch)
                        name=f'{current_model}_seed_{current_seed}_{end:04d}.json'
                        path=output/'generation_batches'/name
                        if path.exists():bundle=read_json(path)
                        else:
                            while True:
                                bundle=remote(a.remote_artifact_root+'/evaluation/generation_batches/'+name,optional=True)
                                if bundle is not None:break
                                producer=remote(a.remote_artifact_root+'/raw_producer_status.json',optional=True)
                                if producer and producer['status'] in ('FAILED','PAUSED'):
                                    status(producer['status'],reason='raw_producer_'+producer['status'].lower());return 1
                                status('WAITING_FOR_RAW',raw_producer=producer)
                                time.sleep(15)
                            atomic_write_json(path,bundle)
                        if (bundle.get('bundle_hash')!=content_hash({k:v for k,v in bundle.items() if k!='bundle_hash'}) or
                            bundle['protocol_hash']!=protocol['protocol_hash'] or bundle['seed']!=current_seed or
                            bundle['configuration_hash']!=experiment.hash or bundle['question_ids']!=[question_id(i) for i in batch] or
                            len(bundle['raw_records'])!=len(batch)):
                            raise ValueError('Raw handoff identity/integrity changed')
                        if state is not None and state!=bundle['state']:raise ValueError('State changed within pass')
                        state=bundle['state']
                        if state['state_id']!='base_'+current_model or state['configuration_hash']!=experiment.hash:
                            raise ValueError('Untouched baseline state changed')
                        generator=VLLMEvaluationGenerator(output)
                        for item,raw in zip(batch,bundle['raw_records']):
                            if (raw['identity']!=generator.identity(state,item,current_seed) or raw.get('simulation') is not False or
                                raw['record_hash']!=content_hash({k:v for k,v in raw.items() if k!='record_hash'}) or
                                raw['request_hash']!=content_hash(raw['identity']) or raw['raw_hash']!=content_hash(raw['raw_output'])):
                                raise ValueError('Raw cache is not exact real frozen generation')
                            destination=output/'raw'/state['state_id']/(raw['request_hash']+'.json')
                            if destination.exists() and read_json(destination)!=raw:raise ValueError('Raw bytes changed')
                            if not destination.exists():atomic_write_json(destination,raw)
                    destination=output/f'base_{current_model}_seed_{current_seed}.json'
                    generator=VLLMEvaluationGenerator(output)
                    last=0
                    def hook(path,metadata):
                        nonlocal last
                        if metadata.get('kind')=='evaluation_item':
                            last+=1
                            if last%10==0:status('SCORING',last)
                    def barrier(records):
                        first=((len(records)-1)//BACKEND['batch_size'])*BACKEND['batch_size']
                        bundle={'state':state,'seed':current_seed,'protocol_hash':protocol['protocol_hash'],
                            'configuration_hash':experiment.hash,'source_commit':experiment.code,'records':records[first:]}
                        bundle['bundle_hash']=content_hash(bundle)
                        path=output/'batches'/f'{current_model}_seed_{current_seed}_{len(records):04d}.json'
                        atomic_write_json(path,bundle);publish(path,'baseline_scored_bundle')
                        status('SCORING',len(records),drive_replication='VERIFIED')
                    status('SCORING')
                    with artifact_hook(hook):
                        result=evaluate_model(state,protocol,generator,sandbox=sandbox,cache_root=output,
                            seed=current_seed,defer_judging=True,batch_completed=barrier)
                    atomic_write_json(destination,result);publish(destination,'baseline_capability_package')
                    if result['status']!='complete':raise RuntimeError('Capability scoring incomplete')
                    completed_passes+=1
            passes={m:{s:read_json(output/f'base_{m}_seed_{s}.json') for s in (42,43,44)} for m in ('7b','14b')}
            report=baseline_repeatability(passes,protocol['items'])
            path=experiment.artifacts/'preflight'/'baseline_repeatability.json'
            atomic_write_json(path,report);publish(path,'baseline_repeatability')
            status('COMPLETE' if report['ready_for_training'] else 'GATE_FAILED',
                repeatability_gate_passed=report['ready_for_training'],drive_replication='VERIFIED')
            publish(status_path,'baseline_status')
            return 0 if report['ready_for_training'] else 1
        except Exception as error:
            status('FAILED',last if 'last' in locals() else 0,error_type=type(error).__name__)
            publish(status_path,'baseline_status');raise

if __name__=='__main__':raise SystemExit(main())
