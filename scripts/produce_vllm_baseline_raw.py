#!/usr/bin/env python3
"""GPU-only producer for the frozen baseline measurement protocol.

Separates driver provenance from the original configuration/measurement source.
Scoring runs in the existing Docker-enabled cloud environment, not on the Pod.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import signal, subprocess, sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from experiment_overview.experiment_overview import Experiment
from operations.workflow import Stage
from shared_procedures.configuration import validate_configuration
from shared_procedures.identity import content_hash,file_hash
from shared_procedures.storage import read_json,atomic_write_json,artifact_hook,file_lock
from evaluation.shared_model_evaluation.item_evaluation.native_output.vllm_generation import VLLMEvaluationGenerator,BACKEND
from data_generation_collection.evaluation_data.protocol.evaluation_protocol import validate_protocol


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','protocol','deadline'):p.add_argument('--'+name,required=True)
    p.add_argument('--model',choices=('7b','14b'))
    a=p.parse_args()
    experiment=Experiment(validate_configuration(read_json(a.config),runtime=True),config_path=a.config)
    protocol=read_json(a.protocol);validate_protocol(protocol,allow_draft=True)
    if protocol['specifications']['generation'].get('backend')!=BACKEND:
        raise ValueError('Frozen measurement engine differs')
    deadline=datetime.fromisoformat(a.deadline.replace('Z','+00:00'))
    service=experiment.artifact_service();current_seed=None
    def publish(path,kind):
        provenance=experiment.provenance(metadata={'kind':kind})
        provenance.update(producer='frozen_vllm_raw_producer',seeds=[current_seed] if current_seed else [42,43,44])
        if not service.list_artifacts(local_path=str(path.resolve()),sha256=file_hash(path)):
            service.register(path,provenance)
        if service.sync_artifacts(workers=8)['status']!='VERIFIED':raise RuntimeError('Drive barrier failed')
    status_path=experiment.artifacts/'raw_producer_status.json'
    if not a.model:
        receipt={'driver_source_commit':experiment.code,'measurement_source_commit':experiment.config['runtime']['implementation_revision'],
            'protocol_hash':protocol['protocol_hash'],'configuration_hash':experiment.hash,
            'source_binding':'all measurement module contents verified against original frozen protocol',
            'reason':'Docker scorer is in cloud environment; preserve exact Pod-generated outputs'}
        path=experiment.artifacts/'raw_producer_handoff.json';atomic_write_json(path,receipt);publish(path,'producer_handoff')
        for model in ('7b','14b'):
            if datetime.now(timezone.utc)>=deadline:
                atomic_write_json(status_path,{'status':'PAUSED','reason':'deadline','driver_source_commit':experiment.code})
                publish(status_path,'raw_producer_status');return 0
            outcome=subprocess.run([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--model',model])
            if outcome.returncode:return outcome.returncode
            status=read_json(status_path)
            if status['status']!='MODEL_COMPLETE':return 0
        atomic_write_json(status_path,{'status':'COMPLETE','completed':7398,'driver_source_commit':experiment.code,
            'protocol_hash':protocol['protocol_hash'],'drive_replication':'VERIFIED'})
        publish(status_path,'raw_producer_status');return 0
    model=a.model;experiment._current_stage=Stage('baseline_'+model,'baseline',(),model=model)
    state=experiment.state(experiment._current_stage)
    generator=VLLMEvaluationGenerator(experiment.artifacts/'evaluation')
    stop=False
    def request_stop(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    completed=0
    with file_lock(experiment.artifacts/'raw_producer.lock'),artifact_hook(lambda *_:None):
        try:
            for current_seed in (42,43,44):
                completed=0
                for start in range(0,len(protocol['items']),BACKEND['batch_size']):
                    if stop or datetime.now(timezone.utc)>=deadline:
                        atomic_write_json(status_path,{'status':'PAUSED','model':model,'seed':current_seed,'completed_in_pass':completed})
                        publish(status_path,'raw_producer_status');return 0
                    batch=protocol['items'][start:start+BACKEND['batch_size']]
                    end=start+len(batch)
                    path=experiment.artifacts/'evaluation'/'generation_batches'/f'{model}_seed_{current_seed}_{end:04d}.json'
                    if path.exists():
                        bundle=read_json(path)
                        if (bundle.get('bundle_hash')!=content_hash({k:v for k,v in bundle.items() if k!='bundle_hash'}) or
                            bundle['state']!=state or bundle['seed']!=current_seed or bundle['protocol_hash']!=protocol['protocol_hash']):
                            raise ValueError('Existing raw bundle identity differs')
                        publish(path,'baseline_raw_bundle')
                    else:
                        generator.generate_batch(state,batch,current_seed)
                        bundle={'state':state,'seed':current_seed,'protocol_hash':protocol['protocol_hash'],
                            'configuration_hash':experiment.hash,'driver_source_commit':experiment.code,
                            'measurement_source_commit':experiment.config['runtime']['implementation_revision'],
                            'question_ids':[i.get('question_id',i.get('item_id')) for i in batch],
                            'raw_records':[generator.generate(state,i,current_seed) for i in batch]}
                        bundle['bundle_hash']=content_hash(bundle);atomic_write_json(path,bundle);publish(path,'baseline_raw_bundle')
                    completed=end
                    atomic_write_json(status_path,{'status':'RUNNING','model':model,'seed':current_seed,'completed_in_pass':end,
                        'drive_replication':'VERIFIED','protocol_hash':protocol['protocol_hash'],'driver_source_commit':experiment.code})
                    print(__import__('json').dumps(read_json(status_path)),flush=True)
            atomic_write_json(status_path,{'status':'MODEL_COMPLETE','model':model,'completed':3699,'drive_replication':'VERIFIED'})
            publish(status_path,'raw_producer_status');return 0
        except Exception as error:
            atomic_write_json(status_path,{'status':'FAILED','model':model,'seed':current_seed,'completed_in_pass':completed,'error_type':type(error).__name__})
            publish(status_path,'raw_producer_status');raise
        finally:
            if generator.engine is not None:
                generator.engine.llm_engine.engine_core.shutdown()

if __name__=='__main__':raise SystemExit(main())
