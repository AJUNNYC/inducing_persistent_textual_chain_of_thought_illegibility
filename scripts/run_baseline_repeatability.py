#!/usr/bin/env python3
"""Run six unchanged native baseline passes with per-artifact Drive replication.

Uses the committed evaluator, native decoder, scorers and judge client. A UTC
limit or SIGTERM pauses before the next question; completed outputs stay usable.
This worker never starts training or stops Pod billing.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from experiment_overview.experiment_overview import Experiment
from operations.workflow import Stage
from shared_procedures.configuration import load_configuration, validate_configuration
from shared_procedures.identity import content_hash, file_hash
from shared_procedures.storage import artifact_hook, atomic_write_json, read_json, notify_artifact, file_lock
from evaluation.shared_model_evaluation.model_evaluation import evaluate_model
from evaluation.shared_model_evaluation.item_evaluation.native_output.output_generation import NativeGenerator
from data_generation_collection.evaluation_data.protocol.evaluation_protocol import build_evaluation_protocol
from data_generation_collection.evaluation_data.protocol.analysis_plan.baseline_repeatability import baseline_repeatability


class BaselinePaused(RuntimeError):
    pass


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--evaluation-package',required=True)
    parser.add_argument('--mbpp-evidence',required=True)
    parser.add_argument('--provider-env',required=True)
    parser.add_argument('--deadline',required=True)
    args=parser.parse_args()
    # This protected file is operational only and never registered as an artifact.
    for name,value in read_json(args.provider_env).items():
        if name not in ('MODEL_OPENAI_API_KEY','ANTHROPIC_API_KEY'):
            raise ValueError('Unexpected provider binding')
        os.environ[name]=value
    config=load_configuration(args.config,require_runtime=True)
    experiment=Experiment(config,config_path=args.config)
    validate_configuration(experiment.config,runtime=True)
    if experiment.config['runtime']['implementation_revision'] != experiment.code:
        raise ValueError('Frozen baseline implementation does not match executed source')
    deadline=datetime.fromisoformat(args.deadline.replace('Z','+00:00'))
    if deadline.utcoffset() is None or deadline.utcoffset().total_seconds()!=0:
        raise ValueError('Baseline deadline must use UTC')
    stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    package=read_json(args.evaluation_package)
    panel=package['capability_panel']
    protocol=build_evaluation_protocol(panel,package['legibility_subset'],
                configuration_hash=experiment.hash,preflight_reports={'mbpp_sandbox':read_json(args.mbpp_evidence)})
    output=experiment.artifacts/'evaluation'
    output.mkdir(parents=True,exist_ok=True)
    status_path=experiment.artifacts/'baseline_status.json'
    current_model=None;current_seed=None;completed=0
    service=experiment.artifact_service()
    def publish(path,metadata):
        path=Path(path)
        provenance=experiment.provenance(metadata=metadata)
        provenance.update(seeds=[current_seed] if current_seed is not None else [42,43,44],
                          producer='native_baseline_repeatability_worker')
        if not service.list_artifacts(local_path=str(path.resolve()),sha256=file_hash(path)):
            service.register(path,provenance)
        if service.sync_artifacts(workers=4)['status'] != 'VERIFIED':
            raise RuntimeError('Baseline artifact Drive replication failed')
        if metadata.get('kind')=='evaluation_item' and metadata.get('status')=='complete':
            nonlocal completed
            completed+=1
            atomic_write_json(status_path,{'status':'RUNNING','model':current_model,'seed':current_seed,
                'completed_items_this_process':completed,'total_items':7398,'at':datetime.now(timezone.utc).isoformat(),
                'source_commit':experiment.code,'configuration_hash':experiment.hash,'drive_replication':'VERIFIED'})
            print(json.dumps({'status':'ITEM_SAVED','model':current_model,'seed':current_seed,'completed_items_this_process':completed}),flush=True)
    native=NativeGenerator(output)
    class PausableNative:
        simulation=False
        def identity(self,*a,**kw):return native.identity(*a,**kw)
        def generate(self,*a,**kw):
            if stopped or datetime.now(timezone.utc)>=deadline:
                raise BaselinePaused('Baseline paused at a saved question boundary')
            return native.generate(*a,**kw)
    with file_lock(experiment.artifacts/'baseline_worker.lock'), artifact_hook(publish):
        try:
            atomic_write_json(status_path,{'status':'STARTING','source_commit':experiment.code,'configuration_hash':experiment.hash,
                'total_items':7398,'deadline':args.deadline,'at':datetime.now(timezone.utc).isoformat()})
            notify_artifact(status_path,{'kind':'baseline_status'})
            for current_model in ('7b','14b'):
                experiment._current_stage=Stage('baseline_'+current_model,'baseline',(),model=current_model)
                state=experiment.state(experiment._current_stage)
                for current_seed in (42,43,44):
                    destination=output/f'base_{current_model}_seed_{current_seed}.json'
                    if destination.exists():
                        previous=read_json(destination)
                        if previous.get('package_hash') != content_hash({k:v for k,v in previous.items() if k!='package_hash'}):
                            raise ValueError('Saved baseline package integrity differs')
                        if previous['state']!=state or previous['protocol_hash']!=protocol['protocol_hash'] or previous['seed']!=current_seed:
                            raise ValueError('Saved baseline package identity differs')
                        if previous['status']=='complete':continue
                    print(json.dumps({'status':'PASS_STARTING','model':current_model,'seed':current_seed}),flush=True)
                    result=evaluate_model(state,protocol,PausableNative(),experiment.judge() if current_seed==42 else None,
                                sandbox=experiment.sandbox(),cache_root=output,seed=current_seed)
                    atomic_write_json(destination,result)
                    notify_artifact(destination,{'kind':'baseline_package'})
                    if result['status']!='complete':
                        if stopped or datetime.now(timezone.utc)>=deadline:
                            atomic_write_json(status_path,{'status':'PAUSED','model':current_model,'seed':current_seed,
                                'completed_items_this_process':completed,'reason':'deadline_or_stop','at':datetime.now(timezone.utc).isoformat()})
                            notify_artifact(status_path,{'kind':'baseline_status'})
                            return 0
                        raise RuntimeError('Baseline evaluation incomplete; original failure retained')
            passes={m:{s:read_json(output/f'base_{m}_seed_{s}.json') for s in (42,43,44)} for m in ('7b','14b')}
            report=baseline_repeatability(passes,panel)
            destination=experiment.artifacts/'preflight'/'baseline_repeatability.json'
            atomic_write_json(destination,report);notify_artifact(destination,{'kind':'baseline_repeatability'})
            atomic_write_json(status_path,{'status':'COMPLETE','result':report['status'],'ready_for_training':report['ready_for_training'],
                'completed_items':7398,'at':datetime.now(timezone.utc).isoformat()})
            notify_artifact(status_path,{'kind':'baseline_status'})
            return 0 if report['ready_for_training'] else 1
        except Exception as error:
            atomic_write_json(status_path,{'status':'FAILED','error_type':type(error).__name__,
                'model':current_model,'seed':current_seed,'completed_items_this_process':completed,
                'at':datetime.now(timezone.utc).isoformat()})
            try:notify_artifact(status_path,{'kind':'baseline_status'})
            except Exception:pass
            raise

if __name__=='__main__':
    raise SystemExit(main())
