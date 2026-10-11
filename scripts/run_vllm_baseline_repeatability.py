#!/usr/bin/env python3
"""One-GPU sequential baseline diagnostics; whole-batch verified Drive barriers.

Judge calls are explicitly deferred, not counted as successful. Saved seed-42
raw outputs can later be judged by the full evaluator without regeneration.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import signal, subprocess, sys, time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from experiment_overview.experiment_overview import Experiment
from operations.workflow import Stage
from shared_procedures.configuration import validate_configuration
from shared_procedures.identity import content_hash, file_hash
from shared_procedures.storage import read_json, atomic_write_json, artifact_hook, file_lock
from evaluation.shared_model_evaluation.model_evaluation import evaluate_model
from evaluation.shared_model_evaluation.item_evaluation.native_output.vllm_generation import VLLMEvaluationGenerator, BACKEND
from data_generation_collection.evaluation_data.protocol.evaluation_protocol import build_evaluation_protocol
from data_generation_collection.evaluation_data.protocol.analysis_plan.baseline_repeatability import baseline_repeatability

class Paused(RuntimeError): pass

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('config','evaluation-package','mbpp-evidence','deadline'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--model',choices=('7b','14b'))
    args=parser.parse_args()
    config=validate_configuration(read_json(args.config),runtime=True)
    experiment=Experiment(config,config_path=args.config)
    if config['runtime'].get('evaluation_backend')!='vllm' or config['runtime']['implementation_revision']!=experiment.code:
        raise ValueError('Executed source/backend differs from frozen run configuration')
    deadline=datetime.fromisoformat(args.deadline.replace('Z','+00:00'))
    if deadline.utcoffset() is None or deadline.utcoffset().total_seconds()!=0:
        raise ValueError('Deadline must use UTC')
    output=experiment.artifacts/'evaluation'
    output.mkdir(parents=True,exist_ok=True)
    package=read_json(args.evaluation_package)
    protocol=build_evaluation_protocol(package['capability_panel'],package['legibility_subset'],
        preflight_reports={'mbpp_sandbox':read_json(args.mbpp_evidence)},generation_backend='vllm')
    service=experiment.artifact_service()
    current_seed=None
    def publish(path,kind):
        provenance=experiment.provenance(metadata={'kind':kind})
        provenance.update(producer='vllm_baseline_capability_worker',seeds=[current_seed] if current_seed else [42,43,44])
        if not service.list_artifacts(local_path=str(path.resolve()),sha256=file_hash(path)):
            service.register(path,provenance)
        if service.sync_artifacts(workers=8)['status']!='VERIFIED':
            raise RuntimeError('Drive integrity barrier failed')
    if not args.model:
        import importlib.metadata
        inventory={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']}
        runtime={'packages':inventory,'package_inventory_hash':content_hash(inventory),
            'backend':BACKEND,'evaluation_lock_hash':file_hash(ROOT/'requirements-vllm.txt')}
        atomic_write_json(experiment.artifacts/'evaluation_runtime.json',runtime)
        publish(experiment.artifacts/'evaluation_runtime.json','evaluation_runtime')
        # Separate processes release each entire engine before loading the next.
        atomic_write_json(experiment.artifacts/'measurement_protocol.json',protocol)
        atomic_write_json(experiment.artifacts/'configuration.json',config)
        for path in (experiment.artifacts/'measurement_protocol.json',experiment.artifacts/'configuration.json'):
            publish(path,'measurement_manifest')
        for model in ('7b','14b'):
            if datetime.now(timezone.utc)>=deadline:return 0
            command=[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--model',model]
            outcome=subprocess.run(command)
            if outcome.returncode:return outcome.returncode
            if not all((output/f'base_{model}_seed_{s}.json').exists() and
                read_json(output/f'base_{model}_seed_{s}.json')['status']=='complete' for s in (42,43,44)):
                return 0
        passes={m:{s:read_json(output/f'base_{m}_seed_{s}.json') for s in (42,43,44)} for m in ('7b','14b')}
        report=baseline_repeatability(passes,package['capability_panel'])
        path=experiment.artifacts/'preflight'/'baseline_repeatability.json'
        atomic_write_json(path,report);publish(path,'baseline_repeatability')
        print(__import__('json').dumps({'status':report['status'],'repeatability_gate':report['ready_for_training'],
            'full_evaluation':'pending_legibility_judgments'}),flush=True)
        return 0 if report['ready_for_training'] else 1
    model=args.model
    experiment._current_stage=Stage('baseline_'+model,'baseline',(),model=model)
    state=experiment.state(experiment._current_stage)
    stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    status_path=experiment.artifacts/f'baseline_{model}_status.json'
    total=0;started=time.monotonic();batch_started=started;timings=[]
    class Generator(VLLMEvaluationGenerator):
        def generate_batch(self,*a,**kw):
            nonlocal batch_started
            if stopped or datetime.now(timezone.utc)>=deadline:raise Paused('Saved whole-batch boundary')
            batch_started=time.monotonic()
            return super().generate_batch(*a,**kw)
    generator=Generator(output)
    # After a Drive restore, bundles reconstruct the identical local raw/item
    # caches. No native output adoption and no new sampling of saved answers.
    for saved in sorted((output/'batches').glob(f'{model}_seed_*.json')):
        bundle=read_json(saved)
        if (bundle.get('bundle_hash')!=content_hash({k:v for k,v in bundle.items() if k!='bundle_hash'})
            or bundle['state']!=state or bundle['protocol_hash']!=protocol['protocol_hash']
            or bundle['backend']!=BACKEND or bundle['configuration_hash']!=experiment.hash):
            raise ValueError('Restored batch identity/integrity mismatch')
        for record in bundle['records']:
            raw=record.get('raw')
            if raw:
                path=output/'raw'/state['state_id']/(raw['request_hash']+'.json')
                if not path.exists():atomic_write_json(path,raw)
            path=output/'capability'/state['state_id']/(content_hash(record['identity'])+'.json')
            if not path.exists():atomic_write_json(path,record)
    # Individual files are private scratch derivatives. Each immutable bundle
    # contains every complete scored item and its full raw token/text record.
    def barrier(records):
        nonlocal total
        batch=records[-min(BACKEND['batch_size'],len(records)-total):]
        if not batch:return
        path=output/'batches'/f'{model}_seed_{current_seed}_{len(records):04d}.json'
        bundle={'state':state,'seed':current_seed,'protocol_hash':protocol['protocol_hash'],
            'configuration_hash':experiment.hash,'source_commit':experiment.code,'backend':BACKEND,'records':batch}
        bundle['bundle_hash']=content_hash(bundle)
        atomic_write_json(path,bundle)
        generation_and_scoring=time.monotonic()-batch_started
        upload_started=time.monotonic();publish(path,'baseline_batch_bundle')
        total=len(records)
        timings.append({'items':len(batch),'generation_and_scoring_seconds':generation_and_scoring,
            'drive_seconds':time.monotonic()-upload_started,'generated_tokens':sum((r.get('raw') or {}).get('generated_tokens',0) for r in batch)})
        status={'status':'RUNNING','model':model,'seed':current_seed,'completed_items_in_pass':total,
            'expected_items_in_pass':len(protocol['items']),'drive_replication':'VERIFIED','timings':timings,
            'source_commit':experiment.code,'protocol_hash':protocol['protocol_hash'],'judging':'deferred',
            'at':datetime.now(timezone.utc).isoformat()}
        atomic_write_json(status_path,status)
        print(__import__('json').dumps({k:status[k] for k in ('status','model','seed','completed_items_in_pass','drive_replication')}
            | {'last_batch':timings[-1]}),flush=True)
    try:
        with file_lock(experiment.artifacts/f'baseline_{model}.lock'),artifact_hook(lambda *_:None):
            for current_seed in (42,43,44):
                destination=output/f'base_{model}_seed_{current_seed}.json'
                if destination.exists():
                    previous=read_json(destination)
                    if (previous['state']!=state or previous['protocol_hash']!=protocol['protocol_hash'] or
                        previous.get('package_hash')!=content_hash({k:v for k,v in previous.items() if k!='package_hash'})):
                        raise ValueError('Saved pass identity/integrity changed')
                    if previous['status']=='complete':continue
                total=0
                result=evaluate_model(state,protocol,generator,sandbox=experiment.sandbox(),cache_root=output,
                    seed=current_seed,defer_judging=True,batch_completed=barrier)
                atomic_write_json(destination,result);publish(destination,'baseline_capability_package')
                if result['status']!='complete':raise RuntimeError('Capability execution incomplete; original evidence retained')
        atomic_write_json(status_path,{'status':'COMPLETE','model':model,'items':3*len(protocol['items']),
            'judging':'deferred','elapsed_seconds':time.monotonic()-started,'drive_replication':'VERIFIED'})
        publish(status_path,'baseline_status')
        return 0
    except Paused:
        atomic_write_json(status_path,{'status':'PAUSED','model':model,'seed':current_seed,'completed_items_in_pass':total})
        publish(status_path,'baseline_status');return 0
    except Exception as error:
        atomic_write_json(status_path,{'status':'FAILED','model':model,'seed':current_seed,
            'completed_items_in_pass':total,'error_type':type(error).__name__})
        publish(status_path,'baseline_status')
        raise

if __name__=='__main__':raise SystemExit(main())
