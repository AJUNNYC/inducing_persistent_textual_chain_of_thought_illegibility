#!/usr/bin/env python3
"""Run a selected certified SFT queue with unchanged gates and Drive checkpoints.

Consumes the previously validated forty-condition worklist; does not rerun data
construction or GPU controls. Every fit still checks its own exact dataset bytes
and the mandatory live preflight report. SIGTERM/deadline pauses only after a
whole optimizer update and a full checkpoint verified in Google Drive.
"""
import argparse
from datetime import datetime, timezone
import gc
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from experiment_overview.experiment_overview import Experiment
from operations.workflow import Stage
from shared_procedures.configuration import validate_configuration
from shared_procedures.identity import content_hash
from shared_procedures.preflight import load_preflight_evidence, preflight_requirements, require_preflights
from shared_procedures.recovery_procedure import configure_worker_recovery
from shared_procedures.storage import artifact_hook, atomic_write_json, file_lock, read_json
from training.sft.data_handoff import validate_handoff
from training.sft.individual_fit.sft_fit import SFTFitSpec, run_sft_fit
from training.sft.sft_training import load_dataset_bundle


def certified_tasks(config, certificate, bundle, worklist, *, family, model, level=None):
    """Reuse certified condition identities, retaining the original producer."""
    validate_handoff(certificate, config)
    if (bundle.get('simulation') is not False or bundle.get('status') != 'verified'
            or bundle.get('config_hash') != certificate['producer_configuration_hash']
            or bundle.get('content_hash') != content_hash(bundle.get('datasets'))
            or bundle.get('content_hash') != certificate['bundle_hash']):
        raise ValueError('Production manifest differs from the certified corpus')
    conditions = {(row['family'],row['model_id'],row['level']):row for row in bundle['datasets']}
    expected = {(f,m,l) for f in ('rot13','kazitext') for m in ('7b','14b') for l in range(1,11)}
    if len(bundle['datasets']) != 40 or set(conditions) != expected or len(worklist) != 40:
        raise ValueError('Certified SFT matrix must contain exactly forty conditions')
    seen = set()
    selected = []
    for task in worklist:
        spec = SFTFitSpec.from_dict(task['spec'])
        key = (spec.family,spec.model_id,spec.level)
        if (key in seen or key not in conditions or task['dataset'] != conditions[key]
                or (task['family'],task['model_id'],task['level']) != key
                or task['run_id'] != spec.run_id
                or task.get('initialization') != 'untouched_base_or_exact_full_checkpoint'):
            raise ValueError('Worklist differs from the immutable certified matrix')
        validate_handoff(certificate, config, spec)
        seen.add(key)
        if spec.family == family and spec.model_id == model and (level is None or spec.level == level):
            selected.append(task)
    if seen != expected or not selected:
        raise ValueError('Requested certified SFT condition is unavailable')
    return sorted(selected, key=lambda task:task['level'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config','certificate','bundle','worklist','preflight','deadline'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--family', choices=('rot13','kazitext'), default='rot13')
    parser.add_argument('--model', choices=('7b','14b'), default='7b')
    parser.add_argument('--level', type=int, choices=range(1,11))
    parser.add_argument('--execution-benchmark', help='Hash-verified accepted target-GPU benchmark receipt.')
    parser.add_argument('--execution-profile', help='Accepted profile name in the benchmark plan.')
    parser.add_argument('--migration-checkpoint', help='Verified original full-state checkpoint; preserved unchanged.')
    parser.add_argument('--migration-receipt', help='Preserved pause receipt containing the original exact identity.')
    parser.add_argument('--recovery-benchmark', help='Paired target-GPU continuation measurement from the original checkpoint.')
    parser.add_argument('--validate-only', action='store_true',
                        help='Validate saved launch bindings without GPU work or Drive requests.')
    args = parser.parse_args()
    config = validate_configuration(read_json(args.config),runtime=True)
    experiment = Experiment(config,config_path=args.config)
    if config['runtime']['implementation_revision'] != experiment.code:
        raise ValueError('Executed source differs from the frozen SFT consumer')
    if not (config['spending'].get('authorized') and config['spending'].get('authorization_evidence')
            and config['spending'].get('authorized_usd',0) > 0):
        raise ValueError('Current spending authorization is required')
    certificate = read_json(args.certificate)
    tasks = certified_tasks(config,certificate,read_json(args.bundle),read_json(args.worklist),
                            family=args.family,model=args.model,level=args.level)
    execution_profile = None
    execution_benchmark = None
    if bool(args.execution_benchmark) != bool(args.execution_profile):
        raise ValueError('Execution activation requires both benchmark and profile')
    if args.execution_benchmark:
        from training.sft.execution_profile import SFTExecutionProfile
        from training.sft.runtime_migration import validate_execution_benchmark
        execution_benchmark = read_json(args.execution_benchmark)
        settings = execution_benchmark['plan']['profiles'][args.execution_profile]
        execution_profile = SFTExecutionProfile(**settings)
        approval = validate_execution_benchmark(execution_benchmark,experiment.code,execution_profile,
                                               profile_name=args.execution_profile)
        if (approval['configuration_hash'] != experiment.hash or len(tasks) != 1
                or approval['condition'] != {'family':args.family,'model':args.model,'level':args.level}):
            raise ValueError('Activation evidence must match the one selected certified fit')
    migration = None
    if any((args.migration_checkpoint,args.migration_receipt,args.recovery_benchmark)):
        if not all((args.migration_checkpoint,args.migration_receipt,args.recovery_benchmark,
                    args.execution_benchmark,args.execution_profile,args.level)):
            raise ValueError('Runtime migration requires original recovery, paired measurements and one accepted fit')
        migration = read_json(args.migration_receipt)
        if migration.get('receipt_hash') != content_hash({k:v for k,v in migration.items() if k != 'receipt_hash'}):
            raise ValueError('Original pause receipt integrity differs')
    preflight = load_preflight_evidence(args.preflight,config)
    require_preflights(preflight,preflight_requirements('training'),expected_configuration_hash=experiment.hash)
    if preflight.get('ready_models') is not None:
        if args.model not in preflight['ready_models'] or execution_profile is not None or migration is not None:
            raise ValueError('Model-scoped baseline controls cannot authorize another model, an optimization, or runtime migration')
    if preflight.get('ready_scope') is not None:
        expected_scope = {'family':args.family,'model':args.model,'level':args.level}
        if (len(tasks) != 1 or preflight['ready_scope'] != expected_scope
                or execution_profile is None):
            raise ValueError('Target-condition preflight cannot authorize another condition or execution profile')
    deadline = datetime.fromisoformat(args.deadline.replace('Z','+00:00'))
    if deadline.utcoffset() is None or deadline.utcoffset().total_seconds() != 0:
        raise ValueError('Deadline must use UTC')
    if datetime.now(timezone.utc) >= deadline:
        raise ValueError('Deadline already passed; SFT was not launched')
    if args.validate_only:
        print(__import__('json').dumps({'status':'READY','source_code_commit':experiment.code,
            'configuration_hash':experiment.hash,'handoff_certificate_hash':certificate['certificate_hash'],
            'preflight_report_hash':preflight['report_hash'],'required_gates':len(preflight_requirements('training')),
            'certified_conditions':len(certificate['conditions']),'selected_conditions':len(tasks),
            'family':args.family,'model':args.model,'deadline':deadline.isoformat(),
            'gpu_work_started':False,'drive_requests_made':False}),flush=True)
        return 0
    stop = {'requested':False}
    def request_pause(_signum,_frame):
        stop['requested'] = True
    signal.signal(signal.SIGTERM,request_pause)
    signal.signal(signal.SIGINT,request_pause)
    def should_pause():
        return stop['requested'] or datetime.now(timezone.utc) >= deadline
    configure_worker_recovery(experiment.artifacts,worker_id=f'sft_{args.family}_{args.model}')
    output_root = experiment.artifacts/'training'/'sft'
    status_path = output_root/f'worker_status_{args.family}_{args.model}.json'
    completed = []
    current = None
    def status(state, **extra):
        report = {'status':state,'source_code_commit':experiment.code,'configuration_hash':experiment.hash,
            'handoff_certificate_hash':certificate['certificate_hash'],'family':args.family,'model':args.model,
            'required_fits':len(tasks),'completed_fits':len(completed),'completed_run_ids':list(completed),
            'current_run_id':current,'deadline':deadline.isoformat(),'at':datetime.now(timezone.utc).isoformat(),**extra}
        atomic_write_json(status_path,report)
        print(__import__('json').dumps(report),flush=True)
    def preserve(path, metadata):
        experiment.publish_permanent(path,metadata)
        if Path(path).name == 'manifest.json' and metadata.get('progress'):
            status('RUNNING',completed_steps=metadata['progress']['completed_steps'],drive_status='VERIFIED')
    with file_lock(output_root/f'.worker_{args.family}_{args.model}.lock',timeout=1),artifact_hook(preserve):
        try:
            status('STARTING',drive_status='PENDING')
            # Authenticate and preserve the launch identity before any GPU fit.
            experiment.publish_permanent(status_path,{'kind':'sft_queue_start'})
            for task in tasks:
                if should_pause():
                    status('PAUSED',drive_status='VERIFIED')
                    return 3
                spec = SFTFitSpec.from_dict(task['spec'])
                current = spec.run_id
                experiment._current_stage = Stage(spec.run_id,'sft',(),model=spec.model_id,
                                                   branch=spec.family,level=spec.level,gpu_count=1)
                status('RUNNING',completed_steps=0,drive_status='VERIFIED')
                train, validation = load_dataset_bundle(task['dataset'])
                output = output_root/spec.family/spec.model_id/f'level_{spec.level:02d}'
                if migration is not None:
                    from training.sft.runtime_migration import migrate_checkpoint
                    identity = {**spec.identity(),'code_revision':experiment.code,
                                'dependency_hash':config['runtime']['dependency_lock_hash'],
                                'execution_profile':execution_profile.to_dict()}
                    original = migration['checkpoint_manifest']
                    target = output/f"checkpoint_{original['progress']['completed_steps']:06d}"
                    lineage = migrate_checkpoint(args.migration_checkpoint,target,original['identity'],identity,
                        execution_profile,execution_benchmark,
                        recovery_report=read_json(args.recovery_benchmark))
                    lineage_path = output/'runtime_migration.json'
                    atomic_write_json(lineage_path,lineage)
                    experiment.publish_permanent(lineage_path,{'identity':identity,'kind':'sft_runtime_migration'})
                    migration = None
                result = run_sft_fit(spec,train,validation,output,config,preflight_report=preflight,
                                    should_pause=should_pause,handoff_certificate=certificate,
                                    execution_profile=execution_profile,execution_benchmark=execution_benchmark)
                # Checkpoint notifications are synchronous Drive barriers. Final
                # adapter publication has a separate verified payload inventory.
                adapter = output/'final_adapter'
                paths = [output/'fit_status.json'] if result['status'] == 'paused' else [
                    *(path for path in sorted(adapter.rglob('*')) if path.name != 'manifest.json'),
                    adapter/'manifest.json',output/'fit_result.json']
                for path in paths:
                    if path.is_file():
                        experiment.publish_permanent(path,{'identity':result['identity']})
                if result['status'] == 'paused':
                    status('PAUSED',completed_steps=result['progress']['completed_steps'],drive_status='VERIFIED')
                    return 3
                if result['status'] != 'complete':
                    raise RuntimeError('SFT fit did not return verified completion')
                completed.append(spec.run_id)
                del train,validation
                gc.collect()
                import torch
                torch.cuda.empty_cache()
            current = None
            status('COMPLETE',drive_status='VERIFIED')
            experiment.publish_permanent(status_path,{'kind':'sft_queue_completion'})
            return 0
        except Exception as error:
            status('FAILED',error_type=type(error).__name__,detail=str(error),drive_status='REQUIRED')
            raise


if __name__ == '__main__':
    raise SystemExit(main())
