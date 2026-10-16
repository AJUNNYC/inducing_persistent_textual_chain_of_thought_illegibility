#!/usr/bin/env python3
"""Wait for verified epoch2 completion, then repeat only frozen level10 GPQA.

This operations dispatcher never changes the frozen evaluator or original data.
The original baseline's 198 raw JSON files are copied byte-for-byte into a new
namespace and synchronously verified in Drive before the native paired CLI runs.
Only the continued level10 adapter requires model inference.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
FROZEN_REVISION = "5e6be1b0848e0e88725323591baddef37d69d8ae"
STATE = "rot13_14b_level_10"


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def process_start(pid, proc_root=Path('/proc')):
    try:
        # Process names may contain spaces or parentheses.
        return (proc_root / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def same_process_alive(pid, start, proc_root=Path('/proc')):
    if start is None or process_start(pid, proc_root) != start:
        return False
    try:
        return (proc_root / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except (OSError, IndexError):
        return False


def authenticated_source(args):
    """Authenticate native entrypoints against the original sealed campaign."""
    plan = json.loads(args.original_campaign_plan.read_text())
    sealed = {k: v for k, v in plan.items() if k != 'plan_hash'}
    digest = hashlib.sha256(json.dumps(sealed, sort_keys=True, separators=(',', ':'),
                                      ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    expected_names = {'scripts/run_gpqa_sft_campaign.py', 'scripts/run_supplemental_gpqa.py'}
    expected = plan.get('implementation_sha256', {})
    baseline = json.loads(args.baseline_protocol.read_text())
    original = json.loads(args.original_level10_protocol.read_text())
    if (plan.get('plan_hash') != digest or plan.get('source_code_commit') != FROZEN_REVISION
            or set(expected) != expected_names
            or any(sha(args.frozen_source / path) != expected[path] for path in expected_names)
            or Path(plan.get('output_root', '')).resolve() != args.original_output_root
            or not any(j.get('state_id') == 'base_14b' and j.get('protocol_hash') == baseline.get('protocol_hash')
                       for j in plan.get('order', []))
            or not any(j.get('state_id') == STATE and j.get('protocol_hash') == original.get('protocol_hash')
                       for j in plan.get('order', []))):
        raise RuntimeError('Frozen evaluation entrypoints differ from the original authenticated campaign')
    return plan


def sft_completion_gate(output, verify):
    """Read the native full-state gate and checksum-bearing Drive records."""
    output = Path(output)
    status = json.loads((output / 'fit_status.json').read_text()) if (output / 'fit_status.json').exists() else {}
    if status.get('status') == 'failed':
        raise RuntimeError('SFT failed; evaluation has not been launched')
    if status.get('status') != 'complete' or not (output / 'fit_result.json').exists():
        return None
    identity = json.loads((output / 'run_identity.json').read_text())
    result = verify(output, identity)
    progress = result['progress']
    if (identity.get('run_id') != 'rot13_14b_level_10_continuation_epoch_02'
            or identity.get('training_budget') != 'efficient_additional_epoch_v1'
            or progress.get('completed_steps') != 625
            or progress.get('cumulative_completed_steps') != 1250):
        raise RuntimeError('SFT completion is not the declared finished level10 extra epoch')
    database = output / 'drive_registry.sqlite3'
    if not database.exists():
        return None
    with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True, timeout=30) as db:
        records = [json.loads(r[0]) for r in db.execute('SELECT record FROM artifacts')]
    if not records or any(r.get('sync_status') != 'VERIFIED' for r in records):
        return None
    required = [output / 'fit_result.json', output / 'run_identity.json',
                output / 'checkpoint_000625' / 'manifest.json',
                output / 'checkpoint_000625' / 'full_state.pt',
                *[p for p in (output / 'final_adapter').rglob('*') if p.is_file()]]
    for path in required:
        matching = [r for r in records if r.get('local_path') == str(path.resolve())
                    and r.get('sha256') == sha(path)
                    and r.get('configuration_hash') == identity['configuration_hash']
                    and r.get('source_code_commit') == identity['code_revision']
                    and r.get('drive_file_id')]
        if not matching:
            return None
    return {'status': 'VERIFIED', 'run_id': identity['run_id'],
            'identity_hash': hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':'),
                                                     ensure_ascii=False).encode()).hexdigest(),
            'fit_result_sha256': sha(output / 'fit_result.json'),
            'checkpoint_hash': result['checkpoint_hash'],
            'adapter_manifest_hash': result['adapter_manifest_hash'],
            'configuration_hash': identity['configuration_hash'],
            'source_code_commit': identity['code_revision'],
            'completed_steps': 625, 'cumulative_completed_steps': 1250,
            'drive_artifacts_verified': len(records)}


def frozen_api(source):
    source = Path(source).resolve()
    sys.path.insert(0, str(source / 'src'))
    import evaluation.supplemental_gpqa as gpqa
    import shared_procedures.storage as storage
    import operations.artifacts as artifacts
    import operations.drive as drive
    for imported in (gpqa, storage, artifacts, drive):
        if not Path(imported.__file__).resolve().is_relative_to(source / 'src'):
            raise RuntimeError('Evaluation imported a different scientific source')
    campaign = module(source / 'scripts/run_gpqa_sft_campaign.py', 'frozen_gpqa_after_sft_campaign')
    return SimpleNamespace(gpqa=gpqa, storage=storage, artifacts=artifacts, drive=drive,
                           campaign=campaign, source=source)


def prepared_protocol(api, baseline, original, adapter, model_cache):
    """Native helpers retain original model, prompts, seeds, scorer and settings."""
    api.gpqa.validate_gpqa_protocol(baseline)
    api.gpqa.validate_gpqa_protocol(original)
    if (baseline['version'] != api.gpqa.BASELINE_VERSION
            or baseline['panel']['variant'] != 'gpqa_diamond'
            or baseline['panel']['item_count'] != 198
            or baseline['panel'] != original['panel']
            or baseline['generation'] != api.gpqa.generation_contract('vllm', 'long_32000')
            or baseline['generation'] != original['generation']
            or [s['state_id'] for s in original['states']] != ['base_14b', STATE]):
        raise RuntimeError('Original comparison is not the same full long-budget level10 GPQA')
    states = api.gpqa.comparison_states(adapter, model_cache)
    if (states[1]['state_id'] != STATE or states[1]['family'] != 'rot13'
            or states[1]['model'] != '14b' or states[1]['level'] != 10):
        raise RuntimeError('Only the completed continued ROT13 14B level10 adapter is selected')
    new = api.gpqa.freeze_gpqa_protocol(baseline['panel'], states, backend='vllm',
        generation_profile='long_32000', baseline_protocol=baseline)
    if (new['generation'] != original['generation']
            or new['states'][1]['state_id'] != original['states'][1]['state_id']
            or new['protocol_hash'] == original['protocol_hash']):
        raise RuntimeError('Repeat must retain generation/seeds and use a distinct adapter namespace')
    return new


def raw_copy_plan(api, baseline, original_root, output):
    generator = api.gpqa.GPQAVLLMGenerator(Path(original_root) / baseline['protocol_hash'], baseline)
    copies = []
    state = baseline['states'][0]
    for item in baseline['panel']['items']:
        identity = generator.identity(state, item)
        name = api.gpqa.content_hash(identity) + '.json'
        relative = Path(baseline['protocol_hash']) / 'raw' / 'base_14b' / name
        source, target = Path(original_root) / relative, Path(output) / relative
        raw = api.storage.read_json(source)
        api.campaign.sealed(raw, 'record_hash')
        if raw['identity'] != identity or raw['request_hash'] != api.gpqa.content_hash(identity):
            raise RuntimeError('Baseline raw identity differs before byte-preserving reuse')
        copies.append({'source': str(source), 'destination': str(target), 'sha256': sha(source)})
    if len(copies) != 198:
        raise RuntimeError('Full baseline reuse requires all198 cached raw responses')
    return copies


def provenance(protocol, source=FROZEN_REVISION):
    key = protocol['protocol_hash']
    return {'experiment_id': key, 'branch': 'supplemental_gpqa', 'model': '14b',
            'run_id': key[:16], 'stage': 'supplemental_capability',
            'producer': 'run_supplemental_gpqa', 'source_code_commit': source,
            'configuration_hash': key, 'dataset_revision': protocol['panel']['source']['revision'],
            'model_revision': protocol['states'][0]['model_revision'], 'seeds': [42]}


def service_for(api, output):
    token, folder = os.environ.get('GOOGLE_OAUTH_TOKEN_FILE'), os.environ.get('GOOGLE_DRIVE_FOLDER_ID')
    if not token or not folder:
        raise RuntimeError('Private Drive configuration is missing')
    return api.artifacts.ArtifactService(output, Path(output) / 'drive_registry.json',
        api.drive.DriveStore(api.drive.drive_service(token), folder), drive_recovery=True)


def adopt_raw(api, baseline, copies, service):
    """Copy immutable bytes, then use native registration/Drive verification."""
    for copy in copies:
        source, target = Path(copy['source']), Path(copy['destination'])
        if sha(source) != copy['sha256']:
            raise RuntimeError('Verified baseline bytes changed during adoption')
        if target.exists():
            if sha(target) != copy['sha256']:
                raise RuntimeError('New baseline cache contains different bytes; preserving it')
        else:
            api.storage.atomic_write_bytes(target, source.read_bytes())
        raw = api.storage.read_json(target)
        service.register(target, {**provenance(baseline), 'kind': 'evaluation_raw',
            'state_id': 'base_14b', 'request_hash': raw['request_hash'],
            'raw_hash': raw['raw_hash'], 'simulation': False})
    if service.sync_artifacts(workers=8)['status'] != 'VERIFIED':
        raise RuntimeError('Permanent artifact synchronization incomplete; registry retains pending/failed records')


def operation_comparison(api, old_report, new_report):
    old = {r['question_id']: r for r in old_report['records'] if r['state_id'] == STATE}
    new = {r['question_id']: r for r in new_report['records'] if r['state_id'] == STATE}
    if len(old) != 198 or old.keys() != new.keys():
        raise RuntimeError('Original/continued paired question coverage differs')
    transitions = {'both_correct': 0, 'gained_correct': 0, 'lost_correct': 0, 'both_incorrect': 0}
    for key in old:
        a, b = old[key]['score']['correct'], new[key]['score']['correct']
        label = ('both_correct' if a and b else 'gained_correct' if b else
                 'lost_correct' if a else 'both_incorrect')
        transitions[label] += 1
    return {'version': 'gpqa_level10_original_vs_continued_v1',
        'scope': 'same_frozen_supplemental_gpqa_descriptive_comparison',
        'original_protocol_hash': old_report['protocol_hash'],
        'continued_protocol_hash': new_report['protocol_hash'],
        'original_summary': old_report['summaries'][STATE], 'continued_summary': new_report['summaries'][STATE],
        'accuracy_difference_continued_minus_original':
            new_report['summaries'][STATE]['accuracy'] - old_report['summaries'][STATE]['accuracy'],
        'paired_transitions': transitions, 'questions': 198, 'state_id': STATE,
        'same_prompts_sampling_seeds_and_scorer': True, 'new_inference_states': [STATE],
        'baseline_regenerated': False, 'original_artifacts_modified': False}


def execute_phase(args):
    authenticated_source(args)
    api = frozen_api(args.frozen_source)
    baseline, original = (api.storage.read_json(p) for p in (args.baseline_protocol, args.original_level10_protocol))
    for protocol in (baseline, original):
        api.campaign.completion_barrier(protocol, args.original_output_root,
                                        save_to_drive=True, source_code_commit=FROZEN_REVISION)
    gate = api.storage.read_json(args.output_root / 'handoff' / 'sft_completion.json')
    if (gate.get('status') != 'VERIFIED'
            or gate['fit_result_sha256'] != sha(args.sft_output / 'fit_result.json')):
        raise RuntimeError('SFT handoff changed after verified completion')
    protocol = prepared_protocol(api, baseline, original, args.sft_output / 'final_adapter', args.model_cache)
    if protocol['states'][1]['adapter_manifest_hash'] != gate['adapter_manifest_hash']:
        raise RuntimeError('Evaluation adapter differs from completed SFT handoff')
    protocol_path = args.output_root / 'handoff' / 'protocol.json'
    api.gpqa.publish_immutable(protocol_path, protocol)
    copies = raw_copy_plan(api, baseline, args.original_output_root, args.output_root)
    adoption = {'version': 'gpqa_cached_baseline_adoption_v1', 'source_code_commit': FROZEN_REVISION,
                'baseline_protocol_hash': baseline['protocol_hash'], 'copies': copies,
                'original_completion_receipt_sha256': sha(args.original_output_root / baseline['protocol_hash'] / 'completion_receipt.json'),
                'new_predictions_generated': 0}
    api.gpqa.publish_immutable(args.output_root / 'handoff' / 'baseline_adoption.json', adoption)
    service = service_for(api, args.output_root)
    adopt_raw(api, baseline, copies, service)
    plan = {'version': 'gpqa_after_sft_continuation_plan_v1', 'evaluation_source_commit': FROZEN_REVISION,
            'operations_source_commit': args.operations_commit, 'dispatcher_sha256': sha(__file__),
            'sft_completion': gate, 'protocol_hash': protocol['protocol_hash'],
            'original_level10_protocol_hash': original['protocol_hash'],
            'baseline_protocol_hash': baseline['protocol_hash'], 'new_inference_states': [STATE],
            'questions': 198, 'baseline_regenerated': False}
    api.gpqa.publish_immutable(args.output_root / 'handoff' / 'plan.json', plan)
    native = [sys.executable, str(args.frozen_source / 'scripts/run_supplemental_gpqa.py'),
              'run', '--protocol', str(protocol_path), '--output-root', str(args.output_root), '--save-to-drive']
    subprocess.run(native, check=True, cwd=args.frozen_source)
    receipt = api.campaign.completion_barrier(protocol, args.output_root,
                                            save_to_drive=True, source_code_commit=FROZEN_REVISION)
    report = api.storage.read_json(args.output_root / protocol['protocol_hash'] / 'comparison.json')
    old_report = api.storage.read_json(args.original_output_root / original['protocol_hash'] / 'comparison.json')
    comparison = operation_comparison(api, old_report, report)
    final_path = args.output_root / 'handoff' / 'comparison.json'
    api.gpqa.publish_immutable(final_path, comparison)
    ops = {**provenance(protocol, args.operations_commit), 'producer': 'run_gpqa_after_sft_continuation',
           'stage': 'post_sft_gpqa_handoff', 'evaluation_source_commit': FROZEN_REVISION}
    for path in sorted((args.output_root / 'handoff').glob('*.json')):
        service.register(path, ops)
    if service.sync_artifacts()['status'] != 'VERIFIED':
        raise RuntimeError('Permanent artifact synchronization incomplete; registry retains pending/failed records')
    service.export_manifest()
    return receipt


def common_command(args, phase, python):
    command = [str(python), str(Path(__file__).absolute()), '--phase', phase]
    for name in ('frozen_source', 'eval_python', 'training_source', 'training_python', 'sft_output',
                 'original_output_root', 'original_campaign_plan', 'baseline_protocol', 'original_level10_protocol', 'model_cache',
                 'output_root', 'operations_commit'):
        command += ['--' + name.replace('_', '-'), str(getattr(args, name))]
    return command


def supervise(args):
    recovery = module(ROOT / 'scripts/run_gpqa_campaign_with_recovery.py', 'post_sft_native_recovery')
    args.output_root.mkdir(parents=True, exist_ok=True)
    monitor = args.output_root / 'campaigns' / 'after_sft'
    monitor.mkdir(parents=True, exist_ok=True)
    status_path = monitor / 'status.json'
    old_status = json.loads(status_path.read_text()) if status_path.exists() else {}
    recoveries = old_status.get('recoveries', 0)
    start_ticks = old_status.get('training_process_start') if old_status else process_start(args.sft_pid)
    if old_status and old_status.get('training_pid') != args.sft_pid:
        raise RuntimeError('Saved dispatcher belongs to a different training process')
    child, stopping = None, {'signal': None}
    previous = {}
    def status(value, **extra):
        record = {'status': value, 'pid': os.getpid(), 'at_utc': datetime.now(timezone.utc).isoformat(),
                  'training_pid': args.sft_pid, 'training_process_start': start_ticks,
                  'recoveries': recoveries, 'evaluation_source_commit': FROZEN_REVISION,
                  'operations_source_commit': args.operations_commit, **extra}
        recovery.private_json(status_path, record)
        print(json.dumps(record), flush=True)
    for number in (signal.SIGINT, signal.SIGTERM):
        previous[number] = signal.signal(number, lambda number, _frame: stopping.update(signal=number))
    environment = os.environ.copy()
    environment.update(SOURCE_CODE_COMMIT=FROZEN_REVISION, PYTHONPATH=str(args.frozen_source / 'src'))
    def owned_command(command, log, *, env=None, capture=False, timeout=None):
        nonlocal child
        child = subprocess.Popen(command, cwd=args.frozen_source, env=env,
            stdout=subprocess.PIPE if capture else log, stderr=log, text=capture,
            start_new_session=True)
        deadline = time.monotonic() + timeout if timeout else float('inf')
        while child.poll() is None:
            if stopping['signal'] or time.monotonic() >= deadline:
                recovery.terminate_owned_group(child, 10)
                child = None
                if stopping['signal']:
                    return 128 + stopping['signal'], ''
                raise TimeoutError('Owned completion gate/repair exceeded its timeout')
            time.sleep(min(args.poll_seconds, 1))
        result = child.stdout.read() if capture else ''
        code = child.returncode
        recovery.terminate_owned_group(child, 10)
        child = None
        return code, result
    try:
        with recovery.exclusive_lock(monitor / 'dispatcher.lock'):
            gate_path = args.output_root / 'handoff' / 'sft_completion.json'
            while True:
                if stopping['signal']:
                    status('STOPPED'); return 128 + stopping['signal']
                with (monitor / 'sft_gate.log').open('a') as log:
                    gate_environment = os.environ.copy()
                    gate_environment['CUDA_VISIBLE_DEVICES'] = ''
                    code, gate_text = owned_command(common_command(args, 'sft-gate', args.training_python),
                        log, env=gate_environment, capture=True, timeout=600)
                if stopping['signal']:
                    status('STOPPED'); return 128 + stopping['signal']
                if code:
                    raise RuntimeError('SFT completion validation failed; inspect private gate log')
                gate = json.loads(gate_text)
                if gate and not same_process_alive(args.sft_pid, start_ticks):
                    gate_path.parent.mkdir(exist_ok=True)
                    if gate_path.exists() and json.loads(gate_path.read_text()) != gate:
                        raise RuntimeError('Existing verified SFT handoff differs')
                    recovery.private_json(gate_path, gate)
                    break
                if not same_process_alive(args.sft_pid, start_ticks) and not gate:
                    raise RuntimeError('Training process exited before verified completion')
                status('WAITING_FOR_TRAINING_EXIT' if gate else 'WAITING_FOR_SFT')
                time.sleep(args.poll_seconds)
            while not stopping['signal']:
                authenticated_source(args)
                recovery.assert_idle(args.output_root)
                failures = recovery.FreshFailures(args.output_root)
                with (monitor / 'worker.log').open('a') as log:
                    child = recovery.launch(common_command(args, 'execute', args.eval_python),
                        source_root=args.frozen_source, environment=environment, log=log)
                    status('RUNNING', worker_pid=child.pid)
                    detected = False
                    while True:
                        detected |= failures.poll()
                        code = child.poll()
                        if stopping['signal'] or detected or code is not None:
                            break
                        time.sleep(args.poll_seconds)
                    if code == 0:
                        recovery.terminate_owned_group(child, 10)
                        child = None; status('COMPLETE'); return 0
                    state = recovery.registry_state(args.output_root)
                    recovery.terminate_owned_group(child, 10); child = None
                if stopping['signal']:
                    status('STOPPED'); return 128 + stopping['signal']
                if not detected or not recovery.recoverable(state) or recoveries >= args.max_recoveries:
                    status('FAILED', reason='unrecognized_failure_or_recovery_limit', registry=state); return 1
                recoveries += 1; status('REPAIRING', registry=state)
                authenticated_source(args)
                repair_command = [str(args.eval_python), str(ROOT / 'scripts/run_gpqa_campaign_with_recovery.py'),
                    '--source-root', str(args.frozen_source), '--output-root', str(args.output_root), '--repair-only']
                with (monitor / 'repair.log').open('a') as log:
                    repaired, _ = owned_command(repair_command, log, env=environment, timeout=600)
                if stopping['signal']:
                    status('STOPPED'); return 128 + stopping['signal']
                if repaired:
                    status('FAILED', reason='native_incremental_repair_failed'); return 1
                status('REPAIRED', registry=recovery.registry_state(args.output_root))
            return 128 + stopping['signal']
    except Exception as error:
        status('FAILED', reason=type(error).__name__, detail=str(error)); return 1
    finally:
        if child is not None:
            recovery.terminate_owned_group(child, 10)
        for number, handler in previous.items():
            signal.signal(number, handler)


def main(argv=None):
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('frozen-source', 'eval-python', 'training-source', 'training-python', 'sft-output',
                 'original-output-root', 'original-campaign-plan', 'baseline-protocol', 'original-level10-protocol', 'model-cache', 'output-root'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--operations-commit', required=True)
    parser.add_argument('--sft-pid', type=int)
    parser.add_argument('--poll-seconds', type=float, default=15)
    parser.add_argument('--max-recoveries', type=int, default=12)
    parser.add_argument('--phase', choices=('wait', 'sft-gate', 'execute'), default='wait', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    for name in ('frozen_source', 'training_source', 'sft_output', 'original_output_root',
                 'original_campaign_plan', 'baseline_protocol', 'original_level10_protocol', 'model_cache', 'output_root'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    for name in ('eval_python', 'training_python'):
        setattr(args, name, getattr(args, name).expanduser().absolute())
    if args.phase == 'sft-gate':
        sys.path.insert(0, str(args.training_source / 'src'))
        from training.sft.continuation import verified_completion
        print(json.dumps(sft_completion_gate(args.sft_output, verified_completion))); return 0
    if args.phase == 'execute':
        if os.environ.get('SOURCE_CODE_COMMIT') != FROZEN_REVISION:
            raise RuntimeError('Frozen evaluation source identity must remain unchanged')
        execute_phase(args); return 0
    if args.sft_pid is None or args.sft_pid <= 0 or args.poll_seconds <= 0 or args.max_recoveries < 0:
        parser.error('A positive SFT PID/poll interval and nonnegative recovery limit are required')
    for protected in (args.original_output_root, args.sft_output):
        if (args.output_root == protected or args.output_root.is_relative_to(protected)
                or protected.is_relative_to(args.output_root)):
            raise RuntimeError('New evaluation root must be separate from original evaluation and training')
    if len(args.operations_commit) != 40 or any(c not in '0123456789abcdef' for c in args.operations_commit):
        parser.error('operations-commit must be an immutable Git revision')
    return supervise(args)


if __name__ == '__main__':
    raise SystemExit(main())
