#!/usr/bin/env python3
"""Versioned final-field repair using preserved real outputs, without inference.

Original packages and failed reports remain immutable. Only scoring derivatives
change; references never enter answer extraction and no ordinary output is retried.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from experiment_overview.experiment_overview import Experiment
from shared_procedures.configuration import validate_configuration
from shared_procedures.identity import content_hash, file_hash
from shared_procedures.storage import atomic_write_json, read_json, file_lock
from evaluation.shared_model_evaluation.item_evaluation.item_evaluation import evaluate_item
from evaluation.shared_model_evaluation.model_evaluation import evaluate_model
from evaluation.shared_model_evaluation.item_evaluation.native_output.vllm_generation import VLLMEvaluationGenerator
from evaluation.shared_model_evaluation.item_evaluation.capability_scoring.capability_scoring import SCORER_VERSION, FINAL_FIELD_CONTRACT
from data_generation_collection.evaluation_data.protocol.evaluation_protocol import build_evaluation_protocol
from data_generation_collection.evaluation_data.protocol.analysis_plan.baseline_repeatability import baseline_repeatability


class CacheOnlyGenerator(VLLMEvaluationGenerator):
    """A missing saved output is a failure, never a new inference request."""
    def _load(self, state):
        raise RuntimeError('Rescoring cannot generate a missing saved response')

    def _generate_cuda(self, *args, **kwargs):
        raise RuntimeError('Rescoring cannot generate a missing saved response')


def validate_scoring_only_revision(old, new):
    for key in ('generation', 'prompts', 'prompt_hashes', 'panel_hash', 'legibility_ids', 'sandbox', 'judges', 'opacity'):
        if new[key] != old[key]:
            raise ValueError('Rescoring may not change generation, panel, judges or sandbox: ' + key)
    allowed = {'evaluation/shared_model_evaluation/item_evaluation/capability_scoring/capability_scoring.py',
        'data_generation_collection/evaluation_data/protocol/evaluation_protocol.py'}
    if old['code_hashes'].keys() != new['code_hashes'].keys():
        raise ValueError('Rescoring cannot change the measurement source inventory')
    changed = {name for name in old['code_hashes'] if old['code_hashes'][name] != new['code_hashes'][name]}
    if changed - allowed:
        raise ValueError('Rescoring changed code outside the final-field repair')


def checked_record(value, key):
    if value.get(key) != content_hash({k:v for k,v in value.items() if k != key}):
        raise ValueError('Preserved artifact integrity differs: ' + key)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'source-root', 'output-root'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError('Cloud scoring uses one to eight independent restricted executors')
    source = Path(args.source_root).resolve()
    destination = Path(args.output_root).resolve()
    if source == destination or destination.is_relative_to(source):
        raise ValueError('Repair must retain the original evaluation separately')
    os.environ['ARTIFACT_ROOT'] = str(destination)
    config = validate_configuration(read_json(args.config), runtime=True)
    experiment = Experiment(config, config_path=args.config)
    old_protocol = read_json(source / 'measurement_protocol.json')
    if old_protocol['protocol_hash'] != content_hash(old_protocol['specifications']):
        raise ValueError('Original protocol integrity differs')
    old_report = checked_record(read_json(source / 'preflight/baseline_repeatability.json'), 'report_hash')
    protocol = build_evaluation_protocol({'items':old_protocol['items']}, old_protocol['legibility_ids'],
        analysis_plan=old_protocol.get('analysis_plan'),
        preflight_reports=old_protocol['preflight_reports'],
        configuration_hash=old_protocol['specifications'].get('configuration_hash'), generation_backend='vllm')
    validate_scoring_only_revision(old_protocol['specifications'], protocol['specifications'])
    sandbox = experiment.sandbox()
    if sandbox.identity != protocol['specifications']['sandbox']['identity'] or sandbox.available()['status'] != 'passed':
        raise RuntimeError('Validated cloud executor unavailable or changed')
    output = destination / 'evaluation'
    output.mkdir(parents=True, exist_ok=True)
    generator = CacheOnlyGenerator(output)
    service = experiment.artifact_service()
    completed = 0
    def status(kind, **extras):
        result = {'status':kind, 'completed':completed, 'total':7398,
            'observed_at':datetime.now(timezone.utc).isoformat(), 'source_commit':experiment.code,
            'protocol_hash':protocol['protocol_hash'], 'judging':'deferred', **extras}
        atomic_write_json(destination / 'coordinator_status.json', result)
        print(json.dumps(result), flush=True)
    def publish(path, kind):
        provenance = experiment.provenance(metadata={'kind':kind})
        provenance.update(producer='preserved_output_rescoring', seeds=[42,43,44])
        if not service.list_artifacts(local_path=str(path.resolve()), sha256=file_hash(path)):
            service.register(path, provenance)
        if service.sync_artifacts(workers=8)['status'] != 'VERIFIED':
            raise RuntimeError('Repair evidence Drive barrier failed')
    with file_lock(destination / 'rescoring.lock'):
        atomic_write_json(destination / 'measurement_protocol.json', protocol)
        publish(destination / 'measurement_protocol.json', 'versioned_measurement_manifest')
        passes = {}
        identities = []
        status('SCORING')
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for model in ('7b','14b'):
                passes[model] = {}
                for seed in (42,43,44):
                    package = checked_record(read_json(source / 'evaluation' / f'base_{model}_seed_{seed}.json'), 'package_hash')
                    if (package['protocol_hash'] != old_protocol['protocol_hash'] or package['seed'] != seed
                        or package['status'] != 'complete' or len(package['records']) != 1233
                        or package['state']['state_id'] != 'base_' + model
                        or package['state']['configuration_hash'] != experiment.hash):
                        raise ValueError('Preserved full pass identity differs')
                    state = package['state']
                    for item, record in zip(protocol['items'], package['records']):
                        raw = checked_record(record['raw'], 'record_hash')
                        identity = generator.identity(state, item, seed)
                        if (raw['identity'] != identity or raw['request_hash'] != content_hash(identity)
                            or raw['raw_hash'] != content_hash(raw['raw_output'])
                            or raw.get('simulation') is not False or record['question_id'] != identity['question_id']):
                            raise ValueError('Preserved response does not match unchanged generation')
                        path = output / 'raw' / state['state_id'] / (raw['request_hash'] + '.json')
                        if path.exists() and read_json(path) != raw:
                            raise ValueError('Raw bytes changed')
                        if not path.exists():
                            atomic_write_json(path, raw)
                        identities.append({'request_hash':raw['request_hash'], 'raw_hash':raw['raw_hash']})
                    for start in range(0, 1233, 128):
                        batch = protocol['items'][start:start+128]
                        def score(item):
                            return evaluate_item(state, item, protocol, generator, sandbox=sandbox,
                                cache_root=output, seed=seed, defer_judging=True)
                        records = list(executor.map(score, batch))
                        if any(record['status'] != 'complete' for record in records):
                            raise RuntimeError('Restricted capability scoring is incomplete')
                        completed += len(records)
                        status('SCORING', model=model, seed=seed, completed_in_pass=start+len(batch))
                    result = evaluate_model(state, protocol, generator, sandbox=sandbox,
                        cache_root=output, seed=seed, defer_judging=True)
                    path = output / f'base_{model}_seed_{seed}.json'
                    atomic_write_json(path, result)
                    publish(path, 'rescored_baseline_package')
                    passes[model][seed] = result
                    status('PASS_COMPLETE', model=model, seed=seed, drive_replication='VERIFIED')
        report = baseline_repeatability(passes, protocol['items'])
        path = destination / 'preflight/baseline_repeatability.json'
        atomic_write_json(path, report)
        publish(path, 'rescored_baseline_repeatability')
        receipt = {'schema_version':'preserved_baseline_rescoring_v1',
            'source_commit':experiment.code, 'source_root':str(source),
            'configuration_hash':experiment.hash, 'original_protocol_hash':old_protocol['protocol_hash'],
            'protocol_hash':protocol['protocol_hash'], 'original_report_hash':old_report['report_hash'],
            'report_hash':report['report_hash'], 'scorer_version':SCORER_VERSION,
            'final_field_contract':FINAL_FIELD_CONTRACT, 'preserved_raw_count':len(identities),
            'preserved_raw_identity_hash':content_hash(identities), 'new_generations':0,
            'scientific_thresholds_changed':False, 'corpus_changed':False,
            'original_failure_retained':True, 'legibility_judging':'deferred',
            'ready_for_training':report['ready_for_training'], 'drive_replication':'VERIFIED'}
        receipt['receipt_hash'] = content_hash(receipt)
        path = destination / 'repair_receipt.json'
        atomic_write_json(path, receipt)
        publish(path, 'baseline_scoring_repair_receipt')
        status('COMPLETE' if report['ready_for_training'] else 'GATE_FAILED',
            repeatability_gate_passed=report['ready_for_training'], drive_replication='VERIFIED')
        publish(destination / 'coordinator_status.json', 'baseline_repair_status')
        return 0 if report['ready_for_training'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
