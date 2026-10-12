#!/usr/bin/env python3
"""Certify the existing Drive-audited corpus for an explicitly frozen consumer.

Reuses completed Drive readback; verifies current local bytes and production
worklist invariants. No generation, transformation, GPU or remote Pod required.
"""
import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from shared_procedures.configuration import configuration_hash, validate_configuration
from shared_procedures.identity import content_hash, file_hash
from shared_procedures.storage import atomic_write_json
from training.sft.data_handoff import compatibility_projection, implementation_bindings, validate_handoff
from training.sft.sft_training import build_worklist
from training.sft.individual_fit.sft_fit import SFTFitSpec


def _git_bytes(commit, name):
    if (not isinstance(commit, str) or len(commit) != 40
            or any(c not in '0123456789abcdef' for c in commit)):
        raise ValueError('Reuse requires an immutable original consumer commit')
    path = Path(name)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Unsafe original source binding')
    return subprocess.check_output(['git', 'show', commit + ':' + name], cwd=ROOT)


def _validation_nodes(source):
    names = {'SFTFitSpec', 'SFTRecipe', 'dataset_hash', 'validate_datasets',
             'MODELS', 'LEVELS', 'TARGET_MODULES', 'LOSS_REDUCTION',
             'IMPLEMENTATION_VERSION', '_hash', 'render_sft_completion', 'tokenize_completion'}
    result = {}
    for node in ast.parse(source).body:
        identifiers = {getattr(node, 'name', '')}
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            identifiers.update(t.id for t in targets if isinstance(t, ast.Name))
        for name in identifiers & names:
            result[name] = ast.dump(node, include_attributes=False)
    if result.keys() != names:
        raise ValueError('Original dataset-validation source is incomplete')
    return result


def _module_imports(source):
    """Resolve original module-level import names without banning new helpers."""
    result = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split('.')[0]
                result[bound] = ('import', alias.name if alias.asname else bound)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                result[alias.asname or alias.name] = ('from', node.level, node.module, alias.name)
    return result


def _validate_bundle(bundle):
    if (bundle.get('simulation') is not False or bundle.get('status') != 'verified'
            or bundle.get('content_hash') != content_hash(bundle['datasets'])):
        raise ValueError('Certified bundle metadata or content integrity differs')


def reuse_certified_worklist(certificate, tasks, producer, consumer, bundle, audit, audit_hash):
    """Reuse validated record identities only after immutable bytes/source checks.

    The caller must still hash every audited corpus artifact before this path.
    Original row validation is reusable only when its exact implementation,
    configurations, all data bytes, and every scheduled condition are unchanged.
    """
    _validate_bundle(bundle)
    if (certificate.get('schema_version') != 'sft_producer_consumer_handoff_v1'
            or certificate.get('status') != 'PASSED'
            or certificate.get('certificate_hash') != content_hash({
                k: v for k, v in certificate.items() if k != 'certificate_hash'})
            or certificate.get('scientific_settings_unchanged') is not True
            or certificate.get('producer_records_modified') is not False
            or certificate.get('record_count') != 404000):
        raise ValueError('Original handoff certificate integrity differs')
    original = certificate['consumer_configuration']
    validate_configuration(original, runtime=True)
    old_commit = certificate['consumer_commit']
    if (old_commit != original['runtime']['implementation_revision']
            or certificate['producer_commit'] != producer['runtime']['implementation_revision']
            or certificate['producer_configuration'] != producer
            or certificate['producer_configuration_hash'] != configuration_hash(producer)
            or certificate['consumer_configuration_hash'] != configuration_hash(original)
            or compatibility_projection(original) != compatibility_projection(consumer)
            or compatibility_projection(producer) != compatibility_projection(original)
            or certificate['drive_readback_report_sha256'] != audit_hash
            or certificate['drive_index_id'] != audit['index_file_id']
            or certificate['verified_artifacts'] != audit['artifacts']
            or certificate['bundle_hash'] != bundle['content_hash']):
        raise ValueError('Original handoff scientific/data binding differs')
    bindings = certificate.get('implementation_bindings', {})
    current_bindings = implementation_bindings()
    extras = current_bindings.keys() - bindings.keys()
    if (not bindings.keys() <= current_bindings.keys()
            or any(not name.startswith('src/training/sft/') or not name.endswith('.py')
                   or Path(name).is_absolute() or '..' in Path(name).parts for name in extras)):
        raise ValueError('Original source binding inventory differs')
    for name, digest in bindings.items():
        if hashlib.sha256(_git_bytes(old_commit, name)).hexdigest() != digest:
            raise ValueError('Original consumer source bytes differ: ' + name)
    for name in ('src/training/sft/sft_training.py', 'src/training/sft/data_handoff.py',
                 'src/shared_procedures/configuration.py', 'src/shared_procedures/identity.py',
                 'requirements.lock'):
        if _git_bytes(old_commit, name) != (ROOT / name).read_bytes():
            raise ValueError('Certified dataset/compatibility implementation changed: ' + name)
    fit_file = 'src/training/sft/individual_fit/sft_fit.py'
    original_fit = _git_bytes(old_commit, fit_file)
    current_fit = (ROOT / fit_file).read_bytes()
    old_imports, current_imports = _module_imports(original_fit), _module_imports(current_fit)
    if (_validation_nodes(original_fit) != _validation_nodes(current_fit)
            or any(current_imports.get(name) != binding for name, binding in old_imports.items())):
        raise ValueError('Certified dataset validation or rendering changed')
    expected_keys = [(f, m, l) for f in ('rot13', 'kazitext')
                     for m in ('7b', '14b') for l in range(1, 11)]
    datasets = {(d['family'], d['model_id'], d['level']): d for d in bundle['datasets']}
    conditions = {(d['family'], d['model_id'], d['level']): d for d in certificate['conditions']}
    if (len(tasks) != 40 or len(bundle['datasets']) != 40 or len(certificate['conditions']) != 40
            or set(datasets) != set(expected_keys) or set(conditions) != set(expected_keys)):
        raise ValueError('Original handoff matrix is incomplete or duplicated')
    for task, key in zip(tasks, expected_keys):
        family, model, level = key
        model_config = original['models'][model]
        spec = SFTFitSpec(family, model, level, model_config['revision'],
                          model_config['tokenizer_revision'], certificate['consumer_configuration_hash'],
                          conditions[key]['dataset_hash'], certificate['producer_configuration_hash'],
                          certificate['certificate_hash'])
        expected = {'spec': spec.identity(), 'dataset': datasets[key], 'family': family,
                    'model_id': model, 'level': level, 'run_id': spec.run_id,
                    'initialization': 'untouched_base_or_exact_full_checkpoint'}
        if task != expected or datasets[key].get('config_hash', datasets[key].get('configuration_hash')) != configuration_hash(producer):
            raise ValueError('Original certified worklist metadata differs')
    return deepcopy(tasks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('producer-config', 'consumer-config', 'audit', 'bundle', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--reuse-certificate')
    parser.add_argument('--reuse-worklist')
    args = parser.parse_args()
    if bool(args.reuse_certificate) != bool(args.reuse_worklist):
        parser.error('--reuse-certificate and --reuse-worklist must be supplied together')
    producer = json.loads(Path(args.producer_config).read_text())
    consumer = json.loads(Path(args.consumer_config).read_text())
    validate_configuration(consumer, runtime=True)
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if consumer['runtime']['implementation_revision'] != commit:
        raise ValueError('Consumer must name the actual frozen source commit')
    if compatibility_projection(producer) != compatibility_projection(consumer):
        raise ValueError('Scientific settings changed between producer and consumer')
    audit = json.loads(Path(args.audit).read_text())
    bundle = json.loads(Path(args.bundle).read_text())
    _validate_bundle(bundle)
    if (audit['status'] != 'PASSED' or audit['verified_files'] != 122
            or audit['validation']['record_count'] != 404000
            or audit['generation_commit'] != producer['runtime']['implementation_revision']
            or bundle['config_hash'] != configuration_hash(producer)):
        raise ValueError('Original Drive readback/source binding differs')
    if len(audit['artifacts']) != 122 or len({entry['path'] for entry in audit['artifacts']}) != 122:
        raise ValueError('All 122 distinct audited artifacts are required')
    for entry in audit['artifacts']:
        relative = Path(entry['path'])
        if relative.is_absolute() or '..' in relative.parts or not str(relative).startswith('data/vllm_sft_preparation/'):
            raise ValueError('Unsafe audit path')
        path = Path('/workspace')/relative
        if path.stat().st_size != entry['bytes'] or file_hash(path) != entry['sha256']:
            raise ValueError('Current corpus bytes differ from Drive readback')
    if Path(args.bundle).resolve() != Path('/workspace/data/vllm_sft_preparation/transformed/manifest.json'):
        raise ValueError('Bundle must be the independently audited corpus manifest')
    print('122 corpus artifacts match completed Drive readback', flush=True)
    if args.reuse_certificate:
        tasks = reuse_certified_worklist(json.loads(Path(args.reuse_certificate).read_text()),
            json.loads(Path(args.reuse_worklist).read_text()), producer, consumer,
            bundle, audit, file_hash(args.audit))
        print('Unchanged certified record identities reused after full byte/source verification', flush=True)
    else:
        tasks = build_worklist(producer, bundle['datasets'])
    certificate = {
        'schema_version':'sft_producer_consumer_handoff_v1', 'status':'PASSED',
        'checked_at':datetime.now(timezone.utc).isoformat(),
        'producer_configuration':producer, 'consumer_configuration':consumer,
        'producer_configuration_hash':configuration_hash(producer),
        'consumer_configuration_hash':configuration_hash(consumer),
        'producer_commit':producer['runtime']['implementation_revision'], 'consumer_commit':commit,
        'implementation_bindings':implementation_bindings(),
        'drive_readback_report_sha256':file_hash(args.audit),
        'drive_index_id':audit['index_file_id'], 'bundle_hash':bundle['content_hash'],
        'verified_artifacts':audit['artifacts'], 'record_count':404000,
        'conditions':[{k:t['spec'][k] for k in ('family','model_id','level','dataset_hash')} for t in tasks],
        'scientific_settings_unchanged':True, 'producer_records_modified':False,
        'scope':'Data compatibility only; does not replace baseline repeatability or launch preflights.'}
    certificate['certificate_hash'] = content_hash(certificate)
    validate_handoff(certificate, consumer)
    atomic_write_json(Path(args.output)/'certificate.json', certificate)
    # Specs preserve original producer hashes while naming the actual consumer.
    for task in tasks:
        task['spec'].update(configuration_hash=certificate['consumer_configuration_hash'],
                            producer_configuration_hash=certificate['producer_configuration_hash'],
                            handoff_certificate_hash=certificate['certificate_hash'])
        validate_handoff(certificate, consumer, SFTFitSpec.from_dict(task['spec']))
    atomic_write_json(Path(args.output)/'worklist.json', tasks)
    print(json.dumps({'status':'PASSED','conditions':len(tasks),'records':404000,
                      'certificate_hash':certificate['certificate_hash']}),flush=True)

if __name__ == '__main__':
    main()
