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
from shared_procedures.configuration import (configuration_hash, validate_configuration,
    sft_training_budget, sft_execution_selection)
from shared_procedures.identity import content_hash, file_hash
from shared_procedures.storage import atomic_write_json
from training.sft.data_handoff import (compatibility_projection, implementation_bindings, validate_handoff,
    training_amendment)
from training.sft.sft_training import build_worklist, select_worklist
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


def _named_node(source, name):
    for node in ast.parse(source).body:
        if getattr(node, 'name', None) == name:
            return node
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                return node
    raise ValueError('Original dataset-validation source is incomplete: ' + name)


def _unchanged_nodes(original, current, names):
    return all(ast.dump(_named_node(original, name), include_attributes=False)
               == ast.dump(_named_node(current, name), include_attributes=False) for name in names)


def _budget_source_compatible(original_fit, current_fit):
    """The budget extension may change recipes, never dataset validation/rendering.

    Frozen defaults, original spec fields/properties and every corpus-validation
    function remain identical. Serialized original identities are independently
    matched against all original certified tasks below.
    """
    names = {'dataset_hash', 'validate_datasets', 'MODELS', 'LEVELS', 'TARGET_MODULES',
             'LOSS_REDUCTION', 'IMPLEMENTATION_VERSION', '_hash',
             'render_sft_completion', 'tokenize_completion'}
    if not _unchanged_nodes(original_fit, current_fit, names):
        return False
    for name in ('SFTRecipe', 'SFTFitSpec'):
        old, new = _named_node(original_fit, name), _named_node(current_fit, name)
        old_fields = {node.target.id: ast.dump(node, include_attributes=False) for node in old.body
                      if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)}
        new_fields = {node.target.id: ast.dump(node, include_attributes=False) for node in new.body
                      if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)}
        if any(new_fields.get(key) != value for key, value in old_fields.items()):
            return False
        if name == 'SFTRecipe' and set(new_fields) != set(old_fields):
            return False
        if name == 'SFTFitSpec' and set(new_fields) - set(old_fields) not in (set(), {'training_budget'}):
            return False
        preserved = ('to_dict',) if name == 'SFTRecipe' else ('run_id', 'strength', 'model_name', 'from_dict')
        for method in preserved:
            a = next(node for node in old.body if getattr(node, 'name', None) == method)
            b = next(node for node in new.body if getattr(node, 'name', None) == method)
            if ast.dump(a, include_attributes=False) != ast.dump(b, include_attributes=False):
                return False
    return True


def _worklist_validation_compatible(original, current):
    """Permit only a named recipe argument and post-validation task selection."""
    if original == current:
        return True
    if not _unchanged_nodes(original, current, ('load_dataset_bundle',)):
        return False
    old = _named_node(original, 'build_worklist')
    new = _named_node(current, 'build_worklist')
    if ast.dump(old, include_attributes=False) == ast.dump(new, include_attributes=False):
        return True
    # Original cross-condition checks and full-count validation stay verbatim.
    for node in ast.walk(new):
        if isinstance(node, ast.ImportFrom) and node.module == 'shared_procedures.configuration':
            node.names = [alias for alias in node.names if alias.name != 'sft_training_budget']
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'SFTFitSpec':
            for keyword in list(node.keywords):
                if keyword.arg == 'training_budget':
                    expected = ast.parse('sft_training_budget(dict(config))', mode='eval').body
                    if ast.dump(keyword.value, include_attributes=False) != ast.dump(expected, include_attributes=False):
                        return False
                    node.keywords.remove(keyword)
        elif isinstance(node, ast.Return) and isinstance(node.value, ast.Call):
            expected = ast.parse('select_worklist(config, result)', mode='eval').body
            if ast.dump(node.value, include_attributes=False) == ast.dump(expected, include_attributes=False):
                node.value = ast.Name(id='result', ctx=ast.Load())
    return ast.dump(old, include_attributes=False) == ast.dump(new, include_attributes=False)


def _configuration_data_compatible(original, current):
    # Every original frozen field, identity exclusion, hashing and live source
    # prerequisite remains identical. Only the named budget/scope validator is new.
    return _unchanged_nodes(original, current, ('FIXED', 'MODEL_IDS', 'KAZITEXT_ALPHAS',
        'PROJECTION_TARGETS', 'OPERATIONAL_IDENTITY_EXCLUSIONS', 'scientific_configuration',
        'configuration_hash', 'snapshot_hash', 'runtime_requirements', '_reject_secrets', '_get'))


def _validate_bundle(bundle):
    if (bundle.get('simulation') is not False or bundle.get('status') != 'verified'
            or bundle.get('content_hash') != content_hash(bundle['datasets'])):
        raise ValueError('Certified bundle metadata or content integrity differs')


def reuse_certified_worklist(certificate, tasks, producer, consumer, bundle, audit, audit_hash):
    """Reuse validated record identities only after immutable bytes/source checks.

    The caller must still hash every audited corpus artifact before this path.
    Original row validation is reusable only when its exact implementation and
    all data bytes are unchanged. Named training budgets and execution scopes
    are amended explicitly; they do not alter the forty-condition corpus.
    """
    _validate_bundle(bundle)
    if (certificate.get('schema_version') != 'sft_producer_consumer_handoff_v1'
            or certificate.get('status') != 'PASSED'
            or certificate.get('certificate_hash') != content_hash({
                k: v for k, v in certificate.items() if k != 'certificate_hash'})
            or certificate.get('scientific_settings_unchanged') is not (
                training_amendment(certificate['producer_configuration'], certificate['consumer_configuration']) is None)
            or certificate.get('producer_records_modified') is not False
            or certificate.get('record_count') != 404000):
        raise ValueError('Original handoff certificate integrity differs')
    original = certificate['consumer_configuration']
    validate_configuration(original, runtime=True)
    old_amendment = training_amendment(certificate['producer_configuration'], original)
    if (old_amendment is not None and (
            certificate.get('consumer_training_amendment') != old_amendment
            or certificate.get('corpus_scientific_settings_unchanged') is not True
            or certificate.get('selected_conditions') != sft_execution_selection(original))):
        raise ValueError('Original handoff budget/scope amendment differs')
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
            or any((not name.startswith('src/training/sft/') or not name.endswith('.py'))
                   and name != 'requirements_sft_published.lock'
                   or Path(name).is_absolute() or '..' in Path(name).parts for name in extras)):
        raise ValueError('Original source binding inventory differs')
    for name, digest in bindings.items():
        if hashlib.sha256(_git_bytes(old_commit, name)).hexdigest() != digest:
            raise ValueError('Original consumer source bytes differ: ' + name)
    for name in ('src/shared_procedures/identity.py', 'requirements.lock'):
        if _git_bytes(old_commit, name) != (ROOT / name).read_bytes():
            raise ValueError('Certified dataset/compatibility implementation changed: ' + name)
    training_file = 'src/training/sft/sft_training.py'
    config_file = 'src/shared_procedures/configuration.py'
    if (not _worklist_validation_compatible(_git_bytes(old_commit, training_file), (ROOT/training_file).read_bytes())
            or not _configuration_data_compatible(_git_bytes(old_commit, config_file), (ROOT/config_file).read_bytes())):
        raise ValueError('Certified dataset/compatibility implementation changed')
    fit_file = 'src/training/sft/individual_fit/sft_fit.py'
    original_fit = _git_bytes(old_commit, fit_file)
    current_fit = (ROOT / fit_file).read_bytes()
    old_imports, current_imports = _module_imports(original_fit), _module_imports(current_fit)
    if ((_validation_nodes(original_fit) != _validation_nodes(current_fit)
             and not _budget_source_compatible(original_fit, current_fit))
            or any(current_imports.get(name) != binding for name, binding in old_imports.items())):
        raise ValueError('Certified dataset validation or rendering changed')
    expected_keys = [(f, m, l) for f in ('rot13', 'kazitext')
                     for m in ('7b', '14b') for l in range(1, 11)]
    datasets = {(d['family'], d['model_id'], d['level']): d for d in bundle['datasets']}
    conditions = {(d['family'], d['model_id'], d['level']): d for d in certificate['conditions']}
    old_selection = {(row['family'], row['model_id'], row['level']) for row in sft_execution_selection(original)}
    if (len(tasks) != len(old_selection) or len(bundle['datasets']) != 40 or len(certificate['conditions']) != 40
            or set(datasets) != set(expected_keys) or set(conditions) != set(expected_keys)):
        raise ValueError('Original handoff matrix is incomplete or duplicated')
    originals_by_key = {(task['family'], task['model_id'], task['level']): task for task in tasks}
    if set(originals_by_key) != old_selection:
        raise ValueError('Original certified worklist metadata differs')
    full_tasks = []
    for key in expected_keys:
        family, model, level = key
        model_config = original['models'][model]
        spec = SFTFitSpec(family, model, level, model_config['revision'],
                          model_config['tokenizer_revision'], certificate['consumer_configuration_hash'],
                          conditions[key]['dataset_hash'], certificate['producer_configuration_hash'],
                          certificate['certificate_hash'], training_budget=sft_training_budget(original))
        expected = {'spec': spec.identity(), 'dataset': datasets[key], 'family': family,
                    'model_id': model, 'level': level, 'run_id': spec.run_id,
                    'initialization': 'untouched_base_or_exact_full_checkpoint'}
        if ((key in old_selection and originals_by_key[key] != expected)
                or datasets[key].get('config_hash', datasets[key].get('configuration_hash')) != configuration_hash(producer)):
            raise ValueError('Original certified worklist metadata differs')
        full_tasks.append(expected)
    return deepcopy(full_tasks)


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
        'scientific_settings_unchanged':training_amendment(producer, consumer) is None,
        'corpus_scientific_settings_unchanged':True, 'producer_records_modified':False,
        'consumer_training_amendment':training_amendment(producer, consumer),
        'selected_conditions':sft_execution_selection(consumer),
        'scope':'Data compatibility only; does not replace baseline repeatability or launch preflights.'}
    certificate['certificate_hash'] = content_hash(certificate)
    validate_handoff(certificate, consumer)
    atomic_write_json(Path(args.output)/'certificate.json', certificate)
    # Specs preserve original producer hashes while naming the actual consumer.
    for task in tasks:
        model_config = consumer['models'][task['model_id']]
        spec = SFTFitSpec(task['family'], task['model_id'], task['level'], model_config['revision'],
            model_config['tokenizer_revision'], certificate['consumer_configuration_hash'],
            task['spec']['dataset_hash'], certificate['producer_configuration_hash'], certificate['certificate_hash'],
            training_budget=sft_training_budget(consumer))
        task['spec'], task['run_id'] = spec.identity(), spec.run_id
        validate_handoff(certificate, consumer, spec)
    tasks = select_worklist(consumer, tasks)
    atomic_write_json(Path(args.output)/'worklist.json', tasks)
    print(json.dumps({'status':'PASSED','conditions':len(certificate['conditions']),
                      'selected_fits':len(tasks),'records':404000,
                      'certificate_hash':certificate['certificate_hash']}),flush=True)

if __name__ == '__main__':
    main()
