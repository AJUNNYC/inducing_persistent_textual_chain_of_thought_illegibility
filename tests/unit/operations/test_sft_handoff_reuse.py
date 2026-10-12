"""Reuse cannot alter certified conditions or bypass immutable source evidence."""
from copy import deepcopy
import hashlib
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
import certify_sft_handoff as handoff
from shared_procedures.configuration import FIXED, configuration_hash
from shared_procedures.identity import content_hash, file_hash
from training.sft.individual_fit.sft_fit import MODELS, SFTFitSpec


def fixture(monkeypatch):
    producer = {}
    for key, value in FIXED.items():
        target = producer
        parts = key.split('.')
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = deepcopy(value)
    producer['models'] = {m: {'id': n, 'revision': 'a'*40, 'tokenizer_revision': 'b'*40}
                          for m, n in MODELS.items()}
    for dataset in producer['datasets'].values():
        dataset['revision'] = 'd'*40
    producer['interventions']['kazitext'].update(udpipe_sha256='e'*64,
        aspell_dictionary_provenance={'settings': {'language': 'en'},
                                     'files': [{'name': 'dictionary', 'sha256': 'f'*64}]})
    producer['runtime'] = {'implementation_revision': '1'*40}
    old = deepcopy(producer)
    old['runtime'].update(implementation_revision='2'*40, dependency_lock_hash=file_hash(ROOT/'requirements.lock'))
    consumer = deepcopy(old)
    consumer['runtime']['implementation_revision'] = '3'*40
    audit = {'index_file_id': 'unit-fixture', 'artifacts': []}
    bundle = {'simulation': False, 'status': 'verified', 'datasets': []}
    conditions = []
    for family in ('rot13', 'kazitext'):
        for model in MODELS:
            for level in range(1, 11):
                row = {'family': family, 'model_id': model, 'level': level}
                bundle['datasets'].append({**row, 'config_hash': configuration_hash(producer)})
                conditions.append({**row, 'dataset_hash': '9'*64})
    bundle['content_hash'] = content_hash(bundle['datasets'])
    bindings = handoff.implementation_bindings()
    sources = {name: (ROOT/name).read_bytes() for name in bindings}
    monkeypatch.setattr(handoff, '_git_bytes', lambda commit, name: sources[name])
    cert = {'schema_version': 'sft_producer_consumer_handoff_v1', 'status': 'PASSED',
        'producer_configuration': producer, 'consumer_configuration': old,
        'producer_configuration_hash': configuration_hash(producer),
        'consumer_configuration_hash': configuration_hash(old),
        'producer_commit': '1'*40, 'consumer_commit': '2'*40,
        'implementation_bindings': bindings, 'conditions': conditions,
        'drive_readback_report_sha256': '7'*64, 'drive_index_id': audit['index_file_id'],
        'verified_artifacts': audit['artifacts'], 'bundle_hash': bundle['content_hash'],
        'scientific_settings_unchanged': True, 'producer_records_modified': False, 'record_count': 404000}
    cert['certificate_hash'] = content_hash(cert)
    tasks = []
    for row, dataset in zip(conditions, bundle['datasets']):
        spec = SFTFitSpec(row['family'], row['model_id'], row['level'], 'a'*40, 'b'*40,
            configuration_hash(old), row['dataset_hash'], configuration_hash(producer), cert['certificate_hash'])
        tasks.append({'spec': spec.identity(), 'dataset': deepcopy(dataset), 'family': row['family'],
            'model_id': row['model_id'], 'level': row['level'], 'run_id': spec.run_id,
            'initialization': 'untouched_base_or_exact_full_checkpoint'})
    return cert, tasks, producer, consumer, bundle, audit, '7'*64, sources


def test_unchanged_certified_identity_reuses_only_copied_worklist(monkeypatch):
    values = fixture(monkeypatch)
    result = handoff.reuse_certified_worklist(*values[:-1])
    assert result == values[1]
    result[0]['spec']['dataset_hash'] = '0'*64
    assert values[1][0]['spec']['dataset_hash'] == '9'*64


@pytest.mark.parametrize('field,value', [
    ('family', 'kazitext'), ('model_id', '14b'), ('level', 2),
    ('run_id', 'wrong'), ('initialization', 'old_adapter')])
def test_task_metadata_tampering_is_rejected(monkeypatch, field, value):
    values = fixture(monkeypatch)
    values[1][0][field] = value
    with pytest.raises(ValueError, match='worklist metadata'):
        handoff.reuse_certified_worklist(*values[:-1])


@pytest.mark.parametrize('field,value', [
    ('dataset_hash', '0'*64), ('model_revision', 'f'*40), ('tokenizer_revision', 'f'*40),
    ('configuration_hash', '0'*64), ('producer_configuration_hash', '0'*64),
    ('handoff_certificate_hash', '0'*64), ('strength', 9), ('recipe', {})])
def test_spec_identity_tampering_is_rejected(monkeypatch, field, value):
    values = fixture(monkeypatch)
    values[1][0]['spec'][field] = value
    with pytest.raises(ValueError, match='worklist metadata'):
        handoff.reuse_certified_worklist(*values[:-1])


@pytest.mark.parametrize('mutation', ['integrity', 'audit', 'bundle', 'source', 'matrix', 'dataset'])
def test_original_certificate_data_and_source_binding_are_required(monkeypatch, mutation):
    values = fixture(monkeypatch)
    cert, tasks, _, _, bundle, audit, _, sources = values
    if mutation == 'integrity':
        cert['certificate_hash'] = '0'*64
    elif mutation == 'audit':
        audit['index_file_id'] = 'another-index'
    elif mutation == 'bundle':
        bundle['content_hash'] = '0'*64
    elif mutation == 'source':
        sources['src/training/sft/sft_training.py'] += b'\n# changed\n'
    elif mutation == 'matrix':
        tasks[1] = deepcopy(tasks[0])
    else:
        tasks[0]['dataset']['config_hash'] = '0'*64
    with pytest.raises(ValueError):
        handoff.reuse_certified_worklist(*values[:-1])


def test_validation_ast_change_rejected_even_with_correct_original_source_hash(monkeypatch):
    values = fixture(monkeypatch)
    cert, _, _, _, _, _, _, sources = values
    name = 'src/training/sft/individual_fit/sft_fit.py'
    sources[name] = sources[name].replace(b'if len(train_records) != 10000', b'if len(train_records) != 9999')
    cert['implementation_bindings'][name] = hashlib.sha256(sources[name]).hexdigest()
    cert['certificate_hash'] = content_hash({k:v for k,v in cert.items() if k != 'certificate_hash'})
    with pytest.raises(ValueError, match='validation or rendering'):
        handoff.reuse_certified_worklist(*values[:-1])


def test_git_source_requires_immutable_commit():
    with pytest.raises(ValueError, match='immutable'):
        handoff._git_bytes('main', 'requirements.lock')


@pytest.mark.parametrize('mutation', ['stale_hash', 'simulation', 'unverified'])
def test_bundle_integrity_and_native_verified_status_are_required(monkeypatch, mutation):
    values = fixture(monkeypatch)
    bundle = values[4]
    if mutation == 'stale_hash':
        bundle['datasets'][0]['new_metadata'] = 'mutated with stale content_hash'
    elif mutation == 'simulation':
        bundle['simulation'] = True
    else:
        bundle['status'] = 'complete'
    with pytest.raises(ValueError, match='bundle metadata or content integrity'):
        handoff.reuse_certified_worklist(*values[:-1])


def test_new_runtime_helper_source_binding_is_allowed(monkeypatch):
    values = fixture(monkeypatch)
    bindings = handoff.implementation_bindings()
    bindings['src/training/sft/new_runtime_helper.py'] = '0'*64
    monkeypatch.setattr(handoff, 'implementation_bindings', lambda: bindings)
    assert len(handoff.reuse_certified_worklist(*values[:-1])) == 40


@pytest.mark.parametrize('name', ['src/shared_procedures/new_dependency.py',
                                 'src/training/sft/../../outside.py',
                                 'src/training/sft/runtime_helper.so'])
def test_new_nonruntime_binding_is_rejected(monkeypatch, name):
    values = fixture(monkeypatch)
    bindings = handoff.implementation_bindings()
    bindings[name] = '0'*64
    monkeypatch.setattr(handoff, 'implementation_bindings', lambda: bindings)
    with pytest.raises(ValueError, match='binding inventory'):
        handoff.reuse_certified_worklist(*values[:-1])


def test_original_binding_removal_is_rejected(monkeypatch):
    values = fixture(monkeypatch)
    bindings = handoff.implementation_bindings()
    bindings.pop('src/training/sft/execution_profile.py')
    monkeypatch.setattr(handoff, 'implementation_bindings', lambda: bindings)
    with pytest.raises(ValueError, match='binding inventory'):
        handoff.reuse_certified_worklist(*values[:-1])


def test_original_import_redirection_rejected_despite_unchanged_validation_ast(monkeypatch):
    values = fixture(monkeypatch)
    cert, _, _, _, _, _, _, sources = values
    name = 'src/training/sft/individual_fit/sft_fit.py'
    sources[name] = sources[name].replace(b'import hashlib\n', b'import wrong_hashlib as hashlib\n')
    cert['implementation_bindings'][name] = hashlib.sha256(sources[name]).hexdigest()
    cert['certificate_hash'] = content_hash({k:v for k,v in cert.items() if k != 'certificate_hash'})
    with pytest.raises(ValueError, match='validation or rendering'):
        handoff.reuse_certified_worklist(*values[:-1])


def test_new_unique_runtime_import_does_not_invalidate_original_eligibility(monkeypatch):
    values = fixture(monkeypatch)
    cert, _, _, _, _, _, _, sources = values
    name = 'src/training/sft/individual_fit/sft_fit.py'
    # The current module adds a unique runtime-only import relative to the
    # fixture's original module, while preserving all original import names.
    sources[name] = sources[name].replace(b'import time\n', b'')
    cert['implementation_bindings'][name] = hashlib.sha256(sources[name]).hexdigest()
    cert['certificate_hash'] = content_hash({k:v for k,v in cert.items() if k != 'certificate_hash'})
    for task in values[1]:
        task['spec']['handoff_certificate_hash'] = cert['certificate_hash']
    assert len(handoff.reuse_certified_worklist(*values[:-1])) == 40
