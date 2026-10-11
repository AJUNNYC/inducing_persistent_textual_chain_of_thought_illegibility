"""Negative integrity checks for operational reuse, with explicit tiny fixtures."""
from copy import deepcopy
from pathlib import Path
import pytest

from operations.collection_inputs import (
    SCIENTIFIC_MODULES, candidate_index, check_inventory, check_pinned_identities,
    check_source_record, function_digest, main, mbpp_image, validate_consumer_certificate,
)
from shared_procedures.identity import content_hash, file_hash
from shared_procedures.configuration import load_configuration, configuration_hash
from shared_procedures.storage import atomic_write_json
from data_generation_collection.sft_data.clean_trace_collection.candidate_manifest import prepare_candidate_manifest


class FixtureTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


def fixture_manifest():
    rows = [
        {'conversation_id': 'one', 'language': 'English', 'conversation': [{'role': 'user', 'content': 'First question'}]},
        {'conversation_id': 'duplicate', 'language': 'en', 'conversation': [{'role': 'user', 'content': ' FIRST  QUESTION '}]},
        {'conversation_id': 'language', 'language': 'French', 'conversation': [{'role': 'user', 'content': 'Other input'}]},
        {'conversation_id': 'overlap', 'language': 'English', 'conversation': [{'role': 'user', 'content': 'Benchmark input'}]},
    ]
    manifest = prepare_candidate_manifest(rows, {'7b': FixtureTokenizer(), '14b': FixtureTokenizer()},
                                          'a' * 40, ['Benchmark input'], authorized=True)
    return rows, manifest


def test_independent_source_audit_checks_all_fixture_decisions():
    rows, manifest = fixture_manifest()
    records = candidate_index(manifest, len(rows))
    seen = set()
    reasons = []
    for index, row in enumerate(rows):
        _, reason = check_source_record(records[index], row, index, {'7b': 2, '14b': 2},
                                        'a' * 40, seen, {'benchmark input'})
        reasons.append(reason)
    assert reasons == [None, 'normalized_duplicate', 'non_english_source_label', 'detectable_evaluation_overlap']


@pytest.mark.parametrize('field,value', [
    ('prompt', 'Tampered prompt'), ('source_content_hash', 'b' * 64),
    ('deduplication_identity', 'c' * 64), ('input_token_counts', {'7b': 3, '14b': 2}),
    ('source_revision', 'd' * 40), ('source_id', 'different_original'),
])
def test_source_prompt_hash_count_and_revision_tampering_rejected(field, value):
    rows, manifest = fixture_manifest()
    record = deepcopy(manifest['candidates'][0])
    record[field] = value
    with pytest.raises(RuntimeError, match='binding differs'):
        check_source_record(record, rows[0], 0, {'7b': 2, '14b': 2}, 'a' * 40, set(), set())


def test_recorded_counts_cannot_substitute_for_independent_native_counts():
    rows, manifest = fixture_manifest()
    with pytest.raises(RuntimeError, match='binding differs'):
        check_source_record(manifest['candidates'][0], rows[0], 0, {'7b': 201, '14b': 2}, 'a' * 40, set(), set())


@pytest.mark.parametrize('mutation', ['partition', 'exclusion_order', 'missing', 'collision'])
def test_complete_population_partition_and_order_are_required(mutation):
    _, manifest = fixture_manifest()
    if mutation == 'partition':
        manifest['candidates'].append(manifest['exclusions'].pop())
        manifest['count'] += 1
    elif mutation == 'exclusion_order':
        manifest['exclusions'].reverse()
    elif mutation == 'missing':
        manifest['exclusions'].pop()
    else:
        manifest['exclusions'][0]['source_index'] = manifest['exclusions'][1]['source_index']
    with pytest.raises(RuntimeError, match='partition|ordering|coverage|incomplete'):
        candidate_index(manifest, 4)


def test_changed_source_or_tokenizer_pin_cannot_reuse_old_inputs():
    old = {'models': {'7b': {'tokenizer_revision': 'a' * 40}}, 'datasets': {'lmsys': {'revision': 'b' * 40}}}
    for part, name, field in [('models', '7b', 'tokenizer_revision'), ('datasets', 'lmsys', 'revision')]:
        changed = deepcopy(old)
        changed[part][name][field] = 'c' * 40
        with pytest.raises(RuntimeError, match='Pinned scientific'):
            check_pinned_identities(changed, old)


def minimal_inventory(root):
    paths = ['data/candidates/manifest.json', 'data/evaluation/capability_panel.json',
             'data/evaluation/legibility_subset.json', 'data/evaluation/official_pools.json',
             'evidence/mbpp_reference_tests.json', 'preparation_evidence.json']
    files = []
    for relative in paths:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'explicit small integrity fixture')
        files.append({'relative_path': relative, 'sha256': file_hash(path), 'size_bytes': path.stat().st_size})
    return {'schema_version': 'immutable_cpu_prerequisite_inventory_v1', 'status': 'verified_cpu_prerequisites',
            'files': files, 'scientific_module_blobs': [{'path': path, 'sha256': 'a' * 64} for path in sorted(SCIENTIFIC_MODULES)]}


def test_restored_inventory_detects_corruption_and_escape(tmp_path):
    root = tmp_path / 'restored'
    inventory = minimal_inventory(root)
    check_inventory(root, inventory)
    (root / inventory['files'][0]['relative_path']).write_bytes(b'corruption')
    with pytest.raises(RuntimeError, match='bytes differ'):
        check_inventory(root, inventory)
    inventory['files'][0]['relative_path'] = '../outside'
    with pytest.raises(RuntimeError, match='unsafe'):
        check_inventory(root, inventory)


def test_scientific_blob_inventory_cannot_omit_the_candidate_constructor(tmp_path):
    inventory = minimal_inventory(tmp_path)
    inventory['scientific_module_blobs'].pop()
    with pytest.raises(RuntimeError, match='Scientific producer inventory'):
        check_inventory(tmp_path, inventory)


def test_configured_mbpp_image_and_storage_function_identity_are_bound():
    assert mbpp_image({'operations': {'mbpp_image': 'operational-image'}}) == 'operational-image'
    assert mbpp_image({'evaluation': {'mbpp': {'image': 'selected-image'}},
                       'operations': {'mbpp_image': 'other-image'}}) == 'selected-image'
    assert function_digest('def read_json(path):\n return path\n', 'read_json') != function_digest(
        'def read_json(path):\n return None\n', 'read_json')


def test_existing_validation_namespace_is_never_modified(tmp_path):
    existing = tmp_path / 'prior'
    existing.mkdir()
    retained = existing / 'certificate.json'
    retained.write_text('immutable original fixture')
    before = retained.read_bytes()
    code = main(['--consumer-commit', 'a' * 40, '--config', str(tmp_path / 'unused.yaml'),
                 '--producer-root', str(tmp_path / 'unused_inputs'), '--output-dir', str(existing),
                 '--source-evidence', str(tmp_path / 'unused_native.json'),
                 '--canonical-source-evidence', str(tmp_path / 'unused_canonical.json'),
                 '--asset-cache', str(tmp_path / 'unused_cache')])
    assert code == 1 and retained.read_bytes() == before
    assert list(existing.iterdir()) == [retained]


@pytest.mark.parametrize('mutation', ['integrity', 'consumer_source', 'validator_source'])
def test_rehashed_or_tampered_certificate_cannot_bypass_consumer_identity(tmp_path, mutation):
    config = load_configuration(Path(__file__).resolve().parents[3] / 'config.yaml')
    config['runtime']['implementation_revision'] = 'a' * 40
    # Deliberately incomplete tiny evidence reaches the identity rejection only.
    certificate = {'schema_version': 'cpu_prerequisite_consumer_validation_v1',
                   'status': 'verified_complete_consumer_validation', 'consumer_commit': 'a' * 40,
                   'consumer_configuration_hash': configuration_hash(config),
                   'validator_implementation_sha256': 'b' * 64}
    certificate['certificate_hash'] = content_hash(certificate)
    if mutation == 'integrity':
        certificate['consumer_commit'] = 'c' * 40
        expected = 'integrity'
    elif mutation == 'consumer_source':
        certificate['consumer_commit'] = 'c' * 40
        certificate['certificate_hash'] = content_hash({k: v for k, v in certificate.items() if k != 'certificate_hash'})
        expected = 'identity'
    else:
        expected = 'implementation'
    path = tmp_path / 'incomplete_certificate.json'
    atomic_write_json(path, certificate)
    with pytest.raises(RuntimeError, match=expected):
        validate_consumer_certificate(path, config, consumer_commit='a' * 40,
                                      input_root=tmp_path / 'unused_inputs', asset_cache=tmp_path / 'unused_cache')
