"""Full, offline consumer validation of immutable originally produced artifacts.

This creates a validation certificate only. It never changes producer metadata,
scientific payloads, authorization flags, stage states, or collection outputs.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

DEFAULT_PROJECT = Path(__file__).resolve().parents[2]
MODULE_SCHEMA = 'cpu_prerequisite_consumer_validation_v1'
SCIENTIFIC_MODULES = {
    'src/data_generation_collection/sft_data/clean_trace_collection/candidate_manifest.py',
    'src/data_generation_collection/evaluation_data/capability_panel/capability_panel_construction.py',
    'src/data_generation_collection/evaluation_data/legibility_subset/legibility_subset_construction.py',
    'src/data_generation_collection/evaluation_data/evaluation_data_preparation.py',
    'src/evaluation/shared_model_evaluation/item_evaluation/capability_scoring/mbpp_sandbox.py',
    'src/operations/assets.py', 'src/shared_procedures/identity.py', 'src/shared_procedures/response_parsing.py',
}

from shared_procedures.configuration import load_configuration, configuration_hash
from shared_procedures.identity import content_hash, derive_seed, file_hash
from shared_procedures.storage import atomic_write_json, read_json
from operations.assets import model_load_kwargs, validate_asset
from data_generation_collection.sft_data.clean_trace_collection.candidate_manifest import (
    load_candidate_manifest, normalize_prompt, SOURCE, NORMALIZATION_VERSION, EXTRACTION_VERSION,
)
from data_generation_collection.evaluation_data.capability_panel.capability_panel_construction import (
    construct_capability_panel, validate_capability_panel,
)
from data_generation_collection.evaluation_data.legibility_subset.legibility_subset_construction import (
    construct_legibility_subset, validate_legibility_subset,
)
from evaluation.shared_model_evaluation.item_evaluation.capability_scoring.mbpp_sandbox import (
    verified_reference_callback, DEFAULT_MBPP_IMAGE,
)


def utc():
    return datetime.now(timezone.utc).isoformat()


def require(condition, description):
    if not condition:
        raise RuntimeError(description)


def function_digest(body, name):
    tree = ast.parse(body)
    matches = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]
    require(len(matches) == 1, 'Scientific helper definition differs')
    return hashlib.sha256(ast.dump(matches[0], include_attributes=False).encode()).hexdigest()


def mbpp_image(config):
    return (config.get('evaluation', {}).get('mbpp', {}).get('image')
            or config.get('operations', {}).get('mbpp_image') or DEFAULT_MBPP_IMAGE)


def check_inventory(root, inventory):
    """Recheck every immutable byte without trusting absolute acquisition paths."""
    root = Path(root).resolve()
    require(inventory.get('schema_version') == 'immutable_cpu_prerequisite_inventory_v1'
            and inventory.get('status') == 'verified_cpu_prerequisites', 'Producer inventory is incomplete')
    paths = set()
    for item in inventory.get('files', []):
        relative = Path(item['relative_path'])
        path = root / relative
        require(not relative.is_absolute() and '..' not in relative.parts
                and relative.as_posix() not in paths and path.resolve().is_relative_to(root)
                and path.is_file() and not path.is_symlink(), 'Producer inventory path is unsafe')
        require(path.stat().st_size == item['size_bytes'] and file_hash(path) == item['sha256'],
                'Original artifact bytes differ')
        paths.add(relative.as_posix())
    required = {'data/candidates/manifest.json', 'data/evaluation/capability_panel.json',
                'data/evaluation/legibility_subset.json', 'data/evaluation/official_pools.json',
                'evidence/mbpp_reference_tests.json', 'preparation_evidence.json'}
    require(required <= paths, 'Producer inventory lacks required inputs')
    blobs = inventory.get('scientific_module_blobs', [])
    require(len({row['path'] for row in blobs}) == len(blobs)
            and SCIENTIFIC_MODULES <= {row['path'] for row in blobs}
            and {row['path'] for row in blobs} <= SCIENTIFIC_MODULES | {
                'src/shared_procedures/configuration.py', 'src/shared_procedures/storage.py'},
            'Scientific producer inventory is incomplete')
    return inventory


def check_pinned_identities(config, old_config):
    require(config['models'] == old_config['models'] and config['datasets'] == old_config['datasets'],
            'Pinned scientific input identities differ')


def tokenizer_bindings(config, asset_cache):
    """Verify pinned tokenizer files without invoking Hub metadata or model loading."""
    root = Path(asset_cache).resolve()
    bindings = {}
    for size, spec in config['models'].items():
        relative = Path('models/by_repository') / spec['id'].replace('/', '--') / spec['tokenizer_revision'] / 'tokenizer'
        manifest = validate_asset(root / relative)
        require(manifest.get('repo_id') == spec['id'] and manifest.get('selected_revision') == spec['tokenizer_revision']
                and manifest.get('variant') == 'tokenizer', 'Pinned tokenizer cache binding differs')
        bindings[size] = {'relative_path': (relative / 'asset_manifest.json').as_posix(),
                          'sha256': file_hash(root / relative / 'asset_manifest.json'),
                          'manifest_hash': manifest['manifest_hash']}
    require(set(bindings) == {'7b', '14b'}, 'Both pinned tokenizer caches are required')
    return bindings


def validate_consumer_certificate(certificate_path, configuration, *, consumer_commit, input_root, asset_cache):
    """Validate prior full-audit evidence and its exact current bytes for stage reuse.

    All checks are local. This never starts an upstream fetch, Docker process,
    native tokenizer computation, or paid work. Absence or mismatch fails closed.
    Paths can move after verified Drive restoration; immutable bytes cannot.
    """
    certificate = read_json(certificate_path)
    require(certificate.get('certificate_hash') == content_hash({key: value for key, value in certificate.items()
                                                               if key != 'certificate_hash'}), 'Consumer certificate integrity differs')
    require(certificate.get('schema_version') == MODULE_SCHEMA
            and certificate.get('status') == 'verified_complete_consumer_validation', 'Complete consumer validation is absent')
    require(certificate.get('consumer_commit') == consumer_commit == configuration['runtime']['implementation_revision']
            and certificate.get('consumer_configuration_hash') == configuration_hash(configuration), 'Consumer certificate identity differs')
    require(certificate.get('validator_implementation_sha256') == file_hash(__file__), 'Consumer validator implementation differs')
    require(configuration.get('operations', {}).get('lmsys_authorization') is True, 'Actual LMSYS license authorization is absent')
    require(certificate.get('pinned_models') == configuration['models']
            and certificate.get('pinned_datasets') == configuration['datasets'], 'Pinned scientific input identities differ')
    require(certificate.get('configured_mbpp_image') == mbpp_image(configuration), 'Configured MBPP sandbox differs')
    require(certificate.get('source_rows_independently_checked') == 1000000
            and certificate.get('source_shards_independently_reverified') == 6
            and certificate.get('capability_count') == 1233 and certificate.get('legibility_count') == 200
            and type(certificate.get('accepted_count')) is int and type(certificate.get('excluded_count')) is int
            and certificate['accepted_count'] + certificate['excluded_count'] == 1000000
            and sum(certificate.get('exclusion_counts', {}).values()) == certificate['excluded_count'],
            'Consumer full-population validation is incomplete')
    for key in ('both_native_tokenizers_rechecked_for_all_first_user_turns', 'current_mbpp_sandbox_binding_verified',
                'original_artifact_bytes_preserved', 'original_producer_metadata_preserved', 'configuration_flags_unchanged'):
        require(certificate.get(key) is True, 'Consumer validation checks are incomplete')
    for key in ('gpu_stage_statuses_written', 'scientific_model_outputs_generated', 'generation_ready',
                'credential_values_logged', 'source_content_logged'):
        require(certificate.get(key) is False, 'Consumer validation scope differs')
    root = Path(input_root).resolve()
    inventory_path = root / 'immutable_prerequisite_inventory.json'
    require(file_hash(inventory_path) == certificate.get('original_inventory_sha256'), 'Original inventory binding differs')
    inventory = check_inventory(root, read_json(inventory_path))
    require(certificate.get('original_inventory') == inventory
            and certificate.get('input_files') == inventory['files']
            and certificate.get('original_producer_commit') == inventory['original_producer_commit']
            and certificate.get('original_configuration_hash') == inventory['scientific_configuration_hash']
            and certificate['accepted_count'] == inventory['candidate_count']
            and certificate['excluded_count'] == inventory['excluded_count'], 'Original producer evidence differs')
    original_evidence = read_json(root / 'preparation_evidence.json')
    require(original_evidence.get('status') == 'verified_cpu_prerequisites'
            and original_evidence.get('source_commit') == certificate['original_producer_commit']
            and original_evidence.get('candidate_manifest_hash') == certificate['candidate_manifest_content_hash']
            and original_evidence.get('candidate_file_sha256') == certificate['candidate_file_sha256']
            and original_evidence.get('candidate_file_sha256') == file_hash(root / 'data/candidates/manifest.json'),
            'Original candidate evidence differs')
    required_modules = [row for row in inventory['scientific_module_blobs']
                        if not row['path'].endswith(('configuration.py', 'storage.py'))]
    require(certificate.get('unchanged_scientific_modules') == [{'path': row['path'], 'sha256': row['sha256']}
                                                               for row in required_modules], 'Scientific blob evidence differs')
    for row in required_modules:
        require(file_hash(DEFAULT_PROJECT / row['path']) == row['sha256'], 'Current scientific module changed')
    storage = (DEFAULT_PROJECT / 'src/shared_procedures/storage.py').read_bytes()
    require(certificate.get('unchanged_storage_helpers') == {name: function_digest(storage, name)
                                                           for name in ('read_json', 'atomic_write_json')}, 'Scientific storage helper changed')
    require(certificate.get('tokenizer_asset_bindings') == tokenizer_bindings(configuration, asset_cache),
            'Current tokenizer cache differs from the complete audit')
    panel = read_json(root / 'data/evaluation/capability_panel.json')
    subset = read_json(root / 'data/evaluation/legibility_subset.json')
    validate_capability_panel(panel, require_verified=True)
    validate_legibility_subset(subset, panel, require_verified=True)
    require(certificate['capability_hash'] == panel['content_hash'] and certificate['legibility_hash'] == subset['content_hash'],
            'Current panel payload differs')
    report = read_json(root / 'evidence/mbpp_reference_tests.json')
    require(report['source_file_sha256'] == file_hash(root / 'data/evaluation/official_pools.json')
            and report['report_hash'] == certificate['mbpp_reference_report_hash'], 'Current MBPP source evidence differs')
    verified_reference_callback(root / 'evidence/mbpp_reference_tests.json', panel=panel, image=mbpp_image(configuration))
    return certificate


def candidate_index(manifest, expected_count):
    """Require complete population coverage, exact partitions and prospective order."""
    records = [None] * expected_count
    require(manifest['count'] == len(manifest['candidates']), 'Accepted population differs')
    for partition, required_status in (('candidates', 'eligible'), ('exclusions', 'excluded')):
        for row in manifest[partition]:
            index = row.get('source_index')
            require(type(index) is int and 0 <= index < len(records) and records[index] is None,
                    'Source-index coverage differs')
            require(row.get('status') == required_status
                    and (('reason' in row and bool(row['reason'])) if partition == 'exclusions' else 'reason' not in row),
                    'Candidate/exclusion partition differs')
            records[index] = row
    require(all(row is not None for row in records), 'Source-index population is incomplete')
    require([row['source_index'] for row in manifest['exclusions']]
            == sorted(row['source_index'] for row in manifest['exclusions']), 'Excluded source ordering differs')
    shuffled = sorted(manifest['candidates'], key=lambda row: (derive_seed(42, 'candidates', row['example_id']), row['example_id']))
    require([row['example_id'] for row in shuffled] == [row['example_id'] for row in manifest['candidates']],
            'Prospective accepted ordering differs')
    require([row.get('candidate_order') for row in manifest['candidates']] == list(range(len(shuffled))),
            'Accepted position sequence differs')
    return records


def check_source_record(recorded, row, source_index, counts, revision, seen, overlaps):
    """Compare one original source record against independently measured token counts."""
    source_id = str(row.get('conversation_id', row.get('id', source_index)))
    text = next((turn.get('content') for turn in row.get('conversation', []) if turn.get('role') == 'user'), None)
    normalized = normalize_prompt(text) if isinstance(text, str) else None
    expected = {'example_id': 'lmsys_' + source_id, 'source_id': source_id, 'source_index': source_index,
        'prompt': text, 'source_dataset': SOURCE, 'source_split': 'train', 'source_revision': revision,
        'source_content_hash': content_hash(text or ''), 'deduplication_identity': content_hash(normalized or ''),
        'input_token_counts': counts, 'normalization_version': NORMALIZATION_VERSION, 'extraction_version': EXTRACTION_VERSION}
    require(set(counts) == ({'7b', '14b'} if text else set())
            and all(type(value) is int and value >= 0 for value in counts.values()), 'Native tokenizer coverage differs')
    require(all(recorded.get(key) == value for key, value in expected.items()),
            'Original row/hash/native token-count binding differs')
    reason = None
    if str(row.get('language', '')).casefold() not in {'english', 'en'}:
        reason = 'non_english_source_label'
    elif not isinstance(text, str) or not text.strip():
        reason = 'missing_first_user_turn'
    elif any(value > 200 for value in counts.values()):
        reason = 'input_exceeds_200_tokens'
    elif normalized in seen:
        reason = 'normalized_duplicate'
    elif normalized in overlaps:
        reason = 'detectable_evaluation_overlap'
    if reason:
        require(recorded.get('status') == 'excluded' and recorded.get('reason') == reason, 'Recorded exclusion decision differs')
    else:
        require(recorded.get('status') == 'eligible' and 'reason' not in recorded, 'Recorded acceptance decision differs')
        seen.add(normalized)
    return source_id, reason


def revalidate_collection_inputs(args):
    started = time.monotonic()
    output = Path(args.output_dir)
    require(not output.exists(), 'Consumer validation namespace already exists')
    output.mkdir(mode=0o700, parents=True)
    args._created_output_dir = True
    progress = output / 'validation_progress.json'

    def report(phase, **details):
        record = {'phase': phase, 'at_utc': utc(), 'consumer_commit': args.consumer_commit,
                  'elapsed_seconds': round(time.monotonic() - started, 1),
                  'scientific_model_outputs_generated': False, 'source_content_logged': False,
                  'credential_values_logged': False, **details}
        atomic_write_json(progress, record)
        print(json.dumps(record, sort_keys=True), flush=True)

    project = Path(args.project).resolve()
    original = Path(args.producer_root).resolve()
    config = load_configuration(args.config)
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=project, text=True).strip()
    require(actual == args.consumer_commit == config['runtime']['implementation_revision'], 'Consumer source freeze differs')
    require(not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=normal'], cwd=project, text=True).strip(), 'Consumer checkout is dirty')
    require(config['operations'].get('lmsys_authorization') is True, 'Actual LMSYS license authorization is absent')
    require(file_hash(project / 'requirements.lock') == config['runtime']['dependency_lock_hash'], 'Consumer dependency freeze differs')
    require(file_hash(project / 'src/operations/collection_inputs.py') == file_hash(__file__), 'Loaded validator differs from consumer source')
    inventory_path = original / 'immutable_prerequisite_inventory.json'
    inventory = check_inventory(original, read_json(inventory_path))
    producer = inventory['original_producer_commit']
    require(len(producer) == 40 and all(char in '0123456789abcdef' for char in producer), 'Original producer differs')
    import yaml
    producer_configs = [original / item['relative_path'] for item in inventory['files']
                        if item['sha256'] == inventory['runtime_config_file_sha256']]
    require(len(producer_configs) == 1, 'Producer runtime configuration binding differs')
    old_config = yaml.safe_load(producer_configs[0].read_text())
    check_pinned_identities(config, old_config)
    unchanged_modules = []
    for row in inventory['scientific_module_blobs']:
        path = row['path']
        if path.endswith(('configuration.py', 'storage.py')):
            continue
        body = subprocess.check_output(['git', 'show', actual + ':' + path], cwd=project)
        require(hashlib.sha256(body).hexdigest() == row['sha256'], 'Scientific producer/consumer module changed')
        unchanged_modules.append({'path': path, 'sha256': row['sha256']})
    storage_path = 'src/shared_procedures/storage.py'
    producer_storage = subprocess.check_output(['git', 'show', producer + ':' + storage_path], cwd=project)
    consumer_storage = subprocess.check_output(['git', 'show', actual + ':' + storage_path], cwd=project)
    storage_helpers = {}
    for name in ('read_json', 'atomic_write_json'):
        digest = function_digest(producer_storage, name)
        require(digest == function_digest(consumer_storage, name), 'Scientific storage helper changed')
        storage_helpers[name] = digest
    report('input_identities_verified', original_producer_commit=producer)

    panel = read_json(original / 'data/evaluation/capability_panel.json')
    subset = read_json(original / 'data/evaluation/legibility_subset.json')
    pools = read_json(original / 'data/evaluation/official_pools.json')
    references = original / 'evidence/mbpp_reference_tests.json'
    reference_report = read_json(references)
    require(file_hash(original / 'data/evaluation/official_pools.json') == reference_report['source_file_sha256'], 'MBPP source binding differs')
    callback = verified_reference_callback(references, panel=panel, image=mbpp_image(config))
    revisions = {key: value['revision'] for key, value in config['datasets'].items() if key != 'lmsys'}
    rebuilt_panel = construct_capability_panel(pools, revisions, verify_mbpp_reference=callback)
    rebuilt_subset = construct_legibility_subset(rebuilt_panel)
    require(panel == rebuilt_panel and subset == rebuilt_subset, 'Complete official panel reconstruction differs')
    validate_capability_panel(panel, require_verified=True)
    validate_legibility_subset(subset, panel, require_verified=True)
    report('complete_panels_revalidated', capability_count=1233, legibility_count=200,
           selected_original_mbpp_suites=203)

    source_evidence = read_json(args.source_evidence)
    canonical_evidence = read_json(args.canonical_source_evidence)
    require(source_evidence['status'] == 'verified' and source_evidence['revision'] == config['datasets']['lmsys']['revision'], 'Source cache binding differs')
    source_manifest = validate_asset(args.canonical_source_cache or canonical_evidence['cache_path'])
    require(source_manifest['manifest_hash'] == inventory['licensed_source_manifest_hash'], 'Source manifest differs')
    shards = sorted(source_evidence['artifact_inventory'], key=lambda row: row['source_path'])
    source_root = Path(args.native_source_cache or source_evidence['native_cache_path'])
    for row in shards:
        path = source_root / (content_hash(row['source_path']) + '.parquet')
        require(path.stat().st_size == row['size_bytes'] and file_hash(path) == row['sha256'], 'Immutable source shard differs')
    require(len(shards) == 6 and sum(row['row_count'] for row in shards) == 1000000, 'Source population differs')

    asset_cache = Path(args.asset_cache).resolve()
    os.environ['HF_HOME'] = str(asset_cache)
    os.environ['HF_HUB_CACHE'] = str(asset_cache / 'hub')
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from transformers import AutoTokenizer
    tokenizers = {size: AutoTokenizer.from_pretrained(spec['id'], revision=spec['tokenizer_revision'],
        use_fast=True, trust_remote_code=False,
        **model_load_kwargs(spec['id'], spec['tokenizer_revision'], asset_cache, tokenizer_only=True, offline=True))
        for size, spec in config['models'].items()}
    frozen_tokenizers = tokenizer_bindings(config, asset_cache)
    manifest_path = original / 'data/candidates/manifest.json'
    manifest = load_candidate_manifest(manifest_path)
    require(manifest['count'] == len(manifest['candidates']) == inventory['candidate_count'], 'Accepted population differs')
    require(len(manifest['exclusions']) == inventory['excluded_count'], 'Excluded population differs')
    require(manifest['source_revision'] == config['datasets']['lmsys']['revision'] and manifest['source_dataset'] == SOURCE and manifest['seed'] == 42, 'Manifest source/seed differs')
    records = candidate_index(manifest, 1000000)
    overlaps = {normalize_prompt(row.get('question', row.get('prompt', row.get('text', '')))) for rows in pools.values() for row in rows}
    require(manifest['evaluation_overlap_hash'] == content_hash(sorted(overlaps)), 'Overlap population differs')
    report('candidate_structure_revalidated', accepted_count=manifest['count'], source_population=len(records))

    import pyarrow.parquet as parquet
    seen, source_ids, reasons = set(), set(), Counter()
    source_index, next_report = 0, 16384
    for shard in shards:
        path = source_root / (content_hash(shard['source_path']) + '.parquet')
        for batch in parquet.ParquetFile(path).iter_batches(batch_size=256):
            rows = batch.to_pylist()
            texts = [next((turn.get('content') for turn in row.get('conversation', []) if turn.get('role') == 'user'), None) for row in rows]
            text_positions = [position for position, text in enumerate(texts) if text]
            native_counts = {}
            for size, tokenizer in tokenizers.items():
                counts = (tokenizer([texts[position] for position in text_positions], add_special_tokens=False,
                                   truncation=False, padding=False, return_attention_mask=False, return_token_type_ids=False)['input_ids']
                          if text_positions else [])
                require(len(counts) == len(text_positions), 'Native tokenizer batch coverage differs')
                native_counts[size] = {position: len(tokens) for position, tokens in zip(text_positions, counts)}
            for position, row in enumerate(rows):
                recorded = records[source_index]
                counts = {size: native_counts[size][position] for size in tokenizers} if texts[position] else {}
                source_id, reason = check_source_record(recorded, row, source_index, counts,
                    config['datasets']['lmsys']['revision'], seen, overlaps)
                require(source_id not in source_ids, 'Original source identity collision')
                source_ids.add(source_id)
                if reason:
                    reasons[reason] += 1
                source_index += 1
            if source_index >= next_report:
                report('independent_native_row_audit', checked_source_rows=source_index, expected_source_rows=1000000)
                next_report = source_index + 16384
    require(source_index == 1000000 and sum(reasons.values()) == len(manifest['exclusions']), 'Full source audit did not complete')
    certificate = {'schema_version': MODULE_SCHEMA, 'status': 'verified_complete_consumer_validation',
        'completed_at_utc': utc(), 'original_producer_commit': producer, 'consumer_commit': actual,
        'original_inventory_sha256': file_hash(inventory_path), 'original_configuration_hash': inventory['scientific_configuration_hash'],
        'consumer_configuration_hash': configuration_hash(config), 'consumer_config_file_sha256': file_hash(args.config),
        'validator_implementation_sha256': file_hash(__file__), 'input_files': inventory['files'],
        'pinned_models': config['models'], 'pinned_datasets': config['datasets'],
        'tokenizer_asset_bindings': frozen_tokenizers,
        'original_inventory': inventory,
        'configured_mbpp_image': mbpp_image(config),
        'unchanged_scientific_modules': unchanged_modules, 'unchanged_storage_helpers': storage_helpers,
        'source_rows_independently_checked': source_index, 'source_shards_independently_reverified': len(shards),
        'both_native_tokenizers_rechecked_for_all_first_user_turns': True,
        'accepted_count': len(manifest['candidates']), 'excluded_count': len(manifest['exclusions']), 'exclusion_counts': dict(reasons),
        'candidate_manifest_content_hash': manifest['content_hash'], 'candidate_file_sha256': file_hash(manifest_path),
        'capability_count':1233,'legibility_count':200,'capability_hash':panel['content_hash'],'legibility_hash':subset['content_hash'],
        'mbpp_reference_report_hash':reference_report['report_hash'],'current_mbpp_sandbox_binding_verified':True,
        'original_artifact_bytes_preserved':True,'original_producer_metadata_preserved':True,'configuration_flags_unchanged':True,
        'gpu_stage_statuses_written':False,'scientific_model_outputs_generated':False,'generation_ready':False,
        'credential_values_logged':False,'source_content_logged':False,'elapsed_seconds':round(time.monotonic()-started,1)}
    check_inventory(original, inventory)
    certificate['certificate_hash'] = content_hash(certificate)
    pending = output / '.consumer_validation_pending.json'
    atomic_write_json(pending, certificate)
    validate_consumer_certificate(pending, config, consumer_commit=actual,
                                  input_root=original, asset_cache=asset_cache)
    atomic_write_json(output / 'consumer_validation.json', certificate)
    pending.unlink()
    report('complete', accepted_count=len(manifest['candidates']), excluded_count=len(manifest['exclusions']),
           certificate_sha256=file_hash(output / 'consumer_validation.json'))
    return certificate


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--consumer-commit', required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    parser.add_argument('--producer-root', type=Path, required=True)
    parser.add_argument('--source-evidence', type=Path, required=True)
    parser.add_argument('--canonical-source-evidence', type=Path, required=True)
    parser.add_argument('--asset-cache', type=Path, required=True)
    parser.add_argument('--native-source-cache', type=Path)
    parser.add_argument('--canonical-source-cache', type=Path)
    arguments = parser.parse_args(argv)
    arguments._created_output_dir = False
    try:
        revalidate_collection_inputs(arguments)
    except Exception as error:
        failure = {'status':'blocked','error_type':type(error).__name__,'at_utc':utc(),
                   'scientific_model_outputs_generated':False,'source_content_logged':False,'credential_values_logged':False}
        if arguments._created_output_dir and arguments.output_dir.exists():
            atomic_write_json(arguments.output_dir / 'consumer_validation_failure.json', failure)
        print(json.dumps(failure), flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
