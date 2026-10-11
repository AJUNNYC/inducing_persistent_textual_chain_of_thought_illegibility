#!/usr/bin/env python3
"""Generate one model's raw responses and deliver verified batches to its consumer.

Uses the native frozen generator. This producer does not classify acceptance,
change targets, or assert that another machine's runtime checks ran locally.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from data_generation_collection.sft_data.clean_trace_collection.candidate_batch.clean_trace_batch import NativeCleanGenerator, GENERATION_SETTINGS
from operations.artifacts import ArtifactService
from operations.drive import DriveStore, drive_service, file_hash
from shared_procedures.configuration import load_configuration, configuration_hash
from shared_procedures.identity import content_hash
from shared_procedures.storage import atomic_write_json, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='Frozen consumer configuration')
    parser.add_argument('--candidates', required=True, help='Ordered subset of the already audited manifest')
    parser.add_argument('--model', choices=('7b', '14b'), required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--artifact-root', required=True)
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--deadline', required=True)
    parser.add_argument('--consumer-credentials', required=True, help='Protected Jupyter connection JSON')
    parser.add_argument('--consumer-data-root', required=True)
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.deadline.replace('Z', '+00:00'))
    if deadline.utcoffset() is None or deadline.utcoffset().total_seconds() != 0:
        raise ValueError('Absolute UTC deadline required')
    config = load_configuration(args.config)
    if not config['spending']['authorized']:
        raise ValueError('Existing compute authorization required')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    producer = copy.deepcopy(config)
    producer['runtime']['implementation_revision'] = commit
    consumer_hash, producer_hash = configuration_hash(config), configuration_hash(producer)
    selected = read_json(args.candidates)
    candidates = selected['candidates']
    if (selected['consumer_configuration_hash'] != consumer_hash
            or selected['selected_candidates_hash'] != content_hash(candidates)
            or len({row['example_id'] for row in candidates}) != len(candidates)):
        raise ValueError('Transferred candidate identity differs from the frozen consumer')
    data, artifacts = Path(args.data_root), Path(args.artifact_root)
    raw_root = data / 'clean_traces' / 'raw' / 'production' / args.model
    raw_root.mkdir(parents=True, exist_ok=True)
    store = DriveStore(drive_service(os.environ['GOOGLE_OAUTH_TOKEN_FILE']),
        os.environ['GOOGLE_DRIVE_FOLDER_ID'], state_root=artifacts / 'state' / 'drive')
    store.create_project_folder()
    service = ArtifactService(data.parent, artifacts / 'state' / 'artifact_registry.json', store)
    import requests
    auth = read_json(args.consumer_credentials)
    session = requests.Session()
    session.headers['Authorization'] = 'token ' + auth['token']
    destination = Path(args.consumer_data_root) / 'clean_traces' / 'raw' / 'production' / args.model
    if not destination.is_absolute() or '..' in destination.parts or not destination.is_relative_to('/workspace'):
        raise ValueError('Consumer destination must be below /workspace')
    api = auth['base_url'].rstrip('/') + '/api/contents/'
    from urllib.parse import quote
    import base64

    def deliver(path):
        record = read_json(path)
        registered = service.register(path, provenance)
        if registered is None or registered['sync_status'] != 'VERIFIED':
            raise RuntimeError('Delivery requires a verified permanent Drive copy')
        record.update(generation_original_producer_commit=commit,
            generation_original_configuration_hash=producer_hash,
            generation_original_artifact_id=registered['artifact_id'],
            generation_original_drive_file_id=registered['drive_file_id'],
            generation_original_file_sha256=registered['sha256'],
            cache_adoption_mode='dedicated_model_producer_bound_to_unchanged_consumer')
        remote = (destination / path.name).relative_to('/').as_posix()
        url = api + quote(remote)
        existing = session.get(url, params={'content': 1, 'format': 'base64'}, timeout=60)
        if existing.status_code == 200:
            previous = json.loads(base64.b64decode(existing.json()['content']))
            if (previous['candidate_hash'] != record['candidate_hash']
                    or previous['config_hash'] != consumer_hash
                    or previous['raw_hash'] != record['raw_hash']):
                raise RuntimeError('Consumer already holds a different immutable response')
            return
        if existing.status_code != 404:
            existing.raise_for_status()
        payload = json.dumps(record, sort_keys=True, ensure_ascii=False).encode()
        session.put(url, json={'type': 'file', 'format': 'base64',
            'content': base64.b64encode(payload).decode()}, timeout=60).raise_for_status()

    provenance = {'experiment_id': producer_hash, 'configuration_hash': producer_hash,
        'source_code_commit': commit, 'dataset_revision': content_hash(config['datasets']),
        'model_revision': config['models'][args.model]['revision'], 'model': args.model,
        'seeds': [42], 'branch': 'shared', 'run_id': 'parallel_' + args.model,
        'stage': 'collect_clean', 'producer': 'native_model_generation_worker'}
    generator = NativeCleanGenerator(config['models'], cache_root=args.cache_root)
    stop = lambda: datetime.now(timezone.utc) >= deadline or (data / 'STOP_REQUESTED').exists()
    status = artifacts / 'progress.json'
    completed = 0
    batch_size = generator.batch_size_for(args.model)
    if service.sync_artifacts(workers=8)['status'] != 'VERIFIED':
        raise RuntimeError('Previous Drive replication incomplete')
    for start in range(0, len(candidates), batch_size):
        if stop():
            break
        batch = candidates[start:start + batch_size]
        pending = []
        paths = []
        for row in batch:
            path = raw_root / (content_hash(row['example_id']) + '.json')
            paths.append(path)
            if path.exists():
                r = read_json(path)
                if r['candidate_hash'] != content_hash(row) or r['config_hash'] != consumer_hash:
                    raise ValueError('Existing native raw response belongs to another candidate/configuration')
            else:
                pending.append(row)
        print(json.dumps({'event': 'GENERATING_BATCH', 'model': args.model,
            'completed': completed, 'batch_size': len(pending)}), flush=True)
        generated = iter(generator.generate_many(args.model, pending, stop_requested=stop))
        for row in pending:
            raw = dict(next(generated))
            record = {**raw, 'example_id': row['example_id'], 'model_id': args.model,
                'candidate_hash': content_hash(row), 'config_hash': consumer_hash,
                'raw_hash': content_hash(raw['raw_generation']),
                'generation_settings': dict(GENERATION_SETTINGS),
                'producer_commit': commit, 'producer_configuration_hash': producer_hash,
                'consumer_commit': config['runtime']['implementation_revision'],
                'consumer_binding': 'identical_scientific_settings_except_producer_revision'}
            atomic_write_json(raw_root / (content_hash(row['example_id']) + '.json'), record)
        if next(generated, None) is not None:
            raise RuntimeError('Native generator returned unexpected extra responses')
        for path in paths:
            service.register(path, provenance)
        if service.sync_artifacts(workers=8)['status'] != 'VERIFIED':
            raise RuntimeError('Whole batch must reach Drive before generation continues')
        for path in paths:
            deliver(path)
        completed += len(batch)
        atomic_write_json(status, {'model': args.model, 'completed': completed,
            'requested': len(candidates), 'drive_verified': completed,
            'consumer_delivered': completed, 'producer_commit': commit,
            'consumer_configuration_hash': consumer_hash, 'batch_size': batch_size,
            'updated_at': datetime.now(timezone.utc).isoformat()})
        print(json.dumps({'event': 'BATCH_DRIVE_VERIFIED_AND_DELIVERED',
            'model': args.model, 'completed': completed}), flush=True)
    print(json.dumps({'event': 'COMPLETE' if completed == len(candidates) else 'PAUSED',
        'completed': completed, 'pod_billing_stopped': False}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Transport exceptions can contain authentication data.
        print(json.dumps({'status': 'FAILED', 'error_type': type(error).__name__}), flush=True)
        raise SystemExit(1)
