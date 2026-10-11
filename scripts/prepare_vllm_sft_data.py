#!/usr/bin/env python3
"""Accept cached vLLM pairs, issue exact-deficit refill work, and build SFT data.

This CPU stage never generates a model response. It uses the existing scientific
validator and builders, checks fresh raw provenance, and verifies Drive copies
at each chunk. Model workers consume the emitted refill manifest separately.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import threading
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from shared_procedures.identity import content_hash
from shared_procedures.storage import atomic_write_json, read_json, file_lock, artifact_hook
from shared_procedures.configuration import configuration_hash
from operations.artifacts import ArtifactService
from operations.drive import DriveStore, drive_service
from experiment_overview.experiment_overview import Experiment
from data_generation_collection.sft_data.clean_trace_collection.candidate_batch.clean_trace_batch import NativeCleanGenerator, process_clean_batch
from data_generation_collection.sft_data.sft_data_preparation import build_sft_datasets, validate_accepted_pool, validate_sft_bundle

GENERATION_COMMIT = 'dc520cb73bf43b59a9566404dd4a7aa8358f1cc3'
GENERATION_HASH = '3757493e93d7f3ac31d9215bd493ae4f13cae0bed7c97e773f4ca4ef5be2ed54'
PAIR_CONTEXT = {}


class CachedOnly(NativeCleanGenerator):
    def generate_many(self, *args, **kwargs):
        raise RuntimeError('Missing fresh raw cache; run the emitted refill work first')


class CachedKazi:
    def __init__(self, backend, root):
        self.backend, self.root = backend, Path(root)

    def transform(self, reasoning, example_id, level, *, model_id):
        path = self.root/model_id/content_hash(example_id)/f'{level:02d}.json'
        if path.exists():
            value = read_json(path)
            if (value['source_hash'] != content_hash(reasoning)
                    or value['content_hash'] != content_hash(value['reasoning'])
                    or value['transformation'] != self.backend.provenance):
                raise ValueError('Cached official transform source/runtime differs')
            return value
        value = self.backend.transform(reasoning, example_id, level, model_id=model_id)
        atomic_write_json(path, value)
        return value


def initialize_pair_worker():
    from multiprocessing.util import Finalize
    backend = PAIR_CONTEXT['experiment'].kazi()
    PAIR_CONTEXT['kazitext'] = CachedKazi(backend, PAIR_CONTEXT['data']/'transform_cache')
    Finalize(None, backend.close, exitpriority=20)


def inspect_pair(candidate):
    data = PAIR_CONTEXT['data']
    raws = {}
    for model in ('7b','14b'):
        raw = read_json(data/'clean_traces/raw/production'/model/(content_hash(candidate['example_id'])+'.json'))
        if (raw['producer_commit'] != GENERATION_COMMIT or raw['config_hash'] != GENERATION_HASH
                or raw['runtime']['generation_backend'] != 'vllm_seeded_batch_v1'
                or raw['candidate_hash'] != content_hash(candidate)
                or raw['model_id'] != model
                or raw['model_revision'] != PAIR_CONTEXT['generator'].models[model]['revision']
                or raw['tokenizer_revision'] != PAIR_CONTEXT['generator'].models[model]['tokenizer_revision']
                or raw['raw_hash'] != content_hash(raw['raw_generation'])):
            raise ValueError('Raw cache is not the matching fresh production response')
        raws[model] = raw
    for folder in ('clean_traces', 'rejected_responses'):
        path = data/'clean_traces'/folder/'production'/(content_hash(candidate['example_id'])+'.json')
        if path.exists():
            record = read_json(path)
            if (record['candidate'] != candidate or record['config_hash'] != GENERATION_HASH
                    or set(record['traces']) != {'7b','14b'}
                    or any(record['traces'][m]['raw_hash'] != raws[m]['raw_hash'] for m in raws)):
                raise ValueError('Existing pair is not bound to these fresh raw responses')
            return record
    return process_clean_batch([candidate], PAIR_CONTEXT['generator'], PAIR_CONTEXT['tokenizers'],
                               kazitext=PAIR_CONTEXT['kazitext'], output_dir=data/'clean_traces',
                               config_hash=GENERATION_HASH)[0]


def validate_production_models(source, config):
    for pair in source['pairs']:
        for model, trace in pair['traces'].items():
            if (trace['producer_commit'] != GENERATION_COMMIT or trace['simulation'] is not False
                    or trace['execution_mode'] != 'production'
                    or trace['runtime']['generation_backend'] != 'vllm_seeded_batch_v1'
                    or trace['model_revision'] != config['models'][model]['revision']
                    or trace['tokenizer_revision'] != config['models'][model]['tokenizer_revision']):
                raise ValueError('Accepted pool contains a foreign engine/model/tokenizer source')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['accept', 'build', 'shard'], required=True)
    for key in ('config', 'candidates', 'full-candidates', 'data-root', 'artifact-root', 'cache-root'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--round', type=int, default=0)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--chunk-size', type=int, default=128)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--stop', type=int)
    a = p.parse_args()
    if not 1 <= a.workers <= 16 or not 1 <= a.chunk_size <= 128 or a.round < 0:
        p.error('Invalid execution chunk/worker/round limits')
    config = read_json(a.config)
    if configuration_hash(config) != GENERATION_HASH:
        raise ValueError('Input configuration is not the frozen fresh vLLM run')
    data, artifacts = Path(a.data_root), Path(a.artifact_root)
    os.environ.update(DATA_ROOT=str(data), ARTIFACT_ROOT=str(artifacts),
                      MODEL_CACHE_ROOT=a.cache_root, TOKENIZERS_PARALLELISM='false')
    experiment = Experiment(config, config_path=a.config)
    # Use verified local snapshot paths, avoiding Transformers' unneeded remote
    # model-info probe even when all tokenizer bytes are already cached.
    from transformers import AutoTokenizer
    from operations.assets import model_load_kwargs
    tokenizers = {}
    for model, spec in config['models'].items():
        model_load_kwargs(spec['id'], spec['tokenizer_revision'], a.cache_root,
                          tokenizer_only=True, offline=True)
        snapshot = Path(a.cache_root)/'hub'/('models--'+spec['id'].replace('/','--'))/'snapshots'/spec['tokenizer_revision']
        tokenizers[model] = AutoTokenizer.from_pretrained(str(snapshot),
                            trust_remote_code=False, local_files_only=True)
    commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    store = DriveStore(drive_service(os.environ['GOOGLE_OAUTH_TOKEN_FILE']),
                      os.environ['GOOGLE_DRIVE_FOLDER_ID'], state_root=artifacts/'state/drive')
    store.create_project_folder()
    service = ArtifactService('/workspace', artifacts/'state/artifact_registry.json', store)
    provenance = {'experiment_id':GENERATION_HASH, 'configuration_hash':GENERATION_HASH,
                  'source_code_commit':commit, 'dataset_revision':config['datasets']['lmsys']['revision'],
                  'model_revision':config['models']['7b']['revision'], 'model':'shared', 'seeds':[42],
                  'branch':'shared', 'run_id':'vllm_sft_preparation', 'stage':'prepare_sft',
                  'producer':'repository_cached_vllm_acceptance'}
    def publish(paths):
        for path in paths:
            service.register(path, provenance)
        if service.sync_artifacts(workers=4)['status'] != 'VERIFIED':
            raise RuntimeError('Preparation chunk has not reached verified Drive storage')
    local = threading.local()
    backends = []
    guard = threading.Lock()
    def kazi():
        if not hasattr(local, 'kazi'):
            backend = experiment.kazi()
            with guard:
                backends.append(backend)
            local.kazi = CachedKazi(backend, data/'transform_cache')
        return local.kazi
    try:
        with file_lock(data/'preparation.lock'):
            if a.phase == 'build':
                source = read_json(data/'accepted/accepted_pool.json')
                validate_production_models(source,config)
                completed_files = []
                def dataset_ready(path, metadata):
                    if metadata.get('kind') != 'sft_dataset':
                        return
                    publish([path])
                    completed_files.append(str(path))
                    state = {'status':'BUILDING', 'drive_verified_dataset_files':len(completed_files),
                             'expected_dataset_files':80, 'last_verified_file':str(path)}
                    atomic_write_json(artifacts/'preparation_progress.json',state)
                    print(json.dumps(state),flush=True)
                with artifact_hook(dataset_ready):
                    bundle = build_sft_datasets(source, tokenizers, data/'transformed',
                                               config_hash=GENERATION_HASH, kazitext=kazi())
                validate_sft_bundle(bundle, source)
                publish([path for path in (data/'transformed').rglob('*') if path.is_file()])
                entries = [r for r in service.list_artifacts() if
                           Path(r['local_path']).is_relative_to(data/'transformed') or
                           Path(r['local_path']) == data/'accepted/accepted_pool.json']
                catalog = {'status':'VERIFIED', 'configuration_hash':GENERATION_HASH,
                           'accepted_pool_hash':source['content_hash'], 'conditions':40,
                           'artifacts':[{k:r[k] for k in ('relative_path','sha256','size',
                                        'artifact_id','drive_file_id','drive_folder_id')} for r in entries]}
                index = artifacts/'dataset_drive_index.json'
                atomic_write_json(index,catalog)
                publish([index])
                index_record = next(r for r in service.list_artifacts() if r['local_path']==str(index))
                state = {'status':'COMPLETE', 'conditions':40, 'records':404000,
                         'configuration_hash':GENERATION_HASH, 'source_code_commit':commit,
                         'drive_folder_id':index_record['drive_folder_id'],
                         'drive_index_file_id':index_record['drive_file_id']}
                atomic_write_json(artifacts/'preparation_progress.json', state)
                publish([artifacts/'preparation_progress.json'])
                print(json.dumps(state), flush=True)
                return
            selected, full = read_json(a.candidates), read_json(a.full_candidates)
            candidates = selected['candidates']
            if selected['selected_candidates_hash'] != content_hash(candidates):
                raise ValueError('Refill candidate identities differ')
            if a.phase == 'shard':
                stop = len(candidates) if a.stop is None else a.stop
                if not 0 <= a.start < stop <= len(candidates):
                    raise ValueError('Invalid shard range')
                generator = CachedOnly(config['models'], cache_root=a.cache_root)
                PAIR_CONTEXT.update(experiment=experiment, generator=generator,
                                    tokenizers=tokenizers, data=data)
                output = data/'verification_shards'/f'round_{a.round:03d}'
                output.mkdir(parents=True, exist_ok=True)
                with ProcessPoolExecutor(max_workers=a.workers,
                                         mp_context=multiprocessing.get_context('fork'),
                                         initializer=initialize_pair_worker) as pool:
                    for start in range(a.start, stop, a.chunk_size):
                        current = candidates[start:min(start+a.chunk_size, stop)]
                        records = list(pool.map(inspect_pair, current))
                        target = output/f'chunk_{start:05d}.json'
                        archive = output/f'transfer_{start:05d}.zip'
                        atomic_write_json(target, {'records':records,
                            'configuration_hash':GENERATION_HASH,
                            'selected_candidates_hash':selected['selected_candidates_hash'],
                            'start':start, 'stop':start+len(current),
                            'acceptance_source_commit':commit})
                        temporary = archive.with_suffix('.zip.tmp')
                        with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
                            for record in records:
                                folder = 'clean_traces' if record['status']=='accepted' else 'rejected_responses'
                                path = data/'clean_traces'/folder/'production'/(content_hash(record['example_id'])+'.json')
                                z.write(path, str(path.relative_to(data)))
                                for model in ('7b','14b'):
                                    for path in (data/'transform_cache'/model/content_hash(record['example_id'])).glob('*.json'):
                                        z.write(path, str(path.relative_to(data)))
                        temporary.replace(archive)
                        publish([target, archive])
                        import hashlib
                        state = {'status':'SHARD_PROGRESS', 'start':start,
                                 'stop':start+len(current), 'archive':str(archive),
                                 'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}
                        atomic_write_json(output/f'ready_{start:05d}.json',state)
                        print(json.dumps(state),flush=True)
                print(json.dumps({'status':'SHARD_COMPLETE','start':a.start,'stop':stop}),flush=True)
                return
            rounds = data/'acceptance_rounds'
            previous = [read_json(rounds/f'round_{i:03d}/summary.json') for i in range(a.round)]
            cursor = sum(r['processed_count'] for r in previous)
            if candidates != full['candidates'][cursor:cursor+len(candidates)]:
                raise ValueError('Candidate order differs from the audited source')
            accepted_before = sum(r['accepted_count'] for r in previous)
            if len(candidates) != 10100-accepted_before:
                raise ValueError('This round must contain exactly the accepted-pair deficit')
            output = rounds/f'round_{a.round:03d}'
            output.mkdir(parents=True, exist_ok=True)
            generator = CachedOnly(config['models'], cache_root=a.cache_root)
            PAIR_CONTEXT.update(experiment=experiment, generator=generator,
                                tokenizers=tokenizers, data=data)
            accepted, rejected, chunks = [], [], []
            with ProcessPoolExecutor(max_workers=a.workers,
                                     mp_context=multiprocessing.get_context('fork'),
                                     initializer=initialize_pair_worker) as pool:
                for start in range(0, len(candidates), a.chunk_size):
                    current = candidates[start:start+a.chunk_size]
                    target = output/f'chunk_{start:05d}.json'
                    archive = output/f'transforms_{start:05d}.zip'
                    if target.exists():
                        records = read_json(target)['records']
                        if [r['example_id'] for r in records] != [c['example_id'] for c in current]:
                            raise ValueError('Existing acceptance chunk identity differs')
                    else:
                        records = list(pool.map(inspect_pair, current))
                        atomic_write_json(target, {'records':records, 'configuration_hash':GENERATION_HASH,
                                                  'acceptance_source_commit':commit})
                    if not archive.exists():
                        temporary = archive.with_suffix('.zip.tmp')
                        with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
                            for c in current:
                                for model in ('7b','14b'):
                                    for f in (data/'transform_cache'/model/content_hash(c['example_id'])).glob('*.json'):
                                        z.write(f, str(f.relative_to(data/'transform_cache')))
                        temporary.replace(archive)
                    publish([target, archive])
                    accepted.extend(r for r in records if r['status']=='accepted')
                    rejected.extend(r['example_id'] for r in records if r['status']=='rejected')
                    chunks.append(str(target))
                    state = {'status':'ACCEPTING', 'round':a.round, 'processed':start+len(current),
                             'requested':len(candidates), 'accepted_this_round':len(accepted),
                             'accepted_total':accepted_before+len(accepted), 'drive_verified_chunks':len(chunks)}
                    atomic_write_json(artifacts/'preparation_progress.json', state)
                    print(json.dumps(state), flush=True)
            summary = {'round':a.round, 'processed_count':len(candidates), 'accepted_count':len(accepted),
                       'rejected_ids':rejected, 'chunks':chunks, 'selected_candidates_hash':content_hash(candidates)}
            atomic_write_json(output/'summary.json',summary)
            total = accepted_before+len(accepted)
            if total < 10100:
                next_start = cursor+len(candidates)
                next_candidates = full['candidates'][next_start:next_start+10100-total]
                if len(next_candidates) != 10100-total:
                    raise RuntimeError('Audited candidate source cannot supply exact deficit')
                refill = {**selected, 'candidates':next_candidates,
                          'selected_candidates_hash':content_hash(next_candidates),
                          'refill_round':a.round+1, 'accepted_count_before_round':total}
                refill_path = artifacts/f'refill_candidates_{a.round+1:03d}.json'
                atomic_write_json(refill_path,refill)
                state = {'status':'REFILL_REQUIRED', 'round':a.round, 'accepted_total':total,
                         'deficit':10100-total, 'refill_candidates':str(refill_path)}
                publish([output/'summary.json', refill_path])
            else:
                all_pairs = []
                for item in previous+[summary]:
                    for path in item['chunks']:
                        all_pairs.extend(r for r in read_json(path)['records'] if r['status']=='accepted')
                ids = [r['example_id'] for r in all_pairs]
                source = {'status':'verified', 'schema_version':'accepted_pool_v1', 'pairs':all_pairs,
                          'train_ids':ids[:10000], 'validation_ids':ids[10000:], 'train_count':10000,
                          'validation_count':100, 'rejected_ids':sum([r['rejected_ids'] for r in previous],[])+rejected,
                          'simulation':False, 'batches':previous+[summary],
                          'candidate_manifest_hash':full['content_hash'], 'config_hash':GENERATION_HASH,
                          'content_hash':content_hash(all_pairs), 'acceptance_source_commit':commit}
                validate_accepted_pool(source,GENERATION_HASH)
                validate_production_models(source,config)
                source_path = data/'accepted/accepted_pool.json'
                atomic_write_json(source_path,source)
                publish([output/'summary.json',source_path])
                state = {'status':'ACCEPTED_POOL_VERIFIED', 'accepted_total':10100}
            atomic_write_json(artifacts/'preparation_progress.json',state)
            publish([artifacts/'preparation_progress.json'])
            print(json.dumps(state), flush=True)
    finally:
        for backend in backends:
            backend.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status':'FAILED','error_type':type(error).__name__}), flush=True)
        raise
