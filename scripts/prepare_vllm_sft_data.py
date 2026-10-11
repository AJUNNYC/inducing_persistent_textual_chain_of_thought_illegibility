#!/usr/bin/env python3
"""Accept cached vLLM pairs, issue exact-deficit refill work, and build SFT data.

This CPU stage never generates a model response. It uses the existing scientific
validator and builders, checks fresh raw provenance, and verifies Drive copies
at each chunk. Model workers consume the emitted refill manifest separately.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from shared_procedures.identity import content_hash
from shared_procedures.storage import atomic_write_json, read_json, file_lock
from shared_procedures.configuration import configuration_hash
from operations.artifacts import ArtifactService
from operations.drive import DriveStore, drive_service
from experiment_overview.experiment_overview import Experiment
from data_generation_collection.sft_data.clean_trace_collection.candidate_batch.clean_trace_batch import NativeCleanGenerator, process_clean_batch
from data_generation_collection.sft_data.sft_data_preparation import build_sft_datasets, validate_accepted_pool, validate_sft_bundle

GENERATION_COMMIT = 'dc520cb73bf43b59a9566404dd4a7aa8358f1cc3'
GENERATION_HASH = '3757493e93d7f3ac31d9215bd493ae4f13cae0bed7c97e773f4ca4ef5be2ed54'


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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['accept', 'build'], required=True)
    for key in ('config', 'candidates', 'full-candidates', 'data-root', 'artifact-root', 'cache-root'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--round', type=int, default=0)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--chunk-size', type=int, default=128)
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
                bundle = build_sft_datasets(source, tokenizers, data/'transformed',
                                           config_hash=GENERATION_HASH, kazitext=kazi())
                validate_sft_bundle(bundle, source)
                publish([path for path in (data/'transformed').rglob('*') if path.is_file()])
                state = {'status':'COMPLETE', 'conditions':40, 'records':404000,
                         'configuration_hash':GENERATION_HASH, 'source_code_commit':commit}
                atomic_write_json(artifacts/'preparation_progress.json', state)
                publish([artifacts/'preparation_progress.json'])
                print(json.dumps(state), flush=True)
                return
            selected, full = read_json(a.candidates), read_json(a.full_candidates)
            candidates = selected['candidates']
            if selected['selected_candidates_hash'] != content_hash(candidates):
                raise ValueError('Refill candidate identities differ')
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
            def inspect(candidate):
                for model in ('7b','14b'):
                    raw = read_json(data/'clean_traces/raw/production'/model/(content_hash(candidate['example_id'])+'.json'))
                    if (raw['producer_commit'] != GENERATION_COMMIT or raw['config_hash'] != GENERATION_HASH
                            or raw['runtime']['generation_backend'] != 'vllm_seeded_batch_v1'
                            or raw['candidate_hash'] != content_hash(candidate)):
                        raise ValueError('Raw cache is not the matching fresh production response')
                return process_clean_batch([candidate], generator, tokenizers, kazitext=kazi(),
                                           output_dir=data/'clean_traces', config_hash=GENERATION_HASH)[0]
            accepted, rejected, chunks = [], [], []
            with ThreadPoolExecutor(max_workers=a.workers) as pool:
                for start in range(0, len(candidates), a.chunk_size):
                    current = candidates[start:start+a.chunk_size]
                    target = output/f'chunk_{start:05d}.json'
                    archive = output/f'transforms_{start:05d}.zip'
                    if target.exists():
                        records = read_json(target)['records']
                        if [r['example_id'] for r in records] != [c['example_id'] for c in current]:
                            raise ValueError('Existing acceptance chunk identity differs')
                    else:
                        records = list(pool.map(inspect, current))
                        atomic_write_json(target, {'records':records, 'configuration_hash':GENERATION_HASH,
                                                  'acceptance_source_commit':commit})
                    if not archive.exists():
                        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
                            for c in current:
                                for model in ('7b','14b'):
                                    for f in (data/'transform_cache'/model/content_hash(c['example_id'])).glob('*.json'):
                                        z.write(f, str(f.relative_to(data/'transform_cache')))
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
