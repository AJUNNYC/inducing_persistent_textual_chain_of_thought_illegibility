#!/usr/bin/env python3
"""Isolated inference benchmark; never adopts outputs into production collection.

Install in a separate environment: vllm==0.11.0 transformers==4.57.3
PyYAML google-api-python-client google-auth-httplib2 google-auth-oauthlib.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from shared_procedures.configuration import load_configuration, configuration_hash
from shared_procedures.identity import content_hash, derive_seed
from shared_procedures.response_parsing import parse_native_output
from shared_procedures.storage import atomic_write_json, read_json
from operations.artifacts import ArtifactService
from operations.drive import DriveStore, drive_service


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--candidates', required=True)
    p.add_argument('--cache-root', required=True)
    p.add_argument('--output-root', required=True)
    p.add_argument('--native-raw-root', required=True)
    p.add_argument('--native-batch-start', required=True)
    p.add_argument('--model', choices=['7b', '14b'], default='14b')
    a = p.parse_args()
    config = load_configuration(a.config)
    if not config['spending']['authorized']:
        raise ValueError('Existing compute authorization required')
    selected = read_json(a.candidates)
    candidates = selected['candidates']
    if selected['selected_candidates_hash'] != content_hash(candidates):
        raise ValueError('Candidate identity differs')
    spec = config['models'][a.model]
    hub = Path(a.cache_root) / 'hub' / ('models--' + spec['id'].replace('/', '--')) / 'snapshots'
    model, tokenizer_path = hub / spec['revision'], hub / spec['tokenizer_revision']
    if not model.is_dir() or not tokenizer_path.is_dir():
        raise ValueError('Previously verified pinned model snapshots required')
    root = Path(a.output_root)
    root.mkdir(parents=True, exist_ok=True)
    native_rows = []
    for c in candidates[32:64]:
        path = Path(a.native_raw_root) / (content_hash(c['example_id']) + '.json')
        row = read_json(path)
        native_rows.append((c, row, path.stat().st_mtime))
    start = datetime.fromisoformat(a.native_batch_start).timestamp()
    native_seconds = max(t for _, _, t in native_rows) - start
    if native_seconds <= 0:
        raise ValueError('Native baseline timestamps out of order')
    import vllm
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    import torch
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=False, local_files_only=True)
    generation_config = read_json(model / 'generation_config.json')
    eos = generation_config.get('eos_token_id', tokenizer.eos_token_id)
    eos = eos if isinstance(eos, list) else [eos]
    t = time.perf_counter()
    engine = LLM(model=str(model), tokenizer=str(tokenizer_path), dtype='bfloat16',
        trust_remote_code=False, max_model_len=4096, max_num_seqs=128,
        gpu_memory_utilization=0.9, max_num_batched_tokens=8192,
        generation_config='vllm', enable_prefix_caching=False)
    startup = time.perf_counter() - t
    def run(batch, name):
        prompts = [tokenizer.apply_chat_template([{'role':'user','content':c['prompt']}],
            tokenize=False, add_generation_prompt=True) for c in batch]
        params = [SamplingParams(temperature=0.6, top_p=0.95, top_k=-1,
            repetition_penalty=1.0, max_tokens=3072,
            seed=derive_seed(42, 'clean_generation', c['example_id'], a.model),
            stop_token_ids=eos, skip_special_tokens=True) for c in batch]
        begin = time.perf_counter()
        outputs = engine.generate(prompts, params, use_tqdm=False)
        elapsed = time.perf_counter() - begin
        rows = []
        for c, prompt, output in zip(batch, prompts, outputs):
            o = output.outputs[0]
            prefix = '<think>' if prompt.rstrip().endswith('<think>') and not o.text.lstrip().startswith('<think>') else ''
            rows.append({'example_id':c['example_id'], 'raw_generation':o.text,
                'generation_token_ids':list(o.token_ids), 'generated_token_count':len(o.token_ids),
                'finish_reason':o.finish_reason, 'native_reasoning_prefix':prefix,
                'parsed':parse_native_output(prefix + o.text), 'benchmark_only':True})
        atomic_write_json(root / (name + '_responses.json'), rows)
        tokens = sum(r['generated_token_count'] for r in rows)
        return {'batch_size':len(rows), 'generation_seconds':elapsed,
            'responses_per_second':len(rows)/elapsed, 'output_tokens':tokens,
            'output_tokens_per_second':tokens/elapsed,
            'native_format_valid':sum(r['parsed']['status']=='valid' for r in rows),
            'stopped':sum(r['finish_reason']=='stop' for r in rows),
            'length_limited':sum(r['finish_reason']=='length' for r in rows)}
    # Same 32 prompts as the last completed native batch, then larger scheduling load.
    matched = run([c for c, _, _ in native_rows], 'matched32')
    larger = run(candidates[:128], 'batch128')
    native_tokens = sum(r['generated_token_count'] for _, r, _ in native_rows)
    baseline = {'batch_size':32, 'generation_seconds_approximate':native_seconds,
        'timing_method':'last raw batch completion minus previous batch Drive/delivery completion; includes small orchestration overhead',
        'output_tokens':native_tokens, 'output_tokens_per_second_approximate':native_tokens/native_seconds,
        'native_format_valid':sum(parse_native_output(r.get('native_reasoning_prefix','')+r['raw_generation'])['status']=='valid' for _, r, _ in native_rows),
        'stopped':sum(r['finish_reason']=='stop' for _, r, _ in native_rows)}
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    report = {'status':'BENCHMARK_COMPLETE', 'model':spec, 'source_code_commit':commit,
        'vllm_version':vllm.__version__, 'torch_version':torch.__version__,
        'model_startup_seconds':startup, 'dtype':'bfloat16',
        'settings':{'max_new_tokens':3072,'temperature':0.6,'top_p':0.95},
        'native_reference':baseline,'vllm_matched32':matched,'vllm_batch128':larger,
        'matched_tokens_per_second_speedup_approximate':matched['output_tokens_per_second']/baseline['output_tokens_per_second_approximate'],
        'estimated_10100_generation_hours_at_batch128_rate':10100/larger['responses_per_second']/3600,
        'limitations':['Small sample does not establish task correctness or quality equivalence.',
                      'Projection excludes setup, Drive replication, delivery, acceptance filtering and refills.',
                      'Native baseline timing is approximate; engine outputs need not be identical.'],
        'completed_at':datetime.now(timezone.utc).isoformat()}
    atomic_write_json(root/'report.json', report)
    # Keep benchmark artifacts separate from production and permanently replicate them.
    store = DriveStore(drive_service(os.environ['GOOGLE_OAUTH_TOKEN_FILE']),
        os.environ['GOOGLE_DRIVE_FOLDER_ID'], state_root=root/'drive_state')
    store.create_project_folder()
    service = ArtifactService(root.parent, root/'registry/artifact_registry.json', store)
    provenance = {'experiment_id':configuration_hash(config),'configuration_hash':configuration_hash(config),
        'source_code_commit':commit,'dataset_revision':content_hash(config['datasets']),
        'model_revision':spec['revision'],'model':a.model,'seeds':[42],
        'branch':'shared','run_id':root.name,'stage':'generation_benchmark','producer':'vllm_benchmark'}
    for name in ['report.json','matched32_responses.json','batch128_responses.json']:
        service.register(root/name, provenance)
    result = service.sync_artifacts(workers=3)
    if result['status'] != 'VERIFIED':
        raise RuntimeError('Benchmark Drive backup incomplete')
    print(json.dumps({**report,'drive_backup':'VERIFIED'}),flush=True)


if __name__ == '__main__':
    main()
