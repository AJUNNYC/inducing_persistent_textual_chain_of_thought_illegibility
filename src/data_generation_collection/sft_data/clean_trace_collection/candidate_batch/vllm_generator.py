"""Untouched BF16 construction models with explicit vLLM inference provenance."""
from pathlib import Path
import os

from .clean_trace_batch import NativeCleanGenerator, CleanDataPause, GENERATION_SETTINGS
from shared_procedures.identity import derive_seed
from shared_procedures.storage import read_json

BACKEND = {'name': 'vllm', 'version': '0.11.0', 'transformers_version': '4.57.3',
           'torch_version': '2.8.0+cu128', 'dtype': 'bfloat16', 'max_model_len': 4096,
           'max_num_seqs': 128, 'max_num_batched_tokens': 8192,
           'gpu_memory_utilization': 0.9, 'enable_prefix_caching': False,
           'prompt_encoding': 'native_chat_template_explicit_token_ids_v1'}


class VLLMCleanGenerator(NativeCleanGenerator):
    """Native record contract; independent per-request seeds, no shared RNG claim."""
    def __init__(self, models, *, cache_root, batch_size=128, **kwargs):
        super().__init__(models, cache_root=cache_root, batch_size=batch_size, **kwargs)
        self.engine = None

    def _load(self, model_id):
        if self.active == model_id:
            return
        if self.engine is not None:
            raise RuntimeError('Use one vLLM model per worker process')
        import torch
        import transformers
        import vllm
        from vllm import LLM
        from transformers import AutoTokenizer
        from operations.assets import model_load_kwargs
        from ..candidate_manifest import validate_revision
        if (vllm.__version__ != BACKEND['version']
                or transformers.__version__ != BACKEND['transformers_version']
                or torch.__version__ != BACKEND['torch_version']):
            raise RuntimeError('vLLM core dependency versions differ from the run contract')
        spec = self.models[model_id]
        validate_revision(spec['revision'])
        validate_revision(spec['tokenizer_revision'])
        # Reuse pinned verified local assets; never download a moving model revision.
        model_load_kwargs(spec['name'], spec['revision'], self.cache_root, offline=True)
        model_load_kwargs(spec['name'], spec['tokenizer_revision'], self.cache_root,
                          tokenizer_only=True, offline=True)
        hub = Path(self.cache_root) / 'hub' / ('models--' + spec['name'].replace('/', '--')) / 'snapshots'
        model_path, tokenizer_path = hub / spec['revision'], hub / spec['tokenizer_revision']
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path),
            trust_remote_code=False, local_files_only=True)
        eos = read_json(model_path / 'generation_config.json').get('eos_token_id', self.tokenizer.eos_token_id)
        self.eos_ids = eos if isinstance(eos, list) else [eos]
        self.engine = LLM(model=str(model_path), tokenizer=str(tokenizer_path),
            dtype='bfloat16', trust_remote_code=False, generation_config='vllm',
            **{k: BACKEND[k] for k in ('max_model_len', 'max_num_seqs',
                'max_num_batched_tokens', 'gpu_memory_utilization', 'enable_prefix_caching')})
        self.active = model_id

    def generate_many(self, model_id, candidates, *, stop_requested=None):
        candidates = list(candidates)
        if not candidates:
            return
        stop = stop_requested or self.stop_requested
        if stop and stop():
            raise CleanDataPause({'phase':'before_model_load', 'model_id':model_id})
        self._load(model_id)
        from vllm import SamplingParams
        spec = self.models[model_id]
        size = self.batch_size_for(model_id)
        for start in range(0, len(candidates), size):
            batch = candidates[start:start+size]
            if stop and stop():
                raise CleanDataPause({'phase':'before_generation_batch', 'model_id':model_id})
            rendered = [self.tokenizer.apply_chat_template([{'role':'user','content':c['prompt']}],
                tokenize=False, add_generation_prompt=True) for c in batch]
            inputs = [{'prompt_token_ids':self.tokenizer.encode(p, add_special_tokens=False)} for p in rendered]
            if any(len(p['prompt_token_ids']) + 3072 > BACKEND['max_model_len'] for p in inputs):
                raise ValueError('Full prompt and response budget exceed context; no truncation permitted')
            seeds = [derive_seed(42, 'clean_generation', c['example_id'], model_id) for c in batch]
            params = [SamplingParams(temperature=0.6, top_p=0.95, top_k=-1,
                repetition_penalty=1.0, max_tokens=3072, seed=s,
                stop_token_ids=self.eos_ids, skip_special_tokens=True) for s in seeds]
            outputs = self.engine.generate(inputs, params, use_tqdm=False)
            if len(outputs) != len(batch):
                raise RuntimeError('vLLM response count differs from the submitted batch')
            for row, (c, prompt, seed, output) in enumerate(zip(batch, rendered, seeds, outputs)):
                ids = inputs[row]['prompt_token_ids']
                if list(output.prompt_token_ids) != ids or len(output.outputs) != 1:
                    raise RuntimeError('vLLM prompt identity or response multiplicity differs')
                o = output.outputs[0]
                if o.finish_reason not in ('stop', 'length') or len(o.token_ids) > 3072:
                    raise RuntimeError('vLLM returned an invalid response boundary or budget')
                prefix = '<think>' if prompt.rstrip().endswith('<think>') and not o.text.lstrip().startswith('<think>') else ''
                yield {'raw_generation':o.text, 'finish_reason':o.finish_reason,
                    'generated_token_count':len(o.token_ids), 'generation_token_ids':list(o.token_ids),
                    'model_revision':spec['revision'], 'tokenizer_revision':spec['tokenizer_revision'],
                    'model_name':spec['name'], 'seed':seed, 'native_reasoning_prefix':prefix,
                    'generation_settings':dict(GENERATION_SETTINGS), 'simulation':False,
                    'execution_mode':'production', 'generation_batch_end':row == len(batch)-1,
                    'runtime':{**BACKEND, 'generation_backend':'vllm_seeded_batch_v1',
                        'generation_batch_size':len(batch), 'generation_batch_row':row}}
