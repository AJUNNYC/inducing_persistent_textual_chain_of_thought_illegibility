"""Pinned batched BF16 measurement backend for both bases and exact adapters."""
from pathlib import Path
from .output_generation import NativeGenerator, MODEL_IDS, verify_adapter_payload
from shared_procedures.identity import content_hash, derive_seed
from shared_procedures.storage import read_json

VERSION = 'vllm_evaluation_generation_v1'
BACKEND = {'name':'vllm', 'version':'0.11.0', 'transformers_version':'4.57.3',
    'torch_version':'2.8.0+cu128', 'dtype':'bfloat16', 'max_model_len':8192,
    'max_num_seqs':128, 'max_num_batched_tokens':8192, 'gpu_memory_utilization':0.9,
    'enable_prefix_caching':False, 'enable_lora':True, 'max_lora_rank':32,
    'prompt_encoding':'native_chat_template_explicit_token_ids_v1', 'batch_size':128}

class VLLMEvaluationGenerator(NativeGenerator):
    """Keep immutable raw-cache semantics; never adopt Transformers outputs.

    One model size per process. Adapter requests are verified before use. All
    states use the same LoRA-enabled BF16 engine, including untouched bases.
    """
    def __init__(self, cache_root):
        super().__init__(cache_root)
        self.batch_size = BACKEND['batch_size']
        self.engine = None
        self._pending = {}
        self._adapters = {}

    def identity(self, state, item, seed=42):
        identity = super().identity(state, item, seed)
        identity.update(version=VERSION, backend=dict(BACKEND))
        return identity

    def _load(self, state):
        key = (state['model'],state['model_revision'],state['tokenizer_revision'])
        if self.engine is not None:
            if key != self._loaded_identity:
                raise RuntimeError('Use a fresh process for the next model size')
            return
        import torch, transformers, vllm
        from transformers import AutoTokenizer
        from vllm import LLM
        from operations.assets import model_load_kwargs
        if (vllm.__version__,transformers.__version__,torch.__version__) != (
                BACKEND['version'],BACKEND['transformers_version'],BACKEND['torch_version']):
            raise RuntimeError('Evaluation engine dependencies differ from frozen contract')
        model_id = MODEL_IDS[state['model']]
        cache = state['cache_root']
        model_load_kwargs(model_id,state['model_revision'],cache,offline=True)
        model_load_kwargs(model_id,state['tokenizer_revision'],cache,tokenizer_only=True,offline=True)
        snapshots = Path(cache)/'hub'/('models--'+model_id.replace('/','--'))/'snapshots'
        model_path = snapshots/state['model_revision']
        self._tokenizer = AutoTokenizer.from_pretrained(str(snapshots/state['tokenizer_revision']),
            trust_remote_code=False,local_files_only=True)
        eos = read_json(model_path/'generation_config.json').get('eos_token_id',self._tokenizer.eos_token_id)
        self.eos_ids = eos if isinstance(eos,list) else [eos]
        self.engine = LLM(model=str(model_path),tokenizer=str(snapshots/state['tokenizer_revision']),
            dtype='bfloat16',trust_remote_code=False,generation_config='vllm',
            **{k:BACKEND[k] for k in ('max_model_len','max_num_seqs','max_num_batched_tokens',
                'gpu_memory_utilization','enable_prefix_caching','enable_lora','max_lora_rank')})
        self._loaded_identity = key

    def generate_batch(self, state, items, seed=42):
        items = list(items)
        missing = []
        for item in items:
            identity = self.identity(state,item,seed)
            path = self.cache_root/'raw'/state['state_id']/(content_hash(identity)+'.json')
            if path.exists():
                # Parent validates the complete record and adapter bytes.
                super().generate(state,item,seed)
            else:
                missing.append((item,identity))
        if not missing:
            return
        self._load(state)
        from vllm import SamplingParams
        adapter = None
        if state.get('adapter_path'):
            verify_adapter_payload(state)
            from vllm.lora.request import LoRARequest
            key = content_hash(state)
            if key not in self._adapters:
                self._adapters[key] = LoRARequest(key,len(self._adapters)+1,state['adapter_path'])
            adapter = self._adapters[key]
        rendered = [self._tokenizer.apply_chat_template([{'role':'user','content':i['public_prompt']}],
            tokenize=False,add_generation_prompt=True) for _,i in missing]
        inputs = [{'prompt_token_ids':self._tokenizer.encode(p,add_special_tokens=False)} for p in rendered]
        if any(len(p['prompt_token_ids'])+3072 > BACKEND['max_model_len'] for p in inputs):
            raise ValueError('Full prompt and response exceed context; truncation is forbidden')
        seeds = [derive_seed('native_evaluation',seed,state['state_id'],i['question_id']) for _,i in missing]
        params = [SamplingParams(temperature=.6,top_p=.95,top_k=-1,repetition_penalty=1.,
            max_tokens=3072,seed=s,stop_token_ids=self.eos_ids,skip_special_tokens=False) for s in seeds]
        outputs = self.engine.generate(inputs,params,lora_request=adapter,use_tqdm=False)
        if len(outputs) != len(missing):
            raise RuntimeError('Response multiplicity changed')
        for (item,identity),prompt,inp,effective_seed,result in zip(missing,rendered,inputs,seeds,outputs):
            if list(result.prompt_token_ids) != inp['prompt_token_ids'] or len(result.outputs)!=1:
                raise RuntimeError('Prompt identity or completion multiplicity changed')
            completion = result.outputs[0]
            if completion.finish_reason not in ('stop','length') or len(completion.token_ids)>3072:
                raise RuntimeError('Invalid stopping boundary or response budget')
            decoded = self._tokenizer.decode(completion.token_ids,skip_special_tokens=False)
            text = decoded
            if self._tokenizer.eos_token and text.endswith(self._tokenizer.eos_token):
                text = text[:-len(self._tokenizer.eos_token)]
            prefix = '<think>\n' if prompt.endswith('<think>\n') else ('<think>' if prompt.endswith('<think>') else '')
            runtime = {**BACKEND,'model_commit':state['model_revision'],
                'tokenizer_commit':state['tokenizer_revision'],'generation_batch_size':len(missing)}
            self._pending[effective_seed] = dict(raw_output=text,decoded_generated_text=decoded,
                generated_token_ids=list(completion.token_ids),generated_tokens=len(completion.token_ids),
                finish_reason=completion.finish_reason,native_prefix=prefix,native_chat_hash=content_hash(prompt),
                effective_seed=effective_seed,runtime=runtime,runtime_observation_hash=content_hash(runtime))
            super().generate(state,item,seed)

    def _generate_cuda(self,state,prompt,settings,seed):
        if seed not in self._pending:
            raise RuntimeError('Batched generation must precede scoring')
        return self._pending.pop(seed)

    def generate(self,state,item,seed=42):
        identity = self.identity(state,item,seed)
        if not (self.cache_root/'raw'/state['state_id']/(content_hash(identity)+'.json')).exists():
            self.generate_batch(state,[item],seed)
        return super().generate(state,item,seed)


def evaluation_generator(cache_root, backend='native'):
    if backend == 'vllm':
        return VLLMEvaluationGenerator(cache_root)
    if backend != 'native':
        raise ValueError('Unknown measurement backend')
    return NativeGenerator(cache_root)
