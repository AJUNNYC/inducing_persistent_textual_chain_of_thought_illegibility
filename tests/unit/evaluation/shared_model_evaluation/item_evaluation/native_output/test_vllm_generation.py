"""Isolated backend contract tests; these are not live scientific evidence."""
import sys
from types import SimpleNamespace
import pytest
from evaluation.shared_model_evaluation.item_evaluation.native_output.vllm_generation import VLLMEvaluationGenerator, BACKEND, VERSION
from evaluation.shared_model_evaluation.item_evaluation.native_output.output_generation import NativeGenerator
from shared_procedures.identity import derive_seed
STATE={'state_id':'base_7b','model':'7b','family':'base','model_revision':'a'*40,
       'tokenizer_revision':'b'*40,'checkpoint_hash':'c'*64,'runtime_hash':'d'*64}
ITEM={'question_id':'q','domain':'gsm8k','question':'What is 2 + 2?'}

class Tokenizer:
    eos_token='<eos>'
    def apply_chat_template(self,*a,**kw):return 'public prompt<think>\n'
    def encode(self,*a,**kw):return [1,2]
    def decode(self,*a,**kw):return 'Reasoning.</think>4<eos>'

def make_generator(tmp_path,monkeypatch):
    params=[]
    monkeypatch.setitem(sys.modules,'vllm',SimpleNamespace(SamplingParams=lambda **kw: params.append(kw) or kw))
    generator=VLLMEvaluationGenerator(tmp_path)
    monkeypatch.setattr(generator,'_load',lambda state:None)
    generator._tokenizer=Tokenizer();generator.eos_ids=[9]
    def generate(inputs,settings,**kw):
        assert kw['lora_request'] is None
        return [SimpleNamespace(prompt_token_ids=p['prompt_token_ids'],outputs=[SimpleNamespace(
            token_ids=[3,4,9],finish_reason='stop')]) for p in inputs]
    generator.engine=SimpleNamespace(generate=generate)
    return generator,params

def test_batch_retains_individual_budget_seeds_and_exact_cache(tmp_path,monkeypatch):
    generator,params=make_generator(tmp_path,monkeypatch)
    second={**ITEM,'question_id':'q2'}
    generator.generate_batch(STATE,[ITEM,second],42)
    assert len(params)==2 and all(p['max_tokens']==3072 for p in params)
    assert [p['seed'] for p in params]==[derive_seed('native_evaluation',42,'base_7b',q) for q in ('q','q2')]
    record=generator.generate(STATE,ITEM)
    assert record['identity']['version']==VERSION and record['native_prefix']=='<think>\n'
    assert record['raw_output']=='Reasoning.</think>4' and record['simulation'] is False
    assert record['identity']!=NativeGenerator(tmp_path).identity(STATE,ITEM)
    generator.generate_batch(STATE,[ITEM,second],42)
    assert len(params)==2

def test_context_overflow_rejected_without_truncation(tmp_path,monkeypatch):
    generator,params=make_generator(tmp_path,monkeypatch)
    generator._tokenizer.encode=lambda *a,**kw:[1]*(BACKEND['max_model_len']-3072+1)
    with pytest.raises(ValueError,match='truncation'):
        generator.generate_batch(STATE,[ITEM])
    assert not params
