import pytest
from evaluation.shared_model_evaluation.item_evaluation.native_output.output_generation import NativeGenerator, render_prompt, validate_state


def state(**changes):
    return dict(state_id='base_7b', model='7b', family='base', model_revision='a'*40,
                tokenizer_revision='b'*40, checkpoint_hash='c'*64, runtime_hash='d'*64, **changes)


def test_identical_raw_reused_including_bad_ordinary_output(tmp_path):
    calls=[]
    def backend(s,p,c,seed):
        calls.append(seed)
        return {'raw_output':'malformed ordinary output','finish_reason':'length','generated_tokens':3072}
    generator=NativeGenerator(tmp_path,backend=backend)
    item={'question_id':'q1','domain':'gsm8k','question':'A public question'}
    first=generator.generate(state(),item)
    assert generator.generate(state(),item)==first
    assert len(calls)==1 and first['simulation'] is True
    assert first['identity']['settings']['max_new_tokens']==3072
    assert generator.generate(state(),item,seed=43)['request_hash'] != first['request_hash']
    assert len(calls)==2


def test_state_identity_change_invalidates_raw_cache(tmp_path):
    calls=[]
    def backend(*args):
        calls.append(True)
        return {'raw_output':'<think>x</think>A','finish_reason':'stop','generated_tokens':4}
    g=NativeGenerator(tmp_path,backend=backend)
    item={'question_id':'q','domain':'gsm8k','question':'A question'}
    s=state()
    assert g.generate(s,item)['request_hash'] != g.generate({**s,'tokenizer_revision':'e'*40},item)['request_hash']
    assert len(calls)==2
    with pytest.raises(ValueError,match='diagnostic'):
        g.identity({**s,'family':'rl','state_id':'rl_7b_step_010','adapter_path':'/adapter','step':10},item,43)


def test_stopping_metadata_tamper_is_preserved_and_rejected(tmp_path):
    from shared_procedures.storage import read_json, atomic_write_json
    calls=[]
    def backend(*args):
        calls.append(True)
        return {'raw_output':'<think>x</think>1','finish_reason':'length','generated_tokens':3072}
    generator=NativeGenerator(tmp_path,backend=backend)
    item={'question_id':'q','domain':'gsm8k','question':'A question'}
    generator.generate(state(),item)
    path=next((tmp_path/'raw'/'base_7b').glob('*.json'))
    corrupt=read_json(path)
    corrupt['finish_reason']='stop'
    atomic_write_json(path,corrupt)
    with pytest.raises(ValueError,match='integrity'):
        generator.generate(state(),item)
    assert len(calls)==1
    assert read_json(path)==corrupt


def test_mbpp_renderer_cannot_read_reference_or_tests():
    item={'domain':'mbpp','question':'Add two inputs','interface':{'signature':'def add(x, y=1):'},
          'reference_code':'PRIVATE_SOLUTION','scoring_tests':['SECRET_ASSERTION']}
    prompt,_=render_prompt(item)
    assert 'def add(x, y=1):' in prompt
    assert 'PRIVATE' not in prompt and 'SECRET' not in prompt
    with pytest.raises(ValueError):
        render_prompt({**item,'interface':'def f():\n return 123'})


def test_pinned_model_identity_required():
    s=state()
    with pytest.raises(ValueError): validate_state({**s,'model_revision':'main'})
    with pytest.raises(ValueError): validate_state({**s,'model_id':'another/model'})


def test_baseline_cannot_silently_load_trained_adapter():
    with pytest.raises(ValueError,match='Untouched baseline'):
        validate_state({**state(),'adapter_path':'/a/trained/adapter'})
    with pytest.raises(ValueError,match='mismatched size'):
        validate_state({**state(),'state_id':'base_14b'})
