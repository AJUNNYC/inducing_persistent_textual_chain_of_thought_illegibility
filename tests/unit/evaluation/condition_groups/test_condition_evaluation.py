import importlib
import pytest
module=importlib.import_module('evaluation.condition_groups.condition_evaluation')
BASES=[{'state_id':'base_7b','model':'7b'},{'state_id':'base_14b','model':'14b'}]


def test_frozen_matrix_cardinality():
    assert len(module.expected_condition_ids('sft'))==40
    assert len(module.expected_condition_ids('rl'))==20
    with pytest.raises(ValueError): module.evaluate_conditions('rl',[],BASES,{},None)


def test_baseline_incomplete_prevents_trained_comparison(monkeypatch):
    calls=[]
    def runner(state,*args,**kw):
        calls.append(state['state_id']);return {'status':'incomplete','state_id':state['state_id']}
    monkeypatch.setattr(module,'evaluate_model',runner)
    state={'state_id':'rl_7b_step_010','model':'7b','family':'rl','step':10}
    result=module.evaluate_conditions('rl',[state],BASES,{},None,require_full_matrix=False)
    assert calls==['base_7b'] and result['missing_conditions']==['rl_7b_step_010']


def test_conditions_preserve_family_and_checkpoint_labels(monkeypatch):
    monkeypatch.setattr(module,'evaluate_model',lambda state,*args,**kw:{'status':'complete','state_id':state['state_id']})
    state={'state_id':'rl_7b_step_010','model':'7b','family':'rl','step':20}
    with pytest.raises(ValueError,match='snapshot'):
        module.evaluate_conditions('rl',[state],BASES,{},None,require_full_matrix=False)
