import importlib
import pytest
module=importlib.import_module('evaluation.evaluation')


def test_all_scientific_states_required(monkeypatch):
    monkeypatch.setattr(module,'validate_protocol',lambda p:None)
    with pytest.raises(ValueError,match='62'): module.run_evaluation([],{},None)


def test_failed_group_cannot_reach_analysis(monkeypatch):
    monkeypatch.setattr(module,'validate_protocol',lambda p:None)
    states=[{'state_id':'base_'+size,'model':size,'family':'base'} for size in ['7b','14b']]
    for family in ['rot13','kazitext']:
        states.extend({'state_id':f'{family}_{size}_level_{level:02d}','family':family,'model':size,'level':level}
            for size in ['7b','14b'] for level in range(1,11))
    states.extend({'state_id':f'rl_{size}_step_{step:03d}','family':'rl','model':size,'step':step}
        for size in ['7b','14b'] for step in range(10,101,10))
    calls=[]
    monkeypatch.setattr(module,'evaluate_conditions',lambda *a,**kw:{'status':'incomplete','missing_conditions':['missing']})
    result=module.run_evaluation(states,{'protocol_hash':'x'},None,analysis=lambda *a,**k:calls.append(True))
    assert result['status']=='incomplete' and result['analysis'] is None and not calls
