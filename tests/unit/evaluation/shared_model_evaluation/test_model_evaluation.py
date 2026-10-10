import importlib
from shared_procedures.identity import content_hash
module=importlib.import_module('evaluation.shared_model_evaluation.model_evaluation')


def test_runner_retains_fixed_panel_and_stops_incomplete_infrastructure(monkeypatch):
    monkeypatch.setattr(module,'validate_state',lambda state:None)
    monkeypatch.setattr(module,'validate_protocol',lambda protocol,**kw:None)
    calls=[]
    def item_runner(state,item,*args,**kw):
        calls.append(item['question_id'])
        return {'question_id':item['question_id'],'domain':item['domain'],'status':'incomplete' if len(calls)==2 else 'complete',
                'correct':None if len(calls)==2 else False,'selected':False,'legibility':{'status':'not_scheduled'},
                'scientific_evidence_status':'recorded'}
    monkeypatch.setattr(module,'evaluate_item',item_runner)
    protocol={'protocol_hash':'x','items':[{'question_id':str(i),'domain':'gsm8k'} for i in range(3)]}
    result=module.evaluate_model({'state_id':'base_7b','model':'7b'},protocol,None)
    assert result['status']=='incomplete' and calls==['0','1']
    assert result['missing_question_ids']==['2'] and result['incomplete_question_ids']==['1']
    assert result['capability_summary']['gsm8k']['denominator']==3
    assert result['capability_summary']['gsm8k']['accuracy'] is None


def test_ordinary_wrong_outcomes_complete_and_accuracy_zero(monkeypatch):
    monkeypatch.setattr(module,'validate_state',lambda state:None)
    monkeypatch.setattr(module,'validate_protocol',lambda protocol,**kw:None)
    monkeypatch.setattr(module,'evaluate_item',lambda state,item,*args,**kw:{'question_id':item['question_id'],
        'domain':'gsm8k','status':'complete','correct':False,'selected':False,'legibility':{},'scientific_evidence_status':'recorded'})
    result=module.evaluate_model({'state_id':'base_7b','model':'7b'},{'protocol_hash':'x','items':[{'question_id':'q','domain':'gsm8k'}]},None)
    assert result['status']=='complete' and result['capability_summary']['gsm8k']['accuracy']==0
