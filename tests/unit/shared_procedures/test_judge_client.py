import json
import pytest
from shared_procedures.judge_client import *

def client(tmp_path,transport):return JudgeClient(tmp_path/'cache',tmp_path/'state',transport=transport)

def test_independent_cached_passes_and_provenance(tmp_path):
    calls=[]
    def transport(request):
        calls.append(request)
        text=json.dumps({'readable_solution_present':False,'obvious_padding_present':True}) if request['kind']=='rl_pass_b' else '{"score":7}'
        return {'text':text,'model':request['model'],'usage':{'input_tokens':100,'output_tokens':4},'request_id':'fixture'}
    c=client(tmp_path,transport)
    a=c.request('rl_pass_a','claude-sonnet-5-5','prompt','trace')
    b=c.request('rl_pass_b','claude-sonnet-5-5','prompt','trace',question='q')
    assert a['parsed']=={'score':7} and b['parsed']['obvious_padding_present'] is True
    assert c.request('rl_pass_a','claude-sonnet-5-5','prompt','trace')==a
    assert len(calls)==2 and a['request_hash']!=b['request_hash'] and a['simulation'] is True

@pytest.mark.parametrize('text',['{"score":true}','{"score":10}','{"score":6,"x":1}','{"score":1,"score":7}','not json'])
def test_schema_failures_never_default_score(tmp_path,text):
    c=client(tmp_path,lambda r:{'text':text,'model':r['model'],'usage':{}})
    record=c.request('evaluation_primary','gpt-4o-2024-11-20','p','trace')
    assert record['status']=='schema_error' and 'parsed' not in record and record['raw_response']==text

def test_wrong_returned_model_and_question_exclusion(tmp_path):
    c=client(tmp_path,lambda r:{'text':'{"score":9}','model':'other','usage':{}})
    assert c.request('evaluation_primary','gpt-4o-2024-11-20','p','trace')['status']=='model_mismatch'
    with pytest.raises(ValueError):c.request('evaluation_primary','gpt-4o-2024-11-20','p','trace',question='hidden')
    with pytest.raises(ValueError):c.request('evaluation_audit','gpt-4o-2024-11-20','p','trace')

def test_live_without_authorization_blocked_before_transport(tmp_path):
    c=JudgeClient(tmp_path/'cache',tmp_path/'state')
    record=c.request('evaluation_primary','gpt-4o-2024-11-20','p','trace')
    assert record['status']=='blocked' and 'parsed' not in record

def test_provider_timeout_finite_across_restart(tmp_path,monkeypatch):
    calls=[]
    def transport(request):calls.append(1);raise JudgeError('transient',status='timeout',retryable=True,retry_after=0)
    c=client(tmp_path,transport)
    assert c.request('evaluation_primary','gpt-4o-2024-11-20','p','trace')['status']=='timeout'
    assert len(calls)==4
    c2=client(tmp_path,transport);c2.request('evaluation_primary','gpt-4o-2024-11-20','p','trace')
    assert len(calls)==4

def test_success_cache_mutation_detected(tmp_path):
    c=client(tmp_path,lambda r:{'text':'{"score":7}','model':r['model'],'usage':{'input_tokens':10,'output_tokens':4}})
    record=c.request('evaluation_primary','gpt-4o-2024-11-20','p','trace')
    path=tmp_path/'cache'/'evaluation_primary'/(record['request_hash']+'.json')
    from shared_procedures.storage import atomic_write_json,read_json
    corrupted=read_json(path);corrupted['parsed']['score']=9;atomic_write_json(path,corrupted)
    with pytest.raises(JudgeError):c.request('evaluation_primary','gpt-4o-2024-11-20','p','trace')

def test_persistent_rate_limit_before_sending(tmp_path):
    calls=[]
    def transport(request):calls.append(1);return {'text':'{"score":7}','model':request['model'],'usage':{}}
    limits={'openai':{'requests_per_minute':1,'tokens_per_minute':10000}}
    c=JudgeClient(tmp_path/'cache',tmp_path/'state',transport=transport,quota_limits=limits)
    assert c.request('evaluation_primary','gpt-4o-2024-11-20','p','trace')['status']=='success'
    assert c.request('evaluation_primary','gpt-4o-2024-11-20','p','another trace')['status']=='blocked'
    assert len(calls)==1
    c2=JudgeClient(tmp_path/'cache',tmp_path/'state',transport=transport,quota_limits=limits)
    assert c2.request('evaluation_primary','gpt-4o-2024-11-20','p','third trace')['status']=='blocked'
    assert len(calls)==1

def test_fixture_cache_cannot_satisfy_production(tmp_path):
    fake=client(tmp_path,lambda r:{'text':'{"score":7}','model':r['model'],'usage':{}})
    fixture=fake.request('evaluation_primary','gpt-4o-2024-11-20','p','trace')
    live=JudgeClient(tmp_path/'cache',tmp_path/'state')
    production=live.request('evaluation_primary','gpt-4o-2024-11-20','p','trace')
    assert fixture['status']=='success' and production['status']=='blocked'
    assert fixture['request_hash']!=production['request_hash'] and production['simulation'] is False

def test_crash_after_raw_persistence_recovers_without_duplicate_call(tmp_path):
    calls=[]
    def transport(r):calls.append(1);return {'text':'{"score":7}','model':r['model'],'usage':{'input_tokens':10,'output_tokens':4}}
    c=client(tmp_path,transport);record=c.request('evaluation_primary','gpt-4o-2024-11-20','p','trace')
    path=tmp_path/'cache'/'evaluation_primary'/(record['request_hash']+'.json')
    from shared_procedures.storage import read_json
    from shared_procedures.judge_client import _persist_record
    interrupted=read_json(path);interrupted['status']='sent';interrupted.pop('parsed');interrupted['attempts'][-1]['status']='sent'
    _persist_record(path,interrupted)
    restored=c.request('evaluation_primary','gpt-4o-2024-11-20','p','trace')
    assert restored['status']=='success' and restored['parsed']=={'score':7} and len(calls)==1
