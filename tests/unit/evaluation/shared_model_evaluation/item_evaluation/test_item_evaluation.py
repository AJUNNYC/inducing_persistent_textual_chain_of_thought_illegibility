from evaluation.shared_model_evaluation.item_evaluation.item_evaluation import evaluate_item
from shared_procedures.identity import content_hash


STATE={'state_id':'base_7b','model':'7b','family':'base'}
ITEM={'question_id':'q','domain':'gsm8k','question':'task','reference_answer':'4'}
PROTOCOL={'protocol_hash':'f'*64,'legibility_ids':['q']}

class Generator:
    def __init__(self,raw='<think>Some complete reasoning.</think>4',finish='stop'):
        self.calls=0;self.raw=raw;self.finish=finish
    def generate(self,*args,**kw):
        self.calls+=1
        return {'raw_output':self.raw,'raw_hash':content_hash(self.raw),'finish_reason':self.finish,'simulation':True}

class Client:
    def __init__(self): self.calls=[]
    def request(self,kind,*args,**kw):
        self.calls.append(kind)
        return {'status':'success','simulation':True,'parsed':{'readable_solution':'not_detected','padding':'uncertain'}
                if kind=='evaluation_audit' else {'score':7}}


def test_branches_share_one_raw_and_exact_cache(tmp_path):
    g=Generator();c=Client()
    first=evaluate_item(STATE,ITEM,PROTOCOL,g,c,cache_root=tmp_path)
    assert first['correct'] is True and first['qualified_opacity'] is True
    assert first['status']=='complete' and first['simulation'] is True
    assert evaluate_item(STATE,ITEM,PROTOCOL,g,c,cache_root=tmp_path)==first
    assert g.calls==1 and len(c.calls)==3


def test_ordinary_malformed_retains_denominator_no_resample(tmp_path):
    g=Generator('no reasoning delimiter');c=Client()
    result=evaluate_item(STATE,ITEM,PROTOCOL,g,c,cache_root=tmp_path)
    assert result['status']=='complete' and result['correct'] is False
    assert result['qualified_opacity'] is False and result['legibility']['status']=='not_assessable'
    assert not c.calls
    evaluate_item(STATE,ITEM,PROTOCOL,g,c,cache_root=tmp_path)
    assert g.calls==1


def test_nonselection_and_diagnostics_never_judged():
    c=Client()
    result=evaluate_item(STATE,ITEM,{**PROTOCOL,'legibility_ids':[]},Generator(),c)
    assert result['legibility']['status']=='not_scheduled' and result['selected'] is False
    diagnostic=evaluate_item(STATE,ITEM,PROTOCOL,Generator(),c,seed=43)
    assert diagnostic['legibility']['status']=='not_scheduled' and not c.calls


def test_generation_infrastructure_failure_is_incomplete():
    class Failed:
        def generate(self,*args,**kw): raise OSError('not available')
    result=evaluate_item(STATE,ITEM,PROTOCOL,Failed())
    assert result['correct'] is None and result['status']=='incomplete' and result['raw'] is None


def test_deferred_baseline_diagnostics_do_not_claim_judge_evidence(tmp_path):
    g=Generator();c=Client()
    diagnostic=evaluate_item(STATE,ITEM,PROTOCOL,g,c,cache_root=tmp_path,defer_judging=True)
    assert diagnostic['status']=='complete' and diagnostic['legibility']['status']=='deferred'
    assert diagnostic['correct'] is True and diagnostic['scientific_evidence_status']=='indeterminate'
    assert not c.calls and diagnostic['qualified_opacity'] is None
    full=evaluate_item(STATE,ITEM,PROTOCOL,g,c,cache_root=tmp_path)
    assert full['legibility']['status']=='complete' and len(c.calls)==3
