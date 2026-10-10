from evaluation.shared_model_evaluation.item_evaluation.legibility_judging.legibility_judging import judge_legibility, qualified_opacity, summarize_legibility


class Client:
    def __init__(self, readable='not_detected'): self.calls=[];self.readable=readable
    def request(self,kind,model,prompt,trace,**kw):
        self.calls.append((kind,trace,kw))
        assert trace==' exact\ntrace '
        assert 'Final' not in trace
        if kind=='evaluation_audit':
            assert kw['question']=='task'
            return {'status':'success','parsed':{'readable_solution':self.readable,'padding':'present'}}
        assert 'question' not in kw
        return {'status':'success','parsed':{'score':9 if kind=='evaluation_primary' else 1}}


def test_three_independent_calls_padding_not_penalized():
    c=Client();record=judge_legibility('task',' exact\ntrace ',c)
    assert len(c.calls)==3 and record['qualified_opacity'] is True
    assert record['primary_score']==9 and record['secondary_score']==1
    assert record['padding']=='present'
    assert len({kind for kind,_,_ in c.calls})==3


def test_uncertain_readable_audit_blocks_qualification():
    assert judge_legibility('task',' exact\ntrace ',Client('uncertain'))['qualified_opacity'] is None
    assert qualified_opacity(assessable=True,primary_score=8,readable_solution='present') is False
    assert qualified_opacity(assessable=True,primary_score=8,readable_solution=None) is None


def test_unassessable_or_unselected_never_scored():
    c=Client()
    assert judge_legibility('task',None,c)['qualified_opacity'] is False
    assert judge_legibility('task',' exact\ntrace ',c,selected=False)['status']=='not_scheduled'
    assert not c.calls


def test_missingness_and_undefined_kappa_kept():
    records=[{'domain':domain,'status':'complete','primary_score':6,'secondary_score':6,
        'qualified_opacity':True,'padding':'uncertain'} for domain in ['gsm8k','mmlu','arc_easy','mbpp'] for _ in range(50)]
    result=summarize_legibility(records)
    assert result['quadratic_weighted_kappa'] is None
    assert result['exact_agreement']==1 and result['confusion_matrix'][5][5]==200
    assert result['macro_mean']['primary']==6
    records[0].update(primary_score=None,qualified_opacity=None,status='incomplete')
    result=summarize_legibility(records)
    assert result['denominator']==200 and result['unresolved_qualified_count']==1
