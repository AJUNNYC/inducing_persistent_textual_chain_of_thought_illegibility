from evaluation.shared_model_evaluation.item_evaluation.legibility_judging.readable_solution_audit.readable_solution_audit import audit_readable_solution, AUDIT_VERSION


def test_exact_schema_information_boundary_and_uncertainty():
    class Client:
        def request(self,kind,model,prompt,trace,**kw):
            assert kind=='evaluation_audit' and model=='gpt-4o-2024-11-20'
            assert kw['question']=='PUBLIC TASK' and trace=='EXACT TRACE'
            assert kw['settings']=={'temperature':0,'max_tokens':64}
            assert kw['schema']['additionalProperties'] is False
            assert 'attempted solution' in prompt
            return {'status':'success','parsed':{'readable_solution':'uncertain','padding':'not_detected'}}
    record=audit_readable_solution('PUBLIC TASK','EXACT TRACE',Client())
    assert record['resolved'] is False and record['prompt_version']==AUDIT_VERSION


def test_provider_failure_never_becomes_not_detected():
    class Client:
        def request(self,*args,**kw): return {'status':'refusal'}
    record=audit_readable_solution('task','trace',Client())
    assert record['readable_solution'] is None and record['padding'] is None
    assert record['status']=='refusal'
