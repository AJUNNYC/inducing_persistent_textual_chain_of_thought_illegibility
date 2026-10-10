"""Fixture-only cross-owner contracts, never empirical model/judge evidence."""
import pytest
from evaluation.shared_model_evaluation.item_evaluation.native_output.output_generation import NativeGenerator
from evaluation.shared_model_evaluation.item_evaluation.item_evaluation import evaluate_item
from shared_procedures.judge_client import JudgeClient
from shared_procedures.records import validate_record
from shared_procedures.identity import content_hash
from data_generation_collection.evaluation_data.protocol.analysis_plan.prospective_analysis_plan import create_analysis_plan
from data_generation_collection.evaluation_data.protocol.analysis_plan.simultaneous_bounds.simultaneous_bounds import qualified_opacity
from evaluation.results_analysis.results_analysis import analyze_results
from data_generation_collection.evaluation_data.protocol.evaluation_protocol import normalize_preflight_reports


STATE={'state_id':'base_7b','model':'7b','family':'base','model_revision':'a'*40,
       'tokenizer_revision':'b'*40,'checkpoint_hash':'c'*64,'runtime_hash':'d'*64}


def test_native_scorer_judge_schema_analysis_information_boundary(tmp_path):
    requests=[]
    def transport(request):
        requests.append(request)
        assert request['trace']==' exact fixture reasoning '
        assert 'FINAL_PRIVATE' not in str(request) and 'REFERENCE_PRIVATE' not in str(request)
        if request['kind']=='evaluation_audit':
            assert request['question']=='What is two plus two?'
            text='{"readable_solution":"not_detected","padding":"uncertain"}'
        else:
            assert request['question'] is None
            text='{"score":7}'
        return {'text':text,'model':request['model'],'usage':{'input_tokens':30,'output_tokens':10},'request_id':'fixture'}
    client=JudgeClient(tmp_path/'judges',tmp_path/'state',transport=transport)
    generator=NativeGenerator(tmp_path/'evaluation',backend=lambda *args:{
        'raw_output':'<think> exact fixture reasoning </think>Final answer: \\boxed{4}',
        'finish_reason':'stop','generated_tokens':16})
    item={'question_id':'q','domain':'gsm8k','question':'What is two plus two?','reference_answer':'4'}
    protocol={'protocol_hash':'f'*64,'legibility_ids':['q']}
    record=evaluate_item(STATE,item,protocol,generator,client,cache_root=tmp_path/'evaluation')
    validate_record('evaluation_record',record)
    assert record['correct'] is True and record['qualified_opacity'] is True and record['simulation'] is True
    assert qualified_opacity(record)==(1,'resolved')
    assert len(requests)==3
    assert all(request['settings']['max_tokens']==64 for request in requests)
    repeated=evaluate_item(STATE,item,protocol,generator,client,cache_root=tmp_path/'evaluation')
    assert repeated==record and len(requests)==3


def test_missing_primary_audit_does_not_become_opacity_or_incorrect_answer(tmp_path):
    def transport(request):
        return {'text':'{"readable_solution":"uncertain","padding":"not_detected"}' if request['kind']=='evaluation_audit' else '{"score":8}',
                'model':request['model'],'usage':{'input_tokens':1,'output_tokens':1},'request_id':'fixture'}
    client=JudgeClient(tmp_path/'judges',tmp_path/'state',transport=transport)
    generator=NativeGenerator(tmp_path/'evaluation',backend=lambda *args:{
        'raw_output':'<think>trace</think>4','finish_reason':'stop','generated_tokens':5})
    item={'question_id':'q','domain':'gsm8k','question':'task','reference_answer':'4'}
    record=evaluate_item(STATE,item,{'protocol_hash':'x','legibility_ids':['q']},generator,client)
    validate_record('evaluation_record',record)
    assert record['correct'] is True and record['qualified_opacity'] is None
    assert qualified_opacity(record)==(0,'readable_solution_unresolved')
    plan=create_analysis_plan([{'question_id':'q','domain':'gsm8k','population_size':20,'stratum':'tercile_1'}],['q'],{'fixture':True})
    result=analyze_results({'base_7b':[record]},plan,confirmatory=True)
    assert result['simulation'] is True and result['confirmatory_authorized'] is False
    assert result['bounds'] is None


def test_preflight_hash_and_configuration_alias_binding():
    report={'configuration_hash':'x','checks':[{'name':'model_runtime','status':'blocked','evidence':{'live':False}}]}
    report['report_hash']=content_hash(report)
    mapped,configuration=normalize_preflight_reports(report)
    assert configuration=='x' and mapped['model_runtime']['status']=='blocked'
    report['checks'][0]['status']='passed'
    with pytest.raises(ValueError,match='integrity'): normalize_preflight_reports(report)
    mapped,_=normalize_preflight_reports({'native_generation':{'status':'blocked'},'judge_compatibility':{'status':'blocked'}})
    assert set(mapped)=={'model_runtime','judge_services'}
