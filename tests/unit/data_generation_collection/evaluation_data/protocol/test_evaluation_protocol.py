import pytest
from pathlib import Path
from shared_procedures.identity import content_hash,file_hash
from evaluation.shared_model_evaluation.item_evaluation.capability_scoring.mbpp_sandbox import _selected_reference_binding,DockerMBPPSandbox,DEFAULT_MBPP_IMAGE
from data_generation_collection.evaluation_data.protocol.evaluation_protocol import build_evaluation_protocol, freeze_evaluation_protocol, validate_protocol, REQUIRED_PREFLIGHTS


def panel():
    items=[];selected=[]
    for domain,count in [('gsm8k',300),('mmlu',399),('arc',331),('mbpp',203)]:
        for i in range(count):
            row={'question_id':f'{domain}_{i}','domain':domain,'question':'A public task',
                 'source_revision':'a'*40,'reference_answer':'1'}
            if domain in {'mmlu','arc'}: row['choices']=[{'label':'1','text':'one'},{'label':'2','text':'two'}]
            if domain=='mbpp': row.update(interface={'name':'add','signature':'def add(a,b):'},public_interface=[{'name':'add','signature':'def add(a,b):'}],scoring_tests=['assert add(1,2)==3'],reference_code='def add(a,b): return a+b',source_content_hash='c'*64,public_interface_hash='d'*64)
            items.append(row)
            if i<50: selected.append(row['question_id'])
    return {'items':items}, {'question_ids':selected}


def test_protocol_cannot_release_without_launch_gates():
    capability,legibility=panel()
    draft=build_evaluation_protocol(capability,legibility)
    assert draft['status']=='draft' and draft['specifications']['generation']['settings']['max_new_tokens']==3072
    assert draft['specifications']['judges']['audit']['settings']['max_tokens']==64
    assert set(REQUIRED_PREFLIGHTS)<=set(draft['blocked_preflights'])
    with pytest.raises(RuntimeError): freeze_evaluation_protocol(capability,legibility)
    validate_protocol(draft,allow_draft=True)


def test_complete_prospective_spec_and_integrity():
    capability,legibility=panel()
    reports={name:{'status':'passed','evidence':{'live':True,'simulation':False,'configuration_hash':'e'*64}} for name in REQUIRED_PREFLIGHTS}
    reports['mbpp_sandbox']['evidence'].update(selected_reference_status='passed',selected_count=203,selected_passed_count=203,
        selected_binding_hash=content_hash(_selected_reference_binding(capability)),sandbox_identity=DockerMBPPSandbox(DEFAULT_MBPP_IMAGE).identity,implementation_hash=file_hash(Path('src/evaluation/shared_model_evaluation/item_evaluation/capability_scoring/mbpp_sandbox.py')))
    protocol=freeze_evaluation_protocol(capability,legibility,analysis_plan={'status':'frozen','plan_hash':'b'*64},preflight_reports=reports)
    validate_protocol(protocol)
    protocol['items'][0]['question']='tampered'
    with pytest.raises(ValueError,match='integrity'): validate_protocol(protocol)


def test_fixed_counts_unique_and_nested_ids_enforced():
    capability,legibility=panel()
    capability['items'].pop()
    with pytest.raises(ValueError,match='1,233'): build_evaluation_protocol(capability,legibility)
    capability,legibility=panel();legibility['question_ids'][0]='not_selected'
    with pytest.raises(ValueError,match='nested'): build_evaluation_protocol(capability,legibility)


def test_wrong_selected_reference_certificate_does_not_freeze():
    capability,legibility=panel()
    reports={name:{'status':'passed','evidence':{'live':True,'simulation':False,'configuration_hash':'e'*64}} for name in REQUIRED_PREFLIGHTS}
    reports['mbpp_sandbox']['evidence'].update(selected_reference_status='passed',selected_count=203,selected_passed_count=203,
        selected_binding_hash='f'*64,sandbox_identity=DockerMBPPSandbox(DEFAULT_MBPP_IMAGE).identity,implementation_hash=file_hash(Path('src/evaluation/shared_model_evaluation/item_evaluation/capability_scoring/mbpp_sandbox.py')))
    with pytest.raises(RuntimeError,match='mbpp_reference_binding'):
        freeze_evaluation_protocol(capability,legibility,analysis_plan={'status':'frozen','plan_hash':'b'*64},preflight_reports=reports)


@pytest.mark.parametrize('stale_field',['harness_hash','implementation_hash'])
def test_stale_executor_evidence_does_not_certify_current_code(stale_field):
    capability,legibility=panel()
    reports={name:{'status':'passed','evidence':{'live':True,'simulation':False,'configuration_hash':'e'*64}} for name in REQUIRED_PREFLIGHTS}
    identity=dict(DockerMBPPSandbox(DEFAULT_MBPP_IMAGE).identity)
    implementation_hash=file_hash(Path('src/evaluation/shared_model_evaluation/item_evaluation/capability_scoring/mbpp_sandbox.py'))
    if stale_field=='harness_hash': identity['harness_hash']='0'*64
    else: implementation_hash='0'*64
    reports['mbpp_sandbox']['evidence'].update(selected_reference_status='passed',selected_count=203,selected_passed_count=203,
        selected_binding_hash=content_hash(_selected_reference_binding(capability)),sandbox_identity=identity,implementation_hash=implementation_hash)
    with pytest.raises(RuntimeError,match='mbpp_reference_binding'):
        freeze_evaluation_protocol(capability,legibility,analysis_plan={'status':'frozen','plan_hash':'b'*64},preflight_reports=reports)
