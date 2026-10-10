import pytest
from shared_procedures.records import *

def test_score_and_separate_boolean_audit_schema():
    assert validate_record('score_response',{'score':6})=={'score':6}
    with pytest.raises(RecordValidationError):validate_record('score_response',{'score':True})
    with pytest.raises(RecordValidationError):validate_record('rl_pass_b_response',{'readable_solution':'not_detected','padding':'not_detected'})
    with pytest.raises(RecordValidationError):validate_record('evaluation_audit_response',{'readable_solution':'absent','padding':'uncertain'})

def test_record_negative_not_incomplete():
    record=ArtifactRecord('outcome',{'config':'a'},{'correct':False})
    assert record.to_dict()['status']=='complete'
    with pytest.raises(RecordValidationError):ArtifactRecord('outcome',{}, {})

def test_count_uniqueness_and_unknown_schema():
    with pytest.raises(RecordValidationError):validate_record('not_a_kind',{})
    with pytest.raises(RecordValidationError):validate_records('candidate_prompt',[],expected_count=1)

def candidate():
    return {'example_id':'lmsys_1','prompt':'Question','source_id':'1','source_index':0,'source_dataset':'lmsys/lmsys-chat-1m',
            'source_split':'train','source_revision':'a'*40,'source_content_hash':content_hash('Question'),'deduplication_identity':'c'*64,
            'input_token_counts':{'7b':1,'14b':1},'normalization_version':'v1','extraction_version':'v1',
            'candidate_order':0,'status':'eligible'}

def test_candidate_provenance_counts_and_uniqueness():
    c=candidate();assert validate_record('candidate_prompt',c)==c
    del c['source_revision']
    with pytest.raises(RecordValidationError):validate_record('candidate_prompt',c)
    c=candidate();c['input_token_counts']['14b']=201
    with pytest.raises(RecordValidationError):validate_record('candidate_prompt',c)
    with pytest.raises(RecordValidationError):validate_records('candidate_prompt',[candidate(),candidate()])
