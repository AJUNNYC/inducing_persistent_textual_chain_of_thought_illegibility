import pytest
from shared_procedures.identity import content_hash
from shared_procedures.storage import read_json
from data_generation_collection.sft_data.clean_trace_collection.clean_trace_collection import collect_clean_traces


def manifest(n):
    records=[{'example_id':str(i),'prompt':'q'+str(i)} for i in range(n)]
    return {'candidates':records,'status':'verified','content_hash':content_hash(records)}


def processor(candidates,*args,**kwargs):
    # Explicit paired-generation test fixture: failures occur only on first IDs.
    return [{'example_id':c['example_id'],'prompt':c['prompt'],'status':'rejected' if int(c['example_id'])<2 else 'accepted',
             'traces':{m:{'status':'accepted'} for m in ('7b','14b')}} for c in candidates]


def test_initial_10100_and_exact_deficit_new_ids(tmp_path):
    result=collect_clean_traces(manifest(10102),None,{},kazitext=None,output_dir=tmp_path,config_hash='c'*64,batch_processor=processor)
    assert [b['size'] for b in result['batches']]==[10100,2]
    assert result['train_count']==10000 and result['validation_count']==100
    assert not set(result['train_ids'])&set(result['validation_ids'])
    assert len({i for b in result['batches'] for i in b['ids']})==10102
    assert result['rejected_ids']==['0','1']


def test_shortage_persists_failure_no_quota_change(tmp_path):
    with pytest.raises(ValueError,match='shortage'):
        collect_clean_traces(manifest(10101),None,{},kazitext=None,output_dir=tmp_path,config_hash='c'*64,batch_processor=processor)
    state=read_json(tmp_path/'collection_status.json')
    assert state['status']=='failed' and state['required_deficit']==2 and state['accepted_count']==10098
    assert not (tmp_path/'accepted_pool.json').exists()


def test_pause_records_exact_next_generation_position_without_frozen_pool(tmp_path):
    calls=[]
    def generate(row,model,settings):
        calls.append((row['example_id'],model))
        return {'raw_generation':'<think>one two three</think>answer','finish_reason':'stop',
                'generated_token_count':10,'model_revision':'a'*40,'tokenizer_revision':'b'*40}
    result=collect_clean_traces(manifest(10100),generate,{'7b':None,'14b':None},kazitext=None,
        output_dir=tmp_path,config_hash='c'*64,stop_requested=lambda:len(calls)==1)
    assert result['status']=='paused' and result['launch_ready'] is False
    assert result['next_batch_start_cursor']==0 and result['next_candidate_position']==1
    assert result['next_operation']['model_id']=='7b'
    assert result['next_operation']['raw_completed_ids']=={'7b':['0'],'14b':[]}
    assert result['required_deficit']==10100 and len(calls)==1
    assert read_json(tmp_path/'collection_status.json')==result
    assert not (tmp_path/'accepted_pool.json').exists()


def test_default_external_stop_file_pauses_before_loading_or_generating(tmp_path):
    (tmp_path/'STOP_REQUESTED').write_text('budget controller requested stop')
    def forbidden(*args,**kwargs):raise AssertionError('work must not start')
    result=collect_clean_traces(manifest(10100),None,{},kazitext=None,output_dir=tmp_path,
        config_hash='c'*64,batch_processor=forbidden)
    assert result['status']=='paused' and result['next_candidate_position']==0
