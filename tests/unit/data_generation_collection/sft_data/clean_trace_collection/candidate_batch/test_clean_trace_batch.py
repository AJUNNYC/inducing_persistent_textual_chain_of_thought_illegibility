import pytest
from shared_procedures.identity import content_hash
from data_generation_collection.sft_data.clean_trace_collection.candidate_batch.clean_trace_batch import process_clean_batch,validate_clean_trace,total_sequence_tokens,native_completion
from data_generation_collection.sft_data.clean_trace_collection.candidate_manifest import prepare_candidate_manifest


class Tokenizer:
    def encode(self,text,**kwargs):return list(range(len(text.split())))
    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=False,**kwargs):
        texts=[]
        for m in messages:
            text=m['content']
            if m['role']=='assistant' and '</think>' in text:text=text.split('</think>')[-1]
            texts.append('<'+m['role']+'>'+text+'<end>')
        value=''.join(texts)
        return self.encode(value) if tokenize else value


class FixtureNoise:
    def transform(self,reasoning,example_id,level,model_id):
        # Explicit isolated test double: no fake scientific KaziText artifact.
        return {'reasoning':reasoning,'content_hash':content_hash(reasoning),'source_hash':content_hash(reasoning),
                'status':'verified','protected_content_valid':True,'changed_word_fraction':0}


def raw(text='<think>one two three</think> answer',finish='stop'):
    return {'raw_generation':text,'finish_reason':finish,'generated_token_count':20,
            'model_revision':'a'*40,'tokenizer_revision':'b'*40,'seed':42}


def candidate():
    return {'example_id':'id','prompt':'question','source_content_hash':content_hash('question')}


def test_joint_retention_raw_cache_and_ordinary_rejection_not_resampled(tmp_path):
    calls=[]
    def generator(c,m,settings):
        calls.append((c['example_id'],m));assert settings['temperature']==.6 and settings['max_new_tokens']==3072
        return raw(finish='length' if m=='14b' else 'stop')
    kwargs={'kazitext':FixtureNoise(),'output_dir':tmp_path,'config_hash':'c'*64}
    result=process_clean_batch([candidate()],generator,{m:Tokenizer() for m in ('7b','14b')},**kwargs)
    assert result[0]['status']=='rejected' and set(result[0]['traces'])=={'7b','14b'}
    assert len(calls)==2
    process_clean_batch([candidate()],generator,{m:Tokenizer() for m in ('7b','14b')},**kwargs)
    assert len(calls)==2
    assert len(list((tmp_path/'raw').rglob('*.json')))==2


def test_short_wrong_reasoning_is_structurally_accepted():
    result=validate_clean_trace(candidate(),'7b',raw('<think>one two wrong</think> 999'),Tokenizer(),kazitext=FixtureNoise(),config_hash='c'*64)
    assert result['status']=='accepted' and len(result['preflight_variants'])==20
    assert result['final_answer']==' 999'


def test_native_history_template_cannot_drop_reasoning_from_token_budget():
    tokenizer=Tokenizer()
    long=' '.join(['one']*4100)
    count=total_sequence_tokens(tokenizer,'question',native_completion(long,' answer'))
    assert count>4096
    result=validate_clean_trace(candidate(),'7b',raw('<think>'+long+'</think> answer'),tokenizer,kazitext=FixtureNoise(),config_hash='c'*64)
    assert result['status']=='rejected' and 'clean_sequence_exceeds_4096' in result['rejection_reasons']


def test_nonempty_separation_three_words_and_all_condition_checks():
    for text in ['answer only','<think>one two</think>answer','<think>one two three</think>']:
        result=validate_clean_trace(candidate(),'7b',raw(text),Tokenizer(),kazitext=FixtureNoise(),config_hash='c'*64)
        assert result['status']=='rejected'
    class BrokenNoise(FixtureNoise):
        def transform(self,*args,**kwargs):raise RuntimeError('required profile unavailable')
    with pytest.raises(RuntimeError):validate_clean_trace(candidate(),'7b',raw(),Tokenizer(),kazitext=BrokenNoise(),config_hash='c'*64)


def test_operational_pause_syncs_completed_raw_and_resume_never_regenerates(tmp_path, monkeypatch):
    from data_generation_collection.sft_data.clean_trace_collection.candidate_batch import clean_trace_batch as module
    from data_generation_collection.sft_data.clean_trace_collection.candidate_batch.clean_trace_batch import CleanDataPause
    calls=[];synced=[]
    rows=[{**candidate(),'example_id':name} for name in ('first','second')]
    def generator(row,model,settings):
        calls.append((row['example_id'],model));return raw()
    monkeypatch.setattr(module,'notify_artifact',lambda path,meta:synced.append(path))
    kwargs={'kazitext':FixtureNoise(),'output_dir':tmp_path,'config_hash':'c'*64}
    with pytest.raises(CleanDataPause) as paused:
        process_clean_batch(rows,generator,{m:Tokenizer() for m in ('7b','14b')},stop_requested=lambda:len(calls)==1,**kwargs)
    assert paused.value.progress['example_id']=='second'
    assert paused.value.progress['model_id']=='7b'
    assert paused.value.progress['raw_completed_ids']=={'7b':['first'],'14b':[]}
    assert len(synced)==1 and synced[0].is_file()
    resumed=process_clean_batch(rows,generator,{m:Tokenizer() for m in ('7b','14b')},stop_requested=lambda:False,**kwargs)
    assert len(calls)==4 and len(set(calls))==4
    assert [row['example_id'] for row in resumed]==['first','second']
    assert all(row['status']=='accepted' for row in resumed)
