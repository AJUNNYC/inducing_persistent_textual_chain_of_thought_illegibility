import pytest
import torch
from shared_procedures.identity import content_hash
from data_generation_collection.sft_data.clean_trace_collection.candidate_batch.clean_trace_batch import process_clean_batch,validate_clean_trace,total_sequence_tokens,native_completion
from data_generation_collection.sft_data.clean_trace_collection.candidate_manifest import prepare_candidate_manifest


def test_batched_sampling_keeps_each_seed_independent_of_companions_and_global_rng():
    from data_generation_collection.sft_data.clean_trace_collection.candidate_batch.clean_trace_batch import IndependentSeedSampler
    scores = torch.tensor([[0., 1., 2., 0., -1., 1., 0., 2.]])
    single = IndependentSeedSampler([42], 'cpu', [7], 2)
    batched = IndependentSeedSampler([123, 42, 456], 'cpu', [7], 2)
    ids = torch.tensor([[1, 2]])
    for _ in range(20):
        torch.manual_seed(987)
        expected = single(ids, scores).argmax(-1)
        torch.manual_seed(654)
        actual = batched(ids.repeat(3, 1), scores.repeat(3, 1)).argmax(-1)
        assert actual[1] == expected[0]


def test_batched_sampling_uses_frozen_native_temperature_and_top_p():
    from transformers import TemperatureLogitsWarper, TopPLogitsWarper
    from data_generation_collection.sft_data.clean_trace_collection.candidate_batch.clean_trace_batch import IndependentSeedSampler
    ids = torch.tensor([[1, 2]])
    scores = torch.tensor([[0., 1., 2., 3., 0., 1., 0., -1.]])
    expected_rng = torch.Generator().manual_seed(42)
    sampler = IndependentSeedSampler([42], 'cpu', [7], 2)
    for _ in range(10):
        probabilities = TopPLogitsWarper(.95)(ids, TemperatureLogitsWarper(.6)(ids, scores)).softmax(-1)
        expected = torch.multinomial(probabilities, 1, generator=expected_rng)
        assert sampler(ids, scores).argmax(-1).item() == expected.item()


def test_batched_generation_keeps_whole_prompts_order_and_per_row_eos():
    from types import SimpleNamespace
    from data_generation_collection.sft_data.clean_trace_collection.candidate_batch.clean_trace_batch import NativeCleanGenerator
    from shared_procedures.identity import derive_seed
    class Encoded(dict):
        def to(self, device): return self
        @property
        def input_ids(self): return self['input_ids']
    class BatchTokenizer:
        pad_token_id = 0
        eos_token_id = 9
        def apply_chat_template(self, messages, **kwargs): return messages[0]['content']+'<think>'
        def __call__(self, prompts, **kwargs):
            assert kwargs['padding'] and kwargs['add_special_tokens'] is False
            return Encoded(input_ids=torch.tensor([[0, 1, 2], [1, 2, 3]]), attention_mask=torch.tensor([[0, 1, 1], [1, 1, 1]]))
        def decode(self, ids, **kwargs): return 'one two three</think> answer'
    class BatchModel:
        device = 'cpu'
        generation_config = SimpleNamespace(eos_token_id=9)
        def generate(self, **kwargs):
            assert kwargs['input_ids'].shape[0] == 2
            assert kwargs['max_new_tokens'] == 3072 and kwargs['temperature'] == .6 and kwargs['top_p'] == .95
            assert kwargs['attention_mask'][0, 0] == 0
            return torch.tensor([[0, 1, 2, 5, 9, 0], [1, 2, 3, 5, 6, 9]])
    models={size:{'id':name,'revision':'a'*40,'tokenizer_revision':'b'*40} for size,name in
        [('7b','deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'),('14b','deepseek-ai/DeepSeek-R1-Distill-Qwen-14B')]}
    generator=NativeCleanGenerator(models,batch_size=32)
    generator.model, generator.tokenizer=BatchModel(),BatchTokenizer()
    generator._load=lambda model:None
    candidates=[{**candidate(),'example_id':str(index)} for index in range(2)]
    responses=list(generator.generate_many('7b',candidates))
    assert [r['generation_token_ids'] for r in responses]==[[5,9],[5,6,9]]
    assert [r['seed'] for r in responses]==[derive_seed(42,'clean_generation',c['example_id'],'7b') for c in candidates]
    assert [r['generation_batch_end'] for r in responses]==[False,True]
    assert all(r['finish_reason']=='stop' for r in responses)


def test_large_batches_split_at_256_without_changing_response_limits(monkeypatch):
    from types import SimpleNamespace
    from data_generation_collection.sft_data.clean_trace_collection.candidate_batch.clean_trace_batch import NativeCleanGenerator
    class Encoded(dict):
        def to(self,device):return self
        @property
        def input_ids(self):return self['input_ids']
    class BatchTokenizer:
        pad_token_id=0
        eos_token_id=9
        def apply_chat_template(self,messages,**kwargs):return messages[0]['content']+'<think>'
        def __call__(self,prompts,**kwargs):
            return Encoded(input_ids=torch.tensor([[1,2]]*len(prompts)),attention_mask=torch.ones(len(prompts),2))
        def decode(self,ids,**kwargs):return 'one two three</think> answer'
    sizes=[]
    class BatchModel:
        device='cpu'
        generation_config=SimpleNamespace(eos_token_id=9)
        def generate(self,**kwargs):
            count=kwargs['input_ids'].shape[0];sizes.append(count)
            assert kwargs['max_new_tokens']==3072 and kwargs['temperature']==.6 and kwargs['top_p']==.95
            return torch.cat([kwargs['input_ids'],torch.tensor([[5,9]]*count)],dim=1)
    models={size:{'id':name,'revision':'a'*40,'tokenizer_revision':'b'*40} for size,name in
        [('7b','deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'),('14b','deepseek-ai/DeepSeek-R1-Distill-Qwen-14B')]}
    monkeypatch.setenv('CLEAN_GENERATION_BATCH_SIZE_7B','256')
    monkeypatch.setenv('CLEAN_GENERATION_BATCH_SIZE_14B','64')
    generator=NativeCleanGenerator(models,batch_size=32)
    assert generator.batch_size_for('7b')==256 and generator.batch_size_for('14b')==64
    generator.model,generator.tokenizer=BatchModel(),BatchTokenizer();generator._load=lambda model:None
    candidates=[{**candidate(),'example_id':str(i)} for i in range(257)]
    responses=list(generator.generate_many('7b',candidates))
    assert sizes==[256,1] and len(responses)==257
    assert [i for i,r in enumerate(responses) if r['generation_batch_end']]==[255,256]
    assert [r['runtime']['generation_batch_size'] for r in responses]==[256]*256+[1]
    monkeypatch.setenv('CLEAN_GENERATION_BATCH_SIZE_7B','257')
    with pytest.raises(ValueError,match='between 1 and 256'):NativeCleanGenerator(models)


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
