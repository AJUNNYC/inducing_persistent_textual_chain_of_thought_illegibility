import pytest
from shared_procedures.identity import content_hash
from data_generation_collection.sft_data.sft_data_preparation import validate_accepted_pool,_publish_records,load_sft_records,build_sft_datasets,validate_sft_bundle


def test_invalid_frozen_pool_not_repaired_with_replacement_ids(tmp_path):
    with pytest.raises(ValueError):validate_accepted_pool({'status':'verified','pairs':[]})
    with pytest.raises(ValueError):build_sft_datasets({'status':'verified','pairs':[]},{},tmp_path,config_hash='a'*64)
    assert not list(tmp_path.rglob('*.jsonl'))


def test_atomic_jsonl_actual_record_readback_and_existing_identity(tmp_path):
    records=[{'example_id':'id','prompt':'exact\nbytes','completion':'<think>one two three</think> answer'}]
    path=tmp_path/'train.jsonl'
    digest=_publish_records(path,records)
    assert digest==content_hash(path.read_bytes())
    assert load_sft_records(path)==records
    assert _publish_records(path,records)==digest
    with pytest.raises(ValueError):_publish_records(path,[{**records[0],'completion':'changed'}])


def test_full_forty_dataset_handoff_common_10000_100_and_model_specific_sources(tmp_path):
    from data_generation_collection.sft_data.clean_trace_collection.candidate_batch.clean_trace_batch import validate_clean_trace
    from data_generation_collection.sft_data.rot13.rot13_data_construction import transform_rot13
    class FixtureTokenizer:
        def encode(self,text,**kwargs):return list(range(len(text.split())))
        def apply_chat_template(self,messages,**kwargs):
            return ''.join('<'+m['role']+'>'+m['content']+'<end>' for m in messages)
    class FixtureNoise:
        def transform(self,reasoning,example_id,level,model_id):
            return {'reasoning':reasoning,'content_hash':content_hash(reasoning),'source_hash':content_hash(reasoning),
                    'status':'verified','protected_content_valid':True,'changed_word_fraction':0}
    tokenizers={m:FixtureTokenizer() for m in ('7b','14b')}
    noise=FixtureNoise()
    common=[]
    template={}
    for model,reasoning in [('7b','one two three'),('14b','four five six')]:
        c={'example_id':'template','prompt':'question','source_content_hash':content_hash('question')}
        raw={'raw_generation':'<think>'+reasoning+'</think> final','finish_reason':'stop','generated_token_count':10,
             'model_revision':'a'*40,'tokenizer_revision':'b'*40}
        template[model]=validate_clean_trace(c,model,raw,tokenizers[model],kazitext=noise,config_hash='c'*64)
    for index in range(10100):
        example_id=str(index)
        # All3-word region sizes1 mean anchors/masks identical across IDs.
        traces={model:{**trace,'example_id':example_id} for model,trace in template.items()}
        common.append({'example_id':example_id,'prompt':'question','source_content_hash':content_hash('question'),'status':'accepted','traces':traces})
    pool={'status':'verified','pairs':common,'train_ids':[str(i) for i in range(10000)],
          'validation_ids':[str(i) for i in range(10000,10100)],'config_hash':'c'*64,'content_hash':content_hash(common)}
    bundle=build_sft_datasets(pool,tokenizers,tmp_path,config_hash='c'*64,kazitext=noise)
    assert bundle['count']==40
    assert {(d['family'],d['model_id'],d['level']) for d in bundle['datasets']}=={(f,m,k) for f in ('rot13','kazitext') for m in ('7b','14b') for k in range(1,11)}
    hashes={m:template[m]['clean_trace_hash'] for m in template}
    for dataset in bundle['datasets']:
        train,validation=load_sft_records(dataset['train_path']),load_sft_records(dataset['validation_path'])
        assert len(train)==10000 and len(validation)==100
        assert [r['example_id'] for r in train]==pool['train_ids']
        assert [r['example_id'] for r in validation]==pool['validation_ids']
        assert {r['clean_trace_hash'] for r in train+validation}=={hashes[dataset['model_id']]}
        assert all(r['final_answer']==' final' and r['prompt']=='question' and r['transformed_token_count']<=4096 for r in train+validation)
        assert dataset['content_hashes']['train']==content_hash(__import__('pathlib').Path(dataset['train_path']).read_bytes())
    assert hashes['7b']!=hashes['14b']
    verified=validate_sft_bundle(bundle,pool,tokenizers=tokenizers,kazitext=noise,recompute=True)
    assert verified['dataset_count']==40 and verified['record_count']==404000
    assert verified['recomputed_transforms'] is True
    altered=dict(bundle['datasets'][0]['content_hashes']);altered['train']='0'*64
    tampered={**bundle,'datasets':[{**bundle['datasets'][0],'content_hashes':altered},*bundle['datasets'][1:]]}
    tampered['content_hash']=content_hash(tampered['datasets'])
    with pytest.raises(ValueError,match='file checksum'):
        validate_sft_bundle(tampered,pool)
