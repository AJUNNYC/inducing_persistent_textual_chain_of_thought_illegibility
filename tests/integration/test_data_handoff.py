"""Actual source-record builders feed SFT consumers; external outputs are fixtures.

The owner unit test independently writes/reopens all404000production-count
records. This integration checks the same full worklist in memory, without
repeating those disk writes or claiming fixture outputs as empirical results.
"""
import copy
from pathlib import Path
import pytest
import yaml
from shared_procedures.configuration import configuration_hash
from shared_procedures.identity import content_hash
from shared_procedures.records import validate_record
from data_generation_collection.sft_data.clean_trace_collection.candidate_batch.clean_trace_batch import validate_clean_trace
from data_generation_collection.sft_data.rot13.rot13_data_construction import transform_rot13
from data_generation_collection.sft_data.sft_data_preparation import build_sft_record
from training.sft.sft_training import build_worklist
from training.sft.individual_fit.sft_fit import SFTInputError


class FixtureTokenizer:
    def encode(self,text,**kwargs):return list(range(len(text.split())))
    def apply_chat_template(self,messages,**kwargs):
        return ''.join('<'+m['role']+'>'+m['content']+'<end>' for m in messages)


class FixtureOfficialNoise:
    def transform(self,reasoning,example_id,level,model_id):
        return {'reasoning':reasoning,'content_hash':content_hash(reasoning),'source_hash':content_hash(reasoning),
                'status':'verified','protected_content_valid':True,'changed_word_fraction':0}


def test_source_generation_transform_record_schema_and_forty_fit_worklist():
    config=yaml.safe_load((Path(__file__).parents[2]/'config.yaml').read_text())
    # Exercise the full historical matrix independently of the deployment selection.
    config['sft']['selection']={'models':['7b','14b'],
                                'families':['rot13','kazitext'],'levels':list(range(1,11))}
    config_hash=configuration_hash(config)
    tokenizer=FixtureTokenizer();noise=FixtureOfficialNoise()
    candidate={'example_id':'template','prompt':'exact prompt','source_content_hash':content_hash('exact prompt')}
    sources={}
    for model,text in [('7b','one two three'),('14b','four five six')]:
        raw={'raw_generation':'<think>'+text+'</think> final','finish_reason':'stop','generated_token_count':10,
             'model_revision':config['models'][model]['revision'],'tokenizer_revision':config['models'][model]['tokenizer_revision'],'seed':42}
        sources[model]=validate_clean_trace(candidate,model,raw,tokenizer,kazitext=noise,config_hash=config_hash)
        validate_record('clean_generation',sources[model])
    datasets=[]
    for family in ('rot13','kazitext'):
        for model in ('7b','14b'):
            for level in range(1,11):
                trace=sources[model]
                transformed=transform_rot13(trace['reasoning'],'template',level) if family=='rot13' else noise.transform(trace['reasoning'],'template',level,model)
                pair={**candidate,'status':'accepted','traces':sources}
                records={}
                for split,start,count in [('train',0,10000),('validation',10000,100)]:
                    sample=build_sft_record(pair,trace,transformed,tokenizer,model_id=model,family=family,level=level,split=split,
                                           source_manifest_hash='a'*64,config_hash=config_hash)
                    validate_record('sft_training_record',sample)
                    records[split]=[{**sample,'example_id':str(i)} for i in range(start,start+count)]
                datasets.append({'status':'verified','family':family,'model_id':model,'level':level,'config_hash':config_hash,
                                 'train_records':records['train'],'validation_records':records['validation']})
    worklist=build_worklist(config,datasets)
    assert len(worklist)==40
    assert all(w['initialization']=='untouched_base_or_exact_full_checkpoint' for w in worklist)
    assert len({w['run_id'] for w in worklist})==40
    assert {w['spec']['model_revision'] for w in worklist}=={config['models'][m]['revision'] for m in ('7b','14b')}
    # Mutate one source final while retaining a structurally coherent target.
    row=datasets[1]['train_records'][0]
    datasets[1]['train_records'][0]={**row,'final_answer':' altered','completion':'<think>'+row['reasoning']+'</think> altered'}
    with pytest.raises(SFTInputError,match='Clean traces'):
        build_worklist(config,datasets)
