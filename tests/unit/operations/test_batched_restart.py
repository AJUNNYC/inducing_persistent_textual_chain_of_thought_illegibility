"""Batched restart preserves original verified responses and their provenance."""
import importlib.util
from copy import deepcopy
from pathlib import Path
import pytest

from operations.artifacts import ArtifactService
from operations.drive import file_hash
from shared_procedures.configuration import load_configuration, configuration_hash
from shared_procedures.identity import content_hash, derive_seed
from shared_procedures.storage import atomic_write_json, read_json
from data_generation_collection.sft_data.clean_trace_collection.candidate_batch.clean_trace_batch import GENERATION_SETTINGS

ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('prepare_batched_collection',ROOT/'scripts/prepare_batched_collection.py')
restart=importlib.util.module_from_spec(spec);spec.loader.exec_module(restart)


def configurations(tmp_path):
    old=load_configuration(ROOT/'config.yaml')
    old['runtime'].update(implementation_revision='a'*40,data_root=str(tmp_path/'old/data'),cache_root=str(tmp_path/'cache'))
    for model in old['models'].values():model.update(revision='c'*40,tokenizer_revision='d'*40)
    new=deepcopy(old)
    new['runtime'].update(implementation_revision='b'*40,data_root=str(tmp_path/'new/data'))
    return old,new


def test_rebinding_rejects_model_revision_changes(tmp_path):
    old,new=configurations(tmp_path)
    restart.unchanged_science(old,new)
    new['models']['7b']['revision']='e'*40
    with pytest.raises(RuntimeError,match='identical scientific settings'):restart.unchanged_science(old,new)


@pytest.mark.parametrize('finish',['stop','length'])
@pytest.mark.parametrize('previous_adoption',[False,True])
def test_original_whole_response_is_not_regenerated_or_relabelled_as_new_generation(tmp_path,monkeypatch,finish,previous_adoption):
    old,new=configurations(tmp_path)
    parent=tmp_path/'parent.json';atomic_write_json(parent,{'fixture_complete_audit':True})
    calls=[]
    def certificate(path,config,**kwargs):
        calls.append(kwargs['consumer_commit'])
        return {'consumer_commit':'a'*40,'consumer_configuration_hash':configuration_hash(old),'certificate_hash':'fixture'}
    monkeypatch.setattr(restart,'validate_consumer_certificate',certificate)
    monkeypatch.setattr(restart.subprocess,'check_output',lambda *args,**kwargs:'b'*40+'\n')
    source=tmp_path/'old/data/clean_traces/raw/production/7b/id.json'
    raw={'model_id':'7b','model_name':old['models']['7b']['id'],'model_revision':'c'*40,'tokenizer_revision':'d'*40,
         'example_id':'id','candidate_hash':'fixture','config_hash':configuration_hash(old),'simulation':False,
         'execution_mode':'production','seed':derive_seed(42,'clean_generation','id','7b'),
         'raw_generation':'original whole response','raw_hash':content_hash('original whole response'),
         'generation_settings':GENERATION_SETTINGS,'generation_token_ids':[1,2],'generated_token_count':2,'finish_reason':finish}
    if previous_adoption:
        raw.update(generation_original_producer_commit='f'*40,generation_original_drive_file_id='original_drive')
    atomic_write_json(source,raw);original_bytes=source.read_bytes()
    service=ArtifactService(tmp_path/'old',tmp_path/'old/registry.json')
    provenance={'experiment_id':'fixture','branch':'shared','model':'all','run_id':'clean','stage':'collect_clean',
        'producer':'experiment_overview','source_code_commit':'a'*40,'configuration_hash':configuration_hash(old),
        'dataset_revision':'c'*40,'model_revision':'d'*40,'seeds':[42]}
    record=service.register(source,provenance)
    record.update(sync_status='VERIFIED',drive_file_id='fixture_drive_file',drive_folder_id='fixture_folder')
    with service._connect() as db:service._put(db,record)
    report=restart.prepare(old,new,old_certificate=parent,input_root=tmp_path,certificate_output=tmp_path/'new_certificate.json',
        old_registry=service.registry,old_data_root=tmp_path/'old/data',new_data_root=tmp_path/'new/data',report_output=tmp_path/'report.json')
    adopted=read_json(tmp_path/'new/data/clean_traces/raw/production/7b/id.json')
    assert adopted['raw_generation']==raw['raw_generation'] and adopted['generation_token_ids']==raw['generation_token_ids']
    assert adopted['finish_reason']==finish and adopted['generation_original_producer_commit']=='a'*40
    assert adopted['generation_original_file_sha256']==file_hash(source)
    if previous_adoption:
        assert adopted['cache_adoption_history'][-1]=={'generation_original_producer_commit':'f'*40,
                                                      'generation_original_drive_file_id':'original_drive'}
    assert adopted['config_hash']==configuration_hash(new) and source.read_bytes()==original_bytes
    assert calls==['a'*40,'b'*40] and report['adopted_response_count']==1
