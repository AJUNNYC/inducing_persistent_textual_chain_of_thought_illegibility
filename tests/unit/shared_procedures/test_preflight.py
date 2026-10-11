from pathlib import Path
import pytest
from shared_procedures.preflight import *
from shared_procedures.configuration import load_configuration
CONFIG=Path(__file__).resolve().parents[3]/'config.yaml'

def test_absent_live_requirements_block_launch(tmp_path):
    report=run_preflights(load_configuration(CONFIG),output_dir=tmp_path)
    assert report['ready'] is False and (tmp_path/'report.json').exists()
    with pytest.raises(PreflightBlocked):require_preflights(report,['model_runtime','judge_services'])

def test_mock_pass_does_not_supply_live_evidence():
    report=run_preflights(load_configuration(CONFIG),checks={'judge_services':CheckResult('judge_services','passed','mock',{'simulation':True})})
    assert next(x for x in report['checks'] if x['name']=='judge_services')['status']=='blocked'

def test_configuration_identity_and_missing_named_gate():
    report=run_preflights(load_configuration(CONFIG))
    with pytest.raises(PreflightBlocked):require_preflights(report,['configuration'],expected_configuration_hash='other')
    with pytest.raises(PreflightBlocked):require_preflights(report,['missing'])

def test_execute_live_without_cuda_or_authorization_is_honestly_blocked(tmp_path,monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.delenv('ILLEGIBILITY_WORKER_RECOVERY_PATH',raising=False)
    monkeypatch.setenv('ARTIFACT_ROOT',str(tmp_path/'artifacts'))
    config=load_configuration(CONFIG);config['runtime']['artifact_root']=str(tmp_path/'artifacts');config['runtime']['data_root']=str(tmp_path/'data')
    report=run_preflights(config,execute_live=True)
    rows={r['name']:r for r in report['checks']}
    assert rows['model_runtime']['status']=='blocked'
    assert rows['loss_mask_gradients']['status']=='blocked'
    assert rows['full_checkpoint_resume']['status']=='blocked'
    assert rows['judge_services']['status']=='blocked'
    assert rows['durable_storage']['status']=='blocked' and rows['durable_storage']['evidence']['local_flush_verified'] is True
    assert report['ready'] is False

def test_actual_dictionary_mismatch_fails_clean_gate_before_tokenizer_recompute(tmp_path,monkeypatch):
    import sys,types
    import data_generation_collection.sft_data.sft_data_preparation as data
    import data_generation_collection.sft_data.kazitext.kazitext_data_construction as kazitext
    from shared_procedures.preflight import _run_clean_checks
    from shared_procedures.storage import atomic_write_json
    config=load_configuration(CONFIG)
    config['runtime'].update(data_root=str(tmp_path/'data'),cache_root=str(tmp_path/'cache'))
    for path in (tmp_path/'data'/'accepted'/'accepted_pool.json',tmp_path/'data'/'transformed'/'sft_manifest.json'):
        atomic_write_json(path,{})
    (tmp_path/'cache'/'kazitext').mkdir(parents=True)
    monkeypatch.setattr(data,'validate_accepted_pool',lambda *_:{'pairs':[{'traces':{'7b':{'simulation':False},'14b':{'simulation':False}}}]})
    closed=[]
    class ActualDictionaryMismatch:
        provenance={'aspell_dictionaries':[{'name':'changed.rws','sha256':'a'*64}],'aspell_settings':{'lang':'en'}}
        def __init__(self,*args,**kwargs):pass
        def close(self):closed.append(True)
    monkeypatch.setattr(kazitext,'OfficialKaziText',ActualDictionaryMismatch)
    monkeypatch.setitem(sys.modules,'transformers',types.SimpleNamespace(AutoTokenizer=object()))
    result=_run_clean_checks(config)
    assert result.status=='failed' and 'dictionary' in result.detail
    assert closed==[True]

def test_generation_readiness_does_not_claim_training_or_run_training_checks(tmp_path):
    from shared_procedures.preflight import preflight_requirements
    callbacks=[]
    checks={name:CheckResult(name,'passed','explicit fixture evidence',{'live':True,'control_preflight':True}) for name in preflight_requirements('generation')}
    def outside_scope(_):callbacks.append('called');raise AssertionError('generation invoked a training gate')
    for name in set(NAMES)-set(preflight_requirements('generation')):checks[name]=outside_scope
    report=run_preflights(load_configuration(CONFIG),checks=checks,output_dir=tmp_path,scope='generation')
    assert report['ready'] is True and report['scope']=='generation'
    assert report['training_ready'] is False and report['study_ready'] is False
    assert callbacks==[]
    require_preflights(report,preflight_requirements('generation'))
    with pytest.raises(PreflightBlocked):require_preflights(report,preflight_requirements('training'))
    assert all(r['status']=='skipped' and r['required'] is False for r in report['checks'] if r['name'] not in report['required_names'])

def test_generation_mock_runtime_never_satisfies_launch():
    checks={name:CheckResult(name,'passed','fixture',{'live':True,'simulation':True}) for name in preflight_requirements('generation')}
    assert run_preflights(load_configuration(CONFIG),checks=checks,scope='generation')['ready'] is False
    with pytest.raises(ValueError):run_preflights(load_configuration(CONFIG),scope='unknown')


@pytest.fixture
def native_clean_control_case(tmp_path,monkeypatch):
    """Exercise the actual native generator with explicit CPU tensor/SDK doubles."""
    import builtins,sys,types,torch
    import operations.assets as assets
    from shared_procedures import storage
    config=load_configuration(CONFIG)
    config['runtime'].update(artifact_root=str(tmp_path/'artifacts'),cache_root=str(tmp_path/'cache'),
                             implementation_revision='a'*40,dependency_lock_hash='b'*64)
    config['interventions']['kazitext']['udpipe_sha256']='c'*64
    config['spending'].update(authorized=True,authorized_usd=1,authorization_evidence='isolated fixture authorization')
    state={'failure':None,'loads':[],'templates':[],'generation_calls':[],'native_loads':[]}
    class Tokenizer:
        pad_token_id=0;eos_token_id=9
        def apply_chat_template(self,messages,*,tokenize,add_generation_prompt,**kwargs):
            assert add_generation_prompt and len(messages)==1 and messages[0]['role']=='user'
            state['templates'].append(tokenize)
            return torch.tensor([[10,11,12]]) if tokenize else '<user>'+messages[0]['content']+'</user><think>'
        def decode(self,ids,skip_special_tokens):
            assert skip_special_tokens
            return 'Missing boundaries' if state['failure']=='malformed' else 'Three pages plus one page make four pages.</think>\n4'
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight=torch.nn.Parameter(torch.zeros(1,dtype=torch.float32 if state['failure']=='dtype' else torch.bfloat16))
            self.config=types.SimpleNamespace(quantization_config=None)
            self.generation_config=types.SimpleNamespace(eos_token_id=[9,99])
            self.is_loaded_in_4bit=state['failure']=='quantized'
            self.peft_config={'fixture_adapter':True} if state['failure']=='adapter' else None
            self.hf_device_map={'':'explicit_cpu_fixture'}
        @property
        def device(self):return self.weight.device
        @property
        def dtype(self):return self.weight.dtype
        def generate(self,encoded,**kwargs):
            assert kwargs=={'do_sample':True,'temperature':.6,'top_p':.95,'max_new_tokens':3072,'top_k':0,
                            'repetition_penalty':1.,'pad_token_id':9}
            state['generation_calls'].append(kwargs)
            return torch.tensor([[10,11,12,7,8,8 if state['failure']=='eos' else 9]])
    def load_model(name,**kwargs):
        assert kwargs['torch_dtype']==torch.bfloat16 and kwargs['device_map']=='auto'
        assert kwargs['trust_remote_code'] is False and 'quantization_config' not in kwargs
        state['loads'].append((name,kwargs));return Model()
    def cached(name,revision,cache_root,**kwargs):
        assert cache_root==config['runtime']['cache_root']
        state['native_loads'].append((name,revision,kwargs));return {'local_files_only':True}
    monkeypatch.setattr(assets,'model_load_kwargs',cached)
    monkeypatch.setitem(sys.modules,'transformers',types.SimpleNamespace(
        AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=load_model),
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *args,**kwargs:Tokenizer()),set_seed=lambda seed:None))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    monkeypatch.setattr(storage,'notify_artifact',lambda *args:None)
    native_import=builtins.__import__
    def no_training(name,*args,**kwargs):
        if name.startswith(('training.sft.individual_fit.sft_fit','peft')):
            raise AssertionError('Generation imported the training/LoRA runtime')
        return native_import(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',no_training)
    return config,state


def test_generation_controls_use_actual_native_clean_bf16_path_without_training(native_clean_control_case):
    from shared_procedures.preflight import _run_model_checks
    config,state=native_clean_control_case
    result=_run_model_checks(config,include_training=False)
    check=result['model_runtime']
    assert check.status=='passed' and check.evidence['runtime_kind']=='native_clean_base_bfloat16'
    assert check.evidence['training_runtime_loaded'] is False
    assert len(state['loads'])==len(state['generation_calls'])==2 and state['templates']==[False,True,False,True]
    assert [name for name,_ in state['loads']]==[config['models'][size]['id'] for size in ('7b','14b')]
    assert len(state['native_loads'])==4  # Exact model and tokenizer provisioning for both pinned bases.
    for row in check.evidence['models']:
        assert row['runtime']['base_dtype']=='torch.bfloat16' and row['runtime']['floating_parameter_dtypes']==['torch.bfloat16']
        assert row['runtime']['quantization'] is None and row['runtime']['adapter_loaded'] is False
        assert row['runtime']['optimizer_loaded'] is False and row['parser_status']=='valid' and row['eos_complete'] is True
        raw=read_json(row['raw_path'])
        assert raw['control_preflight'] is True and raw['scientific_result'] is False and raw['execution_mode']=='control_preflight'
    assert all(result[name].status=='skipped' and result[name].evidence['live'] is False
               for name in ('loss_mask_gradients','full_checkpoint_resume'))


@pytest.mark.parametrize('failure',['dtype','quantized','adapter','eos','malformed'])
def test_generation_control_rejects_wrong_runtime_or_incomplete_native_response(native_clean_control_case,failure):
    from shared_procedures.preflight import _run_model_checks
    config,state=native_clean_control_case
    state['failure']=failure
    with pytest.raises(PreflightBlocked,match='BF16 base|EOS completion|reasoning/final response'):
        _run_model_checks(config,include_training=False)


@pytest.fixture
def drive_primary_case(tmp_path,monkeypatch):
    """Real local JSON/registry integrity with explicit offline Drive doubles."""
    from copy import deepcopy
    import operations.drive as drive
    from operations.artifacts import ArtifactService
    from shared_procedures.storage import publish_artifact
    config=load_configuration(CONFIG)
    config['runtime'].update(artifact_root=str(tmp_path/'artifacts'),implementation_revision='a'*40)
    identity={'configuration_hash':configuration_hash(config),'purpose':'durability_probe_v1'}
    original=tmp_path/'original_probe.json'
    payload=publish_artifact(original,{'payload':'durable_probe_v1'},identity=identity)
    provenance={'experiment_id':identity['configuration_hash'],'branch':'shared','model':'all','run_id':'fixture',
        'stage':'runtime','producer':'explicit_offline_test','source_code_commit':'a'*40,
        'configuration_hash':identity['configuration_hash'],'dataset_revision':content_hash(config['datasets']),
        'model_revision':content_hash(config['models']),'seeds':[42]}
    record=ArtifactService(tmp_path,tmp_path/'registry.json').register(original,provenance)
    record.update(sync_status='VERIFIED',drive_file_id='fixture_file',drive_folder_id='fixture_folder')
    independent={'status':'INDEPENDENT_DRIVE_RETRIEVAL_VERIFIED','source_commit':'a'*40,
        'configuration_hash':identity['configuration_hash'],'probe_identity':identity,
        'content_hash':payload['content_hash'],'observed_at':'2026-10-11T00:00:00+00:00',
        'pod_filesystem_read_for_retrieval':False,
        **{key:record[key] for key in ('artifact_id','drive_file_id','sha256','md5','size')}}
    config['runtime']['durability_evidence']={'mode':'google_drive_primary','live':True,
        'registered_probe':record,'independent_restore':independent}
    calls=[]
    class OfflineDrive:
        def __init__(self,*args,**kwargs):calls.append(('construct',args,kwargs))
        def create_project_folder(self):calls.append(('project',))
        def restore_registered_artifact(self,selected,destination,*,expected_provenance):
            assert selected==record and not destination.exists()
            assert not destination.is_relative_to(tmp_path/'artifacts')
            assert expected_provenance['source_code_commit']=='a'*40
            assert original.is_file()  # Local presence must not supply a restore shortcut.
            calls.append(('restore',str(destination)))
            destination.write_bytes(original.read_bytes())
            return destination
    monkeypatch.setattr(drive,'DriveStore',OfflineDrive)
    monkeypatch.setattr(drive,'drive_service',lambda token: 'explicit_offline_service')
    monkeypatch.setenv('GOOGLE_OAUTH_TOKEN_FILE',str(tmp_path/'private_fixture.json'))
    monkeypatch.setenv('GOOGLE_DRIVE_FOLDER_ID','fixture_project')
    return config,record,independent,original,calls


def test_drive_primary_requires_fresh_remote_restore_each_launch_without_volume_claims(drive_primary_case):
    from shared_procedures.preflight import _run_durable_checks
    config,record,independent,original,calls=drive_primary_case
    first=_run_durable_checks(config);second=_run_durable_checks(config)
    assert first.status==second.status=='passed'
    assert first.evidence['mode']=='google_drive_primary' and first.evidence['live'] is True
    assert first.evidence['sha256']==record['sha256'] and first.evidence['probe_identity']==independent['probe_identity']
    assert first.evidence['runpod_storage_required'] is False
    assert 'stop_survival_verified' not in first.evidence
    restored=[call[1] for call in calls if call[0]=='restore']
    assert len(restored)==2 and restored[0]!=restored[1]
    assert all(not Path(path).exists() for path in restored)
    assert not (Path(config['runtime']['artifact_root'])/'preflight/durable_storage/probe.json').exists()


@pytest.mark.parametrize('mutation',['missing_independent','simulated','source','identity','sha256','content_hash','local_read'])
def test_drive_primary_incomplete_or_incompatible_evidence_blocks_before_network(drive_primary_case,mutation):
    from shared_procedures.preflight import _run_durable_checks
    config,record,independent,original,calls=drive_primary_case
    evidence=config['runtime']['durability_evidence']
    if mutation=='missing_independent':evidence.pop('independent_restore')
    elif mutation=='simulated':independent['simulation']=True
    elif mutation=='source':record['source_code_commit']='b'*40
    elif mutation=='identity':independent['probe_identity']['purpose']='different'
    elif mutation=='sha256':independent['sha256']='c'*64
    elif mutation=='content_hash':independent['content_hash']='d'*64
    else:independent['pod_filesystem_read_for_retrieval']=True
    assert _run_durable_checks(config).status in {'failed','blocked'}
    assert calls==[]


def test_drive_primary_unavailable_or_corrupt_drive_blocks_and_sanitizes_exception(drive_primary_case,monkeypatch):
    from shared_procedures.preflight import _run_durable_checks
    import operations.drive as drive
    config,record,independent,original,calls=drive_primary_case
    def fail(*args,**kwargs):raise ConnectionError('PUBLIC_FIXTURE_EXCEPTION_DETAIL_MUST_NOT_LEAK')
    monkeypatch.setattr(drive.DriveStore,'restore_registered_artifact',fail)
    result=_run_durable_checks(config)
    assert result.status=='blocked' and result.evidence['live'] is False
    assert result.evidence['error_type']=='ConnectionError'
    assert 'PUBLIC_FIXTURE_EXCEPTION_DETAIL_MUST_NOT_LEAK' not in str(result.to_dict())


def test_drive_primary_full_file_with_wrong_json_dependency_is_rejected(drive_primary_case):
    from shared_procedures.preflight import _run_durable_checks
    from shared_procedures.storage import atomic_write_json,read_json
    config,record,independent,original,calls=drive_primary_case
    document=read_json(original);document['identity']['configuration_hash']='b'*64
    atomic_write_json(original,document)
    result=_run_durable_checks(config)
    assert result.status=='blocked' and result.evidence['live'] is False


def test_drive_primary_missing_oauth_credential_blocks_before_restore(drive_primary_case,monkeypatch):
    from shared_procedures.preflight import _run_durable_checks
    config,record,independent,original,calls=drive_primary_case
    monkeypatch.delenv('GOOGLE_OAUTH_TOKEN_FILE')
    result=_run_durable_checks(config)
    assert result.status=='blocked' and result.evidence['live'] is False and calls==[]


def test_unknown_durability_mode_is_not_legacy_or_primary_readiness(tmp_path):
    from shared_procedures.preflight import _run_durable_checks
    config=load_configuration(CONFIG)
    config['runtime']['durability_evidence']={'mode':'unknown','live':True}
    assert _run_durable_checks(config).status=='failed'
