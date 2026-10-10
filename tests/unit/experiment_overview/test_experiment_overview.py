"""Root orchestration fixtures never supply live scientific/paid evidence."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import os

import pytest
import yaml

import experiment_overview.experiment_overview as program
from operations.workflow import Stage
from shared_procedures.identity import content_hash
from shared_procedures.configuration import snapshot_hash
from shared_procedures.storage import atomic_write_json


@pytest.fixture(autouse=True)
def restore_hub_environment(monkeypatch):
    for name in ('HF_HOME','HF_HUB_CACHE','HF_DATASETS_CACHE','DATA_ROOT','ARTIFACT_ROOT','MODEL_CACHE_ROOT'):
        monkeypatch.setenv(name,os.environ.get(name,''))


@pytest.fixture
def experiment(tmp_path,monkeypatch):
    config=yaml.safe_load((program.ROOT/'config.yaml').read_text())
    config['runtime'].update(data_root='data',artifact_root='artifacts',cache_root='cache',
        dependency_lock_hash='1'*64,implementation_revision='a'*40)
    for source in [*config['models'].values(),*config['datasets'].values()]:
        source['revision']='b'*40
        if 'tokenizer_revision' in source:source['tokenizer_revision']='c'*40
    config['interventions']['kazitext']['udpipe_sha256']='d'*64
    monkeypatch.setenv('SOURCE_CODE_COMMIT','a'*40)
    monkeypatch.setattr(program,'code_revision',lambda:'a'*40)
    for variable in ('DATA_ROOT','ARTIFACT_ROOT','MODEL_CACHE_ROOT'):
        monkeypatch.delenv(variable,raising=False)
    return program.Experiment(config,config_path=tmp_path/'config.yaml')


def test_paths_are_resolved_from_config_location_and_env_is_frozen(experiment,tmp_path,monkeypatch):
    assert experiment.data == tmp_path/'data'
    assert experiment.config['runtime']['artifact_root'] == str(tmp_path/'artifacts')
    assert os.environ['HF_HOME'] == str(tmp_path/'cache')
    assert os.environ['HF_HUB_CACHE'] == str(tmp_path/'cache'/'hub')
    monkeypatch.setenv('DATA_ROOT',str(tmp_path/'persistent_data'))
    changed=program.Experiment(experiment.config,config_path=tmp_path/'config.yaml')
    assert changed.data == tmp_path/'persistent_data' and changed.hash == experiment.hash
    assert snapshot_hash(changed.config) != snapshot_hash(experiment.config)


def test_discovery_cli_plan_and_hardware_use_actual_cpu_environment(tmp_path,monkeypatch,capsys):
    config=yaml.safe_load((program.ROOT/'config.yaml').read_text())
    config['runtime'].update(data_root=str(tmp_path/'data'),artifact_root=str(tmp_path/'artifacts'),cache_root=str(tmp_path/'cache'))
    path=tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(config))
    assert program.main(['--config',str(path),'plan','--target','rl_7b']) == 0
    plan=json.loads(capsys.readouterr().out)
    assert plan[-1]['stage_id']=='rl_7b'
    assert not any(row['stage_id']=='rl_14b' for row in plan)
    assert program.main(['--config',str(path),'hardware']) == 0
    actual=json.loads(capsys.readouterr().out)
    assert actual['cpu_count']>0 and actual['disk_free_bytes']>0


def test_transitive_paid_dependency_requires_spending_before_drive(experiment,monkeypatch):
    called=[]
    monkeypatch.setattr(experiment,'artifact_service',lambda:called.append('unexpected_drive_access'))
    monkeypatch.setattr(program.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout=''))
    with pytest.raises(RuntimeError,match='spending authorization'):
        experiment.execute(['sync'])
    assert called == []


def test_container_discovery_without_git_does_not_invent_a_commit(monkeypatch):
    monkeypatch.delenv('SOURCE_CODE_COMMIT',raising=False)
    monkeypatch.setattr(program.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=128,stdout=''))
    assert program.code_revision().startswith('unversioned_')
    monkeypatch.setenv('SOURCE_CODE_COMMIT','main')
    with pytest.raises(ValueError,match='immutable'):
        program.code_revision()


def test_prepare_panels_calls_real_child_contract_and_withholds_tests_from_prompt(experiment,monkeypatch):
    import data_generation_collection.evaluation_data.evaluation_data_preparation as preparation
    captured={}
    def child(pools,revisions,*,output_dir,verify_mbpp_reference):
        assert verify_mbpp_reference('def f(): return 1',['assert f()==1']) is True
        captured.update(pools=pools,revisions=revisions)
        result={'capability_panel':{'status':'verified'},'legibility_subset':{'status':'verified'}}
        atomic_write_json(Path(output_dir)/'evaluation_package.json',result)
        return result
    monkeypatch.setattr(preparation,'prepare_evaluation_data',child)
    monkeypatch.setattr(experiment,'source_pools',lambda:{'gsm8k':[]})
    monkeypatch.setattr(experiment,'sandbox',lambda:SimpleNamespace(execute=lambda code,tests:{'correct':True}))
    result=experiment.action(Stage('panels','prepare_panels',()))
    assert result['status']=='VERIFIED' and captured['pools']=={'gsm8k':[]}


def test_failed_actual_statistical_evidence_remains_failed(experiment,monkeypatch):
    import data_generation_collection.evaluation_data.protocol.analysis_plan.statistical_validation as statistics
    atomic_write_json(experiment.data/'evaluation'/'analysis_plan.json',{'fixture':True})
    def child(plan,*,output_path):
        assert plan['fixture'] is True
        return {'status':'failed','ready_for_confirmatory_launch':False,'report_hash':'fixture'}
    monkeypatch.setattr(statistics,'run_statistical_validation',child)
    result=experiment.action(Stage('statistical_validation','validate_statistics',()))
    assert result['status']=='FAILED'
    assert (experiment.artifacts/'preflight'/'statistical_validation.json').is_file()


def test_selected_rl_model_and_full_hf_revision_keys_are_sent(experiment,monkeypatch):
    import training.rl_grpo.rl_training as rl
    captured={}
    def child(manifest,output,**kwargs):
        captured.update(kwargs)
        assert kwargs['model_sizes']==('14b',)
        assert set(kwargs['model_revisions']) == {model['id'] for model in experiment.config['models'].values()}
        return {'status':'completed','runs':[],'model_sizes':['14b'],'scientific_snapshots':10}
    monkeypatch.setattr(rl,'run_rl_training',child)
    monkeypatch.setattr(experiment,'judge',lambda:'explicit_fixture_judge')
    atomic_write_json(experiment.data/'rl'/'manifest.json',{'fixture':'questions'})
    atomic_write_json(experiment.artifacts/'preflight'/'report.json',{'fixture':'not_live'})
    result=experiment.action(Stage('rl_14b','rl',(),model='14b',branch='rl',gpu_count=1))
    assert result['status']=='VERIFIED'
    assert (experiment.artifacts/'training'/'rl'/'14b'/'trajectory_report.json').is_file()
    assert not (experiment.artifacts/'training'/'rl'/'7b'/'trajectory_report.json').exists()


def test_sft_state_consumes_verified_adapter_and_exact_identity(experiment,monkeypatch):
    import training.sft.individual_fit.sft_fit as fit
    model=experiment.config['models']['7b']
    identity={'configuration_hash':experiment.hash,'model_revision':model['revision'],'tokenizer_revision':model['tokenizer_revision']}
    result={'identity':identity,'adapter_path':'fixture_adapter','checkpoint_hash':'1'*64,'adapter_manifest_hash':'2'*64}
    output=experiment.artifacts/'training'/'sft'/'rot13'/'7b'/'level_01'
    atomic_write_json(output/'fit_result.json',result)
    monkeypatch.setattr(fit,'verify_fit_result',lambda path,expected:result)
    state=experiment.state(Stage('evaluate_rot13_7b_level_01','evaluate',(),model='7b',branch='rot13',level=1))
    assert state['adapter_manifest_hash']=='2'*64 and state['checkpoint_hash']=='1'*64
    assert state['level']==1 and state['model']=='7b'


def test_cached_source_bytes_are_verified_before_reuse(experiment):
    revisions={key:value['revision'] for key,value in experiment.config['datasets'].items() if key!='lmsys'}
    pools={'gsm8k':[{'question':'original'}]}
    path=experiment.data/'evaluation'/'official_pools_cache.json'
    atomic_write_json(path,{'revisions':revisions,'pools':pools,'content_hash':content_hash(pools)})
    assert experiment.source_pools()==pools
    corrupt={'revisions':revisions,'pools':{'gsm8k':[{'question':'corrupted'}]},'content_hash':content_hash(pools)}
    atomic_write_json(path,corrupt)
    with pytest.raises(RuntimeError,match='content changed'):
        experiment.source_pools()


def test_source_wrapper_preserves_existing_official_export(experiment,monkeypatch):
    import data_generation_collection.evaluation_data.capability_panel.capability_panel_construction as panel
    exported={'gsm8k':[{'question':'official source fixture'}]}
    plain=experiment.data/'evaluation'/'official_pools.json'
    atomic_write_json(plain,exported)
    original=plain.read_bytes()
    calls=[]
    def verified_loader(revisions,*,cache_root):
        calls.append((revisions,cache_root))
        return exported
    monkeypatch.setattr(panel,'load_evaluation_sources',verified_loader)
    assert experiment.source_pools()==exported
    assert experiment.source_pools()==exported and len(calls)==1
    assert plain.read_bytes()==original
    wrapped=json.loads((plain.parent/'official_pools_cache.json').read_text())
    assert wrapped['content_hash']==content_hash(exported)
    assert wrapped['revisions']['gsm8k']==experiment.config['datasets']['gsm8k']['revision']


def test_kazitext_constructs_with_actual_auxiliary_provisioner_contract(experiment,monkeypatch):
    import data_generation_collection.sft_data.kazitext.kazitext_data_construction as kazi
    assets={'source_dir':'fixture-source','udpipe_path':'fixture-ewt','aspell_dictionary_files':['fixture-dictionary']}
    calls=[]
    monkeypatch.setattr(kazi,'provision_kazitext_assets',lambda destination,**kwargs:assets)
    expected=experiment.config['interventions']['kazitext']['aspell_dictionary_provenance']
    backend=SimpleNamespace(provenance={'aspell_dictionaries':expected['files'],'aspell_settings':expected['settings']},close=lambda:None)
    monkeypatch.setattr(kazi,'OfficialKaziText',lambda source,udpipe,**kwargs:calls.append((source,udpipe,kwargs)) or backend)
    assert experiment.kazi() is backend
    assert calls == [('fixture-source','fixture-ewt',{'expected_udpipe_sha256':'d'*64,'aspell_dictionary_files':['fixture-dictionary']})]


def test_native_dictionary_mismatch_blocks_before_generation(experiment,monkeypatch):
    import data_generation_collection.sft_data.kazitext.kazitext_data_construction as kazi
    assets={'source_dir':'fixture-source','udpipe_path':'fixture-ewt','aspell_dictionary_files':['fixture-dictionary']}
    monkeypatch.setattr(kazi,'provision_kazitext_assets',lambda *args,**kwargs:assets)
    closed=[]
    backend=SimpleNamespace(provenance={'aspell_dictionaries':[],'aspell_settings':{}},close=lambda:closed.append(True))
    monkeypatch.setattr(kazi,'OfficialKaziText',lambda *args,**kwargs:backend)
    with pytest.raises(RuntimeError,match='dictionary content/settings'):
        experiment.kazi()
    assert closed==[True]
