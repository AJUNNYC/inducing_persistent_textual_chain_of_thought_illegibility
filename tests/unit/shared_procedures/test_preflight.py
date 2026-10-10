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
