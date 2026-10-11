"""Hermetic overview lifecycle checks; fixtures never claim live scientific readiness."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest
import yaml

import experiment_overview.experiment_overview as program
from operations.artifacts import ArtifactService
from operations.drive import file_hash
from operations.workflow import Stage,StageRunner,experiment_graph,topological_order
from shared_procedures.configuration import snapshot_hash
from shared_procedures.cost_accounting import CostLedger
from shared_procedures.identity import content_hash
from shared_procedures.preflight import CheckResult,PreflightBlocked,preflight_requirements,run_preflights
from shared_procedures.recovery_procedure import RecoveryManager
from shared_procedures.storage import artifact_hook,atomic_write_json,read_json


class FixtureDrive:
    """Stores actual uploaded bytes in memory; never contacts Google/OAuth."""
    def __init__(self,fail_name=None):self.uploads=[];self.fail_name=fail_name
    def create_project_folder(self):return 'fixture_project'
    def ensure_path(self,parts):return 'fixture_folder'
    def upload_artifact(self,path,**metadata):
        path=Path(path)
        if path.name==self.fail_name:raise RuntimeError('fixture upload unavailable')
        self.uploads.append({'path':path,'bytes':path.read_bytes(),**metadata})
        return {'id':metadata['artifact_id']}
    def verify_registered_artifact(self,record):return {'id':record['drive_file_id']}


@pytest.fixture
def experiment(tmp_path,monkeypatch):
    # Restore only nonsecret root/cache/worker settings touched by orchestration.
    for name in ('HF_HOME','HF_HUB_CACHE','HF_DATASETS_CACHE','DATA_ROOT','ARTIFACT_ROOT','MODEL_CACHE_ROOT',
                 'ILLEGIBILITY_WORKER_ID','ILLEGIBILITY_WORKER_RECOVERY_PATH'):
        monkeypatch.delenv(name,raising=False)
    config=yaml.safe_load((program.ROOT/'config.yaml').read_text())
    config['runtime'].update(data_root='data',artifact_root='artifacts',cache_root='cache',
                             dependency_lock_hash='1'*64,implementation_revision='a'*40)
    config['spending'].update(authorized=True,authorized_usd=10,authorization_evidence='explicit unit fixture')
    monkeypatch.setattr(program,'code_revision',lambda:'a'*40)
    exp=program.Experiment(config,config_path=tmp_path/'config.yaml')
    exp._artifact_service=ArtifactService(tmp_path,exp.artifacts/'state'/'artifact_registry.json',FixtureDrive())
    return exp


def test_worker_boundary_backs_up_real_sqlite_ledgers_and_registry_last(experiment,monkeypatch):
    monkeypatch.setenv('ILLEGIBILITY_WORKER_ID','generation')
    recovery=RecoveryManager.for_worker(experiment.artifacts)
    recovery.reserve_retry('assets:source42:range1');recovery.reserve_retry('judge:source42:pass_a')
    cost=CostLedger(experiment.artifacts/'state'/'spend.json',authorization=experiment.config['spending'])
    cost.reserve('uncertain_request',1,category='api')
    cost.reserve('completed_request',1,category='api');cost.commit('completed_request',.25,usage={'tokens':4})
    experiment.publish_worker_state()
    uploads=experiment._artifact_service.drive.uploads
    assert [entry['path'].name for entry in uploads]==['retries.sqlite3','spend.sqlite3','artifact_registry.json']
    assert all(entry['path'].is_relative_to(experiment.artifacts/'operations'/'worker_state'/'generation') for entry in uploads)
    with sqlite3.connect(uploads[0]['path']) as database:
        assert database.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        assert database.execute('SELECT retries FROM worker').fetchone()[0]==2
        assert dict(database.execute('SELECT operation_id,retries FROM operations'))=={'assets:source42:range1':1,'judge:source42:pass_a':1}
    with sqlite3.connect(uploads[1]['path']) as database:
        assert database.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        assert database.execute('SELECT spent,reserved FROM totals').fetchone()==('0.25','1')
        assert database.execute("SELECT status FROM entries WHERE operation_id='uncertain_request'").fetchone()[0]=='reserved'
    manifest=json.loads(uploads[-1]['bytes'])
    for entry in uploads[:2]:
        record=manifest['artifacts'][entry['artifact_id']]
        assert record['sync_status']=='VERIFIED' and record['sha256']==content_hash(entry['bytes'])
        assert record['drive_file_id']==entry['artifact_id']
    assert recovery.counters()['worker_retries']==2 and cost.totals()['committed_usd']==1.25


def test_worker_snapshots_are_immutable_and_separate_by_worker(experiment,monkeypatch):
    monkeypatch.setenv('ILLEGIBILITY_WORKER_ID','rot13')
    RecoveryManager.for_worker(experiment.artifacts).reserve_retry('rot13:run42:update1')
    experiment.publish_worker_state()
    first=experiment._artifact_service.drive.uploads[0]
    original=first['bytes'];digest=file_hash(first['path'])
    RecoveryManager.for_worker(experiment.artifacts).reserve_retry('rot13:run42:update2')
    experiment.publish_worker_state()
    assert first['path'].read_bytes()==original and file_hash(first['path'])==digest
    rot13_backups=[row for row in experiment._artifact_service.drive.uploads if row['path'].name=='retries.sqlite3']
    assert len({row['path'] for row in rot13_backups})==2
    monkeypatch.setenv('ILLEGIBILITY_WORKER_ID','kazitext')
    RecoveryManager.for_worker(experiment.artifacts).reserve_retry('kazitext:run42:update1')
    experiment.publish_worker_state()
    last=next(row for row in reversed(experiment._artifact_service.drive.uploads) if row['path'].name=='retries.sqlite3')
    assert last['path'].is_relative_to(experiment.artifacts/'operations'/'worker_state'/'kazitext')
    with sqlite3.connect(last['path']) as database:assert database.execute('SELECT retries FROM worker').fetchone()[0]==1
    assert first['path'].read_bytes()==original


@pytest.mark.parametrize('worker',['../escape','worker/name','.hidden',''])
def test_invalid_worker_namespace_cannot_publish_state(experiment,monkeypatch,worker):
    monkeypatch.setenv('ILLEGIBILITY_WORKER_ID',worker)
    with pytest.raises(ValueError,match='safe worker identity'):experiment.publish_worker_state()
    assert experiment._artifact_service.drive.uploads==[]


def test_failed_ledger_replication_stops_before_publishing_registry(experiment):
    RecoveryManager.for_worker(experiment.artifacts).reserve_retry('storage:source42:retry')
    experiment._artifact_service.drive=FixtureDrive(fail_name='retries.sqlite3')
    with pytest.raises(RuntimeError,match='replication is incomplete'):experiment.publish_worker_state()
    assert experiment._artifact_service.drive.uploads==[]
    failed=experiment._artifact_service.list_artifacts(sync_status='FAILED')
    assert len(failed)==1 and Path(failed[0]['local_path']).name=='retries.sqlite3'
    assert list((experiment.artifacts/'operations'/'worker_state').rglob('retries.sqlite3'))
    assert not list((experiment.artifacts/'operations'/'worker_state').rglob('artifact_registry.json'))


def test_paused_stage_preserves_progress_and_stops_downstream_then_resumes(tmp_path):
    graph={'clean':Stage('clean','collect',()),'train':Stage('train','training',('clean',))}
    runner=StageRunner(graph,tmp_path/'workflow','a'*64,'b'*40)
    progress=tmp_path/'progress.json';trained=tmp_path/'training.json';calls=[]
    def pause(stage):
        calls.append('pause');atomic_write_json(progress,{'completed_ids':['one'],'status':'paused'})
        return {'status':'PAUSED','output_files':[str(progress)]}
    def train(stage):
        calls.append('train');atomic_write_json(trained,{'fixture':True})
        return {'status':'VERIFIED','output_files':[str(trained)]}
    first=runner.run(['train'],{'collect':pause,'training':train})
    assert [row['status'] for row in first]==['PAUSED'] and calls==['pause']
    assert 'completion_hash' not in runner.status('clean') and not trained.exists()
    def resume(stage):
        assert read_json(progress)['completed_ids']==['one']
        calls.append('resume');atomic_write_json(progress,{'completed_ids':['one','two'],'status':'complete'})
        return {'status':'VERIFIED','output_files':[str(progress)]}
    second=runner.run(['train'],{'collect':resume,'training':train})
    assert [row['status'] for row in second]==['COMPLETED','COMPLETED']
    assert calls==['pause','resume','train']
    assert second[-1]['dependency_hashes']['clean']==second[0]['completion_hash']


@pytest.mark.parametrize('mutation',['changed','deleted'])
def test_paused_progress_integrity_is_checked_before_reporting_reusable_state(tmp_path,mutation):
    runner=StageRunner({'clean':Stage('clean','collect',())},tmp_path/'workflow','a'*64,'b'*40)
    progress=tmp_path/'progress.json';atomic_write_json(progress,{'status':'paused','completed_ids':['one']})
    runner.run(['clean'],{'collect':lambda _:{'status':'PAUSED','output_files':[str(progress)]}})
    if mutation=='changed':atomic_write_json(progress,{'status':'paused','completed_ids':['altered']})
    else:progress.unlink()
    assert runner.status('clean')['status']=='INVALIDATED'


def test_generation_preflight_graph_and_action_keep_training_gates_out_of_scope(experiment,monkeypatch):
    order=topological_order(experiment_graph(),['clean'])
    assert order.index('candidates')<order.index('generation_preflight')<order.index('clean')
    assert not any(name in order for name in ('preflight','baseline_7b','baseline_14b','statistical_validation'))
    import shared_procedures.preflight as preflight
    captured={}
    def controls(config,**kwargs):
        captured.update(kwargs)
        checks={name:CheckResult(name,'passed','hermetic scope-contract fixture',{'live':True,'fixture':True,'scientific_results':False}) for name in preflight_requirements('generation')}
        return run_preflights(config,checks,output_dir=kwargs['output_dir'],scope=kwargs['scope'])
    monkeypatch.setattr(preflight,'run_preflights',controls)
    result=experiment.action(Stage('generation_preflight','generation_preflight',()))
    assert captured=={'output_dir':experiment.artifacts/'preflight'/'generation','execute_live':True,'scope':'generation'}
    report=read_json(result['output_files'][0])
    assert result['status']=='VERIFIED' and report['ready'] is True
    assert report['training_ready'] is False and report['study_ready'] is False
    assert (experiment.artifacts/'preflight'/'report.json').exists() is False


def test_failed_generation_gate_blocks_clean_launch(experiment,monkeypatch):
    import shared_procedures.preflight as preflight
    def unavailable(config,**kwargs):return run_preflights(config,scope='generation')
    monkeypatch.setattr(preflight,'run_preflights',unavailable)
    with pytest.raises(PreflightBlocked,match='mandatory launch gates'):experiment.action(Stage('generation_preflight','generation_preflight',()))
    assert not (experiment.data/'accepted'/'accepted_pool.json').exists()


@pytest.mark.parametrize('outcome',['paused','failed'])
def test_clean_boundary_closes_native_backend_and_never_freezes_a_partial_pool(experiment,monkeypatch,outcome):
    import data_generation_collection.sft_data.clean_trace_collection.clean_trace_collection as collection
    atomic_write_json(experiment.data/'candidates'/'manifest.json',{'fixture':True})
    closed=[];backend=SimpleNamespace(close=lambda:closed.append(True))
    monkeypatch.setattr(experiment,'kazi',lambda:backend)
    monkeypatch.setattr(experiment,'tokenizers',lambda:{'7b':'fixture','14b':'fixture'})
    progress=experiment.data/'clean_traces'/'collection_status.json'
    atomic_write_json(progress,{'status':'paused','completed_ids':['one']})
    def collect(*args,**kwargs):
        assert kwargs['kazitext'] is backend
        if outcome=='failed':raise ValueError('fixture transform failure')
        return {'status':'paused','accepted_count':0,'next_candidate_position':1}
    monkeypatch.setattr(collection,'collect_clean_traces',collect)
    if outcome=='failed':
        with pytest.raises(ValueError,match='transform failure'):experiment.action(Stage('clean','collect_clean',()))
    else:
        result=experiment.action(Stage('clean','collect_clean',()))
        assert result['status']=='PAUSED' and str(progress) in result['output_files']
        assert result['resume']['next_candidate_position']==1
    assert closed==[True]
    assert not (experiment.data/'accepted'/'accepted_pool.json').exists()


def test_frozen_native_dictionary_settings_block_even_when_file_hashes_match(experiment,monkeypatch):
    import data_generation_collection.sft_data.kazitext.kazitext_data_construction as native
    expected=deepcopy(experiment.config['interventions']['kazitext']['aspell_dictionary_provenance'])
    changed=deepcopy(expected['settings']);changed['ignore-case']=not changed['ignore-case']
    closed=[]
    backend=SimpleNamespace(provenance={'aspell_dictionaries':expected['files'],'aspell_settings':changed},close=lambda:closed.append(True))
    monkeypatch.setattr(native,'provision_kazitext_assets',lambda *a,**k:{'source_dir':'fixture-source','udpipe_path':'fixture-ewt','aspell_dictionary_files':['fixture-file']})
    monkeypatch.setattr(native,'OfficialKaziText',lambda *a,**k:backend)
    with pytest.raises(RuntimeError,match='dictionary content/settings'):experiment.kazi()
    assert closed==[True]


def test_all_roots_are_canonical_from_config_location_and_scientific_identity_survives_relocation(experiment,tmp_path,monkeypatch):
    original_hash=experiment.hash;original_snapshot=snapshot_hash(experiment.config)
    override={'DATA_ROOT':tmp_path/'persistent'/'data','ARTIFACT_ROOT':tmp_path/'persistent'/'artifacts','MODEL_CACHE_ROOT':tmp_path/'persistent'/'cache'}
    for key,path in override.items():monkeypatch.setenv(key,str(path/'child'/'..'))
    moved=program.Experiment(experiment.config,config_path=tmp_path/'config.yaml')
    assert moved.data==override['DATA_ROOT'] and moved.artifacts==override['ARTIFACT_ROOT'] and moved.cache==override['MODEL_CACHE_ROOT']
    assert moved.hash==original_hash and snapshot_hash(moved.config)!=original_snapshot
    assert os.environ['HF_HOME']==str(moved.cache) and os.environ['HF_HUB_CACHE']==str(moved.cache/'hub')
    assert os.environ['HF_DATASETS_CACHE']==str(moved.cache/'datasets_sdk')
    for variable,field in [('DATA_ROOT','data_root'),('ARTIFACT_ROOT','artifact_root'),('MODEL_CACHE_ROOT','cache_root')]:
        assert moved.config['runtime'][field]==os.environ[variable]
    monkeypatch.chdir(tmp_path.parent)
    for variable in override:monkeypatch.delenv(variable)
    relative=deepcopy(experiment.config);relative['runtime'].update(data_root='data/../real_data',artifact_root='./real_artifacts',cache_root='real_cache')
    base=tmp_path/'separate_config'
    relocated=program.Experiment(relative,config_path=base/'config.yaml')
    assert relocated.data==base/'real_data' and relocated.artifacts==base/'real_artifacts' and relocated.cache==base/'real_cache'

@pytest.mark.parametrize('name',['kazitext_auxiliary_manifest.json','english.udpipe.provenance.json'])
def test_external_cache_auxiliary_provenance_is_copied_and_verified_in_drive(experiment,name):
    source=experiment.cache/'kazitext_auxiliary'/name
    atomic_write_json(source,{'status':'verified','source':'fixture'})
    original=source.read_bytes()
    experiment.publish_permanent(source,{'artifact_type':'auxiliary_provenance','simulation':False})
    destination=experiment.artifacts/'configurations'/'auxiliary_provenance'/file_hash(source)/name
    assert source.read_bytes()==original and destination.read_bytes()==original
    records=experiment._artifact_service.list_artifacts(local_path=str(destination))
    assert len(records)==1 and records[0]['sync_status']=='VERIFIED'
    assert experiment._artifact_service.drive.uploads[-1]['bytes']==original


@pytest.mark.parametrize('cached,name',[(True,'private.json'),(False,'english.udpipe.provenance.json')])
def test_external_artifacts_still_require_known_auxiliary_cache_provenance(experiment,tmp_path,cached,name):
    source=(experiment.cache if cached else tmp_path/'unrelated')/name
    atomic_write_json(source,{'fixture':True})
    with pytest.raises(ValueError,match='outside configured'):
        experiment.publish_permanent(source,{'artifact_type':'auxiliary_provenance'})
    assert experiment._artifact_service.drive.uploads==[]


def test_paused_handler_replicates_progress_and_worker_state_before_returning(experiment,monkeypatch):
    stage=Stage('clean','collect_clean',())
    experiment.runner=StageRunner({'clean':stage},experiment.artifacts/'state'/'workflow',experiment.hash,experiment.code)
    RecoveryManager.for_worker(experiment.artifacts).reserve_retry('storage:control:retry')
    path=experiment.data/'clean_traces'/'collection_status.json'
    atomic_write_json(path,{'status':'paused','completed_ids':['one']})
    monkeypatch.setattr(experiment,'action',lambda _:{'status':'PAUSED','output_files':[str(path)]})
    with artifact_hook(experiment.publish_permanent):
        result=experiment.runner.run(['clean'],{'collect_clean':experiment._execute_stage})
    assert result[0]['status']=='PAUSED' and experiment._current_stage is None
    uploads=experiment._artifact_service.drive.uploads
    assert [row['path'].name for row in uploads]==['collection_status.json','retries.sqlite3','artifact_registry.json']
    manifest=json.loads(uploads[-1]['bytes'])
    assert all(row['artifact_id'] in manifest['artifacts'] for row in uploads[:-1])
    assert 'completion_hash' not in result[0]

def test_transform_failure_closes_native_backend_and_cannot_publish_dataset_manifest(experiment,monkeypatch):
    import data_generation_collection.sft_data.sft_data_preparation as preparation
    atomic_write_json(experiment.data/'accepted'/'accepted_pool.json',{'fixture':True})
    closed=[];backend=SimpleNamespace(close=lambda:closed.append(True))
    monkeypatch.setattr(experiment,'kazi',lambda:backend)
    monkeypatch.setattr(experiment,'tokenizers',lambda:{'7b':'fixture','14b':'fixture'})
    def fail(*args,**kwargs):
        assert kwargs['kazitext'] is backend
        raise ValueError('fixture protected-content violation')
    monkeypatch.setattr(preparation,'build_sft_datasets',fail)
    with pytest.raises(ValueError,match='protected-content violation'):experiment.action(Stage('sft_data','transform',()))
    assert closed==[True] and not (experiment.data/'transformed'/'sft_manifest.json').exists()
