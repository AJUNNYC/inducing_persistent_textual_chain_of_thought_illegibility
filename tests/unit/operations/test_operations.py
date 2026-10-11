"""Offline integration doubles exercise transport failures and integrity boundaries."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import pytest
from operations.artifacts import ArtifactService
from operations.assets import resolve_revision, validate_asset
from operations.drive import DriveStore, _private_json, file_hash
from operations.runpod import RunPodClient
from operations.workflow import Stage, StageRunner, experiment_graph, topological_order
from shared_procedures.storage import atomic_write_json


def provenance():
    return {"experiment_id": "fixture", "branch": "rot13", "model": "7b", "run_id": "level_01",
            "stage": "training", "producer": "test", "source_code_commit": "a" * 40,
            "configuration_hash": "b" * 64, "dataset_revision": "c" * 40, "model_revision": "d" * 40, "seeds": [42]}


def test_parallel_uploads_use_independent_transports_and_preserve_failed_records(tmp_path):
    import threading
    import time
    shared={'active':0,'peak':0,'forks':set()}
    lock=threading.Lock()
    class ParallelDrive:
        def create_project_folder(self): return 'folder'
        def ensure_path(self, parts): return 'folder'
        def fork(self):
            with lock:shared['forks'].add(threading.get_ident())
            return ParallelDrive()
        def upload_artifact(self, path, **kwargs):
            with lock:
                shared['active']+=1
                shared['peak']=max(shared['peak'],shared['active'])
            try:
                time.sleep(.02)
                if Path(path).name=='2.json':raise TimeoutError('fixture')
                return {'id':kwargs['artifact_id']}
            finally:
                with lock:shared['active']-=1
    service=ArtifactService(tmp_path,tmp_path/'registry.json',ParallelDrive())
    for index in range(8):
        file=tmp_path/f'{index}.json';file.write_text(str(index));service.register(file,provenance())
    report=service.sync_artifacts(workers=4)
    assert report['status']=='FAILED' and len(report['failed'])==1
    assert len(service.list_artifacts(sync_status='VERIFIED'))==7
    failed=service.list_artifacts(sync_status='FAILED')[0]
    assert failed['sync_error']=='TimeoutError'
    assert shared['peak']>1 and len(shared['forks'])>1


def test_artifact_compatibility_and_secret_rejection(tmp_path):
    service = ArtifactService(tmp_path, tmp_path / "registry.json")
    output = tmp_path / "adapter.bin"
    output.write_bytes(b"real fixture weights")
    record = service.register(output, provenance())
    assert service.register(output, provenance())["artifact_id"] == record["artifact_id"]
    assert len(service.list_artifacts()) == 1
    assert service.sync_artifacts()["status"] == "BLOCKED"
    assert service.retrieve_checkpoint(record["artifact_id"], tmp_path / "restored.bin", provenance()) == output
    with pytest.raises(RuntimeError, match="compatibility"):
        service.retrieve_checkpoint(record["artifact_id"], tmp_path / "restored.bin", {**provenance(), "model": "14b"})
    output.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="verified"):
        service.retrieve_checkpoint(record["artifact_id"], tmp_path / "restored.bin", provenance())
    secret = tmp_path / ".env"
    secret.write_text("DUMMY_KEY=fixture")
    with pytest.raises(ValueError, match="Credential"):
        service.register(secret, provenance())


def test_failed_upload_persists_retryable_state(tmp_path):
    class BrokenDrive:
        def create_project_folder(self): return "folder"
        def ensure_path(self, parts): return "folder"
        def upload_artifact(self, *args, **kwargs): raise TimeoutError("fixture failure")
    service = ArtifactService(tmp_path, tmp_path / "registry.json", BrokenDrive())
    file = tmp_path / "record.json"
    file.write_text('{"scientific": "fixture"}')
    service.register(file, provenance())
    result = service.sync_artifacts()
    assert result["status"] == "FAILED"
    assert service.list_artifacts()[0]["sync_status"] == "FAILED"
    assert service.list_artifacts()[0]["sync_error"] == "TimeoutError"


def test_entire_graph_and_transitive_invalidation(tmp_path):
    graph = experiment_graph()
    assert sum(v.kind == "sft" for v in graph.values()) == 40
    assert sum(v.kind == "rl" for v in graph.values()) == 2
    assert sum(v.kind == "evaluate" for v in graph.values()) == 60
    order = topological_order(graph, ["sync"])
    assert order.index("baseline_repeatability") < order.index("rot13_7b_level_01")
    assert order.index("statistical_validation") < order.index("rl_14b")
    simple = {"upstream": Stage("upstream", "a", ()), "downstream": Stage("downstream", "b", ("upstream",))}
    runner = StageRunner(simple, tmp_path / "state", "config", "code")
    calls = []
    def action(stage):
        calls.append(stage.stage_id)
        path = tmp_path / f"{stage.stage_id}_data.json"
        path.write_text(stage.stage_id)
        return {"status": "VERIFIED", "output_files": [str(path)]}
    runner.run(["downstream"], {"a": action, "b": action})
    runner.run(["downstream"], {"a": action, "b": action})
    assert calls == ["upstream", "downstream"]
    (tmp_path / "upstream_data.json").write_text("corrupt")
    assert runner.status("downstream")["status"] == "INVALIDATED"


def test_missing_outputs_and_cycle_are_failures(tmp_path):
    graph = {"a": Stage("a", "a", ())}
    runner = StageRunner(graph, tmp_path, "config", "code")
    with pytest.raises(RuntimeError, match="output"):
        runner.run(["a"], {"a": lambda _: {"status": "VERIFIED"}})
    assert runner.status("a")["status"] != "COMPLETED"
    with pytest.raises(ValueError, match="Circular"):
        topological_order({"a": Stage("a", "a", ("b",)), "b": Stage("b", "b", ("a",))}, ["a"])


def test_asset_cache_detects_corruption(tmp_path):
    file = tmp_path / "tokenizer.json"
    file.write_text("fixture-tokenizer")
    atomic_write_json(tmp_path / "asset_manifest.json", {"revision": "a" * 40, "files": (files := {"tokenizer.json": {"size": file.stat().st_size, "sha256": file_hash(file)}}), "manifest_hash": __import__("shared_procedures.identity",fromlist=["content_hash"]).content_hash(files)})
    assert validate_asset(tmp_path)["revision"] == "a" * 40
    file.write_text("corrupted")
    with pytest.raises(RuntimeError, match="checksum"):
        validate_asset(tmp_path)
    api = SimpleNamespace(repo_info=lambda *a, **k: SimpleNamespace(sha="c" * 40, siblings=[SimpleNamespace(size=20)]))
    resolved = resolve_revision({"repo_id": "fixture/model", "repo_type": "model", "revision": None}, api)
    assert resolved["size_bytes"] == 20 and resolved["tokenizer_revision"] == "c" * 40


def test_runpod_payload_is_concretely_approved():
    class Session:
        def __init__(self): self.headers = {}; self.calls = []
        def request(self, method, url, **kwargs):
            self.calls.append((method, url, kwargs))
            return SimpleNamespace(ok=True, content=b"{}", json=lambda: {"id": "fixture_pod"})
    session = Session()
    client = RunPodClient("explicit_test_key", session)
    payload = {"imageName": "fixture@sha256:abc", "name": "test", "gpuTypeIds": ["A100"], "gpuCount": 1, "containerDiskInGb": 30, "volumeMountPath": "/workspace"}
    with pytest.raises(RuntimeError, match="explicit"):
        client.create_pod(payload, {})
    authorization = {"approved": True, "authorization_id": "fixture", "approved_payload": deepcopy(payload), "max_total_usd": 1, "max_runtime_seconds": 60}
    assert client.create_pod(payload, authorization)["id"] == "fixture_pod"
    payload["gpuCount"] = 3
    with pytest.raises(RuntimeError, match="differs"):
        client.create_pod(payload, authorization)
    assert len(session.calls) == 1


def test_oauth_private_permissions_and_remote_hash(tmp_path):
    token = tmp_path / "private_credentials.json"
    _private_json(token, {"refresh_token": "isolated_fixture"})
    assert token.stat().st_mode & 0o777 == 0o600
    file = tmp_path / "output.bin"
    file.write_bytes(b"fixture")
    remote = {"id": "fixture", "size": str(file.stat().st_size), "md5Checksum": file_hash(file, "md5"), "appProperties": {"sha256": file_hash(file)}}
    class Service:
        def files(self): return self
        def get(self, **kwargs): return self
        def execute(self): return remote
    store = DriveStore(Service())
    assert store.verify_artifact("fixture", file)["id"] == "fixture"
    remote["md5Checksum"] = "bad"
    with pytest.raises(RuntimeError, match="checksum"):
        store.verify_artifact("fixture", file)


def test_incremental_index_and_explicit_remote_reconciliation(tmp_path):
    class CountingDrive:
        def __init__(self): self.uploads = 0; self.verifications = 0
        def create_project_folder(self): pass
        def ensure_path(self, parts): return 'folder'
        def upload_artifact(self, path, **kwargs):
            self.uploads += 1
            return {'id':str(self.uploads)}
        def verify_registered_artifact(self, record): self.verifications += 1
    drive=CountingDrive()
    service=ArtifactService(tmp_path,tmp_path/'registry.json',drive)
    for index in range(25):
        path=tmp_path/f'raw_{index}.json';path.write_text(str(index))
        service.register(path,provenance())
        assert service.sync_artifacts()['status']=='VERIFIED'
    assert drive.uploads==25 and drive.verifications==0
    assert not service.registry.exists()  # Hook updates are indexed, not full JSON rewrites.
    service=ArtifactService(tmp_path,tmp_path/'registry.json',drive)
    assert len(service.list_artifacts(sync_status='VERIFIED'))==25
    assert service.sync_artifacts(verify_existing=True)['scope']=='full_reconciliation'
    assert drive.verifications==25 and service.registry.is_file()
    assert len(json.loads(service.registry.read_text())['artifacts'])==25


def test_persisted_pod_controller_stops_before_cost_limit(tmp_path):
    client=RunPodClient(api_key='fixture')
    elapsed=[100.0];stops=[]
    client.inspect_pod=lambda pod:({'desiredStatus':'EXITED','runtime':None,'costPerHr':2} if stops else {'desiredStatus':'RUNNING','costPerHr':2})
    client.stop_pod=lambda pod:stops.append(pod)
    authorization={'approved':True,'authorization_id':'fixture','pod_id':'fixture_pod','max_runtime_seconds':90,
                   'max_total_usd':0.04,'authorized_hourly_usd':2,'price_evidence':'fixture_quote','billing_started_at_unix':100}
    def pause(seconds):elapsed[0]+=seconds
    state=client.guarded_lifecycle('fixture_pod',authorization,tmp_path/'lifecycle.json',clock=lambda:elapsed[0],pause=pause)
    assert state['status']=='STOPPED' and stops==['fixture_pod']
    assert state['elapsed_seconds']==60 and state['stop_reason']=='authorized_runtime_or_cost_limit'
    assert client.guarded_lifecycle('fixture_pod',authorization,tmp_path/'lifecycle.json',clock=lambda:elapsed[0],pause=pause)==state
    assert stops==['fixture_pod']


def test_pause_preserves_progress_and_blocks_children_until_resume(tmp_path):
    graph={'generation':Stage('generation','generate',()),'fit':Stage('fit','fit',('generation',))}
    runner=StageRunner(graph,tmp_path/'workflow','fixed_config','fixed_commit')
    output=tmp_path/'progress.json';output.write_text('{"completed_candidates":1}')
    calls=[]
    def generation(stage):
        calls.append(stage.stage_id)
        return {'status':'PAUSED' if len(calls)==1 else 'VERIFIED','output_files':[str(output)]}
    def fit(stage):
        calls.append(stage.stage_id)
        result=tmp_path/'adapter.bin';result.write_bytes(b'explicit_fixture')
        return {'status':'VERIFIED','output_files':[str(result)]}
    first=runner.run(['fit'],{'generate':generation,'fit':fit})
    assert first[-1]['status']=='PAUSED' and calls==['generation']
    assert runner.status('generation')['status']=='PAUSED'
    assert runner.status('fit')['status']=='WAITING_FOR_DEPENDENCIES'
    second=runner.run(['fit'],{'generate':generation,'fit':fit})
    assert second[-1]['status']=='COMPLETED' and calls==['generation','generation','fit']


def test_image_inventory_runs_without_git_and_excludes_runtime(tmp_path):
    from operations.validation import tracked_inventory
    (tmp_path/'src').mkdir();(tmp_path/'src'/'example.py').write_text('def f(): return 1')
    (tmp_path/'artifacts').mkdir();(tmp_path/'artifacts'/'.gitkeep').touch()
    (tmp_path/'artifacts'/'raw.json').write_text('{"fixture":true}')
    (tmp_path/'.env').write_text('FIXTURE=private')
    (tmp_path/'.env.example').write_text('FIXTURE=')
    (tmp_path/'.venv').mkdir();(tmp_path/'.venv'/'binary').touch()
    assert tracked_inventory(tmp_path)==['.env.example','artifacts/.gitkeep','src/example.py']


class MemoryCollectionDrive(DriveStore):
    """Explicit offline transport fixture; production integrity/restore code is real."""
    def __init__(self,recovery_path):
        from shared_procedures.recovery_procedure import RecoveryManager
        self.recovery=RecoveryManager(recovery_path)
        self.folder_id='project';self._verified_folders={'project'}
        self.metadata={};self.payloads={};self.folders={};self.publications=[]
        self.fail_payload=False
        self.service=SimpleNamespace(files=lambda:SimpleNamespace(get=lambda fileId,**kwargs:
            SimpleNamespace(execute=lambda:deepcopy(self.metadata[fileId]))))

    def create_project_folder(self):return self.folder_id

    def ensure_path(self,parts):
        folder='/'.join(parts)
        self.folders[folder]=folder;self._verified_folders.add(folder)
        return folder

    def list_artifacts(self,folder_id=None):
        return [deepcopy(remote) for remote in self.metadata.values() if remote['parents']==[folder_id]]

    def resolve_artifact_id(self,artifact_id):
        found=[remote for remote in self.metadata.values() if remote['appProperties']['artifact_id']==artifact_id]
        if len(found)>1:raise RuntimeError('Duplicate artifact identity')
        return deepcopy(found[0]) if found else None

    def upload_artifact(self,path,*,artifact_id,folder_id,config_hash):
        if self.fail_payload and not folder_id.startswith('recovery/'):
            raise ConnectionError('Explicit fixture interruption')
        path=Path(path);existing=self.resolve_artifact_id(artifact_id)
        if existing:
            self.verify_artifact(existing['id'],path)
            return existing
        identifier=f'file_{len(self.metadata)}'
        remote={'id':identifier,'name':path.name,'size':str(path.stat().st_size),
            'md5Checksum':file_hash(path,'md5'),'parents':[folder_id],
            'appProperties':{'artifact_id':artifact_id,'project_id':self.folder_id,
                'sha256':file_hash(path),'config_hash':config_hash,'upload_status':'verified',
                'worker_id':__import__('os').environ.get('ILLEGIBILITY_WORKER_ID','coordinator'),
                'worker_retry_floor':str(self.recovery.counters()['worker_retries']),
                'operation_retries':str(self.recovery.counters()['operations'].get('upload_'+artifact_id,0))}}
        self.metadata[identifier]=remote;self.payloads[identifier]=path.read_bytes()
        self.publications.append(('descriptor' if folder_id.startswith('recovery/') else 'payload',artifact_id))
        return deepcopy(remote)

    def download_artifact(self,artifact_id,destination,*,config_hash=None):
        from shared_procedures.storage import atomic_write_bytes
        remote=self.resolve_artifact_id(artifact_id)
        if not remote or remote['appProperties']['config_hash'] != config_hash:
            raise RuntimeError('Incompatible fixture remote artifact')
        atomic_write_bytes(destination,self.payloads[remote['id']])
        self.verify_artifact(remote['id'],destination)
        return Path(destination)


def collection_provenance():
    return {**provenance(),'stage':'collect_clean','branch':'shared','model':'all',
            'model_revision':'d'*64,'dataset_revision':'c'*64,'run_id':'clean'}


def collection_expected():
    source=collection_provenance()
    return {key:source[key] for key in ('configuration_hash','source_code_commit','dataset_revision','model_revision','seeds','model')}


def publish_native_snapshot(service,drive,root):
    snapshot=Path(root)/'artifacts/operations/worker_state/coordinator/native_boundary/retries.sqlite3'
    drive.recovery.snapshot_database(snapshot)
    service.register(snapshot,{**collection_provenance(),'stage':'generation_preflight','run_id':'generation_preflight'})
    assert service.sync_artifacts()['status']=='VERIFIED'


def test_drive_only_scratch_loss_restores_raw_without_native_generation(tmp_path):
    import shutil
    from data_generation_collection.sft_data.clean_trace_collection.candidate_batch.clean_trace_batch import NativeCleanGenerator,process_clean_batch
    from shared_procedures.identity import content_hash
    from shared_procedures.storage import artifact_hook
    class Tokenizer:
        def encode(self,text,**kwargs):return list(range(len(text.split())))
        def apply_chat_template(self,messages,tokenize=False,**kwargs):
            rendered=''.join(message['content'] for message in messages)
            return self.encode(rendered) if tokenize else rendered
    class Noise:
        def transform(self,reasoning,*args,**kwargs):
            return {'reasoning':reasoning,'content_hash':content_hash(reasoning),'source_hash':content_hash(reasoning),
                'status':'verified','protected_content_valid':True,'changed_word_fraction':0}
    class FixtureNative(NativeCleanGenerator):
        # Native class membership exercises production cache paths; this is
        # explicitly a fixture and supplies no live model/scientific evidence.
        def __init__(self):
            self.models={model:{'revision':'e'*40,'tokenizer_revision':'f'*40} for model in ('7b','14b')}
            self.stop_requested=None;self.calls=[];self.forbid=False
        def generate_many(self,model,candidates,**kwargs):
            if self.forbid:raise AssertionError('Restored raw response was regenerated')
            for row in candidates:
                self.calls.append((model,row['example_id']))
                yield {'raw_generation':'<think>one two three</think> answer','finish_reason':'stop',
                    'generated_token_count':10,'model_revision':'e'*40,'tokenizer_revision':'f'*40,'seed':42}
    original=tmp_path/'old';data=original/'data';root=data/'clean_traces'
    drive=MemoryCollectionDrive(tmp_path/'transport/retries.json')
    service=ArtifactService(original,original/'artifacts/state/artifact_registry.json',drive,drive_recovery=True)
    publish_native_snapshot(service,drive,original)
    generator=FixtureNative();row={'example_id':'fixture-only','prompt':'question','source_content_hash':content_hash('question')}
    options={'kazitext':Noise(),'config_hash':'b'*64}
    with artifact_hook(service.completion_hook(collection_provenance())):
        first=process_clean_batch([row],generator,{model:Tokenizer() for model in ('7b','14b')},output_dir=root,**options)
    assert len(generator.calls)==2 and first[0]['status']=='accepted'
    assert [kind for kind,_ in drive.publications]==['descriptor','payload']*4
    assert not service.registry.exists()  # No stage-boundary full registry export.
    shutil.rmtree(original)  # Lose every byte, registry DB and WAL of the old worker.
    fresh=tmp_path/'fresh';fresh_data=fresh/'data'
    recovered=ArtifactService(fresh,fresh/'artifacts/state/artifact_registry.json',drive,drive_recovery=True)
    report=recovered.restore_collection(collection_expected(),data_root=fresh_data)
    assert report['restored_artifacts']==3 and report['native_retry_state']=='VERIFIED'
    assert len(recovered.list_artifacts(sync_status='VERIFIED'))==3
    generator.forbid=True
    repeated=process_clean_batch([row],generator,{model:Tokenizer() for model in ('7b','14b')},output_dir=fresh_data/'clean_traces',**options)
    assert repeated==first and len(generator.calls)==2
    recovered.assert_collection_reconciled(collection_expected())


def test_descriptor_survives_loss_after_remote_upload_before_sqlite_commit(tmp_path,monkeypatch):
    import shutil
    original=tmp_path/'old';path=original/'data/clean_traces/raw/production/7b/fixture.json'
    atomic_write_json(path,{'fixture':'full-response-cache'})
    drive=MemoryCollectionDrive(tmp_path/'transport/retries.json')
    service=ArtifactService(original,original/'artifacts/state/registry.json',drive,drive_recovery=True)
    publish_native_snapshot(service,drive,original)
    service.register(path,collection_provenance())
    original_put=service._put
    def interrupted_commit(db,record):
        if record['sync_status']=='VERIFIED':raise OSError('Explicit fixture process loss')
        original_put(db,record)
    monkeypatch.setattr(service,'_put',interrupted_commit)
    with pytest.raises(OSError):service.sync_artifacts()
    assert [kind for kind,_ in drive.publications]==['descriptor','payload']*2
    shutil.rmtree(original)
    fresh=tmp_path/'fresh'
    recovered=ArtifactService(fresh,fresh/'artifacts/state/registry.json',drive,drive_recovery=True)
    assert recovered.restore_collection(collection_expected(),data_root=fresh/'data')['restored_artifacts']==1
    assert json.loads((fresh/'data/clean_traces/raw/production/7b/fixture.json').read_text())=={'fixture':'full-response-cache'}


def test_uncertain_payload_never_becomes_a_missing_cache_generation(tmp_path):
    path=tmp_path/'old/data/clean_traces/raw/production/7b/fixture.json'
    atomic_write_json(path,{'fixture':'response'})
    drive=MemoryCollectionDrive(tmp_path/'transport/retries.json')
    service=ArtifactService(tmp_path/'old',tmp_path/'old/artifacts/state/registry.json',drive,drive_recovery=True)
    publish_native_snapshot(service,drive,tmp_path/'old')
    drive.fail_payload=True
    service.register(path,collection_provenance())
    assert service.sync_artifacts()['status']=='FAILED'
    fresh=tmp_path/'fresh'
    recovered=ArtifactService(fresh,fresh/'artifacts/state/registry.json',drive,drive_recovery=True)
    with pytest.raises(RuntimeError,match='unresolved uploads'):
        recovered.restore_collection(collection_expected(),data_root=fresh/'data')
    with pytest.raises(RuntimeError,match='recover-collection'):
        recovered.assert_collection_reconciled(collection_expected())
    assert not list((fresh/'data').rglob('*.json'))


def test_descriptor_tampering_and_unsafe_destination_are_rejected(tmp_path):
    from shared_procedures.identity import canonical_json
    path=tmp_path/'old/data/clean_traces/raw/production/7b/fixture.json'
    atomic_write_json(path,{'fixture':'response'})
    drive=MemoryCollectionDrive(tmp_path/'transport/retries.json')
    service=ArtifactService(tmp_path/'old',tmp_path/'old/artifacts/state/registry.json',drive,drive_recovery=True)
    publish_native_snapshot(service,drive,tmp_path/'old')
    service.register(path,collection_provenance());assert service.sync_artifacts()['status']=='VERIFIED'
    descriptor=next(remote for remote in drive.metadata.values() if remote['parents'][0].startswith('recovery/'))
    value=json.loads(drive.payloads[descriptor['id']]);value['artifact']['relative_path']='../../outside.json'
    payload=canonical_json(value)+b'\n'
    drive.payloads[descriptor['id']]=payload
    descriptor['size']=str(len(payload));descriptor['md5Checksum']=hashlib.md5(payload).hexdigest()
    descriptor['appProperties']['sha256']=hashlib.sha256(payload).hexdigest()
    fresh=tmp_path/'fresh'
    recovered=ArtifactService(fresh,fresh/'artifacts/state/registry.json',drive,drive_recovery=True)
    with pytest.raises(RuntimeError,match='provenance/path'):
        recovered.restore_collection(collection_expected(),data_root=fresh/'data')
    assert not (tmp_path/'outside.json').exists()


def test_original_flattened_registry_keeps_complete_asset_provenance(tmp_path):
    from operations.artifacts import validate_registered_record
    path=tmp_path/'native_inputs/model_descriptor.json'
    atomic_write_json(path,{'fixture':'original producer input'})
    original={**provenance(),'artifact_kind':'source_asset','upstream_filename':'dataset/train.parquet',
        'license':'CC-BY-NC-SA-4.0','source_revision':'e'*40,
        'model_descriptor':{'revision':'f'*40,'tokenizer_revision':'1'*40}}
    service=ArtifactService(tmp_path,tmp_path/'registry.json')
    record=service.register(path,original)
    legacy={key:value for key,value in record.items() if key!='provenance'}
    validate_registered_record(legacy)
    assert ArtifactService._recovery_descriptor(legacy)['artifact']['provenance']==original
    legacy['license']='changed-license'
    with pytest.raises(RuntimeError,match='identity differs'):
        validate_registered_record(legacy)


def test_native_restore_imports_actual_retry_floor_without_reset_or_double_charge(tmp_path):
    import shutil
    from shared_procedures.recovery_procedure import RecoveryManager
    original=tmp_path/'old';fresh=tmp_path/'fresh'
    path=original/'data/clean_traces/raw/production/7b/fixture.json'
    atomic_write_json(path,{'fixture':'response'})
    drive=MemoryCollectionDrive(original/'artifacts/state/retries.json')
    for index in range(7):drive.recovery.reserve_retry(f'old_transport_{index}')
    service=ArtifactService(original,original/'artifacts/state/registry.json',drive,drive_recovery=True)
    service.register(path,collection_provenance());assert service.sync_artifacts()['status']=='VERIFIED'
    snapshot=original/'artifacts/operations/worker_state/coordinator/fixture/retries.sqlite3'
    drive.recovery.snapshot_database(snapshot)
    service.register(snapshot,collection_provenance());assert service.sync_artifacts()['status']=='VERIFIED'
    shutil.rmtree(original)
    drive.recovery=RecoveryManager(fresh/'artifacts/state/retries.json')
    for _ in range(2):drive.recovery.reserve_retry('new_restore_transport')
    recovered=ArtifactService(fresh,fresh/'artifacts/state/registry.json',drive,drive_recovery=True)
    report=recovered.restore_collection(collection_expected(),data_root=fresh/'data')
    assert report['worker_retry_floor']==7 and len(report['non_collection_descriptor_ids'])==1
    assert drive.recovery.counters()['worker_retries']==9
    recovered.assert_collection_reconciled(collection_expected())
    recovered.restore_collection(collection_expected(),data_root=fresh/'data')
    assert drive.recovery.counters()['worker_retries']==9
    recovered.assert_collection_reconciled(collection_expected())


@pytest.mark.parametrize('kind',['descriptor','payload'])
def test_native_resume_cannot_invent_missing_old_retry_metadata(tmp_path,kind):
    path=tmp_path/'old/data/clean_traces/raw/production/7b/fixture.json'
    atomic_write_json(path,{'fixture':'response'})
    drive=MemoryCollectionDrive(tmp_path/'transport/retries.json')
    service=ArtifactService(tmp_path/'old',tmp_path/'old/artifacts/state/registry.json',drive,drive_recovery=True)
    publish_native_snapshot(service,drive,tmp_path/'old')
    service.register(path,collection_provenance());assert service.sync_artifacts()['status']=='VERIFIED'
    remote=next(value for value in drive.metadata.values()
        if value['parents'][0].startswith('recovery/') is (kind=='descriptor'))
    remote['appProperties'].pop('worker_retry_floor')
    fresh=tmp_path/'fresh'
    recovered=ArtifactService(fresh,fresh/'artifacts/state/registry.json',drive,drive_recovery=True)
    with pytest.raises(RuntimeError,match='authoritative worker retry metadata'):
        recovered.restore_collection(collection_expected(),data_root=fresh/'data')


def test_coherent_native_snapshot_preserves_asset_operation_retry_ceiling(tmp_path):
    import shutil
    from shared_procedures.recovery_procedure import RecoveryManager,RecoveryStopped,RetryableFailure
    original=tmp_path/'old';fresh=tmp_path/'fresh'
    drive=MemoryCollectionDrive(original/'artifacts/state/retries.json')
    operation='metadata:model:fixture/model:'+'e'*40
    drive.recovery._begin_attempt(operation)
    drive.recovery.reserve_retry(operation);drive.recovery.reserve_retry(operation)
    service=ArtifactService(original,original/'artifacts/state/registry.json',drive,drive_recovery=True)
    publish_native_snapshot(service,drive,original)
    raw=original/'data/clean_traces/raw/production/7b/fixture.json'
    atomic_write_json(raw,{'fixture':'response'})
    service.register(raw,collection_provenance());assert service.sync_artifacts()['status']=='VERIFIED'
    shutil.rmtree(original)
    drive.recovery=RecoveryManager(fresh/'artifacts/state/retries.json')
    recovered=ArtifactService(fresh,fresh/'artifacts/state/registry.json',drive,drive_recovery=True)
    recovered.restore_collection(collection_expected(),data_root=fresh/'data')
    assert drive.recovery.counters()['operations'][operation]==2
    assert operation in drive.recovery.counters()['started_operations']
    attempts=[]
    def fail():attempts.append(1);raise RetryableFailure('Explicit fixture transient error')
    with pytest.raises(RecoveryStopped):drive.recovery.run(operation,fail,sleep=lambda _:None)
    assert attempts==[1] and drive.recovery.counters()['operations'][operation]==3
