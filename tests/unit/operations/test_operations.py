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
