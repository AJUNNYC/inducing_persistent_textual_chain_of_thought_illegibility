"""Drive protocol checks using an explicit in-memory SDK, never live credentials."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from operations.drive import DriveStore, INTENDED_ACCOUNT, PROJECT_TITLE, SCOPE, drive_service, file_hash
from shared_procedures.recovery_procedure import RecoveryStopped


def install_oauth_sdk(monkeypatch, credentials):
    calls = {"loaded": [], "build": [], "refreshed": 0}
    modules = {name: ModuleType(name) for name in (
        "google", "google.oauth2", "google.oauth2.credentials", "google.auth",
        "google.auth.transport", "google.auth.transport.requests", "google.auth.exceptions",
        "googleapiclient", "googleapiclient.discovery")}

    class Credentials:
        @staticmethod
        def from_authorized_user_file(path, scopes=None):
            calls["loaded"].append((path, scopes))
            # Match the SDK: supplied scopes replace the serialized scope list.
            credentials.scopes = list(scopes) if scopes else json.loads(Path(path).read_text())["scopes"]
            return credentials

    def refresh(request):
        calls["refreshed"] += 1
        credentials.valid = True
        credentials.token = "refreshed_fixture_token"

    credentials.refresh = refresh
    credentials.granted_scopes = getattr(credentials, "granted_scopes", None)
    credentials.has_scopes = lambda scopes: set(scopes) <= set(credentials.scopes)
    credentials.to_json = lambda: json.dumps({"scopes": credentials.scopes,
        "token": credentials.token, "refresh_token": credentials.refresh_token,
        "client_id": "fixture", "client_secret": "fixture"})
    modules["google.oauth2.credentials"].Credentials = Credentials
    modules["google.auth.transport.requests"].Request = lambda: object()
    modules["google.auth.exceptions"].RefreshError = type("RefreshError", (Exception,), {})
    def build(*args, **kwargs):
        calls["build"].append((args, kwargs))
        return "fixture_service"
    modules["googleapiclient.discovery"].build = build
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    # Live token introspection is a separate network gate. This explicit fixture
    # checks its call and scope validator without accepting a real token.
    from operations import drive
    monkeypatch.setattr(drive, "_validate_live_scope", lambda cred: drive._validate_scope_set(cred.scopes))
    return calls


def test_expired_restricted_oauth_refresh_is_persisted_privately(tmp_path, monkeypatch):
    token = tmp_path / "private_credentials.json"
    token.write_text(json.dumps({"scopes": [SCOPE], "refresh_token": "fixture_refresh"}))
    credentials = SimpleNamespace(valid=False, scopes=[SCOPE], token="expired_fixture",
                                  refresh_token="fixture_refresh")
    calls = install_oauth_sdk(monkeypatch, credentials)
    assert drive_service(token) == "fixture_service"
    assert calls["refreshed"] == 1
    assert json.loads(token.read_text())["token"] == "refreshed_fixture_token"
    assert token.stat().st_mode & 0o777 == 0o600
    assert calls["loaded"][0][1] is None  # Serialized consent is never overwritten.
    assert calls["build"][0][1]["cache_discovery"] is False


def test_existing_broad_consent_cannot_be_narrowed_by_loader_argument(tmp_path, monkeypatch):
    token = tmp_path / "private_credentials.json"
    broad = "https://www.googleapis.com/auth/drive"
    token.write_text(json.dumps({"scopes": [SCOPE, broad], "refresh_token": "fixture_refresh"}))
    credentials = SimpleNamespace(valid=True, scopes=[SCOPE, broad], token="fixture",
                                  refresh_token="fixture_refresh")
    calls = install_oauth_sdk(monkeypatch, credentials)
    with pytest.raises(RuntimeError, match="scope|Scope|consent|restrict"):
        drive_service(token)
    assert not calls["build"]


def test_refresh_returning_broad_granted_scopes_is_rejected(tmp_path, monkeypatch):
    token = tmp_path / "private_credentials.json"
    original = json.dumps({"scopes": [SCOPE], "refresh_token": "fixture_refresh"})
    token.write_text(original)
    credentials = SimpleNamespace(valid=False, scopes=[SCOPE], token="expired_fixture",
        refresh_token="fixture_refresh", granted_scopes=[SCOPE, "https://www.googleapis.com/auth/drive"])
    calls = install_oauth_sdk(monkeypatch, credentials)
    with pytest.raises(RuntimeError, match="scope|Scope|consent|restrict"):
        drive_service(token)
    assert calls["refreshed"] == 1 and not calls["build"]
    assert token.read_text() == original


@pytest.mark.parametrize('ok,actual_scopes', [(True, SCOPE),
    (True, SCOPE + ' https://www.googleapis.com/auth/drive'), (False, SCOPE)])
def test_actual_access_token_scope_is_introspected_without_scope_override(monkeypatch, ok, actual_scopes):
    from operations.drive import _validate_live_scope
    import requests
    requests_seen = []
    def get(url, **kwargs):
        requests_seen.append((url, kwargs))
        return SimpleNamespace(ok=ok, json=lambda: {'scope': actual_scopes})
    monkeypatch.setattr(requests, 'get', get)
    if ok and actual_scopes == SCOPE:
        _validate_live_scope(SimpleNamespace(token='explicit_fixture_token'))
    else:
        with pytest.raises(RuntimeError):
            _validate_live_scope(SimpleNamespace(token='explicit_fixture_token'))
    assert requests_seen == [('https://oauth2.googleapis.com/tokeninfo',
                              {'params': {'access_token': 'explicit_fixture_token'}, 'timeout': 30})]


class HttpFailure(Exception):
    def __init__(self, status):
        self.resp = SimpleNamespace(status=status)


class FakeRequest:
    def __init__(self, value):
        self.value = value
    def execute(self, **kwargs):
        return deepcopy(self.value() if callable(self.value) else self.value)


class UploadRequest:
    def __init__(self, service, file_id, media):
        self.service, self.file_id, self.media = service, file_id, media
    def next_chunk(self, num_retries=0):
        assert num_retries == 0  # Exactly one shared bounded retry controller.
        self.service.chunk_calls += 1
        if self.service.chunk_failures:
            raise self.service.chunk_failures.pop(0)
        path = Path(self.media.filename)
        remote = self.service.metadata[self.file_id]
        remote.update(size=str(path.stat().st_size), md5Checksum=file_hash(path, "md5"))
        return None, deepcopy(remote)


class DriveSDK:
    def __init__(self, *, email=INTENDED_ACCOUNT):
        self.email = email
        self.metadata = {
            "project": {"id": "project", "name": PROJECT_TITLE,
                        "mimeType": "application/vnd.google-apps.folder", "parents": ["root"], "trashed": False},
            "child": {"id": "child", "name": "checkpoints", "mimeType": "application/vnd.google-apps.folder",
                      "parents": ["project"], "trashed": False},
            "outside": {"id": "outside", "name": "unrelated", "mimeType": "application/vnd.google-apps.folder",
                        "parents": ["root"], "trashed": False}}
        self.creates, self.upload_updates, self.chunk_calls = [], [], 0
        self.chunk_failures = []
    def files(self): return self
    def about(self): return SimpleNamespace(get=lambda **kw: FakeRequest({"user": {"emailAddress": self.email}}))
    def get(self, fileId, **kwargs): return FakeRequest(lambda: self.metadata[fileId])
    def list(self, q, **kwargs):
        if "artifact_id" in q:
            matches = [v for v in self.metadata.values() if v.get("appProperties", {}).get("artifact_id") == "fixture_artifact"
                       and v.get("appProperties", {}).get("project_id") == "project"]
        else:
            matches = []
        return FakeRequest({"files": matches})
    def create(self, body, media_body=None, **kwargs):
        self.creates.append(deepcopy(body))
        file_id = "new_" + str(len(self.creates))
        self.metadata[file_id] = {**deepcopy(body), "id": file_id}
        return UploadRequest(self, file_id, media_body) if media_body else FakeRequest({"id": file_id})
    def update(self, fileId, body, media_body=None, **kwargs):
        self.metadata[fileId].update(deepcopy(body))
        if media_body:
            self.upload_updates.append(fileId)
            return UploadRequest(self, fileId, media_body)
        return FakeRequest({"id": fileId})


def install_media_sdk(monkeypatch):
    module = ModuleType("googleapiclient.http")
    module.MediaFileUpload = lambda filename, **kwargs: SimpleNamespace(filename=filename, **kwargs)
    monkeypatch.setitem(sys.modules, "googleapiclient.http", module)


def store_and_source(tmp_path, monkeypatch, sdk=None):
    install_media_sdk(monkeypatch)
    sdk = sdk or DriveSDK()
    store = DriveStore(sdk, "project", state_root=tmp_path / "recovery")
    store.create_project_folder()
    run = store.recovery.run
    monkeypatch.setattr(store.recovery, "run", lambda operation, call: run(operation, call, sleep=lambda _: None))
    source = tmp_path / "checkpoint.bin"
    source.write_bytes(b"explicit synthetic adapter fixture")
    return store, sdk, source


def test_project_account_and_explicit_folder_identity_are_checked(tmp_path):
    sdk = DriveSDK(email="another.account@example.com")
    store = DriveStore(sdk, "project", state_root=tmp_path / "state")
    with pytest.raises(RuntimeError, match="account"):
        store.create_project_folder()
    sdk.email = INTENDED_ACCOUNT
    sdk.metadata["project"]["name"] = "Different project"
    with pytest.raises(RuntimeError, match="canonical"):
        store.create_project_folder()


def test_pending_upload_after_restart_repairs_same_file_id(tmp_path, monkeypatch):
    store, sdk, source = store_and_source(tmp_path, monkeypatch)
    sdk.metadata["pending"] = {"id": "pending", "name": source.name, "parents": ["child"],
        "appProperties": {"artifact_id": "fixture_artifact", "project_id": "project",
                          "config_hash": "a" * 64, "sha256": file_hash(source), "upload_status": "pending"}}
    response = store.upload_artifact(source, artifact_id="fixture_artifact", folder_id="child", config_hash="a" * 64)
    assert response["id"] == "pending"
    assert sdk.upload_updates == ["pending"] and not sdk.creates
    assert sdk.metadata["pending"]["appProperties"]["upload_status"] == "verified"
    # A verified repeat is read-only and does not restart media or create duplicates.
    assert store.upload_artifact(source, artifact_id="fixture_artifact", folder_id="child", config_hash="a" * 64)["id"] == "pending"
    assert sdk.chunk_calls == 1


def test_transient_upload_retries_are_bounded_and_durable(tmp_path, monkeypatch):
    store, sdk, source = store_and_source(tmp_path, monkeypatch)
    sdk.chunk_failures = [HttpFailure(503)] * 5
    with pytest.raises(RecoveryStopped, match="exhausted"):
        store.upload_artifact(source, artifact_id="fixture_artifact", folder_id="child", config_hash="a" * 64)
    assert sdk.chunk_calls == 4
    assert store.recovery.counters()["operations"]["upload_fixture_artifact"] == 3
    restarted = DriveStore(sdk, "project", state_root=tmp_path / "recovery")
    restarted.create_project_folder()
    with pytest.raises(RecoveryStopped, match="exhausted"):
        restarted.upload_artifact(source, artifact_id="fixture_artifact", folder_id="child", config_hash="a" * 64)
    assert sdk.chunk_calls == 4


def test_permanent_upload_error_does_not_consume_retry_allowance(tmp_path, monkeypatch):
    store, sdk, source = store_and_source(tmp_path, monkeypatch)
    sdk.chunk_failures = [HttpFailure(403)]
    with pytest.raises(HttpFailure):
        store.upload_artifact(source, artifact_id="fixture_artifact", folder_id="child", config_hash="a" * 64)
    assert sdk.chunk_calls == 1
    assert store.recovery.counters()["worker_retries"] == 0


def test_upload_rejects_folder_outside_canonical_project(tmp_path, monkeypatch):
    store, sdk, source = store_and_source(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="project|descend|canonical|folder"):
        store.upload_artifact(source, artifact_id="fixture_artifact", folder_id="outside", config_hash="a" * 64)
    assert not sdk.creates and not sdk.upload_updates and sdk.chunk_calls == 0
