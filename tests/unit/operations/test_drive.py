"""Drive protocol checks using an explicit in-memory SDK, never live credentials."""
from copy import deepcopy
import json
import io
from pathlib import Path
import re
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
    monkeypatch.setattr(drive, "_proxy_aware_http", lambda cred: None)
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
        return SimpleNamespace(status_code=200 if ok else 403, json=lambda: {'scope': actual_scopes}, close=lambda: None)
    monkeypatch.setattr(requests, 'get', get)
    if ok and actual_scopes == SCOPE:
        _validate_live_scope(SimpleNamespace(token='explicit_fixture_token'))
    else:
        with pytest.raises(RuntimeError):
            _validate_live_scope(SimpleNamespace(token='explicit_fixture_token'))
    assert requests_seen == [('https://oauth2.googleapis.com/tokeninfo',
                              {'params': {'access_token': 'explicit_fixture_token'}, 'timeout': 30,
                               'allow_redirects': False, 'verify': True})]


def test_actual_token_introspection_rejects_redirect_even_with_plausible_scope_json(monkeypatch):
    from operations.drive import _validate_live_scope
    import requests
    closed=[]
    monkeypatch.setattr(requests,'get',lambda *args,**kwargs: SimpleNamespace(
        status_code=302,json=lambda:{'scope':SCOPE},close=lambda:closed.append(True)))
    with pytest.raises(RuntimeError,match='scope verification unavailable'):
        _validate_live_scope(SimpleNamespace(token='explicit_public_fixture_token'))
    assert closed==[True]


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
        self.service.payloads[self.file_id] = path.read_bytes()
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
        self.payloads, self.download_calls = {}, []
    def files(self): return self
    def about(self): return SimpleNamespace(get=lambda **kw: FakeRequest({"user": {"emailAddress": self.email}}))
    def get(self, fileId, **kwargs): return FakeRequest(lambda: self.metadata[fileId])
    def get_media(self, fileId, **kwargs):
        self.download_calls.append(fileId)
        return FakeRequest(lambda: self.payloads[fileId])
    def list(self, q, **kwargs):
        if "artifact_id" in q:
            artifact_id = re.search(r"key='artifact_id' and value='([^']+)'", q).group(1)
            matches = [v for v in self.metadata.values() if v.get("appProperties", {}).get("artifact_id") == artifact_id
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
    class MediaDownload:
        def __init__(self, stream, request, **kwargs):
            self.stream, self.request = stream, request
        def next_chunk(self, num_retries=0):
            assert num_retries == 0
            self.stream.write(self.request.execute())
            return None, True
    module.MediaIoBaseDownload = MediaDownload
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


def registered_fixture(tmp_path, monkeypatch):
    from operations.artifacts import ArtifactService
    store, sdk, source = store_and_source(tmp_path, monkeypatch)
    expected = {'configuration_hash': 'a' * 64, 'source_code_commit': 'b' * 40,
        'dataset_revision': 'c' * 40, 'model_revision': 'd' * 40, 'model': 'all', 'seeds': [42]}
    provenance = {**expected, 'experiment_id': 'fixture', 'branch': 'shared', 'run_id': 'fixture',
        'stage': 'runtime', 'producer': 'explicit_offline_test'}
    service = ArtifactService(tmp_path, tmp_path / 'registry.json', store)
    record = service.register(source, provenance)
    assert service.sync_artifacts()['status'] == 'VERIFIED'
    return store, sdk, source, service.list_artifacts()[0], expected


def test_registered_restore_uses_drive_even_if_original_local_copy_exists(tmp_path, monkeypatch):
    store, sdk, source, record, expected = registered_fixture(tmp_path, monkeypatch)
    destination = tmp_path / 'fresh_restore' / 'checkpoint.bin'
    assert store.restore_registered_artifact(record, destination, expected_provenance=expected) == destination
    assert destination.read_bytes() == source.read_bytes()
    assert sdk.download_calls == [record['drive_file_id']]
    with pytest.raises(RuntimeError, match='new path'):
        store.restore_registered_artifact(record, destination, expected_provenance=expected)
    assert sdk.download_calls == [record['drive_file_id']]


@pytest.mark.parametrize('change', ['artifact_id', 'upload_status', 'sha256', 'md5', 'size', 'parents', 'trashed'])
def test_registered_restore_rejects_changed_remote_publication(tmp_path, monkeypatch, change):
    store, sdk, source, record, expected = registered_fixture(tmp_path, monkeypatch)
    remote = sdk.metadata[record['drive_file_id']]
    if change in ('artifact_id', 'upload_status', 'sha256'):
        remote['appProperties'][change] = 'different'
    else:
        remote[{'md5': 'md5Checksum'}.get(change, change)] = {
            'md5': 'e' * 32, 'size': str(record['size'] + 1), 'parents': ['outside'], 'trashed': True}[change]
    destination = tmp_path / 'fresh.bin'
    with pytest.raises(RuntimeError):
        store.restore_registered_artifact(record, destination, expected_provenance=expected)
    assert not destination.exists() and not sdk.download_calls


def test_registered_restore_checks_downloaded_bytes_and_removes_corrupt_output(tmp_path, monkeypatch):
    store, sdk, source, record, expected = registered_fixture(tmp_path, monkeypatch)
    sdk.payloads[record['drive_file_id']] = b'corrupted remote media bytes'
    destination = tmp_path / 'fresh.bin'
    with pytest.raises(RuntimeError, match='checksum|integrity'):
        store.restore_registered_artifact(record, destination, expected_provenance=expected)
    assert not destination.exists() and not list(tmp_path.glob('.download_*'))


def test_registered_restore_requires_complete_matching_source_provenance(tmp_path, monkeypatch):
    store, sdk, source, record, expected = registered_fixture(tmp_path, monkeypatch)
    for incompatible in ({**expected, 'source_code_commit': 'f' * 40},
                         {key: value for key, value in expected.items() if key != 'seeds'}):
        with pytest.raises(RuntimeError, match='provenance'):
            store.restore_registered_artifact(record, tmp_path / 'fresh.bin', expected_provenance=incompatible)
    assert not sdk.download_calls


def test_registered_restore_rejects_inconsistent_artifact_formula_despite_matching_remote_fields(tmp_path,monkeypatch):
    store,sdk,source,record,expected=registered_fixture(tmp_path,monkeypatch)
    record['artifact_id']='f'*64
    sdk.metadata[record['drive_file_id']]['appProperties']['artifact_id']=record['artifact_id']
    with pytest.raises(RuntimeError,match='identity|provenance|artifact'):
        store.restore_registered_artifact(record,tmp_path/'fresh.bin',expected_provenance=expected)
    assert not sdk.download_calls


class ResponseFixture:
    def __init__(self, *, status=200, chunks=(b'explicit_offline_response',), headers=None):
        self.status_code, self.headers, self.reason = status, headers or {'Content-Type': 'application/json'}, 'fixture'
        self.chunks, self.closed = chunks, False
    def iter_content(self, chunk_size):
        assert chunk_size == 64 * 1024
        return iter(self.chunks)
    def close(self): self.closed = True


def transport_fixture(response=None, failure=None):
    from operations.drive import RequestsDriveTransport
    calls = []
    response = response or ResponseFixture()
    def request(method, uri, **kwargs):
        calls.append((method, uri, kwargs))
        if failure is not None: raise failure
        return response
    transport = object.__new__(RequestsDriveTransport)
    transport.session = SimpleNamespace(request=request)
    return transport, response, calls


def test_proxy_transport_returns_sdk_response_with_tls_no_redirects_or_hidden_retries():
    transport, response, calls = transport_fixture()
    result, content = transport.request('https://www.googleapis.com/drive/v3/files', method='GET')
    assert result.status == 200 and content == b'explicit_offline_response' and response.closed
    assert len(calls) == 1
    assert calls[0][2] == {'data': None, 'headers': {'Accept-Encoding': 'identity'},
        'timeout': (30, 60), 'allow_redirects': False, 'verify': True, 'stream': True}


@pytest.mark.parametrize('uri', [
    'http://www.googleapis.com/drive/v3/files', 'https://evil.example/drive/v3/files',
    'https://user@www.googleapis.com/drive/v3/files', 'https://www.googleapis.com:80/drive/v3/files',
    'https://www.googleapis.com/drive/v3/files#fragment',
])
def test_proxy_transport_rejects_unapproved_destinations_before_request(uri):
    transport, response, calls = transport_fixture()
    with pytest.raises(RuntimeError, match='approved Google HTTPS'):
        transport.request(uri)
    assert not calls


@pytest.mark.parametrize('status', [301, 302, 303, 307])
def test_proxy_transport_rejects_redirects_without_following_them(status):
    transport, response, calls = transport_fixture(ResponseFixture(status=status))
    with pytest.raises(RuntimeError, match='redirect'):
        transport.request('https://www.googleapis.com/drive/v3/files')
    assert response.closed and len(calls) == 1


def test_proxy_transport_preserves_resumable_upload_308_and_bounded_file_chunk():
    transport, response, calls = transport_fixture(ResponseFixture(status=308, chunks=()))
    result, content = transport.request('https://www.googleapis.com/upload/drive/v3/files', method='PUT',
        body=io.BytesIO(b'fixture'), headers={'Content-Length': '7'})
    assert result.status == 308 and content == b'' and calls[0][2]['data'] == b'fixture'
    assert response.closed


def test_proxy_transport_rejects_unbounded_stream_or_response_and_closes_response():
    from operations.drive import MAX_TRANSPORT_BYTES
    transport, response, calls = transport_fixture()
    for headers in ({}, {'Content-Length': str(MAX_TRANSPORT_BYTES + 1)}, {'Content-Length': 'bad'}):
        with pytest.raises(RuntimeError, match='bounded chunk'):
            transport.request('https://www.googleapis.com/upload/drive/v3/files', body=io.BytesIO(b'fixture'), headers=headers)
    assert not calls
    transport, response, calls = transport_fixture(ResponseFixture(chunks=(b'x' * MAX_TRANSPORT_BYTES, b'x')))
    with pytest.raises(RuntimeError, match='response exceeds'):
        transport.request('https://www.googleapis.com/drive/v3/files')
    assert response.closed and len(calls) == 1


@pytest.mark.parametrize('kind,expected', [('SSLError', RuntimeError), ('Timeout', TimeoutError), ('ConnectionError', ConnectionError)])
def test_proxy_transport_sanitizes_errors_and_does_not_retry(kind, expected):
    import requests
    failure = getattr(requests.exceptions, kind)('PUBLIC_FIXTURE_EXCEPTION_DETAIL')
    transport, response, calls = transport_fixture(failure=failure)
    with pytest.raises(expected) as caught:
        transport.request('https://www.googleapis.com/drive/v3/files')
    assert 'PUBLIC_FIXTURE_EXCEPTION_DETAIL' not in str(caught.value) and len(calls) == 1


def test_auth_request_rejects_token_redirects_including_upload_status(monkeypatch):
    from operations.drive import _BoundedAuthRequest
    request = object.__new__(_BoundedAuthRequest)
    request.session = transport_fixture(ResponseFixture(status=308))[0].session
    with pytest.raises(RuntimeError, match='redirect'):
        request('https://oauth2.googleapis.com/token', method='POST', body=b'public_fixture')


@pytest.mark.parametrize('status,content', [
    (503, b'{"error":"public_fixture"}'),
    (400, b'{"error":"temporarily_unavailable"}'),
])
def test_actual_google_auth_refresh_cannot_add_hidden_transient_retries(status,content):
    from google.oauth2.credentials import Credentials
    from operations.drive import _BoundedAuthRequest
    request = object.__new__(_BoundedAuthRequest)
    transport, response, calls = transport_fixture(ResponseFixture(status=status,chunks=(content,)))
    request.session = transport.session
    credentials = Credentials(token=None, refresh_token='explicit_public_fixture_refresh',
        token_uri='https://oauth2.googleapis.com/token', client_id='explicit_public_fixture_id',
        client_secret='explicit_public_fixture_secret', scopes=[SCOPE])
    with pytest.raises(ConnectionError, match='shared recovery'):
        credentials.refresh(request)
    assert len(calls) == 1 and response.closed


def test_proxy_env_selects_requests_transport_and_service_keeps_original_scopes(tmp_path, monkeypatch):
    from operations import drive
    token = tmp_path / 'private_fixture.json'
    token.write_text(json.dumps({'scopes': [SCOPE], 'refresh_token': 'fixture_refresh'}))
    credentials = SimpleNamespace(valid=True, scopes=[SCOPE], token='fixture', refresh_token='fixture_refresh')
    calls = install_oauth_sdk(monkeypatch, credentials)
    marker = object()
    monkeypatch.setattr(drive, '_proxy_aware_http', lambda cred: marker)
    assert drive_service(token) == 'fixture_service'
    assert calls['build'][0][1] == {'http': marker, 'cache_discovery': False, 'static_discovery': True}
    assert calls['loaded'][0][1] is None
    # Exercise selection independently of the offline OAuth SDK doubles.
    monkeypatch.undo()
    for name in ('HTTPS_PROXY', 'https_proxy', 'HTTP_PROXY', 'http_proxy', 'ALL_PROXY', 'all_proxy'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(drive, 'RequestsDriveTransport', lambda cred: marker)
    assert drive._proxy_aware_http(credentials) is None
    monkeypatch.setenv('HTTPS_PROXY', 'http://explicit_offline_proxy:8080')
    assert drive._proxy_aware_http(credentials) is marker
