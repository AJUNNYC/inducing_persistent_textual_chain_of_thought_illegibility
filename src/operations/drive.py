"""Restricted Drive OAuth and integrity-checked resumable artifact transport.

Only drive.file is requested. Client and token files stay outside artifacts.
"""
from __future__ import annotations
import hashlib
import json
import os
import re
import tempfile
import time
import threading
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit
from shared_procedures.recovery_procedure import RecoveryManager, RetryableFailure

SCOPE = "https://www.googleapis.com/auth/drive.file"
PROJECT_TITLE = "Inducing Persistent Textual Chain-of-Thought Illegibility"
INTENDED_ACCOUNT = "antoniojun.nyc@gmail.com"
GOOGLE_TRANSPORT_HOSTS = {"www.googleapis.com", "oauth2.googleapis.com", "accounts.google.com"}
MAX_TRANSPORT_BYTES = 9 * 1024 * 1024  # Accommodate each fixed 8-MiB media chunk.


def _checked_google_uri(uri):
    try:
        value = urlsplit(uri)
        accepted = (value.scheme == "https" and value.hostname in GOOGLE_TRANSPORT_HOSTS
                    and value.username is None and value.password is None and not value.fragment
                    and value.port in (None, 443))
    except (TypeError, ValueError):
        accepted = False
    if not accepted:
        raise RuntimeError("Drive transport requires an approved Google HTTPS destination")


def _bounded_request_body(body, headers):
    if hasattr(body, "read"):
        try:
            length = next(int(value) for key, value in headers.items() if key.lower() == "content-length")
        except (ValueError, StopIteration):
            raise RuntimeError("Drive streaming request lacks a bounded chunk length") from None
        if not 0 <= length <= MAX_TRANSPORT_BYTES:
            raise RuntimeError("Drive streaming request lacks a bounded chunk length")
        body = body.read(length)
        if len(body) != length:
            raise RuntimeError("Drive streaming chunk length differs")
    if body is not None and (not isinstance(body, (str, bytes))
        or len(body.encode('utf-8') if isinstance(body, str) else body) > MAX_TRANSPORT_BYTES):
        raise RuntimeError("Drive request body exceeds the bounded chunk size")
    return body


def _google_request(session, uri, method, body, headers, *, resumable=False):
    """One TLS-verified request; retries belong to the shared recovery manager."""
    import requests
    _checked_google_uri(uri)
    selected = dict(headers or {})
    selected['Accept-Encoding'] = 'identity'
    body = _bounded_request_body(body, selected)
    response = None
    try:
        response = session.request(method, uri, data=body, headers=selected,
            timeout=(30, 60), allow_redirects=False, verify=True, stream=True)
        if 300 <= response.status_code < 400 and not (resumable and response.status_code == 308):
            raise RuntimeError("Unexpected Google transport redirect")
        content = bytearray()
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if len(content) + len(chunk) > MAX_TRANSPORT_BYTES:
                raise RuntimeError("Google transport response exceeds the bounded chunk size")
            content.extend(chunk)
        return response.status_code, dict(response.headers), response.reason, bytes(content)
    except requests.exceptions.SSLError:
        raise RuntimeError("Google Drive TLS validation failed") from None
    except requests.exceptions.Timeout:
        raise TimeoutError("Google Drive HTTPS transport timed out") from None
    except requests.exceptions.ConnectionError:
        raise ConnectionError("Google Drive HTTPS transport unavailable") from None
    finally:
        if response is not None:
            response.close()


class _BoundedAuthRequest:
    def __init__(self):
        import requests
        self.session = requests.Session()

    def __call__(self, url, method="GET", body=None, headers=None, timeout=60, **kwargs):
        status, response_headers, _, content = _google_request(self.session, url, method, body, headers)
        # Google Auth otherwise retries transient token responses internally.
        # Raising before its response loop leaves all retries to shared recovery.
        retryable = status in (408, 429, 500, 502, 503, 504)
        if status != 200:
            try:
                document = json.loads(content)
                if isinstance(document, dict):
                    retryable = retryable or any(document.get(key) in
                        {'internal_failure', 'server_error', 'temporarily_unavailable'}
                        for key in ('error', 'error_description'))
            except (ValueError, TypeError):
                pass
        if retryable:
            raise ConnectionError("Google OAuth endpoint unavailable; shared recovery must retry")
        return SimpleNamespace(status=status, data=content, headers=response_headers)


class RequestsDriveTransport:
    """googleapiclient HTTP interface honoring standard HTTPS proxy settings."""
    def __init__(self, credentials):
        from google.auth.transport.requests import AuthorizedSession
        self.credentials = credentials
        self.auth_request = _BoundedAuthRequest()
        self.session = AuthorizedSession(credentials, max_refresh_attempts=0,
            refresh_timeout=60, auth_request=self.auth_request)
        self.timeout = 60

    def request(self, uri, method="GET", body=None, headers=None, redirections=5,
                connection_type=None, **kwargs):
        import httplib2
        status, response_headers, reason, content = _google_request(
            self.session, uri, method, body, headers, resumable=True)
        metadata = {key.lower(): value for key, value in response_headers.items()}
        metadata['status'] = str(status)
        result = httplib2.Response(metadata)
        result.reason = reason
        return result, content

    def close(self):
        self.session.close()
        self.auth_request.session.close()


def _proxy_aware_http(credentials):
    if any(os.environ.get(name) for name in ('HTTPS_PROXY', 'https_proxy', 'HTTP_PROXY', 'http_proxy', 'ALL_PROXY', 'all_proxy')):
        return RequestsDriveTransport(credentials)
    return None


def _private_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".oauth_")
    try:
        with os.fdopen(fd, "w") as stream:
            os.fchmod(stream.fileno(), 0o600)
            json.dump(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _validate_scope_set(scopes) -> None:
    if not isinstance(scopes, (list, tuple, set)) or set(scopes) != {SCOPE}:
        raise RuntimeError('OAuth granted scopes must contain exactly drive.file; reauthorize the existing desktop client')


def _validate_live_scope(credentials) -> None:
    import requests
    # Do not log request/response exceptions: the token is a confidential argument.
    result = None
    try:
        result=requests.get('https://oauth2.googleapis.com/tokeninfo',params={'access_token':credentials.token},timeout=30,
                            allow_redirects=False,verify=True)
        if result.status_code != 200: raise ValueError('Token introspection failed')
        scopes=result.json().get('scope','').split()
    except Exception:
        raise RuntimeError('Google token scope verification unavailable; authorization not accepted') from None
    finally:
        if result is not None:
            try: result.close()
            except Exception: pass
    _validate_scope_set(scopes)


def authorize(client_file: str | Path, token_file: str | Path, port: int = 8765) -> None:
    from google_auth_oauthlib.flow import InstalledAppFlow
    flow = InstalledAppFlow.from_client_secrets_file(str(client_file), scopes=[SCOPE])
    credentials = flow.run_local_server(host="localhost", port=port, open_browser=True,
                                       authorization_prompt_message="Complete Google consent using the authorization URL: {url}",
                                       access_type="offline", prompt="consent", login_hint=INTENDED_ACCOUNT)
    if not credentials.refresh_token:
        raise RuntimeError("Offline refresh authorization was not granted; repeat consent")
    _validate_live_scope(credentials)
    _private_json(Path(token_file), json.loads(credentials.to_json()))


def drive_service(token_file: str | Path):
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from google.auth.exceptions import RefreshError
    from googleapiclient.discovery import build
    path = Path(token_file)
    if not path.is_file():
        raise RuntimeError("GOOGLE_OAUTH_TOKEN_FILE missing; run drive-authorize with the existing desktop OAuth client")
    saved=json.loads(path.read_text())
    _validate_scope_set(saved.get('scopes'))
    credentials = Credentials.from_authorized_user_file(str(path))
    if not credentials.has_scopes([SCOPE]):
        raise RuntimeError("Existing OAuth credentials lack drive.file scope; reauthorize")
    http = _proxy_aware_http(credentials)
    if not credentials.valid:
        if not credentials.refresh_token:
            raise RuntimeError("OAuth refresh token unavailable; reauthorize")
        try:
            credentials.refresh(http.auth_request if http is not None else Request())
        except RefreshError:
            raise RuntimeError("Google OAuth authorization expired or was revoked; repeat drive-authorize") from None
        if credentials.granted_scopes is not None: _validate_scope_set(credentials.granted_scopes)
        _validate_live_scope(credentials)
        _private_json(path, json.loads(credentials.to_json()))
    _validate_live_scope(credentials)
    if http is not None:
        return build("drive", "v3", http=http, cache_discovery=False, static_discovery=True)
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _query_literal(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


class DriveStore:
    """One serialized caller per local registry prevents duplicate publication."""
    def __init__(self, service, folder_id: str | None = None, *, state_root: str | Path | None = None):
        self.service = service
        self.folder_id = folder_id
        self._verified_folders = set()
        self._path_cache = {}
        self._path_lock = threading.RLock()
        self._project_verified = False
        root = Path(state_root or Path(os.environ.get("ARTIFACT_ROOT", "artifacts")) / "state" / "drive")
        self.recovery = RecoveryManager(root / "retries.json") if state_root is not None else RecoveryManager.for_worker()

    @staticmethod
    def _retryable(error):
        status = getattr(getattr(error, "resp", None), "status", None)
        return status in (408, 429, 500, 502, 503, 504) or isinstance(error, (TimeoutError, ConnectionError))

    def account_and_quota(self) -> dict:
        value = self.service.about().get(fields="user(emailAddress),storageQuota").execute()
        email = value.get("user", {}).get("emailAddress")
        if email != INTENDED_ACCOUNT:
            raise RuntimeError("Authorized Google Drive account differs from the designated account")
        return value

    def _list(self, query: str, fields: str = "id,name,mimeType,size,md5Checksum,appProperties,parents") -> list[dict]:
        found, token = [], None
        while True:
            result = self.service.files().list(q=query, fields=f"nextPageToken,files({fields})",
                                               pageSize=1000, pageToken=token).execute()
            found.extend(result.get("files", []))
            token = result.get("nextPageToken")
            if not token:
                return found

    def create_folder(self, name: str, parent_id: str | None = None, *, root: bool = False) -> str:
        parent = parent_id or ("root" if root else self.folder_id)
        if not parent:
            raise RuntimeError("Project folder has not been initialized")
        query = f"trashed = false and mimeType = 'application/vnd.google-apps.folder' and name = '{_query_literal(name)}' and '{_query_literal(parent)}' in parents"
        found = self._list(query, "id,name")
        if len(found) > 1:
            raise RuntimeError("Ambiguous duplicate Drive folders; supply canonical folder ID")
        if found:
            return found[0]["id"]
        return self.service.files().create(body={"name": name, "mimeType": "application/vnd.google-apps.folder", "parents": [parent]}, fields="id").execute()["id"]

    def create_project_folder(self) -> str:
        if self._project_verified:
            return self.folder_id
        self.account_and_quota()
        if self.folder_id:
            value = self.service.files().get(fileId=self.folder_id, fields="id,name,mimeType,trashed").execute()
            if value.get("trashed") or value.get("name") != PROJECT_TITLE or value.get("mimeType") != "application/vnd.google-apps.folder":
                raise RuntimeError("GOOGLE_DRIVE_FOLDER_ID is not the canonical project folder")
        else:
            self.folder_id = self.create_folder(PROJECT_TITLE, root=True)
        self._verified_folders.add(self.folder_id)
        self._project_verified = True
        return self.folder_id

    def fork(self):
        """Independent HTTP transport; shared verified paths and retry ledger."""
        token = os.environ.get('GOOGLE_OAUTH_TOKEN_FILE')
        if not token:
            raise RuntimeError('Parallel Drive publication requires the private OAuth file')
        selected = DriveStore(drive_service(token), self.folder_id)
        selected.recovery = self.recovery
        selected._verified_folders = self._verified_folders
        selected._path_cache, selected._path_lock = self._path_cache, self._path_lock
        selected._project_verified = self._project_verified
        return selected

    def verify_folder_ancestry(self, folder_id: str) -> None:
        if not self.folder_id: raise RuntimeError("Canonical project folder is not initialized")
        current=folder_id;seen=set()
        while current not in self._verified_folders:
            if current in seen: raise RuntimeError("Drive folder ancestry contains a cycle")
            seen.add(current)
            value=self.service.files().get(fileId=current,fields="id,mimeType,parents,trashed").execute()
            if value.get("trashed") or value.get("mimeType")!="application/vnd.google-apps.folder" or len(value.get("parents",[]))!=1:
                raise RuntimeError("Drive destination is outside the canonical project")
            current=value["parents"][0]
            if current=="root": raise RuntimeError("Drive destination is outside the canonical project")
        self._verified_folders.update(seen)

    def ensure_path(self, parts: list[str]) -> str:
        parent = self.folder_id or self.create_project_folder()
        for part in parts:
            if not part or part in (".", "..") or "/" in part:
                raise ValueError("Invalid canonical Drive path component")
            with self._path_lock:
                key = (parent, part)
                if key not in self._path_cache:
                    self._path_cache[key] = self.create_folder(part, parent)
                    self._verified_folders.add(self._path_cache[key])
                parent = self._path_cache[key]
        return parent

    def list_artifacts(self, folder_id: str | None = None) -> list[dict]:
        parent = folder_id or self.folder_id
        if not parent:
            raise RuntimeError("Project folder has not been initialized")
        return self._list(f"trashed = false and '{_query_literal(parent)}' in parents")

    def resolve_artifact_id(self, artifact_id: str) -> dict | None:
        if not self.folder_id:
            raise RuntimeError("Project folder must be verified before artifact lookup")
        query = f"trashed = false and appProperties has {{ key='artifact_id' and value='{_query_literal(artifact_id)}' }}"
        query += f" and appProperties has {{ key='project_id' and value='{_query_literal(self.folder_id)}' }}"
        found = self._list(query)
        if len(found) > 1:
            raise RuntimeError("Duplicate artifact identity in Drive")
        return found[0] if found else None

    def upload_artifact(self, path: str | Path, *, artifact_id: str, folder_id: str, config_hash: str) -> dict:
        from googleapiclient.http import MediaFileUpload
        if not self.folder_id:
            raise RuntimeError("Project folder must be verified before upload")
        worker_id = os.environ.get('ILLEGIBILITY_WORKER_ID','coordinator')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}',worker_id):
            raise RuntimeError('Drive publication requires a stable worker identity')
        self.verify_folder_ancestry(folder_id)
        path = Path(path)
        checksum, size = file_hash(path), path.stat().st_size
        existing = self.resolve_artifact_id(artifact_id)
        if existing:
            props = existing.get("appProperties", {})
            if existing.get("parents") != [folder_id] or props.get("config_hash") != config_hash or props.get("sha256") != checksum:
                raise RuntimeError("Artifact identity collides with incompatible provenance")
            if props.get("upload_status") == "verified":
                self.verify_artifact(existing["id"], path)
                return existing
        media = MediaFileUpload(str(path), resumable=True, chunksize=8 * 1024 * 1024)
        properties = {"artifact_id": artifact_id, "sha256": checksum, "config_hash": config_hash,
                      "project_id": self.folder_id, "upload_status": "pending"}
        if existing:
            request = self.service.files().update(fileId=existing["id"], body={"appProperties": properties}, media_body=media,
                                                  fields="id,name,size,md5Checksum,appProperties,parents")
        else:
            # A restart resolves this same pending ID and repairs it instead of creating a duplicate.
            request = self.service.files().create(body={"name": path.name, "parents": [folder_id], "appProperties": properties},
                                                  media_body=media, fields="id,name,size,md5Checksum,appProperties,parents")
        response = None
        def transfer():
            nonlocal response
            while response is None:
                try:
                    _, response = request.next_chunk(num_retries=0)
                except Exception as error:
                    if self._retryable(error):
                        raise RetryableFailure("Recoverable Drive upload transport failure") from None
                    raise
            return response
        response = self.recovery.run("upload_"+artifact_id, transfer)
        self.verify_artifact(response["id"], path)
        # Publish the actual indexed retry total only after this operation has
        # completed. A lost native collection worker can retain that known floor
        # without uploading a growing SQLite ledger after every raw response.
        with self.recovery._database() as connection:
            retry_floor = connection.execute('SELECT retries FROM worker WHERE singleton=1').fetchone()[0]
            operation_retries = connection.execute('SELECT retries FROM operations WHERE operation_id=?',
                ('upload_'+artifact_id,)).fetchone()[0]
        properties['worker_id'] = worker_id
        properties['worker_retry_floor'] = str(retry_floor)
        properties['operation_retries'] = str(operation_retries)
        properties["upload_status"] = "verified"
        self.service.files().update(fileId=response["id"], body={"appProperties": properties}, fields="id").execute()
        return {**response, "appProperties": properties}

    def verify_registered_artifact(self, record: dict) -> dict:
        remote = self.service.files().get(fileId=record["drive_file_id"], fields="id,size,md5Checksum,appProperties,parents,trashed").execute()
        self.verify_folder_ancestry(record["drive_folder_id"])
        props = remote.get("appProperties", {})
        if remote.get("trashed") or remote.get("parents") != [record["drive_folder_id"]] or int(remote.get("size", -1)) != record["size"]:
            raise RuntimeError("Remote registered artifact is missing or changed")
        if remote.get("md5Checksum") != record["md5"] or props.get("sha256") != record["sha256"] or props.get("config_hash") != record["configuration_hash"] or props.get("project_id") != self.folder_id:
            raise RuntimeError("Remote registered artifact integrity/provenance differs")
        if remote.get("id") != record["drive_file_id"] or props.get("artifact_id") != record["artifact_id"] or props.get("upload_status") != "verified":
            raise RuntimeError("Remote registered artifact identity/publication differs")
        return remote

    def restore_registered_artifact(self, record: dict, destination: str | Path, *, expected_provenance: dict) -> Path:
        """Recover from Drive into a new path, without consulting any local copy.

        The caller supplies a separately preserved registry record and expected
        immutable provenance. A VERIFIED registry flag alone supplies no proof:
        current remote metadata and the newly downloaded complete bytes are checked.
        """
        from operations.artifacts import validate_registered_record
        validate_registered_record(record)
        critical = {"configuration_hash", "source_code_commit", "dataset_revision", "model_revision", "seeds", "model"}
        if (not isinstance(record, dict) or not isinstance(expected_provenance, dict)
            or critical - expected_provenance.keys()
            or any(expected_provenance.get(key) in (None, "", []) for key in critical)
            or any(record.get(key) != value for key, value in expected_provenance.items())):
            raise RuntimeError("Drive recovery requires matching complete critical provenance")
        if (record.get("sync_status") != "VERIFIED" or record.get("completion_status") != "COMPLETED"
            or not record.get("drive_file_id") or not record.get("drive_folder_id")):
            raise RuntimeError("Drive recovery requires a complete registered remote copy")
        destination = Path(destination)
        if destination.exists() or destination.is_symlink():
            raise RuntimeError("Drive recovery destination must be a new path")
        self.verify_registered_artifact(record)
        remote = self.resolve_artifact_id(record["artifact_id"])
        if remote is None or remote.get("id") != record["drive_file_id"]:
            raise RuntimeError("Drive recovery artifact lookup differs from the registry")
        try:
            restored = self.download_artifact(record["artifact_id"], destination,
                                              config_hash=record["configuration_hash"])
            if (restored != destination or not destination.is_file() or destination.is_symlink()
                or destination.stat().st_size != record["size"]
                or file_hash(destination) != record["sha256"] or file_hash(destination, "md5") != record["md5"]):
                raise RuntimeError("Drive recovery complete-file integrity differs from the registry")
            self.verify_registered_artifact(record)
            return destination
        except Exception:
            destination.unlink(missing_ok=True)
            raise

    def verify_artifact(self, file_id: str, local_path: str | Path) -> dict:
        path = Path(local_path)
        remote = self.service.files().get(fileId=file_id, fields="id,size,md5Checksum,appProperties,parents").execute()
        if int(remote.get("size", -1)) != path.stat().st_size or remote.get("md5Checksum") != file_hash(path, "md5"):
            raise RuntimeError("Drive artifact size/checksum mismatch")
        if self.folder_id and remote.get("appProperties", {}).get("project_id") != self.folder_id:
            raise RuntimeError("Drive artifact is outside the canonical project")
        if remote.get("appProperties", {}).get("sha256") != file_hash(path):
            raise RuntimeError("Drive artifact provenance checksum mismatch")
        return remote

    def download_artifact(self, artifact_id: str, destination: str | Path, *, config_hash: str | None = None) -> Path:
        from googleapiclient.http import MediaIoBaseDownload
        remote = self.resolve_artifact_id(artifact_id)
        if not remote or (config_hash and remote.get("appProperties", {}).get("config_hash") != config_hash):
            raise RuntimeError("Compatible artifact not found")
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            self.verify_artifact(remote["id"], destination)
            return destination
        fd, temporary = tempfile.mkstemp(dir=destination.parent, prefix=".download_")
        try:
            with os.fdopen(fd, "wb") as stream:
                download = MediaIoBaseDownload(stream, self.service.files().get_media(fileId=remote["id"]), chunksize=8 * 1024 * 1024)
                done = False
                def transfer():
                    nonlocal done
                    while not done:
                        try:
                            _, done = download.next_chunk(num_retries=0)
                        except Exception as error:
                            if self._retryable(error):
                                raise RetryableFailure("Recoverable Drive download transport failure") from None
                            raise
                self.recovery.run("download_"+artifact_id, transfer)
                stream.flush()
                os.fsync(stream.fileno())
            self.verify_artifact(remote["id"], temporary)
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return destination

    def roundtrip(self, folder_id: str, config_hash: str) -> dict:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "connectivity_probe.bin"
            source.write_bytes(os.urandom(128))
            artifact_id = hashlib.sha256(source.read_bytes()).hexdigest()
            remote = self.upload_artifact(source, artifact_id=artifact_id, folder_id=folder_id, config_hash=config_hash)
            restored = self.download_artifact(artifact_id, Path(temporary) / "restored.bin", config_hash=config_hash)
            if restored.read_bytes() != source.read_bytes():
                raise RuntimeError("Drive roundtrip differs")
            self.service.files().delete(fileId=remote["id"]).execute()
            return {"status": "VERIFIED", "bytes": 128, "scope": "authenticated upload/download/checksum/delete only"}
