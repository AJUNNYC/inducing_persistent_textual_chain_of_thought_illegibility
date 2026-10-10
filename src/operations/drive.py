"""Restricted Drive OAuth and integrity-checked resumable artifact transport.

Only drive.file is requested. Client and token files stay outside artifacts.
"""
from __future__ import annotations
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from shared_procedures.recovery_procedure import RecoveryManager, RetryableFailure

SCOPE = "https://www.googleapis.com/auth/drive.file"
PROJECT_TITLE = "Inducing Persistent Textual Chain-of-Thought Illegibility"
INTENDED_ACCOUNT = "antoniojun.nyc@gmail.com"


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
    try:
        result=requests.get('https://oauth2.googleapis.com/tokeninfo',params={'access_token':credentials.token},timeout=30)
        if not result.ok: raise ValueError('Token introspection failed')
        scopes=result.json().get('scope','').split()
    except Exception:
        raise RuntimeError('Google token scope verification unavailable; authorization not accepted') from None
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
    if not credentials.valid:
        if not credentials.refresh_token:
            raise RuntimeError("OAuth refresh token unavailable; reauthorize")
        try:
            credentials.refresh(Request())
        except RefreshError:
            raise RuntimeError("Google OAuth authorization expired or was revoked; repeat drive-authorize") from None
        if credentials.granted_scopes is not None: _validate_scope_set(credentials.granted_scopes)
        _validate_live_scope(credentials)
        _private_json(path, json.loads(credentials.to_json()))
    _validate_live_scope(credentials)
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
        self.account_and_quota()
        if self.folder_id:
            value = self.service.files().get(fileId=self.folder_id, fields="id,name,mimeType,trashed").execute()
            if value.get("trashed") or value.get("name") != PROJECT_TITLE or value.get("mimeType") != "application/vnd.google-apps.folder":
                raise RuntimeError("GOOGLE_DRIVE_FOLDER_ID is not the canonical project folder")
        else:
            self.folder_id = self.create_folder(PROJECT_TITLE, root=True)
        self._verified_folders.add(self.folder_id)
        return self.folder_id

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
            parent = self.create_folder(part, parent)
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
        return remote

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
