"""Immutable HTTP provisioning into the standard, shared Hugging Face cache.

The Hub SDK performs its own retries. Provisioning therefore uses explicit TLS
requests, validates upstream Git/LFS object digests, and returns local-only
Transformers arguments. Snapshot symlinks and public-cache hardlinks share one
physical blob, including when tokenizer and weight revisions differ.
"""
from __future__ import annotations

import fnmatch
import hashlib
import http.client
import json
import os
import re
import shutil
import socket
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath

from shared_procedures.identity import content_hash
from shared_procedures.recovery_procedure import RecoveryManager, RetryableFailure
from shared_procedures.storage import atomic_write_json, file_lock, read_json
from operations.drive import file_hash

REVISION = re.compile(r"[0-9a-f]{40}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
TOKENIZER_FILES = ("tokenizer*", "vocab*", "merges*", "special_tokens_map.json", "added_tokens.json", "config.json", "generation_config.json")


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        if urllib.parse.urlsplit(new_url).scheme != "https":
            raise RuntimeError("Upstream download attempted a non-TLS redirect")
        redirected = super().redirect_request(request, response, code, message, headers, new_url)
        if redirected is not None and urllib.parse.urlsplit(new_url).netloc != urllib.parse.urlsplit(request.full_url).netloc:
            redirected.remove_header("Authorization")
        return redirected


_open_url = urllib.request.build_opener(_SafeRedirect()).open


def _relative_name(name: str) -> str:
    if not isinstance(name, str) or not name or "\\" in name:
        raise RuntimeError("Unsafe upstream asset path")
    value = PurePosixPath(name)
    if value.is_absolute() or any(part in (".", "..", ".git") for part in name.split("/")) or "" in name.split("/"):
        raise RuntimeError("Unsafe upstream asset path")
    return value.as_posix()


def _repository_name(repo_id: str, repo_type: str) -> str:
    if repo_type not in ("model", "dataset") or not isinstance(repo_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", repo_id):
        raise ValueError("Asset requires a valid Hugging Face model/dataset repository")
    return ("models" if repo_type == "model" else "datasets") + "--" + repo_id.replace("/", "--")


def _request(url: str, *, headers=None):
    selected = {"User-Agent": "persistent-illegibility-verified-provisioner/1", **(headers or {})}
    token = os.environ.get("HF_TOKEN")
    if token:
        selected["Authorization"] = "Bearer " + token
    request = urllib.request.Request(url, headers=selected)
    try:
        return _open_url(request, timeout=60)
    except urllib.error.HTTPError as error:
        if error.code in (408, 425, 429, 500, 502, 503, 504):
            retry_after = error.headers.get("Retry-After")
            try:
                seconds = float(retry_after) if retry_after is not None else None
            except ValueError:
                seconds = None
            raise RetryableFailure(f"Transient upstream HTTP {error.code}", retry_after=seconds) from error
        raise RuntimeError(f"Immutable upstream asset unavailable: HTTP {error.code}") from error
    except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.IncompleteRead, http.client.RemoteDisconnected) as error:
        # Certificate/configuration failures need correction, not repeated traffic.
        if isinstance(error, urllib.error.URLError) and isinstance(error.reason, Exception):
            import ssl
            if isinstance(error.reason, ssl.SSLError):
                raise RuntimeError("Upstream TLS validation failed") from error
        raise RetryableFailure("Transient upstream connection failure") from error


def _metadata(repo_id: str, repo_type: str, revision: str | None, *, manager: RecoveryManager) -> dict:
    plural = "models" if repo_type == "model" else "datasets"
    ref = "/revision/" + urllib.parse.quote(revision, safe="") if revision else ""
    url = f"https://huggingface.co/api/{plural}/{repo_id}{ref}?blobs=true"
    def operation():
        try:
            with _request(url) as response:
                result = json.load(response)
        except (http.client.IncompleteRead, TimeoutError, ConnectionError) as error:
            raise RetryableFailure("Upstream metadata connection interrupted") from error
        except (ValueError, UnicodeError) as error:
            raise RuntimeError("Invalid immutable upstream metadata") from error
        if not isinstance(result, dict) or not REVISION.fullmatch(result.get("sha", "")) or not isinstance(result.get("siblings"), list):
            raise RuntimeError("Upstream metadata lacks immutable commit/file identities")
        if revision and result["sha"] != revision:
            raise RuntimeError("Upstream resolved another immutable repository commit")
        return result
    return manager.run(f"metadata:{repo_type}:{repo_id}:{revision or 'freeze-head'}", operation)


def assets_from_configuration(config: dict) -> list[dict]:
    assets = []
    for key, model in config["models"].items():
        assets.append({"asset_id": f"model_{key}", "repo_id": model["id"], "repo_type": "model",
                       "revision": model["revision"], "tokenizer_revision": model["tokenizer_revision"],
                       "destination": f"models/{key}", "license_authorization_required": False,
                       "provisioner": "operations.assets.provision_asset", "expected_size_status": "query metadata before download"})
    for key, dataset in config["datasets"].items():
        assets.append({"asset_id": f"dataset_{key}", "repo_id": dataset["id"], "repo_type": "dataset",
                       "revision": dataset["revision"], "destination": f"datasets/{key}",
                       "license_authorization_required": key == "lmsys", "provisioner": "operations.assets.provision_asset",
                       "expected_size_status": "query metadata before download"})
    return assets


def resolve_revision(asset: dict, api=None, *, cache_root: str | Path | None = None) -> dict:
    """Freeze moving input only during discovery; production loaders use commits.

    ``api`` remains an explicit metadata fixture seam; the production branch
    performs bounded HTTP, without instantiating the Hub SDK.
    """
    result = dict(asset)
    _repository_name(result["repo_id"], result["repo_type"])
    if result.get("revision") is None:
        if api is not None:
            metadata = api.repo_info(result["repo_id"], repo_type=result["repo_type"], files_metadata=True)
            result["revision"] = metadata.sha
            result["size_bytes"] = sum(s.size or 0 for s in metadata.siblings)
        else:
            root = _cache_root(cache_root)
            metadata = _metadata(result["repo_id"], result["repo_type"], None, manager=RecoveryManager.for_worker(artifact_root=root))
            result["revision"] = metadata["sha"]
            result["size_bytes"] = sum(s.get("size", 0) for s in metadata["siblings"])
    if not isinstance(result.get("revision"), str) or not REVISION.fullmatch(result["revision"]):
        raise ValueError("Asset requires an immutable repository commit")
    if result["repo_type"] == "model":
        result["tokenizer_revision"] = result.get("tokenizer_revision") or result["revision"]
        if not isinstance(result["tokenizer_revision"], str) or not REVISION.fullmatch(result["tokenizer_revision"]):
            raise ValueError("Tokenizer requires an immutable repository commit")
    return result


def _manifest_hash(manifest: dict) -> str:
    if manifest.get("manifest_version") == 2:
        return content_hash({k: v for k, v in manifest.items() if k != "manifest_hash"})
    # Legacy public validation is kept for reading old inventories. Provisioning
    # never accepts this unbound-identity format as a reusable model asset.
    return content_hash(manifest.get("files", {}))


def validate_asset(path: str | Path) -> dict:
    path = Path(path)
    marker = path / "asset_manifest.json"
    if not marker.is_file():
        raise RuntimeError("Asset lacks a completed integrity manifest")
    manifest = read_json(marker)
    if not REVISION.fullmatch(manifest.get("revision", "")):
        raise RuntimeError("Asset manifest revision is not immutable")
    if manifest.get("manifest_hash") != _manifest_hash(manifest):
        raise RuntimeError("Asset manifest hash differs")
    if not isinstance(manifest.get("files"), dict) or not manifest["files"]:
        raise RuntimeError("Asset manifest contains no downloaded files")
    for relative, metadata in manifest["files"].items():
        relative = _relative_name(relative)
        file = path / relative
        if not file.resolve().is_relative_to(path.resolve()) or not file.is_file():
            raise RuntimeError("Asset file missing or escapes cache root")
        if file.stat().st_size != metadata["size"] or file_hash(file) != metadata["sha256"]:
            raise RuntimeError("Cached upstream asset checksum differs")
    return manifest


def _cache_root(value: str | Path | None) -> Path:
    return Path(value or os.environ.get("MODEL_CACHE_ROOT") or os.environ.get("HF_HOME") or "/workspace/model_cache").expanduser().resolve()


def _upstream_file(sibling: dict) -> dict:
    name = _relative_name(sibling.get("rfilename"))
    size = sibling.get("size")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise RuntimeError("Upstream file metadata lacks its exact byte size")
    lfs = sibling.get("lfs")
    if lfs is not None:
        digest = lfs.get("sha256") or lfs.get("oid")
        if not isinstance(digest, str) or not SHA256.fullmatch(digest) or lfs.get("size") != size:
            raise RuntimeError("Upstream LFS size/digest is invalid")
        algorithm = "sha256"
    else:
        digest = sibling.get("blobId")
        if not isinstance(digest, str) or not REVISION.fullmatch(digest):
            raise RuntimeError("Upstream Git blob digest is invalid")
        algorithm = "git_blob_sha1"
    return {"path": name, "size": size, "upstream_hash": digest, "upstream_algorithm": algorithm}


def _verify_upstream(path: Path, expected: dict) -> dict:
    if not path.is_file() or path.stat().st_size != expected["size"]:
        raise RuntimeError("Upstream blob byte size differs")
    sha = hashlib.sha256()
    git = hashlib.sha1(f"blob {expected['size']}\0".encode()) if expected["upstream_algorithm"] == "git_blob_sha1" else None
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
            if git is not None:
                git.update(chunk)
    actual = sha.hexdigest() if expected["upstream_algorithm"] == "sha256" else git.hexdigest()
    if actual != expected["upstream_hash"]:
        raise RuntimeError("Upstream Git/LFS object digest differs")
    return {**expected, "sha256": sha.hexdigest()}


def _download_blob(asset: dict, revision: str, expected: dict, blob: Path, manager: RecoveryManager) -> dict:
    if blob.exists():
        return _verify_upstream(blob, expected)
    prefix = "datasets/" if asset["repo_type"] == "dataset" else ""
    endpoint = "resolve" if expected["upstream_algorithm"] == "sha256" else "raw"
    name = urllib.parse.quote(expected["path"], safe="/")
    url = f"https://huggingface.co/{prefix}{asset['repo_id']}/{endpoint}/{revision}/{name}"
    partial = blob.with_name(blob.name + ".incomplete")
    blob.parent.mkdir(parents=True, exist_ok=True)
    def operation():
        if partial.exists() and partial.stat().st_size > expected["size"]:
            raise RuntimeError("Partial upstream download exceeds immutable size")
        offset = partial.stat().st_size if partial.exists() else 0
        if offset != expected["size"]:
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            try:
                with _request(url, headers=headers) as response:
                    status = response.status
                    if status not in (200, 206):
                        raise RuntimeError("Unexpected upstream download response")
                    if status == 206:
                        supplied = response.headers.get("Content-Range", "")
                        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", supplied)
                        if not match or int(match[1]) != offset or int(match[3]) != expected["size"]:
                            raise RuntimeError("Upstream range response does not match immutable file")
                    else:
                        offset = 0
                    with partial.open("ab" if offset else "wb") as output:
                        for chunk in iter(lambda: response.read(1024 * 1024), b""):
                            if output.tell() + len(chunk) > expected["size"]:
                                raise RuntimeError("Upstream download exceeds immutable byte size")
                            output.write(chunk)
                        output.flush()
                        os.fsync(output.fileno())
            except (TimeoutError, ConnectionError, http.client.IncompleteRead, http.client.RemoteDisconnected, socket.timeout) as error:
                raise RetryableFailure("Upstream blob transfer interrupted") from error
        if not partial.exists():
            # An empty Git blob is still a valid source artifact.
            partial.touch()
        if partial.stat().st_size < expected["size"]:
            raise RetryableFailure("Upstream blob transfer ended before immutable size")
        verified = _verify_upstream(partial, expected)
        os.replace(partial, blob)
        return verified
    return manager.run(f"blob:{asset['repo_type']}:{asset['repo_id']}:{revision}:{expected['path']}:{expected['upstream_hash']}", operation)


def _hardlink(blob: Path, destination: Path, expected: dict) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if not destination.is_file() or destination.is_symlink():
            raise RuntimeError("Verified asset payload path is not a regular file")
        _verify_upstream(destination, expected)
        return
    os.link(blob, destination)


def _snapshot_link(blob: Path, destination: Path, expected: dict) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        _verify_upstream(destination, expected)
        return
    descriptor, name = tempfile.mkstemp(prefix='.snapshot_publish_',dir=destination.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.unlink()
        temporary.symlink_to(os.path.relpath(blob, destination.parent))
        os.replace(temporary, destination)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def provision_asset(asset: dict, cache_root: str | Path, *, authorization: dict | None = None, offline: bool = False, tokenizer_only: bool = False) -> dict:
    if asset.get("license_authorization_required") and not (authorization or {}).get(asset["asset_id"], False):
        raise RuntimeError(f"Explicit source/license authorization required for {asset['asset_id']}")
    root = _cache_root(cache_root)
    if offline and not REVISION.fullmatch(asset.get("revision") or ""):
        raise RuntimeError("Offline provisioning requires a previously frozen immutable revision")
    asset = resolve_revision(asset, cache_root=root)
    if tokenizer_only and asset["repo_type"] != "model":
        raise ValueError("Tokenizer-only mode is applicable to models")
    variant = "tokenizer" if tokenizer_only else "full"
    selected_revision = asset.get("tokenizer_revision", asset["revision"]) if tokenizer_only else asset["revision"]
    repo = _repository_name(asset["repo_id"], asset["repo_type"])
    repository = root / "hub" / repo
    canonical = repository / "verified" / selected_revision / variant
    public = root / _relative_name(asset["destination"]) / asset["revision"] / variant
    if tokenizer_only and selected_revision != asset['revision']:
        public = public / selected_revision
    if not public.resolve().is_relative_to(root):
        raise RuntimeError("Asset destination escapes canonical cache root")
    root.mkdir(parents=True, exist_ok=True)
    with file_lock(repository / "provision.lock"):
        expected_identity = {"repo_id": asset["repo_id"], "repo_type": asset["repo_type"], "revision": selected_revision, "variant": variant}
        if (canonical / "asset_manifest.json").is_file():
            frozen = validate_asset(canonical)
            if frozen.get("manifest_version") != 2 or any(frozen.get(k) != v for k, v in expected_identity.items()):
                raise RuntimeError("Canonical asset cache identity mismatch")
        else:
            manager = RecoveryManager.for_worker(artifact_root=root)
            metadata_path = repository / "metadata" / f"{selected_revision}.json"
            if metadata_path.is_file():
                saved = read_json(metadata_path)
                if saved.get("identity") != {k: expected_identity[k] for k in ("repo_id", "repo_type", "revision")} or saved.get("metadata_hash") != content_hash(saved.get("metadata")):
                    raise RuntimeError("Cached upstream metadata identity/integrity differs")
                metadata = saved["metadata"]
            else:
                if offline:
                    raise RuntimeError("Requested offline asset metadata is not cached")
                metadata = _metadata(asset["repo_id"], asset["repo_type"], selected_revision, manager=manager)
                atomic_write_json(metadata_path, {"identity": {k: expected_identity[k] for k in ("repo_id", "repo_type", "revision")}, "metadata": metadata, "metadata_hash": content_hash(metadata)})
            if metadata.get("sha") != selected_revision:
                raise RuntimeError("Cached metadata commit identity differs")
            selected = [_upstream_file(s) for s in metadata["siblings"]
                        if not tokenizer_only or any(fnmatch.fnmatch(s.get("rfilename", ""), p) for p in TOKENIZER_FILES)]
            names = [f["path"] for f in selected]
            if not selected or len(set(names)) != len(names):
                raise RuntimeError("Upstream file inventory is empty or contains duplicate paths")
            if tokenizer_only and not any(n.startswith("tokenizer") or n.startswith("vocab") for n in names):
                raise RuntimeError("Repository lacks tokenizer payloads")
            missing_bytes = sum(f["size"] for f in selected if not (repository / "blobs" / f["upstream_hash"]).exists())
            if shutil.disk_usage(root).free < missing_bytes + 128 * 1024 * 1024:
                raise RuntimeError("Insufficient cache disk space for immutable upstream asset")
            files = {}
            for expected in selected:
                blob = repository / "blobs" / expected["upstream_hash"]
                if offline and not blob.is_file():
                    raise RuntimeError("Requested offline asset blob is not cached")
                verified = _download_blob(asset, selected_revision, expected, blob, manager)
                _hardlink(blob, canonical / expected["path"], expected)
                files[expected["path"]] = {k: v for k, v in verified.items() if k != "path"}
            frozen = {"manifest_version": 2, **expected_identity, "files": files,
                      "metadata_hash": content_hash(metadata), "size_bytes": sum(f["size"] for f in selected),
                      "upstream_integrity": "immutable_commit_and_git_or_lfs_object_digest"}
            frozen["manifest_hash"] = _manifest_hash(frozen)
            atomic_write_json(canonical / "asset_manifest.json", frozen)
            validate_asset(canonical)
        # Every process receives an actual standard Hub snapshot. This path is
        # rechecked even when a public inventory already exists.
        for name, expected in frozen["files"].items():
            blob = repository / "blobs" / expected["upstream_hash"]
            _verify_upstream(blob, expected)
            _snapshot_link(blob, repository / "snapshots" / selected_revision / name, expected)
        if (public / "asset_manifest.json").is_file():
            manifest = validate_asset(public)
            if manifest.get("manifest_version") != 2 or any(manifest.get(k) != v for k, v in asset.items()) or manifest.get("variant") != variant or manifest.get("canonical_manifest_hash") != frozen["manifest_hash"]:
                raise RuntimeError("Public asset cache identity mismatch")
            return manifest
        for name, expected in frozen["files"].items():
            _hardlink(repository / "blobs" / expected["upstream_hash"], public / name, expected)
        manifest = {**asset, "manifest_version": 2, "variant": variant, "selected_revision": selected_revision,
                    "files": frozen["files"], "canonical_manifest_hash": frozen["manifest_hash"],
                    "snapshot_relative": f"hub/{repo}/snapshots/{selected_revision}", "size_bytes": frozen["size_bytes"]}
        manifest["manifest_hash"] = _manifest_hash(manifest)
        atomic_write_json(public / "asset_manifest.json", manifest)
        return validate_asset(public)


def model_load_kwargs(model_id: str, revision: str, cache_root: str | Path | None = None, *, tokenizer_only: bool = False, offline: bool = False) -> dict:
    """Provision/verify one pinned runtime payload, then prohibit SDK networking."""
    root = _cache_root(cache_root)
    _repository_name(model_id, "model")
    if not isinstance(revision, str) or not REVISION.fullmatch(revision):
        raise ValueError("Runtime model/tokenizer loading requires an immutable commit")
    asset = {"asset_id": "runtime_" + model_id.replace("/", "--"), "repo_id": model_id, "repo_type": "model",
             "revision": revision, "tokenizer_revision": revision,
             "destination": "models/by_repository/" + model_id.replace("/", "--"), "license_authorization_required": False}
    provision_asset(asset, root, offline=offline, tokenizer_only=tokenizer_only)
    return {"cache_dir": str(root / "hub"), "local_files_only": True}
