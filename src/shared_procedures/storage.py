"""Locked, durable local publication; local fsync is not provider survival evidence."""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import fcntl
import json
import os
import tempfile
import time
from typing import Any, Iterator
from .identity import canonical_json, content_hash

class ArtifactIntegrityError(ValueError):
    pass

@contextmanager
def file_lock(path: str | Path, timeout: float = 10) -> Iterator[None]:
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open('a+b')
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f'writer lock unavailable: {lock_path}')
                time.sleep(min(0.05, max(0, deadline-time.monotonic())))
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()

def fsync_directory(path: str | Path) -> None:
    descriptor = os.open(Path(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

def atomic_write_bytes(path: str | Path, payload: bytes) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f'.{target.name}.', dir=target.parent)
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        fsync_directory(target.parent)
        if target.read_bytes() != payload:
            raise ArtifactIntegrityError(f'write verification failed: {target}')
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target

def atomic_write_json(path: str | Path, value: Any) -> Path:
    payload = canonical_json(value) + b'\n'
    return atomic_write_bytes(path, payload)

def read_json(path: str | Path) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ArtifactIntegrityError(f'duplicate JSON field: {key}')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=pairs,
                      parse_constant=lambda v: (_ for _ in ()).throw(ArtifactIntegrityError(f'nonfinite JSON: {v}')))

def publish_artifact(path: str | Path, data: Any, *, identity: dict, expected_count: int | None = None) -> dict:
    if expected_count is not None and len(data) != expected_count:
        raise ArtifactIntegrityError('artifact count mismatch')
    record = {'identity': identity, 'data': data, 'content_hash': content_hash(data), 'status': 'complete'}
    with file_lock(str(path)+'.lock'):
        target = Path(path)
        if target.exists() and read_json(target) != record:
            raise ArtifactIntegrityError('immutable artifact already exists with different contents')
        atomic_write_json(target, record)
        return verify_artifact(target, expected_identity=identity, expected_count=expected_count)

def verify_artifact(path: str | Path, *, expected_identity: dict | None = None, expected_count: int | None = None) -> dict:
    value = read_json(path)
    if value.get('status') != 'complete' or content_hash(value.get('data')) != value.get('content_hash'):
        raise ArtifactIntegrityError('artifact content/status invalid')
    if expected_identity is not None and value.get('identity') != expected_identity:
        raise ArtifactIntegrityError('artifact dependency identity mismatch')
    if expected_count is not None and len(value['data']) != expected_count:
        raise ArtifactIntegrityError('artifact count mismatch')
    return value

def append_jsonl(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(str(target)+'.lock'):
        with target.open('ab') as handle:
            handle.write(canonical_json(value)+b'\n')
            handle.flush()
            os.fsync(handle.fileno())
        fsync_directory(target.parent)

_ARTIFACT_HOOKS = []

def register_artifact_hook(callback) -> None:
    """Register durable replication callback; failures propagate to the orchestrator."""
    if callback not in _ARTIFACT_HOOKS:
        _ARTIFACT_HOOKS.append(callback)

def notify_artifact(path: str | Path, metadata: dict) -> None:
    source=Path(path)
    if source.is_dir():
        from .identity import file_hash
        inventory=metadata.get('files')
        if not isinstance(inventory,dict) or not inventory:
            raise ArtifactIntegrityError('directory replication requires a verified file inventory')
        files=[]
        for name,info in inventory.items():
            relative=Path(name)
            file=source/relative
            if relative.is_absolute() or '..' in relative.parts or file.is_symlink() or not file.is_file():
                raise ArtifactIntegrityError('unsafe or missing directory artifact payload')
            if file_hash(file)!=info.get('sha256') or file.stat().st_size!=info.get('size_bytes'):
                raise ArtifactIntegrityError('directory artifact changed before replication')
            files.append(file)
        # Replicate payload first, completion manifest last. Hook failures propagate;
        # the exact idempotent checkpoint save can retry its replication on re-entry.
        manifest=source/'manifest.json'
        if manifest.exists(): files.append(manifest)
    else:
        files=[source]
    for file in files:
        for callback in tuple(_ARTIFACT_HOOKS):
            callback(file,metadata)

def unregister_artifact_hook(callback) -> None:
    if callback in _ARTIFACT_HOOKS:
        _ARTIFACT_HOOKS.remove(callback)

@contextmanager
def artifact_hook(callback):
    register_artifact_hook(callback)
    try:
        yield
    finally:
        unregister_artifact_hook(callback)
