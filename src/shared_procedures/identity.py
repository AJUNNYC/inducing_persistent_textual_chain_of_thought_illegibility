"""Versioned canonical identities, never Python's randomized hash."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any

IDENTITY_VERSION = 'sha256_canonical_json_v1'

def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')

def content_hash(value: Any) -> str:
    data = value if isinstance(value, bytes) else value.encode('utf-8') if isinstance(value, str) else canonical_json(value)
    return hashlib.sha256(data).hexdigest()

def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def derive_seed(*parts: Any) -> int:
    return int(content_hash({'version': 'seed_v1', 'parts': parts})[:8], 16)

def artifact_identity(kind: str, **dependencies: Any) -> str:
    return content_hash({'kind': kind, 'identity_version': IDENTITY_VERSION, 'dependencies': dependencies})
