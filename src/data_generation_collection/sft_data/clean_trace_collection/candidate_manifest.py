"""Authorized LMSYS extraction, exact normalized overlap and frozen ordering."""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from pathlib import Path

from shared_procedures.identity import content_hash, derive_seed
from shared_procedures.storage import atomic_write_json, read_json

NORMALIZATION_VERSION = "nfkc_casefold_whitespace_v1"
EXTRACTION_VERSION = "lmsys_first_user_turn_v1"
SOURCE = "lmsys/lmsys-chat-1m"


def normalize_prompt(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def validate_revision(revision: str) -> str:
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Source revision must be an immutable 40-character commit SHA")
    return revision


def input_token_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def prepare_candidate_manifest(rows: Iterable[Mapping], tokenizers: Mapping,
                               source_revision: str, evaluation_questions: Iterable[str],
                               *, authorized: bool, output_path=None) -> dict:
    validate_revision(source_revision)
    if not authorized:
        raise PermissionError("Authorized LMSYS source access is required")
    if set(tokenizers) != {"7b", "14b"}:
        raise ValueError("Both construction tokenizers are required for the input cap")
    overlaps = {normalize_prompt(q) for q in evaluation_questions}
    accepted, excluded, seen, source_ids = [], [], set(), set()
    for index, row in enumerate(rows):
        source_id = str(row.get("conversation_id", row.get("id", index)))
        example_id = f"lmsys_{source_id}"
        if source_id in source_ids:
            raise ValueError("Duplicate original source identity")
        source_ids.add(source_id)
        turns = row.get("conversation", [])
        turn = next((t for t in turns if t.get("role") == "user"), None)
        text = turn.get("content") if turn else None
        reason = None
        if str(row.get("language", "")).casefold() not in {"english", "en"}:
            reason = "non_english_source_label"
        elif not isinstance(text, str) or not text.strip():
            reason = "missing_first_user_turn"
        normalized = normalize_prompt(text) if isinstance(text, str) else None
        counts = {model: input_token_count(tokenizer, text) for model, tokenizer in tokenizers.items()} if text else {}
        if reason is None and any(n > 200 for n in counts.values()):
            reason = "input_exceeds_200_tokens"
        if reason is None and normalized in seen:
            reason = "normalized_duplicate"
        if reason is None and normalized in overlaps:
            reason = "detectable_evaluation_overlap"
        record = {"example_id": example_id, "source_id": source_id, "source_index": index,
                  "prompt": text, "source_dataset": SOURCE, "source_split": "train",
                  "source_revision": source_revision, "source_content_hash": content_hash(text or ""),
                  "deduplication_identity": content_hash(normalized or ""), "input_token_counts": counts,
                  "normalization_version": NORMALIZATION_VERSION, "extraction_version": EXTRACTION_VERSION}
        if reason:
            excluded.append({**record, "status": "excluded", "reason": reason})
        else:
            seen.add(normalized)
            accepted.append(record)
    # Prospective cryptographic shuffle independent of source streaming chunks.
    accepted.sort(key=lambda r: (derive_seed(42, "candidates", r["example_id"]), r["example_id"]))
    for position, record in enumerate(accepted):
        record.update(candidate_order=position, status="eligible")
    result = {"schema_version": "candidate_manifest_v1", "status": "verified", "seed": 42,
              "source_revision": source_revision, "source_dataset": SOURCE,
              "candidates": accepted, "exclusions": excluded,
              "evaluation_overlap_hash": content_hash(sorted(overlaps)),
              "content_hash": content_hash(accepted), "count": len(accepted)}
    if output_path:
        atomic_write_json(output_path, result)
        if read_json(output_path) != result:
            raise IOError("Candidate manifest readback failed")
    return result


def load_candidate_manifest(path) -> dict:
    manifest = read_json(path)
    candidates = manifest["candidates"]
    if manifest.get("status") != "verified" or manifest.get("content_hash") != content_hash(candidates):
        raise ValueError("Invalid candidate manifest integrity")
    ids = [r["example_id"] for r in candidates]
    if len(ids) != len(set(ids)) or [r["candidate_order"] for r in candidates] != list(range(len(ids))):
        raise ValueError("Candidate manifest IDs/order invalid")
    for r in candidates:
        if r["source_content_hash"] != content_hash(r["prompt"]) or any(n > 200 for n in r["input_token_counts"].values()):
            raise ValueError("Candidate prompt invariant failed")
    return manifest


def acquire_pinned_parquet(dataset: str, revision: str, *, configuration=None,
                            split=None, cache_root=None) -> list[dict]:
    """Download immutable official parquet/Arrow with upstream LFS/git-blob integrity.

    Requests has zero SDK retries here. Persistent bounded recovery surrounds
    this operation at the worker boundary, so no library default consumes an
    unrecorded retry budget. Verified completed shards survive failed siblings.
    """
    import hashlib
    import os
    import tempfile
    from urllib.parse import urlparse
    import requests
    from huggingface_hub import get_token
    validate_revision(revision)
    cache = Path(cache_root or os.environ.get("HF_HOME", "/workspace/illegibility_assets")) / "verified_sources" / content_hash(dataset) / revision
    cache.mkdir(parents=True, exist_ok=True)
    headers = {}
    token = get_token()
    if token:
        headers["Authorization"] = "Bearer " + token
    session = requests.Session()
    url = f"https://huggingface.co/api/datasets/{dataset}/tree/{revision}?recursive=true&expand=true"
    entries, visited = [], set()
    while url:
        if url in visited or urlparse(url).hostname != "huggingface.co":
            raise ValueError("Invalid official source pagination identity")
        visited.add(url)
        response = session.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        rows = response.json()
        if not isinstance(rows, list):
            raise ValueError("Official dataset tree returned invalid metadata")
        entries.extend(rows)
        url = response.links.get("next", {}).get("url")
    artifacts = []
    for entry in sorted(entries, key=lambda e: e["path"]):
        source_path = entry["path"]
        if entry.get("type") != "file" or not source_path.endswith((".parquet", ".arrow")):
            continue
        config = source_path.split("/")[0]
        source_split = "saved_dataset" if source_path.endswith(".arrow") else Path(source_path).stem.split("-", 1)[0]
        if configuration is not None and config != configuration or split is not None and source_split != split:
            continue
        local = cache / (content_hash(source_path) + Path(source_path).suffix)
        lfs = entry.get("lfs")
        expected_sha = lfs.get("oid") if lfs else None
        expected_size = lfs.get("size", entry.get("size")) if lfs else entry.get("size")
        def verify(path):
            body = path.read_bytes()
            if expected_size is not None and len(body) != expected_size:
                raise ValueError("Official dataset artifact byte count mismatch")
            if expected_sha:
                if content_hash(body) != expected_sha:
                    raise ValueError("Official dataset LFS checksum mismatch")
            else:
                git_hash = hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest()
                if git_hash != entry["oid"]:
                    raise ValueError("Official dataset Git blob checksum mismatch")
            if source_path.endswith(".parquet") and (not body.startswith(b"PAR1") or not body.endswith(b"PAR1")):
                raise ValueError("Official dataset download is not a parquet artifact")
            if source_path.endswith(".arrow") and not body.startswith((b"ARROW1", b"\xff\xff\xff\xff")):
                raise ValueError("Official dataset download is not an Arrow IPC artifact")
            return content_hash(body)
        if not local.exists():
            fd, temporary = tempfile.mkstemp(prefix=".source_", dir=cache)
            try:
                with os.fdopen(fd, "wb") as stream:
                    with session.get(f"https://huggingface.co/datasets/{dataset}/resolve/{revision}/{source_path}", headers=headers, stream=True, timeout=60) as download:
                        download.raise_for_status()
                        for chunk in download.iter_content(1024 * 1024):
                            stream.write(chunk)
                    stream.flush()
                    os.fsync(stream.fileno())
                verify(Path(temporary))
                os.replace(temporary, local)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        digest = verify(local)
        artifacts.append({"path": str(local), "source_path": source_path,
                          "configuration": config, "split": source_split, "format": Path(source_path).suffix[1:],
                          "source_revision": revision, "sha256": digest,
                          "upstream_lfs_sha256": expected_sha, "upstream_git_blob": entry["oid"]})
    if not artifacts:
        raise ValueError("Pinned official source has no requested parquet frame")
    atomic_write_json(cache / (content_hash([configuration, split]) + ".json"),
                      {"dataset": dataset, "revision": revision, "status": "verified", "artifacts": artifacts})
    return artifacts


def iter_pinned_parquet_rows(dataset: str, revision: str, *, configuration=None,
                              split=None, cache_root=None):
    import pyarrow.parquet as parquet
    import pyarrow.ipc as ipc
    artifacts = acquire_pinned_parquet(dataset, revision, configuration=configuration,
                                      split=split, cache_root=cache_root)
    for artifact in artifacts:
        batches = ipc.open_stream(artifact["path"]) if artifact["format"] == "arrow" else parquet.ParquetFile(artifact["path"]).iter_batches(batch_size=512)
        for batch in batches:
            for row in batch.to_pylist():
                yield {**row, "source_split": artifact["split"]}


# Keep the public loader's streaming contract while using checksummed shards.
def load_lmsys_source(revision: str, *, authorized: bool, streaming: bool = True,
                       cache_root=None):
    validate_revision(revision)
    if not authorized:
        raise PermissionError("Authorized LMSYS access is required; license consent is not inferred")
    rows = iter_pinned_parquet_rows(SOURCE, revision, split="train", cache_root=cache_root)
    return rows if streaming else list(rows)
