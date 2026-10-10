"""Canonical permanent-artifact registry and compatibility-aware retrieval."""
from __future__ import annotations
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from shared_procedures.identity import content_hash
from shared_procedures.storage import atomic_write_json, read_json, file_lock
from operations.drive import DriveStore, file_hash

REQUIRED_PROVENANCE = {"experiment_id", "branch", "model", "run_id", "stage", "producer",
                       "source_code_commit", "configuration_hash", "dataset_revision", "model_revision", "seeds"}
SECRET_SUFFIXES = {".pem", ".key"}


class ArtifactService:
    def __init__(self, root: str | Path, registry: str | Path, drive: DriveStore | None = None):
        self.root = Path(root).resolve()
        self.registry = Path(registry)
        self.drive = drive
        self.registry.parent.mkdir(parents=True, exist_ok=True)
        self.database = self.registry.with_suffix('.sqlite3')
        with file_lock(str(self.registry) + '.init.lock'):
            existed = self.database.exists()
            with self._connect() as db:
                db.execute("CREATE TABLE IF NOT EXISTS artifacts (artifact_id TEXT PRIMARY KEY, local_path TEXT NOT NULL, sha256 TEXT NOT NULL, sync_status TEXT NOT NULL, record TEXT NOT NULL)")
                db.execute("CREATE INDEX IF NOT EXISTS pending_artifacts ON artifacts(sync_status)")
                db.execute("CREATE INDEX IF NOT EXISTS local_artifacts ON artifacts(local_path,sha256)")
                if not existed and self.registry.exists():
                    legacy = read_json(self.registry)
                    for record in legacy['artifacts'].values():
                        self._put(db, record)

    def _connect(self):
        db = sqlite3.connect(self.database, timeout=30)
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('PRAGMA synchronous=FULL')
        return db

    @staticmethod
    def _put(db, record):
        db.execute("INSERT INTO artifacts VALUES (?,?,?,?,?) ON CONFLICT(artifact_id) DO UPDATE SET local_path=excluded.local_path,sha256=excluded.sha256,sync_status=excluded.sync_status,record=excluded.record",
                   (record['artifact_id'], record['local_path'], record['sha256'], record['sync_status'], json.dumps(record, sort_keys=True)))

    def _get(self, artifact_id):
        with self._connect() as db:
            row = db.execute('SELECT record FROM artifacts WHERE artifact_id=?', (artifact_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def _read(self) -> dict:
        # Materialization is reserved for explicit inventories, never raw-file hooks.
        return {"version": 2, "artifacts": {r['artifact_id']: r for r in self.list_artifacts()}}

    def export_manifest(self) -> Path:
        atomic_write_json(self.registry, self._read())
        return self.registry

    def _safe_path(self, path: str | Path) -> Path:
        path = Path(path).resolve()
        if not path.is_file() or not path.is_relative_to(self.root):
            raise ValueError("Artifact must be a regular file below the declared artifact root")
        if path.suffix in SECRET_SUFFIXES or any(p.startswith(".env") or p in (".secrets", ".git") or "oauth" in p.lower() or "token" == p.lower() for p in path.parts):
            raise ValueError("Credential/private paths cannot become artifacts")
        return path

    def register(self, path: str | Path, provenance: dict, *, retention: str = "required") -> dict:
        path = self._safe_path(path)
        if REQUIRED_PROVENANCE - provenance.keys():
            raise ValueError(f"Missing artifact provenance names: {sorted(REQUIRED_PROVENANCE - provenance.keys())}")
        for key in REQUIRED_PROVENANCE - {"seeds"}:
            if not isinstance(provenance[key], str) or not provenance[key]:
                raise ValueError(f"Unresolved artifact identity: {key}")
        for key, lengths in (("source_code_commit", (40,)), ("configuration_hash", (64,)), ("dataset_revision", (40,64)), ("model_revision", (40,64))):
            if not re.fullmatch("[0-9a-f]+", provenance[key]) or len(provenance[key]) not in lengths:
                raise ValueError(f"Artifact {key} is not an immutable identity")
        if not isinstance(provenance["seeds"], list) or not provenance["seeds"] or any(type(seed) is not int for seed in provenance["seeds"]):
            raise ValueError("Artifact seeds must be resolved integers")
        identity = {"provenance": provenance, "sha256": file_hash(path), "path": path.relative_to(self.root).as_posix()}
        artifact_id = content_hash(identity)
        record = {**provenance, "artifact_id": artifact_id, "local_path": str(path), "relative_path": identity["path"],
                  "sha256": identity["sha256"], "md5": file_hash(path, "md5"), "size": path.stat().st_size,
                  "created_at": datetime.now(timezone.utc).isoformat(), "completion_status": "COMPLETED",
                  "validation_status": "content_hashed", "sync_status": "PENDING", "retention": retention,
                  "drive_file_id": None, "drive_folder_id": None}
        with self._connect() as db:
            existing = db.execute('SELECT record FROM artifacts WHERE artifact_id=?', (artifact_id,)).fetchone()
            if existing:
                return json.loads(existing[0])
            self._put(db, record)
        return record

    def list_artifacts(self, **filters) -> list[dict]:
        indexed = {'artifact_id', 'local_path', 'sha256', 'sync_status'}
        clauses = [f'{key}=?' for key in filters if key in indexed]
        values = [value for key,value in filters.items() if key in indexed]
        query = 'SELECT record FROM artifacts' + (' WHERE ' + ' AND '.join(clauses) if clauses else '') + ' ORDER BY artifact_id'
        with self._connect() as db:
            records = [json.loads(row[0]) for row in db.execute(query, values)]
        return [r for r in records if all(r.get(k) == v for k,v in filters.items())]

    def sync_artifacts(self, *, verify_existing: bool = False) -> dict:
        if self.drive is None:
            with self._connect() as db:
                pending = db.execute("SELECT COUNT(*) FROM artifacts WHERE sync_status!='VERIFIED'").fetchone()[0]
            return {"status": "BLOCKED", "reason": "Google Drive authorization is required", "pending": pending}
        self.drive.create_project_folder()
        completed, failed = [], []
        # A cross-process publication lock prevents duplicate remote creation. Readers
        # remain independent and each verified upload is committed immediately.
        with file_lock(str(self.registry) + '.sync.lock'):
            records = self.list_artifacts() if verify_existing else self.list_artifacts(sync_status='PENDING') + self.list_artifacts(sync_status='FAILED')
            for record in records:
                artifact_id = record['artifact_id']
                try:
                    if record['sync_status'] == 'VERIFIED':
                        self.drive.verify_registered_artifact(record)
                    else:
                        path = self._safe_path(record['local_path'])
                        if file_hash(path) != record['sha256'] or path.stat().st_size != record['size']:
                            raise RuntimeError('Local artifact changed after registration')
                        folder = self.drive.ensure_path([record['stage'], record['branch'], record['model'], record['run_id']])
                        remote = self.drive.upload_artifact(path, artifact_id=artifact_id, folder_id=folder, config_hash=record['configuration_hash'])
                        record.update(sync_status='VERIFIED', drive_file_id=remote['id'], drive_folder_id=folder)
                    record.pop('sync_error', None)
                    completed.append(artifact_id)
                except Exception as error:
                    record.update(sync_status='FAILED', sync_error=type(error).__name__)
                    failed.append(artifact_id)
                with self._connect() as db:
                    self._put(db, record)
        if verify_existing:
            self.export_manifest()
        return {"status": "FAILED" if failed else "VERIFIED", "uploaded": completed, "failed": failed,
                "scope": "full_reconciliation" if verify_existing else "pending_incremental"}

    def retrieve_checkpoint(self, artifact_id: str, destination: str | Path, expected_provenance: dict) -> Path:
        critical = {"configuration_hash", "source_code_commit", "dataset_revision", "model_revision", "seeds", "model"}
        if critical - expected_provenance.keys() or any(expected_provenance[key] in (None, "", []) for key in critical):
            raise RuntimeError("Artifact compatibility requires complete critical expected provenance")
        record = self._get(artifact_id)
        if record is None or any(record.get(k) != v for k, v in expected_provenance.items()):
            raise RuntimeError("Artifact compatibility failed")
        local = Path(record["local_path"])
        if local.is_file() and file_hash(local) == record["sha256"]:
            return local
        if self.drive is None or record["sync_status"] != "VERIFIED":
            raise RuntimeError("No verified artifact copy is available")
        restored = self.drive.download_artifact(artifact_id, destination, config_hash=record["configuration_hash"])
        if file_hash(restored) != record["sha256"]:
            raise RuntimeError("Retrieved artifact failed manifest checksum")
        return restored

    def completion_hook(self, provenance: dict):
        def publish(path, metadata):
            self.register(path, {**provenance, **metadata})
            result = self.sync_artifacts()
            if result["status"] != "VERIFIED":
                raise RuntimeError("Permanent artifact synchronization incomplete; registry retains pending/failed records")
        return publish
