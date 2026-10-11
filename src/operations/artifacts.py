"""Canonical permanent-artifact registry and compatibility-aware retrieval."""
from __future__ import annotations
import json
import os
import re
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from shared_procedures.identity import content_hash
from shared_procedures.storage import atomic_write_json, read_json, file_lock
from operations.drive import DriveStore, file_hash

REQUIRED_PROVENANCE = {"experiment_id", "branch", "model", "run_id", "stage", "producer",
                       "source_code_commit", "configuration_hash", "dataset_revision", "model_revision", "seeds"}
SECRET_SUFFIXES = {".pem", ".key"}
REGISTRY_METADATA = {'artifact_id','local_path','relative_path','sha256','md5','size','created_at',
    'completion_status','validation_status','sync_status','retention','drive_file_id','drive_folder_id',
    'sync_error','recovery_descriptor_id'}


def _record_provenance(record: dict) -> dict:
    if 'provenance' in record:
        return record['provenance']
    # Earlier registries flattened the original complete provenance, including
    # asset/license/source metadata. Infer it only by removing generated fields;
    # the caller must then prove the exact original artifact identity below.
    return {key:value for key,value in record.items() if key not in REGISTRY_METADATA}


def validate_registered_record(record: dict) -> None:
    """Validate the exact immutable registry identity, including extra provenance."""
    if not isinstance(record,dict):
        raise RuntimeError('Artifact recovery record is invalid')
    provenance = _record_provenance(record)
    if not isinstance(provenance,dict) or REQUIRED_PROVENANCE - provenance.keys():
        raise RuntimeError('Artifact recovery provenance is incomplete')
    for key in REQUIRED_PROVENANCE - {'seeds'}:
        if not isinstance(provenance[key],str) or not provenance[key] or record.get(key) != provenance[key]:
            raise RuntimeError('Artifact recovery provenance differs from its registry record')
    if (not isinstance(provenance['seeds'],list) or not provenance['seeds']
        or any(type(seed) is not int for seed in provenance['seeds']) or record.get('seeds') != provenance['seeds']):
        raise RuntimeError('Artifact recovery seeds are invalid')
    for key,pattern in (('source_code_commit',r'[0-9a-f]{40}'),
                        ('configuration_hash',r'[0-9a-f]{64}'),
                        ('dataset_revision',r'(?:[0-9a-f]{40}|[0-9a-f]{64})'),
                        ('model_revision',r'(?:[0-9a-f]{40}|[0-9a-f]{64})'),
                        ('artifact_id',r'[0-9a-f]{64}'),('sha256',r'[0-9a-f]{64}'),('md5',r'[0-9a-f]{32}')):
        if not isinstance(record.get(key),str) or not re.fullmatch(pattern,record[key]):
            raise RuntimeError('Artifact recovery immutable identity is invalid')
    value = record.get('relative_path')
    if not isinstance(value,str) or not value:
        raise RuntimeError('Artifact recovery relative path is missing')
    relative = Path(value)
    if (relative.is_absolute() or '..' in relative.parts or not relative.parts
        or relative.as_posix() != value or relative.suffix in SECRET_SUFFIXES
        or any(part.startswith('.env') or part in ('.secrets','.git') or 'oauth' in part.lower()
               or part.lower()=='token' for part in relative.parts)):
        raise RuntimeError('Artifact recovery relative path is unsafe')
    if type(record.get('size')) is not int or record['size'] < 0:
        raise RuntimeError('Artifact recovery size is invalid')
    identity = {'provenance':provenance,'sha256':record['sha256'],'path':value}
    if record['artifact_id'] != content_hash(identity):
        raise RuntimeError('Artifact recovery identity differs from its provenance/path/checksum')


class ArtifactService:
    def __init__(self, root: str | Path, registry: str | Path, drive: DriveStore | None = None, *, drive_recovery: bool = False):
        self.root = Path(root).resolve()
        self.registry = Path(registry)
        self.drive = drive
        self.drive_recovery = drive_recovery
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
                  "provenance": provenance,
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

    @staticmethod
    def _recovery_descriptor(record: dict) -> dict:
        """Small immutable discovery record; never includes scientific payload bytes."""
        validate_registered_record(record)
        provenance = _record_provenance(record)
        return {'version':1,'artifact':{'provenance':provenance,
            **{key:record[key] for key in ('artifact_id','relative_path','sha256','md5','size','retention')}}}

    @staticmethod
    def _descriptor_id(artifact_id: str) -> str:
        return content_hash({'kind':'permanent_artifact_recovery_v1','artifact_id':artifact_id})

    def _publish_recovery_descriptor(self, record: dict) -> None:
        folder = self.drive.ensure_path(['recovery',record['source_code_commit'],record['configuration_hash'],record['stage']])
        descriptor_id = self._descriptor_id(record['artifact_id'])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/f'{record["artifact_id"]}.json'
            atomic_write_json(path,self._recovery_descriptor(record))
            self.drive.upload_artifact(path,artifact_id=descriptor_id,folder_id=folder,
                                       config_hash=record['configuration_hash'])
        record['recovery_descriptor_id'] = descriptor_id

    def assert_collection_reconciled(self, expected_provenance: dict) -> None:
        """A new/partially lost worker may not silently replay existing paid work."""
        marker = self.registry.with_name('collection_recovery.json')
        ignored_descriptors = set()
        if marker.exists():
            evidence = read_json(marker)
            manager = self.drive.recovery
            if (evidence.get('native_retry_state') != 'VERIFIED'
                or evidence.get('configuration_hash') != expected_provenance['configuration_hash']
                or evidence.get('source_code_commit') != expected_provenance['source_code_commit']
                or evidence.get('worker_id') != os.environ.get('ILLEGIBILITY_WORKER_ID','coordinator')
                or not manager.database_path.is_file()):
                raise RuntimeError('Recovered collection retry ledger reconciliation is incomplete')
            with manager._database() as connection:
                row = connection.execute('SELECT snapshot FROM imports WHERE source=?',
                    (evidence.get('retry_import_source'),)).fetchone()
                retries = connection.execute('SELECT retries FROM worker WHERE singleton=1').fetchone()[0]
            if (row is None or json.loads(row[0]).get('worker_retries') != evidence.get('worker_retry_floor')
                or retries < evidence['worker_retry_floor']):
                raise RuntimeError('Recovered collection retry ledger reconciliation differs')
            known = json.loads(row[0])
            counters = manager.counters()
            if (retries>=100 or any(counters['operations'].get(key,0)<count for key,count in known['operations'].items())
                or not set(known.get('started_operations',[])).issubset(counters['started_operations'])):
                raise RuntimeError('Recovered collection operation retry allowance differs or is exhausted')
            ignored_descriptors = set(evidence.get('non_collection_descriptor_ids',[]))
        folder = self.drive.ensure_path(['recovery',expected_provenance['source_code_commit'],
                                        expected_provenance['configuration_hash'],'collect_clean'])
        local = {record.get('recovery_descriptor_id'):record for record in self.list_artifacts()
                 if all(record.get(key)==value for key,value in expected_provenance.items())
                 and record.get('stage')=='collect_clean'}
        for remote in self.drive.list_artifacts(folder):
            descriptor_id = remote.get('appProperties',{}).get('artifact_id')
            if descriptor_id in ignored_descriptors:
                continue
            record = local.get(descriptor_id)
            if record is None or record.get('sync_status') != 'VERIFIED':
                raise RuntimeError('Existing Drive collection requires recover-collection and ledger reconciliation before paid work')
            path = Path(record['local_path'])
            if Path(record['relative_path']).name=='collection_status.json':
                continue  # Resume reconstructs this mutable cursor from raw caches.
            if not path.is_file() or path.is_symlink() or file_hash(path) != record['sha256']:
                raise RuntimeError('Missing or changed collection scratch requires recover-collection before paid work')

    @staticmethod
    def _remote_retry_floor(remote: dict, worker_id: str) -> int:
        properties = remote.get('appProperties',{})
        value = properties.get('worker_retry_floor')
        if (properties.get('worker_id') != worker_id or not isinstance(value,str)
            or not re.fullmatch(r'(?:0|[1-9][0-9]?|100)',value)):
            raise RuntimeError('Native collection recovery lacks authoritative worker retry metadata')
        return int(value)

    @staticmethod
    def _remote_operation_retry(remote: dict) -> tuple[str,int]:
        properties = remote.get('appProperties',{})
        value = properties.get('operation_retries')
        artifact_id = properties.get('artifact_id')
        if (not isinstance(value,str) or not re.fullmatch(r'[0-3]',value)
            or not isinstance(artifact_id,str) or not re.fullmatch(r'[0-9a-f]{64}',artifact_id)):
            raise RuntimeError('Native collection recovery lacks authoritative operation retry metadata')
        return 'upload_'+artifact_id,int(value)

    @staticmethod
    def _read_retry_snapshot(path: Path) -> dict:
        connection = sqlite3.connect(f'file:{path}?mode=ro',uri=True)
        try:
            if connection.execute('PRAGMA integrity_check').fetchone()!=('ok',):
                raise RuntimeError('Native retry snapshot SQLite integrity failed')
            rows = list(connection.execute('SELECT operation_id,retries,started FROM operations'))
            worker = connection.execute('SELECT retries FROM worker WHERE singleton=1').fetchone()[0]
        except (sqlite3.DatabaseError,TypeError,IndexError):
            raise RuntimeError('Native retry snapshot schema is invalid') from None
        finally:
            connection.close()
        state = {'worker_retries':worker,'operations':{key:count for key,count,_ in rows},
                 'started_operations':[key for key,_,started in rows if started]}
        from shared_procedures.recovery_procedure import _validate_export
        _validate_export(state)
        if worker>100 or any(count>3 or started not in (0,1) for _,count,started in rows):
            raise RuntimeError('Native retry snapshot exceeds immutable retry limits')
        return state

    @staticmethod
    def _merge_retry_state(manager, state: dict, *, source: str | None=None, known_state: dict | None=None) -> None:
        """Merge proven nondecreasing counters without recharging an intact ledger."""
        if state['worker_retries']>100 or any(count>3 for count in state['operations'].values()):
            raise RuntimeError('Recovered native retry allowance is exhausted')
        with manager._database() as connection:
            connection.execute('BEGIN IMMEDIATE')
            try:
                connection.execute('UPDATE worker SET retries=MAX(retries,?) WHERE singleton=1',(state['worker_retries'],))
                started = set(state.get('started_operations',[]))
                for key in set(state['operations'])|started:
                    connection.execute('INSERT INTO operations VALUES (?,?,?) ON CONFLICT(operation_id) DO UPDATE SET retries=MAX(operations.retries,excluded.retries),started=MAX(operations.started,excluded.started)',
                        (key,state['operations'].get(key,0),int(key in started)))
                if source:
                    connection.execute('INSERT INTO imports VALUES (?,?) ON CONFLICT(source) DO UPDATE SET snapshot=excluded.snapshot',
                        (source,json.dumps(known_state or state,sort_keys=True)))
                connection.execute('COMMIT')
            except BaseException:
                connection.execute('ROLLBACK');raise

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
                    # Publish discovery first. If the process disappears after the
                    # payload upload but before the local SQLite commit, a fresh
                    # machine can still find and validate that exact remote copy.
                    if self.drive_recovery and not (record['stage']=='collect_clean'
                        and Path(record['relative_path']).name=='collection_status.json'):
                        self._publish_recovery_descriptor(record)
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

    def restore_collection(self, expected_provenance: dict, *, data_root: str | Path) -> dict:
        """Discover compatible immutable collection files with no local inventory.

        Import coherent native operation counters and completed transport deltas.
        Paid API/spending/training state is outside this narrowly scoped command.
        """
        critical = {'configuration_hash','source_code_commit','dataset_revision','model_revision','seeds','model'}
        if (self.drive is None or critical - expected_provenance.keys()
            or any(expected_provenance[key] in (None,'',[]) for key in critical)):
            raise RuntimeError('Collection recovery requires Drive and complete expected provenance')
        data_root = Path(data_root).resolve()
        if not data_root.is_relative_to(self.root):
            raise ValueError('Collection data root must be below the artifact service root')
        self.drive.create_project_folder()
        manager = self.drive.recovery
        baseline = manager.counters()
        found = []
        for stage in ('generation_preflight','provision','prepare_panels','prepare_candidates','collect_clean'):
            folder = self.drive.ensure_path(['recovery',expected_provenance['source_code_commit'],
                                            expected_provenance['configuration_hash'],stage])
            found.extend(self.drive.list_artifacts(folder))
        planned = {}
        unresolved = []
        worker_id = os.environ.get('ILLEGIBILITY_WORKER_ID','coordinator')
        retry_floor = 0
        ignored_descriptors = []
        known_operations = {}
        known_started = set()
        for remote in found:
            props = remote.get('appProperties',{})
            artifact_name = remote.get('name','')
            if (props.get('upload_status') != 'verified' or props.get('project_id') != self.drive.folder_id
                or props.get('config_hash') != expected_provenance['configuration_hash']
                or not re.fullmatch(r'[0-9a-f]{64}\.json',artifact_name)
                or props.get('artifact_id') != self._descriptor_id(artifact_name[:-5])):
                raise RuntimeError('Native collection recovery descriptor publication is incomplete')
            payload = self.drive.resolve_artifact_id(artifact_name[:-5])
            if payload is None or payload.get('appProperties',{}).get('upload_status') != 'verified':
                raise RuntimeError('Collection recovery has unresolved uploads; paid regeneration is blocked')
            for completed in (remote,payload):
                retry_floor = max(retry_floor,self._remote_retry_floor(completed,worker_id))
                key,count = self._remote_operation_retry(completed)
                known_operations[key] = max(known_operations.get(key,0),count)
                known_started.add(key)
        # Seed known transport budgets before any recovery media transfer. Full
        # native operation identities arrive in coherent stage snapshots below.
        self._merge_retry_state(manager,{'worker_retries':retry_floor,'operations':known_operations,
                                       'started_operations':list(known_started)})
        transfer_baseline = manager.counters()
        snapshots = []
        with tempfile.TemporaryDirectory() as temporary:
            for remote in found:
                props = remote.get('appProperties',{})
                descriptor_id = props.get('artifact_id')
                if (not isinstance(descriptor_id,str) or not re.fullmatch('[0-9a-f]{64}',descriptor_id)
                    or props.get('config_hash') != expected_provenance['configuration_hash']
                    or props.get('upload_status') != 'verified'):
                    raise RuntimeError('Collection recovery descriptor is incomplete or incompatible')
                retry_floor = max(retry_floor,self._remote_retry_floor(remote,worker_id))
                path = self.drive.download_artifact(descriptor_id,Path(temporary)/f'{descriptor_id}.json',
                                                    config_hash=expected_provenance['configuration_hash'])
                descriptor = read_json(path)
                if not isinstance(descriptor,dict) or descriptor.get('version') != 1 or not isinstance(descriptor.get('artifact'),dict):
                    raise RuntimeError('Collection recovery descriptor schema is invalid')
                artifact = descriptor['artifact']
                provenance = artifact.get('provenance')
                relative = Path(artifact.get('relative_path',''))
                if (not isinstance(provenance,dict) or REQUIRED_PROVENANCE - provenance.keys()
                    or any(provenance.get(key) != value for key,value in expected_provenance.items())
                    or relative.is_absolute() or '..' in relative.parts or not relative.parts
                    or type(artifact.get('size')) is not int or artifact['size'] < 0):
                    raise RuntimeError('Collection recovery descriptor provenance/path is invalid')
                identity = {'provenance':provenance,'sha256':artifact.get('sha256'),'path':relative.as_posix()}
                artifact_id = content_hash(identity)
                if artifact.get('artifact_id') != artifact_id or descriptor_id != self._descriptor_id(artifact_id):
                    raise RuntimeError('Collection recovery descriptor identity differs')
                destination = self.root/relative
                payload = self.drive.resolve_artifact_id(artifact_id)
                if payload is None or payload.get('appProperties',{}).get('upload_status') != 'verified':
                    unresolved.append(artifact_id)
                    continue
                record = {**provenance,**artifact,'local_path':str(destination),
                    'created_at':datetime.now(timezone.utc).isoformat(),'completion_status':'COMPLETED',
                    'validation_status':'content_hashed','sync_status':'VERIFIED',
                    'drive_file_id':payload.get('id'),
                    'drive_folder_id':payload.get('parents',[None])[0] if len(payload.get('parents',[]))==1 else None,
                    'recovery_descriptor_id':descriptor_id}
                validate_registered_record(record)
                self.drive.verify_registered_artifact(record)
                retry_floor = max(retry_floor,self._remote_retry_floor(payload,worker_id))
                if (len(relative.parts)>=5 and relative.parts[-5:-3]==('operations','worker_state')
                    and relative.parts[-3]==worker_id and relative.name=='retries.sqlite3'):
                    snapshot_path = Path(temporary)/f'{artifact_id}.sqlite3'
                    self.drive.restore_registered_artifact(record,snapshot_path,expected_provenance=expected_provenance)
                    snapshot = self._read_retry_snapshot(snapshot_path)
                    retry_floor = max(retry_floor,snapshot['worker_retries'])
                    for key,count in snapshot['operations'].items():
                        known_operations[key] = max(known_operations.get(key,0),count)
                    known_started.update(snapshot['started_operations'])
                    snapshots.append(artifact_id)
                if not destination.resolve().is_relative_to(data_root):
                    ignored_descriptors.append(descriptor_id)
                    continue
                data_relative = destination.relative_to(data_root)
                allowed = (data_relative.parts[0] in ('evaluation','candidates')
                    or (len(data_relative.parts)>1 and data_relative.parts[0]=='clean_traces'
                        and data_relative.parts[1] in ('raw','clean_traces','rejected_responses','accepted_pool.json')))
                if not allowed:
                    ignored_descriptors.append(descriptor_id)
                    continue  # Mutable status/cursor files are reconstructed from raw caches.
                if destination.is_symlink() or any(parent.is_symlink() for parent in destination.parents if parent != self.root):
                    raise RuntimeError('Collection recovery cannot traverse symbolic links')
                existing = planned.get(str(destination))
                if existing and existing['artifact_id'] != artifact_id:
                    raise RuntimeError('Collection recovery has conflicting immutable versions')
                planned[str(destination)] = record
        if unresolved:
            raise RuntimeError('Collection recovery has unresolved uploads; paid regeneration is blocked')
        if planned and not snapshots:
            raise RuntimeError('Native collection recovery requires a matching coherent retry snapshot')
        known_state = {'worker_retries':retry_floor,'operations':known_operations,
                       'started_operations':list(known_started)}
        current = manager.counters()
        # Preserve newly charged recovery-read retries as actual deltas, rather
        # than resetting old native or transport operation allowances.
        deltas = {key:count-transfer_baseline['operations'].get(key,0)
                  for key,count in current['operations'].items()}
        original_extras = sum(max(0,count-known_operations.get(key,0)) for key,count in baseline['operations'].items())
        merged = {'worker_retries':max(baseline['worker_retries'],retry_floor+original_extras)
                      + current['worker_retries']-transfer_baseline['worker_retries'],
            'operations':{key:max(known_operations.get(key,0),baseline['operations'].get(key,0))+deltas.get(key,0)
                for key in set(known_operations)|set(current['operations'])|set(baseline['operations'])},
            'started_operations':list(known_started|set(current['started_operations'])|set(baseline['started_operations']))}
        retry_import_source = 'drive_native_collection:'+content_hash({'project':self.drive.folder_id,
            'configuration_hash':expected_provenance['configuration_hash'],'worker':worker_id})
        self._merge_retry_state(manager,merged,source=retry_import_source,known_state=known_state)
        restored = []
        for destination_string,record in sorted(planned.items()):
            destination = Path(destination_string)
            if destination.exists():
                path = self._safe_path(destination)
                if (file_hash(path) != record['sha256'] or file_hash(path,'md5') != record['md5']
                    or path.stat().st_size != record['size']):
                    raise RuntimeError('Existing collection recovery destination differs')
                self.drive.verify_registered_artifact(record)
            else:
                self.drive.restore_registered_artifact(record,destination,expected_provenance=expected_provenance)
            with self._connect() as db:
                self._put(db,record)
            restored.append(record['artifact_id'])
        result = {'status':'RESTORED','restored_artifacts':len(restored),
            'scope':'immutable_collection_files_only',
            'native_retry_state':'VERIFIED' if restored else 'NOT_APPLICABLE',
            'worker_id':worker_id,'worker_retry_floor':retry_floor,'retry_import_source':retry_import_source,
            'configuration_hash':expected_provenance['configuration_hash'],
            'source_code_commit':expected_provenance['source_code_commit'],
            'non_collection_descriptor_ids':ignored_descriptors,
            'native_retry_snapshot_ids':snapshots,
            'operational_ledgers':'paid API/spending/training state not imported',
            'resume_requires':'same frozen manifest/config/model/source and current externally anchored GPU spending authorization'}
        if restored:
            atomic_write_json(self.registry.with_name('collection_recovery.json'),result)
        return result

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
