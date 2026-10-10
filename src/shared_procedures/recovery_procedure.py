"""Indexed durable retry accounting, with 3 retries per operation / 100 per worker."""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import os
import random
import re
import sqlite3
import tempfile
import time
from typing import Callable, TypeVar
from .identity import canonical_json,content_hash
from .storage import read_json,fsync_directory

T=TypeVar('T')
class RecoveryStopped(RuntimeError):
    pass
class RetryableFailure(RuntimeError):
    def __init__(self,message: str,*,retry_after: float | None=None):
        super().__init__(message)
        self.retry_after=retry_after

def worker_recovery_path(artifact_root: str | Path | None=None, *, worker_id: str | None=None) -> Path:
    """Resolve one persistent retry ledger for every production subsystem of a worker."""
    identifier=worker_id or os.getenv('ILLEGIBILITY_WORKER_ID','coordinator')
    if not isinstance(identifier,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}',identifier):
        raise RecoveryStopped('worker identity must be a stable safe path component')
    explicit=os.getenv('ILLEGIBILITY_WORKER_RECOVERY_PATH')
    if explicit:return Path(explicit).expanduser().resolve()
    root=Path(os.getenv('ARTIFACT_ROOT') or artifact_root or 'artifacts')
    return (root/'state'/'workers'/identifier/'retries.json').resolve()

def configure_worker_recovery(artifact_root: str | Path, *, worker_id: str | None=None) -> Path:
    """Bind startup; explicit fixture RecoveryManager(path) remains isolated."""
    path=worker_recovery_path(artifact_root,worker_id=worker_id)
    os.environ['ILLEGIBILITY_WORKER_ID']=worker_id or os.getenv('ILLEGIBILITY_WORKER_ID','coordinator')
    os.environ['ILLEGIBILITY_WORKER_RECOVERY_PATH']=str(path)
    return path

def _validate_export(state: dict) -> None:
    if not isinstance(state,dict) or not isinstance(state.get('operations'),dict) or type(state.get('worker_retries')) is not int or state['worker_retries']<0:
        raise RecoveryStopped('persistent retry ledger is invalid; counters cannot be reset')
    if any(not isinstance(key,str) or not key or type(value) is not int or value<0 for key,value in state['operations'].items()):
        raise RecoveryStopped('persistent operation retry counters are invalid')
    if not isinstance(state.get('started_operations',[]),list) or any(not isinstance(key,str) or not key for key in state.get('started_operations',[])):
        raise RecoveryStopped('persistent operation identities are invalid')

class RecoveryManager:
    def __init__(self,path: str | Path | None=None,*,max_operation_retries: int=3,max_worker_retries: int=100):
        if max_operation_retries!=3 or max_worker_retries!=100:
            raise RecoveryStopped('finite source retry limits must be 3 per operation / 100 per worker')
        self.path=Path(path) if path is not None else worker_recovery_path()
        # Keep legacy JSON untouched; existing callers retain the same logical path.
        self.database_path=Path(str(self.path)+'.sqlite3')
        self.operation_limit=max_operation_retries;self.worker_limit=max_worker_retries

    @classmethod
    def for_worker(cls,artifact_root: str | Path | None=None,*,worker_id: str | None=None):
        return cls(worker_recovery_path(artifact_root,worker_id=worker_id))

    @contextmanager
    def _database(self):
        self.database_path.parent.mkdir(parents=True,exist_ok=True)
        connection=sqlite3.connect(self.database_path,timeout=30,isolation_level=None)
        try:
            connection.execute('PRAGMA journal_mode=WAL')
            connection.execute('PRAGMA synchronous=FULL')
            connection.execute('CREATE TABLE IF NOT EXISTS operations (operation_id TEXT PRIMARY KEY, retries INTEGER NOT NULL CHECK(retries>=0), started INTEGER NOT NULL CHECK(started IN (0,1)))')
            connection.execute('CREATE TABLE IF NOT EXISTS worker (singleton INTEGER PRIMARY KEY CHECK(singleton=1), retries INTEGER NOT NULL CHECK(retries>=0))')
            connection.execute('CREATE TABLE IF NOT EXISTS imports (source TEXT PRIMARY KEY, snapshot TEXT NOT NULL)')
            connection.execute('INSERT OR IGNORE INTO worker VALUES (1,0)')
            # Import a legacy JSON counter once. No hot operation exports/scans the ledger.
            if self.path.exists() and connection.execute('SELECT 1 FROM imports WHERE source=?',(str(self.path.resolve()),)).fetchone() is None:
                state=read_json(self.path);_validate_export(state)
                connection.execute('BEGIN IMMEDIATE')
                try:
                    self._import(connection,str(self.path.resolve()),state)
                    connection.execute('COMMIT')
                except BaseException:
                    connection.execute('ROLLBACK');raise
            yield connection
        except sqlite3.DatabaseError as error:
            raise RecoveryStopped('durable retry database failed; budget cannot be reset') from error
        finally:connection.close()

    @staticmethod
    def _import(connection,source: str,state: dict) -> None:
        import json
        _validate_export(state)
        row=connection.execute('SELECT snapshot FROM imports WHERE source=?',(source,)).fetchone()
        previous=json.loads(row[0]) if row else {'worker_retries':0,'operations':{}}
        if state['worker_retries']<previous['worker_retries'] or any(state['operations'].get(key,0)<value for key,value in previous['operations'].items()):
            raise RecoveryStopped('legacy retry counters decreased; refusing budget reset')
        delta=state['worker_retries']-previous['worker_retries']
        connection.execute('UPDATE worker SET retries=retries+? WHERE singleton=1',(delta,))
        started=set(state.get('started_operations',[]))
        for key in set(state['operations'])|started:
            change=state['operations'].get(key,0)-previous['operations'].get(key,0)
            connection.execute('INSERT INTO operations VALUES (?,?,?) ON CONFLICT(operation_id) DO UPDATE SET retries=operations.retries+excluded.retries,started=MAX(operations.started,excluded.started)',(key,change,int(key in started)))
        snapshot={'worker_retries':state['worker_retries'],'operations':state['operations']}
        connection.execute('INSERT INTO imports VALUES (?,?) ON CONFLICT(source) DO UPDATE SET snapshot=excluded.snapshot',(source,canonical_json(snapshot).decode()))

    def counters(self) -> dict:
        """Explicit audit export; operational paths use indexed single-row reads."""
        import json
        with self._database() as connection:
            connection.execute('BEGIN')
            rows=list(connection.execute('SELECT operation_id,retries,started FROM operations ORDER BY operation_id'))
            worker=connection.execute('SELECT retries FROM worker WHERE singleton=1').fetchone()[0]
            imports={source:json.loads(snapshot) for source,snapshot in connection.execute('SELECT source,snapshot FROM imports')}
            connection.execute('COMMIT')
        return {'operations':{key:count for key,count,_ in rows if count},'worker_retries':worker,
                'started_operations':[key for key,_,started in rows if started],**({'imported_ledgers':imports} if imports else {})}

    def _begin_attempt(self,operation_id: str,*,retry_only: bool=False,before_attempt: Callable[[],None] | None=None) -> int:
        if not isinstance(operation_id,str) or not operation_id:raise RecoveryStopped('stable operation identity required')
        with self._database() as connection:
            connection.execute('BEGIN IMMEDIATE')
            try:
                row=connection.execute('SELECT retries,started FROM operations WHERE operation_id=?',(operation_id,)).fetchone()
                count,started=row if row else (0,0)
                retry=retry_only or bool(started)
                worker=connection.execute('SELECT retries FROM worker WHERE singleton=1').fetchone()[0]
                if worker>=self.worker_limit or (retry and count>=self.operation_limit):
                    raise RecoveryStopped('persistent retry allowance exhausted')
                # Guards run before an attempt is registered or a retry is charged.
                # They must be short local checks and must not recursively use this ledger.
                if before_attempt is not None:before_attempt()
                connection.execute('INSERT INTO operations VALUES (?,?,?) ON CONFLICT(operation_id) DO UPDATE SET retries=excluded.retries,started=MAX(operations.started,excluded.started)',
                                   (operation_id,count+int(retry),int(started or not retry_only)))
                if retry:connection.execute('UPDATE worker SET retries=retries+1 WHERE singleton=1')
                connection.execute('COMMIT')
                return count+int(retry)
            except BaseException:
                connection.execute('ROLLBACK');raise

    def reserve_retry(self,operation_id: str) -> int:
        return self._begin_attempt(operation_id,retry_only=True)

    def run(self,operation_id: str,operation: Callable[[],T],*,sleep: Callable[[float],None]=time.sleep,
            before_attempt: Callable[[],None] | None=None) -> T:
        while True:
            retry=self._begin_attempt(operation_id,before_attempt=before_attempt)
            try:return operation()
            except RetryableFailure as failure:
                if retry>=self.operation_limit:raise RecoveryStopped('persistent retry allowance exhausted') from failure
                delay=failure.retry_after if failure.retry_after is not None else min(30,2**retry)+random.random()
                if not isinstance(delay,(int,float)) or not 0<=delay<=60:
                    raise RecoveryStopped('Retry-After exceeds bounded worker wait; pause and resume later') from failure
                sleep(delay)
            # Unknown, quota, frozen-input and ordinary model outcomes do not automatically retry.

    def snapshot_database(self,destination: str | Path) -> Path:
        """Create a coherent atomic SQLite backup for independent durable publication."""
        target=Path(destination);target.parent.mkdir(parents=True,exist_ok=True)
        descriptor,name=tempfile.mkstemp(prefix='.'+target.name+'.',dir=target.parent);os.close(descriptor)
        temporary=Path(name)
        try:
            with self._database() as connection:
                backup=sqlite3.connect(temporary)
                try:connection.backup(backup)
                finally:backup.close()
            with temporary.open('rb') as handle:os.fsync(handle.fileno())
            os.replace(temporary,target);fsync_directory(target.parent)
            return target
        finally:
            if temporary.exists():temporary.unlink()

def recover(operation_id: str,operation: Callable[[],T],manager: RecoveryManager,**kwargs) -> T:
    return manager.run(operation_id,operation,**kwargs)

def migrate_worker_recovery(legacy_paths: list[str | Path], *, artifact_root: str | Path | None=None,
                            worker_id: str | None=None) -> dict:
    """Idempotently retain nondecreasing prior production counters; originals remain."""
    manager=RecoveryManager.for_worker(artifact_root,worker_id=worker_id)
    target=manager.path.resolve()
    sources=sorted({Path(path).resolve() for path in legacy_paths if Path(path).exists() or Path(str(path)+'.sqlite3').exists()}-{target})
    with manager._database() as connection:
        connection.execute('BEGIN IMMEDIATE')
        try:
            for source in sources:
                state=RecoveryManager(source).counters() if Path(str(source)+'.sqlite3').exists() else read_json(source)
                _validate_export(state)
                manager._import(connection,str(source),state)
            connection.execute('COMMIT')
        except BaseException:
            connection.execute('ROLLBACK');raise
    return manager.counters()
