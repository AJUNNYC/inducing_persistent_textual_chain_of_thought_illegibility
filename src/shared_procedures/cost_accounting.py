"""Indexed durable reservations and actual expenses under explicit authorization."""
from __future__ import annotations
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
import json
import os
import sqlite3
import tempfile
from typing import Any
from .identity import canonical_json
from .storage import read_json,fsync_directory
from .execution_state import timestamp

class SpendingBlocked(RuntimeError):
    pass

def _money(value: Any) -> Decimal:
    try:result=Decimal(str(value))
    except Exception as error:raise SpendingBlocked('invalid cost') from error
    if not result.is_finite() or result<0:raise SpendingBlocked('cost must be finite and nonnegative')
    return result

def _encoded(value: Any) -> str:return canonical_json(value).decode()

class CostLedger:
    def __init__(self,path: str | Path,*,authorization: dict | None=None):
        self.path=Path(path);self.database_path=Path(str(self.path)+'.sqlite3')
        self.authorization=authorization or {'authorized':False,'authorized_usd':0,'authorization_evidence':None}

    @contextmanager
    def _database(self):
        self.database_path.parent.mkdir(parents=True,exist_ok=True)
        connection=sqlite3.connect(self.database_path,timeout=30,isolation_level=None)
        try:
            connection.execute('PRAGMA journal_mode=WAL');connection.execute('PRAGMA synchronous=FULL')
            connection.execute('CREATE TABLE IF NOT EXISTS entries (operation_id TEXT PRIMARY KEY, status TEXT NOT NULL CHECK(status IN (\'reserved\',\'charged\')), amount TEXT NOT NULL, category TEXT NOT NULL, usage TEXT NOT NULL, at TEXT NOT NULL)')
            connection.execute('CREATE TABLE IF NOT EXISTS aggregates (category TEXT PRIMARY KEY, spent TEXT NOT NULL, reserved TEXT NOT NULL)')
            connection.execute('CREATE TABLE IF NOT EXISTS totals (singleton INTEGER PRIMARY KEY CHECK(singleton=1), spent TEXT NOT NULL, reserved TEXT NOT NULL)')
            connection.execute('CREATE TABLE IF NOT EXISTS events (sequence INTEGER PRIMARY KEY AUTOINCREMENT, payload TEXT NOT NULL)')
            connection.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
            connection.execute("INSERT OR IGNORE INTO totals VALUES (1,'0','0')")
            if self.path.exists() and connection.execute("SELECT 1 FROM metadata WHERE key='legacy_imported'").fetchone() is None:
                connection.execute('BEGIN IMMEDIATE')
                try:
                    if connection.execute("SELECT 1 FROM metadata WHERE key='legacy_imported'").fetchone() is None:self._import_legacy(connection,read_json(self.path))
                    connection.execute('COMMIT')
                except BaseException:
                    connection.execute('ROLLBACK');raise
            yield connection
        except sqlite3.DatabaseError as error:
            raise SpendingBlocked('durable spending database failed; charges cannot be reset') from error
        finally:connection.close()

    @staticmethod
    def _update_total(connection,category: str,*,spent: Decimal=Decimal(0),reserved: Decimal=Decimal(0)) -> None:
        total=connection.execute('SELECT spent,reserved FROM totals WHERE singleton=1').fetchone()
        a,b=_money(total[0])+spent,_money(total[1])+reserved
        if a<0 or b<0:raise SpendingBlocked('durable spending totals are inconsistent')
        connection.execute('UPDATE totals SET spent=?,reserved=? WHERE singleton=1',(str(a),str(b)))
        row=connection.execute('SELECT spent,reserved FROM aggregates WHERE category=?',(category,)).fetchone() or ('0','0')
        a,b=_money(row[0])+spent,_money(row[1])+reserved
        if a<0 or b<0:raise SpendingBlocked('durable category totals are inconsistent')
        connection.execute('INSERT INTO aggregates VALUES (?,?,?) ON CONFLICT(category) DO UPDATE SET spent=excluded.spent,reserved=excluded.reserved',(category,str(a),str(b)))

    @staticmethod
    def _event(connection,value: dict) -> None:connection.execute('INSERT INTO events(payload) VALUES (?)',(_encoded(value),))

    def _import_legacy(self,connection,state: dict) -> None:
        if not isinstance(state,dict) or any(not isinstance(state.get(key),dict) for key in ('charges','reservations')) or not isinstance(state.get('events',[]),list):
            raise SpendingBlocked('legacy spending ledger is invalid; refusing charge reset')
        if set(state['charges'])&set(state['reservations']):raise SpendingBlocked('legacy charge is both reserved and spent')
        for field,status in (('charges','charged'),('reservations','reserved')):
            for operation_id,value in state[field].items():
                amount=_money(value['amount_usd']);category=value['category']
                if not isinstance(operation_id,str) or not operation_id or not isinstance(category,str) or not category:raise SpendingBlocked('legacy spending identity is invalid')
                if connection.execute('SELECT 1 FROM entries WHERE operation_id=?',(operation_id,)).fetchone():raise SpendingBlocked('legacy charge conflicts with current indexed ledger')
                connection.execute('INSERT INTO entries VALUES (?,?,?,?,?,?)',(operation_id,status,str(amount),category,_encoded(value.get('usage',{})),value.get('at',timestamp())))
                self._update_total(connection,category,**({'spent':amount} if status=='charged' else {'reserved':amount}))
        for event in state.get('events',[]):self._event(connection,event)
        connection.execute("INSERT INTO metadata VALUES ('legacy_imported',?)",(str(self.path.resolve()),))

    def read(self) -> dict:
        """Explicit audit export. Hot paths never scan entries or rewrite event history."""
        with self._database() as connection:
            connection.execute('BEGIN')
            state={'charges':{},'reservations':{},'events':[]}
            for key,status,amount,category,usage,at in connection.execute('SELECT operation_id,status,amount,category,usage,at FROM entries ORDER BY operation_id'):
                record={'amount_usd':amount,'category':category,'at':at}
                if status=='charged':record['usage']=json.loads(usage)
                else:record['status']='reserved'
                state['charges' if status=='charged' else 'reservations'][key]=record
            state['events']=[json.loads(row[0]) for row in connection.execute('SELECT payload FROM events ORDER BY sequence')]
            connection.execute('COMMIT')
        return state

    def totals(self) -> dict:
        with self._database() as connection:spent,reserved=connection.execute('SELECT spent,reserved FROM totals WHERE singleton=1').fetchone()
        a,b=_money(spent),_money(reserved)
        return {'spent_usd':float(a),'reserved_usd':float(b),'committed_usd':float(a+b)}

    def reserve(self,operation_id: str,amount_usd: Any,*,category: str) -> dict:
        amount=_money(amount_usd);authorization=self.authorization
        if authorization.get('authorized') is not True or not authorization.get('authorization_evidence'):
            raise SpendingBlocked('paid work requires current explicit authorization evidence')
        ceiling=_money(authorization.get('authorized_usd',0));fraction=_money(authorization.get('scheduling_stop_fraction',0.95))
        if ceiling<=0:raise SpendingBlocked('current authorization must include a positive spending ceiling')
        if fraction>1:raise SpendingBlocked('scheduling stop fraction cannot exceed authorized ceiling')
        if not isinstance(operation_id,str) or not operation_id or not isinstance(category,str) or not category:raise SpendingBlocked('stable spending operation and category required')
        with self._database() as connection:
            connection.execute('BEGIN IMMEDIATE')
            try:
                if connection.execute('SELECT 1 FROM entries WHERE operation_id=?',(operation_id,)).fetchone():
                    raise SpendingBlocked('operation already charged or reserved; inspect uncertain billing before replay')
                spent,reserved=connection.execute('SELECT spent,reserved FROM totals WHERE singleton=1').fetchone();total=_money(spent)+_money(reserved)
                if total+amount>ceiling*fraction:raise SpendingBlocked('authorized scheduling ceiling exhausted including in-flight charges')
                if category in {'rl_pass_a','rl_pass_b'} and authorization.get('reward_service_allowance_usd') is not None:
                    rows=connection.execute("SELECT spent,reserved FROM aggregates WHERE category IN ('rl_pass_a','rl_pass_b')").fetchall()
                    reward_total=sum((_money(a)+_money(b) for a,b in rows),Decimal(0))
                    if reward_total+amount>_money(authorization['reward_service_allowance_usd']):raise SpendingBlocked('authorized local reward-service allowance exhausted')
                for warning in authorization.get('warning_fractions',[0.75,0.8,0.9]):
                    threshold=ceiling*_money(warning)
                    if total<threshold<=total+amount:self._event(connection,{'status':'spending_warning','fraction':warning,'committed_usd':str(total+amount),'at':timestamp()})
                reservation={'amount_usd':str(amount),'category':category,'at':timestamp(),'status':'reserved'}
                connection.execute('INSERT INTO entries VALUES (?,?,?,?,?,?)',(operation_id,'reserved',str(amount),category,'{}',reservation['at']))
                self._update_total(connection,category,reserved=amount);self._event(connection,{'operation_id':operation_id,**reservation})
                connection.execute('COMMIT');return reservation
            except BaseException:
                connection.execute('ROLLBACK');raise

    def commit(self,operation_id: str,actual_usd: Any,*,usage: dict | None=None) -> dict:
        amount=_money(actual_usd);encoded_usage=_encoded(usage or {})
        with self._database() as connection:
            connection.execute('BEGIN IMMEDIATE')
            try:
                row=connection.execute('SELECT status,amount,category,usage,at FROM entries WHERE operation_id=?',(operation_id,)).fetchone()
                if row is None:raise SpendingBlocked('unreserved charge')
                status,old_amount,category,old_usage,at=row
                if status=='charged':
                    if _money(old_amount)!=amount or json.loads(old_usage)!=(usage or {}):raise SpendingBlocked('charge replay differs')
                    connection.execute('COMMIT');return {'amount_usd':old_amount,'category':category,'usage':json.loads(old_usage),'at':at}
                charge={'amount_usd':str(amount),'category':category,'usage':usage or {},'at':timestamp()}
                connection.execute("UPDATE entries SET status='charged',amount=?,usage=?,at=? WHERE operation_id=?",(str(amount),encoded_usage,charge['at'],operation_id))
                self._update_total(connection,category,spent=amount,reserved=-_money(old_amount));self._event(connection,{'operation_id':operation_id,'status':'charged',**charge})
                connection.execute('COMMIT');return charge
            except BaseException:
                connection.execute('ROLLBACK');raise

    def release(self,operation_id: str,*,confirmed_unbilled: bool=False) -> None:
        if not confirmed_unbilled:raise SpendingBlocked('timeout/unknown billing cannot release in-flight reservation')
        with self._database() as connection:
            connection.execute('BEGIN IMMEDIATE')
            try:
                row=connection.execute('SELECT status,amount,category FROM entries WHERE operation_id=?',(operation_id,)).fetchone()
                if row and row[0]=='reserved':
                    connection.execute('DELETE FROM entries WHERE operation_id=?',(operation_id,));self._update_total(connection,row[2],reserved=-_money(row[1]))
                self._event(connection,{'operation_id':operation_id,'status':'confirmed_unbilled','at':timestamp()})
                connection.execute('COMMIT')
            except BaseException:
                connection.execute('ROLLBACK');raise

    def record_resource(self,operation_id: str,*,category: str,rate_usd: Any,units: Any,metadata: dict) -> dict:
        amount=_money(rate_usd)*_money(units)
        self.reserve(operation_id,amount,category=category);return self.commit(operation_id,amount,usage=metadata)

    def snapshot_database(self,destination: str | Path) -> Path:
        """Create a coherent atomic SQLite backup, including committed WAL state."""
        target=Path(destination);target.parent.mkdir(parents=True,exist_ok=True)
        descriptor,name=tempfile.mkstemp(prefix='.'+target.name+'.',dir=target.parent);os.close(descriptor);temporary=Path(name)
        try:
            with self._database() as connection:
                backup=sqlite3.connect(temporary)
                try:connection.backup(backup)
                finally:backup.close()
            with temporary.open('rb') as handle:os.fsync(handle.fileno())
            os.replace(temporary,target);fsync_directory(target.parent);return target
        finally:
            if temporary.exists():temporary.unlink()

def estimate_judge_cost(usage: dict,prices: dict) -> Decimal:
    if set(('input_tokens','output_tokens'))-usage.keys():raise SpendingBlocked('provider token usage missing; cost cannot be invented')
    for field in ('input_tokens','output_tokens'):
        if type(usage[field]) is not int or usage[field]<0:raise SpendingBlocked('invalid provider usage')
    return (_money(usage['input_tokens'])*_money(prices['input_per_million'])+_money(usage['output_tokens'])*_money(prices['output_per_million']))/Decimal(1000000)

def request_forecast() -> dict:
    return {'evaluation_judge_requests_before_retries':37200,'rl_reward_requests_before_retries':12800,
            'native_capability_generations':76446,'baseline_diagnostic_generations':4932}
