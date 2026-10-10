import pytest
from shared_procedures.recovery_procedure import *

def test_retries_persist_and_reentry_cannot_reset(tmp_path):
    path=tmp_path/'r.json';r=RecoveryManager(path);calls=[]
    def operation():calls.append(1);raise RetryableFailure('known transient')
    with pytest.raises(RecoveryStopped):r.run('x',operation,sleep=lambda _:None)
    assert len(calls)==4 and r.counters()['operations']['x']==3
    with pytest.raises(RecoveryStopped):RecoveryManager(path).run('x',operation,sleep=lambda _:None)
    assert len(calls)==4

def test_unknown_and_bad_model_outcomes_no_retry(tmp_path):
    r=RecoveryManager(tmp_path/'r.json')
    with pytest.raises(ValueError):r.run('x',lambda:(_ for _ in ()).throw(ValueError('frozen source')))
    assert r.counters()['worker_retries']==0
    assert r.run('ordinary',lambda:{'correct':False})=={'correct':False}

def test_worker_limit_and_nonfinite_configuration(tmp_path):
    with pytest.raises(RecoveryStopped):RecoveryManager(tmp_path/'r.json',max_operation_retries=4)
    r=RecoveryManager(tmp_path/'r.json')
    for i in range(100):r.reserve_retry(str(i))
    with pytest.raises(RecoveryStopped):r.reserve_retry('new')

def test_worker_budget_is_shared_by_judge_storage_assets_and_survives_restart(tmp_path,monkeypatch):
    monkeypatch.setenv('ILLEGIBILITY_WORKER_RECOVERY_PATH',str(tmp_path/'worker.json'))
    monkeypatch.setenv('ILLEGIBILITY_WORKER_ID','generation')
    managers={stage:RecoveryManager.for_worker(tmp_path/stage) for stage in ('judge','storage','assets')}
    calls=[]
    for index in range(33):
        stage=('judge','storage','assets')[index%3]
        def transient(stage=stage):calls.append(stage);raise RetryableFailure('fixture transient',retry_after=0)
        with pytest.raises(RecoveryStopped):managers[stage].run(f'{stage}:run42:operation{index}',transient,sleep=lambda _:None)
    assert len(calls)==132 and managers['judge'].counters()['worker_retries']==99
    with pytest.raises(RecoveryStopped):managers['assets'].run('assets:run42:last',transient,sleep=lambda _:None)
    assert RecoveryManager.for_worker().counters()['worker_retries']==100
    before=len(calls)
    with pytest.raises(RecoveryStopped):RecoveryManager.for_worker().run('storage:new',lambda:calls.append('forbidden'))
    assert len(calls)==before
    # An explicit fixture path remains isolated even while production is bound.
    fixture=RecoveryManager(tmp_path/'fixture.json');fixture.reserve_retry('fixture')
    assert fixture.counters()['worker_retries']==1

def test_legacy_migration_is_atomic_idempotent_and_never_resets_budget(tmp_path,monkeypatch):
    from shared_procedures.storage import atomic_write_json,read_json
    path=tmp_path/'old.json';original={'operations':{'judge:x':2},'worker_retries':2,'started_operations':['judge:x']}
    atomic_write_json(path,original)
    legacy=RecoveryManager(path)
    assert legacy.counters()['worker_retries']==2
    legacy.reserve_retry('judge:x')
    assert read_json(path)==original  # Original audit retained rather than rewritten.
    monkeypatch.setenv('ILLEGIBILITY_WORKER_RECOVERY_PATH',str(tmp_path/'worker.json'))
    assert migrate_worker_recovery([path])['worker_retries']==3
    assert migrate_worker_recovery([path])['worker_retries']==3
    with pytest.raises(RecoveryStopped):RecoveryManager.for_worker().run('judge:x',lambda:None)

def test_blocked_before_attempt_does_not_consume_a_retry(tmp_path):
    manager=RecoveryManager(tmp_path/'worker.json');calls=[]
    def blocked():raise ValueError('quota unavailable before transport')
    for _ in range(2):
        with pytest.raises(ValueError):manager.run('request',lambda:calls.append(True),before_attempt=blocked)
    assert manager.counters()['worker_retries']==0 and manager.counters()['started_operations']==[]
    assert calls==[]

def test_recovery_hot_path_never_exports_or_rewrites_the_full_ledger(tmp_path,monkeypatch):
    manager=RecoveryManager(tmp_path/'indexed.json')
    monkeypatch.setattr(manager,'counters',lambda:(_ for _ in ()).throw(AssertionError('hot path exported all counters')))
    for index in range(2000):assert manager.run(f'operation:{index}',lambda:index)==index
    import sqlite3
    with sqlite3.connect(manager.database_path) as connection:
        assert connection.execute('SELECT COUNT(*) FROM operations').fetchone()[0]==2000
        plan=connection.execute('EXPLAIN QUERY PLAN SELECT retries FROM operations WHERE operation_id=?',('operation:1000',)).fetchall()
        assert any('SEARCH' in row[3] and 'INDEX' in row[3] for row in plan)
    backup=manager.snapshot_database(tmp_path/'backup.sqlite3')
    with sqlite3.connect(backup) as connection:assert connection.execute('PRAGMA integrity_check').fetchone()[0]=='ok'

def test_cross_process_retry_transactions_cannot_overrun_worker_limit(tmp_path):
    import json,subprocess,sys
    script='''import json,sys
from shared_procedures.recovery_procedure import RecoveryManager,RecoveryStopped
r=RecoveryManager(sys.argv[1]);count=0
for i in range(40):
    try:r.reserve_retry(sys.argv[2]+":"+str(i));count+=1
    except RecoveryStopped:break
print(json.dumps({"count":count}))
'''
    path=tmp_path/'concurrent.json'
    processes=[subprocess.Popen([sys.executable,'-c',script,str(path),str(i)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for i in range(4)]
    results=[]
    for process in processes:
        out,error=process.communicate(timeout=45)
        assert process.returncode==0,error
        results.append(json.loads(out)['count'])
    assert sum(results)==100 and RecoveryManager(path).counters()['worker_retries']==100
