import pytest
from shared_procedures.cost_accounting import *

def authorized():return {'authorized':True,'authorized_usd':10,'authorization_evidence':'account approval test fixture','scheduling_stop_fraction':0.95}

def test_missing_auth_and_overrun_blocked(tmp_path):
    with pytest.raises(SpendingBlocked):CostLedger(tmp_path/'x').reserve('a',1,category='api')
    ledger=CostLedger(tmp_path/'x',authorization=authorized());ledger.reserve('a',9,category='api')
    with pytest.raises(SpendingBlocked):ledger.reserve('b',1,category='gpu')
    assert ledger.totals()['reserved_usd']==9

def test_charge_and_timeout_reservation(tmp_path):
    ledger=CostLedger(tmp_path/'x',authorization=authorized());ledger.reserve('a',1,category='api')
    with pytest.raises(SpendingBlocked):ledger.release('a')
    ledger.commit('a',.5,usage={'tokens':4});assert ledger.totals()['spent_usd']==.5
    with pytest.raises(SpendingBlocked):ledger.commit('a',.6)
    with pytest.raises(SpendingBlocked):ledger.reserve('a',1,category='api')

def test_usage_and_cost_nonfinite_rejected(tmp_path):
    with pytest.raises(SpendingBlocked):estimate_judge_cost({}, {'input_per_million':1,'output_per_million':1})
    ledger=CostLedger(tmp_path/'x',authorization=authorized())
    with pytest.raises(SpendingBlocked):ledger.reserve('a',float('inf'),category='gpu')

def test_warning_and_local_reward_allowance(tmp_path):
    auth=authorized();auth.update(reward_service_allowance_usd=1,warning_fractions=[.75,.8,.9])
    ledger=CostLedger(tmp_path/'x',authorization=auth);ledger.reserve('gpu',7.5,category='gpu')
    assert any(row['status']=='spending_warning' for row in ledger.read()['events'])
    ledger.reserve('a',.8,category='rl_pass_a')
    with pytest.raises(SpendingBlocked):ledger.reserve('b',.3,category='rl_pass_b')

def test_indexed_ledger_import_preserves_uncertain_charges_and_replay(tmp_path):
    from shared_procedures.storage import atomic_write_json,read_json
    path=tmp_path/'legacy.json'
    old={'charges':{'completed':{'amount_usd':'1.25','category':'api','usage':{'tokens':8},'at':'old'}},
         'reservations':{'timeout':{'amount_usd':'2','category':'api','at':'old','status':'reserved'}},'events':[{'status':'old'}]}
    atomic_write_json(path,old)
    ledger=CostLedger(path,authorization=authorized())
    assert ledger.totals()=={'spent_usd':1.25,'reserved_usd':2.0,'committed_usd':3.25}
    assert ledger.commit('completed',1.25,usage={'tokens':8})['at']=='old'
    with pytest.raises(SpendingBlocked):ledger.release('timeout')
    assert CostLedger(path,authorization=authorized()).totals()==ledger.totals()
    assert read_json(path)==old

def test_hot_spending_paths_are_indexed_and_do_not_export_history(tmp_path,monkeypatch):
    ledger=CostLedger(tmp_path/'indexed.json',authorization=authorized())
    monkeypatch.setattr(ledger,'read',lambda:(_ for _ in ()).throw(AssertionError('hot path exported history')))
    for index in range(1000):
        ledger.reserve(str(index),'.001',category='api');ledger.commit(str(index),'.0005',usage={'tokens':1})
    assert ledger.totals()=={'spent_usd':.5,'reserved_usd':0.0,'committed_usd':.5}
    import sqlite3
    with sqlite3.connect(ledger.database_path) as connection:
        assert connection.execute('SELECT COUNT(*) FROM entries').fetchone()[0]==1000
        assert connection.execute('SELECT COUNT(*) FROM events').fetchone()[0]==2000
        assert any('SEARCH' in row[3] and 'INDEX' in row[3] for row in connection.execute('EXPLAIN QUERY PLAN SELECT amount FROM entries WHERE operation_id=?',('500',)))
    backup=ledger.snapshot_database(tmp_path/'backup.sqlite3')
    with sqlite3.connect(backup) as connection:
        assert connection.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        assert connection.execute('SELECT spent FROM totals').fetchone()[0]=='0.5000'

def test_concurrent_process_spending_guard_cannot_exceed_authorization(tmp_path):
    import json,subprocess,sys
    script='''import json,sys
from shared_procedures.cost_accounting import CostLedger,SpendingBlocked
l=CostLedger(sys.argv[1],authorization={"authorized":True,"authorized_usd":10,"authorization_evidence":"fixture","scheduling_stop_fraction":.95});count=0
for i in range(20):
    try:l.reserve(sys.argv[2]+":"+str(i),.5,category="api");count+=1
    except SpendingBlocked:break
print(json.dumps({"count":count}))
'''
    path=tmp_path/'concurrent.json'
    processes=[subprocess.Popen([sys.executable,'-c',script,str(path),str(i)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for i in range(4)]
    counts=[]
    for process in processes:
        out,error=process.communicate(timeout=45)
        assert process.returncode==0,error
        counts.append(json.loads(out)['count'])
    assert sum(counts)==19
    assert CostLedger(path,authorization=authorized()).totals()['committed_usd']==9.5
