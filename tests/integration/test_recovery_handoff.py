"""Actual CPU storage/optimizer restart handoff, distinct from paid GPU preflight."""
import random
from pathlib import Path
import pytest
import torch
from shared_procedures.checkpointing import save_checkpoint,load_checkpoint,verify_checkpoint,CheckpointError
from shared_procedures.execution_state import ExecutionState,StateError
from shared_procedures.recovery_procedure import RecoveryManager,RecoveryStopped,RetryableFailure
from shared_procedures.storage import read_json,atomic_write_json
from shared_procedures.identity import content_hash

def test_partial_optimizer_rollback_keeps_original_steps_and_persistent_attempts(tmp_path):
    identity={'model':'cpu_fixture','config_hash':content_hash({'fixture':True}),'source_hash':content_hash([1,2])}
    model=torch.nn.Linear(1,1);optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,1,.9)
    x=torch.ones((1,1))
    def step():
        optimizer.zero_grad();model(x).square().mean().backward();optimizer.step();scheduler.step()
    step()
    original={k:v.clone() for k,v in model.state_dict().items()}
    manifest=save_checkpoint(tmp_path/'checkpoint',identity,{'adapter':model.state_dict(),'optimizer':optimizer.state_dict(),
        'scheduler':scheduler.state_dict(),'rng':{'torch_cpu':torch.get_rng_state()},'progress':{'completed_steps':1,'sampler_position':1}})
    worker=ExecutionState(tmp_path/'job.json',identity);worker.transition('running');worker.update_progress(1)
    manager=RecoveryManager(tmp_path/'retry.json')
    step()  # Simulate mutation interrupted before completion publication.
    assert any(not torch.equal(model.state_dict()[key],value) for key,value in original.items())
    manager.reserve_retry('logical_update_2')
    restored=load_checkpoint(tmp_path/'checkpoint',identity)
    model.load_state_dict(restored['adapter']);optimizer.load_state_dict(restored['optimizer']);scheduler.load_state_dict(restored['scheduler'])
    torch.set_rng_state(restored['rng']['torch_cpu'])
    assert all(torch.equal(model.state_dict()[key],value) for key,value in original.items())
    assert worker.read()['completed_steps']==1
    assert RecoveryManager(tmp_path/'retry.json').counters()['operations']['logical_update_2']==1
    step();worker.update_progress(2);worker.transition('paused')
    reopened=ExecutionState(tmp_path/'job.json',identity)
    assert reopened.read()['completed_steps']==2 and reopened.read()['attempts']==1
    with pytest.raises(StateError):ExecutionState(tmp_path/'job.json',{**identity,'model':'other'})

def test_incomplete_checkpoint_rejected_without_reset_or_output_overwrite(tmp_path):
    path=tmp_path/'checkpoint';path.mkdir();atomic_write_json(path/'manifest.json',{'status':'complete','identity':{'run':'x'}})
    with pytest.raises(CheckpointError):verify_checkpoint(path,{'run':'x'})
    assert read_json(path/'manifest.json')['identity']=={'run':'x'}
    manager=RecoveryManager(tmp_path/'retry.json')
    calls=[]
    def unavailable():calls.append(1);raise RetryableFailure('known transport fault')
    with pytest.raises(RecoveryStopped):manager.run('fetch',unavailable,sleep=lambda _:None)
    with pytest.raises(RecoveryStopped):RecoveryManager(tmp_path/'retry.json').run('fetch',unavailable,sleep=lambda _:None)
    assert len(calls)==4
