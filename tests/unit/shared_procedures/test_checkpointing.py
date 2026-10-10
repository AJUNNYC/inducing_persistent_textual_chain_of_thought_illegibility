import copy
import random
import pytest
import numpy as np
import torch
from shared_procedures.checkpointing import *

def state():
    return {'adapter':{'weight':torch.tensor([1.,2.])},'optimizer':{'state':{0:{'exp_avg':torch.ones(2)}}},
            'scheduler':{'last_epoch':3},'rng':{'python':random.getstate(),'numpy':np.random.get_state(),
            'torch_cpu':torch.get_rng_state()},'progress':{'completed_steps':3,'sampler_position':48}}

def test_full_state_roundtrip_identity_and_corruption(tmp_path):
    directory=tmp_path/'step';identity={'model':'7b','config_hash':'a','data_hash':'b'}
    manifest=save_checkpoint(directory,identity,state());assert manifest['progress']['completed_steps']==3
    loaded=load_checkpoint(directory,identity);assert torch.equal(loaded['adapter']['weight'],torch.tensor([1.,2.]))
    assert loaded['scheduler']['last_epoch']==3
    with pytest.raises(CheckpointError):load_checkpoint(directory,{'model':'14b'})
    with (directory/'full_state.pt').open('ab') as f:f.write(b'corrupt')
    with pytest.raises(CheckpointError):load_checkpoint(directory,identity)

def test_adapter_only_and_incomplete_progress_rejected(tmp_path):
    with pytest.raises(CheckpointError):save_checkpoint(tmp_path/'a',{'run':'x'},{'adapter':{}})
    s=state();del s['progress']['sampler_position']
    with pytest.raises(CheckpointError):save_checkpoint(tmp_path/'b',{'run':'x'},s)

def test_actual_optimizer_resume_matches_uninterrupted(tmp_path):
    model=torch.nn.Linear(2,1);optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,1,.99)
    x=torch.tensor([[1.,2.]]);target=torch.tensor([[3.]])
    def step(m,o,s):o.zero_grad();((m(x)-target)**2).mean().backward();o.step();s.step()
    step(model,optimizer,scheduler)
    snapshot={'adapter':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
              'rng':{'torch_cpu':torch.get_rng_state()},'progress':{'completed_steps':1,'sampler_position':1}}
    save_checkpoint(tmp_path/'c',{'run':'fixture'},snapshot)
    step(model,optimizer,scheduler)
    resumed=torch.nn.Linear(2,1);opt=torch.optim.AdamW(resumed.parameters(),lr=.01);sch=torch.optim.lr_scheduler.StepLR(opt,1,.99)
    saved=load_checkpoint(tmp_path/'c',{'run':'fixture'});resumed.load_state_dict(saved['adapter']);opt.load_state_dict(saved['optimizer']);sch.load_state_dict(saved['scheduler'])
    step(resumed,opt,sch)
    for a,b in zip(model.parameters(),resumed.parameters()):assert torch.equal(a,b)

def test_latest_matches_identity_and_retains_two(tmp_path):
    for index in (1,2,3):
        s=state();s['progress']['completed_steps']=index;save_checkpoint(tmp_path/f'step_{index}',{'run':'a'},s)
    s=state();s['progress']['completed_steps']=100;save_checkpoint(tmp_path/'later_other',{'run':'b'},s)
    assert latest_checkpoint(tmp_path,{'run':'a'}).name=='step_3'
    retain_checkpoints(tmp_path,{'run':'a'});assert not (tmp_path/'step_1').exists()
    assert (tmp_path/'later_other').exists()

def test_idempotent_exact_state_republish_and_mutation_guard(tmp_path):
    original=state();manifest=save_checkpoint(tmp_path/'c',{'run':'a'},original)
    assert save_checkpoint(tmp_path/'c',{'run':'a'},original)==manifest
    changed=copy.deepcopy(original);changed['adapter']['weight'][0]=9
    with pytest.raises(CheckpointError):save_checkpoint(tmp_path/'c',{'run':'a'},changed)

def test_checkpoint_replication_payload_before_manifest(tmp_path):
    import shared_procedures.storage as storage
    files=[]
    def hook(path,metadata):files.append(path.name)
    storage.register_artifact_hook(hook)
    try:save_checkpoint(tmp_path/'c',{'run':'a'},state())
    finally:storage._ARTIFACT_HOOKS.remove(hook)
    assert files==['full_state.pt','manifest.json']

def test_no_verified_recovery_cannot_silently_fresh_start(tmp_path):
    directory=tmp_path/'claimed';directory.mkdir()
    from shared_procedures.storage import atomic_write_json
    atomic_write_json(directory/'manifest.json',{'identity':{'run':'a'},'status':'complete'})
    with pytest.raises(CheckpointError):latest_checkpoint(tmp_path,{'run':'a'})
