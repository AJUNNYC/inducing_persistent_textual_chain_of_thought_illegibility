"""Focused CPU checks for real AdamW recovery, schedule and epoch2 serialization."""
import copy
from types import SimpleNamespace

import pytest
import torch

from shared_procedures.identity import content_hash
from shared_procedures.checkpointing import _same_state
from training.sft import continuation as c
from training.sft.individual_fit.sft_fit import SFTInputError, exposure_order


def initialized_optimizer():
    parameter = torch.nn.Parameter(torch.tensor([0.4, -0.2]))
    optimizer = torch.optim.AdamW([parameter], lr=2e-4, weight_decay=0)
    for _ in range(4):
        parameter.grad = torch.tensor([0.3, -0.1])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    optimizer.param_groups[0]['lr'] = 0.0
    return parameter, optimizer


def test_schedule_restarts_nonzero_preserving_real_adamw_moments():
    _, parent = initialized_optimizer()
    saved = copy.deepcopy(parent.state_dict())
    parameter = torch.nn.Parameter(torch.tensor([0.4, -0.2]))
    optimizer = torch.optim.AdamW([parameter], lr=0.7)
    scheduler = c.restore_optimizer_scheduler(optimizer, {'optimizer': saved}, continuing_phase=False)
    assert _same_state(optimizer.state_dict()['state'], saved['state'])
    assert scheduler.last_epoch == 0
    assert optimizer.param_groups[0]['lr'] == pytest.approx(2e-4 / 19)
    assert c.learning_rate_scale(18) == c.learning_rate_scale(19) == 1
    assert c.learning_rate_scale(625) == 0
    assert all(0 < c.learning_rate_scale(i) <= 1 for i in range(625))


def test_scheduler_resume_preserves_next_adamw_update_exactly():
    parameter, parent = initialized_optimizer()
    scheduler = c.restore_optimizer_scheduler(parent, {'optimizer': copy.deepcopy(parent.state_dict())},
                                               continuing_phase=False)
    for _ in range(23):
        parameter.grad = torch.tensor([0.2, -0.4])
        parent.step(); scheduler.step(); parent.zero_grad(set_to_none=True)
    state = {'optimizer': copy.deepcopy(parent.state_dict()),
             'scheduler': copy.deepcopy(scheduler.state_dict()), 'progress': {'completed_steps': 23}}
    restored_parameter = torch.nn.Parameter(parameter.detach().clone())
    restored = torch.optim.AdamW([restored_parameter], lr=0.9)
    restored_scheduler = c.restore_optimizer_scheduler(restored, state, continuing_phase=True)
    for p, opt, sched in ((parameter, parent, scheduler), (restored_parameter, restored, restored_scheduler)):
        p.grad = torch.tensor([0.6, -0.7]);opt.step();sched.step()
    assert torch.equal(parameter, restored_parameter)
    assert _same_state(parent.state_dict(), restored.state_dict())
    assert scheduler.state_dict() == restored_scheduler.state_dict()


def test_empty_moments_and_wrong_resume_cursor_rejected():
    p = torch.nn.Parameter(torch.tensor([1.0]));o = torch.optim.AdamW([p], lr=2e-4)
    with pytest.raises(SFTInputError, match='initialized AdamW'):
        c.restore_optimizer_scheduler(o, {'optimizer':o.state_dict()}, continuing_phase=False)
    _, o = initialized_optimizer();s = c.restart_scheduler(o)
    state = {'optimizer':copy.deepcopy(o.state_dict()),'scheduler':s.state_dict(),'progress':{'completed_steps':4}}
    with pytest.raises(SFTInputError, match='scheduler/LR'):
        c.restore_optimizer_scheduler(o,state,continuing_phase=True)


class CharacterTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        prefix = '<bos><user>' + messages[0]['content'] + '<assistant>'
        return prefix + ('<think>\n' if add_generation_prompt else messages[1]['content'] + '<eos>')
    def encode(self,text,add_special_tokens=False):
        return [ord(char) for char in text]
    def __call__(self,text,**kwargs):
        assert kwargs['truncation'] is False
        return {'input_ids':[ord(char) for char in text],
                'offset_mapping':[(i,i+1) for i in range(len(text))]}


def test_derived_newline_preserves_target_bytes_and_original_mask_boundary():
    record={'example_id':'a','prompt':'Question?','reasoning':'Ny cun orgn','final_answer':'42',
            'completion':'<think>Ny cun orgn</think>42','transformed_token_count':1}
    original=copy.deepcopy(record)
    x=c.tokenize_continuation(record,CharacterTokenizer())
    prefix='<bos><user>Question?<assistant>'
    assert ''.join(chr(i) for i in x['input_ids']) == prefix+'<think>\nNy cun orgn</think>42<eos>'
    assert x['labels'][:len(prefix)]==[-100]*len(prefix)
    assert x['labels'][len(prefix):]==x['input_ids'][len(prefix):]
    assert record==original
    oversized={**record,'reasoning':'a'*4096,'completion':'<think>'+'a'*4096+'</think>42'}
    with pytest.raises(SFTInputError,match='truncation is forbidden'):
        c.tokenize_continuation(oversized,CharacterTokenizer())


def test_epoch2_pause_resume_does_not_repeat_epoch1_or_completed_update(monkeypatch):
    order=exposure_order(10000,42,epochs=2)[10000:]
    identity={'exposure_order_hash':content_hash(order)}
    runtime={'physical_microbatch':1,'gradient_accumulation':16,'gradient_checkpointing':False}
    examples=[{'example_id':i,'input_ids':[i]} for i in range(10000)]
    scheduler=SimpleNamespace(last_epoch=0);seen=[];saved=[]
    def update(_model,_optimizer,sched,group,*args,**kwargs):
        seen.append([r['example_id'] for r in group]);sched.last_epoch+=1
    monkeypatch.setattr(c,'one_optimizer_update',update)
    model=SimpleNamespace(train=lambda:None)
    result=c.train_additional_epoch(model,None,scheduler,examples,0,identity=identity,runtime=runtime,
        recipe=None,save=lambda x:saved.append(copy.deepcopy(x)),should_pause=lambda:len(seen)>=1)
    assert result['status']=='paused';assert seen==[order[:16]]
    assert saved[-1]['cumulative_completed_steps']==626
    first=saved[-1]
    result=c.train_additional_epoch(model,None,scheduler,examples,0,identity=identity,runtime=runtime,
        recipe=None,progress=first,save=lambda x:saved.append(copy.deepcopy(x)),
        should_pause=lambda:len(seen)>=2)
    assert seen==[order[:16],order[16:32]]
    assert result['cumulative_completed_steps']==627


def test_completed_fit_republish_preserves_bytes_and_never_loads_gpu(tmp_path, monkeypatch):
    import json
    from safetensors.torch import save_file
    from shared_procedures.storage import atomic_write_json
    from shared_procedures.checkpointing import save_checkpoint
    from training.sft.individual_fit.sft_fit import TARGET_MODULES
    output=tmp_path/'continuation';output.mkdir()
    identity={'run_id':'fixture_extra_epoch','model_name':'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B',
              'configuration_hash':'a'*64}
    progress={'completed_steps':625,'sampler_position':10000,'cumulative_completed_steps':1250}
    weights={'lora_test':torch.ones(2)}
    state={'adapter':weights,'optimizer':{'fixture':'moments'},'scheduler':{},
           'rng':{'fixture':'rng'},'progress':progress}
    checkpoint=save_checkpoint(output/'checkpoint_000625',identity,state)
    adapter=output/'final_adapter';adapter.mkdir()
    save_file(weights,str(adapter/'adapter_model.safetensors'))
    atomic_write_json(adapter/'adapter_config.json',{'r':8,'lora_alpha':16,'lora_dropout':0.05,
        'target_modules':list(TARGET_MODULES),'base_model_name_or_path':identity['model_name']})
    files={p.name:c.file_hash(p) for p in adapter.iterdir()}
    manifest={'identity':identity,'files':files,'checkpoint_hash':checkpoint['checkpoint_hash']}
    manifest['manifest_hash']=content_hash(manifest)
    atomic_write_json(adapter/'manifest.json',manifest)
    result={'status':'complete','identity':identity,'progress':{**progress,'status':'complete'},
        'checkpoint_hash':checkpoint['checkpoint_hash'],'adapter_manifest_hash':manifest['manifest_hash'],
        'adapter_files':files,'validation_count':100,'validation_loss':0.7,
        'elapsed_training_seconds_this_invocation':42.0}
    atomic_write_json(output/'fit_result.json',result)
    before=(output/'fit_result.json').read_bytes()
    monkeypatch.setattr(c,'prepare_inputs',lambda *args,**kwargs:(identity,{},[],[]))
    def forbidden(*args,**kwargs):
        raise AssertionError('completed fit must not initialize GPU/runtime')
    monkeypatch.setattr(c,'_runtime',forbidden)
    observed=c.run_continuation(tmp_path/'parent',tmp_path/'parent_checkpoint',[],[],output,
        source_revision='a'*40,cache_root=tmp_path/'cache')
    assert observed==result
    assert (output/'fit_result.json').read_bytes()==before
    assert json.loads((output/'fit_status.json').read_text())['status']=='complete'
