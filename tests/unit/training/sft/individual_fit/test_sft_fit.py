"""Declared CPU fixtures test real loss, gradients and exact recovery, not a GPU fit."""
import copy
import hashlib
from types import SimpleNamespace

import pytest

from training.sft.individual_fit.sft_fit import (
    LOSS_REDUCTION, SFTFitSpec, SFTInputError, SFTRecipe, capture_rng,
    collate_completion, completion_loss, dataset_hash, exposure_order,
    restore_rng, tokenize_completion, train_optimizer_updates, validate_datasets,
    validation_loss, verify_mask_gradients,
)


class CharacterTokenizer:
    pad_token_id = 0
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False and add_generation_prompt is False
        return '<bos><user>' + messages[0]['content'] + '<assistant>' + messages[1]['content'] + '<eos>'
    def __call__(self, text, **kwargs):
        assert kwargs['truncation'] is False
        return {'input_ids': [ord(c) for c in text], 'offset_mapping': [(i, i + 1) for i in range(len(text))]}


def record(identifier='example', split='train'):
    return {'example_id': identifier, 'prompt': 'Question?', 'completion': '<think>alpha beta gamma</think>42',
            'reasoning': 'alpha beta gamma', 'final_answer': '42', 'split': split,
            'model_id': '7b', 'method': 'rot13', 'level': 1, 'clean_trace_hash': '1' * 64,
            'seed': 42, 'status': 'verified', 'source_content_hash':hashlib.sha256(b'Question?').hexdigest(),
            'source_manifest_hash':'2'*64,'model_revision':'a'*40,'tokenizer_revision':'b'*40,'config_hash':'c'*64}


def spec(data_hash='1' * 64):
    return SFTFitSpec('rot13', '7b', 1, 'a' * 40, 'b' * 40, 'c' * 64, data_hash)


def test_recipe_is_frozen_and_independent_identity():
    SFTRecipe().validate()
    with pytest.raises(SFTInputError):
        SFTRecipe(epochs=3).validate()
    with pytest.raises(SFTInputError):
        SFTFitSpec('rot13', '7b', 1, 'main', 'main', 'a' * 64, 'b' * 64)
    assert spec().run_id != SFTFitSpec('rot13', '7b', 2, 'a'*40, 'b'*40, 'c'*64, '1'*64).run_id
    assert spec().identity()['recipe']['target_modules'] == ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']


def test_full_assistant_target_prompt_and_padding_masks():
    tokenizer = CharacterTokenizer()
    first = tokenize_completion(record(), tokenizer)
    shorter = tokenize_completion({**record('short'), 'prompt': 'Q'}, tokenizer)
    prefix_count = len('<bos><user>Question?<assistant>')
    assert first['labels'][:prefix_count] == [-100] * prefix_count
    assert first['labels'][prefix_count:] == first['input_ids'][prefix_count:]
    assert first['supervised_token_count'] == len(record()['completion'] + '<eos>')
    batch = collate_completion([first, shorter], 0)
    assert batch['labels'][1, -1].item() == -100
    assert batch['attention_mask'][1, -1].item() == 0
    assert LOSS_REDUCTION == 'mean_supervised_tokens_per_example_then_mean_examples'


def test_length_and_boundary_fail_closed_without_truncation():
    with pytest.raises(SFTInputError, match='truncation is forbidden'):
        tokenize_completion(record(), CharacterTokenizer(), max_length=10)
    class CrossingTokenizer(CharacterTokenizer):
        def __call__(self, text, **kwargs):
            boundary = text.index('<think>')
            return {'input_ids': [1], 'offset_mapping': [(boundary - 1, boundary + 1)]}
    with pytest.raises(SFTInputError, match='crosses'):
        tokenize_completion(record(), CrossingTokenizer())
    class StrippingTemplate(CharacterTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return super().apply_chat_template(messages, **kwargs).replace('<think>', '')
    preserved = tokenize_completion(record(), StrippingTemplate())
    assert preserved['supervised_token_count'] == len(record()['completion'] + '<eos>')


def test_masked_cross_entropy_and_zero_prompt_logit_gradients():
    import torch
    import torch.nn.functional as F
    torch.manual_seed(11)
    logits = torch.randn(2, 5, 4, requires_grad=True)
    labels = torch.tensor([[-100,-100,2,1,3],[-100,-100,-100,0,-100]])
    actual = completion_loss(logits, labels)
    expected = (F.cross_entropy(logits[0, 1:4], torch.tensor([2,1,3])) + F.cross_entropy(logits[1, 2:3], torch.tensor([0]))) / 2
    assert torch.allclose(actual, expected)
    actual.backward()
    assert logits.grad[:, 0].abs().sum().item() == 0
    assert logits.grad[1, :2].abs().sum().item() == 0
    assert logits.grad[0, 1:4].abs().sum().item() > 0
    assert logits.grad[:, 4].abs().sum().item() == 0
    with pytest.raises(SFTInputError):
        completion_loss(logits, torch.full((2,5), -100))


def test_frozen_counts_hashes_and_partition_integrity():
    train = [record(f'train_{i}') for i in range(10000)]
    heldout = [record(f'validation_{i}', 'validation') for i in range(100)]
    identity = spec(dataset_hash(train, heldout))
    assert validate_datasets(identity, train, heldout)['train_count'] == 10000
    with pytest.raises(SFTInputError, match='exactly'):
        validate_datasets(identity, train[:-1], heldout)
    duplicate = copy.deepcopy(train)
    duplicate[0] = duplicate[1]
    with pytest.raises(SFTInputError, match='Duplicate'):
        validate_datasets(identity, duplicate, heldout)
    corrupt = copy.deepcopy(train)
    corrupt[0]['final_answer'] = '43'
    with pytest.raises(SFTInputError, match='exact final'):
        validate_datasets(identity, corrupt, heldout)
    modified = copy.deepcopy(train)
    modified[0]['prompt'] = 'Different'
    modified[0]['source_content_hash'] = hashlib.sha256(b'Different').hexdigest()
    with pytest.raises(SFTInputError, match='Actual'):
        validate_datasets(identity, modified, heldout)
    mixed = copy.deepcopy(train)
    mixed[0]['source_manifest_hash'] = '3'*64
    with pytest.raises(SFTInputError, match='one immutable jointly accepted'):
        validate_datasets(spec(dataset_hash(mixed, heldout)), mixed, heldout)


def test_fault_restart_budget_survives_seed_replay_and_paused_continuations(tmp_path,monkeypatch):
    from training.sft.individual_fit.sft_fit import _record_fit_attempt
    from shared_procedures.recovery_procedure import RecoveryManager, RecoveryStopped
    from shared_procedures.storage import atomic_write_json, read_json
    output=tmp_path/'fit'
    output.mkdir()
    ledger=tmp_path/'worker_retries.json'
    monkeypatch.setenv('ILLEGIBILITY_WORKER_RECOVERY_PATH',str(ledger))
    identity=spec().identity()
    _record_fit_attempt(output,identity,None,None)
    for _ in range(3):
        _record_fit_attempt(output,identity,None,None)
    with pytest.raises(RecoveryStopped,match='allowance exhausted'):
        _record_fit_attempt(output,identity,None,None)
    assert RecoveryManager(ledger).counters()['worker_retries']==3
    assert len(read_json(output/'attempts.json')['attempts'])==4
    for _ in range(4):
        atomic_write_json(output/'fit_status.json',{'status':'paused','identity':identity,'progress':{'completed_steps':100}})
        _record_fit_attempt(output,identity,output/'checkpoint_000100',{'completed_steps':100})
    assert RecoveryManager(ledger).counters()['worker_retries']==3
    assert read_json(output/'attempts.json')['attempts'][-1]['voluntary_continuation'] is True
    _record_fit_attempt(output,identity,output/'checkpoint_000100',{'completed_steps':100})
    assert RecoveryManager(ledger).counters()['worker_retries']==4


def tiny_runtime():
    import torch
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(128, 4)
            self.output = torch.nn.Linear(4, 128)
            self.dropout = torch.nn.Dropout(0.05)
        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(logits=self.output(self.dropout(self.embedding(input_ids))))
    torch.manual_seed(42)
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 - step / 10)
    examples = [tokenize_completion(record(str(i)), CharacterTokenizer()) for i in range(16)]
    return model, optimizer, scheduler, examples


def test_two_epoch_exposure_and_validation_cannot_update_weights():
    import torch
    model, optimizer, scheduler, examples = tiny_runtime()
    saved = []
    result = train_optimizer_updates(model, optimizer, scheduler, examples, 0, save=lambda progress:saved.append(progress))
    assert result['example_presentations'] == 32 and result['completed_steps'] == 2
    assert len(saved) == 1 and saved[0]['completed_steps'] == 2
    before = {name: value.clone() for name,value in model.state_dict().items()}
    value = validation_loss(model, examples[:3], 0)
    assert value > 0
    assert all(torch.equal(before[name], value) for name,value in model.state_dict().items())
    order = exposure_order(10000)
    assert len(order) == 20000 and set(order[:10000]) == set(order[10000:]) == set(range(10000))


def test_actual_backward_preflight_restores_rng_and_does_not_update():
    import torch
    model, _, _, examples = tiny_runtime()
    before = {name: value.clone() for name,value in model.state_dict().items()}
    rng_before = torch.get_rng_state().clone()
    evidence = verify_mask_gradients(model, examples[0], 0, require_lora=False)
    assert evidence['status'] == 'passed' and evidence['nonzero_gradient_parameters']
    assert torch.equal(rng_before, torch.get_rng_state())
    assert all(torch.equal(before[name], value) for name,value in model.state_dict().items())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_verified_full_state_resume_matches_uninterrupted_real_optimizer(tmp_path):
    import torch
    from shared_procedures.checkpointing import load_checkpoint, save_checkpoint
    model, optimizer, scheduler, examples = tiny_runtime()
    train_optimizer_updates(model, optimizer, scheduler, examples, 0)
    uninterrupted = copy.deepcopy(model.state_dict())
    model, optimizer, scheduler, examples = tiny_runtime()
    identity = {'fixture':'cpu_actual_optimizer','data_hash':dataset_hash([record()],[])}
    def save(progress):
        save_checkpoint(tmp_path/'checkpoint', identity, {'adapter':model.state_dict(), 'optimizer':optimizer.state_dict(),
            'scheduler':scheduler.state_dict(), 'rng':capture_rng(), 'progress':progress})
    paused = train_optimizer_updates(model, optimizer, scheduler, examples, 0, save=save, should_pause=lambda:True)
    assert paused['status'] == 'paused' and paused['completed_steps'] == 1
    loaded = load_checkpoint(tmp_path/'checkpoint', identity)
    model, optimizer, scheduler, examples = tiny_runtime()
    model.load_state_dict(loaded['adapter'])
    optimizer.load_state_dict(loaded['optimizer'])
    scheduler.load_state_dict(loaded['scheduler'])
    restore_rng(loaded['rng'])
    result = train_optimizer_updates(model, optimizer, scheduler, examples, 0, progress=loaded['progress'])
    assert result['completed_steps'] == 2
    assert all(torch.equal(uninterrupted[name], value) for name,value in model.state_dict().items())
    with pytest.raises(Exception, match='identity'):
        load_checkpoint(tmp_path/'checkpoint', {'fixture':'wrong_fit'})
    invalid = {**loaded['progress'], 'sampler_position':17}
    with pytest.raises(SFTInputError, match='boundary'):
        train_optimizer_updates(model, optimizer, scheduler, examples, 0, progress=invalid)


def test_actual_peft_adapter_publication_binds_full_state_and_detects_corruption(tmp_path):
    """Tiny CPU PEFT fixture; the validator counters are deliberately artificial."""
    import json
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from shared_procedures.checkpointing import save_checkpoint
    from shared_procedures.storage import atomic_write_json
    from training.sft.individual_fit.sft_fit import TARGET_MODULES, _publish_adapter, _verified_checkpoints, verify_fit_result
    tiny = LlamaForCausalLM(LlamaConfig(vocab_size=128,hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=4,max_position_embeddings=256))
    tiny.config._name_or_path = spec().model_name
    tiny.name_or_path = spec().model_name
    model = get_peft_model(tiny,LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,target_modules=list(TARGET_MODULES),task_type='CAUSAL_LM'))
    tokenized = tokenize_completion(record(),CharacterTokenizer())
    assert verify_mask_gradients(model,tokenized,0)['status'] == 'passed'
    class SavingTokenizer:
        def save_pretrained(self,path):
            (path/'tokenizer_config.json').write_text('{"fixture":"not_a_scientific_tokenizer"}')
    # Full state has real tensors; artificial progress exercises completion validation only.
    identity = {**spec().identity(),'fixture':'tiny_peft_cpu_not_scientific_training'}
    progress = {'completed_steps':1250,'sampler_position':20000,'example_presentations':20000}
    state = {'adapter':get_peft_model_state_dict(model,save_embedding_layers=False),'optimizer':{'fixture':True},'scheduler':{'fixture':True},
             'rng':capture_rng(),'progress':progress}
    save_checkpoint(tmp_path/'checkpoint_001250',identity,state)
    manifest = _publish_adapter(tmp_path,model,SavingTokenizer(),identity,None)
    result = {'status':'complete','run_id':spec().run_id,'identity':identity,
        'progress':progress,'validation_loss':1.0,'validation_count':100,'adapter_files':manifest['files'],
        'adapter_manifest_hash':manifest['manifest_hash'],'checkpoint_hash':manifest['checkpoint_hash']}
    atomic_write_json(tmp_path/'fit_result.json',result)
    assert verify_fit_result(tmp_path,identity)['status'] == 'complete'
    atomic_write_json(tmp_path/'fit_result.json',{**result,'checkpoint_hash':'0'*64})
    with pytest.raises(SFTInputError,match='bound'):
        verify_fit_result(tmp_path,identity)
    atomic_write_json(tmp_path/'fit_result.json',result)
    # Valid earlier state remains usable if a higher alleged checkpoint is corrupt.
    damaged = tmp_path/'checkpoint_001251'
    damaged.mkdir()
    (damaged/'manifest.json').write_text('{}')
    assert _verified_checkpoints(tmp_path,identity) == [(1250,tmp_path/'checkpoint_001250')]
    assert (tmp_path/'checkpoint_integrity_failures.json').is_file()
    with (tmp_path/'final_adapter'/'adapter_model.safetensors').open('ab') as handle:
        handle.write(b'corrupt')
    with pytest.raises(SFTInputError,match='corrupt'):
        verify_fit_result(tmp_path,identity)


def test_checkpoint_cadence_at_hundred_and_final_completed_update():
    import torch
    class MicroModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(4,4))
        def forward(self,input_ids,attention_mask):
            return SimpleNamespace(logits=self.weight[input_ids])
    model = MicroModel()
    optimizer = torch.optim.SGD(model.parameters(),lr=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1)
    example = {'input_ids':[0,1,2], 'labels':[-100,1,2]}
    saved = []
    # 808 declared CPU fixture examples, two passes, 101 actual optimizer updates.
    result = train_optimizer_updates(model,optimizer,scheduler,[example]*808,0,save=lambda progress:saved.append(progress['completed_steps']))
    assert result['completed_steps'] == 101 and result['example_presentations'] == 1616
    assert saved == [100,101]
