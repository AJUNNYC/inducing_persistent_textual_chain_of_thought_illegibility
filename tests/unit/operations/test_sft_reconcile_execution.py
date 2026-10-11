"""Synthetic guard fixtures do not certify a hardware optimization."""
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest


SCRIPT = Path(__file__).resolve().parents[3] / 'scripts' / 'reconcile_sft_launch.py'
SPEC = importlib.util.spec_from_file_location('sft_reconcile_execution_guards', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


ORIGINAL_NUMERICS = '''
def completion_loss(logits, labels):
    shifted_logits, shifted_labels = logits[:, :-1, :].float(), labels[:, 1:]
    counts = (shifted_labels != -100).sum(dim=1)
    token_losses = functional.cross_entropy(shifted_logits.transpose(1, 2), shifted_labels, ignore_index=-100, reduction="none")
    return (token_losses.sum(dim=1) / counts).mean()

def collate_completion(examples, pad_token_id, device=None):
    for example in examples:
        padding = length - len(example["input_ids"])
        result["input_ids"].append(list(example["input_ids"]) + [pad_token_id] * padding)
        result["attention_mask"].append([1] * len(example["input_ids"]) + [0] * padding)
        result["labels"].append(list(example["labels"]) + [-100] * padding)
'''

GUARDED_NUMERICS = ORIGINAL_NUMERICS.replace(
    'counts = (shifted_labels != -100).sum(dim=1)',
    'if validated_counts is None:\n        counts = (shifted_labels != -100).sum(dim=1)\n'
    '    else:\n        counts = validated_counts.counts').replace(
    'for example in examples:', 'for example in examples:\n        labels = list(example["labels"])').replace(
    'append(list(example["labels"]) + [-100]', 'append(labels + [-100]')

REPLACEMENT_WRAPPER = '''
def profile_completion_loss(logits, labels, execution_profile=None):
    if loss_implementation == 'spatial_ce':
        return completion_loss(logits, labels)
    shifted_logits, shifted_labels = logits[:, :-1, :].float(), labels[:, 1:]
    counts = (shifted_labels != -100).sum(dim=1)
    if loss_implementation == 'flat_ce':
        token_losses = functional.cross_entropy(shifted_logits.reshape(-1, shifted_logits.shape[-1]), shifted_labels.reshape(-1), ignore_index=-100, reduction="none").reshape(shifted_labels.shape)
    else:
        token_losses = fused_token_cross_entropy(shifted_logits.reshape(-1, shifted_logits.shape[-1]), shifted_labels.reshape(-1), ignore_index=-100).reshape(shifted_labels.shape)
    return (token_losses.sum(dim=1) / counts).mean()
'''

SPATIAL_HELPER_FIXTURE = '''
def forward(ctx, logits, targets, ignore_index):
    log_probabilities = torch.log_softmax(logits, dim=1, dtype=logits.dtype)
    losses = functional.nll_loss(log_probabilities, targets, ignore_index=ignore_index, reduction="none")
    return losses

def backward():
    derivative = target_gradient - libdevice.exp(log_probability) * summed_gradient
'''


def set_numerics(tmp_path, monkeypatch, current):
    path = tmp_path / 'src/training/sft/individual_fit/sft_fit.py'
    path.parent.mkdir(parents=True)
    path.write_text(current)
    monkeypatch.setattr(MODULE, 'ROOT', tmp_path)
    monkeypatch.setattr(MODULE, 'original_file', lambda commit, name: ORIGINAL_NUMERICS.encode())


def test_added_count_guards_preserve_original_scientific_loss_and_mask(tmp_path, monkeypatch):
    set_numerics(tmp_path, monkeypatch, GUARDED_NUMERICS)
    bindings = MODULE.compatible_sft_numerics('original-source')
    assert len(bindings) == 7
    assert all(len(value) == 64 for value in bindings.values())


def test_flat_ce_dispatch_preserves_shift_mask_and_per_example_weighting(tmp_path, monkeypatch):
    set_numerics(tmp_path, monkeypatch, GUARDED_NUMERICS + REPLACEMENT_WRAPPER)
    bindings = MODULE.compatible_sft_numerics('original-source', loss_implementation='flat_ce')
    assert 'flat_ce_dispatch' in bindings
    assert bindings['replacement_reduction'] == bindings['reduction']


@pytest.mark.parametrize('before,after', [
    ('/ counts', '/ (counts + 1)'),
    ('ignore_index=-100', 'ignore_index=-1'),
    ('labels[:, 1:]', 'labels[:, :-1]'),
    ('shifted_labels.reshape(-1)', 'shifted_labels.transpose(0, 1).reshape(-1)'),
])
def test_replacement_dispatch_cannot_change_the_loss_contract(tmp_path, monkeypatch, before, after):
    set_numerics(tmp_path, monkeypatch, GUARDED_NUMERICS + REPLACEMENT_WRAPPER.replace(before, after))
    with pytest.raises(ValueError, match='frozen shift|mask|weighting'):
        MODULE.compatible_sft_numerics('original-source', loss_implementation='flat_ce')


def test_fused_dispatch_requires_bound_kernel_module_source(tmp_path, monkeypatch):
    set_numerics(tmp_path, monkeypatch, GUARDED_NUMERICS + REPLACEMENT_WRAPPER)
    kernel = tmp_path / 'src/training/sft/fused_loss.py'
    kernel.write_text('# Declared fixture bytes; not a kernel measurement.\n')
    bindings = MODULE.compatible_sft_numerics('original-source', loss_implementation='fused_ce')
    assert bindings['fused_ce_module'] == MODULE.file_hash(kernel)


def test_original_spatial_forward_with_new_backward_retains_source_proof(tmp_path, monkeypatch):
    wrapper = REPLACEMENT_WRAPPER.replace('    else:\n        token_losses = fused_token_cross_entropy(',
        '    elif loss_implementation == "spatial_fused_backward":\n'
        '        token_losses = spatial_token_cross_entropy(shifted_logits.transpose(1, 2), shifted_labels, ignore_index=-100)\n'
        '    else:\n        token_losses = fused_token_cross_entropy(')
    set_numerics(tmp_path, monkeypatch, GUARDED_NUMERICS + wrapper)
    helper = tmp_path/'src/training/sft/spatial_loss.py'
    helper.write_text(SPATIAL_HELPER_FIXTURE)
    bindings = MODULE.compatible_sft_numerics('original-source', loss_implementation='spatial_fused_backward')
    assert bindings['replacement_reduction'] == bindings['reduction']
    assert bindings['spatial_loss_module'] == MODULE.file_hash(helper)
    assert 'native_spatial_forward' in bindings


@pytest.mark.parametrize('before,after', [
    ('dim=1', 'dim=-1'),
    ('dtype=logits.dtype', 'dtype=torch.bfloat16'),
    ('reduction="none"', 'reduction="mean"'),
    ('ignore_index=ignore_index', 'ignore_index=-1'),
    ('libdevice.exp(log_probability)', 'tl.exp(log_probability)'),
])
def test_spatial_backward_source_proof_rejects_changed_native_contract(tmp_path, monkeypatch, before, after):
    set_numerics(tmp_path, monkeypatch, GUARDED_NUMERICS)
    helper = tmp_path/'src/training/sft/spatial_loss.py'
    helper.write_text(SPATIAL_HELPER_FIXTURE.replace(before, after))
    with pytest.raises(ValueError, match='Spatial loss changed'):
        MODULE.spatial_native_forward_contract()


@pytest.mark.parametrize('before,after', [
    ('/ counts', '/ (counts + 1)'),
    ('ignore_index=-100', 'ignore_index=-1'),
    ('labels[:, 1:]', 'labels[:, :-1]'),
    ('[0] * padding', '[1] * padding'),
    ('[-100] * padding', '[0] * padding'),
])
def test_hardware_evidence_cannot_override_changed_objective_or_masks(tmp_path, monkeypatch, before, after):
    set_numerics(tmp_path, monkeypatch, GUARDED_NUMERICS.replace(before, after))
    with pytest.raises(ValueError, match='mask|cross-entropy|reduction'):
        MODULE.compatible_sft_numerics('original-source')


def acceptance_fixture(tmp_path, monkeypatch):
    config = {'models': {'7b': {}}, 'runtime': {}}
    monkeypatch.setattr(MODULE, 'configuration_hash', lambda value: 'config-hash')
    plan = {'profiles': {'cached_inputs': {'cache_collated_inputs': True}}}
    path = tmp_path / 'report.json'
    path.write_text(json.dumps({'plan': plan}))
    accepted = {'accepted': True, 'configuration_hash': 'config-hash',
                'handoff_certificate_hash': 'certificate-hash',
                'condition': {'family': 'rot13', 'model': '7b', 'level': 1},
                'recovery_validated': True, 'original_checkpoint_hash': 'checkpoint-hash',
                'source_commit': 'source', 'profile_name': 'cached_inputs'}
    validator = types.ModuleType('training.sft.runtime_migration')
    calls = []
    def validate(report, source, profile, **kwargs):
        calls.append((report, source, profile, kwargs))
        return accepted
    validator.validate_execution_benchmark = validate
    monkeypatch.setitem(sys.modules, 'training.sft.runtime_migration', validator)
    return config, path, accepted, calls


def test_accepted_controls_are_scoped_to_the_measured_model_and_condition(tmp_path, monkeypatch):
    config, path, _, calls = acceptance_fixture(tmp_path, monkeypatch)
    accepted, checks = MODULE.accepted_execution_controls(path, 'cached_inputs', config,
        {'certificate_hash': 'certificate-hash'}, 'source')
    assert set(checks) == {'model_runtime', 'loss_mask_gradients', 'full_checkpoint_resume'}
    for row in checks.values():
        assert row.status == 'passed'
        assert row.evidence['models_retested'] == ['7b']
        assert row.evidence['other_models_retested'] is False
        assert row.evidence['condition'] == accepted['condition']
        assert row.evidence['scientific_equivalence_established'] is False
    assert calls[0][2].cache_collated_inputs is True
    assert calls[0][3]['profile_name'] == 'cached_inputs'


@pytest.mark.parametrize('field,value,reason', [
    ('accepted', False, 'configuration or certificate'),
    ('configuration_hash', 'other-config', 'configuration or certificate'),
    ('handoff_certificate_hash', 'other-certificate', 'configuration or certificate'),
    ('condition', {'family': 'rot13', 'model': '14b', 'level': 1}, 'one valid'),
    ('recovery_validated', False, 'paired continuation'),
    ('original_checkpoint_hash', None, 'paired continuation'),
])
def test_current_fit_or_recovery_binding_cannot_be_missing(tmp_path, monkeypatch, field, value, reason):
    config, path, accepted, _ = acceptance_fixture(tmp_path, monkeypatch)
    accepted[field] = value
    with pytest.raises(ValueError, match=reason):
        MODULE.accepted_execution_controls(path, 'cached_inputs', config,
            {'certificate_hash': 'certificate-hash'}, 'source')


def test_configured_profile_must_equal_measured_profile(tmp_path, monkeypatch):
    config, path, _, _ = acceptance_fixture(tmp_path, monkeypatch)
    config['runtime']['sft_execution_profile'] = {'attention_implementation': 'sdpa'}
    with pytest.raises(ValueError, match='differs from the measured'):
        MODULE.accepted_execution_controls(path, 'cached_inputs', config,
            {'certificate_hash': 'certificate-hash'}, 'source')


@pytest.mark.parametrize('options', [
    ['--execution-profile', 'cached_inputs'],
    ['--execution-benchmark', 'report.json'],
    ['--recovery-benchmark', 'recovery.json'],
])
def test_partial_execution_evidence_arguments_are_rejected_before_reading_files(options):
    required = [part for name in ('config', 'certificate', 'control-config', 'baseline-root', 'output')
                for part in ('--'+name, 'unused-fixture')]
    with pytest.raises(SystemExit):
        MODULE.main([*required, *options])
