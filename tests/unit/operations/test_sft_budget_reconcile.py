"""CPU evidence checks: a shorter budget is explicit, not stale GPU approval."""
import ast
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from shared_procedures.configuration import load_configuration

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location('budget_reconcile', ROOT / 'scripts/reconcile_sft_launch.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def budgets():
    legacy = load_configuration(ROOT / 'config.yaml')
    legacy['sft'].pop('training_budget', None)
    legacy['sft'].pop('selection', None)
    legacy['sft']['epochs'] = 2
    pilot = deepcopy(legacy)
    pilot['sft'].update(training_budget='one_epoch_pilot_v1', epochs=1)
    return legacy, pilot


def test_budget_amendment_is_explicit_and_requires_fresh_native_evidence():
    legacy, pilot = budgets()
    result = MODULE.declared_training_budget_change(pilot, legacy)
    assert result['before_budget'] == 'legacy_two_epoch_v1'
    assert result['after_budget'] == 'one_epoch_pilot_v1'
    assert result['after']['selection'] is None
    assert result['other_sft_settings_unchanged'] is True
    assert result['fresh_native_sft_evidence_required'] is True
    assert result['scientific_training_results_equivalent'] is False


def test_legacy_controls_do_not_need_a_fictional_amendment():
    legacy, _ = budgets()
    assert MODULE.declared_training_budget_change(legacy, legacy) is None


def test_budget_amendment_cannot_hide_a_learning_rate_change():
    legacy, pilot = budgets()
    pilot['sft']['learning_rate'] = 0.001
    with pytest.raises(ValueError, match='learning_rate'):
        MODULE.declared_training_budget_change(pilot, legacy)


def test_historical_recipe_field_definitions_remain_exact(monkeypatch):
    source = (ROOT / 'src/training/sft/individual_fit/sft_fit.py').read_bytes()
    monkeypatch.setattr(MODULE, 'original_file', lambda *_: source)
    result = MODULE.compatible_recipe_fields('declared_cpu_fixture')
    assert result['legacy_recipe']['epochs'] == 2
    assert result['pilot_recipe']['epochs'] == 1
    assert result['native_controls_cover_budget_dispatch'] is True
    changed = source.replace(b'epochs: int = 2', b'epochs: int = 3')
    monkeypatch.setattr(MODULE, 'original_file', lambda *_: changed)
    with pytest.raises(ValueError, match='Historical SFT recipe fields'):
        MODULE.compatible_recipe_fields('declared_cpu_fixture')


def test_model_control_normalization_allows_only_budget_argument(monkeypatch):
    path = ROOT / 'src/shared_procedures/preflight.py'
    old = ast.parse(path.read_text())
    for call in ast.walk(old):
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == 'SFTFitSpec':
            call.keywords = [key for key in call.keywords if key.arg != 'training_budget']
    source = ast.unparse(old).encode()
    monkeypatch.setattr(MODULE, 'original_file', lambda *_: source)
    result = MODULE.compatible_model_control_budget('declared_cpu_fixture')
    assert 'original_control_function' in result
    changed = source.replace(b"control_prompt = 'What is two plus two?'", b"control_prompt = 'Different control'")
    assert changed != source
    monkeypatch.setattr(MODULE, 'original_file', lambda *_: changed)
    with pytest.raises(ValueError, match='Original model controls changed'):
        MODULE.compatible_model_control_budget('declared_cpu_fixture')
