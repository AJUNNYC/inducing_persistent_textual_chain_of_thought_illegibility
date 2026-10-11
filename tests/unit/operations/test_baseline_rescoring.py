"""Repair guards use isolated fixtures; these tests are not scientific evidence."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[3] / 'scripts' / 'rescore_vllm_baselines.py'
SPEC = importlib.util.spec_from_file_location('baseline_rescoring_guards', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def measurement_fixture():
    return {
        'generation': {'version': 'frozen', 'settings': {'max_new_tokens': 3072}},
        'prompts': {'gsm8k': 'A frozen public prompt'},
        'prompt_hashes': {'gsm8k': 'a' * 64},
        'panel_hash': 'b' * 64, 'legibility_ids': ['one'],
        'sandbox': {'identity': 'restricted-image'},
        'judges': {'primary': 'frozen-model'}, 'opacity': {'primary_score_floor': 6},
        'code_hashes': {
            'evaluation/shared_model_evaluation/item_evaluation/capability_scoring/capability_scoring.py': 'c' * 64,
            'evaluation/shared_model_evaluation/item_evaluation/native_output/vllm_generation.py': 'd' * 64,
            'shared_procedures/response_parsing.py': 'e' * 64,
        },
    }


@pytest.mark.parametrize('field, value', [
    ('generation', {'version': 'frozen', 'settings': {'max_new_tokens': 1024}}),
    ('prompts', {'gsm8k': 'A different public prompt'}),
])
def test_answer_repair_cannot_reuse_outputs_with_changed_inference_contract(field, value):
    original = measurement_fixture()
    changed = deepcopy(original)
    changed[field] = value
    with pytest.raises(ValueError, match='may not change'):
        MODULE.validate_scoring_only_revision(original, changed)


def test_generation_source_change_cannot_hide_behind_unchanged_declared_version():
    original = measurement_fixture()
    changed = deepcopy(original)
    changed['code_hashes']['evaluation/shared_model_evaluation/item_evaluation/native_output/vllm_generation.py'] = 'f' * 64
    with pytest.raises(ValueError, match='outside the final-field repair'):
        MODULE.validate_scoring_only_revision(original, changed)


def test_missing_raw_cache_cannot_silently_regenerate_or_load_engine(tmp_path):
    generator = MODULE.CacheOnlyGenerator(tmp_path)
    state = {'state_id':'base_7b','model':'7b','family':'base','model_revision':'a'*40,
             'tokenizer_revision':'b'*40,'checkpoint_hash':'c'*64,'runtime_hash':'d'*64}
    item = {'question_id':'q1','domain':'gsm8k','question':'A public fixture question'}
    with pytest.raises(RuntimeError, match='cannot generate a missing saved response'):
        generator.generate(state, item)
    assert generator.engine is None
    assert not list(tmp_path.rglob('*.json'))
