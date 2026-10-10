"""Cross-owner frozen source/recipes/queues; model and provider outcomes are fixtures."""
from copy import deepcopy
import pytest
from shared_procedures.configuration import configuration_hash
from data_generation_collection.rl_data.rl_data_preparation import prepare_rl_data
from training.rl_grpo.setup.rl_setup import RLRecipe, MODELS as RL_MODELS, validate_rl_manifest
from training.rl_grpo.rl_training import run_rl_training
from training.sft.sft_training import build_worklist
from training.sft.individual_fit.sft_fit import SFTRecipe, MODELS as SFT_MODELS, SFTInputError
from tests.unit.training.sft.test_sft_training import fixture_config,fixture_records
from tests.unit.training.rl_grpo.setup.test_rl_setup import FixtureRuntime
from tests.unit.training.rl_grpo.training_cycle.rewards.sonnet_judgment.test_sonnet_opacity_judgment import FixtureJudge


def test_preparation_handoff_is_same_800_frozen_order_for_both_rl_runs(tmp_path):
    rows=[{'id':str(i),'question':f'Fixture arithmetic prompt {i}?','answer':f'Reference derivation #### {i}'} for i in range(900)]
    source=prepare_rl_data(rows,'a'*40,[],output_path=tmp_path/'rl-manifest.json')
    assert validate_rl_manifest(source)==source['questions']
    result=run_rl_training(source,tmp_path/'rl-runs',judge=FixtureJudge(),
        model_revisions={model:'a'*40 for model in RL_MODELS},runtime_identity={'fixture':True},
        runtime_factory=FixtureRuntime,execution_mode='fixture',stop_requested=lambda:True)
    assert result['status']=='paused' and result['runs'][0]['completed_steps']==0
    assert result['execution_mode']=='fixture'
    changed=deepcopy(source)
    changed['questions'][0],changed['questions'][1]=changed['questions'][1],changed['questions'][0]
    with pytest.raises(ValueError):validate_rl_manifest(changed)


def test_all40_sft_work_items_plus_two_rl_recipes_keep_independent_original_bases(monkeypatch):
    import training.sft.sft_training as scheduling
    # Explicit full-count synthetic records test the real worklist validation, not GPU fits.
    config=fixture_config()
    monkeypatch.setattr(scheduling,'load_dataset_bundle',fixture_records)
    bundles=[{'family':family,'model_id':size,'level':level,'status':'verified','config_hash':configuration_hash(config)}
             for family in ('rot13','kazitext') for size in SFT_MODELS for level in range(1,11)]
    worklist=build_worklist(config,bundles)
    assert len(worklist)==40 and len({task['run_id'] for task in worklist})==40
    assert set(task['spec']['model_name'] for task in worklist)==set(RL_MODELS)
    assert all(task['initialization']=='untouched_base_or_exact_full_checkpoint' for task in worklist)
    assert {task['spec']['recipe']['epochs'] for task in worklist}=={2}
    assert {task['spec']['recipe']['lora_rank'] for task in worklist}=={8}
    rl_recipe=RLRecipe().to_dict()
    assert rl_recipe['lora_rank']==32 and rl_recipe['logical_steps']==100
    assert rl_recipe['questions_per_step']==8 and rl_recipe['completions_per_question']==4
    assert SFTRecipe().to_dict()['learning_rate']==2e-4 and rl_recipe['learning_rate']==5e-6
    corrupt=deepcopy(bundles)
    corrupt[-1]['model_id']='7b'
    with pytest.raises(SFTInputError):build_worklist(config,corrupt)
