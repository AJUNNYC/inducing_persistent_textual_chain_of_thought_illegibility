"""CPU budget/recovery checks; these are not GPU speed or outcome evidence."""
from dataclasses import replace
from types import SimpleNamespace
import sys

import pytest

from training.sft.individual_fit import sft_fit


def spec(budget=sft_fit.LEGACY_TRAINING_BUDGET):
    return sft_fit.SFTFitSpec("rot13", "14b", 10, "a" * 40, "b" * 40,
                           "c" * 64, "d" * 64, training_budget=budget)


def test_named_budgets_preserve_every_other_scientific_recipe_field():
    legacy = sft_fit.SFTRecipe()
    pilot = sft_fit.SFTRecipe.for_budget("one_epoch_pilot_v1")
    assert sft_fit.SFTRecipe.for_budget("legacy_two_epoch_v1") == legacy
    assert pilot == replace(legacy, epochs=1)
    assert legacy.optimizer_steps(10000) == 1250
    assert legacy.warmup_steps(10000) == 38
    assert pilot.optimizer_steps(10000) == 625
    assert pilot.warmup_steps(10000) == 19
    for name in ("one_epoch", "three_epoch_v1", "", None):
        with pytest.raises(sft_fit.SFTInputError, match="named SFT training budget"):
            sft_fit.SFTRecipe.for_budget(name)
    for field, value in (("epochs", True), ("learning_rate", 1e-4),
                         ("max_length", 3072), ("effective_batch_size", 32)):
        with pytest.raises(sft_fit.SFTInputError, match="declared frozen budget"):
            replace(pilot, **{field: value}).validate()


def test_legacy_identity_is_unchanged_and_pilot_budget_is_explicit():
    legacy = spec().identity()
    assert "training_budget" not in legacy
    assert sft_fit.SFTFitSpec.from_dict(legacy).identity() == legacy
    pilot = spec(sft_fit.PILOT_TRAINING_BUDGET).identity()
    assert pilot["training_budget"] == "one_epoch_pilot_v1"
    assert pilot["recipe"]["epochs"] == 1
    assert pilot["recipe"] == {**legacy["recipe"], "epochs": 1}
    assert sft_fit.SFTFitSpec.from_dict(pilot).identity() == pilot
    assert sft_fit._hash(legacy) != sft_fit._hash(pilot)
    with pytest.raises(sft_fit.SFTInputError, match="named training budget"):
        sft_fit._identity_recipe({**pilot, "recipe": legacy["recipe"]})


def test_one_epoch_production_budget_stops_at_625_with_full_frozen_exposure(monkeypatch):
    import torch
    model = torch.nn.Linear(1, 1)
    examples = [{"input_ids": [index], "example_id": index} for index in range(10000)]
    visited, saved = [], []
    def update(model, optimizer, scheduler, group, pad_token_id, **kwargs):
        assert len(group) == 16
        visited.extend(example["example_id"] for example in group)
    monkeypatch.setattr(sft_fit, "one_optimizer_update", update)
    result = sft_fit.train_optimizer_updates(model, None, None, examples, 0,
        recipe=spec(sft_fit.PILOT_TRAINING_BUDGET).recipe,
        save=lambda progress: saved.append(progress["completed_steps"]))
    assert result["status"] == "complete"
    assert result["completed_steps"] == result["total_optimizer_steps"] == 625
    assert result["sampler_position"] == result["example_presentations"] == 10000
    assert saved == [100, 200, 300, 400, 500, 600, 625]
    assert len(visited) == len(set(visited)) == 10000
    assert visited == sft_fit.exposure_order(10000, epochs=1)
    assert visited == sft_fit.exposure_order(10000)[:10000]


def test_epoch_budget_rejects_two_epoch_checkpoint_and_sampler_progress(tmp_path, monkeypatch):
    import torch
    from shared_procedures.checkpointing import load_checkpoint, save_checkpoint
    from shared_procedures.checkpointing import CheckpointError
    model = torch.nn.Linear(1, 1)
    examples = [{"input_ids": [i]} for i in range(16)]
    monkeypatch.setattr(sft_fit, "one_optimizer_update", lambda *args, **kwargs: None)
    legacy_progress = sft_fit.train_optimizer_updates(model, None, None, examples, 0,
        should_pause=lambda: True)
    state = {"adapter": model.state_dict(), "optimizer": {}, "scheduler": {},
             "rng": sft_fit.capture_rng(), "progress": legacy_progress}
    checkpoint = tmp_path / "checkpoint_000001"
    save_checkpoint(checkpoint, spec().identity(), state)
    with pytest.raises(CheckpointError, match="identity"):
        load_checkpoint(checkpoint, spec(sft_fit.PILOT_TRAINING_BUDGET).identity())
    with pytest.raises(sft_fit.SFTInputError, match="sampler/exposure identity"):
        sft_fit.train_optimizer_updates(model, None, None, examples, 0,
            progress=legacy_progress, recipe=spec(sft_fit.PILOT_TRAINING_BUDGET).recipe)


def test_pilot_full_state_resume_matches_uninterrupted_real_cpu_training(tmp_path):
    import copy
    import torch
    from shared_procedures.checkpointing import load_checkpoint, save_checkpoint
    from tests.unit.training.sft.individual_fit.test_sft_fit import tiny_runtime
    recipe = spec(sft_fit.PILOT_TRAINING_BUDGET).recipe
    model, optimizer, scheduler, examples = tiny_runtime()
    examples = examples + examples
    uninterrupted = sft_fit.train_optimizer_updates(model, optimizer, scheduler, examples, 0,
                                                    recipe=recipe)
    expected = copy.deepcopy(model.state_dict())
    expected_scheduler = copy.deepcopy(scheduler.state_dict())
    expected_rng = torch.get_rng_state().clone()
    model, optimizer, scheduler, _ = tiny_runtime()
    identity = spec(sft_fit.PILOT_TRAINING_BUDGET).identity()
    def save(progress):
        save_checkpoint(tmp_path / "checkpoint_000001", identity,
            {"adapter": model.state_dict(), "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(), "rng": sft_fit.capture_rng(), "progress": progress})
    paused = sft_fit.train_optimizer_updates(model, optimizer, scheduler, examples, 0,
        recipe=recipe, save=save, should_pause=lambda: True)
    assert paused["completed_steps"] == 1 and paused["status"] == "paused"
    state = load_checkpoint(tmp_path / "checkpoint_000001", identity)
    model, optimizer, scheduler, _ = tiny_runtime()
    model.load_state_dict(state["adapter"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    sft_fit.restore_rng(state["rng"])
    result = sft_fit.train_optimizer_updates(model, optimizer, scheduler, examples, 0,
                                            recipe=recipe, progress=state["progress"])
    assert result == uninterrupted
    assert scheduler.state_dict() == expected_scheduler
    assert all(torch.equal(value, expected[name]) for name, value in model.state_dict().items())
    assert torch.equal(torch.get_rng_state(), expected_rng)


@pytest.mark.parametrize("budget, expected_steps, expected_warmup", [
    (sft_fit.LEGACY_TRAINING_BUDGET, 1250, 38),
    (sft_fit.PILOT_TRAINING_BUDGET, 625, 19),
])
def test_real_cosine_scheduler_uses_the_declared_runtime_budget(tmp_path, monkeypatch,
                                                            budget, expected_steps, expected_warmup):
    """Run the production scheduler construction with declared CPU model stubs."""
    import torch
    import transformers
    import peft
    import operations.assets
    from training.sft.execution_profile import SFTExecutionProfile
    class RuntimeModel(torch.nn.Linear):
        def __init__(self):
            super().__init__(1, 1)
            self.config = SimpleNamespace(use_cache=True)
            self.is_gradient_checkpointing = False
        def gradient_checkpointing_enable(self, **kwargs):
            self.is_gradient_checkpointing = True
        def gradient_checkpointing_disable(self):
            self.is_gradient_checkpointing = False
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda _: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "declared CPU stub")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(total_memory=0))
    monkeypatch.setattr(operations.assets, "model_load_kwargs", lambda *args, **kwargs: {})
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        lambda *args, **kwargs: SimpleNamespace(pad_token_id=0))
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained",
                        lambda *args, **kwargs: RuntimeModel())
    monkeypatch.setattr(peft, "prepare_model_for_kbit_training", lambda model, **kwargs: model)
    monkeypatch.setattr(peft, "get_peft_model", lambda model, _: model)
    optimizer_stub = lambda params, **kwargs: torch.optim.AdamW(params, **kwargs)
    monkeypatch.setitem(sys.modules, "bitsandbytes",
                        SimpleNamespace(optim=SimpleNamespace(PagedAdamW32bit=optimizer_stub)))
    monkeypatch.setattr(sft_fit.importlib.metadata, "version", lambda _: "0.48.2")
    _, _, optimizer, scheduler, runtime = sft_fit._runtime(
        spec(budget), "cuda:0", cache_root=tmp_path, execution_profile=SFTExecutionProfile())
    assert runtime["total_optimizer_steps"] == expected_steps
    assert runtime["warmup_steps"] == expected_warmup
    assert optimizer.param_groups[0]["lr"] == 0
    for _ in range(expected_warmup):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2e-4)
    for _ in range(expected_steps - expected_warmup):
        optimizer.step()
        scheduler.step()
    assert scheduler.last_epoch == expected_steps
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-15)


def test_declared_epoch_mismatch_is_rejected_before_runtime():
    pilot = spec(sft_fit.PILOT_TRAINING_BUDGET)
    for settings in ({"epochs": 1}, {"training_budget": "one_epoch_pilot_v1", "epochs": 2},
                     {"training_budget": "one_epoch_pilot_v1", "epochs": True}):
        with pytest.raises(sft_fit.SFTInputError, match="named training budget"):
            sft_fit.run_sft_fit(pilot, [], [], "unused", {"sft": settings})


def test_unselected_fit_is_rejected_before_runtime():
    configuration = {"sft": {"training_budget": "one_epoch_pilot_v1", "epochs": 1,
        "selection": {"models": ["14b"], "families": ["rot13", "kazitext"], "levels": [10]}}}
    with pytest.raises(sft_fit.SFTInputError, match="declared execution selection"):
        sft_fit.run_sft_fit(replace(spec(sft_fit.PILOT_TRAINING_BUDGET), level=5),
                            [], [], "unused", configuration)
