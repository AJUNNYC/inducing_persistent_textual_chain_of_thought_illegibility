"""Declared fixture engine; these outputs never represent real model execution."""
from dataclasses import replace
from pathlib import Path
import pytest
from shared_procedures.identity import content_hash
from shared_procedures.checkpointing import verify_checkpoint
from training.rl_grpo.setup.rl_setup import RLRecipe, MODELS, setup_rl_run, validate_rl_manifest


def fixture_manifest():
    questions = [{"question_id": f"q{i}", "question": f"What is {i}+1?", "reference_answer": str(i+1),
                  "source_split": "train", "source_revision": "fixture-source", "order": i} for i in range(800)]
    return {"questions": questions, "status": "verified", "source_split": "train", "seed": 42,
            "content_hash": content_hash(questions), "source_revision": "fixture-source",
            "batches": [[f"q{i}" for i in range(start,start+8)] for start in range(0,800,8)]}


class FixtureRuntime:
    """Explicit fixture counts real transactions while generating known wrong outcomes."""
    created = []
    fail_at = None
    def __init__(self, **kwargs):
        self.step = 0
        self.samples = 0
        self.update_calls = 0
        self.question_history = []
        self.closed = False
        FixtureRuntime.created.append(self)
    def policy_hash(self):
        return content_hash({"fixture_step": self.step})
    def capture_state(self):
        return {"adapter": {"fixture_step": self.step}, "optimizer": {"fixture_steps": self.step},
                "scheduler": {"fixture_steps": self.step}, "rng": {"fixture_seed": 42}}
    def restore_state(self, state):
        self.step = state["adapter"]["fixture_step"]
    def sample(self, prompt, *, seed):
        self.samples += 1
        return {"raw_output": "A structurally complete fixture trace.\nFinal answer: \\boxed{-999}",
                "finish_reason": "eos", "truncated": False, "generated_tokens": 4,
                "prompt_token_ids": [1], "completion_token_ids": [2,3,4,5],
                "old_log_probs": [-1]*4, "reference_log_probs": [-1]*4}
    def update(self, rollouts, rewards):
        self.update_calls += 1
        self.question_history.append([rollouts[index]["question_id"] for index in range(0,32,4)])
        if self.fail_at == self.step + 1:
            self.step += 1  # Fixture deliberately mutates partial state before raising.
            self.fail_at = None
            raise RuntimeError("fixture partial optimizer fault")
        self.step += 1
        return {"logical_optimizer_updates": 1, "loss": 0.0, "kl": 0.0}
    def export_adapter(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path)/"fixture-adapter.json").write_text(str(self.step))
    def close(self):
        self.closed = True


def fixture_state(tmp_path):
    return setup_rl_run(MODELS[0], fixture_manifest(), tmp_path, model_revision="a"*40,
                       runtime_identity={"kind": "fixture"}, runtime_factory=FixtureRuntime, execution_mode="fixture")


def test_recipe_rejects_scientific_and_default_changes():
    RLRecipe().validate()
    for field, value in (("max_completion_length",1024),("learning_rate",1e-5),("num_iterations",2),
                         ("mask_truncated_completions",True),("adam_epsilon",1e-7)):
        with pytest.raises(ValueError):
            replace(RLRecipe(), **{field:value}).validate()


def test_manifest_exact_order_counts_hash(tmp_path):
    manifest = fixture_manifest()
    assert len(validate_rl_manifest(manifest)) == 800
    for mutate in (lambda m:m.update(seed=43), lambda m:m.update(content_hash="bad"),
                   lambda m:m["batches"][0].reverse()):
        altered = fixture_manifest()
        mutate(altered)
        with pytest.raises(ValueError):
            validate_rl_manifest(altered)


def test_real_launch_cannot_use_missing_preflights(tmp_path):
    with pytest.raises(RuntimeError, match="real preflight"):
        setup_rl_run(MODELS[0], fixture_manifest(), tmp_path, model_revision="a"*40,
                     runtime_identity={"lock": "fixture"})


def test_full_boundary_restore_and_invalid_existing_state(tmp_path):
    state = fixture_state(tmp_path)
    assert verify_checkpoint(tmp_path/"checkpoints/step_000")["progress"]["completed_steps"] == 0
    state.runtime.step = 10
    state.completed_steps = 10
    state.save()
    resumed = fixture_state(tmp_path)
    assert resumed.completed_steps == 10 and resumed.runtime.step == 10
    assert resumed.attempt_id != state.attempt_id
    (tmp_path/"checkpoints/step_010/full_state.pt").write_bytes(b"corrupt")
    with pytest.raises(Exception, match="integrity"):
        fixture_state(tmp_path)


def test_changed_identity_never_fresh_starts(tmp_path):
    fixture_state(tmp_path)
    with pytest.raises(ValueError, match="identity differs"):
        setup_rl_run(MODELS[0], fixture_manifest(), tmp_path, model_revision="b"*40,
                     runtime_identity={"kind":"fixture"},runtime_factory=FixtureRuntime,execution_mode="fixture")


def test_injected_engine_cannot_be_labeled_production(tmp_path):
    with pytest.raises(ValueError, match="injected engines are fixtures"):
        setup_rl_run(MODELS[0], fixture_manifest(), tmp_path, model_revision="a"*40,
                     runtime_identity={"kind":"fixture"}, runtime_factory=FixtureRuntime, execution_mode="production")


def test_executable_actual_grpo_preflight_blocks_before_download_on_cpu(tmp_path,monkeypatch):
    import torch
    from training.rl_grpo.setup.rl_setup import run_grpo_preflight
    from tests.unit.training.sft.test_sft_training import fixture_config
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    result=run_grpo_preflight(fixture_config(),tmp_path)
    assert result["status"]=="blocked" and result["name"]=="grpo_update"
    assert result["evidence"]["live"] is False
    assert result["evidence"]["scientific_results"] is False
    assert not list(tmp_path.iterdir())
