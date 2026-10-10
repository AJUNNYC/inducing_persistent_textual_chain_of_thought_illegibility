from shared_procedures.storage import read_json
from shared_procedures.recovery_procedure import RecoveryManager
from training.rl_grpo.training_cycle.grpo_training_cycle import grpo_training_cycle
from tests.unit.training.rl_grpo.setup.test_rl_setup import fixture_state, fixture_manifest
from tests.unit.training.rl_grpo.training_cycle.rewards.sonnet_judgment.test_sonnet_opacity_judgment import FixtureJudge


def test_exact_order_100_boundaries_and_zero_variance_kept(tmp_path):
    state=fixture_state(tmp_path)
    result=grpo_training_cycle(state,fixture_manifest()["questions"],judge=FixtureJudge())
    assert result["status"]=="completed" and state.completed_steps==100
    assert state.runtime.samples==3200 and state.runtime.update_calls==100
    assert state.runtime.question_history==[[f"q{i}" for i in range(start,start+8)] for start in range(0,800,8)]
    assert len(list((tmp_path/"checkpoints").glob("step_???/manifest.json")))==11
    final=read_json(next((tmp_path/"updates").glob("100_*.json")))
    assert len(final["groups"])==8 and all(g["zero_variance"] for g in final["groups"])
    assert final["reward_distribution"] == [0]*32
    assert final["repetition_indicators"] == [0]*32


def test_replay_restores_optimizer_and_persistent_retry_budget(tmp_path):
    state=fixture_state(tmp_path)
    state.runtime.fail_at=2
    manager=RecoveryManager(tmp_path/"worker-retries.json")
    result=grpo_training_cycle(state,fixture_manifest()["questions"],judge=FixtureJudge(),recovery_manager=manager,
                               stop_requested=lambda:state.completed_steps==3)
    assert result["status"]=="paused" and state.runtime.step==state.completed_steps==3
    assert state.runtime.update_calls==5  # step1, partial2, replay1,2,3
    assert state.runtime.samples==96  # same policy outputs reused during replay
    assert manager.counters()["worker_retries"]==1
    assert next(iter(manager.counters()["operations"].values()))==1


def test_required_judge_failure_pauses_before_any_update(tmp_path):
    class CorrectFixtureJudge(FixtureJudge):
        def request(self,*args,**kwargs):return {"status":"failed"}
    state=fixture_state(tmp_path)
    original=state.runtime.sample
    def sample(prompt,*,seed):
        value=original(prompt,seed=seed)
        value["raw_output"]="Trace.\nFinal answer: \\boxed{1}"
        return value
    state.runtime.sample=sample
    result=grpo_training_cycle(state,fixture_manifest()["questions"],judge=CorrectFixtureJudge())
    assert result["status"]=="paused" and result["reason"]=="required_judge_unavailable"
    assert state.completed_steps==0 and state.runtime.update_calls==0
    assert state.runtime.samples==32
