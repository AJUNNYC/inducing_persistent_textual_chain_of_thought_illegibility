import pytest
from shared_procedures.identity import content_hash
from training.rl_grpo.training_cycle.rewards.rl_reward import compute_rl_reward
from tests.unit.training.rl_grpo.training_cycle.rewards.sonnet_judgment.test_sonnet_opacity_judgment import FixtureJudge


def rollout(text,**kwargs):
    return {"completion_id":"c","question_id":"q","question":"public question","policy_hash":"p","raw_output":text,**kwargs}


@pytest.mark.parametrize("text,truncated,reward", [("reasoning without marker",False,-1),
    ("Trace.\nFinal answer: \\boxed{2}",False,0),("Trace.\nFinal answer: \\boxed{1",True,-1),
    ("Final answer: \\boxed{1}",False,-1),("Trace.\nFinal answer: \\boxed{1}\nFinal answer: \\boxed{1}",False,-1)])
def test_local_gate_avoids_all_apis(text,truncated,reward):
    judge=FixtureJudge()
    result=compute_rl_reward(rollout(text,truncated=truncated),"1",judge=judge)
    assert result["reward"] == reward and judge.calls == []


@pytest.mark.parametrize("score,readable,padding,reward",[(9,False,False,1.5),(9,True,False,1.125),
                                                        (9,False,True,1),(1,False,False,1)])
def test_exact_frozen_reward(score,readable,padding,reward):
    judge=FixtureJudge(score,readable,padding)
    result=compute_rl_reward(rollout("Actual reasoning.\nFinal answer: \\boxed{1/2}"),"0.5",judge=judge)
    assert result["reward"] == reward and len(judge.calls)==2
    assert result["raw_output_hash"] == content_hash("Actual reasoning.\nFinal answer: \\boxed{1/2}")


def test_invalid_reference_stops_before_api():
    judge=FixtureJudge()
    with pytest.raises(ValueError):compute_rl_reward(rollout("x"),"broken reference",judge=judge)
    assert judge.calls==[]
