import pytest
from training.rl_grpo.training_cycle.rewards.sonnet_judgment.sonnet_opacity_judgment import (
    judge_opacity, validate_pass_a, validate_pass_b, RewardInfrastructureError, SETTINGS)


class FixtureJudge:
    def __init__(self, score=9, readable=False, padding=False):
        self.calls=[]
        self.score, self.readable, self.padding = score, readable, padding
    def request(self, kind, model, prompt, trace, **kwargs):
        self.calls.append({"kind":kind,"model":model,"prompt":prompt,"trace":trace,**kwargs})
        parsed = {"score":self.score} if kind == "rl_pass_a" else {
            "readable_solution_present":self.readable,"obvious_padding_present":self.padding}
        return {"status":"success","parsed":parsed,"returned_model":model,"verification_level":"fixture"}


@pytest.mark.parametrize("readable,padding,expected", [(False,False,1),(True,False,.25),(False,True,1),(True,True,.25)])
def test_independent_pass_visibility_and_boolean_semantics(readable,padding,expected):
    judge = FixtureJudge(readable=readable,padding=padding)
    result=judge_opacity("PUBLIC QUESTION","TRACE",client=judge)
    assert result["opacity"] == expected
    assert result["padding_gate"] == (0 if padding else 1)
    a,b=judge.calls
    assert a["kind"] == "rl_pass_a" and "question" not in a
    assert b["question"] == "PUBLIC QUESTION" and b["trace"] == "TRACE"
    assert "score" not in b and "temperature" not in a["settings"]
    assert SETTINGS["max_tokens"] == 64
    assert "set false when uncertain" in b["prompt"]


@pytest.mark.parametrize("value", [{"score":True},{"score":0},{"score":10},{"score":2,"extra":0}])
def test_strict_score(value):
    with pytest.raises(RewardInfrastructureError): validate_pass_a(value)


@pytest.mark.parametrize("value", [{"readable_solution_present":"unknown","obvious_padding_present":False},
                                  {"readable_solution_present":False},
                                  {"readable_solution_present":False,"obvious_padding_present":False,"extra":False}])
def test_training_never_accepts_tristate(value):
    with pytest.raises(RewardInfrastructureError): validate_pass_b(value)


def test_failed_pass_has_no_default():
    class FailedJudge(FixtureJudge):
        def request(self,*args,**kwargs):return {"status":"failed"}
    with pytest.raises(RewardInfrastructureError): judge_opacity("question","trace",client=FailedJudge())
