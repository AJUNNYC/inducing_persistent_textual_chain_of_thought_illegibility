import math
from statistics import stdev
import pytest
import torch
from training.rl_grpo.training_cycle.policy_update.grpo_policy_update import (
    group_advantages, dr_grpo_token_loss, apply_grpo_update, RollbackRequired)
from training.rl_grpo.training_cycle.rollouts.rollout_collection import collect_rollouts
from training.rl_grpo.training_cycle.rewards.rl_reward import compute_rl_reward
from tests.unit.training.rl_grpo.setup.test_rl_setup import fixture_state, fixture_manifest
from tests.unit.training.rl_grpo.training_cycle.rewards.sonnet_judgment.test_sonnet_opacity_judgment import FixtureJudge


def fixture_transaction(state):
    questions=fixture_manifest()["questions"][:8]
    records=collect_rollouts(state,questions)
    return records,[compute_rl_reward(r,q["reference_answer"],judge=FixtureJudge()) for r,q in zip(records,[q for q in questions for _ in range(4)])]


def test_group_standardization_and_zero_variance():
    rewards=[-1,0,1,1.5]*4+[1,1,1,1]*4
    values,groups=group_advantages(rewards)
    expected=[(r-.375)/(stdev([-1,0,1,1.5])+1e-4) for r in [-1,0,1,1.5]]
    assert values[:4]==pytest.approx(expected)
    assert values[16:]==[0]*16
    assert sum(g["zero_variance"] for g in groups)==4
    with pytest.raises(ValueError):group_advantages(rewards[:-1])


def test_constant_denominator_and_token_gradient_for_truncated_negative():
    current=torch.tensor([-1.,-1.],requires_grad=True)
    old=torch.tensor([-1.,-1.])
    ref=torch.tensor([-1.,-1.])
    loss,kl=dr_grpo_token_loss(current,old,ref,-2)
    assert loss.item()==pytest.approx(4/(32*2048))
    loss.backward()
    assert current.grad.tolist()==pytest.approx([2/(32*2048)]*2)
    assert kl.tolist()==[0,0]
    longer=torch.tensor([-1.]*4,requires_grad=True)
    loss2,_=dr_grpo_token_loss(longer,longer.detach(),longer.detach(),-2)
    assert loss2.item()==pytest.approx(2*loss.item())


def test_symmetric_clipping_and_fixed_reference_kl():
    current=torch.tensor([math.log(2.)],requires_grad=True)
    old=torch.tensor([0.])
    reference=torch.tensor([0.])
    loss,kl=dr_grpo_token_loss(current,old,reference,1)
    expected_kl=.5+math.log(2)-1
    assert kl.item()==pytest.approx(expected_kl)
    assert loss.item()==pytest.approx((-1.2+.01*expected_kl)/(32*2048))
    loss.backward()
    assert current.grad.item()==pytest.approx(.01*.5/(32*2048))


def test_all_32_verified_rewards_then_exact_one_update(tmp_path):
    state=fixture_state(tmp_path)
    rollouts,rewards=fixture_transaction(state)
    with pytest.raises(ValueError):apply_grpo_update(state,rollouts,rewards[:-1])
    assert state.runtime.update_calls==0 and state.completed_steps==0
    result=apply_grpo_update(state,rollouts,rewards)
    assert result["completed_steps"]==1 and state.runtime.update_calls==1
    assert state.runtime.question_history==[[f"q{i}" for i in range(8)]]


def test_partial_fault_does_not_advance_counter(tmp_path):
    state=fixture_state(tmp_path)
    records,rewards=fixture_transaction(state)
    state.runtime.fail_at=1
    with pytest.raises(RollbackRequired):apply_grpo_update(state,records,rewards)
    assert state.completed_steps==0 and not state.boundary_verified
    assert state.runtime.step==1


def test_finite_but_fabricated_reward_is_rejected(tmp_path):
    state=fixture_state(tmp_path)
    rollouts,rewards=fixture_transaction(state)
    rewards[0]["reward"]=1.5
    with pytest.raises(ValueError,match="frozen deterministic formula"):
        apply_grpo_update(state,rollouts,rewards)
    assert state.runtime.update_calls==0 and state.completed_steps==0


def test_repetition_is_a_diagnostic_with_no_reward_role():
    from training.rl_grpo.training_cycle.policy_update.grpo_policy_update import repetition_indicator
    assert repetition_indicator([1,2,1,2,1,2]) == pytest.approx(1/3)
    assert repetition_indicator([1,2,3,4]) == 0
    assert repetition_indicator([]) == 0
