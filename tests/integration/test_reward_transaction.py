"""Real persistence/client/update orchestration; external model/provider are explicit fixtures."""
import json
import re
import pytest
from shared_procedures.judge_client import JudgeClient
from shared_procedures.storage import read_json
from training.rl_grpo.setup.rl_setup import MODELS, setup_rl_run
from training.rl_grpo.training_cycle.grpo_training_cycle import grpo_training_cycle
from training.rl_grpo.training_cycle.policy_update.grpo_policy_update import apply_grpo_update
from training.rl_grpo.training_cycle.rollouts.rollout_collection import collect_rollouts
from tests.unit.training.rl_grpo.setup.test_rl_setup import FixtureRuntime, fixture_manifest


class CorrectFixtureRuntime(FixtureRuntime):
    """Deterministic synthetic output; does not implement or impersonate a real model."""
    def sample(self,prompt,*,seed):
        result=super().sample(prompt,seed=seed)
        answer=int(re.search(r'What is (\d+)\+1',prompt).group(1))+1
        result['raw_output']=f'Fixture reasoning with seed {seed}.\nFinal answer: \\boxed{{{answer}}}'
        return result


class RecordingProviderFixture:
    def __init__(self):
        self.requests=[]
        self.fail_b_once=True
    def __call__(self,request):
        self.requests.append(request)
        if request['kind']=='rl_pass_a':
            parsed={'score':9}
        elif self.fail_b_once:
            self.fail_b_once=False
            # Deliberate incompatible evaluation tri-state response causes an infrastructure pause.
            parsed={'readable_solution_present':'uncertain','obvious_padding_present':False}
        else:
            parsed={'readable_solution_present':False,'obvious_padding_present':False}
        return {'text':json.dumps(parsed),'model':request['model'],
                'usage':{'input_tokens':10,'output_tokens':5},'request_id':f'fixture_{len(self.requests)}',
                'finish_reason':'end_turn'}


def test_paused_reward_transaction_reuses_raw_and_successful_pass_then_updates_once(tmp_path):
    transport=RecordingProviderFixture()
    client=JudgeClient(tmp_path/'judge-cache',tmp_path/'judge-state',transport=transport)
    state=setup_rl_run(MODELS[0],fixture_manifest(),tmp_path/'run',model_revision='a'*40,
        runtime_identity={'fixture':True},runtime_factory=CorrectFixtureRuntime,execution_mode='fixture')
    first=grpo_training_cycle(state,fixture_manifest()['questions'],judge=client,
                              stop_requested=lambda:state.completed_steps==1)
    assert first['status']=='paused' and first['reason']=='required_judge_unavailable'
    assert state.completed_steps==0 and state.runtime.update_calls==0 and state.runtime.samples==32
    successful_a=list((tmp_path/'judge-cache/rl_pass_a').glob('*.json'))
    assert len(successful_a)==1 and read_json(successful_a[0])['status']=='success'
    assert read_json(successful_a[0])['simulation'] is True
    second=grpo_training_cycle(state,fixture_manifest()['questions'],judge=client,
                               stop_requested=lambda:state.completed_steps==1)
    assert second['status']=='paused' and second['reason']=='controlled_stop'
    assert state.runtime.samples==32 and state.runtime.update_calls==1 and state.completed_steps==1
    assert sum(r['kind']=='rl_pass_a' for r in transport.requests)==32
    assert sum(r['kind']=='rl_pass_b' for r in transport.requests)==33
    for request in transport.requests:
        assert 'Final answer:' not in request['trace'] and 'reference_answer' not in request
        assert request['question'] is None if request['kind']=='rl_pass_a' else request['question'].startswith('What is')
        assert 'temperature' not in request['settings']
    rewards=list((tmp_path/'run/rewards').glob('*.json'))
    assert len(rewards)==32 and all(read_json(path)['reward']['reward']==1.5 for path in rewards)


def test_incomplete_or_wrong_identity_rewards_never_enter_optimizer(tmp_path):
    from tests.unit.training.rl_grpo.training_cycle.policy_update.test_grpo_policy_update import fixture_transaction
    from tests.unit.training.rl_grpo.setup.test_rl_setup import fixture_state
    state=fixture_state(tmp_path/'run')
    rollouts,rewards=fixture_transaction(state)
    with pytest.raises(ValueError):apply_grpo_update(state,rollouts,rewards[:31])
    rewards[-1]['completion_id']='a different rollout'
    with pytest.raises(ValueError):apply_grpo_update(state,rollouts,rewards)
    assert state.runtime.update_calls==0 and state.completed_steps==0
