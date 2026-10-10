import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from evaluation.shared_model_evaluation.item_evaluation.capability_scoring.mbpp_sandbox import DockerMBPPSandbox

IMAGE='python@sha256:'+'a'*64


def test_hard_container_boundaries():
    sandbox=DockerMBPPSandbox(IMAGE)
    command=sandbox.command(Path('/tmp/ephemeral_inputs'),'mbpp_test')
    for option in ['--network=none','--user=0:0','--read-only','--cap-drop=ALL',
                   '--security-opt=no-new-privileges:true','--cpus=1','--memory=1g','--memory-swap=1g','--ulimit=cpu=5:5']:
        assert option in command
    mounts=[x for x in command if x.startswith('type=bind')]
    assert mounts==['type=bind,source=/tmp/ephemeral_inputs,target=/input,readonly']
    assert not any('privileged' in x for x in command)
    with pytest.raises(ValueError): DockerMBPPSandbox('python:latest')


def test_model_early_exit_does_not_pass(tmp_path):
    def runner(command,**kw):
        return SimpleNamespace(returncode=0,stdout='' if command[1]=='run' else 'ok',stderr='')
    sandbox=DockerMBPPSandbox(IMAGE,runner=runner,scratch_root=tmp_path)
    result=sandbox.execute('import os; os._exit(0)\ndef fail(): return False',['assert fail() is True'])
    assert result['status']=='outcome' and result['correct'] is False


def test_executor_launch_failure_missing_not_incorrect(tmp_path):
    def runner(command,**kw):
        return SimpleNamespace(returncode=125 if command[1]=='run' else 0,stdout='',stderr='daemon unavailable')
    result=DockerMBPPSandbox(IMAGE,runner=runner,scratch_root=tmp_path).execute('def f(): return 1',['assert f()==1'])
    assert result['status']=='infrastructure_failure' and result['correct'] is None


def test_exact_code_and_original_tests_reach_disposable_inputs(tmp_path):
    def runner(command,**kw):
        if command[1]!='run': return SimpleNamespace(returncode=0,stdout='ok',stderr='')
        mount=command[command.index('--mount')+1]
        path=Path(mount.split('source=')[1].split(',')[0])
        assert (path/'solution.py').read_text()=='def f(): return 1'
        assert json.loads(kw['input'])=={'tests':['assert f()==1'],'entrypoints':['f']}
        assert not (path/'tests.json').exists()
        assert not (path/'nonce').exists()
        return SimpleNamespace(returncode=0,stdout=json.dumps({'status':'outcome','correct':True,'reason':'passed'}),stderr='')
    assert DockerMBPPSandbox(IMAGE,runner=runner,scratch_root=tmp_path).execute('def f(): return 1',['assert f()==1'])['correct'] is True
    assert not list(tmp_path.iterdir())
