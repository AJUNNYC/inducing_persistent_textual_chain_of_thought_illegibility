import pytest
from shared_procedures.execution_state import *

def test_terminal_and_identity_guard(tmp_path):
    path=tmp_path/'state.json';state=ExecutionState(path,{'config':'a'})
    state.transition('running');state.update_progress(1);state.transition('complete')
    with pytest.raises(StateError):state.transition('running')
    with pytest.raises(StateError):ExecutionState(path,{'config':'b'})
    assert state.read()['completed_steps']==1

def test_partial_and_progress_guard(tmp_path):
    state=ExecutionState(tmp_path/'s.json',{'job':'x'});state.transition('running');state.update_progress(2,partial_group={'n':3})
    with pytest.raises(StateError):state.update_progress(1)
    with pytest.raises(StateError):state.transition('complete',partial_group={'n':3})
    state.transition('paused');state.transition('running')
    assert state.read()['attempts']==2
