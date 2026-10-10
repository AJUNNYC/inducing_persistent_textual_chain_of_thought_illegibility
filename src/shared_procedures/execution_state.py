"""Persistent state transitions with immutable job identity and one writer."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from .storage import atomic_write_json, read_json, file_lock

class StateError(RuntimeError):
    pass

TRANSITIONS = {'pending':{'running','paused','failed'},'running':{'paused','failed','complete'},
               'paused':{'running','failed'},'failed':set(),'complete':set()}

def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()

class ExecutionState:
    def __init__(self,path: str | Path,identity: dict):
        self.path=Path(path)
        self.identity=identity
        if not identity:
            raise StateError('immutable job identity required')
        with file_lock(str(self.path)+'.lock'):
            if self.path.exists():
                if read_json(self.path).get('identity') != identity:
                    raise StateError('job exists under another identity')
            else:
                atomic_write_json(self.path,{'identity':identity,'status':'pending','completed_steps':0,
                    'attempts':0,'operation_retries':{},'worker_retries':0,'history':[],'partial_group':None})

    def read(self) -> dict:
        value=read_json(self.path)
        if value['identity'] != self.identity:
            raise StateError('job identity changed')
        return value

    def transition(self,status: str,*,detail: str='',completed_steps: int | None=None,partial_group: Any=None) -> dict:
        with file_lock(str(self.path)+'.lock'):
            state=self.read()
            if status not in TRANSITIONS.get(state['status'],set()):
                raise StateError(f'invalid transition {state["status"]}->{status}')
            if completed_steps is not None:
                if type(completed_steps) is not int or completed_steps < state['completed_steps']:
                    raise StateError('completed steps cannot advance incorrectly or move backwards')
                state['completed_steps']=completed_steps
            if status=='complete' and partial_group is not None:
                raise StateError('cannot complete a partial update')
            if status=='running':
                state['attempts']+=1
            state['history'].append({'from':state['status'],'to':status,'detail':detail,'at':timestamp()})
            state.update(status=status,partial_group=partial_group)
            atomic_write_json(self.path,state)
            return state

    def update_progress(self,completed_steps: int,*,partial_group: Any=None) -> dict:
        with file_lock(str(self.path)+'.lock'):
            state=self.read()
            if state['status']!='running' or type(completed_steps) is not int or completed_steps<state['completed_steps']:
                raise StateError('invalid completed-operation progress')
            state.update(completed_steps=completed_steps,partial_group=partial_group)
            atomic_write_json(self.path,state)
            return state

    def claim(self,timeout: float=0):
        return file_lock(str(self.path)+'.writer.lock',timeout=timeout)
