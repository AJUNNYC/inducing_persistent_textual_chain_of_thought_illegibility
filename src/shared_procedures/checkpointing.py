"""Full training-state checkpoint publication and verified restoration."""
from __future__ import annotations
from pathlib import Path
import os
import shutil
import tempfile
from typing import Any
from .identity import content_hash, file_hash
from .storage import ArtifactIntegrityError, atomic_write_json, file_lock, fsync_directory, read_json, notify_artifact

REQUIRED_STATE = {'adapter','optimizer','scheduler','rng','progress'}

class CheckpointError(ArtifactIntegrityError):
    pass

def _validate_state(state: dict) -> None:
    if not isinstance(state,dict) or not REQUIRED_STATE <= state.keys():
        raise CheckpointError('checkpoint requires adapter, optimizer, scheduler, RNG and progress state')
    if any(state[k] is None for k in REQUIRED_STATE):
        raise CheckpointError('null full checkpoint state')
    progress = state['progress']
    if not isinstance(progress,dict) or type(progress.get('completed_steps')) is not int or progress['completed_steps'] < 0:
        raise CheckpointError('invalid completed optimizer step')
    if 'sampler_position' not in progress:
        raise CheckpointError('sampler/data position is required')
    if not isinstance(state['rng'],dict) or not state['rng']:
        raise CheckpointError('missing RNG state')

def _same_state(left: Any, right: Any) -> bool:
    import torch
    if isinstance(left,torch.Tensor) and isinstance(right,torch.Tensor):
        return left.shape==right.shape and left.dtype==right.dtype and torch.equal(left.cpu(),right.cpu())
    try:
        import numpy as np
        if isinstance(left,np.ndarray) and isinstance(right,np.ndarray):
            return left.dtype==right.dtype and np.array_equal(left,right)
    except ImportError:
        pass
    if isinstance(left,dict) and isinstance(right,dict):
        return left.keys()==right.keys() and all(_same_state(left[k],right[k]) for k in left)
    if isinstance(left,(tuple,list)) and isinstance(right,type(left)):
        return len(left)==len(right) and all(_same_state(a,b) for a,b in zip(left,right))
    try:
        return bool(left==right)
    except (TypeError,ValueError,RuntimeError):
        return False

def save_checkpoint(directory: str | Path, identity: dict, state: dict) -> dict:
    _validate_state(state)
    if not isinstance(identity,dict) or not identity:
        raise CheckpointError('exact checkpoint identity required')
    import torch
    target = Path(directory)
    target.parent.mkdir(parents=True,exist_ok=True)
    with file_lock(str(target)+'.lock'):
        if target.exists():
            existing = verify_checkpoint(target,expected_identity=identity)
            if existing['progress'] != state['progress']:
                raise CheckpointError('immutable checkpoint progress differs')
            if not _same_state(load_checkpoint(target,identity),state):
                raise CheckpointError('immutable checkpoint full state differs')
            notify_artifact(target,existing)
            return existing
        temporary = Path(tempfile.mkdtemp(prefix=f'.{target.name}.',dir=target.parent))
        try:
            payload = temporary/'full_state.pt'
            torch.save(state,payload)
            with payload.open('rb') as handle:
                os.fsync(handle.fileno())
            manifest = {'version':'full_state_v1','status':'complete','identity':identity,
                        'progress':state['progress'],'state_keys':sorted(state),
                        'files':{'full_state.pt':{'sha256':file_hash(payload),'size_bytes':payload.stat().st_size}}}
            manifest['checkpoint_hash'] = content_hash(manifest)
            atomic_write_json(temporary/'manifest.json',manifest)
            # Verify actual deserializability and full state before atomic directory publication.
            verify_checkpoint(temporary,expected_identity=identity)
            fsync_directory(temporary)
            os.rename(temporary,target)
            fsync_directory(target.parent)
            verified = verify_checkpoint(target,expected_identity=identity)
            notify_artifact(target,verified)
            return verified
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

def verify_checkpoint(directory: str | Path, expected_identity: dict | None = None) -> dict:
    root = Path(directory)
    if not root.is_dir() or root.is_symlink():
        raise CheckpointError('checkpoint directory absent or unsafe')
    manifest = read_json(root/'manifest.json')
    supplied_hash = manifest.get('checkpoint_hash')
    content = {k:v for k,v in manifest.items() if k != 'checkpoint_hash'}
    if supplied_hash != content_hash(content) or manifest.get('status') != 'complete':
        raise CheckpointError('checkpoint manifest integrity/status failure')
    if expected_identity is not None and manifest.get('identity') != expected_identity:
        raise CheckpointError('checkpoint exact identity mismatch')
    if not REQUIRED_STATE <= set(manifest.get('state_keys',[])):
        raise CheckpointError('adapter-only checkpoint cannot resume training')
    if set(manifest.get('files',{})) != {'full_state.pt'}:
        raise CheckpointError('unexpected checkpoint payload inventory')
    for name, info in manifest['files'].items():
        file = root/name
        if file.is_symlink() or not file.is_file() or file.stat().st_size != info.get('size_bytes') or file_hash(file) != info.get('sha256'):
            raise CheckpointError('checkpoint payload integrity failure')
    import torch
    # Pickled tensors and Python/NumPy RNG require full-state mode. Only integrity-verified
    # trusted experiment checkpoints may enter this API; downloaded untrusted pickle is excluded.
    state = torch.load(root/'full_state.pt',map_location='cpu',weights_only=False)
    _validate_state(state)
    if state['progress'] != manifest['progress']:
        raise CheckpointError('checkpoint progress disagrees with payload')
    return manifest

def load_checkpoint(directory: str | Path, expected_identity: dict | None = None) -> dict[str,Any]:
    verify_checkpoint(directory,expected_identity)
    import torch
    state = torch.load(Path(directory)/'full_state.pt',map_location='cpu',weights_only=False)
    _validate_state(state)
    return state

def latest_checkpoint(parent: str | Path, identity: dict) -> Path | None:
    candidates = []
    invalid_matching = []
    for directory in Path(parent).iterdir() if Path(parent).exists() else []:
        if directory.is_dir() and (directory/'manifest.json').exists():
            try:
                manifest = verify_checkpoint(directory,identity)
            except (CheckpointError,ValueError,OSError):
                try: claimed=read_json(directory/'manifest.json').get('identity')
                except (ValueError,OSError): claimed=identity
                if claimed==identity: invalid_matching.append(directory)
                continue
            candidates.append((manifest['progress']['completed_steps'],directory))
    if not candidates and invalid_matching:
        raise CheckpointError('claimed recovery state exists but no matching full checkpoint verifies')
    return max(candidates,key=lambda row:row[0])[1] if candidates else None

def retain_checkpoints(parent: str | Path, identity: dict, *, keep: int = 2, preserve: tuple[str,...] = ()) -> list[str]:
    if keep < 2:
        raise CheckpointError('at least two full recovery states must be retained')
    valid = []
    for directory in Path(parent).iterdir():
        if directory.is_dir() and (directory/'manifest.json').exists():
            try:
                record = verify_checkpoint(directory,identity)
            except (CheckpointError,ValueError,OSError):
                continue
            valid.append((record['progress']['completed_steps'],directory))
    removed=[]
    for _,directory in sorted(valid,key=lambda row:row[0],reverse=True)[keep:]:
        if directory.name not in preserve:
            shutil.rmtree(directory)
            removed.append(str(directory))
    return removed
