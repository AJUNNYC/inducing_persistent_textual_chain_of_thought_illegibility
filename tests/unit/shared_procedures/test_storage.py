import json
import multiprocessing
import pytest
from shared_procedures.storage import *

def test_atomic_roundtrip_and_duplicate_rejection(tmp_path):
    path=tmp_path/'record.json';atomic_write_json(path,{'unicode':'é','x':1})
    assert read_json(path)=={'unicode':'é','x':1}
    path.write_text('{"x":1,"x":2}')
    with pytest.raises(ArtifactIntegrityError):read_json(path)

def test_artifact_dependency_and_corruption(tmp_path):
    path=tmp_path/'a.json';publish_artifact(path,[1,2],identity={'configuration':'a'},expected_count=2)
    with pytest.raises(ArtifactIntegrityError):verify_artifact(path,expected_identity={'configuration':'b'})
    record=read_json(path);record['data'].append(3);atomic_write_json(path,record)
    with pytest.raises(ArtifactIntegrityError):verify_artifact(path)

def test_no_overwrite_immutable_artifact(tmp_path):
    path=tmp_path/'a.json';publish_artifact(path,[1],identity={'x':1})
    with pytest.raises(ArtifactIntegrityError):publish_artifact(path,[2],identity={'x':1})

def test_exclusive_writer(tmp_path):
    path=tmp_path/'writer.lock'
    with file_lock(path):
        # A separate open file description cannot acquire the same namespace.
        with pytest.raises(TimeoutError):
            with file_lock(path,timeout=0):pass

def test_hook_failure_propagates(tmp_path):
    import shared_procedures.storage as storage
    def fail(path,metadata):raise RuntimeError('backup unavailable')
    storage.register_artifact_hook(fail)
    try:
        with pytest.raises(RuntimeError):storage.notify_artifact(tmp_path/'artifact',{})
    finally:storage._ARTIFACT_HOOKS.remove(fail)
