import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('followup',Path(__file__).resolve().parents[3]/'scripts/watch_gpqa_continuation.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_only_fully_uploaded_training_releases_evaluation():
    assert m.training_ready({'phase':'COMPLETE_DRIVE_VERIFIED','completed_updates':625})
    for value in ({'phase':'RUNNING','completed_updates':625},
                  {'phase':'COMPLETE_DRIVE_VERIFIED','completed_updates':600},
                  {'phase':'FAILED'},{}):
        assert not m.training_ready(value)


def records():
    return [{'local_path':f'/original/{"a"*64}/raw/base_14b/{i:064x}.json',
             'sha256':'b'*64,'sync_status':'VERIFIED'} for i in range(198)]


def test_full_baseline_is_reused_only_with_complete_distinct_verified_inventory():
    m.check_records(records(),'a'*64)
    with pytest.raises(ValueError):m.check_records(records()[:-1],'a'*64)
    rows=records();rows[-1]=rows[0]
    with pytest.raises(ValueError):m.check_records(rows,'a'*64)
    rows=records();rows[0]['sync_status']='FAILED'
    with pytest.raises(ValueError):m.check_records(rows,'a'*64)
    with pytest.raises(ValueError):m.check_records(records(),'c'*64)


def test_existing_handoff_json_is_written_atomically(tmp_path):
    p=tmp_path/'state.json';m.write(p,{'phase':'WAITING'})
    m.write(p,{'phase':'PAUSED'})
    assert m.read(p)=={'phase':'PAUSED'}
    assert not p.with_suffix('.tmp').exists()
