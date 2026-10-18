import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('queue',Path(__file__).resolve().parents[3]/'scripts/run_sft_gpqa_queue.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_training_gate_rejects_wrong_level_and_unverified_backup(tmp_path):
    p=tmp_path/'endpoints.json';config={'level':3,'training_endpoints':str(p)}
    for level,status in [(7,'VERIFIED'),(3,'PENDING')]:
        p.write_text(json.dumps({'status':'COMPLETE','drive_status':status,'adapters':[{'level':level}]}))
        with pytest.raises(RuntimeError):m.training_complete(config)
    p.write_text(json.dumps({'status':'COMPLETE','drive_status':'VERIFIED','adapters':[{'level':3}]}))
    m.training_complete(config)


def test_evaluation_success_requires_both_receipt_and_full_question_count(tmp_path):
    config={'level':7,'output':str(tmp_path)};(tmp_path/'handoff').mkdir()
    status={'phase':'COMPLETE_DRIVE_VERIFIED','level':7,'questions':198}
    receipt={'status':'COMPLETE','drive_status':'PENDING'}
    (tmp_path/'followup_status.json').write_text(json.dumps(status))
    p=tmp_path/'handoff/completion_verified.json';p.write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError):m.evaluation_complete(config)
    receipt['drive_status']='VERIFIED';p.write_text(json.dumps(receipt));m.evaluation_complete(config)
    status['questions']=128;(tmp_path/'followup_status.json').write_text(json.dumps(status))
    with pytest.raises(RuntimeError):m.evaluation_complete(config)
