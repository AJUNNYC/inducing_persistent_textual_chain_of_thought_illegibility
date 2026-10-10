import pytest
from data_generation_collection import data_generation as module


def test_all_branch_handoffs_overlap_whole_frame_and_explicit_prepared(monkeypatch,tmp_path):
    observed={}
    monkeypatch.setattr(module,'prepare_evaluation_data',lambda *a,**k:{'launch_ready':False,'content_hash':'e'})
    def rl(rows,revision,questions,**kwargs):
        observed['overlap']=questions
        return {'status':'verified','content_hash':'r'}
    monkeypatch.setattr(module,'prepare_rl_data',rl)
    monkeypatch.setattr(module,'prepare_candidate_manifest',lambda *a,**k:{'fixture':'candidates'})
    monkeypatch.setattr(module,'prepare_sft_data',lambda *a,**k:{'status':'verified','content_hash':'s'})
    result=module.prepare_data(evaluation_pools={'domain':[{'question':'selected'},{'question':'not selected'}]},evaluation_revisions={},gsm8k_train=[],
      gsm8k_train_revision='a'*40,lmsys_rows=[],lmsys_revision='b'*40,lmsys_authorized=True,
      tokenizers={},generator=None,kazitext=None,output_root=tmp_path,config_hash='c'*64)
    assert result['status']=='prepared' and not result['launch_ready']
    assert observed['overlap']==['selected','not selected']


def test_failure_never_released_as_completed(monkeypatch,tmp_path):
    monkeypatch.setattr(module,'prepare_evaluation_data',lambda *a,**k:(_ for _ in ()).throw(ValueError('source frame invalid')))
    with pytest.raises(ValueError):module.prepare_data(evaluation_pools={},evaluation_revisions={},gsm8k_train=[],gsm8k_train_revision='a'*40,
       lmsys_rows=[],lmsys_revision='b'*40,lmsys_authorized=False,tokenizers={},generator=None,kazitext=None,output_root=tmp_path,config_hash='c'*64)
    assert not (tmp_path/'data_manifest.json').exists()
