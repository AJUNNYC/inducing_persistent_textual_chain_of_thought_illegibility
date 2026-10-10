import pytest
from shared_procedures.identity import content_hash
from data_generation_collection.evaluation_data import evaluation_data_preparation as module


def test_protocol_draft_is_prepared_never_claimed_training_ready(monkeypatch,tmp_path):
    panel={'status':'verified','content_hash':'a'*64}
    legibility={'status':'verified','content_hash':'b'*64}
    monkeypatch.setattr(module,'construct_capability_panel',lambda *args,**kwargs:panel)
    monkeypatch.setattr(module,'construct_legibility_subset',lambda *args,**kwargs:legibility)
    result=module.prepare_evaluation_data({}, {}, output_dir=tmp_path,
             protocol_builder=lambda *args,**kwargs:{'status':'draft'})
    assert result['status']=='prepared' and not result['launch_ready']
    result=module.prepare_evaluation_data({}, {}, output_dir=tmp_path,
             protocol_builder=lambda *args,**kwargs:{'status':'frozen'})
    assert result['status']=='verified' and result['launch_ready']


def test_child_failure_propagates_no_parent_artifact(monkeypatch,tmp_path):
    def fail(*args,**kwargs):raise ValueError('impossible fixed quotas')
    monkeypatch.setattr(module,'construct_capability_panel',fail)
    with pytest.raises(ValueError):module.prepare_evaluation_data({}, {}, output_dir=tmp_path)
    assert not (tmp_path/'evaluation_package.json').exists()
