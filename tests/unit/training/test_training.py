"""Branch selection cannot accidentally launch an unselected paid trajectory."""
from pathlib import Path
import pytest
import yaml
import training.training as module


def test_selected_rl_receives_both_immutable_identity_maps_and_one_model(tmp_path,monkeypatch):
    config=yaml.safe_load((Path(__file__).parents[3]/'config.yaml').read_text())
    calls=[]
    def record(manifest,root,**kwargs):
        calls.append(kwargs);return {'status':'fixture_only'}
    monkeypatch.setattr(module,'run_rl_training',record)
    report=module.train(config=config,rl_manifest={'fixture':True},output_root=tmp_path,judge=object(),
        preflight_report={'fixture':True},runtime_identity={'fixture':True},families=(),rl_model_sizes=('14b',))
    assert calls[0]['model_sizes']==('14b',)
    assert calls[0]['model_revisions']=={v['id']:v['revision'] for v in config['models'].values()}
    assert calls[0]['tokenizer_revisions']=={v['id']:v['tokenizer_revision'] for v in config['models'].values()}
    assert 'sft' not in report
    with pytest.raises(ValueError,match='reward client'):
        module.train(config=config,rl_manifest={},output_root=tmp_path,preflight_report={},runtime_identity={},families=())
    with pytest.raises(ValueError,match='At least one'):
        module.train(config=config,output_root=tmp_path,preflight_report={},runtime_identity={},families=(),include_rl=False)
