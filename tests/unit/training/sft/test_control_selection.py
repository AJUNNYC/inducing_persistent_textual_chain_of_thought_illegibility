import pytest
from shared_procedures.configuration import sft_execution_selection, ConfigurationError
from training.sft.individual_fit.sft_fit import SFTFitSpec, SFTInputError
from training.sft.level_continuation import ContinuationSpec, spec_from_identity

def test_zero_control_cannot_alias_level_ten():
    s=SFTFitSpec('rot13','14b',0,'a'*40,'a'*40,'b'*64,'c'*64,training_budget='efficient_one_epoch_v1')
    assert s.strength==0 and s.run_id=='rot13_14b_level_00'
    assert s.recipe.epochs==1 and s.recipe.learning_rate==2e-4
    assert sft_execution_selection({'sft':{'selection':{'models':['14b'],'families':['rot13'],'levels':[0]}}})==[{'family':'rot13','model_id':'14b','level':0}]

def test_zero_kazitext_not_silently_allowed():
    with pytest.raises(SFTInputError):SFTFitSpec('kazitext','14b',0,'a'*40,'a'*40,'b'*64,'c'*64)
    with pytest.raises(ConfigurationError):sft_execution_selection({'sft':{'selection':{'models':['14b'],'families':['rot13','kazitext'],'levels':[0]}}})

def test_kazitext_continuation_roundtrip_has_distinct_family():
    c=ContinuationSpec(5,1,5e-5,family='kazitext')
    assert spec_from_identity({'continuation_policy':c.policy()})==c
    assert c.policy()['condition']['family']=='kazitext'
    assert c.policy()!=ContinuationSpec(5,1,5e-5).policy()
