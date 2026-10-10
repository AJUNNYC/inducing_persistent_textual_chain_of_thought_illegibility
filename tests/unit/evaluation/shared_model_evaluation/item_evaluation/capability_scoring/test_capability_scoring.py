import pytest
from evaluation.shared_model_evaluation.item_evaluation.capability_scoring.capability_scoring import score_capability, extract_final_code, parse_choice_label


def parsed(answer,**kw): return {'status':'valid','reasoning':'a reasoning trace','final_answer':answer,**kw}


def test_numeric_exact_match_without_inferring_answer():
    item={'domain':'gsm8k','reference_answer':'0.5'}
    assert score_capability(item,parsed('Final answer: \\boxed{1/2}'))['correct'] is True
    assert score_capability(item,parsed('It must be 0.5'))['correct'] is False
    assert score_capability(item,parsed('0.5',truncated=True))['reason']=='truncated_output'
    assert score_capability(item,{**parsed('0.5'),'status':'malformed'})['correct'] is False
    with pytest.raises(ValueError): score_capability({'domain':'gsm8k','reference_answer':'unknown'},parsed('1'))


def test_choice_label_ambiguity_and_wrong_label():
    item={'domain':'arc','choices':[{'label':'1','text':'one'},{'label':'2','text':'two'}],'reference_answer':'2'}
    assert score_capability(item,parsed('Final answer: 2'))['correct'] is True
    for answer in ['1','2 or 1','The answer is 2','3']:
        assert score_capability(item,parsed(answer))['correct'] is False
    assert parse_choice_label('\\boxed{B}',['A','B'])=='B'


def test_mbpp_no_repairs_and_broken_executor_is_missing():
    code='def add(a,b):\n  return a+b'
    assert extract_final_code('```python\n'+code+'\n```')==code
    assert extract_final_code('```python\n'+code+'\n```\nmore text') is None
    assert extract_final_code('```python\n```python\nx\n```\n```') is None
    item={'domain':'mbpp','public_interface':[{'name':'add'}],'scoring_tests':['assert add(1,2)==3']}
    assert score_capability(item,parsed(code))['correct'] is None
    class Executor:
        def execute(self,actual,tests,*,entrypoints):
            assert entrypoints==['add']
            assert actual==code
            return {'status':'infrastructure_failure','correct':None,'reason':'daemon_down'}
    result=score_capability(item,parsed(code),sandbox=Executor())
    assert result['status']=='infrastructure_failure' and result['correct'] is None
