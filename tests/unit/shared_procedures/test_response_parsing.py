import pytest
from shared_procedures.response_parsing import *

def test_exact_boundaries_and_final_bytes():
    text='<think> a\nb </think>\n final  '
    out=parse_native_output(text)
    assert out['status']=='valid' and out['reasoning']==' a\nb ' and out['final_answer']=='\n final  '

@pytest.mark.parametrize('text',['x','<think>x</think>','<think></think>42','<think>x</think><think>y</think>42','prefix<think>x</think>42'])
def test_malformed_native_no_guess(text):assert parse_native_output(text)['status']=='malformed'

@pytest.mark.parametrize('text,expected',[('1,234.50','2469/2'),('-0','0'),(' 1 / 2.0. ','1/2'),('0.5','1/2'),('+3!','3'),('−2','-2'),('1e-3','1/1000')])
def test_exact_numeric(text,expected):assert normalize_numeric_answer(text)==expected

@pytest.mark.parametrize('text',['1,23','two','1/0','NaN','42 or 43','42 dollars','1 2'])
def test_ambiguous_numeric_rejected(text):assert normalize_numeric_answer(text) is None

def test_rl_marker_and_truncation_before_marker():
    valid='some reasoning\nFinal answer: \\boxed{0.5}'
    assert parse_rl_output(valid)['numeric_answer']=='1/2'
    assert parse_rl_output(valid,truncated=True)['status']=='valid'
    assert parse_rl_output('some reasoning\nFinal answer: \\boxed{',truncated=True)['status']=='invalid'
    assert parse_rl_output(valid+'\nFinal answer: boxed{1}')['status']=='invalid'

def test_mbpp_single_fence_no_repairs():
    assert extract_final_code('```python\ndef f():\n return 1\n```')=='def f():\n return 1'
    assert extract_final_code('```python\nx\n```\nextra') is None
    assert extract_choice_label('Answer: B')=='B'
    assert extract_choice_label('B or C') is None
