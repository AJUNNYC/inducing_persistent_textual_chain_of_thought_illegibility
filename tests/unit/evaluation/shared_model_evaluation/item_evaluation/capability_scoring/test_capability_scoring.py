import pytest
from evaluation.shared_model_evaluation.item_evaluation.capability_scoring.capability_scoring import (
    score_capability, extract_final_code, parse_choice_label, extract_scalar_final_field,
    ANSWER_EXTRACTION_VERSION,
)


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
    assert extract_final_code('```python\n'+code+'\n```\nmore text') == code
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


@pytest.mark.parametrize('answer', [
    'An explanatory solution with intermediate numbers 7 and 9.\nFinal answer: \\boxed{13}',
    'An explanatory solution.\n**Final Answer:**\n\\[\n\\boxed{13}\n\\]',
    'An explanatory solution.\n\\[\n\\boxed{13}\n\\]',
    'An explanatory solution.\n\\boxed{13}',
])
def test_designated_numeric_field_matches_separate_line_contract(answer):
    result = score_capability({'domain':'gsm8k','reference_answer':'13'}, parsed(answer))
    assert result['correct'] is True
    assert result['answer'] == '13'
    assert result['dependencies']['answer_extraction_version'] == ANSWER_EXTRACTION_VERSION
    # The extracted field is independent of the hidden reference.
    wrong = score_capability({'domain':'gsm8k','reference_answer':'14'}, parsed(answer))
    assert wrong['answer'] == result['answer'] and wrong['correct'] is False


@pytest.mark.parametrize('answer', [
    'Final answer: \\boxed{13}\nFinal answer: \\boxed{13}',
    'Earlier \\boxed{12}.\n\\boxed{13}',
    'Final answer: \\boxed{13}\nActually use 14.',
    'Final answer: 13 dollars',
    '\\boxed{13\\text{ dollars}}',
    'The result is 13',
    'Final answer: \\boxed{13}\n\\boxed{14}',
    'Final answer: \\boxed{13',
    '\\boxed{13}\n\\]',
])
def test_no_ambiguous_unit_bearing_or_inferred_numeric_answers(answer):
    assert score_capability({'domain':'gsm8k','reference_answer':'13'}, parsed(answer))['correct'] is False


def test_final_choice_line_accepts_preceding_explanation_but_not_option_prose():
    item={'domain':'mmlu','choices':['first','second'],'reference_answer':'B'}
    assert score_capability(item,parsed('I considered A, but it is false.\nFinal answer: B'))['correct'] is True
    for answer in ['Explanation.\nFinal answer: B. second',
                   'Final answer: A\nFinal answer: B',
                   'Explanation.\n\\boxed{A}\n\\boxed{B}',
                   'The correct choice is B.']:
        assert score_capability(item,parsed(answer))['correct'] is False


def test_reasoning_is_never_an_alternative_answer_and_cutoff_stays_invalid():
    item={'domain':'gsm8k','reference_answer':'13'}
    only_reasoning=parsed('No final result',reasoning='Final answer: \\boxed{13}')
    assert score_capability(item,only_reasoning)['correct'] is False
    truncated=score_capability(item,parsed('Explanation.\n\\boxed{13}',truncated=True))
    assert truncated['reason']=='truncated_output' and truncated['correct'] is False


def test_mbpp_unique_final_solution_fence_preserves_code_bytes_and_never_repairs():
    code='def add(a, b):\n\treturn a+b  '
    answer='Explanation before the solution.\n```python\n'+code+'\n```\nExplanation afterwards.'
    assert extract_final_code(answer)==code
    for ambiguous in [answer+'\n```python\ndef add(a,b):\n return 0\n```',
                      '```javascript\n'+code+'\n```',
                      '```python\n'+code,
                      '```python\n\n```']:
        assert extract_final_code(ambiguous) is None
    class Executor:
        def execute(self, actual, tests, *, entrypoints):
            assert actual==code
            return {'status':'outcome','correct':False,'reason':'syntax_error'}
    item={'domain':'mbpp','public_interface':[{'name':'add'}],'scoring_tests':['assert add(1,2)==3']}
    result=score_capability(item,parsed(answer),sandbox=Executor())
    assert result['reason']=='syntax_error' and result['correct'] is False
    class PlainFinalExecutor:
        def execute(self,actual,tests,*,entrypoints):
            # Plain final contents are forwarded unchanged; reasoning code is never substituted.
            assert actual=='No final code'
            return {'status':'outcome','correct':False,'reason':'syntax_error'}
    missing_code=score_capability(item,parsed('No final code',reasoning='```python\n'+code+'\n```'),sandbox=PlainFinalExecutor())
    assert missing_code['status']=='outcome' and missing_code['reason']=='syntax_error'
