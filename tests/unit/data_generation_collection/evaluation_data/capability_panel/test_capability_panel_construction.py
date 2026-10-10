import pytest
from collections import Counter
from data_generation_collection.evaluation_data.capability_panel.capability_panel_construction import construct_capability_panel,extract_mbpp_interface,validate_capability_panel


def pools():
    return {
      'gsm8k':[{'question':f'GSM {i}','answer':'word '*(i%50+1)+'#### 1'} for i in range(1319)],
      'mmlu':[{'question':f'MMLU {s} {i}','subject':f'subject_{s:02d}','subject_index':i,'error_type':'ok' if i%10 else 'bad_question_clarity','choices':['a','b','c','d'],'answer':i%4} for s in range(57) for i in range(100)],
      'arc':[{'id':i,'question':f'ARC {i}','choices':{'label':['A','B'],'text':['yes','no']},'answerKey':'A'} for i in range(2376)],
      'mbpp':[{'task_id':i,'prompt':f'MBPP {i}','code':'def required(x, y=2):\n return x+y\ndef helper(z):\n return z',
               'test_list':['assert required(3) == 5','assert required(3, 4) == 7']} for i in range(427)],
    }


def test_full_fixed_counts_strata_actual_clean_populations_and_hidden_interfaces(tmp_path):
    revisions={d:'a'*40 for d in ('gsm8k','mmlu','arc','mbpp')}
    # Isolated source-verification fixture; no claim to a live MBPP sandbox.
    result=construct_capability_panel(pools(),revisions,verify_mbpp_reference=lambda code,tests:True,output_path=tmp_path/'panel.json')
    assert Counter(r['domain'] for r in result['items'])=={'gsm8k':300,'mmlu':399,'arc':331,'mbpp':203}
    assert [s['population_size'] for s in result['sampling']['gsm_strata']]==[440,440,439]
    assert len(result['sampling']['mmlu_subjects'])==57
    assert {s['population_size'] for s in result['sampling']['mmlu_subjects']}=={90}
    mbpp=next(r for r in result['items'] if r['domain']=='mbpp')
    assert mbpp['public_interface'][0]['signature']=='def required(x, y=2):'
    assert 'helper' not in str(mbpp['public_interface'])
    assert 'assert' not in str(mbpp['public_interface']) and 'return' not in str(mbpp['public_interface'])
    assert validate_capability_panel(result)==result
    assert result==construct_capability_panel(pools(),revisions,verify_mbpp_reference=lambda code,tests:True)


def test_source_test_verification_is_required_not_inferred():
    panel=construct_capability_panel(pools(),{d:'a'*40 for d in ('gsm8k','mmlu','arc','mbpp')})
    assert panel['status']=='prepared' and not panel['mbpp_reference_tests_verified']
    with pytest.raises(ValueError):validate_capability_panel(panel)
    with pytest.raises(ValueError):construct_capability_panel(pools(),{d:'a'*40 for d in ('gsm8k','mmlu','arc','mbpp')},verify_mbpp_reference=lambda c,t:False)


def test_impossible_subject_quota_is_not_resampled():
    data=pools()
    for row in data['mmlu']:
        if row['subject']=='subject_00' and row['subject_index']>5:row['error_type']='bad_question_clarity'
    with pytest.raises(ValueError,match='quota'):
        construct_capability_panel(data,{d:'a'*40 for d in data})


def test_interfaces_allow_only_literal_defaults():
    assert extract_mbpp_interface('def f(a, /, *, x=None):\n return a',['assert f(1)==1'])[0]['defaults']=={'x':'None'}
    with pytest.raises(ValueError):extract_mbpp_interface('def f(a=secret()):\n return a',['assert f()==1'])
    with pytest.raises(ValueError):extract_mbpp_interface('def helper(a):\n return a',['assert other(1)==1'])
