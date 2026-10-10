import pytest
from shared_procedures.storage import read_json
from data_generation_collection.rl_data.rl_data_preparation import prepare_rl_data,validate_rl_manifest


def test_exact_numeric_800_fixed_100_by8_and_exclusions(tmp_path):
    rows=[{'id':i,'question':f'question {i}','answer':f'work steps #### {i+1:,}'} for i in range(804)]
    rows.extend([{'id':9000,'question':'bad reference','answer':'nonnumeric'},
                 {'id':9001,'question':'question 1','answer':'#### 2'}])
    result=prepare_rl_data(rows,'a'*40,['question 0'],output_path=tmp_path/'manifest.json')
    assert len(result['questions'])==800 and len(result['batches'])==100
    assert all(len(batch)==8 for batch in result['batches'])
    assert [q['order'] for q in result['questions']]==list(range(800))
    assert all('reasoning' not in q and 'answer' not in q for q in result['questions'])
    assert {r['reason'] for r in result['exclusions']}=={'detectable_evaluation_overlap','invalid_numeric_reference','normalized_duplicate'}
    assert validate_rl_manifest(read_json(tmp_path/'manifest.json'))==result
    assert result==prepare_rl_data(rows,'a'*40,['question 0'])
    result['batches'][0].reverse()
    with pytest.raises(ValueError):validate_rl_manifest(result)


def test_shortage_and_unpinned_sources_stop():
    with pytest.raises(ValueError):prepare_rl_data([],'a'*40,[])
    with pytest.raises(ValueError):prepare_rl_data([],'main',[])
