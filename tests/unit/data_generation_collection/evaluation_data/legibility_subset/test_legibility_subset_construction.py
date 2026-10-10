from collections import Counter
import pytest
from shared_procedures.identity import content_hash
from data_generation_collection.evaluation_data.legibility_subset.legibility_subset_construction import construct_legibility_subset,validate_legibility_subset


def panel():
    items=[]
    for j in range(3):
        items.extend({'question_id':f'g_{j}_{i}','domain':'gsm8k','stratum':f'tercile_{j+1}','population_size':440} for i in range(100))
    for s in range(57):
        items.extend({'question_id':f'm_{s}_{i}','domain':'mmlu','subject':str(s),'stratum':str(s),'population_size':90} for i in range(7))
    for d,n in [('arc',331),('mbpp',203)]:
        items.extend({'question_id':f'{d}_{i}','domain':d,'stratum':d,'population_size':n+100,'reference_tests_verified':True} for i in range(n))
    sampling={'fixture':'explicit_unit_fixture'}
    return {'status':'verified','items':items,'content_hash':content_hash(items),'sampling':sampling,'sampling_hash':content_hash(sampling)}


def test_nested_200_50_each_17_17_16_and_fixed_50_subjects():
    source=panel()
    subset=construct_legibility_subset(source)
    assert len(subset['items'])==200 and len(set(subset['question_ids']))==200
    assert Counter(i['domain'] for i in subset['items'])=={d:50 for d in ('gsm8k','mmlu','arc','mbpp')}
    assert sorted(subset['sampling']['gsm_allocation'].values())==[16,17,17]
    assert len(subset['sampling']['mmlu_subjects'])==50
    assert subset['sampling']['reuse_capability_raw_generations'] is True
    assert subset==construct_legibility_subset(source)
    assert validate_legibility_subset(subset,source)==subset
    subset['items'][0]['question_id']='not_in_parent'
    with pytest.raises(ValueError):validate_legibility_subset(subset,source)
