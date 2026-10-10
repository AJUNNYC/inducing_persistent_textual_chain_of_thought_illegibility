import codecs
import pytest
from data_generation_collection.sft_data.rot13.rot13_data_construction import eligible_word_spans, rot13_masks, transform_rot13


def test_original_eligibility_preserves_numbers_non_ascii_and_nested_masks():
    text='alpha βeta 123 gamma delta epsilon zeta!'
    spans=eligible_word_spans(text)
    assert [text[a:b] for a,b in spans]==['alpha','βeta','gamma','delta','epsilon','zeta']
    plan=rot13_masks(text,'id')
    assert plan['region_sizes']==[2,2,2]
    previous=set()
    for level in range(1,11):
        result=transform_rot13(text,'id',level)
        selected=set(result['selected_word_indices'])
        assert previous<=selected
        assert abs(len(selected)-level*len(spans)/10)<=1.5
        assert '123' in result['reasoning'] and 'β' in result['reasoning']
        for region in plan['orderings']:
            subset=sorted(selected & set(region))
            if subset:assert subset==list(range(subset[0],subset[-1]+1))
        previous=selected
    assert result['reasoning']==codecs.encode(text,'rot_13')
    assert result==transform_rot13(text,'id',10)


def test_short_repeated_masks_permitted_half_up_exact():
    results=[transform_rot13('one two three','short',k) for k in range(1,11)]
    assert [r['encoded_word_count'] for r in results]==[0,0,0,0,3,3,3,3,3,3]
    assert results[0]['repeated_masks']
    with pytest.raises(ValueError):transform_rot13('one two 123','id',1)
    with pytest.raises(ValueError):transform_rot13('one two three','id',False)
