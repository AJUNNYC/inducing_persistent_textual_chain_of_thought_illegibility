import pytest
from data_generation_collection.evaluation_data.protocol.analysis_plan.prospective_analysis_plan import create_analysis_plan, build_gate_registry, verify_analysis_plan


def test_full_family_and_transfer_choices():
    gates, windows = build_gate_registry()
    assert len(windows) == 216 and len(gates) == 10800
    assert len({x['window_id'] for x in windows}) == 216
    assert all(len(w['transfer_pairs']) == 3 for w in windows)
    assert all(set(w['gate_ids']['transfer']) == {'mmlu', 'arc', 'mbpp'} for w in windows)
    assert sum(len(w['levels']) == 3 for w in windows) == 48


def test_population_and_hash_requirements():
    rows = [{'item_id': 'a', 'domain': 'arc', 'stratum': 'arc', 'population_size': 3}]
    p = create_analysis_plan(rows, ['a'], {'fixture': True})
    verify_analysis_plan(p)
    p['draws'] = 2
    with pytest.raises(ValueError, match='hash'):
        verify_analysis_plan(p)
    with pytest.raises(ValueError, match='population_size'):
        create_analysis_plan([{'item_id':'a','domain':'arc'}], ['a'], {'fixture':True})
    with pytest.raises(ValueError, match='nested'):
        create_analysis_plan(rows, ['unknown'], {'fixture':True})
    with pytest.raises(ValueError, match='count'):
        create_analysis_plan(rows, ['a'])
