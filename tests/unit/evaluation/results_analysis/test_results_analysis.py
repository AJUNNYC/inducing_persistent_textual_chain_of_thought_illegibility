import pytest
from data_generation_collection.evaluation_data.protocol.analysis_plan.prospective_analysis_plan import create_analysis_plan
from evaluation.results_analysis.results_analysis import analyze_results
from evaluation.results_analysis.results_analysis import judge_agreement, raw_score_difference_intervals
from data_generation_collection.evaluation_data.protocol.analysis_plan.simultaneous_bounds.simultaneous_bounds import assemble_outcomes


def test_unknown_baseline_not_zero_favorable_difference_and_all_windows_reported():
    rows=[{'item_id':f'{d}{i}','domain':d,'stratum':d,'population_size':20} for d in ('gsm8k','mmlu','arc','mbpp') for i in range(3)]
    plan=create_analysis_plan(rows,[r['item_id'] for r in rows if not r['item_id'].endswith('2')],{'fixture':True})
    records={'base_7b':{r['item_id']:{'correct':True,'assessable':True,'primary_score':None,'readable_solution':'not_detected'} for r in rows}}
    for l in range(1,11):
        records[f'rot13_7b_level_{l:02d}']={r['item_id']:{'correct':True,'assessable':True,'primary_score':7,'secondary_score':6,'readable_solution':'not_detected','padding':'present'} for r in rows}
    report=analyze_results(records,plan,confirmatory=True)
    assert len(report['criteria']['windows'])==216
    assert report['construction_success'] is False and report['confirmatory_authorized'] is False
    assert any(g['reason']=='required_evidence_unresolved' for g in report['scalar_gates'] if g['gate_id'].startswith('opacity_increase:rot13:7b'))
    assert all(g['estimate'] is None for g in report['scalar_gates'] if g['gate_id'].startswith('opacity_increase:rot13:7b'))
    summary=next(s for s in report['state_summaries'] if s['state_id']=='rot13_7b_level_01')['domains']['arc']
    assert summary['qualified_padding_fraction']==1
    assert summary['judge_agreement']['mean_absolute_difference']==1


def test_ordinal_confusion_weighted_kappa_and_undefined_single_category():
    agreement=judge_agreement([(1,2),(2,1),(3,3)])
    assert agreement['confusion_matrix'][0][1]==1
    assert agreement['primary_minus_secondary_distribution']=={'-1':1,'0':1,'1':1}
    assert agreement['quadratically_weighted_kappa']==pytest.approx(.5)
    assert judge_agreement([(7,6),(7,8)])['quadratically_weighted_kappa']==pytest.approx(0.)
    identical_constants=judge_agreement([(7,7),(7,7)])
    assert identical_constants['quadratically_weighted_kappa'] is None
    assert identical_constants['kappa_undefined_reason']=='zero_expected_disagreement'
    with pytest.raises(ValueError,match='integer'):
        judge_agreement([(6.5,7)])


def test_raw_score_intervals_are_paired_and_missingness_remains_indeterminate():
    rows=[{'item_id':str(i),'domain':'arc','stratum':'arc','population_size':30} for i in range(8)]
    plan=create_analysis_plan(rows,['0','1','2','3'],{'fixture':True})
    def r(score): return {'correct':True,'assessable':True,'primary_score':score,'secondary_score':score,'readable_solution':'not_detected'}
    records={'base_7b':{str(i):r(i+1) for i in range(4)},
             'rot13_7b_level_01':{str(i):r(i+2+(i%2)) for i in range(4)}}
    intervals=raw_score_difference_intervals(assemble_outcomes(records,plan),plan,draws=100)
    assert len(intervals)==480
    measured=[x for x in intervals if x['paired_denominator']]
    assert len(measured)==2 and all(x['estimate']==1.5 for x in measured)
    assert all(x['simultaneous'] is False and x['lower']<=x['estimate']<=x['upper'] for x in measured)
    records['rot13_7b_level_01']['0']['primary_score']=None
    intervals=raw_score_difference_intervals(assemble_outcomes(records,plan),plan,draws=100)
    primary=next(x for x in intervals if x['judge']=='primary_score' and x['paired_denominator'])
    assert primary['status']=='indeterminate' and primary['paired_denominator']==3 and primary['original_denominator']==4
