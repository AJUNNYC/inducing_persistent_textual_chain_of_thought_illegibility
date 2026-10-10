import numpy as np
from data_generation_collection.evaluation_data.protocol.analysis_plan.prospective_analysis_plan import create_analysis_plan
from data_generation_collection.evaluation_data.protocol.analysis_plan.simultaneous_bounds.simultaneous_bounds import qualified_opacity, maximum_standardized_bounds, design_weights, assemble_outcomes, metric_arrays, compute_simultaneous_bounds


def test_option_a_missingness_padding_and_readable_solution():
    r = {'assessable':True,'primary_score':6,'readable_solution':'not_detected','padding':'uncertain'}
    assert qualified_opacity(r) == (1,'resolved')
    assert qualified_opacity({**r,'readable_solution':'present'}) == (0,'resolved')
    assert qualified_opacity({**r,'readable_solution':'uncertain'}) == (0,'readable_solution_unresolved')
    assert qualified_opacity({**r,'primary_score':None}) == (0,'primary_unavailable')
    assert qualified_opacity({'assessable':False}) == (0,'unassessable')
    assert qualified_opacity(None) == (0,'missing_record')
    assert qualified_opacity({'assessable':False,'generation_status':'infrastructure_failure'}) == (0,'generation_unavailable')


def test_maximum_direction_quantile_and_degenerate_status():
    points = np.array([.6,.2,.9])
    reps = np.array([[.7,.1,.9],[.5,.3,.9],[.8,.2,.9],[.6,.4,.9]])
    registry = [{'gate_id':str(i),'floor':.5,'natural_lower':0} for i in range(3)]
    result = maximum_standardized_bounds(points,reps,registry)
    se = reps[:,:2].std(axis=0,ddof=1)
    expected = max(0,np.quantile(np.max((reps[:,:2]-points[:2])/se,axis=1),.95,method='higher'))
    assert result['critical_value'] == expected
    assert np.isclose(result['gates'][0]['lower_bound'],max(0,.6-expected*se[0]))
    assert result['gates'][2]['status'] == 'indeterminate'
    assert result['gates'][2]['lower_bound'] == 0
    r = maximum_standardized_bounds(points,reps,registry,[True,False,False])
    assert r['gates'][0]['reason'] == 'required_evidence_unresolved'


def test_design_fixed_subjects_nested_allocation_and_finite_census():
    items = [{'item_id':str(i),'domain':'gsm8k','stratum':str(i//4),'population_size':4} for i in range(12)]
    items += [{'item_id':f'm{i}','domain':'mmlu','stratum':str(i//3),'population_size':10} for i in range(9)]
    plan = create_analysis_plan(items,['0','1','4','5','8','9','m0','m3'],{'fixture':True})
    cap, leg = design_weights(plan,300)
    assert np.all(cap[:,:12] == 1)  # whole first-stage strata are censuses
    assert all(np.allclose(cap[:,s['indices']].sum(axis=1),len(s['indices'])) for s in plan['sample_design'])
    assert np.all(leg[:,[2,3,6,7,10,11,13,14,16,17,18,19,20]] == 0)
    assert np.any(leg[:,0] != 1)  # nested sampling still carries variance
    cap2,leg2=design_weights(plan,300)
    assert np.array_equal(cap,cap2) and np.array_equal(leg,leg2)


def test_intersections_and_same_baseline_conditional_denominator():
    rows=[{'item_id':str(i),'domain':'arc','stratum':'arc','population_size':20} for i in range(6)]
    plan=create_analysis_plan(rows,[str(i) for i in range(6)],{'fixture':True})
    def record(c,h=0): return {'correct':c,'assessable':True,'primary_score':7 if h else 2,'readable_solution':'not_detected'}
    states={'base_7b':{str(i):record(i<4,i==0) for i in range(6)}}
    for l in range(1,11):
        states[f'rot13_7b_level_{l:02d}']={str(i):record(i<4 and i != l-1, i!=0) for i in range(6)}
    arrays=metric_arrays(assemble_outcomes(states,plan),plan)
    gidx={g['gate_id']:i for i,g in enumerate(plan['gate_registry'])}
    window=next(w for w in plan['windows'] if w['window_id']=='rot13_7b_01_03')
    common=next(g for g in plan['gate_registry'] if g['kind']=='common_retention' and g['domain']=='arc' and g['window_id']==window['window_id'])
    assert arrays['point'][gidx[common['gate_id']]] == .25
    conditional=next(g for g in plan['gate_registry'] if g['kind']=='opacity_increase' and g['domain']=='arc' and g['conditional'] and g['window_id']==window['window_id'] and g['level']==1)
    j=gidx[conditional['gate_id']]
    assert arrays['observed_denominator'][j] == 1 and arrays['point'][j] == 1
    assert arrays['original_denominator'][j] == 6


def test_exact_linear_dedup_keeps_resampled_retention_and_conditional_ratios():
    rows=[{'item_id':str(i),'domain':'arc','stratum':'arc','population_size':30} for i in range(8)]
    plan=create_analysis_plan(rows,['0','1','2','3'],{'fixture':True})
    def r(c,h): return {'correct':c,'assessable':True,'primary_score':7 if h else 2,'readable_solution':'not_detected'}
    records={'base_7b':{str(i):r(i<6,i==0) for i in range(8)}}
    for l in range(1,11):
        records[f'rot13_7b_level_{l:02d}']={str(i):r(i<6 and i!=l-1,i%2==1) for i in range(8)}
    result=compute_simultaneous_bounds(records,plan,draws=64,retain_replicates=True)
    cap,leg=design_weights(plan,64)
    idx={g['gate_id']:i for i,g in enumerate(plan['gate_registry'])}
    actual=result['replicates'][:,idx['retention:rot13:7b:arc:1']]
    expected=cap[:,1:6].sum(axis=1)/cap[:,:6].sum(axis=1)
    assert np.allclose(actual,expected,rtol=0,atol=1e-14)
    conditional=next(g for g in plan['gate_registry'] if g['kind']=='opacity_increase' and g['domain']=='arc' and g['conditional'] and g['window_id']=='rot13_7b_01_03' and g['level']==1)
    assert np.allclose(result['replicates'][:,idx[conditional['gate_id']]],1)
    assert len(result['gates'])==10800
    assert all(d['unique_denominators']<d['scalar_gates'] for d in result['linear_vector_deduplication'])
