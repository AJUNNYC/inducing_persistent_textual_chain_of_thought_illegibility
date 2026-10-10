from data_generation_collection.evaluation_data.protocol.analysis_plan.prospective_analysis_plan import build_gate_registry
from evaluation.results_analysis.construction_criteria.construction_success_criteria import evaluate_construction_criteria


def test_same_nonmath_pair_entire_window_not_different_pairs_each_level():
    registry,windows=build_gate_registry()
    plan={'windows':windows}
    gates=[{'gate_id':g['gate_id'],'estimate':1,'floor':g['floor'],'status':'descriptive'} for g in registry]
    target=next(w for w in windows if w['window_id']=='rot13_7b_01_03')
    indexed={g['gate_id']:g for g in gates}
    for domain,level in [('mmlu',1),('arc',2),('mbpp',3)]:
        gid=next(g['gate_id'] for g in registry if g['branch']=='rot13' and g['model']=='7b' and g['domain']==domain and g['kind']=='opacity' and g['level']==level and g['conditional'] and g['window_id']==target['window_id'])
        indexed[gid]['estimate']=0
    report=evaluate_construction_criteria(plan,gates)
    w=next(w for w in report['windows'] if w['window_id']==target['window_id'])
    assert w['status']=='failed' and w['qualifying_pairs']==[]
    assert report['construction_success'] is False


def test_missing_gate_and_crossing_bound_no_success():
    registry,windows=build_gate_registry()
    gates=[{'gate_id':g['gate_id'],'estimate':1,'floor':g['floor'],'status':'supported','lower_bound':g['floor']-.01} for g in registry]
    result=evaluate_construction_criteria({'windows':windows},gates,confirmatory=True,authorized=True)
    assert result['status']=='indeterminate' and result['construction_success'] is False
