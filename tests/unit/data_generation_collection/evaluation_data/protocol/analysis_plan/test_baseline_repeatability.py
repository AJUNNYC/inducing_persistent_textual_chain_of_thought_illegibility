from data_generation_collection.evaluation_data.protocol.analysis_plan.baseline_repeatability import baseline_repeatability


def test_retention_lost_successes_and_common_overlap():
    panel=[{'item_id':f'{d}{i}','domain':d} for d in ('gsm8k','mmlu','arc','mbpp') for i in range(10)]
    passes={m:{s:{r['item_id']:{'correct':int(r['item_id'][-1])<8} for r in panel} for s in (42,43,44)} for m in ('7b','14b')}
    result=baseline_repeatability(passes,panel,require_full_panel=False)
    assert result['status']=='passed' and result['ready_for_training'] is False
    passes['7b'][43]['arc0']={'correct':False}
    passes['7b'][43]['arc1']={'correct':False}
    passes['7b'][43]['arc8']={'correct':True}
    result=baseline_repeatability(passes,panel,require_full_panel=False)
    arc=result['models'][0]['domains']['arc']
    assert arc['repeat_retention']['43']==.75 and arc['status']=='failed'
    assert arc['newly_correct_ids']['43']==['arc8']
    assert result['retention_measurement']=='uncalibrated'


def test_missing_and_empty_baseline_are_indeterminate():
    panel=[{'item_id':d,'domain':d} for d in ('gsm8k','mmlu','arc','mbpp')]
    passes={m:{s:{r['item_id']:{'correct':False} for r in panel} for s in (42,43,44)} for m in ('7b','14b')}
    assert baseline_repeatability(passes,panel,require_full_panel=False)['status']=='indeterminate'
    passes['7b'][42]['arc']={'correct':True}
    del passes['7b'][44]['arc']
    result=baseline_repeatability(passes,panel,require_full_panel=False)
    assert result['models'][0]['domains']['arc']['invalid_or_unavailable_ids']['44']==['arc']
