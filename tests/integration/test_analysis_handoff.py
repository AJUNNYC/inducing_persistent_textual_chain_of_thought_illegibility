"""Real producer/consumer handoff with explicitly simulated providers."""
import pytest
from data_generation_collection.evaluation_data.protocol.analysis_plan.prospective_analysis_plan import create_analysis_plan
from evaluation.shared_model_evaluation.item_evaluation.item_evaluation import evaluate_item
from evaluation.results_analysis.results_analysis import analyze_results
from shared_procedures.identity import content_hash


class FixtureGenerator:
    def generate(self, state, item, seed=42):
        output='<think>Some full reasoning with an unreadable continuation.</think>4'
        return {'raw_output':output,'raw_hash':content_hash(output),'finish_reason':'stop','simulation':True}


class FixtureJudge:
    def request(self,kind,*args,**kwargs):
        return {'status':'success','simulation':True,
                'parsed': {'readable_solution':'not_detected','padding':'uncertain'} if kind=='evaluation_audit' else {'score':7}}


def test_evaluation_record_fields_feed_complete_analysis_and_fixture_never_confirms():
    items=[{'question_id':f'q{i}','item_id':f'q{i}','domain':'gsm8k','stratum':'tercile_1','population_size':20,
            'question':f'Fixture question {i}','reference_answer':'4'} for i in range(4)]
    plan=create_analysis_plan(items,['q0','q1'],{'fixture':True})
    protocol={'protocol_hash':'a'*64,'legibility_ids':['q0','q1']}
    state={'state_id':'base_7b','model':'7b','family':'base'}
    records=[evaluate_item(state,item,protocol,FixtureGenerator(),FixtureJudge()) for item in items]
    assert all(r['correct'] is True and r['simulation'] is True for r in records)
    assert all(r['qualified_opacity'] is True for r in records[:2])
    report=analyze_results({'base_7b':{'records':records}},plan,confirmatory=True)
    assert report['simulation'] is True and report['construction_success'] is False
    assert len(report['criteria']['windows'])==216 and len(report['scalar_gates'])==10800
    base=next(s for s in report['state_summaries'] if s['state_id']=='base_7b')['domains']['gsm8k']
    assert base['capability_denominator']==4 and base['legibility_denominator']==2 and base['qualified_count']==2
    wrong=[{**r,'seed':43} for r in records]
    with pytest.raises(ValueError,match='seed42'):
        analyze_results({'base_7b':{'records':wrong}},plan)
    corrupted={'state_id':'base_7b','records':records,'package_hash':'f'*64}
    with pytest.raises(ValueError,match='integrity'):
        analyze_results({'base_7b':corrupted},plan)


def test_infrastructure_missing_baseline_generation_does_not_create_zero_opacity():
    class Broken:
        def generate(self,*args,**kwargs): raise OSError('Fixture generation failure')
    item={'question_id':'q','item_id':'q','domain':'gsm8k','stratum':'tercile_1','population_size':20,
          'question':'Fixture','reference_answer':'4'}
    plan=create_analysis_plan([item],['q'],{'fixture':True})
    protocol={'protocol_hash':'a'*64,'legibility_ids':['q']}
    state={'state_id':'base_7b','model':'7b','family':'base'}
    baseline=evaluate_item(state,item,protocol,Broken())
    trained=evaluate_item({'state_id':'rot13_7b_level_01','model':'7b','family':'rot13','level':1},item,protocol,FixtureGenerator(),FixtureJudge())
    report=analyze_results({'base_7b':[baseline],'rot13_7b_level_01':[trained]},plan)
    gate=next(g for g in report['scalar_gates'] if g['gate_id']=='opacity_increase:rot13:7b:pooled:1')
    assert gate['status']=='indeterminate' and gate['reason']=='required_evidence_unresolved'
    assert gate['estimate'] is None
