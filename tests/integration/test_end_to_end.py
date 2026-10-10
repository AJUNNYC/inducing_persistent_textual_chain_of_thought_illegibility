"""Durable fixture workflow executes real transforms, scoring, family analysis and export.

Model/provider outputs are explicit fixtures. This never certifies live preflights.
"""
from operations.workflow import Stage,StageRunner
from operations.reporting import generate_report
from shared_procedures.storage import atomic_write_json,read_json
from shared_procedures.identity import content_hash
from data_generation_collection.sft_data.rot13.rot13_data_construction import transform_rot13
from data_generation_collection.evaluation_data.protocol.analysis_plan.prospective_analysis_plan import create_analysis_plan
from evaluation.shared_model_evaluation.item_evaluation.item_evaluation import evaluate_item
from evaluation.results_analysis.results_analysis import analyze_results


def test_fixture_workflow_runs_to_report_and_rejects_corrupt_completion(tmp_path):
    item={'question_id':'q','item_id':'q','domain':'gsm8k','stratum':'tercile_1','population_size':10,'question':'Fixture','reference_answer':'4'}
    plan=create_analysis_plan([item],['q'],{'fixture':True})
    calls=[]
    graph={s.stage_id:s for s in [Stage('transform','transform',()),Stage('evaluate','evaluate',('transform',)),Stage('report','report',('evaluate',))]}
    class Generator:
        def generate(self,state,item,seed=42):
            output='<think>'+read_json(tmp_path/'transform.json')['reasoning']+'</think>4'
            return {'raw_output':output,'raw_hash':content_hash(output),'finish_reason':'stop','simulation':True}
    class Judge:
        def request(self,kind,*args,**kwargs):
            return {'status':'success','simulation':True,'parsed':{'readable_solution':'not_detected','padding':'uncertain'} if kind=='evaluation_audit' else {'score':7}}
    def action(stage):
        calls.append(stage.stage_id)
        if stage.kind=='transform':
            path=tmp_path/'transform.json';atomic_write_json(path,transform_rot13('one two three four five six','fixture',8))
            files=[path]
        elif stage.kind=='evaluate':
            row=evaluate_item({'state_id':'base_7b','model':'7b','family':'base'},item,{'protocol_hash':'a'*64,'legibility_ids':['q']},Generator(),Judge())
            path=tmp_path/'analysis.json';atomic_write_json(path,analyze_results({'base_7b':[row]},plan,confirmatory=True,draws=8));files=[path]
        else: files=generate_report(read_json(tmp_path/'analysis.json'),tmp_path/'report')
        return {'status':'VERIFIED','output_files':[str(p) for p in files]}
    runner=StageRunner(graph,tmp_path/'state',config_hash='c'*64,code_revision='d'*40)
    actions={kind:action for kind in graph}
    runner.run(['report'],actions)
    report=read_json(tmp_path/'report'/'results.json')
    assert report['simulation'] and not report['confirmatory_authorized'] and not report['construction_success']
    assert len(report['criteria']['windows'])==216 and len(report['scalar_gates'])==10800
    assert len(list((tmp_path/'report').glob('*.png')))==6
    assert 'Simulation fixture' in (tmp_path/'report'/'results.md').read_text()
    calls.clear();runner.run(['report'],actions);assert calls==[]
    (tmp_path/'transform.json').write_text('corrupt')
    runner.run(['report'],actions)
    assert calls==['transform']  # Identical restored content permits downstream reuse.
    calls.clear()
    revised=StageRunner(graph,tmp_path/'state',config_hash='e'*64,code_revision='d'*40)
    revised.run(['report'],actions)
    assert calls==['transform','evaluate','report']
