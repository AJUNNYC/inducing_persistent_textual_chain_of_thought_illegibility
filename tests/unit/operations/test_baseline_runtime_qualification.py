"""Native baseline qualification cannot authorize a different runtime or profile."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
import reconcile_sft_launch as reconciliation
from shared_procedures.configuration import FIXED,configuration_hash
from shared_procedures.identity import content_hash,file_hash
from training.sft.execution_profile import SFTExecutionProfile
from training.sft.individual_fit.sft_fit import MODELS


def fixture_report():
    config={}
    for key,value in FIXED.items():
        target=config;parts=key.split('.')
        for part in parts[:-1]:target=target.setdefault(part,{})
        target[parts[-1]]=deepcopy(value)
    config['models']={size:{'id':name,'revision':'a'*40,'tokenizer_revision':'b'*40} for size,name in MODELS.items()}
    config['runtime']={}
    sources=[*sorted((ROOT/'src/training/sft').rglob('*.py')),
             ROOT/'src/shared_procedures/preflight.py',ROOT/'src/shared_procedures/checkpointing.py',
             ROOT/'requirements.lock',ROOT/'scripts/qualify_sft_baseline_runtime.py']
    report={'schema_version':'sft_baseline_runtime_controls_v1','status':'PASSED',
            'source_code_commit':'fixture-source','consumer_configuration_hash':configuration_hash(config),
            'control_configuration_hash':configuration_hash(config),'model_ids':['7b'],
            'models':{'7b':config['models']['7b']},'execution_profile':SFTExecutionProfile().to_dict(),
            'dependency_hash':file_hash(ROOT/'requirements.lock'),
            'source_file_hashes':{str(p.relative_to(ROOT)):file_hash(p) for p in sources},
            'hardware':{'gpu':'fixture GPU'},'training_started':False,'optimization_activated':False,'checks':{}}
    for name in ('model_runtime','loss_mask_gradients','full_checkpoint_resume'):
        model={'model':config['models']['7b'],'runtime':{'execution_profile':SFTExecutionProfile().to_dict()}} if name=='model_runtime' else {'model':'7b'}
        report['checks'][name]={'name':name,'status':'passed','detail':'Unit fixture only',
            'evidence':{'live':True,'simulation':False,'control_preflight':True,
                        'scientific_results':False,'models':[model]},'required':True}
    return config,report


def write_report(tmp_path,report):
    report['report_hash']=content_hash({k:v for k,v in report.items() if k!='report_hash'})
    p=tmp_path/'unit_only_report.json';p.write_text(json.dumps(report));return p


def test_valid_binding_keeps_only_the_measured_model(tmp_path):
    config,report=fixture_report()
    bound,checks=reconciliation.baseline_runtime_controls(write_report(tmp_path,report),config,'fixture-source')
    assert bound['model_ids']==['7b']
    assert set(checks)=={'model_runtime','loss_mask_gradients','full_checkpoint_resume'}


@pytest.mark.parametrize('mutation',[
    lambda r:r.update(model_ids=['7b','14b']),
    lambda r:r.update(consumer_configuration_hash='another-config'),
    lambda r:r.update(source_code_commit='another-source'),
    lambda r:r['source_file_hashes'].update({'requirements.lock':'0'*64}),
    lambda r:r['execution_profile'].update(gradient_checkpointing=False),
    lambda r:r['checks']['model_runtime']['evidence']['models'][0].update(model={'id':'other'}),
    lambda r:r['checks']['model_runtime']['evidence']['models'][0]['runtime']['execution_profile'].update(loss_implementation='spatial_scheduled_ce'),
    lambda r:r['checks']['full_checkpoint_resume'].update(status='failed'),
    lambda r:r['checks']['loss_mask_gradients']['evidence'].update(simulation=True),
    lambda r:r['checks']['loss_mask_gradients']['evidence']['models'][0].update(model='14b'),
])
def test_rehashed_incompatible_evidence_is_rejected(tmp_path,mutation):
    config,report=fixture_report();mutation(report)
    with pytest.raises(ValueError):
        reconciliation.baseline_runtime_controls(write_report(tmp_path,report),config,'fixture-source')
