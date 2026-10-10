"""Declared dataset/fit fixtures validate scheduling; no trained models are made."""
from copy import deepcopy
import hashlib

import pytest

import training.sft.sft_training as scheduling
from training.sft.individual_fit.sft_fit import MODELS, SFTInputError
from shared_procedures.configuration import FIXED, configuration_hash


def fixture_config():
    config = {}
    for key, value in FIXED.items():
        current = config
        parts = key.split('.')
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = deepcopy(value)
    config['sft']['loss_reduction'] = 'mean_supervised_tokens_per_example_then_mean_examples'
    config['models'] = {size:{'id':name, 'revision':'a'*40, 'tokenizer_revision':'b'*40} for size,name in MODELS.items()}
    return config


def fixture_records(bundle):
    def make(i, split):
        return {'example_id':f'{split}_{i}', 'prompt':'Question?', 'completion':'<think>alpha beta gamma</think>42',
            'reasoning':'alpha beta gamma','final_answer':'42','split':split, 'model_id':bundle['model_id'],
            'method':bundle['family'],'level':bundle['level'],'clean_trace_hash':'1'*64,'seed':42,'status':'verified',
            'source_content_hash':hashlib.sha256(b'Question?').hexdigest(),'source_manifest_hash':'2'*64,
            'model_revision':'a'*40,'tokenizer_revision':'b'*40,'config_hash':bundle['config_hash']}
    return [make(i,'train') for i in range(10000)], [make(i,'validation') for i in range(100)]


@pytest.fixture(scope='module')
def fixtures():
    patch = pytest.MonkeyPatch()
    patch.setattr(scheduling, 'load_dataset_bundle', fixture_records)
    config = fixture_config()
    bundles = [{'family':family,'model_id':size,'level':level,'status':'verified','config_hash':configuration_hash(config)}
        for family in ('rot13','kazitext') for size in MODELS for level in range(1,11)]
    worklist = scheduling.build_worklist(config,bundles)
    yield config,bundles,worklist
    patch.undo()


def test_worklist_forty_independent_conditions_share_untouched_bases(fixtures):
    config,bundles,worklist = fixtures
    assert len(worklist) == 40 and len({job['run_id'] for job in worklist}) == 40
    assert all(job['initialization'] == 'untouched_base_or_exact_full_checkpoint' for job in worklist)
    assert len({job['spec']['model_revision'] for job in worklist}) == 1
    assert [job['level'] for job in worklist[:10]] == list(range(1,11))
    with pytest.raises(SFTInputError,match='forty'):
        scheduling.build_worklist(config,bundles[:-1])
    with pytest.raises(SFTInputError,match='Duplicate'):
        scheduling.build_worklist(config,[bundles[0],*bundles[:-1]])


def test_queue_sequential_and_stops_at_child_failure_preserving_partial(fixtures,tmp_path):
    config,_,worklist = fixtures
    calls = []
    released = []
    def runner(spec,*args,**kwargs):
        calls.append(spec.run_id)
        if spec.level == 3:
            raise RuntimeError('declared simulated disk fault')
        return {'status':'complete','run_id':spec.run_id,'identity':spec.identity()}
    report = scheduling.run_family_queue('rot13',worklist,tmp_path,config,fit_runner=runner,release_gpu=released.append)
    assert report['status'] == 'failed' and report['completed_fits'] == 2
    assert len(calls) == 3 and released == ['rot13']
    assert report['execution_mode'] == 'injected_fixture'
    assert report['failure']['run_id'] == 'rot13_7b_level_03'
    assert (tmp_path/'rot13'/'queue_report.json').is_file()


def test_worklist_rejects_condition_specific_presentation_order(fixtures,monkeypatch):
    config,bundles,_=fixtures
    def reordered(bundle):
        train,heldout=fixture_records(bundle)
        if bundle['model_id']=='7b' and bundle['family']=='rot13' and bundle['level']==2:
            train.reverse()
        return train,heldout
    monkeypatch.setattr(scheduling,'load_dataset_bundle',reordered)
    with pytest.raises(SFTInputError,match='Ordered source IDs'):
        scheduling.build_worklist(config,bundles)


def test_one_family_failure_does_not_stop_other_independent_queue(fixtures,tmp_path,monkeypatch):
    config,bundles,worklist = fixtures
    # Reuse the already validated full-count fixture worklist.
    monkeypatch.setattr(scheduling,'build_worklist',lambda config,datasets:worklist)
    calls = []
    def runner(spec,*args,**kwargs):
        calls.append(spec.run_id)
        if spec.family == 'rot13':
            raise RuntimeError('declared family-local fault')
        return {'status':'complete','run_id':spec.run_id,'identity':spec.identity()}
    result = scheduling.sft_training(config,bundles,tmp_path,fit_runner=runner)
    assert result['status'] == 'partial' and result['completed_fits'] == 20
    assert result['queues']['rot13']['status'] == 'failed'
    assert result['queues']['kazitext']['status'] == 'complete'
    assert len(calls) == 21


def test_pause_and_invalid_queue_cannot_release_incomplete_adapters(fixtures,tmp_path):
    config,_,worklist = fixtures
    def runner(spec,*args,**kwargs):
        return {'status':'paused','run_id':spec.run_id}
    report = scheduling.run_family_queue('rot13',worklist,tmp_path,config,fit_runner=runner)
    assert report['status'] == 'paused' and report['completed_fits'] == 0
    with pytest.raises(SFTInputError,match='twenty'):
        scheduling.run_family_queue('rot13',worklist[:-1][1:],tmp_path,config,fit_runner=runner)
