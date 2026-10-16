"""CPU-only handoff gates, native baseline barriers and exact cached reuse."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import sqlite3

import pytest

from operations.artifacts import ArtifactService
from shared_procedures.identity import content_hash, file_hash
from shared_procedures.storage import atomic_write_json, read_json

ROOT = Path(__file__).resolve().parents[3]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value

D = load('gpqa_after_sft_dispatcher', ROOT / 'scripts/run_gpqa_after_sft_continuation.py')
H = load('gpqa_after_sft_test_helpers', ROOT / 'tests/unit/evaluation/test_gpqa_campaign.py')


def test_training_gate_waits_for_complete_and_exact_drive_checksum_records(tmp_path):
    output = tmp_path / 'training'; output.mkdir()
    atomic_write_json(output / 'fit_status.json', {'status': 'running'})
    assert D.sft_completion_gate(output, lambda *_: pytest.fail('must not inspect unfinished fit')) is None
    identity = {'run_id': 'rot13_14b_level_10_continuation_epoch_02',
                'training_budget': 'efficient_additional_epoch_v1', 'configuration_hash': 'b' * 64,
                'code_revision': 'a' * 40}
    result = {'progress': {'completed_steps': 625, 'cumulative_completed_steps': 1250},
              'checkpoint_hash': 'c' * 64, 'adapter_manifest_hash': 'd' * 64}
    atomic_write_json(output / 'fit_status.json', {'status': 'complete'})
    atomic_write_json(output / 'run_identity.json', identity)
    atomic_write_json(output / 'fit_result.json', result)
    for relative in ('checkpoint_000625/manifest.json', 'checkpoint_000625/full_state.pt',
                     'final_adapter/manifest.json', 'final_adapter/adapter_model.safetensors'):
        p = output / relative; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b'CPU checksum fixture')
    assert D.sft_completion_gate(output, lambda *_: result) is None
    service = ArtifactService(output, output / 'drive_registry.json')
    provenance = {'experiment_id': 'b' * 64, 'configuration_hash': 'b' * 64, 'source_code_commit': 'a' * 40,
                  'branch': 'rot13', 'model': '14b', 'run_id': identity['run_id'], 'stage': 'sft',
                  'producer': 'fixture', 'dataset_revision': 'e' * 64, 'model_revision': 'f' * 40, 'seeds': [42]}
    files = [output / 'fit_result.json', output / 'run_identity.json',
             *[p for p in (output / 'checkpoint_000625').iterdir()],
             *[p for p in (output / 'final_adapter').iterdir()]]
    for path in files:
        record = service.register(path, provenance)
        record.update(sync_status='VERIFIED', drive_file_id='cpu-fixture-file')
        with service._connect() as db:
            service._put(db, record)
    gate = D.sft_completion_gate(output, lambda *_: result)
    assert gate['status'] == 'VERIFIED' and gate['completed_steps'] == 625
    (output / 'checkpoint_000625/full_state.pt').write_bytes(b'changed payload')
    assert D.sft_completion_gate(output, lambda *_: result) is None


def test_exact_training_process_identity_rejects_pid_reuse(tmp_path):
    proc = tmp_path / 'proc'; entry = proc / '17152'; entry.mkdir(parents=True)
    def stat(start):
        return '17152 (name with (parentheses)) ' + ' '.join(['S'] + ['0'] * 18 + [str(start)] + ['0'] * 4)
    (entry / 'stat').write_text(stat(12345))
    assert D.process_start(17152, proc) == '12345'
    assert D.same_process_alive(17152, '12345', proc)
    (entry / 'stat').write_text(stat(56789))
    assert not D.same_process_alive(17152, '12345', proc)
    (entry / 'stat').write_text(stat(12345).replace(') S ', ') Z '))
    assert not D.same_process_alive(17152, '12345', proc)


@pytest.fixture
def frozen_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(H.CAMPAIGN, 'code_revision', lambda: D.FROZEN_REVISION)
    args = H.inputs(tmp_path)
    _, protocols = H.CAMPAIGN.campaign_plan(args)
    baseline, original = protocols[0], protocols[-1]
    H.payload_fixture(args, baseline, drive=True)
    H.payload_fixture(args, original, drive=True)
    api = D.frozen_api(ROOT)
    old_adapter = Path(original['states'][1]['adapter_path'])
    adapter = tmp_path / 'continued' / 'final_adapter'
    shutil.copytree(old_adapter, adapter)
    manifest = read_json(adapter / 'manifest.json')
    manifest['identity'].update(configuration_hash=content_hash('continued recipe'),
        training_budget='efficient_additional_epoch_v1', run_id='rot13_14b_level_10_continuation_epoch_02')
    manifest['checkpoint_hash'] = content_hash('continued checkpoint')
    manifest['manifest_hash'] = content_hash({k: v for k, v in manifest.items() if k != 'manifest_hash'})
    atomic_write_json(adapter / 'manifest.json', manifest)
    atomic_write_json(adapter.parent / 'fit_result.json', {'status': 'complete', 'identity': manifest['identity'],
        'adapter_manifest_hash': manifest['manifest_hash'], 'checkpoint_hash': manifest['checkpoint_hash']})
    return args, api, baseline, original, adapter


def test_native_compatibility_keeps_state_item_seeds_and_changes_adapter_namespace(frozen_inputs):
    args, api, baseline, original, adapter = frozen_inputs
    for protocol in (baseline, original):
        api.campaign.completion_barrier(protocol, args.output_root, save_to_drive=True,
                                        source_code_commit=D.FROZEN_REVISION)
    new = D.prepared_protocol(api, baseline, original, adapter, args.model_cache)
    assert new['generation'] == original['generation']
    assert new['generation']['settings']['max_new_tokens'] == 32000
    assert new['generation']['backend']['max_model_len'] == 40960
    assert new['generation']['backend']['max_num_seqs'] == 8
    assert new['states'][1]['state_id'] == original['states'][1]['state_id'] == D.STATE
    assert new['protocol_hash'] != original['protocol_hash']
    assert new['states'][1]['adapter_manifest_hash'] != original['states'][1]['adapter_manifest_hash']
    assert new['baseline_reuse']['state'] == baseline['states'][0]
    from shared_procedures.identity import derive_seed
    for item in new['panel']['items']:
        assert derive_seed('native_evaluation', 42, new['states'][1]['state_id'], item['question_id']) == derive_seed(
            'native_evaluation', 42, original['states'][1]['state_id'], item['question_id'])
    assert read_json(adapter / 'manifest.json')['identity']['training_budget'] == 'efficient_additional_epoch_v1'


def test_baseline_adoption_copies_only_198_raw_bytes_with_original_provenance(frozen_inputs, tmp_path):
    args, api, baseline, original, _ = frozen_inputs
    output = tmp_path / 'new-output'
    copies = D.raw_copy_plan(api, baseline, args.output_root, output)
    registered = []
    class NativeServiceFixture:
        def register(self, path, provenance):
            registered.append((path, provenance))
        def sync_artifacts(self, **kwargs):
            return {'status': 'VERIFIED'}
    D.adopt_raw(api, baseline, copies, NativeServiceFixture())
    assert len(registered) == len(copies) == 198
    for copy, (_, provenance) in zip(copies, registered):
        assert Path(copy['source']).read_bytes() == Path(copy['destination']).read_bytes()
        assert file_hash(copy['source']) == copy['sha256']
        assert provenance['source_code_commit'] == D.FROZEN_REVISION
        assert provenance['configuration_hash'] == baseline['protocol_hash']
    assert not (output / baseline['protocol_hash'] / 'completion_receipt.json').exists()
    assert not (output / baseline['protocol_hash'] / 'drive_manifest.json').exists()
    first = Path(copies[0]['destination']); first.write_bytes(b'bad cache')
    with pytest.raises(RuntimeError, match='different bytes'):
        D.adopt_raw(api, baseline, copies, NativeServiceFixture())
    assert first.read_bytes() == b'bad cache'


def test_changed_original_protocol_rejected_before_generation(frozen_inputs):
    args, api, baseline, original, adapter = frozen_inputs
    changed = deepcopy(original)
    changed['generation']['settings']['max_new_tokens'] = 3072
    with pytest.raises((ValueError, RuntimeError)):
        D.prepared_protocol(api, baseline, changed, adapter, args.model_cache)


def test_frozen_entrypoint_authentication_uses_original_sealed_plan(frozen_inputs, tmp_path):
    args, _, baseline, original, _ = frozen_inputs
    plan, _ = H.CAMPAIGN.campaign_plan(args)
    args.original_campaign_plan = tmp_path / 'original_plan.json'
    args.frozen_source = ROOT
    args.original_output_root = Path(args.output_root)
    args.baseline_protocol = args.original_output_root / baseline['protocol_hash'] / 'protocol.json'
    args.original_level10_protocol = args.original_output_root / original['protocol_hash'] / 'protocol.json'
    atomic_write_json(args.original_campaign_plan, plan)
    assert D.authenticated_source(args) == plan
    altered = deepcopy(plan)
    altered['implementation_sha256']['scripts/run_supplemental_gpqa.py'] = '0' * 64
    altered['plan_hash'] = content_hash({k: v for k, v in altered.items() if k != 'plan_hash'})
    atomic_write_json(args.original_campaign_plan, altered)
    with pytest.raises(RuntimeError, match='authenticated campaign'):
        D.authenticated_source(args)


def test_original_versus_continued_comparison_uses_native_scored_paired_items():
    def report(correct):
        rows = [{'state_id': D.STATE, 'question_id': str(i), 'score': {'correct': i in correct}}
                for i in range(198)]
        return {'protocol_hash': content_hash(sorted(correct)), 'records': rows,
                'summaries': {D.STATE: {'denominator': 198, 'observed': 198,
                                       'correct_count': len(correct), 'accuracy': len(correct) / 198}}}
    summary = D.operation_comparison(None, report({0, 1}), report({1, 2, 3}))
    assert summary['paired_transitions'] == {'both_correct': 1, 'gained_correct': 2,
                                            'lost_correct': 1, 'both_incorrect': 194}
    assert summary['baseline_regenerated'] is False
