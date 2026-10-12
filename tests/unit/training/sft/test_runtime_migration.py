"""Declared CPU fixtures exercise migration guards, never live GPU approval."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from shared_procedures.checkpointing import _same_state, load_checkpoint, save_checkpoint
from shared_procedures.identity import content_hash, file_hash
from training.sft.execution_profile import SFTExecutionProfile
from training.sft.individual_fit.sft_fit import SFTFitSpec, capture_rng, restore_rng
from training.sft.runtime_migration import (
    IMMUTABLE_IDENTITY_FIELDS, REQUIRED_SOURCE_FILES, ROOT, MigrationError,
    migrate_checkpoint, validate_execution_benchmark, validate_identity_compatibility,
)


def identities():
    old = {**SFTFitSpec("rot13", "7b", 1, "a" * 40, "a" * 40, "b" * 64, "c" * 64,
                       "d" * 64, "e" * 64).identity(),
           "code_revision": "f" * 40, "dependency_hash": "1" * 64}
    new = {**old, "code_revision": "2" * 40, "configuration_hash": "3" * 64,
           "handoff_certificate_hash": "4" * 64,
           "execution_profile": SFTExecutionProfile(cache_collated_inputs=True).to_dict()}
    return old, new


def rehash(report):
    plan = report["plan"]
    plan["plan_hash"] = content_hash({key: value for key, value in plan.items() if key != "plan_hash"})
    report["plan_hash"] = plan["plan_hash"]
    for row in [*report["profiles"].values(), *report["profile_runs"].values()]:
        row["plan_hash"] = plan["plan_hash"]
    report["receipt_hash"] = content_hash({key: value for key, value in report.items() if key != "receipt_hash"})
    return report


def measured_fixture(identity, manifest=None):
    profile = SFTExecutionProfile(cache_collated_inputs=True).to_dict()
    binding = None if manifest is None else {key: deepcopy(manifest[key])
                                            for key in ("checkpoint_hash", "identity", "progress", "files")}
    hardware = {"gpu": "DECLARED_CPU_TEST_GPU_FIXTURE", "vram_bytes": 96 * 2**30, "cuda": "fixture"}
    plan = {"schema_version": "isolated_sft_execution_benchmark_v2", "source_code_commit": identity["code_revision"],
            "consumer_bindings_valid": True, "binding_errors": [], "source_tree_validation": {"errors": []},
            "source_file_hashes": {name: file_hash(ROOT / name) for name in REQUIRED_SOURCE_FILES},
            "configuration_hash": identity["configuration_hash"],
            "handoff_certificate_hash": identity["handoff_certificate_hash"],
            "condition": {"family": "rot13", "model": "7b", "level": 1, "spec": identity},
            "profiles": {"baseline": SFTExecutionProfile().to_dict(), "cuda_cached": profile},
            "warmup_updates": 1, "measured_updates": 5, "paired_repetitions": 2,
            "technical_tolerances": {"loss_atol": 1e-5, "loss_rtol": 1e-4,
                "gradient_relative_l2": 1e-3, "update_relative_l2": 1e-3,
                "gradient_cosine_floor": 0.999, "minimum_speedup": 1.05,
                "require_identical_rng_states": True,
                "speedup_rule": "minimum speedup across corresponding logical update pairs in every repetition"},
            "recovery_checkpoint": binding}
    def row(name, repetition):
        settings = plan["profiles"][name]
        recovery = None if binding is None else {**deepcopy(binding), "full_state_restore_verified": True,
                    **{key: True for key in ("adapter_verified", "optimizer_verified", "scheduler_verified",
                                            "rng_verified", "progress_verified")}}
        return {"status": "MEASURED", "profile": name, "settings": settings,
                "runtime": {"execution_profile": settings}, "finite": True, "hardware": hardware,
                "packages": {"torch": "declared_cpu_fixture"}, "recovery_checkpoint": recovery,
                "sample_ids": ["fixture_" + str(i) for i in range(96)], "tokenized_records_hash": "5" * 64,
                "initial_adapter_hash": "6" * 64, "initial_rng_hash": "7" * 64, "final_rng_hash": "8" * 64,
                "wall_seconds_per_update": [10, 20, 30, 40, 50] if name == "baseline" else [8, 16, 24, 32, 40]}
    rows = {f"{name}_{repeat}": row(name, repeat) for repeat in (1, 2) for name in plan["profiles"]}
    vector = {"compatible": True, "relative_l2": 0.0, "cosine": 1.0, "bitwise_equal": True}
    comparison = {"technical_gate": "PASSED", "numerical_screen_passed": True,
                  "speedup_threshold_passed": True, "final_rng_states_identical": True,
                  "losses_within_declared_tolerance": True, "invariant_failures": [],
                  "gradients": vector, "adapter_updates": vector,
                  "paired_update_speedups": [1.25] * 5, "minimum_paired_update_speedup": 1.25,
                  "observed_minimum_speedup": 0.25, "reference_profile": "baseline"}
    repetitions = [{**deepcopy(comparison), "reference_run_id": f"baseline_{i}",
                    "candidate_run_id": f"cuda_cached_{i}"} for i in (1, 2)]
    report = {"schema_version": "isolated_sft_execution_benchmark_v2", "status": "MEASURED", "plan": plan,
              "profiles": {name: deepcopy(rows[name + "_1"]) for name in plan["profiles"]},
              "profile_runs": rows, "comparisons": {"cuda_cached": {**comparison, "repetitions": repetitions}}}
    return rehash(report)


def incremental_fixture(identity, manifest=None):
    """Declared no-checkpoint parent versus one new loss-layout candidate."""
    report = measured_fixture(identity, manifest)
    reference = SFTExecutionProfile(gradient_checkpointing=False).to_dict()
    candidate = SFTExecutionProfile(gradient_checkpointing=False, loss_implementation="flat_ce").to_dict()
    report["plan"]["reference_profile"] = "no_gradient_checkpointing"
    report["plan"]["profiles"] = {"no_gradient_checkpointing": reference, "flat_ce": candidate}
    for container in (report["profiles"], report["profile_runs"]):
        for row in container.values():
            is_reference = row["profile"] == "baseline"
            settings = reference if is_reference else candidate
            row["profile"] = "no_gradient_checkpointing" if is_reference else "flat_ce"
            row["settings"] = settings
            row["runtime"] = {"execution_profile": settings, "loss_implementation": settings["loss_implementation"]}
            row["actual_execution_policy"] = {"loss_implementation": settings["loss_implementation"]}
            row["longest_sequence_probe"] = {"loss_implementation": settings["loss_implementation"]}
    report["profiles"] = {"no_gradient_checkpointing": report["profiles"]["baseline"],
                           "flat_ce": report["profiles"]["cuda_cached"]}
    comparison = report["comparisons"]["cuda_cached"]
    comparison["reference_profile"] = "no_gradient_checkpointing"
    for repeat in comparison["repetitions"]:
        repeat["reference_profile"] = "no_gradient_checkpointing"
    report["comparisons"] = {"flat_ce": comparison}
    return rehash(report)


def spatial_backward_fixture(identity):
    report = incremental_fixture(identity)
    settings = SFTExecutionProfile(gradient_checkpointing=False, loss_implementation='spatial_fused_backward').to_dict()
    name = 'spatial_fused_backward'
    report['plan']['profiles'][name] = settings
    del report['plan']['profiles']['flat_ce']
    helper = 'src/training/sft/spatial_loss.py'
    digest = file_hash(ROOT/helper)
    report['plan']['source_file_hashes'][helper] = digest
    for container in (report['profiles'], report['profile_runs']):
        for row in container.values():
            if row['profile'] == 'flat_ce':
                row['profile'] = name
                row['settings'] = settings
                row['runtime']['execution_profile'] = settings
                for field in ('runtime', 'actual_execution_policy', 'longest_sequence_probe'):
                    row[field]['loss_implementation'] = 'spatial_fused_backward'
                    row[field]['loss_helper_sha256'] = digest
    report['profiles'][name] = report['profiles'].pop('flat_ce')
    report['comparisons'][name] = report['comparisons'].pop('flat_ce')
    return rehash(report)


def scheduled_spatial_fixture(identity, manifest=None):
    """Declared accepted spatial-backward parent versus scheduling only."""
    report = incremental_fixture(identity, manifest)
    reference_name = 'spatial_fused_backward_no_gradient_checkpointing'
    candidate_name = 'spatial_scheduled_ce_no_gradient_checkpointing'
    reference = SFTExecutionProfile(gradient_checkpointing=False, loss_implementation='spatial_fused_backward').to_dict()
    candidate = SFTExecutionProfile(gradient_checkpointing=False, loss_implementation='spatial_scheduled_ce').to_dict()
    helpers = {'spatial_fused_backward': 'src/training/sft/spatial_loss.py',
               'spatial_scheduled_ce': 'src/training/sft/scheduled_spatial_loss.py'}
    report['plan']['reference_profile'] = reference_name
    report['plan']['profiles'] = {reference_name: reference, candidate_name: candidate}
    for path in helpers.values():
        report['plan']['source_file_hashes'][path] = file_hash(ROOT/path)
    for container in (report['profiles'], report['profile_runs']):
        for row in container.values():
            is_reference = row['profile'] == 'no_gradient_checkpointing'
            settings = reference if is_reference else candidate
            row['profile'] = reference_name if is_reference else candidate_name
            row['settings'] = settings
            row['runtime']['execution_profile'] = settings
            for field in ('runtime', 'actual_execution_policy', 'longest_sequence_probe'):
                row[field]['loss_implementation'] = settings['loss_implementation']
                row[field]['loss_helper_sha256'] = report['plan']['source_file_hashes'][helpers[settings['loss_implementation']]]
    report['profiles'] = {reference_name: report['profiles']['no_gradient_checkpointing'],
                          candidate_name: report['profiles']['flat_ce']}
    comparison = report['comparisons'].pop('flat_ce')
    comparison['reference_profile'] = reference_name
    for repetition in comparison['repetitions']:
        repetition['reference_profile'] = reference_name
    report['comparisons'][candidate_name] = comparison
    return rehash(report)


def checkpointed_scheduled_fixture(identity, attention='eager'):
    """A separate native baseline comparison retains checkpointing throughout."""
    report = measured_fixture(identity)
    name = 'spatial_scheduled_ce_checkpointed'
    settings = SFTExecutionProfile(attention_implementation=attention,
                                   loss_implementation='spatial_scheduled_ce').to_dict()
    report['plan']['reference_profile'] = 'baseline'
    report['plan']['profiles'][name] = settings
    del report['plan']['profiles']['cuda_cached']
    for helper in ('spatial_loss.py', 'scheduled_spatial_loss.py'):
        path = 'src/training/sft/' + helper
        report['plan']['source_file_hashes'][path] = file_hash(ROOT/path)
    digest = report['plan']['source_file_hashes']['src/training/sft/scheduled_spatial_loss.py']
    for container in (report['profiles'], report['profile_runs']):
        for row in container.values():
            if row['profile'] == 'cuda_cached':
                row['profile'] = name
                row['settings'] = settings
                row['runtime']['execution_profile'] = settings
                for field in ('runtime', 'actual_execution_policy', 'longest_sequence_probe'):
                    row.setdefault(field, {}).update(loss_implementation='spatial_scheduled_ce',
                                                      loss_helper_sha256=digest)
    report['profiles'][name] = report['profiles'].pop('cuda_cached')
    report['comparisons'][name] = report['comparisons'].pop('cuda_cached')
    return rehash(report)


def causal_gradient_fixture(identity, manifest=None):
    """Declared exact-state scheduling parent versus causal gradient layout."""
    report = incremental_fixture(identity, manifest)
    reference_name = 'spatial_scheduled_ce_no_gradient_checkpointing'
    candidate_name = 'causal_contiguous_gradient_ce_no_gradient_checkpointing'
    reference = SFTExecutionProfile(gradient_checkpointing=False, loss_implementation='spatial_scheduled_ce').to_dict()
    candidate = SFTExecutionProfile(gradient_checkpointing=False, loss_implementation='causal_contiguous_gradient_ce').to_dict()
    helpers = {'spatial_fused_backward': 'src/training/sft/spatial_loss.py',
               'spatial_scheduled_ce': 'src/training/sft/scheduled_spatial_loss.py',
               'causal_contiguous_gradient_ce': 'src/training/sft/causal_gradient_loss.py'}
    report['plan']['reference_profile'] = reference_name
    report['plan']['profiles'] = {reference_name: reference, candidate_name: candidate}
    report['plan']['technical_tolerances']['require_exact_state_comparison'] = True
    for path in helpers.values():
        report['plan']['source_file_hashes'][path] = file_hash(ROOT/path)
    for container in (report['profiles'], report['profile_runs']):
        for row in container.values():
            is_reference = row['profile'] == 'no_gradient_checkpointing'
            settings = reference if is_reference else candidate
            row['profile'] = reference_name if is_reference else candidate_name
            row['settings'] = settings
            row['runtime']['execution_profile'] = settings
            row['losses'] = [1.25] * 96
            for field in ('initial_optimizer_state_hash', 'initial_scheduler_state_hash', 'clipped_gradient_hash',
                          'adapter_delta_hash', 'final_adapter_hash', 'final_optimizer_state_hash',
                          'final_scheduler_state_hash'):
                row[field] = 'a' * 64
            for field in ('runtime', 'actual_execution_policy', 'longest_sequence_probe'):
                row[field]['loss_implementation'] = settings['loss_implementation']
                row[field]['loss_helper_sha256'] = report['plan']['source_file_hashes'][helpers[settings['loss_implementation']]]
    report['profiles'] = {reference_name: report['profiles']['no_gradient_checkpointing'],
                          candidate_name: report['profiles']['flat_ce']}
    comparison = report['comparisons'].pop('flat_ce')
    for row in [comparison, *comparison['repetitions']]:
        row.update(reference_profile=reference_name, exact_state_comparison_passed=True,
                   final_optimizer_states_identical=True, final_scheduler_states_identical=True,
                   losses_exactly_identical=True, final_adapters_identical=True,
                   maximum_loss_absolute_difference=0.0)
    report['comparisons'][candidate_name] = comparison
    return rehash(report)


@pytest.mark.parametrize("field", IMMUTABLE_IDENTITY_FIELDS)
def test_scientific_checkpoint_identity_cannot_change(field):
    old, new = identities()
    new[field] = {"changed": True}
    with pytest.raises(MigrationError, match="Immutable checkpoint identity changed"):
        validate_identity_compatibility(old, new)


def test_legacy_profile_defaults_and_explicit_identity_lineage():
    old, new = identities()
    old["execution_profile"] = {"attention_implementation": "eager", "validated_loss_counts": False,
                                "defer_finite_checks": False}
    compatible = validate_identity_compatibility(old, new)
    assert compatible["source_profile"] == SFTExecutionProfile().to_dict()
    assert compatible["target_profile"] == new["execution_profile"]
    assert compatible["scientific_identity_unchanged"] is True
    new["unspecified_override"] = True
    with pytest.raises(MigrationError, match="Undeclared"):
        validate_identity_compatibility(old, new)


def test_paired_update_screen_accepts_matching_work_despite_unpaired_length_range():
    _, new = identities()
    result = validate_execution_benchmark(measured_fixture(new), new["code_revision"], new["execution_profile"])
    assert result["accepted"] is True
    assert result["hardware"]["gpu"] == "DECLARED_CPU_TEST_GPU_FIXTURE"
    assert result["configuration_hash"] == new["configuration_hash"]
    assert result["condition"] == {"family": "rot13", "model": "7b", "level": 1}
    assert result["recovery_validated"] is False
    assert result["scientific_equivalence_established"] is False


def test_declared_no_checkpoint_reference_measures_incremental_flat_ce_gain():
    _, new = identities()
    new["execution_profile"] = SFTExecutionProfile(gradient_checkpointing=False, loss_implementation="flat_ce").to_dict()
    report = incremental_fixture(new)
    result = validate_execution_benchmark(report, new["code_revision"], new["execution_profile"])
    assert result["reference_profile"] == "no_gradient_checkpointing"
    assert result["reference_execution_profile"]["gradient_checkpointing"] is False
    assert result["reference_execution_profile"]["loss_implementation"] == "spatial_ce"
    assert result["execution_profile"]["loss_implementation"] == "flat_ce"


def test_spatial_backward_candidate_binds_actual_loaded_helper_and_original_reference():
    _, new = identities()
    new['execution_profile'] = SFTExecutionProfile(gradient_checkpointing=False, loss_implementation='spatial_fused_backward').to_dict()
    report = spatial_backward_fixture(new)
    result = validate_execution_benchmark(report, new['code_revision'], new['execution_profile'])
    assert result['reference_execution_profile']['loss_implementation'] == 'spatial_ce'
    assert result['execution_profile']['loss_implementation'] == 'spatial_fused_backward'
    assert result['scientific_equivalence_established'] is False


@pytest.mark.parametrize('field', ('runtime', 'actual_execution_policy', 'longest_sequence_probe'))
def test_spatial_backward_candidate_rejects_missing_or_replaced_loaded_helper(field):
    _, new = identities()
    new['execution_profile'] = SFTExecutionProfile(gradient_checkpointing=False, loss_implementation='spatial_fused_backward').to_dict()
    report = spatial_backward_fixture(new)
    report['profile_runs']['cuda_cached_2'][field]['loss_helper_sha256'] = 'another-loaded-module'
    rehash(report)
    with pytest.raises(MigrationError, match='loaded spatial-loss helper'):
        validate_execution_benchmark(report, new['code_revision'], new['execution_profile'])


def test_spatial_backward_helper_cannot_be_omitted_from_executed_source_bindings():
    _, new = identities()
    new['execution_profile'] = SFTExecutionProfile(gradient_checkpointing=False, loss_implementation='spatial_fused_backward').to_dict()
    report = spatial_backward_fixture(new)
    del report['plan']['source_file_hashes']['src/training/sft/spatial_loss.py']
    rehash(report)
    with pytest.raises(MigrationError, match='source-file bindings'):
        validate_execution_benchmark(report, new['code_revision'], new['execution_profile'])


def test_scheduled_ce_binds_both_loaded_helpers_and_incremental_accepted_reference():
    _, new = identities()
    new['execution_profile'] = SFTExecutionProfile(gradient_checkpointing=False, loss_implementation='spatial_scheduled_ce').to_dict()
    result = validate_execution_benchmark(scheduled_spatial_fixture(new), new['code_revision'], new['execution_profile'])
    assert result['reference_profile'] == 'spatial_fused_backward_no_gradient_checkpointing'
    assert result['reference_execution_profile']['loss_implementation'] == 'spatial_fused_backward'
    assert result['execution_profile']['loss_implementation'] == 'spatial_scheduled_ce'
    assert result['scientific_equivalence_established'] is False


@pytest.mark.parametrize('attention', ('eager', 'sdpa'))
def test_checkpointed_scheduled_ce_accepts_direct_native_baseline_measurement(attention):
    _, new = identities()
    new['execution_profile'] = SFTExecutionProfile(attention_implementation=attention,
        loss_implementation='spatial_scheduled_ce').to_dict()
    result = validate_execution_benchmark(checkpointed_scheduled_fixture(new, attention),
                                          new['code_revision'], new['execution_profile'])
    assert result['reference_execution_profile'] == SFTExecutionProfile().to_dict()
    assert result['execution_profile']['gradient_checkpointing'] is True
    assert result['execution_profile']['attention_implementation'] == attention
    assert result['scientific_equivalence_established'] is False


@pytest.mark.parametrize('field,value', (
    ('attention_implementation', 'sdpa'), ('loss_implementation', 'spatial_fused_backward'),
    ('gradient_checkpointing', False), ('validated_loss_counts', True),
    ('defer_finite_checks', True), ('cache_collated_inputs', True)))
def test_checkpointed_scheduled_ce_rejects_any_nonbaseline_reference(field, value):
    _, new = identities()
    new['execution_profile'] = SFTExecutionProfile(loss_implementation='spatial_scheduled_ce').to_dict()
    report = checkpointed_scheduled_fixture(new)
    report['plan']['profiles']['baseline'][field] = value
    rehash(report)
    with pytest.raises(MigrationError, match='unchanged eager spatial-CE baseline'):
        validate_execution_benchmark(report, new['code_revision'], new['execution_profile'])


@pytest.mark.parametrize('field', ('validated_loss_counts', 'defer_finite_checks', 'cache_collated_inputs'))
def test_checkpointed_scheduled_ce_cannot_bundle_unmeasured_execution_flags(field):
    _, new = identities()
    new['execution_profile'] = SFTExecutionProfile(loss_implementation='spatial_scheduled_ce',
                                                **{field: True}).to_dict()
    report = checkpointed_scheduled_fixture(new)
    report['plan']['profiles']['spatial_scheduled_ce_checkpointed'][field] = True
    rehash(report)
    with pytest.raises(MigrationError, match='unchanged eager spatial-CE baseline'):
        validate_execution_benchmark(report, new['code_revision'], new['execution_profile'])


def test_checkpointed_scheduled_ce_keeps_native_numerical_acceptance_gate():
    _, new = identities()
    new['execution_profile'] = SFTExecutionProfile(loss_implementation='spatial_scheduled_ce').to_dict()
    report = checkpointed_scheduled_fixture(new)
    report['comparisons']['spatial_scheduled_ce_checkpointed']['numerical_screen_passed'] = False
    rehash(report)
    with pytest.raises(MigrationError, match='numerical, speed or RNG screen'):
        validate_execution_benchmark(report, new['code_revision'], new['execution_profile'])


@pytest.mark.parametrize('run', ('baseline_2', 'cuda_cached_2'))
@pytest.mark.parametrize('field', ('runtime', 'actual_execution_policy', 'longest_sequence_probe'))
def test_scheduled_ce_cannot_relabel_the_actual_reference_or_candidate_helper(run, field):
    _, new = identities()
    new['execution_profile'] = SFTExecutionProfile(gradient_checkpointing=False, loss_implementation='spatial_scheduled_ce').to_dict()
    report = scheduled_spatial_fixture(new)
    report['profile_runs'][run][field]['loss_helper_sha256'] = 'unmeasured-loaded-module'
    rehash(report)
    with pytest.raises(MigrationError, match='loaded spatial-loss helper'):
        validate_execution_benchmark(report, new['code_revision'], new['execution_profile'])


@pytest.mark.parametrize('helper', ('spatial_loss.py', 'scheduled_spatial_loss.py'))
def test_scheduled_ce_requires_both_helper_source_dependencies(helper):
    _, new = identities()
    new['execution_profile'] = SFTExecutionProfile(gradient_checkpointing=False, loss_implementation='spatial_scheduled_ce').to_dict()
    report = scheduled_spatial_fixture(new)
    del report['plan']['source_file_hashes']['src/training/sft/'+helper]
    rehash(report)
    with pytest.raises(MigrationError, match='source-file bindings'):
        validate_execution_benchmark(report, new['code_revision'], new['execution_profile'])


@pytest.mark.parametrize('change', ('older_reference', 'attention', 'counts', 'checkpointing'))
def test_scheduled_ce_must_compare_only_scheduling_against_the_accepted_profile(change):
    _, new = identities()
    new['execution_profile'] = SFTExecutionProfile(gradient_checkpointing=False, loss_implementation='spatial_scheduled_ce').to_dict()
    report = scheduled_spatial_fixture(new)
    reference = report['plan']['profiles']['spatial_fused_backward_no_gradient_checkpointing']
    if change == 'older_reference':
        reference['loss_implementation'] = 'spatial_ce'
    elif change == 'attention':
        reference['attention_implementation'] = 'sdpa'
    elif change == 'counts':
        reference['validated_loss_counts'] = True
    else:
        reference['gradient_checkpointing'] = True
    rehash(report)
    with pytest.raises(MigrationError, match='unchanged accepted spatial-backward reference'):
        validate_execution_benchmark(report, new['code_revision'], new['execution_profile'])


def test_causal_gradient_candidate_requires_exact_actual_states_and_accepted_schedule():
    _, new = identities()
    new['execution_profile'] = SFTExecutionProfile(gradient_checkpointing=False, loss_implementation='causal_contiguous_gradient_ce').to_dict()
    result = validate_execution_benchmark(causal_gradient_fixture(new), new['code_revision'], new['execution_profile'])
    assert result['reference_execution_profile']['loss_implementation'] == 'spatial_scheduled_ce'
    assert result['exact_state_comparison_required'] is True
    assert result['exact_state_comparison_passed'] is True
    assert result['scientific_equivalence_established'] is False


@pytest.mark.parametrize('field', ('initial_optimizer_state_hash', 'initial_scheduler_state_hash',
    'clipped_gradient_hash', 'adapter_delta_hash', 'final_adapter_hash', 'final_optimizer_state_hash',
    'final_scheduler_state_hash'))
def test_causal_layout_exact_flag_cannot_hide_different_actual_state_hash(field):
    _, new = identities()
    new['execution_profile'] = SFTExecutionProfile(gradient_checkpointing=False, loss_implementation='causal_contiguous_gradient_ce').to_dict()
    report = causal_gradient_fixture(new)
    report['profile_runs']['cuda_cached_2'][field] = 'b' * 64
    rehash(report)
    with pytest.raises(MigrationError, match='exact sampled losses and training states'):
        validate_execution_benchmark(report, new['code_revision'], new['execution_profile'])


@pytest.mark.parametrize('failure', ('aggregate_flag', 'pair_flag', 'bitwise_gradient', 'bitwise_delta', 'actual_loss'))
def test_causal_layout_cannot_accept_tolerance_only_or_inexact_measurements(failure):
    _, new = identities()
    new['execution_profile'] = SFTExecutionProfile(gradient_checkpointing=False, loss_implementation='causal_contiguous_gradient_ce').to_dict()
    report = causal_gradient_fixture(new)
    comparison = report['comparisons']['causal_contiguous_gradient_ce_no_gradient_checkpointing']
    if failure == 'aggregate_flag':
        comparison['exact_state_comparison_passed'] = False
    elif failure == 'pair_flag':
        comparison['repetitions'][1]['exact_state_comparison_passed'] = False
    elif failure == 'bitwise_gradient':
        comparison['repetitions'][1]['gradients']['bitwise_equal'] = False
    elif failure == 'bitwise_delta':
        comparison['repetitions'][1]['adapter_updates']['bitwise_equal'] = False
    else:
        report['profile_runs']['cuda_cached_2']['losses'][0] += 1e-10
    rehash(report)
    with pytest.raises(MigrationError, match='exact sampled losses and training states'):
        validate_execution_benchmark(report, new['code_revision'], new['execution_profile'])


@pytest.mark.parametrize('helper', ('spatial_loss.py', 'scheduled_spatial_loss.py', 'causal_gradient_loss.py'))
def test_causal_layout_requires_every_original_and_new_helper_source(helper):
    _, new = identities()
    new['execution_profile'] = SFTExecutionProfile(gradient_checkpointing=False, loss_implementation='causal_contiguous_gradient_ce').to_dict()
    report = causal_gradient_fixture(new)
    del report['plan']['source_file_hashes']['src/training/sft/'+helper]
    rehash(report)
    with pytest.raises(MigrationError, match='source-file bindings'):
        validate_execution_benchmark(report, new['code_revision'], new['execution_profile'])


@pytest.mark.parametrize('run', ('baseline_2', 'cuda_cached_2'))
@pytest.mark.parametrize('field', ('runtime', 'actual_execution_policy', 'longest_sequence_probe'))
def test_causal_layout_binds_actual_loaded_reference_and_candidate_helper(run, field):
    _, new = identities()
    new['execution_profile'] = SFTExecutionProfile(gradient_checkpointing=False, loss_implementation='causal_contiguous_gradient_ce').to_dict()
    report = causal_gradient_fixture(new)
    report['profile_runs'][run][field]['loss_helper_sha256'] = 'b' * 64
    rehash(report)
    with pytest.raises(MigrationError, match='loaded spatial-loss helper'):
        validate_execution_benchmark(report, new['code_revision'], new['execution_profile'])


def test_causal_layout_rejects_a_modified_previously_accepted_reference(tmp_path, monkeypatch):
    import training.sft.runtime_migration as module
    _, new = identities()
    new['execution_profile'] = SFTExecutionProfile(gradient_checkpointing=False, loss_implementation='causal_contiguous_gradient_ce').to_dict()
    report = causal_gradient_fixture(new)
    for path in report['plan']['source_file_hashes']:
        destination = tmp_path/path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT/path).read_bytes())
    path = 'src/training/sft/scheduled_spatial_loss.py'
    with (tmp_path/path).open('a') as stream:
        stream.write('\n# Fixture changes previously accepted reference bytes.\n')
    report['plan']['source_file_hashes'][path] = file_hash(tmp_path/path)
    rehash(report)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    with pytest.raises(MigrationError, match='unchanged accepted scheduled reference'):
        validate_execution_benchmark(report, new['code_revision'], new['execution_profile'])


def test_legacy_v2_profiles_without_loss_field_keep_spatial_default():
    _, new = identities(); report = measured_fixture(new)
    for settings in report["plan"]["profiles"].values():
        settings.pop("loss_implementation")
    for row in [*report["profiles"].values(), *report["profile_runs"].values()]:
        row["settings"].pop("loss_implementation", None)
        row["runtime"]["execution_profile"].pop("loss_implementation", None)
    rehash(report)
    result = validate_execution_benchmark(report, new["code_revision"], new["execution_profile"])
    assert result["reference_profile"] == "baseline"
    assert result["execution_profile"]["loss_implementation"] == "spatial_ce"


@pytest.mark.parametrize("location", ["aggregate", "repetition", "actual_run", "reference_runtime"])
def test_reference_cannot_be_relabelled_as_the_slower_original_baseline(location):
    _, new = identities()
    new["execution_profile"] = SFTExecutionProfile(gradient_checkpointing=False, loss_implementation="flat_ce").to_dict()
    report = incremental_fixture(new)
    if location == "aggregate":
        report["comparisons"]["flat_ce"]["reference_profile"] = "baseline"
    elif location == "repetition":
        report["comparisons"]["flat_ce"]["repetitions"][1]["reference_profile"] = "baseline"
    elif location == "actual_run":
        report["profile_runs"]["baseline_2"]["profile"] = "baseline"
    else:
        report["profile_runs"]["baseline_2"]["runtime"]["execution_profile"] = SFTExecutionProfile().to_dict()
    rehash(report)
    with pytest.raises(MigrationError, match="reference|paired repetition|Paired measurements"):
        validate_execution_benchmark(report, new["code_revision"], new["execution_profile"])


@pytest.mark.parametrize("field", ["runtime", "actual_execution_policy", "longest_sequence_probe"])
def test_new_loss_mode_requires_actual_execution_and_native_probe_evidence(field):
    _, new = identities()
    new["execution_profile"] = SFTExecutionProfile(gradient_checkpointing=False, loss_implementation="flat_ce").to_dict()
    report = incremental_fixture(new)
    report["profile_runs"]["cuda_cached_2"][field].pop("loss_implementation")
    rehash(report)
    with pytest.raises(MigrationError, match="loss execution"):
        validate_execution_benchmark(report, new["code_revision"], new["execution_profile"])


def test_recovered_parent_must_match_the_declared_incremental_reference():
    old, new = identities()
    new["execution_profile"] = SFTExecutionProfile(gradient_checkpointing=False, loss_implementation="flat_ce").to_dict()
    manifest = {"checkpoint_hash": "fixture-checkpoint", "identity": old,
                "progress": {"completed_steps": 400, "sampler_position": 6400}, "files": {}}
    report = incremental_fixture(new, manifest)
    with pytest.raises(MigrationError, match="original checkpoint execution profile"):
        validate_execution_benchmark(report, new["code_revision"], new["execution_profile"])
    reference = SFTExecutionProfile(gradient_checkpointing=False).to_dict()
    report["plan"]["recovery_checkpoint"]["identity"]["execution_profile"] = reference
    for row in [*report["profiles"].values(), *report["profile_runs"].values()]:
        row["recovery_checkpoint"]["identity"]["execution_profile"] = reference
    rehash(report)
    assert validate_execution_benchmark(report, new["code_revision"], new["execution_profile"])["recovery_validated"] is True


@pytest.mark.parametrize("flag", ("numerical_screen_passed", "speedup_threshold_passed", "final_rng_states_identical"))
def test_failed_repetition_cannot_be_hidden_by_passed_aggregate(flag):
    _, new = identities(); report = measured_fixture(new)
    report["comparisons"]["cuda_cached"]["repetitions"][1][flag] = False
    rehash(report)
    with pytest.raises(MigrationError, match="paired repetition"):
        validate_execution_benchmark(report, new["code_revision"], new["execution_profile"])


def test_stale_source_weakened_bounds_and_lying_paired_speed_reject():
    _, new = identities()
    with pytest.raises(MigrationError, match="source binding"):
        validate_execution_benchmark(measured_fixture(new), "9" * 40, new["execution_profile"])
    report = measured_fixture(new); report["plan"]["technical_tolerances"]["minimum_speedup"] = 1.0; rehash(report)
    with pytest.raises(MigrationError, match="weakened"):
        validate_execution_benchmark(report, new["code_revision"], new["execution_profile"])
    report = measured_fixture(new); report["profile_runs"]["cuda_cached_2"]["wall_seconds_per_update"][0] = 11; rehash(report)
    with pytest.raises(MigrationError, match="Paired update speed"):
        validate_execution_benchmark(report, new["code_revision"], new["execution_profile"])


@pytest.mark.parametrize('incremental_loss', [None, 'flat_ce', 'spatial_scheduled_ce', 'causal_contiguous_gradient_ce'])
def test_actual_full_state_migration_preserves_next_optimizer_update_and_parent(tmp_path, incremental_loss):
    torch.manual_seed(42)
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.2), torch.nn.Linear(2, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.99)
    def update(m, o, s):
        o.zero_grad(); m(torch.tensor([[1., 2.]])).square().mean().backward(); o.step(); s.step()
    update(model, optimizer, scheduler)
    old, new = identities()
    if incremental_loss:
        reference_loss = ('spatial_scheduled_ce' if incremental_loss == 'causal_contiguous_gradient_ce'
                          else 'spatial_fused_backward' if incremental_loss == 'spatial_scheduled_ce' else 'spatial_ce')
        old['execution_profile'] = SFTExecutionProfile(gradient_checkpointing=False, loss_implementation=reference_loss).to_dict()
        new['execution_profile'] = SFTExecutionProfile(gradient_checkpointing=False, loss_implementation=incremental_loss).to_dict()
    original = tmp_path / "original" / "checkpoint_000001"
    saved = save_checkpoint(original, old, {"adapter": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "rng": capture_rng(),
        "progress": {"completed_steps": 1, "sampler_position": 16, "example_presentations": 16,
                     "processed_tokens": 112, "total_optimizer_steps": 1250, "exposure_order_hash": "9" * 64},
        "runtime": {"gpu_name": "H100_DECLARED_FIXTURE"}})
    parent_hashes = {path.name: file_hash(path) for path in original.iterdir()}
    report = (causal_gradient_fixture(new, saved) if incremental_loss == 'causal_contiguous_gradient_ce'
              else scheduled_spatial_fixture(new, saved) if incremental_loss == 'spatial_scheduled_ce'
              else incremental_fixture(new, saved) if incremental_loss else measured_fixture(new, saved))
    target = tmp_path / "migrated" / "checkpoint_000001"
    receipt = migrate_checkpoint(original, target, old, new, new["execution_profile"], report,
                                 recovery_report=report)
    inherited = load_checkpoint(target, new)
    unchanged = load_checkpoint(original, old)
    assert all(_same_state(inherited[key], unchanged[key]) for key in ("adapter", "optimizer", "scheduler", "rng"))
    assert inherited["progress"]["sampler_position"] == 16
    assert inherited["progress"]["execution_profile"] == new["execution_profile"]
    assert {path.name: file_hash(path) for path in original.iterdir()} == parent_hashes
    assert receipt["parent_checkpoint_hash"] == saved["checkpoint_hash"]
    assert receipt["exact_state_readback_verified"] is True
    restore_rng(unchanged["rng"]); update(model, optimizer, scheduler)
    resumed = deepcopy(model); resumed.load_state_dict(inherited["adapter"])
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.01)
    resumed_scheduler = torch.optim.lr_scheduler.StepLR(resumed_optimizer, 1, 0.99)
    resumed_optimizer.load_state_dict(inherited["optimizer"]); resumed_scheduler.load_state_dict(inherited["scheduler"])
    restore_rng(inherited["rng"]); update(resumed, resumed_optimizer, resumed_scheduler)
    assert _same_state(model.state_dict(), resumed.state_dict())
    assert _same_state(optimizer.state_dict(), resumed_optimizer.state_dict())
    assert _same_state(scheduler.state_dict(), resumed_scheduler.state_dict())


def test_fresh_state_screen_cannot_migrate_official_recovery(tmp_path):
    old, new = identities()
    target = tmp_path / "must_not_exist"
    with pytest.raises(MigrationError, match="verified parent recovery"):
        migrate_checkpoint(tmp_path / "not_read", target, old, new, new["execution_profile"], measured_fixture(new))
    assert not target.exists()
