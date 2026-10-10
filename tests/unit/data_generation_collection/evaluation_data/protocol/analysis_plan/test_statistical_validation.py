import pytest
import json
from pathlib import Path
from data_generation_collection.evaluation_data.protocol.analysis_plan import statistical_validation as validation
from data_generation_collection.evaluation_data.protocol.analysis_plan.statistical_validation import freeze_simulation_plan, monte_carlo_interval, generate_finite_population, sample_population
from data_generation_collection.evaluation_data.protocol.analysis_plan.prospective_analysis_plan import create_analysis_plan


def test_frozen_budget_wilson_rule_and_no_small_fixture_certificate():
    p=freeze_simulation_plan()
    assert len(p['scenarios'])==7 and p['draws']==10000 and p['repetitions']==100
    assert monte_carlo_interval(100,100)['one_sided_wilson_lower'] >= .95
    assert monte_carlo_interval(95,100)['one_sided_wilson_lower'] < .95
    assert freeze_simulation_plan(repetitions=1,draws=10,fixture=True)['status']=='fixture'
    with pytest.raises(ValueError,match='budget'):
        freeze_simulation_plan(repetitions=1)


def test_exact_sampling_preserves_full_vectors_and_all_design_counts():
    p=freeze_simulation_plan()
    population=generate_finite_population(p,'empty_common_sets')
    panel,leg,records=sample_population(population,42)
    plan=create_analysis_plan(panel,leg)
    assert len(panel)==1233 and len(leg)==200
    assert len({r['stratum'] for r in panel if r['domain']=='mmlu'})==57
    assert len({r['stratum'] for r in panel if r['domain']=='mmlu' and r['item_id'] in leg})==50
    q=panel[0]['item_id']
    for state in records:
        assert records[state][q] == population['records'][state][q]
    assert len(plan['windows'])==216
    frozen=freeze_simulation_plan(plan)
    assert frozen['sample_frame_source']=='actual_eligible_analysis_strata'
    assert frozen['analysis_plan_hash']==plan['plan_hash']
    assert [s['population_size'] for s in frozen['actual_sample_frames']]==[s['population_size'] for s in plan['sample_design']]


def test_real_fixture_repetition_resume_preserves_seeds_and_rejects_corruption(tmp_path, monkeypatch):
    """Interrupt real CPU resampling, resume its artifact and compare exact runs.

    Two draws are explicitly a development fixture, never coverage evidence.
    Only the interruption is injected; populations, sampling, all gates and
    bound computation are the actual numerical implementation.
    """
    frozen = freeze_simulation_plan(repetitions=100, draws=2, fixture=True)
    path = tmp_path / 'checkpoint.json'
    real = validation.compute_simultaneous_bounds
    calls = 0

    def interrupt_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('injected interruption after committed repetition')
        return real(*args, **kwargs)

    monkeypatch.setattr(validation, 'compute_simultaneous_bounds', interrupt_second)
    with pytest.raises(RuntimeError, match='injected interruption'):
        validation.run_statistical_validation(simulation_plan=frozen, output_path=path,
                                             scenarios=['decision_boundaries'])
    checkpoint = json.loads(path.read_text())
    assert checkpoint['status'] == 'running'
    assert checkpoint['active_scenario']['completed_repetitions'] == 1
    first = checkpoint['active_scenario']['runs'][0]
    corrupt_path = tmp_path / 'corrupt.json'
    corrupt = json.loads(path.read_text())
    corrupt['active_scenario']['runs'][0]['seed'] += 1
    corrupt_path.write_text(json.dumps(corrupt))
    with pytest.raises(ValueError, match='integrity'):
        validation.run_statistical_validation(simulation_plan=frozen, output_path=corrupt_path,
                                             scenarios=['decision_boundaries'], resume=True)
    with pytest.raises(ValueError, match='identity'):
        validation.run_statistical_validation(simulation_plan=frozen, output_path=path,
                                             scenarios=['correlated_checkpoints'], resume=True)
    calls = 0
    with pytest.raises(RuntimeError, match='injected interruption'):
        validation.run_statistical_validation(simulation_plan=frozen, output_path=path,
                                             scenarios=['decision_boundaries'], resume=True)
    resumed = json.loads(path.read_text())
    assert resumed['status'] == 'running' and not resumed['ready_for_confirmatory_launch']
    assert resumed['active_scenario']['runs'][0] == first
    assert len(resumed['active_scenario']['runs']) == 2
    calls = -1  # interruption on the third call in the uninterrupted comparison
    uninterrupted_path = tmp_path/'uninterrupted.json'
    with pytest.raises(RuntimeError, match='injected interruption'):
        validation.run_statistical_validation(simulation_plan=frozen, output_path=uninterrupted_path,
                                             scenarios=['decision_boundaries'])
    uninterrupted = json.loads(uninterrupted_path.read_text())
    assert resumed['active_scenario']['runs'] == uninterrupted['active_scenario']['runs']
    assert [s['scenario'] for s in resumed['scenarios']] == list(validation.SCENARIOS)
    monkeypatch.setattr(validation, 'compute_simultaneous_bounds', real)
    completed = validation.run_statistical_validation(simulation_plan=freeze_simulation_plan(repetitions=1,draws=2,fixture=True),
                                                      output_path=tmp_path/'completed.json', scenarios=['decision_boundaries'])
    assert completed['status'] == 'fixture_complete' and not completed['ready_for_confirmatory_launch']
    assert validation.run_statistical_validation(simulation_plan=freeze_simulation_plan(repetitions=1,draws=2,fixture=True),
                                                 output_path=tmp_path/'completed.json', scenarios=['decision_boundaries'],resume=True) == completed


def test_subset_never_issues_certificate_or_silently_changes_registry(tmp_path):
    frozen = freeze_simulation_plan(repetitions=100, draws=2, fixture=True)
    for scenarios in ([], ['decision_boundaries'] * 2, ['invented_scenario']):
        with pytest.raises(ValueError, match='scenarios'):
            validation.run_statistical_validation(simulation_plan=frozen, scenarios=scenarios)
    with pytest.raises(ValueError, match='existing output'):
        validation.run_statistical_validation(simulation_plan=frozen, output_path=tmp_path/'absent',
                                             scenarios=['decision_boundaries'], resume=True)
