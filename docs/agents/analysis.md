# analysis.md

Role identifier: `analysis`. Reports to: `coordinator`.

Implement baseline-repeatability diagnostics, the prospective statistical plan/simulation validation, complete paired-window statistics and honest descriptive/confirmatory classification. All 216 windows and required scalar gates matter; unknown baseline evidence, zero variance and empty subsets are not favorable results.

## Required reading

Read the full controlling PDF as needed for your scope, including H.1/H.9 precedence and authority. Read the common overview, architecture, contracts, decisions, validation and delegation documents. Read every assigned module specification and active Mermaid source, not just its title. Do not apply historical overridden PDF wording as a second rule.

## Assigned modules

| Implementation | Module specification | Source |
|---|---|---|
| `src/data_generation_collection/evaluation_data/protocol/analysis_plan/simultaneous_bounds/simultaneous_bounds.py` | [simultaneous_bounds](../modules/data_generation_collection/evaluation_data/protocol/analysis_plan/simultaneous_bounds/simultaneous_bounds.md) | G.10; H.6,H.8 |
| `src/data_generation_collection/evaluation_data/protocol/analysis_plan/prospective_analysis_plan.py` | [prospective_analysis_plan](../modules/data_generation_collection/evaluation_data/protocol/analysis_plan/prospective_analysis_plan.md) | G; H.6,H.8,H.9 |
| `src/evaluation/results_analysis/construction_criteria/construction_success_criteria.py` | [construction_success_criteria](../modules/evaluation/results_analysis/construction_criteria/construction_success_criteria.md) | G; H.6,H.8 |
| `src/evaluation/results_analysis/results_analysis.py` | [results_analysis](../modules/evaluation/results_analysis/results_analysis.md) | D.1.2,D.2.5; G; H.6,H.8 |
| `src/data_generation_collection/evaluation_data/protocol/analysis_plan/baseline_repeatability.py` | [baseline_repeatability](../modules/data_generation_collection/evaluation_data/protocol/analysis_plan/baseline_repeatability.md) | G.9; H.6.4,H.8.2 |
| `src/data_generation_collection/evaluation_data/protocol/analysis_plan/statistical_validation.py` | [statistical_validation](../modules/data_generation_collection/evaluation_data/protocol/analysis_plan/statistical_validation.md) | G.10; H.8 |

## Exclusive implementation write list

These are the complete initial implementation-file assignments. Code files are currently empty; populate only within your scope. Editing a shared source document, another owner's implementation or the canonical path layout requires coordinator review and synchronized updates.

```text
schemas/qualification_record.json
src/data_generation_collection/evaluation_data/protocol/analysis_plan/baseline_repeatability.py
src/data_generation_collection/evaluation_data/protocol/analysis_plan/prospective_analysis_plan.py
src/data_generation_collection/evaluation_data/protocol/analysis_plan/simultaneous_bounds/simultaneous_bounds.py
src/data_generation_collection/evaluation_data/protocol/analysis_plan/statistical_validation.py
src/evaluation/results_analysis/construction_criteria/construction_success_criteria.py
src/evaluation/results_analysis/results_analysis.py
tests/preflight/test_baseline_repeatability.py
tests/preflight/test_statistical_coverage.py
tests/unit/data_generation_collection/evaluation_data/protocol/analysis_plan/simultaneous_bounds/test_simultaneous_bounds.py
tests/unit/data_generation_collection/evaluation_data/protocol/analysis_plan/test_baseline_repeatability.py
tests/unit/data_generation_collection/evaluation_data/protocol/analysis_plan/test_prospective_analysis_plan.py
tests/unit/data_generation_collection/evaluation_data/protocol/analysis_plan/test_statistical_validation.py
tests/unit/evaluation/results_analysis/construction_criteria/test_construction_success_criteria.py
tests/unit/evaluation/results_analysis/test_results_analysis.py
```

## Required handoff

Return actual changed paths, source-to-requirement coverage, public interfaces and typed sample records, exact tests executed and their evidence, unresolved assumptions, remaining live preflights, and all recorded implementation conventions. A mock result is labeled as such. Confirm that no frozen scientific setting, source question count, intervention/reward criterion or empirical success threshold changed. A known implementation bug can be repaired within the design with versioned dependency invalidation; a mandatory unresolved preflight or external authorization need must be reported, not invented away.

## Stop conditions

Stop affected work requiring invalid frozen input, impossible quotas, unbounded retry, unsupported scientific substitution or unavailable external authorization. Preserve completed valid artifacts and explicit partial/failed/indeterminate states. Ordinary model failure is not a request to resample. Do not claim the experiment is implemented because the directory tree or an empty test file exists.
