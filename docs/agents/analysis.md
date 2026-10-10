# analysis.md

Role identifier: `analysis`. Reports to: `coordinator`.

Implement baseline-repeatability diagnostics, the prospective statistical plan/simulation validation, complete paired-window statistics and honest descriptive/confirmatory classification. All 216 windows and required scalar gates matter; unknown baseline evidence, zero variance and empty subsets are not favorable results.

## Current implementation and evidence

All six assigned scientific modules and qualification schema are implemented. The focused engineering suite has 20 passing tests and two blocked-live preflight skips, including actual evaluator-to-analysis fixture handoffs. Live untouched-model repeatability and trained-state measurements do not exist. The actual eligible-frame statistical simulation rejected the candidate after 36 boundary repetitions: 34 jointly covered, two failures, zero structural failures; fixed Wilson futility leaves 64 repetitions and six scenarios unrun. Final checkpoint/resume orchestration was frozen at 2026-10-10 22:38:34 UTC and completed revalidation of the unchanged numerical procedure at one NumPy thread in 627.21 seconds: the same two failures and exact futility rejection after 36 repetitions. Descriptive kappa denominator handling was corrected and its three unit tests rerun successfully; all analysis source was frozen at 22:46:45 UTC. Hashes: implementation `95e615ae076ee9672de28013176e0d3e00185af57fa88f21f7f424def00e1c9a`, simulation `fec05bc79d525074718509205ef26783599f4ede69677c1da75dc9fc2f839688`. Resume integrity tests passed using real reduced-draw CPU resampling; only process interruptions were injected. They do not establish scientific coverage. Both rejection and interrupted evidence are retained in `docs/evidence`. Final current report hash is `3e99f06fd472b1fab0f16de38e1451a202dafe9debc1ba5755de1414d5ff2cf8`. There is no active analysis job or passing certificate; training remains blocked.

Public entrypoints are `create_analysis_plan(panel, legibility_ids, config=None)`, `baseline_repeatability(passes_by_model, panel, require_full_panel=True)`, `freeze_simulation_plan(analysis_plan=None, ...)`, `run_statistical_validation(analysis_plan=None, ...)`, `compute_simultaneous_bounds(records_by_state, plan, ...)` and `analyze_results(records_by_state, plan, ...)`. Module cards document exact return keys, frozen numerical conventions and failure boundaries. The actual prospective plan at `data/evaluation/analysis_plan.json` has 216 windows and 10,800 deduplicated scalar gates. Keep actual source eligibility sizes; MMLU strata are 43–100 clean questions, not a fabricated uniform 100.

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

These are the complete initial implementation-file assignments, now populated. Preserve ownership when repairing code. Editing a shared source document, another owner's implementation or the canonical path layout requires coordinator review and synchronized updates. The coordinator additionally delegated `tests/integration/test_analysis_handoff.py` and the status/API updates in the six assigned module cards and this role brief.

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
