# coordinator.md

> Historical implementation handoff from 10 October 2026. Its test counts and then-blocked integrations describe that handoff, not current execution status. See the [current procedure](../procedure_update.md) and [current validation report](../validation_report.md) for completed generation, Drive verification and remaining training gates.

Role identifier: `coordinator`. Owns overall integration.

Own integration and source consistency. Read every guide, workflow, source page and subagent completion report. Do not do scientific redesign to turn a blocker into a passing launch. Coordinate the six responsibility owners and preserve exact filenames, shared interfaces and independent validation evidence.

## Required reading

Read the full controlling PDF as needed for your scope, including H.1/H.9 precedence and authority. Read the common overview, architecture, contracts, decisions, validation and delegation documents. Read every assigned module specification and active Mermaid source, not just its title. Do not apply historical overridden PDF wording as a second rule.

## Assigned modules

| Implementation | Module specification | Source |
|---|---|---|
| `src/experiment_overview/experiment_overview.py` | [experiment_overview](../modules/experiment_overview/experiment_overview.md) | Sections 1–5; E.1–E.2; G.1; H.1,H.9 |
| `src/training/training.py` | [training](../modules/training/training.md) | 2; C.2–C.3; E.1.1; E.2.2; H.7,H.9 |

## Exclusive implementation write list

These are the original implementation assignments. The implemented operational extensions are recorded in the canonical manifest; delegated integration ownership is documented in handoffs. Editing a shared source document, another owner's implementation or the canonical path layout requires coordinator review and synchronized updates.

```text
pyproject.toml
scripts/run_experiment.py
scripts/validate_repository.py
src/data_generation_collection/__init__.py
src/data_generation_collection/evaluation_data/__init__.py
src/data_generation_collection/evaluation_data/capability_panel/__init__.py
src/data_generation_collection/evaluation_data/legibility_subset/__init__.py
src/data_generation_collection/evaluation_data/protocol/__init__.py
src/data_generation_collection/evaluation_data/protocol/analysis_plan/__init__.py
src/data_generation_collection/evaluation_data/protocol/analysis_plan/simultaneous_bounds/__init__.py
src/data_generation_collection/rl_data/__init__.py
src/data_generation_collection/sft_data/__init__.py
src/data_generation_collection/sft_data/clean_trace_collection/__init__.py
src/data_generation_collection/sft_data/clean_trace_collection/candidate_batch/__init__.py
src/data_generation_collection/sft_data/kazitext/__init__.py
src/data_generation_collection/sft_data/rot13/__init__.py
src/evaluation/__init__.py
src/evaluation/condition_groups/__init__.py
src/evaluation/results_analysis/__init__.py
src/evaluation/results_analysis/construction_criteria/__init__.py
src/evaluation/shared_model_evaluation/__init__.py
src/evaluation/shared_model_evaluation/item_evaluation/__init__.py
src/evaluation/shared_model_evaluation/item_evaluation/capability_scoring/__init__.py
src/evaluation/shared_model_evaluation/item_evaluation/legibility_judging/__init__.py
src/evaluation/shared_model_evaluation/item_evaluation/legibility_judging/readable_solution_audit/__init__.py
src/evaluation/shared_model_evaluation/item_evaluation/native_output/__init__.py
src/experiment_overview/__init__.py
src/experiment_overview/experiment_overview.py
src/shared_procedures/__init__.py
src/training/__init__.py
src/training/rl_grpo/__init__.py
src/training/rl_grpo/setup/__init__.py
src/training/rl_grpo/training_cycle/__init__.py
src/training/rl_grpo/training_cycle/policy_update/__init__.py
src/training/rl_grpo/training_cycle/rewards/__init__.py
src/training/rl_grpo/training_cycle/rewards/sonnet_judgment/__init__.py
src/training/rl_grpo/training_cycle/rollouts/__init__.py
src/training/sft/__init__.py
src/training/sft/individual_fit/__init__.py
src/training/training.py
tests/integration/test_analysis_handoff.py
tests/integration/test_data_handoff.py
tests/integration/test_end_to_end.py
tests/integration/test_evaluation_handoff.py
tests/integration/test_recovery_handoff.py
tests/integration/test_reward_transaction.py
tests/integration/test_training_handoff.py
tests/unit/experiment_overview/test_experiment_overview.py
tests/unit/training/test_training.py
```

## Required handoff

Return actual changed paths, source-to-requirement coverage, public interfaces and typed sample records, exact tests executed and their evidence, unresolved assumptions, remaining live preflights, and all recorded implementation conventions. A mock result is labeled as such. Confirm that no frozen scientific setting, source question count, intervention/reward criterion or empirical success threshold changed. A known implementation bug can be repaired within the design with versioned dependency invalidation; a mandatory unresolved preflight or external authorization need must be reported, not invented away.

## Stop conditions

Stop affected work requiring invalid frozen input, impossible quotas, unbounded retry, unsupported scientific substitution or unavailable external authorization. Preserve completed valid artifacts and explicit partial/failed/indeterminate states. Ordinary model failure is not a request to resample. Do not claim the experiment is implemented because the directory tree or an empty test file exists.
