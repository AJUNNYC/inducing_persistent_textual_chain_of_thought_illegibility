# repository_layout.md

This is the complete static repository tree, generated from the canonical path manifest. All displayed files and folders exist in the local scaffold. Code, tests, implementation configuration, prompts and schemas are deliberately blank; documentation, source PDFs, diagrams and structure/provenance metadata are populated. Runtime folders contain markers only. Git internal metadata and files generated during future scientific execution are not part of this static inventory.

Every entry has an exact Mermaid path label; see [repository_manifest.json](../spec/repository_manifest.json) for its owner and diagram/node. Workflow code, diagram and module-specification relative stems match exactly. Dynamic naming templates are separately specified in [architecture.md](architecture.md).

```text
inducing_persistent_textual_chain_of_thought_illegibility/
├── artifacts/
│   ├── evaluation/
│   │   ├── audits/
│   │   │   └── .gitkeep
│   │   ├── capability/
│   │   │   └── .gitkeep
│   │   ├── legibility/
│   │   │   └── .gitkeep
│   │   └── raw/
│   │       └── .gitkeep
│   ├── judge_cache/
│   │   ├── evaluation_audit/
│   │   │   └── .gitkeep
│   │   ├── evaluation_gpt4o/
│   │   │   └── .gitkeep
│   │   ├── evaluation_sonnet/
│   │   │   └── .gitkeep
│   │   ├── rl_pass_a/
│   │   │   └── .gitkeep
│   │   └── rl_pass_b/
│   │       └── .gitkeep
│   ├── manifests/
│   │   └── .gitkeep
│   ├── preflight/
│   │   └── .gitkeep
│   ├── reports/
│   │   └── .gitkeep
│   ├── state/
│   │   └── .gitkeep
│   └── training/
│       ├── rl/
│       │   └── .gitkeep
│       └── sft/
│           ├── kazitext/
│           │   └── .gitkeep
│           └── rot13/
│               └── .gitkeep
├── data/
│   ├── candidates/
│   │   └── .gitkeep
│   ├── clean_traces/
│   │   └── .gitkeep
│   ├── evaluation/
│   │   └── .gitkeep
│   ├── manifests/
│   │   └── .gitkeep
│   ├── rejected_responses/
│   │   └── .gitkeep
│   ├── rl/
│   │   └── .gitkeep
│   └── transformed/
│       ├── kazitext/
│       │   └── .gitkeep
│       └── rot13/
│           └── .gitkeep
├── docs/
│   ├── agents/
│   │   ├── analysis.md
│   │   ├── coordinator.md
│   │   ├── data_preparation.md
│   │   ├── evaluation.md
│   │   ├── rl.md
│   │   ├── sft.md
│   │   └── shared_infrastructure.md
│   ├── modules/
│   │   ├── data_generation_collection/
│   │   │   ├── evaluation_data/
│   │   │   │   ├── capability_panel/
│   │   │   │   │   └── capability_panel_construction.md
│   │   │   │   ├── legibility_subset/
│   │   │   │   │   └── legibility_subset_construction.md
│   │   │   │   ├── protocol/
│   │   │   │   │   ├── analysis_plan/
│   │   │   │   │   │   ├── simultaneous_bounds/
│   │   │   │   │   │   │   └── simultaneous_bounds.md
│   │   │   │   │   │   ├── baseline_repeatability.md
│   │   │   │   │   │   ├── prospective_analysis_plan.md
│   │   │   │   │   │   └── statistical_validation.md
│   │   │   │   │   └── evaluation_protocol.md
│   │   │   │   └── evaluation_data_preparation.md
│   │   │   ├── rl_data/
│   │   │   │   └── rl_data_preparation.md
│   │   │   ├── sft_data/
│   │   │   │   ├── clean_trace_collection/
│   │   │   │   │   ├── candidate_batch/
│   │   │   │   │   │   └── clean_trace_batch.md
│   │   │   │   │   ├── candidate_manifest.md
│   │   │   │   │   └── clean_trace_collection.md
│   │   │   │   ├── kazitext/
│   │   │   │   │   └── kazitext_data_construction.md
│   │   │   │   ├── rot13/
│   │   │   │   │   └── rot13_data_construction.md
│   │   │   │   └── sft_data_preparation.md
│   │   │   └── data_generation.md
│   │   ├── evaluation/
│   │   │   ├── condition_groups/
│   │   │   │   └── condition_evaluation.md
│   │   │   ├── results_analysis/
│   │   │   │   ├── construction_criteria/
│   │   │   │   │   └── construction_success_criteria.md
│   │   │   │   └── results_analysis.md
│   │   │   ├── shared_model_evaluation/
│   │   │   │   ├── item_evaluation/
│   │   │   │   │   ├── capability_scoring/
│   │   │   │   │   │   ├── capability_scoring.md
│   │   │   │   │   │   └── mbpp_sandbox.md
│   │   │   │   │   ├── legibility_judging/
│   │   │   │   │   │   ├── readable_solution_audit/
│   │   │   │   │   │   │   └── readable_solution_audit.md
│   │   │   │   │   │   └── legibility_judging.md
│   │   │   │   │   ├── native_output/
│   │   │   │   │   │   └── output_generation.md
│   │   │   │   │   └── item_evaluation.md
│   │   │   │   └── model_evaluation.md
│   │   │   └── evaluation.md
│   │   ├── experiment_overview/
│   │   │   └── experiment_overview.md
│   │   ├── shared_procedures/
│   │   │   ├── checkpointing.md
│   │   │   ├── configuration.md
│   │   │   ├── cost_accounting.md
│   │   │   ├── execution_state.md
│   │   │   ├── identity.md
│   │   │   ├── judge_client.md
│   │   │   ├── preflight.md
│   │   │   ├── records.md
│   │   │   ├── recovery_procedure.md
│   │   │   ├── response_parsing.md
│   │   │   └── storage.md
│   │   └── training/
│   │       ├── rl_grpo/
│   │       │   ├── setup/
│   │       │   │   └── rl_setup.md
│   │       │   ├── training_cycle/
│   │       │   │   ├── policy_update/
│   │       │   │   │   └── grpo_policy_update.md
│   │       │   │   ├── rewards/
│   │       │   │   │   ├── sonnet_judgment/
│   │       │   │   │   │   └── sonnet_opacity_judgment.md
│   │       │   │   │   └── rl_reward.md
│   │       │   │   ├── rollouts/
│   │       │   │   │   └── rollout_collection.md
│   │       │   │   └── grpo_training_cycle.md
│   │       │   └── rl_training.md
│   │       ├── sft/
│   │       │   ├── individual_fit/
│   │       │   │   └── sft_fit.md
│   │       │   └── sft_training.md
│   │       └── training.md
│   ├── architecture.md
│   ├── contracts.md
│   ├── decisions.md
│   ├── delegation.md
│   ├── implementation_prompt.md
│   ├── overview.md
│   ├── prompts.md
│   ├── protocol_reference.md
│   ├── repository_layout.md
│   ├── requirements.md
│   ├── runbook.md
│   ├── source_map.md
│   ├── validation.md
│   └── validation_report.md
├── prompts/
│   ├── evaluation_arc_easy.txt
│   ├── evaluation_gsm8k.txt
│   ├── evaluation_legibility.txt
│   ├── evaluation_mbpp.txt
│   ├── evaluation_mmlu.txt
│   ├── evaluation_readable_solution.txt
│   ├── rl_generation.txt
│   ├── rl_pass_a.txt
│   └── rl_pass_b.txt
├── schemas/
│   ├── candidate_prompt.json
│   ├── clean_generation.json
│   ├── evaluation_audit_response.json
│   ├── evaluation_record.json
│   ├── judge_record.json
│   ├── optimizer_checkpoint.json
│   ├── qualification_record.json
│   ├── rl_pass_b_response.json
│   ├── score_response.json
│   └── sft_training_record.json
├── scripts/
│   ├── run_experiment.py
│   ├── run_preflight.py
│   └── validate_repository.py
├── spec/
│   ├── archive/
│   │   ├── experiment_original.pdf
│   │   ├── mermaid_readme.txt
│   │   └── mermaid_source.json
│   ├── mermaid/
│   │   ├── repository/
│   │   │   ├── delegation.mmd
│   │   │   ├── docs_layout.mmd
│   │   │   ├── repository_layout.mmd
│   │   │   ├── runtime_layout.mmd
│   │   │   ├── runtime_rules.mmd
│   │   │   ├── spec_layout.mmd
│   │   │   ├── src_data_generation_collection.mmd
│   │   │   ├── src_evaluation.mmd
│   │   │   ├── src_experiment_overview.mmd
│   │   │   ├── src_layout.mmd
│   │   │   ├── src_shared_procedures.mmd
│   │   │   ├── src_training.mmd
│   │   │   ├── support_layout.mmd
│   │   │   ├── tests_layout.mmd
│   │   │   └── unit_tests_layout.mmd
│   │   ├── workflows/
│   │   │   ├── data_generation_collection/
│   │   │   │   ├── evaluation_data/
│   │   │   │   │   ├── capability_panel/
│   │   │   │   │   │   └── capability_panel_construction.mmd
│   │   │   │   │   ├── legibility_subset/
│   │   │   │   │   │   └── legibility_subset_construction.mmd
│   │   │   │   │   ├── protocol/
│   │   │   │   │   │   ├── analysis_plan/
│   │   │   │   │   │   │   ├── simultaneous_bounds/
│   │   │   │   │   │   │   │   └── simultaneous_bounds.mmd
│   │   │   │   │   │   │   ├── baseline_repeatability.mmd
│   │   │   │   │   │   │   ├── prospective_analysis_plan.mmd
│   │   │   │   │   │   │   └── statistical_validation.mmd
│   │   │   │   │   │   └── evaluation_protocol.mmd
│   │   │   │   │   └── evaluation_data_preparation.mmd
│   │   │   │   ├── rl_data/
│   │   │   │   │   └── rl_data_preparation.mmd
│   │   │   │   ├── sft_data/
│   │   │   │   │   ├── clean_trace_collection/
│   │   │   │   │   │   ├── candidate_batch/
│   │   │   │   │   │   │   └── clean_trace_batch.mmd
│   │   │   │   │   │   ├── candidate_manifest.mmd
│   │   │   │   │   │   └── clean_trace_collection.mmd
│   │   │   │   │   ├── kazitext/
│   │   │   │   │   │   └── kazitext_data_construction.mmd
│   │   │   │   │   ├── rot13/
│   │   │   │   │   │   └── rot13_data_construction.mmd
│   │   │   │   │   └── sft_data_preparation.mmd
│   │   │   │   └── data_generation.mmd
│   │   │   ├── evaluation/
│   │   │   │   ├── condition_groups/
│   │   │   │   │   └── condition_evaluation.mmd
│   │   │   │   ├── results_analysis/
│   │   │   │   │   ├── construction_criteria/
│   │   │   │   │   │   └── construction_success_criteria.mmd
│   │   │   │   │   └── results_analysis.mmd
│   │   │   │   ├── shared_model_evaluation/
│   │   │   │   │   ├── item_evaluation/
│   │   │   │   │   │   ├── capability_scoring/
│   │   │   │   │   │   │   ├── capability_scoring.mmd
│   │   │   │   │   │   │   └── mbpp_sandbox.mmd
│   │   │   │   │   │   ├── legibility_judging/
│   │   │   │   │   │   │   ├── readable_solution_audit/
│   │   │   │   │   │   │   │   └── readable_solution_audit.mmd
│   │   │   │   │   │   │   └── legibility_judging.mmd
│   │   │   │   │   │   ├── native_output/
│   │   │   │   │   │   │   └── output_generation.mmd
│   │   │   │   │   │   └── item_evaluation.mmd
│   │   │   │   │   └── model_evaluation.mmd
│   │   │   │   └── evaluation.mmd
│   │   │   ├── experiment_overview/
│   │   │   │   └── experiment_overview.mmd
│   │   │   ├── shared_procedures/
│   │   │   │   ├── checkpointing.mmd
│   │   │   │   ├── configuration.mmd
│   │   │   │   ├── cost_accounting.mmd
│   │   │   │   ├── execution_state.mmd
│   │   │   │   ├── identity.mmd
│   │   │   │   ├── judge_client.mmd
│   │   │   │   ├── preflight.mmd
│   │   │   │   ├── records.mmd
│   │   │   │   ├── recovery_procedure.mmd
│   │   │   │   ├── response_parsing.mmd
│   │   │   │   └── storage.mmd
│   │   │   └── training/
│   │   │       ├── rl_grpo/
│   │   │       │   ├── setup/
│   │   │       │   │   └── rl_setup.mmd
│   │   │       │   ├── training_cycle/
│   │   │       │   │   ├── policy_update/
│   │   │       │   │   │   └── grpo_policy_update.mmd
│   │   │       │   │   ├── rewards/
│   │   │       │   │   │   ├── sonnet_judgment/
│   │   │       │   │   │   │   └── sonnet_opacity_judgment.mmd
│   │   │       │   │   │   └── rl_reward.mmd
│   │   │       │   │   ├── rollouts/
│   │   │       │   │   │   └── rollout_collection.mmd
│   │   │       │   │   └── grpo_training_cycle.mmd
│   │   │       │   └── rl_training.mmd
│   │   │       ├── sft/
│   │   │       │   ├── individual_fit/
│   │   │       │   │   └── sft_fit.mmd
│   │   │       │   └── sft_training.mmd
│   │   │       └── training.mmd
│   │   └── readme.md
│   ├── experiment.pdf
│   ├── module_manifest.json
│   ├── repository_manifest.json
│   ├── requirements_manifest.json
│   └── source_provenance.json
├── src/
│   ├── data_generation_collection/
│   │   ├── evaluation_data/
│   │   │   ├── capability_panel/
│   │   │   │   ├── __init__.py
│   │   │   │   └── capability_panel_construction.py
│   │   │   ├── legibility_subset/
│   │   │   │   ├── __init__.py
│   │   │   │   └── legibility_subset_construction.py
│   │   │   ├── protocol/
│   │   │   │   ├── analysis_plan/
│   │   │   │   │   ├── simultaneous_bounds/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   └── simultaneous_bounds.py
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── baseline_repeatability.py
│   │   │   │   │   ├── prospective_analysis_plan.py
│   │   │   │   │   └── statistical_validation.py
│   │   │   │   ├── __init__.py
│   │   │   │   └── evaluation_protocol.py
│   │   │   ├── __init__.py
│   │   │   └── evaluation_data_preparation.py
│   │   ├── rl_data/
│   │   │   ├── __init__.py
│   │   │   └── rl_data_preparation.py
│   │   ├── sft_data/
│   │   │   ├── clean_trace_collection/
│   │   │   │   ├── candidate_batch/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── clean_trace_batch.py
│   │   │   │   ├── __init__.py
│   │   │   │   ├── candidate_manifest.py
│   │   │   │   └── clean_trace_collection.py
│   │   │   ├── kazitext/
│   │   │   │   ├── __init__.py
│   │   │   │   └── kazitext_data_construction.py
│   │   │   ├── rot13/
│   │   │   │   ├── __init__.py
│   │   │   │   └── rot13_data_construction.py
│   │   │   ├── __init__.py
│   │   │   └── sft_data_preparation.py
│   │   ├── __init__.py
│   │   └── data_generation.py
│   ├── evaluation/
│   │   ├── condition_groups/
│   │   │   ├── __init__.py
│   │   │   └── condition_evaluation.py
│   │   ├── results_analysis/
│   │   │   ├── construction_criteria/
│   │   │   │   ├── __init__.py
│   │   │   │   └── construction_success_criteria.py
│   │   │   ├── __init__.py
│   │   │   └── results_analysis.py
│   │   ├── shared_model_evaluation/
│   │   │   ├── item_evaluation/
│   │   │   │   ├── capability_scoring/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── capability_scoring.py
│   │   │   │   │   └── mbpp_sandbox.py
│   │   │   │   ├── legibility_judging/
│   │   │   │   │   ├── readable_solution_audit/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   └── readable_solution_audit.py
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── legibility_judging.py
│   │   │   │   ├── native_output/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── output_generation.py
│   │   │   │   ├── __init__.py
│   │   │   │   └── item_evaluation.py
│   │   │   ├── __init__.py
│   │   │   └── model_evaluation.py
│   │   ├── __init__.py
│   │   └── evaluation.py
│   ├── experiment_overview/
│   │   ├── __init__.py
│   │   └── experiment_overview.py
│   ├── shared_procedures/
│   │   ├── __init__.py
│   │   ├── checkpointing.py
│   │   ├── configuration.py
│   │   ├── cost_accounting.py
│   │   ├── execution_state.py
│   │   ├── identity.py
│   │   ├── judge_client.py
│   │   ├── preflight.py
│   │   ├── records.py
│   │   ├── recovery_procedure.py
│   │   ├── response_parsing.py
│   │   └── storage.py
│   └── training/
│       ├── rl_grpo/
│       │   ├── setup/
│       │   │   ├── __init__.py
│       │   │   └── rl_setup.py
│       │   ├── training_cycle/
│       │   │   ├── policy_update/
│       │   │   │   ├── __init__.py
│       │   │   │   └── grpo_policy_update.py
│       │   │   ├── rewards/
│       │   │   │   ├── sonnet_judgment/
│       │   │   │   │   ├── __init__.py
│       │   │   │   │   └── sonnet_opacity_judgment.py
│       │   │   │   ├── __init__.py
│       │   │   │   └── rl_reward.py
│       │   │   ├── rollouts/
│       │   │   │   ├── __init__.py
│       │   │   │   └── rollout_collection.py
│       │   │   ├── __init__.py
│       │   │   └── grpo_training_cycle.py
│       │   ├── __init__.py
│       │   └── rl_training.py
│       ├── sft/
│       │   ├── individual_fit/
│       │   │   ├── __init__.py
│       │   │   └── sft_fit.py
│       │   ├── __init__.py
│       │   └── sft_training.py
│       ├── __init__.py
│       └── training.py
├── tests/
│   ├── integration/
│   │   ├── test_analysis_handoff.py
│   │   ├── test_data_handoff.py
│   │   ├── test_end_to_end.py
│   │   ├── test_evaluation_handoff.py
│   │   ├── test_recovery_handoff.py
│   │   ├── test_reward_transaction.py
│   │   └── test_training_handoff.py
│   ├── preflight/
│   │   ├── test_baseline_repeatability.py
│   │   ├── test_clean_data.py
│   │   ├── test_configuration.py
│   │   ├── test_durable_storage.py
│   │   ├── test_full_checkpoint_resume.py
│   │   ├── test_grpo_update.py
│   │   ├── test_judge_services.py
│   │   ├── test_loss_mask_gradients.py
│   │   ├── test_mbpp_sandbox.py
│   │   ├── test_model_runtime.py
│   │   ├── test_spending_authorization.py
│   │   └── test_statistical_coverage.py
│   └── unit/
│       ├── data_generation_collection/
│       │   ├── evaluation_data/
│       │   │   ├── capability_panel/
│       │   │   │   └── test_capability_panel_construction.py
│       │   │   ├── legibility_subset/
│       │   │   │   └── test_legibility_subset_construction.py
│       │   │   ├── protocol/
│       │   │   │   ├── analysis_plan/
│       │   │   │   │   ├── simultaneous_bounds/
│       │   │   │   │   │   └── test_simultaneous_bounds.py
│       │   │   │   │   ├── test_baseline_repeatability.py
│       │   │   │   │   ├── test_prospective_analysis_plan.py
│       │   │   │   │   └── test_statistical_validation.py
│       │   │   │   └── test_evaluation_protocol.py
│       │   │   └── test_evaluation_data_preparation.py
│       │   ├── rl_data/
│       │   │   └── test_rl_data_preparation.py
│       │   ├── sft_data/
│       │   │   ├── clean_trace_collection/
│       │   │   │   ├── candidate_batch/
│       │   │   │   │   └── test_clean_trace_batch.py
│       │   │   │   ├── test_candidate_manifest.py
│       │   │   │   └── test_clean_trace_collection.py
│       │   │   ├── kazitext/
│       │   │   │   └── test_kazitext_data_construction.py
│       │   │   ├── rot13/
│       │   │   │   └── test_rot13_data_construction.py
│       │   │   └── test_sft_data_preparation.py
│       │   └── test_data_generation.py
│       ├── evaluation/
│       │   ├── condition_groups/
│       │   │   └── test_condition_evaluation.py
│       │   ├── results_analysis/
│       │   │   ├── construction_criteria/
│       │   │   │   └── test_construction_success_criteria.py
│       │   │   └── test_results_analysis.py
│       │   ├── shared_model_evaluation/
│       │   │   ├── item_evaluation/
│       │   │   │   ├── capability_scoring/
│       │   │   │   │   ├── test_capability_scoring.py
│       │   │   │   │   └── test_mbpp_sandbox.py
│       │   │   │   ├── legibility_judging/
│       │   │   │   │   ├── readable_solution_audit/
│       │   │   │   │   │   └── test_readable_solution_audit.py
│       │   │   │   │   └── test_legibility_judging.py
│       │   │   │   ├── native_output/
│       │   │   │   │   └── test_output_generation.py
│       │   │   │   └── test_item_evaluation.py
│       │   │   └── test_model_evaluation.py
│       │   └── test_evaluation.py
│       ├── experiment_overview/
│       │   └── test_experiment_overview.py
│       ├── shared_procedures/
│       │   ├── test_checkpointing.py
│       │   ├── test_configuration.py
│       │   ├── test_cost_accounting.py
│       │   ├── test_execution_state.py
│       │   ├── test_identity.py
│       │   ├── test_judge_client.py
│       │   ├── test_preflight.py
│       │   ├── test_records.py
│       │   ├── test_recovery_procedure.py
│       │   ├── test_response_parsing.py
│       │   └── test_storage.py
│       └── training/
│           ├── rl_grpo/
│           │   ├── setup/
│           │   │   └── test_rl_setup.py
│           │   ├── training_cycle/
│           │   │   ├── policy_update/
│           │   │   │   └── test_grpo_policy_update.py
│           │   │   ├── rewards/
│           │   │   │   ├── sonnet_judgment/
│           │   │   │   │   └── test_sonnet_opacity_judgment.py
│           │   │   │   └── test_rl_reward.py
│           │   │   ├── rollouts/
│           │   │   │   └── test_rollout_collection.py
│           │   │   └── test_grpo_training_cycle.py
│           │   └── test_rl_training.py
│           ├── sft/
│           │   ├── individual_fit/
│           │   │   └── test_sft_fit.py
│           │   └── test_sft_training.py
│           └── test_training.py
├── .gitignore
├── config.yaml
├── pyproject.toml
├── readme.md
└── requirements.lock
```
