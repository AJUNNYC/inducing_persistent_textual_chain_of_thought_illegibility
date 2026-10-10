# evaluation.md

Role identifier: `evaluation`. Reports to: `coordinator`.

Implement the frozen native protocol, shared runners, strict scoring and isolated MBPP execution. Keep raw outputs shared across capability and independent judging. Preserve exact prompt information boundaries, fixed denominators and the separate tri-state audit without changing the RL reward.

## Required reading

Read the full controlling PDF as needed for your scope, including H.1/H.9 precedence and authority. Read the common overview, architecture, contracts, decisions, validation and delegation documents. Read every assigned module specification and active Mermaid source, not just its title. Do not apply historical overridden PDF wording as a second rule.

## Assigned modules

| Implementation | Module specification | Source |
|---|---|---|
| `src/data_generation_collection/evaluation_data/protocol/evaluation_protocol.py` | [evaluation_protocol](../modules/data_generation_collection/evaluation_data/protocol/evaluation_protocol.md) | D.1–D.2; G; H.4–H.6,H.9 |
| `src/evaluation/condition_groups/condition_evaluation.py` | [condition_evaluation](../modules/evaluation/condition_groups/condition_evaluation.md) | C.3.6; D.1–D.2; G; H.5–H.6 |
| `src/evaluation/shared_model_evaluation/item_evaluation/native_output/output_generation.py` | [output_generation](../modules/evaluation/shared_model_evaluation/item_evaluation/native_output/output_generation.md) | D.1.2,D.2.4; E.2.3,E.2.7; H.4,H.9 |
| `src/evaluation/shared_model_evaluation/item_evaluation/capability_scoring/capability_scoring.py` | [capability_scoring](../modules/evaluation/shared_model_evaluation/item_evaluation/capability_scoring/capability_scoring.md) | D.1.2; E.2.3,E.2.7; H.4 |
| `src/evaluation/shared_model_evaluation/item_evaluation/legibility_judging/readable_solution_audit/readable_solution_audit.py` | [readable_solution_audit](../modules/evaluation/shared_model_evaluation/item_evaluation/legibility_judging/readable_solution_audit/readable_solution_audit.md) | H.4–H.6 |
| `src/evaluation/shared_model_evaluation/item_evaluation/legibility_judging/legibility_judging.py` | [legibility_judging](../modules/evaluation/shared_model_evaluation/item_evaluation/legibility_judging/legibility_judging.md) | D.2; H.5–H.6 |
| `src/evaluation/shared_model_evaluation/item_evaluation/item_evaluation.py` | [item_evaluation](../modules/evaluation/shared_model_evaluation/item_evaluation/item_evaluation.md) | D.1–D.2; E.2.3,E.2.7; H.4–H.6 |
| `src/evaluation/shared_model_evaluation/model_evaluation.py` | [model_evaluation](../modules/evaluation/shared_model_evaluation/model_evaluation.md) | 4; D.1–D.2; E.2.7; H.4–H.6 |
| `src/evaluation/evaluation.py` | [evaluation](../modules/evaluation/evaluation.md) | 4; D.1–D.2; E.2.7; G; H.4–H.6,H.9 |
| `src/evaluation/shared_model_evaluation/item_evaluation/capability_scoring/mbpp_sandbox.py` | [mbpp_sandbox](../modules/evaluation/shared_model_evaluation/item_evaluation/capability_scoring/mbpp_sandbox.md) | D.1.2; E.2.7; H.4 |

## Exclusive implementation write list

These are the complete initial implementation-file assignments. Code files are currently empty; populate only within your scope. Editing a shared source document, another owner's implementation or the canonical path layout requires coordinator review and synchronized updates.

```text
prompts/evaluation_arc_easy.txt
prompts/evaluation_gsm8k.txt
prompts/evaluation_legibility.txt
prompts/evaluation_mbpp.txt
prompts/evaluation_mmlu.txt
prompts/evaluation_readable_solution.txt
src/data_generation_collection/evaluation_data/protocol/evaluation_protocol.py
src/evaluation/condition_groups/condition_evaluation.py
src/evaluation/evaluation.py
src/evaluation/shared_model_evaluation/item_evaluation/capability_scoring/capability_scoring.py
src/evaluation/shared_model_evaluation/item_evaluation/capability_scoring/mbpp_sandbox.py
src/evaluation/shared_model_evaluation/item_evaluation/item_evaluation.py
src/evaluation/shared_model_evaluation/item_evaluation/legibility_judging/legibility_judging.py
src/evaluation/shared_model_evaluation/item_evaluation/legibility_judging/readable_solution_audit/readable_solution_audit.py
src/evaluation/shared_model_evaluation/item_evaluation/native_output/output_generation.py
src/evaluation/shared_model_evaluation/model_evaluation.py
tests/unit/data_generation_collection/evaluation_data/protocol/test_evaluation_protocol.py
tests/unit/evaluation/condition_groups/test_condition_evaluation.py
tests/unit/evaluation/shared_model_evaluation/item_evaluation/capability_scoring/test_capability_scoring.py
tests/unit/evaluation/shared_model_evaluation/item_evaluation/capability_scoring/test_mbpp_sandbox.py
tests/unit/evaluation/shared_model_evaluation/item_evaluation/legibility_judging/readable_solution_audit/test_readable_solution_audit.py
tests/unit/evaluation/shared_model_evaluation/item_evaluation/legibility_judging/test_legibility_judging.py
tests/unit/evaluation/shared_model_evaluation/item_evaluation/native_output/test_output_generation.py
tests/unit/evaluation/shared_model_evaluation/item_evaluation/test_item_evaluation.py
tests/unit/evaluation/shared_model_evaluation/test_model_evaluation.py
tests/unit/evaluation/test_evaluation.py
```

## Required handoff

Return actual changed paths, source-to-requirement coverage, public interfaces and typed sample records, exact tests executed and their evidence, unresolved assumptions, remaining live preflights, and all recorded implementation conventions. A mock result is labeled as such. Confirm that no frozen scientific setting, source question count, intervention/reward criterion or empirical success threshold changed. A known implementation bug can be repaired within the design with versioned dependency invalidation; a mandatory unresolved preflight or external authorization need must be reported, not invented away.

## Stop conditions

Stop affected work requiring invalid frozen input, impossible quotas, unbounded retry, unsupported scientific substitution or unavailable external authorization. Preserve completed valid artifacts and explicit partial/failed/indeterminate states. Ordinary model failure is not a request to resample. Do not claim the experiment is implemented because the directory tree or an empty test file exists.
