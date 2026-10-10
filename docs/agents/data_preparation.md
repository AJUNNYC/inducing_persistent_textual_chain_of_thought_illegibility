# data_preparation.md

Role identifier: `data_preparation`. Reports to: `coordinator`.

Own authorized source manifests, structural paired clean data, exact transforms and frozen evaluation/RL question selections. Preserve common IDs, model-specific traces, all-condition validity and no semantic-quality filter. Candidate filtering and short-mask cases must follow the sources, not a desired result.

## Required reading

Read the full controlling PDF as needed for your scope, including H.1/H.9 precedence and authority. Read the common overview, architecture, contracts, decisions, validation and delegation documents. Read every assigned module specification and active Mermaid source, not just its title. Do not apply historical overridden PDF wording as a second rule.

## Assigned modules

| Implementation | Module specification | Source |
|---|---|---|
| `src/data_generation_collection/sft_data/clean_trace_collection/candidate_batch/clean_trace_batch.py` | [clean_trace_batch](../modules/data_generation_collection/sft_data/clean_trace_collection/candidate_batch/clean_trace_batch.md) | 3.1; C.1.3; C.2.3; E.2.4; H.2.3,H.3 |
| `src/data_generation_collection/sft_data/clean_trace_collection/clean_trace_collection.py` | [clean_trace_collection](../modules/data_generation_collection/sft_data/clean_trace_collection/clean_trace_collection.md) | 3.1; B.1; E.2.2–E.2.4; H.2.3,H.3 |
| `src/data_generation_collection/sft_data/rot13/rot13_data_construction.py` | [rot13_data_construction](../modules/data_generation_collection/sft_data/rot13/rot13_data_construction.md) | 3.2; C.1.1–C.1.3; C.2.3; E.2.4; H.3 |
| `src/data_generation_collection/sft_data/kazitext/kazitext_data_construction.py` | [kazitext_data_construction](../modules/data_generation_collection/sft_data/kazitext/kazitext_data_construction.md) | C.1.4; C.2.3; E.2.4; H.2–H.3 |
| `src/data_generation_collection/sft_data/sft_data_preparation.py` | [sft_data_preparation](../modules/data_generation_collection/sft_data/sft_data_preparation.md) | 3.1; B.1; C.1–C.2.3; E.2.2–E.2.4; H.2–H.3 |
| `src/data_generation_collection/rl_data/rl_data_preparation.py` | [rl_data_preparation](../modules/data_generation_collection/rl_data/rl_data_preparation.md) | C.3.2,C.3.5; E.2.2; H.7 |
| `src/data_generation_collection/evaluation_data/capability_panel/capability_panel_construction.py` | [capability_panel_construction](../modules/data_generation_collection/evaluation_data/capability_panel/capability_panel_construction.md) | D.1.2; H.4 |
| `src/data_generation_collection/evaluation_data/legibility_subset/legibility_subset_construction.py` | [legibility_subset_construction](../modules/data_generation_collection/evaluation_data/legibility_subset/legibility_subset_construction.md) | D.2.3; H.8.2 |
| `src/data_generation_collection/evaluation_data/evaluation_data_preparation.py` | [evaluation_data_preparation](../modules/data_generation_collection/evaluation_data/evaluation_data_preparation.md) | D.1.2; D.2.3; E.2.2; H.4,H.8.2 |
| `src/data_generation_collection/data_generation.py` | [data_generation](../modules/data_generation_collection/data_generation.md) | 3.1; B.1; C.1; C.2.3; E.2.2–E.2.4; H.2–H.3 |
| `src/data_generation_collection/sft_data/clean_trace_collection/candidate_manifest.py` | [candidate_manifest](../modules/data_generation_collection/sft_data/clean_trace_collection/candidate_manifest.md) | 3.1; B.1; E.2.2–E.2.4; H.3 |

## Exclusive implementation write list

These are the complete initial implementation-file assignments. Code files are currently empty; populate only within your scope. Editing a shared source document, another owner's implementation or the canonical path layout requires coordinator review and synchronized updates.

```text
src/data_generation_collection/data_generation.py
src/data_generation_collection/evaluation_data/capability_panel/capability_panel_construction.py
src/data_generation_collection/evaluation_data/evaluation_data_preparation.py
src/data_generation_collection/evaluation_data/legibility_subset/legibility_subset_construction.py
src/data_generation_collection/rl_data/rl_data_preparation.py
src/data_generation_collection/sft_data/clean_trace_collection/candidate_batch/clean_trace_batch.py
src/data_generation_collection/sft_data/clean_trace_collection/candidate_manifest.py
src/data_generation_collection/sft_data/clean_trace_collection/clean_trace_collection.py
src/data_generation_collection/sft_data/kazitext/kazitext_data_construction.py
src/data_generation_collection/sft_data/rot13/rot13_data_construction.py
src/data_generation_collection/sft_data/sft_data_preparation.py
tests/unit/data_generation_collection/evaluation_data/capability_panel/test_capability_panel_construction.py
tests/unit/data_generation_collection/evaluation_data/legibility_subset/test_legibility_subset_construction.py
tests/unit/data_generation_collection/evaluation_data/test_evaluation_data_preparation.py
tests/unit/data_generation_collection/rl_data/test_rl_data_preparation.py
tests/unit/data_generation_collection/sft_data/clean_trace_collection/candidate_batch/test_clean_trace_batch.py
tests/unit/data_generation_collection/sft_data/clean_trace_collection/test_candidate_manifest.py
tests/unit/data_generation_collection/sft_data/clean_trace_collection/test_clean_trace_collection.py
tests/unit/data_generation_collection/sft_data/kazitext/test_kazitext_data_construction.py
tests/unit/data_generation_collection/sft_data/rot13/test_rot13_data_construction.py
tests/unit/data_generation_collection/sft_data/test_sft_data_preparation.py
tests/unit/data_generation_collection/test_data_generation.py
```

## Required handoff

Return actual changed paths, source-to-requirement coverage, public interfaces and typed sample records, exact tests executed and their evidence, unresolved assumptions, remaining live preflights, and all recorded implementation conventions. A mock result is labeled as such. Confirm that no frozen scientific setting, source question count, intervention/reward criterion or empirical success threshold changed. A known implementation bug can be repaired within the design with versioned dependency invalidation; a mandatory unresolved preflight or external authorization need must be reported, not invented away.

## Stop conditions

Stop affected work requiring invalid frozen input, impossible quotas, unbounded retry, unsupported scientific substitution or unavailable external authorization. Preserve completed valid artifacts and explicit partial/failed/indeterminate states. Ordinary model failure is not a request to resample. Do not claim the experiment is implemented because the directory tree or an empty test file exists.
