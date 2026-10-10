# clean_trace_batch.md

**Owner:** `data_preparation`. **Implementation status:** blank placeholder, not implemented or tested.

| Artifact | Exact repository path |
|---|---|
| Implementation | `src/data_generation_collection/sft_data/clean_trace_collection/candidate_batch/clean_trace_batch.py` |
| Workflow/support diagram | [`spec/mermaid/workflows/data_generation_collection/sft_data/clean_trace_collection/candidate_batch/clean_trace_batch.mmd`](../../../../../../spec/mermaid/workflows/data_generation_collection/sft_data/clean_trace_collection/candidate_batch/clean_trace_batch.mmd) |
| This specification | `docs/modules/data_generation_collection/sft_data/clean_trace_collection/candidate_batch/clean_trace_batch.md` |
| Unit test slot | `tests/unit/data_generation_collection/sft_data/clean_trace_collection/candidate_batch/test_clean_trace_batch.py` |

**Scientific sources:** 3.1; C.1.3; C.2.3; E.2.4; H.2.3,H.3. [p. 3](../../../../../../spec/experiment.pdf#page=3) · [p. 25](../../../../../../spec/experiment.pdf#page=25) · [p. 30](../../../../../../spec/experiment.pdf#page=30) · [p. 55](../../../../../../spec/experiment.pdf#page=55) · [p. 70](../../../../../../spec/experiment.pdf#page=70) · [p. 71](../../../../../../spec/experiment.pdf#page=71).

## Responsibility and controlling details

Native clean generation: temperature 0.6, `top_p=0.95`, at most 3072 new tokens, no added system prompt, native reasoning formatting and no opacity instruction. Both untouched models independently respond to the same candidate. Save both raw responses, finish reasons, token counts, model/tokenizer/runtime/seed identity and failures before derived processing.

Each output must be completed and separable into nonempty reasoning and nonempty final response; reasoning has at least three original word spans containing an ASCII Latin letter. Apply the exact final transformation algorithms, profile settings, seeds and protected-span rules to preflight every variant. Clean and all variants must fit the **total prompt-plus-response 4096-token bound** for the relevant model tokenizer. Preserve final answers and prompts; KaziText protects numerical and mathematical spans. A failure for either model or any required variant rejects that shared candidate before freeze.

Do not add a semantic-correctness judge, reasoning-quality score, minimum 50-word rule, minimum five-step rule, or distinct-mask requirement. Short/wrong/uninformative but structurally acceptable traces may remain. Log actual lengths, eligible word counts, encoded fractions and repeated masks. Ordinary cutoff/parse/rejection outcomes are not execution retries.

## Interfaces and dependent procedures

Use the common [artifact contracts](../../../../../contracts.md), [frozen prompt instructions](../../../../../prompts.md), [decision ledger](../../../../../decisions.md) and [validation requirements](../../../../../validation.md). Typed signatures/serialization are delegated implementation details; their scientific meaning is not. The named owner defines this module with its consumers before dependent integration.

No separate child procedure is declared in this diagram. This does not exempt shared recovery, storage, provenance or validation contracts.

## Original procedural constraints

The following source comments retain the procedural detail of the supplied diagram. Obsolete introductory editing instructions and raw link directives are omitted from this list; all links are resolved in the path-synchronized diagram below. Appendix H controls any earlier conflicting language.
- One clean trace candidate batch
- Execution contract: known faults use bounded recovery; unclassified/persistent faults stop the worker.
- Frozen invalid inputs are fatal. Unrecoverable child failures propagate; never release incomplete results.
- One clean-trace candidate batch
- Cache folders are siblings. Raw responses include generation provenance.
- Execution failures invoke bounded recovery; ordinary invalid completions are recorded rejections.
- Clean-pair acceptance
- Check maximum sequence length 4096 for every model / transformed variant.
- Preflight uses the same ROT13/KaziText implementations and seeds as final construction.
- Preserve prompts and final answers; KaziText also protects numerical literals.
- Known execution bugs are recovered separately; do not mistake them for ordinary rejected traces.
- Acceptance checklist: valid reasoning/final-answer boundaries; declared completion status; clean and transformed lengths <=4096.
- Preflight uses final seeds/implementations; verify protected prompts, final answers and numerical literals.
- All acceptance checks are applied to both model traces. Never resample ordinary rejected responses until they pass.
- Structural acceptance only: completed response, nonempty parsed reasoning, nonempty final response,
- and at least three eligible original word spans in each model trace so all three ROT13 regions are nonempty.
- Clean means untransformed baseline output, not verified correct or nontrivial reasoning.
- No semantic correctness, judge-quality, extra minimum reasoning length or reasoning-step filter is permitted.
- Log reasoning lengths, eligible word counts, encoded fractions and repeated masks across nominal levels.
- Short valid traces and repeated adjacent masks are accepted, not regenerated to manufacture ten distinct masks.

## Completion and failure boundary

Implement and test `src/data_generation_collection/sft_data/clean_trace_collection/candidate_batch/clean_trace_batch.py` against its specified behavior in `tests/unit/data_generation_collection/sft_data/clean_trace_collection/candidate_batch/test_clean_trace_batch.py`. Validate every emitted artifact and preserve its source/configuration identity. Do not claim completeness from a placeholder or passing import. Propagate required child failures and maintain ordinary model outcomes separately from infrastructure faults. Mandatory real preflights and permission gates remain in force. Unspecified allowed conventions must be documented/frozen before outcomes; new scientific choices are not silently authorized.

## Exact path-synchronized diagram

The following diagram is byte-for-byte the active `.mmd` content at generation time. Its procedural content is from the supplied workflow or the cited PDF support requirement; file labels and role assignment are the requested repository organization.

```mermaid
---
title: "spec/mermaid/workflows/data_generation_collection/sft_data/clean_trace_collection/candidate_batch/clean_trace_batch.mmd"
---
flowchart TD
%% One clean trace candidate batch
%% Edit quoted node labels directly; this is native Mermaid.
%% Execution contract: known faults use bounded recovery; unclassified/persistent faults stop the worker.
%% Frozen invalid inputs are fatal. Unrecoverable child failures propagate; never release incomplete results.
%% One clean-trace candidate batch
%% Cache folders are siblings. Raw responses include generation provenance.
%% Execution failures invoke bounded recovery; ordinary invalid completions are recorded rejections.
%% Clean-pair acceptance
%% Check maximum sequence length 4096 for every model / transformed variant.
%% Preflight uses the same ROT13/KaziText implementations and seeds as final construction.
%% Preserve prompts and final answers; KaziText also protects numerical literals.
%% Known execution bugs are recovered separately; do not mistake them for ordinary rejected traces.
%% Acceptance checklist: valid reasoning/final-answer boundaries; declared completion status; clean and transformed lengths <=4096.
%% Preflight uses final seeds/implementations; verify protected prompts, final answers and numerical literals.
%% All acceptance checks are applied to both model traces. Never resample ordinary rejected responses until they pass.
%% Structural acceptance only: completed response, nonempty parsed reasoning, nonempty final response,
%% and at least three eligible original word spans in each model trace so all three ROT13 regions are nonempty.
%% Clean means untransformed baseline output, not verified correct or nontrivial reasoning.
%% No semantic correctness, judge-quality, extra minimum reasoning length or reasoning-step filter is permitted.
%% Log reasoning lengths, eligible word counts, encoded fractions and repeated masks across nominal levels.
%% Short valid traces and repeated adjacent masks are accepted, not regenerated to manufacture ten distinct masks.
    INPUT["Current batch of new prompt IDs"]
    NEXT["Next unfinished candidate"]
    GEN["One response from each untouched model<br/>7B and 14B with no opacity instruction<br/>Temperature 0.6 and top_p 0.95<br/>At most 3,072 new tokens"]
    CHECK["Check complete, nonempty reasoning and final response; at least 3 eligible words; every transformed record fits 4096 tokens"]
    VALID{"Both traces meet every acceptance rule?"}
    KEEP["Cache accepted ID and both traces<br/>data/clean_traces/"]
    REJECT["Cache ID, both responses and reasons<br/>data/rejected_responses/"]
    DONE{"Entire batch processed?"}
    OUT["Return complete batch acceptance records"]
    INPUT --> NEXT --> GEN --> CHECK --> VALID
    VALID -->|Yes| KEEP --> DONE
    VALID -->|No| REJECT --> DONE
    DONE -->|No| NEXT
    DONE -->|Yes| OUT

%% Repository module: src/data_generation_collection/sft_data/clean_trace_collection/candidate_batch/clean_trace_batch.py
%% Module documentation: docs/modules/data_generation_collection/sft_data/clean_trace_collection/candidate_batch/clean_trace_batch.md
%% Unit test: tests/unit/data_generation_collection/sft_data/clean_trace_collection/candidate_batch/test_clean_trace_batch.py
%% Owner: data_preparation
%% Source sections: 3.1; C.1.3; C.2.3; E.2.4; H.2.3,H.3
```
