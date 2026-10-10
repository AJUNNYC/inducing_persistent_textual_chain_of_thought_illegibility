# capability_scoring.md

**Owner:** `evaluation`. **Implementation status:** blank placeholder, not implemented or tested.

| Artifact | Exact repository path |
|---|---|
| Implementation | `src/evaluation/shared_model_evaluation/item_evaluation/capability_scoring/capability_scoring.py` |
| Workflow/support diagram | [`spec/mermaid/workflows/evaluation/shared_model_evaluation/item_evaluation/capability_scoring/capability_scoring.mmd`](../../../../../../spec/mermaid/workflows/evaluation/shared_model_evaluation/item_evaluation/capability_scoring/capability_scoring.mmd) |
| This specification | `docs/modules/evaluation/shared_model_evaluation/item_evaluation/capability_scoring/capability_scoring.md` |
| Unit test slot | `tests/unit/evaluation/shared_model_evaluation/item_evaluation/capability_scoring/test_capability_scoring.py` |

**Scientific sources:** D.1.2; E.2.3,E.2.7; H.4. [p. 5](../../../../../../spec/experiment.pdf#page=5) · [p. 40](../../../../../../spec/experiment.pdf#page=40) · [p. 41](../../../../../../spec/experiment.pdf#page=41) · [p. 55](../../../../../../spec/experiment.pdf#page=55) · [p. 57](../../../../../../spec/experiment.pdf#page=57) · [p. 72](../../../../../../spec/experiment.pdf#page=72).

## Responsibility and controlling details

Use the domain's frozen answer extraction/scoring rule and exact cache dependency identity: raw output hash, parser/scorer version and reference/test version. GSM8K uses normalized numeric exact match; MMLU/ARC use choice-label exact match. Missing or malformed final answers follow the source's unsuccessful-outcome rule rather than being inferred from reasoning.

MBPP extracts the designated final code field, removes at most one surrounding Python fence and otherwise executes exact content. No indentation fixes, renaming, missing-code insertion or final-answer fallback from reasoning. Execute against original withheld tests in the isolated sandbox with one CPU, one GiB, five CPU seconds and ten wall seconds. Syntax/test/task-defined resource failures are unsuccessful model outcomes. A broken executor is infrastructure failure and cannot be silently marked incorrect. Code correctness is execution-based pass@1, not LLM judging or a safety guarantee.

## Interfaces and dependent procedures

Use the common [artifact contracts](../../../../../contracts.md), [frozen prompt instructions](../../../../../prompts.md), [decision ledger](../../../../../decisions.md) and [validation requirements](../../../../../validation.md). Typed signatures/serialization are delegated implementation details; their scientific meaning is not. The named owner defines this module with its consumers before dependent integration.

No separate child procedure is declared in this diagram. This does not exempt shared recovery, storage, provenance or validation contracts.

## Original procedural constraints

The following source comments retain the procedural detail of the supplied diagram. Obsolete introductory editing instructions and raw link directives are omitted from this list; all links are resolved in the path-synchronized diagram below. Appendix H controls any earlier conflicting language.
- Capability scoring
- Missing / malformed answers follow the frozen scoring policy; preserve these outcomes.
- MBPP wrong code, test failures or task-defined timeouts are outcomes; broken sandbox is infrastructure failure.
- Score dependency identity includes raw hash, parser/scorer and reference/test versions.
- Known infrastructure failures use bounded recovery; exhausted failures return incomplete, not incorrect.
- MBPP uses task plus the frozen public interface, with scoring tests and reference implementations withheld from generation.
- Extract the designated final code field; optionally remove one surrounding Python code fence only.
- Execute exactly that code. Never repair indentation, rename functions or insert missing code.
- Syntax errors, failing tests and task-defined timeouts are unsuccessful model outcomes; broken executor is infrastructure failure.
- No test feedback or repair loop. Report this project-specific withheld-test MBPP variant explicitly.

## Completion and failure boundary

Implement and test `src/evaluation/shared_model_evaluation/item_evaluation/capability_scoring/capability_scoring.py` against its specified behavior in `tests/unit/evaluation/shared_model_evaluation/item_evaluation/capability_scoring/test_capability_scoring.py`. Validate every emitted artifact and preserve its source/configuration identity. Do not claim completeness from a placeholder or passing import. Propagate required child failures and maintain ordinary model outcomes separately from infrastructure faults. Mandatory real preflights and permission gates remain in force. Unspecified allowed conventions must be documented/frozen before outcomes; new scientific choices are not silently authorized.

## Exact path-synchronized diagram

The following diagram is byte-for-byte the active `.mmd` content at generation time. Its procedural content is from the supplied workflow or the cited PDF support requirement; file labels and role assignment are the requested repository organization.

```mermaid
---
title: "spec/mermaid/workflows/evaluation/shared_model_evaluation/item_evaluation/capability_scoring/capability_scoring.mmd"
---
flowchart TD
%% Capability scoring
%% Edit quoted node labels directly; this is native Mermaid source.
%% Contract: ../../../../shared_procedures/recovery_procedure.mmd applies to execution failures.
%% Missing / malformed answers follow the frozen scoring policy; preserve these outcomes.
%% MBPP wrong code, test failures or task-defined timeouts are outcomes; broken sandbox is infrastructure failure.
%% Score dependency identity includes raw hash, parser/scorer and reference/test versions.
%% Known infrastructure failures use bounded recovery; exhausted failures return incomplete, not incorrect.
%% MBPP uses task plus the frozen public interface, with scoring tests and reference implementations withheld from generation.
%% Extract the designated final code field; optionally remove one surrounding Python code fence only.
%% Execute exactly that code. Never repair indentation, rename functions or insert missing code.
%% Syntax errors, failing tests and task-defined timeouts are unsuccessful model outcomes; broken executor is infrastructure failure.
%% No test feedback or repair loop. Report this project-specific withheld-test MBPP variant explicitly.
    INPUT["Parsed answer and frozen scoring specification"]
    CACHE["Check exact score dependency identity"]
    HIT{"Valid score cached?"}
    DOMAIN{"Which task type?"}
    NUM["GSM8K<br/>Numeric exact match"]
    CHOICE["MMLU / ARC<br/>Fixed parsing of choice labels"]
    PARSECODE["Extract final code; remove at most one Python fence; make no repairs"]
    CODE["MBPP<br/>Isolated tests with no network or host access<br/>1 CPU and 1 GiB memory<br/>5 CPU seconds and 10 wall seconds"]
    SAVE["Save score and output status"]
    OUT["Return verified capability outcome"]
    INPUT --> CACHE --> HIT
    HIT -->|Yes| OUT
    HIT -->|No| DOMAIN
    DOMAIN -->|Math| NUM --> SAVE
    DOMAIN -->|Multiple choice| CHOICE --> SAVE
    DOMAIN -->|Code| PARSECODE --> CODE --> SAVE
    SAVE --> OUT

%% Repository module: src/evaluation/shared_model_evaluation/item_evaluation/capability_scoring/capability_scoring.py
%% Module documentation: docs/modules/evaluation/shared_model_evaluation/item_evaluation/capability_scoring/capability_scoring.md
%% Unit test: tests/unit/evaluation/shared_model_evaluation/item_evaluation/capability_scoring/test_capability_scoring.py
%% Owner: evaluation
%% Source sections: D.1.2; E.2.3,E.2.7; H.4
```
