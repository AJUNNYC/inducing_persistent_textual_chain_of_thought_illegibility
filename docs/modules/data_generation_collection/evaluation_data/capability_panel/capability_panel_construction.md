# capability_panel_construction.md

**Owner:** `data_preparation`. **Implementation status:** blank placeholder, not implemented or tested.

| Artifact | Exact repository path |
|---|---|
| Implementation | `src/data_generation_collection/evaluation_data/capability_panel/capability_panel_construction.py` |
| Workflow/support diagram | [`spec/mermaid/workflows/data_generation_collection/evaluation_data/capability_panel/capability_panel_construction.mmd`](../../../../../spec/mermaid/workflows/data_generation_collection/evaluation_data/capability_panel/capability_panel_construction.mmd) |
| This specification | `docs/modules/data_generation_collection/evaluation_data/capability_panel/capability_panel_construction.md` |
| Unit test slot | `tests/unit/data_generation_collection/evaluation_data/capability_panel/test_capability_panel_construction.py` |

**Scientific sources:** D.1.2; H.4. [p. 38](../../../../../spec/experiment.pdf#page=38) · [p. 40](../../../../../spec/experiment.pdf#page=40) · [p. 41](../../../../../spec/experiment.pdf#page=41) · [p. 72](../../../../../spec/experiment.pdf#page=72).

## Responsibility and controlling details

## Fixed selection and scoring manifest

| Domain | Source and selection | Count | Scoring |
|---|---|---:|---|
| GSM8K | Official test population 1319; count whitespace-delimited reference-solution words, sort with stable index tie-break, divide into reference-length terciles and sample 100 without replacement in each | 300 | Normalized numeric exact match |
| MMLU-Redux-2.0 | Source-described clean-labeled eligibility; stable IDs and seven sampled without replacement in each of all 57 subjects | 399 | Exact choice label |
| ARC-Easy | Official test population 2376; stable ID sort then uniform without-replacement fixed sample | 331 | Exact choice label |
| Sanitized MBPP | 427 released hand-verified tasks; stable ID sort then uniform without-replacement fixed sample | 203 | Isolated execution pass@1 |

Use seed 42 and persist exact selected IDs, source revisions, exclusions, references and sampling metadata. The combined count is 1233. Selection is model-independent; never sample based on how any baseline or trained state answers. MMLU's actual post-eligibility populations and GSM stratum boundaries must be recorded for design-aware inference. Do not interpret seven-item subject scores as reliable individual-subject estimates.

Mechanically extract MBPP required callable names, parameters and defaults into a frozen public-interface manifest. Keep the original task description public. Do not expose hidden scoring tests, reference implementation bodies, helper algorithms, assertions, expected outputs or solution comments to generation or audit. Verify source tests and interface extraction before freezing. This is a project-specific withheld-test evaluation, not a directly comparable reproduction of original test-visible MBPP scores.

Validate counts, per-stratum/subject quotas, uniqueness, valid references and exclusions. If quotas are impossible, stop; no replacement benchmark or changed count is authorized. The source's roughly five-percentage-point precision calculations motivate panel sizing but are not a promise about realized or conditional confidence bounds.

## Interfaces and dependent procedures

Use the common [artifact contracts](../../../../contracts.md), [frozen prompt instructions](../../../../prompts.md), [decision ledger](../../../../decisions.md) and [validation requirements](../../../../validation.md). Typed signatures/serialization are delegated implementation details; their scientific meaning is not. The named owner defines this module with its consumers before dependent integration.

| Source node | Called procedure (exact path) | Specification |
|---|---|---|
| `REPAIR` | `src/shared_procedures/recovery_procedure.py` | [recovery_procedure](../../../shared_procedures/recovery_procedure.md) |

## Original procedural constraints

The following source comments retain the procedural detail of the supplied diagram. Obsolete introductory editing instructions and raw link directives are omitted from this list; all links are resolved in the path-synchronized diagram below. Appendix H controls any earlier conflicting language.
- Capability panel construction
- Record pool revisions, source IDs and exclusions; validate MBPP tests before freezing.
- MBPP prompts expose only the task and mechanically extracted public function interface.
- The interface manifest contains required entry point names, parameters and defaults from the pinned source.
- Withhold every scoring test, reference implementation, helper body, algorithm fragment,
- assertion, expected output and solution comment from prompts and judge inputs.
- This is a project-specific withheld-test MBPP variant, not directly comparable to published test-visible scores.

## Completion and failure boundary

Implement and test `src/data_generation_collection/evaluation_data/capability_panel/capability_panel_construction.py` against its specified behavior in `tests/unit/data_generation_collection/evaluation_data/capability_panel/test_capability_panel_construction.py`. Validate every emitted artifact and preserve its source/configuration identity. Do not claim completeness from a placeholder or passing import. Propagate required child failures and maintain ordinary model outcomes separately from infrastructure faults. Mandatory real preflights and permission gates remain in force. Unspecified allowed conventions must be documented/frozen before outcomes; new scientific choices are not silently authorized.

## Exact path-synchronized diagram

The following diagram is byte-for-byte the active `.mmd` content at generation time. Its procedural content is from the supplied workflow or the cited PDF support requirement; file labels and role assignment are the requested repository organization.

```mermaid
---
title: "spec/mermaid/workflows/data_generation_collection/evaluation_data/capability_panel/capability_panel_construction.mmd"
---
flowchart TD
%% Capability panel construction
%% Edit quoted node labels directly; this is native Mermaid source.
%% Contract: ../../../shared_procedures/recovery_procedure.mmd applies to execution failures.
%% Record pool revisions, source IDs and exclusions; validate MBPP tests before freezing.
%% MBPP prompts expose only the task and mechanically extracted public function interface.
%% The interface manifest contains required entry point names, parameters and defaults from the pinned source.
%% Withhold every scoring test, reference implementation, helper body, algorithm fragment,
%% assertion, expected output and solution comment from prompts and judge inputs.
%% This is a project-specific withheld-test MBPP variant, not directly comparable to published test-visible scores.
%% Child: REPAIR => ../../../shared_procedures/recovery_procedure.mmd
    INPUT["Fix eligibility / overlap rules<br/>Sampling seed 42"]
    GSM["GSM8K test: 300<br/>100 per tercile of reference length"]
    MMLU["MMLU Redux 2.0: 399<br/>7 items with clean labels per subject<br/>57 subjects"]
    ARC["ARC Easy test: 331<br/>Uniform fixed sample"]
    MBPP["Sanitized MBPP: 203<br/>From 427 verified tasks"]
    INTERFACE["Freeze required public function interfaces; withhold scoring tests and solutions"]
    JOIN["Wait for all four domain selections"]
    CHECK{"Counts, uniqueness, references<br/>and exclusions valid?"}
    REPAIR["src/shared_procedures/recovery_procedure.py<br/>Shared bounded recovery<br/>Repair construction before freezing"]
    OUT["Return 1,233 verified items"]
    STOP["Stop: impossible quotas or rules"]
    INPUT --> GSM --> JOIN
    INPUT --> MMLU --> JOIN
    INPUT --> ARC --> JOIN
    INPUT --> MBPP --> INTERFACE --> JOIN
    JOIN --> CHECK
    CHECK -->|Yes| OUT
    CHECK -->|No| REPAIR
    REPAIR -->|Same rules and retry permitted| INPUT
    REPAIR -->|Not permitted| STOP
    classDef detail fill:#e5eef9,stroke:#365f8a,stroke-width:2px
    class REPAIR detail
    classDef stop fill:#fbe9e7,stroke:#b45448
    class STOP stop

%% Repository module: src/data_generation_collection/evaluation_data/capability_panel/capability_panel_construction.py
%% Module documentation: docs/modules/data_generation_collection/evaluation_data/capability_panel/capability_panel_construction.md
%% Unit test: tests/unit/data_generation_collection/evaluation_data/capability_panel/test_capability_panel_construction.py
%% Owner: data_preparation
%% Source sections: D.1.2; H.4
```
