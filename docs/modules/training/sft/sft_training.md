# sft_training.md

**Owner:** `sft`. **Implementation status:** full-count queue fixtures pass. The source 05f42b0 continuation reached a Drive-verified safe pause at step 401 for 7B ROT13 level 1 after inheriting step 214 on RTX PRO 6000 Blackwell 96GB, with accepted no-recomputation evidence and Drive-verified child state. Parent bytes remain preserved; the forty-fit study is incomplete.

Source `05f42b0` has accepted no-recomputation evidence and READY affected-runtime controls for 7B ROT13 level 1 on RTX PRO 6000 Blackwell 96GB; the production continuation is now safely paused at step 401 with verified Drive state. CUDA input caching failed the speed gate and remains disabled. Activation evidence is scoped to this model/family/level. Explicit continuation lineage inherits full step-214 state while preserving parent bytes; a passing first-fit screen does not authorize the entire forty-fit matrix.

| Artifact | Exact repository path |
|---|---|
| Implementation | `src/training/sft/sft_training.py` |
| Workflow/support diagram | [`spec/mermaid/workflows/training/sft/sft_training.mmd`](../../../../spec/mermaid/workflows/training/sft/sft_training.mmd) |
| This specification | `docs/modules/training/sft/sft_training.md` |
| Unit test slot | `tests/unit/training/sft/test_sft_training.py` |

**Scientific sources:** C.2; E.1.1–E.1.3; E.2.2,E.2.5. [p. 28](../../../../spec/experiment.pdf#page=28) · [p. 29](../../../../spec/experiment.pdf#page=29) · [p. 30](../../../../spec/experiment.pdf#page=30) · [p. 49](../../../../spec/experiment.pdf#page=49) · [p. 50](../../../../spec/experiment.pdf#page=50) · [p. 54](../../../../spec/experiment.pdf#page=54) · [p. 56](../../../../spec/experiment.pdf#page=56).

## Responsibility and controlling details

Create two independent family queues, one for ROT13 and one for KaziText, each covering both models at ten fixed levels: twenty independent fits per queue and forty total. Both call the same `sft_fit` implementation. Every level initializes from its untouched base; a previous level is not a warm start. Only an exact-fit matching full checkpoint may resume a fit.

Each queue runs its fits sequentially on its assigned GPU. The launcher can serialize selected levels on one GPU. The current source 05f42b0 acceptance selects only 7B ROT13 level 1; remaining levels require their condition-specific affected evidence before activation. The full forty-fit scientific matrix remains unchanged. Concurrent queues require available GPUs. A failed fit stops the affected queue and preserves partial results; another independent queue may complete valid work. Mark a fit complete only after its final adapter, fixed validation loss and provenance have verified. Release each GPU when its own queue is complete. The nominal exposure is 20,000 example presentations per fit and approximately 1250 optimizer updates, not a hardcoded substitute for correct batching/epoch semantics.

## Interfaces and dependent procedures

Use the common [artifact contracts](../../../contracts.md), [frozen prompt instructions](../../../prompts.md), [decision ledger](../../../decisions.md) and [validation requirements](../../../validation.md). Typed signatures/serialization are delegated implementation details; their scientific meaning is not. The named owner defines this module with its consumers before dependent integration.

| Source node | Called procedure (exact path) | Specification |
|---|---|---|
| `ROTFIT` | `src/training/sft/individual_fit/sft_fit.py` | [sft_fit](individual_fit/sft_fit.md) |
| `KAZFIT` | `src/training/sft/individual_fit/sft_fit.py` | [sft_fit](individual_fit/sft_fit.md) |

## Implemented interface and evidence

Actual flat-layout CE failed the numerical screen before the first update and is rejected despite faster timings. Fused CE remains pending. The third `spatial_fused_backward` candidate retains stock spatial forward and changes only tiled backward; focused checks and GPU evidence remain pending. Every loss screen references the accepted no-recomputation profile and restores the same latest preserved full state, so speed comparisons are incremental. The official 7B ROT13 level-1 fit is safely paused at step 401 with Drive-verified state. Previous approval does not authorize a new loss implementation or source: condition-bound acceptance and affected launch controls remain mandatory before a separately recorded continuation.

`build_worklist(config, datasets, handoff_certificate=certificate)` reads the forty verified JSONL conditions and binds actual ordered record bytes to exact fit specifications. It requires the same ordered train/validation IDs, prompt bytes and common accepted-pool hash across every condition; model-specific clean-trace hashes and final-answer bytes remain identical across that model's families/strengths. Different model traces are never interchangeable. `load_dataset_bundle(bundle)` verifies persisted JSONL content hashes before loading.

`run_family_queue(family, worklist, output_root, config, *, preflight_report=None, fit_runner=run_sft_fit, release_gpu=None, should_pause=None)` runs twenty sequential independent fits. A child failure or pause preserves verified prior fits and an explicit partial queue report. An injected runner is labeled `injected_fixture`, and cannot supply scientific results. `sft_training(..., families=('rot13','kazitext'))`, also exported as `run_sft_training`, runs selected queues with the same reusable engine. The single-process default is sequential; independent external workers may run queues on separately assigned GPUs. A failed family does not erase another family's work. A release callback runs when the queue returns.

The [queue tests](../../../../tests/unit/training/sft/test_sft_training.py) use all forty full-count dataset conditions to verify the worklist, exact ordered exposure inputs, original-base identities, sequential failure/partial reporting, independent family progress and paused incomplete adapter handling. They do not train the construction models. Exact commands/results and live limitations are recorded in `artifacts/preflight/sft_implementation_report.json`.

## Original procedural constraints

The following source comments retain the procedural detail of the supplied diagram. Obsolete introductory editing instructions and raw link directives are omitted from this list; all links are resolved in the path-synchronized diagram below. Appendix H controls any earlier conflicting language.
- SFT training schedule
- Each queue covers 7B and 14B at all 10 levels; fits are independent and start from untouched bases.
- Within each queue, select the next unfinished fit until all 20 are verified.
- A failed fit stops that queue and yields a partial report; the other queue may finish valid work.

## Completion and failure boundary

Implement and test `src/training/sft/sft_training.py` against its specified behavior in `tests/unit/training/sft/test_sft_training.py`. Validate every emitted artifact and preserve its source/configuration identity. Do not claim completeness from a placeholder or passing import. Propagate required child failures and maintain ordinary model outcomes separately from infrastructure faults. Mandatory real preflights and permission gates remain in force. Unspecified allowed conventions must be documented/frozen before outcomes; new scientific choices are not silently authorized.

The [certified launch handoff](../../../sft_data_handoff.md) retains original consumer `773cfd3` and its parent checkpoint at step 214. The historical [fd23db9 comparison](../../../evidence/sft_optimization_pro6000_benchmark.json) rejected deferred checks and SDPA. The later [paired recovery screen](../../../evidence/sft_optimization_pro6000_checkpoint_recovery.json), [accepted profile](../../../evidence/sft_optimization_pro6000_selected_profile.json) and [READY controls](../../../evidence/sft_optimization_pro6000_readiness.json) authorize only the current 7B ROT13 level-1 continuation; other conditions retain original execution defaults. Compatible upstream evidence is reused explicitly. Full checkpoints and final payloads are Drive-verified. Trained adapter measurements explicitly reuse the frozen vLLM protocol.

## Exact path-synchronized diagram

The following diagram is byte-for-byte the active `.mmd` content at generation time. Its procedural content is from the supplied workflow or the cited PDF support requirement; file labels and role assignment are the requested repository organization.

```mermaid
---
title: "spec/mermaid/workflows/training/sft/sft_training.mmd"
---
flowchart TD
%% SFT training schedule
%% Edit quoted node labels directly; this is native Mermaid source.
%% Source 05f42b0 recovery accepted no recomputation only for 7B ROT13 level 1 on RTX PRO 6000 Blackwell 96GB.
%% CUDA input caching failed speed; other conditions retain the original execution defaults.
%% CE policy defaults to spatial_ce; measured flat_ce failed numerical bounds and is rejected.
%% Fused CE and spatial-forward/fused-backward remain pending; no loss candidate is approved.
%% Production source 05f42b0 safely paused at step 401; full state is verified in Drive.
%% Loss trials compare against active no recomputation; preserve latest full-state progress and RNG.
%% Contract: ../../shared_procedures/recovery_procedure.mmd applies to execution failures.
%% Each queue covers 7B and 14B at all 10 levels; fits are independent and start from untouched bases.
%% Within each queue, select the next unfinished fit until all 20 are verified.
%% Current certified launcher serializes selected model/levels on one GPU.
%% Whole optimizer updates and full checkpoints must pass verified Drive barriers.
%% Later adapter evaluation explicitly reuses the frozen vLLM measurement contract.
%% Accepted execution evidence is scoped to one model/family/level; do not release the whole matrix from one screen.
%% Migration inherits verified full-state progress and preserves parent checkpoint bytes.
%% A failed fit stops that queue and yields a partial report; the other queue may finish valid work.
%% Child: ROTFIT => individual_fit/sft_fit.mmd
%% Child: KAZFIT => individual_fit/sft_fit.mmd
    INPUT["40 verified condition datasets"]
    CERT["Bind original producer to certified consumer<br/>40-condition worklist and exact dataset hashes"]
    GATES["Require source-compatible launch evidence<br/>Reuse unchanged baselines and upstream controls"]
    PLAN["Two independent family queues<br/>Select 7B ROT13 first on the current single GPU"]
    ROT["ROT13 queue: assigned GPU<br/>20 independent fits; 10 per model"]
    KAZ["KaziText queue: when a GPU is available<br/>20 independent fits; 10 per model"]
    ROTFIT["src/training/sft/individual_fit/sft_fit.py<br/>Individual SFT fit<br/>Repeat for the ROT13 queue"]
    KAZFIT["src/training/sft/individual_fit/sft_fit.py<br/>Individual SFT fit<br/>Repeat for the KaziText queue"]
    ROTDONE["Verify 20 fits<br/>Release assigned GPU"]
    KAZDONE["Verify 20 fits<br/>Release assigned GPU"]
    WAIT["Wait for both verified queues"]
    OUT["Release all 40 final adapters"]
    BENCH["New source or hardware: paired candidate measurements<br/>Active-profile comparison; scoped acceptance and full-state lineage"]
    INPUT --> CERT --> GATES --> PLAN
    CERT -.->|Migration| BENCH --> GATES
    PLAN --> ROT --> ROTFIT --> ROTDONE --> WAIT
    PLAN --> KAZ --> KAZFIT --> KAZDONE --> WAIT
    WAIT --> OUT
    classDef detail fill:#e5eef9,stroke:#365f8a,stroke-width:2px
    class ROTFIT,KAZFIT detail

%% Repository module: src/training/sft/sft_training.py
%% Module documentation: docs/modules/training/sft/sft_training.md
%% Unit test: tests/unit/training/sft/test_sft_training.py
%% Owner: sft
%% Source sections: C.2; E.1.1–E.1.3; E.2.2,E.2.5
```
