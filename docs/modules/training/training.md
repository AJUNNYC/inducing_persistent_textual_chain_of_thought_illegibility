# training.md

**Owner:** `coordinator`. **Current status:** All ten 14B ROT13 levels are complete with verified Drive adapters, full state and fit results. The nine fresh levels 1–9 each reached 625 updates under scientific source `9de8813031da737453f491723a10b305679dc42d`; retained level 10 preserves source `2ea4fb9a7737cff806226366e73743eec5935fad`. KaziText level 10 remains safely paused at 444/625 with verified recovery state and no automatic restart. No other KaziText fit or 7B fit is newly completed by this queue. The separately identified 16 October L10 extra epoch is also complete: 625 additional/1,250 cumulative updates and 26 Drive-verified artifacts. Only its new-adapter GPQA run is active; a completion-biased 42-request snapshot provisionally shows sustained ROT13 in reviewed passages, alongside looping and garbled decoded terminology. Official full-panel results remain pending and original levels 7-9 stay paused. Extra exposure, schedule restart and native-newline alignment changed jointly; no isolated epoch effect or capability benefit is established. See the [current SFT record](sft/sft_training.md).

| Artifact | Exact repository path |
|---|---|
| Implementation | `src/training/training.py` |
| Workflow/support diagram | [`spec/mermaid/workflows/training/training.mmd`](../../../spec/mermaid/workflows/training/training.mmd) |
| This specification | `docs/modules/training/training.md` |
| Unit test slot | `tests/unit/training/test_training.py` |

**Scientific sources:** 2; C.2–C.3; E.1.1; E.2.2; H.7,H.9. [p. 2](../../../spec/experiment.pdf#page=2) · [p. 4](../../../spec/experiment.pdf#page=4) · [p. 28](../../../spec/experiment.pdf#page=28) · [p. 29](../../../spec/experiment.pdf#page=29) · [p. 30](../../../spec/experiment.pdf#page=30) · [p. 35](../../../spec/experiment.pdf#page=35) · [p. 49](../../../spec/experiment.pdf#page=49) · [p. 54](../../../spec/experiment.pdf#page=54) · [p. 77](../../../spec/experiment.pdf#page=77) · [p. 80](../../../spec/experiment.pdf#page=80).

## Responsibility and controlling details

The active `efficient_one_epoch_v1` SFT default uses all 10,000 examples once per explicitly selected condition: 625 updates and 19 warmup updates, BF16 base/head, exact full-gradient CCE, SDPA and plain AdamW. Published policy v2 requests physical microbatch 2 × accumulation 8, with capacity fallback 1 × 16 and unchanged effective batch 16. Each optimizer update retains the same ordered 16 examples and per-example weight 1/16; actual batch/accumulation/recomputation is recorded and fixed on resume. All ten 14B ROT13 levels are complete; all ten KaziText conditions remain uncompleted, including the paused level 10. Available data still spans forty model/family/level conditions, and explicit legacy NF4 recipes remain separately identified. See the [training redesign](../../sft_short_pilot.md) and [SFT module](sft/sft_training.md).

The original full matrix comprises forty SFT fits, two independent 100-step RL trajectories with twenty snapshots, and two bases, yielding62 states if every required output exists. This historical accounting is not a current launch instruction. RL and final confirmatory evaluation remain deferred. Preserve the original Sonnet reward, Dr. GRPO semantics and original final evaluation thresholds; this redesign changes only the declared SFT recipe. Never infer completed branches from a join node or relabel old full checkpoints as efficient-run state.

Selected fits start independently from untouched bases, retain disjoint Drive outputs and may run on separate assigned workers after explicit resume. Native model/mask/gradient/capacity/full-state guards execute during normal startup before updates; the completed ROT1314B L10 receipt certifies that condition only, not future startup on another worker or condition. Preserve partial/failed/deferred states honestly.

## Interfaces and dependent procedures

Use the common [artifact contracts](../../contracts.md), [frozen prompt instructions](../../prompts.md), [decision ledger](../../decisions.md) and [validation requirements](../../validation.md). Typed signatures/serialization are delegated implementation details; their scientific meaning is not. The named owner defines this module with its consumers before dependent integration.

| Source node | Called procedure (exact path) | Specification |
|---|---|---|
| `SFT` | `src/training/sft/sft_training.py` | [sft_training](sft/sft_training.md) |
| `RL` | `src/training/rl_grpo/rl_training.py` | [rl_training](rl_grpo/rl_training.md) |

## Original procedural constraints

The following source comments retain the procedural detail of the supplied diagram. Obsolete introductory editing instructions and raw link directives are omitted from this list; all links are resolved in the path-synchronized diagram below. Appendix H controls any earlier conflicting language.
- Training
- Execution contract: known faults use bounded recovery; unclassified/persistent faults stop the worker.
- Frozen invalid inputs are fatal. Unrecoverable child failures propagate; never release incomplete results.
- RL uses the same fixed 800 questions and order for both sizes. Four rollouts form each GRPO group.
- Independent SFT family queues run on their own GPUs; RL model trajectories run sequentially.
- Exact-condition full checkpoints may resume a run; another level's adapter is never a warm start.
- No training-time reasoning ablations. Earlier valid outputs are preserved if a worker stops.
- RL uses GRPO with Dr. GRPO loss normalization and the fixed settings in rl-setup.mmd.
- The post-training GPT4o artifact audit does not change the Sonnet two-pass training reward.

## Completion and failure boundary

Implement and test `src/training/training.py` against its specified behavior in `tests/unit/training/test_training.py`. Validate every emitted artifact and preserve its source/configuration identity. Do not claim completeness from a placeholder or passing import. Propagate required child failures and maintain ordinary model outcomes separately from infrastructure faults. Mandatory real preflights and permission gates remain in force. Unspecified allowed conventions must be documented/frozen before outcomes; new scientific choices are not silently authorized.

## Exact path-synchronized diagram

The following diagram is byte-for-byte the active `.mmd` content at generation time. Its procedural content is from the supplied workflow or the cited PDF support requirement; file labels and role assignment are the requested repository organization.

```mermaid
---
title: "spec/mermaid/workflows/training/training.mmd"
---
flowchart TD
%% Dated13Oct21:38UTC evidence: actualGPQA49/198 complete/Drive verified;133 truncated+3 malformed at3072 tokens.
%% Current14Oct: allten14B ROT13 fits complete625/Driveverified; L1-9source9de, retainedL10source2ea.
%% RetainedROT13L10 keeps old2ea4fb9 identity; laterdocumentation cannot relabel executed workers.
%% Training
%% Active default: efficient_one_epoch_v1, per explicitly selected SFT condition.
%% Published policy v2: request physical 2 x 8; capacity fallback 1 x 16, same effective 16.
%% Actual batch/accumulation/recomputation mode stays fixed throughout each fit and exact resume.
%% Committed coordinator requires untouched 14B Diamond 198 baseline+verified Drive barrier before 19 new fits.
%% Executednine-levelROT13queue isCOMPLETE; KaziTextpaused444/625, noautomaticrestart.
%% Authorization $240 recorded; no inferred new price or rate-based cutoff; explicit deadline/safe pause separate.
%% Forty datasets available; alltenROT13 fitscomplete, alltenKaziText fitsuncompleted.
%% Historical$240SFT authorization retained; original GPQA32k controls retained.
%% 16Oct L10 extra epoch COMPLETE625/cumulative1250/26Driveverified; newL10 GPQA running.
%% Provisional42completed snapshot: sustained ROT13 in reviewed positive passages and looping; official198pending.
%% Legacy full-grid62state accounting is preserved, not an active launch instruction.
%% Edit quoted node labels directly; this is native Mermaid.
%% Execution contract: known faults use bounded recovery; unclassified/persistent faults stop the worker.
%% Frozen invalid inputs are fatal. Unrecoverable child failures propagate; never release incomplete results.
%% RL uses the same fixed 800 questions and order for both sizes. Four rollouts form each GRPO group.
%% Independent SFT family queues run on their own GPUs; RL model trajectories run sequentially.
%% Exact-condition full checkpoints may resume a run; another level's adapter is never a warm start.
%% No training-time reasoning ablations. Earlier valid outputs are preserved if a worker stops.
%% RL uses GRPO with Dr. GRPO loss normalization and the fixed settings in rl_grpo/setup/rl_setup.mmd.
%% The post-training GPT4o artifact audit does not change the Sonnet two-pass training reward.
%% Child: SFT => sft/sft_training.mmd
%% Child: RL => rl_grpo/rl_training.mmd
    INPUT["Verified data and pinned training recipes"]
    SFT["src/training/sft/sft_training.py<br/>Selected independent SFT fits<br/>All ten14B ROT13 fits complete625 and Drive-verified<br/>L1-9source9de; retainedL10source2ea<br/>KaziTextpaused444/625; noautomaticrestart<br/>40 datasets and 7B history retained<br/>Default: efficient one epoch / 625 updates"]
    SFTOUT["Only selected verified adapters<br/>Each starts from its untouched base"]
    RL["src/training/rl_grpo/rl_training.py<br/>Deferred RL design: Dr. GRPO<br/>2 independent 100-step trajectories; unchanged reward"]
    RLOUT["20 RL checkpoints<br/>Steps 10, 20, ... 100 for each model"]
    BASE["Retain the 2 untouched base identities"]
    WAIT["Join verified selected outputs<br/>Record partial and deferred branches explicitly"]
    OUT["Selected outputs only<br/>Historical full-grid total: 62 states<br/>No claimed completion of deferred jobs"]
    INPUT --> SFT --> SFTOUT --> WAIT
    INPUT --> RL --> RLOUT --> WAIT
    INPUT --> BASE --> WAIT
    WAIT --> OUT
    classDef detail fill:#e5eef9,stroke:#365f8a,stroke-width:2px
    class SFT,RL detail

%% Repository module: src/training/training.py
%% Module documentation: docs/modules/training/training.md
%% Unit test: tests/unit/training/test_training.py
%% Owner: coordinator
%% Source sections: 2; C.2–C.3; E.1.1; E.2.2; H.7,H.9
```
