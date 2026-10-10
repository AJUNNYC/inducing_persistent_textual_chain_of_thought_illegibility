# training.md

**Owner:** `coordinator`. **Implementation status:** blank placeholder, not implemented or tested.

| Artifact | Exact repository path |
|---|---|
| Implementation | `src/training/training.py` |
| Workflow/support diagram | [`spec/mermaid/workflows/training/training.mmd`](../../../spec/mermaid/workflows/training/training.mmd) |
| This specification | `docs/modules/training/training.md` |
| Unit test slot | `tests/unit/training/test_training.py` |

**Scientific sources:** 2; C.2–C.3; E.1.1; E.2.2; H.7,H.9. [p. 2](../../../spec/experiment.pdf#page=2) · [p. 4](../../../spec/experiment.pdf#page=4) · [p. 28](../../../spec/experiment.pdf#page=28) · [p. 29](../../../spec/experiment.pdf#page=29) · [p. 30](../../../spec/experiment.pdf#page=30) · [p. 35](../../../spec/experiment.pdf#page=35) · [p. 49](../../../spec/experiment.pdf#page=49) · [p. 54](../../../spec/experiment.pdf#page=54) · [p. 77](../../../spec/experiment.pdf#page=77) · [p. 80](../../../spec/experiment.pdf#page=80).

## Responsibility and controlling details

Coordinate the forty SFT conditions and the two independent 100-step RL trajectories over verified data and pinned recipes. SFT families call one shared fit procedure and start each level from an untouched base. RL calls the shared Dr. GRPO-normalized setup/cycle and preserves the original Sonnet reward. Retain the two original base identities for the common evaluator, yielding 62 states when all required outputs exist.

The SFT queues run independently on their own GPUs and the two RL model runs are sequential on the RL GPU. Only verified exact-condition full checkpoints may resume; another condition is not a warm start. Preserve valid earlier outputs after failure. No reasoning ablation, extra trajectory or adaptive training setting is introduced. The diagram's join describes a complete matrix; missing required children produce an explicit partial result rather than invented completion.

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
%% Training
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
    SFT["src/training/sft/sft_training.py<br/>SFT training schedule<br/>40 independent fits<br/>ROT13 and KaziText for 7B and 14B<br/>2 epochs per fit"]
    SFTOUT["40 final adapters<br/>Every fit starts from its untouched base"]
    RL["src/training/rl_grpo/rl_training.py<br/>RL training: GRPO with Dr. GRPO normalization; 2 runs of 100 steps; unchanged Sonnet reward"]
    RLOUT["20 RL checkpoints<br/>Steps 10, 20, ... 100 for each model"]
    BASE["Retain the 2 untouched base identities"]
    WAIT["Wait for all verified trained states<br/>and matching base identities"]
    OUT["62 states: 2 bases + 40 SFT + 20 RL<br/>No reasoning ablations during training"]
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
