# rl.md

Role identifier: `rl`. Reports to: `coordinator`.

Implement the source-prescribed two trajectories and reusable eight-by-four rollout/reward/update cycle. Preserve the boolean two-pass Sonnet reward exactly and H.7 Dr. GRPO conventions. Never update with missing rewards or resample ordinary bad responses.

## Required reading

Read the full controlling PDF as needed for your scope, including H.1/H.9 precedence and authority. Read the common overview, architecture, contracts, decisions, validation and delegation documents. Read every assigned module specification and active Mermaid source, not just its title. Do not apply historical overridden PDF wording as a second rule.

## Assigned modules

| Implementation | Module specification | Source |
|---|---|---|
| `src/training/rl_grpo/setup/rl_setup.py` | [rl_setup](../modules/training/rl_grpo/setup/rl_setup.md) | C.3.5; E.1.2–E.1.3; E.2.6; H.7 |
| `src/training/rl_grpo/training_cycle/rollouts/rollout_collection.py` | [rollout_collection](../modules/training/rl_grpo/training_cycle/rollouts/rollout_collection.md) | C.3.2,C.3.5; E.1.4,E.2.6; H.7 |
| `src/training/rl_grpo/training_cycle/rewards/sonnet_judgment/sonnet_opacity_judgment.py` | [sonnet_opacity_judgment](../modules/training/rl_grpo/training_cycle/rewards/sonnet_judgment/sonnet_opacity_judgment.md) | C.3.3–C.3.4; E.1.4; H.6.3 |
| `src/training/rl_grpo/training_cycle/rewards/rl_reward.py` | [rl_reward](../modules/training/rl_grpo/training_cycle/rewards/rl_reward.md) | C.3.2–C.3.5; E.1.4; H.6.3,H.7 |
| `src/training/rl_grpo/training_cycle/policy_update/grpo_policy_update.py` | [grpo_policy_update](../modules/training/rl_grpo/training_cycle/policy_update/grpo_policy_update.md) | C.3.5; E.1.3–E.1.4; E.2.6; H.7 |
| `src/training/rl_grpo/training_cycle/grpo_training_cycle.py` | [grpo_training_cycle](../modules/training/rl_grpo/training_cycle/grpo_training_cycle.md) | C.3.4–C.3.5; E.1.3–E.1.4; E.2.6; H.7 |
| `src/training/rl_grpo/rl_training.py` | [rl_training](../modules/training/rl_grpo/rl_training.md) | C.3; E.1.1–E.1.4; E.2.6; H.7 |

## Exclusive implementation write list

These are the complete initial implementation-file assignments. Assigned code files are implemented; modify only within your current authorized scope. Editing a shared source document, another owner's implementation or the canonical path layout requires coordinator review and synchronized updates.

```text
prompts/rl_generation.txt
prompts/rl_pass_a.txt
prompts/rl_pass_b.txt
src/training/rl_grpo/rl_training.py
src/training/rl_grpo/setup/rl_setup.py
src/training/rl_grpo/training_cycle/grpo_training_cycle.py
src/training/rl_grpo/training_cycle/policy_update/grpo_policy_update.py
src/training/rl_grpo/training_cycle/rewards/rl_reward.py
src/training/rl_grpo/training_cycle/rewards/sonnet_judgment/sonnet_opacity_judgment.py
src/training/rl_grpo/training_cycle/rollouts/rollout_collection.py
tests/unit/training/rl_grpo/setup/test_rl_setup.py
tests/unit/training/rl_grpo/test_rl_training.py
tests/unit/training/rl_grpo/training_cycle/policy_update/test_grpo_policy_update.py
tests/unit/training/rl_grpo/training_cycle/rewards/sonnet_judgment/test_sonnet_opacity_judgment.py
tests/unit/training/rl_grpo/training_cycle/rewards/test_rl_reward.py
tests/unit/training/rl_grpo/training_cycle/rollouts/test_rollout_collection.py
tests/unit/training/rl_grpo/training_cycle/test_grpo_training_cycle.py
```

## Required handoff

Return actual changed paths, source-to-requirement coverage, public interfaces and typed sample records, exact tests executed and their evidence, unresolved assumptions, remaining live preflights, and all recorded implementation conventions. A mock result is labeled as such. Confirm that no frozen scientific setting, source question count, intervention/reward criterion or empirical success threshold changed. A known implementation bug can be repaired within the design with versioned dependency invalidation; a mandatory unresolved preflight or external authorization need must be reported, not invented away.

## Stop conditions

Stop affected work requiring invalid frozen input, impossible quotas, unbounded retry, unsupported scientific substitution or unavailable external authorization. Preserve completed valid artifacts and explicit partial/failed/indeterminate states. Ordinary model failure is not a request to resample. Do not claim the experiment is implemented because the directory tree or an empty test file exists.

## Implemented handoff (10 October 2026)

The seven assigned RL implementation modules and three frozen prompt files now contain real implementations. Unit files are populated, and the coordinator additionally authorized `tests/integration/test_reward_transaction.py` and `tests/integration/test_training_handoff.py`. No scientific setting, source question count, reward criterion, model identity or empirical success threshold was changed. No scientific model output or live provider success is fabricated.

Public entry points are `setup_rl_run`, `collect_rollouts`, `compute_rl_reward`, `judge_opacity`, `apply_grpo_update`, `grpo_training_cycle` and `run_rl_training`. All external record boundaries use JSON-compatible mappings. `run_rl_training` supports a unique nonempty `model_sizes` subset of `7b`/`14b`; the default runs both in canonical order. Separate immutable model and tokenizer revisions are accepted. Production uses the actual lazy Transformers/PEFT engine and a live budgeted `JudgeClient`; injected engines and providers are explicit fixtures. Actual tokenizer/model loading first provisions the verified shared immutable Hub cache and then sets SDK loading to `local_files_only`; no duplicate implicit SDK download is permitted. Production uses shared worker-wide persistent recovery counters; explicit fixture paths remain isolated. Shared preflight reports are bound to the exact global configuration hash.

Frozen permitted runtime conventions are Torch AdamW (betas 0.9/0.999, epsilon 1e-8, weight decay zero), constant learning rate, NF4 double quantization/BF16, all seven attention/MLP LoRA projection targets, dropout zero, gradient checkpointing, microbatch one with 32 completion contributions before one update, gradient clipping norm one and group sample standard deviation plus 1e-4. Native prompt and parser literal serialization is versioned. Diagnostic duplicate-token-fourgram fractions do not enter rewards or loss. Scientific adapters use lowercase model namespaces and `step_###` directories; all scheduled full recovery states are retained.

Executed local tests verify strict boolean Sonnet semantics and visibility, invalid/wrong no-API gating, independent pass persistence across a schema failure, every-32-reward barriers, exact logged eight-by-four grouping/order, numerical clipped Dr. GRPO loss and gradients, zero-variance outcomes, full-state rollback/replay with persistent counters, selected-stage execution, both trajectories' twenty original-step fixture snapshots, and exact earlier-step restoration. They use declared external doubles; real CPU tensor operations and checkpoint/transaction persistence execute normally. The actual-model `run_grpo_preflight` is implemented and CPU blocking before any download is tested. Actual CUDA gradients, full model resumption, provider schema/availability, storage lifecycle, baseline repeatability, statistical validation and spending authorization remain launch requirements.

The dedicated RL verification command in `docs/implementation_acceptance.md` passed all 54 tests in 194.66 seconds after canonical runtime path correction and the actual preflight's deliberate state-perturbation recovery check. After shared asset loading and worker recovery integration, the same RL/integration files together with the initially collected Drive/RunPod cases passed 72 tests in 270.47 seconds. The final Drive/RunPod scope, including three added actual token-introspection endpoint contract cases, passed 21 tests in 0.71 seconds. These overlapping observations must not be added as distinct global test counts. The completed two-model trajectory resume test passed in the stabilized 72-test run; its scientific source identity guards were retained. Compilation and whitespace verification passed.

The coordinator additionally delegated a read-only operations audit, `src/operations/runpod.py`, `tests/unit/operations/test_drive.py`, `tests/unit/operations/test_runpod.py`, the acceptance report and final module/requirement registers. The audit identified restrictive OAuth consent overriding, unverified folder ancestry, nonfinite authorization and stop-acknowledgment issues. The coordinator repaired Drive scope/ancestry and authorization guards; this agent implemented bounded observed termination confirmation, truthful `STOP_UNCONFIRMED`, exception cleanup and saved-state revalidation. These operations checks use explicit provider SDK/HTTP doubles, not real account or paid Pod success.

The synchronized registers preserve all 49 original scientific workflow/support module entries and all 82 scientific/infrastructure requirement wordings and source provenance. They add eight user/H.9 operations modules and eight operational requirements with existing source/card/diagram/executable verification mappings. The 90 requirements distinguish local implementation from pending live/statistical/final-audit evidence. The GPU/API budget integration gap remains explicit: separate ceilings require allocation before combined paid work. A pending authorization under ignored runtime artifacts records the user-authorized existing H100 limit of 90 USD and 108000 seconds; actual Pod ID, quote and billing evidence are unresolved, and no new Pod is authorized.

The final actual-frame statistical report was subsequently observed as FAILED, with hash `79ffded9ddc2cd7af529abe3af092008b1777fca4df178b936373743e34a9ddc`. The prospective decision-boundary scenario had 34/36 joint successes, two failures and a futility stop; the remaining six scenarios were not run. The acceptance and requirement registers now distinguish this actual mandatory rejection from local fixture test passes. Confirmatory launch remains blocked. No trained outcome was inspected or scientific success asserted.

The subsequent secure Jupyter consent delegation added `scripts/create_google_consent_notebook.py`, `docs/operations/jupyter_consent.md` and `tests/unit/operations/test_consent_notebook.py`. The generator produces a standalone notebook using the dedicated `illegibility_oauth` kernel and existing protected desktop-client path, hidden callback entry, exact state/loopback checks, Google HTTPS endpoints, refresh/token-scope/intended-account verification and atomic private token storage. Seventeen boundary tests passed in 0.37 seconds. Actual user consent and live Drive verification remain separate coordinator observations; no private values are embedded in the notebook or repository.
