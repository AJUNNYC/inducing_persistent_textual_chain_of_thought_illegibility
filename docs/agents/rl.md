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

These are the complete initial implementation-file assignments. Code files are currently empty; populate only within your scope. Editing a shared source document, another owner's implementation or the canonical path layout requires coordinator review and synchronized updates.

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
