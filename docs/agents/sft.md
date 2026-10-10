# sft.md

Role identifier: `sft`. Reports to: `coordinator`.

Implement one fit engine for forty independent conditions and two sequential family queues, using the fixed QLoRA recipe and no cross-level warm start, early stopping or adaptive exposure. Integrate actual loss-mask, gradient and checkpoint evidence before launch.

## Required reading

Read the full controlling PDF as needed for your scope, including H.1/H.9 precedence and authority. Read the common overview, architecture, contracts, decisions, validation and delegation documents. Read every assigned module specification and active Mermaid source, not just its title. Do not apply historical overridden PDF wording as a second rule.

## Assigned modules

| Implementation | Module specification | Source |
|---|---|---|
| `src/training/sft/individual_fit/sft_fit.py` | [sft_fit](../modules/training/sft/individual_fit/sft_fit.md) | C.2.2–C.2.4; E.1.3; E.2.4,E.2.6; H.9 |
| `src/training/sft/sft_training.py` | [sft_training](../modules/training/sft/sft_training.md) | C.2; E.1.1–E.1.3; E.2.2,E.2.5 |

## Exclusive implementation write list

These are the complete initial implementation-file assignments. Code files are currently empty; populate only within your scope. Editing a shared source document, another owner's implementation or the canonical path layout requires coordinator review and synchronized updates.

```text
src/training/sft/individual_fit/sft_fit.py
src/training/sft/sft_training.py
tests/unit/training/sft/individual_fit/test_sft_fit.py
tests/unit/training/sft/test_sft_training.py
```

## Required handoff

Return actual changed paths, source-to-requirement coverage, public interfaces and typed sample records, exact tests executed and their evidence, unresolved assumptions, remaining live preflights, and all recorded implementation conventions. A mock result is labeled as such. Confirm that no frozen scientific setting, source question count, intervention/reward criterion or empirical success threshold changed. A known implementation bug can be repaired within the design with versioned dependency invalidation; a mandatory unresolved preflight or external authorization need must be reported, not invented away.

## Stop conditions

Stop affected work requiring invalid frozen input, impossible quotas, unbounded retry, unsupported scientific substitution or unavailable external authorization. Preserve completed valid artifacts and explicit partial/failed/indeterminate states. Ordinary model failure is not a request to resample. Do not claim the experiment is implemented because the directory tree or an empty test file exists.
