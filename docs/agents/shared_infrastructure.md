# shared_infrastructure.md

Role identifier: `shared_infrastructure`. Reports to: `coordinator`.

Own common services and preflight coordination. Keep scientific rules in their prescribed consumers while providing typed records, stable identities, durable state, bounded failure handling and request-specific transport/cache. Define public interfaces jointly with producer/consumer owners before integration.

## Required reading

Read the full controlling PDF as needed for your scope, including H.1/H.9 precedence and authority. Read the common overview, architecture, contracts, decisions, validation and delegation documents. Read every assigned module specification and active Mermaid source, not just its title. Do not apply historical overridden PDF wording as a second rule.

## Assigned modules

| Implementation | Module specification | Source |
|---|---|---|
| `src/shared_procedures/recovery_procedure.py` | [recovery_procedure](../modules/shared_procedures/recovery_procedure.md) | E.1–E.2; H.9 |
| `src/shared_procedures/configuration.py` | [configuration](../modules/shared_procedures/configuration.md) | E.2.3; H.1,H.9 |
| `src/shared_procedures/records.py` | [records](../modules/shared_procedures/records.md) | E.2.3; H.5–H.6,H.9 |
| `src/shared_procedures/identity.py` | [identity](../modules/shared_procedures/identity.md) | C.1.3; E.1.4; E.2.3 |
| `src/shared_procedures/storage.py` | [storage](../modules/shared_procedures/storage.md) | E.1.1,E.1.3; E.2.3,E.2.5 |
| `src/shared_procedures/checkpointing.py` | [checkpointing](../modules/shared_procedures/checkpointing.md) | E.1.3; E.2.3,E.2.6 |
| `src/shared_procedures/judge_client.py` | [judge_client](../modules/shared_procedures/judge_client.md) | C.3.3–C.3.4; D.2.4–D.2.6; E.1.4; H.5 |
| `src/shared_procedures/response_parsing.py` | [response_parsing](../modules/shared_procedures/response_parsing.md) | 3.1; C.1.3,C.3.2; D.1.2,D.2.4; H.4,H.9 |
| `src/shared_procedures/cost_accounting.py` | [cost_accounting](../modules/shared_procedures/cost_accounting.md) | 5; B.1.4; C.2.4; E.1.4–E.1.5,E.2.5; H.6.4 |
| `src/shared_procedures/preflight.py` | [preflight](../modules/shared_procedures/preflight.md) | 5; E.1–E.2.8; G.9–G.10; H.4,H.7–H.9 |
| `src/shared_procedures/execution_state.py` | [execution_state](../modules/shared_procedures/execution_state.md) | E.1–E.2; H.9 |

## Exclusive implementation write list

These are the implementation-file assignments; the coordinator owns `requirements.lock` and packaging finalization under the current delegation. Implementation files are now populated and locally validated; modify only within your scope. Editing a shared source document, another owner's implementation or the canonical path layout requires coordinator review and synchronized updates.

```text
config.yaml
schemas/candidate_prompt.json
schemas/clean_generation.json
schemas/evaluation_audit_response.json
schemas/evaluation_record.json
schemas/judge_record.json
schemas/optimizer_checkpoint.json
schemas/rl_pass_b_response.json
schemas/score_response.json
schemas/sft_training_record.json
scripts/run_preflight.py
src/shared_procedures/checkpointing.py
src/shared_procedures/configuration.py
src/shared_procedures/cost_accounting.py
src/shared_procedures/execution_state.py
src/shared_procedures/identity.py
src/shared_procedures/judge_client.py
src/shared_procedures/preflight.py
src/shared_procedures/records.py
src/shared_procedures/recovery_procedure.py
src/shared_procedures/response_parsing.py
src/shared_procedures/storage.py
tests/preflight/test_clean_data.py
tests/preflight/test_configuration.py
tests/preflight/test_durable_storage.py
tests/preflight/test_full_checkpoint_resume.py
tests/preflight/test_grpo_update.py
tests/preflight/test_judge_services.py
tests/preflight/test_loss_mask_gradients.py
tests/preflight/test_mbpp_sandbox.py
tests/preflight/test_model_runtime.py
tests/preflight/test_spending_authorization.py
tests/unit/shared_procedures/test_checkpointing.py
tests/unit/shared_procedures/test_configuration.py
tests/unit/shared_procedures/test_cost_accounting.py
tests/unit/shared_procedures/test_execution_state.py
tests/unit/shared_procedures/test_identity.py
tests/unit/shared_procedures/test_judge_client.py
tests/unit/shared_procedures/test_preflight.py
tests/unit/shared_procedures/test_records.py
tests/unit/shared_procedures/test_recovery_procedure.py
tests/unit/shared_procedures/test_response_parsing.py
tests/unit/shared_procedures/test_storage.py
```

## Required handoff

Return actual changed paths, source-to-requirement coverage, public interfaces and typed sample records, exact tests executed and their evidence, unresolved assumptions, remaining live preflights, and all recorded implementation conventions. A mock result is labeled as such. Confirm that no frozen scientific setting, source question count, intervention/reward criterion or empirical success threshold changed. A known implementation bug can be repaired within the design with versioned dependency invalidation; a mandatory unresolved preflight or external authorization need must be reported, not invented away.

## Stop conditions

Stop affected work requiring invalid frozen input, impossible quotas, unbounded retry, unsupported scientific substitution or unavailable external authorization. Preserve completed valid artifacts and explicit partial/failed/indeterminate states. Ordinary model failure is not a request to resample. Do not claim the experiment is implemented because the directory tree or an empty test file exists.


## Current implementation handoff

All eleven shared modules, assigned schemas/configuration, preflight CLI and assigned unit/preflight tests are implemented. The coordinator additionally assigned `tests/integration/test_recovery_handoff.py` for actual CPU partial-update recovery and persistent retries. Public APIs and mode/provenance boundaries are documented in each corresponding module card. GPU/provider controls are explicit nonexperimental preflights, never replacements for the fixed scientific corpus or outcomes.

Scientific configuration identity excludes only the explicitly enumerated operational keys in `OPERATIONAL_IDENTITY_EXCLUSIONS`; full resolved snapshot hashes remain immutable audits. The fixed scientific choices, accepted counts, models/datasets, alpha/fraction grid, SFT/GRPO recipes, reward criteria and construction thresholds are unchanged. Original PDFs are preserved.

Evidence is tiered: actual CPU unit/integration tests, actual statistical simulation (failed mandatory coverage), actual MBPP sandbox evidence when available, and unavailable GPU/API/storage lifecycle/funding gates. `run_preflights(..., execute_live=True)` performs the supplied live checks when resources and current authorization exist, and records explicit blocked/failed results otherwise. No missing external fact is supplied as a successful preflight.

Recovery and spending use indexed SQLite rather than whole-ledger JSON updates. Public logical paths remain compatible; `<path>.sqlite3` holds durable state and legacy JSON is preserved/imported. Production `RecoveryManager.for_worker` shares one 100-retry budget across judge/storage/asset/training operations, while explicit fixture paths remain isolated. Atomic SQLite backup APIs permit coherent independent replication. Actual negative tests cover cross-process budget contention, legacy migration, blocked pre-attempt guards, immutable spending replay, and indexed behavior with thousands of stored operations.

Generation and training launch scopes are explicit. The generation gate does not depend on its own not-yet-generated corpus, baseline/statistical outcomes or paid judge controls. It requires actual selected-model and official transformation controls, durable storage and funding evidence. Training retains all thirteen gates, including real full-corpus validation and the failed statistical gate until resolved within the design. Scoped readiness is separate from training/study readiness.

Final focused verification: shared unit/recovery/preflight scope passed 96 tests with 11 explicit resource/evidence skips in 30.90 seconds. Cross-consumer operations/SFT/RL-setup/recovery scope passed 73 tests in 88.99 seconds; these overlapping counts are not a combined global total. Machine-readable evidence is generated at `artifacts/preflight/shared_infrastructure/validation.json`. The coordinator owns source commits, dependency freeze, consolidated validation and sanitized publication.
