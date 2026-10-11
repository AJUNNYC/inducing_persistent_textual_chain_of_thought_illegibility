# data_preparation.md

> Historical implementation handoff from 10 October 2026. Its test counts and then-blocked integrations describe that handoff, not current execution status. See the [current procedure](../procedure_update.md) and [current validation report](../validation_report.md) for completed generation, Drive verification and remaining training gates.

Role identifier: `data_preparation`. Reports to: `coordinator`.

Own authorized source manifests, structural paired clean data, exact transforms and frozen evaluation/RL question selections. Preserve common IDs, model-specific traces, all-condition validity and no semantic-quality filter. Candidate filtering and short-mask cases must follow the sources, not a desired result.

## Required reading

Read the full controlling PDF as needed for your scope, including H.1/H.9 precedence and authority. Read the common overview, architecture, contracts, decisions, validation and delegation documents. Read every assigned module specification and active Mermaid source, not just its title. Do not apply historical overridden PDF wording as a second rule.

## Assigned modules

| Implementation | Module specification | Source |
|---|---|---|
| `src/data_generation_collection/sft_data/clean_trace_collection/candidate_batch/clean_trace_batch.py` | [clean_trace_batch](../modules/data_generation_collection/sft_data/clean_trace_collection/candidate_batch/clean_trace_batch.md) | 3.1; C.1.3; C.2.3; E.2.4; H.2.3,H.3 |
| `src/data_generation_collection/sft_data/clean_trace_collection/clean_trace_collection.py` | [clean_trace_collection](../modules/data_generation_collection/sft_data/clean_trace_collection/clean_trace_collection.md) | 3.1; B.1; E.2.2–E.2.4; H.2.3,H.3 |
| `src/data_generation_collection/sft_data/rot13/rot13_data_construction.py` | [rot13_data_construction](../modules/data_generation_collection/sft_data/rot13/rot13_data_construction.md) | 3.2; C.1.1–C.1.3; C.2.3; E.2.4; H.3 |
| `src/data_generation_collection/sft_data/kazitext/kazitext_data_construction.py` | [kazitext_data_construction](../modules/data_generation_collection/sft_data/kazitext/kazitext_data_construction.md) | C.1.4; C.2.3; E.2.4; H.2–H.3 |
| `src/data_generation_collection/sft_data/sft_data_preparation.py` | [sft_data_preparation](../modules/data_generation_collection/sft_data/sft_data_preparation.md) | 3.1; B.1; C.1–C.2.3; E.2.2–E.2.4; H.2–H.3 |
| `src/data_generation_collection/rl_data/rl_data_preparation.py` | [rl_data_preparation](../modules/data_generation_collection/rl_data/rl_data_preparation.md) | C.3.2,C.3.5; E.2.2; H.7 |
| `src/data_generation_collection/evaluation_data/capability_panel/capability_panel_construction.py` | [capability_panel_construction](../modules/data_generation_collection/evaluation_data/capability_panel/capability_panel_construction.md) | D.1.2; H.4 |
| `src/data_generation_collection/evaluation_data/legibility_subset/legibility_subset_construction.py` | [legibility_subset_construction](../modules/data_generation_collection/evaluation_data/legibility_subset/legibility_subset_construction.md) | D.2.3; H.8.2 |
| `src/data_generation_collection/evaluation_data/evaluation_data_preparation.py` | [evaluation_data_preparation](../modules/data_generation_collection/evaluation_data/evaluation_data_preparation.md) | D.1.2; D.2.3; E.2.2; H.4,H.8.2 |
| `src/data_generation_collection/data_generation.py` | [data_generation](../modules/data_generation_collection/data_generation.md) | 3.1; B.1; C.1; C.2.3; E.2.2–E.2.4; H.2–H.3 |
| `src/data_generation_collection/sft_data/clean_trace_collection/candidate_manifest.py` | [candidate_manifest](../modules/data_generation_collection/sft_data/clean_trace_collection/candidate_manifest.md) | 3.1; B.1; E.2.2–E.2.4; H.3 |

## Exclusive implementation write list

These are the complete initial implementation-file assignments. Code files are currently empty; populate only within your scope. Editing a shared source document, another owner's implementation or the canonical path layout requires coordinator review and synchronized updates.

```text
src/data_generation_collection/data_generation.py
src/data_generation_collection/evaluation_data/capability_panel/capability_panel_construction.py
src/data_generation_collection/evaluation_data/evaluation_data_preparation.py
src/data_generation_collection/evaluation_data/legibility_subset/legibility_subset_construction.py
src/data_generation_collection/rl_data/rl_data_preparation.py
src/data_generation_collection/sft_data/clean_trace_collection/candidate_batch/clean_trace_batch.py
src/data_generation_collection/sft_data/clean_trace_collection/candidate_manifest.py
src/data_generation_collection/sft_data/clean_trace_collection/clean_trace_collection.py
src/data_generation_collection/sft_data/kazitext/kazitext_data_construction.py
src/data_generation_collection/sft_data/rot13/rot13_data_construction.py
src/data_generation_collection/sft_data/sft_data_preparation.py
tests/unit/data_generation_collection/evaluation_data/capability_panel/test_capability_panel_construction.py
tests/unit/data_generation_collection/evaluation_data/legibility_subset/test_legibility_subset_construction.py
tests/unit/data_generation_collection/evaluation_data/test_evaluation_data_preparation.py
tests/unit/data_generation_collection/rl_data/test_rl_data_preparation.py
tests/unit/data_generation_collection/sft_data/clean_trace_collection/candidate_batch/test_clean_trace_batch.py
tests/unit/data_generation_collection/sft_data/clean_trace_collection/test_candidate_manifest.py
tests/unit/data_generation_collection/sft_data/clean_trace_collection/test_clean_trace_collection.py
tests/unit/data_generation_collection/sft_data/kazitext/test_kazitext_data_construction.py
tests/unit/data_generation_collection/sft_data/rot13/test_rot13_data_construction.py
tests/unit/data_generation_collection/sft_data/test_sft_data_preparation.py
tests/unit/data_generation_collection/test_data_generation.py
```

## Required handoff

Return actual changed paths, source-to-requirement coverage, public interfaces and typed sample records, exact tests executed and their evidence, unresolved assumptions, remaining live preflights, and all recorded implementation conventions. A mock result is labeled as such. Confirm that no frozen scientific setting, source question count, intervention/reward criterion or empirical success threshold changed. A known implementation bug can be repaired within the design with versioned dependency invalidation; a mandatory unresolved preflight or external authorization need must be reported, not invented away.

## Stop conditions

Stop affected work requiring invalid frozen input, impossible quotas, unbounded retry, unsupported scientific substitution or unavailable external authorization. Preserve completed valid artifacts and explicit partial/failed/indeterminate states. Ordinary model failure is not a request to resample. Do not claim the experiment is implemented because the directory tree or an empty test file exists.

## Completed implementation handoff

All eleven assigned source modules and their eleven unit-test files contain concrete implementations; `tests/integration/test_data_handoff.py` was additionally assigned by the coordinator. The coordinator authorized updates to these eleven module cards and this role document. No other owner's implementation, initializer, scientific source diagram or experiment PDF was changed by this role.

Public interfaces use JSON-compatible mappings and canonical installed-package imports (`data_generation_collection`, `shared_procedures`, `training`, `operations`). SFT records use the shared schema names, and evaluation/RL item identities use `question_id` with `item_id`/`example_id` aliases where their consumers require them. Public API contracts and delegated algorithms are documented in each assigned module card. All original scientific models, source revisions, quotas, intervention levels, generation/training limits, reward criteria and empirical success thresholds remain unchanged.

The actual checksummed public frames are GSM8K test1319/train7473; MMLU-Redux-2.0 5700, of which5330 have `error_type == "ok"`; ARC-Easy test2376; sanitized MBPP427 across its splits. Actual ignored artifacts are `data/evaluation/official_pools.json`, `data/evaluation/capability_panel_prepared.json` (1233), `data/evaluation/legibility_subset_prepared.json` (200) and `data/rl/manifest.json` (800 questions, 100 ordered batches of eight). The original 203 selected MBPP reference checks are an evaluation-owner prerequisite; selected IDs are never resampled after a failure. Evaluation source preparation does not authorize launch before the prospective protocol and mandatory preflights pass.

The official unmodified KaziText source/profile, individual English-EWT UD2.5 model and recursively resolved actual English Aspell files were acquired/verified. `provision_kazitext_assets` provides an executable reusable provisioning path. The model SHA-256 is `784bd0fa85e3d831fd02a55290d0acfd05c953159dc38cc33d52e1b28add9957`. Actual CPU dictionary provenance and paths are in `/workspace/illegibility_aux/kazitext_auxiliary_manifest.json`; stochastic identities contain names/hashes/settings rather than machine-specific paths. Genuine CPU runtime verification executed all20 model/level requests twice (40 native transforms), passing exact replay, numerical/math protection and lexical-boundary protection. `artifacts/preflight/kazitext_auxiliary_runtime.json` is diagnostic-only actual-engine evidence, not generated scientific data.

Executed verification: the data-owner fast unit selection passed32 tests with the one full-count fixture deselected (6.24s). The SFT data unit file plus actual producer-to-SFT consumer integration passed4 tests (220.78s), including actual serialization/readback of all404000 fixture records, common10000/100 IDs, model-specific source hashes, shared-schema validation, full40-fit worklist construction and source-final corruption rejection. Those external model/tokenizer/noise outputs are explicit simulation fixtures. Separate production/fixture raw cache identities, directories and propagated simulation flags prevent their use as live clean-data evidence.

Remaining real prerequisites: authorized LMSYS gated-source access, frozen untouched construction model assets, compatible GPU runtime and mandatory real clean-data/training/evaluation/service/storage/statistical preflights. No authorized LMSYS candidate manifest, scientific paired10100 clean pool or scientific SFT datasets were fabricated. Root/shared infrastructure own the complete launch gate and durable-storage lifecycle; this role returns all concrete blockers with implemented source/provisioning paths.

The strengthened full-count SFT unit run passed3 tests (306.73s), including exact recomputation of all404000 actual serialized fixture records and fatal checksum corruption detection. Graceful operational pause support additionally passed raw-sync/cache-resume and exact next-position checks: `NativeCleanGenerator(..., stop_requested=callback)`, `process_clean_batch(..., stop_requested=callback)`, `collect_clean_traces(..., stop_requested=callback, stop_file=path)` and `prepare_sft_data(..., stop_requested=callback, stop_file=path)`. The collector also observes `output_dir/STOP_REQUESTED` by default. It persists a `status="paused"` collection mapping with exact next model/candidate operation, completed raw/pair IDs, committed batch cursor and frozen source/configuration identity. It does not publish a partial accepted pool or intervention dataset. The caller must branch on paused status, record its workflow state and let the budget controller stop billing; remove the stop request only when resumption is authorized. No automatic signal handler is installed. Abrupt process termination may lose the in-flight generation, while completed durable/synced raw responses remain reusable.

Sanitized tracked evidence is `docs/evidence/data_preparation.json` (five verified public-source manifests,64 shards, actual frame/panel/RL counts and fixture test outcomes) and `docs/evidence/kazitext_auxiliary_runtime.json` (40 actual native requests, exact replay/protection, verified assets and all50 effective non-locator Aspell options). Neither contains credentials, local auxiliary paths, full source texts or invented scientific model outcomes.

Generation-stage readiness uses the separate public `run_transformation_preflight(config, output_dir)` in the KaziText module. Its canonical `name="transformation_runtime"` result checks both actual immutable native tokenizers and all40 intervention variants through the actual official auxiliary engine and intact assistant renderer. The evidence is explicitly a runtime control (`live=True`, `simulation=False`, `control_preflight=True`, `scientific_results=False`), never a scientific clean-data result. The subsequent full-corpus gate still verifies the actual10100 paired pool and all404000 serialized intervention records. Root main enforces exact frozen dictionary/settings comparison before scientific generation and durable paused-stage handling.

Final actual CPU transformation control executed successfully: `docs/evidence/transformation_runtime.json` records all40 verified native-tokenizer/intervention conditions, exact replay/protection and intact native target lengths. Both tokenizers were acquired through immutable verified model-cache manifests and subsequently loaded with networking disabled. This additional evidence contains diagnostic controls and immutable hashes only; it does not claim base-model inference or clean-corpus readiness.
