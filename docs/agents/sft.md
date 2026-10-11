# sft.md

> Historical implementation handoff from 10 October 2026. Its test counts and then-blocked integrations describe that handoff, not current execution status. See the [current procedure](../procedure_update.md) and [current validation report](../validation_report.md) for completed generation, Drive verification and remaining training gates.

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

## Implemented handoff

The initial SFT implementation and test assignments are complete. The coordinator additionally delegated `src/operations/assets.py`, `tests/unit/operations/test_assets.py` and its module card for the verified shared cache. The root orchestration files `src/experiment_overview/experiment_overview.py`, `src/training/training.py` and `tests/unit/experiment_overview/test_experiment_overview.py` were repaired against real child signatures and released back to the coordinator before final lifecycle edits. No child commit was created.

### Coverage and interfaces

- C.2.2–C.2.4: actual single Transformers/PEFT/BitsAndBytes fit engine, fixed QLoRA recipe, completion-only masks over the entire assistant target, exact fixed two-epoch exposure and held-out loss without selection. `SFTFitSpec`, `run_sft_fit`, `render_sft_completion`, `tokenize_completion`, `completion_loss` and `verify_fit_result` are implemented in the assigned fit module.
- E.1.3/E.2.4/E.2.6: exact source/configuration/model/tokenizer/code/dependency identities, atomic complete-state checkpoints, latest two recovery points, optimizer/scheduler/RNG/sampler restoration, final safetensors/full-checkpoint tensor verification and persistent fault budgets. A result's checkpoint hash and the checkpoint's exposure counters must agree with actual completion.
- C.2/E.1.1–E.1.3/E.2.2/E.2.5: forty independent conditions, two sequential twenty-fit family queues, original-base initialization, ordered shared source IDs, child failure/partial reports and GPU-release callbacks. `build_worklist`, `load_dataset_bundle`, `run_family_queue`, `sft_training` and alias `run_sft_training` are implemented.
- H.9 and operational E/H requirements: immutable HTTPS upstream metadata, Git/LFS whole-object digest verification, interrupted range resume, shared worker retry ledger, native Hub snapshots and one physical blob per object. `model_load_kwargs` provisions/verifies and returns `{'cache_dir': '<canonical_parent>/hub', 'local_files_only': True}`. Model and tokenizer revisions may differ. Native clean-generation, SFT, RL and evaluation owners integrated this boundary.

An input fit identity is a mapping with `family:str`, `model_id:'7b'|'14b'`, `level:int`, immutable `model_revision:str[40]`, `tokenizer_revision:str[40]`, `configuration_hash:str[64]`, and `dataset_hash:str[64]`. `SFTFitSpec.identity()` additionally supplies run ID, full HF model ID, strength, implementation version and fixed recipe. A completed fit mapping contains actual `progress`, `validation_loss:float`, `validation_count:100`, `adapter_path:str`, `adapter_files:dict[path,sha256]`, `adapter_manifest_hash:str[64]`, `checkpoint_hash:str[64]`, runtime and exact identity. Production completion requires the corresponding verified final adapter and full checkpoint; no scientific complete mapping was fabricated during implementation.

### Frozen conventions and evidence

Prospectively approved convention_009 fixes per-example supervised-token means, then an example mean; right padding, no sequence packing, eager attention, nonreentrant gradient checkpointing, seeded two-epoch exposure and upward-rounded 3% warmup (38 of 1,250 updates). `native_training_template_v1` preserves native role/end wrappers while inserting the complete training target. Both official native tokenizer templates strip historical assistant reasoning; training rendering preserves the whole reasoning and final answer with no added system message. Source length validation calls the same renderer. Native evaluation generation uses the published generation template.

Executed CPU tests perform actual masked CE/backward passes, tiny-model optimizer/validation work, exact interrupted dropout/optimizer/scheduler/RNG recovery, all seven real PEFT projection targets, safetensors equality/integrity and 100/final checkpoint cadence. Full-count source fixtures exercise all forty conditions and ordered inputs. Cache tests perform real byte digests, native Hub local lookup, shared-inode reuse, distinct tokenizer pins, range resume, finite persisted retries, corruption and token-safe redirect checks. The final focused command is:

```bash
PYTHONPATH=src /workspace/illegibility_env/bin/python -m pytest \
  tests/unit/operations/test_assets.py tests/unit/training/sft -q \
  --junitxml=/workspace/scratch/sft_assets_final_pytest.xml
```

Exact final results and implementation hashes are recorded in `artifacts/preflight/sft_implementation_report.json`. Earlier combined cache/SFT/root orchestration execution passed 40 tests in 107.51 seconds; later root lifecycle changes belong to the coordinator's consolidated suite.

Actual official tokenizer acquisition and local-only loading also passed for 7B commit `916b56a44061fd5cd7d6a8fb632557ed4f724f60` and 14B commit `1df8507178afcc1bef68cd8c393f61a886323761`. Each published tokenizer JSON SHA-256 is `88145e3c3249adc2546ede277e9819d6e405e19072456e4b521cbc724bd60773`; tokenizer configuration SHA-256 is `8ac8c85fb242563c2260baec0909debd69d718af6a0b3d90e6cab62b4d341cd5`. Native selected tokenizer/cache evidence is `/workspace/scratch/asset_model_cache/validation.json`; it establishes CPU acquisition/rendering, not a CUDA fit.

### Live boundary and ownership release

No frozen scientific hyperparameter, source count, intervention grid, reward rule or empirical/statistical construction threshold was changed. Fixture identities and artificial completion counters in the export validator test are explicitly labeled; they establish software invariants only. Actual construction-model weight loading, CUDA loss/gradient and full-state roundtrip, memory/capacity, full joint corpus, judge calls, baseline repeatability, GRPO update, isolated MBPP execution, durable cloud lifecycle/independent backup and spending evidence remain real launch requirements. Mandatory statistical validation must pass its frozen criteria; a failed report blocks confirmatory launch. No GPU fit, provider call or cloud durability success is claimed by this handoff.

The SFT/cache implementation is frozen for coordinator integration. Root orchestration ownership remains released. Further edits require coordination so final source hashes and tests stay synchronized.
