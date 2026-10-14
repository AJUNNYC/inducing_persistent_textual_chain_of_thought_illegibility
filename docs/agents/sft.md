# sft.md

## Current status: all ten 14B ROT13 fits complete; GPQA long-profile evaluation next

All ten **14B ROT13 SFT levels are complete**, with final adapters, recovery states and results verified in Google Drive. The nine fresh levels 1–9 each reached 625 updates under scientific source `9de8813031da737453f491723a10b305679dc42d`; retained level 10 keeps source `2ea4fb9a7737cff806226366e73743eec5935fad`. The priority queue was observed complete on 14 October 2026, with zero pending or failed publications and its GPU idle. Each independent fit used all 10,000 certified training examples once, a 625-update cosine horizon and 19 warmup updates. KaziText level 10 remains paused at 444/625 with verified full recovery state and no automatic restart. All forty prepared condition datasets and historical 7B results remain available; their availability does not imply completed fits. The next selected work is GPQA Diamond for the untouched 14B base plus all ten ROT13 levels, using the separate `long_32000` evaluation profile. Training, data, recipe and original four-domain evaluation budgets are unchanged. See [the campaign guide](../gpqa_14b_rot13_campaign.md) and [the training record](../sft_completed_rot13_14b_level10.md).

The declared efficient baseline uses an unquantized frozen BF16 base and BF16 head, SDPA with efficient-kernel preference, direct full-gradient Cut Cross-Entropy (`cce_exact`) with FP32 loss and hidden-gradient accumulation, and plain AdamW with FP32 optimizer states. CCE is pinned to upstream commit `3de376c106a1916bc5e1b619f9c77c87a461ee1c`; `filter_eps=None`, `filter_e_grad=False` and `filter_c_grad=False` disable gradient filtering. No Qwen patch or full token-by-vocabulary logits matrix is used. Policy `published_bf16_lora_full_gradient_cce_v2` requests microbatch 2/accumulation 8. A non-updating duplicated-longest-example capacity guard tries 2/off, 1/off, 2/on, then 1/on, where on/off denotes activation checkpointing; only CUDA OOM permits fallback. The selected actual mode is recorded and cannot change during exact recovery.

Keep completion-only masks and per-example weighting, all seven LoRA targets (rank 8/alpha 16/dropout 0.05), learning rate `2e-4`, requested microbatch 2/accumulation 8/effective batch 16 (capacity fallback 1/16/16), seed 42, all 10,000/100 records, full 4096-token eligibility without truncation and verified Drive saves. Checkpoints occur at updates 100, 200, 300, 400, 500, 600 and final 625. Each update retains the same ordered 16 examples with per-example weight 1/16; execution grouping, padding, dropout draws and floating-point rounding may differ. No measured batching speedup or bitwise-equivalent outcome is claimed. This prospectively declared scientific baseline changes exposure, precision/head numerics, optimizer and execution; it does not prove identical final results. The original NF4 `legacy_two_epoch_v1` and intermediate NF4 `one_epoch_pilot_v1` remain explicit alternatives, with incompatible checkpoints and separate identities.

The user requested implementation without another optimization test or timing campaign. Normal native model/mask/gradient/capacity/full-state guards passed for the completed 14B ROT13 level-10 fit on RTX PRO 6000 Blackwell 96GB. Actual execution was microbatch 1 / accumulation 16 with activation checkpointing off. The recorded fit interval was 5,793.143751 seconds (96m33s); worker launch through completion was 6,322.565654 seconds (105m23s). At $2.99/hour these imply $4.81 and $5.25, respectively, calculated from runtime rather than invoices. That standalone L10 timing remains a dated measurement, not a bound for other levels or hardware. Archived PRO step-987 and A100 update-28/30 states remain in Drive, never warm starts for this recipe. One epoch removes 50% of nominal example passes versus the same-condition two-epoch recipe; no one-hour finish or measured hardware speedup is promised. Diagnostics use the 100 certified validation IDs and base companions; final test panels and original confirmatory thresholds remain reserved. No additional training after the completed 625-update cosine horizon is an exact continuation of the old 1,250-update schedule.

See the [completion, provenance and budget record](../sft_completed_rot13_14b_level10.md). The historical untouched 14B GPQA Diamond baseline remains **49/198 (24.75%)**, with 133 truncated and three malformed outputs under its 3,072-token cap; all 198 count. Its Drive-verified records remain separate from the new `long_32000` comparison. A new matching untouched baseline must run once before all ten completed ROT13 adapters, yielding 2,178 unique outputs. No trained-state GPQA result is claimed yet. Only GPQA receives the larger response/context allowance; GSM8K, MMLU, ARC-Easy and MBPP retain their existing 3,072-token measurement contract. The [GPQA contract](../modules/evaluation/supplemental_gpqa.md) and [campaign guide](../gpqa_14b_rot13_campaign.md) describe strict profile/cache isolation and verified Drive publication. KaziText remains suspended and unscheduled.

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
