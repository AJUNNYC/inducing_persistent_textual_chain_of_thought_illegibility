# Efficient SFT deployment and available condition ownership

## Current priority: original one-pass GPQA levels 7–9 first

Complete the original **one-pass** 14B ROT13 GPQA evaluation before further
training. The matching `long_32000` baseline and original levels 1–6 and 10
are complete; original levels 7–9 are next. Reuse their original adapters and
compatible frozen outputs. The published supervisor launched at
21:56:31.846611 UTC, with original level 7 confirmed running at 21:56:48.129439
UTC. Scientific evaluator `5e6be1b` and operational supervisor `2a0f7e2` retain
the two recorded recoveries. Base and levels 1–6 were revalidated without
inference; original L10's completed receipt is retained. Training does not
restart automatically afterward. Current status is
`ONE_PASS_GPQA_PRIORITY_RUNNING_LEVEL7_SFT_CONTINUATION_PAUSED64`.

The separate high-rate L10 pilot's frozen 198-question evaluation is complete:
base **99/198 (50.00%)**, original one-pass L10 **98/198 (49.49%)**, pilot
**52/198 (26.26%)**. Pilot categories are 52 correct, 44 parsed wrong, 9 malformed
and 93 truncated. The fixed 73-word detector finds 168/198 marker-positive
responses, including 34 frozen-correct; this establishes ROT13 presence only.
Historical first-128 manual and wordfreq analyses retain their prefix scope.

The low-learning-rate v2 training campaign launched at **21:32:26 UTC on
16 October** from `a1962353a4301744dc6db5adafb0e6b212354679`, with level-1
updates observed. A whole-update `SIGTERM` pause was requested at **21:54:29 UTC**.
The campaign and fit are **`PAUSED` at level-1 checkpoint `000064`**, with
64 added/689 cumulative updates. Full state, manifest and identity are
Drive-verified; checkpoint hash
`eb52d7e7d4a23f8f3ac482e1a5efce2d44e85767428d7a94263caa2bb465a780`.
The GPU was released before evaluation resumed. Future all-ten training remains
authorized; no automatic SFT resume is installed.

## Authorized matched two-total-epoch ROT13 continuation

The authorized new campaign adds **one pass to each original 14B ROT13 level
1–10 checkpoint**: 625 added/1,250 cumulative updates per level, peak `5e-5`,
19 nonzero warmup updates then cosine, and native thinking newline. Reuse each
level's unchanged 10,000/100 records and its own verified original adapter,
AdamW and RNG. Preserve originals and the completed high-rate L10 pilot in
separate namespaces; the pilot cannot replace this campaign's L10 endpoint.
The [SFT module](modules/training/sft/sft_training.md#planned-matched-two-total-epoch-campaign-levels-110)
controls policy/API and the direct validation command.

One GPU owns a serial level queue. Two GPUs may own disjoint level shards, each
running one level at a time with separate outputs and verified completion
receipts. This is independent level parallelism, not splitting a model or
changing effective batch. No queue advances past an unverified endpoint. The
high-rate L10 pilot evaluation is complete; original one-pass GPQA levels 7–9
take priority, and future matched GPQA requires all ten verified v2 endpoints
plus frozen identity review before launch. Do not regenerate existing compatible
baseline or one-epoch outputs.

The measured original L10 fit was 96.6 minutes; the separate high-rate pilot
was 4h51m. The slowdown's cause remains unknown. Neither measurement establishes
new per-level times, a campaign finish time or a two-GPU speedup.

## Restore and plan the selected levels

For later selected training, run from executed implementation `a1962353a4301744dc6db5adafb0e6b212354679`,
using its exact checkout or byte-verified export and matching
`SOURCE_CODE_COMMIT` as described in the [SFT commands](modules/training/sft/sft_training.md#planned-matched-two-total-epoch-campaign-levels-110).
Later documentation revisions do not change this execution identity.

The original-parent inventory has 141 sealed metadata-verified artifacts across
all ten levels. Inventory preparation alone did not establish payload restoration
or worker startup; the later staging and launch observations are distinct.
The inventory identity is retained in [source provenance](../spec/source_provenance.json).

Current Pod staging is now complete: all 141 files were downloaded and fully
verified; restored manifest `12a4420192b129c45637772d8a6d04bf21d3b7bdb0ba66a326446329493c36a5`.
The published campaign command produced a valid all-ten plan without GPU work.
The v2 worker subsequently launched and paused at verified checkpoint `000064`
as described above. Original one-pass evaluation now takes priority.

The private sealed inventory binds each original checkpoint and exact dataset
files. The restoration command verifies Drive metadata and downloaded bytes;
it performs no GPU training:

```bash
python scripts/restore_sft_level_continuation_inputs.py \
  --inventory /workspace/private/original_rot13_parents.json \
  --destination-root /workspace/continuation_inputs --workers 2
```

Successful restoration writes `sft_level_continuation_inputs_manifest.json`
with status `VERIFIED`. Once all payloads are present, adding `--validate-only`
rechecks those local bytes without network, writes or GPU work; it is not an
inventory-only dry run before download. Then inspect a serial all-level plan
without starting work:

```bash
python scripts/run_sft_level_continuation_campaign.py \
  --inputs-manifest /workspace/continuation_inputs/sft_level_continuation_inputs_manifest.json \
  --levels 1,2,3,4,5,6,7,8,9,10 --additional-epochs 1 --learning-rate 5e-5 \
  --output /workspace/continuation_campaign/all_levels \
  --model-cache /workspace/models --device cuda:0 --plan-only
```

For two GPUs, inspect two separate plans with **odd levels**
`--levels 1,3,5,7,9` and **even levels** `--levels 2,4,6,8,10`, distinct output
roots, and their assigned device on each worker. This spreads high levels
between shards; it does not establish equal runtimes. Any disjoint partition
is allowed. Keep all other policy flags identical. `--python` may name the pinned GPU
interpreter. Removing `--plan-only` starts only that selected serial queue after
a later selected training resume. These examples retain the reusable commands;
the current evaluation-first priority does not schedule a training restart.

Each queue has explicit `PLANNED`, `RUNNING`, `PAUSED`, `FAILED` or `COMPLETE`
status. A failed or paused level prevents later levels from starting. Signals
request a whole-update safe pause and verified Drive recovery. A completed job
is skipped only after its original parent/policy, full state, adapter, sealed
receipt and live Drive copy verify. Once every selected level completes,
`evaluation_endpoints.json` lists that shard's endpoints; it does not launch
GPQA or establish that other shards are complete. The all-ten evaluation barrier
must combine and verify both disjoint shards.

## Historical original one-epoch completion and GPQA handoff, 14 October

All ten **14B ROT13 SFT levels are complete**, with final adapters, recovery states and results verified in Google Drive. The nine fresh levels 1–9 each reached 625 updates under scientific source `9de8813031da737453f491723a10b305679dc42d`; retained level 10 keeps source `2ea4fb9a7737cff806226366e73743eec5935fad`. The priority queue was observed complete on 14 October 2026, with zero pending or failed publications and its GPU idle. Each independent fit used all 10,000 certified training examples once, a 625-update cosine horizon and 19 warmup updates. KaziText level 10 remains paused at 444/625 with verified full recovery state and no automatic restart. All forty prepared condition datasets and historical 7B results remain available; their availability does not imply completed fits. At the 14 October handoff, the next selected work was GPQA Diamond for the untouched 14B base plus all ten ROT13 levels, using the separate `long_32000` evaluation profile. Training, data, recipe and original four-domain evaluation budgets are unchanged. See [the campaign guide](gpqa_14b_rot13_campaign.md) and [the training record](sft_completed_rot13_14b_level10.md).

The declared efficient baseline uses an unquantized frozen BF16 base and BF16 head, SDPA with efficient-kernel preference, direct full-gradient Cut Cross-Entropy (`cce_exact`) with FP32 loss and hidden-gradient accumulation, and plain AdamW with FP32 optimizer states. CCE is pinned to upstream commit `3de376c106a1916bc5e1b619f9c77c87a461ee1c`; `filter_eps=None`, `filter_e_grad=False` and `filter_c_grad=False` disable gradient filtering. No Qwen patch or full token-by-vocabulary logits matrix is used. Policy `published_bf16_lora_full_gradient_cce_v2` requests microbatch 2/accumulation 8. A non-updating duplicated-longest-example capacity guard tries 2/off, 1/off, 2/on, then 1/on, where on/off denotes activation checkpointing; only CUDA OOM permits fallback. The selected actual mode is recorded and cannot change during exact recovery.

Keep completion-only masks and per-example weighting, all seven LoRA targets (rank 8/alpha 16/dropout 0.05), learning rate `2e-4`, requested microbatch 2/accumulation 8/effective batch 16 (capacity fallback 1/16/16), seed 42, all 10,000/100 records, full 4096-token eligibility without truncation and verified Drive saves. Checkpoints occur at updates 100, 200, 300, 400, 500, 600 and final 625. Each update retains the same ordered 16 examples with per-example weight 1/16; execution grouping, padding, dropout draws and floating-point rounding may differ. No measured batching speedup or bitwise-equivalent outcome is claimed. This prospectively declared scientific baseline changes exposure, precision/head numerics, optimizer and execution; it does not prove identical final results. The original NF4 `legacy_two_epoch_v1` and intermediate NF4 `one_epoch_pilot_v1` remain explicit alternatives, with incompatible checkpoints and separate identities.

The user requested implementation without another optimization test or timing campaign. Normal native model/mask/gradient/capacity/full-state guards passed for the completed 14B ROT13 level-10 fit on RTX PRO 6000 Blackwell 96GB. Actual execution was microbatch 1 / accumulation 16 with activation checkpointing off. The recorded fit interval was 5,793.143751 seconds (96m33s); worker launch through completion was 6,322.565654 seconds (105m23s). At $2.99/hour these imply $4.81 and $5.25, respectively, calculated from runtime rather than invoices. That standalone L10 timing remains a dated measurement, not a bound for other levels or hardware. Archived PRO step-987 and A100 update-28/30 states remain in Drive, never warm starts for this recipe. One epoch removes 50% of nominal example passes versus the same-condition two-epoch recipe; no one-hour finish or measured hardware speedup is promised. Diagnostics use the 100 certified validation IDs and base companions; final test panels and original confirmatory thresholds remain reserved. No additional training after the completed 625-update cosine horizon is an exact continuation of the old 1,250-update schedule.

See the [completion, provenance and budget record](sft_completed_rot13_14b_level10.md). The historical untouched 14B GPQA Diamond baseline remains **49/198 (24.75%)**, with 133 truncated and three malformed outputs under its 3,072-token cap; all 198 count. Its Drive-verified records remain separate from the new `long_32000` comparison. The original launch plan required a matching untouched baseline once before all ten completed ROT13 adapters, yielding 2,178 unique outputs. The completed baseline and original-adapter outputs are now preserved; only original levels 7–9 remain to evaluate. Only GPQA receives the larger response/context allowance; GSM8K, MMLU, ARC-Easy and MBPP retain their existing 3,072-token measurement contract. The [GPQA contract](modules/evaluation/supplemental_gpqa.md) and [campaign guide](gpqa_14b_rot13_campaign.md) describe strict profile/cache isolation and verified Drive publication. KaziText remains suspended and unscheduled.

## Historical fresh-fit selection and independent workers

The original selected queue contained fresh 14B ROT13 level 10 and 14B KaziText level 10, each from the untouched pinned base. ROT13 level 10 is now complete and retained under its immutable source/configuration identity; it need not be retrained merely to add evaluation. The unchanged inventory has forty conditions; the broader prepared scope is 20 independent 14B fits across both families and ten levels. All ten ROT13 fits are now complete; ten KaziText fits are uncompleted, including the safely paused level 10. Prepared 7B data/history are retained outside this scope. New workers must own disjoint remaining fits and must not silently repeat or extend the completed 625-update fit.

Each selected fit starts fresh with separate adapter, optimizer, scheduler, RNG/sampler and immutable Drive namespace. Separate GPUs can run independent selected fits concurrently; completion time is the longest assigned queue, not the sum of concurrent durations. No resource allocation or restart is authorized by this document alone. The existing user-owned CPU Colab session remains unaffected.

## Historical handoff: ten original ROT13 adapters complete

All ten 14B ROT13 levels are complete with verified Drive adapters, full state and fit results. The nine fresh levels 1–9 each reached 625 updates under scientific source `9de8813031da737453f491723a10b305679dc42d`; retained level 10 preserves source `2ea4fb9a7737cff806226366e73743eec5935fad`. KaziText level 10 remains safely paused at 444/625 with verified recovery state and no automatic restart. No other KaziText fit or 7B fit is newly completed by this queue.

The nine-level priority queue was observed complete on 14 October with all final publications verified, no pending/failed artifacts and its GPU idle. Its lifecycle helper called the existing certified single-fit CLI without changing science. The original 19-fit and nine-fit plans and dated startup receipts remain historical evidence, not current running status.

At the 14 October handoff, the next selected work was [GPQA Diamond baseline plus all ten completed ROT13 levels](gpqa_14b_rot13_campaign.md), using the separate `long_32000` profile. No new SFT fit or KaziText continuation is scheduled by that evaluation. The historical short-budget baseline cannot substitute for the new matching long-profile baseline.
## Declared published execution

The newly declared `efficient_one_epoch_v1` scientific baseline uses BF16 LoRA, SDPA, direct exact CCE and plain AdamW. It is based on published methods, not conditional activation of the historical custom NF4 kernels. Fresh startup selects 2/off, 1/off, 2/on, then 1/on using a duplicated-longest-example capacity guard, preserving effective batch 16 and full targets. The selected microbatch/accumulation/recomputation mode is fixed during recovery; it is not changed halfway through a fit. Actual model/mask/gradient/full-state guards run during startup before updates. No comparative optimization test or benchmark campaign is scheduled at the user's request.

Install the hash-locked base plus the immutable upstream CCE dependency with `scripts/setup_environment.py --profile gpu-published --skip-tests`. The lock is `requirements_sft_published.lock`; the pinned revision, full-gradient flags, head/loss precisions and actual attention/recomputation settings are part of runtime provenance. SDPA preference does not claim an observed fused kernel.

Archived source a6e3295 accepted a different FP32-head custom loss only for 7B ROT13 level 1 on PRO 6000 96GB. The prior A100 candidates never completed acceptance. Both assignments ended at zero compute units; full states at updates 28/30 were saved in Drive first. The PRO step-987 state is also archived. None is a warm start or proof of the efficient recipe's speed, memory or final efficacy.

## Runtime and cost boundaries

One epoch halves nominal example presentations and updates relative to the same-condition two-epoch recipe. Numerical and runtime changes may reduce additional costs, but the standalone level-10 receipt is a condition-specific timing measurement, not a hardware comparison or bound for the whole matrix. The uploaded 58.7-minute notebook motivates short training and BF16/SDPA execution; it used prompt-plus-response loss and a different length distribution. It does not establish a one-hour finish or validated equivalent encoding/capability.

The historical pre-priority projection for all 20 prepared 14B conditions was: 32.18 fit-hours / $96.23, or 35.13 worker-hours / $105.02 at $2.99/hour if all match the completed ROT13 L10 runtime. At that planning stage, retaining level 10 left 19 fits: 30.57 fit-hours / $91.42, or 33.37 worker-hours / $99.77 (approximately $100). These are one-condition extrapolations, not measured bounds. The newly authorized SFT ceiling is $240; these estimates are not measured bounds or invoices and exclude installation and idle time. Track actual runtime and observed spending; do not turn the historical rate projection into an imposed rate-based cutoff. RL and final evaluation are outside this SFT estimate. See the [budget record](sft_completed_rot13_14b_level10.md).

## Durable state and interpretation

Use the existing [central Drive project](https://drive.google.com/drive/folders/176a8Tm6l4WhfakY_0nCOo4ICeT7KPSmx); local caches are disposable. Save full state at updates 100, 200, 300, 400, 500, 600 and final 625, and verify actual bytes before retention or completion advances. Preserve original source/data identities alongside the new consumer recipe/configuration, precision, dependency and hardware records. Credentials stay private.

Diagnostics use the 100 certified validation IDs and untouched-base companions. Keep final test panels reserved and original confirmatory thresholds unchanged. The single completed level-10 fit cannot establish untrained levels, an intermediate response curve or the original minimum-three-contiguous-level construction window. Extra epochs after final 625 need a separately declared continuation or fresh legacy fit; they cannot inherit the old 1,250-update schedule by relabelling.
