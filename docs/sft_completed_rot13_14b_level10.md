# Completed 14B ROT13 SFT and historical level-10 budget

## Current priority: original one-pass evaluation first

The current priority is to **finish the original one-pass GPQA evaluations first**.
The matching base and original levels 1–6 and 10 are complete; original levels
7–9 remain. The frozen `5e6be1b` campaign has resumed with compatible cached
protocols/results and the existing published supervisor; do not regenerate the
baseline or mix the high-rate pilot into the original one-pass series.

The matched two-total-epoch 14B ROT13 levels 1–10 campaign remains authorized:
one extra pass from each own original full checkpoint, peak `5e-5`, existing
data/encoding and native thinking newline. Source
`a1962353a4301744dc6db5adafb0e6b212354679` launched at 21:32:26 UTC on 16 October
and level 1 began updates. The whole-optimizer-update pause requested at 21:54:29 UTC completed safely
at level 1 checkpoint `000064`: **64 added/689 cumulative updates**, with full state,
manifest and identity verified in Drive. Checkpoint hash:
`eb52d7e7d4a23f8f3ac482e1a5efce2d44e85767428d7a94263caa2bb465a780`.
The campaign and fit are **PAUSED**, and GPU handover followed verification.
The existing supervisor resumed original GPQA at 21:56:31.846611 UTC; original
**level 7 was RUNNING at 21:56:48.129439 UTC** (supervisor 29664/campaign 29665),
using published operations `2a0f7e2` and scientific source `5e6be1b`. Two existing
recoveries remain recorded. Complete base and L1–6 states were revalidated without
inference; original L10 retains its complete 198-question receipt and will be skipped
when visited. Training stays paused and **no automatic training resume is
installed** after evaluation. Original endpoints and
the completed high-rate L10 pilot stay separate. See the
[continuation policy/API/commands](modules/training/sft/sft_training.md#planned-matched-two-total-epoch-campaign-levels-110)
and [worker plan](sft_multi_gpu_execution_plan.md).

## Current completion and evaluation handoff

All ten 14B ROT13 levels are complete with verified Drive adapters, full state and fit results. The nine fresh levels 1–9 each reached 625 updates under scientific source `9de8813031da737453f491723a10b305679dc42d`; retained level 10 preserves source `2ea4fb9a7737cff806226366e73743eec5935fad`. KaziText level 10 remains safely paused at 444/625 with verified recovery state and no automatic restart. No other KaziText fit or 7B fit is newly completed by this queue. The priority queue was observed complete on 14 October 2026. The original [GPQA campaign](gpqa_14b_rot13_campaign.md) retains its 32,000-token baseline and completed original-adapter results. Its remaining original levels 7-9 are the current priority. The sections below retain the dated level-10 measurement, original budget forecasts and subsequent priority change.

On 16 October, the separate L10 extra epoch completed 625 additional updates
(1,250 cumulative), 10,000 additional examples (20,000 cumulative) and 17,996,003
tokens. Validation loss was 0.6151581262983382 on 100 cases. The final adapter,
full state and 26 required Drive artifacts are verified; the trainer exited
with code zero. Fit time was 17,470.2693 seconds (4h51m), about three times the
original 96.6-minute fit; the slowdown's cause is unknown. Training source
`3a89691` and the separate endpoint remain unchanged. Exposure, schedule and
newline alignment changed jointly.

The completed high-rate L10 extra-epoch pilot's frozen **198-question/32,000-token
GPQA evaluation is COMPLETE**, with native final artifacts and Drive verification
recorded on **16 October 2026 at 21:30:25 UTC**. All 198 questions remain in the
strict denominator:

| State, same 198 questions | Frozen scorer correct | Accuracy |
| --- | ---: | ---: |
| Matching untouched base | 99/198 | 50.00% |
| Original one-epoch L10 | 98/198 | 49.49% |
| High-rate extra-epoch L10 pilot | 52/198 | 26.26% |

The pilot has **52 correct, 44 parsed wrong, 9 malformed and 93 truncated**
outputs. The fixed 73-marker screen is positive for **168/198** responses;
**34/168** marker-positive responses are strictly correct. Marker presence
establishes occurrence, not encoding coverage, reasoning validity or formal L10
compliance. No all-198 human visible-choice or wordfreq analysis is claimed.
The 42-request and first-128 manual/contextual/lexical findings below remain
historical subset diagnostics and must not be extrapolated to all 198.

Frozen evaluator `5e6be1b0848e0e88725323591baddef37d69d8ae`, protocol
`57898f795997a55ebfce68cc68d2cc155312e6cd138b4bb2a927fe6cba863f9b`,
native prompts/options/item seeds, sampling, strict scorer and 32,000-token cap
were retained. Completion receipt SHA-256:
`e7171ac99e17f245a3257241a9c08fc1f5a09ffdde808b50844a4e0c6ee0cbfd`.
The private final responses/scores bundle SHA-256 is
`d59f1b16a990a42f71fb574f0b23aca271da5bbb5a50f8b5a9651ac3c555b1f9`;
validation SHA-256 is
`b41aa6f48fa6000f3dd51bc68615bc54f3b6887f048cb4e6a17f19be75016136`.
The [10-file final diagnostics and launch bundle](https://drive.google.com/drive/folders/1qZCc7iu3nT98yaDrbYiN19ju-TrMhhd_)
is Drive-verified. Licensed prompts, responses and token values remain private.

The pilot combines extra exposure, cosine restart at peak `2e-4` and native
thinking newline; these outcomes do not isolate an epoch effect. Its correctness
is lower than both matching controls under the frozen scorer. Original
construction criteria and four-domain evaluation budgets remain unchanged.

Historical first128 manual/contextual and wordfreq findings are retained in the
[diagnosis](sft_level10_training_diagnosis.md); their 56.50% lexical ROT13-only
share and 48.05% repeated-block sensitivity do not describe all198 responses.

## Condition-specific completion: 12 October 2026

`rot13_14b_level_10` completed **625 optimizer updates and 10,000 example presentations** under `efficient_one_epoch_v1` and `published_bf16_lora_full_gradient_cce_v2`. The actual GPU was NVIDIA RTX PRO 6000 Blackwell Server Edition with 101,975,851,008 reported VRAM bytes. Native startup selected **physical microbatch 1, accumulation 16, effective batch 16, activation checkpointing off** after the requested microbatch 2 failed with CUDA OOM. The accepted mode remained fixed throughout the fit.

The native completion-mask/adapter-gradient, non-updating capacity, and actual initialization checkpoint save/load guards passed. The longest training example was 4,095 tokens; mean training length was 1,798.6003 tokens. These observations certify this condition on this runtime, not every future condition or GPU. The full test suite and an optimization comparison campaign were not rerun for the launch.

The final validation loss was **0.7635798774 over the 100 certified validation examples**. This is a loss diagnostic, not proof of encoding efficacy, capability retention, opacity, or construction success. Final capability/legibility measurements remain outstanding.

| Immutable identity | Value |
| --- | --- |
| Executed worker source | `2ea4fb9a7737cff806226366e73743eec5935fad` |
| Consumer configuration | `6f7e50488aa8e8379dfb7f303bc790cef38493b1a260be3b27e5e8d13ad666ec` |
| Handoff certificate | `a34e70567bb2c65b3ce3de45624a075b8818cd51ed8c62f954ae14b6edeb610c` |
| Condition dataset | `7bce352922c2a205046ecd600f118802f8cab8360f2cf744024ef94e74ebeba4` |
| Base/tokenizer revision | `1df8507178afcc1bef68cd8c393f61a886323761` |
| Final adapter manifest | `0921af25d94fd23b5d748a5b5e5e4d74f0ab57542a0946c10cd493380ab8b9aa` |
| Final checkpoint state | `99912dfd758442f1e91d3a92a8aa31121ecda8bc1828aa2a2a03f82ecc6c0175` |

The [public completion receipt](evidence/sft_rot13_14b_level10_completion.json) and [native runtime evidence](evidence/sft_rot13_14b_level10_runtime_evidence.json) are byte-preserved exports from the actual run. The receipt reports worker `COMPLETE` and Drive `VERIFIED`. The [condition's Drive folder](https://drive.google.com/drive/folders/1dnkVaKj7eBZhWDOUTLf62inXRCzeo2R0) contains the final adapter, full recovery checkpoint, and result report; all permanent outputs were verified before completion. No credential material or licensed corpus rows are included in the public evidence.

## Measured runtime and calculated cost

The reported fit interval was **5,793.143751 seconds**, or **1 hour 36 minutes 33 seconds**. Worker launch at 08:31:52.712399 UTC through recorded completion at 10:17:15.278053 UTC was **6,322.565654 seconds**, or **1 hour 45 minutes 23 seconds**. These correspond to 04:31–06:17 on 12 October in America/New_York. The fit interval should not be presented as a separately measured pure forward/backward interval; the broader interval includes startup and final publication.

At the user's quoted **$2.99/hour**, the corresponding calculated costs are **$4.81 for the fit interval** and **$5.25 for worker launch through completion**. The $4.81 is contained in $5.25, not added to it. These are duration-times-rate calculations, not account invoices. Environment installation before launch, idle time after completion, and historical attempts are additional and not reconciled by these receipts. Worker exit releases GPU memory; only the account operator stopping the Pod stops provider billing.

## New SFT authorization: 13 October 2026

The user has **$240 authorized for SFT**. This replaces the earlier provisional request for $250; it is new authorization, not a claim that $240 has already been billed. The [historical prospective 19-fit 14B campaign plan](evidence/authorized_sft_14b_campaign_2026_10_13.json) explicitly records `PLANNED_NOT_LAUNCHED`, excludes the completed adapter, and preserves the finalized recipe. It is not an active launch certificate or running queue. The plan was prospective at its creation; initial execution was separately recorded in the [live campaign receipt](evidence/sft_14b_campaign_training_active_2026_10_13.json). The scientific worker source is `9de8813031da737453f491723a10b305679dc42d`, independently of this later documentation commit. RL and final evaluation are outside these SFT projections.

The user subsequently narrowed active training to **14B ROT13 and 14B KaziText, each across ten levels: 20 conditions**. The completed adapter is retained, leaving **19 selected fits**. All 40 prepared datasets and historical 7B results remain available, but 7B fits are outside this active queue. The [previous 39-fit plan](evidence/authorized_sft_campaign_2026_10_13.json) is retained unchanged as superseded historical evidence. At the same per-fit duration as this single measured 14B ROT13 condition:

| Conditional planning scope | Fit interval only | Worker interval including observed overhead |
| --- | --- | --- |
| All 20 selected 14B fits | 32.18 GPU-hours / $96.23 | 35.13 GPU-hours / $105.02 |
| Remaining 19 selected 14B fits | 30.57 GPU-hours / $91.42 | 33.37 GPU-hours / $99.77 |

The remaining 19 worker projection (approximately $100) leaves approximately **$140.23** of the new $240 authorization for setup, runtime variation, and idle time. It is **not a guaranteed bound**: KaziText, other levels, other GPUs, and different startup batching may take different time. The $240 authorization remains recorded. The historical $2.99/hour scenario corresponds to 80.27 billed GPU-hours, including chargeable setup/idle; it is not an imposed rate or automated runtime cutoff. The current execution instruction is to proceed without setting a new budget rate. Multiple GPUs reduce elapsed time only if work is balanced; combined GPU-hours and cost remain chargeable.

The historical 150-hour planning forecast and extrapolations from a two-update old-recipe benchmark are not actual spending. Do not present old approximately $740 or $2,000 conversation figures as measured invoices or as established estimates for this finalized recipe.

## Historical 13 October priority: ROT13 levels 1–9; KaziText on hold

The completedROT13 14B L10 adapter keeps its original 2ea4fb9/config 6f7e identity. The new user instruction prioritizes **nine fresh 14B ROT13 fits, levels 1–9 ascending**. The broader prepared 14B scope remains 20 conditions; source 9de8813/config 44e9/certificate/data/recipe are unchanged. KaziText 14B L10 was safely paused at **444/625 updates, 7,104 example presentations**, with its full checkpoint verified in Drive at 13 Oct 21:58 UTC. Its checkpoint state hash is `570620362ba00923e3d5b63da10e8d7e8b2711ce4149ec9c4ea9a21ec468ff17`. This is a complete recovery checkpoint, not a completed fit. No KaziText fit is scheduled automatically.

The [new priority/pause receipt](evidence/rot13_14b_priority_execution_2026_10_13.json) records the nine-job order and retained/paused states. A private operational lifecycle helper calls the already published [single-fit CLI](../scripts/run_certified_sft.py) with explicit `--family rot13 --model 14b --level` selection for each level; it does not change scientific code, configuration, data, recipe or checkpoints. The first wrapper failed before GPU initialization. Recovery reconstructed the exact 496-byte coordinator pause payload at its original SHA, changed only the registry's `local_path`, then performed real Drive verification. It did not mark an unverified payload successful, reset retry budgets, delete records or change immutable identity. The [public reconciliation record](evidence/rot13_14b_pause_status_reconciliation_2026_10_13.json) retains the failure and recovery evidence.

At the historical startup observation, priority wrapper 6682/child 6684 started level 1 at 13 Oct 22:03:36 UTC. Native mask/gradient/capacity and actual checkpoint 0 recovery passed. The full initialization state is verified in Drive; GPU activity was 98% at 22:05:44 UTC and 94% at 22:06:10 UTC. ROT13 L1 was **training** at that observation, using microbatch 1/accumulation 16 with checkpointing off. Only startup checkpoint 0 had been observed then, not an exact completed-update count. Those dated observations are superseded for current progress by the completed nine-fit queue. The [sealed active-priority receipt](evidence/sft_14b_rot13_priority_active_2026_10_13.json) records three busyGPU observations across62.75 seconds and preserves the executedsource9de identity. Ordinary per-fit native capacity/mask/recovery safeguards remain required; the scientific suite and GPQA baseline are not repeated. The prior 19-fit ordering and active-KaziText receipt remain historical observations, not the current automatic queue. Future authorization is needed to reactivate KaziText.

The actual GPQA baseline remains49/198 and Drive-verified,with133 truncated/3 malformed under the unchanged3,072-token cap. Its previously verified outputs are reused rather than regenerated for the priority change. The $240 SFT authorization remains recorded,with no imposednew hourlyrate or rate-based cutoff. Explicit deadlines,safe user pauses and verified checkpoint barriers remain separate controls.

## Keeping the completed level without a training confound

Do not retrain or extend this level solely because funding or an evaluation panel changed. Its immutable source, recipe, dataset, seed, actual execution mode, and final endpoint identify a valid completed fit. Future fits should use the same declared scientific recipe, start from their own untouched base, and record condition-specific runtime choices. Later documentation or evaluation-only changes do not retroactively change the completed worker source. Any actual training-algorithm change must receive its own identity and comparability assessment; an old full state cannot simply be relabelled.

Adding GPQA does not require another epoch or a replacement adapter. If multiple training seeds are later needed, declare and retain them as independent replicates rather than discarding this completed result after seeing evaluation outcomes. No GPQA result was used to select or extend this original epoch-one endpoint. The separately identified 16 October endpoint follows inspected GPQA results and remains an exploratory comparison.

## Historical 13 October GPQA baseline and evaluation preparation

On 13 October the user clarified that GPQA is **appended** to GSM8K, MMLU-Redux-2.0, ARC-Easy, and sanitized MBPP; it does not replace those domains. The [José et al. reference implementation](https://github.com/Jozdien/cot_legibility/blob/89195c70288b4b141460447f5fde02f67b38b043/src/inference/datasets.py#L17) maps `gpqa` to **`gpqa_diamond` (198 questions)**. The official acquisition revision is [`Idavidrein/gpqa`, `83022cefff930aea54f654c0b282e74b9eeda5c6`](https://huggingface.co/datasets/Idavidrein/gpqa/tree/83022cefff930aea54f654c0b282e74b9eeda5c6). The official Diamond source CSV has been downloaded through the standard Hugging Face SDK using the existing access; no new user authentication is needed. Its size is 1,373,492 bytes and its SHA-256 is `41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305`. The actual preparation CLI produced the complete 198-question panel with hash `322e4335322a93adc18dc0e22e982bf6bbcd1a07223ccf1b08e25dfb7a5076de`. The [preparation receipt](evidence/gpqa_preparation_receipt.json) reports `PREPARED`, eleven targeted checks passed, and zero trained states evaluated. The actual untouched 14B GPQA seed 42 baseline is now complete and Drive-verified: **49/198 = 24.75%**. It contains **133 truncated outputs and 3 malformed answers under the unchanged 3,072-token cap**, plus 13 parsed incorrect answers. All 198 remain in the headline denominator; this is budget-limited output/scorer performance, not unrestricted GPQA accuracy. The [sanitized result receipt](evidence/gpqa_14b_baseline_completion_2026_10_13.json) preserves the worker identity and first 128 timeout-cache recovery. Trained 14B GPQA evaluation remains unrun and must retain the same panel/budget/scorer. Existing four-domain base measurements are retained, while their deferred judge assessments are still required for full evaluation. See the [variant-source evidence](evidence/gpqa_variant_source.json) and [supplemental GPQA procedure](modules/evaluation/supplemental_gpqa.md).

This matches the reference's dataset variant, not its complete measurement pipeline: the reference prompt omits choices and uses language-model correctness grading, while this project's planned supplement uses four-option native final-field scoring. Retain the prepared panel’s acquisition, IDs and choice order. Before generation, freeze the exact comparison states, prompt/extraction, inference budget and scoring rules; use the same frozen questions and protocol for untouched bases and trained states. Audit detectable overlap without silently changing the already certified SFT data or claiming retroactive exclusion.

GPQA Diamond is a supplemental descriptive capability stress test. Existing four-domain confirmatory thresholds and windows remain unchanged unless a separate scientific amendment defines new criteria. Adding this evaluation panel does not require retraining the completed adapter and cannot establish the original minimum-three-contiguous-level construction window from one trained level.

## Historical 13 October 14B baseline and analysis scope

The [14B baseline report](14b_baseline_results.md) preserves the actual four-domain seed-42 capability results and passing three-seed repeatability evidence. Those outputs are reused, not regenerated because active SFT now selects only 14B. Baseline legibility judgments remain deferred; GPQA requires its own untouched 14B baseline paired with trained-state evaluation on the prepared panel.

At the historical13October snapshot, the user selected **20 SFT conditions for14B only**, with19 fits then remaining; it does not automatically rewrite the entire confirmatory workflow. Modular single-state evaluation and a separately frozen 14B GPQA base/adapter comparison can use this scope. The full main campaign and analysis still encode the original 62-state accounting, both model sizes, RL snapshots, and 216-window reporting. A scope-specific aggregation/reporting decision remains pending before claiming a complete 14B-only confirmatory campaign. The original criteria remain unchanged; unrun states are missing/indeterminate, not successful. That historical scope did not require7B training or repeated calibration/full tests/GPU benchmarks. Current original one-pass GPQA priority and separately paused continuation are recorded above.
