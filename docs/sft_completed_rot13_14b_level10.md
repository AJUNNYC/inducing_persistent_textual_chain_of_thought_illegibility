# Completed 14B ROT13 level 10 and remaining SFT budget

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

The user has **$240 authorized for SFT**. This replaces the earlier provisional request for $250; it is new authorization, not a claim that $240 has already been billed. The [current prospective 19-fit 14B campaign plan](evidence/authorized_sft_14b_campaign_2026_10_13.json) explicitly records `PLANNED_NOT_LAUNCHED`, excludes the completed adapter, and preserves the finalized recipe. It is not an active launch certificate or running queue. The plan was prospective at its creation; actual execution is now separately recorded in the [live campaign receipt](evidence/sft_14b_campaign_training_active_2026_10_13.json). The scientific worker source is `9de8813031da737453f491723a10b305679dc42d`, independently of this later documentation commit. RL and final evaluation are outside these SFT projections.

The user subsequently narrowed active training to **14B ROT13 and 14B KaziText, each across ten levels: 20 conditions**. The completed adapter is retained, leaving **19 selected fits**. All 40 prepared datasets and historical 7B results remain available, but 7B fits are outside this active queue. The [previous 39-fit plan](evidence/authorized_sft_campaign_2026_10_13.json) is retained unchanged as superseded historical evidence. At the same per-fit duration as this single measured 14B ROT13 condition:

| Conditional planning scope | Fit interval only | Worker interval including observed overhead |
| --- | --- | --- |
| All 20 selected 14B fits | 32.18 GPU-hours / $96.23 | 35.13 GPU-hours / $105.02 |
| Remaining 19 selected 14B fits | 30.57 GPU-hours / $91.42 | 33.37 GPU-hours / $99.77 |

The remaining 19 worker projection (approximately $100) leaves approximately **$140.23** of the new $240 authorization for setup, runtime variation, and idle time. It is **not a guaranteed bound**: KaziText, other levels, other GPUs, and different startup batching may take different time. The $240 authorization remains recorded. The historical $2.99/hour scenario corresponds to 80.27 billed GPU-hours, including chargeable setup/idle; it is not an imposed rate or automated runtime cutoff. The current execution instruction is to proceed without setting a new budget rate. Multiple GPUs reduce elapsed time only if work is balanced; combined GPU-hours and cost remain chargeable.

The historical 150-hour planning forecast and extrapolations from a two-update old-recipe benchmark are not actual spending. Do not present old approximately $740 or $2,000 conversation figures as measured invoices or as established estimates for this finalized recipe.

## Baseline before the remaining SFT queue

Use the committed [supplemental GPQA CLI](../scripts/run_supplemental_gpqa.py) `freeze-base` command to bind only the untouched pinned 14B model to the prepared Diamond 198 panel; no trained adapter is required. Run that baseline first and verify all raw records, scores, the 198-item report and completion receipt in Google Drive. A locally written protocol or launched process is not a completed baseline. Wait for the baseline CLI to exit successfully and release GPU workers before invoking the training coordinator. Existing four-domain base outputs and repeatability evidence are retained; this new GPQA baseline is an additional measurement, not a repeat of those domains.

The committed [14B SFT coordinator](../scripts/run_14b_sft_campaign.py) gates training on the actual GPQA completion and Drive receipt, then follows this sequence:

1. Keep the completed14B ROT13 level 10 final adapter/full state under its original source/configuration identity; verify compatible bytes without retraining it.
2. Train fresh 14B KaziText level 10 from the untouched pinned base.
3. For each level 1 through 9 in ascending order, train fresh 14B ROT13 and then fresh 14B KaziText.

All 19 new fits retain the finalized 625-update, one-epoch recipe, disjoint output identities, and verified Drive recovery. One fit’s adapter is never the next level’s initialization. The comparison protocol for evaluating trained adapters remains separate; completing the baseline does not imply any trained-state GPQA result or complete confirmatory evaluation. The GPQA baseline completed 49/198 real outputs and passed the Drive barrier. The coordinator started 13 Oct 20:58:33 UTC; KaziText 14B L10 is training with native guards passed, actual 1 × 16 and activation checkpointing off. At 21:31:09 UTC, checkpoint 200/625 was verified in Drive; the 21:38:59 UTC receipt also confirms a live process and GPU activity. This is a dated progress snapshot, not per-update telemetry or fit completion. The remaining 18 new fits have not started in this snapshot, and the original ROT13 L10 completion remains unchanged.

The $240 SFT authorization remains recorded. The user instructed proceeding without imposing a budget rate: the coordinator does not infer a new hourly price or set a rate-based automatic cutoff. The $2.99/hour tables above are historical-rate planning scenarios, not launch parameters. Explicit worker deadlines and user-requested safe pauses remain separate controls; checkpoint/Drive barriers continue to apply.

## Keeping the completed level without a training confound

Do not retrain or extend this level solely because funding or an evaluation panel changed. Its immutable source, recipe, dataset, seed, actual execution mode, and final endpoint identify a valid completed fit. Future fits should use the same declared scientific recipe, start from their own untouched base, and record condition-specific runtime choices. Later documentation or evaluation-only changes do not retroactively change the completed worker source. Any actual training-algorithm change must receive its own identity and comparability assessment; an old full state cannot simply be relabelled.

Adding GPQA does not require another epoch or a replacement adapter. If multiple training seeds are later needed, declare and retain them as independent replicates rather than discarding this completed result after seeing evaluation outcomes. No GPQA result was used to select or extend this adapter.

## GPQA Diamond: untouched baseline complete, trained evaluation pending

On 13 October the user clarified that GPQA is **appended** to GSM8K, MMLU-Redux-2.0, ARC-Easy, and sanitized MBPP; it does not replace those domains. The [José et al. reference implementation](https://github.com/Jozdien/cot_legibility/blob/89195c70288b4b141460447f5fde02f67b38b043/src/inference/datasets.py#L17) maps `gpqa` to **`gpqa_diamond` (198 questions)**. The official acquisition revision is [`Idavidrein/gpqa`, `83022cefff930aea54f654c0b282e74b9eeda5c6`](https://huggingface.co/datasets/Idavidrein/gpqa/tree/83022cefff930aea54f654c0b282e74b9eeda5c6). The official Diamond source CSV has been downloaded through the standard Hugging Face SDK using the existing access; no new user authentication is needed. Its size is 1,373,492 bytes and its SHA-256 is `41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305`. The actual preparation CLI produced the complete 198-question panel with hash `322e4335322a93adc18dc0e22e982bf6bbcd1a07223ccf1b08e25dfb7a5076de`. The [preparation receipt](evidence/gpqa_preparation_receipt.json) reports `PREPARED`, eleven targeted checks passed, and zero trained states evaluated. The actual untouched 14B GPQA seed 42 baseline is now complete and Drive-verified: **49/198 = 24.75%**. It contains **133 truncated outputs and 3 malformed answers under the unchanged 3,072-token cap**, plus 13 parsed incorrect answers. All 198 remain in the headline denominator; this is budget-limited output/scorer performance, not unrestricted GPQA accuracy. The [sanitized result receipt](evidence/gpqa_14b_baseline_completion_2026_10_13.json) preserves the worker identity and first 128 timeout-cache recovery. Trained 14B GPQA evaluation remains unrun and must retain the same panel/budget/scorer. Existing four-domain base measurements are retained, while their deferred judge assessments are still required for full evaluation. See the [variant-source evidence](evidence/gpqa_variant_source.json) and [supplemental GPQA procedure](modules/evaluation/supplemental_gpqa.md).

This matches the reference's dataset variant, not its complete measurement pipeline: the reference prompt omits choices and uses language-model correctness grading, while this project's planned supplement uses four-option native final-field scoring. Retain the prepared panel’s acquisition, IDs and choice order. Before generation, freeze the exact comparison states, prompt/extraction, inference budget and scoring rules; use the same frozen questions and protocol for untouched bases and trained states. Audit detectable overlap without silently changing the already certified SFT data or claiming retroactive exclusion.

GPQA Diamond is a supplemental descriptive capability stress test. Existing four-domain confirmatory thresholds and windows remain unchanged unless a separate scientific amendment defines new criteria. Adding this evaluation panel does not require retraining the completed adapter and cannot establish the original minimum-three-contiguous-level construction window from one trained level.

## Existing 14B baseline and analysis scope

The [14B baseline report](14b_baseline_results.md) preserves the actual four-domain seed-42 capability results and passing three-seed repeatability evidence. Those outputs are reused, not regenerated because active SFT now selects only 14B. Baseline legibility judgments remain deferred; GPQA requires its own untouched 14B baseline paired with trained-state evaluation on the prepared panel.

The active change selects **20 SFT conditions for 14B only**, with 19 remaining fits; it does not automatically rewrite the entire confirmatory workflow. Modular single-state evaluation and a separately frozen 14B GPQA base/adapter comparison can use this scope. The full main campaign and analysis still encode the original 62-state accounting, both model sizes, RL snapshots, and 216-window reporting. A scope-specific aggregation/reporting decision remains pending before claiming a complete 14B-only confirmatory campaign. The original criteria remain unchanged; unrun states are missing/indeterminate, not successful. No 7B training is required by the current SFT queue, and this scope update does not request repeated calibration, full tests, GPU benchmarks, or evaluation runs.
