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

The user has **$240 authorized for SFT**. This replaces the earlier provisional request for $250; it is new authorization, not a claim that $240 has already been billed. The [prospective 39-fit campaign plan](evidence/authorized_sft_campaign_2026_10_13.json) explicitly records `PLANNED_NOT_LAUNCHED`, excludes the completed adapter, and preserves the finalized recipe. It is not an active launch certificate or running queue. No new GPU was allocated or training worker launched by this documentation update. RL and final evaluation are outside these SFT projections.

The completed adapter is retained, so the full forty-condition inventory has **39 remaining uncompleted fits**. At the same per-fit duration as this single measured 14B ROT13 condition:

| Conditional planning scope | Fit interval only | Worker interval including observed overhead |
| --- | --- | --- |
| All 40 fits | 64.37 GPU-hours / $192.46 | 70.25 GPU-hours / $210.05 |
| Remaining 39 fits | 62.76 GPU-hours / $187.65 | 68.49 GPU-hours / $204.80 |

The remaining-39 worker projection leaves approximately **$35.20** of the new $240 authorization for setup, runtime variation, and idle time. It is **not a guaranteed bound**: KaziText, other levels, 7B, other GPUs, and different startup batching may take different time. Track observed costs as fits complete and preserve verified state before the authorized ceiling. At a constant $2.99/hour, $240 corresponds to at most **80.27 billed GPU-hours**, with all charged setup/idle time consuming that allocation. Multiple GPUs reduce elapsed time only if work is balanced; combined GPU-hours and cost remain chargeable.

The historical 150-hour planning forecast and extrapolations from a two-update old-recipe benchmark are not actual spending. Do not present old approximately $740 or $2,000 conversation figures as measured invoices or as established estimates for this finalized recipe.

## Keeping the completed level without a training confound

Do not retrain or extend this level solely because funding or an evaluation panel changed. Its immutable source, recipe, dataset, seed, actual execution mode, and final endpoint identify a valid completed fit. Future fits should use the same declared scientific recipe, start from their own untouched base, and record condition-specific runtime choices. Later documentation or evaluation-only changes do not retroactively change the completed worker source. Any actual training-algorithm change must receive its own identity and comparability assessment; an old full state cannot simply be relabelled.

Adding GPQA does not require another epoch or a replacement adapter. If multiple training seeds are later needed, declare and retain them as independent replicates rather than discarding this completed result after seeing evaluation outcomes. No GPQA result was used to select or extend this adapter.

## GPQA Diamond appended evaluation: panel prepared, evaluation pending

On 13 October the user clarified that GPQA is **appended** to GSM8K, MMLU-Redux-2.0, ARC-Easy, and sanitized MBPP; it does not replace those domains. The [José et al. reference implementation](https://github.com/Jozdien/cot_legibility/blob/89195c70288b4b141460447f5fde02f67b38b043/src/inference/datasets.py#L17) maps `gpqa` to **`gpqa_diamond` (198 questions)**. The official acquisition revision is [`Idavidrein/gpqa`, `83022cefff930aea54f654c0b282e74b9eeda5c6`](https://huggingface.co/datasets/Idavidrein/gpqa/tree/83022cefff930aea54f654c0b282e74b9eeda5c6). The official Diamond source CSV has been downloaded through the standard Hugging Face SDK using the existing access; no new user authentication is needed. Its size is 1,373,492 bytes and its SHA-256 is `41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305`. The actual preparation CLI produced the complete 198-question panel with hash `322e4335322a93adc18dc0e22e982bf6bbcd1a07223ccf1b08e25dfb7a5076de`. The [preparation receipt](evidence/gpqa_preparation_receipt.json) reports `PREPARED`, ten targeted checks passed, and zero trained states evaluated. Comparison-state freeze, generation, and evaluation remain pending. See the [variant-source evidence](evidence/gpqa_variant_source.json) and [supplemental GPQA procedure](modules/evaluation/supplemental_gpqa.md).

This matches the reference's dataset variant, not its complete measurement pipeline: the reference prompt omits choices and uses language-model correctness grading, while this project's planned supplement uses four-option native final-field scoring. Retain the prepared panel’s acquisition, IDs and choice order. Before generation, freeze the exact comparison states, prompt/extraction, inference budget and scoring rules; use the same frozen questions and protocol for untouched bases and trained states. Audit detectable overlap without silently changing the already certified SFT data or claiming retroactive exclusion.

GPQA Diamond is a supplemental descriptive capability stress test. Existing four-domain confirmatory thresholds and windows remain unchanged unless a separate scientific amendment defines new criteria. Adding this evaluation panel does not require retraining the completed adapter and cannot establish the original minimum-three-contiguous-level construction window from one trained level.
