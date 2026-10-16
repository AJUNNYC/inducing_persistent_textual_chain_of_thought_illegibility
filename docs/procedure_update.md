# Current procedure: 16 October 2026

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
The campaign and fit are **PAUSED**, and GPU handover followed verification. The
[seven-file pause and handoff bundle](https://drive.google.com/drive/folders/12ncXrN2UjJb_rDKX7sSfjR9nLFRfFCOt)
is Drive-verified.
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

## Historical original completion and GPQA handoff, 14 October

All ten **14B ROT13 SFT levels are complete**, with final adapters, recovery states and results verified in Google Drive. The nine fresh levels 1–9 each reached 625 updates under scientific source `9de8813031da737453f491723a10b305679dc42d`; retained level 10 keeps source `2ea4fb9a7737cff806226366e73743eec5935fad`. The priority queue was observed complete on 14 October 2026, with zero pending or failed publications and its GPU idle. Each independent fit used all 10,000 certified training examples once, a 625-update cosine horizon and 19 warmup updates. KaziText level 10 remains paused at 444/625 with verified full recovery state and no automatic restart. All forty prepared condition datasets and historical 7B results remain available; their availability does not imply completed fits. At that 14 October handoff, the next selected work was GPQA Diamond for the untouched 14B base plus all ten ROT13 levels, using the separate `long_32000` evaluation profile. Training, data, recipe and original four-domain evaluation budgets are unchanged. See [the campaign guide](gpqa_14b_rot13_campaign.md) and [the training record](sft_completed_rot13_14b_level10.md).

The declared efficient baseline uses an unquantized frozen BF16 base and BF16 head, SDPA with efficient-kernel preference, direct full-gradient Cut Cross-Entropy (`cce_exact`) with FP32 loss and hidden-gradient accumulation, and plain AdamW with FP32 optimizer states. CCE is pinned to upstream commit `3de376c106a1916bc5e1b619f9c77c87a461ee1c`; `filter_eps=None`, `filter_e_grad=False` and `filter_c_grad=False` disable gradient filtering. No Qwen patch or full token-by-vocabulary logits matrix is used. Policy `published_bf16_lora_full_gradient_cce_v2` requests microbatch 2/accumulation 8. A non-updating duplicated-longest-example capacity guard tries 2/off, 1/off, 2/on, then 1/on, where on/off denotes activation checkpointing; only CUDA OOM permits fallback. The selected actual mode is recorded and cannot change during exact recovery.

Keep completion-only masks and per-example weighting, all seven LoRA targets (rank 8/alpha 16/dropout 0.05), learning rate `2e-4`, requested microbatch 2/accumulation 8/effective batch 16 (capacity fallback 1/16/16), seed 42, all 10,000/100 records, full 4096-token eligibility without truncation and verified Drive saves. Checkpoints occur at updates 100, 200, 300, 400, 500, 600 and final 625. Each update retains the same ordered 16 examples with per-example weight 1/16; execution grouping, padding, dropout draws and floating-point rounding may differ. No measured batching speedup or bitwise-equivalent outcome is claimed. This prospectively declared scientific baseline changes exposure, precision/head numerics, optimizer and execution; it does not prove identical final results. The original NF4 `legacy_two_epoch_v1` and intermediate NF4 `one_epoch_pilot_v1` remain explicit alternatives, with incompatible checkpoints and separate identities.

The user requested implementation without another optimization test or timing campaign. Normal native model/mask/gradient/capacity/full-state guards passed for the completed 14B ROT13 level-10 fit on RTX PRO 6000 Blackwell 96GB. Actual execution was microbatch 1 / accumulation 16 with activation checkpointing off. The recorded fit interval was 5,793.143751 seconds (96m33s); worker launch through completion was 6,322.565654 seconds (105m23s). At $2.99/hour these imply $4.81 and $5.25, respectively, calculated from runtime rather than invoices. That standalone L10 timing remains a dated measurement, not a bound for other levels or hardware. Archived PRO step-987 and A100 update-28/30 states remain in Drive, never warm starts for this recipe. One epoch removes 50% of nominal example passes versus the same-condition two-epoch recipe; no one-hour finish or measured hardware speedup is promised. Diagnostics use the 100 certified validation IDs and base companions; final test panels and original confirmatory thresholds remain reserved. No additional training after the completed 625-update cosine horizon is an exact continuation of the old 1,250-update schedule.

See the [completion, provenance and budget record](sft_completed_rot13_14b_level10.md). The historical untouched 14B GPQA Diamond baseline remains **49/198 (24.75%)**, with 133 truncated and three malformed outputs under its 3,072-token cap; all 198 count. Its Drive-verified records remain separate from the new `long_32000` comparison. The original 14 October launch plan required one matching untouched baseline and ten adapters, totaling 2,178 outputs. The complete matching baseline and original L1-6/L10 results are now reused; original L7-9 are prioritized as recorded above. Only GPQA receives the larger response/context allowance; GSM8K, MMLU, ARC-Easy and MBPP retain their existing 3,072-token measurement contract. The [GPQA contract](modules/evaluation/supplemental_gpqa.md) and [campaign guide](gpqa_14b_rot13_campaign.md) describe strict profile/cache isolation and verified Drive publication. KaziText remains suspended and unscheduled.

## Archived interruption: A100 comparison stopped on 11 October 2026

Both Colab A100 assignments ended; the account reports 0.00 compute units and rejects a new A100 allocation. ROT13 and KaziText baseline attempts are safely paused at updates 28 and 30, with full state verified in Drive before runtime loss. Source `8a0f853` implements checkpointed A100 loss candidates, per-level empirical qualification and verified setup reuse. Native candidate comparisons remain incomplete: no optimized fit has started and no A100 speedup is approved. The user requested fresh optimized restarts from untouched bases; archived partial fits are benchmark evidence only. There is no user-imposed wall-clock deadline. See [the interruption record](evidence/a100_optimization_interruption.json) and [restart procedure](colab_sft_restart.md).

## Prior implementation and evidence scope

The following implementation details and dated measurements retain their original scope; the completed efficient fit and current authorization are described above.

This records the user's approved change to SFT corpus generation and the executed storage and RL preparation steps. It replaces the earlier native Transformers SFT generation procedure for corpus construction. Training recipes, GRPO rollout generation, scientific success floors and validation acceptance rules remain unchanged. The later approved evaluation migration uses batched vLLM for both rebuilt baselines and trained states, as described below. The subsequently approved evaluation-only concentration guard is documented in [the statistical method specification](statistical_concentration_guard.md).

## Historical execution proposals and optimization scope

The [four-worker execution plan](sft_multi_gpu_execution_plan.md) records a **proposal**, not a deployed configuration: two PRO 6000 workers for 14B and two Colab A100 workers for the remaining 7B fits. The currently owned 7B ROT13 level-1 fit is excluded from their queues. All 40 independent fits and scientific settings remain fixed, with distinct verified Drive outputs and no duplicate ownership. Runtime/cost forecasts are estimates; A100/14B capacity, throughput and execution profiles are unvalidated. Current source a6e3295 approval remains limited to the measured 7B ROT13 level-1 condition.

The final [memory-copy trial](evidence/sft_memory_copy_pro6000_benchmark.json), source `07f8a68`, preserved bitwise losses, gradients, updates, final adapter/optimizer/scheduler and RNG in both pairs but was rejected solely on speed. Median times were 6.91222→6.77876 seconds (1.93% less) and 6.91286→6.92438 seconds (0.17% more); minimum matched ratio 0.99834x failed the required 1.05x. Peak allocated memory did not improve. A total of 359 affected CPU checks passed without repeating unrelated suites, providers or statistics. Accepted archived source `a6e3295` remains selected. The [actual resume](evidence/sft_memory_copy_pro6000_training_resume.json) records archived source a6e3295 training RUNNING from verified step 673 in the same existing lineage, with passed native gradients, verified Drive recovery and unchanged deadline. No migration to the rejected source occurred. Optimization rounds are closed. All parents, scientific settings and exactly 40 planned SFT fits are preserved; approval remains only 7B ROT13 level 1 on the measured Blackwell96GB GPU.

Source `a6e3295` selects the [scheduled spatial-CE profile](evidence/sft_scheduled_forward_pro6000_selected_profile.json) only for **7B ROT13 level 1** on RTX PRO 6000 Blackwell 96GB. Two counterbalanced step-509 recovery pairs measured 8.9243→6.7200 and 8.8950→6.6727 seconds per update: 24.70–24.98% less median time versus the previously accepted spatial-forward/tiled-backward profile. All 96 losses per pair, gradients, adapter updates, final adapter/optimizer/scheduler and RNG were bitwise identical. The same 3,190-token probe passed with the reference loss and 84.28-GiB allocated peak. [READY controls](evidence/sft_scheduled_forward_pro6000_readiness.json) bind that scope; 236 focused CPU checks passed without repeating unrelated suites, providers or statistics. The [earlier startup](evidence/sft_scheduled_forward_pro6000_training_resume.json) recorded source a6e3295 running from restored step 509, with passed native mask/gradient checks and verified Drive child state. [Parent recovery](evidence/sft_scheduled_forward_pro6000_pause_509.json), original science and Drive-save cadence remain unchanged. Earlier flat/fused forward candidates remain rejected. This direct incremental measurement does not establish a full-fit ETA, final-result equivalence or approval of other conditions.

The previous source `218900d` selected the [spatial-forward/tiled-backward profile](evidence/sft_loss_optimization_pro6000_selected_profile.json) only for **7B ROT13 level 1**, RTX PRO 6000 Blackwell 96GB. Two paired step-401 recovery repetitions measured 13.52% and 11.65% less median update time versus accepted no recomputation, minimum matched ratio 1.11456x; losses, gradients, updates, final adapter/optimizer/scheduler and RNG were bitwise identical. [Flat and fully fused forward CE](evidence/sft_loss_optimization_pro6000_parallel_ce.json) were faster but failed the frozen numerical bounds and remain rejected. Its [recorded readiness receipt](evidence/sft_loss_optimization_pro6000_readiness.json) binds the selected source/GPU/condition; 165 affected CPU checks passed, with unrelated suites and statistical simulations reused rather than repeated. Its [recorded continuation startup](evidence/sft_loss_optimization_pro6000_training_resume.json) restores step 401 under separate lineage and verifies Drive child state while preserving both parent checkpoints. Eager attention, FP32 head/logits/CE, masks, loss weighting, data, epochs, optimizer, microbatch 1/logical batch 16 and checkpoint cadence remain unchanged. The short screen does not guarantee final-result identity, a full-fit ETA, other levels or 14B.

The earlier source `05f42b0` passed the [paired recovery screen](evidence/sft_optimization_pro6000_checkpoint_recovery.json) for disabling activation recomputation on **7B ROT13 level 1**, RTX PRO 6000 Blackwell Server Edition 96GB. Two repetitions, each one warmup plus five measured logical updates restored from step 214, measured median throughput ratios of 1.2695x and 1.2806x (about 21–22% less update time), with bitwise loss, gradient, adapter, optimizer, scheduler and RNG agreement. CUDA input caching retained bitwise agreement but failed the speed gate and is not selected. The [earlier accepted profile](evidence/sft_optimization_pro6000_selected_profile.json) and [historical readiness receipt](evidence/sft_optimization_pro6000_readiness.json) bind this one condition and their recorded source/hardware. Sixty-nine focused checks and affected GPU controls passed; the full matrix and statistical simulations were not repeated. Eager attention, microbatch 1, logical batch 16 and the scientific recipe remain unchanged. The earlier [continuation launch](evidence/sft_optimization_pro6000_training_resume.json) recorded startup from inherited step 214 with a Drive-verified child checkpoint and separate source/profile lineage. Original parent bytes remain unchanged; the launch receipt does not claim later completed optimizer steps.

The original H100 7B ROT13 state is [preserved at optimizer step 214](evidence/sft_training_pause_214.json), with verified full recovery state in Drive. [Source fd23db9 was measured on RTX PRO 6000 Blackwell Server Edition 96GB](evidence/sft_optimization_pro6000_benchmark.json) using fresh isolated adapters, one warmup and two measured logical updates per profile. Baseline median update time was 11.89899 seconds. Redundant-check removal retained bitwise losses, gradients, adapter updates and final RNG, but its 1.0020x speed ratio failed the 5% speed requirement. SDPA was 0.63% slower and failed the numerical screen; no fused kernel was independently observed. Neither candidate is activated, and this sample does not establish full-fit speed or final-result equivalence. The [Drive index](evidence/sft_optimization_pro6000_drive_index.json) preserves actual reports and the earlier cache-lock failure. The original checkpoint remains untouched. The subsequent source 05f42b0 recovery and readiness receipts above supply the affected migration controls while reusing compatible upstream evidence; [optimization precedent and limits](sft_optimizations.md) define the scope.

## Original consumer launch readiness

All eight model/domain repeatability gates passed after the [versioned final-field scoring repair](final_field_scoring.md) rescored the same 7,398 saved baseline responses. No response was regenerated, no success threshold changed, and the original failed baseline evidence remains retained. The [source-bound launch receipt](evidence/sft_launch_readiness.json) records all thirteen mandatory gates passed for consumer `773cfd3` and certificate `e8b47049…`. Compatible completed controls and the original Drive-audited corpus were reused with their original identities preserved.

The [existing H100 validation](evidence/sft_launch_pod_validation.json) accepted the exact certified forty-condition worklist and the ten selected ROT13 7B fits through `scripts/run_certified_sft.py --validate-only`. No scientific optimizer update had started at this readiness handoff. The training recipe and dataset bytes remain unchanged; full legibility judging, trained-state evaluation and RL training remain pending. Subsequent measurement must use the frozen vLLM evaluation protocol; the generic native evaluation default does not select that protocol automatically.

## SFT corpus generation

Use separate H100 80-GB Pods for the untouched pinned 7B and 14B models. Each runs `scripts/run_vllm_generation_worker.py` from frozen generation commit `dc520cb73bf43b59a9566404dd4a7aa8358f1cc3`. Both consume the same ordered, audited initial 10,100 LMSYS prompt IDs. The configuration hash is `3757493e93d7f3ac31d9215bd493ae4f13cae0bed7c97e773f4ca4ef5be2ed54`; run ID is `vllm_fresh_dc520cb`.

The approved backend is vLLM 0.11.0 with Transformers 4.57.3 and Torch 2.8.0+cu128. Use BF16 weights without quantization, 128 requests per chunk, maximum model length 4096, maximum concurrent sequences 128, scheduler token budget 8192, GPU memory utilization 0.9, and prefix caching disabled. The scheduler token budget is concurrent processing capacity, not a shared output allowance: **every request retains its own 3072-new-token cap**. Temperature remains 0.6 and top_p remains 0.95. Preserve native chat-template prompt token IDs, EOS IDs, pinned model/tokenizer revisions, and independently derived per-example seeds. Prompt eligibility remains at most 200 input tokens.

The vLLM environment is separate from the native Torch 2.9 training environment. Its actual installed package inventory and hash are saved in each immutable run manifest; the native dependency lock is not claimed to describe this engine. Identical weights and sampling parameters do not imply bitwise-identical sampled outputs across engines.

Generate the corpus afresh from prompt one. Old native responses and benchmark responses remain archived and are excluded from the fresh training corpus. Both training models must use only matching fresh vLLM production responses. This is a generation backend change, not a change to the SFT or RL optimizer.

## Drive is the permanent store

The central project folder is [Inducing Persistent Textual Chain-of-Thought Illegibility](https://drive.google.com/drive/folders/176a8Tm6l4WhfakY_0nCOo4ICeT7KPSmx). RunPod container disks are disposable scratch; no Pod volume or network volume is relied upon for preservation.

The worker saves raw responses locally, registers each response with its source/configuration identity, uploads through the repository ArtifactService, and verifies the Drive copy before starting another batch. Immutable run metadata is uploaded before GPU work. Reported progress counts only batches that passed this Drive barrier.

| Artifact | Folder within the central project |
|---|---|
| Fresh 7B responses and metadata | `collect_clean/shared/7b/vllm_fresh_dc520cb` ([open](https://drive.google.com/drive/folders/1bBj2WwDFzyymXyu1FAeKvhZWvqozXAQK)) |
| Fresh 14B responses and metadata | `collect_clean/shared/14b/vllm_fresh_dc520cb` ([open](https://drive.google.com/drive/folders/1Dgdm8NnBawhNd2C_KuE4RjTS3UEuMTZf)) |
| Shared RL question manifest | `prepare_rl/shared/shared/rl_questions_800` ([open](https://drive.google.com/drive/folders/1l63teeYctaMnHhmTHJBOxfDf5dIDn1-q)) |

Intentional pauses drain a whole batch. Restart only after the prior worker has exited, using the original source, package inventory, arguments, configuration, and output roots. The worker checks existing raw identities and skips compatible saved responses; pending uploads must be verified before new generation. The observed 7B interruption was two Drive upload timeouts, not a CUDA memory failure; resuming repaired those uploads without regenerating saved responses. Stopping a worker does not stop Pod billing; the account operator controls the Pod console.

An authenticated Jupyter connection operates each Pod; RunPod account API keys are unavailable under the PI account's permissions. Google Drive uses the existing restricted drive.file OAuth authorization. Credentials remain in protected files outside Git and experimental artifacts. Do not put tokens in documentation, manifests, or connection logs.

## Raw generation is not final dataset acceptance

10,100 initial raw responses per model are not 10,100 accepted pairs. Once the two caches are complete, join by the original prompt ID and apply the existing structural completion, reasoning/final parsing, eligible-word, protection, and clean/all-transformed length checks. Do not add semantic correctness or quality filtering.

Refills use new audited IDs in fixed order, with the batch size exactly equal to the accepted-pair deficit. Rejected ordinary completions are retained rather than repeatedly sampled. Generate refill responses through separate model-specific vLLM workers with recorded provenance; the existing single-model engine cannot load both models in one process. `scripts/coordinate_vllm_sft_preparation.py` now coordinates acceptance, exact-deficit refill workers, and final construction. `scripts/prepare_vllm_sft_data.py` calls the existing native pair validator against cached responses, retains exact official KaziText outputs, verifies each acceptance chunk and transform archive in Drive, and emits the next deficit manifest. CPU processes run independent official noise workers; no response is resampled during acceptance. Initial-round verification is split across both existing Pods: the second Pod processes candidate offsets 2048 onward and publishes completed pair records and official transform caches to Drive. `scripts/coordinate_verification_shard.py` checks transfer hashes and imports those caches into the primary Pod without overwriting differing results. The primary coordinator preserves the original candidate order, final acceptance totals, and exact-deficit refills. Final paired acceptance and exact-deficit refills completed on 11 October 2026. Across 11,639 candidate IDs, 10,100 pairs were accepted and 1,539 rejected. Freeze the shared 10,000 training and 100 validation IDs only after enough pairs pass all checks. Then construct and verify all ROT13 and KaziText condition datasets and their Drive copies. Never mix old native or benchmark responses into these caches.

## Evaluation and RL data

The frozen 1,233-item capability panel and nested 200-item legibility subset were prepared and backed up before the backend change. Their IDs are reused unchanged. All six untouched-model capability passes are complete and the repaired repeatability gate passed; full legibility judging and trained-model evaluations remain pending. The SFT generation migration originally left evaluation unchanged. The subsequently approved measurement migration now uses pinned BF16 vLLM for both rebuilt untouched baselines and trained adapter evaluations. Preserve all public prompts, per-item seeds and the full 3,072-token allowance. Run 7B then 14B on one GPU. Capability repeatability explicitly defers judging; later judge the same saved seed-42 outputs. Native baseline attempts and the original failed scoring reports remain archived. See [the active worker contract](baseline_repeatability_worker.md) and [final-field scoring specification](final_field_scoring.md).

The repository's `prepare_rl_data` function has now selected 800 unique GSM8K training questions from revision `740312add88f781978c0658806c59bc2815b9866`. It checked valid numeric references, duplicate questions, and overlap against the complete official evaluation frames, then fixed seed-42 ordering into 100 consecutive groups of eight. Manifest content hash: `bc4c6b1b50906a9ee1c3ee47c9f4a6b72a7dd6dacc852ac58a1b090688b4a0a8`. Preparation source commit: `a03a70dbd520a34cf0dda67e73d0de2337a87140`. The manifest was uploaded and verified in the shared RL folder linked above.

Both RL trajectories use this same ordered manifest. Preparation does not create imitation reasoning targets or policy responses. The current policy generates four rollouts per question during GRPO; eight questions and 32 rollouts form one logical update. The existing RL temperature, token cap, reward rules, optimizer, and checkpoint cadence remain unchanged. RL training has not started.

## Validation and monitoring

The vLLM benchmark checked actual generation, formatting, token limits, and GPU memory on the existing hardware. Matched 14B batches of 32 took approximately 40.53 seconds with vLLM versus 342.78 seconds with native generation; the 128-request vLLM sample took 52.64 seconds without OOM. These are generation-only samples, not quality equivalence or end-to-end throughput guarantees. Every production batch separately passes Drive integrity verification.

At the user's request, the full test suite and compatible completed launch controls were not repeated. Final acceptance, cross-model pairing, all transformed datasets and Drive copies are verified. The baseline capability passes and repaired repeatability gate are complete, and all thirteen SFT launch gates now pass through explicit evidence reconciliation. Prior prospective statistical validation recorded 34/36 jointly covered repetitions with two coverage failures; the approved concentration-guard revision subsequently passed all seven full-budget scenarios (700/700 jointly covered repetitions). [The passing report](evidence/statistical_guard_v2_report.json), the original failed evidence and the [current launch receipt](evidence/sft_launch_readiness.json) remain distinct. No trained-model result is claimed.

`scripts/monitor_vllm_generation.py` checks both Pods every 30 seconds, records safe status snapshots locally, and reports progress, terminal events, connection errors, and ten-minute stalls. It does not restart workers or send autonomous chat notifications. Its ETA covers initial raw generation and Drive replication only, excluding refills, transformations, training, and evaluation.

The active generation source remains frozen even when documentation or the monitor is pushed to newer GitHub commits. Repeated test counts and transient generation totals are not presented as completed scientific results.

## Integrated protocol and workflow sources

The active [clean collection diagram](../spec/mermaid/workflows/data_generation_collection/sft_data/clean_trace_collection/clean_trace_collection.mmd), [candidate batch diagram](../spec/mermaid/workflows/data_generation_collection/sft_data/clean_trace_collection/candidate_batch/clean_trace_batch.mmd), and [RL preparation diagram](../spec/mermaid/workflows/data_generation_collection/rl_data/rl_data_preparation.mmd) describe the updated procedure. Their module documents mirror the Mermaid sources.

The active [experiment.pdf](../spec/experiment.pdf) incorporates these procedures directly into Sections 3.1, C.3.2, E.1.3, E.2.3, E.2.8, H.1, H.3, H.7 and H.10. It has 81 pages, with no appended procedure amendment or separate amendment PDF. Existing page citations remain valid. The historical source in `spec/archive/experiment_original.pdf` remains unchanged. Editable replacement text lives in `spec/protocol_page_updates.json`; rebuild the active PDF, extraction and provenance with `python scripts/build_protocol_pdf.py` using PyMuPDF.

## Dataset publication and retrieval

Construction uses the repository's existing `build_sft_datasets` and `validate_sft_bundle` boundaries. Cached official KaziText transforms must match their reasoning hashes and runtime provenance before reuse. Each train/validation JSONL reaches verified Drive storage before construction advances; all forty condition manifests and the bundle are then verified. The final `dataset_drive_index.json` maps logical paths, checksums, sizes and artifact IDs to Drive file IDs, so repeated filenames can be downloaded unambiguously. Completion means 40 conditions, each with 10,000 training and 100 validation records; training itself remains a separate gated stage.

## Configurable coordination

Both coordinators accept `--topology /absolute/path/to/topology.json`. The checked-in `deployment/coordinator.json` preserves the current two-Pod assignments, but the routing code has no fixed Pod count or model-name branches. List connection file paths under `pods`, choose `primary_pod`, and assign generation runtimes and model IDs under `generation_workers`. Each generation worker has a dedicated Pod; the primary may be a separate CPU-only Pod. Configure any number of disjoint verification ranges under `verification_shards`, including their source Pod, round, chunk size, data root and worker state path. The importer monitors shards concurrently, uses separate transfer filenames, and rejects overlapping ranges, duplicate model assignments and unknown Pods before connecting. Credentials remain in protected files; this JSON contains paths only.

This generalizes coordination, not the scientific experiment. The current generation and accepted-pair workers still implement the pinned 7B/14B study. A different model family must provide a compatible worker and validation recipe; adding its name to the topology does not alter the experiment or authenticate/provision a Pod. Shard workers and cached prerequisites must exist before starting the importer. Existing running controllers retain their loaded configuration; these changes do not restart live verification.

## Verified final dataset handoff

Preparation completed at 04:00 Eastern on 11 October 2026. The shared pool contains exactly 10,000 training IDs and 100 validation IDs. Acceptance rounds processed/accepted: 10,100/8,752; 1,348/1,180; 168/149; 19/17; 2/0; 2/2. All responses, including refills, came from the frozen fresh vLLM workers; rejected prompt IDs were never resampled.

All 40 conditions (two models, two interventions, ten levels) are built and validated: 80 train/validation JSONLs, 40 condition manifests, a bundle manifest, and the accepted-pool manifest. The 404,000 dataset records are repeated condition-specific transformations of the shared IDs, not 404,000 unique prompts. The download index lists all 122 artifact paths with SHA-256 digests and verified Drive file IDs.

- [Final dataset folder](https://drive.google.com/drive/folders/1kNpUWZ7Z9zXPF-0cu-zifoPpLh2lLg8A)
- [Verified download index](https://drive.google.com/file/d/1jVgApFjn1sDsBaTTWooGn3XdvF8av75O/view)

Dataset completion and the passing revised statistical validation establish their respective components. The current reconciled SFT handoff additionally satisfies all thirteen launch gates; training completion and full legibility evaluation are still separate results. Pod billing remains controlled manually through the account operator.

## Independent Drive readback audit

A separate CPU-only audit passed at 08:03 Eastern on 11 October 2026, with no Pod connections or GPU use. It downloaded all 122 indexed artifacts (7,000,839,153 bytes) directly from Drive and checked every file's SHA-256 and size. All 40 condition manifests matched the bundle. The native accepted-pool and dataset validators passed again on all 10,100 paired sources and 404,000 records, including split order, both models' revisions, source identities, completions, recorded transformation hashes and token/protection invariants. Production provenance identified the frozen vLLM backend and excluded fixture/simulation traces.

The audit checks integrity and frozen record invariants. It does not grade answer quality or recompute the official transformations/tokenizer lengths. It does not rerun or waive prospective statistical validation.

- [Machine-readable audit report](evidence/vllm_sft_drive_readback_audit.json)
- [Verified audit report in Drive](https://drive.google.com/file/d/18-T8LwpuQJM9oG6FGcoDZesqcCFepcxk/view)
- Repeat with `scripts/audit_sft_drive.py --credentials /protected/google_token.json --index-id 1jVgApFjn1sDsBaTTWooGn3XdvF8av75O --report /workspace/scratch/drive_audit.json`. Credential values never appear in the report.
