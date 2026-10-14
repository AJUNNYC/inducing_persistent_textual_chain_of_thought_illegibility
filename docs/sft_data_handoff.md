# SFT producer-to-consumer data handoff

## Current status: all ten 14B ROT13 fits complete; GPQA long-profile evaluation next

All ten **14B ROT13 SFT levels are complete**, with final adapters, recovery states and results verified in Google Drive. The nine fresh levels 1–9 each reached 625 updates under scientific source `9de8813031da737453f491723a10b305679dc42d`; retained level 10 keeps source `2ea4fb9a7737cff806226366e73743eec5935fad`. The priority queue was observed complete on 14 October 2026, with zero pending or failed publications and its GPU idle. Each independent fit used all 10,000 certified training examples once, a 625-update cosine horizon and 19 warmup updates. KaziText level 10 remains paused at 444/625 with verified full recovery state and no automatic restart. All forty prepared condition datasets and historical 7B results remain available; their availability does not imply completed fits. The next selected work is GPQA Diamond for the untouched 14B base plus all ten ROT13 levels, using the separate `long_32000` evaluation profile. Training, data, recipe and original four-domain evaluation budgets are unchanged. See [the campaign guide](gpqa_14b_rot13_campaign.md) and [the training record](sft_completed_rot13_14b_level10.md).

The declared efficient baseline uses an unquantized frozen BF16 base and BF16 head, SDPA with efficient-kernel preference, direct full-gradient Cut Cross-Entropy (`cce_exact`) with FP32 loss and hidden-gradient accumulation, and plain AdamW with FP32 optimizer states. CCE is pinned to upstream commit `3de376c106a1916bc5e1b619f9c77c87a461ee1c`; `filter_eps=None`, `filter_e_grad=False` and `filter_c_grad=False` disable gradient filtering. No Qwen patch or full token-by-vocabulary logits matrix is used. Policy `published_bf16_lora_full_gradient_cce_v2` requests microbatch 2/accumulation 8. A non-updating duplicated-longest-example capacity guard tries 2/off, 1/off, 2/on, then 1/on, where on/off denotes activation checkpointing; only CUDA OOM permits fallback. The selected actual mode is recorded and cannot change during exact recovery.

Keep completion-only masks and per-example weighting, all seven LoRA targets (rank 8/alpha 16/dropout 0.05), learning rate `2e-4`, requested microbatch 2/accumulation 8/effective batch 16 (capacity fallback 1/16/16), seed 42, all 10,000/100 records, full 4096-token eligibility without truncation and verified Drive saves. Checkpoints occur at updates 100, 200, 300, 400, 500, 600 and final 625. Each update retains the same ordered 16 examples with per-example weight 1/16; execution grouping, padding, dropout draws and floating-point rounding may differ. No measured batching speedup or bitwise-equivalent outcome is claimed. This prospectively declared scientific baseline changes exposure, precision/head numerics, optimizer and execution; it does not prove identical final results. The original NF4 `legacy_two_epoch_v1` and intermediate NF4 `one_epoch_pilot_v1` remain explicit alternatives, with incompatible checkpoints and separate identities.

The user requested implementation without another optimization test or timing campaign. Normal native model/mask/gradient/capacity/full-state guards passed for the completed 14B ROT13 level-10 fit on RTX PRO 6000 Blackwell 96GB. Actual execution was microbatch 1 / accumulation 16 with activation checkpointing off. The recorded fit interval was 5,793.143751 seconds (96m33s); worker launch through completion was 6,322.565654 seconds (105m23s). At $2.99/hour these imply $4.81 and $5.25, respectively, calculated from runtime rather than invoices. That standalone L10 timing remains a dated measurement, not a bound for other levels or hardware. Archived PRO step-987 and A100 update-28/30 states remain in Drive, never warm starts for this recipe. One epoch removes 50% of nominal example passes versus the same-condition two-epoch recipe; no one-hour finish or measured hardware speedup is promised. Diagnostics use the 100 certified validation IDs and base companions; final test panels and original confirmatory thresholds remain reserved. No additional training after the completed 625-update cosine horizon is an exact continuation of the old 1,250-update schedule.

See the [completion, provenance and budget record](sft_completed_rot13_14b_level10.md). The historical untouched 14B GPQA Diamond baseline remains **49/198 (24.75%)**, with 133 truncated and three malformed outputs under its 3,072-token cap; all 198 count. Its Drive-verified records remain separate from the new `long_32000` comparison. A new matching untouched baseline must run once before all ten completed ROT13 adapters, yielding 2,178 unique outputs. No trained-state GPQA result is claimed yet. Only GPQA receives the larger response/context allowance; GSM8K, MMLU, ARC-Easy and MBPP retain their existing 3,072-token measurement contract. The [GPQA contract](modules/evaluation/supplemental_gpqa.md) and [campaign guide](gpqa_14b_rot13_campaign.md) describe strict profile/cache isolation and verified Drive publication. KaziText remains suspended and unscheduled.

## Published-method reconciliation: ready to initialize, native checks pending

After freezing the consumer source/configuration and issuing its matching data certificate, use the committed CPU reconciliation entrypoint with the actual file paths:

```bash
python scripts/reconcile_sft_launch.py --published-method \
  --config "$SFT_CONSUMER_CONFIG" \
  --certificate "$SFT_HANDOFF_CERTIFICATE" \
  --control-config "$SFT_ORIGINAL_CONTROL_CONFIG" \
  --output "$SFT_LAUNCH_OUTPUT"
```

`SFT_ORIGINAL_CONTROL_CONFIG` must be the configuration whose hash matches the preserved `docs/evidence/sft_runtime_checks_report.json`; it supplies provenance for compatible upstream reuse. `--baseline-root`, historical native-control overrides and execution/recovery benchmark arguments are not used in this mode. The efficient configuration must name the executed frozen Git commit and its certificate must bind that same consumer.

A successful reconciliation emits `readiness_receipt.json` with **`READY_TO_INITIALIZE`**, `native_runtime_preflight_status=PENDING_AT_LAUNCH`, `native_checks_passed=false`, `training_ready=false`, `study_ready=false` and `training_started=false`. Its external evidence concerns certified data, durable storage and compatible statistical controls; current spending authorization remains a launch requirement. Native model/mask/gradient/capacity/full-state guards execute normally at actual startup before optimizer updates. Reconciliation does not allocate a GPU, select conditions, run a benchmark or prove training success. Subsequent actual completion and native evidence exist for 14B ROT13 level 10 only; do not replace generic initialization status with approval of another condition.

## Archived interruption: A100 comparison stopped on 11 October 2026

Both Colab A100 assignments ended; the account reports 0.00 compute units and rejects a new A100 allocation. ROT13 and KaziText baseline attempts are safely paused at updates 28 and 30, with full state verified in Drive before runtime loss. Source `8a0f853` implements checkpointed A100 loss candidates, per-level empirical qualification and verified setup reuse. Native candidate comparisons remain incomplete: no optimized fit has started and no A100 speedup is approved. The user requested fresh optimized restarts from untouched bases; archived partial fits are benchmark evidence only. There is no user-imposed wall-clock deadline. See [the interruption record](evidence/a100_optimization_interruption.json) and [restart procedure](colab_sft_restart.md).

## Prior implementation and evidence scope

The following implementation details and dated measurements retain their original scope; the completed efficient fit and current authorization are described above.

The prepared corpus keeps its original generation configuration, source commit,
model traces and transformed targets. A repaired training implementation has its
own configuration and native dependency lock. These identities are explicitly
separated, rather than rewriting records or pretending current code is the old
producer.

`scripts/certify_sft_handoff.py` checks all 122 local artifact sizes and SHA-256
hashes against the completed independent Google Drive readback. It validates the
40-condition production worklist, including ordered shared IDs, model-specific
clean traces, exact final answers and 10,000/100 splits. Scientific configuration
projections must agree; only implementation revision and dependency lock hash
may differ beyond existing operational exclusions.

The certificate binds both configurations, original Drive proof, consumer source
files, native lock and exact ordered record hashes. `build_worklist` accepts the
certificate explicitly. Pass the same certificate to `run_family_queue` or
`run_sft_fit`; a fit with producer identity fails without its matching certificate.
The certificate identity enters fit/checkpoint identity. All existing record,
model, target and preflight validations remain in force.

Run the committed script with `--producer-config`, `--consumer-config`, `--audit`,
`--bundle` and `--output`. Consumer implementation revision must name the actual
source commit and dependency lock hash must match `requirements.lock`. Store the
certificate and worklist in Drive. Compatibility certification does not satisfy
baseline repeatability or authorize a scientific fit by itself.

The certificate, worklist, consumer configuration and launch evidence for the paused fit are in the [verified Drive launch folder](https://drive.google.com/drive/folders/1x7_brQz7cCutiTZ42adyQ4C5ilZQsrGV). Its consumer source remains frozen at `773cfd3cc6a776262e9ca65ca10667aaad456b69`. The [certificate](evidence/sft_data_handoff_773cfd3_certificate.json) retains the original corpus and training settings. The newer [optimization implementation](sft_optimizations.md) changes training source bytes and requires a fresh consumer certificate plus affected target-GPU evidence; it is not covered by this original certificate. The prior `a50de2c` handoff remains historical in its original folder.

## Certified single-GPU launch

`scripts/run_certified_sft.py` consumes the consumer configuration, certificate,
production `transformed/manifest.json`, certified worklist, canonical preflight
report and UTC deadline. `--family rot13 --model 7b` runs the ten independent
encoding levels sequentially; `--level 1` selects only the first fit. Each fit
uses a fresh base or an exact full checkpoint. The fixed optimizer, loss, counts
remain unchanged within the historical legacy handoff. The final efficient recipe changes exposure, base/head precision, loss implementation, optimizer and execution. Actual completion: fresh 14B ROT13 level 10. Active training selects 20 14B conditions across both families/all ten levels, with 19 uncompleted; $240 SFT authorized 13 Oct. Bind their own consumer recipe/configuration identities; do not use legacy readiness as efficient-runtime approval. Checkpoints and final adapter payloads pass
verified Drive barriers. SIGTERM and the deadline pause at a whole optimizer
update. In historical NF4 mode, `--validate-only` checks the original thirteen gates. In efficient published mode it checks only the external initialization bindings; native startup guards remain pending, and no model or training starts.

`scripts/reconcile_sft_launch.py` reuses compatible passed control evidence,
retaining its original source/configuration hashes and adding verified current
source/recipe bindings. Historical NF4 reconciliation changed implementation identity only and required its repaired baseline gate. The efficient `--published-method` route above declares a new recipe and does not inherit those native passes.
Scoring, prompt and GPU controls are not declared passed through fixture tests.

The SFT consumer configuration is separate from the vLLM measurement protocol.
Subsequent adapter evaluations must explicitly use the saved frozen vLLM
contract; the generic experiment evaluator must not silently default to native
inference. Capability-only repeatability does not complete legibility judging.


## Validated launch on 11 October 2026

The repaired baseline gate passed for all four domains of both models using the exact 7,398 preserved outputs. The [launch reconciliation receipt](evidence/sft_launch_readiness.json) records all thirteen mandatory training gates passed for consumer `773cfd3`, and the [actual H100 validation](evidence/sft_launch_pod_validation.json) returned READY with forty certified conditions and ten selected 7B ROT13 fits. This snapshot preceded the first fit, now paused at step 214. Compatible GPU and statistical controls retained their original evidence and explicit source/recipe bindings. These receipts do not authorize new execution profiles or replacement hardware.

From the pinned Git archive `/workspace/sft_launch_773cfd3` on the existing H100, the tested launch validation is:

```bash
SOURCE_CODE_COMMIT=773cfd3cc6a776262e9ca65ca10667aaad456b69 \
ARTIFACT_ROOT=/workspace/artifacts/sft_launch_773cfd3 \
/workspace/illegibility_gpu_env/bin/python scripts/run_certified_sft.py \
  --config /workspace/bootstrap/sft_launch_configuration_773cfd3.json \
  --certificate /workspace/artifacts/sft_data_handoff_773cfd3/certificate.json \
  --bundle /workspace/data/vllm_sft_preparation/transformed/manifest.json \
  --worklist /workspace/artifacts/sft_data_handoff_773cfd3/worklist.json \
  --preflight /workspace/artifacts/sft_launch_773cfd3/preflight/report.json \
  --deadline 2026-10-11T20:20:00Z --family rot13 --model 7b --validate-only
```

For training, remove `--validate-only` and provide the existing private `GOOGLE_OAUTH_TOKEN_FILE` and central `GOOGLE_DRIVE_FOLDER_ID` bindings. The deadline must still be in the future; the worker preserves full checkpoints to Drive before pausing. The frozen measurement contract is `/workspace/artifacts/sft_launch_773cfd3/measurement_protocol.json`; its configuration binding differs explicitly from the original baseline binding, while measurement settings, panels and code hashes agree. Legibility judging remains pending.


## Original H100 first-fit launch

The [launch receipt](evidence/sft_training_launch.json) records 7B ROT13 level 1 training on the existing H100, after native model initialization and successful actual completion-loss/adapter-gradient verification. The original ten independent levels remain queued. The launch-only `HF_HUB_OFFLINE` flag was removed after it blocked a tokenizer metadata request; model revisions and cached weights are unchanged. Verified Drive checkpoints occur every 100 optimizer updates, and the authorized 20:20 UTC deadline pauses after a whole update and preserved full checkpoint. The readiness snapshot above predates this launch.


## Safe pause and resumption

The [four-worker execution plan](sft_multi_gpu_execution_plan.md) records a **proposal**, not a deployed configuration: two PRO 6000 workers for 14B and two Colab A100 workers for the remaining 7B fits. The currently owned 7B ROT13 level-1 fit is excluded from their queues. All 40 independent fits and scientific settings remain fixed, with distinct verified Drive outputs and no duplicate ownership. Runtime/cost forecasts are estimates; A100/14B capacity, throughput and execution profiles are unvalidated. Current source a6e3295 approval remains limited to the measured 7B ROT13 level-1 condition.

The final [memory-copy trial](evidence/sft_memory_copy_pro6000_benchmark.json), source `07f8a68`, preserved bitwise losses, gradients, updates, final adapter/optimizer/scheduler and RNG in both pairs but was rejected solely on speed. Median times were 6.91222→6.77876 seconds (1.93% less) and 6.91286→6.92438 seconds (0.17% more); minimum matched ratio 0.99834x failed the required 1.05x. Peak allocated memory did not improve. A total of 359 affected CPU checks passed without repeating unrelated suites, providers or statistics. Accepted archived source `a6e3295` remains selected. The [actual resume](evidence/sft_memory_copy_pro6000_training_resume.json) records archived source a6e3295 training RUNNING from verified step 673 in the same existing lineage, with passed native gradients, verified Drive recovery and unchanged deadline. No migration to the rejected source occurred. Optimization rounds are closed. All parents, scientific settings and exactly 40 planned SFT fits are preserved; approval remains only 7B ROT13 level 1 on the measured Blackwell96GB GPU.

Source `a6e3295` selects the [scheduled spatial-CE profile](evidence/sft_scheduled_forward_pro6000_selected_profile.json) only for **7B ROT13 level 1** on RTX PRO 6000 Blackwell 96GB. Two counterbalanced step-509 recovery pairs measured 8.9243→6.7200 and 8.8950→6.6727 seconds per update: 24.70–24.98% less median time versus the previously accepted spatial-forward/tiled-backward profile. All 96 losses per pair, gradients, adapter updates, final adapter/optimizer/scheduler and RNG were bitwise identical. The same 3,190-token probe passed with the reference loss and 84.28-GiB allocated peak. [READY controls](evidence/sft_scheduled_forward_pro6000_readiness.json) bind that scope; 236 focused CPU checks passed without repeating unrelated suites, providers or statistics. The [earlier startup](evidence/sft_scheduled_forward_pro6000_training_resume.json) recorded source a6e3295 running from restored step 509, with passed native mask/gradient checks and verified Drive child state. [Parent recovery](evidence/sft_scheduled_forward_pro6000_pause_509.json), original science and Drive-save cadence remain unchanged. Earlier flat/fused forward candidates remain rejected. This direct incremental measurement does not establish a full-fit ETA, final-result equivalence or approval of other conditions.

The previous source `218900d` selected the [spatial-forward/tiled-backward profile](evidence/sft_loss_optimization_pro6000_selected_profile.json) only for **7B ROT13 level 1**, RTX PRO 6000 Blackwell 96GB. Two paired step-401 recovery repetitions measured 13.52% and 11.65% less median update time versus accepted no recomputation, minimum matched ratio 1.11456x; losses, gradients, updates, final adapter/optimizer/scheduler and RNG were bitwise identical. [Flat and fully fused forward CE](evidence/sft_loss_optimization_pro6000_parallel_ce.json) were faster but failed the frozen numerical bounds and remain rejected. Its [recorded readiness receipt](evidence/sft_loss_optimization_pro6000_readiness.json) binds the selected source/GPU/condition; 165 affected CPU checks passed, with unrelated suites and statistical simulations reused rather than repeated. Its [recorded continuation startup](evidence/sft_loss_optimization_pro6000_training_resume.json) restores step 401 under separate lineage and verifies Drive child state while preserving both parent checkpoints. Eager attention, FP32 head/logits/CE, masks, loss weighting, data, epochs, optimizer, microbatch 1/logical batch 16 and checkpoint cadence remain unchanged. The short screen does not guarantee final-result identity, a full-fit ETA, other levels or 14B.

The earlier source `05f42b0` has [accepted no-recomputation evidence](evidence/sft_optimization_pro6000_selected_profile.json) and [READY affected-runtime controls](evidence/sft_optimization_pro6000_readiness.json) for 7B ROT13 level 1 on RTX PRO 6000 Blackwell 96GB. CUDA input caching failed the speed gate and remains disabled. The launcher accepts `--execution-benchmark` and `--execution-profile` only with passed source/GPU/condition-matched evidence. Migrating the original step-214 fit additionally requires `--migration-checkpoint`, `--migration-receipt` and `--recovery-benchmark`, selecting that one condition. It verifies parent identity and full-state bytes, preserves immutable scientific fields, inherits adapter/optimizer/scheduler/sampler/RNG state and writes separate consumer lineage while retaining original checkpoint bytes. The two paired 96-example sequences had bitwise state agreement, with 1.2695x and 1.2806x median throughput ratios. The full matrix and statistical validation were not repeated; compatible upstream evidence remains reusable. The [continuation worker](evidence/sft_optimization_pro6000_training_resume.json) launched from inherited step 214 with verified Drive child state; parent bytes remain unchanged.

The original H100 7B ROT13 fit was [paused at step 214](evidence/sft_training_pause_214.json), with the whole optimizer boundary and full recovery state preserved in Drive. For an unchanged-source resume, restore the registered `full_state.pt` and `manifest.json` into their original checkpoint directory using their exact provenance; restore the certified corpus, config, certificate and worklist from the existing Drive indices. Execute pinned source `773cfd3` with the same consumer identity and a new authorized future deadline. The fit automatically restores its latest matching verified full checkpoint, including optimizer, scheduler, RNG and sampler position. The previous 20:20 UTC deadline is historical and must not be extended implicitly. Stopping the training process does not stop provider billing; the PI account operator stops the Pod through the console.

Replacement hardware must additionally pass the affected runtime checks and record an explicit migration. Changing source, attention kernels or execution profile is not an exact continuation under the old certificate. The historical source `fd23db9` benchmark used fresh adapters without reading the paused checkpoint. The source `05f42b0` recovery screen instead loads its immutable full state into isolated comparisons, evaluates the next ordered updates, and never rewrites or advances the official checkpoint.

The [actual source fd23db9 Blackwell benchmark](evidence/sft_optimization_pro6000_benchmark.json) completed fresh baseline, deferred-check and SDPA profiles on RTX PRO 6000 Blackwell Server Edition 96GB. It neither read nor modified the official step 214 state. Deferred checks retained bitwise numerical agreement but missed the 5% speed requirement; SDPA failed the numerical screen and was slightly slower. Neither profile is approved for production. The [verified Drive index](evidence/sft_optimization_pro6000_drive_index.json) preserves both the measured attempt and the earlier cache-lock failure. Benchmark input/source bindings and longest-example gradient probes do not replace the affected production migration gates or authorize the old checkpoint under a new certificate.
