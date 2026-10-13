# SFT execution optimizations

## Current SFT status: 14B ROT13 level 10 complete; $240 authorized for remaining work

The default `efficient_one_epoch_v1` recipe processes **all 10,000 certified training examples once per selected fit**: 625 optimizer updates, a complete 625-update cosine schedule and 19 warmup updates. The first fresh fit, 14B ROT13 level 10, completed all 625 updates on 12 October 2026; KaziText 14B L10 was safely paused at 444/625 complete updates on 13 Oct 21:58 UTC, with full state verified in Drive. It is now on hold with no automatic restart. The user prioritized the nine remaining ROT13 levels 1–9; ROT13 level 1 is now training: native guards and actual checkpoint 0 recovery passed, initial full state was Drive-verified, and the live PID 6684 showed 98%/94% GPU activity at 22:05/22:06 UTC. Only checkpoint 0 is yet observed; do not infer an exact optimizer-update count before the next saved checkpoint. Both 7B/14B, ROT13/KaziText and all ten levels remain available in the forty-condition Drive-verified inventory. The active scope is now 14B only: ROT13 and KaziText across all ten levels, 20 conditions total. The completed adapter is retained, leaving 19 selected fits uncompleted; the 7B data and historical results remain available but outside the active training queue. On 13 October the user authorized $240 for SFT; the original 19-fit campaign started 13 Oct 20:58:33 UTC after the verified GPQA baseline and was safely paused for the user’s priority change; the current automatic queue has only nine ROT13 fits. The scientific worker remains source 9de8813 independently of this documentation commit. See [the training redesign](sft_short_pilot.md).

The declared efficient baseline uses an unquantized frozen BF16 base and BF16 head, SDPA with efficient-kernel preference, direct full-gradient Cut Cross-Entropy (`cce_exact`) with FP32 loss and hidden-gradient accumulation, and plain AdamW with FP32 optimizer states. CCE is pinned to upstream commit `3de376c106a1916bc5e1b619f9c77c87a461ee1c`; `filter_eps=None`, `filter_e_grad=False` and `filter_c_grad=False` disable gradient filtering. No Qwen patch or full token-by-vocabulary logits matrix is used. Policy `published_bf16_lora_full_gradient_cce_v2` requests microbatch 2/accumulation 8. A non-updating duplicated-longest-example capacity guard tries 2/off, 1/off, 2/on, then 1/on, where on/off denotes activation checkpointing; only CUDA OOM permits fallback. The selected actual mode is recorded and cannot change during exact recovery.

Keep completion-only masks and per-example weighting, all seven LoRA targets (rank 8/alpha 16/dropout 0.05), learning rate `2e-4`, requested microbatch 2/accumulation 8/effective batch 16 (capacity fallback 1/16/16), seed 42, all 10,000/100 records, full 4096-token eligibility without truncation and verified Drive saves. Checkpoints occur at updates 100, 200, 300, 400, 500, 600 and final 625. Each update retains the same ordered 16 examples with per-example weight 1/16; execution grouping, padding, dropout draws and floating-point rounding may differ. No measured batching speedup or bitwise-equivalent outcome is claimed. This prospectively declared scientific baseline changes exposure, precision/head numerics, optimizer and execution; it does not prove identical final results. The original NF4 `legacy_two_epoch_v1` and intermediate NF4 `one_epoch_pilot_v1` remain explicit alternatives, with incompatible checkpoints and separate identities.

The user requested implementation without another optimization test or timing campaign. Normal native model/mask/gradient/capacity/full-state guards passed for the completed 14B ROT13 level-10 fit on RTX PRO 6000 Blackwell 96GB. Actual execution was microbatch 1 / accumulation 16 with activation checkpointing off. The recorded fit interval was 5,793.143751 seconds (96m33s); worker launch through completion was 6,322.565654 seconds (105m23s). At $2.99/hour these imply $4.81 and $5.25, respectively, calculated from runtime rather than invoices. Other conditions remain unmeasured. Archived PRO step-987 and A100 update-28/30 states remain in Drive, never warm starts for this recipe. One epoch removes 50% of nominal example passes versus the same-condition two-epoch recipe; no one-hour finish or measured hardware speedup is promised. Diagnostics use the 100 certified validation IDs and base companions; final test panels and original confirmatory thresholds remain reserved. No additional training after the completed 625-update cosine horizon is an exact continuation of the old 1,250-update schedule.

See the [completion, provenance and budget record](sft_completed_rot13_14b_level10.md). GPQA Diamond (198 questions) is appended to the existing capability suite, matching the José et al. dataset variant; the actual untouched14B seed42 baseline completed49/198 (24.75%), with 133 truncated outputs and3 malformed answers under the unchanged3,072-token cap; all 198 count in the denominator. The full result is Drive-verified; trained-state GPQA remains unrun. It is a supplemental stress test and does not change the original four-domain criteria. The GPQA baseline remains verified. Current priority is ROT13 levels 1–9 ascending, each a fresh 625-update fit; the completed ROT13 L10 adapter is retained unchanged. KaziText is suspended and not scheduled automatically. Source 9de8813/config 44e9/certificate/data/recipe remain unchanged; a private operational lifecycle helper calls the published single-fit CLI with an explicit level. See the [priority/pause receipt](evidence/rot13_14b_priority_execution_2026_10_13.json). Actual ROT13 L1 native evidence and two busy GPU snapshots establish training activity; later saved checkpoints will establish exact progress.

## Archived interruption: A100 comparison stopped on 11 October 2026

Both Colab A100 assignments ended; the account reports 0.00 compute units and rejects a new A100 allocation. ROT13 and KaziText baseline attempts are safely paused at updates 28 and 30, with full state verified in Drive before runtime loss. Source `8a0f853` implements checkpointed A100 loss candidates, per-level empirical qualification and verified setup reuse. Native candidate comparisons remain incomplete: no optimized fit has started and no A100 speedup is approved. The user requested fresh optimized restarts from untouched bases; archived partial fits are benchmark evidence only. There is no user-imposed wall-clock deadline. See [the interruption record](evidence/a100_optimization_interruption.json) and [restart procedure](colab_sft_restart.md).

## Prior implementation and evidence scope

The following implementation details and dated measurements retain their original scope; the completed efficient fit and current authorization are described above.

The decision rule is to change execution in order of least expected effect on the
trained result, require established precedent, and measure correctness and speed
on the intended GPU before enabling a candidate. Source `a6e3295` now selects
`spatial_scheduled_ce_no_gradient_checkpointing` only for **7B ROT13 level 1**
on RTX PRO 6000 Blackwell Server Edition 96GB. It preserves native spatial FP32
forward arithmetic while changing token scheduling and stored row statistics.
Two paired repetitions measured an additional **24.70–24.98% less median update
time** versus the previously accepted spatial-forward/tiled-backward profile,
with bitwise losses, gradients, updates and final adapter/optimizer/scheduler/RNG agreement.
Earlier faster flat/fused forward candidates remain rejected. This short screen
does not approve other levels, 14B, final-result equivalence or a full-fit ETA.

The historical `legacy_two_epoch_v1` recipe used two epochs over the same 10,000 examples, logical batch
16, NF4 with double quantization, BF16 computation, FP32 cross-entropy, the same
LoRA targets and dropout, optimizer, scheduler, clipping, sequence limits and
completion masks. Execution optimization within either named recipe does not shorten responses, remove examples,
change the loss weighting, reorder the exposure sequence or reduce Drive saves.
Activation checkpointing is the original default. Disabling recomputation is an
explicit execution-profile override, permitted only after its measured screen
and production migration gates pass; it is not concealed as unchanged execution.

## Proposed worker allocation

The [canonical four-worker plan](sft_multi_gpu_execution_plan.md) records the
new scheduling and price discussion without reopening optimization trials. Two
PRO 6000 workers would handle the 14B families; two Colab A100 workers would
handle 7B ROT13 levels 2–10 and 7B KaziText levels 1–10. The currently owned 7B
ROT13 level-1 fit stays on its existing worker. Additional workers are not
deployed or validated, and no condition may be trained twice. The 150/75/about
50-hour forecasts are conditional estimates, not full-fit benchmarks. A100
compute-unit consumption and all A100/14B rates/capacity remain unmeasured;
the plan records quoted prices and the conditional 11.91-CU/hour break-even.
The checkpointed 7B baseline is a capacity candidate for Colab, whereas current
optimized helpers require Blackwell SM120 and pinned Triton 3.5.0. Distinct
immutable fit outputs remain in the shared Drive project. The current accepted
profile, full scientific recipe and forty-fit matrix are unchanged.

## Ordered candidates

| Order | Change | Evidence and effect on results | Status |
|---|---|---|---|
| 1 | Validate supervised-token counts on CPU and consolidate loss-finiteness checks before clipping and the optimizer update. | Removes redundant host decisions while retaining the same FP32 logits, cross-entropy, per-example denominator and reduction. Nonfinite losses still reject the whole update before either optimizer or scheduler advances. PyTorch recommends avoiding CUDA-dependent Python control flow. | Implemented and measured as `deferred_checks`. Actual GPU losses, gradients, adapter updates and final RNG agreed bitwise, but median speedup was only 1.0020x and failed the 5% speed screen. Not activated. |
| 2 | Cache this fit's exact collated input IDs, attention masks and labels as immutable CUDA tensors. | Removes repeated host preparation/transfers. Tensor contents, microbatch shapes, sample order and RNG remain unchanged; ownership and tensor-version guards reject mutation or substitution. | Measured as `cuda_input_cache` in source `05f42b0`. Paired numerical/state comparisons were bitwise identical, but the minimum matched-update speed ratio was 0.9692x, failing the 1.05x requirement. Not activated. CPU preloading and compact supervised-position CE remain future investigations. |
| 3 | Replace eager attention with built-in SDPA. | Transformers supports this for Qwen2 and recommends it for efficient training. It retains the attention definition, but fused floating-point kernels can produce different numerical results. | Implemented and measured as `sdpa`. Median speedup was 0.9937x; the numerical screen failed. No fused kernel was independently observed. Not activated. |
| 4 | Disable activation checkpointing if worst-case examples fit in VRAM. | Removes recomputation after identical PEFT preparation, base freezing, parameter casting and LoRA setup. Original protocol page29 calls checkpointing a memory-saving execution setting, not a behavioral intervention. Compare actual losses, gradients, updates and RNG. | Accepted as `no_gradient_checkpointing` for source `05f42b0`, RTX PRO 6000 Blackwell 96GB, 7B ROT13 level 1. Two paired repetitions measured 1.2695x and 1.2806x median throughput ratios, with bitwise numerical/state/RNG agreement. Other conditions retain the original checkpointing default. |
| 5 | Flatten the token CE input from spatial `[batch, vocabulary, tokens]` to `[batch*tokens, vocabulary]`. | PyTorch supports both CE layouts, but real GPU measurements show that their floating-point results differ beyond the frozen bound before the first update. | Source `6d21fff` completed two paired repetitions. Native/flat median seconds were 9.4263/5.8296 and 9.5075/5.6417. Faster timings did not clear the numerical screen; rejected and not activated. |
| 6 | Fuse only FP32 token cross-entropy with the pinned Triton runtime. | Liger provides online softmax/CE precedent, but changing the forward normalization order exceeded the frozen numerical bounds on this workload. | Source `6d21fff` measured median ratios 1.6751x and 1.7008x, but maximum loss difference was 0.00595903 and gradient/update relative L2 differences were 2.169%/1.922%, above 0.1% bounds. Rejected and not activated. |
| 7 | Retain stock spatial CE forward and tile only its backward gradient. | Keeps `torch.log_softmax(..., dim=1)` and unreduced `F.nll_loss`; the tiled backward follows the pinned PyTorch epilogue with libdevice exponential and the one nonzero target gradient. | Source `218900d` accepted for 7B ROT13 level 1 on the measured Blackwell96GB GPU. Two paired median ratios were 1.1564x/1.1318x, minimum matched ratio 1.1146x; losses, gradients, updates and final full state/RNG were bitwise identical. |
| 8 | Schedule the same spatial CE forward over smaller token blocks and retain only target losses/row statistics. | Follow pinned PyTorch's serial FP32 class accumulation and libdevice exp/log sequence; reconstruct log-probabilities during the already validated tiled backward. Scheduling and storage change rather than intentionally changing normalization order. Actual equality must still be checked. | Source `a6e3295` accepted for the selected Blackwell 7B ROT13 level-1 condition. Direct median times fell 8.9243→6.7200 and 8.8950→6.6727 seconds; bitwise numerical/state/RNG agreement, minimum matched ratio 1.31995x. |
| 9 | Write the complete causal LM-head gradient directly in contiguous token/vocabulary layout. | Move the causal shift inside custom autograd, reuse the accepted scheduled forward, and write the full `[batch,tokens,vocabulary]` gradient with its unused last row zeroed. This targets separate slice-backward and gradient-layout copies while preserving the same tensor values and arithmetic. Layout can still affect downstream execution, requiring measurement. | Source `07f8a68` passed 359 affected CPU checks and both numerical/state comparisons, but failed the speed gate: median time improved 1.93% in one pair and was 0.17% slower in the other; minimum matched ratio 0.99834x versus required 1.05x. Rejected; accepted source `a6e3295` and its existing lineage remain selected. |
| 10 | Increase physical microbatch while keeping logical batch 16. | Official guidance favors larger physical batches over excessive accumulation. However, batching changes floating-point grouping, padding and LoRA dropout RNG assignment even when the logical batch is unchanged. | Implemented in the new published policy v2: request 2 × 8, capacity fallback 1 × 16; preserve the same 16 examples and per-example weighting. Historical NF4 recipes remain unchanged. No batching speedup measured. |

Epoch count, training exposure, BF16 versus NF4 construction, optimizer precision,
learning rate, LoRA rank/dropout and effective batch size are scientific recipe
changes, not execution optimizations. Fusing the LM-head projection with CE and
compilation remain later investigations. The new token-CE candidate retains the
existing full LM-head projection and FP32 logits; it does not introduce either
of those additional changes.
Published speedups from other models are not promises for this workload.

The objective is **mean supervised-token loss within each example, then mean
across examples**. Stock model loss or token-global averaging must not silently
replace it. A faster implementation must preserve the current weighting, causal
shift and assistant-end supervision.

## Final causal contiguous-gradient trial — rejected

The final memory-copy trial used source `07f8a68` and
`causal_contiguous_gradient_ce_no_gradient_checkpointing`, with the accepted
scheduled spatial-CE profile as its reference. The
[step-673 pause](evidence/sft_memory_copy_pro6000_pause_673.json) preserved the
existing source `a6e3295` fit in Drive. Each isolated profile restored the same
adapter, optimizer, scheduler, sampler and RNG; the trial never advanced the
production checkpoint. Two counterbalanced pairs each used one warmup plus
five measured updates over 96 ordered examples.

| Repetition | Reference median seconds | Candidate median seconds | Median time change |
|---|---:|---:|---|
| 1 | 6.91222 | 6.77876 | 1.93% less time |
| 2 | 6.91286 | 6.92438 | 0.17% more time |

The [final report](evidence/sft_memory_copy_pro6000_benchmark.json) records
bitwise equality for every measured loss, gradients, adapter updates, final
adapter, optimizer, scheduler and RNG in both pairs. The candidate failed
**only the frozen speed requirement**: minimum matched ratio 0.99834x versus
at least 1.05x on every matched update. It is rejected and has not replaced the
accepted scheduled spatial-CE implementation. Peak allocated memory did not
improve: both longest 3,190-token probes were approximately 84.28 GiB, with the
same loss, 0.37006774545. A different allocator reservation does not establish a
lower peak tensor-memory requirement. No benefit is inferred from the intended
copy removal.

The [helper](../src/training/sft/causal_gradient_loss.py) moves the causal logit
shift inside custom autograd, reuses accepted scheduled forward with its
coalesced input copy, and writes the full contiguous token/vocabulary gradient
directly in 32-class by 32-time tiles, zeroing only the unused last row. This
removes the separate slice-backward full-buffer/scatter and gradient-layout
copy while preserving the head/GEMM, vocabulary, masks, per-example objective,
FP32 arithmetic and RNG. The measured agreement supports that implementation's
numerical behavior; it did not produce the required speed gain.

[PyTorch's pinned transpose documentation](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/torch/_torch_docs.py)
states that a strided transpose shares underlying storage.
[Its contiguous documentation](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/torch/_tensor_docs.py)
describes a contiguous tensor containing the same data, and
[Triton's pinned matrix tutorial](https://raw.githubusercontent.com/triton-lang/triton/v3.5.0/python/tutorials/03-matrix-multiplication.py)
demonstrates explicit multi-dimensional pointer/stride arithmetic. These were
precedent for the storage change, not workload-specific performance evidence.
The actual paired report retains the failed speed result rather than treating
source-level copy removal as proof of improvement.

A total of 359 affected CPU checks passed; no unrelated suite, provider,
statistical simulation or full matrix was repeated. Report hash
`cb7afe170ab23f0c1295b75748a5a9d370964427ae4228840e160a3e71b40997`
remains preserved. The accepted profile is still archived source `a6e3295`;
the [actual resume](evidence/sft_memory_copy_pro6000_training_resume.json)
records the worker RUNNING from restored step 673 via `--resume-existing` in the
**same existing lineage**, with passed native gradients, verified Drive
recovery and the original deadline unchanged. Original and frozen step-673
checkpoint hashes agree. No later completed optimizer step is claimed by this
startup observation. The older step-509 migration record remains historical;
there is no new migration to the rejected source. Parents at steps 214, 401,
509 and 673 remain unchanged, and all scientific settings are retained.

Optimization rounds are closed; no further candidates are scheduled. Deferred
entries are an inventory, not additional planned optimization trials. The full
study remains exactly 40 SFT fits; current approval still covers only 7B ROT13
level 1 on the measured Blackwell GPU. The accepted profile remains selected; the current session is paused.
The rejected copy-removal candidate is not activated.

## Current scheduled spatial-forward profile

The [step-509 pause](evidence/sft_scheduled_forward_pro6000_pause_509.json)
preserves the previously accepted source `218900d` fit's full state in Drive.
The [new benchmark](evidence/sft_scheduled_forward_pro6000_benchmark.json)
restored that same adapter, optimizer, scheduler, sampler and RNG into every
isolated run. It used counterbalanced reference/candidate/candidate/reference
order, forming two paired repetitions, each one warmup plus five measured
logical updates and 96 ordered examples. The declared reference was
`spatial_fused_backward_no_gradient_checkpointing`: the gain below is additional
to that profile, not a fresh comparison against the original checkpointed fit.

| Repetition | Reference median seconds | Scheduled CE median seconds | Throughput ratio | Less median update time |
|---|---:|---:|---:|---:|
| 1 | 8.92434 | 6.71998 | 1.32803x | 24.70% |
| 2 | 8.89501 | 6.67273 | 1.33304x | 24.98% |

Every matched update passed the frozen 5% improvement requirement; the minimum
ratio was 1.31995x. All 96 losses in each repetition were bitwise identical,
as were clipped-gradient and adapter-delta hashes, final adapter, optimizer,
scheduler and RNG states. The longest 3,190-token probe passed with the same
loss, 0.38299718499, and the same 84.28 GiB allocated/87.32 GiB reserved peak as
its reference. This is direct measured evidence for the selected continuation;
no memory improvement or final-result identity is inferred. Comparisons against
earlier screens use different checkpoint states and sampled update sequences,
so combined historical speed estimates are extrapolations rather than a direct
fresh paired measurement.

The [scheduled helper](../src/training/sft/scheduled_spatial_loss.py) preserves
pinned PyTorch's class-index order for serial FP32 accumulation, the same
max/exp/sum/log and target-loss arithmetic, and libdevice exp/log operations.
It schedules 64 tokens per CUDA thread block, saves target losses plus row statistics, and
reconstructs `input - max_input - logsum` in the validated tiled backward.
For shifted sequences of at most 64 tokens, it explicitly uses original
spatial forward, whose class reduction can use multiple threads. This is a
prospective shape route, not exception-driven fallback. Unsupported execution
fails rather than silently choosing another kernel. Full FP32 head/logits,
vocabulary, causal masks, per-example weighting and RNG behavior stay fixed.
The [pinned PyTorch spatial kernel](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/aten/src/ATen/native/cuda/SoftMax.cu)
is algorithmic precedent; the actual paired report separately validates this
implementation on the selected workload and GPU.

The [accepted profile](evidence/sft_scheduled_forward_pro6000_selected_profile.json)
and [READY affected controls](evidence/sft_scheduled_forward_pro6000_readiness.json)
bind source `a6e3295`, the exact GPU and this one condition. A total of 236
focused CPU checks passed; unrelated suites, provider and statistical checks
were not repeated. Report hash
`63711f97c0d29bd9d3672e961399f62112ecce60064244f9dab7b4e51ed7d162`
and selection hash
`d4c8aa552c2e67ea989d63ecff85b7f7866468caef4aac494aa864b42f85877e`
remain bound separately from documentation commits. The benchmark itself
retains `approved_for_production=false` and
`scientific_equivalence_established=false`; the shared validator performs
condition-scoped selection rather than automatic benchmark activation.
The [actual startup](evidence/sft_scheduled_forward_pro6000_training_resume.json)
records source a6e3295 training RUNNING from restored step 509. Native completion
mask/adapter-gradient checks passed, the selected helper hash agrees, and child
state is verified in Drive. The observation does not claim a later completed
optimizer step; the existing deadline remains unchanged. Parent checkpoints at steps 214, 401 and 509
remain unchanged. Science, exposure order and Drive-save cadence are unchanged.

## Historical recomputation continuation screen

Source `05f42b0` adds explicit cache and recomputation controls while preserving
eager attention, NF4/BF16, original finite-loss checks, physical microbatch 1,
logical batch 16 and the optimizer recipe. Sixty-nine focused checks passed; the
full suite was not repeated. Cache tensors are prepared once for the selected
fit and checked against their exact tokenized example, device, shape and version.
No other condition may reuse that fit's cache.

The [actual GPU recovery screen](evidence/sft_optimization_pro6000_checkpoint_recovery.json)
used two paired repetitions, each with one warmup
and five measured logical updates per profile. A continuation comparison restores
the same verified step 214 adapter, optimizer, scheduler, sampler and RNG into
each isolated profile; the official checkpoint remains read-only. The frozen
speed rule requires at least 5% improvement on every corresponding measured
update pair, alongside numerical and identical-RNG requirements. A combined
cache/no-recomputation profile also requires the separate cache and recomputation
comparisons and was not selected after the cache candidate failed.

| Candidate | Baseline / candidate median seconds, repetitions 1 and 2 | Minimum matched-update speed ratio | Decision |
|---|---|---:|---|
| CUDA input cache | 11.9242 / 11.7858; 11.9491 / 12.0956 | 0.9692x | Bitwise state agreement; failed speed gate |
| No activation recomputation | 11.9095 / 9.3814; 11.9435 / 9.3264 | 1.2344x | Passed numerical, RNG and speed gates |

For the accepted candidate, the median throughput ratios were 1.2695x and
1.2806x, corresponding to about 21–22% less measured update time. Every matched
update met the frozen 5% improvement rule. Each paired 96-example continuation
sequence had identical losses, clipped-gradient and adapter-delta hashes, final
adapter, optimizer and scheduler hashes, and final RNG state. This is direct
evidence for the measured continuation, not a guarantee of identical final model
behavior or a full-fit ETA.

The longest native example in this condition was 3,190 tokens and passed the
actual finite-loss/nonzero-adapter-gradient probe. Without recomputation its
probe allocated 86.09 GiB and reserved 89.13 GiB of the GPU's 94.97 GiB; memory
headroom is narrow. This result does not establish that 14B or other levels fit.
Eager attention, original loss checks, physical microbatch 1, accumulation 16,
and every scientific recipe field remain unchanged. No attention-kernel change
or combined cache profile was selected.

The [accepted profile](evidence/sft_optimization_pro6000_selected_profile.json)
has acceptance hash
`578a7eed0b174017882649a3fb0d0c0d831866012e94bbbb0a1f861325c4d963`
and binds the unmodified report hash
`11c055bf2edf87738d0e31eb24eb7cbb9a79279670679a797fa3738401a39417`.
The [historical readiness receipt](evidence/sft_optimization_pro6000_readiness.json)
is `READY` for this one condition after affected GPU controls; the entire matrix
and statistical simulations were not repeated. The benchmark itself retains
`approved_for_production=false`: selection is a separate hash-validated decision,
not automatic benchmark approval. The [verified Drive index](evidence/sft_optimization_pro6000_recovery_drive_index.json)
preserves source, inputs, reports and launch evidence in the
[recovery folder](https://drive.google.com/drive/folders/1MPZAIU9kj53i3uzFxhXG35OCtdMXuiYj).

## Historical forward and backward trials

The [step-401 pause](evidence/sft_loss_optimization_pro6000_pause_401.json)
preserves the source `05f42b0` fit's full state in Drive. All loss trials restore
that same adapter, optimizer, scheduler, sampler and RNG; isolated measurements
do not advance the official parent. The declared reference is
`no_gradient_checkpointing` with original `spatial_ce`, so measured gains are
incremental to the previous recomputation optimization. Each comparison uses
two paired repetitions of one warmup plus five measured logical updates,
96 ordered examples per profile/repetition, with the frozen numerical and speed
bounds. Every matched update must improve by at least 5%; relative L2 gradient
and update differences must each stay within 0.1%. Loss tolerance is
`1e-5 + 1e-4*abs(reference_loss)`, with identical-RNG requirements retained.

| Candidate and source | Native / candidate median seconds, repetitions 1 and 2 | Numerical outcome | Decision |
|---|---|---|---|
| Flat CE, `6d21fff` | 9.4263 / 5.8296; 9.5075 / 5.6417 | Max loss difference 0.00435412; gradient L2 2.360%; update L2 1.804% | Rejected |
| Fully fused token CE, `6d21fff` | 9.4186 / 5.6228; 9.5238 / 5.5995 | Max loss difference 0.00595903; gradient L2 2.169%; update L2 1.922% | Rejected |
| Stock spatial forward / tiled backward, `218900d` | 9.4654 / 8.1854; 9.4791 / 8.3750 | Bitwise losses, gradients, updates, final adapter/optimizer/scheduler and RNG | Accepted for selected condition |

The [rejected forward-candidate report](evidence/sft_loss_optimization_pro6000_parallel_ce.json)
retains both failures. Each candidate reproduced its own loss sequence across
repetitions, but both differed from native before the first optimizer update:
0.5324794054 versus 0.5325533152, absolute difference 0.0000739098 against an
allowed 0.0000632479. Flat CE exceeded the loss bound on 86/96 examples and
fused CE on 87/96. Identical starting weights/data/RNG and faster timing did not
override these failures. Preserving FP32 dtype and the mathematical formula was
insufficient to preserve native floating-point results.

The [accepted spatial-backward report](evidence/sft_loss_optimization_pro6000_spatial_backward.json)
records median throughput ratios 1.15637x and 1.13183x: **13.52% and 11.65% less
median update time**, respectively. The minimum matched ratio was 1.11456x,
about 10.28% less update time. Every paired loss and gradient/update comparison
was bitwise equal, as were final adapter, optimizer, scheduler and RNG hashes.
This establishes the measured continuation behavior, not final-result
identity or a full-fit ETA. The actual longest 3,190-token forward/backward
probe passed, allocating 84.28 GiB and reserving 87.32 GiB, versus 86.09 GiB
allocated for its no-recomputation reference. Memory evidence remains scoped
to this condition on the measured 94.97-GiB device.

`loss_implementation` explicitly supports `spatial_ce`, `flat_ce`, `fused_ce`
and `spatial_fused_backward`. The selected
[spatial helper](../src/training/sft/spatial_loss.py) retains actual stock
`torch.log_softmax(..., dim=1)` and unreduced `F.nll_loss` forward. Its tiled
backward follows `gradOutput - exp(log_probability)*summed_grad`, using libdevice
exponential and the single nonzero target gradient per supervised token. It
retains FP32 head/logits/CE, all vocabulary classes, causal shift, original masks
and per-example weighting, with no extra RNG or hidden fallback. Locked Triton
3.5.0 supplies the existing runtime; no extra package was installed for it.

[PyTorch 2.9.0 SoftMax.cu](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/aten/src/ATen/native/cuda/SoftMax.cu)
contains distinct spatial reduction paths, including serial class accumulation
when `blockDim.x == 1`, and the stated backward epilogue. That source explains
why layout changes can alter arithmetic across the 152,064-class vocabulary;
no independent GPU kernel-dispatch observation is claimed. Retaining stock
forward plus using that backward formula supplies precedent; the actual paired
GPU report separately supplies workload evidence.

The shared validator [selected this profile](evidence/sft_loss_optimization_pro6000_selected_profile.json)
with source/GPU/condition bindings; [historical affected launch controls](evidence/sft_loss_optimization_pro6000_readiness.json)
are READY. A total of 165 affected CPU checks
passed in 4.08 seconds; the full unrelated suite, matrix and statistical
simulations were not repeated. Benchmark report flags remain
`approved_for_production=false` because separate validated selection performs
activation. The selected receipt binds report hash
`f3252b23ceb80bf7836e7eca18477940b1d2a8224851fc4eba7a05844c3a1df2`
and acceptance hash
`78d5dcee6af0235ac8ef55744caaf1f15959e6ab45c559172749df0f2019091a`.
The [actual continuation startup](evidence/sft_loss_optimization_pro6000_training_resume.json)
records the worker running from restored step 401, with the selected runtime
profile, passed completion-mask/adapter-gradient checks and Drive-verified child state.
The launch observation does not claim a later completed optimizer step.
Original step-214 and step-401 parent bytes remain preserved under separate
lineage. Dataset bytes, scientific settings and Drive-save cadence are unchanged.

## What the empirical checks establish

The focused CPU fixtures exercise the actual loss and optimizer loop, including
unequal completion lengths, masked prompt/padding positions, dropout RNG,
corrupted count metadata, and nonfinite losses. Bitwise agreement on those
fixtures supports the first candidate's arithmetic and failure behavior. It does
not establish CUDA equality, RTX performance or identical final scientific
outcomes.

[Actual CPU audit](evidence/sft_optimization_cpu_audit.json) records 36 passing
focused tests and 31 fewer host scalar extractions per 16-example update (36
versus 5). The [earlier plan-only receipt](evidence/sft_optimization_benchmark_plan.json)
retains its original pending GPU/consumer bindings as historical planning evidence.

### Historical first RTX PRO 6000 comparison

The [measured receipt](evidence/sft_optimization_pro6000_benchmark.json) binds
source `fd23db9`, the pinned 7B ROT13 level-1 corpus, and the actual Blackwell
Server Edition GPU. Packages were Torch 2.9.0 with CUDA 12.8, Transformers 4.57.3,
PEFT 0.18.0 and bitsandbytes 0.48.2. Each fresh profile used the same first 48
records in frozen exposure order: one warmup and two measured logical updates.
All profiles also passed the longest native 3,190-token example's actual finite
loss and nonzero-adapter-gradient probe.

| Profile | Median seconds per update | Baseline/candidate speed ratio | Outcome |
|---|---:|---:|---|
| Baseline eager | 11.89899 | 1.0000x | Comparison reference |
| Deferred checks | 11.87521 | 1.0020x | Bitwise losses, gradients, adapter updates, final adapter and RNG; only 0.2% faster, failing the frozen 5% speed requirement |
| SDPA | 11.97400 | 0.9937x | 0.63% slower; numerical screen failed |

SDPA's maximum absolute loss difference was 0.004179, relative gradient L2
difference 0.064823 and relative adapter-update L2 difference 0.060809. This is a
failed execution comparison, not evidence of worse final model capability or an
invalid SDPA algorithm. The selected SDPA kernel was not independently observed,
so the receipt does not establish that FlashAttention ran.

Both candidate technical gates are `FAILED`; exit code 3 reports the rejected
screen rather than a crashed training worker. Production approval remains false.
The first attempt, `gpu_benchmark_01`, stopped on a cache lock before GPU
measurement and is retained separately. The [actual measurement reports](https://drive.google.com/drive/folders/1MsTcGg-78j-Wg6O7pxKc8EwhGBLBDy73)
and the earlier failed attempt have verified copies in the central project, with
both folders recorded in the [preservation index](evidence/sft_optimization_pro6000_drive_index.json).
The official step-214 state was neither read nor modified by these fresh runs.

The isolated benchmark in
[benchmark_sft_optimizations.py](../scripts/benchmark_sft_optimizations.py) runs
the baseline and each candidate in separate processes from the same pinned base.
Fresh comparisons share a seed; continuation comparisons restore the same
verified full checkpoint. It uses certified production records, whole logical updates,
unchanged rendering and masks, warmup, synchronized GPU timing, peak memory,
losses, adapter gradients and parameter updates. It never advances or replaces
the paused production fit. The plan freezes comparison bounds before running;
reports retain failures as failures and do not automatically approve a candidate
for production. A short numerical comparison does not prove that final trained
outputs will be identical.

To reproduce the archived final screen, use its recorded source and certified
inputs with a new output directory. No further trial is scheduled. This
planning command does not run GPU work:

```bash
python scripts/benchmark_sft_optimizations.py \
  --config CURRENT_CONFIG.json \
  --certificate CURRENT_CERTIFICATE.json \
  --bundle CERTIFIED_BUNDLE.json \
  --worklist CURRENT_WORKLIST.json \
  --model 7b --family rot13 --level 1 \
  --reference-profile spatial_scheduled_ce_no_gradient_checkpointing \
  --profiles spatial_scheduled_ce_no_gradient_checkpointing causal_contiguous_gradient_ce_no_gradient_checkpointing \
  --warmup-updates 1 --updates 5 --repetitions 2 \
  --recovery-checkpoint VERIFIED_CURRENT_SCHEDULED_CE_CHECKPOINT_DIRECTORY \
  --output NEW_BENCHMARK_DIRECTORY --plan-only
```

Reproduction of the archived comparison uses the same command on its recorded
GPU with a fresh output directory and without `--plan-only`. This is a
reproducibility interface, not a new scheduled optimization. CPU timing does
not substitute for the GPU comparison. Hardware details and actual
dependency versions must accompany the report; an SDPA setting alone does not
prove a fused kernel was selected.

## Preserved parent fit and migration

The [earlier no-recomputation launch](evidence/sft_optimization_pro6000_training_resume.json)
records startup from inherited step 214 with a Drive-verified child
checkpoint. The child records source `05f42b0`, the accepted profile and hardware
migration; the parent checkpoint remains unchanged. The launch snapshot does
not claim a later completed optimizer step. Child serialized checkpoint bytes
may differ because lineage and execution metadata are updated; exact inherited
adapter, optimizer, scheduler, sampler and RNG state is the preserved contract.

The verified step-214 checkpoint, original source, configuration and Drive
evidence remain preserved. A hardware, source or execution-profile change must
be recorded as a migration; it must not masquerade as the original exact resume
identity. Restore full adapter, optimizer, scheduler, sampler and RNG state when
resuming an accepted continuation, and document any numerical execution change.

A new consumer source needs a current handoff certificate and evidence for the
affected GPU/runtime controls. The launcher requires accepted, hash-verified
source/hardware/condition/profile-bound benchmark evidence before selecting an
optimization. A continuation additionally requires the original pause receipt,
verified full checkpoint and paired recovery measurements. Migration creates a
separate lineage record and consumer checkpoint; it preserves parent bytes and
inherits full training state without repeating completed presentations. This is
an explicitly declared continuation, not an exact resume under the old identity.
Unchanged corpus, transformation, judging and statistical evidence can be reused
with explicit compatibility bindings. Google Drive remains the permanent store;
unrelated suites and datasets need not be repeated.

Verified reports and the integrated protocol are preserved in the central project at
[reports/sft_optimizations/94e2fb7](https://drive.google.com/drive/folders/1p2MxgaoO4utvdMEO_D6ijH4g8vrfgjN4).

## Authoritative precedent

- [PyTorch performance tuning guide](https://raw.githubusercontent.com/pytorch/tutorials/main/recipes_source/recipes/tuning_guide.py): “Avoid unnecessary CPU-GPU synchronization”; examples include `cuda_tensor.item()` and CUDA-dependent Python control flow such as `if (cuda_tensor != 0).all()`. It also recommends asynchronous loading and pinned-memory transfers. These are mechanisms to measure, not workload-specific speed guarantees.
- [Transformers v4.57.3 GPU guide](https://raw.githubusercontent.com/huggingface/transformers/v4.57.3/docs/source/en/perf_train_gpu_one.md): recommends data preloading, larger physical batches, and SDPA; describes checkpointing as trading memory for “slower training speed (~20%).”
- [PyTorch v2.9.0 CE documentation](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/torch/nn/modules/loss.py): `ignore_index` “does not contribute to the input gradient.” This permits investigating omitted masked-row loss work, without discarding context or vocabulary classes.
- [Liger v0.4.0 CE implementation](https://raw.githubusercontent.com/linkedin/Liger-Kernel/v0.4.0/src/liger_kernel/ops/cross_entropy.py): documents “Online softmax: 2 loads + 1 store” and a second pass computing gradients. This is algorithmic precedent for the independent FP32 token-CE helper; its original reduction and storage choices are not imported into this fit. [Triton's v3.5.0 fused-softmax tutorial](https://raw.githubusercontent.com/triton-lang/triton/v3.5.0/python/tutorials/02-fused-softmax.py) supplies further fusion precedent. Neither source empirically validates this candidate on the present training workload.
- [Qwen2 v4.57.3 implementation](https://raw.githubusercontent.com/huggingface/transformers/v4.57.3/src/transformers/models/qwen2/modeling_qwen2.py): declares SDPA support and provides selected-position `logits_to_keep`. Changing LM-head projection shape remains a separate candidate requiring empirical comparison.
- [PyTorch v2.9.0 SDPA documentation](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/torch/nn/functional.py): fused floating-point operations can make output differ with the selected backend. [Numerical accuracy guidance](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/docs/source/notes/numerical_accuracy.rst) likewise warns that batched and individual computations need not be bitwise identical.
- [PyTorch v2.9.0 checkpoint documentation](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/docs/source/checkpoint.md): RNG preservation makes dropout outputs agree with non-checkpointed passes. Keep `preserve_rng_state=True`; [non-reentrant checkpointing is recommended](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/torch/utils/checkpoint.py) and already used.
- [PEFT v0.18.0 preparation](https://raw.githubusercontent.com/huggingface/peft/v0.18.0/src/peft/utils/other.py): preparation freezes the base and casts nonquantized BF16/FP16 parameters to FP32. The recomputation candidate performs identical preparation first, then disables only activation checkpointing. The [original protocol page29](../spec/experiment.pdf#page=29) describes that setting as a memory-saving execution choice.
- [PyTorch transfer tutorial](https://raw.githubusercontent.com/pytorch/tutorials/main/intermediate_source/pinmem_nonblock.py): calling `pin_memory()` synchronously immediately before transfer can be slower. Preloading requires a measured implementation rather than adding that call to the hot loop.
- [Liger v0.4.0](https://raw.githubusercontent.com/linkedin/Liger-Kernel/v0.4.0/README.md): reports throughput and memory improvements and convergence testing, with benchmark conditions including LLaMA3-8B BF16/FSDP on eight A100s. This is empirical precedent for later fused-kernel investigation, not direct validation of single-GPU QLoRA here.
- [bitsandbytes 0.48.2 installation table](https://raw.githubusercontent.com/bitsandbytes-foundation/bitsandbytes/0.48.2/docs/source/installation.mdx): Linux x86-64 CUDA 12.8–12.9 builds include `sm120`. This supports the Blackwell compatibility investigation; it does not establish training throughput or replace actual hardware checks.

No published source above directly measures this frozen 7B/14B recipe on an RTX
PRO 6000 Blackwell 96GB. The actual short 7B comparisons now supply local empirical
evidence: the earlier two candidates failed, while the paired step-214 recovery
screen accepted disabling recomputation, and the later step-401 screen accepted
only stock spatial forward with tiled backward for the same selected condition. They
do not certify the full 7B/14B matrix, final-result equivalence or an
H100-versus-Blackwell training-time ratio. The current source a6e3295 readiness receipt separately
binds the declared continuation; it does not change the parent's identity.
