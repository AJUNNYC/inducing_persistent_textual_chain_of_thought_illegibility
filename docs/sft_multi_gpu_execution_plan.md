# Proposed SFT deployment across RunPod and Colab

## Current session: paused on 11 October 2026

At **7:19 PM EDT**, the accepted 7B ROT13 level-1 fit was safely paused at **step 987 of 1,250** through its whole-update handler. The [pause receipt](evidence/sft_pro6000_session_pause_987.json) verifies the full adapter, optimizer, scheduler, RNG and sampler checkpoint in Google Drive. There are 263 optimizer updates remaining; training and final validation are not complete. The Pod may be stopped through the RunPod console. Resume from the verified full checkpoint under the existing scientific identity; if the original deadline has elapsed, a new authorized deadline is required. This session pause supersedes earlier RUNNING snapshots without changing their historical evidence. No additional workers have been deployed.

Status: planning discussion recorded on 11 October 2026. The additional workers
have not been deployed or validated by this plan. The study remains 40 independent
fits: two model sizes, two transformation families and ten levels per family.
RL and final evaluation are outside the runtime estimates below.

## Measured execution versus proposed deployment

The accepted execution profile is archived source `a6e3295`,
`spatial_scheduled_ce_no_gradient_checkpointing`, scoped only to 7B ROT13 level 1
on the measured RTX PRO 6000 Blackwell 96GB. The
[accepted report](evidence/sft_scheduled_forward_pro6000_benchmark.json) and
[selection](evidence/sft_scheduled_forward_pro6000_selected_profile.json) retain
their original scope. The [final memory-copy trial](evidence/sft_memory_copy_pro6000_benchmark.json)
passed exact numerical/state comparisons but failed speed and remains rejected.
The [existing-lineage resume](evidence/sft_memory_copy_pro6000_training_resume.json)
preserved step 673 and the original deadline. These are dated evidence snapshots,
not a live optimizer-step counter. The accepted profile remains selected; the current session is paused as recorded above.

Optimization trials are closed. Allocating independent fits to additional GPU
sessions is a deployment proposal; it does not approve new kernels, hardware,
model sizes or transformation levels through the existing measurements.

## Four-worker allocation

The proposed workers operate concurrently, each with one GPU and an independent
adapter, optimizer, scheduler and sampler. Assign whole fits rather than changing
the physical or effective training batch. The current fit stays on its existing
worker until completion or a separately verified recovery.

| Proposed worker | Device | Conditions to assign |
|---|---|---|
| RunPod worker A | RTX PRO 6000 96GB | 14B ROT13 levels 1–10, after its current 7B fit finishes if using the existing Pod |
| RunPod worker B | RTX PRO 6000 96GB | 14B KaziText levels 1–10 |
| Colab worker A | A100, actual VRAM to be recorded | 7B ROT13 levels 2–10; level 1 remains owned by the current worker |
| Colab worker B | A100, actual VRAM to be recorded | 7B KaziText levels 1–10 |

The 39 proposed assignments plus the already owned 7B ROT13 level-1 fit cover
all 40 conditions exactly once. If the current fit is interrupted, recover its
verified full state under its existing ownership or a recorded migration; do
not independently restart it on another worker. Keep hardware/profile provenance
for every condition so comparisons do not conceal execution differences.

Use an explicit condition-to-worker assignment before launch. There is no
four-worker launch receipt or validated cross-provider job-claim mechanism in
this planning record. The existing [certified launcher](sft_data_handoff.md)
supports independent family/model queues and individual `--level` selection;
the certified worklist, source/configuration bindings and required controls still
apply. Deployment must exclude completed and currently owned conditions.

## Hardware and memory evidence

| 7B execution measured on PRO 6000 | Longest example | Peak allocated | Peak reserved |
|---|---:|---:|---:|
| Original checkpointed baseline | 3,190 tokens | 15.95 GiB | 21.12 GiB |
| Accepted scheduled CE without checkpointing | 3,190 tokens | 84.28 GiB | 87.32 GiB |

Sources: [checkpointed/recomputation report](evidence/sft_optimization_pro6000_checkpoint_recovery.json)
and [final reference measurements](evidence/sft_memory_copy_pro6000_benchmark.json).
These measurements cover this selected condition, not the longest example in
every transformation level or 14B. Allocator reservation is distinct from live
tensor allocation. Total device use also includes runtime overhead.

Re-enabling the original gradient checkpointing is a memory-saving candidate for
Colab. It preserves the scientific recipe but adds recomputation. A 32GB RTX
5090 or a 48GB RTX 6000 Ada is a budget candidate, not a tested replacement.
No controlled same-code H100/A100/5090 versus PRO 6000 speed comparison has been
completed. An H100 or A100 is not automatically faster for this FP32-head workload.

The optimized helpers currently require CUDA compute capability `(12, 0)` and
pinned Triton 3.5.0. A100 and Ada devices cannot silently reuse that profile.
The original native PyTorch loss remains available, but an A100 launch needs its
own compatible execution evidence, longest-record memory check, source-bound
launch controls and recovery proof where applicable. Another SM120 GPU is also
not automatically approved by the measured PRO 6000 receipt. No A100/14B memory
or training throughput result has been recorded here. Reuse compatible completed
data/provider/statistical evidence; do not repeat unrelated suites merely to add
workers.

## Runtime estimates and their assumptions

These are conversation planning estimates, not measured full-fit durations.
They assume 20 fits per model size, a planning budget of 2.5 hours per 7B fit and
5 hours per 14B fit, including approximate checkpoint overhead. The 14B rate is
unmeasured, and transformation length, checkpointing and hardware can alter it.

| Scenario | Planning wall time | Required scheduling assumption |
|---|---:|---|
| One PRO 6000 | 150 hours | 20 × 2.5 hours + 20 × 5 hours, sequential fits |
| Two PRO 6000s | 75 hours | Balance both model sizes between the workers; dedicating one to each size would leave the longer group as the bottleneck |
| Two PRO 6000s plus two Colab A100s | About 50 hours | The two PRO workers finish the estimated 100-hour 14B workload in about 50 hours, and the two A100 workers finish their 7B assignments within that time |

The four-worker estimate is not simply 150/4: A100 throughput is unmeasured and
the workloads differ. It is a conditional planning target, not a guaranteed
deadline. Extra session setup, availability and recovery interruptions are not
independently measured. The active worker's existing deadline is unchanged;
these forecasts do not extend it or launch additional paid sessions.

## Price inputs and cost per completed fit

The user quoted RTX 5090 pricing of **$1.19/hour** and Colab pricing of
**$9.99 per 100 compute units**. These are supplied planning inputs, not a live
price audit. Colab's actual A100 compute-unit consumption per hour remains unknown.
At that unit price, A100 hourly compute cost is `0.0999 × units_per_hour` dollars.
It beats $1.19/hour only below **11.91 units/hour**; for example, 13 units/hour
would cost $1.2987/hour. That example is conditional, not an observed Colab rate.

Compare hourly price multiplied by measured fit duration, then include applicable
storage, taxes and session overhead. A lower hourly price alone does not establish
a lower cost per completed fit. No current PRO 6000 price or new spending ceiling
is inferred by this plan.

## Shared Drive storage and unchanged science

Use the existing [central Google Drive project](https://drive.google.com/drive/folders/176a8Tm6l4WhfakY_0nCOo4ICeT7KPSmx)
for the certified corpus, source/configuration archives, full checkpoints,
adapters and provenance. Every worker must have distinct condition ownership and
output namespaces. Preserve each fit's configuration/source/hardware identity,
full adapter/optimizer/scheduler/RNG/sampler state and verified Drive records.
Local RunPod or Colab storage is a cache; it is not the permanent recovery source.
Credentials remain private and are excluded from Git and public documentation.

Keep the two epochs, 10,000/100 split, source traces, transformed targets, full
sequence limits, FP32 head/logits/loss, NF4/BF16 construction, per-example loss
weighting, masks, LoRA/dropout, optimizer/scheduler, seed, microbatch 1 and logical
batch 16 unchanged. Parallelism changes where independent fits run, not the
recipe or scientific success thresholds. Hardware-specific evidence and explicit
continuation provenance remain required before deployment is called ready.
