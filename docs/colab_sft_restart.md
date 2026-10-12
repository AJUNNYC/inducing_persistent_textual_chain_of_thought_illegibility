# Archived two-A100 restart and efficient per-fit redesign

## Active selection: two fresh 14B level-10 fits; awaiting GPU access

The default `efficient_one_epoch_v1` recipe processes **all 10,000 certified training examples once per selected fit**: 625 optimizer updates, a complete 625-update cosine schedule and 19 warmup updates. The selected launch scope is two independent fresh 14B fits: ROT13 level 10 and KaziText level 10. Both 7B/14B, ROT13/KaziText and all ten levels remain available in the forty-condition Drive-verified inventory. No GPU connection is available and training remains paused pending GPU access. See [the training redesign](sft_short_pilot.md).

The declared efficient baseline uses an unquantized frozen BF16 base and BF16 head, SDPA with efficient-kernel preference, direct full-gradient Cut Cross-Entropy (`cce_exact`) with FP32 loss and hidden-gradient accumulation, and plain AdamW with FP32 optimizer states. CCE is pinned to upstream commit `3de376c106a1916bc5e1b619f9c77c87a461ee1c`; `filter_eps=None`, `filter_e_grad=False` and `filter_c_grad=False` disable gradient filtering. No Qwen patch or full token-by-vocabulary logits matrix is used. Policy `published_bf16_lora_full_gradient_cce_v2` requests microbatch 2/accumulation 8. A non-updating duplicated-longest-example capacity guard tries 2/off, 1/off, 2/on, then 1/on, where on/off denotes activation checkpointing; only CUDA OOM permits fallback. The selected actual mode is recorded and cannot change during exact recovery.

Keep completion-only masks and per-example weighting, all seven LoRA targets (rank 8/alpha 16/dropout 0.05), learning rate `2e-4`, requested microbatch 2/accumulation 8/effective batch 16 (capacity fallback 1/16/16), seed 42, all 10,000/100 records, full 4096-token eligibility without truncation and verified Drive saves. Checkpoints occur at updates 100, 200, 300, 400, 500, 600 and final 625. Each update retains the same ordered 16 examples with per-example weight 1/16; execution grouping, padding, dropout draws and floating-point rounding may differ. No measured batching speedup or bitwise-equivalent outcome is claimed. This prospectively declared scientific baseline changes exposure, precision/head numerics, optimizer and execution; it does not prove identical final results. The original NF4 `legacy_two_epoch_v1` and intermediate NF4 `one_epoch_pilot_v1` remain explicit alternatives, with incompatible checkpoints and separate identities.

The user requested implementation without another optimization test or timing campaign. Normal native model/mask/gradient/full-state guards run during actual startup before any update; they have not passed for this redesign. Historical A100 comparisons ended without an accepted optimization, and 14B speed/memory remain unmeasured. Archived PRO step-987 and A100 update-28/30 states remain in Drive, never warm starts for this recipe. One epoch removes 50% of nominal example passes versus the same-condition two-epoch recipe; no one-hour finish or measured hardware speedup is promised. Diagnostics use the 100 certified validation IDs and base companions; final test panels and original confirmatory thresholds remain reserved. No additional training after the completed 625-update cosine horizon is an exact continuation of the old 1,250-update schedule.

## Historical two-A100 restart


Historical interruption: A100 comparisons were blocked. Both assignments ended, the
account reports 0.00 compute units, and an attempted A100 replacement was
rejected for quota or entitlement. Source `8a0f853` is published; no native
optimization acceptance or optimized official fit is complete. Baseline pause
states at ROT13 update 28 and KaziText update 30 remain in Drive.


The user explicitly requested restarting the ROT13 and KaziText 7B fine-tuning
queues on two Colab A100s. `sft_a100` owns ROT13 levels 1–10;
`sft_a100_kazi` owns KaziText levels 1–10. Each condition starts from the pinned,
untouched base model. This is a fresh restart, not continuation of the prior
step-987 ROT13 fit. Its full checkpoint remains archived and unchanged in Drive.

Before assignment loss, both NVIDIA A100-SXM4-40GB runtimes had remote command access and the pinned
Python 3.12/PyTorch 2.9.0+cu128 environment installed from the hash-locked GitHub
requirements. Missing Aspell development/system packages were installed before
retrying the unchanged setup. Setup skipped the already completed development
suites; the affected native launch controls remain mandatory.

The fresh queues use the published eager baseline: gradient checkpointing on,
FP32 head/logits/spatial cross-entropy, unchanged NF4/double quantization/BF16
compute, masks, per-example loss weighting, LoRA, optimizer, seed, 4096-token
limit, effective batch 16, and two epochs. The Blackwell-only optimized helpers
and the roughly 84-GiB no-checkpointing profile are not activated on the 40GB
A100s. No empirical speedup claim is made for the baseline hardware fallback.

`scripts/qualify_sft_baseline_runtime.py` runs actual native 7B generation,
loss/mask/gradient, and perturbed-full-state recovery controls on each selected
GPU without altering the full production configuration or exercising 14B.
Reconciliation verifies source, dependencies, production configuration, default
execution settings, measured model scope, and original scientific objective.
The resulting `ready_models`/`baseline_only` launch scope cannot authorize 14B,
an optimized profile, or optimizer-state migration. Existing compatible judge,
transformation, corpus, baseline-repeatability, and statistical evidence remains
reusable; no acceptance rule for an optimization is relaxed.

Each worker has separate per-family state, output directories and immutable
Drive checkpoint identities. Dataset downloads must match the certified Drive
index by hash and size. GPU state is disposable; complete adapter, optimizer,
scheduler, RNG and sampler checkpoints must be verified in Drive. Current
status: both native 7B control reports passed, all thirteen required launch gates
passed, and both family queues started on 12 October 2026 using source
`05f47f48f2cc2bbaaff326bb83b3594e5ae973c5`. Each completed its first optimizer
update and verified full state in Drive. A whole-update pause then exact recovery
removed an operator-imposed cutoff; neither queue inherits the old step-987 run.
Those attempts had no intentional time limit. Neither assignment remains active. The legacy required deadline argument
uses `9999-12-31T23:59:59.999999+00:00` as an operational sentinel. Colab resource
limits and explicit stop requests can still interrupt a runtime. No fit or final
validation is yet reported complete. Only 7B baseline execution is qualified;
14B and optimized A100 profiles remain outside these receipts.

## A100 optimization measurements

Both baseline queues were paused at complete updates (ROT13 28, KaziText 30),
with full state verified in Drive. The user requested A100-compatible speedups
and fresh optimized restarts. The candidate scheduled spatial CE retains the
original serial FP32 forward arithmetic and gradient checkpointing; its shared
hardware guard now permits SM80 measurements. The original Blackwell helper
bytes and historical acceptance scope remain preserved. A checkpointed SDPA
combination is a separate candidate, not an automatic production selection.

Candidate comparison keeps at least one warm-up and five measured updates in
each of two counterbalanced repetitions, unchanged loss/gradient/update bounds,
identical RNG and a minimum 1.05x speedup for every corresponding update pair.
Each model/family/level needs its own measured approval and longest-record
memory/gradient probe. Paused baseline checkpoints serve as isolated benchmark
evidence only; optimized official fits restart in a clean namespace from the
pinned base. Compatible provider/corpus/statistical evidence is reused. No
A100 optimization is claimed accepted before its actual measured gate passes.

`scripts/run_optimized_sft_queue.py` applies those gates separately at every
level, publishes full benchmark/launch evidence to Drive, and starts each
accepted fit from the untouched base. It never treats a level-1 timing result
as permission for other levels. Rejected candidates retain their failed evidence
and use the qualified baseline; acceptance thresholds are not relaxed.

The comparison setup exposed redundant full reads of already verified local
model shards. Native SFT now requests private local verification receipts:
first use retains full SHA-256 verification, and reuse requires the same bound
manifest and unchanged file path/device/inode/size/timestamps/owner/permissions.
Hub snapshots must resolve to the verified blob inodes. Changes trigger full
verification or a failure. This reduces setup work; it changes no model tensor,
loss, RNG operation or training setting. The strict uncached API remains the
default for callers that do not request receipts. Unchanged corpus handoffs may
also reuse certified identities after all 122 actual hashes and source/metadata
checks, avoiding repeated parsing of all 404,000 records.
