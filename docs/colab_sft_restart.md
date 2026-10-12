# Two-A100 7B SFT restart

The user explicitly requested restarting the ROT13 and KaziText 7B fine-tuning
queues on two Colab A100s. `sft_a100` owns ROT13 levels 1–10;
`sft_a100_kazi` owns KaziText levels 1–10. Each condition starts from the pinned,
untouched base model. This is a fresh restart, not continuation of the prior
step-987 ROT13 fit. Its full checkpoint remains archived and unchanged in Drive.

Both NVIDIA A100-SXM4-40GB runtimes have remote command access and the pinned
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
Both now have no intentional time limit. The legacy required deadline argument
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
