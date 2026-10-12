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
status: pinned setup complete, certified data staging and affected runtime
qualification in progress; training has not yet started.
