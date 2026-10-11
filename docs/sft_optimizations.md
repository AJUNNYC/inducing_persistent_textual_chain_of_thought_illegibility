# SFT execution optimizations

The decision rule is to change execution in order of least expected effect on the
trained result, require established precedent, and measure correctness and speed
on the intended GPU before enabling a candidate in production. The current
production default remains eager attention with the original loss checks. An
isolated comparison on RTX PRO 6000 Blackwell Server Edition 96GB measured both
candidates; neither cleared the frozen technical screen, so neither is activated.
The short comparison does not establish full-fit throughput or training duration.

The frozen recipe remains two epochs over the same 10,000 examples, logical batch
16, NF4 with double quantization, BF16 computation, FP32 cross-entropy, the same
LoRA targets and dropout, optimizer, scheduler, clipping, sequence limits and
completion masks. Optimization does not shorten responses, remove examples,
change the loss weighting, reorder the exposure sequence or reduce Drive saves.

## Ordered candidates

| Order | Change | Evidence and effect on results | Status |
|---|---|---|---|
| 1 | Validate supervised-token counts on CPU and consolidate loss-finiteness checks before clipping and the optimizer update. | Removes redundant host decisions while retaining the same FP32 logits, cross-entropy, per-example denominator and reduction. Nonfinite losses still reject the whole update before either optimizer or scheduler advances. PyTorch recommends avoiding CUDA-dependent Python control flow. | Implemented and measured as `deferred_checks`. Actual GPU losses, gradients, adapter updates and final RNG agreed bitwise, but median speedup was only 1.0020x and failed the 5% speed screen. Not activated. |
| 2 | Prepare immutable CPU tensors and transfers in advance; investigate compact supervised-position cross-entropy. | Preloading has official precedent. Ignored CE targets contribute no gradient, so their loss work can be omitted mathematically. Keep all prompt tokens in model attention and retain a full-vocabulary denominator for each supervised token. A changed reduction or projection shape can change rounding. | Investigations only; neither is enabled in production. Require identical inputs/masks/order and empirical loss and update comparisons. |
| 3 | Replace eager attention with built-in SDPA. | Transformers supports this for Qwen2 and recommends it for efficient training. It retains the attention definition, but fused floating-point kernels can produce different numerical results. | Implemented and measured as `sdpa`. Median speedup was 0.9937x; the numerical screen failed. No fused kernel was independently observed. Not activated. |
| 4 | Disable activation checkpointing if worst-case examples fit in VRAM. | Removes recomputation. Transformers documents a training-speed cost around 20% for checkpointing; this is general guidance, not a measurement of this fit. Preserve dropout and RNG behavior and compare actual gradients/updates. | Deferred. Original non-reentrant checkpointing and RNG preservation remain enabled. |
| 5 | Increase physical microbatch while keeping logical batch 16. | Official guidance favors larger physical batches over excessive accumulation. However, batching changes floating-point grouping, padding and LoRA dropout RNG assignment even when the logical batch is unchanged. | Deferred. Physical microbatch 1 and accumulation 16 remain fixed. |

Epoch count, training exposure, BF16 versus NF4 construction, optimizer precision,
learning rate, LoRA rank/dropout and effective batch size are scientific recipe
changes, not execution optimizations. Fused LM-head/CE kernels and compilation
are later investigations because they add more numerical and dependency changes.
Published speedups from other models are not promises for this workload.

The objective is **mean supervised-token loss within each example, then mean
across examples**. Stock model loss or token-global averaging must not silently
replace it. A faster implementation must preserve the current weighting, causal
shift and assistant-end supervision.

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

### Actual RTX PRO 6000 comparison

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
the baseline and each candidate in separate processes from the same pinned base
and fresh seed. It uses certified production records, whole logical updates,
unchanged rendering and masks, warmup, synchronized GPU timing, peak memory,
losses, adapter gradients and parameter updates. It never advances or replaces
the paused production fit. The plan freezes comparison bounds before running;
reports retain failures as failures and do not automatically approve a candidate
for production. A short numerical comparison does not prove that final trained
outputs will be identical.

Use a new output directory and the current recertified consumer inputs. This
planning command does not run GPU work:

```bash
python scripts/benchmark_sft_optimizations.py \
  --config CURRENT_CONFIG.json \
  --certificate CURRENT_CERTIFICATE.json \
  --bundle CERTIFIED_BUNDLE.json \
  --worklist CURRENT_WORKLIST.json \
  --model 7b --family rot13 --level 1 \
  --profiles baseline deferred_checks sdpa \
  --warmup-updates 1 --updates 2 \
  --output NEW_BENCHMARK_DIRECTORY --plan-only
```

For a future isolated comparison, run the same command on the intended GPU with
a fresh output directory and without `--plan-only`. A combined candidate may be measured after
the individual candidates, so their separate effects remain visible. CPU timing
does not substitute for this GPU comparison. Hardware details and actual
dependency versions must accompany the report; an SDPA setting alone does not
prove a fused kernel was selected.

## Paused fit and migration

The verified step-214 checkpoint, original source, configuration and Drive
evidence remain preserved. A hardware, source or execution-profile change must
be recorded as a migration; it must not masquerade as the original exact resume
identity. Restore full adapter, optimizer, scheduler, sampler and RNG state when
resuming an accepted continuation, and document any numerical execution change.

A new consumer source needs a current handoff certificate and evidence for the
affected GPU/runtime controls. Unchanged corpus, transformation, judging and
statistical evidence can be reused with explicit compatibility bindings. This
does not require repeating every unrelated test or regenerating the datasets.
Google Drive remains the permanent store for checkpoint and migration evidence.

Verified reports and the integrated protocol are preserved in the central project at
[reports/sft_optimizations/94e2fb7](https://drive.google.com/drive/folders/1p2MxgaoO4utvdMEO_D6ijH4g8vrfgjN4).

## Authoritative precedent

- [PyTorch performance tuning guide](https://raw.githubusercontent.com/pytorch/tutorials/main/recipes_source/recipes/tuning_guide.py): “Avoid unnecessary CPU-GPU synchronization”; examples include `cuda_tensor.item()` and CUDA-dependent Python control flow such as `if (cuda_tensor != 0).all()`. It also recommends asynchronous loading and pinned-memory transfers. These are mechanisms to measure, not workload-specific speed guarantees.
- [Transformers v4.57.3 GPU guide](https://raw.githubusercontent.com/huggingface/transformers/v4.57.3/docs/source/en/perf_train_gpu_one.md): recommends data preloading, larger physical batches, and SDPA; describes checkpointing as trading memory for “slower training speed (~20%).”
- [PyTorch v2.9.0 CE documentation](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/torch/nn/modules/loss.py): `ignore_index` “does not contribute to the input gradient.” This permits investigating omitted masked-row loss work, without discarding context or vocabulary classes.
- [Qwen2 v4.57.3 implementation](https://raw.githubusercontent.com/huggingface/transformers/v4.57.3/src/transformers/models/qwen2/modeling_qwen2.py): declares SDPA support and provides selected-position `logits_to_keep`. Changing LM-head projection shape remains a separate candidate requiring empirical comparison.
- [PyTorch v2.9.0 SDPA documentation](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/torch/nn/functional.py): fused floating-point operations can make output differ with the selected backend. [Numerical accuracy guidance](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/docs/source/notes/numerical_accuracy.rst) likewise warns that batched and individual computations need not be bitwise identical.
- [PyTorch v2.9.0 checkpoint documentation](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/docs/source/checkpoint.md): RNG preservation makes dropout outputs agree with non-checkpointed passes. Keep `preserve_rng_state=True`; [non-reentrant checkpointing is recommended](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/torch/utils/checkpoint.py) and already used.
- [PyTorch transfer tutorial](https://raw.githubusercontent.com/pytorch/tutorials/main/intermediate_source/pinmem_nonblock.py): calling `pin_memory()` synchronously immediately before transfer can be slower. Preloading requires a measured implementation rather than adding that call to the hot loop.
- [Liger v0.4.0](https://raw.githubusercontent.com/linkedin/Liger-Kernel/v0.4.0/README.md): reports throughput and memory improvements and convergence testing, with benchmark conditions including LLaMA3-8B BF16/FSDP on eight A100s. This is empirical precedent for later fused-kernel investigation, not direct validation of single-GPU QLoRA here.
- [bitsandbytes 0.48.2 installation table](https://raw.githubusercontent.com/bitsandbytes-foundation/bitsandbytes/0.48.2/docs/source/installation.mdx): Linux x86-64 CUDA 12.8–12.9 builds include `sm120`. This supports the Blackwell compatibility investigation; it does not establish training throughput or replace actual hardware checks.

No published source above directly measures this frozen 7B/14B recipe on an RTX
PRO 6000 Blackwell 96GB. The actual short 7B comparison now supplies local
empirical evidence and rejected both candidates. It does not certify the full
7B/14B matrix, an exact production-checkpoint migration, final-result equivalence
or an H100-versus-Blackwell training-time ratio.
