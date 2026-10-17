# 14B ROT13 level-5 second-epoch pilot

The selected experiment adds one epoch to the original level-5 one-epoch checkpoint,
using its same 10,000 examples, 100 validation examples and 50% transformation rule.
The additional phase has 625 optimizer updates, peak learning rate 5e-5, 19 warmup
updates and cosine decay. Parent AdamW moments and step counters are restored.
The original one-epoch checkpoint and GPQA evidence remain immutable.

On 2026-10-17, the A40 (46,068 MiB reported VRAM) failed the non-updating longest-example
capacity check with the inherited physical batch 1, accumulation 16 and checkpointing
disabled. No training updates were completed by that attempt.

`--enable-gradient-checkpointing` on the continuation campaign or single-level worker
explicitly enables non-reentrant activation recomputation with RNG preservation.
This option keeps the parent physical batch and effective batch, sequence limit,
completion mask, loss, data ordering and optimizer initialization unchanged. It
changes activation storage and adds recomputation cost; numerical identity across
GPUs or identical final results is not asserted. The option is recorded in the
continuation policy and produces a distinct job identity. Resuming that job must
retain its recorded checkpointing mode. Existing continuations retain their
original default policy and cannot be silently migrated.

Only level 5 is selected. After verified training completion, repeat GPQA Diamond
with the existing `long_32000` profile (32,000 output tokens) and frozen questions,
then compare accuracy and observed encoding/illegibility with the original one-epoch
level 5. No GPQA examples are introduced into training. This document describes the
selected plan; it does not claim training or the follow-up evaluation is complete.
