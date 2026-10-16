# Level-10 training diagnosis and additional-epoch revision

On 16 October 2026 UTC, the user requested independent training diagnosis while
the existing GPQA evaluation continued. This CPU-only audit inspected the actual
14B ROT13 level-10 corpus and frozen training implementation. It did not change
training, evaluation protocols, saved adapters, targets or official scores.

## Actual target and token-mask findings

The 10,000 training and 100 validation records independently reproduce dataset
identity `7bce352922c2a205046ecd600f118802f8cab8360f2cf744024ef94e74ebeba4`
and handoff certificate
`a34e70567bb2c65b3ce3de45624a075b8818cd51ed8c62f954ae14b6edeb610c`.
Every reasoning target equals the full ASCII ROT13 transformation of its own
accepted clean 14B response, covering 5,177,440 eligible words. Final answers
remain byte-identical to their clean originals. Native thinking boundaries are
intact. No empty records, duplicate prompts or IDs, or train/validation prompt
overlap were found. There are 97 extra identical completion rows across distinct
prompts, including one validation completion also present in training; this is
reported separately from prompt overlap and is not a measured causal explanation.

Using the exact level-10 rendering/tokenization functions at source
`2ea4fb9a7737cff806226366e73743eec5935fad` and the pinned tokenizer, all
10,100 records have their entire encoded reasoning and clear final answer
supervised, user prefix masked, and native end token supervised. There are zero
tokenization or truncation failures. Recomputed training length is 17,986,003
tokens, matching the original runtime; maximum sequence length is 4,095 tokens.
Encoded reasoning receives 82.75% of mean per-example supervised-token weight,
clear final answers 16.97%, and structural/boundary tokens the remainder. These
are objective weights, not measured shares of gradient magnitude.

The published CCE path uses causal shift one, ignored labels `-100`, full
gradients without filtering, and equal-example accumulation. The existing real
GPU startup evidence passed the masked-row/LoRA gradient checks; no GPU check
or full test suite was repeated. Relevant rendering, fit and published-loss
source files are identical between level 10 and levels 1–9's training revision.

A small real serialization discrepancy remains: training places reasoning
immediately after `<think>`, whereas native inference prefills `<think>\n`.
Prefixes agree through thinking token 151648; inference adds newline token 198.
The behavioral effect is unmeasured. This does not establish why encoding is
inconsistent, and the running frozen evaluation has not been altered.

These findings rule out the inspected missing-target/missing-reasoning-mask
failure modes. They do not certify factual correctness of the original generated
LMSYS answers, learned free-generation compliance, or capability preservation.
Validation cross-entropy 0.76358 measures teacher-forced prediction, not sustained
encoding of independently generated reasoning.

## Authorized next training phase; not yet launched

The user subsequently prioritized further fine-tuning instead of continuing the
remaining GPQA evaluations. The selected next phase is **one additional 14B
ROT13 level-10 epoch**. This is a new training
phase, not a restart of epoch one or an exact recovery of the original completed
fit. At this documentation revision it is **not yet launched**. Preserve the
original adapter, full checkpoint, GPQA outputs and their immutable identities;
do not rerun completed GPQA measurements or automatically enqueue other levels.

The [continuation CLI](../scripts/continue_sft_epoch.py) requires the verified
original full checkpoint, completed fit/adapter, exact 10,000/100 records and
pinned model cache. The parent is source
`2ea4fb9a7737cff806226366e73743eec5935fad`, identity
`181c1684b106a0373af09fdf57aee3cde6021d7796e45c81408d6f8770c22182`,
checkpoint `99912dfd758442f1e91d3a92a8aa31121ecda8bc1828aa2a2a03f82ecc6c0175`.
An adapter-only export cannot replace AdamW moments, optimizer step counters or
RNG state. The new run is `rot13_14b_level_10_continuation_epoch_02`, policy
`published_sft_additional_epoch_v1`, under its own source/configuration identity
and output directory.

The declared phase makes 625 additional updates using the second seed-42
10,000-example permutation, reaching 1,250 cumulative updates and 20,000 example
presentations. It preserves the parent's adapter, AdamW moments/step counters,
RNG, LoRA settings, full-gradient CCE, effective batch 16 and actual physical
mode. Because the original cosine schedule has ended, it explicitly starts a
new 625-update cosine schedule with 19 warmup updates and the same peak learning
rate `2e-4`; the first update uses `2e-4 / 19`, not the parent's terminal zero.
Learning rate is not increased. Recovery within this phase restores its own
scheduler, sampler cursor and complete state.

The new renderer inserts the native newline after `<think>`. It preserves the
encoded reasoning and clear final-answer bytes, original role-prefix loss
boundary and structural-token supervision. Original dataset records and their
hash remain unchanged; derived tokenized records receive separate hashes/counts
in the new identity. Exact native-prefill equality and full-length eligibility
are required; no truncation is permitted. The CPU rendering audit passed all
10,100 records: 17,996,003 training tokens (one extra newline token per example),
maximum 4,096; 188,481 validation tokens, maximum 4,085; and 10,100 exact native
token-prefill matches. Six focused continuation checks passed; the full test
suite and GPU optimization comparisons were not repeated. These checks establish
implementation boundaries, not a measured training benefit. Checkpoints and final artifacts must
pass actual Drive verification, and a requested pause occurs after a whole
optimizer update.

This combines additional exposure, a schedule restart and a serialization
change. Any improvement cannot be attributed to epoch count alone. Stronger or
more consistent free-generated encoding is an objective, not an established
benefit; answer correctness and capability retention remain to be measured.
Low validation loss does not guarantee either. No higher learning rate, new
training corpus or repeated full GPQA campaign is included in this phase.

## Prepared development screen; prefix comparison deferred

A private fixed 64-question panel was prepared from pinned GSM8K training data,
separate from its formal test pool. Of 7,473 source questions, the 800 reserved
RL questions were excluded, leaving 6,673 eligible. Exact normalized comparisons
found no overlap with either model's SFT training/validation prompts, the full
acquired four-benchmark evaluation pools, or GPQA Diamond. This does not claim
semantic decontamination or exclusion from base-model pretraining.

The earlier proposed diagnostic was a small controlled prefix comparison on a
fixed subset of this panel. It is now **deferred**, not a prerequisite for the
user-prioritized additional epoch.
Compare native `<think>\n` with training-matched `<think>`, then a fixed,
answer-independent ROT13 starter with its plain-English counterpart, using both
the untouched base and original level-10 adapter. Hold questions, sampling and
the 3,072-token ceiling fixed; exclude supplied starter tokens from encoding
counts and retain correctness, truncation and joint-success outcomes.
Teacher forcing supplies encoded context during training, whereas free
generation can enter English and continue there. This makes initiation and
sustained generation worth distinguishing; it does not establish exposure bias
or the newline as a measured cause. A starter-induced improvement also does not
establish spontaneous encoding. This diagnostic is proposed only, has not been
launched, and does not change the frozen GPQA protocol or original training targets.

The prepared post-training development comparison has three states: untouched
14B base, original level-10 adapter, and the separately identified continuation.
It has not been launched. It uses the same
64 questions, ordinary GSM8K prompts without an encoding instruction, fixed
sampling settings, one response per state/question and the existing 3,072-token
GSM8K ceiling. Only GPQA has the separately authorized 32,000-token budget.
Sixty-four questions are a practical screen, not a guaranteed-power sample size.

Report encoding occurrence, sustained coverage and placement, final-answer
correctness, joint encoded-and-correct outcomes, parsing failures and truncation.
Free-generated text has no paired clean original, so estimated encoding coverage
requires uncertainty and raw/decoded inspection; a few dictionary matches do not
establish full level-10 compliance. Keep the panel's answers private and out of
generation prompts. An optional instructed cipher-conversion diagnostic measures
a different ability from spontaneous encoding during reasoning.

Development improvements would inform whether to extend training across levels,
with uncertainty and cost reported. Existing inspected GPQA results remain
descriptive rather than being relabelled as untouched confirmation after
model-development decisions. Decisions about further levels or evaluation
resumption remain separate from this one additional training epoch.
