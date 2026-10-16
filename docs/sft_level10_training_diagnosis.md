# Level-10 training diagnosis and prepared development comparison

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

## Prepared development screen; no training or inference launched

A private fixed 64-question panel was prepared from pinned GSM8K training data,
separate from its formal test pool. Of 7,473 source questions, the 800 reserved
RL questions were excluded, leaving 6,673 eligible. Exact normalized comparisons
found no overlap with either model's SFT training/validation prompts, the full
acquired four-benchmark evaluation pools, or GPQA Diamond. This does not claim
semantic decontamination or exclusion from base-model pretraining.

Before spending on another epoch or a higher learning rate, the recommended next
step is a small controlled prefix comparison on a fixed subset of this panel.
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
launched, and does not change the running frozen GPQA campaign or training targets.

If subsequent development evidence supports extra training, the prepared
continuation comparison has three states: untouched 14B base, original level-10
adapter, and a proposed separately identified continuation. It uses the same
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

The proposed continuation is one additional epoch with the same peak learning
rate `2e-4`, original corpus, masks, LoRA capacity and effective batch. Preserve
compatible AdamW moments from the verified original full checkpoint. A fresh
declared nonzero learning-rate schedule is necessary because the original cosine
schedule has ended. This compares a documented continuation recipe, including
schedule restart, rather than isolating epoch count alone. Unavailable optimizer
state requires an explicit recipe decision; it must not be silently reset.
The newline discrepancy is a separate diagnostic, not an undeclared target change.

The original checkpoint remains immutable. The continuation is proposed only;
the active GPU continues the authorized GPQA queue. Development improvements
would inform whether to extend training across levels, with uncertainty and
cost reported. Existing inspected GPQA results remain descriptive rather than
being relabelled as untouched confirmation after model-development decisions.
