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

## Additional training phase completed, 16 October 2026 UTC

The user subsequently prioritized further fine-tuning instead of continuing the
remaining GPQA evaluations. The selected next phase is **one additional 14B
ROT13 level-10 epoch**. This is a new training
phase, not a restart of epoch one or an exact recovery of the original completed
fit. The worker launched at 06:37 UTC from Git commit
`3a89691a34fb124d2a5235dfb162f55ca723a2f6`; the first additional optimizer update
was observed at 06:40 UTC, with the initial checkpoint verified in Drive.
The phase is now **complete**: 625 additional updates, 1,250 cumulative updates,
10,000 additional example presentations and 20,000 cumulative presentations.
It processed 17,996,003 training tokens and evaluated all 100 validation cases,
with validation loss `0.6151581262983382`. The final adapter, full checkpoint and
26 required Drive artifacts passed actual verification; the trainer exited with
code zero. Capability benefit remains unmeasured; provisional encoding behavior
is reported below. Preserve the original adapter,
full checkpoint, GPQA outputs and their immutable identities; the follow-up
below evaluates only the new endpoint, without regenerating its existing
controls or automatically enqueueing other levels.

The recorded fit interval was `17470.26931851078` seconds, approximately 4 hours
51 minutes. This is about three times the original fit's 96.6 minutes. The cause
of the slowdown is **unknown**; no hardware, attention-backend or workload cause
has been established, and no speedup is claimed.

The completed run binds identity
`8f8a69a26e566bfcc7f889a0087fb40f964e12bda4eed4652cd3f5e15d59e6be`,
configuration `68c5362d33b1767edf5292d5dd71604ae724c04e8a13c5bc80f24aee0da4e0f0`,
adapter manifest `70b7c15e7658b755a48c0f37e45c25a6116e7ef6733ed4460697204c35520e5d`
and full checkpoint
`6ce27df67ae4c51ad909b8da5cc46c64cf2709729c65feddd5cb2d919faed8aa`.
Its executed training source remains `3a89691a34fb124d2a5235dfb162f55ca723a2f6`.

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
change. Any improvement cannot be attributed to epoch count alone. The provisional
snapshot below supports sustained ROT13 in reviewed passages from some completed
requests; it does not establish population-wide encoding or capability retention.
The later fixed-42 diagnostic scores below measure correctness only within that
completion-selected subset.
Low validation loss does not guarantee either. No higher learning rate, new
training corpus or repeated full GPQA campaign is included in this phase.

## Running level-10 GPQA; official results pending

The user authorized the **same full GPQA Diamond evaluation** immediately
after this additional epoch completes. The handoff is now active for only the
new 14B ROT13 level-10 adapter: 198 questions, up to 32,000 new tokens each, the same frozen
`5e6be1b0848e0e88725323591baddef37d69d8ae` evaluation implementation,
native prompts, option permutations, per-item seeds, sampling, engine settings
and strict final-label scorer. The original level-10 outputs (98/198) and matching
32,000-token untouched baseline (99/198) remain the comparison controls; their
compatible cached records are reused rather than regenerated. Levels 7, 8 and 9 remain
paused. No other training or full evaluation campaign is scheduled.

The dispatcher verified all 625 additional updates, the complete continuation
result and 26 Drive artifacts, including the final adapter and full checkpoint.
Training-worker exit and release of its GPU/locks remain required before
evaluation starts. The new adapter and checkpoint are bound under distinct
immutable protocol
`57898f795997a55ebfce68cc68d2cc155312e6cd138b4bb2a927fe6cba863f9b`
and output namespace. The dispatcher must not overwrite the original level-10 raw
outputs, alter the frozen evaluator or relabel training source `3a89691`.
Training failure, pause, missing backup or incompatible bindings stop the
handoff. At 11:31:53 UTC on 16 October, the dispatcher reported `RUNNING` from
operations source `dfcae353a50f723ac2551727904b802e7b34750f`, preparing only this
new level-10 evaluation. That earlier status established preparation only.
Actual new-adapter inference was observed at 11:52:41 UTC, and the full matched
198-question run continues unchanged. Official new GPQA results remain pending.
See the [GPQA campaign guide](gpqa_14b_rot13_campaign.md).

Wrong, malformed and truncated responses remain incorrect under the same
official scorer. Save and verify all 198 raw responses, scores, paired report
and completion receipt in Drive. The comparison remains exploratory because
earlier GPQA behavior informed this training decision; it is not an untouched
confirmatory test or proof that additional epochs caused any observed change.

## Provisional completed-request observations, 13:39:40 UTC

A private snapshot at **2026-10-16 13:39:40.342734 UTC** recovered all 42 requests
already completed in the current in-flight group. A bounded, typed, read-only
observer inspected the native parent process (PID 20998, start ticks
`260047295`), without calling target methods, changing memory, pausing inference
or regenerating responses. This is an observational diagnostic, separate from
the native raw-output/scoring and Drive-completion barriers.

A fixed **73-type ROT13 marker screen flagged 37/42 requests**, with 52,793 marker
occurrences. Independent opening/middle/end review of all 42 outputs found
sustained ROT13 in the reviewed passages of all 37 positives and ordinary
English/math in the five negatives. This provides provisional evidence that the
new adapter emits extended ROT13-form text under unchanged native prompts.
Bounded windows do not establish full-response coverage, prescribed level-10
compliance, or absence of encoding elsewhere in the negatives.

| Observed completion metadata | Count |
| --- | ---: |
| Finished with `stop` | 25 |
| Finished with `length` at the cap | 17 |
| Marker-positive requests with `stop` | 21 |
| Marker-positive capped requests | 16 |
| Marker-negative capped requests | 1 |
| Generated tokens across these 42 requests | 648,941 |

All 17 capped outputs showed pronounced looping in reviewed passages: 16 were
marker-positive and one was ordinary English. Repetition inflates the marker
occurrence count; it is not a token-coverage measure. Decoded passages often
contained garbled, invented or substituted technical terms. Readable connective
syntax after decoding does not establish a valid scientific argument,
comprehension, correct answers or faithful reasoning.

The **42 requests are completion-biased**, not a random or representative sample
of all 198, and their request IDs were not yet mapped to question IDs. At this
snapshot there were **zero new native raw records and zero official scores**;
outputs remained in memory until the request group returned. The separate 13:55
identity/scoring stage below later resolved mapping for exactly these 42 requests. Do not extrapolate
37/42 to the full panel or call it an accuracy or final encoding rate. The
original L10 control had 0/198 positives under the same marker screen and scored
98/198 officially, versus the matching base's 99/198. That historical contrast
is descriptive; the new adapter's full matched accuracy and encoding comparison
remain pending. The combined exposure/schedule/newline revision and earlier
GPQA-informed development preclude an isolated epoch-effect claim.

The [private observational bundle](https://drive.google.com/drive/folders/1m6bTexX5ZGXIfOrQFIw1d_zeq10wXs0N)
has eight verified Drive files, including the snapshot, capture provenance and
independent review. This backup is **not** a native evaluation-completion receipt
or checkpoint. No question text or generated-response snippets are published in
Git. The snapshot SHA-256 is
`bad494dfab6b89852289583214934aa5b8f9c99142da781407c56b4091d41b23`;
independent findings SHA-256 is
`518d998af41c46607d582fcd0a291d7d9f1531da1fb076f2f18da92b5c2ad725`;
observer helper SHA-256 is
`d2ccfccc55ca7eb2c2c301ad8f60c7a1c9e92a57b28e80543946e3591794e52b`.
The helper is a private observational bootstrap, not the canonical evaluator.
The frozen `5e6be1b` source, protocol `57898f79...`, 32,000-token allowance,
native prompts, item seeds and scorer remain unchanged for all 198 questions.

## Provisional fixed-42 correctness, identity capture at 13:55:01 UTC

A second bounded read-only capture at **2026-10-16 13:55:01.947404 UTC** resolved
the identities of the same 42 requests selected at 13:39, before correctness was
known. Full input-token-vector hashes matched uniquely to the frozen 198-prompt
map. Full generated IDs decoded exactly as native output (`skip_special_tokens=False`
with one terminal EOS stripped); all 42 texts matched the earlier snapshot.
Host/native Transformers 4.57.3 and tokenizers 0.22.2 matched. No response was
regenerated or excluded, and no native artifacts were created.

The unchanged frozen `5e6be1b` scorer, including its known bold-final presentation
limitation, produced these **offline provisional scores on the same 42 questions**:

| State | Correct / 42 | Parsed wrong | Malformed | Truncated |
| --- | ---: | ---: | ---: | ---: |
| Matching untouched base | 23 (54.76%) | 17 | 2 | 0 |
| Original L10 | 22 (52.38%) | 15 | 4 | 1 |
| New L10 extra epoch | 11 (26.19%) | 11 | 3 | 17 |

Original-to-new pairs were 9 both correct, 13 correct-to-incorrect, 2
incorrect-to-correct and 18 both incorrect (including malformed/truncated). Among 37 marker positives, 8 were
correct: all were among the 21 `stop` requests; the 16 capped positives were
incorrect. Three of five marker negatives were correct. This subset therefore
shows increased observed encoding alongside lower diagnostic accuracy, with
looping and truncation. It does not establish comprehension, faithful reasoning
or an isolated epoch effect.

Independent local audit passed **554 checks**, replaying the scorer on all 42
new and 84 cached control responses; no new GPU tests ran. These are diagnostic
scores with `raw_hash=None`, **not native official scored records**. Zero new
native raw/official scores existed at this stage. The full 198-question run
continues unchanged; native group-output publication awaits the 128-request
group's return, and all 198 remain required. The 42 remain a
completion-selected, nonrandom subset; their scores are not estimates of final
198-question performance. The earlier unmapped/zero-score statement remains a
dated description of the 13:39 snapshot.

Identity snapshot SHA-256:
`43774975aaa4465b866e8d7a696b12a74e0711083f1022fb84b035da06bfdf11`;
identity helper:
`a2872a6d42828bd6964f77910ed2f32b6ee3d3b9ca1720f2fb4836c42c6c28ab`;
independent correctness audit:
`c0c67743dfbd6d834017aae466cd45170334925d73c3bc7cd193df6e53c22ad9`.
The [private fixed-42 correctness bundle](https://drive.google.com/drive/folders/1-pB32EBJuXU0J7K6BplTv7VrHGsJUacx)
has 13 verified Drive files including its receipt. Prompt/response/token values
remain private. This identity/scoring backup is separate from both the earlier
eight-file observational bundle and native completion.

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
work beyond the single authorized follow-up remain separate from this epoch.
