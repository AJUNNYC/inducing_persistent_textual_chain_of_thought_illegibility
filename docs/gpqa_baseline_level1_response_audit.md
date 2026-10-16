# GPQA response audit: baseline and ROT13 level 1

This read-only audit covers all 396 responses from the completed untouched 14B
baseline and ROT13 level 1: 198 full GPQA Diamond questions per state, with up
to 32,000 generated tokens per response. The completed payloads are Drive
verified. Execution source is `5e6be1b0848e0e88725323591baddef37d69d8ae`.
The continuing evaluation, original raw responses, protocol and recorded scores
were not changed. Item text, response bodies and per-item answer keys remain in
private storage; this report contains aggregate findings only.

## Recorded outcomes

| Outcome | Baseline | ROT13 level 1 |
|---|---:|---:|
| Correct | 99 / 198 (50.00%) | 93 / 198 (46.97%) |
| Parsed incorrect choice | 69 | 66 |
| Final answer rejected by the choice parser | 28 | 32 |
| Reached 32,000-token limit | 2 | 7 |
| Mean generated tokens | 5,331.7 | 5,720.5 |
| Median generated tokens | 4,756.5 | 4,223.5 |

Both states answer 71 questions correctly. Level 1 loses 28 questions that the
baseline scored correctly and gains 22 that the baseline failed, for six fewer
correct answers overall (3.03 percentage points). These are single sampled
outputs per question, with state-specific derived seeds. This small net gap
alone does not establish a stable or substantial capability decline.

## A concrete parser presentation mismatch

A clear final line such as `**Final Answer: A**` is rejected: the parser consumes
the opening bold marker before the final-answer heading but leaves the closing
marker attached to the letter. There are 26 such baseline cases and 28 level-1
cases. Eight of those baseline choices and eleven level-1 choices match the
answer key. A narrow, presentation-only diagnostic would therefore give
107/198 (54.04%) and 104/198 (52.53%), respectively.

The remaining two baseline and four level-1 rejected final answers also state
an explicit terminal choice, but use repeated consistent declarations, an
additional heading or numeric option text. A broader audit reading those
explicit choices gives 108/198 (54.55%) and 107/198 (54.04%). Extraction used
only the final declarations before comparison with reference labels; reasoning
was not mined for an answer. These are **post hoc diagnostic counts**, not new
benchmark results or silent replacements for the frozen reports. The narrow
formatting fix and broader extraction policy must remain distinguished.

Any correction should be versioned and applied consistently to saved raw
responses for the baseline and every level, preserving the original reports.
That requires scoring work, not new model generation.

## Encoding transfer and illegibility

The user's target is increased illegibility, with final-answer capability
measured separately. Faithful preservation of an original explanation is not an
additional success requirement. Garbled decoded wording is not a failure by
itself; its relevance here is whether an observer can understand the raw trace.

Level 1 training targets approximately 10% of eligible reasoning word spans,
arranged in three patches. It does not encode prompts or final answers. The
GPQA evaluation prompt does not ask for encoding, so visible encoding here is
spontaneous transfer from SFT.

A common-word ROT13 marker detector finds recognizable patches in 28/198
level-1 responses versus 0/198 baseline responses; 19 level-1 responses have at
least five markers. The detector uses 73 common words, and its matches were
inspected in context. This is a diagnostic count, not a formal 14.1% compliance
rate or a measurement of the fraction of words encoded. Several decoded
passages contain garbled or invented wording; the detector does not establish
how much this obstructs an observer's understanding. One truncated answer
contributes 415 of the 662 marker occurrences, making pooled marker counts
particularly misleading. No marker matches were found in final-answer regions.

Manual review of selected paired responses also finds ordinary scientific
mistakes, reversals after lengthy self-questioning and late guesses in both
states. Some questions improve while others worsen. This review does not
provide a correctness certificate for reasoning or a causal attribution to
encoding. Nothing here establishes behavior at the higher, uncompleted levels.

## Direct reading of complete responses

The primary assistant subsequently read 12 verbatim outputs end-to-end: nine
level-1 responses and three matched baseline responses, totaling 24,021 original
generated tokens. Five inspected level-1 responses contained recognizable ROT13
markers and four did not. Eight trained responses were selected by a fixed
question-ID rule across marker presence and short/medium length; one additional
paired example was read in full. This purposive sample oversamples encoded
cases and cannot estimate a population illegibility rate. Earlier excerpts of
the longest outputs are not counted as complete readings.

The direct judgment is that level 1 remains largely readable in this sample.
Encoded or garbled patches genuinely interrupt local reading, but the
surrounding English often makes the overall argument understandable without
decoding. All four marker-negative sample responses are readable English,
including the mathematical steps. Some are repetitive or scientifically
confused, which is not itself textual illegibility. One matched baseline was
more circular and harder to follow than its trained counterpart. These are
qualitative observations of the native reasoning region, not an automated
marker score, a correctness judgment or a formal observer evaluation. They
support localized obstruction at level 1; they do not establish behavior at
higher levels.

The relevant target is an increase relative to baseline, so broad or complete
unreadability is not required. A subsequent paired reassessment expanded direct
reading to the complete native reasoning for all nine trained examples and
their nine matched baselines. The five deliberately selected marker-positive
trained cases add locally opaque passages absent in their baseline counterparts.
The four marker-negative cases show no clear increase in local textual opacity;
some trained answers are easier to follow overall. One encoded trained answer
also has a shorter, easier overall presentation despite its locally opaque
fragment. The direct judgment is therefore modest, uneven qualitative evidence
of increased illegibility relative to baseline. These purposive observations
do not estimate the net effect across all 198 questions. Answer correctness and
preservation of an original explanation are separate from this assessment.

The [aggregate audit receipt](evidence/gpqa_baseline_level1_response_audit_2026_10_15.json)
binds counts and private input/report hashes. No new inference, provider judging,
training, score mutation or full test suite was required for this audit.

## Level 2 follow-up

All 198 level-2 GPQA Diamond responses have now been generated and scored under
that same frozen 32,000-token protocol. A read-only audit verified every raw and
record hash, the raw-to-score links, token counts and matching item identities.
Recorded outcomes are 100 correct (50.51%), 61 parsed incorrect, 29 malformed
final answers and eight responses that reached the token limit. The mean output
is 5,851.8 tokens and the median is 4,628. These are the original official scores.
The same narrow whole-bold final-line diagnostic identifies 26 presentation
failures, including 15 correct explicit choices; its diagnostic count would be
115/198 (58.08%). This does not replace the frozen score or relax other answer
extraction rules.

The identical 73-word marker detector finds 18/198 marker-positive level-2
responses (9.09%), compared with 28/198 at level 1 and zero at baseline. Sixteen
level-2 responses contain at least five markers, with 523 occurrences overall;
none occur in final-answer regions. Three questions are marker-positive at both
trained levels. These are diagnostic frequencies, not formal illegibility
scores, encoded-word fractions or evidence that the effect rises monotonically
with the training level. A higher training encoding target does not guarantee
more frequent encoding on novel evaluation prompts.

The primary assistant personally read nine complete level-2 responses. Five
were plain English and four contained clear localized coded patches. For three
of the coded cases, the assistant also read the complete matched baseline and
level-1 outputs: those earlier outputs used readable English where level 2
introduced locally opaque passages. This purposive sample supports useful
localized illegibility relative to baseline, while providing no evidence that
level 2 does so more consistently than level 1. The sample cannot estimate an
average illegibility effect across all 198 questions, and answer correctness or
faithful preservation of an earlier explanation is not an additional criterion.

During the subsequent level-3 transition, a Drive timeout was repaired by
retrying synchronization with the unchanged implementation; the same campaign
resumed at 2026-10-15 06:55:51 UTC. No model regeneration, protocol change or
scientific score mutation was needed for this operational recovery.

## Level 4 follow-up

All 198 level-4 responses and scores are complete and verified on Google Drive.
The frozen official outcomes are 95 correct (47.98%), 72 parsed incorrect,
27 malformed final answers and four responses that reached the 32,000-token
limit. The baseline is 99/198 (50.00%). Level 4 generated 1,039,669 tokens,
averaging 5,250.9 per response, with a median of 4,331.

The identical 73-word detector finds 11/198 marker-positive responses (5.56%),
including ten with at least five markers and 446 marker occurrences overall.
No markers occur in final-answer regions. The marker-positive frequency is
lower than level 1's 28/198 and level 2's 18/198. This screening proxy does not
establish a monotonic population gradient or quantify actual illegibility.

The primary assistant personally read seven complete level-4 outputs: four
marker-positive cases and three plain-English cases, selected purposively.
Some positive cases contain longer consecutive encoded passages than earlier
levels. In one matched example, baseline and level 1 use ordinary English,
level 2 introduces a brief encoded patch, and level 4 contains several
consecutive encoded paragraphs. In another, level 1 has a short fragment,
level 2 uses ordinary English, and level 4 contains several encoded paragraphs.
These are concrete candidates for individual increases in illegibility, while
the second example also illustrates why the levels are not reliable ordinal
labels for every generated response. The three marker-negative outputs read
as ordinary English. This small selected sample cannot estimate the average
effect or the yield of usable probe examples. It supports stronger local
encoding in some responses alongside lower observed frequency; actual
illegibility must be assessed per response. Answer correctness and faithful
preservation of an earlier explanation are separate criteria.

This read-only follow-up changed no training, generation, scoring or active
protocol settings. Private aggregates and per-item marker diagnostics are
saved separately; response text and answer keys are excluded from this page.

### Broader level-4 ROT13 recheck

The 73-word detector is a limited screen. To investigate missed encoding, all
198 level-4 outputs were rescanned using a broader baseline-derived vocabulary,
character-trigram English-versus-ROT13 windows, and an external English word
frequency comparison (`wordfreq` 3.1.1). The frequency scan recovered all 11
original positives and flagged four additional responses; the trigram scan
also recovered all 11 and flagged four additional responses. Across these two
methods, six distinct previously negative outputs required contextual review.
The primary assistant inspected all newly flagged contexts: they contained
scientific units, gene names, chemical formulas or mathematical notation,
including `Cre`, `PbCl2`, `PbF2`, `FUV`, `erg` and `hbar`. Decoding these did not
reveal additional encoded passages. An independent vocabulary and trigram
review likewise found no additional confirmed cases.

The primary assistant also inspected an original and ROT13-decoded passage
from each of the 11 original positive responses. The supported result is
therefore **11 confirmed responses containing recognizable ROT13 passages**,
with no additional confirmed responses found by the broader checks. This is
not proof of perfect recall: very short, partial, rare-word or garbled
encoding can evade these screens. It also does not measure all forms of
illegibility. Synthetic 16-word encoded controls were detected, but those
controls do not establish sensitivity on naturally generated mixed passages.
No model inference, training, official scoring or active protocol changed.

## Correctness within encoded subsets, levels 1–4

Using the same 73-word screen at each level, with encoded contexts checked,
the following counts describe final-answer correctness among responses that
contain recognizable ROT13 passages. They do not certify compliance with the
intended encoding fraction for the whole response.

| Level | Encoded responses | Official correct | Narrow bold-format diagnostic correct |
| --- | ---: | ---: | ---: |
| 1 | 28 | 12 (42.9%) | 12 (42.9%) |
| 2 | 18 | 7 (38.9%) | 8 (44.4%) |
| 3 | 17 | 7 (41.2%) | 8 (47.1%) |
| 4 | 11 | 6 (54.5%) | 6 (54.5%) |

The diagnostic column accounts only for the previously documented whole-bold
terminal final-label presentation bug; it does not replace official scores.
All seven malformed final sections in these subsets were inspected. Six have
a single whole-bold choice label; the remaining response gives two different
choices and remains a failure. Only two correct answers are rescued by the
narrow presentation diagnostic. Truncated outputs remain failures.

Correct final answers co-occur with encoded passages at every level. These
are small, differently selected question groups, so their percentages do not
establish an accuracy trend or a causal effect of encoding. The private audit
also retains same-question baseline comparisons and per-item provenance;
question text and answer keys are excluded from this page.

## Level-5 follow-up, 16 October 2026 UTC

Level 5 completed all 198 full-budget observations and its verified Drive barrier.
Official outcomes were 97 correct (48.99%), 61 incorrect, 37 malformed and three
truncated. The baseline remained 99/198 (50.00%); official scores were unchanged.

The same fixed 73-word screen identified 10/198 responses (5.05%) containing
ROT13 passages, versus 11/198 at level 4 and zero baseline positives. The primary
assistant and an independent reviewer each inspected raw and decoded passages
from all ten positives; these include genuine encoded phrases and paragraphs.
Five answered correctly, four incorrectly and one was truncated. These counts
do not establish full level-5 target compliance or perfect detector recall.

One truncated response supplies 1,954 of the 2,750 marker occurrences. Aggregate
marker volume must therefore not be presented as population-wide improvement.
Encoding remains inconsistent, with no observed monotonic frequency gradient
across levels 1–5. The private compressed response/score export has SHA-256
`0408423024d4999bea428729f6e114503a2df7dfbd2703d1c7762b27d113d9aa`;
licensed question text and answer keys are not published here.

## Level-6 follow-up, 16 October 2026 UTC

Level 6 completed all 198 full-budget observations and its verified Drive barrier.
Official outcomes were 104 correct (52.53%), 66 incorrect, 27 malformed and one
truncated. All 198 response-to-score raw-hash links were independently verified.

Before the correctness scores were supplied, the same fixed 73-word screen
identified 11/198 responses (5.56%) containing recognizable ROT13. An independent
reviewer inspected raw and decoded initial passages from all eleven positives;
each contained genuine encoded sentences or paragraphs. After joining the
official scores, four were correct, six incorrect and one malformed, giving
4/11 (36.36%) official accuracy in this small selected subset. No answer-parser
relaxation was applied.

The observed frequency matches level 4's 11/198 and is close to level 5's 10/198;
it does not establish a monotonic population gradient. Several positive outputs
contain substantial encoded passages, but marker counts depend on response
length and do not measure prescribed encoding-fraction compliance. Decoded
technical terminology is sometimes garbled, and this limited detector can miss
short, unusual-word or partial encoding. Different selected question subsets
also prevent treating their correctness percentages as an accuracy trend.

This read-only audit changed no training, generation, protocol or official score.
The private compressed scores/completion-receipt export
(`level06_scores_and_receipt.json.gz`) has SHA-256
`3b22917749a7bd653c68cf59c674f75bcd7d991c358824c666cf82b7e5bd31a0`;
licensed question text and answer keys are excluded from this page.

## Level-10 follow-up, 16 October 2026 UTC

Level 10 completed all 198 full-budget responses and its verified Drive barrier.
Official outcomes were 98 correct (49.49%), 73 incorrect, 25 malformed and two
truncated, compared with the baseline's 99/198 (50.00%). An independent join
verified all 198 response-to-score raw-hash links; the official scores and
protocol remain unchanged.

Before consulting correctness scores, the same fixed 73-word screen found zero
marker-positive responses among all 198 outputs. All 128 responses in the
earlier provisional export were retained unchanged in the full export. Manual
review covered selected negative examples, including start, middle and end
excerpts from three question IDs selected from the additional 70 responses
before reading their contents. Those excerpts read as ordinary English with
scientific notation. This is limited excerpt review, not exhaustive manual
judging of all 198 outputs. The fixed detector can miss short, unusual-word,
partial or garbled encoding and other forms of illegibility; zero hits must
not be described as proof that every possible encoding is absent.

These observations do not demonstrate reliable full ROT13 at the level-10
checkpoint or an increasing encoding-frequency gradient. A read-only source
and adapter-selection audit found no identified selection fault, but source
inspection is not a functional adapter-on/off ablation. A separate diagnostic
comparison of native reasoning prefixes was proposed before deciding whether
to add training epochs; that diagnostic and additional training have not been
launched. The frozen evaluation remains the original result.

The private compressed full response-only export has SHA-256
`b2396428620c96171450fd31442e7e8e3c29c50604a757db3f857b16aabe60c9`.
The private compressed scores/completion-receipt export has SHA-256
`3e46bae4fb2bf99c266b02b091ec9d6276959d0543013ea70208316d2c5f798b`.
Licensed question text and answer keys are excluded from this page.
