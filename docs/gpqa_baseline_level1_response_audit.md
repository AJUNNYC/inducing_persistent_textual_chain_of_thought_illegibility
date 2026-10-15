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
support localized obstruction rather than broad, consistent illegibility at
level 1; they do not establish behavior at higher levels.

The [aggregate audit receipt](evidence/gpqa_baseline_level1_response_audit_2026_10_15.json)
binds counts and private input/report hashes. No new inference, provider judging,
training, score mutation or full test suite was required for this audit.
