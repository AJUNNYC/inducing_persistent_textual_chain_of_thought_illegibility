# Versioned final-field scoring repair

The original scorer required the entire native final response to be a scalar or
code-only response, although scalar prompts requested a separate final-answer
line. Untouched R1 responses commonly explain the solution after `</think>` and
then supply that line, a boxed answer, or a clearly separated Python solution.
This defect produced the retained failed baseline report. No trained outcomes
had been inspected when the repair was frozen.

`capability_scorer_v2_final_fields` reads only the native final response. For math
and choice tasks it accepts a unique terminal `Final answer:` field, a unique
terminal boxed scalar, or an entire scalar. Marker case, paired bold and math
display wrappers are presentation. Multiple markers/boxes, units, ambiguous
suffixes and prose-inferred answers remain invalid. Numeric normalization and
exact public-label matching are unchanged. It never reads an answer from the
reasoning trace or selects a candidate using the reference answer.

For MBPP, exactly one Python or unlabelled Markdown fence designates the final
code field; surrounding explanation is outside that code field. Its interior
bytes are preserved exactly. An entire unfenced response remains the code field.
Multiple or malformed fences are invalid. No indentation repair, function
renaming, code insertion, reasoning fallback, test feedback or resampling occurs.
Original hidden tests and the existing restricted executor remain unchanged.

Appendix H.9 delegates deterministic parsers and defect repairs before training;
G.9 requires versioning an uncalibrated measurement before trained outcomes.
H.4's exact code, no-repair and final-field restrictions remain in force. The
scientific success thresholds, fixed panels, prompts, generation engine and
seeds, 3072-token budget, SFT data and optimizer are unchanged.

Run `scripts/rescore_vllm_baselines.py --config CONFIG --source-root ORIGINAL
--output-root REPAIR`. It verifies the six original complete packages and every
raw request/text hash, creates a revised measurement identity and fresh scoring
cache, scores up to eight independent isolated code tasks concurrently, and
checks the original repeatability rule. It cannot sample new outputs. Original
packages and failed reports stay in their original folder. Revised packages,
report and repair receipt pass verified Drive barriers before completion.

SFT readiness still requires the actual repeatability result and compatible
training launch evidence; a passing parser test alone does not clear a gate.
