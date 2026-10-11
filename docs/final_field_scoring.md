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

## Actual completed repair

All 7,398 original responses were rescored once under the frozen final-field
contract without a new generation. The revised [repeatability report](evidence/vllm_baseline_rescored_repeatability.json)
passed every model/domain check. The [repair receipt](evidence/vllm_baseline_scoring_repair.json)
binds the original failed report, original request/text identities, revised
measurement protocol and actual executed source `1a72d4a`. Revised packages,
report and receipt are integrity-verified in [Google Drive](https://drive.google.com/drive/folders/1ruy6fVcaVAmfIkZ3asQaGDxljhuMyA-2).

| Model | Domain | Correct counts, seeds 42 / 43 / 44 | Seed-43 / seed-44 retention | Three-seed common retention |
|---|---|---|---|---|
| 7B | GSM8K | 249 / 254 / 250 | 0.9116 / 0.9116 | 0.8514 |
| 7B | MMLU | 240 / 233 / 252 | 0.8042 / 0.8333 | 0.7208 |
| 7B | ARC | 301 / 300 / 305 | 0.9402 / 0.9535 | 0.9003 |
| 7B | MBPP | 122 / 126 / 113 | 0.8934 / 0.8443 | 0.8197 |
| 14B | GSM8K | 246 / 257 / 256 | 0.9228 / 0.9187 | 0.8618 |
| 14B | MMLU | 326 / 320 / 319 | 0.8896 / 0.9172 | 0.8528 |
| 14B | ARC | 320 / 314 / 319 | 0.9594 / 0.9781 | 0.9469 |
| 14B | MBPP | 115 / 117 / 118 | 0.8696 / 0.8783 | 0.8000 |

The unchanged requirements are 0.80 retention in each repeat and 0.60 common
retention. The smallest observed values are 0.8042 and 0.7208. Passing the
repeatability gate does not imply every task answer is correct or certify opacity.
Legibility judging is deferred; no scientific SFT fit has started. Compatible
training evidence was subsequently reconciled to all 13 READY launch checks;
[launch readiness](evidence/sft_launch_readiness.json) records the current binding.

Twenty-three targeted scorer and reuse-guard tests passed. Changed generation
settings, prompt bytes or generation source reject reuse. A missing saved raw
response fails without loading an engine. The native reasoning parser, raw
outputs and their original failed scoring packages remain unchanged. Later base
and trained-state evaluation must use the same revised extraction contract.
