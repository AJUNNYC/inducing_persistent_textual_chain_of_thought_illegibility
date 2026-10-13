# Existing 14B base capability results

These are completed evaluations of the untouched DeepSeek-R1-Distill-Qwen-14B, before SFT. The main baseline uses seed 42; seeds 43 and 44 are repeatability diagnostics. They use the project's frozen subsets, native prompts and final-field scoring, not a full benchmark leaderboard reproduction.

| Benchmark | Seed 42 correct / questions | Accuracy | Seed 43 accuracy | Seed 44 accuracy |
| --- | --- | --- | --- | --- |
| GSM8K | 246 / 300 | 82.0% | 85.7% | 85.3% |
| MMLU-Redux-2.0 | 326 / 399 | 81.7% | 80.2% | 79.9% |
| ARC-Easy | 320 / 331 | 96.7% | 94.9% | 96.4% |
| Sanitized MBPP (withheld tests) | 115 / 203 | 56.7% | 57.6% | 58.1% |

All three 14B passes have 1,233 complete scored items. All four domains passed the prospectively specified repeatability diagnostic. Common-correct retention in the repeatability report is a separate diagnostic, not benchmark accuracy. Three passes flag gross instability; they do not establish a precise repeatability estimate.

The original final-field parser failure remains archived. The repaired scorer reused all 7,398 raw outputs across both models and three seeds, with no regeneration, corpus change or threshold change. The results above use the corrected designated-final-field scores. MBPP with scoring tests withheld from the prompt is not directly comparable to test-visible published MBPP scores.

Baseline legibility judging remains deferred; these completed capability results do not establish baseline opacity. GPQA Diamond's 198-question supplemental panel is prepared, but no GPQA baseline or trained-model score has been generated. Add an untouched 14B GPQA baseline, then use the same frozen questions, options, prompt, decoding and scoring for the preselected adapters. Existing four-domain raw outputs and scores remain usable and need not be regenerated solely because GPQA is appended.

The active training focus is 14B: ROT13 and KaziText across ten levels, 20 independent fits. Retain the completed ROT13 level-10 adapter; 19 fresh fits remain. The 7B corpus and historical baseline results remain archived and available, without active 7B training in this scope.

Sources: [summary receipt](evidence/14b_baseline_summary.json), [authoritative repeatability report](evidence/vllm_baseline_rescored_repeatability.json), [rescoring repair](evidence/vllm_baseline_scoring_repair.json), and [GPQA preparation](evidence/gpqa_preparation_receipt.json). Baseline auditing only read existing records; no new GPU job or provider call was started.
