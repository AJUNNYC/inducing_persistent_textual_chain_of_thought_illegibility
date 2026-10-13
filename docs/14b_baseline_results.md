# Existing 14B base capability results

These are completed evaluations of the untouched DeepSeek-R1-Distill-Qwen-14B. The original four-domain measurements preceded the first fit; the appended GPQA baseline ran before the current 19-fit queue, after the previously completed ROT13 level 10 adapter was retained. The main baseline uses seed 42; seeds 43 and 44 are repeatability diagnostics. They use the project's frozen subsets, native prompts and final-field scoring, not a full benchmark leaderboard reproduction.

| Benchmark | Seed 42 correct / questions | Accuracy | Seed 43 accuracy | Seed 44 accuracy |
| --- | --- | --- | --- | --- |
| GSM8K | 246 / 300 | 82.0% | 85.7% | 85.3% |
| MMLU-Redux-2.0 | 326 / 399 | 81.7% | 80.2% | 79.9% |
| ARC-Easy | 320 / 331 | 96.7% | 94.9% | 96.4% |
| Sanitized MBPP (withheld tests) | 115 / 203 | 56.7% | 57.6% | 58.1% |
| GPQA Diamond (supplement; 3,072-token cap) | 49 / 198 | 24.75% | Not run | Not run |

All three 14B passes have 1,233 complete scored items. All four domains passed the prospectively specified repeatability diagnostic. Common-correct retention in the repeatability report is a separate diagnostic, not benchmark accuracy. Three passes flag gross instability; they do not establish a precise repeatability estimate.

The original final-field parser failure remains archived. The repaired scorer reused all 7,398 raw outputs across both models and three seeds, with no regeneration, corpus change or threshold change. The results above use the corrected designated-final-field scores. MBPP with scoring tests withheld from the prompt is not directly comparable to test-visible published MBPP scores.

Baseline legibility judging remains deferred; these completed capability results do not establish baseline opacity. GPQA Diamond seed 42 completed all 198 actual untouched 14B outputs: **49 correct, 13 incorrect, 133 truncated outputs, and 3 malformed answers**. Thus 136 items had no parsed answer, while 62 were parsed. The headline remains **49/198 = 24.75%**; do not discard failures or report 49/62 as benchmark accuracy. The fixed **3,072-new-token cap**, identical to primary evaluation, is a major limitation: this measures budget-limited native output/final-field performance, not unrestricted GPQA capability. No response budget was raised, no ordinary failure was regenerated until correct, and the trained comparison must retain the same frozen panel, decoding budget and scorer. No trained-model GPQA result exists yet. Existing four-domain raw outputs and scores remain usable and need not be regenerated solely because GPQA is appended.

The active training focus is 14B: ROT13 and KaziText across ten levels, 20 independent fits. Retain the completed ROT13 level-10 adapter; 19 fresh fits remain. The 7B corpus and historical baseline results remain archived and available, without active 7B training in this scope.

Sources: [summary receipt](evidence/14b_baseline_summary.json), [authoritative repeatability report](evidence/vllm_baseline_rescored_repeatability.json), [rescoring repair](evidence/vllm_baseline_scoring_repair.json), and [GPQA preparation](evidence/gpqa_preparation_receipt.json). The historical four-domain baseline audit only read existing records and started no GPU job or provider call. The appended GPQA measurement subsequently ran real GPU inference as recorded below.

## GPQA completion and infrastructure recovery

The [sanitized completion summary](evidence/gpqa_14b_baseline_completion_2026_10_13.json) binds worker scientific source `9de8813031da737453f491723a10b305679dc42d`, protocol `42fb1287bce7968810f9d14d5325b402a8e75649628a78e8835ceb990579bfb6`, the 198-question panel, and actual 49/198 result. All 398 raw/scored/protocol/report payload records are `VERIFIED`; the fetched report and inventory bytes match the completion-receipt hashes. The [baseline's Drive folder](https://drive.google.com/drive/folders/19fz91SREgU0TvnqLMKUaXb1hx-gJUrRW) retains permanent outputs. Public Git evidence excludes licensed questions and raw responses.

An operational timeout interrupted the first attempt after 128 cached results. Recovery preserved and reused those compatible raw records, finished the remaining items, and retained ordinary outcome failures. The baseline process exited 0 and released its GPU workers before the training coordinator started. This was infrastructure recovery, not outcome-selective resampling. Later documentation commits do not change the executed scientific source. Seeds 43 and 44 have not been run for GPQA, and no supplemental repeatability gate is claimed.
