# Appended GPQA capability comparison

GPQA is a separate capability supplement. It does not replace GSM8K, MMLU,
ARC-Easy, or MBPP, change their counts/prompts/caches or 3,072-token budgets,
enter the primary statistical window or multiplicity family, or modify SFT/RL
training and certificates. The opt-in `long_32000` profile changes **GPQA
evaluation only**. Existing trained adapters do not need to be retrained.

## Current evidence: final198 complete; original one-pass evaluation first

The completed high-rate L10 extra-epoch pilot's frozen **198-question/32,000-token
GPQA evaluation is COMPLETE**, with native final artifacts and Drive verification
recorded on **16 October 2026 at 21:30:25 UTC**. All 198 questions remain in the
strict denominator:

| State, same 198 questions | Frozen scorer correct | Accuracy |
| --- | ---: | ---: |
| Matching untouched base | 99/198 | 50.00% |
| Original one-epoch L10 | 98/198 | 49.49% |
| High-rate extra-epoch L10 pilot | 52/198 | 26.26% |

The pilot has **52 correct, 44 parsed wrong, 9 malformed and 93 truncated**
outputs. The fixed 73-marker screen is positive for **168/198** responses;
**34/168** marker-positive responses are strictly correct. Marker presence
establishes occurrence, not encoding coverage, reasoning validity or formal L10
compliance. No all-198 human visible-choice or wordfreq analysis is claimed.
The 42-request and first-128 manual/contextual/lexical findings below remain
historical subset diagnostics and must not be extrapolated to all 198.

Frozen evaluator `5e6be1b0848e0e88725323591baddef37d69d8ae`, protocol
`57898f795997a55ebfce68cc68d2cc155312e6cd138b4bb2a927fe6cba863f9b`,
native prompts/options/item seeds, sampling, strict scorer and 32,000-token cap
were retained. Completion receipt SHA-256:
`e7171ac99e17f245a3257241a9c08fc1f5a09ffdde808b50844a4e0c6ee0cbfd`.
The private final responses/scores bundle SHA-256 is
`d59f1b16a990a42f71fb574f0b23aca271da5bbb5a50f8b5a9651ac3c555b1f9`;
validation SHA-256 is
`b41aa6f48fa6000f3dd51bc68615bc54f3b6887f048cb4e6a17f19be75016136`.
The [10-file final diagnostics and launch bundle](https://drive.google.com/drive/folders/1qZCc7iu3nT98yaDrbYiN19ju-TrMhhd_)
is Drive-verified. Licensed prompts, responses and token values remain private.

The pilot combines extra exposure, cosine restart at peak `2e-4` and native
thinking newline; these outcomes do not isolate an epoch effect. Its correctness
is lower than both matching controls under the frozen scorer. Original
construction criteria and four-domain evaluation budgets remain unchanged.

Current priority is original one-pass levels7-9; base/L1-6/L10 are complete and
reused under frozen evaluator5e6be1b. Low-rate v2 sourcea1962353 launched
at21:32:26 UTC; the whole-update pause completed at L1checkpoint000064,64added/689cumulative,
with fullstate/manifest/identity Drive-verified. Original level7 was RUNNING at
21:56:48.129439 UTC via publishedops2a0f7e2 supervisor29664/campaign29665,
retaining2recoveries; base/L1-6 were revalidated without inference and original
L10 retains its complete 198-question receipt. Future matched training remains authorized
and paused; no automatic training resume is installed.

## Historical first128 native prefix and human/lexical diagnostics

This section retains only the first128 subset evidence. At its collection time
the final70 group and final receipt were pending; the final198 result above now
controls full-panel capability reporting. No manual or lexical finding below is
extrapolated to the 198-response panel.

| Same 128 questions | Frozen correct | Supplementary visible-choice correct |
| --- | ---: | ---: |
| Matching base | 70/128 | 74/128 |
| Original L10 | 61/128 | 70/128 |
| New extra-epoch L10 | 35/128 | 39/128 |

New frozen categories are 35 correct, 28 parsed wrong, 5 malformed and 60
truncated. The human diagnostic credits four correct visible choices among all
five malformed outputs, with one invalid label. Frozen scores stay primary,
and no parser is repaired. The 111 marker positives include 52 `stop` finishers
and 59 `length` outputs. Strict correctness among those finishers is 23/52;
the human diagnostic gives 25/52, or 25/111 when capped outputs are included.
These conditional denominators never replace all 128. Bounded review
corroborates ROT13 occurrence, often with loops and garbled terms; it does not
establish full-response compliance. Archived full 198-question control totals
are distinct from this matched-prefix table.

L10 still targets **100% encoding of eligible reasoning spans**: Unicode word
spans with any ASCII letter, including short words, variables, mixed
alphanumeric/underscore words and Latin LaTeX command names. The narrower
wordfreq diagnostic does not exempt excluded or unassigned words from that
target and cannot establish 100% compliance. Contextual adjudication confirms
at least 34/128 failures (17 ordinary-English, 17 marker-positive). Among 25
marker-positive human-correct finishers, nine have confirmed plaintext, three
are unresolved and 13 have no identified exception after full review. This last
category is observational and does not certify 100% encoding. Together with 14
correct ordinary-English outputs, these make 39 correct visible choices; 29
finished wrong/invalid and 60 were cut off. Frozen scores remain primary.

The user-requested post hoc within-response diagnostic uses English-large data
from wordfreq 3.1.1 at the primary Zipf threshold of 3, chosen before these
measurements. Among all 719,767 alphabetic reasoning-word occurrences, 56.50%
are ROT13-only, 4.82% plain-only, 0.46% both (ambiguous), 9.06% neither (unknown)
and 29.17% short (neutral). Only words at least three letters long receive an
orientation assignment. Final answers after the thinking closer, numerals and
mixed alphanumeric tokens are excluded. Removing repeated exact, non-overlapping
eight-word blocks gives 48.05%; the equal-response mean is 53.37%. Results for
Zipf thresholds 2, 3, 4 and 5 are 43.17%, 56.50%, 53.34% and 41.50%, respectively.
Within marker-positive responses, 61.40% ROT13-only and 0.14% plain-only leave
38.46% unassigned; 99.77% is conditional R/(R+P), not all-text coverage. This
proxy is sensitive to vocabulary. It does not establish exact encoding, a
guaranteed lower bound, target L10 compliance, comprehension or faithful
reasoning. Same-prefix control background assignments are not calibrated
false-positive corrections.

The [24-file private prefix/analysis bundle](https://drive.google.com/drive/folders/1dgMvq2Zl_7CYk8bTHPh_4CbWk5ct9Kid)
is verified. An independent audit exactly reproduces 4,608 per-response tables
and 96 summaries. See the [diagnosis](../../sft_level10_training_diagnosis.md) for
category counts, repetition and threshold sensitivities, selected-subgroup
definitions, exact hashes and the dated history of the 42-request subset. This
backup does not establish full 198-question completion or scientific success.
No question text, individual IDs or prompt/response/token values enter Git.

Frozen evaluator `5e6be1b`, protocol
`57898f795997a55ebfce68cc68d2cc155312e6cd138b4bb2a927fe6cba863f9b`,
native prompts/options/seeds, sampling and the 32,000-token limit remain
unchanged. Wrong, malformed and truncated responses stay incorrect. The
exposure/schedule/newline revision cannot isolate an epoch effect. Original levels 7–9
are now prioritized using existing original states and cached protocols/results;
training remains paused during these evaluations, with no automatic restart. The
[campaign guide](../../gpqa_14b_rot13_campaign.md) retains the native completion
barrier: all 198 raw responses and scores, the report and a verified receipt.

## Dataset and measurement distinction

The José et al. code (`Jozdien/cot_legibility` at
`89195c70288b4b141460447f5fde02f67b38b043`,
`src/inference/datasets.py`) maps `gpqa` to **GPQA Diamond**. This supplement uses
all 198 questions from the official `Idavidrein/gpqa` repository at
`83022cefff930aea54f654c0b282e74b9eeda5c6`, `gpqa_diamond.csv`.
`gpqa_main` (448 questions) is an explicit alternative, never a silent fallback.
See `docs/evidence/gpqa_variant_source.json` for the primary-source evidence.

This matches the dataset variant, not José's complete measurement procedure.
Their implementation renders question text and grades answers with an LLM.
Their [archived 14B configuration](https://github.com/Jozdien/cot_legibility/blob/89195c70288b4b141460447f5fde02f67b38b043/streamlit_runs/20251022_013133_R1-Distill-Qwen-14B_gpqa/config.yaml)
allows 32,000 completion tokens with a requested 28,000 reasoning-token
allowance through OpenRouter. Our long profile allows up to
32,000 **total generated tokens per response**, without a separate reasoning
sub-budget. This is not a reproduction of their scores.
This project's supplement renders four multiple-choice options, uses the native
chat template and reasoning parser, and mechanically scores the designated
final answer label. No provider judge is called and no extra legibility panel is
added by this capability-only supplement.

The official CSV has repeated **incorrect** distractors at zero-based source
rows 89 and 126. All options and all 198 questions are preserved; the panel
records these source quirks. A repeated correct option, duplicate question,
incorrect population count, variant mismatch, or corrupt reference fails.
The literal `Question`, `Correct Answer`, and `Incorrect Answer 1/2/3` columns
are used. Explanations, pre/extra-revision fields, validator feedback, and canary
strings are never rendered in the prompt.

## Frozen generation profiles

Both profiles preserve temperature 0.6, top-p 0.95, seed 42, deterministic
choice permutations, native prompts, pinned model/tokenizer revisions, BF16
weights and designated-final-field scoring. Each item receives one response;
ordinary wrong, malformed or truncated outputs are never regenerated until
correct. The longer cap gives already-trained reasoning behavior more time to
finish; it neither adds reasoning training nor guarantees better accuracy.

| Setting | `legacy_3072` | `long_32000` |
| --- | --- | --- |
| Per-response maximum new tokens | 3,072 | 32,000 |
| vLLM total sequence context | 8,192 | 40,960 |
| vLLM maximum concurrent sequences | 128 | 8 |
| Submitted request group | 128 | 128 |
| vLLM scheduler token budget | 8,192 | 8,192 |
| GPU memory utilization | 0.9 | 0.9 |
| Prefix caching | Disabled | Disabled |
| LoRA enabled, including untouched base | Yes; maximum rank 32 | Yes; maximum rank 32 |
| Backend | Native or pinned vLLM | Pinned vLLM only |

The submitted group size differs from concurrency. The long-profile engine
schedules at most eight sequences concurrently from groups of up to 128
requests. Its scheduler's 8,192 tokens are a processing allowance, not a shared
completion cap. Every response retains its full 32,000-token allowance, with
early stopping at native EOS. Prompts must fit alongside that allowance and
are never truncated to fit.

The evaluation environment remains pinned vLLM 0.11.0, Transformers 4.57.3 and
Torch 2.8.0+cu128. Training has its separate Torch 2.9 environment and unchanged
published loss policy. The selected evaluation GPU is RTX PRO 6000 Blackwell
96GB; an old PyTorch 2.4 template is not the scientific evaluation environment.

## Untouched baseline first, then frozen adapter comparisons

Preparation fixes one deterministic SHA-256 choice permutation per question
(seed 42) and moves the correct reference label with its option. Both the
untouched base and preselected final adapter receive exactly the same questions,
options, prompt, generation profile, sampling settings and backend. Temperature
is 0.6, top-p 0.95 and the response cap is fixed by the selected profile.
There are no retry-until-correct outputs or shortened targets.

Freeze the comparison before inference. It binds the full official source pin,
CSV checksum, panel, prompt, parser/scorer/generator implementation hashes,
generation profile, engine, backend, seed and exact state descriptors.
Source/prompt/budget/engine changes cannot reuse an incompatible supplemental
protocol or raw cache. Original four-domain versions and prompt bytes remain
unchanged.

The baseline can be frozen and evaluated **before training**, without a trained
adapter. `freeze-base` reads official model/tokenizer pins from the project
configuration (or both explicit immutable commits) and permits only the untouched
construction model. Its separate `supplemental_gpqa_baseline_v1` protocol contains
one state, and `baseline.json` is complete only after every panel item has a raw
output and a score. A long-profile evaluation starts with a new matching untouched
baseline; the completed 3,072-token result cannot stand in for that baseline.
It never invents a missing adapter or runs a trained state.

Later paired freezes can opt into `--baseline-protocol` reuse. The original
baseline protocol is reconstructed and checked against the full panel, prompt,
source implementations, generation profile, sampling contract, engine, seed and model/tokenizer
commits. Use the same `--output-root` containing the baseline raw records. Exact
baseline raw identities and bytes are reused without re-labeling them as new
predictions or inferring accuracy from a summary alone. A missing raw cache cannot
be substituted with the baseline's reported accuracy. Existing paired protocols
without this option retain their original paired cache identities.

Select the published `final_adapter` with its matching `fit_result.json` alongside
it. The loader checks the original model/tokenizer commits, LoRA payload bytes,
manifest/checkpoint hash, family/model/level, and original training configuration.
The new evaluation protocol is stored separately; it does not relabel the
adapter with the current evaluation configuration. **Adding GPQA does not
require retraining any completed 14B ROT13 fit.** Levels 1–9 retain scientific
source `9de8813031da737453f491723a10b305679dc42d`; retained level 10 keeps source
`2ea4fb9a7737cff806226366e73743eec5935fad`. All ten fits are complete and were
independent untouched-base fits. GPQA results must not
be used to select a checkpoint for this predeclared comparison. Each SFT level
remains an independent fit from its original base.

## Commands

Use the existing project CPU environment for preparation and freezing. The
account must already have official Hugging Face GPQA access; downloads use the
configured credential and verified asset loader. These commands do not submit
access terms or use an ungated mirror.

```bash
python scripts/run_supplemental_gpqa.py prepare \
  --variant gpqa_diamond \
  --model-cache /workspace/model_cache \
  --output /workspace/data/evaluation/supplemental_gpqa/gpqa_diamond_panel.json

python scripts/run_supplemental_gpqa.py freeze-base \
  --panel /workspace/data/evaluation/supplemental_gpqa/gpqa_diamond_panel.json \
  --config config.yaml --model 14b \
  --model-cache /workspace/model_cache --backend vllm \
  --generation-profile long_32000 \
  --output /workspace/data/evaluation/supplemental_gpqa/base_14b_long32000_protocol.json

python scripts/run_supplemental_gpqa.py run \
  --protocol /workspace/data/evaluation/supplemental_gpqa/base_14b_long32000_protocol.json \
  --output-root /workspace/artifacts/evaluation/gpqa_rot13_14b_long32000 \
  --save-to-drive

python scripts/run_supplemental_gpqa.py freeze \
  --panel /workspace/data/evaluation/supplemental_gpqa/gpqa_diamond_panel.json \
  --adapter /path/to/completed/rot13/14b/level_10/final_adapter \
  --model-cache /workspace/model_cache \
  --backend vllm \
  --generation-profile long_32000 \
  --baseline-protocol /workspace/data/evaluation/supplemental_gpqa/base_14b_long32000_protocol.json \
  --output /workspace/data/evaluation/supplemental_gpqa/rot13_14b_level10_long32000_protocol.json

python scripts/run_supplemental_gpqa.py run \
  --protocol /workspace/data/evaluation/supplemental_gpqa/rot13_14b_level10_long32000_protocol.json \
  --output-root /workspace/artifacts/evaluation/gpqa_rot13_14b_long32000 \
  --validate-only
```

For baseline plus all ten completed ROT13 levels, use the published
`scripts/run_gpqa_sft_campaign.py` and [operator guide](../../gpqa_14b_rot13_campaign.md).
Its original launch validated all ten adapters, ran one matching baseline and
processed levels1–10 ascending. Current resume reuses complete compatible
baseline/L1-6/L10 results and runs only remaining original levels7–9. The workload is **198 × 11 = 2,178
unique responses**. Each paired report refers to the same baseline evidence;
it does not generate another baseline for every level.

The [sanitized completion receipt](../../evidence/sft_14b_rot13_matrix_completion_2026_10_14.json)
records all ten SFT adapters. Fit results and final manifests were downloaded
and byte-verified, and listed payloads matched their Drive metadata; this
independent check did not download all adapter weights again. Restoration
must verify downloaded payload bytes before loading them for inference.

After GPU availability and spending authorization are established, use the
published native GPU environment for `run` without `--validate-only`. Append
`--save-to-drive` to require synchronous verified replication using private
`GOOGLE_OAUTH_TOKEN_FILE` and project `GOOGLE_DRIVE_FOLDER_ID`. This publishes
the frozen protocol first, then raw outputs, scores, and the final comparison.
No secret values are written to the artifact tree. If omitted, the CLI explicitly
reports `drive_status: NOT_REQUESTED`; local publication is not Drive evidence.

After synchronous verified publication, the CLI writes an immutable inventory
`drive_manifest.json` and a local `completion_receipt.json` orchestration barrier.
The receipt binds protocol/panel hashes, source, model/tokenizer commits, seed,
simulation status, report hash and file checksum, and inventory path/checksum.
It is written only after the payloads and inventory have been verified in Drive.
A training coordinator must verify the actual 198 raw records and scores and
their Drive registry entries, not trust the receipt's `COMPLETE` string alone.

`--backend vllm` is available only with the existing pinned vLLM evaluation
runtime, verified cached model snapshots, and its fixed backend contract.
Switching backends or generation profiles requires a separately frozen protocol;
it cannot resume an incompatible protocol under another contract.

## Results and validation

The historical `legacy_3072` untouched 14B run remains complete and Drive-verified:
**49/198 correct (24.75%)**, 13 incorrect, 133 truncated and 3 malformed. All 198
stay in its denominator. Its original source and permanent outputs are preserved;
see [baseline results](../../14b_baseline_results.md). Those short-budget outputs
cannot enter the new 32k paired comparison. Availability of the long-profile
commands alone does not establish completion; the actual completed long-profile
results and verified receipt are recorded above.

Approximately **22 hours** for baseline plus ten levels on one PRO 6000 is a
planning estimate, including ordinary Drive publication, not a measured 32k
benchmark or deadline. Update it from actual throughput after startup.

The base-only report contains the untouched model's accuracy and a null paired
difference; a comparison report contains both accuracies and their difference
on the complete 198-question panel. Malformed or truncated model outputs count
as incorrect under the frozen scorer. Infrastructure errors stop execution and
leave resumable raw records; they cannot be converted into incorrect answers or
an apparently complete denominator. Fixture outputs are labeled simulations.

CPU tests cover choice/reference correspondence, leakage prevention, official
distractor quirks, immutable population and pairing, implementation/prompt
binding, completed-adapter reuse across evaluation configuration changes,
generation-profile isolation and strict matching-baseline reuse. GPQA is
descriptive; it does not establish primary construction success, opacity,
transfer or activation-probe performance. A prepared panel is not an evaluated model, and these
tests do not establish native GPU correctness or GPQA performance.
