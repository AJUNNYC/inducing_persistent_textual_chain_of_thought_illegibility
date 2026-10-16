# Appended GPQA capability comparison

GPQA is a separate capability supplement. It does not replace GSM8K, MMLU,
ARC-Easy, or MBPP, change their counts/prompts/caches or 3,072-token budgets,
enter the primary statistical window or multiplicity family, or modify SFT/RL
training and certificates. The opt-in `long_32000` profile changes **GPQA
evaluation only**. Existing trained adapters do not need to be retrained.

## Active additional-epoch level-10 handoff; results pending

The separate 14B ROT13 L10 extra epoch is complete: 625 additional/1,250 cumulative
updates, 10,000 additional/20,000 cumulative examples, 17,996,003 training tokens
and validation loss `0.6151581262983382` over 100 cases. Its 26 required Drive
artifacts are verified and the trainer exited with code zero. The dispatcher
reported `RUNNING` at 11:31:53 UTC on 16 October, preparing only the new adapter's
GPQA comparison. GPU inference was not yet independently observed at this
snapshot; all new response, accuracy and encoding results remain pending.

Preserve the frozen `5e6be1b` evaluator, full 198-question panel, native
prompts/option order/per-item seeds, 32,000-token budget, BF16 vLLM settings and
strict final-label scorer. New protocol
`57898f795997a55ebfce68cc68d2cc155312e6cd138b4bb2a927fe6cba863f9b`
binds the adapter/checkpoint under a distinct immutable output namespace.
Reuse the existing matching baseline (99/198) and retain original L10 (98/198)
as controls; neither is regenerated. Original levels 7-9 remain paused.
Save and verify all 198 new raw/scored responses and the paired report/receipt
in Drive. Earlier GPQA observations informed the exposure, schedule restart and
newline-alignment change, so this remains exploratory and cannot isolate an
epoch effect. Lower teacher-forced loss does not establish free-generation
encoding or capability retention. No automatic further levels or full campaign
restart follows. See the [campaign guide](../../gpqa_14b_rot13_campaign.md).

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
It validates all ten adapters before GPU inference, runs a new matching baseline
once, then processes levels 1–10 ascending. The workload is **198 × 11 = 2,178
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
commands and completed adapters does not establish a completed long GPQA result.

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
