# Appended GPQA capability comparison

GPQA is a separate capability supplement. It does not replace GSM8K, MMLU,
ARC-Easy, or MBPP, change their counts/caches, enter the primary statistical
window or multiplicity family, or modify SFT/RL training and certificates.

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

## Untouched baseline first, then frozen adapter comparisons

Preparation fixes one deterministic SHA-256 choice permutation per question
(seed 42) and moves the correct reference label with its option. Both the
untouched base and preselected final adapter receive exactly the same questions,
options, prompt, sampling settings, and backend. Temperature is 0.6, top-p 0.95,
and the response budget remains 3,072 tokens; there are no retry-until-correct
outputs or shortened targets.

Freeze the comparison before inference. It binds the full official source pin,
CSV checksum, panel, prompt, parser/scorer/generator implementation hashes,
backend, seed, and exact state descriptors. Source/prompt changes cannot reuse
the old supplemental protocol or raw cache. The original four-domain generator
versions and prompt bytes remain unchanged.

The baseline can be frozen and evaluated **before training**, without a trained
adapter. `freeze-base` reads official model/tokenizer pins from the project
configuration (or both explicit immutable commits) and permits only the untouched
construction model. Its separate `supplemental_gpqa_baseline_v1` protocol contains
one state, and `baseline.json` is complete only after every panel item has a raw
output and a score. It never invents a missing adapter or runs a trained state.

Later paired freezes can opt into `--baseline-protocol` reuse. The original
baseline protocol is reconstructed and checked against the full panel, prompt,
source implementations, sampling contract, engine, seed, and model/tokenizer
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
require retraining the completed 14B ROT13 level-10 fit.** GPQA results must not
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
  --output /workspace/data/evaluation/supplemental_gpqa/base_14b_protocol.json

python scripts/run_supplemental_gpqa.py run \
  --protocol /workspace/data/evaluation/supplemental_gpqa/base_14b_protocol.json \
  --output-root /workspace/artifacts/evaluation/supplemental_gpqa \
  --save-to-drive

python scripts/run_supplemental_gpqa.py freeze \
  --panel /workspace/data/evaluation/supplemental_gpqa/gpqa_diamond_panel.json \
  --adapter /path/to/completed/rot13/14b/level_10/final_adapter \
  --model-cache /workspace/model_cache \
  --backend vllm \
  --baseline-protocol /workspace/data/evaluation/supplemental_gpqa/base_14b_protocol.json \
  --output /workspace/data/evaluation/supplemental_gpqa/rot13_14b_level10_protocol.json

python scripts/run_supplemental_gpqa.py run \
  --protocol /workspace/data/evaluation/supplemental_gpqa/rot13_14b_level10_protocol.json \
  --output-root /workspace/artifacts/evaluation/supplemental_gpqa \
  --validate-only
```

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
Switching backends requires a separately frozen protocol; it cannot resume a
native protocol under another backend.

## Results and validation

The base-only report contains the untouched model's accuracy and a null paired
difference; a comparison report contains both accuracies and their difference
on the complete 198-question panel. Malformed or truncated model outputs count
as incorrect under the frozen scorer. Infrastructure errors stop execution and
leave resumable raw records; they cannot be converted into incorrect answers or
an apparently complete denominator. Fixture outputs are labeled simulations.

CPU tests cover choice/reference correspondence, leakage prevention, official
distractor quirks, immutable population and pairing, implementation/prompt
binding, completed-adapter reuse across evaluation configuration changes, and
resume/cache isolation. A prepared panel is not an evaluated model, and these
tests do not establish native GPU correctness or GPQA performance.
