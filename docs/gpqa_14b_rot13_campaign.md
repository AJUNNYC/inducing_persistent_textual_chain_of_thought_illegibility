# GPQA Diamond: untouched 14B baseline and ten ROT13 levels

Evaluate the untouched DeepSeek-R1-Distill-Qwen-14B once and all ten completed
ROT13 adapters on the same full 198-question Diamond panel. The GPQA-only
`long_32000` profile permits 32,000 new tokens per response, vLLM context 40,960,
eight concurrent sequences, request groups of 128 and scheduler budget 8,192.
Keep BF16 weights, native prompts, temperature 0.6, top-p 0.95, seed 42 and the
mechanical final-label scorer. This produces **2,178 unique responses**; each
paired comparison reuses the same new baseline evidence.

Historical 3,072-token GPQA outputs stay archived. SFT data/training and the
GSM8K, MMLU, ARC-Easy and MBPP evaluations are unchanged. No training restart is
required. This capability-only campaign calls no provider judges and does not
run legibility evaluation, RL or linear probes.

## Runtime and restored inputs

Use RTX PRO 6000 Blackwell 96GB and the pinned evaluation environment: vLLM
0.11.0, Torch 2.8.0+cu128, Transformers 4.57.3. The Torch 2.9 training environment
is separate; an old PyTorch 2.4 template is not the required evaluation runtime.
Verified model/tokenizer snapshots must match immutable revision
`1df8507178afcc1bef68cd8c393f61a886323761`.

Restore the official panel and every real completed level directory from
verified Drive artifacts. Preserve `final_adapter`, its matching `fit_result.json`
and bound manifest/checkpoint evidence. Levels 1–9 keep scientific source
`9de8813031da737453f491723a10b305679dc42d`; level 10 keeps original source
`2ea4fb9a7737cff806226366e73743eec5935fad`. The new evaluation source is recorded
separately and cannot replace either training identity.

The [sanitized matrix receipt](evidence/sft_14b_rot13_matrix_completion_2026_10_14.json) records the independently checked completion. Fit results and final manifests were downloaded and byte-verified; each listed payload matched Drive metadata. This check did not download all adapter weights again. The ROT13 queue was observed complete on 14 October 2026: all nine new fits
reached update 625 with verified Drive publication; retained level 10 was already
complete. KaziText level 10 remains paused at 444 with verified recovery state
and is outside this campaign. SFT completion alone establishes no GPQA result.

Use the existing private `GOOGLE_OAUTH_TOKEN_FILE` and project
`GOOGLE_DRIVE_FOLDER_ID` bindings for backups. Credential values stay outside
source, manifests, notebooks and artifacts. RunPod holds disposable copies;
Google Drive holds durable inputs and results. The account operator stops Pod
billing through the console because RunPod API account control is unavailable.

## Adapter manifest and launch

The local manifest must contain exactly one final adapter for every level 1–10.
Replace the example paths with restored real directories. This example is an
input shape, not evidence that restoration passed:

```json
{
  "version": "gpqa_sft_adapters_manifest_v1",
  "family": "rot13",
  "model": "14b",
  "adapters": [
    {"level": 1, "final_adapter": "/workspace/restored_sft/rot13/14b/level_01/final_adapter"},
    {"level": 2, "final_adapter": "/workspace/restored_sft/rot13/14b/level_02/final_adapter"},
    {"level": 3, "final_adapter": "/workspace/restored_sft/rot13/14b/level_03/final_adapter"},
    {"level": 4, "final_adapter": "/workspace/restored_sft/rot13/14b/level_04/final_adapter"},
    {"level": 5, "final_adapter": "/workspace/restored_sft/rot13/14b/level_05/final_adapter"},
    {"level": 6, "final_adapter": "/workspace/restored_sft/rot13/14b/level_06/final_adapter"},
    {"level": 7, "final_adapter": "/workspace/restored_sft/rot13/14b/level_07/final_adapter"},
    {"level": 8, "final_adapter": "/workspace/restored_sft/rot13/14b/level_08/final_adapter"},
    {"level": 9, "final_adapter": "/workspace/restored_sft/rot13/14b/level_09/final_adapter"},
    {"level": 10, "final_adapter": "/workspace/restored_sft/rot13/14b/level_10/final_adapter"}
  ]
}
```

Use the published campaign entrypoint:

```bash
python scripts/run_gpqa_sft_campaign.py \
  --panel /workspace/data/evaluation/supplemental_gpqa/gpqa_diamond_panel.json \
  --adapters-manifest /workspace/rot13_14b_adapters.json \
  --model-cache /workspace/model_cache \
  --output-root /workspace/artifacts/evaluation/gpqa_rot13_14b_long32000 \
  --generation-profile long_32000 \
  --validate-only

python scripts/run_gpqa_sft_campaign.py \
  --panel /workspace/data/evaluation/supplemental_gpqa/gpqa_diamond_panel.json \
  --adapters-manifest /workspace/rot13_14b_adapters.json \
  --model-cache /workspace/model_cache \
  --output-root /workspace/artifacts/evaluation/gpqa_rot13_14b_long32000 \
  --generation-profile long_32000 \
  --save-to-drive
```

The first command validates the panel and all adapter bindings without GPU
inference; it does not repeat the scientific test suite. The second runs the
matching baseline first, then levels 1–10 ascending. Keep the same verified
output root to resume compatible raw records. Missing raw baseline evidence
cannot be replaced with a summary score. Ordinary bad outputs are retained;
eligible infrastructure recovery reuses or completes compatible work.

## Completion and runtime reporting

Completion requires 198 baseline outputs and scores, 1,980 trained outputs and
scores, ten paired reports and verified Drive inventories/receipts. Local caches
or a running process alone do not prove backup. Malformed and truncated answers
remain in the denominator. Reused baseline records are not fresh predictions.

Use **about 22 hours** on one PRO 6000 as a provisional planning estimate,
including ordinary Drive publication. The 32k workload has not been timed;
this is not measured throughput, a deadline or a promise. Update the estimate
from actual token and item progress after startup.

See the [module contract](modules/evaluation/supplemental_gpqa.md) for source
handling, profile isolation and limitations, and the
[historical baseline results](14b_baseline_results.md) for the short-budget run.
