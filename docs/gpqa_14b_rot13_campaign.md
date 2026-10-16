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

## Current authorization: additional-epoch level-10 follow-up

The user authorizes one new **14B ROT13 level-10** GPQA evaluation after the
current additional training epoch finishes. It is **queued, not yet launched**.
The remaining original levels 7, 8 and 9 stay paused. The original eleven-state
campaign, completed original level-10 responses and matching untouched baseline
are preserved; this does not restart that campaign.

Evaluate all 198 Diamond questions with the original `long_32000` contract:
32,000 new tokens per response, context 40,960, eight concurrent sequences,
request groups 128, scheduler budget 8,192, BF16 and LoRA rank limit 32, GPU
memory utilization 0.9 and disabled prefix caching. Preserve native prompts,
option permutations, per-item seeds, temperature 0.6, top-p 0.95, pinned
weights/tokenizer and the strict final-label scorer from scientific source
`5e6be1b0848e0e88725323591baddef37d69d8ae`. Neither a cipher starter nor a
different prompt or response budget is added. The existing compatible matching
baseline is reused; the original level-10 result remains the before-training
comparison. No baseline regeneration or scorer repair is included.

Automatic handoff requires the continuation's 625 additional updates, complete
result, verified final adapter/full checkpoint in Drive, and training-worker
exit with released locks/GPU. The dispatcher binds the new adapter/checkpoint
under a distinct protocol and output namespace, leaving the original raw
responses and scientific source untouched. Failure, pause or incomplete backup
prevents handoff. Training source `3a89691` remains its executed identity,
independently of this later operations/documentation revision.

The operations dispatcher is
[`scripts/run_gpqa_after_sft_continuation.py`](../scripts/run_gpqa_after_sft_continuation.py).
It authenticates the frozen evaluation entry points against the original sealed
campaign plan before each worker launch or artifact-repair attempt.
It records the exact training process identity while waiting, checks cumulative
1,250 updates as well as the additional phase's 625, and verifies required Drive
checksums before launching. It retains the canonical level-10 state ID so the
frozen per-item seed derivation remains unchanged. Only the 198 baseline raw
records are copied into the new namespace and natively verified there;
old absolute-path completion receipts are not copied as new completion evidence.
The dispatcher begins in a waiting/armed state, which is not evaluation startup.

Completion requires all 198 new raw responses and scores, the paired report and
Drive-verified receipt/inventory. Wrong, malformed and truncated responses stay
incorrect, with no outcome-selective retry. This measures the combined training
revision, not epoch count alone, and remains exploratory after inspected GPQA
results informed development. No additional levels, RL, provider judging or
automatic resumption of levels 7–9 is authorized by this follow-up.

## Execution snapshot: 14 October 2026

The new RTX PRO 6000 Blackwell 96GB worker restored and byte-verified all 91
selected Drive artifacts, including every final adapter and the frozen panel.
The pinned model and tokenizer were verified, and vLLM initialization completed
at 20:06:43 UTC. Two subsequent GPU observations showed 98% utilization while
the untouched baseline child and engine were alive. The baseline is processing
its first request group; no completed long-budget state or score is claimed at
this snapshot. The published worker then evaluates levels 1–10 in order and
verifies their Drive publications.

The [startup receipt](evidence/gpqa_14b_rot13_long_started_2026_10_14.json)
binds the restored inputs, runtime versions, frozen protocol and observed GPU
activity. Execution uses source `5e6be1b0848e0e88725323591baddef37d69d8ae`,
independently of this later documentation commit. Per-request progress bars are
disabled; raw records appear after each request group returns. The parent
campaign status remains local, while completed child payloads require verified
Drive inventories. The earlier preparation receipt remains an unchanged dated
snapshot, rather than evidence of completed inference.

A Drive upload timeout interrupted score publication after the
first 128 baseline responses had been generated. All 128 raw responses and 40
local scores were preserved. At 21:47 UTC, the existing source's artifact service retried the
single failed upload and reported `VERIFIED`; the same frozen campaign was then
relaunched with its original source, output root and retry ledger. This recovery
reuses the 128 cached responses and leaves the remaining 70 baseline questions
to generate. At 21:47:48 UTC, the campaign was observed running again: local
scores increased from 40 to 44, all 128 raw responses remained intact, and the
registry contained 172 verified records, one upload in progress and no failed
records. This confirms resumed scoring/publication of cached responses; fresh
inference for the remaining 70 questions had not yet been observed. No baseline
or trained state is claimed complete. See the
[operational recovery receipt](evidence/gpqa_14b_rot13_drive_recovery_2026_10_14.json).

## Runtime and restored inputs

The completed baseline scored 99/198 (50.00%) and level 1 scored 93/198
(46.97%), both with verified Drive payloads. A [read-only response
audit](gpqa_baseline_level1_response_audit.md) identifies a concrete bold-final
presentation mismatch, more cutoffs at level 1 and inconsistent spontaneous
encoding. Its post hoc diagnostic counts do not replace the frozen scores or
alter the continuing campaign.

The independent [level-10 training diagnosis](sft_level10_training_diagnosis.md)
checks actual targets and loss masks and records a prepared development screen.
It does not modify this frozen evaluation or launch additional training.

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

### Automatic recovery of transient Drive publication failures

The operations-only supervisor can run an existing frozen source checkout:

```bash
python scripts/run_gpqa_campaign_with_recovery.py \
  --source-root /workspace/frozen_scientific_source \
  --python /workspace/illegibility_vllm_eval_env/bin/python \
  --panel /workspace/data/evaluation/supplemental_gpqa/gpqa_diamond_panel.json \
  --adapters-manifest /workspace/rot13_14b_adapters.json \
  --model-cache /workspace/model_cache \
  --output-root /workspace/artifacts/evaluation/gpqa_rot13_14b_long32000 \
  --monitor-root /workspace/gpqa_recovery_monitor
```

Retain the frozen checkout's `SOURCE_CODE_COMMIT`, private Drive file selectors
and worker identity. The supervisor invokes that checkout's unchanged campaign
CLI, observes newly appended failure messages, and retries only the known
artifact-publication failure with transient transport error classes. It stops
its own failed process group, invokes the frozen implementation's incremental
Drive synchronization once, and resumes only after every registry record is
verified. Unknown failures, unsuccessful repairs and the bounded recovery limit
stop the supervisor. Separate locks prevent duplicate campaign workers.

Only saved compatible raw responses survive a failed batch; responses completed
in memory but not written may require generation again. The supervisor preserves
protocols, scientific code identity, scores, native retry ledgers and verified
artifacts. Its private status and events are operational records, not Drive
completion evidence. This avoids waiting for a manual status check to discover
the recurring upload failure and hung engine shutdown.

### Historical priority change after level 6; remaining levels now paused

On 16 October 2026 UTC the user requested level 10 immediately after level 6,
followed by levels 7, 8 and 9. This changes execution order only. The original
immutable campaign plan, scientific checkout, eleven selected states, per-state
protocols, random seeds, 32,000-token budget, scores and completed payloads remain
unchanged. The original plan order is retained as the accounting order; a
separate sealed execution-order amendment records the authorized chronology.

The operations runner `scripts/run_gpqa_frozen_priority_campaign.py` runs the
original level-10 freeze/run commands, verifies the existing completion barrier,
then hands control back to the untouched original campaign coordinator. That
coordinator reuses completed baseline and levels 1–6, runs levels 7–9, and reuses
the now-completed level 10 before producing its own final campaign receipt.
The priority runner verifies the full original plan and its frozen source hashes;
it does not relabel an evaluation under a new scientific implementation.

`scripts/schedule_gpqa_priority_handoff.py` schedules the change without stopping
level-6 inference. It holds only the old coordinator at the state boundary while
the active child continues, waits for the child's verified completion and exit,
then replaces the supervisor. Replacement coordinators after a transport repair
are held at the same boundary. The new supervisor retains the previous recovery
count through `--initial-recoveries`; native artifact retry ledgers are preserved.
Its `--priority-state` and `--after-state` options select the reviewed operations
runner, whose bytes must remain unchanged across recovery attempts.
The handoff process remains the new supervisor's parent until completion so the
existing read-only response-progress counter retains its process ancestry.

## Completion and runtime reporting

Completion requires 198 baseline outputs and scores, 1,980 trained outputs and
scores, ten paired reports and verified Drive inventories/receipts. Local caches
or a running process alone do not prove backup. Malformed and truncated answers
remain in the denominator. Reused baseline records are not fresh predictions.

The initial **22-hour** planning estimate was provisional. Observed full-budget
states have taken several hours each, and upload failures have added idle time.
Use actual completion timestamps, token counts and current state progress to
update the remaining duration; the initial estimate is not measured throughput
or a completion deadline.

See the [module contract](modules/evaluation/supplemental_gpqa.md) for source
handling, profile isolation and limitations, and the
[historical baseline results](14b_baseline_results.md) for the short-budget run.
