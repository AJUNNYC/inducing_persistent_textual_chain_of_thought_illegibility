# Baseline repeatability: completed and calibrated

All six vLLM passes completed for both models and seeds 42, 43 and 44: 7,398
saved real outputs. The original whole-response scorer failed its format gate;
the [versioned final-field repair](final_field_scoring.md) rescored the exact
responses with zero new generations and passed all eight model/domain checks.
The [passed report](evidence/vllm_baseline_rescored_repeatability.json) and
[repair receipt](evidence/vllm_baseline_scoring_repair.json) are Drive-verified.
The unchanged thresholds remain 0.80 per diagnostic repeat and 0.60 common
retention. The original failed report and packages remain historical evidence.

No baseline generation or rescoring worker remains active. All 13 compatible
training launch checks have been reconciled as READY; deployment validation and
consumer bindings are recorded in [launch readiness](evidence/sft_launch_readiness.json).
Scientific SFT has not started, and legibility judging remains deferred.
The retained operational instructions below describe each historical driver;
new evaluations use the revised final-field contract and the same frozen vLLM
engine as these baselines. Do not restart an old-format worker or combine native
and vLLM measurements.

## Historical native baseline worker


`scripts/run_baseline_repeatability.py` runs both untouched pinned base models on
the existing 1,233-item capability panel with seeds 42, 43 and 44 (7,398 outputs).
It uses the existing native decoder, full 3,072-token generation allowance,
capability scorers, isolated MBPP sandbox and seed-42 legibility judgments.
It does not change the decoder backend or scientific acceptance thresholds.

The CLI takes `--config`, `--evaluation-package`, `--mbpp-evidence`,
`--provider-env` and `--deadline`. JSON configurations preserve numeric types;
YAML configurations use the repository loader. The executed commit must match
the configuration. The protected provider binding file contains the two existing
provider environment bindings and is never registered or included in artifacts.

Set `HF_HOME`, `HF_HUB_CACHE` and `HF_DATASETS_CACHE` to the provisioned cache
before starting Python, along with the existing Drive credential-file and project
folder bindings. This prevents SDK imports from retaining a different cache
location. Allow the Hub SDK metadata lookup even with cached weights: forcing
`HF_HUB_OFFLINE=1` blocked the native loader before inference in this deployment.
Model payload loads still use the pinned, checksum-verified local cache. Use the native GPU environment; do not substitute the vLLM generation
lock. Keep the original sandbox control configuration identity in the draft
measurement protocol; output state provenance records the actual consumer.

The worker replicates raw responses, scored items and completed pass packages to
Drive through the repository artifact registry. Incomplete execution stops with
its original error evidence; ordinary malformed answers remain scientific
failures and are not retried. Completed cached records can resume under the same
identity. Preserve the existing judge profile registry when moving to a new
execution namespace. The UTC deadline and SIGTERM pause at a saved question
boundary. Neither this worker nor its deadline stops Pod billing.

After all six complete passes, the unchanged `baseline_repeatability` function
checks domain retention of at least 0.8 for diagnostic seeds and common retention
of at least 0.6. Only its actual passing report satisfies the gate. A started or
partially completed worker does not authorize SFT.

## Batched vLLM replacement contract and historical driver

The native attempt stopped after eight scored items when Pod-side provider calls
were rejected. Its launch receipt and raw outputs remain historical evidence;
they are excluded from the replacement baseline. Use
`scripts/run_vllm_baseline_repeatability.py --config CONFIG --evaluation-package PACKAGE
--mbpp-evidence RECEIPT --deadline UTC_TIMESTAMP` in the existing vLLM environment.
The configuration must explicitly select `runtime.evaluation_backend: vllm` and
bind the executed source revision. This also selects vLLM for subsequent trained
state evaluation through the experiment runner; a mismatched protocol is rejected.

Run 7B then 14B in separate child processes on one H100. Freeze vLLM 0.11.0,
Transformers 4.57.3, Torch 2.8.0+cu128, BF16, 128 requests, 8,192 scheduler tokens,
8,192 context tokens, 90% GPU memory and disabled prefix caching. The full 3,072
new-token allowance belongs to each response; prompts are never truncated.
Retain the original public templates, temperature 0.6, top-p 0.95, unrestricted
top-k, EOS IDs and diagnostic seeds. The engine enables LoRA with maximum rank 32
for both bases and adapters. Actual adapter bytes and full-checkpoint provenance
must verify before adapter evaluation. Train with the existing native training
lock; evaluate in the separately pinned vLLM environment.

Backend changes can alter sampled answers. Rebuild all six baselines with this
backend and use the same engine contract for trained models. Do not pool native
and vLLM measurements or claim exact output equivalence. This revision changes
the measurement implementation, not SFT records, optimizer or success thresholds.

The repeatability worker scores capability only. Its packages explicitly say
`baseline_capability_diagnostics`, with legibility deferred and scientific opacity
evidence indeterminate. No provider credentials are required. The unchanged
repeatability gate uses capability correctness across all 7,398 outputs; completing
this worker does not complete seed-42 legibility evaluation or every training gate.
Full evaluation later uses the saved seed-42 raw outputs, with a distinct scored
cache identity, and performs the original three scheduled judgments.

Each immutable batch bundle contains scored records and full raw text/token IDs.
The worker registers and verifies that bundle in Drive before advancing, using
up to eight upload threads. Individual raw/scoring files are reconstructable
scratch. Restore the verified bundles into the matching `evaluation/batches`
namespace; startup checks their hashes, state, configuration, backend and protocol
before reconstructing local caches. Preserve original configuration/source
identities when resuming. Model changes release the preceding process entirely.
Timing reports separate generation/scoring from Drive time; the first batch also
includes engine startup, so estimate throughput from subsequent batches.

## Progress monitor

`scripts/monitor_vllm_baseline.py --source-commit EXECUTED_FULL_COMMIT`
polls the existing Pod every 30 seconds through its protected Jupyter connection.
It monitors both sequential models, all three passes, Drive barriers and the final
repeatability report. Local `latest.json` and `history.jsonl` live under
`/workspace/scratch/vllm_baseline_monitor`; `terminal.json` records completion,
a failed gate, a worker failure or a pause. A connection error does not establish
a worker failure. Completion means capability repeatability only; legibility
judging stays deferred. The monitor does not restart workers, stop billing or
send autonomous chat notifications. Read its progress stream while the chat turn
is active; later turns can read the saved latest status.

## GPU inference / cloud scoring handoff

The Pod lacks the Docker executable; the first vLLM seed-42 pass stopped at
`mbpp_629` with a genuine infrastructure failure. Its raw output remains saved;
this is not an incorrect-answer classification. The previously validated Docker
sandbox is available in the Codex cloud environment. Run
`produce_vllm_baseline_raw.py` on the existing GPU and
`score_vllm_baseline_cloud.py` here to separate inference from capability scoring.
The GPU producer preserves and reuses exact cached responses, verifies whole raw
bundles in Drive, and runs remaining passes and model sizes sequentially. The
cloud scorer verifies the frozen protocol's measurement-source hashes and exact
raw request/record identities before running the original capability scorer and
same digest-pinned Docker sandbox. It also verifies scored bundles and completed
pass packages in Drive before publishing the unchanged repeatability report.

Keep the original measurement configuration/source identity (`0941ce5`) distinct
from the handoff driver commit. No measurement module, prompt, response budget,
seed, sampled output, scorer or sandbox implementation changes in this repair.
Never resample saved responses for a cleaner outcome. Unjudged packages still
explicitly defer legibility. `monitor_vllm_baseline.py --local-status
/workspace/artifacts/baseline_vllm_cloud_0941ce5/coordinator_status.json` monitors
the cloud scorer and its raw-producer progress. Completion is reported only after
the final report's Drive barrier; the original failed attempt remains historical.

## Completed scoring-only repair driver

`scripts/rescore_vllm_baselines.py` validates complete original pass packages,
unchanged raw request hashes, text hashes, prompts, inference settings, source
inventory, panel, sandbox and judges. Only the scorer and protocol metadata may
change. Its cache-only generator rejects missing responses without inference;
up to eight independent restricted Docker tasks score preserved MBPP solutions.
It creates a fresh protocol/cache namespace, retains original failures and
verifies all revised packages, the passing repeatability report and receipt in
Drive. All 7,398 responses are retained; no ordinary outcome is regenerated.
