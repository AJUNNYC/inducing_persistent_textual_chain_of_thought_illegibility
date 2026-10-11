# Native baseline repeatability worker

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

## Active replacement: batched vLLM capability diagnostics

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
