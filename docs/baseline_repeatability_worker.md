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
location. Use the native GPU environment; do not substitute the vLLM generation
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
