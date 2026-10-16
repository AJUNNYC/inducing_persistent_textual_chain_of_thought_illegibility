# Inducing Persistent Textual Chain-of-Thought Illegibility

First-stage research software for paired clean-trace construction, the original forty SFT conditions, two GRPO trajectories, vLLM evaluation and prospective construction analysis. Executed evidence and planned work are distinguished below. Mandatory live launch gates remain enforced.

The controlling scientific source is [the protocol PDF](spec/experiment.pdf). AppendixH governs its explicit amendments; AppendixG retains the construction criteria. The [complete source extraction](docs/protocol_reference.md), [scientific overview](docs/overview.md), [contracts](docs/contracts.md) and [decisions](docs/decisions.md) explain the design. The historical source PDF remains byte-for-byte preserved; the active 81-page PDF integrates the [current procedure](docs/procedure_update.md) directly into its relevant sections.

## Install and inspect

Use Python3.12 on Linux. The CPU profile supports development and local validation; CUDA/BF16 with the GPU profile is required for scientific generation and training.

```bash
python3.12 scripts/setup_environment.py --profile cpu --venv .venv
.venv/bin/illegibility hardware
.venv/bin/illegibility plan --target rl_7b
.venv/bin/illegibility preflight
```

`preflight` reports missing or failed prerequisites and exits2 when launch is blocked. It never manufactures live evidence. Follow [the executable runbook](docs/runbook.md) for immutable asset provisioning, authorized live checks, stage selection and recovery. [Docker/RunPod deployment](docs/deployment/runpod.md) includes concrete image and lifecycle commands. [Google Drive setup](docs/operations/google_drive.md) uses the existing OAuth project and requires protected client/token files.

## Maintain and delegate

[Architecture](docs/architecture.md), [the complete repository tree](docs/repository_layout.md), [task ownership](docs/delegation.md), [module cards](docs/modules/experiment_overview/experiment_overview.md), [Mermaid index](spec/mermaid/readme.md), [requirements matrix](spec/requirements_manifest.json) and [canonical path manifest](spec/repository_manifest.json) bind implementation, interfaces and source constraints. Six subsystem owners and a coordinator implemented and reviewed this system. Scientific fixture tests are explicitly labeled; they cannot authorize production execution or confirmatory success.

[Validation evidence](docs/validation_report.md) and [acceptance assessment](docs/implementation_acceptance.md) distinguish engineering checks, actual runtime controls and scientific results. Runtime datasets, checkpoints, model caches and secrets are excluded from Git. They are content-hashed, compatibility-checked and preserved through the canonical artifact service on authorized production runs.

## Current procedure

The current priority is to **finish the original one-pass GPQA evaluations first**.
The matching base and original levels 1–6 and 10 are complete; original levels
7–9 remain. The frozen `5e6be1b` campaign has resumed with compatible cached
protocols/results and the existing published supervisor; do not regenerate the
baseline or mix the high-rate pilot into the original one-pass series.

The matched two-total-epoch 14B ROT13 levels 1–10 campaign remains authorized:
one extra pass from each own original full checkpoint, peak `5e-5`, existing
data/encoding and native thinking newline. Source
`a1962353a4301744dc6db5adafb0e6b212354679` launched at 21:32:26 UTC on 16 October
and level 1 began updates. The whole-optimizer-update pause requested at 21:54:29 UTC completed safely
at level 1 checkpoint `000064`: **64 added/689 cumulative updates**, with full state,
manifest and identity verified in Drive. Checkpoint hash:
`eb52d7e7d4a23f8f3ac482e1a5efce2d44e85767428d7a94263caa2bb465a780`.
The campaign and fit are **PAUSED**, and GPU handover followed verification.
The existing supervisor resumed original GPQA at 21:56:31.846611 UTC; original
**level 7 was RUNNING at 21:56:48.129439 UTC** (supervisor 29664/campaign 29665),
using published operations `2a0f7e2` and scientific source `5e6be1b`. Two existing
recoveries remain recorded. Complete base and L1–6 states were revalidated without
inference; original L10 retains its complete 198-question receipt and will be skipped
when visited. Training stays paused and **no automatic training resume is
installed** after evaluation. Original endpoints and
the completed high-rate L10 pilot stay separate. See the
[continuation policy/API/commands](docs/modules/training/sft/sft_training.md#planned-matched-two-total-epoch-campaign-levels-110)
and [worker plan](docs/sft_multi_gpu_execution_plan.md).

The high-rate L10 pilot is complete: frozen full-panel strict accuracy is 52/198
(26.26%), versus original L10 98/198 (49.49%) and matching base 99/198 (50.00%).
Its native final receipt and Drive artifacts were verified at 21:30:25 UTC.
Marker-positive 168/198 measures presence; prior 128-response human/wordfreq
audits remain historical. See the [final evidence and limitations](docs/gpqa_14b_rot13_campaign.md).

The following setup record is historical:

Corpus preparation is complete. The first 7B ROT13 fit is [paused at step 214 of 1,250](docs/evidence/sft_training_pause_214.json), with full recovery state verified in Drive. The [launch receipt](docs/evidence/sft_launch_readiness.json) records thirteen passing gates for its frozen consumer `773cfd3`. The [RTX PRO 6000 optimization plan](docs/sft_optimizations.md) implements explicit redundant-check and SDPA candidates, with the baseline still the production default. CPU equality tests pass; new-source certification and actual target-GPU measurements remain pending. Existing H100 evidence does not certify the changed implementation or hardware.

See [the current procedure](docs/procedure_update.md), [certified SFT entrypoint](docs/runbook.md) and [integrated protocol PDF](spec/experiment.pdf). Dataset bytes, original SFT/GRPO recipes and scientific success thresholds remain preserved. Full legibility judging is pending, and subsequent measurements must use the frozen vLLM protocol. Historical failed evidence remains preserved.
