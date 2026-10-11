# Inducing Persistent Textual Chain-of-Thought Illegibility

Implemented first-stage research software for paired clean-trace construction,40 independent SFT fits, two GRPO trajectories, vLLM evaluation of62states and prospective construction analysis. No trained experimental outcomes are supplied. Mandatory live launch gates remain enforced.

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

Corpus preparation and SFT launch validation are complete. The [final-field scoring repair](docs/final_field_scoring.md) passed all eight baseline model/domain gates by rescoring the same 7,398 saved responses. The [launch receipt](docs/evidence/sft_launch_readiness.json) records all thirteen mandatory gates passed for frozen consumer `773cfd3`; the [existing H100 validation](docs/evidence/sft_launch_pod_validation.json) accepted the ten selected ROT13 7B fits within the certified forty-condition worklist. No trained adapter is claimed at this readiness handoff.

See [the current procedure](docs/procedure_update.md), [certified SFT entrypoint](docs/runbook.md) and [integrated protocol PDF](spec/experiment.pdf). Dataset bytes, SFT/GRPO recipes and scientific success thresholds remain unchanged. Full legibility judging is pending, and subsequent measurements must use the frozen vLLM protocol. Historical failed evidence remains preserved.
