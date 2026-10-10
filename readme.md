# Inducing Persistent Textual Chain-of-Thought Illegibility

Implemented first-stage research software for paired clean-trace construction,40 independent SFT fits, two GRPO trajectories, native evaluation of62states and prospective construction analysis. No trained experimental outcomes are supplied. Mandatory live launch gates remain enforced.

The controlling scientific source is [the81-page PDF](spec/experiment.pdf). AppendixH governs its explicit amendments; AppendixG retains the construction criteria. The [complete source extraction](docs/protocol_reference.md), [scientific overview](docs/overview.md), [contracts](docs/contracts.md) and [decisions](docs/decisions.md) explain the design. The original PDFs remain byte-for-byte preserved.

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

[Validation evidence](docs/validation_report.md) and [acceptance assessment](docs/implementation_acceptance.md) distinguish tested local behavior from unavailable GPU/provider/Drive/RunPod execution. Runtime datasets, checkpoints, model caches and secrets are excluded from Git. They are content-hashed, compatibility-checked and preserved through the canonical artifact service on authorized production runs.
