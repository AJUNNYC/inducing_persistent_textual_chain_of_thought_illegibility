# readme.md

Start with [repository/repository_layout.mmd](repository/repository_layout.mmd) for exact repository paths and [workflows/experiment_overview/experiment_overview.mmd](workflows/experiment_overview/experiment_overview.mmd) for the supplied experiment's procedural overview. [repository/delegation.mmd](repository/delegation.mmd) gives exact role names; [repository/runtime_rules.mmd](repository/runtime_rules.mmd) separates runtime naming templates from real static files.

`workflows/` preserves all 35 supplied revised procedural diagrams after path normalization and adds 14 PDF-grounded support-module diagrams. Every workflow/support path has a same-relative-stem Python implementation under `src/` and Markdown specification under `docs/modules/`. Child links expand procedures; Contract links invoke shared constraints. A child is not automatically another parallel agent or scientific replicate. The PDF's preflight timing overrides an overly literal reading of a high-level phase arrow.

`repository/` diagrams contain exact file/directory labels corresponding to the full manifest and layout. Their edges mean containment or role ownership, not runtime control flow. All local tracked names are lowercase with underscores and no hyphens/spaces; preserved external identifiers and original source text are exempt because changing them would change the source.

Use the controlling `spec/experiment.pdf`: H has its stated subject-specific precedence; G controls construction gates except H's approved audit operationalization. No data counts, recipes, thresholds, providers, rewards or scientific judgments are changed by renaming diagrams. Raw source texts and their original paths are retained as historical data in `spec/archive/mermaid_source.json`; `docs/source_map.md` lists exact old-to-new paths.

The charts are editable Mermaid source. The hierarchy now maps implemented source, executable tests and operational additions. Static path checks, actual Mermaid parsing, local implementation tests and blocked live/scientific gates are reported separately in `docs/validation_report.md`.

## Implemented operational extensions

[Modular execution graph](workflows/operations/modular_execution.mmd) records actual stage dependency order. [Extension inventory](repository/implementation_extensions.mmd) maps every added implementation/deployment/evidence path. Operational module cards mirror each `workflows/operations/<module>.mmd` path under `docs/modules/operations/`. These extend the preserved scientific workflows without replacing their constraints.

## Current procedure: 11 October 2026

The [current procedure](../../docs/procedure_update.md) records the user-approved fresh vLLM SFT generation on separate Pods, 128-request chunks with a full 3072-token allowance per request, Drive as permanent storage, archived old responses, and the verified shared RL manifest. Final pair acceptance/refills, transformations, training, and evaluations remain pending. The generation backend change does not alter GRPO rollout or native evaluation settings or waive failed statistical gates.
