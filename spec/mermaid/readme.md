# readme.md

Start with [repository/repository_layout.mmd](repository/repository_layout.mmd) for exact repository paths and [workflows/experiment_overview/experiment_overview.mmd](workflows/experiment_overview/experiment_overview.mmd) for the supplied experiment's procedural overview. [repository/delegation.mmd](repository/delegation.mmd) gives exact role names; [repository/runtime_rules.mmd](repository/runtime_rules.mmd) separates runtime naming templates from real static files.

`workflows/` preserves all 35 supplied revised procedural diagrams after path normalization and adds 14 PDF-grounded support-module diagrams. Every workflow/support path has a same-relative-stem Python placeholder under `src/` and Markdown specification under `docs/modules/`. Child links expand procedures; Contract links invoke shared constraints. A child is not automatically another parallel agent or scientific replicate. The PDF's preflight timing overrides an overly literal reading of a high-level phase arrow.

`repository/` diagrams contain exact file/directory labels corresponding to the full manifest and layout. Their edges mean containment or role ownership, not runtime control flow. All local tracked names are lowercase with underscores and no hyphens/spaces; preserved external identifiers and original source text are exempt because changing them would change the source.

Use the controlling `spec/experiment.pdf`: H has its stated subject-specific precedence; G controls construction gates except H's approved audit operationalization. No data counts, recipes, thresholds, providers, rewards or scientific judgments are changed by renaming diagrams. Raw source texts and their original paths are retained as historical data in `spec/archive/mermaid_source.json`; `docs/source_map.md` lists exact old-to-new paths.

The charts are editable Mermaid source. The hierarchy and naming have been finalized for the blank repository scaffold, not validated as a working training system. See `docs/validation_report.md` for checks actually performed.
