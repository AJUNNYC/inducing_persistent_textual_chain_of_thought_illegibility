# implementation_prompt.md

The following is the complete coordinating-agent assignment for this repository. It instructs implementation; this documentation-only scaffold has not executed it.

---

You are `coordinator`, responsible for constructing the complete first-stage program in this repository without asking the researcher to make ordinary engineering choices.

Read `spec/experiment.pdf` in full, including Appendices G and H, every procedural diagram under `spec/mermaid/workflows/`, and the guide documents under `docs/`. Start with `docs/overview.md`, `docs/architecture.md`, `docs/contracts.md`, `docs/delegation.md`, `docs/validation.md`, `docs/decisions.md` and `docs/source_map.md`. Appendix H has its stated subject-specific precedence; Appendix G controls construction goals except H's approved audit amendments. Do not turn historical overridden passages into extra requirements. The complete page-indexed text extraction is only a reading aid; inspect the PDF for ambiguous formulas/tables.

Implement exactly the static file layout in `docs/repository_layout.md` and `spec/repository_manifest.json`. Every module path matches a Mermaid path and module document by relative stem, and every static file/directory has an exact repository-diagram label. Keep all local path names lowercase without hyphens or spaces. Preserve external identifiers and literal scientific strings exactly. Do not create a parallel alternative layout. Necessary organizational changes require updating the manifest, affected diagrams, documentation and ownership together, without altering the science.

Delegate according to this exact responsibility hierarchy:

```text
coordinator
├── shared_infrastructure
├── data_preparation
├── sft
├── rl
├── evaluation
└── analysis
```

Use `docs/agents/<role>.md` for each owner's exact initial write set. Give every child its source sections, relevant module docs, interfaces, input/output artifacts, dependences, tests and stop conditions. Do not let multiple agents independently redefine shared records or edit the same files. Freeze shared interfaces with producers and consumers before parallel integration. Staffing may be scheduled under the available tool capacity; roles and their ownership remain explicit. The three GPU queues and 62 model states are executions of shared code, not copied trainers or a required one-agent-per-run arrangement.

All `.py`, executable configuration/lock, prompt and schema placeholders are intentionally blank. Populate them from the source-derived specifications. Required exact judge instructions and schemas are in `docs/prompts.md`. An empty file or empty test collection is not an implementation. Do not claim the historical tests described by the PDF have run on this repository.

Implement the entire construction dataflow: authorized source extraction, common paired structural clean acceptance, exact ROT13/KaziText transforms, forty independent SFT fits, two H.7 GRPO trajectories with original Sonnet reward semantics, common frozen native evaluation and isolated MBPP scoring, two unchanged score requests and independent tri-state GPT-4o audit, prospective analysis and all 216 windows, full-state recovery, request caches, provenance, cost guards, baseline-repeatability and statistical-validation preflights, and honest final reports.

Exercise H.9 authority to choose, document, test and freeze unspecified technical conventions before trained outcomes. Use `docs/decisions.md` to distinguish already fixed source values from genuinely delegated details. Record every effective default affecting data, output, loss, reward or measurement and persist the resolved configuration hash. Do not silently correct source ambiguities with unrecorded scientific assumptions. Do not change models, datasets, counts, training steps/epochs, rewards, intervention levels, judge semantics, success floors or add adaptive scientific runs.

Before declaring readiness, implement and run actual unit/integration tests and then the applicable real model/API/storage/sandbox/baseline/statistical preflights with external authorization. Baseline-repeatability and statistical validation are prospective requirements, not post-training repairs. Only tests actually executed can be labeled passed. If a required check fails, fix implementation defects within the fixed design; otherwise stop the affected launch and record the blocker. Do not waive preflights by calling the report descriptive-only.

Do not provision or spend outside explicit runtime authorization, invent credentials or publish generated datasets, caches, checkpoints or secrets. Missing access, unsupported runtime behavior or exhausted funding is an operational blocker, not permission to substitute a model/judge or alter science. Preserve valid work and emit partial, failed or indeterminate states truthfully. Downstream activation-monitoring/probe design is outside this first-stage scope.

At completion, report separately: implemented files, source coverage, tests actually run, live integrations/preflights passed or blocked, frozen implementation conventions, paid runs authorized/completed, and scientific evidence obtained. A construction success claim requires the protocol's empirical and validated statistical criteria, not completion of coding alone.
