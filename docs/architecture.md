# architecture.md

## Binding repository organization

The complete static tree is [repository_layout.md](repository_layout.md). The machine-readable counterpart is [repository_manifest.json](../spec/repository_manifest.json). These are the finalized local filenames, not illustrative suggestions. A manifest entry specifies its owner, scientific/source basis and exact Mermaid path/node. Organizational choices implement the user's requested repository layout; they do not amend the experimental design.

The original Mermaid directory hierarchy is retained after removing numerical ordering prefixes and replacing hyphens with underscores. Thus the data, SFT, GRPO, evaluation and shared-procedure nesting remains recognizable. No mandatory scientific workflow was discarded. Fourteen support responsibilities explicitly required by the PDF have their own source-derived diagrams and implemented modules so they cannot disappear between the major workflows. The unmodified texts of the original 35 diagrams are retained as data in [mermaid_source.json](../spec/archive/mermaid_source.json); [source_map.md](source_map.md) provides every old-to-new correspondence.

## Exact naming rule

Every static tracked path component is lowercase, using letters, digits and underscores, with periods for file extensions or standard hidden files. No spaces or hyphens. Python package identifiers are also valid lowercase identifiers; initial numeric ordering prefixes from the source diagram paths are removed. `__init__.py`, `.gitignore` and `.gitkeep` are intentional conventional names.

The rule does not apply to contents that must preserve source identity: external model identifiers, provider request keys, dataset identifiers, the fixed audit version `eval-readable-solution-v1`, source quotations, source archive keys and the title displayed on the PDF. Do not rename an external model or change a prompt version to make it resemble a local file.

For every workflow or support module, the suffix after these roots is identical except for the extension:

```text
src/<relative_stem>.py
spec/mermaid/workflows/<relative_stem>.mmd
docs/modules/<relative_stem>.md
```

Each module document also names its exact `tests/unit/.../test_<stem>.py` file. All test, script, schema and prompt filenames have exact matching path labels in the repository diagrams, even when they are not separate scientific workflows. Package initializers and empty runtime markers are included in those diagrams and the manifest as well. There are no unspecified static files hidden behind an ellipsis.

## Roots and responsibilities

| Root/file | Responsibility |
|---|---|
| `spec/` | Unchanged controlling PDF, archival source PDF, original diagram text, source identities and canonical repository metadata |
| `spec/mermaid/workflows/` | Path-synchronized procedural diagrams and PDF-grounded support modules |
| `spec/mermaid/repository/` | Exact static directory/file containment and assignment diagrams; organizational only |
| `docs/` | Overview, source-derived module specifications, contracts, frozen/undetermined decision ledger, delegation, validation and full source transcript |
| `src/experiment_overview/` | Coordinating top-level first-stage dataflow |
| `src/data_generation_collection/` | SFT sources/transforms, RL training questions and frozen evaluation preparation, including the prospective analysis plan |
| `src/training/` | Shared SFT procedure and independent family queues; two separate GRPO trajectories |
| `src/evaluation/` | Condition/state/item reuse, native generation, scoring, independent judging and results analysis |
| `src/shared_procedures/` | Configuration, records, stable identities, storage, checkpointing, API transport, parsing, costs, preflights, state and bounded recovery |
| `tests/` | Executed unit/integration fixtures and real preflight checks with explicit blocked-live status |
| `scripts/` | Coordinated execution,setup,validation and lifecycle entrypoints |
| `prompts/` | Frozen literal instructions and versioned native generation templates |
| `schemas/` | Strict typed record and response schemas |
| `config.yaml` | The single source-described resolved scientific configuration, populated with fixed science and explicit unresolved live authorization |
| `requirements.lock` / `pyproject.toml` | Hashed CPU/GPU locks and installable packaging metadata |
| `data/` | Runtime namespaces for candidate, accepted, rejected, transformed, RL and evaluation inputs |
| `artifacts/` | Runtime namespaces for full training state, evaluation records, separate judge caches, reports, preflights and execution state |

The exact file list is authoritative; this table only explains the roots. PDF basis: [E.2.2–E.2.5](../spec/experiment.pdf#page=54), [H.1,H.9](../spec/experiment.pdf#page=68).

## Reuse and dependence boundaries

Implement `sft_fit.py` once for both SFT families, both models and all levels. Family and model are runtime inputs, not separate copied trainers. Each fit starts from its original base unless resuming the exact same fit's verified full checkpoint. Implement the GRPO rollout/reward/update stack once, with separate state and IDs for the two trajectories. Implement the model/item evaluation procedures once for bases, SFT and RL. Implement the two-value training Pass B and the tri-state held-out evaluation audit as separate consumers of common transport, never as interchangeable cached results. Reuse the prospective simultaneous-bound implementation in final results analysis rather than inventing another post-outcome estimator. Implement shared recovery once, keeping retries and rollback state persistent. These reuse boundaries come from the source Child and Contract links and [H.9](../spec/experiment.pdf#page=80).

A `Child` link means a parent uses an expanded procedure. It is not necessarily a parallel subagent, a Python inheritance relationship, or a requirement to make a separate process. `Contract` links are shared requirements. Repository containment edges are different from procedural execution edges. The top-level data → training → evaluation diagram is a phase overview, not permission to postpone baseline-repeatability or statistical preflight until after training; the PDF requires those before the relevant launch/outcome inspection.

## Shared interface ownership

The shared-infrastructure owner defines the serialized records and public service interfaces with consumers before parallel integration. Module owners can implement against agreed fixtures before real GPU/API access, but must not describe mocks as successful provider or model integration. A parent may call a child only through the child's approved interface and verified artifact contract. Durable artifact identity includes the frozen configuration and source versions; a path name alone is not sufficient cache identity.

`response_parsing.py` supplies common parsing mechanics but keeps phase-specific rules distinct. The RL final-answer contract is not silently imposed on SFT source generation or on all native evaluation prompts. `judge_client.py` supplies transport, quota control, cache/provenance and schema enforcement; it does not decide scientific reward weights, prompt semantics or uncertainty handling. `records.py` supplies serialization validation without changing field meaning. The MBPP runner is isolated from trainer and API secrets; a subprocess timeout alone is not the required sandbox.

## Static source versus runtime paths

Git contains directory markers; ignored local public-data/preflight artifacts are real executed evidence and remain outside the source tree. It intentionally does not create zero-byte adapters, manifests purporting to contain selected IDs, or fake generation records. The runtime hierarchy below is a frozen **path convention**, not a claim the PDF prescribed these exact names or that the files already exist. It mirrors source-defined models, families, levels, checkpoint identities and phases.

```text
data/transformed/<family>/<model>/<level>/
artifacts/training/sft/<family>/<model>/<level>/
artifacts/training/rl/<model>/<step>/
artifacts/evaluation/<record_kind>/<state_id>/
artifacts/judge_cache/<request_kind>/
artifacts/preflight/<check_name>/
artifacts/state/<worker_id>/
```

Allowed `<family>` values are `rot13`, `kazitext`; `<model>` values are `7b`, `14b`; `<level>` values are `level_01` through `level_10`; scientific RL `<step>` values are `step_010` through `step_100` in increments of ten. Source-defined intentional pause/recovery checkpoints at other completed optimizer steps must also be representable and clearly distinguished from the ten scientific snapshots. `<state_id>` follows `base_<model>`, `<family>_<model>_<level>`, or `rl_<model>_<step>`. Main evaluation and repeatability seed records are distinct, and seed 43/44 diagnostics are not new scientific model states. `<record_kind>` is `raw`, `capability`, `legibility`, or `audits`. `<request_kind>` is one of the five precreated cache directories. Record filenames must use their stable identities and preserve the user's naming rule; their encoding is a delegated convention that must be frozen before use. These templates are also displayed in `spec/mermaid/repository/runtime_rules.mmd`.

Do not use directory names as the sole cache key. Publish completion markers only after write/flush/integrity checks. Use one writer per job namespace and shared immutable sources. Store runtime roots on verified durable storage, with backups as required, not merely wherever a directory called `workspace` happens to exist. See [E.1.3](../spec/experiment.pdf#page=50) and [E.2.3–E.2.6](../spec/experiment.pdf#page=55).

## Changes after this freeze

A future necessary **organizational** change must update the canonical manifest, matching diagrams, module docs, ownership and source map together before dependent integration. No change may silently revise a fixed scientific setting. Ordinary implementation conventions are logged prospectively in `decisions.md` under H.9. Conflicts without an explicit source precedence rule remain documented blockers or delegated technical issues; they are not resolved by whichever document the agent reads last.
