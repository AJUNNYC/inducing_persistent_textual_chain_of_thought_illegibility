# Inducing Persistent Textual Chain-of-Thought Illegibility

Experimental Protocol, Measurement Plan, and Technical Justification

**Documentation and blank implementation scaffold. No experiment code, trained models, generated datasets, live API checks, or statistical validation results are included.** The controlling source is the supplied 81-page updated PDF. The earlier 67-page PDF is preserved for provenance, not as an alternative protocol.

## Start here

Read [docs/overview.md](docs/overview.md) for the scientific design and reading order. The complete [repository hierarchy](docs/repository_layout.md), [architecture](docs/architecture.md), [artifact contracts](docs/contracts.md), [delegation plan](docs/delegation.md), [validation requirements](docs/validation.md), and [decision ledger](docs/decisions.md) form the implementation guide. [docs/implementation_prompt.md](docs/implementation_prompt.md) is the coordinating agent's assignment.

The source documents are [spec/experiment.pdf](spec/experiment.pdf) and the path-synchronized workflows under [spec/mermaid](spec/mermaid/readme.md). [docs/protocol_reference.md](docs/protocol_reference.md) is a complete page-indexed **text extraction**, not a corrected or re-typeset replacement for the PDF. Equations and tables must be checked against the PDF when extraction is ambiguous.

## Specification authority

Appendix H controls the subjects it explicitly resolves. Appendix G controls construction criteria except for H's approved operational amendments. All other fixed PDF requirements remain in force. Mermaid diagrams describe execution flow and add the settings explicitly retained by H.9. Documentation records those requirements without inventing new scientific choices. The exact file layout and ownership assignments are repository organization specified in response to the repository request, not filenames claimed to have been prescribed by the PDF. See [source traceability](docs/source_map.md).

The project covers model-organism construction: data, SFT, RL, native evaluation, construction-criteria analysis, durable state, spending controls and recovery. Downstream activation-monitoring and probe design are **not** implemented by this scaffold. See PDF [H.1, p. 68](spec/experiment.pdf#page=68).

## Naming and one-to-one mapping

All tracked path components are lowercase and contain no hyphens or spaces. Underscores separate words; dots retain file extensions and standard hidden filenames. Existing external identifiers, model names, prompt-version strings, literal source quotations and original source titles are **data**, and must not be rewritten just to satisfy a local filename rule.

Each workflow/support module has the same relative stem under `src/`, `spec/mermaid/workflows/` and `docs/modules/`, with `.py`, `.mmd` and `.md` extensions respectively. Every static file and directory, including tests, documents, configuration placeholders, package initializers and runtime markers, is represented with its exact repository-relative path in the repository diagrams. [spec/repository_manifest.json](spec/repository_manifest.json) binds the path, owner, source and diagram node together.

## Blank files are deliberate

All implementation `.py` files, their tests, command entrypoints, `config.yaml`, `pyproject.toml`, `requirements.lock`, prompt `.txt` files and implementation schema `.json` files are zero-byte placeholders. Required settings, field meanings, exact frozen judge instructions and validation expectations are in the Markdown documentation and original PDF. An empty test suite is not a passing test suite; an empty configuration is not a valid scientific configuration. Only documentation, Mermaid diagrams, provenance/structure metadata, PDF sources and repository housekeeping are populated.

`data/` and `artifacts/` contain only empty directory markers. Runtime traces, tests, references, checkpoint weights, request caches and secrets are not uploaded. The scaffold does not fabricate dataset IDs, hashes of unavailable binaries, credentials, measured results or completed checkpoints.

## Publication and privacy

This scaffold is prepared for the requested owner `AJUNNYC`, with repository name `inducing_persistent_textual_chain_of_thought_illegibility`. No public visibility was specified; do not publish research materials publicly by inference. Use a private repository for the initial publication unless the owner separately authorizes public visibility. The GitHub connection, account permission and remote creation are external operational prerequisites, not scientific parameters.

See [docs/validation_report.md](docs/validation_report.md) for the actual **scaffold checks**, carefully separated from unperformed experiment preflights. The repository's generated path inventories exclude Git's own internal metadata.
