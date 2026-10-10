# delegation.md

## Final responsibility hierarchy

```text
coordinator
├── shared_infrastructure
├── data_preparation
├── sft
├── rl
├── evaluation
└── analysis
```

These exact lowercase role names are shared by [repository diagrams](../spec/mermaid/repository/delegation.mmd), the [repository manifest](../spec/repository_manifest.json), module documents and [agent briefs](agents/coordinator.md). The hierarchy is an explicit ownership organization responding to the user's request, not a claim the PDF required a particular number of simultaneously running coding agents. H.9 explicitly permits the coordinator to choose staffing. A role may be implemented by one subagent or further delegated, but every file still has one owner and a reviewed handoff.

The three GPU queues are **not** this role hierarchy. They execute twenty ROT13 fits, twenty KaziText fits, and two sequential RL trajectories using shared implementation. Likewise the 62 evaluated model states and repeated item calls are runtime instances, not distinct codebases. Source: the Mermaid readme's natural coding division, the repeated Child/Contract links, and [H.9](../spec/experiment.pdf#page=80).

## Responsibilities and separation

| Owner | Scope | Key dependences |
|---|---|---|
| `coordinator` | Top-level first-stage orchestration, whole-repository consistency, integration and final acceptance; shared-path changes; no unilateral scientific revision | All owners; PDF precedence and launch gates |
| `shared_infrastructure` | Configuration, typed record contracts, stable IDs/seeds, storage, checkpointing, provider transport/cache, common phase-aware parsing, costs, execution state, bounded recovery and preflight coordination | Consumers jointly define interfaces; real provider/model/storage tests |
| `data_preparation` | Authorized LMSYS candidates, paired structural clean collection, ROT13/KaziText construction, RL question manifests and capability/legibility manifests | Frozen shared records, parsers/identity, source revisions, benchmark overlap data |
| `sft` | Reusable SFT fit and two independent family queues with exact recipe and fixed exposure | Verified shared datasets, model/runtime, masks, state and checkpoint services |
| `rl` | Setup/rollout/reward/update/trajectory stack; exact boolean Sonnet reward and H.7 semantics | Frozen 800-question order, local parsing, provider/cache, full-state recovery |
| `evaluation` | Evaluation protocol, reused condition/model/item runners, native generations, capability scoring, MBPP sandbox, two scores and independent tri-state audit | Fixed panels/interfaces, raw records, common transport/parsers/state and prospective analysis contract |
| `analysis` | Prospective plan, baseline repeatability, simulation validation, design-aware simultaneous bounds, full outcome reporting and construction criteria | Evaluation runner/records, source sampling metadata, main baseline and all paired observations |

Ownership is by the manifest's exact files, not simply the most obvious top-level folder. In particular, `analysis` owns the prospective analysis and simultaneous inference modules nested under evaluation-data preparation **and** results-analysis modules under evaluation. `evaluation` owns evaluation-protocol preparation inside the data hierarchy. `data_preparation` does not gain permission to edit those files merely because their paths start with the same parent directory. Package initializers and root integration files are coordinator-owned. Shared transport does not own scientific reward or audit semantics.

## Build order and interface gates

**First, source and interface freeze.** The coordinator reads the full controlling PDF, complete procedural Mermaid hierarchy and this guide, then confirms each required workflow/contract has an owner. Shared infrastructure defines record/service interfaces with consumers and records permitted technical conventions in `decisions.md`. Source diagrams are requirements, not tests claiming runtime functionality.

**Second, independent implementation with declared fixtures.** Owners can implement against agreed typed fixtures/mocks in parallel where inputs are stable. Data preparation resolves real sources and manifests. Evaluation and analysis can implement known scoring, missingness, criteria and statistical fixtures before trained outcomes exist. SFT and RL implement against the same contracts and selected runtime semantics. No fixture output is passed off as a real generated corpus, provider call or observed result.

**Third, verified handoff and integration.** Integrate each owner's tested code and schemas; check upstream identities, source counts and downstream failure propagation. Reuse common SFT/evaluation/recovery routines rather than creating copies per model or branch. Coordinator controls integration-test files and resolves interface changes with both producers and consumers. Phase-boundary partial failures remain explicit.

**Fourth, required real preflights.** Source-defined GPU/API/storage/sandbox/baseline/statistical/authorization gates occur before the affected paid launches and before trained outcome inspection where required. The overview's data → training → evaluation arrow does not supersede these earlier gates. Baseline repeatability requires the native evaluator before scientific training; the statistical plan requires prospectively frozen simulations and acceptance criteria. Failure blocks launch, not unilateral changes in datasets, thresholds or rewards.

**Fifth, completion report.** Report separately what was implemented, which tests actually ran, which external integrations passed, which preflights remain blocked, and whether any scientific runs were authorized/completed. Do not claim success from blank files, old PDF scaffold tests or simulated outcomes.

## Subdelegation rule

Before assigning a child subagent, provide its exact read set, exclusive write set, source sections, inputs, outputs, dependence identities, completion tests and stop conditions. A child may request a shared-interface change through its owner but does not independently edit another owner's files. An owner integrates and verifies the child's work; delegation does not transfer authority to alter the scientific design. This is repository organization implementing the source's shared-contract decomposition, not an added experimental treatment.

All owners read `docs/overview.md`, `docs/architecture.md`, `docs/contracts.md`, `docs/decisions.md`, `docs/validation.md` and their module cards. Relevant exact judge text is in `docs/prompts.md`. Original source pages remain the final reference for scientific meaning. The complete write lists are in `docs/agents/` and the machine-readable manifest.
