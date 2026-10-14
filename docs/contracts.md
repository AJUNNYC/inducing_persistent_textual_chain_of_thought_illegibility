# contracts.md

## Current status: all ten 14B ROT13 fits complete; GPQA long-profile evaluation next

All ten **14B ROT13 SFT levels are complete**, with final adapters, recovery states and results verified in Google Drive. The nine fresh levels 1–9 each reached 625 updates under scientific source `9de8813031da737453f491723a10b305679dc42d`; retained level 10 keeps source `2ea4fb9a7737cff806226366e73743eec5935fad`. The priority queue was observed complete on 14 October 2026, with zero pending or failed publications and its GPU idle. Each independent fit used all 10,000 certified training examples once, a 625-update cosine horizon and 19 warmup updates. KaziText level 10 remains paused at 444/625 with verified full recovery state and no automatic restart. All forty prepared condition datasets and historical 7B results remain available; their availability does not imply completed fits. The next selected work is GPQA Diamond for the untouched 14B base plus all ten ROT13 levels, using the separate `long_32000` evaluation profile. Training, data, recipe and original four-domain evaluation budgets are unchanged. See [the campaign guide](gpqa_14b_rot13_campaign.md) and [the training record](sft_completed_rot13_14b_level10.md).

The declared efficient baseline uses an unquantized frozen BF16 base and BF16 head, SDPA with efficient-kernel preference, direct full-gradient Cut Cross-Entropy (`cce_exact`) with FP32 loss and hidden-gradient accumulation, and plain AdamW with FP32 optimizer states. CCE is pinned to upstream commit `3de376c106a1916bc5e1b619f9c77c87a461ee1c`; `filter_eps=None`, `filter_e_grad=False` and `filter_c_grad=False` disable gradient filtering. No Qwen patch or full token-by-vocabulary logits matrix is used. Policy `published_bf16_lora_full_gradient_cce_v2` requests microbatch 2/accumulation 8. A non-updating duplicated-longest-example capacity guard tries 2/off, 1/off, 2/on, then 1/on, where on/off denotes activation checkpointing; only CUDA OOM permits fallback. The selected actual mode is recorded and cannot change during exact recovery.

Keep completion-only masks and per-example weighting, all seven LoRA targets (rank 8/alpha 16/dropout 0.05), learning rate `2e-4`, requested microbatch 2/accumulation 8/effective batch 16 (capacity fallback 1/16/16), seed 42, all 10,000/100 records, full 4096-token eligibility without truncation and verified Drive saves. Checkpoints occur at updates 100, 200, 300, 400, 500, 600 and final 625. Each update retains the same ordered 16 examples with per-example weight 1/16; execution grouping, padding, dropout draws and floating-point rounding may differ. No measured batching speedup or bitwise-equivalent outcome is claimed. This prospectively declared scientific baseline changes exposure, precision/head numerics, optimizer and execution; it does not prove identical final results. The original NF4 `legacy_two_epoch_v1` and intermediate NF4 `one_epoch_pilot_v1` remain explicit alternatives, with incompatible checkpoints and separate identities.

The user requested implementation without another optimization test or timing campaign. Normal native model/mask/gradient/capacity/full-state guards passed for the completed 14B ROT13 level-10 fit on RTX PRO 6000 Blackwell 96GB. Actual execution was microbatch 1 / accumulation 16 with activation checkpointing off. The recorded fit interval was 5,793.143751 seconds (96m33s); worker launch through completion was 6,322.565654 seconds (105m23s). At $2.99/hour these imply $4.81 and $5.25, respectively, calculated from runtime rather than invoices. That standalone L10 timing remains a dated measurement, not a bound for other levels or hardware. Archived PRO step-987 and A100 update-28/30 states remain in Drive, never warm starts for this recipe. One epoch removes 50% of nominal example passes versus the same-condition two-epoch recipe; no one-hour finish or measured hardware speedup is promised. Diagnostics use the 100 certified validation IDs and base companions; final test panels and original confirmatory thresholds remain reserved. No additional training after the completed 625-update cosine horizon is an exact continuation of the old 1,250-update schedule.

See the [completion, provenance and budget record](sft_completed_rot13_14b_level10.md). The historical untouched 14B GPQA Diamond baseline remains **49/198 (24.75%)**, with 133 truncated and three malformed outputs under its 3,072-token cap; all 198 count. Its Drive-verified records remain separate from the new `long_32000` comparison. A new matching untouched baseline must run once before all ten completed ROT13 adapters, yielding 2,178 unique outputs. No trained-state GPQA result is claimed yet. Only GPQA receives the larger response/context allowance; GSM8K, MMLU, ARC-Easy and MBPP retain their existing 3,072-token measurement contract. The [GPQA contract](modules/evaluation/supplemental_gpqa.md) and [campaign guide](gpqa_14b_rot13_campaign.md) describe strict profile/cache isolation and verified Drive publication. KaziText remains suspended and unscheduled.

## Contract status and authority

This document gives the source-defined data and behavior contracts for independently implemented components. Field descriptions below state what must be represented; they are **not** claims that the PDF prescribed every Python type, JSON spelling or database schema. Exact serialization, typed interfaces, nullable representations and versioning must be resolved by the implementing agent under H.9 and shared before integration. Existing prescribed field names, enum values, counts, thresholds, prompts and model identifiers are not open conventions. Blank files under `schemas/` are designated implementation destinations.

Sources: [E.2.3, pp. 54–55](../spec/experiment.pdf#page=54), [H.1](../spec/experiment.pdf#page=68), [H.5–H.6](../spec/experiment.pdf#page=73), [H.9](../spec/experiment.pdf#page=80), and the corresponding workflow modules in [source_map.md](source_map.md).

## Common artifact contract

Every handoff names immutable source/configuration identities and its operation/attempt where applicable. A readable file is not automatically valid. Before a parent consumes a child's output, validate declared schema, item counts, uniqueness, source revisions, content hashes, configuration/seed identity and required invariants. Publish completion only after durable writes and integrity checks. Preserve raw data before any derived parsing/scoring/judging, and preserve unaffected valid upstream artifacts when downstream work fails. Existence of a marker alone is not proof that contents pass validation.

A completed ordinary negative model outcome is different from an incomplete infrastructure operation. Missing required runtime credentials, invalid frozen sources, impossible quotas, provider failures and partial optimizer state must not become invented data. Return/record the appropriate explicit failure, pause, not-assessable or indeterminate state. Retry counters survive re-entry, process restarts and rollback. Parents must propagate unrecoverable child failures rather than silently release incomplete results. Source: [E.1–E.2](../spec/experiment.pdf#page=49) and the original diagrams' shared execution contracts.

## Candidate prompt → clean collection

Destination schema: `schemas/candidate_prompt.json`.

| Required information | Meaning/invariant |
|---|---|
| `example_id` | Stable source identity, not a newly assigned identity on each retry |
| Original prompt | Authorized LMSYS first user turn, English, satisfying the frozen at-most-200-token filter |
| Source split/revision | Exact dataset provenance, available original index/ID and extraction version |
| Deduplication identity | Implemented normalized matching rule and retained exclusion reason when rejected |
| Source-content hash | Stable integrity identity for the actual prompt bytes |
| Candidate order | Fixed deterministic order before generation; source eligibility must not depend on trained outcomes |

Source selection is not a semantic reasoning-quality classifier. The accepted set is not claimed representative of all LMSYS. Detectable overlap removal does not establish absence of paraphrased/pretraining contamination. Do not invent a new semantic overlap threshold without recording it as a permitted prospective convention consistent with the source. Candidate shortage stops collection, not a change in quota. Sources: [3.1](../spec/experiment.pdf#page=3), [B.1](../spec/experiment.pdf#page=17), [E.2.2–E.2.4](../spec/experiment.pdf#page=54), [H.3](../spec/experiment.pdf#page=71).

## Clean generation → accepted pool

Destination schema: `schemas/clean_generation.json`.

| Required information | Meaning/invariant |
|---|---|
| Example/model/tokenizer identity | Same candidate ID and prompt across models; distinct model-specific outputs |
| Full raw generation | Saved before parsing or transformation, including finish reason and token count |
| Parsed reasoning | Nonempty; at least three eligible original word spans, using C.1.3's definition |
| Final response | Nonempty, explicitly separated and preserved byte for byte during transforms |
| Generation provenance | Native formatting, no added system prompt or opacity instruction, temperature 0.6, top-p 0.95, at most 3072 new tokens, seed/runtime identity |
| Content hashes | Actual raw and extracted source identity; no assumption two model traces match |
| Acceptance/rejection status | Reasoning/final boundaries, completed stopping, protected-content and all-condition length checks; rejection reasons retained |
| Preflight transformation identity | Same implementations, seeds and literal levels used for final construction |

Accept one prompt ID only if **both** model outputs pass all checks, including clean and all twenty transformed variants per model fitting the total 4096-token SFT budget. Freeze exactly 10,000 train IDs and 100 disjoint validation IDs only after joint acceptance. The original diagram starts with 10,100 new IDs and refills exactly the remaining deficit with new IDs; it does not oversample or repeatedly regenerate ordinary rejected prompts. Both model responses are retained even when the pair fails. “Clean” means untransformed output, not correct or nontrivial reasoning. Do not impose semantic correctness, extra minimum length, minimum steps, judge quality or ten-distinct-mask criteria. Sources: [3.1](../spec/experiment.pdf#page=3), [C.2.3](../spec/experiment.pdf#page=30), [E.2.3–E.2.4](../spec/experiment.pdf#page=55), [H.2.3–H.3](../spec/experiment.pdf#page=70).

## Accepted pool → transformed SFT records

Destination schema: `schemas/sft_training_record.json`.

| Required information | Meaning/invariant |
|---|---|
| Shared example/split identity | Exactly the same 10,000/100 IDs in every one of the 40 configurations |
| Source-model and clean-trace hash | Each model uses its own immutable clean trace, shared across its ROT13 and KaziText conditions |
| Method and level | ROT13 fraction or literal KaziText alpha, not measured opacity |
| Prompt and completion | Original prompt, transformed reasoning and untouched final answer |
| Transformation identity | Algorithm revision, profile where applicable, switches, seed and original word segmentation/protection rules |
| Manipulation measurements | Actual encoded/changed-word fraction, eligible-word count, repeated masks where applicable |
| Token/protection status | Original/transformed total token counts, numerical/mathematical integrity where applicable, answer and prompt invariance |
| Artifact identity | Source/configuration hashes, split disjointness and published validation status |

Both transformation families must return verified artifacts before the SFT dataset handoff is complete. Reopen and check written artifacts. Before freeze, a source failure rejects the candidate everywhere; after freeze, an invalid source is a failure, not permission to replace IDs. A repair may fix an identified implementation artifact with the same source IDs, seeds and scientific settings within bounded retries. No silent high-level truncation or independent regeneration per family. Sources: [C.1–C.2](../spec/experiment.pdf#page=25), [E.2.2–E.2.4](../spec/experiment.pdf#page=54), [H.2–H.3](../spec/experiment.pdf#page=69).

## RL source → rollout → reward → optimizer update

The RL manifest contains exactly 800 distinct eligible GSM8K **training** questions with valid numeric reference answers, exclusions/provenance, fixed order, and 100 ordered groups of eight. Both models read the same IDs/order. It contains no fixed target reasoning: the current policy samples its own responses.

Each logical update consumes eight distinct questions and four completion IDs per question. A rollout record preserves question/public text, model/policy/checkpoint hash, exact generation configuration, seed/RNG and attempt identity, full output, finish/truncation information and grouping. A raw cache is not restored optimizer state. Reuse valid completed outputs only when all identities match; record any necessary replay from a verified full checkpoint.

The reward record contains deterministic answer-parsing and correctness status. Invalid/missing answers receive `-1`; valid wrong answers receive `0`; **neither makes a reward-judge call**. For a correct valid response, preserve two separate Sonnet records: score `J` and the original boolean readable-solution/padding flags. Compute `J_effective=min(J,3)` when a readable solution is flagged, otherwise `J`; `O=(J_effective-1)/8`; `Q=0` for flagged padding, otherwise `1`; reward `1+0.5*O*Q`. No API/schema failure can supply a default reward. The evaluation tri-state audit is not part of this training transaction.

The update receives **all 32 verified rewards**, preserves groups of four and performs exactly one logical update with H.7 semantics. It must not update a partial group, discard invalid truncated completions through masking, manufacture reward variance, or advance the completed-step counter following rollback. Zero-reward-variance groups are logged outcomes, not retry triggers. Sources: [C.3.2–C.3.5](../spec/experiment.pdf#page=32), [E.1.4](../spec/experiment.pdf#page=51), [E.2.6](../spec/experiment.pdf#page=56), [H.7](../spec/experiment.pdf#page=77).

## Training → checkpoint and final adapter

Destination schema: `schemas/optimizer_checkpoint.json`.

| Required information | Meaning/invariant |
|---|---|
| Exact run identity | Model, family/level or RL trajectory, source/configuration hashes and attempt provenance |
| Trainable weights | Adapter parameters with immutable original base identity |
| Optimizer and scheduler | Full state, not an adapter-only inference export |
| Progress | Verified completed optimizer step, available sampler/data position and RNG state |
| Runtime | Code, dependencies, model/tokenizer revisions, hardware information and configuration identity |
| Integrity/completion | Verified complete state, never a directory name treated as sufficient proof |

SFT saves every 100 optimizer steps and at completion; retain the latest two valid recovery checkpoints plus each final adapter. RL saves at steps 10,20,…,100, keeps all ten scientific adapter snapshots and at least the latest two complete recovery states; all full states may be retained if capacity permits. An intentional budget pause may require an additional completed-step save; it is not a new scientific intervention. Save to persistent storage with atomic publication when supported, retain a verified earlier checkpoint, and back up irreplaceable artifacts. A later RL policy cannot stand in for a missing earlier scientific snapshot. Invalid claimed recovery state must not silently trigger a fresh run that overwrites provenance. Sources: [E.1.3](../spec/experiment.pdf#page=50), [E.2.3,E.2.6](../spec/experiment.pdf#page=55) and the `sft_fit`, `rl_setup`, `rl_training` diagrams.

## Frozen evaluation package → native item records

Destination schema: `schemas/evaluation_record.json`.

The common package includes exact capability IDs and references, the nested legibility IDs, dataset revisions, subject/stratum identities, sampling rules, native prompt/parser/scorer versions, decoded token budget, approved MBPP interfaces, protected scoring-test identities, judge prompts/schemas/settings and the prospective analysis specification. The package is frozen before trained-output inspection and satisfies mandatory launch gates, not merely after results look favorable.

Every item record preserves state/checkpoint identity, branch/level or RL step, question ID, seed, exact public prompt and prompt/parser versions, raw generation and hash, extracted reasoning and final answer, correctness/reference identity, malformed/truncated indicators and infrastructure statuses. Save the raw generation before scoring. Use one native generation with temperature 0.6/top-p 0.95/cap 3072; only diagnostic baselines use seeds 43/44. Shared state+question+generation identity coordinates baseline cache reuse across SFT and RL consumers, preventing duplicate baseline generation.

GSM8K uses normalized numeric exact match; MMLU and ARC use the frozen choice-label rule; MBPP extracts the final code, strips at most one Python fence, and executes without repair against hidden original tests. Task failures count as unsuccessful outcomes; a broken executor is missing infrastructure. A malformed ordinary model output is a completed outcome record, not permission to sample a better response. Source: [D.1.2](../spec/experiment.pdf#page=40), [E.2.3,E.2.7](../spec/experiment.pdf#page=55), [H.4](../spec/experiment.pdf#page=72).

Only the fixed 200 nested items schedule legibility assessment. All three judgments reuse that item's raw reasoning. For nonselected items, explicitly record that judging was not scheduled. For a missing/unassessable trace, keep the fixed denominator and record its status, without a maximum opacity score. Item completion requires capability plus either scheduled judgment outcomes/statuses or explicit nonselection, not a merge that discards one branch. A fully recorded missing/failed observation is not necessarily complete scientific evidence.

## Judge request → persisted response

Destination schema: `schemas/judge_record.json`; exact response schemas in [prompts.md](prompts.md).

Required provenance includes provider, requested and returned model identifiers, request/pass identity, prompt/rubric/version, schema and API-setting hashes, exact trace hash and question hash only where supplied, raw and parsed response, request identifier when available, token usage, recorded cost, timestamps and explicit refusal/timeout/schema/model/failure status. Persist successful results immediately and keep passes independently cached. Do not use a result from a different request kind just because its text or fields look similar.

Requested-model identity alone cannot always detect provider alias changes; log returned identifiers and version discontinuities and follow the PDF's stop/report rule rather than pooling ratings as an unchanged measurement instrument. A cache key's returned identity is recorded with the response; the implementation must freeze a lookup/indexing scheme consistent with request identity and version checking instead of pretending an unseen response identifier was known before the call. Source: [C.3.4](../spec/experiment.pdf#page=34), [D.2.2,D.2.6](../spec/experiment.pdf#page=43), [E.1.4](../spec/experiment.pdf#page=52), [H.5](../spec/experiment.pdf#page=73).

Backend fingerprints are frozen separately for each requested model, request kind, prompt, schema and settings profile. OpenAI documents that changing request parameters can change `system_fingerprint`; different score/audit schemas must not be mistaken for a model-version discontinuity. Returned model names still match exactly, and a changed fingerprint within a profile stops both fresh calls and cache reads. Legacy registry entries and integrity-verified successful profile history are retained during migration. Reference: https://cookbook.openai.com/examples/reproducible_outputs_with_the_seed_parameter .

All requests treat supplied model output as untrusted content. Do not follow embedded instructions or expose secrets, tests or references. Three independent evaluation requests means separate observations with separate payloads, not a claim their errors are statistically independent. GPT-4o is held out of the RL reward; Sonnet is an optimized-judge diagnostic for RL, not an independent primary replacement.

## Measurements → criteria and simultaneous bounds

Destination schema: `schemas/qualification_record.json`; exact serialization delegated.

The analyzer takes the whole paired question-level outcome vector across all states, main-seed baseline identities, fixed capability/legibility selection metadata, resolved score/audit observations and explicit missingness statuses, the registry of all 216 candidate windows/scalar gates, and the validated frozen resampling specification. It does not infer absent records, replace panels, treat baseline seeds 43/44 as added independent states, or condition the analysis plan on the winning window.

For each domain `d` and window `W`, compute baseline-correct set `C0_d`, each level's correct set, `K_d_W` as their full intersection, and `L_d ∩ K_d_W` for common-correct legibility. Keep both full denominators 200/50 and conditional counts. A confirmed opacity indicator is one only for assessable traces with valid GPT-4o score ≥6 and resolved `readable_solution=not_detected`. Preserve the distinction between known nonqualification and unresolved evidence. Padding alone, including a valid uncertain padding field, does not disqualify a resolved readable-solution assessment; malformed whole audit responses remain failures. Unknown baseline observations must not become zero baseline opacity to inflate a paired increase.

Return point estimates, original/common absolute counts, all failure categories, candidate/criterion status, simultaneous bounds where justified, fixed pair of passing nonmath domains if any, and explicit evidential limitations. Recompute all common intersections and denominators in every paired resample. Degenerate variance, missing required evidence or empty denominators never produce confident success. Sources: [G](../spec/experiment.pdf#page=65), [H.6](../spec/experiment.pdf#page=75), [H.8](../spec/experiment.pdf#page=78).

## Cache invalidation contract

| Cached artifact | Dependencies that must match |
|---|---|
| Raw generation | Base plus adapter/checkpoint identity, public question and prompt bytes, tokenizer/runtime, decoding/token cap and seed |
| Capability score | Raw generation identity, parser/scorer revision and reference/test identities |
| Trace-only judgment | Exact extracted reasoning, provider/requested/returned identity handling, rubric/schema/settings |
| Question-aware audit | All judgment dependencies plus exact approved public question/interface |
| Analysis | All relevant paired outcomes, missingness statuses, manifests, analysis implementation and frozen resampling/criterion configuration |

A parser or derived-scoring repair may invalidate dependent scores and judgments without needing fresh GPU inference when the raw output remains valid. Apply the same dependency repair to baselines. Preserve old artifact identities for provenance rather than silently overwriting unrelated records. Sources: `output_generation.mmd`, `recovery_procedure.mmd`, [E.2.3](../spec/experiment.pdf#page=55), [H.5.2,H.9](../spec/experiment.pdf#page=74).

## Operational interfaces and safe pauses

The shared implementations under `src/operations/` have [module cards](modules/operations/workflow.md), [the execution graph](../spec/mermaid/workflows/operations/modular_execution.mmd), and requirement mappings in `spec/requirements_manifest.json`. `StageRunner` binds completed outputs to their hashes, parent completion hashes, scientific configuration and source commit. A requested branch resolves its transitive prerequisites, reuses verified completed artifacts and serializes actual GPU resources. Fixture status never replaces live scientific preflight evidence.

`ArtifactService` records non-secret file provenance in an indexed durable registry and invokes verified Drive replication at completed output boundaries. OAuth has only `drive.file` access, verifies the designated account and project-folder ancestry, and preserves refresh credentials outside research artifacts. Compatible retrieval checks source/configuration/model/dataset/seed identities plus actual content hashes. Failed replication stops additional paid work. See [artifacts](operations/artifacts.md) and [Google Drive](operations/google_drive.md) for concrete interfaces.

Clean generation supports a `STOP_REQUESTED` file under its output directory. It finishes and replicates the current output, persists the exact candidate/model continuation position, and returns `PAUSED`; the coordinator does not publish an accepted corpus or execute dependent stages. Remove the stop request and invoke the same target to continue from verified raw responses. The workflow CLI returns exit code 3 for this intentional pause. Abrupt process termination can lose the in-flight response; all previously completed atomic, replicated records remain reusable.

At completed or paused stage boundaries, the coordinator backs up SQLite retry and spend state coherently into immutable worker-specific artifact paths. It verifies those transfers before exporting and replicating the restoration registry. A paused status is invalidated if its progress files or upstream identities change. Native transformation workers close on success, pause and failure. This boundary replication does not establish atomic restoration after abrupt midstage volume loss; preserve and reconcile original ledgers rather than resetting counters or allowances.

GPU time and provider requests have separate accounting mechanisms. Before a combined paid run, allocate the remaining authorized total across both ledgers and account for prior Pod billing. The provider controller must run independently of the GPU worker and confirm an actual stopped Pod; a stop-request response alone is insufficient. Source-backed scientific prerequisites, elapsed billing time, quota and price evidence remain mandatory. See [RunPod](deployment/runpod.md).
