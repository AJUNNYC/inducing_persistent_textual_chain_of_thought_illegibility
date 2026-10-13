# validation.md

## Current SFT status: 14B ROT13 level 10 complete; $240 authorized for remaining work

The default `efficient_one_epoch_v1` recipe processes **all 10,000 certified training examples once per selected fit**: 625 optimizer updates, a complete 625-update cosine schedule and 19 warmup updates. The first fresh fit, 14B ROT13 level 10, completed all 625 updates on 12 October 2026; KaziText14B level10 is training; at the13Oct21:31UTC snapshot,checkpoint 200/625 was verified in Drive. Its native guards passed with physical 1 × accumulation 16, activation checkpointing off; no completion is claimed. Both 7B/14B, ROT13/KaziText and all ten levels remain available in the forty-condition Drive-verified inventory. The active scope is now 14B only: ROT13 and KaziText across all ten levels, 20 conditions total. The completed adapter is retained, leaving 19 selected fits uncompleted; the 7B data and historical results remain available but outside the active training queue. On 13 October the user authorized $240 for SFT; the live 19-fit campaign started13Oct20:58:33UTC after the verified GPQA baseline. The scientific worker remains source 9de8813 independently of this documentation commit. See [the training redesign](sft_short_pilot.md).

The declared efficient baseline uses an unquantized frozen BF16 base and BF16 head, SDPA with efficient-kernel preference, direct full-gradient Cut Cross-Entropy (`cce_exact`) with FP32 loss and hidden-gradient accumulation, and plain AdamW with FP32 optimizer states. CCE is pinned to upstream commit `3de376c106a1916bc5e1b619f9c77c87a461ee1c`; `filter_eps=None`, `filter_e_grad=False` and `filter_c_grad=False` disable gradient filtering. No Qwen patch or full token-by-vocabulary logits matrix is used. Policy `published_bf16_lora_full_gradient_cce_v2` requests microbatch 2/accumulation 8. A non-updating duplicated-longest-example capacity guard tries 2/off, 1/off, 2/on, then 1/on, where on/off denotes activation checkpointing; only CUDA OOM permits fallback. The selected actual mode is recorded and cannot change during exact recovery.

Keep completion-only masks and per-example weighting, all seven LoRA targets (rank 8/alpha 16/dropout 0.05), learning rate `2e-4`, requested microbatch 2/accumulation 8/effective batch 16 (capacity fallback 1/16/16), seed 42, all 10,000/100 records, full 4096-token eligibility without truncation and verified Drive saves. Checkpoints occur at updates 100, 200, 300, 400, 500, 600 and final 625. Each update retains the same ordered 16 examples with per-example weight 1/16; execution grouping, padding, dropout draws and floating-point rounding may differ. No measured batching speedup or bitwise-equivalent outcome is claimed. This prospectively declared scientific baseline changes exposure, precision/head numerics, optimizer and execution; it does not prove identical final results. The original NF4 `legacy_two_epoch_v1` and intermediate NF4 `one_epoch_pilot_v1` remain explicit alternatives, with incompatible checkpoints and separate identities.

The user requested implementation without another optimization test or timing campaign. Normal native model/mask/gradient/capacity/full-state guards passed for the completed 14B ROT13 level-10 fit on RTX PRO 6000 Blackwell 96GB. Actual execution was microbatch 1 / accumulation 16 with activation checkpointing off. The recorded fit interval was 5,793.143751 seconds (96m33s); worker launch through completion was 6,322.565654 seconds (105m23s). At $2.99/hour these imply $4.81 and $5.25, respectively, calculated from runtime rather than invoices. Other conditions remain unmeasured. Archived PRO step-987 and A100 update-28/30 states remain in Drive, never warm starts for this recipe. One epoch removes 50% of nominal example passes versus the same-condition two-epoch recipe; no one-hour finish or measured hardware speedup is promised. Diagnostics use the 100 certified validation IDs and base companions; final test panels and original confirmatory thresholds remain reserved. No additional training after the completed 625-update cosine horizon is an exact continuation of the old 1,250-update schedule.

See the [completion, provenance and budget record](sft_completed_rot13_14b_level10.md). GPQA Diamond (198 questions) is appended to the existing capability suite, matching the José et al. dataset variant; the actual untouched14B seed42 baseline completed49/198 (24.75%), with 133 truncated outputs and3 malformed answers under the unchanged3,072-token cap; all 198 count in the denominator. The full result is Drive-verified; trained-state GPQA remains unrun. It is a supplemental stress test and does not change the original four-domain criteria. The authorized execution order is now: freeze and run the untouched14B GPQA Diamond baseline on all198 prepared questions, verify its raw outputs/scores/report in Google Drive, retain the completedROT13 14B L10 adapter, then start freshKaziText 14B L10 followed by ROT13 andKaziText at levels1–9 in ascending level order. The committed baseline-only CLI supports evaluation before a trained adapter exists; the separate 19-fit coordinator requires that verified baseline before SFT. The GPQA barrier passed and KaziText L10 started; the rest of this ordered queue remains pending. See the [actual baseline/training receipt](evidence/sft_14b_campaign_training_active_2026_10_13.json). These are dated observations, not completion claims for all 19 fits.

## Four different kinds of evidence

**Repository integrity** checks whether this package's paths,names,source copies,mappings,links and implemented module contracts are consistent. **Implementation unit/integration tests** check actual code against fixtures and contracts. **Runtime preflights** check real selected models, hardware, APIs, storage and sandbox behavior. **Scientific results** are the completed training and frozen measurements. Passing the first does not establish any of the others. The completed implementation audit is in [validation_report.md](validation_report.md); executed results and blocked live checks are recorded separately.

The PDF explicitly distinguishes old offline checks from missing live integrations and scientific validation. See [E.2.1,E.2.8](../spec/experiment.pdf#page=53) and [H.1,H.9](../spec/experiment.pdf#page=68).

## Unit contracts to implement

Each module has an exact unit-test path in its module document and manifest. Tests must check the contract, not merely import a module or assert a hardcoded total. Fixtures may exercise source-prescribed boundary cases without adding new training runs. Concrete required categories follow.

| Category | Required invariants and negative cases | Source |
|---|---|---|
| Identity/configuration | Stable content/config hashes; exact model/run matching; fixed matrix/counts/settings; invalid or missing frozen inputs rejected | E.2.3–E.2.4; H.9 |
| Candidate/clean pairing | Same prompt ID/text across two models; one ordinary response per model; three eligible words; both traces and all transforms pass; empty/missing/cutoff outputs rejected with retained records; exact deficit refills | 3.1; E.2.2–E.2.4; H.3; clean collection diagrams |
| ROT13 | Three near-equal nonempty regions; stable anchors; half-up region counts; contiguous nested masks; full eligible coverage at level ten; original numerals and final answer unchanged; short repeated masks accepted | C.1.3; H.3 |
| KaziText | Fixed revision/profile checksum/switches/literal grid; same canonical order and transform in acceptance/final construction; protected numerical/math spans, prompt and final answer unchanged; invalid loadable binary/profile failures explicit | C.1.4; H.2 |
| SFT | Completion-only masks exclude prompt/padding; entire assistant target supervised; all seven target projections and fixed settings; named-recipe exact exposure (efficient 1 epoch / 625, legacy 2 epochs / 1250); no warm starts/early stopping; validation cannot add training | C.2; sft_fit diagram |
| Reward | Invalid=-1 and wrong=0 with no API calls; correct gated exactly; original boolean flags; independent cached passes; no API default; all 32 rewards before update | C.3; H.6.3 |
| RL semantics | Eight distinct IDs and exactly four completion IDs per group; one update; fixed order; Dr. GRPO/group scaling/clipping/reference/sampling; invalid truncated completions not masked away | H.7 |
| Evaluation selection | 300/399/331/203 counts; seven per all 57 subjects; correct GSM strata; 200 nested unique IDs and 50/domain; same manifests for all 62 states | D.1.2,D.2.3; H.8.2 |
| Native output/cache | Same raw output for scoring/judging; exact adapter/config/tokenizer/seed identity; ordinary failures not resampled; derivative repairs retain valid raw data and invalidate affected baseline derivatives too | E.2.7; output_generation diagram |
| MBPP | Public interface only; scoring tests/reference bodies never in prompts/audit; final-only extraction; at most one fence removed; no function repair; test failure distinct from broken executor | H.4 |
| Judging/audit | Correct request-specific visible fields; exact four frozen instructions; score 1–9; boolean training flags versus tri-state audit; scores not shared as input; cache/request namespace separation | C.3.3; D.2.4; H.5 |
| Criteria | All 216 windows; fixed same nonmath pair throughout; 80/60 retention of original baseline successes; qualified 50% and +20pp on full and common sets; pooled conditional count, not accidental macro-mean | G; H.6,H.8 |
| Missingness | Missing or unknown primary/baseline evidence not zero favorable opacity; absent reasoning not maximally opaque; full denominators preserved; unknown padding alone does not block resolved audit | D.2; H.6 |
| Recovery | Three retries/operation, 100/worker inclusive SDK counts survive restart/rollback; no retry of ordinary model outcomes; invalid frozen input and quota failures stop; last verified checkpoint rather than assumed OOM save | E.1–E.2; H.9 |

These are required behavior categories derived from source rules. The implementing agent must specify exact test fixtures and independently compute expected outputs; this document does not prescribe a new scientific hyperparameter for any test.

## Integration tests and handoffs

| Exact test entrypoint | Integration requirement |
|---|---|
| `tests/integration/test_data_handoff.py` | Common accepted IDs, both source models, all forty verified train/validation variants and disjoint held-out manifests |
| `tests/integration/test_training_handoff.py` | Worklists enumerate forty independent SFT fits and two separate RL trajectories; correct recipe/source identities and shared fit reuse |
| `tests/integration/test_reward_transaction.py` | Rollout persistence → local correctness → two cached passes for correct answers → all 32 verified rewards → one update; faults cannot partially update |
| `tests/integration/test_evaluation_handoff.py` | Shared bases, one raw item output, strict scoring, three independent scheduled judgments and explicit nonselection/missingness |
| `tests/integration/test_analysis_handoff.py` | Full paired state vectors and correct sampling metadata feed all common sets, window gates and simultaneous statistics |
| `tests/integration/test_recovery_handoff.py` | State/cache/checkpoint integrity across interrupts, persistence and bounded retries |
| `tests/integration/test_end_to_end.py` | Complete source-described first-stage pipeline over declared fixtures, with all externally mocked components identified |

Coordinator integration does not authorize scientific execution. These filenames are organizational assignments; the scientific behavior is specified by E.2 and H.9.

## Mandatory real preflights

The checks below are implemented. They report actual passed/failed/blocked evidence; unavailable live services do not become passing fixtures.

| Exact test entrypoint | Evidence required before affected paid launch |
|---|---|
| `tests/preflight/test_configuration.py` | Fully resolved validated scientific config, source/dependency hashes and no ambiguous unauthorized defaults |
| `tests/preflight/test_model_runtime.py` | Exact two construction models/tokenizers/native delimiters; actual compatible precision/runtime/hardware |
| `tests/preflight/test_clean_data.py` | All accepted IDs and every variant meet the frozen structural/protection/length conditions |
| `tests/preflight/test_loss_mask_gradients.py` | Actual model loss masks and nonzero LoRA gradients, not merely names of trainable parameters |
| `tests/preflight/test_full_checkpoint_resume.py` | Full optimizer/scheduler/RNG/sampler recovery at verified exact-run boundaries, including faults |
| `tests/preflight/test_durable_storage.py` | Actual provider-mounted durability, access/locking and intended stop/terminate survival; verified backup handling |
| `tests/preflight/test_judge_services.py` | Real frozen provider calls/schema/output/version logging, quotas, cache/retry/pause behavior and cost accounting |
| `tests/preflight/test_mbpp_sandbox.py` | Reference-test execution and required isolation/resource limits; no secrets/host-data/network access |
| `tests/preflight/test_baseline_repeatability.py` | Both untouched baselines at 42/43/44, same 1233 IDs, domainwise repeat retention/common overlap; no extra legibility scoring |
| `tests/preflight/test_grpo_update.py` | Controlled actual IDs/group fixture demonstrates eight-by-four and exactly one logical optimizer update with H.7 semantics |
| `tests/preflight/test_statistical_coverage.py` | Prospectively frozen simulation scenarios/budget/seeds and quantitative Monte Carlo joint-coverage acceptance under full design/family |
| `tests/preflight/test_spending_authorization.py` | Actual account permission, current envelope, provider limits/rates and documented safe scheduling policy |

Sources: [minimum gates, p. 6](../spec/experiment.pdf#page=6), [E.2.8](../spec/experiment.pdf#page=58), [G.9–G.10](../spec/experiment.pdf#page=67), [H.4,H.7–H.9](../spec/experiment.pdf#page=72).

## Statistical preflight must not be hand-waved

Freeze the actual without-replacement/nested sampling representation, estimands/weights, paired resampling across all states, quantile convention and degenerate/missing-metric family treatment. Simulations must cover thresholds, correlated checkpoints, uneven domains, small/empty common-correct subsets and extreme proportions. Specify population generation, seeds, replication budget and a quantitative acceptance rule for joint one-sided 95% coverage, including Monte Carlo uncertainty. These unspecified numerical implementation details are delegated under H.8.4, not already validated by the paper's reference to bootstrap methods.

Zero estimated variance cannot justify a zero-width confident interval. A missing required denominator or undefined replicate requires the frozen conservative/indeterminate behavior. Do not omit troublesome gates from multiplicity correction and declare the remaining family successful. Failed mandatory validation blocks confirmatory launch, not merely a final sentence in the results. Descriptive reporting may still describe available measurements, but does not waive any mandatory launch gate. Sources: [H.8](../spec/experiment.pdf#page=78) and the prospective-analysis/simultaneous-bounds diagrams.

## Acceptance and failure reporting

A module is ready for integration only after its actual implementation, contract tests and source traceability are reviewed. An integration is ready for paid launch only after applicable actual preflight checks and authorization pass. An experiment is complete only when required jobs, states, generations, observations and reports really exist; explicit partial results remain partial. A construction claim additionally requires the frozen empirical and statistical criteria.

If a mandatory gate fails, record the failure and preserve completed valid work. Fix identified implementation defects within the scientific design and version the repair. Do not lower thresholds, alter dataset quotas, substitute providers/models, resample an unfavorable result or mark a not-run test as passed. Exact sample IDs, raw generation records and all failures must remain auditable.
