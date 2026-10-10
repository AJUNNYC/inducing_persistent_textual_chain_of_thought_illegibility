# decisions.md

## This is not another scientific protocol

Fixed scientific choices below are copied from the supplied PDF and diagrams. Architectural filenames and role assignments implement the current repository request. Permitted implementation conventions are resolved below and tested prospectively, with unresolved live prerequisites retained. H.9 authorizes the implementing agent to select, test, document and freeze them before trained outcomes. Runtime credentials, account permissions and actual spending authorization cannot be fabricated. Executed implementation evidence is distinguished from blocked live verification throughout the repository.

## Source precedence and already resolved conflicts

| Topic | Controlling source and adopted interpretation |
|---|---|
| Overall authority | H.1, p. 68: H controls its explicit amendments; G retains construction criteria except H operationalizes the approved audit; remaining fixed PDF requirements stay in force |
| Model identity | H.1: construction uses the two DeepSeek-R1-Distill-Qwen models; historical Qwen3 probe compatibility is downstream, not permission to change models |
| “Correct clean reasoning” | H.3, p. 71: clean is untransformed output; structural acceptance only, with at least three eligible words per model trace; no semantic correctness/quality filter |
| Validation allocation | E.2.2, p. 54 and H.3: 10,000 train plus 100 separate validation IDs, superseding older passages that call validation size unresolved |
| Missing old KaziText profile | H.2, pp. 69–70: specified official development profile/revision/hash replaces the unavailable pilot profile |
| Inconsistent alpha grid | H.2.1/p. 27 literal table controls: 0.200, 0.301, 0.454, 0.684, 1.031, 1.553, 2.339, 3.524, 5.310, 8.000 |
| Causal usefulness and padding | G plus H.6: usefulness is not a first-stage gate; no detected readable solution is required; padding alone is diagnostic |
| Two versus three evaluation calls | H.5–H.6: two unchanged trace-only scores plus separate tri-state question-aware GPT-4o audit, maximum 37,200 evaluation requests before retries |
| Training versus evaluation audit | C.3.3–C.3.4 and H.6.3: boolean Sonnet reward semantics, readable-solution cap and padding bonus suppression remain unchanged; tri-state audit is evaluation only |
| GRPO trainer defaults | H.7, p. 77 explicitly fixes Dr. GRPO/group scaling and related semantics; remaining compatible implementation details still need locking |
| Evaluation cap and MBPP limits | H.9/H.4 explicitly retain diagram-supplied 3072-token cap and MBPP 1 CPU/1 GiB/5 CPU sec/10 wall sec limits |
| Retry policy | H.9 retains three retries per operation and 100 per worker, including SDK retries, with persistent counters |
| Confirmatory statistics | G.10 plus H.8: full 216-window/scalar family; specified maximum-standardized bootstrap subject to mandatory prospectively fixed simulation validation |
| Historical code status | E.2 describes a prior scaffold and old CPU checks. Those are source history, not tests that have been run on this new blank repository |

Sources: [H.1–H.9](../spec/experiment.pdf#page=68), [G](../spec/experiment.pdf#page=65), [E.2](../spec/experiment.pdf#page=53). The original PDF pages are copied unchanged, not silently edited to make earlier wording disappear.

## Source details requiring explicit handling, not silent correction

**Spending-warning conflict.** E.1.5, p. 52 states warnings at 75% and 90% of the approved envelope. E.2.5, p. 56 states an initial $300 envelope with warnings at $240 and scheduling stop at $285; it also names a $75 local reward-service allowance. H does not explicitly resolve all these warning values. Retain both source statements in the ledger and require the implementer to document a source-consistent conservative operational policy under actual authorization, without claiming the PDF contains a single unambiguous warning schedule. None of these planning figures authorizes unlimited spending or proves the matrix affordable. Missing/conflicting authorization blocks paid scheduling; construction of the guard can still proceed.

**Final-answer display.** C.3.2, p. 32 visibly prints `Final answer:` and `boxed{NUMBER}` while describing a final-answer line containing a boxed numeric expression. The PDF does not provide a separately executable literal template. Freeze and test exact formatter/parser serialization under H.9; do not silently present a guessed backslash convention as verbatim source text.

**MMLU eligibility and inference.** Main/source diagrams specify seven clean-labeled questions per subject. Historical design-stage calculations use the 5700 reannotated pool; they are not a license to pretend post-eligibility subject pool sizes are still 100. Record actual eligible populations and maintain the fixed quota. H.8.2 requires design-aware inference under the actual within-subject and nested selection scheme. Do not claim the source's planning margins are measured achieved precision.

**Runtime API/library behavior.** The PDF fixes provider/model/setting strings and cites a compatibility reference. This documentation reproduces them without rechecking public websites or substituting newer names. The actual environment must verify availability and semantics before paid launch. Incompatibility is an integration blocker, not permission to change the judge, generation rule or reward.

## Delegated convention register

These entries identify what the implementing agent must resolve before outcomes. They are not new user questions and do not justify inventing scientific freedoms.

| ID | Frozen convention | Implementation/evidence | Status |
|---|---|---|---|
| convention_001 | Pinned HF commits in config; authorized public sources verified; LMSYS remains gated | configuration.py; candidate_manifest.py; operations/assets.py | VERIFIED public metadata; BLOCKED LMSYS |
| convention_002 | Python3.12, hashed CPU/GPU locks, CUDA12.8.1 image, explicit trainer semantics | requirements_cpu.lock; requirements.lock; dockerfile | VERIFIED CPU install; GPU runtime BLOCKED |
| convention_003 | Official English-EWT UD2.5 binary SHA784bd0fa…; actual Aspell dependency closure/hashes/settings | kazitext_data_construction.py; config.yaml | VERIFIED native auxiliary loading; deployment dictionaries must match |
| convention_004 | NFKC prompt normalization, stable source content IDs/order, first-user extraction, exact overlap/token caps | candidate_manifest.py; test_candidate_manifest.py | VERIFIED fixture contracts; live gated corpus BLOCKED |
| convention_005 | ASCII eligible spans, region remainders assigned left-to-right, half-up quotas, SHA seed42 tie-breaking | rot13_data_construction.py; test_rot13_data_construction.py | VERIFIED all ten nested masks |
| convention_006 | Immutable math/numeric spans and lexical boundaries, separate seeded official worker PYTHONHASHSEED0 | kazitext_data_construction.py; test_kazitext_data_construction.py | Native replay/protection evidence recorded; structural rejection preserved |
| convention_007 | Strict think/final separation, Decimal numeric canonicalization, literal choice labels, no answer repair | response_parsing.py; capability_scoring.py | VERIFIED parser/scorer adversarial fixtures |
| convention_008 | Literal prescribed rubrics; canonical JSON payload hashes; native chat wrapper preserves entire assistant target | prompts/; render_sft_completion; output_generation.py | VERIFIED official tokenizer controls; live provider compatibility BLOCKED |
| convention_009 | Per-example supervised token mean; microbatch1; native_training_template_v1; right padding; no packing; eager attention; nonreentrant checkpointing; seeded two-epoch exposure | sft_fit.py; config.yaml; test_sft_fit.py | VERIFIED real tiny PEFT CPU gradients and exact-state resume; full GPU fit BLOCKED |
| convention_010 | Explicit AdamW constant schedule, BF16/NF4, sample std correction1, epsilon1e-4, maxnorm1, microbatch1/accum32, all completed tokens and fixed2048 loss denominator | grpo_policy_update.py; rl_setup.py; config.yaml | VERIFIED numeric/update/reward/resume fixtures; actual model GPU preflight BLOCKED |
| convention_011 | Atomic manifests, immutable SHA identities, SQLite artifact index, locks, full optimizer/RNG checkpoints and scoped artifact hooks | storage.py; checkpointing.py; operations/artifacts.py | VERIFIED local crashes/corruption/reuse; Drive lifetime durability BLOCKED |
| convention_012 | Pinned Python3.11 slim sandbox, trusted root supervisor/unprivileged worker, original tests retained, fixed resource caps | mbpp_sandbox.py; test_mbpp_sandbox.py | VERIFIED actual Docker isolation; original reference cap audit may block |
| convention_013 | Paired design-aware Rao-Wu rescaling with actual finite populations/nested selection, float64 higher quantile, conservative missing/zero-variance handling, all10800 gates retained | simultaneous_bounds.py; test_simultaneous_bounds.py | Implemented; mandatory actual-frame coverage validation determines readiness |
| convention_014 | Seven finite-population scenarios,100 repetitions each,10000 draws, seed42 namespaces, one-sided95%Wilson coverage lower bound>=.95 and zero structural errors; exact prospective futility stop | statistical_validation.py; frozen simulation plan | Frozen before trained outcomes; passing certificate unavailable until executed |
| convention_015 | Canonical sorted JSON SHA256, uint32 derived seeds, typed records, lowercase namespaces, scientific projection plus full operational snapshot | identity.py; records.py; configuration.py | VERIFIED producer/consumer and source invalidation fixtures |
| convention_016 | Actual authorization/rates/quotas required; warning75/80/90%; scheduling stop95%; independent Pod billing controller | cost_accounting.py; judge_client.py; operations/runpod.py | VERIFIED local guard tests; live authority/evidence BLOCKED |

For each resolution, record the chosen value/algorithm, source constraint, rationale, code/config revision, fixture evidence, freeze time relative to trained outcome inspection and consequences for dependent caches. “Tested” must name the evidence; do not turn an empty table into a claim of a passing preflight.

## What may not be delegated away

Do not change models, datasets, accepted counts, panel IDs/counts, training epochs/steps, intervention levels, reward weights or gates, construction thresholds, fixed judge prompts/models, or add adaptive scientific runs. No method switch after viewing trained outcomes when statistical validation fails. Fix implementation defects within the design, version the repair and invalidate affected derivatives, including baselines. Unresolvable mandatory preflight failures stop the affected launch. Sources: [H.9](../spec/experiment.pdf#page=80), [G.9–G.11](../spec/experiment.pdf#page=67).

## Freeze and interpretation

All choices were implemented before any trained experimental outcomes existed. Scientific configuration hashes retain code/dependency/model/data/prompt/parser/trainer/sandbox identities. Full operational snapshots separately retain paths, funding and quota evidence. Production runtime requires the actual committed source SHA and lock hash; an unversioned development source cannot authorize paid work. An implementation repair versions affected evidence and derivatives. Historical or wrong-population simulation evidence cannot certify the actual frozen panel. Failed mandatory coverage or reference-executor checks remain launch blockers.
