# decisions.md

## This is not another scientific protocol

Fixed scientific choices below are copied from the supplied PDF and diagrams. Architectural filenames and role assignments implement the current repository request. Remaining permitted implementation conventions have **not** been filled with invented defaults. H.9 authorizes the implementing agent to select, test, document and freeze them before trained outcomes. Runtime credentials, account permissions and actual spending authorization cannot be fabricated. The repository itself contains blank executable/configuration files, so none of the unresolved implementation entries should be mistaken for completed work.

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

| ID | Convention to freeze | Constraint/source | Initial status |
|---|---|---|---|
| convention_001 | Model/tokenizer and dataset revisions; exact authorized extraction access | E.2.3,E.2.8; H.9; construction model identities fixed | not_resolved |
| convention_002 | Compatible Python/package/CUDA/TRL/quantization lock and hardware runtime | E.1.2,E.2.8; H.7–H.9; no unrecorded defaults affecting treatment | not_resolved |
| convention_003 | UDPipe binary checksum, binding and actual English Aspell dictionary provenance | H.2.3; exact model/profile already selected | not_resolved |
| convention_004 | Candidate normalization, deterministic source order, duplicate/overlap audit and tokenizer used for input cap | 3.1,B.1,E.2.2–E.2.4; no new semantic trace-quality filter | not_resolved |
| convention_005 | Original word boundary segmentation, region remainder convention, stable hash seed and expansion tie-breaking | C.1.3,H.9; three regions, half-up counts, contiguous nested masks fixed | not_resolved |
| convention_006 | Math/numeric recognition/protection mechanics and stochastic ordering/hash-seed behavior | C.1.4,H.2.3,H.9; same candidate/final transforms and fixed switches | not_resolved |
| convention_007 | Native trace/final parsing, numeric normalization and choice-label extraction | C.3.2,D.1.2,H.4,H.9; no inferred or repaired answers | not_resolved |
| convention_008 | Literal generation templates, payload serialization and prompt version hashes | C.3.2,D.1–D.2,H.4–H.5,H.9; prescribed judge instructions unchanged | not_resolved |
| convention_009 | Completion-only SFT loss reduction and compatible trainer details | C.2.3,H.9; all fixed SFT settings, masks and exposure preserved | not_resolved |
| convention_010 | Remaining RL optimizer/scheduler/precision/adapter/stabilization/microbatch details | H.7; actual eight-by-four one-update semantics and other fixed settings preserved | not_resolved |
| convention_011 | Durable storage, locking, atomic publication, cache lookup/indexing and checkpoint recovery implementation | E.1–E.2; exact identities, counters and safe boundary semantics required | not_resolved |
| convention_012 | Restricted MBPP executor and mechanical interface manifest extraction | E.2.7,H.4; exact resource caps and leakage prohibitions | not_resolved |
| convention_013 | Exact design-aware resampling weights, finite-population treatment, numerical quantile and degenerate/missing-metric family handling | H.8; 10000 draws, seed 42, complete scalar family, no zero-width certainty | not_resolved |
| convention_014 | Statistical simulation scenarios, seeds, replication budget and quantitative Monte Carlo acceptance rule | H.8.4; freeze before trained outcomes and obey failed-preflight stop | not_resolved |
| convention_015 | Typed service signatures, record serialization, hash namespaces, runtime filenames | E.2.3,H.9; semantic contracts and finalized static paths preserved | not_resolved |
| convention_016 | Operational configuration under actual account authorization, rates/quotas and the documented warning conflict | E.1.4–E.1.5,E.2.5,E.2.8; spending not inferred from scientific scope | awaiting_runtime_authorization |

For each resolution, record the chosen value/algorithm, source constraint, rationale, code/config revision, fixture evidence, freeze time relative to trained outcome inspection and consequences for dependent caches. “Tested” must name the evidence; do not turn an empty table into a claim of a passing preflight.

## What may not be delegated away

Do not change models, datasets, accepted counts, panel IDs/counts, training epochs/steps, intervention levels, reward weights or gates, construction thresholds, fixed judge prompts/models, or add adaptive scientific runs. No method switch after viewing trained outcomes when statistical validation fails. Fix implementation defects within the design, version the repair and invalidate affected derivatives, including baselines. Unresolvable mandatory preflight failures stop the affected launch. Sources: [H.9](../spec/experiment.pdf#page=80), [G.9–G.11](../spec/experiment.pdf#page=67).
