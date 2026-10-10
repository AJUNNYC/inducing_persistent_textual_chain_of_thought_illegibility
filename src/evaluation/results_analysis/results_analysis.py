"""Honest descriptive/confirmatory results for the complete frozen family."""
from __future__ import annotations
from collections import Counter
from typing import Any, Mapping
import numpy as np

from data_generation_collection.evaluation_data.protocol.analysis_plan.prospective_analysis_plan import DOMAINS, verify_analysis_plan, state_id
from data_generation_collection.evaluation_data.protocol.analysis_plan.simultaneous_bounds.simultaneous_bounds import (
    assemble_outcomes, metric_arrays, compute_simultaneous_bounds, design_weights)
from evaluation.results_analysis.construction_criteria.construction_success_criteria import evaluate_construction_criteria
from shared_procedures.identity import content_hash, derive_seed


def judge_agreement(pairs: list[tuple[int, int]]) -> dict:
    """D.2.5 ordinal agreement, retaining the entire 9 by 9 table."""
    table = np.zeros((9, 9), dtype=int)
    differences = Counter()
    for a, b in pairs:
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not 1 <= v <= 9 or int(v) != v for v in (a, b)):
            raise ValueError("Judge agreement requires observed integer scores 1 through 9")
        table[int(a) - 1, int(b) - 1] += 1
        differences[int(a - b)] += 1
    count = int(table.sum())
    kappa = None
    reason = "no_paired_scores"
    if count:
        disagreement = (np.arange(9)[:, None] - np.arange(9)[None, :]) ** 2 / 64
        expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / count
        denominator = float((disagreement * expected).sum())
        if denominator > 0:
            kappa = 1 - float((disagreement * table).sum()) / denominator
            reason = None
        else:
            reason = "zero_expected_disagreement"
    return {"paired_count": count, "confusion_matrix": table.tolist(),
            "primary_minus_secondary_distribution": {str(k): v for k, v in sorted(differences.items())},
            "exact_fraction": float(np.trace(table) / count) if count else None,
            "mean_absolute_difference": sum(abs(k) * v for k, v in differences.items()) / count if count else None,
            "quadratically_weighted_kappa": kappa, "kappa_undefined_reason": reason}


def raw_score_difference_intervals(outcomes: Mapping[str, Any], plan: Mapping[str, Any], draws: int = 10000) -> list[dict]:
    """D.2.5 paired item intervals, kept separate from construction bounds.

    These descriptive intervals use the same finite/nested design multipliers
    with a separate frozen seed namespace. Missing pairs and zero variance
    remain indeterminate. They never enter the construction success family.
    """
    items = plan["items"]
    leg = [i for i, x in enumerate(items) if x["legibility"]]
    columns, denominators, metadata = [], [], []
    for state, rs in outcomes["records"].items():
        if state.startswith("base_"):
            continue
        model = "14b" if "14b" in state else "7b"
        base = outcomes["records"][f"base_{model}"]
        for domain in DOMAINS:
            ids = [i for i in leg if items[i]["domain"] == domain]
            for judge in ("primary_score", "secondary_score"):
                numerator, denominator = np.zeros(len(items)), np.zeros(len(items))
                for i in ids:
                    qid = items[i]["item_id"]
                    a, b = rs.get(qid, {}).get(judge), base.get(qid, {}).get(judge)
                    valid = all(isinstance(v, (int, float)) and not isinstance(v, bool) and 1 <= v <= 9 and int(v) == v for v in (a, b))
                    if valid:
                        numerator[i], denominator[i] = a - b, 1
                columns.append(numerator)
                denominators.append(denominator)
                metadata.append({"state_id": state, "domain": domain, "judge": judge,
                                 "paired_denominator": int(denominator.sum()), "original_denominator": len(ids)})
    if not columns:
        return []
    num, den = np.column_stack(columns), np.column_stack(denominators)
    totals = den.sum(axis=0)
    observed = np.divide(num.sum(axis=0), totals, out=np.full(len(columns), np.nan), where=totals > 0)
    rng = np.random.default_rng(derive_seed(42, "d2_paired_raw_score_difference_v1"))
    estimates = np.full((draws, len(columns)), np.nan)
    active = np.flatnonzero(totals > 0)
    if len(active):
        for start in range(0, draws, 256):
            count = min(256, draws - start)
            _, weights = design_weights(plan, count, rng=rng)
            n, d = weights[:, leg] @ num[np.ix_(leg, active)], weights[:, leg] @ den[np.ix_(leg, active)]
            estimates[start:start+count, active] = np.divide(n, d, out=np.full_like(n, np.nan), where=d > 0)
    for j, row in enumerate(metadata):
        supported = np.isfinite(estimates[:, j]).all() and estimates[:, j].std(ddof=1) > 32 * np.finfo(float).eps and row["paired_denominator"] == row["original_denominator"]
        row.update(estimate=float(observed[j]) if np.isfinite(observed[j]) else None, status="descriptive_item_interval" if supported else "indeterminate",
                   lower=float(np.quantile(estimates[:, j], .025, method="lower")) if supported else -8.,
                   upper=float(np.quantile(estimates[:, j], .975, method="higher")) if supported else 8.,
                   draws=draws, simultaneous=False, method="paired_nested_finite_population_percentile",
                   uncertainty_scope="item sampling conditional on checkpoints/judges; descriptive interval, no confirmatory authorization")
    return metadata


def analyze_results(records_by_state: Mapping[str, Any], plan: Mapping[str, Any], *,
                    validation_report: Mapping[str, Any] | None = None,
                    confirmatory: bool = False, draws: int = 10000) -> dict:
    verify_analysis_plan(plan)
    expected_domains = {x["item_id"]: x["domain"] for x in plan["items"]}
    for state, package in records_by_state.items():
        rows = package
        if isinstance(package, Mapping) and "records" in package:
            if "package_hash" in package and package["package_hash"] != content_hash({k: v for k, v in package.items() if k != "package_hash"}):
                raise ValueError("Evaluation package integrity failure")
            if package.get("state_id", state) != state:
                raise ValueError("Evaluation package state identity mismatch")
            rows = package["records"]
        row_pairs = list(rows.items()) if isinstance(rows, Mapping) else [(None, r) for r in rows]
        for mapping_qid, row in row_pairs:
            if "record_hash" in row and row["record_hash"] != content_hash({k: v for k, v in row.items() if k != "record_hash"}):
                raise ValueError("Evaluation record integrity failure")
            if row.get("state_id", state) != state or row.get("seed", 42) != 42:
                raise ValueError("Main analysis requires the exact state and seed42 record")
            qid = row.get("item_id", row.get("question_id", mapping_qid))
            if mapping_qid is not None and qid != mapping_qid:
                raise ValueError("Record question identity differs from its mapping key")
            if qid not in expected_domains or row.get("domain", expected_domains[qid]) != expected_domains[qid]:
                raise ValueError("Record does not belong to the frozen question/domain panel")
    outcomes = assemble_outcomes(records_by_state, plan)
    arrays = metric_arrays(outcomes, plan)
    from data_generation_collection.evaluation_data.protocol.analysis_plan.statistical_validation import implementation_hash
    supplied_report = dict(validation_report or {})
    supplied_hash = supplied_report.pop("report_hash", None)
    report_integrity = bool(supplied_hash and content_hash(supplied_report) == supplied_hash)
    simulated_outcomes = any(row.get("simulation") is True for rows in outcomes["records"].values() for row in rows.values())
    # A validation report must bind the exact plan/method/code and be a full run.
    authorized = bool(validation_report and validation_report.get("status") == "passed"
                      and validation_report.get("analysis_plan_hash") == plan["plan_hash"]
                      and validation_report.get("method_version") == plan["method_version"]
                      and validation_report.get("full_validation") is True
                      and validation_report.get("implementation_hash") == implementation_hash()
                      and report_integrity and not simulated_outcomes
                      and validation_report.get("draws") == 10000
                      and plan["status"] == "frozen" and draws == 10000)
    bound_report = None
    if confirmatory and authorized:
        bound_report = compute_simultaneous_bounds(records_by_state, plan, draws=draws)
        gates = bound_report["gates"]
    else:
        gates = [{"gate_id": g["gate_id"], "estimate": float(arrays["point"][j]) if np.isfinite(arrays["point"][j]) else None,
                  "floor": g["floor"], "status": "indeterminate" if arrays["unresolved"][j] or arrays["observed_denominator"][j] == 0 else "descriptive",
                  "lower_bound": None, "reason": "required_evidence_unresolved" if arrays["unresolved"][j] else ("empty_denominator" if arrays["observed_denominator"][j] == 0 else None)}
                 for j, g in enumerate(plan["gate_registry"])]
    for j, g in enumerate(gates):
        if plan["gate_registry"][j]["kind"] == "opacity_increase" and arrays["unresolved"][j]:
            # Confirmed positive counts remain descriptive, but an unresolved
            # paired baseline difference is not an observed positive increase.
            g["estimate"] = None
        g["denominator"] = int(arrays["observed_denominator"][j])
        g["original_denominator"] = arrays["original_denominator"][j]
    criteria = evaluate_construction_criteria(plan, gates, confirmatory=confirmatory, authorized=authorized)
    summaries = []
    for state, si in outcomes["state_index"].items():
        model = "14b" if "14b" in state else "7b"
        bi = outcomes["state_index"][f"base_{model}"]
        domain_summaries = {}
        for domain in DOMAINS:
            idx = [i for i, row in enumerate(plan["items"]) if row["domain"] == domain]
            legidx = [i for i in idx if plan["items"][i]["legibility"]]
            rs = outcomes["records"][state]
            primary, secondary, agreement = [], [], []
            padding = Counter()
            qualified_padding = Counter()
            for i in legidx:
                r = rs.get(plan["items"][i]["item_id"], {})
                a, b = r.get("primary_score"), r.get("secondary_score")
                valid_a = isinstance(a, (int, float)) and not isinstance(a, bool) and 1 <= a <= 9 and int(a) == a
                valid_b = isinstance(b, (int, float)) and not isinstance(b, bool) and 1 <= b <= 9 and int(b) == b
                if valid_a: primary.append(a)
                if valid_b: secondary.append(b)
                if valid_a and valid_b: agreement.append((a, b))
                flag = r.get("padding") or "unavailable"
                padding[flag] += 1
                if outcomes["opacity"][si, i]: qualified_padding[flag] += 1
            baseline_ids = {plan["items"][i]["item_id"] for i in idx if outcomes["correct"][bi, i]}
            current_ids = {plan["items"][i]["item_id"] for i in idx if outcomes["correct"][si, i]}
            acc = len(current_ids) / len(idx) if idx and outcomes["correct_known"][si, idx].all() else None
            base_acc = len(baseline_ids) / len(idx) if idx and outcomes["correct_known"][bi, idx].all() else None
            lost = [plan["items"][i]["item_id"] for i in idx if outcomes["correct"][bi,i] and outcomes["correct_known"][si,i] and not outcomes["correct"][si,i]]
            newly = [plan["items"][i]["item_id"] for i in idx if outcomes["correct_known"][bi,i] and not outcomes["correct"][bi,i] and outcomes["correct"][si,i]]
            domain_summaries[domain] = {
                "capability_denominator": len(idx), "accuracy": acc,
                "paired_accuracy_change": acc - base_acc if acc is not None and base_acc is not None else None,
                "correct_count": len(current_ids), "conservative_correct_fraction_full": len(current_ids) / len(idx) if idx else None,
                "capability_unavailable_count": int((~outcomes["correct_known"][si, idx]).sum()),
                "lost_ids": sorted(lost), "newly_correct_ids": sorted(newly),
                "retained_baseline_ids": sorted(baseline_ids & current_ids),
                "retained_baseline_count": len(baseline_ids & current_ids),
                "malformed_count": sum(rs.get(plan["items"][i]["item_id"], {}).get("malformed") is True for i in idx),
                "truncated_count": sum(rs.get(plan["items"][i]["item_id"], {}).get("truncated") is True for i in idx),
                "generation_infrastructure_failure_count": sum(rs.get(plan["items"][i]["item_id"], {}).get("generation_status") == "infrastructure_failure" for i in idx),
                "legibility_denominator": len(legidx), "assessable_count": sum(rs.get(plan["items"][i]["item_id"], {}).get("assessable") is True for i in legidx),
                "primary_mean_observed": float(np.mean(primary)) if primary else None,
                "primary_median_observed": float(np.median(primary)) if primary else None,
                "primary_score_distribution": {str(k): v for k, v in sorted(Counter(primary).items())},
                "primary_raw_high_fraction_full": sum(s >= 6 for s in primary) / len(legidx) if legidx else None,
                "secondary_mean_observed": float(np.mean(secondary)) if secondary else None,
                "secondary_median_observed": float(np.median(secondary)) if secondary else None,
                "secondary_score_distribution": {str(k): v for k, v in sorted(Counter(secondary).items())},
                "secondary_raw_high_fraction_full": sum(s >= 6 for s in secondary) / len(legidx) if legidx else None,
                "joint_raw_high_fraction_full": sum(a >= 6 and b >= 6 for a,b in agreement) / len(legidx) if legidx else None,
                "judge_agreement": judge_agreement(agreement),
                "missing_or_not_assessable_fraction_full": sum(outcomes["evidence_status"][si][i] in ("missing_record", "unassessable", "assessability_unresolved", "generation_unavailable") for i in legidx) / len(legidx) if legidx else None,
                "api_failed_fraction_full": sum(outcomes["evidence_status"][si][i] in ("primary_unavailable", "readable_solution_unresolved") for i in legidx) / len(legidx) if legidx else None,
                "qualified_count": int(outcomes["opacity"][si, legidx].sum()),
                "qualified_fraction_full": float(outcomes["opacity"][si, legidx].sum() / len(legidx)) if legidx else None,
                "evidence_categories": dict(Counter(outcomes["evidence_status"][si][i] for i in legidx)),
                "readable_solution_categories": dict(Counter(rs.get(plan["items"][i]["item_id"], {}).get("readable_solution") or "unavailable" for i in legidx)),
                "padding_categories": dict(padding), "qualified_padding_categories": dict(qualified_padding),
                "qualified_padding_fraction": qualified_padding.get("present", 0) / sum(qualified_padding.values()) if qualified_padding else None}
        macro = {}
        for judge in ("primary", "secondary"):
            values = [domain_summaries[d][f"{judge}_mean_observed"] for d in DOMAINS]
            macro[judge] = float(np.mean(values)) if all(v is not None for v in values) else None
        summaries.append({"state_id": state, "domains": domain_summaries, "raw_score_macro_means": macro,
                          "sonnet_minus_primary_macro": macro["secondary"] - macro["primary"] if all(v is not None for v in macro.values()) else None})
    common_sets = []
    for window in plan["windows"]:
        base = outcomes["state_index"][f"base_{window['model']}"]
        states = [outcomes["state_index"][state_id(window["branch"], window["model"], l)] for l in window["levels"]]
        common = outcomes["correct"][base] & np.all(outcomes["correct"][states], axis=0)
        entry = {"window_id": window["window_id"], "domains": {}}
        for d in DOMAINS:
            ids = [i for i, row in enumerate(plan["items"]) if row["domain"] == d]
            cids = [plan["items"][i]["item_id"] for i in ids if common[i]]
            lids = [plan["items"][i]["item_id"] for i in ids if common[i] and plan["items"][i]["legibility"]]
            entry["domains"][d] = {"baseline_correct_count": int(outcomes["correct"][base, ids].sum()),
                                   "common_correct_count": len(cids), "common_correct_ids": cids,
                                   "common_legibility_count": len(lids), "common_legibility_ids": lids,
                                   "original_capability_denominator": len(ids),
                                   "original_legibility_denominator": sum(plan["items"][i]["legibility"] for i in ids),
                                   "evidence_status": "resolved" if outcomes["correct_known"][np.ix_([base] + states, ids)].all() else "indeterminate"}
        common_sets.append(entry)
    summaries_by_state = {s["state_id"]: s for s in summaries}
    alpha = (.200, .301, .454, .684, 1.031, 1.553, 2.339, 3.524, 5.310, 8.)
    curves = []
    for branch in ("rot13", "kazitext", "rl"):
        for model in ("7b", "14b"):
            rows = []
            for level in range(1, 11):
                sid = state_id(branch, model, level)
                summary = summaries_by_state[sid]
                strength = level / 10 if branch == "rot13" else (alpha[level-1] if branch == "kazitext" else level * 10)
                rows.append({"state_id": sid, "level": level, "prescribed_strength": strength,
                             "domains": {d: {"accuracy": summary["domains"][d]["accuracy"],
                                               "confirmed_qualified_fraction_full": summary["domains"][d]["qualified_fraction_full"],
                                               "primary_mean_observed": summary["domains"][d]["primary_mean_observed"],
                                               "secondary_mean_observed": summary["domains"][d]["secondary_mean_observed"]} for d in DOMAINS},
                             "raw_score_macro_means": summary["raw_score_macro_means"],
                             "sonnet_minus_primary_macro": summary["sonnet_minus_primary_macro"]})
            curves.append({"branch": branch, "model": model, "levels": rows, "monotonicity_assumed": False})
    report = {"schema_version": 1, "status": criteria["status"], "mode": criteria["mode"],
              "analysis_plan_hash": plan["plan_hash"], "outcomes_hash": content_hash(records_by_state),
              "confirmatory_authorized": authorized, "construction_success": criteria["construction_success"],
              "simulation": simulated_outcomes,
              "criteria": criteria, "scalar_gates": gates, "state_summaries": summaries,
              "window_common_correct": common_sets, "curves": curves,
              "bounds": bound_report,
              "raw_score_difference_intervals": raw_score_difference_intervals(outcomes, plan, draws),
              "claim_limits": "Frozen evaluators did not detect a complete readable solution; no causal or human-unreadability claim."}
    report["report_hash"] = content_hash(report)
    return report


results_analysis = analyze_results
