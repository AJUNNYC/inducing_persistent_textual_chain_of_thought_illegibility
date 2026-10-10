"""Real baseline seed diagnostics; ordinary invalid answers remain failures."""
from __future__ import annotations
from typing import Any, Mapping, Sequence

from data_generation_collection.evaluation_data.protocol.analysis_plan.prospective_analysis_plan import DOMAINS, MODELS
from data_generation_collection.evaluation_data.protocol.analysis_plan.simultaneous_bounds.simultaneous_bounds import _state_records
from shared_procedures.identity import content_hash


def baseline_repeatability(passes_by_model: Mapping[str, Any], panel: Sequence[Mapping[str, Any]] | Mapping[str, Any],
                           *, require_full_panel: bool = True) -> dict:
    items = list(panel.get("items", [])) if isinstance(panel, Mapping) else list(panel)
    ids = [str(i.get("item_id", i.get("question_id", ""))) for i in items]
    if len(ids) != len(set(ids)) or any(not x for x in ids):
        raise ValueError("Invalid baseline panel IDs")
    counts = dict(zip(DOMAINS, (300, 399, 331, 203)))
    if require_full_panel and any(sum(i["domain"] == d for i in items) != n for d, n in counts.items()):
        raise ValueError("Repeatability must use the frozen 1233-item panel")
    model_results = []
    live_evidence = True
    for model in MODELS:
        supplied = passes_by_model.get(model, {})
        passes = {seed: _state_records(supplied.get(seed, supplied.get(str(seed), []))) for seed in (42, 43, 44)}
        live_evidence &= all(row.get("simulation") is False and row.get("raw_hash") and row.get("generation_status") == "success"
                             and row.get("seed") == seed for seed, rows in passes.items() for row in rows.values())
        domains = {}
        for d in DOMAINS:
            dids = [qid for qid, item in zip(ids, items) if item["domain"] == d]
            unresolved = {str(seed): [q for q in dids if not isinstance(passes[seed].get(q, {}).get("correct"), bool)] for seed in passes}
            correct = {seed: {q for q in dids if passes[seed].get(q, {}).get("correct") is True} for seed in passes}
            base = correct[42]
            common = set.intersection(*correct.values())
            repeat = {str(s): len(base & correct[s]) / len(base) if base else None for s in (43, 44)}
            common_rate = len(common) / len(base) if base else None
            status = "indeterminate" if not base or any(unresolved.values()) else ("passed" if all(r >= .8 for r in repeat.values()) and common_rate >= .6 else "failed")
            domains[d] = {"status": status, "panel_count": len(dids), "baseline_correct_count": len(base),
                          "correct_counts": {str(s): len(v) for s, v in correct.items()},
                          "repeat_retention": repeat, "common_retention": common_rate,
                          "common_correct_count": len(common), "common_correct_ids": sorted(common),
                          "invalid_or_unavailable_ids": unresolved,
                          "lost_ids": {str(s): sorted(base - correct[s]) for s in (43, 44)},
                          "newly_correct_ids": {str(s): sorted(correct[s] - base) for s in (43, 44)}}
        statuses = [x["status"] for x in domains.values()]
        model_results.append({"model": model, "status": "failed" if "failed" in statuses else ("indeterminate" if "indeterminate" in statuses else "passed"), "domains": domains})
    statuses = [x["status"] for x in model_results]
    status = "failed" if "failed" in statuses else ("indeterminate" if "indeterminate" in statuses else "passed")
    report = {"status": status, "ready_for_training": status == "passed" and require_full_panel and bool(live_evidence),
            "evidence_type": "live_baseline_diagnostics" if require_full_panel and live_evidence else "fixture",
            "evidence": {"live": bool(require_full_panel and live_evidence), "panel_hash": content_hash(items)},
            "models": model_results, "seeds": [42, 43, 44], "main_baseline_seed": 42,
            "retention_measurement": "calibrated_diagnostic" if status == "passed" else "uncalibrated",
            "limitation": "Three passes flag gross instability, not a precise repeatability estimate."}
    report["checks"] = [{"name": "baseline_repeatability", "status": status if report["evidence"]["live"] and status in ("passed", "failed") else "blocked",
                         "detail": "Real untouched model seed diagnostics" if report["evidence"]["live"] else "Actual untouched model generation provenance is absent",
                         "evidence": report["evidence"], "required": True}]
    report["report_hash"] = content_hash(report)
    return report
