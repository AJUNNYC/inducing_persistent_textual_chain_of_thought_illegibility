"""The frozen G/H construction gates, with one fixed transfer pair per window."""
from __future__ import annotations
from typing import Any, Mapping


def _conjunction(statuses: list[str]) -> str:
    if "failed" in statuses:
        return "failed"
    return "indeterminate" if "indeterminate" in statuses else "passed"


def evaluate_construction_criteria(plan: Mapping[str, Any], gates: list[Mapping[str, Any]],
                                   *, confirmatory: bool = False, authorized: bool = False) -> dict:
    indexed = {g["gate_id"]: g for g in gates}
    if len(indexed) != len(gates):
        raise ValueError("Duplicate scalar gate IDs")
    def check(gid):
        g = indexed.get(gid)
        if g is None or g.get("estimate") is None or g.get("status") == "indeterminate":
            return "indeterminate"
        if g["estimate"] < g["floor"]:
            return "failed"
        if confirmatory:
            if not authorized or g.get("status") != "supported" or g.get("lower_bound") is None:
                return "indeterminate"
            if g["lower_bound"] < g["floor"]:
                return "indeterminate"
        return "passed"
    results = []
    for window in plan["windows"]:
        required = _conjunction([check(x) for x in window["gate_ids"]["mandatory"]])
        domain_status = {d: _conjunction([check(x) for x in ids]) for d, ids in window["gate_ids"]["transfer"].items()}
        pairs = [{"domains": pair, "status": _conjunction([domain_status[d] for d in pair])}
                 for pair in window["transfer_pairs"]]
        transfer = "passed" if any(x["status"] == "passed" for x in pairs) else ("indeterminate" if any(x["status"] == "indeterminate" for x in pairs) else "failed")
        status = _conjunction([required, transfer])
        if confirmatory and not authorized and status == "passed":
            status = "indeterminate"
        results.append({"window_id": window["window_id"], "branch": window["branch"],
                        "model": window["model"], "levels": window["levels"], "status": status,
                        "mandatory_status": required, "transfer_domains": domain_status,
                        "transfer_pairs": pairs,
                        "qualifying_pairs": [x["domains"] for x in pairs if x["status"] == "passed"],
                        "gate_ids": window["gate_ids"]})
    qualified = [w["window_id"] for w in results if w["status"] == "passed"]
    overall = "passed" if qualified else ("indeterminate" if any(w["status"] == "indeterminate" for w in results) else "failed")
    return {"status": overall, "mode": "confirmatory" if confirmatory else "descriptive",
            "construction_success": bool(confirmatory and authorized and qualified),
            "descriptive_candidates": qualified if not confirmatory else [],
            "qualifying_windows": qualified if confirmatory and authorized else [],
            "window_count": len(results), "windows": results}


construction_success_criteria = evaluate_construction_criteria
