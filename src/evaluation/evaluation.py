"""Top-level 62-state native evaluation, with separate modular state entrypoints."""
from __future__ import annotations

from data_generation_collection.evaluation_data.protocol.evaluation_protocol import validate_protocol
from evaluation.condition_groups.condition_evaluation import evaluate_conditions, expected_condition_ids


def run_evaluation(states: list[dict], protocol: dict, generator, judge_client=None, *, sandbox=None,
                   cache_root=None, analysis=None, confirmatory: bool = False) -> dict:
    validate_protocol(protocol)
    expected = {"base_7b", "base_14b"} | expected_condition_ids("sft") | expected_condition_ids("rl")
    ids = [state["state_id"] for state in states]
    if len(ids) != 62 or set(ids) != expected:
        raise ValueError("Main evaluation requires all 62 exact unique scientific states")
    baselines = [state for state in states if state["state_id"].startswith("base_")]
    grouped = {}
    for group, family in [("sft", {"rot13", "kazitext"}), ("rl", {"rl"})]:
        conditions = [state for state in states if state.get("family") in family]
        grouped[group] = evaluate_conditions(group, conditions, baselines, protocol, generator,
            judge_client, sandbox=sandbox, cache_root=cache_root)
        if grouped[group]["status"] != "complete":
            return {"status": "incomplete", "groups": grouped, "analysis": None,
                    "reason": "required_evaluation_group_incomplete", "protocol_hash": protocol["protocol_hash"]}
    packages = {**grouped["sft"]["baselines"], **grouped["sft"]["conditions"], **grouped["rl"]["conditions"]}
    analysis_result = None
    if analysis is not None:
        analysis_result = analysis({identifier: package["records"] for identifier,package in packages.items()},
            protocol["analysis_plan"], validation_report=protocol["preflight_reports"].get("statistical_validation", {}).get("evidence", {}).get("validation_report"),
            confirmatory=confirmatory)
    return {"status": "complete", "protocol_hash": protocol["protocol_hash"], "groups": grouped,
            "packages": packages, "analysis": analysis_result,
            "scientific_evidence_status": "indeterminate" if any(package["scientific_evidence_status"] == "indeterminate" for package in packages.values()) else "recorded"}
