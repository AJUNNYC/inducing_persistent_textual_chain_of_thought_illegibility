"""Common SFT/RL grouping with shared exact baseline item-cache ownership."""
from __future__ import annotations

from evaluation.shared_model_evaluation.model_evaluation import evaluate_model


def expected_condition_ids(group: str) -> set[str]:
    if group == "sft":
        return {f"{family}_{model}_level_{level:02d}" for family in ["rot13", "kazitext"]
                for model in ["7b", "14b"] for level in range(1,11)}
    if group == "rl":
        return {f"rl_{model}_step_{step:03d}" for model in ["7b", "14b"] for step in range(10,101,10)}
    raise ValueError("Group must be sft or rl")


def evaluate_conditions(group: str, states: list[dict], baselines: list[dict], protocol: dict,
                         generator, judge_client=None, *, sandbox=None, cache_root=None,
                         require_full_matrix: bool = True) -> dict:
    expected = expected_condition_ids(group)
    ids = [state["state_id"] for state in states]
    if len(ids) != len(set(ids)) or not set(ids) <= expected or (require_full_matrix and set(ids) != expected):
        raise ValueError("Condition list must preserve the exact frozen matrix and unique state IDs")
    if {state["state_id"] for state in baselines} != {"base_7b", "base_14b"} or len(baselines) != 2:
        raise ValueError("Both exact untouched baselines are required")
    for state in states:
        if group == "rl":
            expected_id = f"rl_{state['model']}_step_{int(state.get('step',-1)):03d}"
            if state.get("family") != "rl" or state["state_id"] != expected_id:
                raise ValueError("RL state labels do not identify its original scientific snapshot")
        else:
            level = state.get("level")
            level_string = f"level_{level:02d}" if isinstance(level, int) else str(level)
            if state.get("family") not in {"rot13", "kazitext"} or state["state_id"] != f"{state['family']}_{state['model']}_{level_string}":
                raise ValueError("SFT state labels mismatch family/model/level")
    base_packages, packages = {}, {}
    options = {"sandbox": sandbox, "cache_root": cache_root}
    for state in baselines:
        package = evaluate_model(state, protocol, generator, judge_client, **options)
        base_packages[state["state_id"]] = package
        if package["status"] != "complete":
            return {"status": "incomplete", "group": group, "baselines": base_packages, "conditions": packages,
                    "missing_conditions": ids, "reason": "baseline_incomplete"}
    for index, state in enumerate(states):
        package = evaluate_model(state, protocol, generator, judge_client, **options)
        packages[state["state_id"]] = package
        if package["status"] != "complete":
            return {"status": "incomplete", "group": group, "baselines": base_packages, "conditions": packages,
                    "missing_conditions": ids[index:], "reason": "condition_incomplete"}
    return {"status": "complete", "group": group, "baselines": base_packages, "conditions": packages,
            "missing_conditions": [], "target_count": len(states), "full_matrix": set(ids) == expected}
