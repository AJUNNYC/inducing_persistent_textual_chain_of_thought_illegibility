"""Frozen C.3 reward; malformed/wrong responses never use a remote judge."""
from typing import Any, Mapping
from shared_procedures.identity import content_hash
from shared_procedures.response_parsing import normalize_numeric_answer, parse_rl_output
from .sonnet_judgment.sonnet_opacity_judgment import judge_opacity


def compute_rl_reward(rollout: Mapping[str, Any], reference_answer: str, *, judge: Any) -> dict[str, Any]:
    reference = normalize_numeric_answer(reference_answer)
    if reference is None:
        raise ValueError("Frozen RL reference must be a valid numeric answer")
    parsed = parse_rl_output(rollout["raw_output"], truncated=bool(rollout.get("truncated", False)))
    result = {"completion_id": rollout["completion_id"], "question_id": rollout["question_id"],
              "policy_hash": rollout["policy_hash"], "status": "verified", "parser": parsed,
              "reference_answer": reference, "raw_output_hash": content_hash(rollout["raw_output"]), "judge": None}
    if parsed["status"] != "valid":
        result.update(correctness="invalid", reward=-1.0)
    elif parsed["numeric_answer"] != reference:
        result.update(correctness="wrong", reward=0.0)
    else:
        judgment = judge_opacity(rollout["question"], parsed["reasoning"], client=judge)
        result.update(correctness="correct", judge=judgment,
                      reward=1.0 + 0.5 * judgment["opacity"] * judgment["padding_gate"])
    return result

rl_reward = compute_rl_reward
