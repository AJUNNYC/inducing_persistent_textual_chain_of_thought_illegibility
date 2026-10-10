"""C.3.3/C.3.4 two independent, cached, boolean Sonnet training judgments."""
from pathlib import Path
from typing import Any

MODEL = "claude-sonnet-5-5"
PROMPT_ROOT = Path(__file__).resolve().parents[6] / "prompts"
PASS_A_SCHEMA = {"type": "object", "properties": {"score": {"type": "integer", "minimum": 1, "maximum": 9}}, "required": ["score"], "additionalProperties": False}
PASS_B_SCHEMA = {"type": "object", "properties": {"readable_solution_present": {"type": "boolean"}, "obvious_padding_present": {"type": "boolean"}}, "required": ["readable_solution_present", "obvious_padding_present"], "additionalProperties": False}
SETTINGS = {"thinking": {"type": "between_tools"}, "output_config": {"effort": "low"}, "max_tokens": 64}

class RewardInfrastructureError(RuntimeError):
    """A missing required observation: never converted into a numerical reward."""


def validate_pass_a(value: dict[str, Any]) -> int:
    if set(value) != {"score"} or type(value["score"]) is not int or not 1 <= value["score"] <= 9:
        raise RewardInfrastructureError("Pass A requires exactly one integer score in 1..9")
    return value["score"]


def validate_pass_b(value: dict[str, Any]) -> tuple[bool, bool]:
    if set(value) != {"readable_solution_present", "obvious_padding_present"} or any(type(v) is not bool for v in value.values()):
        raise RewardInfrastructureError("Pass B requires exactly two boolean flags")
    return value["readable_solution_present"], value["obvious_padding_present"]


def judge_opacity(question: str, reasoning: str, *, client: Any) -> dict[str, Any]:
    """Pass A sees trace only; B sees public question plus trace. Client commits each."""
    if not question.strip() or not reasoning.strip():
        raise RewardInfrastructureError("Judgment requires a public question and nonempty trace")
    a = client.request("rl_pass_a", MODEL, (PROMPT_ROOT / "rl_pass_a.txt").read_text().strip(), reasoning,
                       settings=SETTINGS, schema=PASS_A_SCHEMA)
    if a.get("status") != "success":
        raise RewardInfrastructureError("Pass A unavailable: " + str(a.get("status")))
    score = validate_pass_a(a["parsed"])
    b = client.request("rl_pass_b", MODEL, (PROMPT_ROOT / "rl_pass_b.txt").read_text().strip(), reasoning,
                       question=question, settings=SETTINGS, schema=PASS_B_SCHEMA)
    if b.get("status") != "success":
        raise RewardInfrastructureError("Pass B unavailable; committed Pass A remains reusable")
    readable, padding = validate_pass_b(b["parsed"])
    effective = min(score, 3) if readable else score
    return {"status": "verified", "score": score, "effective_score": effective,
            "readable_solution_present": readable, "obvious_padding_present": padding,
            "opacity": (effective - 1) / 8, "padding_gate": 0 if padding else 1,
            "pass_a": a, "pass_b": b}

sonnet_opacity_judgment = judge_opacity
