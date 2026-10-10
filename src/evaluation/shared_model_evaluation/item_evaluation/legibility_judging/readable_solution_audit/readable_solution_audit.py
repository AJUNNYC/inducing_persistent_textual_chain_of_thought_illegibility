"""Independent H.5 question-aware tri-state audit, separate from RL Pass B."""
from __future__ import annotations
from pathlib import Path

AUDIT_VERSION = "eval-readable-solution-v1"
AUDIT_MODEL = "gpt-4o-2024-11-20"
AUDIT_SETTINGS = {"temperature": 0, "max_tokens": 64}
AUDIT_SCHEMA = {"type": "object", "properties": {
    "readable_solution": {"type": "string", "enum": ["present", "not_detected", "uncertain"]},
    "padding": {"type": "string", "enum": ["present", "not_detected", "uncertain"]}},
    "required": ["readable_solution", "padding"], "additionalProperties": False}
PROMPT_PATH = Path(__file__).resolve().parents[6] / "prompts" / "evaluation_readable_solution.txt"


def audit_readable_solution(question: str, reasoning: str, judge_client) -> dict:
    if not isinstance(question, str) or not question or not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError("Audit requires a public question and the whole nonempty reasoning trace")
    prompt = PROMPT_PATH.read_text().rstrip("\n")
    record = judge_client.request("evaluation_audit", AUDIT_MODEL, prompt, reasoning,
        question=question, settings=AUDIT_SETTINGS, schema=AUDIT_SCHEMA)
    response = record.get("parsed") if record.get("status") == "success" else None
    if (not isinstance(response, dict) or set(response) != {"readable_solution", "padding"}
        or any(response[name] not in {"present", "not_detected", "uncertain"} for name in response)):
        return {"status": record.get("status") if record.get("status") != "success" else "schema_error",
                "readable_solution": None, "padding": None, "resolved": False,
                "prompt_version": AUDIT_VERSION, "request": record}
    return {"status": "success", **response, "resolved": response["readable_solution"] != "uncertain",
            "prompt_version": AUDIT_VERSION, "request": record}
