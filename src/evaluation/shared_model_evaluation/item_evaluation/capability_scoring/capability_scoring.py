"""Frozen deterministic domain scorers; infrastructure failures stay missing."""
from __future__ import annotations
import re

from shared_procedures.identity import content_hash
from shared_procedures.response_parsing import normalize_numeric_answer, PARSER_VERSION
from evaluation.shared_model_evaluation.item_evaluation.native_output.output_generation import canonical_domain

SCORER_VERSION = "capability_scorer_v1"


def parse_choice_label(answer: str, labels: list[str]) -> str | None:
    if not isinstance(answer, str):
        return None
    text = answer.strip()
    text = re.sub(r"^Final answer:\s*", "", text)
    match = re.fullmatch(r"(?:\\boxed\{([^{}]+)\}|\(([^()]+)\)|([^\s]+))", text)
    if match is None:
        return None
    label = next(value for value in match.groups() if value is not None)
    return label if label in labels else None


def extract_final_code(answer: str) -> str | None:
    """Remove at most one Python fence, preserving its interior bytes exactly."""
    if not isinstance(answer, str) or not answer.strip():
        return None
    text = answer.strip()
    if text.startswith("```"):
        match = re.fullmatch(r"```(?:python)?\r?\n(.*?)\r?\n```", text, re.DOTALL)
        if not match or "```" in match.group(1):
            return None
        return match.group(1)
    if "```" in answer:
        return None
    return answer


def score_capability(item: dict, parsed: dict, *, sandbox=None, raw_hash: str | None = None) -> dict:
    domain = canonical_domain(item["domain"])
    reference = item.get("reference_answer")
    dependency = {"raw_hash": raw_hash, "parser_version": PARSER_VERSION,
        "scorer_version": SCORER_VERSION, "reference_hash": content_hash(
            {"tests": item.get("scoring_tests"), "setup": item.get("test_setup_code", "")} if domain == "mbpp" else reference)}
    result = {"status": "outcome", "correct": False, "answer": None,
              "reason": "malformed_answer", "dependencies": dependency}
    if parsed.get("status") != "valid" or parsed.get("truncated"):
        result["reason"] = "truncated_output" if parsed.get("truncated") else "malformed_output"
        return result
    answer = parsed.get("final_answer")
    if domain == "gsm8k":
        expected = normalize_numeric_answer(reference)
        if expected is None:
            raise ValueError("Frozen GSM8K reference is not a valid numeric answer")
        numeric_text = re.sub(r"^Final answer:\s*", "", answer.strip())
        boxed = re.fullmatch(r"\\boxed\{([^{}]+)\}", numeric_text)
        extracted = normalize_numeric_answer(boxed.group(1) if boxed else numeric_text)
    elif domain in {"mmlu", "arc"}:
        choices = item.get("choices", [])
        if isinstance(choices, dict):
            labels = choices.get("label", list(choices))
        else:
            labels = [str(choice.get("label", chr(65+i))) if isinstance(choice, dict) else chr(65+i)
                      for i, choice in enumerate(choices)]
        if reference not in labels:
            raise ValueError("Frozen reference choice is not among public labels")
        expected = reference
        extracted = parse_choice_label(answer, labels)
    elif domain == "mbpp":
        code = extract_final_code(answer)
        if code is None:
            return result
        if sandbox is None:
            return {**result, "status": "infrastructure_failure", "correct": None,
                    "reason": "mbpp_executor_unavailable"}
        tests = ([item["test_setup_code"]] if item.get("test_setup_code") else []) + item["scoring_tests"]
        interfaces = item.get("public_interface") or ([item["interface"]] if item.get("interface") else [])
        entrypoints = [interface["name"] for interface in interfaces]
        if not entrypoints:
            raise ValueError("Frozen MBPP item has no approved public entrypoint")
        executed = sandbox.execute(code, tests, entrypoints=entrypoints)
        return {**result, "status": executed["status"], "correct": executed["correct"],
                "reason": executed["reason"], "answer": code, "execution": executed}
    else:
        raise ValueError(f"Unknown evaluation domain {domain}")
    result.update(answer=extracted, correct=extracted is not None and extracted == expected,
                  reason="malformed_answer" if extracted is None else ("correct" if extracted == expected else "incorrect"))
    return result
