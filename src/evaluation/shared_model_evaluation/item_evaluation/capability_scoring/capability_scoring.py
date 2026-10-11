"""Frozen deterministic domain scorers; infrastructure failures stay missing."""
from __future__ import annotations
import re

from shared_procedures.identity import content_hash
from shared_procedures.response_parsing import normalize_numeric_answer, PARSER_VERSION
from evaluation.shared_model_evaluation.item_evaluation.native_output.output_generation import canonical_domain

SCORER_VERSION = "capability_scorer_v2_final_fields"
ANSWER_EXTRACTION_VERSION = "designated_final_fields_v2"
ANSWER_EXTRACTION_CONTRACT = {
    "version": ANSWER_EXTRACTION_VERSION,
    "scope": "native_final_response_only_never_reasoning",
    "scalar": "unique_terminal_final_answer_marker_or_box_or_entire_scalar",
    "presentation": "case_insensitive_final_answer_marker_optional_bold_and_paired_math_wrappers",
    "ambiguity": "reject_multiple_final_answer_markers_or_boxes_and_non_scalar_suffixes",
    "numeric": "existing_exact_rational_decimal_grammar_no_units_or_prose_inference",
    "choice": "one_exact_public_label_no_option_text_or_prose_inference",
    "mbpp": "exactly_one_python_or_unlabelled_fenced_final_code_field_or_entire_plain_code_no_repairs",
}
FINAL_FIELD_VERSION = ANSWER_EXTRACTION_VERSION
FINAL_FIELD_CONTRACT = ANSWER_EXTRACTION_CONTRACT

_FINAL_MARKER = re.compile(r"\bFinal[ \t]+answer(?:\*\*)?[ \t]*:", re.IGNORECASE)
_FINAL_LINE = re.compile(
    r"(?m)^[ \t]*(?:\*\*)?Final[ \t]+answer(?:\*\*)?[ \t]*:(?:\*\*)?[ \t]*",
    re.IGNORECASE,
)
_BOX = re.compile(r"\\boxed\{([^{}]+)\}")


def _unwrap_scalar_presentation(text: str) -> str:
    """Remove only complete, paired display wrappers around a designated field."""
    text = text.strip()
    for opening, closing in (("**", "**"), (r"\[", r"\]"), (r"\(", r"\)"), ("$$", "$$"), ("$", "$")):
        if text.startswith(opening) and text.endswith(closing) and len(text) > len(opening) + len(closing):
            text = text[len(opening):-len(closing)].strip()
    return text


def extract_scalar_final_field(answer: str) -> str | None:
    """Locate an explicit final scalar without consulting a reference answer.

    Native final responses may explain a solution before the requested final-answer
    line. Only that declared field, a unique terminal box, or an entire scalar is
    eligible; no number or label is inferred from explanatory prose.
    """
    if not isinstance(answer, str) or not answer.strip():
        return None
    text = answer.strip()
    markers = list(_FINAL_MARKER.finditer(text))
    box_count = text.count(r"\boxed")
    if len(markers) > 1 or box_count > 1:
        return None
    if markers:
        marker = _FINAL_LINE.search(text)
        if marker is None:
            return None
        field = _unwrap_scalar_presentation(text[marker.end():])
        # Any box must be the whole designated field, not a number embedded in prose.
        if box_count and _BOX.fullmatch(field) is None:
            return None
        return field
    if box_count:
        box = _BOX.search(text)
        if box is None:
            return None
        prefix, suffix = text[:box.start()].rstrip(), text[box.end():].strip()
        if suffix in {"", ".", "!", "?", ";"}:
            return box.group(0)
        for opening, closing in ((r"\[", r"\]"), (r"\(", r"\)"), ("$$", "$$"), ("$", "$"), ("**", "**")):
            if prefix.endswith(opening) and suffix == closing:
                return box.group(0)
        return None
    return _unwrap_scalar_presentation(text)


def parse_choice_label(answer: str, labels: list[str]) -> str | None:
    if not isinstance(answer, str):
        return None
    text = extract_scalar_final_field(answer)
    if text is None:
        return None
    match = re.fullmatch(r"(?:\\boxed\{([^{}]+)\}|\(([^()]+)\)|([^\s]+))", text)
    if match is None:
        return None
    label = next(value for value in match.groups() if value is not None)
    return label if label in labels else None


def extract_final_code(answer: str) -> str | None:
    """Read one designated final-code fence, preserving its interior bytes exactly."""
    if not isinstance(answer, str) or not answer.strip():
        return None
    if "```" in answer:
        if answer.count("```") != 2:
            return None
        match = re.search(r"(?m)^[ \t]*```(?:python)?[ \t]*\r?\n(.*?)\r?\n[ \t]*```[ \t]*(?=\r?$)", answer, re.DOTALL)
        if not match or not match.group(1).strip() or "```" in match.group(1):
            return None
        return match.group(1)
    return answer


def score_capability(item: dict, parsed: dict, *, sandbox=None, raw_hash: str | None = None) -> dict:
    domain = canonical_domain(item["domain"])
    reference = item.get("reference_answer")
    dependency = {"raw_hash": raw_hash, "parser_version": PARSER_VERSION,
        "scorer_version": SCORER_VERSION, "answer_extraction_version": ANSWER_EXTRACTION_VERSION,
        "reference_hash": content_hash(
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
        numeric_text = extract_scalar_final_field(answer)
        boxed = _BOX.fullmatch(numeric_text) if numeric_text is not None else None
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
