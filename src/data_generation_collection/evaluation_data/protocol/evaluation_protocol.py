"""Prospective native measurement specifications, with explicit launch gates."""
from __future__ import annotations
from pathlib import Path
from collections import Counter

from shared_procedures.identity import content_hash, file_hash
from evaluation.shared_model_evaluation.item_evaluation.native_output.output_generation import (
    GENERATION_SETTINGS, GENERATION_VERSION, PROMPT_ROOT, canonical_domain, question_id, render_prompt)
from evaluation.shared_model_evaluation.item_evaluation.capability_scoring.capability_scoring import SCORER_VERSION, PARSER_VERSION
from evaluation.shared_model_evaluation.item_evaluation.capability_scoring.mbpp_sandbox import LIMITS, SANDBOX_VERSION
from evaluation.shared_model_evaluation.item_evaluation.legibility_judging.legibility_judging import (
    SCORE_SCHEMA, PRIMARY_MODEL, SECONDARY_MODEL, PRIMARY_SETTINGS, SECONDARY_SETTINGS)
from evaluation.shared_model_evaluation.item_evaluation.legibility_judging.readable_solution_audit.readable_solution_audit import (
    AUDIT_SCHEMA, AUDIT_MODEL, AUDIT_SETTINGS, AUDIT_VERSION)

PROTOCOL_VERSION = "native_evaluation_protocol_v1"
DOMAIN_COUNTS = {"gsm8k": 300, "mmlu": 399, "arc": 331, "mbpp": 203}
REQUIRED_PREFLIGHTS = ("model_runtime", "judge_services", "mbpp_sandbox", "baseline_repeatability", "statistical_validation", "durable_storage")


def selected_ids(legibility) -> list[str]:
    if isinstance(legibility, dict):
        values = legibility.get("question_ids", legibility.get("item_ids", legibility.get("ids", legibility.get("items"))))
    else:
        values = legibility
    if not isinstance(values, (list, tuple, set)):
        raise ValueError("Legibility package lacks the exact selected IDs")
    return [question_id(value) if isinstance(value, dict) else str(value) for value in values]


def normalize_preflight_reports(reports: dict | None) -> tuple[dict, str | None]:
    supplied = reports or {}
    aliases = {"native_generation": "model_runtime", "judge_compatibility": "judge_services"}
    if "checks" in supplied:
        copied = dict(supplied)
        saved_hash = copied.pop("report_hash", None)
        if saved_hash is None or saved_hash != content_hash(copied):
            raise ValueError("Preflight report integrity hash is missing or mismatched")
        rows = supplied["checks"]
        if len({row["name"] for row in rows}) != len(rows):
            raise ValueError("Duplicate preflight identities")
        normalized = {aliases.get(row["name"], row["name"]): row for row in rows}
        return normalized, supplied.get("configuration_hash")
    normalized = {aliases.get(name, name): row for name,row in supplied.items()}
    hashes = {row.get("evidence", {}).get("configuration_hash") for row in normalized.values() if isinstance(row, dict)} - {None}
    if len(hashes) > 1:
        raise ValueError("Preflight configuration identities disagree")
    return normalized, next(iter(hashes)) if hashes else None


def _gate_passed(row: dict | None) -> bool:
    return bool(isinstance(row, dict) and row.get("status") == "passed"
                and row.get("evidence", {}).get("live") is True
                and row.get("evidence", {}).get("simulation") is not True)


def build_evaluation_protocol(panel: dict | list[dict], legibility, *, analysis_plan: dict | None = None,
                              preflight_reports: dict | None = None, require_ready: bool = False,
                              configuration_hash: str | None = None, generation_backend: str = 'native') -> dict:
    items = panel.get("items") if isinstance(panel, dict) else panel
    if not isinstance(items, list):
        raise ValueError("Capability panel must contain item records")
    ids = [question_id(item) for item in items]
    if len(ids) != len(set(ids)) or dict(Counter(canonical_domain(item["domain"]) for item in items)) != DOMAIN_COUNTS:
        raise ValueError("Capability package must contain exactly the common 1,233 unique items")
    selected = selected_ids(legibility)
    if len(selected) != 200 or len(set(selected)) != 200 or not set(selected) <= set(ids):
        raise ValueError("Legibility panel must have exactly 200 unique nested items")
    by_id = {question_id(item): item for item in items}
    if Counter(canonical_domain(by_id[i]["domain"]) for i in selected) != Counter({domain: 50 for domain in DOMAIN_COUNTS}):
        raise ValueError("Legibility panel must retain exactly 50 items per domain")
    # Validate public rendering and reference/test identity before any generation.
    for item in items:
        if not item.get("source_revision"):
            raise ValueError("Evaluation item lacks pinned dataset provenance")
        render_prompt(item)
        if canonical_domain(item["domain"]) == "mbpp":
            if not item.get("scoring_tests"):
                raise ValueError("MBPP scoring tests are missing")
            for test in ([item["test_setup_code"]] if item.get("test_setup_code") else []) + item["scoring_tests"]:
                compile(test, "<frozen_mbpp_test>", "exec")
        elif item.get("reference_answer") is None:
            raise ValueError("Evaluation reference answer is missing")
    prompts = {name: (PROMPT_ROOT / (name + ".txt")).read_text() for name in [
        "evaluation_gsm8k", "evaluation_mmlu", "evaluation_arc_easy", "evaluation_mbpp",
        "evaluation_legibility", "evaluation_readable_solution"]}
    code_root = Path(__file__).resolve().parents[3]
    sources = [
        Path(__file__),
        code_root / "evaluation/evaluation.py",
        code_root / "evaluation/condition_groups/condition_evaluation.py",
        code_root / "evaluation/shared_model_evaluation/model_evaluation.py",
        code_root / "evaluation/shared_model_evaluation/item_evaluation/item_evaluation.py",
        code_root / "evaluation/shared_model_evaluation/item_evaluation/native_output/output_generation.py",
        code_root / "evaluation/shared_model_evaluation/item_evaluation/capability_scoring/capability_scoring.py",
        code_root / "evaluation/shared_model_evaluation/item_evaluation/capability_scoring/mbpp_sandbox.py",
        code_root / "evaluation/shared_model_evaluation/item_evaluation/legibility_judging/legibility_judging.py",
        code_root / "evaluation/shared_model_evaluation/item_evaluation/legibility_judging/readable_solution_audit/readable_solution_audit.py",
        code_root / "shared_procedures/response_parsing.py"]
    generation = {'version': GENERATION_VERSION, 'settings': GENERATION_SETTINGS, 'seed':42, 'diagnostic_baseline_seeds':[43,44]}
    if generation_backend == 'vllm':
        from evaluation.shared_model_evaluation.item_evaluation.native_output.vllm_generation import VERSION, BACKEND
        generation.update(version=VERSION, backend=dict(BACKEND))
        sources.append(code_root / 'evaluation/shared_model_evaluation/item_evaluation/native_output/vllm_generation.py')
    elif generation_backend != 'native':
        raise ValueError('Unknown measurement backend')
    specifications = {"version": PROTOCOL_VERSION, "panel_hash": content_hash(items),
        "legibility_ids": sorted(selected), "domain_counts": DOMAIN_COUNTS,
        "generation": generation,
        "prompts": prompts, "prompt_hashes": {name: content_hash(text) for name,text in prompts.items()},
        "parser": {"version": PARSER_VERSION, "reasoning": "single_native_think_boundary", "choice": "one_exact_public_label",
                   "mbpp": "final_field_only_remove_at_most_one_python_fence_no_repairs"},
        "scorer": {"version": SCORER_VERSION, "mbpp_variant": "withheld_tests_public_interface_pass_at_one"},
        "sandbox": {"version": SANDBOX_VERSION, "limits": LIMITS},
        "judges": {"primary": {"kind": "evaluation_primary", "model": PRIMARY_MODEL, "settings": PRIMARY_SETTINGS, "schema": SCORE_SCHEMA},
                   "secondary": {"kind": "evaluation_secondary", "model": SECONDARY_MODEL, "settings": SECONDARY_SETTINGS, "schema": SCORE_SCHEMA},
                   "audit": {"kind": "evaluation_audit", "model": AUDIT_MODEL, "settings": AUDIT_SETTINGS, "schema": AUDIT_SCHEMA, "version": AUDIT_VERSION}},
        "opacity": {"primary_score_floor": 6, "readable_solution": "not_detected", "padding": "diagnostic_only"},
        "code_hashes": {str(path.relative_to(code_root)): content_hash(path.read_text()) for path in sources},
        "analysis_plan_hash": (analysis_plan or {}).get("plan_hash")}
    reports, evidence_config_hash = normalize_preflight_reports(preflight_reports)
    if configuration_hash and evidence_config_hash and configuration_hash != evidence_config_hash:
        raise ValueError("Preflight report belongs to another configuration")
    configuration_hash = configuration_hash or evidence_config_hash
    specifications["configuration_hash"] = configuration_hash
    blocked = [name for name in REQUIRED_PREFLIGHTS if not _gate_passed(reports.get(name))]
    from evaluation.shared_model_evaluation.item_evaluation.capability_scoring.mbpp_sandbox import _selected_reference_binding, DockerMBPPSandbox
    selected_binding_hash = content_hash(_selected_reference_binding({"items":items}))
    specifications["sandbox"]["selected_reference_binding_hash"] = selected_binding_hash
    mbpp_evidence = reports.get("mbpp_sandbox", {}).get("evidence", {})
    specifications["sandbox"]["identity"] = mbpp_evidence.get("sandbox_identity")
    sandbox_source_hash = file_hash(code_root / "evaluation/shared_model_evaluation/item_evaluation/capability_scoring/mbpp_sandbox.py")
    identity = mbpp_evidence.get("sandbox_identity")
    current_identity = DockerMBPPSandbox(identity["image"]).identity if isinstance(identity,dict) and isinstance(identity.get("image"),str) else None
    if _gate_passed(reports.get("mbpp_sandbox")) and (
        mbpp_evidence.get("selected_binding_hash") != selected_binding_hash
        or mbpp_evidence.get("selected_reference_status") != "passed"
        or mbpp_evidence.get("selected_count") != 203 or mbpp_evidence.get("selected_passed_count") != 203
        or identity != current_identity or current_identity is None
        or mbpp_evidence.get("implementation_hash") != sandbox_source_hash):
        blocked.append("mbpp_reference_binding")
    if not configuration_hash:
        blocked.append("preflight_configuration_identity")
    if not analysis_plan or analysis_plan.get("status") != "frozen" or not analysis_plan.get("plan_hash"):
        blocked.append("prospective_analysis_plan")
    status = "draft" if blocked else "frozen"
    if require_ready and blocked:
        raise RuntimeError("Evaluation protocol launch gates remain blocked: " + ", ".join(blocked))
    return {"status": status, "version": PROTOCOL_VERSION, "items": items, "legibility_ids": sorted(selected),
            "specifications": specifications, "protocol_hash": content_hash(specifications),
            "preflight_reports": reports, "blocked_preflights": blocked,
            "analysis_plan": analysis_plan, "panel_hash": specifications["panel_hash"],
            "preflight_configuration_hash": configuration_hash, "preflight_reports_hash": content_hash(reports)}


def freeze_evaluation_protocol(panel, legibility, **kwargs) -> dict:
    return build_evaluation_protocol(panel, legibility, **kwargs, require_ready=True)


def validate_protocol(protocol: dict, *, allow_draft: bool = False) -> None:
    if protocol.get("status") != "frozen" and not (allow_draft and protocol.get("status") == "draft"):
        raise ValueError("Evaluation requires a frozen package; only baseline preflights may use its draft")
    if content_hash(protocol.get("specifications")) != protocol.get("protocol_hash"):
        raise ValueError("Evaluation protocol hash mismatch")
    if content_hash(protocol.get("items")) != protocol.get("panel_hash"):
        raise ValueError("Capability panel integrity failure")
    if sorted(protocol.get("legibility_ids", [])) != protocol["specifications"]["legibility_ids"]:
        raise ValueError("Legibility subset integrity failure")
    if not allow_draft:
        if protocol.get("blocked_preflights"):
            raise ValueError("Required evaluation preflights have not passed")
        if any(not _gate_passed(protocol.get("preflight_reports", {}).get(name)) for name in REQUIRED_PREFLIGHTS):
            raise ValueError("Frozen protocol lacks actual mandatory preflight records")
        evidence = protocol["preflight_reports"]["mbpp_sandbox"].get("evidence",{})
        if (evidence.get("selected_binding_hash") != protocol["specifications"]["sandbox"].get("selected_reference_binding_hash")
            or evidence.get("selected_reference_status") != "passed" or evidence.get("selected_count") != 203
            or evidence.get("selected_passed_count") != 203
            or evidence.get("sandbox_identity") != protocol["specifications"]["sandbox"].get("identity")):
            raise ValueError("Frozen MBPP preflight does not certify this exact selected panel/executor")

    if content_hash(protocol.get("preflight_reports", {})) != protocol.get("preflight_reports_hash"):
        raise ValueError("Preflight evidence integrity failure")
    code_root = Path(__file__).resolve().parents[3]
    for relative, expected in protocol["specifications"]["code_hashes"].items():
        if content_hash((code_root / relative).read_text()) != expected:
            raise ValueError("Frozen evaluation implementation changed; invalidate affected derivatives")
