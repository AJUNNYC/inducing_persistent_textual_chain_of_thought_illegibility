"""Join capability and scheduled assessment from exactly one native output."""
from __future__ import annotations
from pathlib import Path

from shared_procedures.identity import content_hash
from shared_procedures.storage import atomic_write_json, read_json, file_lock, notify_artifact
from shared_procedures.response_parsing import parse_native_output
from evaluation.shared_model_evaluation.item_evaluation.native_output.output_generation import question_id, public_question, canonical_domain
from evaluation.shared_model_evaluation.item_evaluation.capability_scoring.capability_scoring import score_capability
from evaluation.shared_model_evaluation.item_evaluation.legibility_judging.legibility_judging import judge_legibility

ITEM_VERSION = "evaluation_item_v1"


def evaluate_item(state: dict, item: dict, protocol: dict, generator, judge_client=None,
                  *, sandbox=None, cache_root: str | Path | None = None, seed: int = 42) -> dict:
    identity = {"version": ITEM_VERSION, "state": state, "question_id": question_id(item),
                "item_hash": content_hash(item), "protocol_hash": protocol["protocol_hash"], "seed": seed,
                "generation_identity": generator.identity(state,item,seed) if hasattr(generator,"identity") else None,
                "judge_execution_mode": "production" if getattr(judge_client,"live",False) else "fixture",
                "sandbox_identity": getattr(sandbox,"identity",None), "sandbox_simulation": getattr(sandbox,"simulation",False)}
    key = content_hash(identity)
    path = Path(cache_root) / "capability" / state["state_id"] / (key + ".json") if cache_root else None
    if path is not None:
        with file_lock(str(path) + ".lock"):
            if path.exists():
                cached = read_json(path)
                check = dict(cached)
                saved_hash = check.pop("record_hash", None)
                if saved_hash != content_hash(check) or cached.get("identity") != identity:
                    raise ValueError("Item record identity/integrity failure")
                if cached.get("status") == "complete":
                    return cached
            record = _evaluate(state, item, protocol, generator, judge_client, sandbox, seed, identity)
            record["record_hash"] = content_hash(record)
            atomic_write_json(path, record)
            if read_json(path) != record:
                raise OSError("Item record publication did not verify")
            notify_artifact(path, {"kind": "evaluation_item", "state_id": state["state_id"], "identity_hash": key,
                                   "status": record["status"], "simulation": record.get("simulation", False)})
            for kind, value in [("legibility", record.get("legibility")),
                                ("audits", record.get("legibility", {}).get("requests", {}).get("audit"))]:
                if value is None:
                    continue
                derivative = {"identity": identity, "raw_hash": record.get("raw_hash"), "data": value,
                              "simulation": record.get("simulation", False)}
                derivative["record_hash"] = content_hash(derivative)
                derivative_path = Path(cache_root) / kind / state["state_id"] / (key + ".json")
                atomic_write_json(derivative_path, derivative)
                notify_artifact(derivative_path, {"kind": "evaluation_"+kind, "state_id": state["state_id"],
                                                  "record_hash": derivative["record_hash"], "simulation": derivative["simulation"]})
            return record
    return _evaluate(state, item, protocol, generator, judge_client, sandbox, seed, identity)


def _evaluate(state, item, protocol, generator, judge_client, sandbox, seed, identity):
    selected = question_id(item) in set(protocol["legibility_ids"]) and seed == 42
    record = {"identity": identity, "state_id": state["state_id"], "model": state["model"],
        "family": state.get("family", "base"), "level": state.get("level"), "step": state.get("step"),
        "question_id": question_id(item), "item_id": question_id(item), "domain": canonical_domain(item["domain"]),
        "seed": seed, "selected": selected, "public_question": public_question(item),
        "correct": None, "primary_score": None, "secondary_score": None, "readable_solution": None,
        "padding": None, "assessable": False, "qualified_opacity": None, "status": "incomplete",
        "simulation": bool(getattr(generator,"simulation",True))}
    try:
        raw = generator.generate(state, item, seed=seed)
    except Exception as error:
        return {**record, "raw": None, "generation_status": "infrastructure_failure",
                "error_type": type(error).__name__, "capability": {"status": "not_available", "correct": None},
                "legibility": {"status": "not_available" if selected else "not_scheduled", "scheduled": selected}}
    native_prefix = raw.get("native_prefix", "")
    if native_prefix not in {"", "<think>", "<think>\n"}:
        raise ValueError("Only a recorded exact native tokenizer prefill may restore the opening boundary")
    parsed = parse_native_output(native_prefix + raw["raw_output"])
    parsed["truncated"] = raw.get("finish_reason") == "length"
    assessable = parsed["status"] == "valid" and bool(parsed.get("reasoning", "").strip()) and not parsed["truncated"]
    try:
        capability = score_capability(item, parsed, sandbox=sandbox, raw_hash=raw["raw_hash"])
    except Exception as error:
        capability = {"status": "infrastructure_failure", "correct": None, "error_type": type(error).__name__}
    legibility = judge_legibility(record["public_question"], parsed.get("reasoning"), judge_client,
                                 selected=selected, assessable=assessable)
    complete = capability["status"] == "outcome" and legibility["status"] in {"complete", "not_assessable", "not_scheduled", "judge_ineligible"}
    record.update(raw=raw, raw_hash=raw["raw_hash"], generation_status="success", parsed=parsed,
        reasoning=parsed.get("reasoning"), final_answer=parsed.get("final_answer"),
        malformed=parsed["status"] != "valid", truncated=parsed["truncated"], capability=capability,
        legibility=legibility, correct=capability.get("correct"), status="complete" if complete else "incomplete")
    for field in ("primary_score", "secondary_score", "readable_solution", "padding", "assessable", "qualified_opacity"):
        record[field] = legibility.get(field)
    record["simulation"] = record["simulation"] or bool(raw.get("simulation",True)) or bool(capability.get("execution", {}).get("simulation")) or any(
        value.get("simulation", value.get("request", {}).get("simulation", False))
        for value in legibility.get("requests", {}).values())
    record["scientific_evidence_status"] = "indeterminate" if selected and record["qualified_opacity"] is None else "recorded"
    return record
