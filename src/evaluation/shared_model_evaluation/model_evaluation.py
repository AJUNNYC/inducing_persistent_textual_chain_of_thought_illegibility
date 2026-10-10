"""Evaluate every fixed item for one exact base, SFT adapter, or RL snapshot."""
from __future__ import annotations
from pathlib import Path
from collections import Counter

from shared_procedures.identity import content_hash
from shared_procedures.storage import atomic_write_json, notify_artifact
from data_generation_collection.evaluation_data.protocol.evaluation_protocol import validate_protocol
from evaluation.shared_model_evaluation.item_evaluation.native_output.output_generation import validate_state, question_id, canonical_domain
from evaluation.shared_model_evaluation.item_evaluation.item_evaluation import evaluate_item
from evaluation.shared_model_evaluation.item_evaluation.legibility_judging.legibility_judging import summarize_legibility


def evaluate_model(state: dict, protocol: dict, generator, judge_client=None, *, sandbox=None,
                   cache_root: str | Path | None = None, seed: int = 42) -> dict:
    validate_state(state)
    validate_protocol(protocol, allow_draft=state.get("family", "base") == "base")
    records = []
    for item in protocol["items"]:
        record = evaluate_item(state, item, protocol, generator, judge_client, sandbox=sandbox,
                               cache_root=cache_root, seed=seed)
        records.append(record)
        # Persistent execution failures stop the worker; retain honest coverage.
        if record["status"] == "incomplete":
            break
    expected_ids = [question_id(item) for item in protocol["items"]]
    present = {record["question_id"] for record in records}
    missing = [identifier for identifier in expected_ids if identifier not in present]
    incomplete = [record["question_id"] for record in records if record["status"] != "complete"]
    capabilities = {}
    for domain in sorted({canonical_domain(item["domain"]) for item in protocol["items"]}):
        subset = [record for record in records if record["domain"] == domain]
        denominator = sum(canonical_domain(item["domain"]) == domain for item in protocol["items"])
        observed = [record for record in subset if type(record["correct"]) is bool]
        capabilities[domain] = {"denominator": denominator, "recorded": len(subset), "observed": len(observed),
            "correct_count": sum(record["correct"] for record in observed),
            "accuracy": sum(record["correct"] for record in observed)/denominator if len(observed)==denominator else None,
            "missing_scoring": denominator-len(observed)}
    legibility_records = [{**record["legibility"], "domain": record["domain"]} for record in records if record["selected"]]
    summary = summarize_legibility(legibility_records) if len(legibility_records) == 200 else None
    package = {"status": "complete" if not missing and not incomplete else "incomplete",
        "state": state, "state_id": state["state_id"], "protocol_hash": protocol["protocol_hash"], "seed": seed,
        "records": records, "expected_item_count": len(expected_ids), "missing_question_ids": missing,
        "incomplete_question_ids": incomplete, "capability_summary": capabilities, "legibility_summary": summary,
        "record_statuses": dict(Counter(record["status"] for record in records)),
        "scientific_evidence_status": "indeterminate" if missing or incomplete or any(record.get("scientific_evidence_status")=="indeterminate" for record in records) else "recorded"}
    package["package_hash"] = content_hash(package)
    if cache_root:
        path = Path(cache_root) / "capability" / state["state_id"] / ("package_" + content_hash({"state": state, "protocol": protocol["protocol_hash"], "seed": seed}) + ".json")
        atomic_write_json(path, package)
        notify_artifact(path, {"kind": "evaluation_package", "state_id": state["state_id"],
                               "package_hash": package["package_hash"], "status": package["status"]})
    return package
