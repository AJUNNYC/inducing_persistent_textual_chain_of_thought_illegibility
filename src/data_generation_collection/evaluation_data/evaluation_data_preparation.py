"""Prepare fixed manifests and propagate honest protocol/preflight readiness."""
from pathlib import Path
from shared_procedures.identity import content_hash
from shared_procedures.storage import atomic_write_json, read_json
from data_generation_collection.evaluation_data.capability_panel.capability_panel_construction import construct_capability_panel
from data_generation_collection.evaluation_data.legibility_subset.legibility_subset_construction import construct_legibility_subset


def prepare_evaluation_data(pools, revisions, *, output_dir, protocol_builder=None,
                            verify_mbpp_reference=None, analysis_plan=None,
                            preflight_reports=None) -> dict:
    root = Path(output_dir)
    panel = construct_capability_panel(pools, revisions, output_path=root / "capability_panel.json", verify_mbpp_reference=verify_mbpp_reference)
    legibility = construct_legibility_subset(panel, output_path=root / "legibility_subset.json", require_verified_parent=panel["status"] == "verified")
    if protocol_builder is None:
        from data_generation_collection.evaluation_data.protocol.evaluation_protocol import build_evaluation_protocol
        protocol_builder = build_evaluation_protocol
    protocol = protocol_builder(panel, legibility, analysis_plan=analysis_plan, preflight_reports=preflight_reports)
    if not isinstance(protocol, dict) or protocol.get("status") not in {"draft", "frozen"}:
        raise ValueError("Evaluation protocol failed its declared child handoff")
    ready = panel["status"] == "verified" and legibility["status"] == "verified" and protocol["status"] == "frozen"
    result = {"status": "verified" if ready else "prepared", "launch_ready": ready,
              "capability_panel": panel, "legibility_subset": legibility, "protocol": protocol,
              "content_hash": content_hash({"panel": panel["content_hash"], "legibility": legibility["content_hash"], "protocol": protocol}),
              "blocked_requirements": ([] if ready else ["complete_verified_protocol_and_applicable_real_preflights"])}
    atomic_write_json(root / "evaluation_package.json", result)
    if read_json(root / "evaluation_package.json") != result:
        raise IOError("Evaluation package readback failed")
    return result

evaluation_data_preparation = prepare_evaluation_data
