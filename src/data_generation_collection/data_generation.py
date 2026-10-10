"""Three scientific source branches with overlap-aware ordering and fail-closed joins."""
from pathlib import Path
from shared_procedures.identity import content_hash
from shared_procedures.storage import atomic_write_json
from data_generation_collection.evaluation_data.evaluation_data_preparation import prepare_evaluation_data
from data_generation_collection.rl_data.rl_data_preparation import prepare_rl_data
from data_generation_collection.sft_data.clean_trace_collection.candidate_manifest import prepare_candidate_manifest
from data_generation_collection.sft_data.sft_data_preparation import prepare_sft_data


def prepare_data(*, evaluation_pools, evaluation_revisions, gsm8k_train,
                 gsm8k_train_revision, lmsys_rows, lmsys_revision, lmsys_authorized,
                 tokenizers, generator, kazitext, output_root, config_hash,
                 verify_mbpp_reference=None, protocol_builder=None,
                 analysis_plan=None, preflight_reports=None) -> dict:
    """Prepare every real source branch; supplied adapters perform actual work.

    No fixtures are synthesized here. The evaluator and training orchestrator
    must still enforce applicable runtime, statistical and spending launch gates.
    """
    root = Path(output_root)
    evaluation = prepare_evaluation_data(evaluation_pools, evaluation_revisions,
                    output_dir=root / "evaluation", protocol_builder=protocol_builder,
                    verify_mbpp_reference=verify_mbpp_reference, analysis_plan=analysis_plan,
                    preflight_reports=preflight_reports)
    # Exclude detectable matches to the entire official evaluation frame, not
    # merely whichever 1233 IDs happen to be sampled into the panel.
    evaluation_questions = [row.get("question", row.get("prompt", row.get("text", ""))) for rows in evaluation_pools.values() for row in rows]
    rl = prepare_rl_data(gsm8k_train, gsm8k_train_revision, evaluation_questions,
                         output_path=root / "rl" / "manifest.json")
    candidates = prepare_candidate_manifest(lmsys_rows, tokenizers, lmsys_revision,
                    evaluation_questions, authorized=lmsys_authorized,
                    output_path=root / "candidates" / "manifest.json")
    sft = prepare_sft_data(candidates, generator, tokenizers, kazitext=kazitext,
                    clean_output_dir=root, transformed_output_dir=root / "transformed", config_hash=config_hash)
    if sft["status"] != "verified" or rl["status"] != "verified":
        raise ValueError("Incomplete source branches cannot be released")
    result = {"status": "verified" if evaluation["launch_ready"] else "prepared",
              "launch_ready": evaluation["launch_ready"], "sft": sft, "rl": rl,
              "evaluation": evaluation, "config_hash": config_hash,
              "content_hash": content_hash({"sft": sft["content_hash"], "rl": rl["content_hash"], "evaluation": evaluation["content_hash"]})}
    atomic_write_json(root / "data_manifest.json", result)
    return result

data_generation = prepare_data
