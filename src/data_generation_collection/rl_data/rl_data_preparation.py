"""Freeze exactly 800 numeric GSM8K training questions in 100 groups of eight."""
from __future__ import annotations

from shared_procedures.identity import content_hash, derive_seed
from shared_procedures.response_parsing import normalize_numeric_answer
from shared_procedures.storage import atomic_write_json, read_json
from data_generation_collection.sft_data.clean_trace_collection.candidate_manifest import normalize_prompt, validate_revision


def load_rl_source(revision: str, *, cache_root=None):
    validate_revision(revision)
    from data_generation_collection.sft_data.clean_trace_collection.candidate_manifest import iter_pinned_parquet_rows
    return iter_pinned_parquet_rows("openai/gsm8k", revision, configuration="main", split="train", cache_root=cache_root)


def prepare_rl_data(rows, source_revision: str, evaluation_questions, *, output_path=None) -> dict:
    validate_revision(source_revision)
    overlap = {normalize_prompt(q) for q in evaluation_questions}
    eligible, exclusions, seen, source_ids = [], [], set(), set()
    for index, row in enumerate(rows):
        source_id = str(row.get("id", index))
        question_id = "gsm8k_train_" + source_id
        if source_id in source_ids:
            raise ValueError("Duplicate GSM8K source ID")
        source_ids.add(source_id)
        question = row.get("question")
        raw_reference = row.get("answer", "")
        answer = normalize_numeric_answer(raw_reference.rsplit("####", 1)[-1]) if "####" in raw_reference else None
        reason = None
        if not isinstance(question, str) or not question.strip():
            reason = "missing_question"
        elif answer is None:
            reason = "invalid_numeric_reference"
        elif normalize_prompt(question) in overlap:
            reason = "detectable_evaluation_overlap"
        elif normalize_prompt(question) in seen:
            reason = "normalized_duplicate"
        if reason:
            exclusions.append({"question_id": question_id, "source_index": index, "reason": reason})
            continue
        seen.add(normalize_prompt(question))
        eligible.append({"question_id": question_id, "example_id": question_id,
                         "question": question, "reference_answer": answer,
                         "source_dataset": "openai/gsm8k", "source_split": "train",
                         "source_revision": source_revision, "source_index": index,
                         "source_content_hash": content_hash({"question": question, "reference_answer": answer})})
    eligible.sort(key=lambda r: (derive_seed(42, "rl_questions", r["question_id"]), r["question_id"]))
    if len(eligible) < 800:
        raise ValueError("GSM8K training pool cannot supply 800 distinct eligible questions")
    questions = eligible[:800]
    for order, record in enumerate(questions):
        record["order"] = order
    batches = [[r["question_id"] for r in questions[start:start + 8]] for start in range(0, 800, 8)]
    result = {"status": "verified", "schema_version": "rl_manifest_v1", "questions": questions,
              "batches": batches, "source_revision": source_revision, "source_split": "train",
              "eligible_population_size": len(eligible), "exclusions": exclusions, "seed": 42,
              "order_version": "sha256_42_rl_questions_v1", "count": 800,
              "evaluation_overlap_hash": content_hash(sorted(overlap)), "content_hash": content_hash(questions)}
    validate_rl_manifest(result)
    if output_path:
        if __import__('pathlib').Path(output_path).exists() and read_json(output_path) != result:
            raise ValueError("Frozen RL manifest differs; cannot replace IDs")
        atomic_write_json(output_path, result)
        if read_json(output_path) != result:
            raise IOError("RL manifest readback mismatch")
    return result


def validate_rl_manifest(manifest):
    rows = manifest.get("questions", [])
    if manifest.get("status") != "verified" or manifest.get("source_split") != "train" or len(rows) != 800 or len({r["question_id"] for r in rows}) != 800:
        raise ValueError("Invalid frozen RL count/split/IDs")
    if manifest.get("content_hash") != content_hash(rows) or [r["order"] for r in rows] != list(range(800)):
        raise ValueError("RL source hash/order invalid")
    batches = [[r["question_id"] for r in rows[start:start + 8]] for start in range(0, 800, 8)]
    if manifest.get("batches") != batches:
        raise ValueError("RL manifest must contain exactly 100 ordered groups of eight")
    for row in rows:
        if normalize_numeric_answer(row["reference_answer"]) is None or "reasoning" in row or "answer" in row:
            raise ValueError("RL records require valid numeric references and no imitation target")
    return manifest

rl_data_preparation = prepare_rl_data
