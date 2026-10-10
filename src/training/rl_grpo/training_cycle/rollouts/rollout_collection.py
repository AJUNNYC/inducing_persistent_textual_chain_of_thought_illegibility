"""Exactly eight frozen questions and four persisted ordinary outcomes per question."""
from pathlib import Path
from typing import Any, Mapping, Sequence
from shared_procedures.identity import content_hash, derive_seed
from shared_procedures.storage import atomic_write_json, read_json
from shared_procedures.recovery_procedure import RecoveryManager

PROMPT_FILE = Path(__file__).resolve().parents[5] / "prompts/rl_generation.txt"


def format_rl_prompt(question: str) -> str:
    if not question.strip():
        raise ValueError("Empty public RL question")
    return PROMPT_FILE.read_text().replace("{question}", question).rstrip()


def validate_rollout_groups(rollouts: Sequence[Mapping[str, Any]], question_ids: Sequence[str] | None = None) -> None:
    if len(rollouts) != 32 or len({r["completion_id"] for r in rollouts}) != 32:
        raise ValueError("Exactly 32 distinct completion records required")
    groups: dict[str, list[Mapping[str, Any]]] = {}
    policies = {r["policy_hash"] for r in rollouts}
    for r in rollouts:
        groups.setdefault(r["question_id"], []).append(r)
        if r.get("raw_output_hash") != content_hash(r["raw_output"]):
            raise ValueError("Persisted raw rollout hash mismatch")
        if r.get("status") != "completed":
            raise ValueError("Missing/incomplete rollout cannot enter an update")
    if len(policies) != 1 or len(groups) != 8 or any(len(group) != 4 for group in groups.values()):
        raise ValueError("Eight distinct questions, four completions, one policy required")
    if any({r["completion_index"] for r in group} != {0, 1, 2, 3} for group in groups.values()):
        raise ValueError("Question group must contain each of four completion indices exactly once")
    if question_ids is not None and list(groups) != list(question_ids):
        raise ValueError("Actual sampled question order differs from frozen group")
    for start in range(0, 32, 4):
        if len({r["question_id"] for r in rollouts[start:start+4]}) != 1:
            raise ValueError("Groups of four must remain contiguous")


def collect_rollouts(state: Any, questions: Sequence[Mapping[str, Any]], *,
                     recovery_manager: RecoveryManager | None = None) -> list[dict[str, Any]]:
    if not state.boundary_verified or not 0 <= state.completed_steps < 100:
        raise RuntimeError("Sampling requires a verified unfinished optimizer boundary")
    if len(questions) != 8 or len({q["question_id"] for q in questions}) != 8:
        raise ValueError("Exactly eight distinct frozen questions required")
    policy_hash = state.policy_hash
    step = state.completed_steps + 1
    records = []
    for group_index, question in enumerate(questions):
        prompt = format_rl_prompt(question["question"])
        for completion_index in range(4):
            seed = derive_seed(42, "rl_rollout_v1", state.identity["model_id"], step,
                               question["question_id"], completion_index, policy_hash)
            identity = {"run_hash": content_hash(state.identity), "policy_hash": policy_hash,
                        "logical_step": step, "question_id": question["question_id"],
                        "question_hash": content_hash(question["question"]), "prompt_hash": content_hash(prompt),
                        "completion_index": completion_index, "seed": seed,
                        "generation_settings": {"temperature": 1.0, "top_p": 1.0, "top_k": 0,
                            "repetition_penalty": 1.0, "max_new_tokens": 2048, "use_vllm": False}}
            completion_id = content_hash(identity)
            destination = state.run_root / "rollouts" / f"step_{step:03d}" / f"{completion_id}.json"
            if destination.exists():
                record = read_json(destination)
                if record.get("identity") != identity or record.get("raw_output_hash") != content_hash(record["raw_output"]):
                    raise ValueError("Existing raw rollout cache is corrupt or incompatible")
            else:
                # A malformed/wrong/truncated sampled response is saved once, never resampled.
                if recovery_manager is None:
                    sampled = state.runtime.sample(prompt, seed=seed)
                else:
                    sampled = recovery_manager.run("rollout:" + completion_id,
                                lambda: state.runtime.sample(prompt, seed=seed))
                if not isinstance(sampled.get("raw_output"), str):
                    raise RuntimeError("Model runtime did not return a raw output string")
                record = {**sampled, "identity": identity, "completion_id": completion_id,
                          "question_id": question["question_id"], "question": question["question"],
                          "reference_answer": question["reference_answer"],
                          "policy_hash": policy_hash, "group_index": group_index,
                          "completion_index": completion_index, "logical_step": step,
                          "seed": seed, "attempt_id": state.attempt_id, "status": "completed",
                          "execution_mode": state.execution_mode,
                          "raw_output_hash": content_hash(sampled["raw_output"])}
                atomic_write_json(destination, record)
                if read_json(destination) != record:
                    raise RuntimeError("Raw rollout persistence verification failed")
            records.append(record)
    validate_rollout_groups(records, [q["question_id"] for q in questions])
    return records

rollout_collection = collect_rollouts
