"""Durable rollout -> all rewards -> one update -> exact checkpoint transaction."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import uuid
import re
from shared_procedures.identity import content_hash
from shared_procedures.storage import atomic_write_json, read_json
from shared_procedures.checkpointing import load_checkpoint, verify_checkpoint
from shared_procedures.recovery_procedure import RecoveryManager
from .rollouts.rollout_collection import collect_rollouts
from .rewards.rl_reward import compute_rl_reward
from .rewards.sonnet_judgment.sonnet_opacity_judgment import RewardInfrastructureError
from .policy_update.grpo_policy_update import apply_grpo_update, RollbackRequired


def _restore_latest(state: Any) -> None:
    candidates = []
    for checkpoint in (state.run_root / "checkpoints").glob("step_*"):
        if not re.fullmatch(r"step_\d{3}", checkpoint.name):
            continue
        manifest = verify_checkpoint(checkpoint, expected_identity=state.identity)
        candidates.append((manifest["progress"]["completed_steps"], checkpoint))
    if not candidates:
        raise RuntimeError("Rollback blocked: no verified full checkpoint")
    completed, path = max(candidates)
    restored = load_checkpoint(path, expected_identity=state.identity)
    state.runtime.restore_state(restored)
    if state.runtime.policy_hash() != restored["runtime"]["policy_hash"]:
        raise RuntimeError("Rollback restored adapter hash mismatch")
    state.completed_steps = completed
    state.boundary_verified = True
    state.attempt_id = uuid.uuid4().hex
    atomic_write_json(state.run_root / "attempt.json", {"attempt_id": state.attempt_id,
                       "restored_step": completed, "status": "replaying", "execution_mode": state.execution_mode})


def grpo_training_cycle(state: Any, questions: Sequence[Mapping[str, Any]], *, judge: Any,
                        on_checkpoint: Callable[[Any], None] | None = None,
                        stop_requested: Callable[[], bool] | None = None,
                        recovery_manager: RecoveryManager | None = None) -> dict[str, Any]:
    """Run 100 exact updates, or return an honest paused state with no partial reward update."""
    if len(questions) != 800 or len({q["question_id"] for q in questions}) != 800:
        raise ValueError("Full fixed 800-question order is required")
    manager = recovery_manager or getattr(judge, "recovery", None) or (
        RecoveryManager.for_worker(state.identity.get("runtime_identity", {}).get("artifact_root"))
        if state.execution_mode == "production" else RecoveryManager(state.run_root.parent / "retries.json"))
    while state.completed_steps < 100:
        step = state.completed_steps + 1
        if stop_requested is not None and stop_requested():
            state.save()
            return {"status": "paused", "reason": "controlled_stop", "completed_steps": state.completed_steps}
        group = questions[state.completed_steps*8:(state.completed_steps+1)*8]
        rollouts = collect_rollouts(state, group, recovery_manager=manager)
        rewards = []
        try:
            for rollout, question in zip(rollouts, [q for q in group for _ in range(4)]):
                cache_identity = {"completion_id": rollout["completion_id"], "raw_output_hash": rollout["raw_output_hash"],
                                  "reference": question["reference_answer"], "reward_version": "c3_boolean_v1"}
                path = state.run_root / "rewards" / f"{content_hash(cache_identity)}.json"
                if path.exists():
                    cached = read_json(path)
                    if cached["identity"] != cache_identity or cached["reward_hash"] != content_hash(cached["reward"]):
                        raise ValueError("Existing verified reward cache integrity mismatch")
                    reward = cached["reward"]
                else:
                    reward = compute_rl_reward(rollout, question["reference_answer"], judge=judge)
                    atomic_write_json(path, {"identity": cache_identity, "reward": reward, "reward_hash": content_hash(reward)})
                rewards.append(reward)
        except RewardInfrastructureError as exc:
            # Every raw rollout and successful pass is already durable. Pause before optimizer update.
            state.save()
            pause = {"status": "paused", "reason": "required_judge_unavailable", "completed_steps": state.completed_steps,
                     "pending_step": step, "completed_rewards": len(rewards), "error": str(exc)}
            atomic_write_json(state.run_root / "progress.json", pause)
            return pause
        try:
            boundary = apply_grpo_update(state, rollouts, rewards)
        except RollbackRequired:
            # Budget identity uses failed logical step, never attempt ID; counters survive replay.
            manager.reserve_retry(f"rl:{state.identity['model_id']}:update:{step}")
            _restore_latest(state)
            continue
        if state.completed_steps % 10 == 0 and on_checkpoint is not None:
            on_checkpoint(state)
        atomic_write_json(state.run_root / "progress.json", boundary)
    return {"status": "completed", "completed_steps": 100, "planned_rollouts": 3200,
            "execution_mode": state.execution_mode}
