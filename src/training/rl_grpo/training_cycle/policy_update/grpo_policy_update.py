"""Explicit H.7 token-level clipped Dr. GRPO, group-standardized rewards."""
from __future__ import annotations
import math
from statistics import mean, stdev
from typing import Any, Mapping, Sequence
from shared_procedures.identity import content_hash
from shared_procedures.storage import atomic_write_json
from shared_procedures.response_parsing import parse_rl_output, normalize_numeric_answer
from ..rewards.sonnet_judgment.sonnet_opacity_judgment import validate_pass_a, validate_pass_b
from ..rollouts.rollout_collection import validate_rollout_groups


class RLHardwareError(RuntimeError):
    """An unsafe GPU OOM requires a new verified hardware preflight before resuming."""


class RollbackRequired(RuntimeError):
    """A partial/uncertain update must restore a verified full optimizer boundary."""


def group_advantages(rewards: Sequence[float]) -> tuple[list[float], list[dict[str, Any]]]:
    if len(rewards) != 32 or any(not math.isfinite(float(r)) or not -1 <= float(r) <= 1.5 for r in rewards):
        raise ValueError("All 32 finite verified frozen rewards required")
    advantages, statistics = [], []
    for group_index in range(8):
        group = [float(value) for value in rewards[group_index*4:group_index*4+4]]
        center, spread = mean(group), stdev(group)
        advantages.extend((value - center) / (spread + 1e-4) for value in group)
        statistics.append({"group_index": group_index, "mean_reward": center, "reward_std": spread,
                           "reward_range": max(group)-min(group), "zero_variance": spread == 0})
    return advantages, statistics


def dr_grpo_token_loss(current_log_probs: Any, old_log_probs: Any, reference_log_probs: Any,
                       advantage: float, *, max_length: int = 2048, batch_size: int = 32):
    """Return this microbatch's contribution to sum(tokens)/(32*2048), plus KL.

    All tensors contain actual completion tokens (EOS included). No realized-length
    normalization and no truncated-completion mask are introduced.
    """
    import torch
    if max_length != 2048 or batch_size != 32:
        raise ValueError("Frozen Dr. GRPO denominator is 32*2048")
    if current_log_probs.ndim != 1 or current_log_probs.shape != old_log_probs.shape or current_log_probs.shape != reference_log_probs.shape:
        raise ValueError("Token-level probability arrays must have identical one-dimensional shape")
    if not 0 < current_log_probs.numel() <= 2048:
        raise ValueError("Completion token count outside fixed cap")
    ratio = torch.exp(current_log_probs - old_log_probs.detach())
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, min=.8, max=1.2) * advantage
    difference = reference_log_probs.detach() - current_log_probs
    kl = torch.exp(difference) - difference - 1
    token_loss = -torch.minimum(unclipped, clipped) + .01 * kl
    return token_loss.sum() / (batch_size * max_length), kl


def repetition_indicator(token_ids: Sequence[int]) -> float:
    """Diagnostic duplicate-token-fourgram fraction v1; never enters the reward."""
    grams = [tuple(token_ids[index:index+4]) for index in range(max(0, len(token_ids)-3))]
    return (len(grams)-len(set(grams)))/len(grams) if grams else 0.0


def validate_reward_transaction(rollouts: Sequence[Mapping[str, Any]], rewards: Sequence[Mapping[str, Any]]) -> None:
    validate_rollout_groups(rollouts)
    if len(rewards) != 32:
        raise ValueError("Missing rewards forbid a partial update")
    for rollout, reward in zip(rollouts, rewards):
        if (reward.get("status") != "verified" or reward.get("completion_id") != rollout["completion_id"]
            or reward.get("raw_output_hash") != rollout["raw_output_hash"]
            or reward.get("question_id") != rollout["question_id"] or reward.get("policy_hash") != rollout["policy_hash"]):
            raise ValueError("Reward identity does not match the exact persisted rollout")
        parsed = parse_rl_output(rollout["raw_output"], truncated=bool(rollout.get("truncated", False)))
        reference = normalize_numeric_answer(rollout["reference_answer"])
        if reference is None or reward.get("reference_answer") != reference or reward.get("parser") != parsed:
            raise ValueError("Reward parser/reference differs from exact frozen rollout")
        expected_status = "invalid" if parsed["status"] != "valid" else "correct" if parsed["numeric_answer"] == reference else "wrong"
        if reward.get("correctness") != expected_status:
            raise ValueError("Reward correctness differs from deterministic source parsing")
        expected_reward = -1.0 if expected_status == "invalid" else 0.0
        judgment = reward.get("judge")
        if expected_status == "correct":
            if not isinstance(judgment, Mapping) or judgment.get("status") != "verified":
                raise ValueError("A correct answer requires both verified judgment passes")
            if judgment["pass_a"].get("status") != "success" or judgment["pass_b"].get("status") != "success":
                raise ValueError("A missing judgment cannot supply a correct-answer bonus")
            score = validate_pass_a(judgment["pass_a"]["parsed"])
            readable, padding = validate_pass_b(judgment["pass_b"]["parsed"])
            effective = min(score, 3) if readable else score
            opacity, gate = (effective-1)/8, 0 if padding else 1
            if (judgment.get("score"), judgment.get("effective_score"), judgment.get("opacity"), judgment.get("padding_gate")) != (score, effective, opacity, gate):
                raise ValueError("Reward opacity derivation differs from frozen boolean score/cap/padding")
            expected_reward = 1.0 + .5 * opacity * gate
        elif judgment is not None:
            raise ValueError("Invalid/wrong responses must not contain API judgments")
        if reward.get("reward") != expected_reward:
            raise ValueError("Reward value differs from frozen deterministic formula")
    group_advantages([reward["reward"] for reward in rewards])


def apply_grpo_update(state: Any, rollouts: list[dict[str, Any]], rewards: list[dict[str, Any]]) -> dict[str, Any]:
    if not state.boundary_verified or state.completed_steps >= 100:
        raise RuntimeError("Update requires an unfinished verified optimizer boundary")
    validate_reward_transaction(rollouts, rewards)
    if {r["policy_hash"] for r in rollouts} != {state.policy_hash} or any(r["logical_step"] != state.completed_steps + 1 for r in rollouts):
        raise ValueError("Rollout policy/step differs from live optimizer state")
    step = state.completed_steps + 1
    log_path = state.run_root / "updates" / f"{step:03d}_{state.attempt_id}.json"
    _, groups = group_advantages([r["reward"] for r in rewards])
    metadata = {"step": step, "attempt_id": state.attempt_id, "question_ids": [rollouts[i]["question_id"] for i in range(0,32,4)],
                "completion_ids": [r["completion_id"] for r in rollouts], "groups": groups,
                "reward_transaction_hash": content_hash(rewards), "execution_mode": state.execution_mode,
                "correctness_counts": {status: sum(r["correctness"] == status for r in rewards) for status in ("invalid", "wrong", "correct")},
                "invalid_rate": sum(r["correctness"] == "invalid" for r in rewards) / 32,
                "reward_distribution": [r["reward"] for r in rewards],
                "opacity_scores": [r["judge"]["score"] if r.get("judge") else None for r in rewards],
                "readable_solution_flags": [r["judge"]["readable_solution_present"] if r.get("judge") else None for r in rewards],
                "padding_flags": [r["judge"]["obvious_padding_present"] if r.get("judge") else None for r in rewards],
                "repetition_indicator_version": "duplicate_token_fourgram_fraction_v1",
                "repetition_indicators": [repetition_indicator(r.get("completion_token_ids", [])) for r in rollouts],
                "response_lengths": [r.get("generated_tokens") for r in rollouts],
                "effective_opacity": [r["judge"]["opacity"] * r["judge"]["padding_gate"] if r.get("judge") else None for r in rewards]}
    atomic_write_json(log_path, {**metadata, "status": "pending"})
    state.boundary_verified = False
    try:
        metrics = state.runtime.update(rollouts, rewards)
        if metrics.get("logical_optimizer_updates") != 1:
            raise RuntimeError("Runtime did not prove exactly one logical optimizer update")
        if not math.isfinite(float(metrics.get("loss", float("nan")))):
            raise RuntimeError("Runtime produced nonfinite update metrics")
    except Exception as exc:
        atomic_write_json(log_path, {**metadata, "status": "rollback_required", "error_type": type(exc).__name__})
        if type(exc).__name__ in {"OutOfMemoryError", "CUDAOutOfMemoryError"} or "CUDA out of memory" in str(exc):
            raise RLHardwareError("GPU OOM: stop worker; retain verified checkpoint and revalidate hardware") from exc
        raise RollbackRequired("Partial/uncertain optimizer state; restore a verified full checkpoint") from exc
    state.completed_steps = step
    state.boundary_verified = True
    checkpoint = None
    try:
        if step % 10 == 0:
            checkpoint = state.save()
        atomic_write_json(log_path, {**metadata, "status": "completed", "metrics": metrics,
                                    "checkpoint": checkpoint, "policy_hash_after": state.policy_hash})
    except Exception as exc:
        # Completed live step exists, but publication failed. Do not execute an extra update.
        raise RollbackRequired("Completed update persistence failed; restore/replay verified state") from exc
    return {"status": "completed", "completed_steps": step, "metrics": metrics, "checkpoint": checkpoint}

grpo_policy_update = apply_grpo_update
