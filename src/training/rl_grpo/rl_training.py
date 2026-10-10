"""The two sequential, independent 7B/14B source-prescribed RL trajectories."""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Mapping, Sequence
from shared_procedures.identity import content_hash, file_hash
from shared_procedures.storage import atomic_write_json, read_json, fsync_directory, file_lock
from shared_procedures.checkpointing import load_checkpoint, verify_checkpoint
from shared_procedures.recovery_procedure import RecoveryManager
from .setup.rl_setup import MODELS, SNAPSHOT_STEPS, setup_rl_run, validate_rl_manifest
from .training_cycle.grpo_training_cycle import grpo_training_cycle


def verify_snapshot(directory: str | Path, *, identity: Mapping[str, Any], step: int) -> dict[str, Any]:
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Scientific snapshot directory absent or unsafe")
    manifest = read_json(root / "snapshot.json")
    supplied = manifest.get("artifact_hash")
    if supplied != content_hash({k:v for k,v in manifest.items() if k != "artifact_hash"}):
        raise ValueError("Scientific RL snapshot manifest hash mismatch")
    if manifest.get("identity") != identity or manifest.get("step") != step or step not in SNAPSHOT_STEPS:
        raise ValueError("Scientific snapshot identity/step differs from original trajectory")
    if manifest.get("status") != "complete" or not manifest.get("files"):
        raise ValueError("Scientific snapshot incomplete")
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p.name != "snapshot.json"}
    if actual != set(manifest["files"]):
        raise ValueError("Scientific snapshot file inventory differs")
    for name, info in manifest["files"].items():
        path = root / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != info["size_bytes"] or file_hash(path) != info["sha256"]:
            raise ValueError("Scientific adapter snapshot file integrity failure")
    if identity["execution_mode"] == "production" and not (root / "adapter_config.json").is_file():
        raise ValueError("Production snapshot requires a real PEFT adapter export")
    return manifest


def publish_scientific_snapshot(state: Any) -> dict[str, Any]:
    step = state.completed_steps
    if step not in SNAPSHOT_STEPS or not state.boundary_verified:
        raise ValueError("Snapshot must be an original scheduled verified optimizer boundary")
    checkpoint = verify_checkpoint(state.run_root / "checkpoints" / f"step_{step:03d}", expected_identity=state.identity)
    target = state.run_root / f"step_{step:03d}"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return verify_snapshot(target, identity=state.identity, step=step)
    with file_lock(str(target) + ".lock"):
        temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
        try:
            state.runtime.export_adapter(temporary)
            files = {str(p.relative_to(temporary)): {"sha256": file_hash(p), "size_bytes": p.stat().st_size}
                     for p in temporary.rglob("*") if p.is_file()}
            manifest = {"version": "rl_scientific_snapshot_v1", "status": "complete", "step": step,
                        "identity": state.identity, "policy_hash": state.policy_hash,
                        "full_checkpoint_hash": checkpoint["checkpoint_hash"], "files": files}
            manifest["artifact_hash"] = content_hash(manifest)
            atomic_write_json(temporary / "snapshot.json", manifest)
            verify_snapshot(temporary, identity=state.identity, step=step)
            for path in temporary.rglob("*"):
                if path.is_file():
                    with path.open("rb") as handle:
                        os.fsync(handle.fileno())
            fsync_directory(temporary)
            os.rename(temporary, target)
            fsync_directory(target.parent)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return verify_snapshot(target, identity=state.identity, step=step)


def verify_all_scientific_snapshots(state: Any) -> list[dict[str, Any]]:
    """Recover a missing export only from that exact original-step full checkpoint."""
    final_state = state.runtime.capture_state()
    final_step = state.completed_steps
    snapshots = []
    try:
        for step in SNAPSHOT_STEPS:
            root = state.run_root / f"step_{step:03d}"
            if not root.exists():
                full = load_checkpoint(state.run_root / "checkpoints" / f"step_{step:03d}", expected_identity=state.identity)
                if full["progress"]["completed_steps"] != step:
                    raise ValueError("A later policy cannot replace a missing earlier snapshot")
                state.runtime.restore_state(full)
                state.completed_steps = step
                if state.policy_hash != full["runtime"]["policy_hash"]:
                    raise ValueError("Original-step backup adapter mismatch")
                publish_scientific_snapshot(state)
            manifest = verify_snapshot(root, identity=state.identity, step=step)
            full_manifest = verify_checkpoint(state.run_root / "checkpoints" / f"step_{step:03d}", expected_identity=state.identity)
            if manifest["full_checkpoint_hash"] != full_manifest["checkpoint_hash"]:
                raise ValueError("Snapshot does not identify its exact original full checkpoint")
            snapshots.append({"step": step, "adapter_path": str(root), "artifact_hash": manifest["artifact_hash"],
                              "checkpoint_hash": full_manifest["checkpoint_hash"]})
    finally:
        state.runtime.restore_state(final_state)
        state.completed_steps = final_step
    return snapshots


def run_rl_training(manifest: Mapping[str, Any], run_root: str | Path, *, judge: Any,
                    model_revisions: Mapping[str, str], runtime_identity: Mapping[str, Any],
                    tokenizer_revisions: Mapping[str, str] | None = None,
                    model_sizes: Sequence[str] = ("7b", "14b"),
                    preflight_evidence: Mapping[str, Any] | None = None,
                    runtime_factory: Callable[..., Any] | None = None,
                    execution_mode: str = "production", stop_requested: Callable[[], bool] | None = None) -> dict[str, Any]:
    if not model_sizes or len(set(model_sizes)) != len(model_sizes) or any(size not in ("7b", "14b") for size in model_sizes):
        raise ValueError("RL model_sizes requires a nonempty unique subset of 7b/14b")
    selected_sizes = [size for size in ("7b", "14b") if size in model_sizes]
    selected_models = [MODELS[("7b", "14b").index(size)] for size in selected_sizes]
    questions = validate_rl_manifest(manifest)
    if execution_mode == "production":
        from shared_procedures.judge_client import JudgeClient
        if not isinstance(judge, JudgeClient) or not judge.live:
            raise ValueError("Production RL requires the live, budgeted provider JudgeClient")
    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    reports = []
    manager = getattr(judge, "recovery", None) or (
        RecoveryManager.for_worker(runtime_identity.get("artifact_root")) if execution_mode == "production"
        else RecoveryManager(root / "retries.json"))
    for model in selected_models:
        state = None
        try:
            model_root = root / model.rsplit("-", 1)[-1].lower()
            state = setup_rl_run(model, manifest, model_root, model_revision=model_revisions[model],
                                 runtime_identity=runtime_identity, tokenizer_revision=(tokenizer_revisions or model_revisions)[model],
                                 preflight_evidence=preflight_evidence,
                                 runtime_factory=runtime_factory, execution_mode=execution_mode)
            # Interrupted export at a scheduled saved boundary can resume before any further update.
            if state.completed_steps in SNAPSHOT_STEPS:
                publish_scientific_snapshot(state)
            trajectory = grpo_training_cycle(state, questions, judge=judge,
                        on_checkpoint=publish_scientific_snapshot, stop_requested=stop_requested, recovery_manager=manager)
            if trajectory["status"] != "completed":
                reports.append({"model_id": model, **trajectory})
                result = {"status": "paused", "runs": reports, "execution_mode": execution_mode, "model_sizes": selected_sizes}
                atomic_write_json(root / "rl_report.json", result)
                return result
            snapshots = verify_all_scientific_snapshots(state)
            reports.append({"model_id": model, **trajectory, "snapshots": snapshots,
                            "run_hash": content_hash(state.identity)})
        except Exception as exc:
            result = {"status": "partial", "runs": reports, "failed_model": model,
                      "completed_steps": state.completed_steps if state else None,
                      "error_type": type(exc).__name__, "error": str(exc), "execution_mode": execution_mode, "model_sizes": selected_sizes}
            atomic_write_json(root / "rl_report.json", result)
            return result
        finally:
            if state is not None:
                state.runtime.close()
    result = {"status": "completed", "runs": reports, "scientific_snapshots": 10 * len(selected_models),
              "planned_rollouts": 3200 * len(selected_models), "execution_mode": execution_mode,
              "required_trajectories": len(selected_models), "model_sizes": selected_sizes,
              "complete_trajectory_set": len(selected_models) == 2}
    atomic_write_json(root / "rl_report.json", result)
    return result

rl_training = run_rl_training
