"""Explicit extra passes for original published 14B ROT13 level checkpoints.

This policy is separate from the fixed level-10 additional-epoch v1 pilot.
Every condition restores its own original one-epoch adapter, AdamW and RNG;
only the declared schedule and native structural newline change. Recovery
within this phase restores its own exact schedule and exposure cursor.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import random
import re
import shutil
import time
from typing import Any, Callable, Mapping, Sequence

from shared_procedures.checkpointing import (
    _same_state, latest_checkpoint, load_checkpoint, save_checkpoint, verify_checkpoint,
)
from shared_procedures.identity import content_hash, file_hash
from shared_procedures.storage import (
    atomic_write_json, file_lock, fsync_directory, notify_artifact, read_json,
)
from training.sft.continuation import _tokenized_data
from training.sft.individual_fit.sft_fit import (
    EFFICIENT_TRAINING_BUDGET, LOSS_REDUCTION, SFTFitSpec, SFTInputError,
    _runtime, _verify_adapter, capture_rng, exposure_order, one_optimizer_update,
    restore_rng, validate_datasets, validation_loss, verify_fit_result,
)
from training.sft.published_execution import PUBLISHED_EXECUTION_POLICY

VERSION = "published_rot13_level_continuation_v2"
TRAIN_COUNT, EFFECTIVE_BATCH, UPDATES_PER_EPOCH = 10000, 16, 625


@dataclass(frozen=True)
class ContinuationSpec:
    level: int
    additional_epochs: int
    learning_rate: float

    def __post_init__(self) -> None:
        if type(self.level) is not int or not 1 <= self.level <= 10:
            raise SFTInputError("Continuation level must be an integer from 1 to 10")
        if type(self.additional_epochs) is not int or self.additional_epochs not in (1, 2):
            raise SFTInputError("Continuation supports one or two explicitly declared extra epochs")
        if (type(self.learning_rate) not in (int, float)
                or not math.isfinite(self.learning_rate) or self.learning_rate <= 0):
            raise SFTInputError("Continuation peak learning rate must be finite and positive")

    @property
    def total_steps(self) -> int:
        return UPDATES_PER_EPOCH * self.additional_epochs

    @property
    def warmup_steps(self) -> int:
        return math.ceil(self.total_steps * 0.03)

    def policy(self) -> dict:
        return {
            "version": VERSION,
            "condition": {"family": "rot13", "model_id": "14b", "level": self.level},
            "additional_epochs": self.additional_epochs,
            "additional_examples": TRAIN_COUNT * self.additional_epochs,
            "additional_optimizer_steps": self.total_steps,
            "effective_batch_size": EFFECTIVE_BATCH,
            "learning_rate_peak": self.learning_rate,
            "warmup_updates": self.warmup_steps,
            "scheduler": "single_nonzero_linear_warmup_then_cosine_added_phase_v2",
            "first_update_learning_rate": self.learning_rate / self.warmup_steps,
            "optimizer_initialization": "preserve_original_parent_adamw_moments_and_step_counters",
            "sampler": "seed42_epoch_permutations_after_original_epoch1",
            "target_text": "unchanged_parent_encoded_reasoning_and_clear_final_answer",
            "rendering": "native_training_template_think_newline_v1",
            "loss_mask": "unchanged_native_role_prefix_boundary_structural_tokens_supervised",
            "physical_execution": "preserve_parent_microbatch_accumulation_and_checkpointing",
            "checkpoint_updates": 100,
            "additional_checkpoint_boundaries": "ceil_half_epoch_at_whole_optimizer_update",
            "parent_updates": UPDATES_PER_EPOCH,
            "parent_examples": TRAIN_COUNT,
            "cumulative_final_updates": UPDATES_PER_EPOCH + self.total_steps,
            "cumulative_final_examples": TRAIN_COUNT * (1 + self.additional_epochs),
        }


def spec_from_identity(identity: Mapping) -> ContinuationSpec:
    policy = identity["continuation_policy"]
    spec = ContinuationSpec(policy["condition"]["level"], policy["additional_epochs"],
                            policy["learning_rate_peak"])
    if policy != spec.policy():
        raise SFTInputError("Continuation identity has a different declared policy")
    return spec


def added_exposure_order(spec: ContinuationSpec) -> list[int]:
    """Reproduce original seed-42 permutations without altering frozen APIs."""
    generator = random.Random(42)
    order = []
    for epoch in range(1 + spec.additional_epochs):
        indices = list(range(TRAIN_COUNT))
        generator.shuffle(indices)
        if epoch:
            order.extend(indices)
    return order


def learning_rate_scale(step: int, spec: ContinuationSpec) -> float:
    if type(step) is not int or not 0 <= step <= spec.total_steps:
        raise SFTInputError("Continuation scheduler cursor is outside the added phase")
    if step < spec.warmup_steps:
        return (step + 1) / spec.warmup_steps
    return 0.5 * (1 + math.cos(math.pi * (step - spec.warmup_steps)
                              / (spec.total_steps - spec.warmup_steps)))


def restore_optimizer_scheduler(optimizer: Any, state: Mapping, spec: ContinuationSpec,
                                *, continuing_phase: bool) -> Any:
    """Keep AdamW moments/step counts; replace only the exhausted parent schedule."""
    from torch.optim.lr_scheduler import LambdaLR
    optimizer.load_state_dict(state["optimizer"])
    if not state["optimizer"].get("state"):
        raise SFTInputError("Original parent lacks initialized AdamW moments")
    if not _same_state(optimizer.state_dict()["state"], state["optimizer"]["state"]):
        raise SFTInputError("Restored AdamW moments or step counters changed")
    saved_lrs = [group["lr"] for group in optimizer.param_groups]
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] = spec.learning_rate
    scheduler = LambdaLR(optimizer, lambda step: learning_rate_scale(step, spec))
    if continuing_phase:
        scheduler.load_state_dict(state["scheduler"])
        for group, saved_lr in zip(optimizer.param_groups, saved_lrs, strict=True):
            group["lr"] = saved_lr
        step = state["progress"]["completed_steps"]
        expected_lr = spec.learning_rate * learning_rate_scale(step, spec)
        if (scheduler.last_epoch != step
                or any(base != spec.learning_rate for base in scheduler.base_lrs)
                or any(not math.isclose(group["lr"], expected_lr, rel_tol=1e-12, abs_tol=1e-15)
                       for group in optimizer.param_groups)):
            raise SFTInputError("Continuation scheduler/LR differs from its completed-update cursor")
    return scheduler


def prepare_inputs(parent_fit: str | Path, parent_checkpoint: str | Path,
                   train_records: Sequence[Mapping], validation_records: Sequence[Mapping],
                   source_revision: str, *, continuation: ContinuationSpec,
                   cache_root: str | Path) -> tuple[dict, dict, list, list]:
    """Bind this level's original complete state and its unchanged ordered data."""
    if not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise SFTInputError("Continuation execution requires an immutable source commit")
    parent_fit, parent_checkpoint = Path(parent_fit).resolve(), Path(parent_checkpoint).resolve()
    result = read_json(parent_fit / "fit_result.json")
    original = result.get("identity", {})
    parent_spec = SFTFitSpec.from_dict(original)
    if ({"family": parent_spec.family, "model_id": parent_spec.model_id, "level": parent_spec.level}
            != continuation.policy()["condition"]
            or parent_spec.training_budget != EFFICIENT_TRAINING_BUDGET
            or original.get("run_id") != parent_spec.run_id
            or original.get("implementation_version") != parent_spec.identity()["implementation_version"]
            or "continuation_policy" in original or "parent" in original
            or original.get("published_execution_policy") != PUBLISHED_EXECUTION_POLICY):
        raise SFTInputError("Continuation requires this level's original published one-epoch parent")
    verify_fit_result(parent_fit, original)
    manifest = verify_checkpoint(parent_checkpoint, original)
    adapter = _verify_adapter(parent_fit / "final_adapter", original)
    progress = manifest["progress"]
    if (manifest["checkpoint_hash"] != result.get("checkpoint_hash")
            or any(progress.get(key) != expected for key, expected in {
                "completed_steps": UPDATES_PER_EPOCH, "total_optimizer_steps": UPDATES_PER_EPOCH,
                "sampler_position": TRAIN_COUNT, "example_presentations": TRAIN_COUNT,
                "exposure_order_hash": content_hash(exposure_order(TRAIN_COUNT, 42, epochs=1)),
            }.items())
            or {key: value for key, value in result["progress"].items() if key != "status"} != progress):
        raise SFTInputError("Parent is not the verified original 625-update full checkpoint")
    data = validate_datasets(parent_spec, train_records, validation_records)
    derived, train, validation = _tokenized_data(parent_spec, train_records, validation_records, cache_root)
    data = {**data, "derived_tokenization": derived}
    parent = {
        "fit_result_sha256": file_hash(parent_fit / "fit_result.json"),
        "identity": original, "identity_hash": content_hash(original),
        "checkpoint_hash": manifest["checkpoint_hash"],
        "checkpoint_payload_sha256": manifest["files"]["full_state.pt"]["sha256"],
        "adapter_manifest_hash": adapter["manifest_hash"],
    }
    order_hash = content_hash(added_exposure_order(continuation))
    config = {"policy": continuation.policy(), "parent": parent,
              "source_code_commit": source_revision, "dataset_hash": data["dataset_hash"],
              "derived_tokenization": derived, "exposure_order_hash": order_hash}
    digest = content_hash(config)
    identity = {
        **original, "implementation_version": VERSION,
        "training_budget": "efficient_level_continuation_v2",
        "run_id": (f"rot13_14b_level_{continuation.level:02d}_continuation_"
                   f"total_epochs_{1 + continuation.additional_epochs:02d}_{digest[:12]}"),
        "configuration_hash": digest, "code_revision": source_revision,
        "parent": parent, "continuation_policy": continuation.policy(),
        "exposure_order_hash": order_hash, "derived_tokenization": derived,
    }
    return identity, data, train, validation


def progress_record(completed: int, processed_tokens: int, identity: Mapping, runtime: Mapping) -> dict:
    spec = spec_from_identity(identity)
    return {
        "completed_steps": completed, "sampler_position": completed * EFFECTIVE_BATCH,
        "example_presentations": completed * EFFECTIVE_BATCH,
        "total_optimizer_steps": spec.total_steps, "processed_tokens": processed_tokens,
        "exposure_order_hash": identity["exposure_order_hash"],
        "cumulative_completed_steps": UPDATES_PER_EPOCH + completed,
        "cumulative_example_presentations": TRAIN_COUNT + completed * EFFECTIVE_BATCH,
        "epoch_index": 2 + min(completed // UPDATES_PER_EPOCH, spec.additional_epochs - 1),
        "completed_additional_epochs": completed // UPDATES_PER_EPOCH,
        "execution_profile": None, "published_execution_policy": PUBLISHED_EXECUTION_POLICY,
        **{key: runtime[key] for key in
           ("physical_microbatch", "gradient_accumulation", "gradient_checkpointing")},
    }


def checkpoint_steps(spec: ContinuationSpec) -> set[int]:
    ordinary = set(range(100, spec.total_steps + 1, 100))
    half_epochs = {math.ceil(TRAIN_COUNT * half / (2 * EFFECTIVE_BATCH))
                   for half in range(1, 2 * spec.additional_epochs + 1)}
    return ordinary | half_epochs | {spec.total_steps}


def train_added_epochs(model: Any, optimizer: Any, scheduler: Any,
                       examples: Sequence[Mapping], pad_token_id: int, *,
                       identity: Mapping, runtime: Mapping, recipe: Any,
                       progress: Mapping | None = None, save: Callable,
                       should_pause: Callable[[], bool] | None = None,
                       on_progress: Callable[[dict], None] | None = None) -> dict:
    spec = spec_from_identity(identity)
    if len(examples) != TRAIN_COUNT:
        raise SFTInputError("Continuation requires the unchanged 10000-example corpus")
    order = added_exposure_order(spec)
    if content_hash(order) != identity["exposure_order_hash"]:
        raise SFTInputError("Continuation exposure order changed")
    completed, processed = (progress or {}).get("completed_steps", 0), (progress or {}).get("processed_tokens", 0)
    if (type(completed) is not int or not 0 <= completed <= spec.total_steps
            or type(processed) is not int or processed < 0):
        raise SFTInputError("Invalid continuation update/token cursor")
    if progress and dict(progress) != progress_record(completed, processed, identity, runtime):
        raise SFTInputError("Continuation recovery sampler/exposure/runtime cursor changed")
    if scheduler.last_epoch != completed:
        raise SFTInputError("Continuation scheduler cursor differs from completed updates")
    model.train()
    boundaries = checkpoint_steps(spec)
    for step in range(completed, spec.total_steps):
        if should_pause and should_pause():
            cursor = progress_record(step, processed, identity, runtime)
            save(cursor)
            return {**cursor, "status": "paused"}
        indices = order[step * EFFECTIVE_BATCH:(step + 1) * EFFECTIVE_BATCH]
        one_optimizer_update(model, optimizer, scheduler, [examples[index] for index in indices],
                             pad_token_id, recipe=recipe, runtime=runtime)
        processed += sum(len(examples[index]["input_ids"]) for index in indices)
        cursor = progress_record(step + 1, processed, identity, runtime)
        if on_progress:
            on_progress(cursor)
        pause = bool(should_pause and should_pause())
        if step + 1 in boundaries or pause:
            save(cursor)
        if pause:
            return {**cursor, "status": "paused"}
    return {**progress_record(spec.total_steps, processed, identity, runtime), "status": "complete"}


def final_checkpoint(output: Path, identity: Mapping) -> Path:
    return output / f"checkpoint_{spec_from_identity(identity).total_steps:06d}"


def _export_adapter(output: Path, model: Any, tokenizer: Any, identity: Mapping) -> dict:
    checkpoint = verify_checkpoint(final_checkpoint(output, identity), dict(identity))
    destination = output / "final_adapter"
    if destination.exists():
        manifest = _verify_adapter(destination, identity)
        if manifest["checkpoint_hash"] != checkpoint["checkpoint_hash"]:
            raise SFTInputError("Continuation adapter binds a different final full checkpoint")
        return manifest
    staging = output / ".final_adapter_staging"
    if staging.exists():
        shutil.rmtree(staging)
    model.save_pretrained(staging, safe_serialization=True, save_embedding_layers=False)
    tokenizer.save_pretrained(staging)
    files = {str(path.relative_to(staging)): file_hash(path) for path in staging.rglob("*") if path.is_file()}
    manifest = {"identity": dict(identity), "files": files, "checkpoint_hash": checkpoint["checkpoint_hash"]}
    manifest["manifest_hash"] = content_hash(manifest)
    atomic_write_json(staging / "manifest.json", manifest)
    _verify_adapter(staging, identity)
    for path in staging.rglob("*"):
        if path.is_file():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
    fsync_directory(staging)
    staging.rename(destination)
    fsync_directory(output)
    return _verify_adapter(destination, identity)


def verified_completion(output: Path, identity: Mapping) -> dict:
    """Verify actual final full state/export; idempotent re-publication needs no GPU."""
    from safetensors.torch import load_file
    import torch
    spec = spec_from_identity(identity)
    result = read_json(output / "fit_result.json")
    manifest = _verify_adapter(output / "final_adapter", identity)
    checkpoint = verify_checkpoint(final_checkpoint(output, identity), dict(identity))
    recovered = load_checkpoint(final_checkpoint(output, identity), dict(identity))
    cursor = {key: value for key, value in result.get("progress", {}).items() if key != "status"}
    if (result.get("status") != "complete" or result.get("identity") != dict(identity)
            or type(cursor.get("processed_tokens")) is not int or cursor["processed_tokens"] < 0
            or cursor != checkpoint["progress"]
            or cursor != progress_record(spec.total_steps, cursor.get("processed_tokens"), identity, recovered["runtime"])
            or result.get("checkpoint_hash") != checkpoint["checkpoint_hash"]
            or manifest.get("checkpoint_hash") != checkpoint["checkpoint_hash"]
            or result.get("adapter_manifest_hash") != manifest["manifest_hash"]
            or result.get("adapter_files") != manifest["files"]
            or result.get("validation_count") != 100
            or type(result.get("validation_loss")) not in (float, int)
            or not math.isfinite(result["validation_loss"])
            or recovered["runtime"].get("published_execution_policy") != PUBLISHED_EXECUTION_POLICY
            or recovered["runtime"].get("continuation_policy") != spec.policy()
            or recovered["runtime"].get("parent_checkpoint_hash") != identity["parent"]["checkpoint_hash"]):
        raise SFTInputError("Existing continuation completion is invalid; preserving its bytes")
    weights = load_file(str(output / "final_adapter" / "adapter_model.safetensors"), device="cpu")
    if (not all(torch.isfinite(tensor).all() for tensor in weights.values())
            or not _same_state(weights, recovered["adapter"])):
        raise SFTInputError("Continuation export differs from completed full-state tensors")
    return result


def publish_completion(output: Path, identity: Mapping) -> None:
    checkpoint = final_checkpoint(output, identity)
    notify_artifact(checkpoint, verify_checkpoint(checkpoint, dict(identity)))
    for path in sorted((output / "final_adapter").rglob("*")):
        if path.is_file():
            notify_artifact(path, {"kind": "sft_level_continuation_final_adapter"})
    notify_artifact(output / "fit_result.json", {"kind": "sft_level_continuation_completion"})


def validate_output_directory(output_dir: str | Path, parent_fit: str | Path,
                              parent_checkpoint: str | Path) -> Path:
    """Reject overlapping namespaces before even operational files are created."""
    output = Path(output_dir).resolve()
    parents = (Path(parent_fit).resolve(), Path(parent_checkpoint).resolve())
    if any(output == parent or output.is_relative_to(parent) or parent.is_relative_to(output)
           for parent in parents):
        raise SFTInputError("Continuation output must be separate from immutable parent artifacts")
    return output


def run_continuation(parent_fit: str | Path, parent_checkpoint: str | Path,
                     train_records: Sequence[Mapping], validation_records: Sequence[Mapping],
                     output_dir: str | Path, *, source_revision: str,
                     continuation: ContinuationSpec, cache_root: str | Path,
                     device: str = "cuda:0", should_pause: Callable[[], bool] | None = None) -> dict:
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    output = validate_output_directory(output_dir, parent_fit, parent_checkpoint)
    parent_fit, parent_checkpoint = Path(parent_fit).resolve(), Path(parent_checkpoint).resolve()
    identity, data, train, validation = prepare_inputs(parent_fit, parent_checkpoint, train_records,
        validation_records, source_revision, continuation=continuation, cache_root=cache_root)
    output.mkdir(parents=True, exist_ok=True)
    with file_lock(output / ".continuation.lock", timeout=1):
        identity_path = output / "run_identity.json"
        if identity_path.exists() and read_json(identity_path) != identity:
            raise SFTInputError("Continuation namespace already has a different exact identity")
        atomic_write_json(identity_path, identity)
        notify_artifact(identity_path, {"kind": "sft_level_continuation_identity"})
        if (output / "fit_result.json").exists():
            result = verified_completion(output, identity)
            publish_completion(output, identity)
            atomic_write_json(output / "fit_status.json", {"status": "complete", "run_id": identity["run_id"],
                "identity_hash": content_hash(identity), "progress": result["progress"], "status_storage": "LOCAL_ONLY"})
            return result
        checkpoint = latest_checkpoint(output, identity)
        state = load_checkpoint(checkpoint, identity) if checkpoint else load_checkpoint(
            parent_checkpoint, identity["parent"]["identity"])
        if (state.get("runtime", {}).get("published_execution_policy") != PUBLISHED_EXECUTION_POLICY
                or checkpoint and (state["runtime"].get("continuation_policy") != continuation.policy()
                    or state["runtime"].get("parent_checkpoint_hash") != identity["parent"]["checkpoint_hash"])):
            raise SFTInputError("Full parent/continuation runtime policy differs")
        parent_spec = SFTFitSpec.from_dict(identity["parent"]["identity"])
        model, tokenizer, optimizer, unused_scheduler, runtime = _runtime(
            parent_spec, device, cache_root=cache_root, train_count=TRAIN_COUNT)
        del unused_scheduler
        set_peft_model_state_dict(model, state["adapter"])
        if not _same_state(get_peft_model_state_dict(model, save_embedding_layers=False), state["adapter"]):
            raise SFTInputError("Restored adapter differs from its verified checkpoint")
        scheduler = restore_optimizer_scheduler(optimizer, state, continuation, continuing_phase=checkpoint is not None)
        micro, accumulation, checkpointing = (state["runtime"].get(key) for key in
            ("physical_microbatch", "gradient_accumulation", "gradient_checkpointing"))
        if (type(micro) is not int or type(accumulation) is not int
                or (micro, accumulation) not in ((1, 16), (2, 8)) or type(checkpointing) is not bool):
            raise SFTInputError("Parent execution mode is not fully recorded")
        from training.sft.published_execution import capacity_setup
        capacity = capacity_setup(model, max(train, key=lambda example: len(example["input_ids"])),
            tokenizer.pad_token_id, optimizer=optimizer, fixed_microbatch_size=micro,
            fixed_gradient_checkpointing=checkpointing)
        runtime.update({key: capacity[key] for key in
            ("physical_microbatch", "gradient_accumulation", "gradient_checkpointing")})
        runtime.update({"continuation_policy": continuation.policy(),
            "parent_checkpoint_hash": identity["parent"]["checkpoint_hash"],
            "optimizer_moments_and_steps_restored": True,
            "warmup_steps": continuation.warmup_steps, "total_optimizer_steps": continuation.total_steps,
            "training_budget": identity["training_budget"]})
        restore_rng(state["rng"])
        progress = state["progress"] if checkpoint else progress_record(0, 0, identity, runtime)
        status_path = output / "fit_status.json"
        def status(cursor: dict, status_value: str = "running") -> None:
            atomic_write_json(status_path, {"status": status_value, "run_id": identity["run_id"],
                "identity_hash": content_hash(identity), "progress": cursor,
                "learning_rates": scheduler.get_last_lr(), "updated_unix": time.time(), "status_storage": "LOCAL_ONLY"})
        def save(cursor: dict) -> None:
            save_checkpoint(output / f"checkpoint_{cursor['completed_steps']:06d}", identity,
                {"adapter": get_peft_model_state_dict(model, save_embedding_layers=False),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "rng": capture_rng(), "progress": cursor, "runtime": runtime})
            status(cursor)
        if not checkpoint:
            save(progress)
        start = time.monotonic()
        try:
            final = train_added_epochs(model, optimizer, scheduler, train, tokenizer.pad_token_id,
                identity=identity, runtime=runtime, recipe=parent_spec.recipe, progress=progress,
                save=save, should_pause=should_pause, on_progress=status)
            if final["status"] == "paused":
                status({key: value for key, value in final.items() if key != "status"}, "paused")
                return {"status": "paused", "identity": identity, "progress": final}
            loss = validation_loss(model, validation, tokenizer.pad_token_id)
            manifest = _export_adapter(output, model, tokenizer, identity)
            result = {"status": "complete", "identity": identity, "run_id": identity["run_id"],
                "progress": final, "runtime": runtime, "data": data, "validation_loss": loss,
                "validation_count": 100, "loss_reduction": LOSS_REDUCTION,
                "adapter_path": str(output / "final_adapter"), "adapter_files": manifest["files"],
                "adapter_manifest_hash": manifest["manifest_hash"], "checkpoint_hash": manifest["checkpoint_hash"],
                "elapsed_training_seconds_this_invocation": time.monotonic() - start}
            atomic_write_json(output / "fit_result.json", result)
            status({key: value for key, value in final.items() if key != "status"}, "publishing")
            verified_completion(output, identity)
            publish_completion(output, identity)
            status({key: value for key, value in final.items() if key != "status"}, "complete")
            return result
        except Exception as error:
            atomic_write_json(status_path, {"status": "failed", "error_type": type(error).__name__,
                "detail": str(error), "recovery": "resume_only_verified_level_continuation_full_checkpoint"})
            raise
