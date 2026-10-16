"""A separately identified second epoch from a completed published SFT checkpoint.

The original one-epoch fit is immutable. Adapter weights, AdamW moments/step
counters and RNG are restored; only the exhausted learning-rate schedule gets
an explicitly declared restart. Recovery inside this phase restores its own
scheduler and second-epoch sampler cursor exactly.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re
import shutil
import time
from typing import Any, Callable, Mapping, Sequence

from shared_procedures.checkpointing import (
    _same_state, latest_checkpoint, load_checkpoint, save_checkpoint, verify_checkpoint,
)
from shared_procedures.identity import canonical_json, content_hash, file_hash
from shared_procedures.storage import (
    atomic_write_json, file_lock, fsync_directory, notify_artifact, read_json,
)
from training.sft.individual_fit.sft_fit import (
    EFFICIENT_TRAINING_BUDGET, LOSS_REDUCTION, SFTFitSpec, SFTInputError,
    _runtime, _verify_adapter, capture_rng, exposure_order, one_optimizer_update,
    restore_rng, render_sft_completion, tokenize_completion, validate_datasets, validation_loss,
)
from training.sft.published_execution import PUBLISHED_EXECUTION_POLICY

POLICY = {
    "version": "published_sft_additional_epoch_v1",
    "condition": {"family": "rot13", "model_id": "14b", "level": 10},
    "additional_epochs": 1, "additional_examples": 10000,
    "additional_optimizer_steps": 625, "effective_batch_size": 16,
    "learning_rate_peak": 2e-4, "warmup_updates": 19,
    "scheduler": "nonzero_linear_warmup_then_cosine_restart_v1",
    "first_update_learning_rate": 2e-4 / 19,
    "optimizer_initialization": "preserve_parent_adamw_moments_and_step_counters",
    "sampler": "second_seed42_epoch_permutation",
    "target_text": "unchanged_parent_encoded_reasoning_and_clear_final_answer",
    "rendering": "native_training_template_think_newline_v1",
    "loss_mask": "unchanged_native_role_prefix_boundary_structural_tokens_supervised",
    "physical_execution": "preserve_parent_microbatch_accumulation_and_checkpointing",
    "parent_updates": 625, "cumulative_final_updates": 1250,
    "parent_examples": 10000, "cumulative_final_examples": 20000,
}


def learning_rate_scale(step: int) -> float:
    """LR used after ``step`` completed updates, including a nonzero first LR."""
    if type(step) is not int or not 0 <= step <= 625:
        raise SFTInputError("Continuation scheduler step must lie in [0,625]")
    if step < 19:
        return (step + 1) / 19
    return 0.5 * (1.0 + math.cos(math.pi * (step - 19) / (625 - 19)))


def restart_scheduler(optimizer: Any) -> Any:
    """Reset LR/scheduler only; never clear or replace AdamW state."""
    from torch.optim.lr_scheduler import LambdaLR
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] = POLICY["learning_rate_peak"]
    return LambdaLR(optimizer, learning_rate_scale)


def restore_optimizer_scheduler(optimizer: Any, state: Mapping[str, Any], *,
                                continuing_phase: bool) -> Any:
    """Preserve initialized AdamW state, replacing only the parent LR schedule."""
    optimizer.load_state_dict(state["optimizer"])
    if not state["optimizer"].get("state"):
        raise SFTInputError("Completed parent checkpoint lacks initialized AdamW moments")
    if not _same_state(optimizer.state_dict()["state"], state["optimizer"]["state"]):
        raise SFTInputError("Restored AdamW moments or step counters changed")
    saved_lrs = [group["lr"] for group in optimizer.param_groups]
    scheduler = restart_scheduler(optimizer)
    if continuing_phase:
        scheduler.load_state_dict(state["scheduler"])
        for group, lr in zip(optimizer.param_groups, saved_lrs, strict=True):
            group["lr"] = lr
        expected_step = state["progress"]["completed_steps"]
        expected_lr = POLICY["learning_rate_peak"] * learning_rate_scale(expected_step)
        if (scheduler.last_epoch != expected_step
                or any(not math.isclose(group["lr"], expected_lr, rel_tol=1e-12, abs_tol=1e-15)
                       for group in optimizer.param_groups)):
            raise SFTInputError("Continuation scheduler/LR differs from its completed-update cursor")
    return scheduler


def restore_training_state(model: Any, optimizer: Any, state: Mapping[str, Any], *,
                           continuing_phase: bool) -> Any:
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    set_peft_model_state_dict(model, state["adapter"])
    if not _same_state(get_peft_model_state_dict(model, save_embedding_layers=False), state["adapter"]):
        raise SFTInputError("Restored continuation adapter differs from its verified checkpoint")
    scheduler = restore_optimizer_scheduler(optimizer, state, continuing_phase=continuing_phase)
    restore_rng(state["rng"])
    return scheduler


def tokenize_continuation(record: Mapping, tokenizer: Any) -> dict:
    """Insert only the native structural newline; preserve reasoning/final bytes."""
    if record["completion"] != "<think>" + record["reasoning"] + "</think>" + record["final_answer"]:
        raise SFTInputError("Parent target structure differs")
    derived = {**record, "completion": "<think>\n" + record["reasoning"] + "</think>" + record["final_answer"]}
    derived.pop("transformed_token_count", None)  # old renderer count is not relabeled
    rendered, _boundary = render_sft_completion(derived["prompt"], derived["completion"], tokenizer)
    native = tokenizer.apply_chat_template([{"role": "user", "content": record["prompt"]}],
                                           tokenize=False, add_generation_prompt=True)
    if not rendered.startswith(native) or not native.endswith("<think>\n"):
        raise SFTInputError("Derived teacher-forced reasoning prefix differs from native inference")
    example = tokenize_completion(derived, tokenizer, max_length=4096)
    native_ids = tokenizer.encode(native, add_special_tokens=False)
    if example["input_ids"][:len(native_ids)] != native_ids:
        raise SFTInputError("Derived training tokens do not retain the exact native inference prefill")
    return example


def _tokenized_data(spec: SFTFitSpec, train_records: Sequence[Mapping],
                    validation_records: Sequence[Mapping], cache_root: str | Path) -> tuple:
    from transformers import AutoTokenizer
    from operations.assets import model_load_kwargs
    tokenizer = AutoTokenizer.from_pretrained(spec.model_name, revision=spec.tokenizer_revision,
        use_fast=True, trust_remote_code=False,
        **model_load_kwargs(spec.model_name, spec.tokenizer_revision, cache_root,
                            tokenizer_only=True, offline=True))
    summary, outputs = {"policy": POLICY["rendering"], "loss_mask": POLICY["loss_mask"],
                        "policy_hash": content_hash({k: POLICY[k] for k in
                            ("rendering", "loss_mask", "target_text")})}, []
    for split, records in (("train", train_records), ("validation", validation_records)):
        examples = [tokenize_continuation(record, tokenizer) for record in records]
        digest = hashlib.sha256()
        for example in examples:
            digest.update(canonical_json(example) + b"\n")
        lengths = [len(e["input_ids"]) for e in examples]
        summary[split] = {"count": len(examples), "tokenized_records_sha256": digest.hexdigest(),
                          "total_tokens": sum(lengths), "min_tokens": min(lengths),
                          "max_tokens": max(lengths), "sequence_truncation": False,
                          "exact_native_token_prefill_matches": len(examples)}
        outputs.append(examples)
    return summary, outputs[0], outputs[1]


def prepare_inputs(parent_fit: str | Path, parent_checkpoint: str | Path,
                   train_records: Sequence[Mapping], validation_records: Sequence[Mapping],
                   source_revision: str, *, cache_root: str | Path) -> tuple[dict, dict, list, list]:
    """Bind actual completed parent, full-state checksum and unchanged corpus."""
    if not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise SFTInputError("Continuation execution requires an immutable source commit")
    parent_fit, parent_checkpoint = Path(parent_fit).resolve(), Path(parent_checkpoint).resolve()
    result = read_json(parent_fit / "fit_result.json")
    original = result.get("identity", {})
    spec = SFTFitSpec.from_dict(original)
    if ({"family": spec.family, "model_id": spec.model_id, "level": spec.level} != POLICY["condition"]
            or spec.training_budget != EFFICIENT_TRAINING_BUDGET
            or original.get("recipe") != spec.recipe.to_dict()
            or original.get("published_execution_policy") != PUBLISHED_EXECUTION_POLICY
            or result.get("status") != "complete"):
        raise SFTInputError("Continuation requires the completed published 14B ROT13 level10 parent")
    adapter = _verify_adapter(parent_fit / "final_adapter", original)
    manifest = verify_checkpoint(parent_checkpoint, original)
    progress = manifest["progress"]
    if (manifest["checkpoint_hash"] != result.get("checkpoint_hash")
            or manifest["checkpoint_hash"] != adapter.get("checkpoint_hash")
            or adapter["manifest_hash"] != result.get("adapter_manifest_hash")
            or result.get("adapter_files") != adapter["files"]
            or any(progress.get(k) != v for k, v in {
                "completed_steps": 625, "total_optimizer_steps": 625,
                "sampler_position": 10000, "example_presentations": 10000,
                "exposure_order_hash": content_hash(exposure_order(10000, 42, epochs=1)),
            }.items())
            or {k: v for k, v in result.get("progress", {}).items() if k != "status"} != progress):
        raise SFTInputError("Parent is not the original verified completed625-update full checkpoint")
    data = validate_datasets(spec, train_records, validation_records)
    derived, train, validation = _tokenized_data(spec, train_records, validation_records, cache_root)
    data = {**data, "derived_tokenization": derived}
    parent = {"fit_result_sha256": file_hash(parent_fit / "fit_result.json"),
              "identity": original, "identity_hash": content_hash(original),
              "checkpoint_hash": manifest["checkpoint_hash"],
              "checkpoint_payload_sha256": manifest["files"]["full_state.pt"]["sha256"],
              "adapter_manifest_hash": adapter["manifest_hash"]}
    order = exposure_order(10000, 42, epochs=2)[10000:]
    config = {"policy": POLICY, "parent": parent, "source_code_commit": source_revision,
              "dataset_hash": data["dataset_hash"], "derived_tokenization": derived,
              "exposure_order_hash": content_hash(order)}
    identity = {**original, "implementation_version": POLICY["version"],
                "training_budget": "efficient_additional_epoch_v1",
                "run_id": "rot13_14b_level_10_continuation_epoch_02",
                "configuration_hash": content_hash(config), "code_revision": source_revision,
                "parent": parent, "continuation_policy": POLICY,
                "exposure_order_hash": content_hash(order), "derived_tokenization": derived}
    return identity, data, train, validation


def _progress(completed: int, processed_tokens: int, identity: Mapping, runtime: Mapping) -> dict:
    return {"completed_steps": completed, "sampler_position": completed * 16,
            "example_presentations": completed * 16, "total_optimizer_steps": 625,
            "processed_tokens": processed_tokens, "exposure_order_hash": identity["exposure_order_hash"],
            "cumulative_completed_steps": 625 + completed,
            "cumulative_example_presentations": 10000 + completed * 16,
            "epoch_index": 2, "execution_profile": None,
            "published_execution_policy": PUBLISHED_EXECUTION_POLICY,
            **{key: runtime[key] for key in
               ("physical_microbatch", "gradient_accumulation", "gradient_checkpointing")}}


def train_additional_epoch(model: Any, optimizer: Any, scheduler: Any,
                           examples: Sequence[Mapping], pad_token_id: int, *,
                           identity: Mapping, runtime: Mapping, recipe: Any,
                           progress: Mapping | None = None, save: Callable,
                           should_pause: Callable[[], bool] | None = None,
                           on_progress: Callable[[dict], None] | None = None) -> dict:
    """Only epoch2 exposures; resume only after a whole completed update."""
    if len(examples) != 10000:
        raise SFTInputError("Continuation requires the unchanged10000-example corpus")
    order = exposure_order(10000, 42, epochs=2)[10000:]
    if content_hash(order) != identity["exposure_order_hash"]:
        raise SFTInputError("Continuation second-epoch order changed")
    completed = (progress or {}).get("completed_steps", 0)
    processed = (progress or {}).get("processed_tokens", 0)
    if type(completed) is not int or not 0 <= completed <= 625:
        raise SFTInputError("Invalid continuation update cursor")
    if progress and dict(progress) != _progress(completed, processed, identity, runtime):
        raise SFTInputError("Continuation recovery sampler/exposure/runtime cursor changed")
    if scheduler.last_epoch != completed:
        raise SFTInputError("Continuation scheduler cursor differs from completed updates")
    model.train()
    for step in range(completed, 625):
        if should_pause and should_pause():
            state = _progress(step, processed, identity, runtime)
            save(state)
            return {**state, "status": "paused"}
        indices = order[step * 16:(step + 1) * 16]
        one_optimizer_update(model, optimizer, scheduler, [examples[i] for i in indices],
                             pad_token_id, recipe=recipe, runtime=runtime)
        processed += sum(len(examples[i]["input_ids"]) for i in indices)
        state = _progress(step + 1, processed, identity, runtime)
        if on_progress:
            on_progress(state)
        pause = bool(should_pause and should_pause())
        if (step + 1) % 100 == 0 or step + 1 == 625 or pause:
            save(state)
        if pause:
            return {**state, "status": "paused"}
    return {**_progress(625, processed, identity, runtime), "status": "complete"}


def _export_adapter(output: Path, model: Any, tokenizer: Any, identity: Mapping) -> dict:
    """Publish the continuation export bound to its own final full state."""
    destination = output / "final_adapter"
    checkpoint = verify_checkpoint(output / "checkpoint_000625", dict(identity))
    if destination.exists():
        manifest = _verify_adapter(destination, identity)
        if manifest["checkpoint_hash"] != checkpoint["checkpoint_hash"]:
            raise SFTInputError("Continuation adapter binds a different full checkpoint")
        return manifest
    staging = output / ".final_adapter_staging"
    if staging.exists():
        shutil.rmtree(staging)
    model.save_pretrained(staging, safe_serialization=True, save_embedding_layers=False)
    tokenizer.save_pretrained(staging)
    files = {str(p.relative_to(staging)): file_hash(p) for p in staging.rglob("*") if p.is_file()}
    manifest = {"identity": dict(identity), "files": files,
                "checkpoint_hash": checkpoint["checkpoint_hash"]}
    manifest["manifest_hash"] = content_hash(manifest)
    atomic_write_json(staging / "manifest.json", manifest)
    _verify_adapter(staging, identity)
    for path in staging.rglob("*"):
        if path.is_file():
            with path.open("rb") as stream:
                import os
                os.fsync(stream.fileno())
    fsync_directory(staging)
    staging.rename(destination)
    fsync_directory(output)
    return _verify_adapter(destination, identity)


def verified_completion(output: Path, identity: Mapping) -> dict:
    """Reuse immutable completed bytes after a publication failure, without GPU."""
    from safetensors.torch import load_file
    result = read_json(output / "fit_result.json")
    checkpoint = verify_checkpoint(output / "checkpoint_000625", dict(identity))
    manifest = _verify_adapter(output / "final_adapter", identity)
    cursor = {k: v for k, v in result.get("progress", {}).items() if k != "status"}
    if (result.get("status") != "complete" or result.get("identity") != dict(identity)
            or cursor != checkpoint["progress"]
            or cursor.get("completed_steps") != 625 or cursor.get("sampler_position") != 10000
            or cursor.get("cumulative_completed_steps") != 1250
            or result.get("checkpoint_hash") != checkpoint["checkpoint_hash"]
            or manifest.get("checkpoint_hash") != checkpoint["checkpoint_hash"]
            or result.get("adapter_manifest_hash") != manifest["manifest_hash"]
            or result.get("adapter_files") != manifest["files"]
            or result.get("validation_count") != 100
            or not isinstance(result.get("validation_loss"), (float, int))
            or not math.isfinite(result["validation_loss"])):
        raise SFTInputError("Existing continuation completion is invalid; preserving its bytes")
    recovered = load_checkpoint(output / "checkpoint_000625", dict(identity))
    weights = load_file(str(output / "final_adapter" / "adapter_model.safetensors"), device="cpu")
    import torch
    if not all(torch.isfinite(tensor).all() for tensor in weights.values()) or not _same_state(weights, recovered["adapter"]):
        raise SFTInputError("Continuation export differs from completed full-state tensors")
    return result


def publish_completion(output: Path, identity: Mapping) -> None:
    manifest = verify_checkpoint(output / "checkpoint_000625", dict(identity))
    notify_artifact(output / "checkpoint_000625", manifest)
    for path in sorted((output / "final_adapter").rglob("*")):
        if path.is_file():
            notify_artifact(path, {"kind": "sft_continuation_final_adapter"})
    notify_artifact(output / "fit_result.json", {"kind": "sft_continuation_completion"})


def run_continuation(parent_fit: str | Path, parent_checkpoint: str | Path,
                     train_records: Sequence[Mapping], validation_records: Sequence[Mapping],
                     output_dir: str | Path, *, source_revision: str,
                     cache_root: str | Path, device: str = "cuda:0",
                     should_pause: Callable[[], bool] | None = None) -> dict:
    """Run/recover a separately named extra epoch, preserving all parent files."""
    from peft import get_peft_model_state_dict
    output, parent_fit, parent_checkpoint = (Path(p).resolve() for p in
                                           (output_dir, parent_fit, parent_checkpoint))
    if (output == parent_fit or output.is_relative_to(parent_fit) or parent_fit.is_relative_to(output)
            or output == parent_checkpoint or parent_checkpoint.is_relative_to(output)
            or output.is_relative_to(parent_checkpoint)):
        raise SFTInputError("Continuation output must be separate from immutable parent artifacts")
    identity, data, train, validation = prepare_inputs(parent_fit, parent_checkpoint, train_records,
                                    validation_records, source_revision, cache_root=cache_root)
    output.mkdir(parents=True, exist_ok=True)
    with file_lock(output / ".continuation.lock", timeout=1):
        identity_path = output / "run_identity.json"
        if identity_path.exists() and read_json(identity_path) != identity:
            raise SFTInputError("Continuation namespace already has a different exact identity")
        atomic_write_json(identity_path, identity)
        notify_artifact(identity_path, {"kind": "sft_continuation_identity"})
        if (output / "fit_result.json").exists():
            result = verified_completion(output, identity)
            publish_completion(output, identity)
            atomic_write_json(output / "fit_status.json", {"status": "complete",
                "run_id": identity["run_id"], "identity_hash": content_hash(identity),
                "progress": result["progress"], "status_storage": "LOCAL_ONLY"})
            return result
        checkpoint = latest_checkpoint(output, identity)
        state = load_checkpoint(checkpoint, identity) if checkpoint else load_checkpoint(
            parent_checkpoint, identity["parent"]["identity"])
        if state.get("runtime", {}).get("published_execution_policy") != PUBLISHED_EXECUTION_POLICY:
            raise SFTInputError("Full parent/continuation runtime policy differs")
        spec = SFTFitSpec.from_dict(identity["parent"]["identity"])
        model, tokenizer, optimizer, _unused_scheduler, runtime = _runtime(
            spec, device, cache_root=cache_root, train_count=10000)
        del _unused_scheduler
        scheduler = restore_training_state(model, optimizer, state, continuing_phase=checkpoint is not None)
        mode = state["runtime"]
        micro, accumulation, checkpointing = (mode.get(key) for key in
                                             ("physical_microbatch", "gradient_accumulation", "gradient_checkpointing"))
        if (type(micro) is not int or type(accumulation) is not int
                or (micro, accumulation) not in ((1, 16), (2, 8)) or type(checkpointing) is not bool):
            raise SFTInputError("Parent execution mode is not fully recorded")
        from training.sft.published_execution import capacity_setup
        capacity = capacity_setup(model, max(train, key=lambda x: len(x["input_ids"])),
                                  tokenizer.pad_token_id, optimizer=optimizer,
                                  fixed_microbatch_size=micro, fixed_gradient_checkpointing=checkpointing)
        runtime.update({key: capacity[key] for key in
                        ("physical_microbatch", "gradient_accumulation", "gradient_checkpointing")})
        runtime["continuation_policy"] = POLICY
        runtime["parent_checkpoint_hash"] = identity["parent"]["checkpoint_hash"]
        runtime["optimizer_moments_and_steps_restored"] = True
        restore_rng(state["rng"])
        progress = state["progress"] if checkpoint else _progress(0, 0, identity, runtime)
        status_path = output / "fit_status.json"
        def status(cursor: dict, status_value: str = "running") -> None:
            atomic_write_json(status_path, {"status": status_value, "run_id": identity["run_id"],
                "identity_hash": content_hash(identity), "progress": cursor,
                "learning_rates": scheduler.get_last_lr(), "updated_unix": time.time(),
                "status_storage": "LOCAL_ONLY"})
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
            final = train_additional_epoch(model, optimizer, scheduler, train, tokenizer.pad_token_id,
                        identity=identity, runtime=runtime, recipe=spec.recipe,
                        progress=progress, save=save, should_pause=should_pause, on_progress=status)
            if final["status"] == "paused":
                status({k: v for k, v in final.items() if k != "status"}, "paused")
                return {"status": "paused", "identity": identity, "progress": final}
            loss = validation_loss(model, validation, tokenizer.pad_token_id)
            manifest = _export_adapter(output, model, tokenizer, identity)
            result = {"status": "complete", "identity": identity, "run_id": identity["run_id"],
                      "progress": final, "runtime": runtime, "data": data,
                      "validation_loss": loss, "validation_count": 100, "loss_reduction": LOSS_REDUCTION,
                      "adapter_path": str(output / "final_adapter"), "adapter_files": manifest["files"],
                      "adapter_manifest_hash": manifest["manifest_hash"], "checkpoint_hash": manifest["checkpoint_hash"],
                      "elapsed_training_seconds_this_invocation": time.monotonic() - start}
            atomic_write_json(output / "fit_result.json", result)
            status({k: v for k, v in final.items() if k != "status"}, "publishing")
            verified_completion(output, identity)
            publish_completion(output, identity)
            status({k: v for k, v in final.items() if k != "status"}, "complete")
            return result
        except Exception as error:
            atomic_write_json(status_path, {"status": "failed", "error_type": type(error).__name__,
                "detail": str(error), "recovery": "resume_only_verified_continuation_full_checkpoint"})
            raise
