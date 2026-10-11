"""One fixed, completion-only QLoRA fit (protocol C.2 and E.1.3).

Heavy dependencies are loaded only at execution. CPU helpers exercise the real
masked cross entropy, batching, and full-state recovery with declared fixtures.
The production entry point never reduces the frozen corpus or recipe.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
from typing import Any, Callable, Mapping, Sequence

from training.sft.execution_profile import SFTExecutionProfile, resolve_execution_profile

MODELS = {
    "7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
}
LEVELS = {
    "rot13": (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    "kazitext": (0.200, 0.301, 0.454, 0.684, 1.031, 1.553, 2.339, 3.524, 5.310, 8.000),
}
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
LOSS_REDUCTION = "mean_supervised_tokens_per_example_then_mean_examples"
IMPLEMENTATION_VERSION = "completion_sft_v1"


class SFTInputError(ValueError):
    """A frozen input is invalid; replacing IDs or restarting is not a repair."""


class SFTRuntimeBlocked(RuntimeError):
    """The selected runtime or mandatory live evidence is unavailable."""


@dataclass(frozen=True)
class SFTRecipe:
    quantization: str = "nf4"
    double_quantization: bool = True
    compute_dtype: str = "bfloat16"
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = TARGET_MODULES
    learning_rate: float = 2e-4
    optimizer: str = "paged_adamw_32bit"
    scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 0.3
    epochs: int = 2
    microbatch_size: int = 1
    gradient_accumulation_steps: int = 16
    effective_batch_size: int = 16
    max_length: int = 4096
    seed: int = 42
    gradient_checkpointing: bool = True
    checkpoint_steps: int = 100
    early_stopping: bool = False
    loss_reduction: str = LOSS_REDUCTION

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["target_modules"] = list(self.target_modules)
        return result

    def validate(self) -> None:
        if self != SFTRecipe():
            raise SFTInputError("SFT recipe differs from frozen C.2 settings")


@dataclass(frozen=True)
class SFTFitSpec:
    family: str
    model_id: str
    level: int
    model_revision: str
    tokenizer_revision: str
    configuration_hash: str
    dataset_hash: str
    producer_configuration_hash: str | None = None
    handoff_certificate_hash: str | None = None

    def __post_init__(self) -> None:
        if self.family not in LEVELS or self.model_id not in MODELS or type(self.level) is not int or not 1 <= self.level <= 10:
            raise SFTInputError("Unknown SFT family/model/level")
        for key in ("model_revision", "tokenizer_revision"):
            value = getattr(self, key)
            if len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
                raise SFTInputError(f"{key} must be an immutable model commit SHA")
        if bool(self.producer_configuration_hash) != bool(self.handoff_certificate_hash):
            raise SFTInputError("Producer identity requires a handoff certificate")
        for value in (self.producer_configuration_hash, self.handoff_certificate_hash):
            if value is not None and (len(value) != 64 or any(c not in "0123456789abcdef" for c in value)):
                raise SFTInputError("Invalid producer/handoff digest")
        for key in ("configuration_hash", "dataset_hash"):
            value = getattr(self, key)
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise SFTInputError(f"{key} must be a SHA-256 digest")

    @property
    def run_id(self) -> str:
        return f"{self.family}_{self.model_id}_level_{self.level:02d}"

    @property
    def strength(self) -> float:
        return LEVELS[self.family][self.level - 1]

    @property
    def model_name(self) -> str:
        return MODELS[self.model_id]

    def identity(self) -> dict[str, Any]:
        return {**asdict(self), "run_id": self.run_id, "model_name": self.model_name,
                "strength": self.strength, "implementation_version": IMPLEMENTATION_VERSION,
                "recipe": SFTRecipe().to_dict()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SFTFitSpec":
        return cls(**{field: value[field] for field in cls.__dataclass_fields__ if field in value})


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def dataset_hash(train_records: Sequence[Mapping[str, Any]], validation_records: Sequence[Mapping[str, Any]]) -> str:
    """Hash actual ordered records, never just their filenames or claimed hashes."""
    return _hash({"train": list(train_records), "validation": list(validation_records)})


def validate_datasets(spec: SFTFitSpec, train_records: Sequence[Mapping[str, Any]],
                      validation_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(train_records) != 10000 or len(validation_records) != 100:
        raise SFTInputError("Every production fit requires exactly 10,000 train and 100 held-out records")
    id_sets = []
    for split, records in (("train", train_records), ("validation", validation_records)):
        ids = []
        for record in records:
            if record.get("status") not in ("verified", "complete"):
                raise SFTInputError("SFT source has not been verified")
            if record.get("split") != split or record.get("method") != spec.family or record.get("level") != spec.level:
                raise SFTInputError("Record split/condition mismatch")
            if record.get("model_id", record.get("model")) != spec.model_id:
                raise SFTInputError("A model must train on its own clean traces")
            if not all(isinstance(record.get(k), str) and record[k] for k in ("example_id", "prompt", "completion", "clean_trace_hash")):
                raise SFTInputError("Missing required SFT source fields")
            for key in ("clean_trace_hash", "source_content_hash", "source_manifest_hash"):
                value = record.get(key)
                if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                    raise SFTInputError(f"Missing immutable SFT source identity: {key}")
            if record["source_content_hash"] != hashlib.sha256(record["prompt"].encode()).hexdigest():
                raise SFTInputError("Frozen prompt content hash mismatch")
            if (record.get("model_revision"), record.get("tokenizer_revision")) != (spec.model_revision, spec.tokenizer_revision):
                raise SFTInputError("SFT source model/tokenizer revision mismatch")
            if record.get("config_hash") != (spec.producer_configuration_hash or spec.configuration_hash):
                raise SFTInputError("SFT source configuration mismatch")
            if record.get("seed") != 42:
                raise SFTInputError("Transformation seed mismatch")
            if "reasoning" not in record or "final_answer" not in record or not record["reasoning"] or not record["final_answer"]:
                raise SFTInputError("Missing whole assistant target")
            if record["completion"] != "<think>" + record["reasoning"] + "</think>" + record["final_answer"]:
                raise SFTInputError("Assistant target does not preserve transformed reasoning and exact final answer")
            ids.append(record["example_id"])
        if len(ids) != len(set(ids)):
            raise SFTInputError("Duplicate SFT example ID")
        id_sets.append(set(ids))
    if id_sets[0] & id_sets[1]:
        raise SFTInputError("Training and validation IDs overlap")
    if len({record['source_manifest_hash'] for record in [*train_records, *validation_records]}) != 1:
        raise SFTInputError("A fit must use one immutable jointly accepted source pool")
    actual_hash = dataset_hash(train_records, validation_records)
    if actual_hash != spec.dataset_hash:
        raise SFTInputError("Actual SFT record bytes differ from frozen dataset identity")
    return {"train_count": 10000, "validation_count": 100, "dataset_hash": actual_hash,
            "train_ids_hash": _hash(sorted(id_sets[0])), "validation_ids_hash": _hash(sorted(id_sets[1]))}


def render_sft_completion(prompt: str, completion: str, tokenizer: Any) -> tuple[str, int]:
    """Keep native role/end wrappers and insert the intact training target.

    R1's inference-history template strips text preceding </think> from an
    assistant message. Training and all-condition length eligibility must bypass
    that history-only transform. A harmless sentinel locates native wrappers;
    no sentinel is sent to the model and the native generation template stays
    unchanged. This frozen rendering convention is native_training_template_v1.
    """
    sentinel = "__sft_assistant_boundary_" + _hash([prompt, completion]) + "__"
    if sentinel in prompt or sentinel in completion:
        raise SFTInputError("Assistant-boundary sentinel collision")
    user = {"role": "user", "content": prompt}
    skeleton = tokenizer.apply_chat_template([user, {"role": "assistant", "content": sentinel}], tokenize=False, add_generation_prompt=False)
    if skeleton.count(sentinel) != 1:
        raise SFTInputError("Native template has no unique intact assistant boundary")
    prefix, suffix = skeleton.split(sentinel)
    return prefix + completion + suffix, len(prefix)


def tokenize_completion(record: Mapping[str, Any], tokenizer: Any, max_length: int = 4096) -> dict[str, Any]:
    """Mask the exact native assistant boundary and retain the entire target.

    Fast-tokenizer offsets verify that no token crosses the prompt boundary.
    Native assistant end tokens are supervised. No reasoning-history stripping
    template is applied to the actual target.
    """
    rendered, boundary = render_sft_completion(record["prompt"], record["completion"], tokenizer)
    try:
        encoded = tokenizer(rendered, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
    except (TypeError, NotImplementedError) as error:
        raise SFTInputError("Native mask verification requires a tokenizer with character offsets") from error
    ids = list(encoded["input_ids"])
    offsets = list(encoded["offset_mapping"])
    if record.get("transformed_token_count") is not None and record["transformed_token_count"] != len(ids):
        raise SFTInputError("Actual native token count differs from frozen transform preflight")
    if len(ids) > max_length:
        raise SFTInputError(f"SFT sequence has {len(ids)} tokens; truncation is forbidden")
    labels = []
    for token, (start, end) in zip(ids, offsets, strict=True):
        if start < boundary < end:
            raise SFTInputError("A token crosses the native prompt/assistant boundary")
        labels.append(token if start >= boundary and end > start else -100)
    target_count = sum(label != -100 for label in labels[1:])
    if not target_count:
        raise SFTInputError("No causally predicted assistant tokens remain")
    return {"example_id": record.get("example_id"), "input_ids": ids,
            "attention_mask": [1] * len(ids), "labels": labels,
            "prompt_token_count": sum(label == -100 for label in labels),
            "supervised_token_count": target_count}


@dataclass(frozen=True)
class _ValidatedCompletionCounts:
    """CPU-checked counts bound to one unchanged collated labels tensor."""
    labels: Any
    counts: Any
    labels_version: int
    counts_version: int


def collate_completion(examples: Sequence[Mapping[str, Any]], pad_token_id: int, device: Any = None,
                       *, validate_counts: bool = True) -> dict[str, Any]:
    import torch
    if not examples:
        raise SFTInputError("Empty completion batch")
    length = max(len(example["input_ids"]) for example in examples)
    result = {key: [] for key in ("input_ids", "attention_mask", "labels")}
    supervised_counts = []
    for example in examples:
        labels = list(example["labels"])
        if len(labels) != len(example["input_ids"]):
            raise SFTInputError("Completion labels do not match input sequence length")
        if validate_counts:
            count = sum(label != -100 for label in labels[1:])
            if not count:
                raise SFTInputError("An example has no supervised shifted target")
            supervised_counts.append(count)
        padding = length - len(example["input_ids"])
        result["input_ids"].append(list(example["input_ids"]) + [pad_token_id] * padding)
        result["attention_mask"].append([1] * len(example["input_ids"]) + [0] * padding)
        result["labels"].append(labels + [-100] * padding)
    batch = {key: torch.tensor(values, dtype=torch.long, device=device) for key, values in result.items()}
    if validate_counts:
        counts = torch.tensor(supervised_counts, dtype=torch.long, device=device)
        batch["validated_completion_counts"] = _ValidatedCompletionCounts(
            batch["labels"], counts, batch["labels"]._version, counts._version)
    return batch


def completion_loss(logits: Any, labels: Any, *, validated_counts: Any = None) -> Any:
    """Mean assistant-token CE within example, then mean across examples.

    Collation can validate counts before device transfer. Its bound metadata
    avoids a device-to-host scalar check while preserving the same integer
    denominators and floating-point CE/reduction operations. Standalone calls
    retain direct validation; stale or substituted metadata is rejected.
    """
    import torch
    import torch.nn.functional as functional
    shifted_logits, shifted_labels = logits[:, :-1, :].float(), labels[:, 1:]
    if validated_counts is None:
        counts = (shifted_labels != -100).sum(dim=1)
        if torch.any(counts == 0):
            raise SFTInputError("An example has no supervised shifted target")
    else:
        if (not isinstance(validated_counts, _ValidatedCompletionCounts)
                or validated_counts.labels is not labels
                or validated_counts.labels_version != labels._version
                or validated_counts.counts_version != validated_counts.counts._version):
            raise SFTInputError("Validated completion counts do not match unchanged collated labels")
        counts = validated_counts.counts
    token_losses = functional.cross_entropy(shifted_logits.transpose(1, 2), shifted_labels, ignore_index=-100, reduction="none")
    return (token_losses.sum(dim=1) / counts).mean()


def exposure_order(count: int, seed: int = 42) -> list[int]:
    """Two seeded epoch permutations, accumulated continuously across boundary."""
    generator = random.Random(seed)
    order = []
    for _ in range(2):
        epoch = list(range(count))
        generator.shuffle(epoch)
        order.extend(epoch)
    return order


def capture_rng() -> dict[str, Any]:
    import numpy as np
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state: Mapping[str, Any]) -> None:
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        if not torch.cuda.is_available() or len(state["torch_cuda"]) != torch.cuda.device_count():
            raise SFTRuntimeBlocked("Checkpoint CUDA RNG topology differs; declare a runtime migration")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def verify_mask_gradients(model: Any, example: Mapping[str, Any], pad_token_id: int, *, require_lora: bool = True) -> dict[str, Any]:
    """Actual backward pass; preserve RNG and clear gradients without updating."""
    import torch
    before = capture_rng()
    model.train()
    model.zero_grad(set_to_none=True)
    try:
        device = next(model.parameters()).device
        batch = collate_completion([example], pad_token_id, device)
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        loss = completion_loss(logits, batch["labels"])
        if not torch.isfinite(loss):
            raise SFTRuntimeBlocked("Actual completion loss is not finite")
        loss.backward()
        names = []
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                if require_lora and "lora_" not in name:
                    raise SFTRuntimeBlocked(f"Unexpected trainable nonadapter parameter: {name}")
                if parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum().item() > 0:
                    names.append(name)
        if not names:
            raise SFTRuntimeBlocked("No nonzero actual adapter gradient")
        if require_lora:
            absent = [projection for projection in TARGET_MODULES if not any(projection in name for name, parameter in model.named_parameters() if parameter.requires_grad)]
            if absent:
                raise SFTRuntimeBlocked(f"LoRA projection families missing: {absent}")
        return {"status": "passed", "loss": loss.item(), "nonzero_gradient_parameters": names,
                "prompt_token_count": example["prompt_token_count"], "supervised_token_count": example["supervised_token_count"],
                "loss_reduction": LOSS_REDUCTION}
    finally:
        model.zero_grad(set_to_none=True)
        restore_rng(before)


def one_optimizer_update(model: Any, optimizer: Any, scheduler: Any,
                         examples: Sequence[Mapping[str, Any]], pad_token_id: int,
                         *, execution_profile: SFTExecutionProfile | None = None) -> tuple[Any, ...]:
    """One real update, also used by the isolated hardware benchmark.

    The optional count metadata preserves CE and reduction operation order.
    Deferred checks synchronize once after all microbatches, before clipping or
    either state update. Any nonfinite loss still requires checkpoint rollback.
    Returned losses are detached; collecting diagnostics does not retain graphs.
    """
    import torch
    profile = resolve_execution_profile(execution_profile)
    recipe = SFTRecipe()
    if not 1 <= len(examples) <= recipe.gradient_accumulation_steps:
        raise SFTInputError("One SFT update requires one to sixteen ordered examples")
    device = next(model.parameters()).device
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for example in examples:
        batch = collate_completion([example], pad_token_id, device, validate_counts=profile.validated_loss_counts)
        counts = batch["validated_completion_counts"] if profile.validated_loss_counts else None
        loss = completion_loss(model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits,
                               batch["labels"], validated_counts=counts)
        if not profile.defer_finite_checks and not torch.isfinite(loss):
            raise SFTRuntimeBlocked("Nonfinite training loss; partial update requires rollback")
        (loss / len(examples)).backward()
        losses.append(loss.detach())
    if profile.defer_finite_checks and not torch.isfinite(torch.stack(losses)).all():
        raise SFTRuntimeBlocked("Nonfinite training loss; partial update requires rollback")
    torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad],
                                  recipe.max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    return tuple(losses)


def train_optimizer_updates(model: Any, optimizer: Any, scheduler: Any,
                            examples: Sequence[Mapping[str, Any]], pad_token_id: int,
                            *, progress: Mapping[str, Any] | None = None,
                            save: Callable[[dict[str, Any]], None] | None = None,
                            should_pause: Callable[[], bool] | None = None,
                            execution_profile: SFTExecutionProfile | None = None) -> dict[str, Any]:
    """Real optimizer loop, shared by GPU production and tiny CPU fixtures.

    Pause happens only after one complete logical optimizer update. Exceptions
    propagate without claiming the potentially partial update as completed.
    """
    profile = resolve_execution_profile(execution_profile)
    recipe = SFTRecipe()
    order = exposure_order(len(examples))
    order_hash = _hash(order)
    total_steps = math.ceil(len(order) / recipe.gradient_accumulation_steps)
    completed = int((progress or {}).get("completed_steps", 0))
    position = int((progress or {}).get("sampler_position", 0))
    if progress and progress.get("exposure_order_hash") != order_hash:
        raise SFTInputError("Resume sampler/exposure identity mismatch")
    if progress and progress.get("execution_profile", SFTExecutionProfile().to_dict()) != profile.to_dict():
        raise SFTInputError("Resume execution profile differs; declare a runtime migration")
    if completed < 0 or completed > total_steps or position != min(completed * 16, len(order)):
        raise SFTInputError("Checkpoint is not at a completed optimizer boundary")
    processed_tokens = int((progress or {}).get("processed_tokens", 0))
    model.train()
    for step in range(completed, total_steps):
        indices = order[position:position + recipe.gradient_accumulation_steps]
        one_optimizer_update(model, optimizer, scheduler, [examples[index] for index in indices],
                             pad_token_id, execution_profile=profile)
        position += len(indices)
        processed_tokens += sum(len(examples[index]["input_ids"]) for index in indices)
        completed = step + 1
        state = {"completed_steps": completed, "sampler_position": position,
                 "exposure_order_hash": order_hash, "processed_tokens": processed_tokens,
                 "total_optimizer_steps": total_steps, "example_presentations": position,
                 "execution_profile": profile.to_dict()}
        pause = bool(should_pause and should_pause())
        if save and (completed % 100 == 0 or completed == total_steps or pause):
            save(state)
        if pause:
            return {**state, "status": "paused"}
    return {"completed_steps": completed, "sampler_position": position,
            "exposure_order_hash": order_hash, "processed_tokens": processed_tokens,
            "total_optimizer_steps": total_steps, "example_presentations": position,
            "execution_profile": profile.to_dict(), "status": "complete"}


def validation_loss(model: Any, examples: Sequence[Mapping[str, Any]], pad_token_id: int) -> float:
    import torch
    if not examples:
        raise SFTInputError("Missing fixed held-out split")
    model.eval()
    values = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for example in examples:
            batch = collate_completion([example], pad_token_id, device)
            value = completion_loss(model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits, batch["labels"])
            if not torch.isfinite(value):
                raise SFTRuntimeBlocked("Nonfinite validation loss")
            values.append(value.item())
    return sum(values) / len(values)


def _runtime(spec: SFTFitSpec, device: str, *, cache_root: str | Path | None = None,
             execution_profile: SFTExecutionProfile | None = None) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    profile = resolve_execution_profile(execution_profile)
    try:
        import numpy as np
        import torch
        import bitsandbytes as bnb
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_cosine_schedule_with_warmup
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as error:
        raise SFTRuntimeBlocked("Install the pinned CUDA training dependency lock") from error
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SFTRuntimeBlocked("Fixed BF16 NF4 training requires a compatible CUDA GPU")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    from operations.assets import model_load_kwargs
    tokenizer = AutoTokenizer.from_pretrained(spec.model_name, revision=spec.tokenizer_revision, use_fast=True, trust_remote_code=False,
        **model_load_kwargs(spec.model_name, spec.tokenizer_revision, cache_root, tokenizer_only=True))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_name, revision=spec.model_revision, trust_remote_code=False,
        **model_load_kwargs(spec.model_name, spec.model_revision, cache_root),
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16, device_map={"": device}, attn_implementation=profile.attention_implementation)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                                           target_modules=list(TARGET_MODULES), bias="none", task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = bnb.optim.PagedAdamW32bit([parameter for parameter in model.parameters() if parameter.requires_grad],
                                         lr=2e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=math.ceil(1250 * 0.03), num_training_steps=1250)
    runtime = {"device": device, "gpu_name": torch.cuda.get_device_name(torch.device(device)),
               "vram_bytes": torch.cuda.get_device_properties(torch.device(device)).total_memory,
               "cuda_version": torch.version.cuda, "attention_implementation": profile.attention_implementation,
               "execution_profile": profile.to_dict(), "use_reentrant": False,
               "optimizer_betas": [0.9, 0.999], "optimizer_epsilon": 1e-8, "warmup_steps": 38,
               "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "bitsandbytes", "numpy")}}
    return model, tokenizer, optimizer, scheduler, runtime


def _verified_checkpoints(directory: Path, identity: Mapping[str, Any]) -> list[tuple[int, Path]]:
    from shared_procedures.checkpointing import CheckpointError, verify_checkpoint
    from shared_procedures.storage import atomic_write_json
    candidates, invalid = [], []
    for path in sorted(directory.glob("checkpoint_*")):
        if path.is_dir():
            try:
                manifest = verify_checkpoint(path, dict(identity))
            except (CheckpointError, ValueError, OSError) as error:
                invalid.append({"path": str(path), "error_type": type(error).__name__, "detail": str(error)})
                continue
            candidates.append((int(manifest["progress"]["completed_steps"]), path))
    if invalid:
        atomic_write_json(directory / "checkpoint_integrity_failures.json", {"identity": dict(identity), "invalid": invalid,
            "recovery": "only previously verified complete states may be used; invalid files preserved"})
        if not candidates:
            raise SFTInputError("Claimed recovery states are invalid; a fresh run cannot replace their provenance")
    return sorted(candidates)


def _verify_adapter(directory: Path, identity: Mapping[str, Any]) -> dict[str, Any]:
    from shared_procedures.storage import read_json
    manifest = read_json(directory / "manifest.json")
    if manifest.get("identity") != dict(identity) or manifest.get("manifest_hash") != _hash({k:v for k,v in manifest.items() if k != "manifest_hash"}):
        raise SFTInputError("Final adapter manifest identity/integrity mismatch")
    files = manifest.get("files", {})
    if "adapter_config.json" not in files or "adapter_model.safetensors" not in files:
        raise SFTInputError("Final adapter inventory lacks LoRA configuration or tensor weights")
    for relative, expected_hash in files.items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise SFTInputError("Unsafe adapter inventory path")
        path = directory / relative_path
        if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise SFTInputError("Final adapter file is missing or corrupt")
    adapter_config = read_json(directory / "adapter_config.json")
    if (adapter_config.get("r"), adapter_config.get("lora_alpha"), adapter_config.get("lora_dropout")) != (8, 16, 0.05):
        raise SFTInputError("Exported adapter differs from fixed LoRA recipe")
    if set(adapter_config.get("target_modules", [])) != set(TARGET_MODULES):
        raise SFTInputError("Exported adapter misses fixed projection families")
    if adapter_config.get("base_model_name_or_path") != identity["model_name"]:
        raise SFTInputError("Exported adapter changed the construction base")
    return manifest


verify_adapter_manifest = _verify_adapter

def verify_fit_result(output_dir: str | Path, expected_identity: Mapping[str, Any]) -> dict[str, Any]:
    from shared_procedures.checkpointing import load_checkpoint, verify_checkpoint
    from shared_procedures.storage import read_json
    from safetensors.torch import load_file
    import torch
    output_dir = Path(output_dir)
    result = read_json(output_dir / "fit_result.json")
    if result.get("status") != "complete" or result.get("identity") != dict(expected_identity):
        raise SFTInputError("Fit completion has a different condition/config/data identity")
    if result.get("progress", {}).get("example_presentations") != 20000 or result["progress"].get("completed_steps") != 1250:
        raise SFTInputError("Fit completion does not establish fixed exposure")
    if not math.isfinite(result["validation_loss"]) or result["validation_count"] != 100:
        raise SFTInputError("Fit lacks the fixed held-out validation computation")
    manifest = _verify_adapter(output_dir / "final_adapter", expected_identity)
    if result.get("adapter_manifest_hash") != manifest["manifest_hash"] or result.get("adapter_files") != manifest["files"]:
        raise SFTInputError("Completion result differs from verified final adapter")
    final_checkpoint = output_dir / "checkpoint_001250"
    recovery_manifest = verify_checkpoint(final_checkpoint, dict(expected_identity))
    if result.get('checkpoint_hash') != manifest['checkpoint_hash'] or manifest["checkpoint_hash"] != recovery_manifest["checkpoint_hash"]:
        raise SFTInputError("Final adapter is not bound to the completed full-state checkpoint")
    if any(recovery_manifest['progress'].get(key) != required for key, required in
           (('completed_steps', 1250), ('sampler_position', 20000), ('example_presentations', 20000))):
        raise SFTInputError("Final full-state checkpoint does not establish fixed exposure")
    recovered = load_checkpoint(final_checkpoint, dict(expected_identity))
    weights = load_file(str(output_dir / "final_adapter" / "adapter_model.safetensors"), device="cpu")
    if set(weights) != set(recovered["adapter"]):
        raise SFTInputError("Final adapter tensor names differ from complete recovery weights")
    for name, tensor in weights.items():
        if not torch.isfinite(tensor).all() or not torch.equal(tensor, recovered["adapter"][name].cpu()):
            raise SFTInputError("Final adapter tensors differ from completed training state")
    return result


def _publish_adapter(output: Path, model: Any, tokenizer: Any, identity: Mapping[str, Any], recovery: Any) -> dict[str, Any]:
    from shared_procedures.checkpointing import verify_checkpoint
    from shared_procedures.storage import atomic_write_json, fsync_directory
    adapter = output / "final_adapter"
    if adapter.exists():
        # An interrupted completion publication can reuse an already verified
        # export. An unknown or corrupt accepted export is never overwritten.
        return _verify_adapter(adapter, identity)
    staging = output / ".final_adapter_staging"
    if staging.exists():
        shutil.rmtree(staging)  # explicitly private, unpublished output only
    model.save_pretrained(staging, safe_serialization=True, save_embedding_layers=False)
    tokenizer.save_pretrained(staging)
    files = {str(path.relative_to(staging)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in staging.rglob("*") if path.is_file()}
    checkpoint = verify_checkpoint(output / "checkpoint_001250", dict(identity))
    manifest = {"identity": dict(identity), "files": files, "checkpoint_hash": checkpoint["checkpoint_hash"]}
    manifest["manifest_hash"] = _hash(manifest)
    atomic_write_json(staging / "manifest.json", manifest)
    _verify_adapter(staging, identity)
    for path in staging.rglob("*"):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    fsync_directory(staging)
    os.replace(staging, adapter)
    fsync_directory(output)
    return _verify_adapter(adapter, identity)


def _record_fit_attempt(output: Path, identity: Mapping[str, Any], resume: str | Path | None,
                        restored_progress: Mapping[str, Any] | None) -> None:
    """Fault restarts share one persistent budget; verified pauses may continue."""
    from shared_procedures.recovery_procedure import RecoveryManager
    from shared_procedures.storage import atomic_write_json
    attempts_path = output / 'attempts.json'
    attempts = json.loads(attempts_path.read_text()) if attempts_path.exists() else {'attempts': []}
    status_path = output / 'fit_status.json'
    previous = json.loads(status_path.read_text()) if status_path.exists() else {}
    voluntary = (previous.get('status') == 'paused' and previous.get('identity') == dict(identity)
                 and restored_progress is not None
                 and previous.get('progress', {}).get('completed_steps') == restored_progress.get('completed_steps'))
    step = restored_progress['completed_steps'] if restored_progress is not None else 0
    if attempts['attempts'] and not voluntary:
        RecoveryManager.for_worker(artifact_root=output.parent.parent).reserve_retry(
            identity['run_id'] + ':recover_step_' + str(step))
    attempts['attempts'].append({'attempt': len(attempts['attempts']) + 1, 'time': time.time(),
                                'resume': str(resume) if resume is not None else None,
                                'replay_from_seed': resume is None and bool(attempts['attempts']),
                                'voluntary_continuation': voluntary})
    atomic_write_json(attempts_path, attempts)
    # An abrupt kill during the continuation must not inherit the previous
    # voluntary pause exemption on its next entry.
    atomic_write_json(status_path, {'status': 'running', 'run_id': identity['run_id'],
                                   'identity': dict(identity), 'restored_completed_steps': step})


def run_sft_fit(spec: SFTFitSpec | Mapping[str, Any], train_records: Sequence[Mapping[str, Any]],
                validation_records: Sequence[Mapping[str, Any]], output_dir: str | Path,
                config: Mapping[str, Any], *, resume: str | Path | None = None,
                preflight_report: Mapping[str, Any] | None = None,
                should_pause: Callable[[], bool] | None = None,
                handoff_certificate: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Train the exact fit, or continue only its verified full recovery state.

    Caller must provide real preflight evidence; this function never turns unit
    fixtures into a paid-launch authorization. Its own live gradient check also
    runs before the first optimizer update and after exact-state restoration.
    """
    from shared_procedures.configuration import configuration_hash
    from shared_procedures.checkpointing import load_checkpoint, save_checkpoint, verify_checkpoint
    from shared_procedures.preflight import require_preflights
    from shared_procedures.storage import atomic_write_json
    if not isinstance(spec, SFTFitSpec):
        spec = SFTFitSpec.from_dict(spec)
    if config.get("sft", {}).get("loss_reduction") != LOSS_REDUCTION:
        raise SFTInputError("Resolved SFT loss reduction differs from frozen engine convention")
    if configuration_hash(dict(config)) != spec.configuration_hash:
        raise SFTInputError("Configuration differs from frozen fit identity")
    if not preflight_report or preflight_report.get("configuration_hash") != spec.configuration_hash:
        raise SFTRuntimeBlocked("Matching real preflight report is required")
    require_preflights(dict(preflight_report), ["model_runtime", "clean_data", "loss_mask_gradients", "full_checkpoint_resume", "durable_storage", "spending_authorization", "baseline_repeatability", "statistical_validation"])
    if spec.producer_configuration_hash:
        from training.sft.data_handoff import validate_handoff
        validate_handoff(handoff_certificate, config, spec)
    data_evidence = validate_datasets(spec, train_records, validation_records)
    identity = {**spec.identity(), "code_revision": config.get("runtime", {}).get("implementation_revision"),
                "dependency_hash": config.get("runtime", {}).get("dependency_lock_hash")}
    if not identity["code_revision"] or not identity["dependency_hash"]:
        raise SFTInputError("Resolved code and dependency identities are required")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (output / ".fit_lock").open("a")
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SFTRuntimeBlocked("Another worker owns this fit namespace") from error
        if (output / "fit_result.json").exists():
            return verify_fit_result(output, identity)
        checkpoints = _verified_checkpoints(output, identity)
        if resume is None and checkpoints:
            resume = checkpoints[-1][1]
        if (output / "run_identity.json").exists():
            old = json.loads((output / "run_identity.json").read_text())
            if old != identity:
                raise SFTInputError("Existing output namespace belongs to another exact fit")
        atomic_write_json(output / "run_identity.json", identity)
        restored = load_checkpoint(resume, identity) if resume is not None else None
        _record_fit_attempt(output, identity, resume, restored['progress'] if restored else None)
        model, tokenizer, optimizer, scheduler, runtime = _runtime(spec, config.get("runtime", {}).get("sft_device", "cuda:0"), cache_root=config.get("runtime", {}).get("cache_root"))
        from peft import get_peft_model_state_dict, set_peft_model_state_dict
        train = [tokenize_completion(record, tokenizer) for record in train_records]
        validation = [tokenize_completion(record, tokenizer) for record in validation_records]
        if restored:
            set_peft_model_state_dict(model, restored["adapter"])
            optimizer.load_state_dict(restored["optimizer"])
            scheduler.load_state_dict(restored["scheduler"])
            restore_rng(restored["rng"])
        gradient_evidence = verify_mask_gradients(model, train[0], tokenizer.pad_token_id)
        atomic_write_json(output / "runtime_evidence.json", {"identity": identity, "runtime": runtime,
            "mask_gradients": gradient_evidence, "data": data_evidence,
            "train_token_lengths": {"min": min(len(r["input_ids"]) for r in train), "max": max(len(r["input_ids"]) for r in train), "mean": sum(len(r["input_ids"]) for r in train) / len(train)}})
        def save(progress: dict[str, Any]) -> None:
            directory = output / f"checkpoint_{progress['completed_steps']:06d}"
            save_checkpoint(directory, identity, {"adapter": get_peft_model_state_dict(model, save_embedding_layers=False),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "rng": capture_rng(),
                "progress": progress, "runtime": runtime})
            verify_checkpoint(directory, identity)
            valid = _verified_checkpoints(output, identity)
            for _, old in valid[:-2]:
                shutil.rmtree(old)
        start = time.monotonic()
        progress = train_optimizer_updates(model, optimizer, scheduler, train, tokenizer.pad_token_id,
                    progress=restored["progress"] if restored else None, save=save, should_pause=should_pause)
        if progress["status"] == "paused":
            result = {"status": "paused", "run_id": spec.run_id, "identity": identity, "progress": progress}
            atomic_write_json(output / "fit_status.json", result)
            return result
        # Final full state already durably published by the optimizer loop.
        loss = validation_loss(model, validation, tokenizer.pad_token_id)
        from shared_procedures.recovery_procedure import RecoveryManager, RetryableFailure
        import errno
        recovery = RecoveryManager.for_worker(artifact_root=output.parent.parent)
        def publish_adapter():
            try:
                return _publish_adapter(output, model, tokenizer, identity, recovery)
            except OSError as error:
                if error.errno in {errno.EIO, errno.ETIMEDOUT, errno.ESTALE, errno.EAGAIN}:
                    raise RetryableFailure("Transient final adapter publication I/O failed") from error
                raise
        adapter_manifest = recovery.run(spec.run_id + ":final_adapter", publish_adapter)
        adapter = output / "final_adapter"
        files = adapter_manifest["files"]
        import torch
        result = {"status": "complete", "run_id": spec.run_id, "identity": identity, "progress": progress,
                  "validation_loss": loss, "validation_count": 100, "loss_reduction": LOSS_REDUCTION,
                  "adapter_path": str(adapter), "adapter_files": files, "adapter_manifest_hash": adapter_manifest["manifest_hash"], "checkpoint_hash": adapter_manifest["checkpoint_hash"], "data": data_evidence,
                  "runtime": runtime, "elapsed_training_seconds": time.monotonic() - start,
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        atomic_write_json(output / "fit_result.json", result)
        return verify_fit_result(output, identity)
    except Exception as error:
        atomic_write_json(output / "fit_status.json", {"status": "failed", "run_id": spec.run_id,
            "error_type": type(error).__name__, "detail": str(error), "recovery": "only a previously verified complete checkpoint may resume"})
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
