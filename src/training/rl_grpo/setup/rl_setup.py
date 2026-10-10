"""Exact-run RL setup and the lazy single-GPU Transformers/PEFT runtime.

H.7 is implemented explicitly rather than inherited from trainer defaults.
Optimizer/scheduler/precision conventions are frozen in RLRecipe before outcomes.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import re
from typing import Any, Callable, Mapping
import uuid

from shared_procedures.identity import content_hash
from shared_procedures.storage import atomic_write_json, read_json
from shared_procedures.checkpointing import save_checkpoint, load_checkpoint, verify_checkpoint

MODELS = ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
SNAPSHOT_STEPS = tuple(range(10, 101, 10))
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
REQUIRED_PREFLIGHTS = ("configuration", "model_runtime", "loss_mask_gradients", "full_checkpoint_resume",
                       "durable_storage", "judge_services", "baseline_repeatability", "grpo_update",
                       "statistical_validation", "spending_authorization")


@dataclass(frozen=True)
class RLRecipe:
    seed: int = 42
    logical_steps: int = 100
    questions_per_step: int = 8
    completions_per_question: int = 4
    lora_rank: int = 32
    lora_alpha: int = 32
    learning_rate: float = 5e-6
    beta: float = 0.01
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    max_completion_length: int = 2048
    loss_type: str = "dr_grpo"
    scale_rewards: str = "group"
    num_iterations: int = 1
    importance_sampling_level: str = "token"
    epsilon: float = 0.2
    epsilon_high: float = 0.2
    mask_truncated_completions: bool = False
    use_vllm: bool = False
    reference_sync: bool = False
    entropy_coefficient: float = 0.0
    optimizer: str = "torch_adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.0
    scheduler: str = "constant"
    precision: str = "bf16"
    quantization: str = "nf4_double_quant"
    lora_dropout: float = 0.0
    max_grad_norm: float = 1.0
    microbatch_size: int = 1
    gradient_accumulation_steps: int = 32
    gradient_checkpointing: bool = True
    reward_std_correction: int = 1
    reward_std_epsilon: float = 1e-4

    def validate(self) -> None:
        # All effective defaults are frozen; changes require a versioned convention repair.
        if asdict(self) != asdict(RLRecipe()):
            raise ValueError("RL recipe differs from frozen scientific/settings convention")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {**asdict(self), "target_modules": list(TARGET_MODULES)}


@dataclass
class RLRunState:
    identity: dict[str, Any]
    run_root: Path
    runtime: Any
    recipe: RLRecipe
    completed_steps: int = 0
    attempt_id: str = ""
    boundary_verified: bool = True
    execution_mode: str = "production"

    @property
    def policy_hash(self) -> str:
        return self.runtime.policy_hash()

    def save(self) -> dict[str, Any]:
        if not self.boundary_verified:
            raise RuntimeError("Cannot save an unsafe partial optimizer update")
        state = self.runtime.capture_state()
        state["progress"] = {"completed_steps": self.completed_steps,
                             "sampler_position": self.completed_steps * 8}
        state["runtime"] = {"attempt_id": self.attempt_id, "execution_mode": self.execution_mode,
                            "runtime_identity": self.identity["runtime_identity"], "policy_hash": self.policy_hash}
        destination = self.run_root / "checkpoints" / f"step_{self.completed_steps:03d}"
        if destination.exists():
            saved = load_checkpoint(destination, expected_identity=self.identity)
            if saved["progress"] != state["progress"] or saved["runtime"]["policy_hash"] != self.policy_hash:
                raise ValueError("Existing immutable checkpoint differs from live optimizer boundary")
            return verify_checkpoint(destination, expected_identity=self.identity)
        manifest = save_checkpoint(destination, self.identity, state)
        verify_checkpoint(destination, expected_identity=self.identity)
        return manifest


def validate_rl_manifest(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    questions = list(manifest.get("questions", []))
    if manifest.get("status") != "verified" or manifest.get("source_split") != "train" or manifest.get("seed") != 42:
        raise ValueError("RL input must be a verified seed-42 GSM8K training manifest")
    if len(questions) != 800 or len({q["question_id"] for q in questions}) != 800:
        raise ValueError("Exactly 800 distinct frozen questions required")
    from shared_procedures.response_parsing import normalize_numeric_answer
    for index, q in enumerate(questions):
        if not q.get("question", "").strip() or normalize_numeric_answer(q.get("reference_answer", "")) is None:
            raise ValueError("Invalid frozen question/reference answer")
        if q.get("order", index) != index or q.get("source_split", "train") != "train":
            raise ValueError("Question order/split differs from frozen training selection")
        if any(key in q for key in ("reasoning", "target_reasoning", "reference_solution")):
            raise ValueError("RL question records must not supply target reasoning")
    if manifest.get("content_hash") != content_hash(questions):
        raise ValueError("RL frozen question content hash mismatch")
    expected_batches = [[q["question_id"] for q in questions[start:start+8]] for start in range(0, 800, 8)]
    if manifest.get("batches", expected_batches) != expected_batches:
        raise ValueError("Frozen RL grouping differs from exact consecutive eight-question order")
    return questions


def implementation_hash() -> str:
    root = Path(__file__).resolve().parents[4]
    owned = sorted((root / "src/training/rl_grpo").rglob("*.py"))
    owned += [root / "prompts" / name for name in ("rl_generation.txt", "rl_pass_a.txt", "rl_pass_b.txt")]
    return content_hash({str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in owned})


def _require_preflights(evidence: Mapping[str, Any] | None, runtime_identity: Mapping[str, Any]) -> None:
    from shared_procedures.preflight import require_preflights, PreflightBlocked
    config_hash = runtime_identity.get("configuration_hash")
    if not isinstance(config_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", config_hash):
        raise RuntimeError("RL launch blocked; real preflight configuration hash is required")
    try:
        require_preflights(dict(evidence or {}), REQUIRED_PREFLIGHTS, expected_configuration_hash=config_hash)
    except PreflightBlocked as exc:
        raise RuntimeError("RL launch blocked; real preflight evidence required: " + str(exc)) from exc


def setup_rl_run(model_id: str, manifest: Mapping[str, Any], run_root: str | Path, *, model_revision: str,
                 runtime_identity: Mapping[str, Any], tokenizer_revision: str | None = None, preflight_evidence: Mapping[str, Any] | None = None,
                 runtime_factory: Callable[..., Any] | None = None,
                 execution_mode: str = "production", recipe: RLRecipe | None = None) -> RLRunState:
    """Load untouched base or verify/restore full exact-run state; corrupt state fails closed."""
    if model_id not in MODELS:
        raise ValueError("Construction model identity is fixed")
    if not re.fullmatch(r"[0-9a-f]{40}", model_revision):
        raise ValueError("Model and tokenizer require an immutable 40-character revision")
    tokenizer_revision = tokenizer_revision or model_revision
    if not re.fullmatch(r"[0-9a-f]{40}", tokenizer_revision):
        raise ValueError("Tokenizer requires an immutable 40-character revision")
    recipe = recipe or RLRecipe()
    recipe.validate()
    validate_rl_manifest(manifest)
    if execution_mode not in ("production", "fixture"):
        raise ValueError("Unknown execution mode")
    if execution_mode == "production":
        if runtime_factory not in (None, TransformersRLRuntime):
            raise ValueError("Production execution requires the real Transformers runtime; injected engines are fixtures")
        _require_preflights(preflight_evidence, runtime_identity)
    elif runtime_factory is None:
        raise ValueError("Fixture mode requires an explicit fixture runtime factory")
    if not runtime_identity:
        raise ValueError("Pinned runtime/dependency identity required")
    identity = {"model_id": model_id, "model_revision": model_revision, "tokenizer_revision": tokenizer_revision,
                "trajectory": "rl", "source_hash": manifest["content_hash"], "recipe": recipe.to_dict(),
                "implementation_hash": implementation_hash(), "runtime_identity": dict(runtime_identity),
                "execution_mode": execution_mode}
    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    identity_file = root / "identity.json"
    if identity_file.exists() and read_json(identity_file) != identity:
        raise ValueError("Existing RL run identity differs; refusing fresh start/overwrite")
    atomic_write_json(identity_file, identity)
    checkpoints = sorted(p for p in (root / "checkpoints").glob("step_*") if re.fullmatch(r"step_\d{3}", p.name))
    restored, completed = None, 0
    if checkpoints:
        verified = []
        for checkpoint in checkpoints:
            info = verify_checkpoint(checkpoint, expected_identity=identity)
            verified.append((info["progress"]["completed_steps"], checkpoint))
        completed, checkpoint = max(verified)
        restored = load_checkpoint(checkpoint, expected_identity=identity)
        if not 0 <= completed <= 100 or restored["progress"]["sampler_position"] != completed * 8:
            raise ValueError("Restored RL position differs from exact optimizer boundary")
    factory = runtime_factory or TransformersRLRuntime
    runtime = factory(model_id=model_id, model_revision=model_revision, tokenizer_revision=tokenizer_revision, recipe=recipe,
                      runtime_identity=dict(runtime_identity))
    state = RLRunState(identity, root, runtime, recipe, attempt_id=uuid.uuid4().hex,
                       execution_mode=execution_mode)
    try:
        if restored is not None:
            runtime.restore_state(restored)
            if restored["runtime"]["policy_hash"] != runtime.policy_hash():
                raise ValueError("Restored adapter hash differs from verified full state")
            state.completed_steps = completed
        else:
            state.save()
    except Exception:
        runtime.close()
        raise
    atomic_write_json(root / "attempt.json", {"attempt_id": state.attempt_id,
                     "restored_step": state.completed_steps, "status": "ready", "execution_mode": execution_mode})
    return state


class TransformersRLRuntime:
    """Real NF4 QLoRA engine: one GPU, sequential sampling and microbatch accumulation.

    Reference log-probabilities disable the fresh adapter on the original frozen base.
    Heavy dependencies and model downloads occur only when this runtime is constructed.
    """
    def __init__(self, *, model_id: str, model_revision: str, tokenizer_revision: str, recipe: RLRecipe,
                 runtime_identity: Mapping[str, Any]):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from operations.assets import model_load_kwargs
        recipe.validate()
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Real RL requires a BF16-compatible CUDA GPU")
        self.torch, self.recipe = torch, recipe
        random.seed(recipe.seed)
        torch.manual_seed(recipe.seed)
        torch.cuda.manual_seed_all(recipe.seed)
        tokenizer_assets = model_load_kwargs(model_id, tokenizer_revision, tokenizer_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=tokenizer_revision,
                                                       trust_remote_code=False, **tokenizer_assets)
        if self.tokenizer.eos_token_id is None or not self.tokenizer.chat_template:
            raise RuntimeError("Exact native tokenizer must supply EOS and chat template")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                          bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        model_assets = model_load_kwargs(model_id, model_revision)
        base = AutoModelForCausalLM.from_pretrained(model_id, revision=model_revision,
                 quantization_config=quantization, torch_dtype=torch.bfloat16,
                 device_map={"": 0}, trust_remote_code=False, attn_implementation="sdpa", **model_assets)
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
        self.model = get_peft_model(base, LoraConfig(r=32, lora_alpha=32, lora_dropout=0.0,
                      target_modules=list(TARGET_MODULES), bias="none", task_type="CAUSAL_LM"))
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable or any(p.requires_grad and "lora_" not in n for n, p in self.model.named_parameters()):
            raise RuntimeError("Only fresh LoRA adapter parameters may be trainable")
        self.optimizer = torch.optim.AdamW(trainable, lr=5e-6, betas=(.9, .999), eps=1e-8, weight_decay=0)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda _: 1.0)
        self.model.config.use_cache = False
        self.runtime_information = {"packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "peft", "bitsandbytes")}, "cuda": torch.version.cuda,
                     "gpu": torch.cuda.get_device_name(0), "vram_bytes": torch.cuda.get_device_properties(0).total_memory,
                     "runtime_identity": dict(runtime_identity)}

    def policy_hash(self) -> str:
        from peft import get_peft_model_state_dict
        digest = hashlib.sha256()
        for name, value in sorted(get_peft_model_state_dict(self.model).items()):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().view(self.torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def sample(self, prompt: str, *, seed: int) -> dict[str, Any]:
        torch = self.torch
        self.model.eval()
        formatted = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                           tokenize=False, add_generation_prompt=True)
        encoded = self.tokenizer(formatted, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        self.model.config.use_cache = True
        # Per-completion forked RNG permits exact cache reuse without advancing global RNG differently.
        try:
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                with torch.no_grad():
                    outputs = self.model.generate(**encoded, do_sample=True, temperature=1.0, top_p=1.0,
                           top_k=0, repetition_penalty=1.0, max_new_tokens=2048,
                           pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id,
                           use_cache=True)
        finally:
            self.model.config.use_cache = False
        prompt_ids = encoded["input_ids"][0].tolist()
        completion_ids = outputs[0, len(prompt_ids):].tolist()
        if not completion_ids:
            raise RuntimeError("Generation engine returned no sampled tokens")
        ended = completion_ids[-1] == self.tokenizer.eos_token_id
        raw = self.tokenizer.decode(completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        with torch.no_grad():
            old = self.log_probs(prompt_ids, completion_ids).detach().cpu().tolist()
            with self.model.disable_adapter():
                reference = self.log_probs(prompt_ids, completion_ids).detach().cpu().tolist()
        return {"raw_output": raw, "prompt_token_ids": prompt_ids, "completion_token_ids": completion_ids,
                "old_log_probs": old, "reference_log_probs": reference, "finish_reason": "eos" if ended else "length",
                "truncated": not ended, "generated_tokens": len(completion_ids),
                "native_prompt_hash": content_hash(formatted)}

    def log_probs(self, prompt_ids: list[int], completion_ids: list[int]):
        torch = self.torch
        inputs = torch.tensor([prompt_ids + completion_ids], device=self.model.device, dtype=torch.long)
        if not prompt_ids:
            raise ValueError("Native prompt must include at least one token")
        output = self.model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False)
        logits = output.logits[0, len(prompt_ids)-1:-1].float()
        targets = torch.tensor(completion_ids, device=logits.device, dtype=torch.long)
        return torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    def update(self, rollouts: list[dict[str, Any]], rewards: list[dict[str, Any]]) -> dict[str, Any]:
        from ..training_cycle.policy_update.grpo_policy_update import group_advantages, dr_grpo_token_loss
        torch = self.torch
        advantages, group_stats = group_advantages([r["reward"] for r in rewards])
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss, total_kl, token_count = 0.0, 0.0, 0
        for rollout, advantage in zip(rollouts, advantages):
            current = self.log_probs(rollout["prompt_token_ids"], rollout["completion_token_ids"])
            old = torch.tensor(rollout["old_log_probs"], device=current.device)
            reference = torch.tensor(rollout["reference_log_probs"], device=current.device)
            # Every actual token, including every truncated completion, contributes.
            loss, kl = dr_grpo_token_loss(current, old, reference, advantage, max_length=2048, batch_size=32)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss: optimizer boundary must be rolled back")
            loss.backward()
            total_loss += float(loss.detach())
            total_kl += float(kl.detach().sum())
            token_count += current.numel()
        gradient_norm = torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], 1.0)
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("Nonfinite LoRA gradients: optimizer boundary must be rolled back")
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        return {"loss": total_loss, "kl": total_kl / token_count, "gradient_norm": float(gradient_norm),
                "logical_optimizer_updates": 1, "groups": group_stats,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved()}

    def capture_state(self) -> dict[str, Any]:
        from peft import get_peft_model_state_dict
        torch = self.torch
        return {"adapter": {n: v.detach().cpu() for n, v in get_peft_model_state_dict(self.model).items()},
                "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
                "rng": {"python": random.getstate(), "torch_cpu": torch.get_rng_state(),
                        "torch_cuda": torch.cuda.get_rng_state_all()}, "hardware": self.runtime_information}

    def restore_state(self, state: Mapping[str, Any]) -> None:
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(self.model, state["adapter"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        random.setstate(state["rng"]["python"])
        self.torch.set_rng_state(state["rng"]["torch_cpu"])
        self.torch.cuda.set_rng_state_all(state["rng"]["torch_cuda"])

    def export_adapter(self, destination: Path) -> None:
        self.model.save_pretrained(destination, safe_serialization=True)
        self.tokenizer.save_pretrained(destination)

    def close(self) -> None:
        self.model = None
        self.optimizer = None
        self.scheduler = None
        import gc
        gc.collect()
        self.torch.cuda.empty_cache()

rl_setup = setup_rl_run


def run_grpo_preflight(config: Mapping[str, Any], output_dir: str | Path) -> dict[str, Any]:
    """Exercise actual 7B/14B native sampling, one update and full-state restoration.

    These synthetic questions and fixed numerical rewards are a controlled hardware
    fixture, never scientific outcomes or a substitute for live reward-service checks.
    Call only on authorized hardware. CUDA absence blocks before model download.
    """
    from shared_procedures.configuration import configuration_hash, validate_configuration
    from ..training_cycle.rollouts.rollout_collection import collect_rollouts
    name = "grpo_update"
    try:
        import torch
    except ImportError:
        return {"name": name, "status": "blocked", "detail": "PyTorch is unavailable", "required": True,
                "evidence": {"live": False, "control_preflight": True, "scientific_results": False}}
    config_hash = configuration_hash(dict(config))
    common = {"configuration_hash": config_hash, "control_preflight": True, "scientific_results": False,
              "controlled_inputs": True, "reward_fixture": [-1.0, 0.0, 1.0, 1.5],
              "reward_service_verified_by_this_check": False}
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        return {"name": name, "status": "blocked", "detail": "Actual BF16-compatible CUDA model preflight requires GPU hardware",
                "required": True, "evidence": {**common, "live": False, "cuda_available": torch.cuda.is_available()}}
    validate_configuration(dict(config))
    for size in ("7b", "14b"):
        model = config["models"][size]
        if any(not isinstance(model.get(key), str) or not re.fullmatch(r"[0-9a-f]{40}", model[key])
               for key in ("revision", "tokenizer_revision")):
            return {"name": name, "status": "blocked", "detail": "Immutable actual model/tokenizer revisions are required",
                    "required": True, "evidence": {**common, "live": False}}
    run_root = Path(output_dir) / ("controlled_run_" + uuid.uuid4().hex)
    questions = [{"question_id": f"controlled_question_{index}", "question": f"Controlled runtime fixture: what is {index}+1?",
                  "reference_answer": str(index+1)} for index in range(8)]
    model_results = []
    runtime = None
    try:
        for size in ("7b", "14b"):
            model = config["models"][size]
            recipe = RLRecipe()
            identity = {"model_id": model["id"], "model_revision": model["revision"],
                        "tokenizer_revision": model["tokenizer_revision"], "trajectory": "controlled_grpo_preflight",
                        "source_hash": content_hash(questions), "recipe": recipe.to_dict(),
                        "implementation_hash": implementation_hash(), "execution_mode": "preflight_fixture",
                        "runtime_identity": {"configuration_hash": config_hash}}
            runtime = TransformersRLRuntime(model_id=model["id"], model_revision=model["revision"],
                 tokenizer_revision=model["tokenizer_revision"], recipe=recipe,
                 runtime_identity=identity["runtime_identity"])
            state = RLRunState(identity, run_root / size, runtime, recipe, attempt_id=uuid.uuid4().hex,
                               execution_mode="preflight_fixture")
            original_hash = state.policy_hash
            original_manifest = state.save()
            original = load_checkpoint(state.run_root / "checkpoints/step_000", expected_identity=identity)
            actual_rollouts = collect_rollouts(state, questions)
            controlled_rewards = [{"reward": reward, "fixture": True} for _ in range(8) for reward in (-1., 0., 1., 1.5)]
            metrics = runtime.update(actual_rollouts, controlled_rewards)
            gradient_norm = float(metrics["gradient_norm"])
            if metrics["logical_optimizer_updates"] != 1 or not gradient_norm > 0:
                raise RuntimeError("Controlled actual model update did not demonstrate nonzero LoRA gradients and one update")
            optimizer_steps = [float(value["step"]) for value in runtime.optimizer.state.values() if "step" in value]
            if not optimizer_steps or set(optimizer_steps) != {1.0} or runtime.scheduler.last_epoch != 1:
                raise RuntimeError("Actual optimizer/scheduler state does not demonstrate exactly one step")
            updated_hash = state.policy_hash
            if updated_hash == original_hash:
                raise RuntimeError("Controlled actual optimizer did not change adapter weights")
            state.completed_steps = 1
            updated_manifest = state.save()
            updated = load_checkpoint(state.run_root / "checkpoints/step_001", expected_identity=identity)
            # Deliberately disturb live state so the recovery check cannot pass vacuously.
            with torch.no_grad():
                next(p for p in runtime.model.parameters() if p.requires_grad).add_(0.125)
            runtime.optimizer.state.clear()
            runtime.scheduler.last_epoch += 7
            torch.manual_seed(127)
            random.seed(127)
            runtime.restore_state(updated)
            def equal_state(left, right):
                if torch.is_tensor(left):
                    return torch.is_tensor(right) and torch.equal(left.detach().cpu(), right.detach().cpu())
                if isinstance(left, dict):
                    return isinstance(right, dict) and left.keys() == right.keys() and all(equal_state(left[key], right[key]) for key in left)
                if isinstance(left, (tuple, list)):
                    return isinstance(right, (tuple, list)) and len(left) == len(right) and all(equal_state(a,b) for a,b in zip(left,right))
                return left == right
            if not equal_state(runtime.optimizer.state_dict(), updated["optimizer"]) or not equal_state(runtime.scheduler.state_dict(), updated["scheduler"]):
                raise RuntimeError("Full completed-step optimizer/scheduler tensors did not restore")
            if state.policy_hash != updated_hash:
                raise RuntimeError("Full completed-step adapter restore differs")
            runtime.restore_state(original)
            if state.policy_hash != original_hash or runtime.optimizer.state:
                raise RuntimeError("Original adapter/optimizer boundary was not restored")
            if runtime.scheduler.state_dict() != original["scheduler"]:
                raise RuntimeError("Original scheduler state was not restored")
            if not torch.equal(torch.get_rng_state(), original["rng"]["torch_cpu"]) or any(
                    not torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(), original["rng"]["torch_cuda"])):
                raise RuntimeError("Original Torch RNG state was not restored")
            if random.getstate() != original["rng"]["python"]:
                raise RuntimeError("Original Python RNG state was not restored")
            model_results.append({"model_id": model["id"], "question_ids": [r["question_id"] for r in actual_rollouts[::4]],
                "completion_ids": [r["completion_id"] for r in actual_rollouts], "logical_optimizer_updates": 1,
                "gradient_norm": gradient_norm, "policy_hash_before": original_hash, "policy_hash_after": updated_hash,
                "restored_original_policy_hash": state.policy_hash, "original_checkpoint_hash": original_manifest["checkpoint_hash"],
                "updated_checkpoint_hash": updated_manifest["checkpoint_hash"], "hardware": runtime.runtime_information})
            runtime.close()
            runtime = None
        result = {"name": name, "status": "passed", "detail": "Both actual models completed controlled eight-by-four update and exact full-state restore",
                  "required": True, "evidence": {**common, "live": True, "simulation": False, "models": model_results}}
    except Exception as exc:
        result = {"name": name, "status": "failed", "detail": f"{type(exc).__name__}: {exc}", "required": True,
                  "evidence": {**common, "live": True, "simulation": False, "models": model_results,
                               "error_type": type(exc).__name__}}
    finally:
        if runtime is not None:
            runtime.close()
    atomic_write_json(run_root / "report.json", result)
    return result
