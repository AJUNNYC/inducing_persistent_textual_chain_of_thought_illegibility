"""Native CUDA inference and immutable, state-specific generation caching.

No retries are made for ordinary model outputs. The literal prompts are an H.9
implementation convention; tests/references never enter the public renderer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import re

from shared_procedures.identity import content_hash, derive_seed, file_hash
from shared_procedures.storage import atomic_write_json, read_json, file_lock, notify_artifact

GENERATION_VERSION = "native_generation_v1"
GENERATION_SETTINGS = {"temperature": 0.6, "top_p": 0.95, "max_new_tokens": 3072,
                       "do_sample": True, "top_k": 0, "repetition_penalty": 1.0}
MODEL_IDS = {"7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
             "14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"}
PROMPT_ROOT = Path(__file__).resolve().parents[5] / "prompts"


def question_id(item: Mapping[str, Any]) -> str:
    value = item.get("question_id", item.get("item_id"))
    if not isinstance(value, str) or not value:
        raise ValueError("Missing stable question_id/item_id")
    return value


def canonical_domain(domain: str) -> str:
    return {"arc_easy": "arc", "arc-easy": "arc", "mmlu_redux": "mmlu",
            "mmlu_redux_2_0": "mmlu", "sanitized_mbpp": "mbpp"}.get(domain, domain)


def public_question(item: Mapping[str, Any]) -> str:
    """Return task and approved public fields only, never the whole item record."""
    domain = canonical_domain(item["domain"])
    text = item.get("question", item.get("prompt", item.get("text")))
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Missing public task question")
    if domain in {"mmlu", "arc"}:
        choices = item.get("choices")
        if isinstance(choices, dict) and {"label", "text"} <= choices.keys():
            choices = list(zip(choices["label"], choices["text"]))
        elif isinstance(choices, dict):
            choices = list(choices.items())
        rendered = []
        for index, choice in enumerate(choices or []):
            if isinstance(choice, dict):
                label, value = choice["label"], choice["text"]
            elif isinstance(choice, (tuple, list)) and len(choice) == 2:
                label, value = choice
            else:
                label, value = chr(65 + index), choice
            if not re.fullmatch(r"[A-Za-z0-9]", str(label)) or not isinstance(value, str):
                raise ValueError("Invalid frozen public choice")
            rendered.append(f"{label}. {value}")
        if not rendered:
            raise ValueError("Multiple-choice task has no public options")
        return text + "\n\nChoices:\n" + "\n".join(rendered)
    if domain == "mbpp":
        interfaces = item.get("public_interface", item.get("interface"))
        interfaces = interfaces if isinstance(interfaces, list) else [interfaces]
        approved = []
        for interface in interfaces:
            if isinstance(interface, dict):
                if interface.get("status", "verified") != "verified":
                    raise ValueError("Unverified MBPP interface")
                interface = interface.get("signature")
            if not isinstance(interface, str) or not interface.startswith("def "):
                raise ValueError("Missing approved function signature")
            if "\n" in interface or not interface.endswith(":") or "#" in interface:
                raise ValueError("Interface must be one bare signature without body/comment")
            approved.append(interface)
        if not approved:
            raise ValueError("Missing approved public interfaces")
        return text + "\n\nRequired public function interface:\n" + "\n".join(approved)

    if domain != "gsm8k":
        raise ValueError(f"Unknown evaluation domain {domain}")
    return text


def render_prompt(item: Mapping[str, Any]) -> tuple[str, str]:
    domain = canonical_domain(item["domain"])
    prompt_domain = "arc_easy" if domain == "arc" else domain
    template = (PROMPT_ROOT / f"evaluation_{prompt_domain}.txt").read_text()
    question = public_question(item)
    if domain in {"mmlu", "arc"}:
        task, choices = question.rsplit("\n\nChoices:\n", 1)
        result = template.format(question=task, choices=choices)
    elif domain == "mbpp":
        task, interface = question.rsplit("\n\nRequired public function interface:\n", 1)
        result = template.format(question=task, interface=interface)
    else:
        result = template.format(question=question)
    return result, content_hash({"version": GENERATION_VERSION, "template": template})


def validate_state(state: Mapping[str, Any]) -> None:
    required = ("state_id", "model", "model_revision", "tokenizer_revision", "checkpoint_hash", "runtime_hash")
    if any(not isinstance(state.get(key), str) or not state[key] for key in required):
        raise ValueError("State lacks immutable checkpoint/model/tokenizer/runtime identity")
    if not re.fullmatch(r"(?:base_(?:7b|14b)|(?:rot13|kazitext)_(?:7b|14b)_level_(?:0[1-9]|10)|rl_(?:7b|14b)_step_(?:0[1-9]0|100))", state["state_id"]):
        raise ValueError("State ID is outside the frozen scientific matrix")
    if state["model"] not in MODEL_IDS:
        raise ValueError("Only the fixed 7b and 14b construction models are permitted")
    if state.get("model_id", MODEL_IDS[state["model"]]) != MODEL_IDS[state["model"]]:
        raise ValueError("Construction model substitution is forbidden")
    for key in ("model_revision", "tokenizer_revision"):
        if state[key] in {"main", "master", "latest"}:
            raise ValueError("Immutable model/tokenizer revisions are required")
    family = state.get("family", "base")
    if family == "base":
        if state["state_id"] != "base_" + state["model"] or state.get("adapter_path"):
            raise ValueError("Untouched baseline cannot contain an adapter or mismatched size")
    elif family in {"rot13", "kazitext"}:
        level = state.get("level")
        label = f"level_{level:02d}" if isinstance(level, int) else str(level)
        if state["state_id"] != f"{family}_{state['model']}_{label}":
            raise ValueError("SFT state metadata differs from its frozen matrix identity")
    elif family == "rl":
        if state["state_id"] != f"rl_{state['model']}_step_{int(state.get('step',-1)):03d}":
            raise ValueError("RL state metadata differs from the original snapshot")
    else:
        raise ValueError("Unknown scientific state family")
    if family != "base" and not state.get("adapter_path"):
        raise ValueError("Trained state lacks its exact adapter path")


def verify_adapter_payload(state: Mapping[str, Any]) -> dict | None:
    """Verify the exact published scientific adapter, including all file bytes."""
    if state.get("family", "base") == "base":
        return None
    root = Path(state["adapter_path"])
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Scientific adapter directory is missing or unsafe")
    rl = state["family"] == "rl"
    manifest = read_json(root / ("snapshot.json" if rl else "manifest.json"))
    hash_field = "artifact_hash" if rl else "manifest_hash"
    declared = manifest.get(hash_field)
    if declared != content_hash({k:v for k,v in manifest.items() if k != hash_field}):
        raise ValueError("Adapter manifest integrity failure")
    if declared != state.get("adapter_manifest_hash"):
        raise ValueError("State descriptor differs from published adapter identity")
    checkpoint = manifest.get("full_checkpoint_hash" if rl else "checkpoint_hash")
    if checkpoint != state["checkpoint_hash"]:
        raise ValueError("Adapter is not derived from the declared full checkpoint")
    identity = manifest.get("identity", {})
    model_id = identity.get("model_id") if rl else identity.get("model_name")
    if (model_id != MODEL_IDS[state["model"]] or identity.get("model_revision") != state["model_revision"]
        or identity.get("tokenizer_revision") != state["tokenizer_revision"]):
        raise ValueError("Adapter changed the declared original model/tokenizer")
    if identity.get("execution_mode") == "fixture":
        raise ValueError("A fixture adapter cannot enter native scientific evaluation")
    if rl and (manifest.get("step") != state.get("step") or manifest.get("status") != "complete"):
        raise ValueError("RL snapshot changed the declared original scientific step")
    files = manifest.get("files", {})
    if not {"adapter_config.json", "adapter_model.safetensors"} <= files.keys():
        raise ValueError("Adapter lacks actual safe weights and PEFT configuration")
    for relative, expected in files.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe scientific adapter file inventory")
        target = root / path
        digest = expected["sha256"] if isinstance(expected, dict) else expected
        if target.is_symlink() or not target.is_file() or file_hash(target) != digest:
            raise ValueError("Adapter file bytes are missing or corrupt")
        if isinstance(expected, dict) and target.stat().st_size != expected["size_bytes"]:
            raise ValueError("Adapter file size is corrupt")
    return manifest


class NativeGenerator:
    """One exact output, shared between scoring and all three judgments.

    ``backend`` is injectable for isolated tests; production defaults to actual
    CUDA Transformers inference and never replaces missing hardware with fixtures.
    """
    def __init__(self, cache_root: str | Path, *, backend=None):
        self.cache_root = Path(cache_root)
        self.backend = backend
        self.simulation = backend is not None
        self._loaded_identity = None
        self._verified_adapter_fingerprints = {}
        self._model = self._tokenizer = None

    def identity(self, state: Mapping[str, Any], item: Mapping[str, Any], seed: int = 42) -> dict:
        validate_state(state)
        if seed not in {42, 43, 44} or (seed != 42 and state.get("family", "base") != "base"):
            raise ValueError("Only baselines may use diagnostic seeds 43/44")
        prompt, version = render_prompt(item)
        return {"version": GENERATION_VERSION, "state": dict(state), "question_id": question_id(item),
                "public_prompt": prompt, "prompt_version": version, "seed": seed,
                "settings": dict(GENERATION_SETTINGS), "execution_mode": "fixture" if self.backend else "production"}

    def generate(self, state: Mapping[str, Any], item: Mapping[str, Any], seed: int = 42) -> dict:
        identity = self.identity(state, item, seed)
        if self.backend is None and state.get("adapter_path"):
            state_key = content_hash(state)
            root = Path(state["adapter_path"])
            fingerprint = tuple(sorted((str(path.relative_to(root)), path.stat().st_size, path.stat().st_mtime_ns)
                for path in root.rglob("*") if path.is_file()))
            if self._verified_adapter_fingerprints.get(state_key) != fingerprint:
                verify_adapter_payload(state)
                self._verified_adapter_fingerprints[state_key] = fingerprint
        key = content_hash(identity)
        path = self.cache_root / "raw" / state["state_id"] / (key + ".json")
        with file_lock(str(path) + ".lock"):
            if path.exists():
                cached = read_json(path)
                if (cached.get("request_hash") == key and cached.get("identity") == identity
                    and cached.get("simulation") is self.simulation
                    and isinstance(cached.get("raw_output"), str)
                    and cached.get("raw_hash") == content_hash(cached["raw_output"])
                    and cached.get("record_hash") == content_hash({k:v for k,v in cached.items() if k != "record_hash"})):
                    return cached
                # Corruption is preserved for diagnosis; do not overwrite claimed evidence.
                raise ValueError("Raw generation cache integrity failure")
            backend = self.backend or self._generate_cuda
            output = backend(state, identity["public_prompt"], identity["settings"],
                             derive_seed("native_evaluation", seed, state["state_id"], question_id(item)))
            if (not isinstance(output, dict) or not isinstance(output.get("raw_output"), str)
                or output.get("finish_reason") not in {"stop", "length"}
                or type(output.get("generated_tokens")) is not int
                or not 0 <= output["generated_tokens"] <= GENERATION_SETTINGS["max_new_tokens"]):
                raise ValueError("Inference backend omitted raw output or mandatory stopping/token metadata")
            record = {**output, "status": "success", "simulation": self.backend is not None, "identity": identity, "request_hash": key,
                      "raw_hash": content_hash(output["raw_output"])}
            record["record_hash"] = content_hash(record)
            atomic_write_json(path, record)
            if read_json(path) != record:
                raise OSError("Raw generation publication did not verify")
            notify_artifact(path, {"kind": "evaluation_raw", "state_id": state["state_id"],
                                   "request_hash": key, "raw_hash": record["raw_hash"], "simulation": record["simulation"]})
            return record

    def _generate_cuda(self, state: Mapping[str, Any], prompt: str, settings: dict, seed: int) -> dict:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        if not torch.cuda.is_available():
            raise RuntimeError("Native model inference requires compatible CUDA hardware")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("Native runtime requires validated BF16 support")
        identity = content_hash(dict(state))
        if self._loaded_identity != identity:
            self._model = self._tokenizer = None
            torch.cuda.empty_cache()
            model_id = MODEL_IDS[state["model"]]
            from operations.assets import model_load_kwargs
            tokenizer_files = model_load_kwargs(model_id, state["tokenizer_revision"], state.get("cache_root"), tokenizer_only=True)
            model_files = model_load_kwargs(model_id, state["model_revision"], state.get("cache_root"))
            tokenizer = AutoTokenizer.from_pretrained(model_id, revision=state["tokenizer_revision"], trust_remote_code=False, **tokenizer_files)
            model = AutoModelForCausalLM.from_pretrained(model_id, revision=state["model_revision"],
                torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=False, **model_files)
            if state.get("adapter_path"):
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, state["adapter_path"], is_trainable=False)
            if any(str(device) in {"cpu", "disk"} for device in getattr(model, "hf_device_map", {}).values()):
                raise RuntimeError("Native CUDA runtime must not silently offload model layers")
            model.eval()
            self._model, self._tokenizer, self._loaded_identity = model, tokenizer, identity
        tokenizer, model = self._tokenizer, self._model
        rendered = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(rendered, return_tensors="pt").to(next(model.parameters()).device)
        set_seed(seed)
        with torch.inference_mode():
            generated = model.generate(**encoded, **settings, pad_token_id=tokenizer.eos_token_id)
        tokens = generated[0, encoded["input_ids"].shape[1]:]
        decoded_text = tokenizer.decode(tokens, skip_special_tokens=False)
        text = decoded_text
        eos = tokenizer.eos_token
        if eos and text.endswith(eos):
            text = text[:-len(eos)]
        prefill = rendered.rsplit(tokenizer.bos_token or "\0", 1)[-1]
        # The actual tokenizer may prefill the opening tag. Retain this known
        # literal prefix only; never infer or manufacture reasoning boundaries.
        if prefill.endswith("<think>\n") or prefill.endswith("<think>"):
            native_prefix = "<think>" + ("\n" if prefill.endswith("\n") else "")
        else:
            native_prefix = ""
        import platform
        import transformers
        runtime = {"python":platform.python_version(), "torch":torch.__version__,
                   "transformers":transformers.__version__, "cuda":torch.version.cuda,
                   "dtype":str(next(model.parameters()).dtype),
                   "devices":{str(index):torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())},
                   "model_commit":getattr(model.config,"_commit_hash",None),
                   "tokenizer_commit":tokenizer.init_kwargs.get("_commit_hash")}
        return {"raw_output": text, "decoded_generated_text":decoded_text,
                "generated_token_ids":tokens.tolist(), "runtime":runtime,
                "runtime_observation_hash":content_hash(runtime), "generated_tokens": len(tokens), "finish_reason":
                "length" if len(tokens) >= settings["max_new_tokens"] and tokens[-1].item() != tokenizer.eos_token_id else "stop",
                "native_chat_hash": content_hash(rendered), "native_prefix": native_prefix, "effective_seed": seed}
