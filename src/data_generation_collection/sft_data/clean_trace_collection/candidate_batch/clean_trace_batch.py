"""Paired native generation, raw persistence and all-condition structural acceptance."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import os

from shared_procedures.identity import content_hash, derive_seed
from shared_procedures.response_parsing import parse_native_output
from shared_procedures.storage import atomic_write_json, read_json, notify_artifact
from data_generation_collection.sft_data.rot13.rot13_data_construction import eligible_word_spans, transform_rot13

GENERATION_SETTINGS = {"temperature": 0.6, "top_p": 0.95, "max_new_tokens": 3072,
                       "do_sample": True, "top_k": 0, "repetition_penalty": 1.0,
                       "system_prompt": None, "opacity_instruction": None}


class CleanDataPause(RuntimeError):
    """An operational pause at a complete-response boundary, never a rejection."""
    def __init__(self, progress: Mapping):
        self.progress = dict(progress)
        super().__init__("Clean-data operation paused at a durable boundary")


def native_completion(reasoning: str, final_answer: str) -> str:
    return "<think>" + reasoning + "</think>" + final_answer


def total_sequence_tokens(tokenizer, prompt: str, completion: str) -> int:
    from training.sft.individual_fit.sft_fit import render_sft_completion
    rendered, _ = render_sft_completion(prompt, completion, tokenizer)
    return len(tokenizer.encode(rendered, add_special_tokens=False, truncation=False))


class NativeCleanGenerator:
    """Real Transformers generation; load one frozen base at a time.

    ``models`` maps 7b/14b to immutable model/tokenizer specs. GPU/runtime and
    authorization preflights belong to the orchestrator before invoking this.
    """
    def __init__(self, models: Mapping, *, device_map="auto", torch_dtype="bfloat16", cache_root=None, stop_requested=None, batch_size=None):
        expected = {"7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"}
        if set(models) != set(expected) or any(spec.get("id", spec.get("name")) != expected[key] for key, spec in models.items()):
            raise ValueError("Only the two frozen untouched construction models are supported")
        self.models = {key: {**spec, "name": spec.get("id", spec.get("name"))} for key, spec in models.items()}
        self.device_map, self.dtype, self.cache_root = device_map, torch_dtype, cache_root
        self.active, self.model, self.tokenizer = None, None, None
        self.stop_requested = stop_requested
        self.batch_size = int(batch_size if batch_size is not None else os.getenv("CLEAN_GENERATION_BATCH_SIZE", "1"))
        if not 1 <= self.batch_size <= 64:
            raise ValueError("Clean generation batch size must be between 1 and 64")

    def _load(self, model_id):
        if self.active == model_id:
            return
        import gc
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from operations.assets import model_load_kwargs
        from data_generation_collection.sft_data.clean_trace_collection.candidate_manifest import validate_revision
        if self.model is not None:
            self.model = self.tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        spec = self.models[model_id]
        validate_revision(spec["revision"])
        validate_revision(spec["tokenizer_revision"])
        tokenizer_options = model_load_kwargs(spec["name"], spec["tokenizer_revision"], self.cache_root, tokenizer_only=True)
        model_options = model_load_kwargs(spec["name"], spec["revision"], self.cache_root)
        self.tokenizer = AutoTokenizer.from_pretrained(spec["name"], revision=spec["tokenizer_revision"], trust_remote_code=False, **tokenizer_options)
        self.model = AutoModelForCausalLM.from_pretrained(spec["name"], revision=spec["revision"],
                        device_map=self.device_map, torch_dtype=getattr(torch, self.dtype), trust_remote_code=False, **model_options)
        self.model.eval()
        self.active = model_id

    def generate_many(self, model_id: str, candidates, *, stop_requested=None):
        candidates = list(candidates)
        if not candidates:
            return
        request_stop = stop_requested or self.stop_requested
        if request_stop and request_stop():
            raise CleanDataPause({"phase": "before_model_load", "model_id": model_id,
                                  "example_id": candidates[0]["example_id"]})
        self._load(model_id)
        import torch
        from transformers import set_seed
        spec = self.models[model_id]
        if self.batch_size > 1:
            yield from self._generate_batches(model_id, candidates, request_stop)
            return
        for candidate in candidates:
            if request_stop and request_stop():
                raise CleanDataPause({"phase": "before_generation", "model_id": model_id,
                                      "example_id": candidate["example_id"]})
            seed = derive_seed(42, "clean_generation", candidate["example_id"], model_id)
            set_seed(seed)
            messages = [{"role": "user", "content": candidate["prompt"]}]
            rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            encoded = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
            with torch.inference_mode():
                output = self.model.generate(encoded, do_sample=True, temperature=0.6, top_p=0.95,
                      max_new_tokens=3072, top_k=0, repetition_penalty=1.0,
                      pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
            token_ids = output[0, encoded.shape[1]:].tolist()
            raw = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            eos = self.model.generation_config.eos_token_id
            eos_ids = eos if isinstance(eos, list) else [eos]
            stopped = bool(token_ids and token_ids[-1] in eos_ids)
            prefix = "<think>" if rendered.rstrip().endswith("<think>") and not raw.lstrip().startswith("<think>") else ""
            yield {"raw_generation": raw, "finish_reason": "stop" if stopped else "length",
                   "generated_token_count": len(token_ids), "generation_token_ids": token_ids,
                   "model_revision": spec["revision"], "tokenizer_revision": spec["tokenizer_revision"],
                   "model_name": spec["name"], "seed": seed, "native_reasoning_prefix": prefix,
                   "generation_settings": GENERATION_SETTINGS, "simulation": False, "execution_mode": "production", "runtime": {"torch": torch.__version__}}

    def _generate_batches(self, model_id, candidates, request_stop):
        import torch
        from transformers import LogitsProcessorList
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        spec = self.models[model_id]
        eos = self.model.generation_config.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        for start in range(0, len(candidates), self.batch_size):
            batch = candidates[start:start + self.batch_size]
            if request_stop and request_stop():
                raise CleanDataPause({"phase": "before_generation_batch", "model_id": model_id,
                                      "example_id": batch[0]["example_id"]})
            rendered = [self.tokenizer.apply_chat_template([{"role": "user", "content": c["prompt"]}],
                tokenize=False, add_generation_prompt=True) for c in batch]
            encoded = self.tokenizer(rendered, add_special_tokens=False, padding=True, return_tensors="pt").to(self.model.device)
            seeds = [derive_seed(42, "clean_generation", c["example_id"], model_id) for c in batch]
            sampler = IndependentSeedSampler(seeds, self.model.device, eos_ids, encoded.input_ids.shape[1])
            with torch.inference_mode():
                outputs = self.model.generate(**encoded, do_sample=True, temperature=0.6, top_p=0.95,
                    max_new_tokens=3072, top_k=0, repetition_penalty=1.0,
                    pad_token_id=self.tokenizer.pad_token_id, logits_processor=LogitsProcessorList([sampler]))
            completions = outputs[:, encoded.input_ids.shape[1]:].tolist()
            for row, (candidate, prompt, seed) in enumerate(zip(batch, rendered, seeds)):
                ids = completions[row]
                stop = next((index for index, token in enumerate(ids) if token in eos_ids), None)
                if stop is not None:
                    ids = ids[:stop + 1]
                raw = self.tokenizer.decode(ids, skip_special_tokens=True)
                prefix = "<think>" if prompt.rstrip().endswith("<think>") and not raw.lstrip().startswith("<think>") else ""
                yield {"raw_generation": raw, "finish_reason": "stop" if stop is not None else "length",
                    "generated_token_count": len(ids), "generation_token_ids": ids,
                    "model_revision": spec["revision"], "tokenizer_revision": spec["tokenizer_revision"],
                    "model_name": spec["name"], "seed": seed, "native_reasoning_prefix": prefix,
                    "generation_settings": GENERATION_SETTINGS, "simulation": False, "execution_mode": "production",
                    "runtime": {"torch": torch.__version__, "generation_backend": "transformers_seeded_batch_v1",
                                "generation_batch_size": len(batch), "generation_batch_row": row},
                    "generation_batch_end": row == len(batch) - 1}


class IndependentSeedSampler:
    """Sample each sequence with its own RNG, independent of batch companions.

    Apply the frozen temperature/top-p policy, then force the selected token.
    Transformers' subsequent warpers and sampling retain that one-token choice.
    No global RNG state is reported as an individual sequence's seed.
    """
    def __init__(self, seeds, device, eos_ids, prompt_width):
        import torch
        self.generators = [torch.Generator(device=device).manual_seed(seed) for seed in seeds]
        self.eos_ids = torch.tensor(eos_ids, device=device)
        self.prompt_width = prompt_width
        self.finished = torch.zeros(len(seeds), dtype=torch.bool, device=device)

    def __call__(self, input_ids, scores):
        import torch
        from transformers import TemperatureLogitsWarper, TopPLogitsWarper
        if scores.shape[0] != len(self.generators):
            raise ValueError("Seeded clean generation requires one sequence per prompt")
        warped = TopPLogitsWarper(0.95)(input_ids, TemperatureLogitsWarper(0.6)(input_ids, scores))
        probabilities = torch.softmax(warped, dim=-1)
        forced = torch.full_like(scores, -float("inf"))
        if input_ids.shape[1] > self.prompt_width:
            self.finished |= torch.isin(input_ids[:, -1], self.eos_ids)
        tokens = torch.cat([torch.multinomial(probabilities[row:row + 1], 1, generator=generator)
                            for row, generator in enumerate(self.generators)], dim=0)
        tokens = torch.where(self.finished[:, None], self.eos_ids[0], tokens)
        forced.scatter_(1, tokens, 0)
        return forced


def validate_clean_trace(candidate: Mapping, model_id: str, raw: Mapping, tokenizer,
                         *, kazitext, config_hash: str) -> dict:
    """No correctness/semantic filter. Infrastructure transform errors propagate."""
    from data_generation_collection.sft_data.clean_trace_collection.candidate_manifest import validate_revision
    validate_revision(raw["model_revision"])
    validate_revision(raw["tokenizer_revision"])
    if raw.get("generation_settings", GENERATION_SETTINGS) != GENERATION_SETTINGS:
        raise ValueError("Clean generation settings differ from frozen native policy")
    if type(raw.get("generated_token_count")) is not int or raw["generated_token_count"] < 0:
        raise ValueError("Invalid generated-token count provenance")
    raw = {**raw, "example_id": candidate["example_id"], "model_id": model_id,
           "prompt": candidate["prompt"], "source_content_hash": candidate["source_content_hash"],
           "raw_hash": content_hash(raw["raw_generation"]), "reasoning": "", "final_answer": "",
           "clean_trace_hash": None, "original_token_count": None,
           "eligible_word_count": None, "preflight_variants": [], "config_hash": config_hash,
           "simulation": raw.get("simulation", True), "execution_mode": raw.get("execution_mode", "fixture")}
    if raw.get("finish_reason") not in {"stop", "eos", "end_turn"}:
        return {**raw, "status": "rejected", "rejection_reasons": ["incomplete_generation"]}
    if raw.get("generated_token_count", 3073) > 3072:
        return {**raw, "status": "rejected", "rejection_reasons": ["generation_budget_exceeded"]}
    text = raw["raw_generation"]
    prefix = raw.get("native_reasoning_prefix", "")
    if prefix not in {"", "<think>"}:
        raise ValueError("Invalid declared native reasoning prefix")
    parsed = parse_native_output(prefix + text)
    if parsed["status"] != "valid":
        return {**raw, "status": "rejected", "rejection_reasons": ["malformed_reasoning_final_boundaries"], "parsed": parsed}
    reasoning, final = parsed["reasoning"], parsed["final_answer"]
    count = len(eligible_word_spans(reasoning))
    if count < 3:
        return {**raw, "status": "rejected", "reasoning": reasoning, "final_answer": final,
                "eligible_word_count": count, "rejection_reasons": ["fewer_than_three_eligible_words"]}
    original_tokens = total_sequence_tokens(tokenizer, candidate["prompt"], native_completion(reasoning, final))
    reasons, checks = [], []
    if original_tokens > 4096:
        reasons.append("clean_sequence_exceeds_4096")
    for level in range(1, 11):
        for method in ("rot13", "kazitext"):
            try:
                transformed = transform_rot13(reasoning, candidate["example_id"], level) if method == "rot13" else kazitext.transform(reasoning, candidate["example_id"], level, model_id=model_id)
            except ValueError as error:
                # Protection violations are ordinary all-condition source rejection.
                # Configuration/loading/provider failures are not swallowed.
                if "integrity failure" not in str(error):
                    raise
                reasons.append(f"{method}_{level:02d}_protected_integrity")
                checks.append({"method": method, "level": level, "status": "rejected", "reason": str(error)})
                continue
            if transformed.get("status") != "verified" or transformed.get("source_hash") != content_hash(reasoning):
                raise ValueError("Transformation handoff has invalid source identity")
            total = total_sequence_tokens(tokenizer, candidate["prompt"], native_completion(transformed["reasoning"], final))
            if total > 4096:
                reasons.append(f"{method}_{level:02d}_sequence_exceeds_4096")
            checks.append({"method": method, "level": level, "content_hash": transformed["content_hash"],
                          "token_count": total, "changed_word_fraction": transformed["changed_word_fraction"],
                          "protected_content_valid": transformed["protected_content_valid"],
                          "repeated_masks": transformed.get("repeated_masks", [])})
    return {**raw, "example_id": candidate["example_id"], "model_id": model_id,
            "prompt": candidate["prompt"], "source_content_hash": candidate["source_content_hash"],
            "raw_hash": content_hash(text), "reasoning": reasoning, "final_answer": final,
            "clean_trace_hash": content_hash({"reasoning": reasoning, "final_answer": final}),
            "eligible_word_count": count, "original_token_count": original_tokens,
            "preflight_variants": checks, "config_hash": config_hash,
            "status": "accepted" if not reasons else "rejected", "rejection_reasons": reasons}


def process_clean_batch(candidates, generator, tokenizers: Mapping, *, kazitext,
                        output_dir, config_hash: str, stop_requested=None) -> list[dict]:
    if set(tokenizers) != {"7b", "14b"}:
        raise ValueError("Both construction tokenizers are required")
    candidates = list(candidates)
    ids = [c["example_id"] for c in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("Candidate batch IDs must be unique")
    root = Path(output_dir)
    simulation = not isinstance(generator, NativeCleanGenerator)
    execution_mode = "fixture" if simulation else "production"
    raw_by_model = {model: {} for model in tokenizers}
    results = []
    stop_requested = stop_requested or getattr(generator, "stop_requested", None)

    def progress():
        return {"raw_completed_ids": {model: list(records) for model, records in raw_by_model.items()},
                "paired_completed_ids": [record["example_id"] for record in results],
                "simulation": simulation, "execution_mode": execution_mode}

    def check_stop(phase, example_id=None, model_id=None):
        if stop_requested and stop_requested():
            raise CleanDataPause({**progress(), "phase": phase, "example_id": example_id, "model_id": model_id})
    for model in ("7b", "14b"):
        pending = []
        for c in candidates:
            path = root / "raw" / execution_mode / model / (content_hash(c["example_id"]) + ".json")
            if path.exists():
                record = read_json(path)
                if (record["candidate_hash"] != content_hash(c) or record["config_hash"] != config_hash
                    or record.get("execution_mode") != execution_mode or record.get("simulation") is not simulation
                    or record["raw_hash"] != content_hash(record["raw_generation"])):
                    raise ValueError("Existing raw generation cache identity mismatch")
                notify_artifact(path, {"kind": "clean_raw", "example_id": c["example_id"], "model_id": model, "config_hash": config_hash, "raw_hash": record["raw_hash"]})
                raw_by_model[model][c["example_id"]] = record
            else:
                pending.append(c)
        if pending:
            check_stop("before_generation", pending[0]["example_id"], model)
        if pending and isinstance(generator, NativeCleanGenerator):
            generated = generator.generate_many(model, pending, stop_requested=stop_requested)
        elif pending and hasattr(generator, "generate_many"):
            generated = generator.generate_many(model, pending)
        else:
            generated = (generator(c, model, dict(GENERATION_SETTINGS)) for c in pending)
        generated = iter(generated)
        draining_batch = False
        for c in pending:
            if not draining_batch:
                check_stop("before_generation", c["example_id"], model)
            try:
                raw = dict(next(generated))
            except CleanDataPause as error:
                raise CleanDataPause({**error.progress, **progress()}) from error
            except StopIteration as error:
                raise RuntimeError("Generator failed to return each pending response") from error
            if not isinstance(raw.get("raw_generation"), str):
                raise ValueError("Generator must return the full raw generation")
            if hasattr(generator, "models") and (raw.get("model_revision") != generator.models[model]["revision"] or raw.get("tokenizer_revision") != generator.models[model]["tokenizer_revision"]):
                raise ValueError("Generator raw model/tokenizer provenance differs from frozen runtime")
            if raw.get("generation_settings", GENERATION_SETTINGS) != GENERATION_SETTINGS:
                raise ValueError("Generator returned an unauthorized generation configuration")
            record = {**raw, "example_id": c["example_id"], "model_id": model,
                      "candidate_hash": content_hash(c), "config_hash": config_hash,
                      "simulation": simulation, "execution_mode": execution_mode,
                      "raw_hash": content_hash(raw["raw_generation"]), "generation_settings": dict(GENERATION_SETTINGS)}
            path = root / "raw" / execution_mode / model / (content_hash(c["example_id"]) + ".json")
            atomic_write_json(path, record)  # Always persisted before parsing.
            if read_json(path) != record:
                raise IOError("Raw generation readback mismatch")
            notify_artifact(path, {"kind": "clean_raw", "example_id": c["example_id"], "model_id": model, "config_hash": config_hash, "raw_hash": record["raw_hash"],
                "defer_until_batch_end": "generation_batch_end" in raw and not raw["generation_batch_end"]})
            draining_batch = "generation_batch_end" in raw and not raw["generation_batch_end"]
            raw_by_model[model][c["example_id"]] = record
        if next(generated, None) is not None:
            raise RuntimeError("Generator returned extra unrequested responses")
    for c in candidates:
        check_stop("before_pair_validation", c["example_id"])
        traces = {model: validate_clean_trace(c, model, raw_by_model[model][c["example_id"]], tokenizers[model], kazitext=kazitext, config_hash=config_hash) for model in ("7b", "14b")}
        accepted = all(t["status"] == "accepted" for t in traces.values())
        result = {"example_id": c["example_id"], "prompt": c["prompt"],
                  "source_content_hash": c["source_content_hash"], "candidate": c,
                  "traces": traces, "status": "accepted" if accepted else "rejected",
                  "config_hash": config_hash, "simulation": simulation, "execution_mode": execution_mode}
        path = root / ("clean_traces" if accepted else "rejected_responses") / execution_mode / (content_hash(c["example_id"]) + ".json")
        if path.exists() and read_json(path) != result:
            raise ValueError("Existing pair differs; versioned repair required")
        atomic_write_json(path, result)
        if read_json(path) != result:
            raise IOError("Pair artifact readback mismatch")
        notify_artifact(path, {"kind": "clean_pair", "example_id": c["example_id"], "status": result["status"], "config_hash": config_hash})
        results.append(result)
    return results

clean_trace_batch = process_clean_batch
