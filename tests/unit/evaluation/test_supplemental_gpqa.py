"""CPU fixtures for GPQA leakage, immutable pairing, and supplemental isolation."""
from copy import deepcopy
from pathlib import Path
import pytest

from evaluation.supplemental_gpqa import (
    SOURCE_REPOSITORY, SOURCE_REVISION, GPQANativeGenerator, GPQAVLLMGenerator, comparison_states,
    evaluate_gpqa, freeze_gpqa_protocol, generation_contract, prepare_gpqa_panel,
    publish_immutable, render_gpqa_prompt, score_gpqa, validate_gpqa_panel, validate_gpqa_protocol,
)
from evaluation.shared_model_evaluation.item_evaluation.native_output.output_generation import NativeGenerator
from shared_procedures.identity import content_hash, file_hash
from shared_procedures.storage import atomic_write_json
from training.sft.individual_fit.sft_fit import TARGET_MODULES


def rows():
    return [{"Question": f"Fixture question {i}?", "Correct Answer": f"right {i}",
             "Incorrect Answer 1": f"wrong one {i}", "Incorrect Answer 2": f"wrong two {i}",
             "Incorrect Answer 3": f"wrong three {i}", "Explanation": "PRIVATE_EXPLANATION_DO_NOT_RENDER"}
            for i in range(198)]


def panel():
    return prepare_gpqa_panel(rows(), variant="gpqa_diamond", source={
        "repo_id": SOURCE_REPOSITORY, "revision": SOURCE_REVISION,
        "filename": "gpqa_diamond.csv", "csv_sha256": "c" * 64, "asset_manifest_hash": "d" * 64})


def seal(record, field):
    record.pop(field, None)
    record[field] = content_hash(record)
    return record


def adapter(tmp_path):
    """Fake weight bytes are a CPU integrity fixture, never scientific evidence."""
    root = tmp_path / "fit" / "final_adapter"
    root.mkdir(parents=True)
    config = {"r": 8, "lora_alpha": 16, "lora_dropout": .05,
              "target_modules": list(TARGET_MODULES),
              "base_model_name_or_path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"}
    atomic_write_json(root / "adapter_config.json", config)
    (root / "adapter_model.safetensors").write_bytes(b"CPU test fixture, not valid model tensors")
    identity = {"family": "rot13", "model_id": "14b", "level": 10,
                "model_name": config["base_model_name_or_path"], "model_revision": "a" * 40,
                "tokenizer_revision": "b" * 40, "configuration_hash": "e" * 64,
                "execution_mode": "production", "original_training_recipe": "unchanged"}
    manifest = seal({"identity": identity, "checkpoint_hash": "f" * 64,
                     "files": {p.name: file_hash(p) for p in root.iterdir()}}, "manifest_hash")
    atomic_write_json(root / "manifest.json", manifest)
    atomic_write_json(root.parent / "fit_result.json", {"status": "complete", "identity": identity,
                       "adapter_manifest_hash": manifest["manifest_hash"], "checkpoint_hash": manifest["checkpoint_hash"]})
    return root


def protocol(tmp_path, backend="native"):
    return freeze_gpqa_protocol(panel(), comparison_states(adapter(tmp_path), tmp_path / "model_cache"), backend=backend)


def test_choice_permutation_preserves_reference_and_does_not_render_private_fields():
    prepared = panel()
    assert prepared == panel()
    assert {i["reference_answer"] for i in prepared["items"]} == set("ABCD")
    for source, item in zip(rows(), prepared["items"]):
        expected = next(c["text"] for c in item["choices"] if c["label"] == item["reference_answer"])
        assert expected == source["Correct Answer"]
        text, _ = render_gpqa_prompt({**item, "reference_solution": "SECRET_SOLUTION", "Explanation": source["Explanation"]})
        assert "PRIVATE_EXPLANATION" not in text and "SECRET_SOLUTION" not in text
        assert "reference_answer" not in text and "Correct Answer" not in text
        assert all(c["text"] in text for c in item["choices"])


def test_population_duplicates_and_choice_reference_mismatches_fail():
    values = rows()
    with pytest.raises(ValueError, match="population"):
        prepare_gpqa_panel(values[:-1], variant="gpqa_diamond", source=panel()["source"])
    values[1]["Question"] = values[0]["Question"]
    with pytest.raises(ValueError, match="duplicates"):
        prepare_gpqa_panel(values, variant="gpqa_diamond", source=panel()["source"])
    modified = panel()
    item = modified["items"][0]
    item["reference_answer"] = next(c for c in "ABCD" if c != item["reference_answer"])
    seal(item, "item_hash")
    seal(modified, "panel_hash")
    with pytest.raises(ValueError, match="mismatched"):
        validate_gpqa_panel(modified)


def test_official_duplicate_distractors_are_preserved_without_ambiguous_correct_choice():
    values = rows()
    values[3]["Incorrect Answer 2"] = values[3]["Incorrect Answer 1"]
    prepared = prepare_gpqa_panel(values, variant="gpqa_diamond", source=panel()["source"])
    assert prepared["item_count"] == 198
    assert prepared["source_quirks"]["duplicate_incorrect_choices_preserved_at_source_indices"] == [3]
    assert len({c["text"] for c in prepared["items"][3]["choices"]}) == 3
    values[3]["Incorrect Answer 1"] = values[3]["Correct Answer"]
    with pytest.raises(ValueError, match="correct answer is duplicated"):
        prepare_gpqa_panel(values, variant="gpqa_diamond", source=panel()["source"])


def test_old_training_configuration_is_reused_without_relabeling(tmp_path):
    frozen = protocol(tmp_path)
    base, trained = frozen["states"]
    assert base["configuration_hash"] == trained["configuration_hash"] == "e" * 64
    assert frozen["protocol_hash"] != trained["configuration_hash"]
    validate_gpqa_protocol(frozen)
    modified = deepcopy(frozen)
    modified["states"][1]["configuration_hash"] = "1" * 64
    modified["states"][0]["configuration_hash"] = "1" * 64
    seal(modified, "protocol_hash")
    with pytest.raises(ValueError, match="original fit"):
        validate_gpqa_protocol(modified)


def test_adapter_condition_corruption_and_nonfinal_selection_fail(tmp_path):
    frozen = protocol(tmp_path)
    modified = deepcopy(frozen)
    modified["states"][1].update(family="kazitext", state_id="kazitext_14b_level_10")
    seal(modified, "protocol_hash")
    with pytest.raises(ValueError, match="original fit"):
        validate_gpqa_protocol(modified)
    with pytest.raises(ValueError, match="final_adapter"):
        comparison_states(Path(frozen["states"][1]["adapter_path"]).parent, tmp_path)
    Path(frozen["states"][1]["adapter_path"], "adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="corrupt"):
        validate_gpqa_protocol(frozen)


def test_frozen_backend_source_code_and_variant_cannot_change(tmp_path, monkeypatch):
    frozen = protocol(tmp_path)
    modified = deepcopy(frozen)
    modified["generation"]["settings"]["max_new_tokens"] = 1024
    seal(modified, "protocol_hash")
    with pytest.raises(ValueError, match="settings"):
        validate_gpqa_protocol(modified)
    modified = deepcopy(frozen)
    modified["generation"]["implementation_sha256"]["src/shared_procedures/response_parsing.py"] = "1" * 64
    seal(modified, "protocol_hash")
    with pytest.raises(ValueError, match="settings"):
        validate_gpqa_protocol(modified)
    modified = panel()
    modified["variant"] = "gpqa_main"
    seal(modified, "panel_hash")
    with pytest.raises(ValueError, match="variant"):
        validate_gpqa_panel(modified)


def test_scorer_reads_only_final_field_and_counts_malformed_or_truncated_wrong():
    item = panel()["items"][0]
    label = item["reference_answer"]
    def raw(text, finish="stop"):
        return {"raw_output": text, "finish_reason": finish, "raw_hash": content_hash(text)}
    correct = f"<think>Explanation.</think>Final answer: {label}"
    assert score_gpqa(item, raw(correct))["correct"] is True
    assert score_gpqa(item, raw(correct, "length"))["reason"] == "truncated_output"
    wrong = next(c for c in "ABCD" if c != label)
    assert not score_gpqa(item, raw(f"<think>Answer {label}.</think>Final answer: {wrong}"))["correct"]
    assert not score_gpqa(item, raw(f"<think>Answer {label}.</think>ambiguous prose"))["correct"]


def test_paired_evaluation_is_immutable_and_does_not_change_primary_identity(tmp_path):
    frozen = protocol(tmp_path)
    base = frozen["states"][0]
    primary_item = {"question_id": "original", "domain": "gsm8k", "question": "2+2?"}
    primary_before = NativeGenerator(tmp_path / "primary").identity(base, primary_item)
    calls = []
    def backend(state, prompt, settings, seed):
        calls.append((state["state_id"], prompt, settings))
        return {"raw_output": "<think>Fixture.</think>Final answer: A", "finish_reason": "stop", "generated_tokens": 10}
    root = tmp_path / "supplement"
    generator = GPQANativeGenerator(root / frozen["protocol_hash"], frozen, backend=backend)
    report = evaluate_gpqa(frozen, root, generator=generator)
    assert report["simulation"] is True and report["primary_suite_changed"] is False
    assert report["checkpoint_selection_performed"] is False
    assert len(calls) == 396
    assert all(x["denominator"] == 198 for x in report["summaries"].values())
    assert [x[1] for x in calls[:198]] == [x[1] for x in calls[198:]]
    assert evaluate_gpqa(frozen, root, generator=generator) == report
    assert len(calls) == 396
    assert primary_before == NativeGenerator(tmp_path / "primary").identity(base, primary_item)
    with pytest.raises(ValueError, match="different contents"):
        publish_immutable(root / frozen["protocol_hash"] / "comparison.json", {"corrupt": True})


def test_vllm_supplement_identity_and_batch_use_the_frozen_public_prompt(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    frozen = protocol(tmp_path, backend="vllm")
    generator = GPQAVLLMGenerator(tmp_path / "vllm", frozen)
    state, item = frozen["states"][0], frozen["panel"]["items"][0]
    captured = []
    class Tokenizer:
        eos_token = "<eos>"
        def apply_chat_template(self, messages, **kwargs):
            captured.append(messages[0]["content"])
            return messages[0]["content"] + "<think>\n"
        def encode(self, *args, **kwargs):
            return [1, 2]
        def decode(self, *args, **kwargs):
            return "Fixture.</think>Final answer: A<eos>"
    params = []
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kw: params.append(kw) or kw))
    monkeypatch.setattr(generator, "_load", lambda state: None)
    generator._tokenizer, generator.eos_ids = Tokenizer(), [9]
    generator.engine = SimpleNamespace(generate=lambda inputs, settings, **kw: [
        SimpleNamespace(prompt_token_ids=p["prompt_token_ids"], outputs=[
            SimpleNamespace(token_ids=[3, 9], finish_reason="stop")]) for p in inputs])
    generator.generate_batch(state, [item])
    raw = generator.generate(state, item)
    assert captured == [render_gpqa_prompt(item)[0]]
    assert raw["identity"]["protocol_hash"] == frozen["protocol_hash"]
    assert raw["identity"]["version"] == frozen["generation"]["version"]
    assert params[0]["max_tokens"] == 3072
    assert raw["raw_output"] == "Fixture.</think>Final answer: A"
