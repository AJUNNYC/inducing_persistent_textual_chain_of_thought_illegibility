"""Isolated, frozen GPQA capability comparison; never a primary-suite domain.

The appended panel does not change training, the four-domain primary estimand,
window selection, multiplicity family, certification, or an existing raw cache.
References and source explanations never enter the public prompt. A comparison
is frozen before inference, uses one exact completed adapter and its untouched
base, and cannot choose a checkpoint based on GPQA outcomes.
"""
from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path
import re
from typing import Mapping

from evaluation.shared_model_evaluation.item_evaluation.capability_scoring.capability_scoring import (
    ANSWER_EXTRACTION_CONTRACT, parse_choice_label,
)
from evaluation.shared_model_evaluation.item_evaluation.native_output.output_generation import (
    GENERATION_SETTINGS, MODEL_IDS, NativeGenerator, validate_state, verify_adapter_payload,
)
from evaluation.shared_model_evaluation.item_evaluation.native_output.vllm_generation import (
    BACKEND, VLLMEvaluationGenerator,
)
from shared_procedures.identity import content_hash, file_hash
from shared_procedures.response_parsing import PARSER_VERSION, parse_native_output
from shared_procedures.storage import atomic_write_json, file_lock, notify_artifact, read_json

SOURCE_REPOSITORY = "Idavidrein/gpqa"
SOURCE_REVISION = "83022cefff930aea54f654c0b282e74b9eeda5c6"
VARIANTS = {"gpqa_main": 448, "gpqa_diamond": 198}
PANEL_VERSION = "supplemental_gpqa_panel_v1"
PROTOCOL_VERSION = "supplemental_gpqa_comparison_v1"
SCORER_VERSION = "supplemental_gpqa_exact_final_label_v1"
NATIVE_VERSION = "supplemental_gpqa_native_generation_v1"
VLLM_VERSION = "supplemental_gpqa_vllm_generation_v1"
PERMUTATION_VERSION = "sha256_order_fixed_seed_42_v1"
PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "evaluation_gpqa.txt"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")


def _seal(record: dict, field: str) -> dict:
    record[field] = content_hash(record)
    return record


def _check_seal(record: Mapping, field: str) -> None:
    if record.get(field) != content_hash({k: v for k, v in record.items() if k != field}):
        raise ValueError(f"GPQA {field} integrity differs")


def _permutation(variant: str, revision: str, index: int, question: str) -> list[int]:
    return sorted(range(4), key=lambda choice: content_hash({
        "version": PERMUTATION_VERSION, "seed": 42, "variant": variant,
        "revision": revision, "source_index": index, "question": question, "choice": choice,
    }))


def _source(source: Mapping, variant: str) -> dict:
    source = dict(source)
    if (variant not in VARIANTS or source.get("repo_id") != SOURCE_REPOSITORY
            or not _REVISION.fullmatch(str(source.get("revision", "")))
            or source.get("filename") != variant + ".csv"
            or not _SHA.fullmatch(str(source.get("csv_sha256", "")))
            or not _SHA.fullmatch(str(source.get("asset_manifest_hash", "")))):
        raise ValueError("GPQA requires its explicit variant and immutable official CSV provenance")
    return source


def load_official_gpqa(variant: str, cache_root: str | Path, *, revision: str = SOURCE_REVISION,
                       offline: bool = False) -> tuple[list[dict], dict]:
    """Provision the gated official source through the existing verified loader.

    No fallback mirror, moving revision, third-party reformatted data, or silent
    main/Diamond substitution is permitted. HF_TOKEN remains in private storage.
    """
    if variant not in VARIANTS or not _REVISION.fullmatch(revision):
        raise ValueError("Select gpqa_main or gpqa_diamond and an immutable source commit")
    from operations.assets import provision_asset
    asset = {"asset_id": "dataset_gpqa_supplement", "repo_id": SOURCE_REPOSITORY,
             "repo_type": "dataset", "revision": revision,
             "destination": "datasets/gpqa_supplement", "license_authorization_required": True}
    # The requested evaluation authorizes the download. This does not submit or
    # accept access terms: upstream gating still requires prior account access.
    manifest = provision_asset(asset, cache_root,
                              authorization={asset["asset_id"]: True}, offline=offline)
    path = Path(cache_root).expanduser().resolve() / asset["destination"] / revision / "full" / (variant + ".csv")
    if path.is_symlink() or not path.is_file():
        raise ValueError("Verified GPQA variant CSV is missing")
    digest = manifest["files"][variant + ".csv"]["sha256"]
    if file_hash(path) != digest:
        raise ValueError("Official GPQA CSV changed after provisioning")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return rows, {"repo_id": SOURCE_REPOSITORY, "revision": revision,
                  "filename": variant + ".csv", "csv_sha256": digest,
                  "asset_manifest_hash": manifest["manifest_hash"]}


def prepare_gpqa_panel(rows: list[dict], *, variant: str, source: Mapping) -> dict:
    """Use the entire official variant, permuting public choices once for all states."""
    source = _source(source, variant)
    if len(rows) != VARIANTS[variant]:
        raise ValueError("GPQA source population differs from the selected official variant")
    items = []
    for index, row in enumerate(rows):
        question = row.get("Question")
        answers = [row.get("Correct Answer")] + [row.get(f"Incorrect Answer {i}") for i in range(1, 4)]
        if (not isinstance(question, str) or not question.strip()
                or any(not isinstance(a, str) or not a.strip() for a in answers)
                or any(answers[0].strip() == a.strip() for a in answers[1:])):
            raise ValueError("GPQA source question/choices are malformed or the correct answer is duplicated")
        permutation = _permutation(variant, source["revision"], index, question)
        item = {"question_id": f"{variant}_{index:04d}_{content_hash(question)[:16]}",
                "domain": "gpqa", "question": question, "source_index": index,
                "source_row_hash": content_hash(row), "source_answers": answers,
                "choices": [{"label": chr(65 + i), "text": answers[j]} for i, j in enumerate(permutation)],
                "choice_permutation": permutation, "reference_answer": chr(65 + permutation.index(0))}
        items.append(_seal(item, "item_hash"))
    panel = _seal({"version": PANEL_VERSION, "scope": "supplemental_capability_only",
                   "variant": variant, "source": source, "seed": 42,
                   "permutation_version": PERMUTATION_VERSION,
                   "source_quirks": {"duplicate_incorrect_choices_preserved_at_source_indices": [
                       i["source_index"] for i in items if len({a.strip() for a in i["source_answers"][1:]}) < 3]},
                   "item_count": len(items), "items": items}, "panel_hash")
    validate_gpqa_panel(panel)
    return panel


def validate_gpqa_panel(panel: Mapping) -> None:
    _check_seal(panel, "panel_hash")
    variant = panel.get("variant")
    source = _source(panel.get("source", {}), variant)
    if (panel.get("version") != PANEL_VERSION or panel.get("scope") != "supplemental_capability_only"
            or panel.get("seed") != 42 or panel.get("permutation_version") != PERMUTATION_VERSION
            or panel.get("item_count") != VARIANTS[variant]
            or not isinstance(panel.get("items"), list) or len(panel["items"]) != VARIANTS[variant]):
        raise ValueError("GPQA panel differs from the fixed full-variant design")
    ids, questions = set(), set()
    for index, item in enumerate(panel["items"]):
        _check_seal(item, "item_hash")
        question, answers = item.get("question"), item.get("source_answers", [])
        if (not isinstance(question, str) or not question.strip() or not isinstance(answers, list)
                or len(answers) != 4 or any(not isinstance(a, str) or not a.strip() for a in answers)
                or any(answers[0].strip() == a.strip() for a in answers[1:])
                or not _SHA.fullmatch(str(item.get("source_row_hash", "")))):
            raise ValueError("Invalid GPQA item source payload")
        permutation = _permutation(variant, source["revision"], index, question)
        identifier = f"{variant}_{index:04d}_{content_hash(question)[:16]}"
        if (item.get("source_index") != index or item.get("domain") != "gpqa"
                or item.get("question_id") != identifier or identifier in ids
                or " ".join(question.split()) in questions
                or item.get("choice_permutation") != permutation
                or item.get("choices") != [{"label": chr(65 + i), "text": answers[j]} for i, j in enumerate(permutation)]
                or item.get("reference_answer") != chr(65 + permutation.index(0))):
            raise ValueError("GPQA panel has duplicates or mismatched choices/references")
        ids.add(identifier)
        questions.add(" ".join(question.split()))
    expected_quirks = {"duplicate_incorrect_choices_preserved_at_source_indices": [
        i["source_index"] for i in panel["items"] if len({a.strip() for a in i["source_answers"][1:]}) < 3]}
    if panel.get("source_quirks") != expected_quirks:
        raise ValueError("GPQA source quirks were repaired or misreported")


def render_gpqa_prompt(item: Mapping) -> tuple[str, str]:
    """Render only the public question/options, never source/reference fields."""
    if item.get("domain") != "gpqa" or not isinstance(item.get("question"), str):
        raise ValueError("Supplemental renderer requires a GPQA question")
    choices = item.get("choices")
    if (not isinstance(choices, list) or len(choices) != 4
            or [c.get("label") for c in choices] != list("ABCD")
            or any(not isinstance(c.get("text"), str) or not c["text"].strip() for c in choices)):
        raise ValueError("GPQA prompt requires four frozen public choices")
    template = PROMPT_PATH.read_text(encoding="utf-8")
    return template.format(question=item["question"], choices="\n".join(
        f"{c['label']}. {c['text']}" for c in choices)), content_hash({"version": PANEL_VERSION, "template": template})


def generation_contract(backend: str) -> dict:
    if backend not in {"native", "vllm"}:
        raise ValueError("Unknown GPQA measurement backend")
    root = Path(__file__).resolve().parents[2]
    implementations = (
        "src/evaluation/supplemental_gpqa.py", "scripts/run_supplemental_gpqa.py",
        "src/shared_procedures/response_parsing.py", "src/shared_procedures/identity.py",
        "src/shared_procedures/storage.py", "src/operations/assets.py",
        "src/evaluation/shared_model_evaluation/item_evaluation/capability_scoring/capability_scoring.py",
        "src/evaluation/shared_model_evaluation/item_evaluation/native_output/output_generation.py",
        "src/evaluation/shared_model_evaluation/item_evaluation/native_output/vllm_generation.py",
        "src/training/sft/individual_fit/sft_fit.py",
    )
    return {"version": NATIVE_VERSION if backend == "native" else VLLM_VERSION,
            "backend": None if backend == "native" else dict(BACKEND),
            "settings": dict(GENERATION_SETTINGS), "prompt_sha256": file_hash(PROMPT_PATH),
            "implementation_sha256": {name: file_hash(root / name) for name in implementations},
            "parser_version": PARSER_VERSION, "scorer_version": SCORER_VERSION,
            "answer_extraction": dict(ANSWER_EXTRACTION_CONTRACT)}


def comparison_states(adapter_path: str | Path, cache_root: str | Path) -> list[dict]:
    """Read the original adapter identity; do not rebind it to a new eval config.

    The exact adapter manifest already binds checkpoint, model, tokenizer, old
    training configuration, source and recipe. Appending an evaluation benchmark
    cannot make that fit scientifically different or require retraining it.
    """
    root = Path(adapter_path).resolve()
    if Path(adapter_path).is_symlink() or root.name != "final_adapter":
        raise ValueError("Select the exact published final_adapter, not a GPQA-selected checkpoint")
    manifest = read_json(root / "manifest.json")
    identity = manifest.get("identity", {})
    if not _SHA.fullmatch(str(identity.get("configuration_hash", ""))):
        raise ValueError("GPQA adapter lacks its immutable original training configuration")
    model = identity.get("model_id")
    if model not in MODEL_IDS or identity.get("model_name") != MODEL_IDS[model]:
        raise ValueError("GPQA comparison requires an original construction-model adapter")
    from training.sft.individual_fit.sft_fit import verify_adapter_manifest
    verify_adapter_manifest(root, identity)
    family, level = identity.get("family"), identity.get("level")
    if family not in {"rot13", "kazitext"} or type(level) is not int or level not in range(1, 11):
        raise ValueError("GPQA comparison currently accepts completed SFT final adapters")
    result = read_json(root.parent / "fit_result.json")
    if (result.get("status") != "complete" or result.get("identity") != identity
            or result.get("adapter_manifest_hash") != manifest.get("manifest_hash")
            or result.get("checkpoint_hash") != manifest.get("checkpoint_hash")):
        raise ValueError("GPQA adapter lacks its matching completed-fit report")
    shared = {"model": model, "model_revision": identity.get("model_revision"),
              "tokenizer_revision": identity.get("tokenizer_revision"),
              "cache_root": str(Path(cache_root).resolve()),
              "configuration_hash": identity.get("configuration_hash"),
              "runtime_hash": content_hash({"scope": PROTOCOL_VERSION})}
    base = {**shared, "family": "base", "state_id": "base_" + model,
            "checkpoint_hash": content_hash({"model_id": MODEL_IDS[model], "model_revision": shared["model_revision"]})}
    trained = {**shared, "family": family, "level": level,
               "state_id": f"{family}_{model}_level_{level:02d}", "adapter_path": str(root),
               "adapter_manifest_hash": manifest["manifest_hash"], "checkpoint_hash": manifest["checkpoint_hash"],
               "training_identity_hash": content_hash(identity)}
    for state in (base, trained):
        validate_state(state)
        if not all(_REVISION.fullmatch(str(state[k])) for k in ("model_revision", "tokenizer_revision")):
            raise ValueError("GPQA states require immutable model and tokenizer commits")
    verify_adapter_payload(trained)
    return [base, trained]


def freeze_gpqa_protocol(panel: Mapping, states: list[dict], *, backend: str) -> dict:
    validate_gpqa_panel(panel)
    contract = generation_contract(backend)
    states = deepcopy(states)
    for state in states:
        state["runtime_hash"] = content_hash(contract)
    protocol = _seal({"version": PROTOCOL_VERSION, "scope": "supplemental_capability_only",
                      "panel": deepcopy(panel), "states": states, "seed": 42,
                      "generation": contract, "checkpoint_selection": "explicit_completed_final_adapter_before_evaluation",
                      "reporting": "paired_descriptive_accuracy_no_primary_window_or_multiplicity_claim"}, "protocol_hash")
    validate_gpqa_protocol(protocol)
    return protocol


def validate_gpqa_protocol(protocol: Mapping, *, verify_adapters: bool = True) -> None:
    _check_seal(protocol, "protocol_hash")
    validate_gpqa_panel(protocol.get("panel", {}))
    contract = protocol.get("generation", {})
    backend = "vllm" if (contract.get("backend") or {}).get("name") == "vllm" else "native"
    if (protocol.get("version") != PROTOCOL_VERSION or protocol.get("scope") != "supplemental_capability_only"
            or protocol.get("seed") != 42 or contract != generation_contract(backend)
            or protocol.get("checkpoint_selection") != "explicit_completed_final_adapter_before_evaluation"
            or protocol.get("reporting") != "paired_descriptive_accuracy_no_primary_window_or_multiplicity_claim"):
        raise ValueError("GPQA protocol settings differ from the frozen supplemental design")
    states = protocol.get("states", [])
    if (not isinstance(states, list) or len(states) != 2 or states[0].get("family") != "base"
            or states[1].get("family") not in {"rot13", "kazitext"}):
        raise ValueError("GPQA requires one untouched base paired with one preselected final SFT adapter")
    for state in states:
        validate_state(state)
        if state.get("runtime_hash") != content_hash(contract):
            raise ValueError("GPQA state runtime binding differs")
    for key in ("model", "model_revision", "tokenizer_revision", "configuration_hash"):
        if states[0].get(key) != states[1].get(key):
            raise ValueError("GPQA base/adapter pairing differs from the original training identity")
    if not all(_REVISION.fullmatch(str(states[0][k])) for k in ("model_revision", "tokenizer_revision")):
        raise ValueError("GPQA model/tokenizer pins are not immutable")
    if verify_adapters:
        trained = states[1]
        manifest = verify_adapter_payload(trained)
        identity = manifest["identity"]
        if (Path(trained["adapter_path"]).name != "final_adapter"
                or trained.get("training_identity_hash") != content_hash(identity)
                or any(trained.get(k) != identity.get(k) for k in ("family", "level", "configuration_hash"))
                or trained["model"] != identity.get("model_id")):
            raise ValueError("GPQA adapter state changed the original fit condition/configuration")


class _GPQAIdentity:
    def identity(self, state, item, seed=42):
        validate_state(state)
        if seed != 42 or state not in self.protocol["states"]:
            raise ValueError("GPQA request differs from its frozen state/seed")
        frozen = self.items.get(item.get("question_id"))
        if frozen != item:
            raise ValueError("GPQA request item differs from its immutable panel")
        prompt, version = render_gpqa_prompt(item)
        return {"version": self.protocol["generation"]["version"], "state": dict(state),
                "question_id": item["question_id"], "public_prompt": prompt,
                "prompt_version": version, "protocol_hash": self.protocol["protocol_hash"], "seed": seed,
                "settings": dict(GENERATION_SETTINGS),
                "execution_mode": "fixture" if self.backend else "production",
                **({"backend": dict(BACKEND)} if isinstance(self, VLLMEvaluationGenerator) else {})}


class GPQANativeGenerator(_GPQAIdentity, NativeGenerator):
    def __init__(self, cache_root, protocol, *, backend=None):
        validate_gpqa_protocol(protocol)
        if protocol["generation"] != generation_contract("native"):
            raise ValueError("GPQA native generator cannot change a frozen backend")
        super().__init__(cache_root, backend=backend)
        self.protocol = deepcopy(protocol)
        self.items = {i["question_id"]: i for i in protocol["panel"]["items"]}


class GPQAVLLMGenerator(_GPQAIdentity, VLLMEvaluationGenerator):
    def __init__(self, cache_root, protocol):
        validate_gpqa_protocol(protocol)
        if protocol["generation"] != generation_contract("vllm"):
            raise ValueError("GPQA vLLM generator cannot change a frozen backend")
        super().__init__(cache_root)
        self.protocol = deepcopy(protocol)
        self.items = {i["question_id"]: i for i in protocol["panel"]["items"]}


def score_gpqa(item: Mapping, raw: Mapping) -> dict:
    prefix = raw.get("native_prefix", "")
    if prefix not in {"", "<think>", "<think>\n"}:
        raise ValueError("Invalid recorded native tokenizer prefill")
    parsed = parse_native_output(prefix + raw["raw_output"])
    reason = "truncated_output" if raw.get("finish_reason") == "length" else "malformed_output"
    answer = None
    if parsed["status"] == "valid" and raw.get("finish_reason") != "length":
        answer = parse_choice_label(parsed["final_answer"], list("ABCD"))
        reason = "malformed_answer" if answer is None else "correct" if answer == item["reference_answer"] else "incorrect"
    return {"status": "outcome", "answer": answer, "correct": answer is not None and answer == item["reference_answer"],
            "reason": reason, "raw_hash": raw["raw_hash"], "item_hash": item["item_hash"],
            "scorer_version": SCORER_VERSION, "parser_version": PARSER_VERSION}


def publish_immutable(path: str | Path, record: Mapping) -> None:
    with file_lock(str(path) + ".lock"):
        if Path(path).exists() and read_json(path) != record:
            raise ValueError("Frozen GPQA artifact already exists with different contents")
        atomic_write_json(path, record)
        notify_artifact(path, {"kind": "supplemental_gpqa", "scope": "supplemental_capability_only"})


def evaluate_gpqa(protocol: Mapping, cache_root: str | Path, *, generator=None) -> dict:
    """Resume the same frozen comparison; failures are missing, never wrong answers."""
    validate_gpqa_protocol(protocol)
    root = Path(cache_root) / protocol["protocol_hash"]
    backend = "vllm" if protocol["generation"]["backend"] else "native"
    generator = generator or (GPQAVLLMGenerator(root, protocol) if backend == "vllm" else GPQANativeGenerator(root, protocol))
    expected_version = protocol["generation"]["version"]
    records, summaries = [], {}
    for state in protocol["states"]:
        observed = []
        items = protocol["panel"]["items"]
        if generator.identity(state, items[0])["version"] != expected_version:
            raise ValueError("GPQA evaluator backend differs from its frozen protocol")
        size = getattr(generator, "batch_size", 1)
        for start in range(0, len(items), size):
            batch = items[start:start + size]
            if hasattr(generator, "generate_batch"):
                generator.generate_batch(state, batch, 42)
            for item in batch:
                raw = generator.generate(state, item, 42)
                score = score_gpqa(item, raw)
                record = _seal({"protocol_hash": protocol["protocol_hash"], "state_id": state["state_id"],
                                "question_id": item["question_id"], "raw_hash": raw["raw_hash"],
                                "simulation": raw.get("simulation", True), "score": score}, "record_hash")
                publish_immutable(root / "scores" / state["state_id"] / (item["question_id"] + ".json"), record)
                observed.append(record)
                records.append(record)
        summaries[state["state_id"]] = {"denominator": len(items), "observed": len(observed),
            "correct_count": sum(r["score"]["correct"] for r in observed),
            "accuracy": sum(r["score"]["correct"] for r in observed) / len(items)}
    base, trained = [s["state_id"] for s in protocol["states"]]
    report = _seal({"version": PROTOCOL_VERSION, "scope": "supplemental_capability_only", "status": "complete",
                   "protocol_hash": protocol["protocol_hash"], "panel_hash": protocol["panel"]["panel_hash"],
                   "variant": protocol["panel"]["variant"], "summaries": summaries, "records": records,
                   "accuracy_difference_trained_minus_base": summaries[trained]["accuracy"] - summaries[base]["accuracy"],
                   "simulation": any(r["simulation"] for r in records),
                   "primary_suite_changed": False, "checkpoint_selection_performed": False}, "report_hash")
    publish_immutable(root / "comparison.json", report)
    return report
