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
from shared_procedures.identity import content_hash, derive_seed, file_hash
from shared_procedures.response_parsing import PARSER_VERSION, parse_native_output
from shared_procedures.storage import atomic_write_json, file_lock, notify_artifact, read_json

SOURCE_REPOSITORY = "Idavidrein/gpqa"
SOURCE_REVISION = "83022cefff930aea54f654c0b282e74b9eeda5c6"
VARIANTS = {"gpqa_main": 448, "gpqa_diamond": 198}
PANEL_VERSION = "supplemental_gpqa_panel_v1"
PROTOCOL_VERSION = "supplemental_gpqa_comparison_v1"
BASELINE_VERSION = "supplemental_gpqa_baseline_v1"
SCORER_VERSION = "supplemental_gpqa_exact_final_label_v1"
NATIVE_VERSION = "supplemental_gpqa_native_generation_v1"
VLLM_VERSION = "supplemental_gpqa_vllm_generation_v1"
VLLM_LONG_VERSION = "supplemental_gpqa_vllm_generation_long_32000_v1"
GENERATION_PROFILES = ("legacy_3072", "long_32000")
LONG_BACKEND = {**BACKEND, "max_model_len": 40960, "max_num_seqs": 8}
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


def generation_contract(backend: str, generation_profile: str = "legacy_3072") -> dict:
    if backend not in {"native", "vllm"}:
        raise ValueError("Unknown GPQA measurement backend")
    if generation_profile not in GENERATION_PROFILES:
        raise ValueError("Unknown GPQA generation profile")
    if generation_profile == "long_32000" and backend != "vllm":
        raise ValueError("GPQA long_32000 requires the frozen vLLM backend")
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
    contract = {"version": NATIVE_VERSION if backend == "native" else VLLM_VERSION,
            "backend": None if backend == "native" else dict(BACKEND),
            "settings": dict(GENERATION_SETTINGS), "prompt_sha256": file_hash(PROMPT_PATH),
            "implementation_sha256": {name: file_hash(root / name) for name in implementations},
            "parser_version": PARSER_VERSION, "scorer_version": SCORER_VERSION,
            "answer_extraction": dict(ANSWER_EXTRACTION_CONTRACT)}
    if generation_profile == "long_32000":
        contract.update(version=VLLM_LONG_VERSION, generation_profile=generation_profile,
                        backend=dict(LONG_BACKEND))
        contract["settings"]["max_new_tokens"] = 32000
    return contract


def _generation_profile(contract: Mapping) -> str:
    return contract.get("generation_profile", "legacy_3072")


# Audited predecessor 5e6be1b: only zero-control admission changed in the
# evaluator/state validator/SFT identity. Prompting, generation and scoring did not.
HISTORICAL_GPQA_IMPLEMENTATIONS = {'scripts/run_supplemental_gpqa.py': 'd2f6805c7211bd76744eab78d22db8f16bc3e2c517a41e5d3fd16c5897f5018c', 'src/evaluation/shared_model_evaluation/item_evaluation/capability_scoring/capability_scoring.py': '87614862e9ad33af1e8a12aa52bc4bd4dc492b7c69c8994b39fb212345af0383', 'src/evaluation/shared_model_evaluation/item_evaluation/native_output/output_generation.py': '1041cac7156c88916cf2c84120b7511d628d6b7eaf6aaa9c53fe84e99e6673f9', 'src/evaluation/shared_model_evaluation/item_evaluation/native_output/vllm_generation.py': 'c973ce8436ad0db2f52a33e7551becaeeb9466bc2308c8fd215a329bf2053433', 'src/evaluation/supplemental_gpqa.py': '07ff7235aadd0dec16e5e34c9f56bea84ae0e2e0bcdbb5f7da938fbd6c44f994', 'src/operations/assets.py': '898909c45b2dc50c8949f6c05df2a1ecbf085e91e1358cd376f043a612da4ff7', 'src/shared_procedures/identity.py': '42bdf7a349557a42d2bc14d842b7b3adcb485412b6ec3b2919d3f7dc2a828887', 'src/shared_procedures/response_parsing.py': 'fa80b63e8eb3e1e6626f347d444e0f6f3f22f9a95ed82a61833f64ac007a2ad3', 'src/shared_procedures/storage.py': '9c13b7c91ddd13667db96a541d24c75d6f5ffd2685ffec8b565bf408af4cf9f8', 'src/training/sft/individual_fit/sft_fit.py': '1a3a0b4c3e470bc83addc7a0eb8037ae1dc7609c7a06dbed146cf2f9c8ab4fd8'}


def _compatible_baseline_contract(old: Mapping, current: Mapping) -> bool:
    if old == current:
        return True
    if old.get("implementation_sha256") != HISTORICAL_GPQA_IMPLEMENTATIONS:
        return False
    if {k: v for k, v in old.items() if k != "implementation_sha256"} != {k: v for k, v in current.items() if k != "implementation_sha256"}:
        return False
    audited = {'src/evaluation/shared_model_evaluation/item_evaluation/native_output/output_generation.py': '088813f0b160acf7caadb524931fb8a5e74a00872eff8ed1dc5bb6c5b673efd1', 'src/training/sft/individual_fit/sft_fit.py': 'fd4b82d76f4884376cacabc3372e1019d3b5453c583dcb2b83112be76279b16b'}
    if any(current["implementation_sha256"].get(name) != digest for name, digest in audited.items()):
        return False
    changed = {name for name, digest in current["implementation_sha256"].items()
               if digest != HISTORICAL_GPQA_IMPLEMENTATIONS.get(name)}
    return changed <= {"src/evaluation/supplemental_gpqa.py",
                       "src/evaluation/shared_model_evaluation/item_evaluation/native_output/output_generation.py",
                       "src/training/sft/individual_fit/sft_fit.py"}


def _validate_generation_contract(contract: Mapping, backend: str, *, historical_baseline: bool = False) -> None:
    profile = _generation_profile(contract)
    current = generation_contract(backend, profile)
    if contract == current or (historical_baseline and _compatible_baseline_contract(contract, current)):
        return
    raise ValueError("GPQA protocol settings differ from the frozen supplemental design")


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
    if family not in {"rot13", "kazitext"} or type(level) is not int or level not in (range(0, 11) if family == "rot13" else range(1, 11)):
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


def baseline_state(model: str, cache_root: str | Path, *, model_revision: str,
                   tokenizer_revision: str) -> dict:
    """Bind an untouched official model without requiring a trained adapter.

    The baseline's model identity deliberately excludes training configuration:
    its raw answers can be reused by later comparisons with matching pins.
    """
    if (model not in MODEL_IDS or not _REVISION.fullmatch(str(model_revision))
            or not _REVISION.fullmatch(str(tokenizer_revision))):
        raise ValueError("GPQA baseline requires an official model and immutable model/tokenizer commits")
    model_identity = {"id": MODEL_IDS[model], "revision": model_revision,
                      "tokenizer_revision": tokenizer_revision}
    return {"state_id": "base_" + model, "family": "base", "model": model,
            "model_id": MODEL_IDS[model], "model_revision": model_revision,
            "tokenizer_revision": tokenizer_revision,
            "cache_root": str(Path(cache_root).resolve()),
            "checkpoint_hash": content_hash(model_identity),
            "configuration_hash": content_hash({"untouched_base": model_identity}),
            "runtime_hash": "pending_freeze"}


def freeze_gpqa_baseline(panel: Mapping, state: dict, *, backend: str,
                         generation_profile: str = "legacy_3072") -> dict:
    validate_gpqa_panel(panel)
    contract = generation_contract(backend, generation_profile)
    state = deepcopy(state)
    state["runtime_hash"] = content_hash(contract)
    protocol = _seal({"version": BASELINE_VERSION, "scope": "supplemental_capability_only",
                      "panel": deepcopy(panel), "states": [state], "seed": 42,
                      "generation": contract, "checkpoint_selection": "untouched_official_base_before_training",
                      "reporting": "baseline_descriptive_accuracy_no_primary_window_or_multiplicity_claim"},
                     "protocol_hash")
    validate_gpqa_protocol(protocol)
    return protocol


def freeze_gpqa_protocol(panel: Mapping, states: list[dict], *, backend: str,
                         baseline_protocol: Mapping | None = None,
                         generation_profile: str = "legacy_3072") -> dict:
    validate_gpqa_panel(panel)
    contract = generation_contract(backend, generation_profile)
    states = deepcopy(states)
    for state in states:
        state["runtime_hash"] = content_hash(contract)
    protocol = {"version": PROTOCOL_VERSION, "scope": "supplemental_capability_only",
                      "panel": deepcopy(panel), "states": states, "seed": 42,
                      "generation": contract, "checkpoint_selection": "explicit_completed_final_adapter_before_evaluation",
                      "reporting": "paired_descriptive_accuracy_no_primary_window_or_multiplicity_claim"}
    if baseline_protocol is not None:
        validate_gpqa_protocol(baseline_protocol)
        if baseline_protocol["version"] != BASELINE_VERSION:
            raise ValueError("GPQA reuse requires a frozen base-only protocol")
        if not _compatible_baseline_contract(baseline_protocol["generation"], contract):
            raise ValueError("GPQA baseline reuse differs in generation profile or implementation")
        protocol["baseline_reuse"] = {
            "protocol_hash": baseline_protocol["protocol_hash"],
            "panel_hash": baseline_protocol["panel"]["panel_hash"],
            "generation": deepcopy(baseline_protocol["generation"]),
            "state": deepcopy(baseline_protocol["states"][0]),
        }
    _seal(protocol, "protocol_hash")
    validate_gpqa_protocol(protocol)
    return protocol


def validate_gpqa_protocol(protocol: Mapping, *, verify_adapters: bool = True) -> None:
    _check_seal(protocol, "protocol_hash")
    validate_gpqa_panel(protocol.get("panel", {}))
    contract = protocol.get("generation", {})
    backend = "vllm" if (contract.get("backend") or {}).get("name") == "vllm" else "native"
    _validate_generation_contract(contract, backend, historical_baseline=protocol.get("version") == BASELINE_VERSION)
    base_only = protocol.get("version") == BASELINE_VERSION
    if (protocol.get("version") not in {PROTOCOL_VERSION, BASELINE_VERSION} or protocol.get("scope") != "supplemental_capability_only"
            or protocol.get("seed") != 42
            or protocol.get("checkpoint_selection") != ("untouched_official_base_before_training" if base_only else
                                                        "explicit_completed_final_adapter_before_evaluation")
            or protocol.get("reporting") != ("baseline_descriptive_accuracy_no_primary_window_or_multiplicity_claim" if base_only else
                                            "paired_descriptive_accuracy_no_primary_window_or_multiplicity_claim")):
        raise ValueError("GPQA protocol settings differ from the frozen supplemental design")
    states = protocol.get("states", [])
    if (not isinstance(states, list) or len(states) != (1 if base_only else 2)
            or any(not isinstance(s, dict) for s in states) or states[0].get("family") != "base"
            or (not base_only and states[1].get("family") not in {"rot13", "kazitext"})):
        raise ValueError("GPQA requires one untouched base paired with one preselected final SFT adapter")
    for state in states:
        validate_state(state)
        if state.get("runtime_hash") != content_hash(contract):
            raise ValueError("GPQA state runtime binding differs")
    if not all(_REVISION.fullmatch(str(states[0][k])) for k in ("model_revision", "tokenizer_revision")):
        raise ValueError("GPQA model/tokenizer pins are not immutable")
    if base_only:
        expected = baseline_state(states[0]["model"], states[0].get("cache_root", ""),
                                  model_revision=states[0]["model_revision"],
                                  tokenizer_revision=states[0]["tokenizer_revision"])
        expected["runtime_hash"] = content_hash(contract)
        if states[0] != expected or "baseline_reuse" in protocol:
            raise ValueError("GPQA baseline differs from the untouched official model identity")
        return
    for key in ("model", "model_revision", "tokenizer_revision", "configuration_hash"):
        if states[0].get(key) != states[1].get(key):
            raise ValueError("GPQA base/adapter pairing differs from the original training identity")
    reuse = protocol.get("baseline_reuse")
    if reuse is not None:
        # Reconstruct the entire baseline protocol rather than trusting a caller's
        # claimed reuse hash or relabeling old predictions as new evidence.
        if not isinstance(reuse, Mapping) or not isinstance(reuse.get("state"), dict):
            raise ValueError("GPQA baseline reuse is malformed")
        try:
            restored = freeze_gpqa_baseline(protocol["panel"], reuse["state"], backend=backend,
                                             generation_profile=_generation_profile(contract))
        except ValueError as error:
            raise ValueError("GPQA baseline reuse changed its original official model identity") from error
        # Restore the historical protocol verbatim rather than relabeling cached answers.
        restored["generation"] = deepcopy(reuse["generation"])
        restored["states"] = [deepcopy(reuse["state"])]
        restored.pop("protocol_hash", None)
        _seal(restored, "protocol_hash")
        validate_gpqa_protocol(restored)
        if (set(reuse) != {"protocol_hash", "panel_hash", "generation", "state"}
                or reuse["protocol_hash"] != restored["protocol_hash"]
                or reuse["panel_hash"] != protocol["panel"]["panel_hash"]
                or not _compatible_baseline_contract(reuse["generation"], contract)
                or any(reuse["state"].get(k) != states[0].get(k)
                       for k in ("model", "model_revision", "tokenizer_revision"))):
            raise ValueError("GPQA baseline reuse differs in panel, engine, prompt, seed, or model/tokenizer pins")
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
        reuse = self.protocol.get("baseline_reuse") if state["family"] == "base" else None
        return {"version": self.protocol["generation"]["version"], "state": dict(state),
                "question_id": item["question_id"], "public_prompt": prompt,
                "prompt_version": version, "protocol_hash": self.protocol["protocol_hash"], "seed": seed,
                "settings": dict(self.protocol["generation"]["settings"]),
                "execution_mode": "fixture" if self.backend else "production",
                **({"backend": deepcopy(self.protocol["generation"]["backend"])}
                   if isinstance(self, VLLMEvaluationGenerator) else {}),
                **({"state": deepcopy(reuse["state"]), "protocol_hash": reuse["protocol_hash"]} if reuse else {})}


class GPQANativeGenerator(_GPQAIdentity, NativeGenerator):
    def __init__(self, cache_root, protocol, *, backend=None):
        validate_gpqa_protocol(protocol)
        if protocol["generation"] != generation_contract("native", _generation_profile(protocol["generation"])):
            raise ValueError("GPQA native generator cannot change a frozen backend")
        super().__init__(cache_root, backend=backend)
        self.protocol = deepcopy(protocol)
        self.items = {i["question_id"]: i for i in protocol["panel"]["items"]}


class GPQAVLLMGenerator(_GPQAIdentity, VLLMEvaluationGenerator):
    def __init__(self, cache_root, protocol):
        validate_gpqa_protocol(protocol)
        if protocol["generation"] != generation_contract("vllm", _generation_profile(protocol["generation"])):
            raise ValueError("GPQA vLLM generator cannot change a frozen backend")
        super().__init__(cache_root)
        self.protocol = deepcopy(protocol)
        self.items = {i["question_id"]: i for i in protocol["panel"]["items"]}
        self.batch_size = protocol["generation"]["backend"]["batch_size"]

    @property
    def _long_profile(self):
        return _generation_profile(self.protocol["generation"]) == "long_32000"

    def _load(self, state):
        if not self._long_profile:
            return super()._load(state)
        key = (state["model"], state["model_revision"], state["tokenizer_revision"])
        if self.engine is not None:
            if key != self._loaded_identity:
                raise RuntimeError("Use a fresh process for the next model size")
            return
        import torch, transformers, vllm
        from transformers import AutoTokenizer
        from vllm import LLM
        from operations.assets import model_load_kwargs
        backend = self.protocol["generation"]["backend"]
        if (vllm.__version__, transformers.__version__, torch.__version__) != (
                backend["version"], backend["transformers_version"], backend["torch_version"]):
            raise RuntimeError("Evaluation engine dependencies differ from frozen contract")
        model_id, cache = MODEL_IDS[state["model"]], state["cache_root"]
        model_load_kwargs(model_id, state["model_revision"], cache, offline=True)
        model_load_kwargs(model_id, state["tokenizer_revision"], cache, tokenizer_only=True, offline=True)
        snapshots = Path(cache) / "hub" / ("models--" + model_id.replace("/", "--")) / "snapshots"
        model_path, tokenizer_path = snapshots / state["model_revision"], snapshots / state["tokenizer_revision"]
        self._model_context_limit = read_json(model_path / "config.json").get("max_position_embeddings")
        if (type(self._model_context_limit) is not int
                or self._model_context_limit < backend["max_model_len"]):
            raise ValueError("Frozen model context cannot contain the full GPQA long_32000 context")
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path), trust_remote_code=False, local_files_only=True)
        eos = read_json(model_path / "generation_config.json").get("eos_token_id", self._tokenizer.eos_token_id)
        self.eos_ids = eos if isinstance(eos, list) else [eos]
        if not self.eos_ids or any(type(token) is not int or token < 0 for token in self.eos_ids):
            raise ValueError("Frozen model lacks valid EOS stopping tokens")
        self.engine = LLM(model=str(model_path), tokenizer=str(tokenizer_path), dtype="bfloat16",
            trust_remote_code=False, generation_config="vllm",
            **{k: backend[k] for k in ("max_model_len", "max_num_seqs", "max_num_batched_tokens",
                "gpu_memory_utilization", "enable_prefix_caching", "enable_lora", "max_lora_rank")})
        self._loaded_identity = key

    def _validate_long_output(self, output, state, effective_seed):
        settings, backend = self.protocol["generation"]["settings"], self.protocol["generation"]["backend"]
        count, tokens = output.get("generated_tokens"), output.get("generated_token_ids")
        runtime = output.get("runtime", {})
        if (not isinstance(output.get("raw_output"), str)
                or not isinstance(output.get("decoded_generated_text"), str)
                or output.get("finish_reason") not in {"stop", "length"}
                or type(count) is not int or not 0 <= count <= settings["max_new_tokens"]
                or not isinstance(tokens, list) or len(tokens) != count
                or any(type(token) is not int or token < 0 for token in tokens)
                or (output["finish_reason"] == "length" and count != settings["max_new_tokens"])
                or output.get("effective_seed") != effective_seed
                or output.get("native_prefix") not in {"", "<think>", "<think>\n"}
                or not _SHA.fullmatch(str(output.get("native_chat_hash", "")))
                or not isinstance(runtime, dict)
                or any(runtime.get(k) != v for k, v in backend.items())
                or runtime.get("model_commit") != state["model_revision"]
                or runtime.get("tokenizer_commit") != state["tokenizer_revision"]
                or type(runtime.get("generation_batch_size")) is not int
                or not 1 <= runtime["generation_batch_size"] <= self.batch_size
                or type(runtime.get("prompt_tokens")) is not int or runtime["prompt_tokens"] < 0
                or runtime["prompt_tokens"] + settings["max_new_tokens"] > backend["max_model_len"]
                or type(runtime.get("model_context_limit")) is not int
                or runtime["model_context_limit"] < backend["max_model_len"]
                or output.get("runtime_observation_hash") != content_hash(runtime)):
            raise ValueError("GPQA long raw output violates its frozen stopping, token, or runtime contract")

    def _long_raw(self, state, item, seed=42):
        """Publish/read long outputs without the primary engine's 3072 validator."""
        identity = self.identity(state, item, seed)
        if state.get("adapter_path"):
            state_key = content_hash(state)
            root = Path(state["adapter_path"])
            fingerprint = tuple(sorted((str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns)
                for p in root.rglob("*") if p.is_file()))
            if self._verified_adapter_fingerprints.get(state_key) != fingerprint:
                verify_adapter_payload(state)
                self._verified_adapter_fingerprints[state_key] = fingerprint
        key = content_hash(identity)
        path = self.cache_root / "raw" / state["state_id"] / (key + ".json")
        effective_seed = derive_seed("native_evaluation", seed, state["state_id"], item["question_id"])
        with file_lock(str(path) + ".lock"):
            if path.exists():
                cached = read_json(path)
                if (cached.get("request_hash") != key or cached.get("identity") != identity
                        or cached.get("status") != "success" or cached.get("simulation") is not False
                        or cached.get("raw_hash") != content_hash(cached.get("raw_output", ""))
                        or cached.get("record_hash") != content_hash({k: v for k, v in cached.items() if k != "record_hash"})):
                    raise ValueError("Raw generation cache integrity failure")
                self._validate_long_output(cached, state, effective_seed)
                return cached
            output = self._generate_cuda(state, identity["public_prompt"], identity["settings"], effective_seed)
            self._validate_long_output(output, state, effective_seed)
            record = {**output, "status": "success", "simulation": False, "identity": identity,
                      "request_hash": key, "raw_hash": content_hash(output["raw_output"])}
            record["record_hash"] = content_hash(record)
            atomic_write_json(path, record)
            if read_json(path) != record:
                raise OSError("Raw generation publication did not verify")
            notify_artifact(path, {"kind": "evaluation_raw", "state_id": state["state_id"],
                                   "request_hash": key, "raw_hash": record["raw_hash"], "simulation": False})
            return record

    def generate_batch(self, state, items, seed=42):
        if not self._long_profile:
            return super().generate_batch(state, items, seed)
        items = list(items)
        if len(items) > self.batch_size:
            raise ValueError("GPQA request batch exceeds its frozen batch size")
        missing = []
        for item in items:
            identity = self.identity(state, item, seed)
            path = self.cache_root / "raw" / state["state_id"] / (content_hash(identity) + ".json")
            if path.exists():
                self._long_raw(state, item, seed)
            else:
                missing.append((item, identity))
        if not missing:
            return
        self._load(state)
        from vllm import SamplingParams
        settings, backend = self.protocol["generation"]["settings"], self.protocol["generation"]["backend"]
        adapter = None
        if state.get("adapter_path"):
            verify_adapter_payload(state)
            from vllm.lora.request import LoRARequest
            key = content_hash(state)
            if key not in self._adapters:
                self._adapters[key] = LoRARequest(key, len(self._adapters) + 1, state["adapter_path"])
            adapter = self._adapters[key]
        rendered = [self._tokenizer.apply_chat_template([{"role": "user", "content": i["public_prompt"]}],
            tokenize=False, add_generation_prompt=True) for _, i in missing]
        inputs = [{"prompt_token_ids": self._tokenizer.encode(p, add_special_tokens=False)} for p in rendered]
        if any(len(p["prompt_token_ids"]) + settings["max_new_tokens"] > backend["max_model_len"] for p in inputs):
            raise ValueError("Full prompt and response exceed context; truncation is forbidden")
        seeds = [derive_seed("native_evaluation", seed, state["state_id"], i["question_id"]) for _, i in missing]
        params = [SamplingParams(temperature=settings["temperature"], top_p=settings["top_p"], top_k=-1,
            repetition_penalty=settings["repetition_penalty"], max_tokens=settings["max_new_tokens"], seed=s,
            stop_token_ids=self.eos_ids, skip_special_tokens=False) for s in seeds]
        outputs = self.engine.generate(inputs, params, lora_request=adapter, use_tqdm=False)
        if len(outputs) != len(missing):
            raise RuntimeError("Response multiplicity changed")
        for (item, identity), prompt, inp, effective_seed, result in zip(missing, rendered, inputs, seeds, outputs):
            if list(result.prompt_token_ids) != inp["prompt_token_ids"] or len(result.outputs) != 1:
                raise RuntimeError("Prompt identity or completion multiplicity changed")
            completion = result.outputs[0]
            if (completion.finish_reason not in {"stop", "length"}
                    or len(completion.token_ids) > settings["max_new_tokens"]
                    or (completion.finish_reason == "length" and len(completion.token_ids) != settings["max_new_tokens"])):
                raise RuntimeError("Invalid stopping boundary or response budget")
            decoded = self._tokenizer.decode(completion.token_ids, skip_special_tokens=False)
            text = decoded
            if self._tokenizer.eos_token and text.endswith(self._tokenizer.eos_token):
                text = text[:-len(self._tokenizer.eos_token)]
            prefix = "<think>\n" if prompt.endswith("<think>\n") else ("<think>" if prompt.endswith("<think>") else "")
            runtime = {**backend, "model_commit": state["model_revision"],
                "tokenizer_commit": state["tokenizer_revision"], "generation_batch_size": len(missing),
                "prompt_tokens": len(inp["prompt_token_ids"]), "model_context_limit": self._model_context_limit}
            self._pending[effective_seed] = dict(raw_output=text, decoded_generated_text=decoded,
                generated_token_ids=list(completion.token_ids), generated_tokens=len(completion.token_ids),
                finish_reason=completion.finish_reason, native_prefix=prefix, native_chat_hash=content_hash(prompt),
                effective_seed=effective_seed, runtime=runtime, runtime_observation_hash=content_hash(runtime))
            self._long_raw(state, item, seed)

    def generate(self, state, item, seed=42):
        if not self._long_profile:
            return super().generate(state, item, seed)
        identity = self.identity(state, item, seed)
        path = self.cache_root / "raw" / state["state_id"] / (content_hash(identity) + ".json")
        if not path.exists():
            self.generate_batch(state, [item], seed)
        return self._long_raw(state, item, seed)


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
        # Opt-in paired reuse reads the original baseline namespace and validates
        # complete raw identities/checksums. Legacy paired cache paths stay intact.
        generator.cache_root = (Path(cache_root) / protocol["baseline_reuse"]["protocol_hash"]
                                if state["family"] == "base" and protocol.get("baseline_reuse") else root)
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
    base_only = protocol["version"] == BASELINE_VERSION
    base = protocol["states"][0]["state_id"]
    difference = None if base_only else summaries[protocol["states"][1]["state_id"]]["accuracy"] - summaries[base]["accuracy"]
    report = _seal({"version": protocol["version"], "scope": "supplemental_capability_only", "status": "complete",
                   "protocol_hash": protocol["protocol_hash"], "panel_hash": protocol["panel"]["panel_hash"],
                   "variant": protocol["panel"]["variant"], "summaries": summaries, "records": records,
                   "accuracy_difference_trained_minus_base": difference,
                   "simulation": any(r["simulation"] for r in records),
                   "primary_suite_changed": False, "checkpoint_selection_performed": False}, "report_hash")
    publish_immutable(root / ("baseline.json" if base_only else "comparison.json"), report)
    return report
