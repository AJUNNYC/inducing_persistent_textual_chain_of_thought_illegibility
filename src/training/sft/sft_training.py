"""Explicitly selected independent fits over a fully certified forty-condition corpus."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .individual_fit.sft_fit import (
    LEVELS, MODELS, SFTFitSpec, SFTInputError, dataset_hash, run_sft_fit,
    validate_datasets, verify_fit_result,
)


def build_worklist(config: Mapping[str, Any], datasets: Sequence[Mapping[str, Any]], *, handoff_certificate=None) -> list[dict[str, Any]]:
    """Bind every verified condition exactly once to its untouched immutable base.

    Datasets contain train_path/validation_path JSONL and optionally already
    loaded train_records/validation_records. Actual content is checked before
    scheduling. Complete source IDs must agree across all forty conditions.
    """
    from shared_procedures.configuration import configuration_hash, sft_training_budget
    config_hash = configuration_hash(dict(config))
    producer_hash = config_hash
    if handoff_certificate is not None:
        from .data_handoff import validate_handoff
        validate_handoff(handoff_certificate, config)
        producer_hash = handoff_certificate["producer_configuration_hash"]
    if len(datasets) != 40:
        raise SFTInputError("The SFT worklist requires all forty verified datasets")
    conditions: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    train_ids = validation_ids = None
    source_by_model = {}
    shared_prompts = None
    shared_source_manifest = None
    for bundle in datasets:
        key = (bundle.get("family"), bundle.get("model_id"), bundle.get("level"))
        if key in conditions or key[0] not in LEVELS or key[1] not in MODELS or type(key[2]) is not int or not 1 <= key[2] <= 10:
            raise SFTInputError("Duplicate or unknown dataset condition")
        if bundle.get("status") not in ("verified", "complete") or bundle.get("config_hash", bundle.get("configuration_hash")) != producer_hash:
            raise SFTInputError("Dataset bundle is unverified or has changed configuration")
        train, validation = load_dataset_bundle(bundle)
        model_config = config["models"][key[1]]
        if model_config.get("id", model_config.get("name")) != MODELS[key[1]]:
            raise SFTInputError("Construction model substitution is forbidden")
        spec = SFTFitSpec(key[0], key[1], key[2], model_config["revision"],
                          model_config["tokenizer_revision"], config_hash, dataset_hash(train, validation),
                          training_budget=sft_training_budget(dict(config)))
        if handoff_certificate is not None:
            from dataclasses import replace
            spec = replace(spec, producer_configuration_hash=producer_hash,
                           handoff_certificate_hash=handoff_certificate["certificate_hash"])
            validate_handoff(handoff_certificate, config, spec)
        validate_datasets(spec, train, validation)
        source = {record["example_id"]: (record["prompt"], record["clean_trace_hash"], record["final_answer"])
                  for record in [*train, *validation]}
        if key[1] in source_by_model and source_by_model[key[1]] != source:
            raise SFTInputError("Clean traces, prompts or final answers changed across a model's conditions")
        source_by_model[key[1]] = source
        prompts = {identifier: values[0] for identifier, values in source.items()}
        if shared_prompts is not None and shared_prompts != prompts:
            raise SFTInputError("Prompt bytes differ between source models")
        shared_prompts = prompts
        source_manifest_hashes = {record["source_manifest_hash"] for record in [*train, *validation]}
        if len(source_manifest_hashes) != 1:
            raise SFTInputError("One condition mixed different frozen accepted pools")
        source_manifest_hash = next(iter(source_manifest_hashes))
        if shared_source_manifest is not None and shared_source_manifest != source_manifest_hash:
            raise SFTInputError("SFT conditions came from different frozen accepted pools")
        shared_source_manifest = source_manifest_hash
        this_train = tuple(record["example_id"] for record in train)
        this_validation = tuple(record["example_id"] for record in validation)
        if train_ids is None:
            train_ids, validation_ids = this_train, this_validation
        elif this_train != train_ids or this_validation != validation_ids:
            raise SFTInputError("Ordered source IDs or train/validation partitions differ across SFT conditions")
        conditions[key] = {"spec": spec.identity(), "dataset": dict(bundle)}
    result = []
    for family in ("rot13", "kazitext"):
        for model_id in ("7b", "14b"):
            for level in range(1, 11):
                key = (family, model_id, level)
                if key not in conditions:
                    raise SFTInputError("A frozen condition is missing")
                result.append({**conditions[key], "family": family, "model_id": model_id, "level": level,
                               "run_id": conditions[key]["spec"]["run_id"], "initialization": "untouched_base_or_exact_full_checkpoint"})
    return select_worklist(config, result)


def select_worklist(config: Mapping[str, Any], worklist: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Only declared fits may run; all forty conditions remain in the corpus."""
    from shared_procedures.configuration import sft_execution_selection
    expected = {(row['family'], row['model_id'], row['level'])
                for row in sft_execution_selection(dict(config))}
    return [dict(task) for task in worklist
            if (task['family'], task['model_id'], task['level']) in expected]


def load_dataset_bundle(bundle: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import json
    result = []
    for split in ("train", "validation"):
        if f"{split}_records" in bundle:
            records = [dict(record) for record in bundle[f"{split}_records"]]
        else:
            path = Path(bundle[f"{split}_path"])
            records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            expected_hash = bundle.get("content_hashes", {}).get(split)
            if expected_hash:
                import hashlib
                if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
                    raise SFTInputError("Persisted SFT JSONL differs from published content hash")
        result.append(records)
    return result[0], result[1]


def run_family_queue(family: str, worklist: Sequence[Mapping[str, Any]], output_root: str | Path,
                     config: Mapping[str, Any], *, preflight_report: Mapping[str, Any] | None = None,
                     fit_runner: Callable[..., dict[str, Any]] = run_sft_fit,
                     release_gpu: Callable[[str], None] | None = None,
                     should_pause: Callable[[], bool] | None = None,
                     handoff_certificate=None) -> dict[str, Any]:
    """Sequential queue: a failed fit stops only its own family.

    A custom runner is an explicit dependency-injection boundary for declared
    integration fixtures, never evidence of live scientific execution. The
    production runner checks persistent full-state/output compatibility.
    """
    from shared_procedures.storage import atomic_write_json
    if family not in LEVELS:
        raise SFTInputError("Unknown SFT queue")
    tasks = [task for task in worklist if task.get("family") == family]
    from shared_procedures.configuration import sft_execution_selection, sft_training_budget
    expected = {(row['model_id'], row['level']) for row in sft_execution_selection(dict(config))
                if row['family'] == family}
    actual = {(task.get("model_id"), task.get("level")) for task in tasks}
    if not expected or len(tasks) != len(expected) or actual != expected:
        raise SFTInputError("Each family queue must contain its declared independent fits (twenty for the historical scope)")
    completed, outcomes = [], []
    status = "complete"
    failure = None
    try:
        for task in tasks:
            spec = SFTFitSpec.from_dict(task["spec"])
            if ((spec.family, spec.model_id, spec.level) != (family, task["model_id"], task["level"])
                    or task['spec'] != spec.identity()):
                raise SFTInputError("Task metadata differs from exact fit identity")
            if spec.training_budget != sft_training_budget(dict(config)):
                raise SFTInputError("Task training budget differs from the declared configuration")
            output = Path(output_root) / family / spec.model_id / f"level_{spec.level:02d}"
            train, validation = load_dataset_bundle(task["dataset"])
            try:
                handoff_args = {"handoff_certificate": handoff_certificate} if handoff_certificate is not None else {}
                result = fit_runner(spec, train, validation, output, config,
                                    preflight_report=preflight_report, should_pause=should_pause, **handoff_args)
                if result.get("status") == "paused":
                    outcomes.append(result)
                    status = "paused"
                    break
                if result.get("status") != "complete" or result.get("run_id") != spec.run_id:
                    raise SFTInputError("Child fit did not return exact verified completion")
                if fit_runner is run_sft_fit:
                    result = verify_fit_result(output, result["identity"])
                outcomes.append(result)
                completed.append(spec.run_id)
            except Exception as error:
                status = "failed"
                failure = {"run_id": spec.run_id, "error_type": type(error).__name__, "detail": str(error)}
                break
    finally:
        if release_gpu is not None:
            release_gpu(family)
    report = {"family": family, "status": status, "required_fits": len(expected), "completed_fits": len(completed),
              "completed_run_ids": completed, "outcomes": outcomes, "failure": failure,
              "execution_mode": "production" if fit_runner is run_sft_fit else "injected_fixture"}
    atomic_write_json(Path(output_root) / family / "queue_report.json", report)
    return report


def sft_training(config: Mapping[str, Any], datasets: Sequence[Mapping[str, Any]], output_root: str | Path,
                 *, families: Sequence[str] | None = None,
                 preflight_report: Mapping[str, Any] | None = None,
                 fit_runner: Callable[..., dict[str, Any]] = run_sft_fit,
                 release_gpu: Callable[[str], None] | None = None,
                 should_pause: Callable[[], bool] | None = None) -> dict[str, Any]:
    """Run selected family queues; external workers may run families concurrently.

    Single-process default is sequential and cannot oversubscribe one GPU. Each
    independently failed family preserves its own results while the other runs.
    """
    from shared_procedures.configuration import sft_execution_selection
    available_families = {row['family'] for row in sft_execution_selection(dict(config))}
    if families is None:
        families = tuple(family for family in LEVELS if family in available_families)
    if len(set(families)) != len(families) or any(family not in available_families for family in families):
        raise SFTInputError("Invalid family selection")
    worklist = build_worklist(config, datasets)
    queues = {family: run_family_queue(family, worklist, output_root, config,
              preflight_report=preflight_report, fit_runner=fit_runner,
              release_gpu=release_gpu, should_pause=should_pause) for family in families}
    selected_families = {task['family'] for task in worklist}
    status = "complete" if set(queues) == selected_families and all(queue["status"] == "complete" for queue in queues.values()) else "partial"
    return {"status": status, "required_fits": len(worklist),
            "completed_fits": sum(queue["completed_fits"] for queue in queues.values()),
            "queues": queues, "worklist": [{key: task[key] for key in ("family", "model_id", "level", "run_id", "initialization")} for task in worklist]}


run_sft_training = sft_training
