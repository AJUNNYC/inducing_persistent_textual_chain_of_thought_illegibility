"""Freeze common source IDs and construct all forty independent SFT datasets."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from shared_procedures.identity import content_hash
from shared_procedures.storage import atomic_write_json, read_json, notify_artifact
from data_generation_collection.sft_data.rot13.rot13_data_construction import transform_rot13
from data_generation_collection.sft_data.clean_trace_collection.candidate_batch.clean_trace_batch import native_completion, total_sequence_tokens
from data_generation_collection.sft_data.clean_trace_collection.clean_trace_collection import collect_clean_traces


def validate_accepted_pool(pool: dict, config_hash: str | None = None) -> dict:
    if pool.get("status") != "verified" or len(pool.get("pairs", [])) != 10100:
        raise ValueError("Frozen source pool must contain exactly 10100 verified pairs")
    if pool.get("content_hash") != content_hash(pool["pairs"]):
        raise ValueError("Accepted source pool hash mismatch")
    if config_hash is not None and pool.get("config_hash") != config_hash:
        raise ValueError("Accepted source/configuration mismatch")
    train, validation = pool["train_ids"], pool["validation_ids"]
    ids = [r["example_id"] for r in pool["pairs"]]
    if len(train) != 10000 or len(validation) != 100 or len(set(ids)) != 10100 or set(train + validation) != set(ids) or set(train) & set(validation):
        raise ValueError("Frozen common split/count/identity invariants failed")
    from data_generation_collection.sft_data.clean_trace_collection.candidate_manifest import validate_revision
    for pair in pool["pairs"]:
        if pair.get("status") != "accepted" or set(pair["traces"]) != {"7b", "14b"}:
            raise ValueError("Frozen pair invalid")
        if pair["source_content_hash"] != content_hash(pair["prompt"]):
            raise ValueError("Frozen prompt hash mismatch")
        for model, trace in pair["traces"].items():
            validate_revision(trace["model_revision"])
            validate_revision(trace["tokenizer_revision"])
            if trace.get("status") != "accepted" or trace.get("prompt") != pair["prompt"] or trace.get("example_id") != pair["example_id"] or trace.get("model_id") != model:
                raise ValueError("Frozen model-specific source identity invalid")
            if trace["clean_trace_hash"] != content_hash({"reasoning": trace["reasoning"], "final_answer": trace["final_answer"]}):
                raise ValueError("Frozen clean-trace hash mismatch")
            checks = trace.get("preflight_variants", [])
            if {(v["method"], v["level"]) for v in checks} != {(m, k) for m in ("rot13", "kazitext") for k in range(1, 11)}:
                raise ValueError("Frozen trace missing all-condition acceptance checks")
            if trace["original_token_count"] > 4096 or any(v.get("token_count", 4097) > 4096 or not v.get("protected_content_valid") for v in checks):
                raise ValueError("Frozen source length/protection invariant failed")
    return pool


def load_sft_records(path) -> list[dict]:
    path = Path(path)
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]
    value = read_json(path)
    return value if isinstance(value, list) else value["records"]


def _publish_records(path: Path, records: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if load_sft_records(path) != records:
            raise ValueError("Existing transformed artifact differs; explicit versioned repair required")
        digest = content_hash(path.read_bytes())
        notify_artifact(path, {"kind": "sft_dataset", "sha256": digest, "record_count": len(records)})
        return digest
    handle, temporary = tempfile.mkstemp(prefix=".dataset_", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    reopened = load_sft_records(path)
    if reopened != records:
        raise IOError("Written transformed dataset readback mismatch")
    digest = content_hash(path.read_bytes())
    notify_artifact(path, {"kind": "sft_dataset", "sha256": digest, "record_count": len(records)})
    return digest


def build_sft_record(pair, trace, transformed, tokenizer, *, model_id: str, family: str,
                     level: int, split: str, source_manifest_hash: str, config_hash: str) -> dict:
    """Shared record boundary after structural acceptance; exact native target."""
    example_id = pair["example_id"]
    if split not in {"train", "validation"} or model_id not in {"7b", "14b"} or family not in {"rot13", "kazitext"} or level not in range(1, 11):
        raise ValueError("Invalid SFT record condition")
    if trace["example_id"] != example_id or trace["model_id"] != model_id or trace["prompt"] != pair["prompt"] or trace["status"] != "accepted":
        raise ValueError("SFT record source/model/prompt identity mismatch")
    preflight = next(v for v in trace["preflight_variants"] if v["method"] == family and v["level"] == level)
    if (transformed.get("status") != "verified" or transformed.get("source_hash") != content_hash(trace["reasoning"])
        or transformed["content_hash"] != content_hash(transformed["reasoning"])
        or transformed["content_hash"] != preflight["content_hash"]):
        raise ValueError("Frozen transform differs from candidate acceptance; stop without replacing IDs")
    completion = native_completion(transformed["reasoning"], trace["final_answer"])
    total = total_sequence_tokens(tokenizer, pair["prompt"], completion)
    if total > 4096 or total != preflight["token_count"] or not transformed["protected_content_valid"]:
        raise ValueError("Frozen transform failed token/protection preflight identity")
    return {"example_id": example_id, "prompt": pair["prompt"], "completion": completion,
            "reasoning": transformed["reasoning"], "final_answer": trace["final_answer"],
            "model_id": model_id, "model": model_id, "method": family, "family": family,
            "level": level, "split": split, "clean_trace_hash": trace["clean_trace_hash"],
            "source_content_hash": pair["source_content_hash"], "seed": 42,
            "identity": {"config_hash": config_hash, "model": model_id, "method": family, "level": level,
                         "source_manifest_hash": source_manifest_hash, "clean_trace_hash": trace["clean_trace_hash"]},
            "model_revision": trace["model_revision"], "tokenizer_revision": trace["tokenizer_revision"],
            "original_token_count": trace["original_token_count"], "transformed_token_count": total,
            "transformation": {k: v for k, v in transformed.items() if k != "reasoning"},
            "config_hash": config_hash, "source_manifest_hash": source_manifest_hash, "status": "verified",
            "simulation": trace.get("simulation", True), "execution_mode": trace.get("execution_mode", "fixture")}


def build_sft_datasets(pool, tokenizers, output_dir, *, config_hash: str, kazitext=None,
                       families=("rot13", "kazitext")) -> dict:
    validate_accepted_pool(pool, config_hash)
    if set(tokenizers) != {"7b", "14b"} or not families or any(f not in {"rot13", "kazitext"} for f in families) or len(set(families)) != len(families):
        raise ValueError("Invalid construction model/family handoff")
    if "kazitext" in families and kazitext is None:
        raise ValueError("Pinned official KaziText runtime is required")
    by_id = {p["example_id"]: p for p in pool["pairs"]}
    datasets = []
    for family in families:
        for model in ("7b", "14b"):
            for level in range(1, 11):
                split_records = {}
                for split, ids in (("train", pool["train_ids"]), ("validation", pool["validation_ids"])):
                    records = []
                    for example_id in ids:
                        pair, trace = by_id[example_id], by_id[example_id]["traces"][model]
                        transformed = transform_rot13(trace["reasoning"], example_id, level) if family == "rot13" else kazitext.transform(trace["reasoning"], example_id, level, model_id=model)
                        record = build_sft_record(pair, trace, transformed, tokenizers[model],
                                  model_id=model, family=family, level=level, split=split,
                                  source_manifest_hash=pool["content_hash"], config_hash=config_hash)
                        records.append(record)
                    split_records[split] = records
                condition = Path(output_dir) / family / model / f"level_{level:02d}"
                hashes = {split: _publish_records(condition / f"{split}.jsonl", records) for split, records in split_records.items()}
                manifest = {"status": "verified", "family": family, "method": family, "model_id": model, "model": model,
                            "level": level, "train_path": str((condition / "train.jsonl").resolve()),
                            "validation_path": str((condition / "validation.jsonl").resolve()),
                            "train_count": 10000, "validation_count": 100,
                            "train_ids": pool["train_ids"], "validation_ids": pool["validation_ids"],
                            "content_hashes": hashes, "config_hash": config_hash, "source_manifest_hash": pool["content_hash"],
                            "simulation": pool.get("simulation", True)}
                manifest["content_hash"] = content_hash(manifest)
                atomic_write_json(condition / "manifest.json", manifest)
                if read_json(condition / "manifest.json") != manifest:
                    raise IOError("Condition manifest readback failed")
                datasets.append(manifest)
    bundle = {"status": "verified", "datasets": datasets, "count": len(datasets),
              "train_ids": pool["train_ids"], "validation_ids": pool["validation_ids"],
              "source_manifest_hash": pool["content_hash"], "config_hash": config_hash,
              "content_hash": content_hash(datasets), "simulation": pool.get("simulation", True)}
    atomic_write_json(Path(output_dir) / ("manifest.json" if len(families) == 2 else f"{families[0]}_manifest.json"), bundle)
    return bundle


def prepare_sft_data(candidate_manifest, generator, tokenizers, *, kazitext,
                     clean_output_dir, transformed_output_dir, config_hash: str,
                     stop_requested=None, stop_file=None) -> dict:
    pool = collect_clean_traces(candidate_manifest, generator, tokenizers, kazitext=kazitext,
                               output_dir=clean_output_dir, config_hash=config_hash,
                               stop_requested=stop_requested, stop_file=stop_file)
    if pool["status"] == "paused":
        return {"status": "paused", "launch_ready": False, "collection": pool, "datasets": None}
    bundle = build_sft_datasets(pool, tokenizers, transformed_output_dir,
                               config_hash=config_hash, kazitext=kazitext)
    if bundle["count"] != 40:
        raise ValueError("Both complete families are required for the SFT handoff")
    return {"status": "verified", "accepted_pool": pool, "datasets": bundle,
            "content_hash": content_hash({"source": pool["content_hash"], "datasets": bundle["content_hash"]})}

sft_data_preparation = prepare_sft_data


def validate_sft_bundle(bundle, pool, *, tokenizers=None, kazitext=None,
                        recompute: bool = False) -> dict:
    """Read every actual condition file and verify frozen handoff identities.

    Recompute=True repeats exact transforms and native token counts from the
    unchanged sources. This is an artifact integrity preflight, not regeneration
    of any model response or scientific resampling.
    """
    validate_accepted_pool(pool, bundle.get("config_hash"))
    datasets = bundle.get("datasets", [])
    expected = {(family, model, level) for family in ("rot13", "kazitext") for model in ("7b", "14b") for level in range(1, 11)}
    if bundle.get("status") != "verified" or len(datasets) != 40 or {(d["family"], d["model_id"], d["level"]) for d in datasets} != expected:
        raise ValueError("SFT bundle requires all forty unique verified conditions")
    if bundle.get("content_hash") != content_hash(datasets) or bundle.get("source_manifest_hash") != pool["content_hash"]:
        raise ValueError("SFT bundle content/source identity mismatch")
    if recompute and (tokenizers is None or set(tokenizers) != {"7b", "14b"} or kazitext is None):
        raise ValueError("Exact native tokenizers and pinned KaziText required to recompute clean-data preflight")
    sources = {pair["example_id"]: pair for pair in pool["pairs"]}
    total = 0
    for dataset in datasets:
        if dataset.get("status") != "verified" or dataset.get("source_manifest_hash") != pool["content_hash"] or dataset.get("config_hash") != bundle["config_hash"]:
            raise ValueError("SFT condition source/configuration mismatch")
        model, family, level = dataset["model_id"], dataset["family"], dataset["level"]
        for split in ("train", "validation"):
            ids = pool[split + "_ids"]
            path = Path(dataset[split + "_path"])
            if not path.is_file() or content_hash(path.read_bytes()) != dataset["content_hashes"][split]:
                raise ValueError("SFT condition file checksum mismatch")
            records = load_sft_records(path)
            if [r["example_id"] for r in records] != ids or dataset.get(split + "_ids") != ids or dataset.get(split + "_count") != len(ids):
                raise ValueError("SFT condition changed common split/order/count")
            for record in records:
                pair, trace = sources[record["example_id"]], sources[record["example_id"]]["traces"][model]
                check = next(v for v in trace["preflight_variants"] if v["method"] == family and v["level"] == level)
                if (record.get("status") != "verified" or record["split"] != split or record["method"] != family or record["model_id"] != model or record["level"] != level
                    or record["prompt"] != pair["prompt"] or record["final_answer"] != trace["final_answer"]
                    or record["clean_trace_hash"] != trace["clean_trace_hash"] or record["source_content_hash"] != pair["source_content_hash"]
                    or record["source_manifest_hash"] != pool["content_hash"] or record["config_hash"] != bundle["config_hash"] or record["seed"] != 42
                    or record["completion"] != native_completion(record["reasoning"], trace["final_answer"])
                    or content_hash(record["reasoning"]) != check["content_hash"] or record["transformed_token_count"] != check["token_count"]
                    or record["transformed_token_count"] > 4096 or not record["transformation"]["protected_content_valid"]
                    or record["model_revision"] != trace["model_revision"] or record["tokenizer_revision"] != trace["tokenizer_revision"]
                    or record["identity"] != {"config_hash": bundle["config_hash"], "model": model, "method": family,
                                               "level": level, "source_manifest_hash": pool["content_hash"], "clean_trace_hash": trace["clean_trace_hash"]}):
                    raise ValueError("Actual SFT record violated frozen source/condition/protection/length invariants")
                if recompute:
                    transformed = transform_rot13(trace["reasoning"], pair["example_id"], level) if family == "rot13" else kazitext.transform(trace["reasoning"], pair["example_id"], level, model_id=model)
                    rebuilt = build_sft_record(pair, trace, transformed, tokenizers[model], model_id=model, family=family,
                                               level=level, split=split, source_manifest_hash=pool["content_hash"], config_hash=bundle["config_hash"])
                    if rebuilt != record:
                        raise ValueError("Exact recomputation differs from frozen training record")
                total += 1
    return {"status": "verified", "dataset_count": 40, "record_count": total,
            "train_count_per_condition": 10000, "validation_count_per_condition": 100,
            "recomputed_transforms": recompute, "source_manifest_hash": pool["content_hash"],
            "config_hash": bundle["config_hash"], "bundle_hash": bundle["content_hash"]}
