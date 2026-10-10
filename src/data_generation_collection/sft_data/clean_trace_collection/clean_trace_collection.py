"""Exact deficit refills; one ordinary completion per candidate/model."""
from __future__ import annotations

from pathlib import Path
from shared_procedures.identity import content_hash
from shared_procedures.storage import atomic_write_json, read_json, notify_artifact
from data_generation_collection.sft_data.clean_trace_collection.candidate_batch.clean_trace_batch import process_clean_batch, CleanDataPause


def collect_clean_traces(candidate_manifest, generator, tokenizers, *, kazitext,
                         output_dir, config_hash: str, batch_processor=process_clean_batch,
                         stop_requested=None, stop_file=None) -> dict:
    candidates = candidate_manifest["candidates"]
    if candidate_manifest.get("status") != "verified" or candidate_manifest.get("content_hash") != content_hash(candidates):
        raise ValueError("Candidate manifest must be verified")
    accepted, rejected, batches, cursor = [], [], [], 0
    root = Path(output_dir)
    stop_path = Path(stop_file) if stop_file is not None else root / "STOP_REQUESTED"
    callback = stop_requested or getattr(generator, "stop_requested", None)
    def requested():
        return stop_path.exists() or bool(callback and callback())
    def pause(progress):
        ids = [candidate["example_id"] for candidate in candidates]
        next_id = progress.get("example_id")
        position = ids.index(next_id) if next_id is not None else cursor
        result = {"status": "paused", "reason": "operational_stop_requested", "launch_ready": False,
                  "accepted_count": len(accepted), "accepted_ids": [r["example_id"] for r in accepted],
                  "rejected_ids": rejected, "required_deficit": 10100 - len(accepted),
                  "next_batch_start_cursor": cursor, "next_candidate_position": position,
                  "next_operation": progress, "batches": batches,
                  "candidate_manifest_hash": candidate_manifest["content_hash"], "config_hash": config_hash,
                  "resume_policy": "same_manifest_config_ids_and_original_raw_caches"}
        path = root / "collection_status.json"
        atomic_write_json(path, result)
        notify_artifact(path, {"kind": "clean_collection_pause", "config_hash": config_hash})
        return result
    while len(accepted) < 10100:
        if requested():
            return pause({"phase": "before_batch", "example_id": candidates[cursor]["example_id"] if cursor < len(candidates) else None})
        deficit = 10100 - len(accepted)
        if len(candidates) - cursor < deficit:
            failure = {"status": "failed", "reason": "insufficient_new_candidate_pool",
                       "accepted_count": len(accepted), "required_deficit": deficit,
                       "remaining_count": len(candidates) - cursor, "batches": batches,
                       "candidate_manifest_hash": candidate_manifest["content_hash"], "config_hash": config_hash}
            atomic_write_json(root / "collection_status.json", failure)
            raise ValueError("Candidate shortage: cannot supply exact deficit refill")
        batch = candidates[cursor:cursor + deficit]
        try:
            options = {"kazitext": kazitext, "output_dir": root, "config_hash": config_hash}
            if batch_processor is process_clean_batch:
                options["stop_requested"] = requested
            records = batch_processor(batch, generator, tokenizers, **options)
        except CleanDataPause as error:
            return pause(error.progress)
        if [r["example_id"] for r in records] != [c["example_id"] for c in batch]:
            raise ValueError("Batch result IDs/order mismatch")
        for record in records:
            if record["status"] == "accepted":
                if set(record["traces"]) != {"7b", "14b"} or any(t["status"] != "accepted" for t in record["traces"].values()):
                    raise ValueError("Invalid jointly accepted pair")
                accepted.append(record)
            elif record["status"] == "rejected":
                rejected.append(record["example_id"])
            else:
                raise ValueError("Incomplete batch state cannot enter frozen pool")
        cursor += deficit
        batches.append({"size": deficit, "ids": [c["example_id"] for c in batch],
                        "cumulative_accepted": len(accepted), "processed_count": cursor})
        atomic_write_json(root / "collection_status.json", {"status": "running", "batches": batches,
                          "accepted_ids": [r["example_id"] for r in accepted], "rejected_ids": rejected,
                          "candidate_manifest_hash": candidate_manifest["content_hash"], "config_hash": config_hash})
    train_ids = [r["example_id"] for r in accepted[:10000]]
    validation_ids = [r["example_id"] for r in accepted[10000:]]
    if len(set(train_ids + validation_ids)) != 10100:
        raise ValueError("Accepted split IDs must be distinct")
    result = {"status": "verified", "schema_version": "accepted_pool_v1", "pairs": accepted,
              "train_ids": train_ids, "validation_ids": validation_ids,
              "train_count": 10000, "validation_count": 100, "rejected_ids": rejected,
              "simulation": any(pair.get("simulation", True) for pair in accepted),
              "batches": batches, "candidate_manifest_hash": candidate_manifest["content_hash"],
              "config_hash": config_hash, "content_hash": content_hash(accepted)}
    path = root / "accepted_pool.json"
    if path.exists() and read_json(path) != result:
        raise ValueError("Existing frozen accepted pool differs; cannot replace source IDs")
    atomic_write_json(path, result)
    if read_json(path) != result:
        raise IOError("Accepted pool readback failed")
    notify_artifact(path, {"kind": "accepted_pool", "source_manifest_hash": result["content_hash"], "config_hash": config_hash})
    atomic_write_json(root / "collection_status.json", {"status": "complete", "pool_hash": result["content_hash"], "batches": batches})
    return result

clean_trace_collection = collect_clean_traces
