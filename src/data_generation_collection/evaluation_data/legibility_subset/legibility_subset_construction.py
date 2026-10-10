"""Nested 200-item selection; reuse the exact capability raw generations."""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path

from shared_procedures.identity import content_hash, derive_seed
from shared_procedures.storage import atomic_write_json, read_json
from data_generation_collection.evaluation_data.capability_panel.capability_panel_construction import validate_capability_panel


def construct_legibility_subset(panel: dict, *, output_path=None, require_verified_parent=True) -> dict:
    validate_capability_panel(panel, require_verified=require_verified_parent)
    by_domain = defaultdict(list)
    for item in panel["items"]:
        by_domain[item["domain"]].append(item)
    def sample(rows, count, namespace):
        ordered = sorted(rows, key=lambda r: r["question_id"])
        if len(ordered) < count:
            raise ValueError("Nested legibility quota is impossible")
        return sorted(random.Random(derive_seed(42, "legibility", namespace)).sample(ordered, count), key=lambda r: r["question_id"])
    selected, gsm = [], defaultdict(list)
    for item in by_domain["gsm8k"]:
        gsm[item["stratum"]].append(item)
    strata = sorted(gsm)
    remainder = random.Random(derive_seed(42, "legibility", "gsm_remainder")).choice(strata)
    allocation = {s: 16 if s == remainder else 17 for s in strata}
    for stratum in strata:
        selected.extend(sample(gsm[stratum], allocation[stratum], "gsm_" + stratum))
    subjects = defaultdict(list)
    for item in by_domain["mmlu"]:
        subjects[item["subject"]].append(item)
    chosen_subjects = sorted(random.Random(derive_seed(42, "legibility", "mmlu_subjects")).sample(sorted(subjects), 50))
    for subject in chosen_subjects:
        selected.extend(sample(subjects[subject], 1, "mmlu_" + subject))
    for domain in ("arc", "mbpp"):
        selected.extend(sample(by_domain[domain], 50, domain))
    selected.sort(key=lambda r: (r["domain"], r["question_id"]))
    sampling = {"seed": 42, "namespace": "legibility", "without_replacement": True,
                "algorithm_version": "nested_sha256_rng_v1", "gsm_allocation": allocation,
                "gsm_remainder_stratum": remainder, "mmlu_subjects": chosen_subjects,
                "mmlu_subject_population": 57, "mmlu_subject_sample_size": 50,
                "mmlu_items_per_selected_subject": 1, "arc_nested_population": 331,
                "mbpp_nested_population": 203, "reuse_capability_raw_generations": True}
    result = {"status": "verified" if panel["status"] == "verified" else "prepared",
              "schema_version": "legibility_subset_v1", "items": selected, "count": 200,
              "question_ids": [r["question_id"] for r in selected],
              "legibility_ids": [r["question_id"] for r in selected],
              "parent_hash": panel["content_hash"], "sampling": sampling,
              "sampling_hash": content_hash(sampling), "content_hash": content_hash(selected)}
    validate_legibility_subset(result, panel, require_verified=require_verified_parent)
    if output_path:
        if Path(output_path).exists() and read_json(output_path) != result:
            raise ValueError("Frozen nested IDs differ; no response-driven reselection permitted")
        atomic_write_json(output_path, result)
        if read_json(output_path) != result:
            raise IOError("Legibility subset readback failed")
    return result


def validate_legibility_subset(subset, panel, *, require_verified=True):
    items = subset.get("items", [])
    if subset.get("status") not in ({"verified"} if require_verified else {"verified", "prepared"}) or len(items) != 200 or len({r["question_id"] for r in items}) != 200:
        raise ValueError("Nested legibility count/IDs invalid")
    if Counter(r["domain"] for r in items) != {d: 50 for d in ("gsm8k", "mmlu", "arc", "mbpp")}:
        raise ValueError("Legibility requires 50 per domain")
    parent = {r["question_id"]: r for r in panel["items"]}
    if any(r["question_id"] not in parent or r != parent[r["question_id"]] for r in items):
        raise ValueError("Legibility items must equal exact frozen capability items")
    if subset.get("parent_hash") != panel["content_hash"] or subset.get("content_hash") != content_hash(items) or subset.get("sampling_hash") != content_hash(subset["sampling"]):
        raise ValueError("Nested source/sampling/content identity failed")
    allocation = Counter(r["stratum"] for r in items if r["domain"] == "gsm8k")
    subjects = Counter(r["subject"] for r in items if r["domain"] == "mmlu")
    if sorted(allocation.values()) != [16, 17, 17] or len(subjects) != 50 or set(subjects.values()) != {1}:
        raise ValueError("Nested stratification invalid")
    return subset

legibility_subset_construction = construct_legibility_subset
