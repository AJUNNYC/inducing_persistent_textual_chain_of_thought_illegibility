"""Frozen four-domain capability selection and withheld-test MBPP interfaces."""
from __future__ import annotations

import ast
import copy
import random
from collections import Counter, defaultdict
from pathlib import Path

from shared_procedures.identity import content_hash, derive_seed
from shared_procedures.response_parsing import normalize_numeric_answer
from shared_procedures.storage import atomic_write_json, read_json
from data_generation_collection.sft_data.clean_trace_collection.candidate_manifest import validate_revision

COUNTS = {"gsm8k": 300, "mmlu": 399, "arc": 331, "mbpp": 203}
POPULATIONS = {"gsm8k": 1319, "mmlu": 5700, "arc": 2376, "mbpp": 427}
SOURCES = {"gsm8k": "openai/gsm8k", "mmlu": "edinburgh-dawg/mmlu-redux-2.0",
           "arc": "allenai/ai2_arc", "mbpp": "google-research-datasets/mbpp"}
SAMPLING_VERSION = "capability_sha256_rng_without_replacement_v1"


def load_evaluation_sources(revisions: dict, *, cache_root=None) -> dict:
    """Read official, upstream-checksummed parquet, with zero hidden retries."""
    from data_generation_collection.sft_data.clean_trace_collection.candidate_manifest import acquire_pinned_parquet, iter_pinned_parquet_rows
    import pyarrow.parquet as parquet
    import pyarrow.ipc as ipc
    for domain in SOURCES:
        validate_revision(revisions[domain])
    pools = {
        "gsm8k": list(iter_pinned_parquet_rows(SOURCES["gsm8k"], revisions["gsm8k"], configuration="main", split="test", cache_root=cache_root)),
        "arc": list(iter_pinned_parquet_rows(SOURCES["arc"], revisions["arc"], configuration="ARC-Easy", split="test", cache_root=cache_root)),
        "mbpp": list(iter_pinned_parquet_rows(SOURCES["mbpp"], revisions["mbpp"], configuration="sanitized", cache_root=cache_root)),
    }
    artifacts = acquire_pinned_parquet(SOURCES["mmlu"], revisions["mmlu"], cache_root=cache_root)
    subjects = sorted({a["configuration"] for a in artifacts})
    if len(subjects) != 57:
        raise ValueError("Pinned MMLU-Redux-2.0 must contain exactly 57 subjects")
    mmlu = []
    for subject in subjects:
        index = 0
        for artifact in artifacts:
            if artifact["configuration"] != subject:
                continue
            batches = ipc.open_stream(artifact["path"]) if artifact["format"] == "arrow" else parquet.ParquetFile(artifact["path"]).iter_batches(batch_size=512)
            for batch in batches:
                for row in batch.to_pylist():
                    mmlu.append({**row, "subject": subject, "subject_index": index, "source_split": artifact["split"]})
                    index += 1
    pools["mmlu"] = mmlu
    return pools


def extract_mbpp_interface(reference_code: str, scoring_tests: list[str]) -> list[dict]:
    """Only direct tested callable signatures; no source/test bodies leave here."""
    tree = ast.parse(reference_code)
    functions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    called = set()
    for test in scoring_tests:
        test_tree = ast.parse(test)
        called.update(node.func.id for node in ast.walk(test_tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name))
    required = sorted(called & functions.keys())
    if not required:
        raise ValueError("Cannot mechanically identify MBPP tested public entry point")
    interfaces = []
    for name in required:
        fn = functions[name]
        args = copy.deepcopy(fn.args)
        defaults = {}
        positional = args.posonlyargs + args.args
        for argument, default in zip(positional[len(positional) - len(args.defaults):], args.defaults):
            try:
                ast.literal_eval(default)
            except (ValueError, TypeError) as error:
                raise ValueError("MBPP public default requires unsupported algorithmic expression") from error
            defaults[argument.arg] = ast.unparse(default)
        for argument, default in zip(args.kwonlyargs, args.kw_defaults):
            if default is not None:
                try:
                    ast.literal_eval(default)
                except (ValueError, TypeError) as error:
                    raise ValueError("MBPP public keyword default is not a literal") from error
                defaults[argument.arg] = ast.unparse(default)
        all_args = positional + args.kwonlyargs + ([args.vararg] if args.vararg else []) + ([args.kwarg] if args.kwarg else [])
        for argument in all_args:
            argument.annotation = None
        # ast.unparse(arguments) serializes names/defaults without function bodies.
        signature = f"def {name}({ast.unparse(args)}):"
        interfaces.append({"name": name, "parameters": [a.arg for a in all_args], "defaults": defaults,
                           "signature": signature, "extraction_version": "ast_direct_tested_callable_v1"})
    return interfaces


def _choice_records(row: dict, *, arc: bool) -> tuple[list[dict], str]:
    raw = row.get("choices")
    if isinstance(raw, dict):
        choices = [{"label": str(label), "text": str(text)} for label, text in zip(raw["label"], raw["text"])]
    elif isinstance(raw, list):
        choices = [{"label": str(c.get("label", chr(65 + i))), "text": str(c["text"])} if isinstance(c, dict)
                   else {"label": chr(65 + i), "text": str(c)} for i, c in enumerate(raw)]
    else:
        raise ValueError("Missing official multiple-choice options")
    answer = row.get("answerKey") if arc else row.get("answer")
    if not arc and isinstance(answer, int) and not isinstance(answer, bool):
        answer = chr(65 + answer)
    answer = str(answer)
    if len({c["label"] for c in choices}) != len(choices) or answer not in {c["label"] for c in choices}:
        raise ValueError("Invalid official choice reference")
    return choices, answer


def normalize_evaluation_item(domain: str, row: dict, index: int, revision: str) -> dict:
    source_id = str(row.get("task_id", row.get("id", row.get("subject_index", index))))
    subject = str(row.get("subject", "")) if domain == "mmlu" else None
    question_id = f"{domain}_{subject + '_' if subject else ''}{source_id}"
    question = row.get("question", row.get("prompt", row.get("text")))
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Missing source question")
    item = {"question_id": question_id, "item_id": question_id, "domain": domain,
            "question": question, "source_dataset": SOURCES[domain], "source_revision": revision,
            "source_index": index, "source_id": source_id,
            "source_split": row.get("source_split", "test")}
    if domain == "gsm8k":
        reference = row.get("answer", "")
        if "####" not in reference:
            raise ValueError("GSM8K source answer lacks official numeric delimiter")
        numeric = normalize_numeric_answer(reference.rsplit("####", 1)[-1])
        if numeric is None:
            raise ValueError("Invalid GSM8K numeric reference")
        item.update(reference_answer=numeric, reference_solution=reference,
                    reference_word_count=len(reference.split()))
    elif domain in {"mmlu", "arc"}:
        choices, answer = _choice_records(row, arc=domain == "arc")
        item.update(choices=choices, reference_answer=answer)
        if domain == "mmlu":
            item.update(subject=subject, error_type=row.get("error_type"))
    else:
        code, tests = row.get("code"), row.get("test_list")
        if not isinstance(code, str) or not isinstance(tests, list) or not tests or any(not isinstance(t, str) for t in tests):
            raise ValueError("Missing official sanitized MBPP code/tests")
        compile(code, "mbpp_reference", "exec")
        for test in tests:
            compile(test, "mbpp_scoring_test", "exec")
        interfaces = extract_mbpp_interface(code, tests)
        imports = row.get("test_imports", [])
        if not isinstance(imports, list) or any(not isinstance(i, str) for i in imports):
            raise ValueError("Invalid original MBPP test imports")
        setup = row.get("test_setup_code", "\n".join(imports))
        if setup:
            compile(setup, "mbpp_test_setup", "exec")
        item.update(public_interface=interfaces, interface=interfaces[0], reference_code=code,
                    scoring_tests=tests, test_setup_code=setup, test_imports=imports, reference_answer=None,
                    scoring_tests_hash=content_hash({"tests": tests, "setup": setup}),
                    public_interface_hash=content_hash(interfaces), mbpp_variant="withheld_test_public_interface_v1")
    item["source_content_hash"] = content_hash(item)
    return item


def _sample(rows, count: int, namespace: str):
    ordered = sorted(rows, key=lambda r: r["question_id"])
    if len(ordered) < count:
        raise ValueError("Fixed source quota is impossible")
    chosen = random.Random(derive_seed(42, "capability", namespace)).sample(ordered, count)
    return sorted(chosen, key=lambda r: r["question_id"])


def construct_capability_panel(pools: dict, revisions: dict, *, output_path=None,
                              verify_mbpp_reference=None) -> dict:
    if set(pools) != set(COUNTS) or set(revisions) != set(COUNTS):
        raise ValueError("All four exact official benchmark sources are required")
    for domain in COUNTS:
        validate_revision(revisions[domain])
        if len(pools[domain]) != POPULATIONS[domain]:
            raise ValueError(f"{domain} pinned source population differs from protocol")
    normalized, exclusions = {}, []
    for domain, rows in pools.items():
        items = []
        for index, row in enumerate(rows):
            if domain == "mmlu" and row.get("error_type") != "ok":
                exclusions.append({"domain": domain, "source_index": index, "source_id": str(row.get("id", row.get("subject_index", index))),
                                   "subject": row.get("subject"), "reason": "mmlu_not_clean_labeled", "error_type": row.get("error_type")})
                continue
            items.append(normalize_evaluation_item(domain, row, index, revisions[domain]))
        if len({i["question_id"] for i in items}) != len(items):
            raise ValueError("Official source identity collision")
        normalized[domain] = items
    selected, gsm_metadata = [], []
    ranked = sorted(normalized["gsm8k"], key=lambda r: (r["reference_word_count"], r["source_index"]))
    sizes = [len(ranked) // 3 + (j < len(ranked) % 3) for j in range(3)]
    cursor = 0
    for j, size in enumerate(sizes):
        stratum = f"tercile_{j + 1}"
        rows = ranked[cursor:cursor + size]
        for row in rows:
            row.update(stratum=stratum, population_size=size)
        selected.extend(_sample(rows, 100, "gsm8k_" + stratum))
        gsm_metadata.append({"stratum": stratum, "population_size": size, "sample_size": 100,
                             "rank_start": cursor, "rank_end": cursor + size,
                             "reference_length_min": rows[0]["reference_word_count"],
                             "reference_length_max": rows[-1]["reference_word_count"]})
        cursor += size
    subjects = defaultdict(list)
    for item in normalized["mmlu"]:
        subjects[item["subject"]].append(item)
    raw_subject_counts = Counter(str(r.get("subject", "")) for r in pools["mmlu"])
    if len(subjects) != 57 or len(raw_subject_counts) != 57 or "" in subjects:
        raise ValueError("All 57 MMLU fixed subjects require clean eligible populations")
    mmlu_metadata = []
    for subject, rows in sorted(subjects.items()):
        for row in rows:
            row.update(stratum=subject, population_size=len(rows))
        selected.extend(_sample(rows, 7, "mmlu_" + subject))
        mmlu_metadata.append({"stratum": subject, "subject": subject,
                              "population_size": len(rows), "original_population_size": raw_subject_counts[subject],
                              "sample_size": 7, "subject_is_fixed_stratum": True})
    for domain in ("arc", "mbpp"):
        rows = normalized[domain]
        for row in rows:
            row.update(stratum=domain, population_size=len(rows))
        selected.extend(_sample(rows, COUNTS[domain], domain))
    selected.sort(key=lambda r: (list(COUNTS).index(r["domain"]), r["question_id"]))
    tests_verified = verify_mbpp_reference is not None
    for item in selected:
        if item["domain"] == "mbpp":
            if verify_mbpp_reference:
                tests = ([item["test_setup_code"]] if item["test_setup_code"] else []) + item["scoring_tests"]
                result = verify_mbpp_reference(item["reference_code"], tests)
                valid = result is True or (isinstance(result, dict) and result.get("status") in {"pass", "passed", "valid"})
                if not valid:
                    raise ValueError("MBPP reference failed original scoring tests or executor preflight")
            item["reference_tests_verified"] = tests_verified
    sampling = {"seed": 42, "namespace": "capability", "algorithm_version": SAMPLING_VERSION,
                "without_replacement": True, "gsm_strata": gsm_metadata,
                "mmlu_subjects": mmlu_metadata, "mmlu_subjects_are_fixed": True,
                "arc": {"population_size": 2376, "sample_size": 331},
                "mbpp": {"population_size": 427, "sample_size": 203},
                "gsm_tercile_remainder": "earliest_ranked_strata", "python_random_algorithm": "mt19937_sample"}
    panel = {"status": "verified" if tests_verified else "prepared", "schema_version": "capability_panel_v1",
             "items": selected, "question_ids": [r["question_id"] for r in selected], "count": 1233,
             "domain_counts": COUNTS, "source_revisions": revisions, "exclusions": exclusions,
             "sampling": sampling, "mbpp_reference_tests_verified": tests_verified,
             "content_hash": content_hash(selected), "sampling_hash": content_hash(sampling)}
    validate_capability_panel(panel, require_verified=tests_verified)
    if output_path:
        if Path(output_path).exists() and read_json(output_path) != panel:
            raise ValueError("Existing frozen capability IDs/references differ")
        atomic_write_json(output_path, panel)
        if read_json(output_path) != panel:
            raise IOError("Capability panel readback failed")
    return panel


def validate_capability_panel(panel: dict, *, require_verified: bool = True) -> dict:
    items = panel.get("items", [])
    if panel.get("status") not in ({"verified"} if require_verified else {"verified", "prepared"}) or len(items) != 1233:
        raise ValueError("Capability panel is not verified with the fixed count")
    if Counter(r["domain"] for r in items) != COUNTS or len({r["question_id"] for r in items}) != 1233 or panel.get("content_hash") != content_hash(items):
        raise ValueError("Capability count/ID/hash invariant failed")
    gsm = Counter(r["stratum"] for r in items if r["domain"] == "gsm8k")
    mmlu = Counter(r["subject"] for r in items if r["domain"] == "mmlu")
    if sorted(gsm.values()) != [100, 100, 100] or len(mmlu) != 57 or set(mmlu.values()) != {7}:
        raise ValueError("Capability stratification quota invalid")
    if panel.get("sampling_hash") != content_hash(panel["sampling"]):
        raise ValueError("Capability sampling identity mismatch")
    if require_verified and any(not r.get("reference_tests_verified") for r in items if r["domain"] == "mbpp"):
        raise ValueError("MBPP source reference tests not verified")
    return panel

capability_panel_construction = construct_capability_panel
