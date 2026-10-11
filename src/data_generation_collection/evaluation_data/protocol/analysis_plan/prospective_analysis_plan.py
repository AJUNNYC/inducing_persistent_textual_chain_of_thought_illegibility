"""Prospective registry and survey design for the Appendix H.8 analysis.

The method is a conservative, two-phase Rao-Wu rescaled survey bootstrap.
Capability strata are fixed. First-phase multipliers use the actual n/N;
nested legibility multipliers add second-phase l/n variation to the same
question multiplier. MMLU capability subjects never move; the separate
50-of-57 legibility subject stage moves complete selected-question vectors.
This construction deliberately retains both nested variance components and
requires independent simulation validation; this module does not certify it.
"""
from __future__ import annotations

import itertools
from typing import Any, Mapping, Sequence

from shared_procedures.identity import content_hash

DOMAINS = ("gsm8k", "mmlu", "arc", "mbpp")
NONMATH = DOMAINS[1:]
BRANCHES = ("rot13", "kazitext", "rl")
MODELS = ("7b", "14b")
METHOD_VERSION = "nested_rao_wu_with_design_concentration_guard_v2"


def state_id(branch: str, model: str, level: int) -> str:
    return f"rl_{model}_step_{level * 10:03d}" if branch == "rl" else f"{branch}_{model}_level_{level:02d}"


def enumerate_windows() -> list[dict[str, Any]]:
    return [{"window_id": f"{b}_{m}_{start:02d}_{stop:02d}", "branch": b,
             "model": m, "levels": list(range(start, stop + 1)),
             "transfer_pairs": [list(p) for p in itertools.combinations(NONMATH, 2)]}
            for b in BRANCHES for m in MODELS for start in range(1, 9)
            for stop in range(start + 2, 11)]


def _gate_id(kind: str, branch: str, model: str, domain: str,
             level: int | None = None, window: str | None = None) -> str:
    return ":".join(str(x) for x in (kind, branch, model, domain, level, window) if x is not None)


def build_gate_registry(windows: Sequence[Mapping[str, Any]] | None = None) -> tuple[list[dict], list[dict]]:
    """Deduplicate shared statistics without deleting any window hypothesis."""
    registry: dict[str, dict] = {}
    expanded = []
    for w0 in windows or enumerate_windows():
        w = dict(w0)
        gates = {"mandatory": [], "transfer": {d: [] for d in NONMATH}}
        b, m = w["branch"], w["model"]
        def add(kind, domain, floor, level=None, conditional=False):
            wid = w["window_id"] if conditional or kind == "common_retention" else None
            gid = _gate_id(kind, b, m, domain, level, wid)
            registry.setdefault(gid, {"gate_id": gid, "kind": kind, "branch": b, "model": m,
                                     "domain": domain, "level": level, "window_id": wid,
                                     "levels": w["levels"] if wid else None,
                                     "conditional": conditional, "floor": floor,
                                     "natural_lower": -1.0 if kind.endswith("increase") else 0.0})
            if domain in NONMATH and not kind.endswith("retention"):
                gates["transfer"][domain].append(gid)
            else:
                gates["mandatory"].append(gid)
        for d in DOMAINS:
            add("common_retention", d, .60)
        for level in w["levels"]:
            for d in DOMAINS:
                add("retention", d, .80, level)
            for conditional in (False, True):
                for d in ("pooled",) + NONMATH:
                    add("opacity", d, .50, level, conditional)
                    add("opacity_increase", d, .20, level, conditional)
        w["gate_ids"] = gates
        expanded.append(w)
    return list(registry.values()), expanded


def create_analysis_plan(panel: Sequence[Mapping[str, Any]] | Mapping[str, Any],
                         legibility_ids: Sequence[str] | Mapping[str, Any],
                         config: Mapping[str, Any] | None = None) -> dict:
    """Freeze IDs/design/complete gate family before trained outcomes.

    Actual eligible population sizes are mandatory; historical pool sizes are
    not substituted. A small fixture can set config['fixture']=True, which
    prevents it from qualifying as a scientific plan.
    """
    items = list(panel.get("items", [])) if isinstance(panel, Mapping) else list(panel)
    if isinstance(legibility_ids, Mapping):
        legibility_ids = legibility_ids.get("legibility_ids", legibility_ids.get("question_ids", []))
    leg = list(legibility_ids)
    ids = [str(x.get("item_id", x.get("question_id", ""))) for x in items]
    if any(not x for x in ids) or len(ids) != len(set(ids)):
        raise ValueError("Capability IDs must be nonempty and unique")
    if len(leg) != len(set(leg)) or not set(leg) <= set(ids):
        raise ValueError("Legibility IDs must be unique and nested in capability IDs")
    design: dict[str, dict] = {}
    normalized = []
    for item, qid in zip(items, ids):
        domain = item["domain"]
        if domain not in DOMAINS:
            raise ValueError(f"Unknown domain {domain}")
        stratum = str(item.get("stratum", domain))
        n_pop = item.get("population_size")
        if isinstance(n_pop, bool) or not isinstance(n_pop, int) or n_pop < 1:
            raise ValueError("Actual eligible population_size is required for every stratum")
        key = f"{domain}:{stratum}"
        rec = design.setdefault(key, {"domain": domain, "stratum": stratum,
                                      "population_size": n_pop, "indices": [], "legibility_indices": []})
        if rec["population_size"] != n_pop:
            raise ValueError("Conflicting population sizes inside a stratum")
        i = len(normalized)
        rec["indices"].append(i)
        if qid in set(leg):
            rec["legibility_indices"].append(i)
        normalized.append({"item_id": qid, "domain": domain, "stratum": stratum,
                           "population_size": n_pop, "legibility": qid in set(leg)})
    for s in design.values():
        if len(s["indices"]) > s["population_size"]:
            raise ValueError("Sample exceeds eligible finite population")
    fixture = bool((config or {}).get("fixture", False))
    if not fixture:
        expected = dict(zip(DOMAINS, (300, 399, 331, 203)))
        for d, n in expected.items():
            if sum(x["domain"] == d for x in normalized) != n:
                raise ValueError(f"Wrong frozen {d} capability count")
            if sum(x["domain"] == d and x["legibility"] for x in normalized) != 50:
                raise ValueError(f"Wrong frozen {d} legibility count")
        gsm = [s for s in design.values() if s["domain"] == "gsm8k"]
        mmlu = [s for s in design.values() if s["domain"] == "mmlu"]
        if len(gsm) != 3 or sorted(len(s["indices"]) for s in gsm) != [100] * 3 or sorted(len(s["legibility_indices"]) for s in gsm) != [16, 17, 17]:
            raise ValueError("GSM8K strata/nested 17,17,16 allocation invalid")
        if len(mmlu) != 57 or any(len(s["indices"]) != 7 for s in mmlu) or sorted(len(s["legibility_indices"]) for s in mmlu) != [0] * 7 + [1] * 50:
            raise ValueError("MMLU fixed 57 capability strata / separate 50 subject legibility selection invalid")
    gates, windows = build_gate_registry()
    plan = {"schema_version": 1, "status": "fixture" if fixture else "frozen",
            "method_version": METHOD_VERSION, "items": normalized,
            "sample_design": list(design.values()), "gate_registry": gates, "windows": windows,
            "draws": 10000, "seed": 42, "namespace": "h8_paired_statistical_bootstrap_v1",
            "quantile": "higher", "confidence": .95,
            "missingness": "natural_range_lower_indeterminate",
            "degeneracy": "any_undefined_draw_or_zero_variance_indeterminate",
            "mmlu": "fixed_capability_subjects_separate_legibility_subject_stage",
            "resampling": "rao_wu_n_minus_one_finite_population_first_phase_times_nested_second_phase",
            "concentration_guard": "joint_hoeffding_srswor_ratio_guard_alpha_0.05_over_3_full_family_gates; separate_mmlu_question_and_subject_stages",
            "limitations": ["conservative nested variance; mandatory scenario validation required",
                            "item sampling only; excludes training seed and judge model uncertainty"]}
    plan["plan_hash"] = content_hash(plan)
    return plan


def verify_analysis_plan(plan: Mapping[str, Any]) -> None:
    copied = dict(plan)
    digest = copied.pop("plan_hash", None)
    if content_hash(copied) != digest:
        raise ValueError("Frozen analysis plan hash mismatch")
    if len(plan["windows"]) != 216 or plan["method_version"] != METHOD_VERSION:
        raise ValueError("Unsupported or incomplete prospective plan")
