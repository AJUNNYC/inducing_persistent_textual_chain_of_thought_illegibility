"""Prospectively frozen finite-population Monte Carlo coverage validation.

Synthetic populations are validation data only. Exact SRSWOR capability and
nested legibility sampling feed the same complete 216-window implementation.
The frozen acceptance rule requires the one-sided 95% Wilson lower confidence
limit for *joint* coverage to be >= .95 in every mandatory scenario. The full
budget is 100 repetitions/scenario and 10,000 bootstrap draws/repetition.
Reduced development fixtures can never create a passing preflight certificate.
"""
from __future__ import annotations
from pathlib import Path
from statistics import NormalDist
from typing import Any, Mapping
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import numpy as np

from data_generation_collection.evaluation_data.protocol.analysis_plan.prospective_analysis_plan import (
    DOMAINS, MODELS, BRANCHES, METHOD_VERSION, create_analysis_plan, state_id, verify_analysis_plan)
from data_generation_collection.evaluation_data.protocol.analysis_plan.simultaneous_bounds.simultaneous_bounds import (
    assemble_outcomes, metric_arrays, compute_simultaneous_bounds)
from shared_procedures.identity import content_hash, derive_seed
from shared_procedures.storage import atomic_write_json, read_json

SCENARIOS = ("decision_boundaries", "correlated_checkpoints", "uneven_domains",
             "extreme_proportions", "small_common_sets", "empty_common_sets", "heterogeneous_strata")


def implementation_hash() -> str:
    here = Path(__file__).parent
    paths = [here / "prospective_analysis_plan.py", here / "simultaneous_bounds/simultaneous_bounds.py",
             here / "statistical_validation.py"]
    return content_hash({p.name: p.read_text() for p in paths})


def freeze_simulation_plan(analysis_plan: Mapping[str, Any] | None = None, *, repetitions: int = 100,
                           draws: int = 10000, fixture: bool = False) -> dict:
    if repetitions < 1 or draws < 2:
        raise ValueError("Positive Monte Carlo budget and at least two bootstrap draws required")
    if not fixture and (repetitions != 100 or draws != 10000):
        raise ValueError("Full prospectively fixed validation budget is 100 repetitions and 10000 draws")
    actual_frames = None
    if analysis_plan is not None:
        verify_analysis_plan(analysis_plan)
        actual_frames = [{"domain": s["domain"], "stratum": s["stratum"],
                          "population_size": s["population_size"],
                          "capability_count": len(s["indices"]),
                          "legibility_count": len(s["legibility_indices"])}
                         for s in analysis_plan["sample_design"]]
    plan = {"schema_version": 1, "status": "fixture" if fixture else "frozen",
            "method_version": METHOD_VERSION, "implementation_hash": implementation_hash(),
            "analysis_plan_hash": analysis_plan.get("plan_hash") if analysis_plan else None,
            "scenarios": list(SCENARIOS), "repetitions": repetitions, "draws": draws,
            "seed": 42, "namespace": "h8_finite_population_coverage_v1",
            "generator": "paired_binary_finite_population_v1",
            "sampling": "exact_stratified_srswor_then_nested_srswor; fixed57_mmlu_subjects; separate50_subjects",
            "population_sizes": {"gsm8k": [440, 440, 439], "mmlu": [100] * 57, "arc": 2376, "mbpp": 427},
            "actual_sample_frames": actual_frames,
            "sample_frame_source": "actual_eligible_analysis_strata" if actual_frames is not None else "generic_validation_fixture_population",
            "scenario_parameters": {
                "decision_boundaries": {"baseline": .72, "retention": .80, "baseline_opacity": .30, "trained_opacity": .50, "dependence": .75},
                "correlated_checkpoints": {"baseline": .72, "retention": .85, "baseline_opacity": .25, "trained_opacity": .60, "dependence": .98},
                "uneven_domains": {"baseline": [.90, .50, .25, .75], "retention": [.95, .80, .65, .85], "baseline_opacity": [.10, .20, .35, .40], "trained_opacity": [.60, .50, .55, .70], "dependence": .6},
                "extreme_proportions": {"baseline": [.995, .01, .99, .02], "retention": .99, "baseline_opacity": 0., "trained_opacity": 1., "dependence": .9},
                "small_common_sets": {"baseline": .25, "retention": .35, "baseline_opacity": .3, "trained_opacity": .5, "dependence": .05},
                "empty_common_sets": {"baseline": .8, "retention": .5, "baseline_opacity": .3, "trained_opacity": .6, "dependence": 1.0, "disjoint_alternating": True},
                "heterogeneous_strata": {"baseline": .7, "retention": .8, "baseline_opacity": .3, "trained_opacity": .5, "dependence": .5, "stratum_heterogeneity": True}},
            "acceptance": {"joint_coverage_target": .95, "monte_carlo_confidence": .95,
                           "rule": "each_mandatory_scenario_one_sided_wilson_lower_at_least_0.95",
                           "structural_failures_allowed": 0,
                           "futility_stop": "stop_scenario_when_even_all_remaining_successes_cannot_pass_fixed_wilson_rule",
                           "mandatory_failure": "stop_confirmatory_preflight_on_first_rejected_mandatory_scenario; preserve_unrun_scenario_registry"},
            "outcomes_inspected": False}
    if actual_frames is not None:
        plan["population_sizes"] = {d: [s["population_size"] for s in actual_frames if s["domain"] == d]
                                    for d in DOMAINS}
    plan["simulation_plan_hash"] = content_hash(plan)
    return plan


def monte_carlo_interval(successes: int, repetitions: int, confidence: float = .95) -> dict:
    if not 0 <= successes <= repetitions or repetitions < 1:
        raise ValueError("Invalid Monte Carlo binomial counts")
    p = successes / repetitions
    z = NormalDist().inv_cdf(confidence)  # one-sided bound
    den = 1 + z * z / repetitions
    center = (p + z * z / (2 * repetitions)) / den
    half = z * np.sqrt(p * (1 - p) / repetitions + z * z / (4 * repetitions * repetitions)) / den
    return {"estimate": p, "standard_error": float(np.sqrt(p * (1 - p) / repetitions)),
            "one_sided_wilson_lower": max(0., float(center - half)),
            "one_sided_wilson_upper": min(1., float(center + half)), "confidence": confidence}


def generate_finite_population(simulation_plan: Mapping[str, Any], scenario: str) -> dict:
    if scenario not in simulation_plan["scenarios"]:
        raise ValueError("Scenario absent from frozen plan")
    sizes = simulation_plan["population_sizes"]
    items = []
    design = []
    if simulation_plan.get("actual_sample_frames") is not None:
        strata = [(s["domain"], s["stratum"], s["population_size"], s["capability_count"], s["legibility_count"])
                  for s in simulation_plan["actual_sample_frames"]]
    else:
        strata = [("gsm8k", f"tercile_{j+1}", n, 100, (17, 17, 16)[j]) for j, n in enumerate(sizes["gsm8k"])]
        strata += [("mmlu", f"subject_{j:02d}", n, 7, 0) for j, n in enumerate(sizes["mmlu"])]
        strata += [("arc", "arc", sizes["arc"], 331, 50), ("mbpp", "mbpp", sizes["mbpp"], 203, 50)]
    for d, s, pop, cap, leg in strata:
        indices = []
        for k in range(pop):
            indices.append(len(items))
            items.append({"item_id": f"sim_{d}_{s}_{k:04d}", "domain": d, "stratum": s, "population_size": pop})
        design.append({"domain": d, "stratum": s, "population_size": pop, "indices": indices,
                       "capability_count": cap, "legibility_count": leg})
    rng = np.random.default_rng(derive_seed(simulation_plan["seed"], simulation_plan["namespace"], scenario, "population"))
    params = simulation_plan["scenario_parameters"][scenario]
    n = len(items)
    domain_index = np.asarray([DOMAINS.index(row["domain"]) for row in items])
    def vector(key):
        value = params[key]
        return np.asarray(value)[domain_index] if isinstance(value, list) else np.full(n, value)
    base_prob = vector("baseline")
    opacity_base = vector("baseline_opacity")
    trained_prob = vector("trained_opacity")
    retention = vector("retention")
    if params.get("stratum_heterogeneity"):
        for j, stratum in enumerate(design):
            ix = stratum["indices"]
            shift = .2 * np.sin(j * 1.618)
            base_prob[ix] = np.clip(base_prob[ix] + shift, .02, .98)
            trained_prob[ix] = np.clip(trained_prob[ix] + shift, .02, .98)
    shared_correct = rng.random(n)
    shared_opacity = rng.random(n)
    common_base = rng.random(n)
    records = {}
    for model in MODELS:
        baseline = common_base < base_prob
        base_op = shared_opacity < opacity_base
        records[f"base_{model}"] = _simulation_records(items, baseline, base_op)
        for branch in BRANCHES:
            for level in range(1, 11):
                corr_uniform = np.where(rng.random(n) < params["dependence"], shared_correct, rng.random(n))
                keep = corr_uniform < retention
                if params.get("disjoint_alternating"):
                    keep = shared_correct < .5 if level % 2 else shared_correct >= .5
                correct = (baseline & keep) | (~baseline & (rng.random(n) < .1))
                op_uniform = np.where(rng.random(n) < params["dependence"], shared_opacity, rng.random(n))
                opaque = op_uniform < trained_prob
                records[state_id(branch, model, level)] = _simulation_records(items, correct, opaque)
    return {"items": items, "sampling_design": design, "records": records, "scenario": scenario,
            "evidence_type": "synthetic_finite_population_validation"}


def _simulation_records(items, correct, opaque):
    return {row["item_id"]: {"question_id": row["item_id"], "domain": row["domain"],
            "correct": bool(c), "assessable": True, "primary_score": 7 if o else 3,
            "secondary_score": 7 if o else 3, "readable_solution": "not_detected",
            "padding": "not_detected"} for row, c, o in zip(items, correct, opaque)}


def sample_population(population: Mapping[str, Any], seed: int) -> tuple[list[dict], list[str], dict]:
    """Exact two-stage without-replacement design, including MMLU distinction."""
    rng = np.random.default_rng(seed)
    panel, leg = [], []
    mmlu = [s for s in population["sampling_design"] if s["domain"] == "mmlu"]
    selected_subjects = set(rng.choice(len(mmlu), 50, replace=False).tolist())
    subject_idx = 0
    for s in population["sampling_design"]:
        chosen = rng.choice(s["indices"], s["capability_count"], replace=False).tolist()
        panel.extend(dict(population["items"][i]) for i in chosen)
        if s["domain"] == "mmlu":
            selected = [int(rng.choice(chosen))] if subject_idx in selected_subjects else []
            subject_idx += 1
        else:
            selected = rng.choice(chosen, s["legibility_count"], replace=False).tolist()
        leg.extend(population["items"][i]["item_id"] for i in selected)
    ids = {row["item_id"] for row in panel}
    records = {state: {qid: row for qid, row in rows.items() if qid in ids} for state, rows in population["records"].items()}
    return panel, leg, records


def population_truth(population: Mapping[str, Any]) -> np.ndarray:
    """Design-weighted finite-population ratio targets for the fixed quotas."""
    panel = population["items"]
    plan = create_analysis_plan(panel, [x["item_id"] for x in panel], {"fixture": True})
    outcomes = assemble_outcomes(population["records"], plan)
    arrays = metric_arrays(outcomes, plan)
    cap_weights = np.zeros(len(panel))
    leg_weights = np.zeros(len(panel))
    for s in population["sampling_design"]:
        cap_weights[s["indices"]] = s["capability_count"] / s["population_size"]
        leg_weights[s["indices"]] = (50 / 57 if s["domain"] == "mmlu" else s["legibility_count"]) / s["population_size"]
    truth = np.full(len(plan["gate_registry"]), np.nan)
    for kind, weights in (("capability", cap_weights), ("legibility", leg_weights)):
        ix = np.flatnonzero(arrays["kind"] == kind)
        num = weights @ arrays["numerator"][:, ix]
        den = weights @ arrays["denominator"][:, ix]
        truth[ix] = np.divide(num, den, out=np.full(len(ix), np.nan), where=den > 0)
    return truth


def _report_digest(report: Mapping[str, Any]) -> str:
    return content_hash({k: v for k, v in report.items() if k not in ("checkpoint_hash", "report_hash")})


def _persist_report(report: dict, output_path: str | Path | None, *, final: bool = False) -> None:
    report.pop("report_hash", None)
    report["checkpoint_hash"] = _report_digest(report)
    if final:
        report["report_hash"] = content_hash(report)
    if output_path is not None:
        atomic_write_json(output_path, report)


def _validate_checkpoint(report: Mapping[str, Any], expected: Mapping[str, Any], frozen: Mapping[str, Any]) -> None:
    if report.get("checkpoint_hash") != _report_digest(report):
        raise ValueError("Simulation checkpoint integrity mismatch")
    if "report_hash" in report and report["report_hash"] != content_hash({k: v for k, v in report.items() if k != "report_hash"}):
        raise ValueError("Simulation final report integrity mismatch")
    for field in ("simulation_plan_hash", "analysis_plan_hash", "implementation_hash", "method_version", "draws",
                  "repetitions", "selected_scenarios", "full_validation"):
        if report.get(field) != expected.get(field):
            raise ValueError(f"Simulation checkpoint identity mismatch: {field}")
    if [s.get("scenario") for s in report.get("scenarios", [])] != list(SCENARIOS):
        raise ValueError("Simulation checkpoint scenario registry mismatch")
    groups = [(s["scenario"], s.get("runs", [])) for s in report["scenarios"]]
    if report.get("active_scenario"):
        active = report["active_scenario"]
        if active.get("name") not in expected["selected_scenarios"]:
            raise ValueError("Simulation checkpoint active scenario mismatch")
        groups.append((active["name"], active.get("runs", [])))
    for scenario, runs in groups:
        if len(runs) > frozen["repetitions"]:
            raise ValueError("Simulation checkpoint exceeds frozen budget")
        for repetition, run in enumerate(runs):
            seed = derive_seed(frozen["seed"], frozen["namespace"], scenario, repetition, "sample")
            if run.get("repetition") != repetition or run.get("seed") != seed:
                raise ValueError("Simulation checkpoint repetition/seed mismatch")
            if not isinstance(run.get("joint_covered"), bool) or not isinstance(run.get("structural_ok"), bool):
                raise ValueError("Simulation checkpoint outcome type mismatch")


def _scenario_summary(scenario: str, runs: list[dict], frozen: Mapping[str, Any]) -> dict:
    successes = sum(int(r["joint_covered"] and r["structural_ok"]) for r in runs)
    diagnostics = {"structural_failures": sum(int(not r["structural_ok"]) for r in runs),
                   "unsupported_gates": sum(r["unsupported_gate_count"] for r in runs),
                   "undefined_population_targets": sum(r["undefined_population_targets"] for r in runs)}
    interval = monte_carlo_interval(successes, len(runs)) if runs else None
    remaining = frozen["repetitions"] - len(runs)
    attainable = monte_carlo_interval(successes + remaining, frozen["repetitions"])["one_sided_wilson_lower"]
    rejected = bool(runs) and (attainable < .95 or diagnostics["structural_failures"] > 0)
    accepted = remaining == 0 and interval["one_sided_wilson_lower"] >= .95 and not diagnostics["structural_failures"]
    return {"scenario": scenario, "status": "passed" if accepted else "failed" if rejected else "running",
            "successes": successes, "repetitions": len(runs), "planned_repetitions": frozen["repetitions"],
            "unrun_repetitions": remaining, "futility_stopped": rejected and remaining > 0,
            "best_attainable_wilson_lower": attainable, "joint_coverage": interval,
            "diagnostics": diagnostics, "runs": runs}


def _validation_repetition(population, frozen, scenario, repetition):
    seed = derive_seed(frozen['seed'], frozen['namespace'], scenario, repetition, 'sample')
    panel, leg, records = sample_population(population, seed)
    plan = create_analysis_plan(panel, leg)
    bounds = compute_simultaneous_bounds(records, plan, draws=frozen['draws'])
    truth = population['validation_truth']
    applicable = np.isfinite(truth)
    lower = np.asarray([g['lower_bound'] for g in bounds['gates']])
    violations = np.flatnonzero(applicable & (lower > truth + 1e-12))
    structural = len(bounds['gates']) == len(plan['gate_registry']) and len(plan['windows']) == 216
    structural &= all(g['status'] == 'indeterminate' for g, undefined in zip(bounds['gates'], ~applicable) if undefined)
    return {'repetition':repetition, 'seed':seed, 'joint_covered':len(violations)==0,
            'structural_ok':bool(structural), 'critical_value':bounds['critical_value'],
            'unsupported_gate_count':bounds['unsupported_gate_count'],
            'undefined_population_targets':int((~applicable).sum()),
            'violations':[{'gate_id':bounds['gates'][j]['gate_id'], 'lower_bound':float(lower[j]),
                           'population_truth':float(truth[j])} for j in violations],
            'violation_count':len(violations)}


def _initialize_validation_worker(population, frozen, scenario):
    global _validation_worker_context
    _validation_worker_context = (population, frozen, scenario)


def _worker_repetition(repetition):
    return _validation_repetition(*_validation_worker_context, repetition)


def run_statistical_validation(analysis_plan: Mapping[str, Any] | None = None, *,
                               simulation_plan: Mapping[str, Any] | None = None,
                               repetitions: int = 100, draws: int = 10000,
                               output_path: str | Path | None = None,
                               fixture: bool = False, scenarios: list[str] | None = None,
                               resume: bool = False, workers: int = 1) -> dict:
    """Run frozen cases, atomically checkpoint every repetition, resume by identity.

    A subset retains the complete mandatory registry and is ``partial`` when
    successful. A rejected mandatory case is ``failed`` even in a subset.
    Checkpoints cannot cross code/plan/selection identities. Nothing is resampled
    because of failure; resumed repetition seeds continue their frozen sequence.
    """
    if workers < 1:raise ValueError('Positive validation worker count required')
    frozen = dict(simulation_plan or freeze_simulation_plan(analysis_plan, repetitions=repetitions, draws=draws, fixture=fixture))
    digest = frozen.pop("simulation_plan_hash", None)
    if content_hash(frozen) != digest:
        raise ValueError("Simulation plan hash mismatch")
    frozen["simulation_plan_hash"] = digest
    if frozen["implementation_hash"] != implementation_hash():
        raise ValueError("Analysis implementation changed since simulation freeze")
    if frozen["scenarios"] != list(SCENARIOS):
        raise ValueError("Frozen mandatory scenario registry mismatch")
    if frozen["status"] == "frozen" and (frozen["draws"] != 10000 or frozen["repetitions"] != 100):
        raise ValueError("Frozen simulation budget mismatch")
    if analysis_plan is not None:
        verify_analysis_plan(analysis_plan)
        if frozen["analysis_plan_hash"] != analysis_plan["plan_hash"]:
            raise ValueError("Simulation/analysis plan identity mismatch")
    selected = list(frozen["scenarios"] if scenarios is None else scenarios)
    if not selected or len(selected) != len(set(selected)) or any(s not in SCENARIOS for s in selected):
        raise ValueError("Select nonempty unique scenarios from the frozen registry")
    full = frozen["status"] == "frozen" and selected == list(SCENARIOS)
    report = {"status": "running", "full_validation": full, "evidence_type": "finite_population_monte_carlo",
              "simulation_plan_hash": digest, "analysis_plan_hash": frozen["analysis_plan_hash"],
              "implementation_hash": frozen["implementation_hash"], "method_version": METHOD_VERSION,
              "draws": frozen["draws"], "repetitions": frozen["repetitions"], "selected_scenarios": selected,
              "scenarios": [{"scenario": s, "status": "not_run", "reason": "not_yet_executed" if s in selected else "not_selected_component",
                             "repetitions": 0, "planned_repetitions": frozen["repetitions"],
                             "unrun_repetitions": frozen["repetitions"], "runs": []} for s in SCENARIOS],
              "evidence": {"live": True, "kind": "actual_cpu_finite_population_simulation", "synthetic_populations": True},
              "checks": [], "preflight_name": "statistical_validation", "ready_for_confirmatory_launch": False}
    if resume:
        if output_path is None or not Path(output_path).is_file():
            raise ValueError("Resume requires an existing output checkpoint")
        previous = read_json(output_path)
        _validate_checkpoint(previous, report, frozen)
        if previous["status"] != "running":
            return previous
        report = previous
    if output_path:
        atomic_write_json(str(output_path) + ".plan.json", frozen)
    _persist_report(report, output_path)
    registry = {s["scenario"]: s for s in report["scenarios"]}
    for scenario in selected:
        if registry[scenario]["status"] in ("passed", "failed"):
            if registry[scenario]["status"] == "failed":
                break
            continue
        active = report.get("active_scenario", {})
        runs = list(active.get("runs", [])) if active.get("name") == scenario else []
        summary = _scenario_summary(scenario, runs, frozen)
        if summary["status"] == "running":
            population = generate_finite_population(frozen, scenario)
            population['validation_truth'] = population_truth(population)
            executor = ProcessPoolExecutor(max_workers=workers, initializer=_initialize_validation_worker,
                                           initargs=(population,frozen,scenario)) if workers > 1 else None
            with executor if executor is not None else nullcontext():
                indices = range(len(runs), frozen['repetitions'])
                outcomes = executor.map(_worker_repetition, indices) if executor else (
                    _validation_repetition(population,frozen,scenario,i) for i in indices)
                for run in outcomes:
                    runs.append(run)
                    report['active_scenario'] = {'name':scenario, 'completed_repetitions':len(runs), 'runs':runs}
                    _persist_report(report, output_path)
                    summary = _scenario_summary(scenario, runs, frozen)
                    if summary['status'] != 'running':
                        if executor:executor.shutdown(wait=True,cancel_futures=True)
                        break
        registry[scenario].clear()
        registry[scenario].update(summary)
        report.pop("active_scenario", None)
        _persist_report(report, output_path)
        if summary["status"] == "failed":
            for remaining in report["scenarios"]:
                if remaining["status"] == "not_run":
                    remaining["reason"] = "confirmatory_preflight_stopped_after_mandatory_rejection"
            break
    statuses = [s["status"] for s in report["scenarios"]]
    if frozen["status"] == "fixture":
        report["status"] = "fixture_complete"
    else:
        report["status"] = "failed" if "failed" in statuses else "passed" if all(s == "passed" for s in statuses) else "partial"
    report["ready_for_confirmatory_launch"] = report["status"] == "passed" and analysis_plan is not None
    report["full_budget_executed"] = all(s["repetitions"] == frozen["repetitions"] for s in report["scenarios"])
    check_status = "passed" if report["ready_for_confirmatory_launch"] else "failed" if report["status"] == "failed" else "blocked"
    report["checks"] = [{"name": "statistical_validation", "status": check_status,
                         "detail": "Every mandatory scenario clears the frozen joint-coverage rule" if check_status == "passed" else "Mandatory simulation validation rejected coverage or remains incomplete",
                         "evidence": report["evidence"], "required": True}]
    _persist_report(report, output_path, final=True)
    return report
