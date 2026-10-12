#!/usr/bin/env python3
"""Qualify execution separately for each certified fit, then train fresh bases.

An isolated, immutable baseline checkpoint supplies actual full-state restore
evidence; it never seeds official training. Every condition retains the existing
paired numerical/RNG/speed gates and synchronous verified Drive barriers.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from experiment_overview.experiment_overview import Experiment
from run_certified_sft import certified_tasks
from shared_procedures.configuration import validate_configuration
from shared_procedures.identity import content_hash
from shared_procedures.preflight import load_preflight_evidence, preflight_requirements, require_preflights
from shared_procedures.storage import atomic_write_json, file_lock, read_json
from training.sft.execution_profile import SFTExecutionProfile
from training.sft.individual_fit import sft_fit
from training.sft.runtime_migration import validate_execution_benchmark
from training.sft.sft_training import load_dataset_bundle


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "certificate", "bundle", "worklist", "control-config", "baseline-root"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--baseline-preflight", help="Matching model baseline readiness; required only for fallback.")
    parser.add_argument("--family", choices=("rot13", "kazitext"), required=True)
    parser.add_argument("--model", choices=("7b", "14b"), default="7b")
    parser.add_argument("--deadline", default=datetime.max.replace(tzinfo=timezone.utc).isoformat())
    parser.add_argument("--profiles", nargs="+", default=["spatial_scheduled_ce_checkpointed"])
    parser.add_argument("--initial-checkpoint", help="Immutable baseline parent for level 1 only; never official initialization.")
    parser.add_argument("--profile-timeout-seconds", type=int, default=3600)
    parser.add_argument("--_prepare-level", type=int, choices=range(1, 11), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    from benchmark_sft_optimizations import PROFILE_SETTINGS
    if (len(set(args.profiles)) != len(args.profiles) or "baseline" in args.profiles
            or any(name not in PROFILE_SETTINGS for name in args.profiles)
            or args.profile_timeout_seconds < 1):
        parser.error("Choose distinct declared candidate profiles and a positive profile timeout")
    deadline = datetime.fromisoformat(args.deadline.replace("Z", "+00:00"))
    if deadline.utcoffset() is None or deadline.utcoffset().total_seconds() != 0:
        parser.error("Deadline must use UTC")
    return args


def condition(task):
    return {"family": task["family"], "model": task["model_id"], "level": task["level"]}


def publish_files(experiment, paths, metadata):
    """Stage payloads together; the final call must verify the whole batch."""
    paths = list(paths)
    for index, path in enumerate(paths):
        experiment.publish_permanent(path, {**metadata, "defer_until_batch_end": index < len(paths) - 1})


def prepare_parent(experiment, task, destination):
    """One exact baseline update, not a production fit or a training warm start."""
    from peft import get_peft_model_state_dict
    from shared_procedures.checkpointing import save_checkpoint, verify_checkpoint
    spec = sft_fit.SFTFitSpec.from_dict(task["spec"])
    identity = {**spec.identity(), "code_revision": experiment.code,
                "dependency_hash": experiment.config["runtime"]["dependency_lock_hash"]}
    if not destination.exists():
        train, validation = load_dataset_bundle(task["dataset"])
        sft_fit.validate_datasets(spec, train, validation)
        model, tokenizer, optimizer, scheduler, runtime = sft_fit._runtime(
            spec, experiment.config["runtime"].get("sft_device", "cuda:0"),
            cache_root=experiment.config["runtime"].get("cache_root"))
        recipe = spec.recipe
        order = sft_fit.exposure_order(len(train), epochs=recipe.epochs, seed=recipe.seed)
        selected = [sft_fit.tokenize_completion(train[index], tokenizer) for index in order[:16]]
        model.train()
        sft_fit.one_optimizer_update(model, optimizer, scheduler, selected, tokenizer.pad_token_id, recipe=recipe)
        progress = {"completed_steps": 1, "sampler_position": 16, "example_presentations": 16,
                    "exposure_order_hash": content_hash(order),
                    "total_optimizer_steps": len(order) // recipe.effective_batch_size,
                    "processed_tokens": sum(len(row["input_ids"]) for row in selected),
                    "execution_profile": SFTExecutionProfile().to_dict()}
        save_checkpoint(destination, identity, {"adapter": get_peft_model_state_dict(model, save_embedding_layers=False),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "rng": sft_fit.capture_rng(), "progress": progress, "runtime": runtime})
    manifest = verify_checkpoint(destination, identity)
    publish_files(experiment, [destination / "full_state.pt", destination / "manifest.json"],
                  {"identity": identity, "kind": "isolated_sft_benchmark_parent"})
    return manifest


def select_candidate(report, config_hash, certificate_hash, code, task, profiles):
    """Never expand one measured condition or accept a rehashed failed gate."""
    if report.get("receipt_hash") != content_hash({k: v for k, v in report.items() if k != "receipt_hash"}):
        raise ValueError("Benchmark receipt integrity differs")
    plan = report["plan"]
    if (plan.get("source_code_commit") != code or plan.get("configuration_hash") != config_hash
            or plan.get("handoff_certificate_hash") != certificate_hash
            or {k: plan.get("condition", {}).get(k) for k in ("family", "model", "level")} != condition(task)
            or plan.get("consumer_bindings_valid") is not True or plan.get("binding_errors") != []):
        raise ValueError("Benchmark does not bind this exact consumer condition")
    if report.get("status") != "MEASURED":
        return None
    accepted = []
    for name in profiles:
        if report.get("comparisons", {}).get(name, {}).get("technical_gate") != "PASSED":
            continue
        profile = SFTExecutionProfile.from_dict(plan["profiles"][name])
        approval = validate_execution_benchmark(report, code, profile, profile_name=name)
        if (approval.get("recovery_validated") is not True or approval["condition"] != condition(task)
                or approval["configuration_hash"] != config_hash or approval["handoff_certificate_hash"] != certificate_hash):
            raise ValueError("Accepted candidate lacks exact-condition full recovery evidence")
        accepted.append((report["comparisons"][name]["minimum_paired_update_speedup"], name))
    return max(accepted)[1] if accepted else None


def training_command(args, preflight, level, benchmark=None, profile=None):
    command = [sys.executable, str(ROOT / "scripts/run_certified_sft.py")]
    for name in ("config", "certificate", "bundle", "worklist", "family", "model", "deadline"):
        command += ["--" + name, str(getattr(args, name))]
    command += ["--preflight", str(preflight), "--level", str(level)]
    if benchmark is not None:
        command += ["--execution-benchmark", str(benchmark), "--execution-profile", profile]
    return command


def completed_condition(experiment, task, level_root):
    """Skip only a final fit verified against this queue's durable decision."""
    output = experiment.artifacts / "training" / "sft" / task["family"] / task["model_id"] / f"level_{task['level']:02d}"
    if not (output / "fit_result.json").exists():
        return False
    decision = read_json(level_root / "selection.json")
    if (decision.get("selection_hash") != content_hash({k: v for k, v in decision.items() if k != "selection_hash"})
            or decision.get("condition") != condition(task) or decision.get("source_code_commit") != experiment.code
            or decision.get("configuration_hash") != experiment.hash
            or decision.get("handoff_certificate_hash") != task["spec"]["handoff_certificate_hash"]):
        raise ValueError("Completed condition differs from its exact queue decision")
    spec = sft_fit.SFTFitSpec.from_dict(task["spec"])
    identity = {**spec.identity(), "code_revision": experiment.code,
                "dependency_hash": experiment.config["runtime"]["dependency_lock_hash"]}
    if decision["execution_profile"] != "baseline":
        report = read_json(level_root / "benchmark" / "report.json")
        profile = SFTExecutionProfile.from_dict(report["plan"]["profiles"][decision["execution_profile"]])
        approval = validate_execution_benchmark(report, experiment.code, profile, profile_name=decision["execution_profile"])
        if (approval["condition"] != condition(task) or approval["benchmark_report_hash"] != decision["benchmark_report_hash"]
                or approval["configuration_hash"] != experiment.hash
                or approval["handoff_certificate_hash"] != decision["handoff_certificate_hash"]):
            raise ValueError("Completed condition optimization evidence differs")
        identity["execution_profile"] = profile.to_dict()
    sft_fit.verify_fit_result(output, identity)
    return True


def pause_child(process):
    if process is not None and process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def main(argv=None):
    args = arguments(argv)
    config = validate_configuration(read_json(args.config), runtime=True)
    if config['sft'].get('training_budget') == 'efficient_one_epoch_v1':
        raise ValueError('The finalized published recipe uses run_certified_sft.py with an explicit model and level; no optimization campaign queue')
    experiment = Experiment(config, config_path=args.config)
    if experiment.code != config["runtime"]["implementation_revision"]:
        raise ValueError("Executed source differs from the frozen consumer")
    certificate = read_json(args.certificate)
    tasks = certified_tasks(config, certificate, read_json(args.bundle), read_json(args.worklist),
                            family=args.family, model=args.model)
    if args.baseline_preflight:
        baseline = load_preflight_evidence(args.baseline_preflight, config)
        require_preflights(baseline, preflight_requirements("training"), expected_configuration_hash=experiment.hash)
        if (baseline.get("ready_scope") is not None
                or (baseline.get("ready_models") is not None and args.model not in baseline["ready_models"])):
            raise ValueError("Baseline fallback must authorize the selected model across levels")
    root = experiment.artifacts / "optimization_queue" / args.family / args.model
    root.mkdir(parents=True, exist_ok=True)
    if args._prepare_level:
        task = next(task for task in tasks if task["level"] == args._prepare_level)
        prepare_parent(experiment, task, root / f"level_{task['level']:02d}" / "parent" / "checkpoint_000001")
        return 0
    deadline = datetime.fromisoformat(args.deadline.replace("Z", "+00:00"))
    stop, active, completed = {"requested": False}, {"process": None}, []
    def request_pause(_signum=None, _frame=None):
        stop["requested"] = True
        pause_child(active["process"])
    signal.signal(signal.SIGTERM, request_pause)
    signal.signal(signal.SIGINT, request_pause)
    def run(command, log):
        with log.open("a") as handle:
            process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            active["process"] = process
            try:
                if stop["requested"]:
                    request_pause()
                while process.poll() is None:
                    if not stop["requested"] and datetime.now(timezone.utc) >= deadline:
                        request_pause()
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        pass
                return process.returncode
            finally:
                active["process"] = None
    def status(phase, task=None, **extra):
        value = {"status": phase, "source_code_commit": experiment.code, "configuration_hash": experiment.hash,
                 "handoff_certificate_hash": certificate["certificate_hash"], "family": args.family, "model": args.model,
                 "required_fits": len(tasks), "completed_fits": len(completed), "completed_levels": list(completed),
                 "current_level": task["level"] if task else None, "at": datetime.now(timezone.utc).isoformat(), **extra}
        path = root / "queue_status.json"
        atomic_write_json(path, value)
        experiment.publish_permanent(path, {"kind": "optimized_sft_queue_status"})
        print(__import__("json").dumps(value), flush=True)
    with file_lock(root / ".queue.lock", timeout=1):
        try:
            for task in tasks:
                if stop["requested"] or datetime.now(timezone.utc) >= deadline:
                    status("PAUSED", task, drive_status="VERIFIED")
                    return 3
                level_root = root / f"level_{task['level']:02d}"
                level_root.mkdir(parents=True, exist_ok=True)
                if completed_condition(experiment, task, level_root):
                    completed.append(task["level"])
                    status("CONDITION_COMPLETE", task, drive_status="VERIFIED", verified_existing_completion=True)
                    continue
                status("QUALIFYING", task)
                parent = (Path(args.initial_checkpoint).resolve() if task["level"] == 1 and args.initial_checkpoint else
                          level_root / "parent" / "checkpoint_000001")
                if task["level"] != 1 or not args.initial_checkpoint:
                    command = [sys.executable, str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:]),
                               "--_prepare-level", str(task["level"])]
                    result = run(command, level_root / "parent.log")
                    if result and not stop["requested"]:
                        raise RuntimeError("Isolated baseline checkpoint preparation failed")
                else:
                    # Imported immutable parents enter the new Drive publication namespace unchanged.
                    from benchmark_sft_optimizations import recovery_binding
                    import shutil
                    recovery_binding(parent, task, config)
                    copied = level_root / "parent" / parent.name
                    if not copied.exists():
                        shutil.copytree(parent, copied)
                    from benchmark_sft_optimizations import recovery_binding as verify_parent
                    verify_parent(copied, task, config)
                    publish_files(experiment, [copied / "full_state.pt", copied / "manifest.json"], {"kind": "isolated_sft_benchmark_parent"})
                    parent = copied
                if stop["requested"]:
                    status("PAUSED", task, drive_status="VERIFIED")
                    return 3
                benchmark = level_root / "benchmark"
                if not (benchmark / "report.json").is_file():
                    if benchmark.exists() and any(benchmark.iterdir()):
                        benchmark.rename(level_root / ("interrupted_benchmark_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")))
                    command = [sys.executable, str(ROOT / "scripts/benchmark_sft_optimizations.py")]
                    for name in ("config", "certificate", "bundle", "worklist", "family", "model"):
                        command += ["--" + name, str(getattr(args, name))]
                    command += ["--level", str(task["level"]), "--profiles", "baseline", *args.profiles,
                        "--reference-profile", "baseline", "--warmup-updates", "1", "--updates", "5", "--repetitions", "2",
                        "--profile-timeout-seconds", str(args.profile_timeout_seconds), "--recovery-checkpoint", str(parent), "--output", str(benchmark)]
                    result = run(command, level_root / "benchmark.log")
                    if stop["requested"]:
                        status("PAUSED", task, drive_status="VERIFIED")
                        return 3
                    if result not in (0, 2, 3) or not (benchmark / "report.json").is_file():
                        raise RuntimeError("Benchmark exited without a complete receipt")
                publish_files(experiment, sorted(path for path in benchmark.rglob("*") if path.is_file()),
                              {"kind": "isolated_sft_execution_benchmark"})
                selected = select_candidate(read_json(benchmark / "report.json"), experiment.hash,
                    certificate["certificate_hash"], experiment.code, task, args.profiles)
                preflight = args.baseline_preflight
                if selected is None and not preflight:
                    raise RuntimeError("No candidate passed and no matching baseline fallback readiness was supplied")
                if selected:
                    launch = level_root / "launch"
                    command = [sys.executable, str(ROOT / "scripts/reconcile_sft_launch.py")]
                    for name in ("config", "certificate", "control-config", "baseline-root"):
                        command += ["--" + name, str(getattr(args, name.replace("-", "_")))]
                    command += ["--execution-benchmark", str(benchmark / "report.json"), "--execution-profile", selected,
                                "--output", str(launch)]
                    result = run(command, level_root / "reconcile.log")
                    if result and not stop["requested"]:
                        raise RuntimeError("Accepted condition failed launch reconciliation")
                    if stop["requested"]:
                        status("PAUSED", task, drive_status="VERIFIED")
                        return 3
                    publish_files(experiment, sorted(path for path in launch.rglob("*") if path.is_file()), {"kind": "sft_launch_evidence"})
                    preflight = launch / "preflight" / "report.json"
                decision = {"condition": condition(task), "source_code_commit": experiment.code, "configuration_hash": experiment.hash,
                    "handoff_certificate_hash": certificate["certificate_hash"], "execution_profile": selected or "baseline",
                    "benchmark_report_hash": read_json(benchmark / "report.json")["receipt_hash"],
                    "official_initialization": "fresh_base_or_exact_current_namespace_checkpoint"}
                decision["selection_hash"] = content_hash(decision)
                decision_path = level_root / "selection.json"
                if decision_path.exists() and read_json(decision_path) != decision:
                    raise ValueError("Existing exact-condition execution decision changed")
                atomic_write_json(decision_path, decision)
                experiment.publish_permanent(decision_path, {"kind": "sft_execution_selection"})
                status("TRAINING", task, execution_profile=selected or "baseline", initialization="fresh_base_or_exact_current_namespace_checkpoint")
                result = run(training_command(args, preflight, task["level"], benchmark / "report.json" if selected else None, selected),
                             level_root / "training.log")
                if result == 3 or stop["requested"]:
                    status("PAUSED", task, drive_status="VERIFIED")
                    return 3
                if result:
                    raise RuntimeError("Official condition failed; queue stopped")
                completed.append(task["level"])
                status("CONDITION_COMPLETE", task, drive_status="VERIFIED")
            status("COMPLETE", drive_status="VERIFIED")
            return 0
        except Exception as error:
            status("FAILED", detail=str(error), drive_status="REQUIRED")
            raise


if __name__ == "__main__":
    raise SystemExit(main())
