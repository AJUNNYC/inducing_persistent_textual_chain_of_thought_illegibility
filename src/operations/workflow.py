"""Dependency-aware execution with immutable inputs and persistent completion checks."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import subprocess
import os
from typing import Callable
from shared_procedures.identity import content_hash
from shared_procedures.storage import atomic_write_json, read_json, file_lock
from operations.drive import file_hash


@dataclass(frozen=True)
class Stage:
    stage_id: str
    kind: str
    dependencies: tuple[str, ...]
    model: str | None = None
    branch: str | None = None
    level: int | None = None
    gpu_count: int = 0
    paid_services: bool = False


def experiment_graph() -> dict[str, Stage]:
    stages = [Stage("assets", "provision", ()), Stage("panels", "prepare_panels", ("assets",)),
              Stage("candidates", "prepare_candidates", ("assets", "panels")),
              Stage("generation_preflight", "generation_preflight", ("candidates",), gpu_count=1),
              Stage("clean", "collect_clean", ("generation_preflight",), gpu_count=1),
              Stage("sft_data", "transform", ("clean",)), Stage("rl_data", "prepare_rl", ("assets", "panels")),
              Stage("analysis_plan", "freeze_analysis", ("panels",)),
              Stage("statistical_validation", "validate_statistics", ("analysis_plan",))]
    for model in ("7b", "14b"):
        stages.append(Stage(f"baseline_{model}", "baseline", ("panels", "analysis_plan"), model=model, gpu_count=1, paid_services=True))
    stages.append(Stage("baseline_repeatability", "repeatability", ("baseline_7b", "baseline_14b")))
    stages.append(Stage("preflight", "preflight", ("sft_data", "rl_data", "baseline_repeatability", "statistical_validation"), paid_services=True))
    evaluations = []
    for model in ("7b", "14b"):
        for branch in ("rot13", "kazitext"):
            for level in range(1, 11):
                job = f"{branch}_{model}_level_{level:02d}"
                stages.append(Stage(job, "sft", ("sft_data", "preflight"), model=model, branch=branch, level=level, gpu_count=1))
                evaluation = f"evaluate_{job}"
                stages.append(Stage(evaluation, "evaluate", (job, "panels"), model=model, branch=branch, level=level, gpu_count=1, paid_services=True))
                evaluations.append(evaluation)
        trajectory = f"rl_{model}"
        stages.append(Stage(trajectory, "rl", ("rl_data", "preflight"), model=model, branch="rl", gpu_count=1, paid_services=True))
        for step in range(10, 101, 10):
            evaluation = f"evaluate_rl_{model}_step_{step:03d}"
            stages.append(Stage(evaluation, "evaluate", (trajectory, "panels"), model=model, branch="rl", level=step, gpu_count=1, paid_services=True))
            evaluations.append(evaluation)
    stages.append(Stage("analysis", "analyze", tuple(evaluations) + ("baseline_7b", "baseline_14b", "statistical_validation")))
    stages.append(Stage("reports", "report", ("analysis",)))
    stages.append(Stage("sync", "sync", ("reports",)))
    graph = {stage.stage_id: stage for stage in stages}
    topological_order(graph, list(graph))
    return graph


def topological_order(graph: dict[str, Stage], targets: list[str]) -> list[str]:
    order, completed, visiting = [], set(), set()
    def visit(stage_id):
        if stage_id not in graph:
            raise ValueError(f"Unknown execution stage: {stage_id}")
        if stage_id in visiting:
            raise ValueError("Circular execution dependencies")
        if stage_id in completed:
            return
        visiting.add(stage_id)
        for parent in graph[stage_id].dependencies:
            visit(parent)
        visiting.remove(stage_id)
        completed.add(stage_id)
        order.append(stage_id)
    for target in targets:
        visit(target)
    return order


class StageRunner:
    def __init__(self, graph: dict[str, Stage], state_root: str | Path, config_hash: str, code_revision: str):
        self.graph = graph
        self.root = Path(state_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.config_hash = config_hash
        self.code_revision = code_revision

    def _path(self, stage_id):
        if stage_id not in self.graph:
            raise ValueError("Unknown stage ID")
        return self.root / f"{stage_id}.json"

    def status(self, stage_id: str) -> dict:
        path = self._path(stage_id)
        if not path.exists():
            dependencies = [d for d in self.graph[stage_id].dependencies if self.status(d)["status"] != "COMPLETED"]
            return {"stage_id": stage_id, "status": "WAITING_FOR_DEPENDENCIES" if dependencies else "NOT_STARTED", "missing_dependencies": dependencies}
        record = read_json(path)
        if record["configuration_hash"] != self.config_hash or record["code_revision"] != self.code_revision:
            return {**record, "status": "INVALIDATED", "reason": "Configuration or implementation changed"}
        if record["status"] in ("COMPLETED", "PAUSED"):
            for file, digest in record.get("output_hashes", {}).items():
                if not Path(file).is_file() or file_hash(Path(file)) != digest:
                    return {**record, "status": "INVALIDATED", "reason": "Output missing or changed"}
            for parent in self.graph[stage_id].dependencies:
                upstream = self.status(parent)
                if upstream["status"] != "COMPLETED" or upstream.get("completion_hash") != record.get("dependency_hashes", {}).get(parent):
                    return {**record, "status": "INVALIDATED", "reason": "Upstream identity changed"}
        return record

    def inspect(self, targets: list[str]) -> list[dict]:
        return [{**asdict(self.graph[s]), **self.status(s)} for s in topological_order(self.graph, targets)]

    def run(self, targets: list[str], actions: dict[str, Callable[[Stage], dict]]) -> list[dict]:
        results = []
        for stage_id in topological_order(self.graph, targets):
            stage = self.graph[stage_id]
            with file_lock(str(self._path(stage_id)) + ".lock"):
                status = self.status(stage_id)
                if status["status"] == "COMPLETED":
                    results.append(status)
                    continue
                parent_status = {d: self.status(d) for d in stage.dependencies}
                if any(v["status"] != "COMPLETED" for v in parent_status.values()):
                    raise RuntimeError(f"Dependencies incomplete for {stage_id}")
                record = {"stage_id": stage_id, "configuration_hash": self.config_hash, "code_revision": self.code_revision,
                          "dependency_hashes": {d: v["completion_hash"] for d, v in parent_status.items()}, "status": "RUNNING"}
                atomic_write_json(self._path(stage_id), record)
                try:
                    action = actions.get(stage.kind)
                    if action is None:
                        raise RuntimeError(f"Execution handler not registered for stage kind {stage.kind}")
                    # One lock per actual GPU resource prevents independent worker overcommit.
                    if stage.gpu_count:
                        with file_lock(self.root / ("gpu_" + content_hash(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])[:16] + ".lock"), timeout=10):
                            result = action(stage)
                    else:
                        result = action(stage)
                    if result.get('status')=='PAUSED':
                        outputs=[Path(p).resolve() for p in result.get('output_files',[])]
                        if not outputs or any(not p.is_file() for p in outputs):
                            raise RuntimeError('Paused stage requires durable progress artifacts')
                        record.update(status='PAUSED',result=result,output_hashes={str(p):file_hash(p) for p in outputs})
                        atomic_write_json(self._path(stage_id),record)
                        results.append(record)
                        return results
                    if result.get("status") not in ("COMPLETED", "VERIFIED"):
                        raise RuntimeError(f"Stage did not satisfy completion: {result.get('status')}")
                    outputs = [Path(p).resolve() for p in result.get("output_files", [])]
                    if not outputs or any(not p.is_file() for p in outputs):
                        raise RuntimeError("Completed stage requires validated output artifacts")
                    record.update(status="COMPLETED", result=result, output_hashes={str(p): file_hash(p) for p in outputs})
                    record["completion_hash"] = content_hash({"identity": record["dependency_hashes"], "outputs": record["output_hashes"], "configuration_hash": self.config_hash, "code_revision": self.code_revision})
                except Exception as error:
                    record.update(status="BLOCKED" if isinstance(error, RuntimeError) else "FAILED", error_type=type(error).__name__, reason=str(error))
                    atomic_write_json(self._path(stage_id), record)
                    raise
                atomic_write_json(self._path(stage_id), record)
                results.append(record)
        return results
