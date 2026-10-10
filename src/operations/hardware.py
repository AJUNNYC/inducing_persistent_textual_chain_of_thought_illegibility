"""Report observed hardware separately from protocol deployment estimates."""
from __future__ import annotations
import os
import platform
import shutil
import subprocess
from pathlib import Path


def inspect_hardware(storage_root: str | Path = ".") -> dict:
    disk = shutil.disk_usage(storage_root)
    result = {"platform": platform.platform(), "python": platform.python_version(),
              "cpu_count": os.cpu_count(), "disk_free_bytes": disk.free,
              "disk_total_bytes": disk.total, "gpus": [], "bf16": False}
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        fields = dict(line.split(":", 1) for line in meminfo.read_text().splitlines())
        result["ram_bytes"] = int(fields["MemTotal"].strip().split()[0]) * 1024
    if shutil.which("nvidia-smi"):
        proc = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        if proc.returncode == 0:
            result["gpus"] = [{"name": f[0].strip(), "vram_mib": int(f[1]), "driver": f[2].strip()} for line in proc.stdout.splitlines() if (f := line.split(","))]
        else:
            result["gpu_query_status"] = "failed"
    try:
        import torch
        result["torch"] = torch.__version__
        result["cuda_runtime"] = torch.version.cuda
        result["cuda_available"] = torch.cuda.is_available()
        result["bf16"] = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    except ImportError:
        result["cuda_available"] = False
    return result


def validate_hardware(report: dict, *, gpu: bool = False, required_disk_bytes: int = 0) -> dict:
    failures = []
    if report["disk_free_bytes"] < required_disk_bytes:
        failures.append("Insufficient disk for selected assets and retained checkpoints")
    if gpu and (not report.get("cuda_available") or not report.get("bf16")):
        failures.append("Protocol training needs compatible CUDA and BF16; hardware runtime not available")
    return {"status": "BLOCKED" if failures else "VERIFIED", "failures": failures,
            "observations": report, "scope": "runtime prerequisites only; peak-memory profile still required"}
