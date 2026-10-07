from __future__ import annotations

import platform
import sys

import torch
import transformers

from src.config import DATA_PROCESSED, FIGURES, METRICS, REPORTS


def system_info() -> dict:
    info = {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
            "platform": platform.platform(terse=True), "cuda_available": torch.cuda.is_available(),
            "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
            "cpu_threads": torch.get_num_threads()}
    try:
        import psutil
        vm = psutil.virtual_memory()
        proc = psutil.Process().memory_info().rss
        info["memory"] = {"process_rss_mb": round(proc / 2**20), "system_total_mb": round(vm.total / 2**20),
                          "system_available_mb": round(vm.available / 2**20)}
    except ImportError:
        info["memory"] = None
    return info


def artifact_status() -> dict:
    checks = {
        "dataset_metadata": (DATA_PROCESSED / "summary.json").exists() and (DATA_PROCESSED / "label_maps.json").exists(),
        "evaluation_metrics": (METRICS / "comparison.json").exists(),
        "error_analysis": (REPORTS / "error_analysis.csv").exists(),
        "figures": any(FIGURES.glob("confusion_intent_*.png")) if FIGURES.exists() else False,
    }
    return checks


def device_name(models: dict) -> str:
    for m in models.values():
        if m.get("device"):
            return m["device"]
    return "cuda" if torch.cuda.is_available() else "cpu"


def health(models: dict) -> dict:
    loaded = [k for k, m in models.items() if m["status"] == "ready"]
    art = artifact_status()
    return {"status": "ok" if loaded else "degraded", "models_loaded": loaded, "models_total": len(models),
            "models": {k: m["status"] for k, m in models.items()}, "artifacts": art, "device": device_name(models),
            "python": sys.version.split()[0]}
