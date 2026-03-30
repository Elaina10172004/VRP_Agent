from __future__ import annotations

import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config" / "runtime_defaults.json"


def _load_config() -> dict:
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    gpu_budget_mb = int(raw["gpu_budget_mb"])
    fast_single_instance_vram_mb = int(raw["fast_single_instance_vram_mb"])
    thinking_single_instance_vram_mb = int(raw["thinking_single_instance_vram_mb"])

    return {
        "gpu_budget_mb": gpu_budget_mb,
        "fast_single_instance_vram_mb": fast_single_instance_vram_mb,
        "thinking_single_instance_vram_mb": thinking_single_instance_vram_mb,
        "fast_default_instance_parallelism": max(1, gpu_budget_mb // fast_single_instance_vram_mb),
        "thinking_default_instance_parallelism": max(1, gpu_budget_mb // thinking_single_instance_vram_mb),
    }


RUNTIME_DEFAULTS = _load_config()

GPU_BUDGET_MB = RUNTIME_DEFAULTS["gpu_budget_mb"]
FAST_SINGLE_INSTANCE_VRAM_MB = RUNTIME_DEFAULTS["fast_single_instance_vram_mb"]
THINKING_SINGLE_INSTANCE_VRAM_MB = RUNTIME_DEFAULTS["thinking_single_instance_vram_mb"]
FAST_DEFAULT_INSTANCE_PARALLELISM = RUNTIME_DEFAULTS["fast_default_instance_parallelism"]
THINKING_DEFAULT_INSTANCE_PARALLELISM = RUNTIME_DEFAULTS["thinking_default_instance_parallelism"]


def default_instance_parallelism(mode: str) -> int:
    normalized = str(mode).strip().lower()
    if normalized == "fast":
        return FAST_DEFAULT_INSTANCE_PARALLELISM
    return THINKING_DEFAULT_INSTANCE_PARALLELISM


def build_runtime_defaults_payload() -> dict:
    return {
        "gpuBudgetMb": GPU_BUDGET_MB,
        "fastSingleInstanceVramMb": FAST_SINGLE_INSTANCE_VRAM_MB,
        "thinkingSingleInstanceVramMb": THINKING_SINGLE_INSTANCE_VRAM_MB,
        "fastMaxParallelInstances": FAST_DEFAULT_INSTANCE_PARALLELISM,
        "thinkingMaxParallelInstances": THINKING_DEFAULT_INSTANCE_PARALLELISM,
    }


if __name__ == "__main__":
    print(json.dumps(build_runtime_defaults_payload(), ensure_ascii=False))
