from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solver_skill import solve_payload


def build_cvrptw_instance(node_count: int = 100, seed: int = 20260330) -> dict:
    rng = random.Random(seed)
    node_xy = []
    node_demand = []
    node_tw = []
    for _ in range(node_count):
        node_xy.append([round(rng.uniform(0, 100), 3), round(rng.uniform(0, 100), 3)])
        node_demand.append(rng.randint(1, 9))
        node_tw.append([0, 1000])

    return {
        "depot_xy": [50.0, 50.0],
        "node_xy": node_xy,
        "node_demand": node_demand,
        "capacity": 40,
        "node_tw": node_tw,
        "service_time": 0,
        "grid_scale": 100,
    }


def build_drl_payload() -> dict:
    return {
        "problem_type": "cvrptw",
        "instance": build_cvrptw_instance(),
        "config": {
            "mode": "fast",
            "drl_samples": 128,
            "seed_trials": 1,
            "enable_lookahead": False,
            "enable_local_search": False,
            "device": "cuda",
        },
    }


def build_lookahead_payload(candidate_k: int) -> dict:
    return {
        "problem_type": "cvrptw",
        "instance": build_cvrptw_instance(),
        "config": {
            "mode": "hybrid",
            "drl_samples": candidate_k,
            "seed_trials": 1,
            "enable_lookahead": True,
            "lookahead_k": candidate_k,
            "lookahead_depth": 2,
            "lookahead_beam_width": 4,
            "enable_local_search": False,
            "device": "cuda",
        },
    }


def query_nvidia_smi() -> dict:
    output = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    parts = [part.strip() for part in output.split(",")]
    return {
        "name": parts[0],
        "memory_total_mb": float(parts[1]),
        "memory_used_mb": float(parts[2]),
        "memory_free_mb": float(parts[3]),
        "driver_version": parts[4],
    }


def query_gpu_counters() -> list[dict]:
    script = r"""
$samples = Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage','\GPU Adapter Memory(*)\Shared Usage' | ForEach-Object {
  $_.CounterSamples | ForEach-Object {
    [PSCustomObject]@{
      InstanceName = $_.InstanceName
      Counter = if ($_.Path -like '*dedicated usage') { 'dedicated' } else { 'shared' }
      ValueMB = [math]::Round($_.CookedValue / 1MB, 2)
    }
  }
}
$samples | ConvertTo-Json -Depth 3
"""
    output = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    parsed = json.loads(output)
    if isinstance(parsed, dict):
        parsed = [parsed]

    grouped: dict[str, dict] = {}
    for item in parsed:
        adapter = grouped.setdefault(item["InstanceName"], {"instance_name": item["InstanceName"], "dedicated_mb": 0.0, "shared_mb": 0.0})
        if item["Counter"] == "dedicated":
            adapter["dedicated_mb"] = float(item["ValueMB"])
        else:
            adapter["shared_mb"] = float(item["ValueMB"])
    return list(grouped.values())


def query_gpu_snapshot() -> dict:
    nvidia = query_nvidia_smi()
    adapters = query_gpu_counters()
    primary = max(adapters, key=lambda item: item["dedicated_mb"])
    return {
        "nvidia": nvidia,
        "adapters": adapters,
        "primary_adapter": primary,
    }


def run_worker(mode: str, value: int | None = None) -> dict:
    command = [sys.executable, str(Path(__file__).resolve())]
    if mode == "drl-worker":
        command.append("drl-worker")
    else:
        command.extend(["lookahead-worker", "--k", str(value or 1)])

    return {
        "command": command,
        "process": subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ),
    }


def monitor_processes(process_entries: list[dict], shared_threshold_mb: float, baseline: dict) -> dict:
    baseline_shared = float(baseline["primary_adapter"]["shared_mb"])
    baseline_dedicated = float(baseline["primary_adapter"]["dedicated_mb"])

    peak_nvidia_used = float(baseline["nvidia"]["memory_used_mb"])
    peak_shared = baseline_shared
    peak_dedicated = baseline_dedicated

    started_at = time.perf_counter()
    while any(entry["process"].poll() is None for entry in process_entries):
        snapshot = query_gpu_snapshot()
        peak_nvidia_used = max(peak_nvidia_used, float(snapshot["nvidia"]["memory_used_mb"]))
        peak_shared = max(peak_shared, float(snapshot["primary_adapter"]["shared_mb"]))
        peak_dedicated = max(peak_dedicated, float(snapshot["primary_adapter"]["dedicated_mb"]))
        time.sleep(0.4)

    duration_s = time.perf_counter() - started_at
    worker_results = []
    exit_codes = []
    for entry in process_entries:
        stdout, stderr = entry["process"].communicate(timeout=15)
        exit_codes.append(entry["process"].returncode)
        parsed_stdout = None
        if stdout.strip():
            try:
                parsed_stdout = json.loads(stdout.strip())
            except json.JSONDecodeError:
                parsed_stdout = {"raw_stdout": stdout.strip()}
        worker_results.append(
            {
                "command": entry["command"],
                "stdout": parsed_stdout,
                "stderr": stderr.strip(),
                "exit_code": entry["process"].returncode,
            }
        )

    shared_delta = peak_shared - baseline_shared
    return {
        "baseline": baseline,
        "peak_nvidia_used_mb": peak_nvidia_used,
        "peak_primary_dedicated_mb": peak_dedicated,
        "peak_primary_shared_mb": peak_shared,
        "shared_delta_mb": shared_delta,
        "duration_s": duration_s,
        "ok": all(code == 0 for code in exit_codes) and shared_delta <= shared_threshold_mb,
        "shared_threshold_mb": shared_threshold_mb,
        "worker_results": worker_results,
    }


def benchmark_drl_concurrency(max_instances: int, shared_threshold_mb: float) -> dict:
    history = []
    best_run = None

    for instance_count in range(1, max_instances + 1):
        baseline = query_gpu_snapshot()
        process_entries = [run_worker("drl-worker") for _ in range(instance_count)]
        run = monitor_processes(process_entries, shared_threshold_mb, baseline)
        run["instances"] = instance_count
        history.append(run)
        if run["ok"]:
            best_run = run
            continue
        break

    return {
        "best_instances": best_run["instances"] if best_run else 0,
        "best_run": best_run,
        "history": history,
    }


def benchmark_lookahead_k(candidate_values: list[int], shared_threshold_mb: float) -> dict:
    history = []
    best_run = None

    for candidate_k in candidate_values:
        baseline = query_gpu_snapshot()
        process_entries = [run_worker("lookahead-worker", candidate_k)]
        run = monitor_processes(process_entries, shared_threshold_mb, baseline)
        run["lookahead_k"] = candidate_k
        history.append(run)
        if run["ok"]:
            best_run = run
            continue
        break

    return {
        "best_k": best_run["lookahead_k"] if best_run else 0,
        "best_run": best_run,
        "history": history,
    }


def worker_drl() -> int:
    payload = build_drl_payload()
    started_at = time.perf_counter()
    result = solve_payload(payload)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "problem_type": result["problem_type"],
                "distance": result["final_solution"]["distance"],
                "duration_s": time.perf_counter() - started_at,
                "meta": result["meta"],
            }
        )
    )
    return 0


def worker_lookahead(candidate_k: int) -> int:
    payload = build_lookahead_payload(candidate_k)
    started_at = time.perf_counter()
    result = solve_payload(payload)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "problem_type": result["problem_type"],
                "distance": result["final_solution"]["distance"],
                "duration_s": time.perf_counter() - started_at,
                "meta": result["meta"],
            }
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark local GPU capacity for DRL concurrency and lookahead-k search.")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("drl-worker")

    lookahead_worker = subparsers.add_parser("lookahead-worker")
    lookahead_worker.add_argument("--k", type=int, required=True)

    parser.add_argument("--max-instances", type=int, default=12, help="Maximum number of concurrent k=128 DRL instances to test.")
    parser.add_argument("--shared-threshold-mb", type=float, default=256.0, help="Maximum tolerated shared-memory increase on the primary GPU adapter.")
    parser.add_argument(
        "--lookahead-k-values",
        default="1,2,4,8,16,32,64,128,256,512",
        help="Comma-separated candidate K values for single-instance lookahead benchmarking.",
    )
    parser.add_argument("--skip-drl", action="store_true", help="Skip the DRL concurrency benchmark.")
    parser.add_argument("--skip-lookahead", action="store_true", help="Skip the lookahead-k benchmark.")
    args = parser.parse_args()

    if args.command == "drl-worker":
        return worker_drl()
    if args.command == "lookahead-worker":
        return worker_lookahead(args.k)

    lookahead_values = [int(item) for item in args.lookahead_k_values.split(",") if item.strip()]
    summary = {
        "gpu": query_nvidia_smi(),
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_device_count": torch.cuda.device_count(),
        "drl_concurrency": None if args.skip_drl else benchmark_drl_concurrency(args.max_instances, args.shared_threshold_mb),
        "lookahead_k": None if args.skip_lookahead else benchmark_lookahead_k(lookahead_values, args.shared_threshold_mb),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
