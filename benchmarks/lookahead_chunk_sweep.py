from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from instance_skill.api import ingest_uploaded_file
from solver_core.cvrptw_lookahead import solve_cvrptw_with_decode_lookahead


def load_instance() -> dict:
    default_path = ROOT / "solomon_data" / "C101.txt"
    ingested = ingest_uploaded_file(str(default_path))
    return ingested["payload"]["instance"]


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


def run_worker(chunk_size: int, *, num_samples: int, top_k: int, confident_prob: float) -> dict:
    return {
        "command": [
            sys.executable,
            str(Path(__file__).resolve()),
            "worker",
            "--chunk-size",
            str(chunk_size),
            "--num-samples",
            str(num_samples),
            "--top-k",
            str(top_k),
            "--confident-prob",
            str(confident_prob),
        ],
        "process": subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "worker",
                "--chunk-size",
                str(chunk_size),
                "--num-samples",
                str(num_samples),
                "--top-k",
                str(top_k),
                "--confident-prob",
                str(confident_prob),
            ],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ),
    }


def monitor_process(process_entry: dict, baseline: dict) -> dict:
    baseline_shared = float(baseline["primary_adapter"]["shared_mb"])
    baseline_dedicated = float(baseline["primary_adapter"]["dedicated_mb"])
    baseline_nvidia_used = float(baseline["nvidia"]["memory_used_mb"])

    peak_shared = baseline_shared
    peak_dedicated = baseline_dedicated
    peak_nvidia_used = baseline_nvidia_used

    started_at = time.perf_counter()
    while process_entry["process"].poll() is None:
        snapshot = query_gpu_snapshot()
        peak_shared = max(peak_shared, float(snapshot["primary_adapter"]["shared_mb"]))
        peak_dedicated = max(peak_dedicated, float(snapshot["primary_adapter"]["dedicated_mb"]))
        peak_nvidia_used = max(peak_nvidia_used, float(snapshot["nvidia"]["memory_used_mb"]))
        time.sleep(0.25)

    monitor_duration_s = time.perf_counter() - started_at
    stdout, stderr = process_entry["process"].communicate(timeout=15)
    if process_entry["process"].returncode != 0:
        raise RuntimeError(stderr.strip() or stdout.strip() or f"Worker failed with code {process_entry['process'].returncode}")

    worker_payload = json.loads(stdout.strip())
    return {
        "baseline_nvidia_used_mb": baseline_nvidia_used,
        "baseline_primary_dedicated_mb": baseline_dedicated,
        "baseline_primary_shared_mb": baseline_shared,
        "peak_nvidia_used_mb": peak_nvidia_used,
        "peak_primary_dedicated_mb": peak_dedicated,
        "peak_primary_shared_mb": peak_shared,
        "delta_nvidia_used_mb": peak_nvidia_used - baseline_nvidia_used,
        "delta_primary_dedicated_mb": peak_dedicated - baseline_dedicated,
        "delta_primary_shared_mb": peak_shared - baseline_shared,
        "monitor_duration_s": monitor_duration_s,
        "worker": worker_payload,
    }


def sweep_chunks(chunk_sizes: list[int], *, num_samples: int, top_k: int, confident_prob: float) -> dict:
    history = []
    for chunk_size in chunk_sizes:
        baseline = query_gpu_snapshot()
        process_entry = run_worker(chunk_size, num_samples=num_samples, top_k=top_k, confident_prob=confident_prob)
        run = monitor_process(process_entry, baseline)
        run["chunk_size"] = chunk_size
        history.append(run)
    return {
        "gpu": query_nvidia_smi(),
        "torch_cuda_available": torch.cuda.is_available(),
        "chunk_sizes": chunk_sizes,
        "num_samples": num_samples,
        "top_k": top_k,
        "confident_prob": confident_prob,
        "history": history,
    }


def worker_main(chunk_size: int, *, num_samples: int, top_k: int, confident_prob: float) -> int:
    instance = load_instance()
    started_at = time.perf_counter()
    result = solve_cvrptw_with_decode_lookahead(
        instance["depot_xy"],
        instance["node_xy"],
        instance["node_demand"],
        instance["capacity"],
        instance["node_tw"],
        instance["service_time"],
        num_samples=num_samples,
        top_k=top_k,
        confident_prob=confident_prob,
        uncertain_chunk_size=chunk_size,
        objective={"primary": "distance"},
        device="cuda",
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "problem_type": result["problem_type"],
                "distance": result["distance"],
                "solve_duration_s": time.perf_counter() - started_at,
                "meta": result.get("meta", {}),
            }
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark different lookahead chunk sizes on a single GPU instance.")
    subparsers = parser.add_subparsers(dest="command")

    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--chunk-size", type=int, required=True)
    worker_parser.add_argument("--num-samples", type=int, default=128)
    worker_parser.add_argument("--top-k", type=int, default=3)
    worker_parser.add_argument("--confident-prob", type=float, default=0.95)

    parser.add_argument("--chunk-sizes", default="8,16,32,64,128,256", help="Comma-separated chunk sizes to test.")
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--confident-prob", type=float, default=0.95)
    args = parser.parse_args()

    if args.command == "worker":
        return worker_main(
            args.chunk_size,
            num_samples=args.num_samples,
            top_k=args.top_k,
            confident_prob=args.confident_prob,
        )

    chunk_sizes = [int(item) for item in str(args.chunk_sizes).split(",") if item.strip()]
    summary = sweep_chunks(
        chunk_sizes,
        num_samples=args.num_samples,
        top_k=args.top_k,
        confident_prob=args.confident_prob,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
