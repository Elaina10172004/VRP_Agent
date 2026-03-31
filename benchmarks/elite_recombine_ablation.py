from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from instance_skill.api import ingest_uploaded_file
from solver_skill.api import solve_payload


def run_case(instance_name: str, enabled: bool) -> dict:
    random.seed(123)
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(123)

    ingested = ingest_uploaded_file(str(ROOT / "solomon_data" / instance_name))
    result = solve_payload(
        {
            "problem_type": ingested["payload"]["problem_type"],
            "instance": ingested["payload"]["instance"],
            "config": {
                "mode": "fast",
                "drl_samples": 128,
                "seed_trials": 8,
                "enable_local_search": True,
                "local_search_rounds": 8,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "objective": {"primary": "distance"},
                "enable_elite_recombine": enabled,
                "elite_pool_size": 32,
                "elite_recombine_polish_rounds": 8,
                "enable_multistart_local_search": enabled,
                "multistart_local_search_candidates": 2,
            },
        }
    )
    return {
        "instance": instance_name,
        "enabled": enabled,
        "final_distance": float(result["final_solution"]["distance"]),
        "vehicle_count": int(result["meta"]["final_score"]["vehicle_count"]),
        "elite_recombine": result["meta"]["elite_recombine"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="A/B benchmark elite-structure recombination on Solomon instances.")
    parser.add_argument(
        "--instances",
        default="R101.txt,R201.txt,RC101.txt,RC208.txt",
        help="Comma-separated Solomon instance filenames.",
    )
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    torch.use_deterministic_algorithms(True, warn_only=True)
    instances = [item.strip() for item in args.instances.split(",") if item.strip()]
    rows = []
    for instance_name in instances:
        baseline = run_case(instance_name, enabled=False)
        experiment = run_case(instance_name, enabled=True)
        rows.append(
            {
                "instance": instance_name,
                "baseline_distance": baseline["final_distance"],
                "elite_distance": experiment["final_distance"],
                "improvement": baseline["final_distance"] - experiment["final_distance"],
                "elite_recombine": experiment["elite_recombine"],
            }
        )

    if args.pretty:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(rows, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
