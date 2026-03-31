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
    warmup = solve_payload(
        {
            "problem_type": ingested["payload"]["problem_type"],
            "instance": ingested["payload"]["instance"],
            "config": {
                "mode": "hybrid",
                "drl_samples": 128,
                "enable_lookahead": True,
                "enable_local_search": False,
                "enable_destroy_repair": False,
                "enable_vehicle_reduction": False,
                "decode_lookahead_as_initial": True,
                "tool_plan": ["construct_initial", "validate_solution", "compare_solutions"],
                "lookahead_top_k": 3,
                "lookahead_confident_prob": 0.95,
                "lookahead_uncertain_chunk_size": 256,
                "objective": {"primary": "distance"},
                "device": "cuda" if torch.cuda.is_available() else "cpu",
            },
        }
    )
    base_solution = warmup["lookahead_solution"] or warmup["final_solution"]
    refined = solve_payload(
        {
            "problem_type": ingested["payload"]["problem_type"],
            "instance": ingested["payload"]["instance"],
            "starting_solution": base_solution,
            "config": {
                "mode": "hybrid",
                "enable_lookahead": False,
                "enable_local_search": True,
                "enable_destroy_repair": True,
                "enable_vehicle_reduction": False,
                "enable_elite_guided_repair": enabled,
                "tool_plan": ["validate_solution"]
                + (["elite_guided_repair", "validate_solution"] if enabled else [])
                + ["destroy_repair", "validate_solution", "improve_solution", "validate_solution", "compare_solutions"],
                "drl_samples": 128,
                "seed_trials": 1,
                "elite_guided_pool_size": 24,
                "elite_guided_polish_rounds": 12,
                "elite_guided_candidate_count": 128,
                "elite_guided_seed_trials": 1,
                "destroy_repair_rounds": 12,
                "local_search_rounds": 20,
                "allow_worse_acceptance": True,
                "acceptance_budget": 6,
                "acceptance_temperature": 0.01,
                "acceptance_decay": 0.9,
                "granular_neighbor_k": 24,
                "regret_k": 3,
                "shaw_remove_count": 8,
                "enable_multistart_refinement": enabled,
                "multistart_refinement_candidates": 2,
                "objective": {"primary": "distance"},
                "device": "cuda" if torch.cuda.is_available() else "cpu",
            },
        }
    )
    return {
        "instance": instance_name,
        "enabled": enabled,
        "final_distance": float(refined["final_solution"]["distance"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="A/B benchmark elite-guided repair on thinking-mode refinement.")
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
                "elite_guided_distance": experiment["final_distance"],
                "improvement": baseline["final_distance"] - experiment["final_distance"],
            }
        )

    if args.pretty:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(rows, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
