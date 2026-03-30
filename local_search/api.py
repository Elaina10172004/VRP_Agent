from __future__ import annotations

from .cvrp import improve_cvrp_solution
from .cvrptw import improve_cvrptw_solution
from .tsp import improve_tsp_solution


def improve_payload(payload: dict) -> dict:
    problem_type = str(payload.get("problem_type", "")).strip().lower()
    instance = payload.get("instance")
    solution = payload.get("solution")
    config = payload.get("config", {})

    if not isinstance(instance, dict):
        raise ValueError("Payload must contain an 'instance' object.")
    if not isinstance(solution, dict):
        raise ValueError("Payload must contain a 'solution' object.")

    if problem_type == "tsp":
        improved = improve_tsp_solution(instance, solution, config)
    elif problem_type == "cvrp":
        improved = improve_cvrp_solution(instance, solution, config)
    elif problem_type == "cvrptw":
        improved = improve_cvrptw_solution(instance, solution, config)
    else:
        raise ValueError(f"Unsupported problem_type: {problem_type}")

    return {
        "problem_type": problem_type,
        "solution": improved,
        "meta": improved.get("meta", {}),
    }
