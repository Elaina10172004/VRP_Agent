from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from local_search.common import (
    build_distance_matrix,
    build_vrp_distance_matrix,
    evaluate_cvrptw_route,
    normalize_demands,
    normalize_routes_payload,
    normalize_service_times,
    normalize_time_windows,
    route_cost,
    route_load,
)
from local_search.tsp import tour_length


def _read_payload(input_path: str | None) -> dict:
    if input_path:
        return json.loads(Path(input_path).read_text(encoding="utf-8"))
    return json.load(sys.stdin)


def _write_output(payload: dict, output_path: str | None, pretty: bool) -> None:
    kwargs = {"ensure_ascii": False}
    if pretty:
        kwargs["indent"] = 2
    content = json.dumps(payload, **kwargs)
    if output_path:
        Path(output_path).write_text(content + ("\n" if pretty else ""), encoding="utf-8")
        return
    sys.stdout.write(content)
    if pretty:
        sys.stdout.write("\n")


def _normalize_tsp_tour(solution: dict, node_count: int) -> list[int]:
    if isinstance(solution.get("tour"), list):
        return [int(node) for node in solution["tour"]]
    if isinstance(solution.get("closed_tour"), list):
        closed = [int(node) for node in solution["closed_tour"]]
        if len(closed) >= 2 and closed[0] == closed[-1]:
            return closed[:-1]
        return closed
    raise ValueError("TSP solution must contain 'tour' or 'closed_tour'.")


def _build_summary(capacity_ok: bool, time_windows_ok: bool, return_to_depot_ok: bool, visited_once_ok: bool) -> dict:
    return {
        "capacity_ok": capacity_ok,
        "time_windows_ok": time_windows_ok,
        "return_to_depot_ok": return_to_depot_ok,
        "visited_once_ok": visited_once_ok,
    }


def _validate_tsp(instance: dict, solution: dict) -> dict:
    points = instance["points"]
    node_count = len(points)
    violations: list[str] = []

    try:
        tour = _normalize_tsp_tour(solution, node_count)
    except ValueError as error:
        return {
            "feasible": False,
            "vehicle_count": 1,
            "distance": 0.0,
            "violations": [str(error)],
            "summary": _build_summary(True, True, False, False),
        }

    visited_once_ok = len(tour) == node_count and sorted(tour) == list(range(node_count))
    if not visited_once_ok:
        violations.append("TSP tour must visit each node exactly once.")

    distance = 0.0
    if visited_once_ok:
        distance = float(tour_length(tour, build_distance_matrix(points)))

    return_to_depot_ok = bool(tour)
    if not return_to_depot_ok:
        violations.append("TSP tour is empty.")

    feasible = visited_once_ok and return_to_depot_ok
    return {
        "feasible": feasible,
        "vehicle_count": 1 if return_to_depot_ok else 0,
        "distance": distance,
        "violations": violations,
        "summary": _build_summary(True, True, return_to_depot_ok, visited_once_ok),
    }


def _check_return_to_depot(solution: dict) -> bool:
    raw_sequence = solution.get("raw_sequence")
    if isinstance(raw_sequence, list) and raw_sequence:
        return int(raw_sequence[0]) == 0 and int(raw_sequence[-1]) == 0
    return solution.get("routes") is not None


def _validate_vrp_common(instance: dict, solution: dict, with_time_windows: bool) -> dict:
    depot_xy = instance["depot_xy"]
    node_xy = instance["node_xy"]
    demands = normalize_demands(instance["node_demand"])
    capacity = float(instance["capacity"])
    routes = normalize_routes_payload(solution)

    node_count = len(node_xy)
    distance_matrix = build_vrp_distance_matrix(depot_xy, node_xy)
    flattened = [customer for route in routes for customer in route]
    visited_once_ok = sorted(flattened) == list(range(node_count))
    vehicle_count = len(routes)
    return_to_depot_ok = _check_return_to_depot(solution)
    violations: list[str] = []

    if not visited_once_ok:
        missing = sorted(set(range(node_count)) - set(flattened))
        duplicated = sorted({customer for customer in flattened if flattened.count(customer) > 1})
        if missing:
            violations.append(f"Missing customers: {missing}")
        if duplicated:
            violations.append(f"Duplicated customers: {duplicated}")

    if not return_to_depot_ok:
        violations.append("Route representation does not return to depot.")

    capacity_ok = True
    for route_index, route in enumerate(routes):
        load = route_load(route, demands)
        if load > capacity:
            capacity_ok = False
            violations.append(f"Route {route_index + 1} exceeds capacity: {load:.3f} > {capacity:.3f}")

    time_windows_ok = True
    if with_time_windows:
        time_windows = normalize_time_windows(instance["node_tw"])
        service_times = normalize_service_times(instance["service_time"], node_count)
        for route_index, route in enumerate(routes):
            evaluation = evaluate_cvrptw_route(route, distance_matrix, demands, float("inf"), time_windows, service_times)
            if not evaluation.feasible:
                time_windows_ok = False
                violations.append(f"Route {route_index + 1} violates time windows.")

    distance = float(sum(route_cost(route, distance_matrix) for route in routes))
    feasible = visited_once_ok and return_to_depot_ok and capacity_ok and time_windows_ok
    return {
        "feasible": feasible,
        "vehicle_count": vehicle_count,
        "distance": distance,
        "violations": violations,
        "summary": _build_summary(capacity_ok, time_windows_ok, return_to_depot_ok, visited_once_ok),
    }


def validate_payload_solution(payload: dict) -> dict:
    problem_type = str(payload.get("problem_type", "")).strip().lower()
    instance = payload.get("instance")
    solution = payload.get("solution")

    if not isinstance(instance, dict):
        raise ValueError("Payload must contain an 'instance' object.")
    if not isinstance(solution, dict):
        raise ValueError("Payload must contain a 'solution' object.")

    if problem_type == "tsp":
        return validate_payload_solution_parts(problem_type, instance, solution)
    if problem_type in {"cvrp", "cvrptw"}:
        return validate_payload_solution_parts(problem_type, instance, solution)
    raise ValueError(f"Unsupported problem_type: {problem_type}")


def validate_payload_solution_parts(problem_type: str, instance: dict, solution: dict) -> dict:
    normalized_problem_type = str(problem_type).strip().lower()
    if normalized_problem_type == "tsp":
        return _validate_tsp(instance, solution)
    if normalized_problem_type == "cvrp":
        return _validate_vrp_common(instance, solution, with_time_windows=False)
    if normalized_problem_type == "cvrptw":
        return _validate_vrp_common(instance, solution, with_time_windows=True)
    raise ValueError(f"Unsupported problem_type: {problem_type}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a TSP/CVRP/CVRPTW solution payload.")
    parser.add_argument("--input", help="Path to input JSON payload. Defaults to stdin.")
    parser.add_argument("--output", help="Path to write output JSON. Defaults to stdout.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    args = parser.parse_args()

    payload = _read_payload(args.input)
    result = validate_payload_solution(payload)
    _write_output(result, args.output, args.pretty)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
