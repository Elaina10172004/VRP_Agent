from __future__ import annotations

import argparse
import json
import sys
from math import hypot
from pathlib import Path

from local_search.common import (
    build_distance_matrix,
    build_vrp_distance_matrix,
    evaluate_cvrptw_route,
    normalize_demands,
    normalize_objective_spec,
    normalize_routes_payload,
    normalize_service_times,
    normalize_time_windows,
    route_distance,
    route_generalized_cost,
    route_load,
    score_cvrp_routes,
    score_cvrptw_routes,
    score_tsp_tour,
)


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


def _normalize_tsp_tour(solution: dict) -> list[int]:
    if isinstance(solution.get("tour"), list):
        return [int(node) for node in solution["tour"]]
    if isinstance(solution.get("closed_tour"), list):
        closed = [int(node) for node in solution["closed_tour"]]
        if len(closed) >= 2 and closed[0] == closed[-1]:
            return closed[:-1]
        return closed
    raise ValueError("TSP solution must contain 'tour' or 'closed_tour'.")


def _build_cvrptw_wait_profile(
    route: list[int],
    distance_matrix: list[list[float]],
    time_windows: list[tuple[float, float]],
    service_times: list[float],
) -> dict:
    time = 0.0
    previous = 0
    stops = []
    total_waiting_time = 0.0
    max_waiting_time = 0.0
    max_waiting_customer = None

    for customer in route:
        node_index = customer + 1
        travel = distance_matrix[previous][node_index]
        arrival = time + travel
        ready, due = time_windows[customer]
        start = max(arrival, ready)
        waiting_time = max(0.0, ready - arrival)
        total_waiting_time += waiting_time
        if waiting_time > max_waiting_time:
            max_waiting_time = waiting_time
            max_waiting_customer = customer
        stops.append(
            {
                "customer_index": customer,
                "arrival_time": arrival,
                "ready_time": ready,
                "due_time": due,
                "service_start": start,
                "waiting_time": waiting_time,
            }
        )
        time = start + service_times[customer]
        previous = node_index

    return {
        "stops": stops,
        "total_waiting_time": total_waiting_time,
        "max_waiting_time": max_waiting_time,
        "max_waiting_customer": max_waiting_customer,
    }


def _node_label(node_index: int) -> str:
    return "depot" if node_index == 0 else f"customer_{node_index - 1}"


def _build_customer_knn(node_xy: list[list[float]], k: int = 3) -> list[list[dict]]:
    coords = [(float(x), float(y)) for x, y in node_xy]
    knn: list[list[dict]] = []
    for index, (x1, y1) in enumerate(coords):
        neighbors = []
        for other_index, (x2, y2) in enumerate(coords):
            if index == other_index:
                continue
            dist = hypot(x1 - x2, y1 - y2)
            neighbors.append(
                {
                    "customer_index": other_index,
                    "distance": dist,
                }
            )
        neighbors.sort(key=lambda item: item["distance"])
        knn.append(neighbors[:k])
    return knn


def _build_longest_edges(routes: list[list[int]], distance_matrix: list[list[float]], top_n: int = 8) -> list[dict]:
    edges = []
    for route_index, route in enumerate(routes, start=1):
        previous = 0
        for customer in route:
            current = customer + 1
            distance = float(distance_matrix[previous][current])
            edges.append(
                {
                    "route_index": route_index,
                    "from": _node_label(previous),
                    "to": _node_label(current),
                    "distance": distance,
                }
            )
            previous = current
        distance = float(distance_matrix[previous][0])
        edges.append(
            {
                "route_index": route_index,
                "from": _node_label(previous),
                "to": "depot",
                "distance": distance,
            }
        )
    edges.sort(key=lambda item: item["distance"], reverse=True)
    return edges[:top_n]


def _build_connection_hotspots(
    routes: list[list[int]],
    node_xy: list[list[float]],
    distance_matrix: list[list[float]],
    top_n: int = 8,
) -> list[dict]:
    knn = _build_customer_knn(node_xy, k=3)
    hotspots = []
    for route_index, route in enumerate(routes, start=1):
        for position, customer in enumerate(route):
            current = customer + 1
            previous = 0 if position == 0 else route[position - 1] + 1
            nxt = 0 if position == len(route) - 1 else route[position + 1] + 1
            prev_distance = float(distance_matrix[previous][current])
            next_distance = float(distance_matrix[current][nxt])

            if previous == 0 and nxt != 0:
                adjusted_connection_cost = next_distance * 2.0
            elif previous != 0 and nxt == 0:
                adjusted_connection_cost = prev_distance * 2.0
            else:
                adjusted_connection_cost = prev_distance + next_distance

            nearest_neighbors = knn[customer]
            knn_reference = 0.0
            if nearest_neighbors:
                knn_reference = 2.0 * float(nearest_neighbors[0]["distance"])

            hotspots.append(
                {
                    "route_index": route_index,
                    "customer_index": customer,
                    "previous_node": _node_label(previous),
                    "next_node": _node_label(nxt),
                    "prev_distance": prev_distance,
                    "next_distance": next_distance,
                    "adjusted_connection_cost": adjusted_connection_cost,
                    "nearest_neighbors": nearest_neighbors,
                    "knn_reference_cost": knn_reference,
                    "excess_over_knn": adjusted_connection_cost - knn_reference,
                }
            )
    hotspots.sort(key=lambda item: item["excess_over_knn"], reverse=True)
    return hotspots[:top_n]


def _analyze_tsp(instance: dict, solution: dict, objective: dict) -> dict:
    points = instance["points"]
    distance_matrix = build_distance_matrix([(float(x), float(y)) for x, y in points])
    tour = _normalize_tsp_tour(solution)
    score = score_tsp_tour(tour, distance_matrix, objective)
    return {
        "problem_type": "tsp",
        "objective": normalize_objective_spec(objective).__dict__,
        "summary": {
            "vehicle_count": score.vehicle_count,
            "distance": score.distance,
            "generalized_cost": score.generalized_cost,
            "duration": score.duration,
        },
        "routes": [
            {
                "route_index": 1,
                "customer_count": len(tour),
                "distance": score.distance,
                "duration": score.duration,
                "generalized_cost": score.generalized_cost,
                "sequence": tour,
            }
        ],
        "hotspots": {},
    }


def _analyze_cvrp(instance: dict, solution: dict, objective: dict) -> dict:
    depot_xy = instance["depot_xy"]
    node_xy = instance["node_xy"]
    demands = normalize_demands(instance["node_demand"])
    distance_matrix = build_vrp_distance_matrix(depot_xy, node_xy)
    routes = normalize_routes_payload(solution)
    objective_spec = normalize_objective_spec(objective)
    total_score = score_cvrp_routes(routes, distance_matrix, objective_spec)

    route_items = []
    longest_route = None
    heaviest_route = None
    for route_index, route in enumerate(routes, start=1):
        distance = route_distance(route, distance_matrix)
        load = route_load(route, demands)
        generalized_cost = route_generalized_cost(route, distance_matrix, objective_spec)
        item = {
            "route_index": route_index,
            "customer_count": len(route),
            "distance": distance,
            "duration": distance,
            "load": load,
            "generalized_cost": generalized_cost,
            "sequence": route,
        }
        route_items.append(item)
        if longest_route is None or distance > longest_route["distance"]:
            longest_route = item
        if heaviest_route is None or load > heaviest_route["load"]:
            heaviest_route = item

    return {
        "problem_type": "cvrp",
        "objective": objective_spec.__dict__,
        "summary": {
            "vehicle_count": total_score.vehicle_count,
            "distance": total_score.distance,
            "generalized_cost": total_score.generalized_cost,
            "duration": total_score.duration,
        },
        "routes": route_items,
        "hotspots": {
            "longest_route": longest_route,
            "heaviest_route": heaviest_route,
            "longest_edges": _build_longest_edges(routes, distance_matrix),
            "worst_connection_points": _build_connection_hotspots(routes, node_xy, distance_matrix),
        },
    }


def _analyze_cvrptw(instance: dict, solution: dict, objective: dict) -> dict:
    depot_xy = instance["depot_xy"]
    node_xy = instance["node_xy"]
    demands = normalize_demands(instance["node_demand"])
    capacity = float(instance["capacity"])
    time_windows = normalize_time_windows(instance["node_tw"])
    service_times = normalize_service_times(instance["service_time"], len(demands))
    distance_matrix = build_vrp_distance_matrix(depot_xy, node_xy)
    routes = normalize_routes_payload(solution)
    objective_spec = normalize_objective_spec(objective)
    total_score = score_cvrptw_routes(routes, distance_matrix, demands, capacity, time_windows, service_times, objective_spec)

    route_items = []
    longest_route = None
    max_waiting_stop = None
    for route_index, route in enumerate(routes, start=1):
        evaluation = evaluate_cvrptw_route(route, distance_matrix, demands, capacity, time_windows, service_times)
        wait_profile = _build_cvrptw_wait_profile(route, distance_matrix, time_windows, service_times)
        generalized_cost = route_generalized_cost(route, distance_matrix, objective_spec, evaluation)
        item = {
            "route_index": route_index,
            "customer_count": len(route),
            "distance": evaluation.distance,
            "duration": evaluation.duration,
            "load": evaluation.load,
            "generalized_cost": generalized_cost,
            "total_waiting_time": wait_profile["total_waiting_time"],
            "max_waiting_time": wait_profile["max_waiting_time"],
            "max_waiting_customer": wait_profile["max_waiting_customer"],
            "sequence": route,
        }
        route_items.append(item)
        if longest_route is None or evaluation.distance > longest_route["distance"]:
            longest_route = item
        if wait_profile["max_waiting_customer"] is not None:
            stop_info = {
                "route_index": route_index,
                "customer_index": wait_profile["max_waiting_customer"],
                "waiting_time": wait_profile["max_waiting_time"],
            }
            if max_waiting_stop is None or stop_info["waiting_time"] > max_waiting_stop["waiting_time"]:
                max_waiting_stop = stop_info

    return {
        "problem_type": "cvrptw",
        "objective": objective_spec.__dict__,
        "summary": {
            "vehicle_count": total_score.vehicle_count,
            "distance": total_score.distance,
            "generalized_cost": total_score.generalized_cost,
            "duration": total_score.duration,
            "max_waiting_time": 0.0 if max_waiting_stop is None else max_waiting_stop["waiting_time"],
        },
        "routes": route_items,
        "hotspots": {
            "longest_route": longest_route,
            "max_waiting_stop": max_waiting_stop,
            "longest_edges": _build_longest_edges(routes, distance_matrix),
            "worst_connection_points": _build_connection_hotspots(routes, node_xy, distance_matrix),
        },
    }


def analyze_solution_parts(problem_type: str, instance: dict, solution: dict, objective: dict | None = None) -> dict:
    normalized_problem_type = str(problem_type).strip().lower()
    if normalized_problem_type == "tsp":
        return _analyze_tsp(instance, solution, objective or {})
    if normalized_problem_type == "cvrp":
        return _analyze_cvrp(instance, solution, objective or {})
    if normalized_problem_type == "cvrptw":
        return _analyze_cvrptw(instance, solution, objective or {})
    raise ValueError(f"Unsupported problem_type: {problem_type}")


def analyze_solution_payload(payload: dict) -> dict:
    problem_type = str(payload.get("problem_type", "")).strip().lower()
    instance = payload.get("instance")
    solution = payload.get("solution")
    objective = payload.get("objective")

    if not isinstance(instance, dict):
        raise ValueError("Payload must contain an 'instance' object.")
    if not isinstance(solution, dict):
        raise ValueError("Payload must contain a 'solution' object.")
    return analyze_solution_parts(problem_type, instance, solution, objective)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze a TSP/CVRP/CVRPTW solution payload.")
    parser.add_argument("--input", help="Path to input JSON payload. Defaults to stdin.")
    parser.add_argument("--output", help="Path to write output JSON. Defaults to stdout.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    args = parser.parse_args()

    payload = _read_payload(args.input)
    result = analyze_solution_payload(payload)
    _write_output(result, args.output, args.pretty)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
