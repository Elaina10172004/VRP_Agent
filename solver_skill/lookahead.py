from __future__ import annotations

from dataclasses import dataclass

from local_search.common import (
    EPS,
    build_distance_matrix,
    build_vrp_distance_matrix,
    clean_routes,
    evaluate_cvrptw_route,
    normalize_demands,
    normalize_routes_payload,
    normalize_service_times,
    normalize_time_windows,
    route_cost,
    route_load,
    routes_to_raw_sequence,
    total_route_cost,
)
from local_search.tsp import tour_length


@dataclass(frozen=True)
class CandidateState:
    key: tuple
    solution: dict
    cost: float
    history: tuple[str, ...]


def _dedupe_best(states: list[CandidateState], beam_width: int) -> list[CandidateState]:
    best_by_key: dict[tuple, CandidateState] = {}
    for state in states:
        previous = best_by_key.get(state.key)
        if previous is None or state.cost < previous.cost - EPS:
            best_by_key[state.key] = state
    ordered = sorted(best_by_key.values(), key=lambda item: item.cost)
    return ordered[:beam_width]


def _tsp_neighbors(tour: list[int], distance_matrix: list[list[float]], operators: list[str], per_operator_limit: int) -> list[tuple[list[int], float, str]]:
    current_cost = tour_length(tour, distance_matrix)
    neighbors: list[tuple[list[int], float, str]] = []

    if "two_opt" in operators:
      two_opt_candidates: list[tuple[list[int], float, str]] = []
      for i in range(len(tour) - 1):
          for j in range(i + 1, len(tour)):
              if i == 0 and j == len(tour) - 1:
                  continue
              candidate = tour[:i] + list(reversed(tour[i : j + 1])) + tour[j + 1 :]
              candidate_cost = tour_length(candidate, distance_matrix)
              if candidate_cost < current_cost - EPS:
                  two_opt_candidates.append((candidate, candidate_cost, "two_opt"))
      two_opt_candidates.sort(key=lambda item: item[1])
      neighbors.extend(two_opt_candidates[:per_operator_limit])

    if "relocate" in operators:
      relocate_candidates: list[tuple[list[int], float, str]] = []
      for from_index in range(len(tour)):
          node = tour[from_index]
          reduced = tour[:from_index] + tour[from_index + 1 :]
          for insert_index in range(len(tour)):
              candidate = reduced[:insert_index] + [node] + reduced[insert_index:]
              if candidate == tour:
                  continue
              candidate_cost = tour_length(candidate, distance_matrix)
              if candidate_cost < current_cost - EPS:
                  relocate_candidates.append((candidate, candidate_cost, "relocate"))
      relocate_candidates.sort(key=lambda item: item[1])
      neighbors.extend(relocate_candidates[:per_operator_limit])

    if "swap" in operators:
      swap_candidates: list[tuple[list[int], float, str]] = []
      for i in range(len(tour) - 1):
          for j in range(i + 1, len(tour)):
              candidate = list(tour)
              candidate[i], candidate[j] = candidate[j], candidate[i]
              candidate_cost = tour_length(candidate, distance_matrix)
              if candidate_cost < current_cost - EPS:
                  swap_candidates.append((candidate, candidate_cost, "swap"))
      swap_candidates.sort(key=lambda item: item[1])
      neighbors.extend(swap_candidates[:per_operator_limit])

    neighbors.sort(key=lambda item: item[1])
    return neighbors[: max(1, per_operator_limit)]


def apply_tsp_lookahead(instance: dict, solution: dict, config: dict | None = None) -> dict:
    config = dict(config or {})
    operators = config.get("operators", ["two_opt", "relocate", "swap"])
    depth = int(config.get("depth", 2))
    beam_width = int(config.get("beam_width", 4))
    per_operator_limit = int(config.get("per_operator_limit", beam_width))
    points = instance["points"]
    tour = [int(node) for node in solution["tour"]]
    distance_matrix = build_distance_matrix([(float(x), float(y)) for x, y in points])
    initial_cost = tour_length(tour, distance_matrix)

    frontier = [CandidateState(tuple(tour), {"tour": tour}, initial_cost, tuple())]
    best = frontier[0]
    for _ in range(depth):
        expanded: list[CandidateState] = []
        for state in frontier:
            neighbors = _tsp_neighbors(state.solution["tour"], distance_matrix, operators, per_operator_limit)
            for candidate_tour, candidate_cost, operator in neighbors:
                expanded.append(
                    CandidateState(
                        tuple(candidate_tour),
                        {"tour": candidate_tour},
                        candidate_cost,
                        state.history + (operator,),
                    )
                )
        if not expanded:
            break
        frontier = _dedupe_best(expanded, beam_width)
        if frontier[0].cost < best.cost - EPS:
            best = frontier[0]

    tour = best.solution["tour"]
    return {
        "problem_type": "tsp",
        "tour": tour,
        "closed_tour": tour + [tour[0]] if tour else [],
        "distance": best.cost,
        "meta": {
            "initial_distance": initial_cost,
            "improved_distance": best.cost,
            "improvement": initial_cost - best.cost,
            "beam_width": beam_width,
            "depth": depth,
            "applied_operators": list(best.history),
        },
    }


def _cvrp_neighbors(
    routes: list[list[int]],
    distance_matrix: list[list[float]],
    demands: list[float],
    capacity: float,
    operators: list[str],
    per_operator_limit: int,
) -> list[tuple[list[list[int]], float, str]]:
    neighbors: list[tuple[list[list[int]], float, str]] = []
    current_cost = total_route_cost(routes, distance_matrix)
    route_costs = [route_cost(route, distance_matrix) for route in routes]
    route_loads = [route_load(route, demands) for route in routes]

    if "two_opt" in operators:
        candidates: list[tuple[list[list[int]], float, str]] = []
        for route_index, route in enumerate(routes):
            if len(route) < 4:
                continue
            for i in range(len(route) - 1):
                for j in range(i + 1, len(route)):
                    candidate_route = route[:i] + list(reversed(route[i : j + 1])) + route[j + 1 :]
                    candidate_cost = route_cost(candidate_route, distance_matrix)
                    delta = candidate_cost - route_costs[route_index]
                    if delta < -EPS:
                        updated = [list(item) for item in routes]
                        updated[route_index] = candidate_route
                        candidates.append((clean_routes(updated), current_cost + delta, "two_opt"))
        candidates.sort(key=lambda item: item[1])
        neighbors.extend(candidates[:per_operator_limit])

    if "relocate" in operators:
        candidates = []
        for from_route_index, from_route in enumerate(routes):
            for from_pos, customer in enumerate(from_route):
                reduced_route = from_route[:from_pos] + from_route[from_pos + 1 :]
                reduced_cost = route_cost(reduced_route, distance_matrix)
                for to_route_index, to_route in enumerate(routes):
                    if from_route_index != to_route_index and route_loads[to_route_index] + demands[customer] > capacity + EPS:
                        continue
                    insertion_base = reduced_route if from_route_index == to_route_index else to_route
                    for insert_pos in range(len(insertion_base) + 1):
                        candidate_target = insertion_base[:insert_pos] + [customer] + insertion_base[insert_pos:]
                        if from_route_index == to_route_index and candidate_target == from_route:
                            continue
                        if from_route_index == to_route_index:
                            delta = route_cost(candidate_target, distance_matrix) - route_costs[from_route_index]
                        else:
                            delta = (
                                reduced_cost
                                + route_cost(candidate_target, distance_matrix)
                                - route_costs[from_route_index]
                                - route_costs[to_route_index]
                            )
                        if delta < -EPS:
                            updated = [list(route) for route in routes]
                            moved = updated[from_route_index].pop(from_pos)
                            if from_route_index == to_route_index:
                                adjusted_insert = insert_pos - 1 if insert_pos > from_pos else insert_pos
                                updated[to_route_index].insert(adjusted_insert, moved)
                            else:
                                updated[to_route_index].insert(insert_pos, moved)
                            candidates.append((clean_routes(updated), current_cost + delta, "relocate"))
        candidates.sort(key=lambda item: item[1])
        neighbors.extend(candidates[:per_operator_limit])

    if "swap" in operators:
        candidates = []
        for left_route_index in range(len(routes) - 1):
            left_route = routes[left_route_index]
            for right_route_index in range(left_route_index + 1, len(routes)):
                right_route = routes[right_route_index]
                for left_pos, left_customer in enumerate(left_route):
                    for right_pos, right_customer in enumerate(right_route):
                        new_left_load = route_loads[left_route_index] - demands[left_customer] + demands[right_customer]
                        new_right_load = route_loads[right_route_index] - demands[right_customer] + demands[left_customer]
                        if new_left_load > capacity + EPS or new_right_load > capacity + EPS:
                            continue
                        candidate_left = list(left_route)
                        candidate_right = list(right_route)
                        candidate_left[left_pos] = right_customer
                        candidate_right[right_pos] = left_customer
                        delta = (
                            route_cost(candidate_left, distance_matrix)
                            + route_cost(candidate_right, distance_matrix)
                            - route_costs[left_route_index]
                            - route_costs[right_route_index]
                        )
                        if delta < -EPS:
                            updated = [list(route) for route in routes]
                            updated[left_route_index] = candidate_left
                            updated[right_route_index] = candidate_right
                            candidates.append((clean_routes(updated), current_cost + delta, "swap"))
        candidates.sort(key=lambda item: item[1])
        neighbors.extend(candidates[:per_operator_limit])

    neighbors.sort(key=lambda item: item[1])
    return neighbors[: max(1, per_operator_limit)]


def apply_cvrp_lookahead(instance: dict, solution: dict, config: dict | None = None) -> dict:
    config = dict(config or {})
    operators = config.get("operators", ["two_opt", "relocate", "swap"])
    depth = int(config.get("depth", 2))
    beam_width = int(config.get("beam_width", 4))
    per_operator_limit = int(config.get("per_operator_limit", beam_width))
    routes = normalize_routes_payload(solution)
    demands = normalize_demands(instance["node_demand"])
    capacity = float(instance["capacity"])
    distance_matrix = build_vrp_distance_matrix(instance["depot_xy"], instance["node_xy"])
    initial_cost = total_route_cost(routes, distance_matrix)

    frontier = [CandidateState(tuple(tuple(route) for route in routes), {"routes": routes}, initial_cost, tuple())]
    best = frontier[0]
    for _ in range(depth):
        expanded: list[CandidateState] = []
        for state in frontier:
            neighbors = _cvrp_neighbors(state.solution["routes"], distance_matrix, demands, capacity, operators, per_operator_limit)
            for candidate_routes, candidate_cost, operator in neighbors:
                expanded.append(
                    CandidateState(
                        tuple(tuple(route) for route in candidate_routes),
                        {"routes": candidate_routes},
                        candidate_cost,
                        state.history + (operator,),
                    )
                )
        if not expanded:
            break
        frontier = _dedupe_best(expanded, beam_width)
        if frontier[0].cost < best.cost - EPS:
            best = frontier[0]

    routes = best.solution["routes"]
    return {
        "problem_type": "cvrp",
        "routes": routes,
        "raw_sequence": routes_to_raw_sequence(routes),
        "distance": best.cost,
        "meta": {
            "initial_distance": initial_cost,
            "improved_distance": best.cost,
            "improvement": initial_cost - best.cost,
            "beam_width": beam_width,
            "depth": depth,
            "applied_operators": list(best.history),
        },
    }


def _cvrptw_neighbors(
    routes: list[list[int]],
    distance_matrix: list[list[float]],
    demands: list[float],
    capacity: float,
    time_windows: list[tuple[float, float]],
    service_times: list[float],
    operators: list[str],
    per_operator_limit: int,
) -> list[tuple[list[list[int]], float, str]]:
    neighbors: list[tuple[list[list[int]], float, str]] = []
    current_cost = sum(
        evaluate_cvrptw_route(route, distance_matrix, demands, capacity, time_windows, service_times).cost
        for route in routes
    )
    route_evals = [
        evaluate_cvrptw_route(route, distance_matrix, demands, capacity, time_windows, service_times)
        for route in routes
    ]

    if "two_opt" in operators:
        candidates: list[tuple[list[list[int]], float, str]] = []
        for route_index, route in enumerate(routes):
            if len(route) < 4:
                continue
            for i in range(len(route) - 1):
                for j in range(i + 1, len(route)):
                    candidate_route = route[:i] + list(reversed(route[i : j + 1])) + route[j + 1 :]
                    candidate_eval = evaluate_cvrptw_route(candidate_route, distance_matrix, demands, capacity, time_windows, service_times)
                    if not candidate_eval.feasible:
                        continue
                    delta = candidate_eval.cost - route_evals[route_index].cost
                    if delta < -EPS:
                        updated = [list(item) for item in routes]
                        updated[route_index] = candidate_route
                        candidates.append((clean_routes(updated), current_cost + delta, "two_opt"))
        candidates.sort(key=lambda item: item[1])
        neighbors.extend(candidates[:per_operator_limit])

    if "relocate" in operators:
        candidates = []
        for from_route_index, from_route in enumerate(routes):
            for from_pos, customer in enumerate(from_route):
                reduced_route = from_route[:from_pos] + from_route[from_pos + 1 :]
                reduced_eval = evaluate_cvrptw_route(reduced_route, distance_matrix, demands, capacity, time_windows, service_times)
                if not reduced_eval.feasible:
                    continue
                for to_route_index, to_route in enumerate(routes):
                    insertion_base = reduced_route if from_route_index == to_route_index else to_route
                    for insert_pos in range(len(insertion_base) + 1):
                        candidate_target = insertion_base[:insert_pos] + [customer] + insertion_base[insert_pos:]
                        if from_route_index == to_route_index and candidate_target == from_route:
                            continue
                        candidate_eval = evaluate_cvrptw_route(candidate_target, distance_matrix, demands, capacity, time_windows, service_times)
                        if not candidate_eval.feasible:
                            continue
                        if from_route_index == to_route_index:
                            delta = candidate_eval.cost - route_evals[from_route_index].cost
                        else:
                            delta = (
                                reduced_eval.cost
                                + candidate_eval.cost
                                - route_evals[from_route_index].cost
                                - route_evals[to_route_index].cost
                            )
                        if delta < -EPS:
                            updated = [list(route) for route in routes]
                            moved = updated[from_route_index].pop(from_pos)
                            if from_route_index == to_route_index:
                                adjusted_insert = insert_pos - 1 if insert_pos > from_pos else insert_pos
                                updated[to_route_index].insert(adjusted_insert, moved)
                            else:
                                updated[to_route_index].insert(insert_pos, moved)
                            candidates.append((clean_routes(updated), current_cost + delta, "relocate"))
        candidates.sort(key=lambda item: item[1])
        neighbors.extend(candidates[:per_operator_limit])

    if "swap" in operators:
        candidates = []
        for left_route_index in range(len(routes) - 1):
            left_route = routes[left_route_index]
            for right_route_index in range(left_route_index + 1, len(routes)):
                right_route = routes[right_route_index]
                for left_pos, left_customer in enumerate(left_route):
                    for right_pos, right_customer in enumerate(right_route):
                        candidate_left = list(left_route)
                        candidate_right = list(right_route)
                        candidate_left[left_pos] = right_customer
                        candidate_right[right_pos] = left_customer
                        left_eval = evaluate_cvrptw_route(candidate_left, distance_matrix, demands, capacity, time_windows, service_times)
                        if not left_eval.feasible:
                            continue
                        right_eval = evaluate_cvrptw_route(candidate_right, distance_matrix, demands, capacity, time_windows, service_times)
                        if not right_eval.feasible:
                            continue
                        delta = (
                            left_eval.cost
                            + right_eval.cost
                            - route_evals[left_route_index].cost
                            - route_evals[right_route_index].cost
                        )
                        if delta < -EPS:
                            updated = [list(route) for route in routes]
                            updated[left_route_index] = candidate_left
                            updated[right_route_index] = candidate_right
                            candidates.append((clean_routes(updated), current_cost + delta, "swap"))
        candidates.sort(key=lambda item: item[1])
        neighbors.extend(candidates[:per_operator_limit])

    neighbors.sort(key=lambda item: item[1])
    return neighbors[: max(1, per_operator_limit)]


def apply_cvrptw_lookahead(instance: dict, solution: dict, config: dict | None = None) -> dict:
    config = dict(config or {})
    operators = config.get("operators", ["two_opt", "relocate", "swap"])
    depth = int(config.get("depth", 2))
    beam_width = int(config.get("beam_width", 4))
    per_operator_limit = int(config.get("per_operator_limit", beam_width))
    routes = normalize_routes_payload(solution)
    demands = normalize_demands(instance["node_demand"])
    capacity = float(instance["capacity"])
    time_windows = normalize_time_windows(instance["node_tw"])
    service_times = normalize_service_times(instance["service_time"], len(demands))
    distance_matrix = build_vrp_distance_matrix(instance["depot_xy"], instance["node_xy"])
    initial_cost = sum(
        evaluate_cvrptw_route(route, distance_matrix, demands, capacity, time_windows, service_times).cost
        for route in routes
    )

    frontier = [CandidateState(tuple(tuple(route) for route in routes), {"routes": routes}, initial_cost, tuple())]
    best = frontier[0]
    for _ in range(depth):
        expanded: list[CandidateState] = []
        for state in frontier:
            neighbors = _cvrptw_neighbors(
                state.solution["routes"],
                distance_matrix,
                demands,
                capacity,
                time_windows,
                service_times,
                operators,
                per_operator_limit,
            )
            for candidate_routes, candidate_cost, operator in neighbors:
                expanded.append(
                    CandidateState(
                        tuple(tuple(route) for route in candidate_routes),
                        {"routes": candidate_routes},
                        candidate_cost,
                        state.history + (operator,),
                    )
                )
        if not expanded:
            break
        frontier = _dedupe_best(expanded, beam_width)
        if frontier[0].cost < best.cost - EPS:
            best = frontier[0]

    routes = best.solution["routes"]
    return {
        "problem_type": "cvrptw",
        "routes": routes,
        "raw_sequence": routes_to_raw_sequence(routes),
        "distance": best.cost,
        "meta": {
            "initial_distance": initial_cost,
            "improved_distance": best.cost,
            "improvement": initial_cost - best.cost,
            "beam_width": beam_width,
            "depth": depth,
            "applied_operators": list(best.history),
        },
    }
