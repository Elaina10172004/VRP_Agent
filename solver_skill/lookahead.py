from __future__ import annotations

from dataclasses import dataclass

from local_search.common import (
    EPS,
    ObjectiveScore,
    clean_routes,
    build_distance_matrix,
    build_vrp_distance_matrix,
    is_better_score,
    normalize_demands,
    normalize_objective_spec,
    normalize_routes_payload,
    normalize_service_times,
    normalize_time_windows,
    route_load,
    routes_to_raw_sequence,
    score_cvrp_routes,
    score_cvrptw_routes,
    score_tsp_tour,
)


@dataclass(frozen=True)
class CandidateState:
    key: tuple
    solution: dict
    score: ObjectiveScore
    history: tuple[str, ...]


def _dedupe_best(states: list[CandidateState], beam_width: int) -> list[CandidateState]:
    best_by_key: dict[tuple, CandidateState] = {}
    for state in states:
        previous = best_by_key.get(state.key)
        if previous is None or is_better_score(state.score, previous.score):
            best_by_key[state.key] = state
    ordered = sorted(best_by_key.values(), key=lambda item: item.score.ranking_key())
    return ordered[:beam_width]


def _tsp_neighbors(
    tour: list[int],
    distance_matrix: list[list[float]],
    operators: list[str],
    per_operator_limit: int,
    objective: dict,
) -> list[tuple[list[int], ObjectiveScore, str]]:
    current_score = score_tsp_tour(tour, distance_matrix, objective)
    neighbors: list[tuple[list[int], ObjectiveScore, str]] = []

    if "two_opt" in operators:
        candidates: list[tuple[list[int], ObjectiveScore, str]] = []
        for i in range(len(tour) - 1):
            for j in range(i + 1, len(tour)):
                if i == 0 and j == len(tour) - 1:
                    continue
                candidate = tour[:i] + list(reversed(tour[i : j + 1])) + tour[j + 1 :]
                candidate_score = score_tsp_tour(candidate, distance_matrix, objective)
                if is_better_score(candidate_score, current_score):
                    candidates.append((candidate, candidate_score, "two_opt"))
        candidates.sort(key=lambda item: item[1].ranking_key())
        neighbors.extend(candidates[:per_operator_limit])

    if "relocate" in operators:
        candidates = []
        for from_index in range(len(tour)):
            node = tour[from_index]
            reduced = tour[:from_index] + tour[from_index + 1 :]
            for insert_index in range(len(tour)):
                candidate = reduced[:insert_index] + [node] + reduced[insert_index:]
                if candidate == tour:
                    continue
                candidate_score = score_tsp_tour(candidate, distance_matrix, objective)
                if is_better_score(candidate_score, current_score):
                    candidates.append((candidate, candidate_score, "relocate"))
        candidates.sort(key=lambda item: item[1].ranking_key())
        neighbors.extend(candidates[:per_operator_limit])

    if "swap" in operators:
        candidates = []
        for i in range(len(tour) - 1):
            for j in range(i + 1, len(tour)):
                candidate = list(tour)
                candidate[i], candidate[j] = candidate[j], candidate[i]
                candidate_score = score_tsp_tour(candidate, distance_matrix, objective)
                if is_better_score(candidate_score, current_score):
                    candidates.append((candidate, candidate_score, "swap"))
        candidates.sort(key=lambda item: item[1].ranking_key())
        neighbors.extend(candidates[:per_operator_limit])

    neighbors.sort(key=lambda item: item[1].ranking_key())
    return neighbors[: max(1, per_operator_limit)]


def apply_tsp_lookahead(instance: dict, solution: dict, config: dict | None = None) -> dict:
    config = dict(config or {})
    operators = config.get("operators", ["two_opt", "relocate", "swap"])
    depth = int(config.get("depth", 2))
    beam_width = int(config.get("beam_width", 4))
    per_operator_limit = int(config.get("per_operator_limit", beam_width))
    objective = normalize_objective_spec(config.get("objective"))
    tour = [int(node) for node in solution["tour"]]
    distance_matrix = build_distance_matrix([(float(x), float(y)) for x, y in instance["points"]])
    initial_score = score_tsp_tour(tour, distance_matrix, objective)

    frontier = [CandidateState(tuple(tour), {"tour": tour}, initial_score, tuple())]
    best = frontier[0]
    for _ in range(depth):
        expanded: list[CandidateState] = []
        for state in frontier:
            neighbors = _tsp_neighbors(state.solution["tour"], distance_matrix, operators, per_operator_limit, objective)
            for candidate_tour, candidate_score, operator in neighbors:
                expanded.append(
                    CandidateState(
                        tuple(candidate_tour),
                        {"tour": candidate_tour},
                        candidate_score,
                        state.history + (operator,),
                    )
                )
        if not expanded:
            break
        frontier = _dedupe_best(expanded, beam_width)
        if is_better_score(frontier[0].score, best.score):
            best = frontier[0]

    best_tour = best.solution["tour"]
    return {
        "problem_type": "tsp",
        "tour": best_tour,
        "closed_tour": best_tour + [best_tour[0]] if best_tour else [],
        "distance": best.score.distance,
        "meta": {
            "initial_distance": initial_score.distance,
            "improved_distance": best.score.distance,
            "initial_score": initial_score.generalized_cost,
            "improved_score": best.score.generalized_cost,
            "improvement": initial_score.generalized_cost - best.score.generalized_cost,
            "beam_width": beam_width,
            "depth": depth,
            "applied_operators": list(best.history),
            "objective": objective.__dict__,
        },
    }


def _cvrp_neighbors(
    routes: list[list[int]],
    distance_matrix: list[list[float]],
    demands: list[float],
    capacity: float,
    operators: list[str],
    per_operator_limit: int,
    objective: dict,
) -> list[tuple[list[list[int]], ObjectiveScore, str]]:
    neighbors: list[tuple[list[list[int]], ObjectiveScore, str]] = []
    current_score = score_cvrp_routes(routes, distance_matrix, objective)
    route_loads = [route_load(route, demands) for route in routes]

    if "two_opt" in operators:
        candidates: list[tuple[list[list[int]], ObjectiveScore, str]] = []
        for route_index, route in enumerate(routes):
            if len(route) < 4:
                continue
            for i in range(len(route) - 1):
                for j in range(i + 1, len(route)):
                    candidate_route = route[:i] + list(reversed(route[i : j + 1])) + route[j + 1 :]
                    updated = [list(item) for item in routes]
                    updated[route_index] = candidate_route
                    candidate_routes = clean_routes(updated)
                    candidate_score = score_cvrp_routes(candidate_routes, distance_matrix, objective)
                    if is_better_score(candidate_score, current_score):
                        candidates.append((candidate_routes, candidate_score, "two_opt"))
        candidates.sort(key=lambda item: item[1].ranking_key())
        neighbors.extend(candidates[:per_operator_limit])

    if "relocate" in operators:
        candidates = []
        for from_route_index, from_route in enumerate(routes):
            for from_pos, customer in enumerate(from_route):
                for to_route_index, to_route in enumerate(routes):
                    if from_route_index != to_route_index and route_loads[to_route_index] + demands[customer] > capacity + EPS:
                        continue
                    for insert_pos in range(len(to_route) + 1):
                        updated = [list(route) for route in routes]
                        moved = updated[from_route_index].pop(from_pos)
                        if from_route_index == to_route_index:
                            adjusted_insert = insert_pos - 1 if insert_pos > from_pos else insert_pos
                            updated[to_route_index].insert(adjusted_insert, moved)
                        else:
                            updated[to_route_index].insert(insert_pos, moved)
                        candidate_routes = clean_routes(updated)
                        if candidate_routes == clean_routes(routes):
                            continue
                        candidate_score = score_cvrp_routes(candidate_routes, distance_matrix, objective)
                        if is_better_score(candidate_score, current_score):
                            candidates.append((candidate_routes, candidate_score, "relocate"))
        candidates.sort(key=lambda item: item[1].ranking_key())
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
                        updated = [list(route) for route in routes]
                        updated[left_route_index][left_pos] = right_customer
                        updated[right_route_index][right_pos] = left_customer
                        candidate_routes = clean_routes(updated)
                        candidate_score = score_cvrp_routes(candidate_routes, distance_matrix, objective)
                        if is_better_score(candidate_score, current_score):
                            candidates.append((candidate_routes, candidate_score, "swap"))
        candidates.sort(key=lambda item: item[1].ranking_key())
        neighbors.extend(candidates[:per_operator_limit])

    neighbors.sort(key=lambda item: item[1].ranking_key())
    return neighbors[: max(1, per_operator_limit)]


def apply_cvrp_lookahead(instance: dict, solution: dict, config: dict | None = None) -> dict:
    config = dict(config or {})
    operators = config.get("operators", ["two_opt", "relocate", "swap"])
    depth = int(config.get("depth", 2))
    beam_width = int(config.get("beam_width", 4))
    per_operator_limit = int(config.get("per_operator_limit", beam_width))
    objective = normalize_objective_spec(config.get("objective"))
    routes = normalize_routes_payload(solution)
    demands = normalize_demands(instance["node_demand"])
    capacity = float(instance["capacity"])
    distance_matrix = build_vrp_distance_matrix(instance["depot_xy"], instance["node_xy"])
    initial_score = score_cvrp_routes(routes, distance_matrix, objective)

    frontier = [
        CandidateState(tuple(tuple(route) for route in routes), {"routes": routes}, initial_score, tuple())
    ]
    best = frontier[0]
    for _ in range(depth):
        expanded: list[CandidateState] = []
        for state in frontier:
            neighbors = _cvrp_neighbors(
                state.solution["routes"],
                distance_matrix,
                demands,
                capacity,
                operators,
                per_operator_limit,
                objective,
            )
            for candidate_routes, candidate_score, operator in neighbors:
                expanded.append(
                    CandidateState(
                        tuple(tuple(route) for route in candidate_routes),
                        {"routes": candidate_routes},
                        candidate_score,
                        state.history + (operator,),
                    )
                )
        if not expanded:
            break
        frontier = _dedupe_best(expanded, beam_width)
        if is_better_score(frontier[0].score, best.score):
            best = frontier[0]

    best_routes = best.solution["routes"]
    return {
        "problem_type": "cvrp",
        "routes": best_routes,
        "raw_sequence": routes_to_raw_sequence(best_routes),
        "distance": best.score.distance,
        "meta": {
            "initial_distance": initial_score.distance,
            "improved_distance": best.score.distance,
            "initial_score": initial_score.generalized_cost,
            "improved_score": best.score.generalized_cost,
            "improvement": initial_score.generalized_cost - best.score.generalized_cost,
            "beam_width": beam_width,
            "depth": depth,
            "applied_operators": list(best.history),
            "objective": objective.__dict__,
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
    objective: dict,
) -> list[tuple[list[list[int]], ObjectiveScore, str]]:
    neighbors: list[tuple[list[list[int]], ObjectiveScore, str]] = []
    current_score = score_cvrptw_routes(routes, distance_matrix, demands, capacity, time_windows, service_times, objective)

    if "two_opt" in operators:
        candidates: list[tuple[list[list[int]], ObjectiveScore, str]] = []
        for route_index, route in enumerate(routes):
            if len(route) < 4:
                continue
            for i in range(len(route) - 1):
                for j in range(i + 1, len(route)):
                    candidate_route = route[:i] + list(reversed(route[i : j + 1])) + route[j + 1 :]
                    updated = [list(item) for item in routes]
                    updated[route_index] = candidate_route
                    candidate_routes = clean_routes(updated)
                    candidate_score = score_cvrptw_routes(candidate_routes, distance_matrix, demands, capacity, time_windows, service_times, objective)
                    if is_better_score(candidate_score, current_score):
                        candidates.append((candidate_routes, candidate_score, "two_opt"))
        candidates.sort(key=lambda item: item[1].ranking_key())
        neighbors.extend(candidates[:per_operator_limit])

    if "relocate" in operators:
        candidates = []
        for from_route_index, from_route in enumerate(routes):
            for from_pos, _customer in enumerate(from_route):
                for to_route_index, to_route in enumerate(routes):
                    for insert_pos in range(len(to_route) + 1):
                        updated = [list(route) for route in routes]
                        moved = updated[from_route_index].pop(from_pos)
                        if from_route_index == to_route_index:
                            adjusted_insert = insert_pos - 1 if insert_pos > from_pos else insert_pos
                            updated[to_route_index].insert(adjusted_insert, moved)
                        else:
                            updated[to_route_index].insert(insert_pos, moved)
                        candidate_routes = clean_routes(updated)
                        if candidate_routes == clean_routes(routes):
                            continue
                        candidate_score = score_cvrptw_routes(candidate_routes, distance_matrix, demands, capacity, time_windows, service_times, objective)
                        if is_better_score(candidate_score, current_score):
                            candidates.append((candidate_routes, candidate_score, "relocate"))
        candidates.sort(key=lambda item: item[1].ranking_key())
        neighbors.extend(candidates[:per_operator_limit])

    if "swap" in operators:
        candidates = []
        for left_route_index in range(len(routes) - 1):
            left_route = routes[left_route_index]
            for right_route_index in range(left_route_index + 1, len(routes)):
                right_route = routes[right_route_index]
                for left_pos, left_customer in enumerate(left_route):
                    for right_pos, right_customer in enumerate(right_route):
                        updated = [list(route) for route in routes]
                        updated[left_route_index][left_pos] = right_customer
                        updated[right_route_index][right_pos] = left_customer
                        candidate_routes = clean_routes(updated)
                        candidate_score = score_cvrptw_routes(candidate_routes, distance_matrix, demands, capacity, time_windows, service_times, objective)
                        if is_better_score(candidate_score, current_score):
                            candidates.append((candidate_routes, candidate_score, "swap"))
        candidates.sort(key=lambda item: item[1].ranking_key())
        neighbors.extend(candidates[:per_operator_limit])

    neighbors.sort(key=lambda item: item[1].ranking_key())
    return neighbors[: max(1, per_operator_limit)]


def apply_cvrptw_lookahead(instance: dict, solution: dict, config: dict | None = None) -> dict:
    config = dict(config or {})
    operators = config.get("operators", ["two_opt", "relocate", "swap"])
    depth = int(config.get("depth", 2))
    beam_width = int(config.get("beam_width", 4))
    per_operator_limit = int(config.get("per_operator_limit", beam_width))
    objective = normalize_objective_spec(config.get("objective"))
    routes = normalize_routes_payload(solution)
    demands = normalize_demands(instance["node_demand"])
    capacity = float(instance["capacity"])
    time_windows = normalize_time_windows(instance["node_tw"])
    service_times = normalize_service_times(instance["service_time"], len(demands))
    distance_matrix = build_vrp_distance_matrix(instance["depot_xy"], instance["node_xy"])
    initial_score = score_cvrptw_routes(routes, distance_matrix, demands, capacity, time_windows, service_times, objective)

    frontier = [
        CandidateState(tuple(tuple(route) for route in routes), {"routes": routes}, initial_score, tuple())
    ]
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
                objective,
            )
            for candidate_routes, candidate_score, operator in neighbors:
                expanded.append(
                    CandidateState(
                        tuple(tuple(route) for route in candidate_routes),
                        {"routes": candidate_routes},
                        candidate_score,
                        state.history + (operator,),
                    )
                )
        if not expanded:
            break
        frontier = _dedupe_best(expanded, beam_width)
        if is_better_score(frontier[0].score, best.score):
            best = frontier[0]

    best_routes = best.solution["routes"]
    return {
        "problem_type": "cvrptw",
        "routes": best_routes,
        "raw_sequence": routes_to_raw_sequence(best_routes),
        "distance": best.score.distance,
        "meta": {
            "initial_distance": initial_score.distance,
            "improved_distance": best.score.distance,
            "initial_score": initial_score.generalized_cost,
            "improved_score": best.score.generalized_cost,
            "improvement": initial_score.generalized_cost - best.score.generalized_cost,
            "beam_width": beam_width,
            "depth": depth,
            "applied_operators": list(best.history),
            "objective": objective.__dict__,
        },
    }
