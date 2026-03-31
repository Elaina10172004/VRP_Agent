from __future__ import annotations

from collections import Counter
from statistics import median

from local_search.common import (
    ObjectiveScore,
    build_distance_matrix,
    build_vrp_distance_matrix,
    clean_routes,
    ensure_points,
    evaluate_cvrptw_route,
    is_better_score,
    normalize_demands,
    normalize_objective_spec,
    normalize_routes_payload,
    normalize_service_times,
    normalize_time_windows,
    route_load,
    score_cvrp_routes,
    score_cvrptw_routes,
    score_tsp_tour,
)


def _edge_key(a: int, b: int, *, directed: bool) -> tuple[int, int]:
    if directed:
        return (a, b)
    return (a, b) if a <= b else (b, a)


def _tsp_edges(tour: list[int]) -> list[tuple[int, int]]:
    if not tour:
        return []
    edges = []
    for index, node in enumerate(tour):
        nxt = tour[(index + 1) % len(tour)]
        edges.append(_edge_key(node, nxt, directed=False))
    return edges


def _vrp_edges(routes: list[list[int]], *, directed: bool) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for route in routes:
        prev = 0
        for customer in route:
            node = customer + 1
            edges.append(_edge_key(prev, node, directed=directed))
            prev = node
        edges.append(_edge_key(prev, 0, directed=directed))
    return edges


def _build_edge_support(problem_type: str, elite_solutions: list[dict]) -> Counter[tuple[int, int]]:
    support: Counter[tuple[int, int]] = Counter()
    directed = problem_type == "cvrptw"
    for solution in elite_solutions:
        if problem_type == "tsp":
            tour = [int(node) for node in solution.get("tour", [])]
            support.update(_tsp_edges(tour))
        else:
            support.update(_vrp_edges(normalize_routes_payload(solution), directed=directed))
    return support


def _route_support(route: list[int], edge_support: Counter[tuple[int, int]], *, directed: bool) -> float:
    if not route:
        return 0.0
    prev = 0
    values: list[float] = []
    for customer in route:
        node = customer + 1
        values.append(float(edge_support[_edge_key(prev, node, directed=directed)]))
        prev = node
    values.append(float(edge_support[_edge_key(prev, 0, directed=directed)]))
    return sum(values) / max(1, len(values))


def _route_key(route: list[int], *, directed: bool) -> tuple[int, ...]:
    if directed:
        return tuple(route)
    reversed_route = tuple(reversed(route))
    route_key = tuple(route)
    return route_key if route_key <= reversed_route else reversed_route


def _collect_supported_routes(
    problem_type: str,
    elite_solutions: list[dict],
    edge_support: Counter[tuple[int, int]],
) -> list[dict]:
    directed = problem_type == "cvrptw"
    best_by_key: dict[tuple[int, ...], dict] = {}
    for solution in elite_solutions:
        if problem_type == "tsp":
            continue
        for route in normalize_routes_payload(solution):
            key = _route_key(route, directed=directed)
            support = _route_support(route, edge_support, directed=directed)
            record = best_by_key.get(key)
            if record is None or support > record["support"]:
                best_by_key[key] = {"route": list(route), "support": support}
    supported_routes = list(best_by_key.values())
    supported_routes.sort(key=lambda item: (-item["support"], -len(item["route"]), item["route"]))
    return supported_routes


def _select_route_assembly(route_pool: list[dict], target_vehicle_count: int | None = None) -> list[list[int]]:
    selected: list[list[int]] = []
    used_customers: set[int] = set()
    for item in route_pool:
        route = item["route"]
        if any(customer in used_customers for customer in route):
            continue
        if target_vehicle_count is not None and len(selected) >= target_vehicle_count:
            break
        selected.append(list(route))
        used_customers.update(route)
    return clean_routes(selected)


def _collect_cvrp_insertions(
    routes: list[list[int]],
    customer: int,
    distance_matrix: list[list[float]],
    demands: list[float],
    capacity: float,
    objective: dict | None,
    include_new_route: bool,
) -> list[tuple[list[list[int]], ObjectiveScore]]:
    objective_spec = normalize_objective_spec(objective)
    candidates: list[tuple[list[list[int]], ObjectiveScore]] = []
    route_loads = [route_load(route, demands) for route in routes]
    for route_index, route in enumerate(routes):
        if route_loads[route_index] + demands[customer] > capacity:
            continue
        for insert_pos in range(len(route) + 1):
            updated = [list(item) for item in routes]
            updated[route_index] = route[:insert_pos] + [customer] + route[insert_pos:]
            candidate_routes = clean_routes(updated)
            candidates.append((candidate_routes, score_cvrp_routes(candidate_routes, distance_matrix, objective_spec)))
    if include_new_route and demands[customer] <= capacity:
        candidate_routes = clean_routes([*routes, [customer]])
        candidates.append((candidate_routes, score_cvrp_routes(candidate_routes, distance_matrix, objective_spec)))
    candidates.sort(key=lambda item: item[1].ranking_key())
    return candidates


def _collect_cvrptw_insertions(
    routes: list[list[int]],
    customer: int,
    distance_matrix: list[list[float]],
    demands: list[float],
    capacity: float,
    time_windows: list[tuple[float, float]],
    service_times: list[float],
    objective: dict | None,
    include_new_route: bool,
) -> list[tuple[list[list[int]], ObjectiveScore]]:
    objective_spec = normalize_objective_spec(objective)
    candidates: list[tuple[list[list[int]], ObjectiveScore]] = []
    for route_index, route in enumerate(routes):
        for insert_pos in range(len(route) + 1):
            updated = [list(item) for item in routes]
            updated[route_index] = route[:insert_pos] + [customer] + route[insert_pos:]
            candidate_routes = clean_routes(updated)
            score = score_cvrptw_routes(candidate_routes, distance_matrix, demands, capacity, time_windows, service_times, objective_spec)
            if score.feasible:
                candidates.append((candidate_routes, score))
    if include_new_route:
        candidate_routes = clean_routes([*routes, [customer]])
        score = score_cvrptw_routes(candidate_routes, distance_matrix, demands, capacity, time_windows, service_times, objective_spec)
        if score.feasible:
            candidates.append((candidate_routes, score))
    candidates.sort(key=lambda item: item[1].ranking_key())
    return candidates


def _regret_insert_cvrp(
    base_routes: list[list[int]],
    customers: list[int],
    distance_matrix: list[list[float]],
    demands: list[float],
    capacity: float,
    objective: dict | None,
    regret_k: int,
) -> tuple[list[list[int]], ObjectiveScore] | None:
    objective_spec = normalize_objective_spec(objective)
    routes = clean_routes(base_routes)
    pending = list(customers)
    while pending:
        chosen_customer: int | None = None
        chosen_routes: list[list[int]] | None = None
        chosen_score: ObjectiveScore | None = None
        chosen_regret = float("-inf")
        for customer in pending:
            candidates = _collect_cvrp_insertions(routes, customer, distance_matrix, demands, capacity, objective_spec, include_new_route=True)
            if not candidates:
                return None
            best_routes, best_score = candidates[0]
            compare_index = min(max(1, regret_k) - 1, len(candidates) - 1)
            regret = candidates[compare_index][1].generalized_cost - best_score.generalized_cost
            if regret > chosen_regret or (
                abs(regret - chosen_regret) <= 1e-9 and (chosen_score is None or is_better_score(best_score, chosen_score))
            ):
                chosen_customer = customer
                chosen_routes = best_routes
                chosen_score = best_score
                chosen_regret = regret
        if chosen_customer is None or chosen_routes is None or chosen_score is None:
            return None
        routes = chosen_routes
        pending.remove(chosen_customer)
    return routes, score_cvrp_routes(routes, distance_matrix, objective_spec)


def _regret_insert_cvrptw(
    base_routes: list[list[int]],
    customers: list[int],
    distance_matrix: list[list[float]],
    demands: list[float],
    capacity: float,
    time_windows: list[tuple[float, float]],
    service_times: list[float],
    objective: dict | None,
    regret_k: int,
) -> tuple[list[list[int]], ObjectiveScore] | None:
    objective_spec = normalize_objective_spec(objective)
    routes = clean_routes(base_routes)
    pending = list(customers)
    while pending:
        chosen_customer: int | None = None
        chosen_routes: list[list[int]] | None = None
        chosen_score: ObjectiveScore | None = None
        chosen_regret = float("-inf")
        for customer in pending:
            candidates = _collect_cvrptw_insertions(
                routes,
                customer,
                distance_matrix,
                demands,
                capacity,
                time_windows,
                service_times,
                objective_spec,
                include_new_route=True,
            )
            if not candidates:
                return None
            best_routes, best_score = candidates[0]
            compare_index = min(max(1, regret_k) - 1, len(candidates) - 1)
            regret = candidates[compare_index][1].generalized_cost - best_score.generalized_cost
            if regret > chosen_regret or (
                abs(regret - chosen_regret) <= 1e-9 and (chosen_score is None or is_better_score(best_score, chosen_score))
            ):
                chosen_customer = customer
                chosen_routes = best_routes
                chosen_score = best_score
                chosen_regret = regret
        if chosen_customer is None or chosen_routes is None or chosen_score is None:
            return None
        routes = chosen_routes
        pending.remove(chosen_customer)
    return routes, score_cvrptw_routes(routes, distance_matrix, demands, capacity, time_windows, service_times, objective_spec)


def _recombine_tsp(elite_solutions: list[dict], objective: dict | None) -> dict | None:
    tours = [[int(node) for node in solution.get("tour", [])] for solution in elite_solutions if solution.get("tour")]
    if not tours:
        return None
    points = ensure_points(elite_solutions[0]["_instance_points"])
    distance_matrix = build_distance_matrix(points)
    edge_support = _build_edge_support("tsp", elite_solutions)
    node_count = len(tours[0])
    remaining = set(range(node_count))
    current = tours[0][0]
    route = [current]
    remaining.remove(current)
    while remaining:
        next_node = min(
            remaining,
            key=lambda node: (
                -edge_support[_edge_key(current, node, directed=False)],
                distance_matrix[current][node],
                node,
            ),
        )
        route.append(next_node)
        remaining.remove(next_node)
        current = next_node
    score = score_tsp_tour(route, distance_matrix, objective)
    return {
        "solution": {
            "problem_type": "tsp",
            "tour": route,
            "closed_tour": route + [route[0]] if route else [],
            "distance": score.distance,
            "meta": {"source": "elite_consensus"},
        },
        "score": score,
        "stats": {
            "elite_count": len(elite_solutions),
            "edge_support_size": len(edge_support),
            "method": "edge_frequency_greedy",
        },
    }


def _recombine_cvrp(instance: dict, elite_solutions: list[dict], objective: dict | None, config: dict) -> dict | None:
    demands = normalize_demands(instance["node_demand"])
    capacity = float(instance["capacity"])
    distance_matrix = build_vrp_distance_matrix(instance["depot_xy"], instance["node_xy"])
    edge_support = _build_edge_support("cvrp", elite_solutions)
    route_pool = _collect_supported_routes("cvrp", elite_solutions, edge_support)
    if not route_pool:
        return None

    vehicle_counts = [len(normalize_routes_payload(solution)) for solution in elite_solutions]
    targets = []
    for value in [min(vehicle_counts), int(median(vehicle_counts)), len(normalize_routes_payload(elite_solutions[0]))]:
        if value not in targets:
            targets.append(value)

    best_solution = None
    best_score = None
    all_customers = set(range(len(demands)))
    regret_k = max(2, int(config.get("regret_k", 3)))
    for target in targets:
        selected_routes = _select_route_assembly(route_pool, target_vehicle_count=target)
        used = {customer for route in selected_routes for customer in route}
        missing = sorted(all_customers - used)
        inserted = _regret_insert_cvrp(selected_routes, missing, distance_matrix, demands, capacity, objective, regret_k)
        if inserted is None:
            continue
        routes, score = inserted
        if best_score is None or is_better_score(score, best_score):
            best_score = score
            best_solution = routes

    if best_solution is None or best_score is None:
        return None

    return {
        "solution": {
            "problem_type": "cvrp",
            "routes": best_solution,
            "raw_sequence": _routes_to_raw_sequence(best_solution),
            "distance": best_score.distance,
            "meta": {"source": "elite_consensus"},
        },
        "score": best_score,
        "stats": {
            "elite_count": len(elite_solutions),
            "route_pool_size": len(route_pool),
            "edge_support_size": len(edge_support),
            "method": "route_assembly_regret_repair",
        },
    }


def _routes_to_raw_sequence(routes: list[list[int]]) -> list[int]:
    raw = [0]
    for route in clean_routes(routes):
        raw.extend(customer + 1 for customer in route)
        raw.append(0)
    return raw


def _recombine_cvrptw(instance: dict, elite_solutions: list[dict], objective: dict | None, config: dict) -> dict | None:
    demands = normalize_demands(instance["node_demand"])
    capacity = float(instance["capacity"])
    distance_matrix = build_vrp_distance_matrix(instance["depot_xy"], instance["node_xy"])
    time_windows = normalize_time_windows(instance["node_tw"])
    service_times = normalize_service_times(instance["service_time"], len(demands))
    edge_support = _build_edge_support("cvrptw", elite_solutions)
    route_pool = _collect_supported_routes("cvrptw", elite_solutions, edge_support)
    if not route_pool:
        return None

    vehicle_counts = [len(normalize_routes_payload(solution)) for solution in elite_solutions]
    targets = []
    for value in [min(vehicle_counts), int(median(vehicle_counts)), len(normalize_routes_payload(elite_solutions[0]))]:
        if value not in targets:
            targets.append(value)

    best_solution = None
    best_score = None
    all_customers = set(range(len(demands)))
    regret_k = max(2, int(config.get("regret_k", 3)))
    for target in targets:
        selected_routes = _select_route_assembly(route_pool, target_vehicle_count=target)
        feasible_selected = True
        for route in selected_routes:
            if not evaluate_cvrptw_route(route, distance_matrix, demands, capacity, time_windows, service_times).feasible:
                feasible_selected = False
                break
        if not feasible_selected:
            continue
        used = {customer for route in selected_routes for customer in route}
        missing = sorted(all_customers - used)
        inserted = _regret_insert_cvrptw(
            selected_routes,
            missing,
            distance_matrix,
            demands,
            capacity,
            time_windows,
            service_times,
            objective,
            regret_k,
        )
        if inserted is None:
            continue
        routes, score = inserted
        if best_score is None or is_better_score(score, best_score):
            best_score = score
            best_solution = routes

    if best_solution is None or best_score is None:
        return None

    return {
        "solution": {
            "problem_type": "cvrptw",
            "routes": best_solution,
            "raw_sequence": _routes_to_raw_sequence(best_solution),
            "distance": best_score.distance,
            "meta": {"source": "elite_consensus"},
        },
        "score": best_score,
        "stats": {
            "elite_count": len(elite_solutions),
            "route_pool_size": len(route_pool),
            "edge_support_size": len(edge_support),
            "method": "route_assembly_regret_repair",
        },
    }


def recombine_elite_candidates(
    problem_type: str,
    instance: dict,
    scored_candidates: list[dict],
    config: dict | None = None,
) -> dict | None:
    config = dict(config or {})
    if len(scored_candidates) < 2:
        return None
    if not bool(config.get("enable_elite_recombine", True)):
        return None

    elite_pool_size = max(2, int(config.get("elite_pool_size", min(24, len(scored_candidates)))))
    elite_items = scored_candidates[:elite_pool_size]
    elite_solutions = [item["solution"] for item in elite_items]
    objective = config.get("objective")

    if problem_type == "tsp":
        tagged = [{**solution, "_instance_points": instance["points"]} for solution in elite_solutions]
        return _recombine_tsp(tagged, objective)
    if problem_type == "cvrp":
        return _recombine_cvrp(instance, elite_solutions, objective, config)
    if problem_type == "cvrptw":
        return _recombine_cvrptw(instance, elite_solutions, objective, config)
    return None
