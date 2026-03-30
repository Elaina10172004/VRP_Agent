from __future__ import annotations

from .common import (
    EPS,
    build_vrp_distance_matrix,
    clean_routes,
    normalize_demands,
    normalize_routes_payload,
    route_cost,
    route_load,
    routes_to_raw_sequence,
    total_route_cost,
)


def best_intra_route_two_opt(routes: list[list[int]], distance_matrix: list[list[float]]) -> tuple[list[list[int]], float] | None:
    best_delta = 0.0
    best_routes: list[list[int]] | None = None
    route_costs = [route_cost(route, distance_matrix) for route in routes]

    for route_index, route in enumerate(routes):
        if len(route) < 4:
            continue
        current_cost = route_costs[route_index]
        for i in range(len(route) - 1):
            for j in range(i + 1, len(route)):
                candidate = route[:i] + list(reversed(route[i : j + 1])) + route[j + 1 :]
                delta = route_cost(candidate, distance_matrix) - current_cost
                if delta < best_delta - EPS:
                    updated = [list(item) for item in routes]
                    updated[route_index] = candidate
                    best_delta = delta
                    best_routes = updated

    if best_routes is None:
        return None
    return clean_routes(best_routes), best_delta


def best_relocate_move(
    routes: list[list[int]],
    distance_matrix: list[list[float]],
    demands: list[float],
    capacity: float,
) -> tuple[list[list[int]], float] | None:
    best_delta = 0.0
    best_routes: list[list[int]] | None = None
    route_costs = [route_cost(route, distance_matrix) for route in routes]
    route_loads = [route_load(route, demands) for route in routes]

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
                        candidate_target_cost = route_cost(candidate_target, distance_matrix)
                        delta = reduced_cost + candidate_target_cost - route_costs[from_route_index] - route_costs[to_route_index]

                    if delta < best_delta - EPS:
                        updated = [list(route) for route in routes]
                        moved_customer = updated[from_route_index].pop(from_pos)
                        if from_route_index == to_route_index:
                            adjusted_insert_pos = insert_pos
                            if insert_pos > from_pos:
                                adjusted_insert_pos -= 1
                            updated[to_route_index].insert(adjusted_insert_pos, moved_customer)
                        else:
                            updated[to_route_index].insert(insert_pos, moved_customer)
                        best_delta = delta
                        best_routes = clean_routes(updated)

    if best_routes is None:
        return None
    return best_routes, best_delta


def best_swap_move(
    routes: list[list[int]],
    distance_matrix: list[list[float]],
    demands: list[float],
    capacity: float,
) -> tuple[list[list[int]], float] | None:
    best_delta = 0.0
    best_routes: list[list[int]] | None = None
    route_costs = [route_cost(route, distance_matrix) for route in routes]
    route_loads = [route_load(route, demands) for route in routes]

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
                    if delta < best_delta - EPS:
                        updated = [list(route) for route in routes]
                        updated[left_route_index] = candidate_left
                        updated[right_route_index] = candidate_right
                        best_delta = delta
                        best_routes = clean_routes(updated)

    if best_routes is None:
        return None
    return best_routes, best_delta


def best_or_opt_move(
    routes: list[list[int]],
    distance_matrix: list[list[float]],
    demands: list[float],
    capacity: float,
    segment_lengths: tuple[int, ...] = (2, 3),
) -> tuple[list[list[int]], float] | None:
    best_delta = 0.0
    best_routes: list[list[int]] | None = None
    route_costs = [route_cost(route, distance_matrix) for route in routes]
    route_loads = [route_load(route, demands) for route in routes]

    for from_route_index, from_route in enumerate(routes):
        for segment_length in segment_lengths:
            if len(from_route) < segment_length:
                continue

            for start in range(len(from_route) - segment_length + 1):
                segment = from_route[start : start + segment_length]
                segment_load = sum(demands[customer] for customer in segment)
                reduced_route = from_route[:start] + from_route[start + segment_length :]
                reduced_cost = route_cost(reduced_route, distance_matrix)

                for to_route_index, to_route in enumerate(routes):
                    if from_route_index != to_route_index and route_loads[to_route_index] + segment_load > capacity + EPS:
                        continue

                    insertion_base = reduced_route if from_route_index == to_route_index else to_route
                    for insert_pos in range(len(insertion_base) + 1):
                        if from_route_index == to_route_index and insert_pos == start:
                            continue

                        candidate_target = insertion_base[:insert_pos] + segment + insertion_base[insert_pos:]
                        if from_route_index == to_route_index and candidate_target == from_route:
                            continue

                        if from_route_index == to_route_index:
                            delta = route_cost(candidate_target, distance_matrix) - route_costs[from_route_index]
                        else:
                            candidate_target_cost = route_cost(candidate_target, distance_matrix)
                            delta = reduced_cost + candidate_target_cost - route_costs[from_route_index] - route_costs[to_route_index]

                        if delta < best_delta - EPS:
                            updated = [list(route) for route in routes]
                            moved_segment = updated[from_route_index][start : start + segment_length]
                            del updated[from_route_index][start : start + segment_length]
                            if from_route_index == to_route_index:
                                adjusted_insert_pos = insert_pos
                                if insert_pos > start:
                                    adjusted_insert_pos -= segment_length
                                updated[to_route_index][adjusted_insert_pos:adjusted_insert_pos] = moved_segment
                            else:
                                updated[to_route_index][insert_pos:insert_pos] = moved_segment
                            best_delta = delta
                            best_routes = clean_routes(updated)

    if best_routes is None:
        return None
    return best_routes, best_delta


def best_two_opt_star_move(
    routes: list[list[int]],
    distance_matrix: list[list[float]],
    demands: list[float],
    capacity: float,
) -> tuple[list[list[int]], float] | None:
    best_delta = 0.0
    best_routes: list[list[int]] | None = None
    route_costs = [route_cost(route, distance_matrix) for route in routes]
    route_loads = [route_load(route, demands) for route in routes]

    for left_route_index in range(len(routes) - 1):
        left_route = routes[left_route_index]
        for right_route_index in range(left_route_index + 1, len(routes)):
            right_route = routes[right_route_index]
            for left_cut in range(len(left_route) + 1):
                left_prefix = left_route[:left_cut]
                left_suffix = left_route[left_cut:]
                left_prefix_load = route_load(left_prefix, demands)
                left_suffix_load = route_load(left_suffix, demands)

                for right_cut in range(len(right_route) + 1):
                    right_prefix = right_route[:right_cut]
                    right_suffix = right_route[right_cut:]
                    right_prefix_load = route_load(right_prefix, demands)
                    right_suffix_load = route_load(right_suffix, demands)

                    candidate_left = left_prefix + right_suffix
                    candidate_right = right_prefix + left_suffix
                    if candidate_left == left_route and candidate_right == right_route:
                        continue

                    candidate_left_load = left_prefix_load + right_suffix_load
                    candidate_right_load = right_prefix_load + left_suffix_load
                    if candidate_left_load > capacity + EPS or candidate_right_load > capacity + EPS:
                        continue

                    delta = (
                        route_cost(candidate_left, distance_matrix)
                        + route_cost(candidate_right, distance_matrix)
                        - route_costs[left_route_index]
                        - route_costs[right_route_index]
                    )
                    if delta < best_delta - EPS:
                        updated = [list(route) for route in routes]
                        updated[left_route_index] = candidate_left
                        updated[right_route_index] = candidate_right
                        best_delta = delta
                        best_routes = clean_routes(updated)

    if best_routes is None:
        return None
    return best_routes, best_delta


def best_cross_exchange_move(
    routes: list[list[int]],
    distance_matrix: list[list[float]],
    demands: list[float],
    capacity: float,
    segment_lengths: tuple[int, ...] = (1, 2),
) -> tuple[list[list[int]], float] | None:
    best_delta = 0.0
    best_routes: list[list[int]] | None = None
    route_costs = [route_cost(route, distance_matrix) for route in routes]
    route_loads = [route_load(route, demands) for route in routes]

    for left_route_index in range(len(routes) - 1):
        left_route = routes[left_route_index]
        for right_route_index in range(left_route_index + 1, len(routes)):
            right_route = routes[right_route_index]
            for left_length in segment_lengths:
                if len(left_route) < left_length:
                    continue
                for left_start in range(len(left_route) - left_length + 1):
                    left_segment = left_route[left_start : left_start + left_length]
                    left_segment_load = sum(demands[customer] for customer in left_segment)
                    left_remaining = left_route[:left_start] + left_route[left_start + left_length :]

                    for right_length in segment_lengths:
                        if len(right_route) < right_length:
                            continue
                        for right_start in range(len(right_route) - right_length + 1):
                            right_segment = right_route[right_start : right_start + right_length]
                            right_segment_load = sum(demands[customer] for customer in right_segment)
                            right_remaining = right_route[:right_start] + right_route[right_start + right_length :]

                            candidate_left_load = route_loads[left_route_index] - left_segment_load + right_segment_load
                            candidate_right_load = route_loads[right_route_index] - right_segment_load + left_segment_load
                            if candidate_left_load > capacity + EPS or candidate_right_load > capacity + EPS:
                                continue

                            candidate_left = left_remaining[:left_start] + right_segment + left_remaining[left_start:]
                            candidate_right = right_remaining[:right_start] + left_segment + right_remaining[right_start:]
                            if candidate_left == left_route and candidate_right == right_route:
                                continue

                            delta = (
                                route_cost(candidate_left, distance_matrix)
                                + route_cost(candidate_right, distance_matrix)
                                - route_costs[left_route_index]
                                - route_costs[right_route_index]
                            )
                            if delta < best_delta - EPS:
                                updated = [list(route) for route in routes]
                                updated[left_route_index] = candidate_left
                                updated[right_route_index] = candidate_right
                                best_delta = delta
                                best_routes = clean_routes(updated)

    if best_routes is None:
        return None
    return best_routes, best_delta


def _best_customer_insertion(
    routes: list[list[int]],
    customer: int,
    distance_matrix: list[list[float]],
    demands: list[float],
    capacity: float,
) -> tuple[list[list[int]], float] | None:
    best_delta = float("inf")
    best_routes: list[list[int]] | None = None
    route_costs = [route_cost(route, distance_matrix) for route in routes]
    route_loads = [route_load(route, demands) for route in routes]

    for route_index, route in enumerate(routes):
        if route_loads[route_index] + demands[customer] > capacity + EPS:
            continue
        for insert_pos in range(len(route) + 1):
            candidate_route = route[:insert_pos] + [customer] + route[insert_pos:]
            delta = route_cost(candidate_route, distance_matrix) - route_costs[route_index]
            if delta < best_delta:
                updated = [list(item) for item in routes]
                updated[route_index] = candidate_route
                best_delta = delta
                best_routes = clean_routes(updated)

    if best_routes is None:
        return None
    return best_routes, best_delta


def best_route_elimination(
    routes: list[list[int]],
    distance_matrix: list[list[float]],
    demands: list[float],
    capacity: float,
) -> tuple[list[list[int]], float] | None:
    if len(routes) < 2:
        return None

    original_cost = total_route_cost(routes, distance_matrix)
    route_order = sorted(range(len(routes)), key=lambda index: (len(routes[index]), route_cost(routes[index], distance_matrix)))
    best_routes: list[list[int]] | None = None
    best_delta = float("inf")

    for route_index in route_order:
        remaining_routes = [list(route) for idx, route in enumerate(routes) if idx != route_index]
        customers_to_reinsert = list(routes[route_index])
        working_routes = clean_routes(remaining_routes)
        feasible = True

        for customer in customers_to_reinsert:
            insertion = _best_customer_insertion(working_routes, customer, distance_matrix, demands, capacity)
            if insertion is None:
                feasible = False
                break
            working_routes, _ = insertion

        if not feasible:
            continue

        delta = total_route_cost(working_routes, distance_matrix) - original_cost
        if delta < best_delta:
            best_delta = delta
            best_routes = clean_routes(working_routes)

    if best_routes is None:
        return None
    return best_routes, best_delta


def reduce_cvrp_vehicle_count(
    instance: dict,
    solution: dict,
    config: dict | None = None,
) -> dict:
    config = dict(config or {})
    depot_xy = instance["depot_xy"]
    node_xy = instance["node_xy"]
    demands = normalize_demands(instance["node_demand"])
    capacity = float(instance["capacity"])
    routes = clean_routes(normalize_routes_payload(solution))
    distance_matrix = build_vrp_distance_matrix(depot_xy, node_xy)

    max_rounds = int(config.get("max_rounds", 10))
    current_routes = routes
    applied = 0

    for _ in range(max_rounds):
        move = best_route_elimination(current_routes, distance_matrix, demands, capacity)
        if move is None or len(move[0]) >= len(current_routes):
            break
        current_routes, _ = move
        applied += 1

    final_distance = total_route_cost(current_routes, distance_matrix)
    return {
        "problem_type": "cvrp",
        "routes": current_routes,
        "raw_sequence": routes_to_raw_sequence(current_routes),
        "distance": final_distance,
        "meta": {
            "vehicle_reduction_rounds": applied,
            "vehicle_count": len(current_routes),
        },
    }


def improve_cvrp_solution(
    instance: dict,
    solution: dict,
    config: dict | None = None,
) -> dict:
    config = dict(config or {})
    depot_xy = instance["depot_xy"]
    node_xy = instance["node_xy"]
    demands = normalize_demands(instance["node_demand"])
    capacity = float(instance["capacity"])
    routes = normalize_routes_payload(solution)

    distance_matrix = build_vrp_distance_matrix(depot_xy, node_xy)
    operators = config.get(
        "operators",
        ["or_opt", "two_opt_star", "cross_exchange", "relocate", "swap", "two_opt"],
    )
    max_rounds = int(config.get("max_rounds", 50))

    current_routes = clean_routes(routes)
    initial_distance = total_route_cost(current_routes, distance_matrix)
    applied_operators: list[str] = []

    for _ in range(max_rounds):
        improved = False
        for operator in operators:
            if operator == "two_opt":
                move = best_intra_route_two_opt(current_routes, distance_matrix)
            elif operator == "relocate":
                move = best_relocate_move(current_routes, distance_matrix, demands, capacity)
            elif operator == "swap":
                move = best_swap_move(current_routes, distance_matrix, demands, capacity)
            elif operator == "or_opt":
                move = best_or_opt_move(current_routes, distance_matrix, demands, capacity)
            elif operator == "two_opt_star":
                move = best_two_opt_star_move(current_routes, distance_matrix, demands, capacity)
            elif operator == "cross_exchange":
                move = best_cross_exchange_move(current_routes, distance_matrix, demands, capacity)
            elif operator == "route_elimination":
                move = best_route_elimination(current_routes, distance_matrix, demands, capacity)
            else:
                continue

            if move is None:
                continue

            next_routes, delta = move
            if delta > -EPS and operator != "route_elimination":
                continue
            if operator == "route_elimination" and len(next_routes) >= len(current_routes) and delta > -EPS:
                continue

            current_routes = next_routes
            applied_operators.append(operator)
            improved = True
            break

        if not improved:
            break

    final_distance = total_route_cost(current_routes, distance_matrix)
    return {
        "problem_type": "cvrp",
        "routes": current_routes,
        "raw_sequence": routes_to_raw_sequence(current_routes),
        "distance": final_distance,
        "meta": {
            "initial_distance": initial_distance,
            "improved_distance": final_distance,
            "improvement": initial_distance - final_distance,
            "iterations": len(applied_operators),
            "applied_operators": applied_operators,
            "vehicle_count": len(current_routes),
        },
    }
