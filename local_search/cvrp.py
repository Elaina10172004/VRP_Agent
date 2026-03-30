from __future__ import annotations

from .common import (
    EPS,
    ObjectiveScore,
    build_vrp_distance_matrix,
    clean_routes,
    is_better_score,
    normalize_demands,
    normalize_objective_spec,
    normalize_routes_payload,
    route_load,
    routes_to_raw_sequence,
    score_cvrp_routes,
)


def best_intra_route_two_opt(
    routes: list[list[int]],
    distance_matrix: list[list[float]],
    objective: dict | None = None,
) -> tuple[list[list[int]], ObjectiveScore] | None:
    objective_spec = normalize_objective_spec(objective)
    current_score = score_cvrp_routes(routes, distance_matrix, objective_spec)
    best_score: ObjectiveScore | None = None
    best_routes: list[list[int]] | None = None

    for route_index, route in enumerate(routes):
        if len(route) < 4:
            continue
        for i in range(len(route) - 1):
            for j in range(i + 1, len(route)):
                candidate_route = route[:i] + list(reversed(route[i : j + 1])) + route[j + 1 :]
                updated = [list(item) for item in routes]
                updated[route_index] = candidate_route
                candidate_routes = clean_routes(updated)
                candidate_score = score_cvrp_routes(candidate_routes, distance_matrix, objective_spec)
                if is_better_score(candidate_score, current_score) and is_better_score(candidate_score, best_score):
                    best_score = candidate_score
                    best_routes = candidate_routes

    if best_routes is None or best_score is None:
        return None
    return best_routes, best_score


def best_relocate_move(
    routes: list[list[int]],
    distance_matrix: list[list[float]],
    demands: list[float],
    capacity: float,
    objective: dict | None = None,
) -> tuple[list[list[int]], ObjectiveScore] | None:
    objective_spec = normalize_objective_spec(objective)
    current_score = score_cvrp_routes(routes, distance_matrix, objective_spec)
    best_score: ObjectiveScore | None = None
    best_routes: list[list[int]] | None = None
    route_loads = [route_load(route, demands) for route in routes]

    for from_route_index, from_route in enumerate(routes):
        for from_pos, customer in enumerate(from_route):
            for to_route_index, to_route in enumerate(routes):
                if from_route_index != to_route_index and route_loads[to_route_index] + demands[customer] > capacity + EPS:
                    continue

                for insert_pos in range(len(to_route) + 1):
                    updated = [list(route) for route in routes]
                    moved_customer = updated[from_route_index].pop(from_pos)
                    if from_route_index == to_route_index:
                        adjusted_insert_pos = insert_pos - 1 if insert_pos > from_pos else insert_pos
                        updated[to_route_index].insert(adjusted_insert_pos, moved_customer)
                    else:
                        updated[to_route_index].insert(insert_pos, moved_customer)
                    candidate_routes = clean_routes(updated)
                    if candidate_routes == clean_routes(routes):
                        continue
                    candidate_score = score_cvrp_routes(candidate_routes, distance_matrix, objective_spec)
                    if is_better_score(candidate_score, current_score) and is_better_score(candidate_score, best_score):
                        best_score = candidate_score
                        best_routes = candidate_routes

    if best_routes is None or best_score is None:
        return None
    return best_routes, best_score


def best_swap_move(
    routes: list[list[int]],
    distance_matrix: list[list[float]],
    demands: list[float],
    capacity: float,
    objective: dict | None = None,
) -> tuple[list[list[int]], ObjectiveScore] | None:
    objective_spec = normalize_objective_spec(objective)
    current_score = score_cvrp_routes(routes, distance_matrix, objective_spec)
    best_score: ObjectiveScore | None = None
    best_routes: list[list[int]] | None = None
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

                    updated = [list(route) for route in routes]
                    updated[left_route_index][left_pos] = right_customer
                    updated[right_route_index][right_pos] = left_customer
                    candidate_routes = clean_routes(updated)
                    candidate_score = score_cvrp_routes(candidate_routes, distance_matrix, objective_spec)
                    if is_better_score(candidate_score, current_score) and is_better_score(candidate_score, best_score):
                        best_score = candidate_score
                        best_routes = candidate_routes

    if best_routes is None or best_score is None:
        return None
    return best_routes, best_score


def best_or_opt_move(
    routes: list[list[int]],
    distance_matrix: list[list[float]],
    demands: list[float],
    capacity: float,
    objective: dict | None = None,
    segment_lengths: tuple[int, ...] = (2, 3),
) -> tuple[list[list[int]], ObjectiveScore] | None:
    objective_spec = normalize_objective_spec(objective)
    current_score = score_cvrp_routes(routes, distance_matrix, objective_spec)
    best_score: ObjectiveScore | None = None
    best_routes: list[list[int]] | None = None
    route_loads = [route_load(route, demands) for route in routes]

    for from_route_index, from_route in enumerate(routes):
        for segment_length in segment_lengths:
            if len(from_route) < segment_length:
                continue

            for start in range(len(from_route) - segment_length + 1):
                segment = from_route[start : start + segment_length]
                segment_load = sum(demands[customer] for customer in segment)

                for to_route_index, to_route in enumerate(routes):
                    if from_route_index != to_route_index and route_loads[to_route_index] + segment_load > capacity + EPS:
                        continue

                    for insert_pos in range(len(to_route) + 1):
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

                        candidate_routes = clean_routes(updated)
                        if candidate_routes == clean_routes(routes):
                            continue
                        candidate_score = score_cvrp_routes(candidate_routes, distance_matrix, objective_spec)
                        if is_better_score(candidate_score, current_score) and is_better_score(candidate_score, best_score):
                            best_score = candidate_score
                            best_routes = candidate_routes

    if best_routes is None or best_score is None:
        return None
    return best_routes, best_score


def best_two_opt_star_move(
    routes: list[list[int]],
    distance_matrix: list[list[float]],
    demands: list[float],
    capacity: float,
    objective: dict | None = None,
) -> tuple[list[list[int]], ObjectiveScore] | None:
    objective_spec = normalize_objective_spec(objective)
    current_score = score_cvrp_routes(routes, distance_matrix, objective_spec)
    best_score: ObjectiveScore | None = None
    best_routes: list[list[int]] | None = None

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

                    updated = [list(route) for route in routes]
                    updated[left_route_index] = candidate_left
                    updated[right_route_index] = candidate_right
                    candidate_routes = clean_routes(updated)
                    candidate_score = score_cvrp_routes(candidate_routes, distance_matrix, objective_spec)
                    if is_better_score(candidate_score, current_score) and is_better_score(candidate_score, best_score):
                        best_score = candidate_score
                        best_routes = candidate_routes

    if best_routes is None or best_score is None:
        return None
    return best_routes, best_score


def best_cross_exchange_move(
    routes: list[list[int]],
    distance_matrix: list[list[float]],
    demands: list[float],
    capacity: float,
    objective: dict | None = None,
    segment_lengths: tuple[int, ...] = (1, 2),
) -> tuple[list[list[int]], ObjectiveScore] | None:
    objective_spec = normalize_objective_spec(objective)
    current_score = score_cvrp_routes(routes, distance_matrix, objective_spec)
    best_score: ObjectiveScore | None = None
    best_routes: list[list[int]] | None = None
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

                            updated = [list(route) for route in routes]
                            updated[left_route_index] = candidate_left
                            updated[right_route_index] = candidate_right
                            candidate_routes = clean_routes(updated)
                            candidate_score = score_cvrp_routes(candidate_routes, distance_matrix, objective_spec)
                            if is_better_score(candidate_score, current_score) and is_better_score(candidate_score, best_score):
                                best_score = candidate_score
                                best_routes = candidate_routes

    if best_routes is None or best_score is None:
        return None
    return best_routes, best_score


def _best_customer_insertion(
    routes: list[list[int]],
    customer: int,
    distance_matrix: list[list[float]],
    demands: list[float],
    capacity: float,
    objective: dict | None = None,
) -> tuple[list[list[int]], ObjectiveScore] | None:
    objective_spec = normalize_objective_spec(objective)
    best_score: ObjectiveScore | None = None
    best_routes: list[list[int]] | None = None
    route_loads = [route_load(route, demands) for route in routes]

    for route_index, route in enumerate(routes):
        if route_loads[route_index] + demands[customer] > capacity + EPS:
            continue
        for insert_pos in range(len(route) + 1):
            updated = [list(item) for item in routes]
            updated[route_index] = route[:insert_pos] + [customer] + route[insert_pos:]
            candidate_routes = clean_routes(updated)
            candidate_score = score_cvrp_routes(candidate_routes, distance_matrix, objective_spec)
            if is_better_score(candidate_score, best_score):
                best_score = candidate_score
                best_routes = candidate_routes

    if best_routes is None or best_score is None:
        return None
    return best_routes, best_score


def best_route_elimination(
    routes: list[list[int]],
    distance_matrix: list[list[float]],
    demands: list[float],
    capacity: float,
    objective: dict | None = None,
) -> tuple[list[list[int]], ObjectiveScore] | None:
    if len(routes) < 2:
        return None

    objective_spec = normalize_objective_spec(objective)
    best_routes: list[list[int]] | None = None
    best_score: ObjectiveScore | None = None
    route_order = sorted(range(len(routes)), key=lambda index: (len(routes[index]), index))

    for route_index in route_order:
        remaining_routes = [list(route) for idx, route in enumerate(routes) if idx != route_index]
        customers_to_reinsert = list(routes[route_index])
        working_routes = clean_routes(remaining_routes)
        feasible = True

        for customer in customers_to_reinsert:
            insertion = _best_customer_insertion(working_routes, customer, distance_matrix, demands, capacity, objective_spec)
            if insertion is None:
                feasible = False
                break
            working_routes, _ = insertion

        if not feasible or len(working_routes) >= len(routes):
            continue

        candidate_score = score_cvrp_routes(working_routes, distance_matrix, objective_spec)
        if is_better_score(candidate_score, best_score):
            best_score = candidate_score
            best_routes = working_routes

    if best_routes is None or best_score is None:
        return None
    return best_routes, best_score


def reduce_cvrp_vehicle_count(
    instance: dict,
    solution: dict,
    config: dict | None = None,
) -> dict:
    config = dict(config or {})
    objective = normalize_objective_spec(config.get("objective"))
    demands = normalize_demands(instance["node_demand"])
    capacity = float(instance["capacity"])
    routes = clean_routes(normalize_routes_payload(solution))
    distance_matrix = build_vrp_distance_matrix(instance["depot_xy"], instance["node_xy"])

    max_rounds = int(config.get("max_rounds", 10))
    current_routes = routes
    current_score = score_cvrp_routes(current_routes, distance_matrix, objective)
    applied = 0

    for _ in range(max_rounds):
        move = best_route_elimination(current_routes, distance_matrix, demands, capacity, objective)
        if move is None:
            break
        next_routes, next_score = move
        if len(next_routes) >= len(current_routes):
            break
        current_routes = next_routes
        current_score = next_score
        applied += 1

    return {
        "problem_type": "cvrp",
        "routes": current_routes,
        "raw_sequence": routes_to_raw_sequence(current_routes),
        "distance": current_score.distance,
        "meta": {
            "vehicle_reduction_rounds": applied,
            "vehicle_count": len(current_routes),
            "final_score": current_score.generalized_cost,
            "objective": objective.__dict__,
        },
    }


def improve_cvrp_solution(
    instance: dict,
    solution: dict,
    config: dict | None = None,
) -> dict:
    config = dict(config or {})
    objective = normalize_objective_spec(config.get("objective"))
    demands = normalize_demands(instance["node_demand"])
    capacity = float(instance["capacity"])
    routes = normalize_routes_payload(solution)
    distance_matrix = build_vrp_distance_matrix(instance["depot_xy"], instance["node_xy"])

    operators = config.get(
        "operators",
        ["or_opt", "two_opt_star", "cross_exchange", "relocate", "swap", "two_opt"],
    )
    max_rounds = int(config.get("max_rounds", 50))

    current_routes = clean_routes(routes)
    current_score = score_cvrp_routes(current_routes, distance_matrix, objective)
    initial_score = current_score
    applied_operators: list[str] = []

    for _ in range(max_rounds):
        improved = False
        for operator in operators:
            if operator == "two_opt":
                move = best_intra_route_two_opt(current_routes, distance_matrix, objective)
            elif operator == "relocate":
                move = best_relocate_move(current_routes, distance_matrix, demands, capacity, objective)
            elif operator == "swap":
                move = best_swap_move(current_routes, distance_matrix, demands, capacity, objective)
            elif operator == "or_opt":
                move = best_or_opt_move(current_routes, distance_matrix, demands, capacity, objective)
            elif operator == "two_opt_star":
                move = best_two_opt_star_move(current_routes, distance_matrix, demands, capacity, objective)
            elif operator == "cross_exchange":
                move = best_cross_exchange_move(current_routes, distance_matrix, demands, capacity, objective)
            elif operator == "route_elimination":
                move = best_route_elimination(current_routes, distance_matrix, demands, capacity, objective)
            else:
                continue

            if move is None:
                continue

            next_routes, next_score = move
            if not is_better_score(next_score, current_score):
                continue

            current_routes = next_routes
            current_score = next_score
            applied_operators.append(operator)
            improved = True
            break

        if not improved:
            break

    return {
        "problem_type": "cvrp",
        "routes": current_routes,
        "raw_sequence": routes_to_raw_sequence(current_routes),
        "distance": current_score.distance,
        "meta": {
            "initial_distance": initial_score.distance,
            "improved_distance": current_score.distance,
            "initial_score": initial_score.generalized_cost,
            "improved_score": current_score.generalized_cost,
            "improvement": initial_score.generalized_cost - current_score.generalized_cost,
            "iterations": len(applied_operators),
            "applied_operators": applied_operators,
            "vehicle_count": len(current_routes),
            "objective": objective.__dict__,
        },
    }
