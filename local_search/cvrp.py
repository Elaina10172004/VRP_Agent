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
from .search_utils import (
    AcceptanceController,
    build_customer_knn,
    candidate_insert_positions,
    candidate_route_indices_for_nodes,
)


def _neighbor_lists(distance_matrix: list[list[float]], config: dict | None = None) -> list[list[int]]:
    config = config or {}
    return build_customer_knn(distance_matrix, int(config.get("granular_neighbor_k", 20)))


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
    neighbor_lists: list[list[int]] | None = None,
) -> tuple[list[list[int]], ObjectiveScore] | None:
    objective_spec = normalize_objective_spec(objective)
    current_score = score_cvrp_routes(routes, distance_matrix, objective_spec)
    best_score: ObjectiveScore | None = None
    best_routes: list[list[int]] | None = None
    route_loads = [route_load(route, demands) for route in routes]

    for from_route_index, from_route in enumerate(routes):
        for from_pos, customer in enumerate(from_route):
            target_route_indices = candidate_route_indices_for_nodes(
                routes,
                [customer],
                neighbor_lists,
                force_include=[from_route_index],
            )
            for to_route_index in target_route_indices:
                to_route = routes[to_route_index]
                if from_route_index != to_route_index and route_loads[to_route_index] + demands[customer] > capacity + EPS:
                    continue

                insert_positions = candidate_insert_positions(to_route, [customer], neighbor_lists)
                for insert_pos in insert_positions:
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
    neighbor_lists: list[list[int]] | None = None,
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

                target_route_indices = candidate_route_indices_for_nodes(
                    routes,
                    segment,
                    neighbor_lists,
                    force_include=[from_route_index],
                )
                for to_route_index in target_route_indices:
                    to_route = routes[to_route_index]
                    if from_route_index != to_route_index and route_loads[to_route_index] + segment_load > capacity + EPS:
                        continue

                    insert_positions = candidate_insert_positions(to_route, segment, neighbor_lists)
                    for insert_pos in insert_positions:
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
    neighbor_lists: list[list[int]] | None = None,
) -> tuple[list[list[int]], ObjectiveScore] | None:
    objective_spec = normalize_objective_spec(objective)
    current_score = score_cvrp_routes(routes, distance_matrix, objective_spec)
    best_score: ObjectiveScore | None = None
    best_routes: list[list[int]] | None = None
    route_loads = [route_load(route, demands) for route in routes]

    for left_route_index in range(len(routes) - 1):
        left_route = routes[left_route_index]
        candidate_right_routes = candidate_route_indices_for_nodes(
            routes,
            left_route,
            neighbor_lists,
            force_include=[left_route_index],
        )
        for right_route_index in candidate_right_routes:
            if right_route_index <= left_route_index:
                continue
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


def _collect_customer_insertions(
    routes: list[list[int]],
    customer: int,
    distance_matrix: list[list[float]],
    demands: list[float],
    capacity: float,
    objective: dict | None = None,
    neighbor_lists: list[list[int]] | None = None,
    include_new_route: bool = False,
) -> list[tuple[list[list[int]], ObjectiveScore]]:
    objective_spec = normalize_objective_spec(objective)
    candidates: list[tuple[list[list[int]], ObjectiveScore]] = []
    route_loads = [route_load(route, demands) for route in routes]

    target_route_indices = candidate_route_indices_for_nodes(routes, [customer], neighbor_lists)
    for route_index in target_route_indices:
        route = routes[route_index]
        if route_loads[route_index] + demands[customer] > capacity + EPS:
            continue
        for insert_pos in candidate_insert_positions(route, [customer], neighbor_lists):
            updated = [list(item) for item in routes]
            updated[route_index] = route[:insert_pos] + [customer] + route[insert_pos:]
            candidate_routes = clean_routes(updated)
            candidate_score = score_cvrp_routes(candidate_routes, distance_matrix, objective_spec)
            candidates.append((candidate_routes, candidate_score))

    if include_new_route and demands[customer] <= capacity + EPS:
        candidate_routes = clean_routes([*routes, [customer]])
        candidate_score = score_cvrp_routes(candidate_routes, distance_matrix, objective_spec)
        candidates.append((candidate_routes, candidate_score))

    candidates.sort(key=lambda item: item[1].ranking_key())
    return candidates


def _best_customer_insertion(
    routes: list[list[int]],
    customer: int,
    distance_matrix: list[list[float]],
    demands: list[float],
    capacity: float,
    objective: dict | None = None,
    neighbor_lists: list[list[int]] | None = None,
    include_new_route: bool = False,
) -> tuple[list[list[int]], ObjectiveScore] | None:
    candidates = _collect_customer_insertions(
        routes,
        customer,
        distance_matrix,
        demands,
        capacity,
        objective,
        neighbor_lists=neighbor_lists,
        include_new_route=include_new_route,
    )
    return candidates[0] if candidates else None


def _shaw_related_customers(
    routes: list[list[int]],
    distance_matrix: list[list[float]],
    demands: list[float],
    seed_customer: int,
) -> list[int]:
    route_of = {}
    for route_index, route in enumerate(routes):
        for customer in route:
            route_of[customer] = route_index

    seed_node = seed_customer + 1
    related = []
    for customer in route_of:
        node_index = customer + 1
        score = distance_matrix[seed_node][node_index] + 0.2 * abs(demands[seed_customer] - demands[customer])
        if route_of.get(customer) == route_of.get(seed_customer):
            score *= 0.7
        related.append((score, customer))
    related.sort(key=lambda item: (item[0], item[1]))
    return [customer for _score, customer in related]


def best_shaw_regret_move(
    routes: list[list[int]],
    distance_matrix: list[list[float]],
    demands: list[float],
    capacity: float,
    objective: dict | None = None,
    config: dict | None = None,
    neighbor_lists: list[list[int]] | None = None,
) -> tuple[list[list[int]], ObjectiveScore] | None:
    config = config or {}
    flattened = [customer for route in routes for customer in route]
    if len(flattened) < 2:
        return None

    objective_spec = normalize_objective_spec(objective)
    remove_count = min(len(flattened), max(2, int(config.get("shaw_remove_count", max(4, len(flattened) // 6 or 2)))))
    regret_k = max(2, int(config.get("regret_k", 3)))

    best_routes: list[list[int]] | None = None
    best_score: ObjectiveScore | None = None

    seed_candidates = []
    for route_index, route in enumerate(routes):
        for customer in route:
            prev_node = 0 if route.index(customer) == 0 else route[route.index(customer) - 1] + 1
            next_node = 0 if route.index(customer) == len(route) - 1 else route[route.index(customer) + 1] + 1
            customer_node = customer + 1
            break_cost = (
                distance_matrix[prev_node][customer_node]
                + distance_matrix[customer_node][next_node]
                - distance_matrix[prev_node][next_node]
            )
            seed_candidates.append((break_cost, route_index, customer))
    seed_candidates.sort(reverse=True)
    seed_pool = [customer for _cost, _route_index, customer in seed_candidates[: min(4, len(seed_candidates))]]
    if not seed_pool:
        seed_pool = flattened[:1]

    for seed_customer in seed_pool:
        removal_order = _shaw_related_customers(routes, distance_matrix, demands, seed_customer)
        removed_customers = removal_order[:remove_count]
        removed_set = set(removed_customers)
        working_routes = clean_routes([[customer for customer in route if customer not in removed_set] for route in routes])
        pending = list(removed_customers)
        feasible = True

        while pending:
            selected_customer: int | None = None
            selected_routes: list[list[int]] | None = None
            selected_score: ObjectiveScore | None = None
            selected_regret = float("-inf")

            for customer in pending:
                insertions = _collect_customer_insertions(
                    working_routes,
                    customer,
                    distance_matrix,
                    demands,
                    capacity,
                    objective_spec,
                    neighbor_lists=neighbor_lists,
                    include_new_route=True,
                )
                if not insertions:
                    feasible = False
                    break

                best_candidate_routes, best_candidate_score = insertions[0]
                compare_index = min(regret_k - 1, len(insertions) - 1)
                regret_value = insertions[compare_index][1].generalized_cost - best_candidate_score.generalized_cost
                if (
                    regret_value > selected_regret + EPS
                    or (
                        abs(regret_value - selected_regret) <= EPS
                        and (selected_score is None or is_better_score(best_candidate_score, selected_score))
                    )
                ):
                    selected_customer = customer
                    selected_routes = best_candidate_routes
                    selected_score = best_candidate_score
                    selected_regret = regret_value

            if not feasible or selected_customer is None or selected_routes is None or selected_score is None:
                feasible = False
                break

            working_routes = selected_routes
            pending.remove(selected_customer)

        if not feasible:
            continue

        candidate_score = score_cvrp_routes(working_routes, distance_matrix, objective_spec)
        if is_better_score(candidate_score, best_score):
            best_routes = working_routes
            best_score = candidate_score

    if best_routes is None or best_score is None:
        return None
    return best_routes, best_score


def best_route_elimination(
    routes: list[list[int]],
    distance_matrix: list[list[float]],
    demands: list[float],
    capacity: float,
    objective: dict | None = None,
    neighbor_lists: list[list[int]] | None = None,
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
            insertion = _best_customer_insertion(
                working_routes,
                customer,
                distance_matrix,
                demands,
                capacity,
                objective_spec,
                neighbor_lists=neighbor_lists,
            )
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
    neighbor_lists = _neighbor_lists(distance_matrix, config)
    applied = 0

    for _ in range(max_rounds):
        move = best_route_elimination(current_routes, distance_matrix, demands, capacity, objective, neighbor_lists=neighbor_lists)
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
    neighbor_lists = _neighbor_lists(distance_matrix, config)
    acceptance = AcceptanceController(
        enabled=bool(config.get("allow_worse_acceptance", False)),
        budget=int(config.get("acceptance_budget", max(0, max_rounds // 6))),
        temperature=float(config.get("acceptance_temperature", 0.01)),
        decay=float(config.get("acceptance_decay", 0.9)),
        random_seed=int(config.get("random_seed", 0)),
    )

    current_routes = clean_routes(routes)
    current_score = score_cvrp_routes(current_routes, distance_matrix, objective)
    best_routes = current_routes
    best_score = current_score
    initial_score = current_score
    applied_operators: list[str] = []
    accepted_worse_operators: list[str] = []

    for iteration in range(max_rounds):
        accepted = False
        for operator in operators:
            if operator == "two_opt":
                move = best_intra_route_two_opt(current_routes, distance_matrix, objective)
            elif operator == "relocate":
                move = best_relocate_move(current_routes, distance_matrix, demands, capacity, objective, neighbor_lists=neighbor_lists)
            elif operator == "swap":
                move = best_swap_move(current_routes, distance_matrix, demands, capacity, objective)
            elif operator == "or_opt":
                move = best_or_opt_move(current_routes, distance_matrix, demands, capacity, objective, neighbor_lists=neighbor_lists)
            elif operator == "two_opt_star":
                move = best_two_opt_star_move(current_routes, distance_matrix, demands, capacity, objective)
            elif operator == "cross_exchange":
                move = best_cross_exchange_move(current_routes, distance_matrix, demands, capacity, objective, neighbor_lists=neighbor_lists)
            elif operator == "route_elimination":
                move = best_route_elimination(current_routes, distance_matrix, demands, capacity, objective, neighbor_lists=neighbor_lists)
            elif operator == "shaw_regret":
                move = best_shaw_regret_move(
                    current_routes,
                    distance_matrix,
                    demands,
                    capacity,
                    objective,
                    config=config,
                    neighbor_lists=neighbor_lists,
                )
            else:
                continue

            if move is None:
                continue

            next_routes, next_score = move
            if not acceptance.consider(next_score, current_score, iteration):
                continue

            accepted_worse = not is_better_score(next_score, current_score)
            current_routes = next_routes
            current_score = next_score
            applied_operators.append(operator)
            if is_better_score(current_score, best_score):
                best_routes = current_routes
                best_score = current_score
            if accepted_worse:
                accepted_worse_operators.append(operator)
            accepted = True
            break

        if not accepted:
            break

    return {
        "problem_type": "cvrp",
        "routes": best_routes,
        "raw_sequence": routes_to_raw_sequence(best_routes),
        "distance": best_score.distance,
        "meta": {
            "initial_distance": initial_score.distance,
            "improved_distance": best_score.distance,
            "initial_score": initial_score.generalized_cost,
            "improved_score": best_score.generalized_cost,
            "improvement": initial_score.generalized_cost - best_score.generalized_cost,
            "iterations": len(applied_operators),
            "applied_operators": applied_operators,
            "accepted_worse_operators": accepted_worse_operators,
            "acceptance": acceptance.snapshot(),
            "vehicle_count": len(best_routes),
            "objective": objective.__dict__,
        },
    }
