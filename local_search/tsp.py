from __future__ import annotations

from .common import EPS, build_distance_matrix, ensure_points, is_better_score, normalize_objective_spec, score_tsp_tour


def tour_length(tour: list[int], distance_matrix: list[list[float]]) -> float:
    total = 0.0
    node_count = len(tour)
    for index in range(node_count):
        total += distance_matrix[tour[index]][tour[(index + 1) % node_count]]
    return total


def best_two_opt_move(tour: list[int], distance_matrix: list[list[float]]) -> tuple[list[int], float] | None:
    node_count = len(tour)
    if node_count < 4:
        return None

    best_delta = 0.0
    best_move: tuple[int, int] | None = None
    for i in range(node_count - 1):
        prev_node = tour[i - 1]
        start_node = tour[i]
        for j in range(i + 1, node_count):
            if i == 0 and j == node_count - 1:
                continue
            end_node = tour[j]
            next_node = tour[(j + 1) % node_count]
            delta = (
                distance_matrix[prev_node][end_node]
                + distance_matrix[start_node][next_node]
                - distance_matrix[prev_node][start_node]
                - distance_matrix[end_node][next_node]
            )
            if delta < best_delta - EPS:
                best_delta = delta
                best_move = (i, j)

    if best_move is None:
        return None

    i, j = best_move
    improved = tour[:i] + list(reversed(tour[i : j + 1])) + tour[j + 1 :]
    return improved, best_delta


def best_relocate_move(tour: list[int], distance_matrix: list[list[float]]) -> tuple[list[int], float] | None:
    current_cost = tour_length(tour, distance_matrix)
    node_count = len(tour)
    best_delta = 0.0
    best_tour: list[int] | None = None

    for from_index in range(node_count):
        node = tour[from_index]
        reduced = tour[:from_index] + tour[from_index + 1 :]
        for insert_index in range(node_count):
            candidate = reduced[:insert_index] + [node] + reduced[insert_index:]
            if candidate == tour:
                continue
            delta = tour_length(candidate, distance_matrix) - current_cost
            if delta < best_delta - EPS:
                best_delta = delta
                best_tour = candidate

    if best_tour is None:
        return None
    return best_tour, best_delta


def best_swap_move(tour: list[int], distance_matrix: list[list[float]]) -> tuple[list[int], float] | None:
    current_cost = tour_length(tour, distance_matrix)
    node_count = len(tour)
    best_delta = 0.0
    best_tour: list[int] | None = None

    for i in range(node_count - 1):
        for j in range(i + 1, node_count):
            candidate = list(tour)
            candidate[i], candidate[j] = candidate[j], candidate[i]
            delta = tour_length(candidate, distance_matrix) - current_cost
            if delta < best_delta - EPS:
                best_delta = delta
                best_tour = candidate

    if best_tour is None:
        return None
    return best_tour, best_delta


def improve_tsp_solution(
    instance: dict,
    solution: dict,
    config: dict | None = None,
) -> dict:
    config = dict(config or {})
    points = ensure_points(instance["points"])
    tour = [int(node) for node in solution.get("tour", [])]
    if len(tour) != len(points):
        raise ValueError("TSP solution tour length does not match instance size.")

    distance_matrix = build_distance_matrix(points)
    objective = normalize_objective_spec(config.get("objective"))
    operators = config.get("operators", ["two_opt", "relocate", "swap"])
    max_rounds = int(config.get("max_rounds", 50))

    current = list(tour)
    initial_score = score_tsp_tour(current, distance_matrix, objective)
    applied_operators: list[str] = []

    for _ in range(max_rounds):
        improved = False
        current_score = score_tsp_tour(current, distance_matrix, objective)
        for operator in operators:
            if operator == "two_opt":
                move = best_two_opt_move(current, distance_matrix)
            elif operator == "relocate":
                move = best_relocate_move(current, distance_matrix)
            elif operator == "swap":
                move = best_swap_move(current, distance_matrix)
            else:
                continue

            if move is None:
                continue

            candidate, _ = move
            candidate_score = score_tsp_tour(candidate, distance_matrix, objective)
            if not is_better_score(candidate_score, current_score):
                continue

            current = candidate
            applied_operators.append(operator)
            improved = True
            break

        if not improved:
            break

    final_score = score_tsp_tour(current, distance_matrix, objective)
    return {
        "problem_type": "tsp",
        "tour": current,
        "closed_tour": current + [current[0]] if current else [],
        "distance": final_score.distance,
        "meta": {
            "initial_distance": initial_score.distance,
            "improved_distance": final_score.distance,
            "initial_score": initial_score.generalized_cost,
            "improved_score": final_score.generalized_cost,
            "improvement": initial_score.generalized_cost - final_score.generalized_cost,
            "iterations": len(applied_operators),
            "applied_operators": applied_operators,
            "objective": objective.__dict__,
        },
    }
