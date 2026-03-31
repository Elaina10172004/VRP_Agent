from __future__ import annotations

from .common import split_depot_sequence


def _normalize_tsp_cycle(nodes: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    cycle = tuple(int(node) for node in nodes)
    if not cycle:
        return tuple()

    min_node = min(cycle)
    start_idx = cycle.index(min_node)
    forward = cycle[start_idx:] + cycle[:start_idx]

    reversed_cycle = tuple(reversed(cycle))
    reversed_start_idx = reversed_cycle.index(min_node)
    backward = reversed_cycle[reversed_start_idx:] + reversed_cycle[:reversed_start_idx]
    return min(forward, backward)


def _normalize_route_tuple(route: list[int] | tuple[int, ...], *, reversible: bool) -> tuple[int, ...]:
    route_tuple = tuple(int(customer) for customer in route)
    if not reversible:
        return route_tuple
    reversed_tuple = tuple(reversed(route_tuple))
    return min(route_tuple, reversed_tuple)


def _normalize_vrp_routes(
    routes: list[list[int]] | tuple[tuple[int, ...], ...],
    *,
    reversible_routes: bool,
) -> tuple[tuple[int, ...], ...]:
    normalized_routes = [
        _normalize_route_tuple(route, reversible=reversible_routes)
        for route in routes
        if len(route) > 0
    ]
    normalized_routes.sort()
    return tuple(normalized_routes)


def canonical_solution_key(problem_type: str, solution: dict) -> tuple:
    normalized_problem_type = str(problem_type).strip().lower()

    if normalized_problem_type == "tsp":
        if isinstance(solution.get("tour"), list):
            return ("tsp", _normalize_tsp_cycle(solution["tour"]))
        if isinstance(solution.get("closed_tour"), list):
            closed_tour = [int(node) for node in solution["closed_tour"]]
            if len(closed_tour) >= 2 and closed_tour[0] == closed_tour[-1]:
                closed_tour = closed_tour[:-1]
            return ("tsp", _normalize_tsp_cycle(closed_tour))
        raise ValueError("TSP solution must contain 'tour' or 'closed_tour'.")

    if normalized_problem_type in {"cvrp", "cvrptw"}:
        if isinstance(solution.get("routes"), list):
            return (
                normalized_problem_type,
                _normalize_vrp_routes(
                    solution["routes"],
                    reversible_routes=normalized_problem_type == "cvrp",
                ),
            )
        if isinstance(solution.get("raw_sequence"), list):
            return (
                normalized_problem_type,
                _normalize_vrp_routes(
                    split_depot_sequence([int(node) for node in solution["raw_sequence"]]),
                    reversible_routes=normalized_problem_type == "cvrp",
                ),
            )
        raise ValueError(f"{normalized_problem_type.upper()} solution must contain 'routes' or 'raw_sequence'.")

    raise ValueError(f"Unsupported problem_type: {problem_type}")


def dedupe_solutions(problem_type: str, solutions: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    unique: list[dict] = []
    for solution in solutions:
        key = canonical_solution_key(problem_type, solution)
        if key in seen:
            continue
        seen.add(key)
        unique.append(solution)
    return unique
