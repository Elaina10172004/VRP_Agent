from __future__ import annotations

from dataclasses import dataclass
from math import hypot


EPS = 1e-9


@dataclass(frozen=True)
class RouteEvaluation:
    feasible: bool
    cost: float
    load: float
    end_time: float = 0.0


def ensure_point(point: list[float] | tuple[float, float]) -> tuple[float, float]:
    if len(point) != 2:
        raise ValueError(f"Expected 2D point, got: {point}")
    return float(point[0]), float(point[1])


def ensure_points(points: list[list[float]] | list[tuple[float, float]]) -> list[tuple[float, float]]:
    return [ensure_point(point) for point in points]


def build_distance_matrix(points: list[tuple[float, float]]) -> list[list[float]]:
    size = len(points)
    matrix = [[0.0] * size for _ in range(size)]
    for i in range(size):
        x1, y1 = points[i]
        for j in range(i + 1, size):
            x2, y2 = points[j]
            dist = hypot(x1 - x2, y1 - y2)
            matrix[i][j] = dist
            matrix[j][i] = dist
    return matrix


def build_vrp_distance_matrix(
    depot_xy: list[float] | tuple[float, float],
    node_xy: list[list[float]] | list[tuple[float, float]],
) -> list[list[float]]:
    points = [ensure_point(depot_xy), *ensure_points(node_xy)]
    return build_distance_matrix(points)


def clean_routes(routes: list[list[int]]) -> list[list[int]]:
    return [list(route) for route in routes if route]


def split_raw_sequence(raw_sequence: list[int]) -> list[list[int]]:
    routes: list[list[int]] = []
    current: list[int] = []
    for node in raw_sequence:
        if int(node) == 0:
            if current:
                routes.append(current)
                current = []
            continue
        current.append(int(node) - 1)
    if current:
        routes.append(current)
    return routes


def routes_to_raw_sequence(routes: list[list[int]]) -> list[int]:
    routes = clean_routes(routes)
    if not routes:
        return [0]

    raw: list[int] = [0]
    for route in routes:
        raw.extend(customer + 1 for customer in route)
        raw.append(0)
    return raw


def route_cost(route: list[int], distance_matrix: list[list[float]]) -> float:
    if not route:
        return 0.0

    total = distance_matrix[0][route[0] + 1]
    for idx in range(len(route) - 1):
        total += distance_matrix[route[idx] + 1][route[idx + 1] + 1]
    total += distance_matrix[route[-1] + 1][0]
    return total


def total_route_cost(routes: list[list[int]], distance_matrix: list[list[float]]) -> float:
    return sum(route_cost(route, distance_matrix) for route in routes)


def route_load(route: list[int], demands: list[float]) -> float:
    return sum(demands[customer] for customer in route)


def normalize_routes_payload(solution: dict) -> list[list[int]]:
    if "routes" in solution and solution["routes"] is not None:
        return clean_routes([[int(customer) for customer in route] for route in solution["routes"]])
    if "raw_sequence" in solution and solution["raw_sequence"] is not None:
        return split_raw_sequence([int(node) for node in solution["raw_sequence"]])
    raise ValueError("Route solution must contain either 'routes' or 'raw_sequence'.")


def normalize_service_times(service_time: float | list[float], customer_count: int) -> list[float]:
    if isinstance(service_time, (int, float)):
        return [float(service_time)] * customer_count
    values = [float(value) for value in service_time]
    if len(values) != customer_count:
        raise ValueError(f"Expected {customer_count} service times, got {len(values)}.")
    return values


def normalize_demands(node_demand: list[float] | tuple[float, ...]) -> list[float]:
    return [float(value) for value in node_demand]


def normalize_time_windows(node_tw: list[list[float]] | list[tuple[float, float]]) -> list[tuple[float, float]]:
    windows: list[tuple[float, float]] = []
    for tw in node_tw:
        if len(tw) != 2:
            raise ValueError(f"Expected [ready, due] pair, got: {tw}")
        ready, due = float(tw[0]), float(tw[1])
        windows.append((ready, due))
    return windows


def evaluate_cvrptw_route(
    route: list[int],
    distance_matrix: list[list[float]],
    demands: list[float],
    capacity: float,
    time_windows: list[tuple[float, float]],
    service_times: list[float],
) -> RouteEvaluation:
    load = 0.0
    time = 0.0
    cost = 0.0
    previous = 0

    for customer in route:
        load += demands[customer]
        if load > capacity + EPS:
            return RouteEvaluation(False, float("inf"), load, time)

        node_index = customer + 1
        travel = distance_matrix[previous][node_index]
        arrival = time + travel
        ready, due = time_windows[customer]
        start = max(arrival, ready)
        if start > due + EPS:
            return RouteEvaluation(False, float("inf"), load, time)

        cost += travel
        time = start + service_times[customer]
        previous = node_index

    cost += distance_matrix[previous][0]
    return RouteEvaluation(True, cost, load, time)
