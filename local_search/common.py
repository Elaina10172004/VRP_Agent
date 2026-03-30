from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot


EPS = 1e-9


@dataclass(frozen=True)
class ObjectiveSpec:
    primary: str = "distance"
    distance_weight: float = 1.0
    duration_weight: float = 0.0
    vehicle_fixed_cost: float = 0.0
    overtime_penalty: float = 0.0
    lateness_penalty: float = 0.0
    unserved_penalty: float = 1_000_000.0


@dataclass(frozen=True)
class ObjectiveScore:
    feasible: bool
    vehicle_count: int
    distance: float
    duration: float
    generalized_cost: float
    lateness: float = 0.0
    overtime: float = 0.0
    unserved_count: int = 0
    violations: int = 0
    objective: ObjectiveSpec = field(default_factory=ObjectiveSpec)

    def ranking_key(self) -> tuple:
        feasible_rank = 0 if self.feasible else 1
        if self.objective.primary == "vehicle_count":
            primary_value = self.vehicle_count
            secondary_value = self.generalized_cost
        else:
            primary_value = self.generalized_cost
            secondary_value = self.vehicle_count
        return (
            feasible_rank,
            primary_value,
            secondary_value,
            self.distance,
            self.duration,
            self.unserved_count,
            self.violations,
        )


@dataclass(frozen=True)
class RouteEvaluation:
    feasible: bool
    distance: float
    load: float
    duration: float = 0.0
    end_time: float = 0.0
    lateness: float = 0.0
    overtime: float = 0.0

    @property
    def cost(self) -> float:
        return self.distance


def _as_float(value, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def normalize_objective_spec(value: ObjectiveSpec | dict | None) -> ObjectiveSpec:
    if isinstance(value, ObjectiveSpec):
        return value
    if not isinstance(value, dict):
        return ObjectiveSpec()

    primary = str(value.get("primary", "distance")).strip().lower()
    if bool(value.get("prioritize_vehicle_count")):
        primary = "vehicle_count"
    if primary not in {"distance", "vehicle_count"}:
        primary = "distance"

    vehicle_fixed_cost = _as_float(
        value.get("vehicle_fixed_cost", value.get("vehicle_count_weight", 0.0)),
        0.0,
    )

    return ObjectiveSpec(
        primary=primary,
        distance_weight=_as_float(value.get("distance_weight", 1.0), 1.0),
        duration_weight=_as_float(value.get("duration_weight", 0.0), 0.0),
        vehicle_fixed_cost=vehicle_fixed_cost,
        overtime_penalty=_as_float(value.get("overtime_penalty", 0.0), 0.0),
        lateness_penalty=_as_float(value.get("lateness_penalty", 0.0), 0.0),
        unserved_penalty=_as_float(value.get("unserved_penalty", 1_000_000.0), 1_000_000.0),
    )


def is_better_score(candidate: ObjectiveScore, incumbent: ObjectiveScore | None) -> bool:
    if incumbent is None:
        return True
    return candidate.ranking_key() < incumbent.ranking_key()


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


def route_distance(route: list[int], distance_matrix: list[list[float]]) -> float:
    if not route:
        return 0.0

    total = distance_matrix[0][route[0] + 1]
    for idx in range(len(route) - 1):
        total += distance_matrix[route[idx] + 1][route[idx + 1] + 1]
    total += distance_matrix[route[-1] + 1][0]
    return total


def route_cost(route: list[int], distance_matrix: list[list[float]]) -> float:
    return route_distance(route, distance_matrix)


def total_route_cost(routes: list[list[int]], distance_matrix: list[list[float]]) -> float:
    return sum(route_distance(route, distance_matrix) for route in routes)


def route_duration(
    route: list[int],
    distance_matrix: list[list[float]],
    evaluation: RouteEvaluation | None = None,
) -> float:
    if evaluation is not None:
        return float(evaluation.duration)
    return route_distance(route, distance_matrix)


def route_load(route: list[int], demands: list[float]) -> float:
    return sum(demands[customer] for customer in route)


def route_feasible(evaluation: RouteEvaluation | None) -> bool:
    return True if evaluation is None else bool(evaluation.feasible)


def route_generalized_cost(
    route: list[int],
    distance_matrix: list[list[float]],
    objective: ObjectiveSpec | dict | None = None,
    evaluation: RouteEvaluation | None = None,
) -> float:
    objective_spec = normalize_objective_spec(objective)
    distance = route_distance(route, distance_matrix) if evaluation is None else float(evaluation.distance)
    duration = route_duration(route, distance_matrix, evaluation)
    lateness = 0.0 if evaluation is None else float(evaluation.lateness)
    overtime = 0.0 if evaluation is None else float(evaluation.overtime)
    return (
        objective_spec.vehicle_fixed_cost
        + objective_spec.distance_weight * distance
        + objective_spec.duration_weight * duration
        + objective_spec.lateness_penalty * lateness
        + objective_spec.overtime_penalty * overtime
    )


def solution_vehicle_count(routes: list[list[int]]) -> int:
    return len(clean_routes(routes))


def solution_distance(
    routes: list[list[int]],
    distance_matrix: list[list[float]],
    route_evaluations: list[RouteEvaluation] | None = None,
) -> float:
    if route_evaluations is None:
        return sum(route_distance(route, distance_matrix) for route in routes)
    return sum(float(evaluation.distance) for evaluation in route_evaluations)


def solution_duration(
    routes: list[list[int]],
    distance_matrix: list[list[float]],
    route_evaluations: list[RouteEvaluation] | None = None,
) -> float:
    if route_evaluations is None:
        return sum(route_duration(route, distance_matrix) for route in routes)
    return sum(float(evaluation.duration) for evaluation in route_evaluations)


def solution_cost(
    routes: list[list[int]],
    distance_matrix: list[list[float]],
    objective: ObjectiveSpec | dict | None = None,
    route_evaluations: list[RouteEvaluation] | None = None,
    unserved_count: int = 0,
) -> float:
    objective_spec = normalize_objective_spec(objective)
    route_costs = 0.0
    if route_evaluations is None:
        for route in routes:
            route_costs += route_generalized_cost(route, distance_matrix, objective_spec)
    else:
        for route, evaluation in zip(routes, route_evaluations, strict=False):
            route_costs += route_generalized_cost(route, distance_matrix, objective_spec, evaluation)
    return route_costs + objective_spec.unserved_penalty * max(0, int(unserved_count))


def solution_score(
    routes: list[list[int]],
    distance_matrix: list[list[float]],
    objective: ObjectiveSpec | dict | None = None,
    route_evaluations: list[RouteEvaluation] | None = None,
    feasible: bool = True,
    unserved_count: int = 0,
    violations: int = 0,
) -> ObjectiveScore:
    objective_spec = normalize_objective_spec(objective)
    evaluations = route_evaluations or []
    distance = solution_distance(routes, distance_matrix, route_evaluations)
    duration = solution_duration(routes, distance_matrix, route_evaluations)
    generalized_cost = solution_cost(routes, distance_matrix, objective_spec, route_evaluations, unserved_count)
    lateness = sum(float(evaluation.lateness) for evaluation in evaluations)
    overtime = sum(float(evaluation.overtime) for evaluation in evaluations)
    return ObjectiveScore(
        feasible=feasible,
        vehicle_count=solution_vehicle_count(routes),
        distance=distance,
        duration=duration,
        generalized_cost=generalized_cost,
        lateness=lateness,
        overtime=overtime,
        unserved_count=max(0, int(unserved_count)),
        violations=max(0, int(violations)),
        objective=objective_spec,
    )


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
    distance = 0.0
    previous = 0

    for customer in route:
        load += demands[customer]
        if load > capacity + EPS:
            return RouteEvaluation(False, float("inf"), load, float("inf"), time)

        node_index = customer + 1
        travel = distance_matrix[previous][node_index]
        arrival = time + travel
        ready, due = time_windows[customer]
        start = max(arrival, ready)
        if start > due + EPS:
            return RouteEvaluation(False, float("inf"), load, float("inf"), time, lateness=start - due)

        distance += travel
        time = start + service_times[customer]
        previous = node_index

    return_travel = distance_matrix[previous][0]
    distance += return_travel
    duration = time + return_travel
    return RouteEvaluation(True, distance, load, duration, time)


def score_tsp_tour(
    tour: list[int],
    distance_matrix: list[list[float]],
    objective: ObjectiveSpec | dict | None = None,
) -> ObjectiveScore:
    objective_spec = normalize_objective_spec(objective)
    if not tour:
        return ObjectiveScore(False, 0, float("inf"), float("inf"), float("inf"), violations=1, objective=objective_spec)
    distance = 0.0
    node_count = len(tour)
    for index in range(node_count):
        distance += distance_matrix[tour[index]][tour[(index + 1) % node_count]]
    generalized_cost = objective_spec.distance_weight * distance + objective_spec.duration_weight * distance
    return ObjectiveScore(
        feasible=True,
        vehicle_count=1,
        distance=distance,
        duration=distance,
        generalized_cost=generalized_cost,
        objective=objective_spec,
    )


def score_cvrp_routes(
    routes: list[list[int]],
    distance_matrix: list[list[float]],
    objective: ObjectiveSpec | dict | None = None,
) -> ObjectiveScore:
    return solution_score(clean_routes(routes), distance_matrix, objective)


def score_cvrptw_routes(
    routes: list[list[int]],
    distance_matrix: list[list[float]],
    demands: list[float],
    capacity: float,
    time_windows: list[tuple[float, float]],
    service_times: list[float],
    objective: ObjectiveSpec | dict | None = None,
) -> ObjectiveScore:
    normalized_routes = clean_routes(routes)
    evaluations = [
        evaluate_cvrptw_route(route, distance_matrix, demands, capacity, time_windows, service_times)
        for route in normalized_routes
    ]
    feasible = all(route_feasible(evaluation) for evaluation in evaluations)
    if not feasible:
        objective_spec = normalize_objective_spec(objective)
        violation_count = sum(0 if evaluation.feasible else 1 for evaluation in evaluations)
        return ObjectiveScore(
            feasible=False,
            vehicle_count=len(normalized_routes),
            distance=float("inf"),
            duration=float("inf"),
            generalized_cost=float("inf"),
            violations=violation_count,
            objective=objective_spec,
        )
    return solution_score(normalized_routes, distance_matrix, objective, evaluations, feasible=True)
