from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, isfinite
from random import Random

from .common import ObjectiveScore, is_better_score


def build_customer_knn(distance_matrix: list[list[float]], neighbor_k: int) -> list[list[int]]:
    customer_count = max(0, len(distance_matrix) - 1)
    if customer_count == 0:
        return []

    limit = max(1, min(int(neighbor_k), max(1, customer_count - 1)))
    neighbors: list[list[int]] = []
    for customer in range(customer_count):
        node_index = customer + 1
        ordered = sorted(
            (other for other in range(customer_count) if other != customer),
            key=lambda other: distance_matrix[node_index][other + 1],
        )
        neighbors.append(ordered[:limit])
    return neighbors


def candidate_route_indices_for_nodes(
    routes: list[list[int]],
    nodes: list[int] | tuple[int, ...],
    neighbor_lists: list[list[int]] | None,
    force_include: list[int] | tuple[int, ...] = (),
) -> list[int]:
    if not routes:
        return []
    if not neighbor_lists:
        return list(range(len(routes)))

    node_set = {int(node) for node in nodes}
    expanded = set(node_set)
    for node in node_set:
        if 0 <= node < len(neighbor_lists):
            expanded.update(neighbor_lists[node])

    candidates = {int(index) for index in force_include if 0 <= int(index) < len(routes)}
    for route_index, route in enumerate(routes):
        if any(customer in expanded for customer in route):
            candidates.add(route_index)

    return sorted(candidates) if candidates else list(range(len(routes)))


def candidate_insert_positions(
    route: list[int],
    nodes: list[int] | tuple[int, ...],
    neighbor_lists: list[list[int]] | None,
) -> list[int]:
    if not route:
        return [0]
    if not neighbor_lists:
        return list(range(len(route) + 1))

    node_set = {int(node) for node in nodes}
    expanded = set(node_set)
    for node in node_set:
        if 0 <= node < len(neighbor_lists):
            expanded.update(neighbor_lists[node])

    positions = {0, len(route)}
    for index, customer in enumerate(route):
        if customer in expanded:
            positions.add(index)
            positions.add(index + 1)
    return sorted(position for position in positions if 0 <= position <= len(route))


@dataclass
class AcceptanceController:
    enabled: bool = False
    budget: int = 0
    temperature: float = 0.01
    decay: float = 0.9
    random_seed: int = 0
    worse_accepts: int = 0
    _rng: Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.budget = max(0, int(self.budget))
        self.temperature = max(1e-6, float(self.temperature))
        self.decay = min(1.0, max(0.01, float(self.decay)))
        self._rng = Random(int(self.random_seed))

    def consider(self, candidate: ObjectiveScore, current: ObjectiveScore, iteration: int) -> bool:
        if is_better_score(candidate, current):
            return True
        if not self.enabled or self.worse_accepts >= self.budget:
            return False
        if not candidate.feasible or not current.feasible:
            return False
        if not (isfinite(candidate.generalized_cost) and isfinite(current.generalized_cost)):
            return False

        delta = candidate.generalized_cost - current.generalized_cost
        if delta <= 0.0:
            return True

        scale = max(1.0, abs(current.generalized_cost)) * self.temperature * (self.decay ** max(0, int(iteration)))
        probability = exp(-delta / max(scale, 1e-6))
        if self._rng.random() < probability:
            self.worse_accepts += 1
            return True
        return False

    def snapshot(self) -> dict:
        return {
            "enabled": self.enabled,
            "budget": self.budget,
            "temperature": self.temperature,
            "decay": self.decay,
            "accepted_worse_moves": self.worse_accepts,
        }
