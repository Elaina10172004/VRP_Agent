# solver_core

Minimal DRL inference layer for:

- `TSP`
- `CVRP`
- `CVRPTW`

It only keeps the code required to load PolyNet checkpoints and produce solutions.

## Usage

```python
from solver_core import TSPSolver, CVRPSolver, CVRPTWSolver

tsp_solver = TSPSolver()
cvrp_solver = CVRPSolver()
cvrptw_solver = CVRPTWSolver()
```

### TSP

```python
result = tsp_solver.solve(points, num_samples=128)
```

- `points`: `(N, 2)` or `(B, N, 2)`
- returns `tour`, `closed_tour`, `distance`

### CVRP

```python
result = cvrp_solver.solve(depot_xy, node_xy, node_demand, capacity, num_samples=128)
```

- `depot_xy`: `(2,)`, `(B, 2)`, or `(B, 1, 2)`
- `node_xy`: `(N, 2)` or `(B, N, 2)`
- `node_demand`: `(N,)` or `(B, N)`
- `capacity`: scalar or `(B,)`
- returns `raw_sequence`, `routes`, `distance`

### CVRPTW

```python
result = cvrptw_solver.solve(
    depot_xy,
    node_xy,
    node_demand,
    capacity,
    node_tw,
    service_time,
    num_samples=128,
    grid_scale=1000.0,
)
```

- `node_tw`: `(N, 2)` or `(B, N, 2)`
- `service_time`: scalar, `(N,)`, `(B,)`, or `(B, N)`
- `grid_scale`: recommended when coordinates/time windows are already in a known scale
- returns `raw_sequence`, `routes`, `distance`

## Notes

- `num_samples` is the DRL rollout count. Your future `fast` mode can call these solvers directly.
- `routes` use zero-based customer indices.
- The remaining checkpoints are:
  - `TSP/models/PolyNet/...`
  - `CVRP/models/PolyNet/...`
  - `PolyNet/pt/checkpoint-1500.pt`
