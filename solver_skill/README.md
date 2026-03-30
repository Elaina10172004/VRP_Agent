# solver_skill

Solve orchestration layer for the VRP agent.

Available actions:

- `construct_initial`
- `validate_solution`
- `reduce_vehicles`
- `apply_lookahead`
- `destroy_repair`
- `improve_solution`
- `compare_solutions`

Default DRL behavior is still `k=128` rollouts and selecting the best rollout, but the post-processing chain is now configurable through `tool_plan` and operator lists instead of being hardcoded.

All ranking now goes through `objective`, so construction selection, lookahead, and local search can optimize for more than pure distance.

## CLI

```bash
python -m solver_skill.cli --input payload.json --output result.json --pretty
```

## Payload

```json
{
  "problem_type": "cvrp",
  "instance": {
    "depot_xy": [0, 0],
    "node_xy": [[1, 0], [2, 0]],
    "node_demand": [1, 1],
    "capacity": 10
  },
  "config": {
    "mode": "hybrid",
    "drl_samples": 128,
    "tool_plan": [
      "construct_initial",
      "validate_solution",
      "reduce_vehicles",
      "validate_solution",
      "apply_lookahead",
      "validate_solution",
      "improve_solution",
      "compare_solutions"
    ],
    "enable_vehicle_reduction": true,
    "objective": {
      "primary": "vehicle_count",
      "vehicle_fixed_cost": 100.0
    },
    "enable_lookahead": true,
    "lookahead_depth": 2,
    "lookahead_beam_width": 4,
    "enable_local_search": true,
    "local_search_rounds": 50,
    "local_search_operators": ["or_opt", "two_opt_star", "cross_exchange", "relocate", "swap", "two_opt"]
  }
}
```

`mode="fast"` keeps the short path and returns the best DRL construction result directly.

Supported objective fields:

- `primary`: `distance` or `vehicle_count`
- `vehicle_fixed_cost`
- `distance_weight`
- `duration_weight`
- `overtime_penalty`
- `lateness_penalty`
- `unserved_penalty`
