# solver_skill

Unified solve pipeline for OptiChat:

1. `DRL` seed generation from `solver_core`
2. optional `lookahead`
3. optional `local_search`

Default DRL behavior is `k=128` rollouts and selecting the best rollout.
In `hybrid` mode, the default chain is `DRL -> lookahead -> local_search` unless a stage is explicitly disabled.

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
    "enable_lookahead": true,
    "lookahead_depth": 2,
    "lookahead_beam_width": 4,
    "enable_local_search": true,
    "local_search_rounds": 50
  }
}
```

`mode="fast"` will skip lookahead and local search and return the DRL seed directly.
