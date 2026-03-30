# local_search

Local improvement scripts for the DRL seed solutions in [`solver_core`](../solver_core/README.md).

Current operators:

- `TSP`: `two_opt`, `relocate`, `swap`
- `CVRP`: intra-route `two_opt`, inter/intra-route `relocate`, inter-route `swap`
- `CVRPTW`: feasible intra-route `two_opt`, feasible inter/intra-route `relocate`, feasible inter-route `swap`

## CLI

```bash
python -m local_search.cli --input payload.json --output improved.json --pretty
```

If `--input` is omitted, the script reads JSON from `stdin`.

## Payload format

```json
{
  "problem_type": "cvrp",
  "instance": {
    "depot_xy": [0.5, 0.5],
    "node_xy": [[0.1, 0.2], [0.8, 0.9]],
    "node_demand": [3, 4],
    "capacity": 10
  },
  "solution": {
    "routes": [[0, 1]]
  },
  "config": {
    "max_rounds": 50,
    "operators": ["two_opt", "relocate", "swap"]
  }
}
```

`solution` can also carry `raw_sequence` for `CVRP/CVRPTW`.

## Notes

- Customer indices in `routes` are zero-based.
- `raw_sequence` follows the same convention as the solver output:
  depot is `0`, customers are `customer_index + 1`.
- For `CVRPTW`, `service_time` can be a scalar or a per-customer list.
