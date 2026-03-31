---
name: upload-instance-solver
description: Use when a user uploads a routing instance file and you need to detect whether it is TSP, CVRP, or CVRPTW, normalize it into this repo's canonical payload/files, and solve it with the local DRL plus optional lookahead/local-search pipeline.
---

# Upload Instance Solver

Use this skill for uploaded instance files such as `.json`, `.txt`, `.vrp`, `.tsp`, `.csv`, and `.tsv`.

## Workflow

1. Prefer the one-step CLI when you just need recognition plus solving:

```powershell
python -m instance_skill.solve_cli --input-file <path> --mode quick --pretty
```

Or for deep mode:

```powershell
python -m instance_skill.solve_cli --input-file <path> --mode thinking --pretty
```

2. If you need only normalization first, run:

```powershell
python -m instance_skill.cli --input-file <path> --pretty
```

3. The normalized outputs are saved under `normalized_instances/<timestamp>_<name>/` and include:
   - original uploaded file copy
   - `payload.json`
   - canonical instance file:
     - `*.tsp` for TSP
     - `*.vrp` for CVRP
     - `*.solomon.txt` for CVRPTW

## Supported detection

- JSON solver payload or JSON instance
- Solomon-style VRPTW text
- TSPLIB TSP
- TSPLIB CVRP
- CSV/TSV tables with headers

## Default solve policy

- `quick` mode:
  - `mode=fast`
  - `seed_trials=8`
  - `drl_samples=128`
  - no lookahead
  - lightweight local search by default: `two_opt`, `relocate`, `swap`
- `thinking` mode:
  - `mode=hybrid`
  - `seed_trials=1`
  - `drl_samples=128`
  - lookahead enabled by default
  - after lookahead, let the model choose follow-up operators and refinement rounds

## Objective extraction

When the user states penalties or priorities in natural language, convert them into `objective` and pass that through the solver chain.

Examples:

- "优先减少车辆数" -> `"primary": "vehicle_count"`
- "每多开一辆车罚 100" -> `"vehicle_fixed_cost": 100.0`
- "迟到要重罚" -> `"lateness_penalty": ...`
- "超时也要罚" -> `"overtime_penalty": ...`

These objective fields affect candidate selection, lookahead ranking, and local-search / destroy-repair comparisons.

## Analysis tools

Use these tools when the model needs to inspect the current solution before deciding the next operator:

```powershell
python -m tools.validate_solution --input payload.json --pretty
python -m tools.analyze_solution --input payload.json --pretty
```

`tools.analyze_solution` returns route-level diagnostics such as:

- per-route distance
- vehicle count
- generalized cost
- route load
- longest route
- for CVRPTW: total waiting time, max waiting time, and the longest-waiting customer

## Concurrency hints

These are reference values derived from earlier single-instance VRAM measurements with a `6 GB` budget:

- `fast`: `6144 / 91 ~= 67` instances
- `thinking`: `6144 / 123 ~= 49` instances

## Code references

- Recognition and canonical save: `instance_skill/api.py`
- One-step uploaded-file solver: `instance_skill/solve_cli.py`
- Main solve pipeline: `solver_skill/api.py`
- Solution analysis: `tools/analyze_solution.py`
- Feasibility validation: `tools/validate_solution.py`
