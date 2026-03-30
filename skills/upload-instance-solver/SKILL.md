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
  - no local search by default
- `thinking` mode:
  - `mode=hybrid`
  - `seed_trials=1`
  - `drl_samples=128`
  - lookahead enabled by default
  - local search disabled by default

## Concurrency hints

These are reference values derived from earlier single-instance VRAM measurements with a `6 GB` budget:

- `fast`: `6144 / 91 ~= 67` instances
- `thinking`: `6144 / 123 ~= 49` instances

## Code references

- Recognition and canonical save: `instance_skill/api.py`
- One-step uploaded-file solver: `instance_skill/solve_cli.py`
- Main solve pipeline: `solver_skill/api.py`
