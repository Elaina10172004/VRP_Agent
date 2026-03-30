# VRP Agent

LLM-driven desktop agent for `TSP`, `CVRP`, and `CVRPTW`.

The repository combines:

- a minimal DRL construction layer extracted from `PolyNet`
- a local-search toolbox for route improvement
- an instance ingestion layer for uploaded benchmark files
- an Electron desktop app with chat, history, visualization, batch solving, and export

## Repository layout

- `solver_core/`: minimal DRL inference code and checkpoints for TSP/CVRP/CVRPTW
- `solver_skill/`: orchestration layer that exposes `construct_initial`, `validate_solution`, `reduce_vehicles`, `apply_lookahead`, `destroy_repair`, `improve_solution`, `compare_solutions`
- `local_search/`: local-search neighborhoods and vehicle-reduction operators
- `instance_skill/`: detect uploaded instances and normalize them into solver payloads
- `tools/validate_solution.py`: standalone feasibility checker and score summary
- `optichat/`: Electron + React desktop frontend
- `benchmarks/`: local benchmark and VRAM measurement scripts
- `config/runtime_defaults.json`: single source of truth for GPU-budget runtime defaults

## Desktop app

Requirements:

- Windows
- Node.js 18+ with `npm`
- Python 3.10+
- PyTorch environment that can load the included checkpoints

One-click start from the repository root:

```bat
start.bat
```

Development mode:

```bat
start.bat --dev
```

Dependency check only:

```bat
start.bat --check
```

## Python CLI

Solve a normalized payload:

```powershell
python -m solver_skill.cli --input payload.json --output result.json --pretty
```

Validate any candidate solution:

```powershell
python -m tools.validate_solution --input validate_payload.json --pretty
```

Improve an existing solution with local search:

```powershell
python -m local_search.cli --input improve_payload.json --output improved.json --pretty
```

Detect and solve an uploaded benchmark file directly:

```powershell
python -m instance_skill.solve_cli --input-file .\solomon_data\C101.txt --mode thinking --pretty
```

## Solver pipeline

`solver_skill` is now an orchestration layer instead of a fixed script. It exposes fine-grained tools that can be arranged by an LLM strategy or by static config:

- `construct_initial`
- `validate_solution`
- `reduce_vehicles`
- `apply_lookahead`
- `destroy_repair`
- `improve_solution`
- `compare_solutions`

`fast` mode keeps the short path:

- `construct_initial -> validate_solution -> compare_solutions`

`thinking` mode can use richer plans, including route elimination and stronger local-search neighborhoods.

## Local-search operators

Currently available:

- `TSP`: `two_opt`, `relocate`, `swap`
- `CVRP`: `two_opt`, `relocate`, `swap`, `or_opt`, `two_opt_star`, `cross_exchange`, `route_elimination`
- `CVRPTW`: `two_opt`, `relocate`, `swap`, `or_opt`, `two_opt_star`, `cross_exchange`, `route_elimination`

## Checkpoints

The repository keeps only the minimum weights required for inference:

- `TSP/models/PolyNet/...`
- `CVRP/models/PolyNet/...`
- `PolyNet/pt/checkpoint-1500.pt`

## Benchmarks

VRAM and single-instance measurements:

```powershell
python .\benchmarks\gpu_capacity.py --help
```

The runtime parallelism defaults used by both Python and Electron are derived from:

- `config/runtime_defaults.json`

