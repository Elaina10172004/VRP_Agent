from __future__ import annotations

from collections.abc import Callable

from local_search import improve_cvrp_solution, improve_cvrptw_solution, improve_tsp_solution
from solver_core import CVRPSolver, CVRPTWSolver, TSPSolver

from .lookahead import apply_cvrp_lookahead, apply_cvrptw_lookahead, apply_tsp_lookahead
from .runtime_defaults import (
    FAST_DEFAULT_INSTANCE_PARALLELISM,
    FAST_SINGLE_INSTANCE_VRAM_MB,
    GPU_BUDGET_MB,
    THINKING_DEFAULT_INSTANCE_PARALLELISM,
    THINKING_SINGLE_INSTANCE_VRAM_MB,
    default_instance_parallelism,
)


def _ensure_list(result: dict | list[dict] | list[list[dict]]) -> list[dict]:
    if isinstance(result, dict):
        return [result]
    if result and isinstance(result[0], list):
        flattened: list[dict] = []
        for group in result:
            flattened.extend(group)
        return flattened
    return list(result)


def _repeat_value(value, count: int):
    return [value for _ in range(count)]


def _build_seed_candidates(
    problem_type: str,
    instance: dict,
    config: dict,
    trial_count: int = 1,
    candidate_count: int = 1,
) -> list[dict]:
    drl_samples = int(config.get("drl_samples", 128))
    greedy = bool(config.get("greedy", False))
    device = config.get("device")
    batch_size = max(1, int(trial_count))
    top_k = max(1, int(candidate_count))

    if problem_type == "tsp":
        solver = TSPSolver(device=device)
        result = solver.solve(
            _repeat_value(instance["points"], batch_size),
            num_samples=drl_samples,
            greedy=greedy,
            return_topk=top_k if top_k > 1 else None,
        )
        return _ensure_list(result)

    if problem_type == "cvrp":
        solver = CVRPSolver(device=device)
        result = solver.solve(
            _repeat_value(instance["depot_xy"], batch_size),
            _repeat_value(instance["node_xy"], batch_size),
            _repeat_value(instance["node_demand"], batch_size),
            _repeat_value(instance["capacity"], batch_size),
            num_samples=drl_samples,
            greedy=greedy,
            return_topk=top_k if top_k > 1 else None,
        )
        return _ensure_list(result)

    if problem_type == "cvrptw":
        solver = CVRPTWSolver(device=device)
        grid_scale = instance.get("grid_scale")
        result = solver.solve(
            _repeat_value(instance["depot_xy"], batch_size),
            _repeat_value(instance["node_xy"], batch_size),
            _repeat_value(instance["node_demand"], batch_size),
            _repeat_value(instance["capacity"], batch_size),
            _repeat_value(instance["node_tw"], batch_size),
            _repeat_value(instance["service_time"], batch_size),
            num_samples=drl_samples,
            greedy=greedy,
            grid_scale=None if grid_scale is None else _repeat_value(grid_scale, batch_size),
            return_topk=top_k if top_k > 1 else None,
        )
        return _ensure_list(result)

    raise ValueError(f"Unsupported problem_type: {problem_type}")


def _apply_lookahead(problem_type: str, instance: dict, solution: dict, config: dict) -> dict:
    lookahead_config = {
        "depth": int(config.get("lookahead_depth", 2)),
        "beam_width": int(config.get("lookahead_beam_width", 4)),
        "per_operator_limit": int(config.get("lookahead_per_operator_limit", int(config.get("lookahead_beam_width", 4)))),
        "operators": config.get("operators", ["two_opt", "relocate", "swap"]),
    }

    if problem_type == "tsp":
        return apply_tsp_lookahead(instance, solution, lookahead_config)
    if problem_type == "cvrp":
        return apply_cvrp_lookahead(instance, solution, lookahead_config)
    if problem_type == "cvrptw":
        return apply_cvrptw_lookahead(instance, solution, lookahead_config)
    raise ValueError(f"Unsupported problem_type: {problem_type}")


def _apply_local_search(problem_type: str, instance: dict, solution: dict, config: dict) -> dict:
    ls_config = {
        "max_rounds": int(config.get("local_search_rounds", 50)),
        "operators": config.get("operators", ["two_opt", "relocate", "swap"]),
    }

    if problem_type == "tsp":
        return improve_tsp_solution(instance, solution, ls_config)
    if problem_type == "cvrp":
        return improve_cvrp_solution(instance, solution, ls_config)
    if problem_type == "cvrptw":
        return improve_cvrptw_solution(instance, solution, ls_config)
    raise ValueError(f"Unsupported problem_type: {problem_type}")


def _sort_candidates(candidates: list[dict]) -> list[dict]:
    return sorted(candidates, key=lambda item: float(item["distance"]))


def _candidate_distances(candidates: list[dict] | None) -> list[float] | None:
    if candidates is None:
        return None
    return [float(item["distance"]) for item in candidates]


def _seed_policy(seed_trials: int, candidate_count_per_seed: int) -> str:
    if seed_trials > 1 and candidate_count_per_seed > 1:
        return "best_of_seed_trials_and_topk_per_seed"
    if seed_trials > 1:
        return "best_of_seed_trials"
    if candidate_count_per_seed > 1:
        return "best_of_topk_per_seed"
    return "single_seed_single_candidate"


def _report_progress(progress: Callable[[str, str, str | None], None] | None, step_id: str, label: str, detail: str | None = None):
    if progress is not None:
        progress(step_id, label, detail)


def solve_payload(payload: dict, progress: Callable[[str, str, str | None], None] | None = None) -> dict:
    problem_type = str(payload.get("problem_type", "")).strip().lower()
    instance = payload.get("instance")
    config = dict(payload.get("config", {}))
    if not isinstance(instance, dict):
        raise ValueError("Payload must contain an 'instance' object.")

    fast_mode = str(config.get("mode", "hybrid")).strip().lower() == "fast"
    mode_name = "fast" if fast_mode else "hybrid"
    default_seed_trials = 8 if fast_mode else 1
    enable_lookahead = bool(config.get("enable_lookahead", not fast_mode))
    enable_local_search = bool(config.get("enable_local_search", False))
    seed_trials = max(1, int(config.get("seed_trials", default_seed_trials)))
    candidate_count_per_seed = max(1, int(config.get("lookahead_k", 1)))

    _report_progress(progress, "seed", "运行 DRL 构造初始解", f"seed_trials={seed_trials}, topk={candidate_count_per_seed}")
    seed_candidates = _sort_candidates(
        _build_seed_candidates(
            problem_type,
            instance,
            config,
            trial_count=seed_trials,
            candidate_count=candidate_count_per_seed,
        )
    )
    seed_solution = seed_candidates[0]
    current_solution = seed_solution
    _report_progress(progress, "seed_done", "DRL 初始解完成", f"best_distance={float(seed_solution['distance']):.3f}")

    lookahead_candidates = None
    lookahead_solution = None
    if not fast_mode and enable_lookahead:
        _report_progress(
            progress,
            "lookahead",
            "执行 Lookahead",
            f"depth={int(config.get('lookahead_depth', 2))}, beam={int(config.get('lookahead_beam_width', 4))}",
        )
        lookahead_candidates = _sort_candidates(
            [_apply_lookahead(problem_type, instance, candidate_solution, config) for candidate_solution in seed_candidates]
        )
        lookahead_solution = lookahead_candidates[0]
        current_solution = lookahead_solution
        _report_progress(progress, "lookahead_done", "Lookahead 完成", f"best_distance={float(lookahead_solution['distance']):.3f}")

    local_search_candidates = None
    local_search_solution = None
    if not fast_mode and enable_local_search:
        _report_progress(
            progress,
            "local_search",
            "执行局部搜索",
            f"rounds={int(config.get('local_search_rounds', 50))}",
        )
        base_candidates = lookahead_candidates if lookahead_candidates is not None else seed_candidates
        local_search_candidates = _sort_candidates(
            [_apply_local_search(problem_type, instance, candidate_solution, config) for candidate_solution in base_candidates]
        )
        local_search_solution = local_search_candidates[0]
        current_solution = local_search_solution
        _report_progress(progress, "local_search_done", "局部搜索完成", f"best_distance={float(local_search_solution['distance']):.3f}")

    _report_progress(progress, "finalize", "整理最终结果", f"final_distance={float(current_solution['distance']):.3f}")

    return {
        "problem_type": problem_type,
        "seed_solution": seed_solution,
        "lookahead_solution": lookahead_solution,
        "local_search_solution": local_search_solution,
        "final_solution": current_solution,
        "meta": {
            "mode": mode_name,
            "drl_samples": int(config.get("drl_samples", 128)),
            "seed_policy": _seed_policy(seed_trials, candidate_count_per_seed),
            "seed_trials": seed_trials,
            "candidate_count_per_seed": candidate_count_per_seed,
            "seed_candidate_distances": _candidate_distances(seed_candidates),
            "enable_lookahead": enable_lookahead and not fast_mode,
            "lookahead_depth": int(config.get("lookahead_depth", 2)),
            "lookahead_beam_width": int(config.get("lookahead_beam_width", 4)),
            "lookahead_candidate_distances": _candidate_distances(lookahead_candidates),
            "enable_local_search": enable_local_search and not fast_mode,
            "local_search_rounds": int(config.get("local_search_rounds", 50)),
            "local_search_candidate_distances": _candidate_distances(local_search_candidates),
            "gpu_budget_mb": GPU_BUDGET_MB,
            "fast_single_instance_vram_mb": FAST_SINGLE_INSTANCE_VRAM_MB,
            "thinking_single_instance_vram_mb": THINKING_SINGLE_INSTANCE_VRAM_MB,
            "default_instance_parallelism": default_instance_parallelism(mode_name),
            "default_instance_parallelism_by_mode": {
                "fast": FAST_DEFAULT_INSTANCE_PARALLELISM,
                "hybrid": THINKING_DEFAULT_INSTANCE_PARALLELISM,
            },
        },
    }
