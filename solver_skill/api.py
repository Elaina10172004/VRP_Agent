from __future__ import annotations

from collections.abc import Callable

from local_search import improve_cvrp_solution, improve_cvrptw_solution, improve_tsp_solution
from local_search.cvrp import reduce_cvrp_vehicle_count
from local_search.cvrptw import reduce_cvrptw_vehicle_count
from solver_core import CVRPSolver, CVRPTWSolver, TSPSolver
from tools.validate_solution import validate_payload_solution_parts

from .lookahead import apply_cvrp_lookahead, apply_cvrptw_lookahead, apply_tsp_lookahead
from .runtime_defaults import (
    FAST_DEFAULT_INSTANCE_PARALLELISM,
    FAST_SINGLE_INSTANCE_VRAM_MB,
    GPU_BUDGET_MB,
    THINKING_DEFAULT_INSTANCE_PARALLELISM,
    THINKING_SINGLE_INSTANCE_VRAM_MB,
    build_runtime_defaults_payload,
    default_instance_parallelism,
)

DEFAULT_LOOKAHEAD_OPERATORS = ["two_opt", "relocate", "swap"]
DEFAULT_IMPROVEMENT_OPERATORS = ["or_opt", "two_opt_star", "cross_exchange", "relocate", "swap", "two_opt"]
DEFAULT_DESTROY_REPAIR_OPERATORS = ["route_elimination", *DEFAULT_IMPROVEMENT_OPERATORS]


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


def _sort_candidates(candidates: list[dict]) -> list[dict]:
    return sorted(candidates, key=lambda item: float(item["distance"]))


def _candidate_distances(candidates: list[dict] | None) -> list[float] | None:
    if candidates is None:
        return None
    return [float(item["distance"]) for item in candidates]


def _report_progress(progress: Callable[[str, str, str | None], None] | None, step_id: str, label: str, detail: str | None = None):
    if progress is not None:
        progress(step_id, label, detail)


def _seed_policy(seed_trials: int, candidate_count_per_seed: int) -> str:
    if seed_trials > 1 and candidate_count_per_seed > 1:
        return "best_of_seed_trials_and_topk_per_seed"
    if seed_trials > 1:
        return "best_of_seed_trials"
    if candidate_count_per_seed > 1:
        return "best_of_topk_per_seed"
    return "single_seed_single_candidate"


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


def construct_initial(
    problem_type: str,
    instance: dict,
    config: dict,
    progress: Callable[[str, str, str | None], None] | None = None,
) -> dict:
    seed_trials = max(1, int(config.get("seed_trials", 1)))
    candidate_count_per_seed = max(1, int(config.get("initial_candidate_count", config.get("lookahead_k", 1))))

    _report_progress(progress, "seed", "构造初始解", f"seed_trials={seed_trials}, topk={candidate_count_per_seed}")
    candidates = _sort_candidates(
        _build_seed_candidates(
            problem_type,
            instance,
            config,
            trial_count=seed_trials,
            candidate_count=candidate_count_per_seed,
        )
    )
    best_solution = candidates[0]
    validation = validate_solution(problem_type, instance, best_solution)
    _report_progress(progress, "seed_done", "初始解完成", f"best_distance={float(best_solution['distance']):.3f}")
    return {
        "candidates": candidates,
        "best_solution": best_solution,
        "validation": validation,
        "seed_trials": seed_trials,
        "candidate_count_per_seed": candidate_count_per_seed,
    }


def validate_solution(problem_type: str, instance: dict, solution: dict) -> dict:
    return validate_payload_solution_parts(problem_type, instance, solution)


def apply_lookahead_action(
    problem_type: str,
    instance: dict,
    solution: dict,
    config: dict,
    progress: Callable[[str, str, str | None], None] | None = None,
) -> dict:
    lookahead_config = {
        "depth": int(config.get("lookahead_depth", 2)),
        "beam_width": int(config.get("lookahead_beam_width", 4)),
        "per_operator_limit": int(config.get("lookahead_per_operator_limit", int(config.get("lookahead_beam_width", 4)))),
        "operators": config.get("lookahead_operators", config.get("operators", DEFAULT_LOOKAHEAD_OPERATORS)),
    }

    _report_progress(progress, "lookahead", "执行 Lookahead", f"depth={lookahead_config['depth']}, beam={lookahead_config['beam_width']}")
    if problem_type == "tsp":
        improved = apply_tsp_lookahead(instance, solution, lookahead_config)
    elif problem_type == "cvrp":
        improved = apply_cvrp_lookahead(instance, solution, lookahead_config)
    elif problem_type == "cvrptw":
        improved = apply_cvrptw_lookahead(instance, solution, lookahead_config)
    else:
        raise ValueError(f"Unsupported problem_type: {problem_type}")

    _report_progress(progress, "lookahead_done", "Lookahead 完成", f"distance={float(improved['distance']):.3f}")
    return improved


def improve_solution(
    problem_type: str,
    instance: dict,
    solution: dict,
    config: dict,
    progress: Callable[[str, str, str | None], None] | None = None,
) -> dict:
    ls_config = {
        "max_rounds": int(config.get("local_search_rounds", 50)),
        "operators": config.get("local_search_operators", config.get("operators", DEFAULT_IMPROVEMENT_OPERATORS)),
    }

    _report_progress(progress, "local_search", "执行改进搜索", f"rounds={ls_config['max_rounds']}")
    if problem_type == "tsp":
        improved = improve_tsp_solution(instance, solution, ls_config)
    elif problem_type == "cvrp":
        improved = improve_cvrp_solution(instance, solution, ls_config)
    elif problem_type == "cvrptw":
        improved = improve_cvrptw_solution(instance, solution, ls_config)
    else:
        raise ValueError(f"Unsupported problem_type: {problem_type}")

    _report_progress(progress, "local_search_done", "改进搜索完成", f"distance={float(improved['distance']):.3f}")
    return improved


def destroy_repair(
    problem_type: str,
    instance: dict,
    solution: dict,
    config: dict,
    progress: Callable[[str, str, str | None], None] | None = None,
) -> dict:
    dr_config = {
        "max_rounds": int(config.get("destroy_repair_rounds", config.get("local_search_rounds", 50))),
        "operators": config.get(
            "destroy_repair_operators",
            DEFAULT_DESTROY_REPAIR_OPERATORS,
        ),
    }

    _report_progress(progress, "lookahead", "执行破坏修复", f"operators={','.join(dr_config['operators'])}")
    if problem_type == "tsp":
        improved = improve_tsp_solution(instance, solution, dr_config)
    elif problem_type == "cvrp":
        improved = improve_cvrp_solution(instance, solution, dr_config)
    elif problem_type == "cvrptw":
        improved = improve_cvrptw_solution(instance, solution, dr_config)
    else:
        raise ValueError(f"Unsupported problem_type: {problem_type}")
    _report_progress(progress, "lookahead_done", "破坏修复完成", f"distance={float(improved['distance']):.3f}")
    return improved


def reduce_vehicles(
    problem_type: str,
    instance: dict,
    solution: dict,
    config: dict,
    progress: Callable[[str, str, str | None], None] | None = None,
) -> dict:
    if problem_type == "tsp":
        return solution

    _report_progress(progress, "lookahead", "执行删路线", f"rounds={int(config.get('vehicle_reduction_rounds', 10))}")
    reduction_config = {"max_rounds": int(config.get("vehicle_reduction_rounds", 10))}
    if problem_type == "cvrp":
        reduced = reduce_cvrp_vehicle_count(instance, solution, reduction_config)
    elif problem_type == "cvrptw":
        reduced = reduce_cvrptw_vehicle_count(instance, solution, reduction_config)
    else:
        raise ValueError(f"Unsupported problem_type: {problem_type}")
    _report_progress(progress, "lookahead_done", "删路线完成", f"vehicles={len(reduced.get('routes', []))}")
    return reduced


def compare_solutions(problem_type: str, instance: dict, solutions: list[dict]) -> dict:
    if not solutions:
        raise ValueError("At least one solution is required.")

    comparisons = []
    for solution in solutions:
        validation = validate_solution(problem_type, instance, solution)
        comparisons.append(
            {
                "solution": solution,
                "validation": validation,
            }
        )

    def ranking_key(item: dict) -> tuple:
        validation = item["validation"]
        feasible_rank = 0 if validation["feasible"] else 1
        return (
            feasible_rank,
            int(validation["vehicle_count"]),
            float(validation["distance"]),
            len(validation["violations"]),
        )

    best = min(comparisons, key=ranking_key)
    return {
        "best_solution": best["solution"],
        "best_validation": best["validation"],
        "comparisons": comparisons,
    }


def _default_tool_plan(config: dict, fast_mode: bool) -> list[str]:
    if fast_mode:
        return ["construct_initial", "validate_solution", "compare_solutions"]

    plan = ["construct_initial", "validate_solution"]
    if bool(config.get("enable_vehicle_reduction", False)):
        plan.extend(["reduce_vehicles", "validate_solution"])
    if bool(config.get("enable_lookahead", True)):
        plan.extend(["apply_lookahead", "validate_solution"])
    if bool(config.get("enable_destroy_repair", False)):
        plan.extend(["destroy_repair", "validate_solution"])
    if bool(config.get("enable_local_search", False)):
        plan.extend(["improve_solution", "validate_solution"])
    plan.append("compare_solutions")
    return plan


def solve_payload(payload: dict, progress: Callable[[str, str, str | None], None] | None = None) -> dict:
    problem_type = str(payload.get("problem_type", "")).strip().lower()
    instance = payload.get("instance")
    config = dict(payload.get("config", {}))
    if not isinstance(instance, dict):
        raise ValueError("Payload must contain an 'instance' object.")

    fast_mode = str(config.get("mode", "hybrid")).strip().lower() == "fast"
    mode_name = "fast" if fast_mode else "hybrid"
    config.setdefault("seed_trials", 8 if fast_mode else 1)
    config.setdefault("enable_lookahead", not fast_mode)
    config.setdefault("enable_local_search", False)

    tool_plan = list(config.get("tool_plan", _default_tool_plan(config, fast_mode)))
    trace: list[dict] = []

    construction = construct_initial(problem_type, instance, config, progress=progress)
    seed_candidates = construction["candidates"]
    seed_solution = construction["best_solution"]
    current_solution = seed_solution
    current_validation = construction["validation"]
    trace.append({"action": "construct_initial", "validation": current_validation})

    lookahead_solution = None
    local_search_solution = None
    comparison_candidates = [seed_solution]

    for action in tool_plan:
        if action == "construct_initial":
            continue
        if action == "validate_solution":
            current_validation = validate_solution(problem_type, instance, current_solution)
            trace.append({"action": action, "validation": current_validation})
            continue
        if action == "reduce_vehicles":
            reduced = reduce_vehicles(problem_type, instance, current_solution, config, progress=progress)
            current_solution = compare_solutions(problem_type, instance, [current_solution, reduced])["best_solution"]
            comparison_candidates.append(reduced)
            trace.append({"action": action, "distance": float(current_solution["distance"])})
            continue
        if action == "apply_lookahead":
            looked_ahead = apply_lookahead_action(problem_type, instance, current_solution, config, progress=progress)
            lookahead_solution = looked_ahead
            current_solution = compare_solutions(problem_type, instance, [current_solution, looked_ahead])["best_solution"]
            comparison_candidates.append(looked_ahead)
            trace.append({"action": action, "distance": float(looked_ahead["distance"])})
            continue
        if action == "destroy_repair":
            repaired = destroy_repair(problem_type, instance, current_solution, config, progress=progress)
            current_solution = compare_solutions(problem_type, instance, [current_solution, repaired])["best_solution"]
            comparison_candidates.append(repaired)
            trace.append({"action": action, "distance": float(repaired["distance"])})
            continue
        if action == "improve_solution":
            improved = improve_solution(problem_type, instance, current_solution, config, progress=progress)
            local_search_solution = improved
            current_solution = compare_solutions(problem_type, instance, [current_solution, improved])["best_solution"]
            comparison_candidates.append(improved)
            trace.append({"action": action, "distance": float(improved["distance"])})
            continue
        if action == "compare_solutions":
            compared = compare_solutions(problem_type, instance, comparison_candidates)
            current_solution = compared["best_solution"]
            current_validation = compared["best_validation"]
            trace.append({"action": action, "distance": float(current_solution["distance"])})
            continue

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
            "seed_policy": _seed_policy(construction["seed_trials"], construction["candidate_count_per_seed"]),
            "seed_trials": construction["seed_trials"],
            "candidate_count_per_seed": construction["candidate_count_per_seed"],
            "seed_candidate_distances": _candidate_distances(seed_candidates),
            "enable_lookahead": bool(config.get("enable_lookahead", False)) and not fast_mode,
            "lookahead_depth": int(config.get("lookahead_depth", 2)),
            "lookahead_beam_width": int(config.get("lookahead_beam_width", 4)),
            "lookahead_candidate_distances": _candidate_distances([lookahead_solution] if lookahead_solution else None),
            "enable_local_search": bool(config.get("enable_local_search", False)) and not fast_mode,
            "local_search_rounds": int(config.get("local_search_rounds", 50)),
            "local_search_candidate_distances": _candidate_distances([local_search_solution] if local_search_solution else None),
            "gpu_budget_mb": GPU_BUDGET_MB,
            "fast_single_instance_vram_mb": FAST_SINGLE_INSTANCE_VRAM_MB,
            "thinking_single_instance_vram_mb": THINKING_SINGLE_INSTANCE_VRAM_MB,
            "default_instance_parallelism": default_instance_parallelism(mode_name),
            "default_instance_parallelism_by_mode": {
                "fast": FAST_DEFAULT_INSTANCE_PARALLELISM,
                "hybrid": THINKING_DEFAULT_INSTANCE_PARALLELISM,
            },
            "tool_plan": tool_plan,
            "tool_trace": trace,
            "final_validation": current_validation,
            "runtime_defaults": build_runtime_defaults_payload(),
        },
    }
