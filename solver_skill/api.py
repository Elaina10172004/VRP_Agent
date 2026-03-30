from __future__ import annotations

from collections.abc import Callable

from local_search import improve_cvrp_solution, improve_cvrptw_solution, improve_tsp_solution
from local_search.common import (
    ObjectiveScore,
    build_distance_matrix,
    build_vrp_distance_matrix,
    ensure_points,
    normalize_demands,
    normalize_objective_spec,
    normalize_routes_payload,
    normalize_service_times,
    normalize_time_windows,
    score_cvrp_routes,
    score_cvrptw_routes,
    score_tsp_tour,
)
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


def _candidate_distances(candidates: list[dict] | None) -> list[float] | None:
    if candidates is None:
        return None
    return [float(item["distance"]) for item in candidates]


def _candidate_scores(scored_items: list[dict] | None) -> list[float] | None:
    if scored_items is None:
        return None
    return [float(item["score"].generalized_cost) for item in scored_items]


def _report_progress(
    progress: Callable[[str, str, str | None], None] | None,
    step_id: str,
    label: str,
    detail: str | None = None,
):
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


def _validation_to_score(validation: dict, objective: dict) -> ObjectiveScore:
    objective_spec = normalize_objective_spec(objective)
    feasible = bool(validation["feasible"])
    vehicle_count = int(validation["vehicle_count"])
    distance = float(validation["distance"])
    generalized_cost = (
        objective_spec.vehicle_fixed_cost * vehicle_count
        + objective_spec.distance_weight * distance
        + objective_spec.duration_weight * distance
    )
    if not feasible:
        generalized_cost = float("inf")
    return ObjectiveScore(
        feasible=feasible,
        vehicle_count=vehicle_count,
        distance=distance,
        duration=distance,
        generalized_cost=generalized_cost,
        violations=len(validation.get("violations", [])),
        objective=objective_spec,
    )


def _score_solution(problem_type: str, instance: dict, solution: dict, objective: dict) -> ObjectiveScore:
    objective_spec = normalize_objective_spec(objective)

    if problem_type == "tsp":
        points = ensure_points(instance["points"])
        distance_matrix = build_distance_matrix(points)
        if isinstance(solution.get("tour"), list):
            tour = [int(node) for node in solution["tour"]]
        elif isinstance(solution.get("closed_tour"), list):
            closed_tour = [int(node) for node in solution["closed_tour"]]
            tour = closed_tour[:-1] if len(closed_tour) >= 2 and closed_tour[0] == closed_tour[-1] else closed_tour
        else:
            raise ValueError("TSP solution must contain 'tour' or 'closed_tour'.")
        return score_tsp_tour(tour, distance_matrix, objective_spec)

    if problem_type == "cvrp":
        distance_matrix = build_vrp_distance_matrix(instance["depot_xy"], instance["node_xy"])
        routes = normalize_routes_payload(solution)
        return score_cvrp_routes(routes, distance_matrix, objective_spec)

    if problem_type == "cvrptw":
        distance_matrix = build_vrp_distance_matrix(instance["depot_xy"], instance["node_xy"])
        routes = normalize_routes_payload(solution)
        demands = normalize_demands(instance["node_demand"])
        capacity = float(instance["capacity"])
        time_windows = normalize_time_windows(instance["node_tw"])
        service_times = normalize_service_times(instance["service_time"], len(demands))
        return score_cvrptw_routes(routes, distance_matrix, demands, capacity, time_windows, service_times, objective_spec)

    raise ValueError(f"Unsupported problem_type: {problem_type}")


def _score_with_validation(problem_type: str, instance: dict, solution: dict, validation: dict, objective: dict) -> ObjectiveScore:
    try:
        score = _score_solution(problem_type, instance, solution, objective)
    except Exception:
        return _validation_to_score(validation, objective)
    if score.feasible != bool(validation["feasible"]):
        return _validation_to_score(validation, objective)
    return score


def _sort_candidates(problem_type: str, instance: dict, objective: dict, candidates: list[dict]) -> list[dict]:
    scored_items = []
    for candidate in candidates:
        validation = validate_solution(problem_type, instance, candidate)
        score = _score_with_validation(problem_type, instance, candidate, validation, objective)
        scored_items.append(
            {
                "solution": candidate,
                "validation": validation,
                "score": score,
            }
        )
    scored_items.sort(key=lambda item: item["score"].ranking_key())
    return scored_items


def construct_initial(
    problem_type: str,
    instance: dict,
    config: dict,
    progress: Callable[[str, str, str | None], None] | None = None,
) -> dict:
    objective = normalize_objective_spec(config.get("objective"))
    seed_trials = max(1, int(config.get("seed_trials", 1)))
    candidate_count_per_seed = max(1, int(config.get("initial_candidate_count", config.get("lookahead_k", 1))))

    _report_progress(progress, "seed", "Construct initial solution", f"seed_trials={seed_trials}, topk={candidate_count_per_seed}")
    scored_candidates = _sort_candidates(
        problem_type,
        instance,
        objective,
        _build_seed_candidates(
            problem_type,
            instance,
            config,
            trial_count=seed_trials,
            candidate_count=candidate_count_per_seed,
        ),
    )
    best_item = scored_candidates[0]
    _report_progress(
        progress,
        "seed_done",
        "Initial solution ready",
        f"score={best_item['score'].generalized_cost:.3f}, distance={best_item['score'].distance:.3f}",
    )
    return {
        "candidates": [item["solution"] for item in scored_candidates],
        "scored_candidates": scored_candidates,
        "best_solution": best_item["solution"],
        "validation": best_item["validation"],
        "best_score": best_item["score"],
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
        "objective": config.get("objective"),
    }

    _report_progress(
        progress,
        "lookahead",
        "Run lookahead",
        f"depth={lookahead_config['depth']}, beam={lookahead_config['beam_width']}",
    )
    if problem_type == "tsp":
        improved = apply_tsp_lookahead(instance, solution, lookahead_config)
    elif problem_type == "cvrp":
        improved = apply_cvrp_lookahead(instance, solution, lookahead_config)
    elif problem_type == "cvrptw":
        improved = apply_cvrptw_lookahead(instance, solution, lookahead_config)
    else:
        raise ValueError(f"Unsupported problem_type: {problem_type}")

    score = _score_solution(problem_type, instance, improved, lookahead_config["objective"])
    _report_progress(progress, "lookahead_done", "Lookahead finished", f"score={score.generalized_cost:.3f}")
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
        "objective": config.get("objective"),
    }

    _report_progress(progress, "local_search", "Run local search", f"rounds={ls_config['max_rounds']}")
    if problem_type == "tsp":
        improved = improve_tsp_solution(instance, solution, ls_config)
    elif problem_type == "cvrp":
        improved = improve_cvrp_solution(instance, solution, ls_config)
    elif problem_type == "cvrptw":
        improved = improve_cvrptw_solution(instance, solution, ls_config)
    else:
        raise ValueError(f"Unsupported problem_type: {problem_type}")

    score = _score_solution(problem_type, instance, improved, ls_config["objective"])
    _report_progress(progress, "local_search_done", "Local search finished", f"score={score.generalized_cost:.3f}")
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
        "operators": config.get("destroy_repair_operators", DEFAULT_DESTROY_REPAIR_OPERATORS),
        "objective": config.get("objective"),
    }

    _report_progress(
        progress,
        "destroy_repair",
        "Run destroy-repair",
        f"operators={','.join(dr_config['operators'])}",
    )
    if problem_type == "tsp":
        improved = improve_tsp_solution(instance, solution, dr_config)
    elif problem_type == "cvrp":
        improved = improve_cvrp_solution(instance, solution, dr_config)
    elif problem_type == "cvrptw":
        improved = improve_cvrptw_solution(instance, solution, dr_config)
    else:
        raise ValueError(f"Unsupported problem_type: {problem_type}")
    score = _score_solution(problem_type, instance, improved, dr_config["objective"])
    _report_progress(progress, "destroy_repair_done", "Destroy-repair finished", f"score={score.generalized_cost:.3f}")
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

    reduction_config = {
        "max_rounds": int(config.get("vehicle_reduction_rounds", 10)),
        "objective": config.get("objective"),
    }
    _report_progress(progress, "reduce_vehicles", "Run route elimination", f"rounds={reduction_config['max_rounds']}")
    if problem_type == "cvrp":
        reduced = reduce_cvrp_vehicle_count(instance, solution, reduction_config)
    elif problem_type == "cvrptw":
        reduced = reduce_cvrptw_vehicle_count(instance, solution, reduction_config)
    else:
        raise ValueError(f"Unsupported problem_type: {problem_type}")
    score = _score_solution(problem_type, instance, reduced, reduction_config["objective"])
    _report_progress(
        progress,
        "reduce_vehicles_done",
        "Route elimination finished",
        f"vehicles={len(reduced.get('routes', []))}, score={score.generalized_cost:.3f}",
    )
    return reduced


def compare_solutions(problem_type: str, instance: dict, solutions: list[dict], config: dict | None = None) -> dict:
    if not solutions:
        raise ValueError("At least one solution is required.")

    objective = normalize_objective_spec((config or {}).get("objective"))
    comparisons = []
    for solution in solutions:
        validation = validate_solution(problem_type, instance, solution)
        score = _score_with_validation(problem_type, instance, solution, validation, objective)
        comparisons.append(
            {
                "solution": solution,
                "validation": validation,
                "score": score,
            }
        )

    best = min(comparisons, key=lambda item: item["score"].ranking_key())
    return {
        "best_solution": best["solution"],
        "best_validation": best["validation"],
        "best_score": best["score"],
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
    objective = normalize_objective_spec(config.get("objective"))
    config["objective"] = objective.__dict__
    config.setdefault("seed_trials", 8 if fast_mode else 1)
    config.setdefault("enable_lookahead", not fast_mode)
    config.setdefault("enable_local_search", False)

    tool_plan = list(config.get("tool_plan", _default_tool_plan(config, fast_mode)))
    trace: list[dict] = []

    construction = construct_initial(problem_type, instance, config, progress=progress)
    seed_candidates = construction["candidates"]
    scored_seed_candidates = construction["scored_candidates"]
    seed_solution = construction["best_solution"]
    current_solution = seed_solution
    current_validation = construction["validation"]
    current_score = construction["best_score"]
    trace.append(
        {
            "action": "construct_initial",
            "validation": current_validation,
            "score": current_score.generalized_cost,
            "vehicle_count": current_score.vehicle_count,
            "distance": current_score.distance,
        }
    )

    lookahead_solution = None
    local_search_solution = None
    comparison_candidates = [seed_solution]

    for action in tool_plan:
        if action == "construct_initial":
            continue
        if action == "validate_solution":
            current_validation = validate_solution(problem_type, instance, current_solution)
            current_score = _score_with_validation(problem_type, instance, current_solution, current_validation, objective)
            trace.append(
                {
                    "action": action,
                    "validation": current_validation,
                    "score": current_score.generalized_cost,
                    "vehicle_count": current_score.vehicle_count,
                    "distance": current_score.distance,
                }
            )
            continue
        if action == "reduce_vehicles":
            reduced = reduce_vehicles(problem_type, instance, current_solution, config, progress=progress)
            compared = compare_solutions(problem_type, instance, [current_solution, reduced], config)
            current_solution = compared["best_solution"]
            current_validation = compared["best_validation"]
            current_score = compared["best_score"]
            comparison_candidates.append(reduced)
            trace.append(
                {
                    "action": action,
                    "score": current_score.generalized_cost,
                    "vehicle_count": current_score.vehicle_count,
                    "distance": current_score.distance,
                }
            )
            continue
        if action == "apply_lookahead":
            looked_ahead = apply_lookahead_action(problem_type, instance, current_solution, config, progress=progress)
            lookahead_solution = looked_ahead
            compared = compare_solutions(problem_type, instance, [current_solution, looked_ahead], config)
            current_solution = compared["best_solution"]
            current_validation = compared["best_validation"]
            current_score = compared["best_score"]
            comparison_candidates.append(looked_ahead)
            trace.append(
                {
                    "action": action,
                    "score": current_score.generalized_cost,
                    "vehicle_count": current_score.vehicle_count,
                    "distance": current_score.distance,
                }
            )
            continue
        if action == "destroy_repair":
            repaired = destroy_repair(problem_type, instance, current_solution, config, progress=progress)
            compared = compare_solutions(problem_type, instance, [current_solution, repaired], config)
            current_solution = compared["best_solution"]
            current_validation = compared["best_validation"]
            current_score = compared["best_score"]
            comparison_candidates.append(repaired)
            trace.append(
                {
                    "action": action,
                    "score": current_score.generalized_cost,
                    "vehicle_count": current_score.vehicle_count,
                    "distance": current_score.distance,
                }
            )
            continue
        if action == "improve_solution":
            improved = improve_solution(problem_type, instance, current_solution, config, progress=progress)
            local_search_solution = improved
            compared = compare_solutions(problem_type, instance, [current_solution, improved], config)
            current_solution = compared["best_solution"]
            current_validation = compared["best_validation"]
            current_score = compared["best_score"]
            comparison_candidates.append(improved)
            trace.append(
                {
                    "action": action,
                    "score": current_score.generalized_cost,
                    "vehicle_count": current_score.vehicle_count,
                    "distance": current_score.distance,
                }
            )
            continue
        if action == "compare_solutions":
            compared = compare_solutions(problem_type, instance, comparison_candidates, config)
            current_solution = compared["best_solution"]
            current_validation = compared["best_validation"]
            current_score = compared["best_score"]
            trace.append(
                {
                    "action": action,
                    "score": current_score.generalized_cost,
                    "vehicle_count": current_score.vehicle_count,
                    "distance": current_score.distance,
                }
            )
            continue

    _report_progress(progress, "finalize", "Finalize solution", f"score={current_score.generalized_cost:.3f}")

    return {
        "problem_type": problem_type,
        "seed_solution": seed_solution,
        "lookahead_solution": lookahead_solution,
        "local_search_solution": local_search_solution,
        "final_solution": current_solution,
        "meta": {
            "mode": mode_name,
            "objective": objective.__dict__,
            "drl_samples": int(config.get("drl_samples", 128)),
            "seed_policy": _seed_policy(construction["seed_trials"], construction["candidate_count_per_seed"]),
            "seed_trials": construction["seed_trials"],
            "candidate_count_per_seed": construction["candidate_count_per_seed"],
            "seed_candidate_distances": _candidate_distances(seed_candidates),
            "seed_candidate_scores": _candidate_scores(scored_seed_candidates),
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
            "final_score": {
                "generalized_cost": current_score.generalized_cost,
                "vehicle_count": current_score.vehicle_count,
                "distance": current_score.distance,
                "duration": current_score.duration,
                "ranking_key": list(current_score.ranking_key()),
            },
            "runtime_defaults": build_runtime_defaults_payload(),
        },
    }
