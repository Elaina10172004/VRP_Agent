from __future__ import annotations

from contextlib import contextmanager

import torch

from local_search.common import normalize_demands, normalize_objective_spec, normalize_service_times, normalize_time_windows, score_cvrptw_routes

from .common import maybe_unbatch, sample_binary_z, split_depot_sequence
from .cvrptw import CVRPTWEnv, CVRPTWSolver


@contextmanager
def _temporary_decode_state(model, batch_indices: torch.Tensor, rollout_indices: torch.Tensor):
    original = {
        "encoded_nodes": model.encoded_nodes,
        "k": model.decoder.k,
        "v": model.decoder.v,
        "single_head_key": model.decoder.single_head_key,
        "z": model.decoder.z,
    }

    branch_encoded_nodes = original["encoded_nodes"].index_select(0, batch_indices).contiguous()
    branch_z = original["z"][batch_indices, rollout_indices].unsqueeze(1).contiguous()
    model.encoded_nodes = branch_encoded_nodes
    model.decoder.set_kv(branch_encoded_nodes, branch_z)
    try:
        yield
    finally:
        model.encoded_nodes = original["encoded_nodes"]
        model.decoder.k = original["k"]
        model.decoder.v = original["v"]
        model.decoder.single_head_key = original["single_head_key"]
        model.decoder.z = original["z"]


def _clone_branch_env(parent_env: CVRPTWEnv, batch_indices: torch.Tensor, rollout_indices: torch.Tensor) -> CVRPTWEnv:
    branch_count = int(batch_indices.numel())
    branch_env = CVRPTWEnv()
    branch_env.problem_size = parent_env.problem_size
    branch_env.batch_size = branch_count
    branch_env.rollout_size = 1
    branch_env.grid_scale = parent_env.grid_scale.index_select(0, batch_indices).clone()
    branch_env.cost_depot_node_xy = parent_env.cost_depot_node_xy.index_select(0, batch_indices).clone()
    branch_env.depot_node_demand = parent_env.depot_node_demand.index_select(0, batch_indices).clone()
    branch_env.depot_node_tw = parent_env.depot_node_tw.index_select(0, batch_indices).clone()
    branch_env.depot_node_service = parent_env.depot_node_service.index_select(0, batch_indices).clone()
    branch_env.distance_matrix = parent_env.distance_matrix.index_select(0, batch_indices).clone()

    device = branch_env.cost_depot_node_xy.device
    branch_env.BATCH_IDX = torch.arange(branch_count, device=device)[:, None]
    branch_env.ROLLOUT_IDX = torch.zeros((branch_count, 1), dtype=torch.long, device=device)

    branch_env.selected_count = int(parent_env.selected_count)
    branch_env.current_node = None
    if parent_env.current_node is not None:
        branch_env.current_node = parent_env.current_node[batch_indices, rollout_indices].unsqueeze(1).clone()
    branch_env.selected_node_list = parent_env.selected_node_list[batch_indices, rollout_indices].unsqueeze(1).clone()
    branch_env.at_the_depot = parent_env.at_the_depot[batch_indices, rollout_indices].unsqueeze(1).clone()
    branch_env.load = parent_env.load[batch_indices, rollout_indices].unsqueeze(1).clone()
    branch_env.visited_ninf_flag = parent_env.visited_ninf_flag[batch_indices, rollout_indices].unsqueeze(1).clone()
    branch_env.ninf_mask = parent_env.ninf_mask[batch_indices, rollout_indices].unsqueeze(1).clone()
    branch_env.finished = parent_env.finished[batch_indices, rollout_indices].unsqueeze(1).clone()
    branch_env.deadlocked = parent_env.deadlocked[batch_indices, rollout_indices].unsqueeze(1).clone()
    branch_env.time = parent_env.time[batch_indices, rollout_indices].unsqueeze(1).clone()

    branch_env.step_state = type(parent_env.step_state)(BATCH_IDX=branch_env.BATCH_IDX, ROLLOUT_IDX=branch_env.ROLLOUT_IDX)
    branch_env.step_state.selected_count = branch_env.selected_count
    branch_env.step_state.load = branch_env.load
    branch_env.step_state.current_node = branch_env.current_node
    branch_env.step_state.ninf_mask = branch_env.ninf_mask
    branch_env.step_state.finished = branch_env.finished
    branch_env.step_state.time = branch_env.time / branch_env.grid_scale[:, None]
    return branch_env


def _rollout_branch_candidates(model, parent_env: CVRPTWEnv, batch_indices: torch.Tensor, rollout_indices: torch.Tensor, branch_actions: torch.Tensor):
    branch_env = _clone_branch_env(parent_env, batch_indices, rollout_indices)
    with _temporary_decode_state(model, batch_indices, rollout_indices):
        state, reward, done = branch_env.step(branch_actions.unsqueeze(1))
        while not done:
            selected, _ = model(state, greedy_construction=True)
            state, reward, done = branch_env.step(selected)

    return (
        reward.squeeze(1).detach(),
        branch_env.selected_node_list.squeeze(1).detach(),
        branch_env.deadlocked.squeeze(1).detach(),
    )


def _can_use_reward_fast_path(objective_spec) -> bool:
    return (
        abs(float(objective_spec.duration_weight)) <= 1e-9
        and abs(float(objective_spec.lateness_penalty)) <= 1e-9
        and abs(float(objective_spec.overtime_penalty)) <= 1e-9
    )


def _can_use_distance_only_fast_path(objective_spec) -> bool:
    return (
        objective_spec.primary == "distance"
        and float(objective_spec.distance_weight) > 0.0
        and abs(float(objective_spec.duration_weight)) <= 1e-9
        and abs(float(objective_spec.vehicle_fixed_cost)) <= 1e-9
        and abs(float(objective_spec.lateness_penalty)) <= 1e-9
        and abs(float(objective_spec.overtime_penalty)) <= 1e-9
    )


def _branch_ranking_key(
    objective_spec,
    *,
    feasible: bool,
    vehicle_count: int,
    distance: float,
    action_prob: float,
) -> tuple:
    generalized_cost = (
        objective_spec.vehicle_fixed_cost * float(vehicle_count)
        + objective_spec.distance_weight * distance
    )
    feasible_rank = 0 if feasible else 1
    if objective_spec.primary == "vehicle_count":
        primary_value = vehicle_count
        secondary_value = generalized_cost
    else:
        primary_value = generalized_cost
        secondary_value = vehicle_count
    return (feasible_rank, primary_value, secondary_value, distance, 0.0, 0, 0, -action_prob)


def _select_actions_lookahead(
    model,
    env: CVRPTWEnv,
    state,
    probs: torch.Tensor,
    *,
    confident_prob: float,
    top_k: int,
    uncertain_chunk_size: int,
    batch_distance_matrices,
    batch_demands,
    batch_capacities,
    batch_time_windows,
    batch_service_times,
    objective,
):
    objective_spec = normalize_objective_spec(objective)
    valid_mask = torch.isfinite(state.ninf_mask)
    valid_counts = valid_mask.sum(dim=2)
    top_probs, top_actions = probs.max(dim=2)
    selected = top_actions.clone()
    active = ~state.finished

    greedy_shortcuts = active & ((top_probs > confident_prob) | (valid_counts <= 1))
    uncertain = torch.nonzero(active & (~greedy_shortcuts), as_tuple=False)
    stats = {
        "greedy_shortcuts": int(greedy_shortcuts.sum().item()),
        "lookahead_steps": 0,
        "rollout_candidates": 0,
    }

    if uncertain.numel() == 0:
        selected[~active] = 0
        return selected, stats

    branch_k = min(max(1, int(top_k)), int(probs.size(2)))
    chunk_size = max(1, int(uncertain_chunk_size))
    branch_index = torch.arange(branch_k, device=probs.device)
    use_distance_only_fast_path = _can_use_distance_only_fast_path(objective_spec)
    use_reward_fast_path = _can_use_reward_fast_path(objective_spec)

    for start in range(0, uncertain.size(0), chunk_size):
        chunk = uncertain[start:start + chunk_size]
        parent_bs = chunk[:, 0]
        parent_rs = chunk[:, 1]
        chunk_probs = probs[parent_bs, parent_rs].masked_fill(~valid_mask[parent_bs, parent_rs], -1.0)
        top_branch_probs, top_branch_actions = torch.topk(chunk_probs, k=branch_k, dim=1)
        branch_counts = valid_counts[parent_bs, parent_rs].clamp_max(branch_k)
        branch_mask = branch_index.unsqueeze(0) < branch_counts.unsqueeze(1)

        branch_parent_bs_t = parent_bs.repeat_interleave(branch_counts)
        branch_parent_rs_t = parent_rs.repeat_interleave(branch_counts)
        branch_actions_t = top_branch_actions[branch_mask]
        branch_probs_t = top_branch_probs[branch_mask]

        branch_reward, branch_sequences, branch_deadlocked = _rollout_branch_candidates(
            model,
            env,
            branch_parent_bs_t,
            branch_parent_rs_t,
            branch_actions_t,
        )

        if use_distance_only_fast_path:
            branch_distance_matrix = torch.full(
                (chunk.size(0), branch_k),
                float("inf"),
                dtype=branch_reward.dtype,
                device=branch_reward.device,
            )
            branch_distance_matrix[branch_mask] = -branch_reward
            best_local = torch.argmin(branch_distance_matrix - top_branch_probs * 1e-6, dim=1)
            selected[parent_bs, parent_rs] = top_branch_actions[torch.arange(chunk.size(0), device=probs.device), best_local]
            stats["lookahead_steps"] += int(chunk.size(0))
            stats["rollout_candidates"] += int(branch_actions_t.numel())
            continue

        prev_nodes = torch.cat([torch.zeros_like(branch_sequences[:, :1]), branch_sequences[:, :-1]], dim=1)
        branch_vehicle_counts = (branch_sequences.ne(0) & prev_nodes.eq(0)).sum(dim=1)
        branch_distances = -branch_reward

        offset = 0
        branch_sequences_cpu = None
        for row_idx in range(chunk.size(0)):
            count = int(branch_counts[row_idx].item())
            if count <= 0:
                continue

            best_local = 0
            batch_idx = int(parent_bs[row_idx].item())
            rollout_idx = int(parent_rs[row_idx].item())
            for local_idx in range(1, count):
                current_idx = offset + local_idx
                best_idx = offset + best_local

                if use_reward_fast_path:
                    current_key = _branch_ranking_key(
                        objective_spec,
                        feasible=not bool(branch_deadlocked[current_idx].item()),
                        vehicle_count=int(branch_vehicle_counts[current_idx].item()),
                        distance=float(branch_distances[current_idx].item()),
                        action_prob=float(branch_probs_t[current_idx].item()),
                    )
                    best_key = _branch_ranking_key(
                        objective_spec,
                        feasible=not bool(branch_deadlocked[best_idx].item()),
                        vehicle_count=int(branch_vehicle_counts[best_idx].item()),
                        distance=float(branch_distances[best_idx].item()),
                        action_prob=float(branch_probs_t[best_idx].item()),
                    )
                else:
                    if branch_sequences_cpu is None:
                        branch_sequences_cpu = branch_sequences.detach().cpu().tolist()

                    if bool(branch_deadlocked[current_idx].item()):
                        current_key = _branch_ranking_key(
                            objective_spec,
                            feasible=False,
                            vehicle_count=int(branch_vehicle_counts[current_idx].item()),
                            distance=float(branch_distances[current_idx].item()),
                            action_prob=float(branch_probs_t[current_idx].item()),
                        )
                    else:
                        current_routes = split_depot_sequence([int(node) for node in branch_sequences_cpu[current_idx]])
                        current_score = score_cvrptw_routes(
                            current_routes,
                            batch_distance_matrices[batch_idx],
                            batch_demands[batch_idx],
                            batch_capacities[batch_idx],
                            batch_time_windows[batch_idx],
                            batch_service_times[batch_idx],
                            objective_spec,
                        )
                        current_key = current_score.ranking_key() + (-float(branch_probs_t[current_idx].item()),)

                    if bool(branch_deadlocked[best_idx].item()):
                        best_key = _branch_ranking_key(
                            objective_spec,
                            feasible=False,
                            vehicle_count=int(branch_vehicle_counts[best_idx].item()),
                            distance=float(branch_distances[best_idx].item()),
                            action_prob=float(branch_probs_t[best_idx].item()),
                        )
                    else:
                        best_routes = split_depot_sequence([int(node) for node in branch_sequences_cpu[best_idx]])
                        best_score = score_cvrptw_routes(
                            best_routes,
                            batch_distance_matrices[batch_idx],
                            batch_demands[batch_idx],
                            batch_capacities[batch_idx],
                            batch_time_windows[batch_idx],
                            batch_service_times[batch_idx],
                            objective_spec,
                        )
                        best_key = best_score.ranking_key() + (-float(branch_probs_t[best_idx].item()),)

                if current_key < best_key:
                    best_local = local_idx

            selected[batch_idx, rollout_idx] = int(branch_actions_t[offset + best_local].item())
            offset += count

        stats["lookahead_steps"] += int(chunk.size(0))
        stats["rollout_candidates"] += int(branch_actions_t.numel())

    selected[~active] = 0
    return selected, stats


def solve_cvrptw_with_decode_lookahead(
    depot_xy,
    node_xy,
    node_demand,
    capacity,
    node_tw,
    service_time,
    *,
    num_samples: int = 128,
    grid_scale=None,
    top_k: int = 3,
    confident_prob: float = 0.95,
    uncertain_chunk_size: int = 128,
    objective=None,
    device=None,
):
    solver = CVRPTWSolver(device=device)
    solver_device = solver.device

    depot_xy = torch.as_tensor(depot_xy, dtype=torch.float32, device=solver_device)
    node_xy = torch.as_tensor(node_xy, dtype=torch.float32, device=solver_device)
    node_demand = torch.as_tensor(node_demand, dtype=torch.float32, device=solver_device)
    capacity = torch.as_tensor(capacity, dtype=torch.float32, device=solver_device)
    node_tw = torch.as_tensor(node_tw, dtype=torch.float32, device=solver_device)
    service_time = torch.as_tensor(service_time, dtype=torch.float32, device=solver_device)
    grid_scale_tensor = None if grid_scale is None else torch.as_tensor(grid_scale, dtype=torch.float32, device=solver_device)

    if depot_xy.ndim == 1:
        depot_xy = depot_xy[None, None, :]
    elif depot_xy.ndim == 2:
        depot_xy = depot_xy[:, None, :]
    if node_xy.ndim == 2:
        node_xy = node_xy[None, :, :]
    if node_demand.ndim == 1:
        node_demand = node_demand[None, :]
    if node_tw.ndim == 2:
        node_tw = node_tw[None, :, :]
    if capacity.ndim == 0:
        capacity = capacity.repeat(node_xy.size(0))
    if service_time.ndim == 0:
        service_time = service_time.repeat(node_xy.size(0), node_xy.size(1))
    elif service_time.ndim == 1:
        if service_time.size(0) == node_xy.size(0):
            service_time = service_time[:, None].repeat(1, node_xy.size(1))
        else:
            service_time = service_time[None, :].repeat(node_xy.size(0), 1)
    if grid_scale_tensor is not None and grid_scale_tensor.ndim == 0:
        grid_scale_tensor = grid_scale_tensor.repeat(node_xy.size(0))

    batch_size, problem_size, _ = node_xy.shape
    solver._load_model()
    model = solver._model
    model_params = solver._model_params

    z, rollout_size = sample_binary_z(
        batch_size=batch_size,
        num_samples=int(num_samples),
        z_dim=int(model_params["z_dim"]),
        device=solver_device,
        problem_size=problem_size,
        force_first_move=bool(model_params["force_first_move"]),
    )

    env = CVRPTWEnv()
    env.load_instances(
        depot_xy=depot_xy,
        node_xy=node_xy,
        node_demand=node_demand,
        capacity=capacity,
        node_tw=node_tw,
        service_time=service_time,
        rollout_size=rollout_size,
        grid_scale=grid_scale_tensor,
    )

    objective_spec = normalize_objective_spec(objective)
    batch_demands = [normalize_demands(node_demand[batch_idx].detach().cpu().tolist()) for batch_idx in range(batch_size)]
    batch_time_windows = [normalize_time_windows(node_tw[batch_idx].detach().cpu().tolist()) for batch_idx in range(batch_size)]
    batch_service_times = [
        normalize_service_times(service_time[batch_idx].detach().cpu().tolist(), len(batch_demands[batch_idx]))
        for batch_idx in range(batch_size)
    ]
    batch_distance_matrices = [env.distance_matrix[batch_idx].detach().cpu().tolist() for batch_idx in range(batch_size)]
    batch_capacities = [float(item) for item in capacity.detach().cpu().tolist()]
    lookahead_stats = {"greedy_shortcuts": 0, "lookahead_steps": 0, "rollout_candidates": 0}

    model.eval()
    with torch.no_grad():
        reset_state, _, _ = env.reset()
        model.pre_forward(reset_state, z)
        state, reward, done = env.pre_step()
        while not done:
            if int(state.selected_count or 0) > 1:
                encoded_last_node = model._get_current_node_or_depot(state) if hasattr(model, "_get_current_node_or_depot") else None
                if encoded_last_node is None:
                    from .cvrptw import _get_encoding, _sanitize_probs

                    encoded_last_node = _get_encoding(model.encoded_nodes, state.current_node)
                    probs = model.decoder(encoded_last_node, state.load, state.time, state.ninf_mask)
                    probs = _sanitize_probs(probs, state.ninf_mask)
                else:
                    probs = model.decoder(encoded_last_node, state.load, state.time, state.ninf_mask)
                selected, step_stats = _select_actions_lookahead(
                    model,
                    env,
                    state,
                    probs,
                    confident_prob=float(confident_prob),
                    top_k=int(top_k),
                    uncertain_chunk_size=int(uncertain_chunk_size),
                    batch_distance_matrices=batch_distance_matrices,
                    batch_demands=batch_demands,
                    batch_capacities=batch_capacities,
                    batch_time_windows=batch_time_windows,
                    batch_service_times=batch_service_times,
                    objective=objective_spec,
                )
                for key, value in step_stats.items():
                    lookahead_stats[key] += int(value)
            else:
                selected, _ = model(state, greedy_construction=True)
            state, reward, done = env.step(selected)

    checkpoint_str = str(solver.checkpoint_path)
    results_per_batch: list[list[dict]] = []
    for batch_row in range(batch_size):
        batch_results: list[dict] = []
        for rollout_idx in range(env.rollout_size):
            raw_sequence = [int(node) for node in env.selected_node_list[batch_row, rollout_idx].detach().cpu().tolist()]
            routes = split_depot_sequence(raw_sequence)
            score = score_cvrptw_routes(
                routes,
                batch_distance_matrices[batch_row],
                batch_demands[batch_row],
                batch_capacities[batch_row],
                batch_time_windows[batch_row],
                batch_service_times[batch_row],
                objective_spec,
            )
            ranking_key = score.ranking_key()
            batch_results.append(
                {
                    "problem_type": "cvrptw",
                    "raw_sequence": raw_sequence,
                    "routes": routes,
                    "distance": float(score.distance),
                    "checkpoint": checkpoint_str,
                    "meta": {
                        "lookahead_top_k": int(top_k),
                        "lookahead_confident_prob": float(confident_prob),
                        "lookahead_uncertain_chunk_size": int(uncertain_chunk_size),
                        "lookahead_stats": dict(lookahead_stats),
                    },
                    "_ranking_key": ranking_key,
                }
            )

        batch_results.sort(key=lambda item: item["_ranking_key"])
        for item in batch_results:
            item.pop("_ranking_key", None)
        results_per_batch.append(batch_results)

    best_results = [batch_results[0] for batch_results in results_per_batch]
    return maybe_unbatch(best_results)
