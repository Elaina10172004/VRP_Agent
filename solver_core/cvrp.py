from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ROOT_DIR, maybe_unbatch, normalize_xy_group, remap_legacy_state_dict_keys, resolve_device, sample_binary_z, select_checkpoint, split_depot_sequence
from .nn import AddAndInstanceNormalization, FeedForward, fast_multi_head_attention, multi_head_attention, reshape_by_heads


DEFAULT_CVRP_MODEL_PARAMS = {
    "embedding_dim": 128,
    "poly_embedding_dim": 256,
    "sqrt_embedding_dim": 128 ** 0.5,
    "encoder_layer_num": 6,
    "qkv_dim": 16,
    "head_num": 8,
    "logit_clipping": 10,
    "ff_hidden_dim": 512,
    "z_dim": 16,
    "use_fast_attention": True,
    "force_first_move": False,
}

DEFAULT_CVRP_CHECKPOINTS = {
    100: ROOT_DIR / "CVRP" / "models" / "PolyNet" / "n100" / "checkpoint-300.pt",
    200: ROOT_DIR / "CVRP" / "models" / "PolyNet" / "n200" / "checkpoint-40.pt",
}


@dataclass
class ResetState:
    depot_xy: torch.Tensor
    node_xy: torch.Tensor
    node_demand: torch.Tensor


@dataclass
class StepState:
    BATCH_IDX: torch.Tensor
    ROLLOUT_IDX: torch.Tensor
    selected_count: int | None = None
    load: torch.Tensor | None = None
    current_node: torch.Tensor | None = None
    ninf_mask: torch.Tensor | None = None
    finished: torch.Tensor | None = None


class CVRPEnv:
    def __init__(self) -> None:
        self.problem_size: int | None = None
        self.rollout_size: int | None = None
        self.batch_size: int | None = None
        self.BATCH_IDX: torch.Tensor | None = None
        self.ROLLOUT_IDX: torch.Tensor | None = None
        self.model_depot_xy: torch.Tensor | None = None
        self.model_node_xy: torch.Tensor | None = None
        self.cost_depot_node_xy: torch.Tensor | None = None
        self.depot_node_demand: torch.Tensor | None = None
        self.selected_count = 0
        self.current_node: torch.Tensor | None = None
        self.selected_node_list: torch.Tensor | None = None
        self.at_the_depot: torch.Tensor | None = None
        self.load: torch.Tensor | None = None
        self.visited_ninf_flag: torch.Tensor | None = None
        self.ninf_mask: torch.Tensor | None = None
        self.finished: torch.Tensor | None = None
        self.reset_state: ResetState | None = None
        self.step_state: StepState | None = None

    def load_instances(
        self,
        depot_xy: torch.Tensor,
        node_xy: torch.Tensor,
        node_demand: torch.Tensor,
        capacity: torch.Tensor,
        rollout_size: int,
    ) -> None:
        self.batch_size = node_xy.size(0)
        self.problem_size = node_xy.size(1)
        self.rollout_size = rollout_size

        normalized, _, _ = normalize_xy_group(depot_xy, node_xy)
        self.model_depot_xy, self.model_node_xy = normalized
        self.cost_depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        depot_demand = torch.zeros((self.batch_size, 1), dtype=torch.float32, device=node_xy.device)
        self.depot_node_demand = torch.cat((depot_demand, node_demand / capacity[:, None]), dim=1)

        self.BATCH_IDX = torch.arange(self.batch_size, device=node_xy.device)[:, None].expand(self.batch_size, rollout_size)
        self.ROLLOUT_IDX = torch.arange(rollout_size, device=node_xy.device)[None, :].expand(self.batch_size, rollout_size)
        self.reset_state = ResetState(self.model_depot_xy, self.model_node_xy, self.depot_node_demand[:, 1:])
        self.step_state = StepState(BATCH_IDX=self.BATCH_IDX, ROLLOUT_IDX=self.ROLLOUT_IDX)

    def reset(self) -> tuple[ResetState, None, bool]:
        self.selected_count = 0
        self.current_node = None
        self.selected_node_list = torch.zeros(
            (self.batch_size, self.rollout_size, 0),
            dtype=torch.long,
            device=self.depot_node_demand.device,
        )
        self.at_the_depot = torch.ones((self.batch_size, self.rollout_size), dtype=torch.bool, device=self.depot_node_demand.device)
        self.load = torch.ones((self.batch_size, self.rollout_size), dtype=torch.float32, device=self.depot_node_demand.device)
        self.visited_ninf_flag = torch.zeros(
            (self.batch_size, self.rollout_size, self.problem_size + 1),
            dtype=torch.float32,
            device=self.depot_node_demand.device,
        )
        self.ninf_mask = torch.zeros_like(self.visited_ninf_flag)
        self.finished = torch.zeros((self.batch_size, self.rollout_size), dtype=torch.bool, device=self.depot_node_demand.device)
        return self.reset_state, None, False

    def pre_step(self) -> tuple[StepState, None, bool]:
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        return self.step_state, None, False

    def step(self, selected: torch.Tensor) -> tuple[StepState, torch.Tensor | None, bool]:
        self.selected_count += 1
        self.current_node = selected
        self.selected_node_list = torch.cat((self.selected_node_list, selected[:, :, None]), dim=2)
        self.at_the_depot = selected == 0

        demand_list = self.depot_node_demand[:, None, :].expand(self.batch_size, self.rollout_size, -1)
        selected_demand = demand_list.gather(dim=2, index=selected[:, :, None]).squeeze(2)
        self.load = self.load - selected_demand
        self.load[self.at_the_depot] = 1.0

        self.visited_ninf_flag[self.BATCH_IDX, self.ROLLOUT_IDX, selected] = float("-inf")
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0

        self.ninf_mask = self.visited_ninf_flag.clone()
        demand_too_large = self.load[:, :, None] + 1e-5 < demand_list
        self.ninf_mask[demand_too_large] = float("-inf")

        newly_finished = (self.visited_ninf_flag == float("-inf")).all(dim=2)
        self.finished = self.finished | newly_finished
        self.ninf_mask[:, :, 0][self.finished] = 0

        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        done = bool(self.finished.all())
        reward = -self._get_travel_distance() if done else None
        return self.step_state, reward, done

    def _get_travel_distance(self) -> torch.Tensor:
        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        all_xy = self.cost_depot_node_xy[:, None, :, :].expand(-1, self.rollout_size, -1, -1)
        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        return ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt().sum(2)


class CVRPModel(nn.Module):
    def __init__(self, **model_params: float | int | bool) -> None:
        super().__init__()
        self.model_params = model_params
        self.force_first_move = bool(model_params["force_first_move"])
        self.encoder = CVRPEncoder(**model_params)
        self.decoder = CVRPDecoder(**model_params)
        self.encoded_nodes: torch.Tensor | None = None

    def pre_forward(self, reset_state: ResetState, z: torch.Tensor) -> None:
        node_xy_demand = torch.cat((reset_state.node_xy, reset_state.node_demand[:, :, None]), dim=2)
        self.encoded_nodes = self.encoder(reset_state.depot_xy, node_xy_demand)
        self.decoder.set_kv(self.encoded_nodes, z)

    def forward(self, state: StepState, greedy_construction: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = state.BATCH_IDX.size(0)
        rollout_size = state.BATCH_IDX.size(1)
        problem_size = self.encoded_nodes.size(1) - 1

        if state.selected_count == 0:
            selected = torch.zeros((batch_size, rollout_size), dtype=torch.long, device=self.encoded_nodes.device)
            prob = torch.ones((batch_size, rollout_size), device=self.encoded_nodes.device)
            return selected, prob

        if state.selected_count == 1 and self.force_first_move:
            selected = torch.arange(problem_size, device=self.encoded_nodes.device).repeat(rollout_size // problem_size)[None, :]
            selected = selected.expand(batch_size, rollout_size)
            prob = torch.ones((batch_size, rollout_size), device=self.encoded_nodes.device)
            return selected, prob

        encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
        probs = self.decoder(encoded_last_node, state.load, state.ninf_mask)
        if greedy_construction:
            selected = probs.argmax(dim=2)
        else:
            while True:
                with torch.no_grad():
                    selected = probs.reshape(batch_size * rollout_size, -1).multinomial(1).squeeze(1).reshape(batch_size, rollout_size)
                prob = probs[state.BATCH_IDX, state.ROLLOUT_IDX, selected].reshape(batch_size, rollout_size)
                if (prob != 0).all():
                    break

        prob = probs[state.BATCH_IDX, state.ROLLOUT_IDX, selected].reshape(batch_size, rollout_size)
        return selected, prob


def _get_encoding(encoded_nodes: torch.Tensor, node_index_to_pick: torch.Tensor) -> torch.Tensor:
    batch_size = node_index_to_pick.size(0)
    rollout_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)
    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, rollout_size, embedding_dim)
    return encoded_nodes.gather(dim=1, index=gathering_index)


class CVRPEncoder(nn.Module):
    def __init__(self, **model_params: float | int | bool) -> None:
        super().__init__()
        embedding_dim = int(model_params["embedding_dim"])
        encoder_layer_num = int(model_params["encoder_layer_num"])
        self.embedding_depot = nn.Linear(2, embedding_dim)
        self.embedding_node = nn.Linear(3, embedding_dim)
        self.layers = nn.ModuleList([CVRPEncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, depot_xy: torch.Tensor, node_xy_demand: torch.Tensor) -> torch.Tensor:
        out = torch.cat((self.embedding_depot(depot_xy), self.embedding_node(node_xy_demand)), dim=1)
        for layer in self.layers:
            out = layer(out)
        return out


class CVRPEncoderLayer(nn.Module):
    def __init__(self, **model_params: float | int | bool) -> None:
        super().__init__()
        embedding_dim = int(model_params["embedding_dim"])
        head_num = int(model_params["head_num"])
        qkv_dim = int(model_params["qkv_dim"])
        self.head_num = head_num
        self.attention_fn = fast_multi_head_attention if model_params["use_fast_attention"] else multi_head_attention
        self.wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.add_norm_1 = AddAndInstanceNormalization(embedding_dim)
        self.feed_forward = FeedForward(embedding_dim, int(model_params["ff_hidden_dim"]))
        self.add_norm_2 = AddAndInstanceNormalization(embedding_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        q = reshape_by_heads(self.wq(inputs), head_num=self.head_num)
        k = reshape_by_heads(self.wk(inputs), head_num=self.head_num)
        v = reshape_by_heads(self.wv(inputs), head_num=self.head_num)
        out_concat = self.attention_fn(q, k, v)
        out1 = self.add_norm_1(inputs, self.multi_head_combine(out_concat))
        out2 = self.feed_forward(out1)
        return self.add_norm_2(out1, out2)


class CVRPDecoder(nn.Module):
    def __init__(self, **model_params: float | int | bool) -> None:
        super().__init__()
        embedding_dim = int(model_params["embedding_dim"])
        poly_embedding_dim = int(model_params["poly_embedding_dim"])
        head_num = int(model_params["head_num"])
        qkv_dim = int(model_params["qkv_dim"])
        z_dim = int(model_params["z_dim"])
        self.model_params = model_params
        self.head_num = head_num
        self.attention_fn = fast_multi_head_attention if model_params["use_fast_attention"] else multi_head_attention
        self.wq_last = nn.Linear(embedding_dim + 1, head_num * qkv_dim, bias=False)
        self.wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.poly_layer_1 = nn.Linear(embedding_dim + z_dim, poly_embedding_dim)
        self.poly_layer_2 = nn.Linear(poly_embedding_dim, embedding_dim)
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None
        self.single_head_key: torch.Tensor | None = None
        self.z: torch.Tensor | None = None

    def set_kv(self, encoded_nodes: torch.Tensor, z: torch.Tensor) -> None:
        self.k = reshape_by_heads(self.wk(encoded_nodes), head_num=self.head_num)
        self.v = reshape_by_heads(self.wv(encoded_nodes), head_num=self.head_num)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        self.z = z

    def forward(self, encoded_last_node: torch.Tensor, load: torch.Tensor, ninf_mask: torch.Tensor) -> torch.Tensor:
        input_cat = torch.cat((encoded_last_node, load[:, :, None]), dim=2)
        q = reshape_by_heads(self.wq_last(input_cat), head_num=self.head_num)
        out_concat = self.attention_fn(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        mh_atten_out = self.multi_head_combine(out_concat)
        poly_out = self.poly_layer_2(F.relu(self.poly_layer_1(torch.cat((mh_atten_out, self.z), dim=2))))
        score = torch.matmul(mh_atten_out + poly_out, self.single_head_key)
        score_scaled = score / float(self.model_params["sqrt_embedding_dim"])
        score_clipped = float(self.model_params["logit_clipping"]) * torch.tanh(score_scaled)
        return F.softmax(score_clipped + ninf_mask, dim=2)


class CVRPSolver:
    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        model_params: dict | None = None,
        device: str | torch.device | None = None,
        amp_inference: bool = True,
    ) -> None:
        self.device = resolve_device(device)
        self.amp_inference = bool(amp_inference)
        self.model_param_overrides = dict(model_params or {})
        self.checkpoint_path = Path(checkpoint_path).resolve() if checkpoint_path is not None else None
        self._loaded_checkpoint: Path | None = None
        self._model: CVRPModel | None = None
        self._model_params: dict | None = None

    def _resolve_checkpoint(self, problem_size: int) -> Path:
        if self.checkpoint_path is not None:
            return self.checkpoint_path
        return select_checkpoint(problem_size, DEFAULT_CVRP_CHECKPOINTS)

    def _load_model(self, problem_size: int) -> None:
        checkpoint_path = self._resolve_checkpoint(problem_size)
        if self._model is not None and self._loaded_checkpoint == checkpoint_path:
            return

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model_params = dict(DEFAULT_CVRP_MODEL_PARAMS)
        model_params.update(self.model_param_overrides)
        model_params["z_dim"] = int(checkpoint.get("z_dim", model_params["z_dim"]))
        model_params["force_first_move"] = bool(checkpoint.get("force_first_move", model_params["force_first_move"]))

        model = CVRPModel(**model_params).to(self.device)
        model.load_state_dict(remap_legacy_state_dict_keys(checkpoint["model_state_dict"]))
        model.eval()

        self._loaded_checkpoint = checkpoint_path
        self._model = model
        self._model_params = model_params

    def solve(
        self,
        depot_xy: torch.Tensor | list,
        node_xy: torch.Tensor | list,
        node_demand: torch.Tensor | list,
        capacity: torch.Tensor | list | float | int,
        num_samples: int = 128,
        greedy: bool = False,
        return_topk: int | None = None,
    ) -> dict | list[dict] | list[list[dict]]:
        depot_xy = torch.as_tensor(depot_xy, dtype=torch.float32, device=self.device)
        node_xy = torch.as_tensor(node_xy, dtype=torch.float32, device=self.device)
        node_demand = torch.as_tensor(node_demand, dtype=torch.float32, device=self.device)
        capacity = torch.as_tensor(capacity, dtype=torch.float32, device=self.device)

        if depot_xy.ndim == 1:
            depot_xy = depot_xy[None, None, :]
        elif depot_xy.ndim == 2:
            depot_xy = depot_xy[:, None, :]
        if node_xy.ndim == 2:
            node_xy = node_xy[None, :, :]
        if node_demand.ndim == 1:
            node_demand = node_demand[None, :]
        if capacity.ndim == 0:
            capacity = capacity.repeat(node_xy.size(0))

        if depot_xy.ndim != 3 or depot_xy.size(1) != 1 or depot_xy.size(2) != 2:
            raise ValueError("depot_xy must have shape (2,), (B, 2), or (B, 1, 2).")
        if node_xy.ndim != 3 or node_xy.size(2) != 2:
            raise ValueError("node_xy must have shape (N, 2) or (B, N, 2).")
        if node_demand.ndim != 2:
            raise ValueError("node_demand must have shape (N,) or (B, N).")
        if not (depot_xy.size(0) == node_xy.size(0) == node_demand.size(0) == capacity.size(0)):
            raise ValueError("Batch sizes of depot_xy, node_xy, node_demand, and capacity must match.")

        batch_size, problem_size, _ = node_xy.shape
        self._load_model(problem_size)
        model = self._model
        model_params = self._model_params

        z, rollout_size = sample_binary_z(
            batch_size=batch_size,
            num_samples=int(num_samples),
            z_dim=int(model_params["z_dim"]),
            device=self.device,
            problem_size=problem_size,
            force_first_move=bool(model_params["force_first_move"]),
        )

        env = CVRPEnv()
        env.load_instances(depot_xy, node_xy, node_demand, capacity, rollout_size)

        with torch.no_grad():
            reset_state, _, _ = env.reset()
            model.pre_forward(reset_state, z)
            state, reward, done = env.pre_step()
            while not done:
                with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_inference and self.device.type == "cuda"):
                    selected, _ = model(state, greedy_construction=greedy)
                state, reward, done = env.step(selected)

        top_k = max(1, min(int(return_topk or 1), reward.size(1)))
        top_reward, top_idx = reward.topk(k=top_k, dim=1)
        batch_index = torch.arange(batch_size, device=self.device)[:, None].expand(batch_size, top_k)
        top_sequences = env.selected_node_list[batch_index, top_idx]

        checkpoint_str = str(self._loaded_checkpoint)
        results_per_batch: list[list[dict]] = []
        for seq_batch, reward_batch in zip(top_sequences.cpu(), top_reward.cpu()):
            batch_results: list[dict] = []
            for seq_tensor, reward_tensor in zip(seq_batch, reward_batch):
                raw_sequence = [int(node) for node in seq_tensor.tolist()]
                batch_results.append(
                    {
                        "problem_type": "cvrp",
                        "raw_sequence": raw_sequence,
                        "routes": split_depot_sequence(raw_sequence),
                        "distance": float(-reward_tensor.item()),
                        "checkpoint": checkpoint_str,
                    }
                )
            results_per_batch.append(batch_results)

        if return_topk and top_k > 1:
            return results_per_batch[0] if batch_size == 1 else results_per_batch

        best_results = [batch_results[0] for batch_results in results_per_batch]
        return maybe_unbatch(best_results)
