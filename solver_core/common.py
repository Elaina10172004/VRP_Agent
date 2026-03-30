from __future__ import annotations

from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parents[1]


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def select_checkpoint(problem_size: int, candidates: dict[int, str | Path]) -> Path:
    if not candidates:
        raise ValueError("No checkpoint candidates were provided.")
    ordered_sizes = sorted(int(size) for size in candidates)
    for size in ordered_sizes:
        if problem_size <= size:
            return Path(candidates[size]).resolve()
    return Path(candidates[ordered_sizes[-1]]).resolve()


def sample_binary_z(
    batch_size: int,
    num_samples: int,
    z_dim: int,
    device: torch.device,
    problem_size: int,
    force_first_move: bool,
) -> tuple[torch.Tensor, int]:
    if force_first_move:
        rollout_size = problem_size * num_samples
        z = torch.randint(
            0,
            2,
            size=(batch_size, problem_size, num_samples, z_dim),
            device=device,
            dtype=torch.float32,
        )
        z = z.transpose(1, 2).reshape(batch_size, rollout_size, z_dim)
        return z, rollout_size

    rollout_size = num_samples
    z = torch.randint(
        0,
        2,
        size=(batch_size, rollout_size, z_dim),
        device=device,
        dtype=torch.float32,
    )
    return z, rollout_size


def normalize_xy_group(
    *xy_tensors: torch.Tensor,
    scale: float | torch.Tensor | None = None,
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    if not xy_tensors:
        raise ValueError("At least one coordinate tensor is required.")

    stacked = torch.cat([tensor.reshape(tensor.size(0), -1, 2) for tensor in xy_tensors], dim=1)
    min_xy = stacked.min(dim=1, keepdim=True).values

    if scale is None:
        max_xy = stacked.max(dim=1, keepdim=True).values
        scale_tensor = (max_xy - min_xy).amax(dim=2).amax(dim=1).clamp_min(1e-6)
    else:
        scale_tensor = torch.as_tensor(scale, dtype=stacked.dtype, device=stacked.device)
        if scale_tensor.ndim == 0:
            scale_tensor = scale_tensor.repeat(stacked.size(0))
        scale_tensor = scale_tensor.reshape(stacked.size(0)).clamp_min(1e-6)

    normalized = [
        (tensor - min_xy) / scale_tensor[:, None, None]
        for tensor in xy_tensors
    ]
    return normalized, min_xy, scale_tensor


def split_depot_sequence(sequence: list[int]) -> list[list[int]]:
    routes: list[list[int]] = []
    current: list[int] = []
    for node in sequence:
        if node == 0:
            if current:
                routes.append(current)
                current = []
            continue
        current.append(node - 1)
    if current:
        routes.append(current)
    return routes


def maybe_unbatch(results: list[dict]) -> dict | list[dict]:
    return results[0] if len(results) == 1 else results


def remap_legacy_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    replacements = (
        (".Wq.", ".wq."),
        (".Wk.", ".wk."),
        (".Wv.", ".wv."),
        (".W_gate.", ".w_gate."),
        ("addAndNormalization1.", "add_norm_1."),
        ("addAndNormalization2.", "add_norm_2."),
        ("add_n_normalization_1.", "add_norm_1."),
        ("add_n_normalization_2.", "add_norm_2."),
        ("feedForward.W1.", "feed_forward.w1."),
        ("feedForward.W2.", "feed_forward.w2."),
        ("feed_forward.W1.", "feed_forward.w1."),
        ("feed_forward.W2.", "feed_forward.w2."),
        ("decoder.Wq_first.", "decoder.wq_first."),
        ("decoder.Wq_last.", "decoder.wq_last."),
        ("decoder.Wk.", "decoder.wk."),
        ("decoder.Wv.", "decoder.wv."),
    )

    remapped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        for old, new in replacements:
            new_key = new_key.replace(old, new)
        remapped[new_key] = value
    return remapped
