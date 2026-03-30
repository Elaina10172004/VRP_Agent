from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def reshape_by_heads(qkv: torch.Tensor, head_num: int) -> torch.Tensor:
    batch_size = qkv.size(0)
    item_count = qkv.size(1)
    qkv = qkv.reshape(batch_size, item_count, head_num, -1)
    return qkv.transpose(1, 2)


def reshape_from_heads(headwise_out: torch.Tensor) -> torch.Tensor:
    out_transposed = headwise_out.transpose(1, 2)
    batch_size = out_transposed.size(0)
    item_count = out_transposed.size(1)
    head_num = out_transposed.size(2)
    key_dim = out_transposed.size(3)
    return out_transposed.reshape(batch_size, item_count, head_num * key_dim)


def multi_head_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rank3_ninf_mask: torch.Tensor | None = None,
    return_headwise: bool = False,
) -> torch.Tensor:
    batch_size = q.size(0)
    head_num = q.size(1)
    item_count = q.size(2)
    key_dim = q.size(3)
    input_count = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float32, device=q.device))
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_size, head_num, item_count, input_count)

    weights = torch.softmax(score_scaled, dim=3)
    out = torch.matmul(weights, v)
    if return_headwise:
        return out
    return reshape_from_heads(out)


def fast_multi_head_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rank3_ninf_mask: torch.Tensor | None = None,
    return_headwise: bool = False,
) -> torch.Tensor:
    batch_size = q.size(0)
    head_num = q.size(1)
    item_count = q.size(2)
    input_count = k.size(2)

    mask = None
    if rank3_ninf_mask is not None:
        mask = rank3_ninf_mask[:, None, :, :].expand(batch_size, head_num, item_count, input_count)

    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    if return_headwise:
        return out
    return reshape_from_heads(out)


class AddAndInstanceNormalization(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        added = input1 + input2
        normalized = self.norm(added.transpose(1, 2))
        return normalized.transpose(1, 2)


class FeedForward(nn.Module):
    def __init__(self, embedding_dim: int, ff_hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.w2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.w2(F.relu(self.w1(inputs)))


class RMSNorm(nn.Module):
    def __init__(self, embedding_dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embedding_dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        rms = inputs.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return inputs * rms * self.weight
