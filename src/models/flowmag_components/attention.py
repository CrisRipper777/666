from __future__ import annotations

import math

import torch
from torch import nn
from torch_geometric.utils import softmax as segment_softmax

from .adapters import AdaptedLinear


class SameModalRelation(nn.Module):
    """One feature-conditioned relation kernel and weighted aggregation."""

    def __init__(self, hidden_dim: int, relation_dim: int, adapter_rank: int):
        super().__init__()
        keys = ("text", "image")
        self.query = AdaptedLinear(hidden_dim, relation_dim, adapter_rank, keys)
        self.key = AdaptedLinear(hidden_dim, relation_dim, adapter_rank, keys)
        self.scale = math.sqrt(relation_dim)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        source_valid: torch.Tensor,
        key: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_nodes = h.size(0)
        src, dst = edge_index
        edge_valid = source_valid[src]
        valid_src = src[edge_valid]
        valid_dst = dst[edge_valid]
        mu = h.new_zeros((num_nodes, h.size(-1)))
        same_valid = torch.zeros(num_nodes, dtype=torch.bool, device=h.device)
        if valid_src.numel() == 0:
            return mu, same_valid, h.new_empty((0,))

        q = self.query(h, key)
        k = self.key(h, key)
        scores = (q[valid_dst] * k[valid_src]).sum(dim=-1) / self.scale
        alpha = segment_softmax(scores, valid_dst, num_nodes=num_nodes)
        mu.index_add_(0, valid_dst, h[valid_src] * alpha.unsqueeze(-1))
        same_valid[valid_dst] = True
        return mu, same_valid, alpha


class CrossModalAttention(nn.Module):
    """Directional multi-head attention with a learned Null source."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        adapter_rank: int,
        *,
        null_cardinality_correction: bool = True,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("cross_heads must divide hidden_dim")
        if attention_dropout != 0:
            raise ValueError("FlowMAG V0.2.1 requires attention_dropout=0")
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = hidden_dim // num_heads
        self.directions = ("t2i", "i2t")
        self.query = AdaptedLinear(hidden_dim, hidden_dim, adapter_rank, self.directions)
        self.key = AdaptedLinear(hidden_dim, hidden_dim, adapter_rank, self.directions)
        self.value = AdaptedLinear(hidden_dim, hidden_dim, adapter_rank, self.directions)
        self.reference = AdaptedLinear(hidden_dim, hidden_dim, adapter_rank, self.directions)
        # Bias-free so a zero real-source correction remains exactly zero.
        self.output = AdaptedLinear(
            hidden_dim, hidden_dim, adapter_rank, self.directions, bias=False
        )
        self.null_key = nn.ParameterDict({
            direction: nn.Parameter(torch.empty(num_heads, self.head_dim))
            for direction in self.directions
        })
        self.null_bias = nn.ParameterDict({
            direction: nn.Parameter(torch.zeros(())) for direction in self.directions
        })
        for value in self.null_key.values():
            nn.init.xavier_uniform_(value)
        self.null_cardinality_correction = bool(null_cardinality_correction)

    def forward(
        self,
        target: torch.Tensor,
        source: torch.Tensor,
        edge_index: torch.Tensor,
        source_valid: torch.Tensor,
        direction: str,
        *,
        collect_diagnostics: bool = True,
        return_attention_diagnostics: bool = False,
    ):
        if direction not in self.directions:
            raise ValueError(f"direction must be one of {self.directions}, got {direction!r}")
        num_nodes = target.size(0)
        src, dst = edge_index
        edge_valid = source_valid[src]
        real_src = src[edge_valid]
        real_dst = dst[edge_valid]

        q = self.query(target, direction).view(num_nodes, self.num_heads, self.head_dim)
        ref = self.reference(target, direction).view(num_nodes, self.num_heads, self.head_dim)
        null_key = self.null_key[direction]
        null_scores = (q * null_key.unsqueeze(0)).sum(dim=-1) / math.sqrt(self.head_dim)
        degrees = torch.zeros(num_nodes, dtype=torch.long, device=target.device)
        if real_src.numel():
            degrees.index_add_(0, real_dst, torch.ones_like(real_dst))
        if self.null_cardinality_correction:
            null_scores = null_scores + torch.log(degrees.clamp_min(1).to(null_scores.dtype)).unsqueeze(-1)
        null_scores = null_scores + self.null_bias[direction]

        if real_src.numel():
            k = self.key(source, direction).view(num_nodes, self.num_heads, self.head_dim)
            real_scores = (
                q[real_dst] * k[real_src]
            ).sum(dim=-1) / math.sqrt(self.head_dim)
            all_scores = torch.cat((real_scores, null_scores), dim=0)
            all_dst = torch.cat((real_dst, torch.arange(num_nodes, device=target.device)))
            all_alpha = segment_softmax(all_scores, all_dst, num_nodes=num_nodes)
            alpha_real = all_alpha[: real_src.numel()]
            alpha_null = all_alpha[real_src.numel() :]

            v = self.value(source, direction).view(num_nodes, self.num_heads, self.head_dim)
            differences = v[real_src] - ref[real_dst]
            messages = alpha_real.unsqueeze(-1) * differences
            cross_raw = target.new_zeros((num_nodes, self.num_heads, self.head_dim))
            cross_raw.index_add_(0, real_dst, messages)
            if collect_diagnostics:
                safe_alpha_real = alpha_real.clamp_min(torch.finfo(alpha_real.dtype).tiny)
                entropy_by_edge_head = -(safe_alpha_real * safe_alpha_real.log())
                entropy_by_node_head = target.new_zeros((num_nodes, self.num_heads))
                entropy_by_node_head.index_add_(0, real_dst, entropy_by_edge_head)
        else:
            alpha_null = torch.ones(
                num_nodes, self.num_heads, dtype=target.dtype, device=target.device
            )
            cross_raw = target.new_zeros((num_nodes, self.num_heads, self.head_dim))
            if collect_diagnostics:
                entropy_by_node_head = target.new_zeros((num_nodes, self.num_heads))

        if collect_diagnostics:
            safe_alpha_null = alpha_null.clamp_min(torch.finfo(alpha_null.dtype).tiny)
            null_entropy_by_head = -(safe_alpha_null * safe_alpha_null.log())
            entropy_by_target = (entropy_by_node_head + null_entropy_by_head).mean(dim=-1)
            positive_degree = degrees > 0
            normalizer = torch.log((degrees + 1).to(entropy_by_target.dtype))
            normalized = entropy_by_target / normalizer.clamp_min(torch.finfo(entropy_by_target.dtype).tiny)
            normalized_entropy_by_target = torch.where(
                positive_degree, normalized, torch.zeros_like(normalized)
            )
            entropy = entropy_by_target.mean()
        else:
            entropy = None
            entropy_by_target = None
            normalized_entropy_by_target = None
        null_mass = alpha_null.mean(dim=-1)
        cross_raw = cross_raw.reshape(num_nodes, self.hidden_dim)
        cross = self.output(cross_raw, direction)
        cross_valid = degrees > 0
        # Make the structural zero invariant explicit even under mixed precision.
        cross = torch.where(cross_valid.unsqueeze(-1), cross, torch.zeros_like(cross))
        if return_attention_diagnostics:
            return (
                cross,
                null_mass,
                entropy,
                cross_valid,
                entropy_by_target,
                normalized_entropy_by_target,
            )
        return cross, null_mass, entropy, cross_valid
