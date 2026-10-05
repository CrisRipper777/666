from __future__ import annotations

import math

import torch
from torch import nn


class AdaptedLinear(nn.Module):
    """A shared linear map with key-specific low-rank residuals."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        keys: tuple[str, ...] | list[str],
        *,
        bias: bool = True,
    ):
        super().__init__()
        if in_features <= 0 or out_features <= 0 or rank <= 0:
            raise ValueError("AdaptedLinear dimensions and rank must be positive")
        if not keys or len(set(keys)) != len(keys):
            raise ValueError("AdaptedLinear keys must be non-empty and unique")
        self.base = nn.Linear(in_features, out_features, bias=bias)
        self.a = nn.ModuleDict({
            key: nn.Linear(in_features, rank, bias=False) for key in keys
        })
        self.b = nn.ModuleDict({
            key: nn.Linear(rank, out_features, bias=False) for key in keys
        })
        for layer in self.a.values():
            nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
        for layer in self.b.values():
            nn.init.zeros_(layer.weight)

    def forward(self, x: torch.Tensor, key: str) -> torch.Tensor:
        if key not in self.a:
            raise KeyError(f"Unknown adapter key {key!r}; available: {tuple(self.a)}")
        return self.base(x) + self.b[key](self.a[key](x))


class ModalityStem(nn.Module):
    """Independent topology-free projection for one input modality."""

    def __init__(self, in_features: int, hidden_dim: int):
        super().__init__()
        if in_features <= 0 or hidden_dim <= 0:
            raise ValueError("stem dimensions must be positive")
        self.layers = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
