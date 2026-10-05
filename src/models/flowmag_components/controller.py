from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def candidate_descriptor(candidate: torch.Tensor, eps: float) -> torch.Tensor:
    """Detached direction and log-magnitude features for controller input."""
    detached = candidate.detach()
    direction = F.layer_norm(detached, (detached.size(-1),), eps=1e-5)
    rms = detached.square().mean(dim=-1, keepdim=True).sqrt()
    return torch.cat((direction, (rms + eps).log()), dim=-1)


class MaskAwareContext(nn.Module):
    """Two-token multimodal context used only by each layer's controller."""

    def __init__(self, hidden_dim: int, router_dim: int):
        super().__init__()
        self.modality_embedding = nn.Parameter(torch.empty(2, hidden_dim))
        self.scalar_scorer = nn.Linear(hidden_dim, 1)
        self.context_projection = nn.Linear(hidden_dim, router_dim)
        self.all_missing_context = nn.Parameter(torch.zeros(router_dim))
        nn.init.normal_(self.modality_embedding, std=0.02)

    def forward(
        self,
        text: torch.Tensor,
        image: torch.Tensor,
        text_valid: torch.Tensor,
        image_valid: torch.Tensor,
    ) -> torch.Tensor:
        tokens = torch.stack((text, image), dim=1) + self.modality_embedding.unsqueeze(0)
        valid = torch.stack((text_valid, image_valid), dim=1)
        logits = self.scalar_scorer(tokens).squeeze(-1)
        all_missing = ~valid.any(dim=1)
        safe_logits = logits.masked_fill(~valid, -torch.inf)
        safe_logits = torch.where(all_missing.unsqueeze(-1), torch.zeros_like(safe_logits), safe_logits)
        weights = torch.softmax(safe_logits, dim=-1) * valid.to(logits.dtype)
        context = self.context_projection((tokens * weights.unsqueeze(-1)).sum(dim=1))
        return torch.where(
            all_missing.unsqueeze(-1), self.all_missing_context.unsqueeze(0), context
        )


class FactorizedController(nn.Module):
    """Frequency choice plus separate same and cross admissions."""

    def __init__(
        self,
        hidden_dim: int,
        router_dim: int,
        tau_start: float,
        init_smooth_prob: float,
        init_same_admission: float,
        init_cross_admission: float,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.modality_embedding = nn.Parameter(torch.empty(2, router_dim))
        self.context_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 2 * router_dim, router_dim),
            nn.GELU(),
            nn.Linear(router_dim, router_dim),
        )
        self.flow_type_embedding = nn.Parameter(torch.empty(2, router_dim))
        self.eps = float(eps)
        candidate_input_dim = 2 * router_dim + hidden_dim + 1
        self.candidate_scorer = nn.Sequential(
            nn.Linear(candidate_input_dim, router_dim),
            nn.GELU(),
            nn.Linear(router_dim, 1),
        )
        nn.init.zeros_(self.candidate_scorer[-1].weight)
        nn.init.zeros_(self.candidate_scorer[-1].bias)
        # With logits divided by tau, a .75/.25 prior needs tau*log(3).
        self.flow_bias = nn.Parameter(torch.tensor([
            float(tau_start) * math.log(init_smooth_prob / (1.0 - init_smooth_prob)),
            0.0,
        ]))
        same_input_dim = router_dim + hidden_dim + 1 + 1
        cross_input_dim = router_dim + hidden_dim + 1 + 1 + 1
        self.same_admission = nn.Sequential(
            nn.Linear(same_input_dim, router_dim), nn.GELU(), nn.Linear(router_dim, 1)
        )
        self.cross_admission = nn.Sequential(
            nn.Linear(cross_input_dim, router_dim), nn.GELU(), nn.Linear(router_dim, 1)
        )
        for module, prior in (
            (self.same_admission, init_same_admission),
            (self.cross_admission, init_cross_admission),
        ):
            nn.init.zeros_(module[-1].weight)
            nn.init.constant_(module[-1].bias, math.log(prior / (1.0 - prior)))
        nn.init.normal_(self.modality_embedding, std=0.02)
        nn.init.normal_(self.flow_type_embedding, std=0.02)

    def target_context(
        self,
        target: torch.Tensor,
        joint_context: torch.Tensor,
        modality_index: int,
    ) -> torch.Tensor:
        target_norm = F.layer_norm(target, (target.size(-1),), eps=1e-5)
        modality = self.modality_embedding[modality_index].expand(target.size(0), -1)
        return self.context_mlp(torch.cat((target_norm, joint_context, modality), dim=-1))

    def frequency(
        self,
        q_ctrl: torch.Tensor,
        smooth: torch.Tensor,
        contrast: torch.Tensor,
        *,
        tau: float,
        rho: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        descriptors = (candidate_descriptor(smooth, self.eps), candidate_descriptor(contrast, self.eps))
        logits = []
        for idx, desc in enumerate(descriptors):
            flow_type = self.flow_type_embedding[idx].expand(q_ctrl.size(0), -1)
            score_input = torch.cat((q_ctrl, desc, flow_type), dim=-1)
            logits.append(self.candidate_scorer(score_input).squeeze(-1) + self.flow_bias[idx])
        logits = torch.stack(logits, dim=-1)
        learned = torch.softmax(logits / float(tau), dim=-1)
        prior = logits.new_tensor([0.75, 0.25]).view(1, 2)
        effective = (1.0 - float(rho)) * prior + float(rho) * learned
        return logits, effective

    def admissions(
        self,
        q_ctrl: torch.Tensor,
        same_flow: torch.Tensor,
        cross_flow: torch.Tensor,
        null_source_mass: torch.Tensor,
        same_valid: torch.Tensor,
        cross_valid: torch.Tensor,
        *,
        rho: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        same_desc = candidate_descriptor(same_flow, self.eps)
        same_input = torch.cat(
            (q_ctrl, same_desc, same_valid.to(q_ctrl.dtype).unsqueeze(-1)), dim=-1
        )
        same_learned = torch.sigmoid(self.same_admission(same_input).squeeze(-1))
        same_effective = (0.8 * (1.0 - float(rho)) + float(rho) * same_learned)
        same_effective = same_effective * same_valid.to(same_effective.dtype)

        cross_desc = candidate_descriptor(cross_flow, self.eps)
        cross_features = torch.cat(
            (
                q_ctrl,
                cross_desc,
                null_source_mass.detach().unsqueeze(-1),
                cross_valid.to(q_ctrl.dtype).unsqueeze(-1),
            ),
            dim=-1,
        )
        cross_learned = torch.sigmoid(self.cross_admission(cross_features).squeeze(-1))
        cross_effective = (0.2 * (1.0 - float(rho)) + float(rho) * cross_learned)
        cross_effective = cross_effective * cross_valid.to(cross_effective.dtype)
        return same_effective, cross_effective
