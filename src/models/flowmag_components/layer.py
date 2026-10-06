from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .adapters import AdaptedLinear
from .attention import CrossModalAttention, SameModalRelation
from .controller import FactorizedController, MaskAwareContext


def _rms(value: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor | None:
    if value.numel() == 0:
        return None
    if mask is not None and not bool(mask.any()):
        return None
    row_count = int(value.size(0))
    row_square_sum = torch.zeros((), dtype=torch.float32, device=value.device)
    selected_rows = torch.zeros_like(row_square_sum)
    # Chunking avoids a second full-size float buffer for large full-graph
    # tensors while keeping the same node-weighted RMS definition.
    chunk_size = 8192
    for start in range(0, row_count, chunk_size):
        stop = min(start + chunk_size, row_count)
        chunk = value[start:stop].detach().float().reshape(stop - start, -1)
        per_row_square_mean = chunk.square().mean(dim=-1)
        if mask is None:
            row_square_sum = row_square_sum + per_row_square_mean.sum()
            selected_rows = selected_rows + (stop - start)
        else:
            weights = mask[start:stop].to(per_row_square_mean.dtype)
            row_square_sum = row_square_sum + (per_row_square_mean * weights).sum()
            selected_rows = selected_rows + weights.sum()
    return (row_square_sum / selected_rows.clamp_min(1)).sqrt()


def _mean(value: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor | None:
    if mask is not None:
        value = value[mask]
    if value.numel() == 0:
        return None
    return value.detach().float().mean()


class FlowMAGLayer(nn.Module):
    """PreNorm same-modal and directional cross-modal update layer."""

    def __init__(
        self,
        hidden_dim: int,
        relation_dim: int,
        router_dim: int,
        adapter_rank: int,
        cross_heads: int,
        ffn_multiplier: int,
        dropout: float,
        attention_dropout: float,
        null_cardinality_correction: bool,
        tau_start: float,
        init_smooth_prob: float,
        init_same_admission: float,
        init_cross_admission: float,
        eps: float,
        same_update_scale: float,
        cross_update_scale: float,
    ):
        super().__init__()
        self.pre_norm = nn.LayerNorm(hidden_dim)
        self.same_relation = SameModalRelation(hidden_dim, relation_dim, adapter_rank)
        self.smooth = AdaptedLinear(hidden_dim, hidden_dim, adapter_rank, ("text", "image"))
        self.contrast = AdaptedLinear(
            hidden_dim, hidden_dim, adapter_rank, ("text", "image"), bias=False
        )
        self.cross_attention = CrossModalAttention(
            hidden_dim,
            cross_heads,
            adapter_rank,
            null_cardinality_correction=null_cardinality_correction,
            attention_dropout=attention_dropout,
        )
        self.context = MaskAwareContext(hidden_dim, router_dim)
        self.controller = FactorizedController(
            hidden_dim,
            router_dim,
            tau_start,
            init_smooth_prob,
            init_same_admission,
            init_cross_admission,
            eps=eps,
        )
        ffn_hidden = int(ffn_multiplier) * hidden_dim
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, hidden_dim),
        )
        self.update_dropout = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)
        self.same_update_scale = float(same_update_scale)
        self.cross_update_scale = float(cross_update_scale)

    def _one_modality(
        self,
        *,
        name: str,
        modality_index: int,
        target: torch.Tensor,
        same_mu: torch.Tensor,
        same_valid: torch.Tensor,
        smooth: torch.Tensor,
        contrast: torch.Tensor,
        cross: torch.Tensor,
        cross_valid: torch.Tensor,
        null_mass: torch.Tensor,
        cross_attention_entropy: torch.Tensor | None,
        normalized_cross_attention_entropy: torch.Tensor | None,
        joint_context: torch.Tensor,
        rho: float,
        tau: float,
        collect_diagnostics: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        q_ctrl = self.controller.target_context(target, joint_context, modality_index)
        frequency_logits, pi = self.controller.frequency(
            q_ctrl, smooth, contrast, tau=tau, rho=rho
        )
        same_flow = pi[:, 0:1] * smooth + pi[:, 1:2] * contrast
        same_admission, cross_admission = self.controller.admissions(
            q_ctrl,
            same_flow,
            cross,
            null_mass,
            same_valid,
            cross_valid,
            rho=rho,
        )
        same_delta = self.same_update_scale * same_admission.unsqueeze(-1) * same_flow
        cross_delta = self.cross_update_scale * cross_admission.unsqueeze(-1) * cross
        graph_delta = same_delta + cross_delta
        middle = target + self.update_dropout(graph_delta)
        ffn_raw = self.ffn(self.ffn_norm(middle))
        ffn_delta = self.ffn_dropout(ffn_raw)
        result = middle + ffn_delta

        if collect_diagnostics:
            smooth_contrast_cos = F.cosine_similarity(smooth, contrast, dim=-1, eps=1e-8)
            state_input_rms = _rms(target)
            middle_rms = _rms(middle)
            same_update_rms = _rms(same_delta, same_valid)
            cross_update_rms = _rms(cross_delta, cross_valid)
            graph_update_rms = _rms(graph_delta)
            ffn_update_rms = _rms(ffn_delta)
            diag = {
                # Compatibility keys retain their historical all-node semantics.
                "pi_smooth": _mean(pi[:, 0]),
                "pi_contrast": _mean(pi[:, 1]),
                "same_admission": _mean(same_admission),
                "cross_admission": _mean(cross_admission),
                "smooth_rms": _rms(smooth),
                "contrast_rms": _rms(contrast),
                "cross_rms": _rms(cross),
                "update_rms": _rms(graph_delta),
                "state_rms": _rms(result),
                "smooth_contrast_cosine": _mean(smooth_contrast_cos, same_valid),
                # Availability and conditional routing are reported separately.
                "same_valid_rate": _mean(same_valid.float()),
                "cross_valid_rate": _mean(cross_valid.float()),
                "no_same_source_rate": _mean((~same_valid).float()),
                "no_cross_source_rate": _mean((~cross_valid).float()),
                "pi_smooth_given_same_valid": _mean(pi[:, 0], same_valid),
                "pi_contrast_given_same_valid": _mean(pi[:, 1], same_valid),
                "same_admission_given_valid": _mean(same_admission, same_valid),
                "smooth_rms_given_valid": _rms(smooth, same_valid),
                "contrast_rms_given_valid": _rms(contrast, same_valid),
                "smooth_contrast_cosine_given_valid": _mean(smooth_contrast_cos, same_valid),
                "cross_admission_given_valid": _mean(cross_admission, cross_valid),
                "cross_rms_given_valid": _rms(cross, cross_valid),
                "null_mass_given_cross_valid": _mean(null_mass, cross_valid),
                "cross_attention_entropy_given_valid": _mean(cross_attention_entropy, cross_valid)
                if cross_attention_entropy is not None else None,
                "normalized_cross_attention_entropy": _mean(
                    normalized_cross_attention_entropy, cross_valid
                ) if normalized_cross_attention_entropy is not None else None,
                "normalized_cross_attention_entropy_given_valid": _mean(
                    normalized_cross_attention_entropy, cross_valid
                ) if normalized_cross_attention_entropy is not None else None,
                "same_update_rms": same_update_rms,
                "cross_update_rms": cross_update_rms,
                "graph_update_rms": graph_update_rms,
                "ffn_update_rms": ffn_update_rms,
                "state_input_rms": state_input_rms,
                "middle_state_rms": middle_rms,
                "state_output_rms": _rms(result),
                "same_update_ratio": same_update_rms / (state_input_rms + self.controller.eps)
                if same_update_rms is not None and state_input_rms is not None else None,
                "cross_update_ratio": cross_update_rms / (state_input_rms + self.controller.eps)
                if cross_update_rms is not None and state_input_rms is not None else None,
                "graph_update_ratio": graph_update_rms / (state_input_rms + self.controller.eps)
                if graph_update_rms is not None and state_input_rms is not None else None,
                "ffn_update_ratio": ffn_update_rms / (middle_rms + self.controller.eps)
                if ffn_update_rms is not None and middle_rms is not None else None,
            }
        else:
            diag = {}
        log_partition = torch.logsumexp(frequency_logits, dim=-1)
        router_z_sum = (log_partition.square() * same_valid.to(log_partition.dtype)).sum()
        router_z_count = same_valid.sum()
        return result, router_z_sum, router_z_count, diag, same_flow

    def forward(
        self,
        h_text: torch.Tensor,
        h_image: torch.Tensor,
        text_valid: torch.Tensor,
        image_valid: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        rho: float,
        tau: float,
        collect_diagnostics: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, object], list[tuple[torch.Tensor, torch.Tensor]]]:
        pre_text = self.pre_norm(h_text)
        pre_image = self.pre_norm(h_image)
        joint_context = self.context(pre_text, pre_image, text_valid, image_valid)

        mu_text, same_valid_text, _ = self.same_relation(
            pre_text, edge_index, text_valid, "text"
        )
        mu_image, same_valid_image, _ = self.same_relation(
            pre_image, edge_index, image_valid, "image"
        )
        smooth_text = self.smooth(mu_text, "text")
        smooth_image = self.smooth(mu_image, "image")
        contrast_text = self.contrast(pre_text - mu_text, "text")
        contrast_image = self.contrast(pre_image - mu_image, "image")
        smooth_text = torch.where(same_valid_text.unsqueeze(-1), smooth_text, torch.zeros_like(smooth_text))
        smooth_image = torch.where(same_valid_image.unsqueeze(-1), smooth_image, torch.zeros_like(smooth_image))
        contrast_text = torch.where(same_valid_text.unsqueeze(-1), contrast_text, torch.zeros_like(contrast_text))
        contrast_image = torch.where(same_valid_image.unsqueeze(-1), contrast_image, torch.zeros_like(contrast_image))

        attention_text = self.cross_attention(
            pre_text, pre_image, edge_index, image_valid, "i2t",
            collect_diagnostics=collect_diagnostics,
            return_attention_diagnostics=collect_diagnostics,
        )
        attention_image = self.cross_attention(
            pre_image, pre_text, edge_index, text_valid, "t2i",
            collect_diagnostics=collect_diagnostics,
            return_attention_diagnostics=collect_diagnostics,
        )
        if collect_diagnostics:
            cross_text, null_i2t, _, cross_valid_text, entropy_i2t, normalized_entropy_i2t = attention_text
            cross_image, null_t2i, _, cross_valid_image, entropy_t2i, normalized_entropy_t2i = attention_image
        else:
            cross_text, null_i2t, _, cross_valid_text = attention_text
            cross_image, null_t2i, _, cross_valid_image = attention_image
            entropy_i2t = normalized_entropy_i2t = None
            entropy_t2i = normalized_entropy_t2i = None

        text_next, text_z_sum, text_z_count, text_diag, _ = self._one_modality(
            name="text",
            modality_index=0,
            target=h_text,
            same_mu=mu_text,
            same_valid=same_valid_text,
            smooth=smooth_text,
            contrast=contrast_text,
            cross=cross_text,
            cross_valid=cross_valid_text,
            null_mass=null_i2t,
            cross_attention_entropy=entropy_i2t,
            normalized_cross_attention_entropy=normalized_entropy_i2t,
            joint_context=joint_context,
            rho=rho,
            tau=tau,
            collect_diagnostics=collect_diagnostics,
        )
        image_next, image_z_sum, image_z_count, image_diag, _ = self._one_modality(
            name="image",
            modality_index=1,
            target=h_image,
            same_mu=mu_image,
            same_valid=same_valid_image,
            smooth=smooth_image,
            contrast=contrast_image,
            cross=cross_image,
            cross_valid=cross_valid_image,
            null_mass=null_t2i,
            cross_attention_entropy=entropy_t2i,
            normalized_cross_attention_entropy=normalized_entropy_t2i,
            joint_context=joint_context,
            rho=rho,
            tau=tau,
            collect_diagnostics=collect_diagnostics,
        )
        diagnostics: dict[str, object] = {"text": text_diag, "image": image_diag} if collect_diagnostics else {}
        return text_next, image_next, diagnostics, [
            (text_z_sum, text_z_count), (image_z_sum, image_z_count)
        ]
