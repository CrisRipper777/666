from __future__ import annotations

from collections.abc import Mapping
import math

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from .flowmag_components.adapters import ModalityStem
from .flowmag_components.auxiliary import FlowMAGSupervisedAuxiliary
from .flowmag_components.layer import FlowMAGLayer


def _probability(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be strictly between 0 and 1, got {value}")
    return value


class Model(nn.Module):
    """FlowMAG V0.2.1 with explicit modality priors and graph-enriched state."""

    requires_full_lp_sampler_depth = True

    def __init__(self, cfg, data_info: Mapping[str, int | None]):
        super().__init__()
        model_cfg = cfg.model
        task_cfg = cfg.task
        self.input_dim = int(data_info["input_dim"])
        self.text_dim = int(data_info.get("text_dim", 0) or 0)
        self.visual_dim = int(data_info.get("visual_dim", 0) or 0)
        if self.text_dim <= 0 or self.visual_dim <= 0:
            raise ValueError(
                "flowmag requires positive text_dim and visual_dim; "
                f"got text_dim={self.text_dim}, visual_dim={self.visual_dim}"
            )
        if self.text_dim + self.visual_dim != self.input_dim:
            raise ValueError(
                "flowmag expects baseline's [text, visual] feature layout: "
                f"input_dim={self.input_dim}, text_dim+visual_dim={self.text_dim + self.visual_dim}"
            )

        self.hidden_dim = int(model_cfg.get("hidden_dim", 256))
        self.num_layers = int(model_cfg.get("num_layers", 2))
        self.prior_dim = int(model_cfg.get("prior_dim_per_modality", 64))
        self.graph_dim = int(model_cfg.get("graph_block_dim", 128))
        self.relation_dim = int(model_cfg.get("relation_dim", 64))
        self.router_dim = int(model_cfg.get("router_dim", 64))
        self.adapter_rank = int(model_cfg.get("adapter_rank", 16))
        self.cross_heads = int(model_cfg.get("cross_heads", 2))
        self.ffn_multiplier = int(model_cfg.get("ffn_multiplier", 2))
        positive = {
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "prior_dim_per_modality": self.prior_dim,
            "graph_block_dim": self.graph_dim,
            "relation_dim": self.relation_dim,
            "router_dim": self.router_dim,
            "adapter_rank": self.adapter_rank,
            "cross_heads": self.cross_heads,
            "ffn_multiplier": self.ffn_multiplier,
        }
        invalid = {key: value for key, value in positive.items() if value <= 0}
        if invalid:
            raise ValueError(f"FlowMAG dimensions must be positive: {invalid}")
        if self.hidden_dim % self.cross_heads:
            raise ValueError("cross_heads must divide hidden_dim")

        self.dropout = float(model_cfg.get("dropout", 0.2))
        self.attention_dropout = float(model_cfg.get("attention_dropout", 0.0))
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.attention_dropout != 0.0:
            raise ValueError("FlowMAG V0.2.1 requires attention_dropout=0")
        self.eps = float(model_cfg.get("eps", 1e-8))
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("eps must be positive")
        self.detach_candidate_descriptor = bool(model_cfg.get("detach_candidate_descriptor", True))
        if not self.detach_candidate_descriptor:
            raise ValueError("V0.2.1 requires detach_candidate_descriptor=true")
        self.null_cardinality_correction = bool(model_cfg.get("null_cardinality_correction", True))
        self.same_update_scale = float(model_cfg.get("same_update_scale", 0.1))
        self.cross_update_scale = float(model_cfg.get("cross_update_scale", 0.1))
        if (
            not math.isfinite(self.same_update_scale)
            or not math.isfinite(self.cross_update_scale)
            or self.same_update_scale < 0
            or self.cross_update_scale < 0
        ):
            raise ValueError("update scales must be non-negative")

        self.warmup_ratio = float(model_cfg.get("warmup_ratio", 0.05))
        self.controller_ramp_ratio = float(model_cfg.get("controller_ramp_ratio", 0.10))
        self.router_temp_start = float(model_cfg.get("router_temp_start", 2.0))
        self.router_temp_end = float(model_cfg.get("router_temp_end", 1.0))
        self.router_temp_anneal_end_ratio = float(model_cfg.get("router_temp_anneal_end_ratio", 0.60))
        self.init_smooth_prob = _probability(model_cfg.get("init_smooth_prob", 0.75), "init_smooth_prob")
        self.init_same_admission = _probability(model_cfg.get("init_same_admission", 0.80), "init_same_admission")
        self.init_cross_admission = _probability(model_cfg.get("init_cross_admission", 0.20), "init_cross_admission")
        if (
            not math.isfinite(self.warmup_ratio)
            or not math.isfinite(self.controller_ramp_ratio)
            or self.warmup_ratio < 0
            or self.controller_ramp_ratio < 0
        ):
            raise ValueError("warmup_ratio and controller_ramp_ratio must be non-negative")
        if not math.isfinite(self.warmup_ratio + self.controller_ramp_ratio) or self.warmup_ratio + self.controller_ramp_ratio > 1.0:
            raise ValueError("warmup_ratio + controller_ramp_ratio must be <= 1")
        if not 0.0 < self.router_temp_start or not 0.0 < self.router_temp_end:
            raise ValueError("router temperatures must be positive")
        if not 0.0 < self.router_temp_anneal_end_ratio <= 1.0:
            raise ValueError("router_temp_anneal_end_ratio must be in (0, 1]")
        self.lambda_prior = float(model_cfg.get("lambda_prior", 0.05))
        self.lambda_graph_start = float(model_cfg.get("lambda_graph_start", 0.10))
        self.graph_aux_decay_end_ratio = float(model_cfg.get("graph_aux_decay_end_ratio", 0.40))
        self.lambda_router_z = float(model_cfg.get("lambda_router_z", 1e-3))
        self.collect_diagnostics = bool(model_cfg.get("collect_diagnostics", False))
        if (
            not math.isfinite(self.lambda_prior)
            or not math.isfinite(self.lambda_graph_start)
            or not math.isfinite(self.lambda_router_z)
            or self.lambda_prior < 0
            or self.lambda_graph_start < 0
            or self.lambda_router_z < 0
        ):
            raise ValueError("auxiliary loss weights must be non-negative")
        if not 0.0 < self.graph_aux_decay_end_ratio <= 1.0:
            raise ValueError("graph_aux_decay_end_ratio must be in (0, 1]")

        self.total_epochs = max(int(task_cfg.get("epochs", 1)), 1)
        self.task = str(task_cfg.get("name", "nc")).lower()
        if self.task not in {"nc", "lp"}:
            raise ValueError(f"flowmag supports task='nc' or 'lp', got {self.task!r}")
        self.stem_text = ModalityStem(self.text_dim, self.hidden_dim)
        self.stem_image = ModalityStem(self.visual_dim, self.hidden_dim)
        self.missing_text = nn.Parameter(torch.empty(self.hidden_dim))
        self.missing_image = nn.Parameter(torch.empty(self.hidden_dim))
        nn.init.normal_(self.missing_text, std=0.02)
        nn.init.normal_(self.missing_image, std=0.02)

        self.layers = nn.ModuleList([
            FlowMAGLayer(
                hidden_dim=self.hidden_dim,
                relation_dim=self.relation_dim,
                router_dim=self.router_dim,
                adapter_rank=self.adapter_rank,
                cross_heads=self.cross_heads,
                ffn_multiplier=self.ffn_multiplier,
                dropout=self.dropout,
                attention_dropout=self.attention_dropout,
                null_cardinality_correction=self.null_cardinality_correction,
                tau_start=self.router_temp_start,
                init_smooth_prob=self.init_smooth_prob,
                init_same_admission=self.init_same_admission,
                init_cross_admission=self.init_cross_admission,
                eps=self.eps,
                same_update_scale=self.same_update_scale,
                cross_update_scale=self.cross_update_scale,
            )
            for _ in range(self.num_layers)
        ])

        self.prior_text_projection = nn.Linear(self.hidden_dim, self.prior_dim)
        self.prior_image_projection = nn.Linear(self.hidden_dim, self.prior_dim)
        self.prior_text_norm = nn.LayerNorm(self.prior_dim)
        self.prior_image_norm = nn.LayerNorm(self.prior_dim)
        self.graph_fuse = nn.Sequential(
            nn.Linear(2 * self.hidden_dim + 2, self.graph_dim),
            nn.GELU(),
            nn.Linear(self.graph_dim, self.graph_dim),
        )
        self.graph_norm = nn.LayerNorm(self.graph_dim)
        self.out_dim = 2 * self.prior_dim + self.graph_dim
        self.supervised_auxiliary = FlowMAGSupervisedAuxiliary(
            self.task,
            self.prior_dim,
            self.graph_dim,
            data_info.get("num_classes"),
            self.lambda_prior,
            self.lambda_graph_start,
        )
        self.set_epoch(0)

    def _progress(self, epoch: int) -> float:
        return min(max(float(epoch) / float(self.total_epochs), 0.0), 1.0)

    def set_epoch(self, epoch: int) -> None:
        """Update warmup/ramp, temperature, and graph-head schedules."""
        self.current_epoch = int(epoch)
        progress = self._progress(epoch)
        warm_end = self.warmup_ratio
        ramp_end = self.warmup_ratio + self.controller_ramp_ratio
        if progress + 1e-12 < warm_end:
            self.phase = "A"
            self.rho = 0.0
        elif self.controller_ramp_ratio > 0 and progress + 1e-12 < ramp_end:
            self.phase = "B"
            self.rho = min(max((progress - warm_end) / self.controller_ramp_ratio, 0.0), 1.0)
        else:
            self.phase = "C"
            self.rho = 1.0

        temp_fraction = min(progress / self.router_temp_anneal_end_ratio, 1.0)
        self.tau = self.router_temp_start + temp_fraction * (
            self.router_temp_end - self.router_temp_start
        )
        graph_fraction = max(1.0 - progress / self.graph_aux_decay_end_ratio, 0.0)
        self.lambda_graph_current = self.lambda_graph_start * graph_fraction
        self.lambda_router_z_effective = self.lambda_router_z * self.rho

    def checkpoint_eligible(self, epoch: int) -> bool:
        """Only allow validation selection after controller ramp completion."""
        return self._progress(epoch) + 1e-12 >= self.warmup_ratio + self.controller_ramp_ratio

    def restore_checkpoint_epoch(self, epoch: int) -> None:
        """Restore non-parameter schedule state for the selected checkpoint."""
        self.set_epoch(epoch)

    def _split_and_stem(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(f"flowmag expected x shape [N, {self.input_dim}], got {tuple(x.shape)}")
        text_raw = x[:, : self.text_dim]
        image_raw = x[:, self.text_dim : self.text_dim + self.visual_dim]
        text = torch.nan_to_num(text_raw, nan=0.0, posinf=0.0, neginf=0.0)
        image = torch.nan_to_num(image_raw, nan=0.0, posinf=0.0, neginf=0.0)
        text_valid = torch.isfinite(text_raw).all(dim=-1) & (text.abs().sum(dim=-1) > 0)
        image_valid = torch.isfinite(image_raw).all(dim=-1) & (image.abs().sum(dim=-1) > 0)
        u_text = self.stem_text(text)
        u_image = self.stem_image(image)
        u_text = torch.where(text_valid.unsqueeze(-1), u_text, self.missing_text.unsqueeze(0))
        u_image = torch.where(image_valid.unsqueeze(-1), u_image, self.missing_image.unsqueeze(0))
        return u_text, u_image, text_valid, image_valid, text, image

    def _compose_output(
        self,
        u_text: torch.Tensor,
        u_image: torch.Tensor,
        h_text: torch.Tensor,
        h_image: torch.Tensor,
        text_valid: torch.Tensor,
        image_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r_text = self.prior_text_norm(self.prior_text_projection(u_text))
        r_image = self.prior_image_norm(self.prior_image_projection(u_image))
        graph_input = torch.cat(
            (
                h_text,
                h_image,
                text_valid.to(h_text.dtype).unsqueeze(-1),
                image_valid.to(h_image.dtype).unsqueeze(-1),
            ),
            dim=-1,
        )
        graph_block = self.graph_norm(self.graph_fuse(graph_input))
        return torch.cat((r_text, r_image, graph_block), dim=-1), r_text, r_image, graph_block

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None = None):
        if edge_index is None:
            raise ValueError("flowmag requires edge_index")
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        u_text, u_image, text_valid, image_valid, _, _ = self._split_and_stem(x)
        h_text, h_image = u_text, u_image
        diagnostics: dict[str, object] = {"layers": []} if self.collect_diagnostics else {}
        router_terms: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in self.layers:
            h_text, h_image, layer_diag, layer_router_terms = layer(
                h_text,
                h_image,
                text_valid,
                image_valid,
                edge_index,
                rho=self.rho,
                tau=self.tau,
                collect_diagnostics=self.collect_diagnostics,
            )
            if self.collect_diagnostics:
                diagnostics["layers"].append(layer_diag)
            router_terms.extend(layer_router_terms)

        z, r_text, r_image, graph_block = self._compose_output(
            u_text, u_image, h_text, h_image, text_valid, image_valid
        )
        if router_terms:
            total = torch.stack([term_sum for term_sum, _ in router_terms]).sum()
            count = torch.stack([term_count for _, term_count in router_terms]).sum()
            router_z = total / count.clamp_min(1).to(total.dtype)
        else:
            router_z = z.sum() * 0.0
        aux_loss = self.lambda_router_z_effective * router_z
        diagnostics.update({
            "phase": self.phase,
            "rho": float(self.rho),
            "tau": float(self.tau),
            "lambda_graph": float(self.lambda_graph_current),
            "lambda_router_z_effective": float(self.lambda_router_z_effective),
        })
        if self.collect_diagnostics:
            top_metrics = torch.stack((
                r_text.detach().float().norm(dim=-1).mean(),
                r_image.detach().float().norm(dim=-1).mean(),
                graph_block.detach().float().norm(dim=-1).mean(),
                router_z.detach().float(),
            )).cpu().tolist()
            diagnostics.update({
                "prior_text_norm": float(top_metrics[0]),
                "prior_image_norm": float(top_metrics[1]),
                "graph_block_norm": float(top_metrics[2]),
                "router_z_raw": float(top_metrics[3]),
            })
        if self.training:
            diagnostics["_supervised"] = {
                "prior_text": r_text,
                "prior_image": r_image,
                "graph": graph_block,
                "text_valid": text_valid,
                "image_valid": image_valid,
            }
        return z, None, None, aux_loss, diagnostics

    def gradient_diagnostics(self) -> dict[str, float | None]:
        """Return per-component gradient L2 norms without changing gradients."""
        groups: dict[str, list[nn.Parameter]] = {
            "stem_text": list(self.stem_text.parameters()),
            "stem_image": list(self.stem_image.parameters()),
            "prior_text_projection": list(self.prior_text_projection.parameters()),
            "prior_image_projection": list(self.prior_image_projection.parameters()),
            "graph_fuse": list(self.graph_fuse.parameters()),
        }
        for layer_index, layer in enumerate(self.layers):
            for group_name in (
                "same_relation", "smooth", "contrast", "cross_attention",
                "context", "controller", "ffn",
            ):
                groups[f"layers.{layer_index}.{group_name}"] = list(
                    getattr(layer, group_name).parameters()
                )

        result: dict[str, float | None] = {}
        active_names: list[str] = []
        squared_norms: list[torch.Tensor] = []
        for name, parameters in groups.items():
            grad_sums = [
                parameter.grad.detach().float().square().sum()
                for parameter in parameters
                if parameter.grad is not None
            ]
            if not grad_sums:
                result[name] = None
                continue
            active_names.append(name)
            squared_norms.append(torch.stack(grad_sums).sum())
        if squared_norms:
            norms_cpu = torch.stack(squared_norms).sqrt().detach().cpu().tolist()
            for name, norm in zip(active_names, norms_cpu, strict=True):
                result[name] = float(norm) if math.isfinite(float(norm)) else None
        return result

    def compute_supervised_aux_loss(
        self,
        *,
        task: str,
        aux_info: dict,
        labels: torch.Tensor | None = None,
        node_index: torch.Tensor | None = None,
        edge_label_index: torch.Tensor | None = None,
        edge_label: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.supervised_auxiliary.compute(
            task=task,
            aux_info=aux_info,
            labels=labels,
            node_index=node_index,
            edge_label_index=edge_label_index,
            edge_label=edge_label,
            lambda_graph_current=self.lambda_graph_current,
        )

    def _finalize_cpu_batch(
        self,
        u_text: torch.Tensor,
        u_image: torch.Tensor,
        h_text: torch.Tensor,
        h_image: torch.Tensor,
        text_valid: torch.Tensor,
        image_valid: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        z, _, _, _ = self._compose_output(
            u_text.to(device),
            u_image.to(device),
            h_text.to(device),
            h_image.to(device),
            text_valid.to(device),
            image_valid.to(device),
        )
        return z.detach().cpu()

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        device: torch.device | None = None,
        batch_size: int = 65536,
    ) -> torch.Tensor:
        """Exact layerwise inference using full one-hop neighborhoods per layer."""
        if edge_index is None:
            raise ValueError("flowmag requires edge_index")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if device is None:
            device = next(self.parameters()).device
        device = torch.device(device)
        self.eval()
        x_cpu = x.detach().cpu()
        edge_cpu = edge_index.detach().cpu().long().contiguous()
        num_nodes = int(x_cpu.size(0))
        u_text = torch.empty((num_nodes, self.hidden_dim), dtype=self.missing_text.dtype, device="cpu")
        u_image = torch.empty_like(u_text)
        text_valid = torch.empty(num_nodes, dtype=torch.bool)
        image_valid = torch.empty(num_nodes, dtype=torch.bool)
        for start in range(0, num_nodes, batch_size):
            stop = min(start + batch_size, num_nodes)
            values = x_cpu[start:stop].to(device)
            ut, ui, mt, mi, _, _ = self._split_and_stem(values)
            u_text[start:stop] = ut.cpu()
            u_image[start:stop] = ui.cpu()
            text_valid[start:stop] = mt.cpu()
            image_valid[start:stop] = mi.cpu()

        h_text, h_image = u_text, u_image
        graph = Data(edge_index=edge_cpu, num_nodes=num_nodes)
        all_nodes = torch.arange(num_nodes, dtype=torch.long)
        for layer in self.layers:
            loader = NeighborLoader(
                graph,
                input_nodes=all_nodes,
                num_neighbors=[-1],
                batch_size=int(batch_size),
                shuffle=False,
            )
            next_text = torch.empty_like(h_text)
            next_image = torch.empty_like(h_image)
            for batch in loader:
                n_id = batch.n_id
                size = int(batch.batch_size)
                t = h_text[n_id].to(device)
                i = h_image[n_id].to(device)
                mt = text_valid[n_id].to(device)
                mi = image_valid[n_id].to(device)
                edge_local = batch.edge_index.to(device)
                out_t, out_i, _, _ = layer(
                    t,
                    i,
                    mt,
                    mi,
                    edge_local,
                    rho=self.rho,
                    tau=self.tau,
                    collect_diagnostics=False,
                )
                seed_nodes = n_id[:size]
                next_text[seed_nodes] = out_t[:size].cpu()
                next_image[seed_nodes] = out_i[:size].cpu()
            h_text, h_image = next_text, next_image

        result = torch.empty((num_nodes, self.out_dim), dtype=self.missing_text.dtype)
        for start in range(0, num_nodes, batch_size):
            stop = min(start + batch_size, num_nodes)
            result[start:stop] = self._finalize_cpu_batch(
                u_text[start:stop],
                u_image[start:stop],
                h_text[start:stop],
                h_image[start:stop],
                text_valid[start:stop],
                image_valid[start:stop],
                device,
            )
        return result
