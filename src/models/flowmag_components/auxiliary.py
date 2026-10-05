from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FlowMAGSupervisedAuxiliary(nn.Module):
    """Train-label-only heads kept outside the encoder's unsupervised loss."""

    def __init__(
        self,
        task: str,
        prior_dim: int,
        graph_dim: int,
        num_classes: int | None,
        lambda_prior: float,
        lambda_graph_start: float,
    ):
        super().__init__()
        self.task = str(task).lower()
        self.lambda_prior = float(lambda_prior)
        self.lambda_graph_start = float(lambda_graph_start)
        if self.task == "nc":
            if num_classes is None or int(num_classes) < 2:
                raise ValueError("NC auxiliary heads require num_classes >= 2")
            self.prior_text_node = nn.Linear(prior_dim, int(num_classes))
            self.prior_image_node = nn.Linear(prior_dim, int(num_classes))
            self.graph_node = nn.Linear(graph_dim, int(num_classes))
        elif self.task == "lp":
            self.prior_text_pair = nn.Linear(prior_dim, 1)
            self.prior_image_pair = nn.Linear(prior_dim, 1)
            self.graph_pair = nn.Linear(graph_dim, 1)
        else:
            raise ValueError(f"FlowMAG supports task='nc' or 'lp', got {task!r}")

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @staticmethod
    def _zero(reference: torch.Tensor) -> torch.Tensor:
        return reference.sum() * 0.0

    def compute(
        self,
        *,
        task: str,
        aux_info: dict,
        labels: torch.Tensor | None = None,
        node_index: torch.Tensor | None = None,
        edge_label_index: torch.Tensor | None = None,
        edge_label: torch.Tensor | None = None,
        lambda_graph_current: float,
    ) -> dict[str, torch.Tensor]:
        task = str(task).lower()
        if task != self.task:
            raise ValueError(f"auxiliary heads were built for {self.task}, got {task}")
        state = aux_info.get("_supervised")
        if not isinstance(state, dict):
            raise ValueError("aux_info lacks training-only '_supervised' representations")
        reference = state["graph"]
        zero = self._zero(reference)
        prior_text = zero
        prior_image = zero
        graph = zero

        if task == "nc":
            if labels is None or node_index is None:
                raise ValueError("NC auxiliary loss requires train labels and node_index")
            node_index = torch.as_tensor(node_index, dtype=torch.long, device=reference.device).reshape(-1)
            labels = torch.as_tensor(labels, dtype=torch.long, device=reference.device).reshape(-1)
            if labels.numel() == state["graph"].size(0):
                selected_labels = labels[node_index]
            elif labels.numel() == node_index.numel():
                selected_labels = labels
            else:
                raise ValueError("NC labels must align with node_index or the full local node matrix")
            present_losses = []
            for modality, head_name, loss_name in (
                ("text", "prior_text_node", "prior_text"),
                ("image", "prior_image_node", "prior_image"),
            ):
                valid = state[f"{modality}_valid"][node_index]
                if bool(valid.any()):
                    logits = getattr(self, head_name)(state[f"prior_{modality}"][node_index][valid])
                    loss_value = F.cross_entropy(logits, selected_labels[valid])
                    present_losses.append((loss_name, loss_value))
            divisor = max(len(present_losses), 1)
            for name, loss_value in present_losses:
                weighted = self.lambda_prior * loss_value / divisor
                if name == "prior_text":
                    prior_text = weighted
                else:
                    prior_image = weighted
            graph_logits = self.graph_node(state["graph"][node_index])
            graph = float(lambda_graph_current) * F.cross_entropy(graph_logits, selected_labels)

        else:
            if edge_label_index is None or edge_label is None:
                raise ValueError("LP auxiliary loss requires current-batch edge_label_index and edge_label")
            edge_label_index = torch.as_tensor(
                edge_label_index, dtype=torch.long, device=reference.device
            )
            edge_label = torch.as_tensor(
                edge_label, dtype=reference.dtype, device=reference.device
            ).reshape(-1)
            if edge_label_index.dim() != 2 or edge_label_index.size(0) != 2:
                raise ValueError("edge_label_index must have shape [2, num_pairs]")
            if edge_label_index.size(1) != edge_label.numel():
                raise ValueError("edge_label_index and edge_label must have matching pair counts")
            src, dst = edge_label_index
            present_losses = []
            for modality, head_name, loss_name in (
                ("text", "prior_text_pair", "prior_text"),
                ("image", "prior_image_pair", "prior_image"),
            ):
                valid = state[f"{modality}_valid"][src] & state[f"{modality}_valid"][dst]
                if bool(valid.any()):
                    pair = state[f"prior_{modality}"][src] * state[f"prior_{modality}"][dst]
                    logits = getattr(self, head_name)(pair[valid]).reshape(-1)
                    present_losses.append((loss_name, F.binary_cross_entropy_with_logits(logits, edge_label[valid])))
            divisor = max(len(present_losses), 1)
            for name, loss_value in present_losses:
                weighted = self.lambda_prior * loss_value / divisor
                if name == "prior_text":
                    prior_text = weighted
                else:
                    prior_image = weighted
            graph_pair = state["graph"][src] * state["graph"][dst]
            graph_logits = self.graph_pair(graph_pair).reshape(-1)
            graph = float(lambda_graph_current) * F.binary_cross_entropy_with_logits(graph_logits, edge_label)

        total = prior_text + prior_image + graph
        return {
            "loss": total,
            "prior_text": prior_text,
            "prior_image": prior_image,
            "graph": graph,
        }
