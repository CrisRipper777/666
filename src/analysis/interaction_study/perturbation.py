from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .diagnostic_host import FeatureOnlyMultimodalDiagnosticHost
from .splits import StudySplit
from .statistics import block_bootstrap_mean_ci, finite_summary


def equal_relative_norm(direction: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Scale a perturbation to the target state's norm, preserving its direction."""
    direction_norm = direction.norm(dim=-1, keepdim=True)
    target_norm = target.norm(dim=-1, keepdim=True)
    return direction / (direction_norm + eps) * (target_norm + eps)


def finite_difference_loss(loss_fn, base: torch.Tensor, direction: torch.Tensor, alpha: float) -> torch.Tensor:
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    return (loss_fn(base + alpha * direction) - loss_fn(base - alpha * direction)) / (2.0 * alpha)


@dataclass
class ReceiverSources:
    receivers: torch.Tensor
    neighbor_ids: torch.Tensor
    random_ids: torch.Tensor
    counts: torch.Tensor
    neighbor_cap: int
    seed: int

    def aggregate(self, states: torch.Tensor, random_control: bool = False) -> torch.Tensor:
        ids = self.random_ids if random_control else self.neighbor_ids
        out = torch.zeros((self.receivers.numel(), states.size(1)), dtype=states.dtype)
        if ids.numel():
            rows = torch.repeat_interleave(torch.arange(self.receivers.numel()), self.counts)
            out.index_add_(0, rows, states[ids])
        return out / self.counts.clamp_min(1).to(states.dtype)[:, None]

    def select(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        positions = {int(node): row for row, node in enumerate(self.receivers.tolist())}
        rows = torch.tensor([positions[int(node)] for node in indices.detach().long().cpu().tolist()], dtype=torch.long)
        return self.neighbor_ids, self.random_ids, rows


def _adjacency(edge_index: torch.Tensor, num_nodes: int) -> list[list[int]]:
    adj: list[set[int]] = [set() for _ in range(num_nodes)]
    edge = edge_index.detach().long().cpu()
    rows = edge.T.contiguous() if edge.size(0) == 2 else edge.contiguous()
    for a, b in rows.tolist():
        if a == b:
            continue
        adj[a].add(b)
        adj[b].add(a)
    return [sorted(row) for row in adj]


def sample_receiver_sources(
    edge_index: torch.Tensor,
    receivers: torch.Tensor,
    num_nodes: int,
    neighbor_cap: int,
    seed: int,
) -> ReceiverSources:
    if neighbor_cap < 1:
        raise ValueError("neighbor_cap must be >= 1")
    receiver_ids = receivers.detach().long().cpu().unique(sorted=True)
    adj = _adjacency(edge_index, int(num_nodes))
    rng = np.random.default_rng(int(seed))
    neighbor_flat: list[int] = []
    random_flat: list[int] = []
    counts: list[int] = []
    for receiver in receiver_ids.tolist():
        neighbors = adj[receiver]
        nonneighbor_count = int(num_nodes) - 1 - len(neighbors)
        count = min(int(neighbor_cap), len(neighbors), nonneighbor_count)
        if count <= 0:
            counts.append(0)
            continue
        if len(neighbors) > count:
            actual = sorted(rng.choice(neighbors, size=count, replace=False).tolist())
        else:
            actual = neighbors
        selected_random: set[int] = set()
        neighbor_set = set(neighbors)
        attempts = 0
        while len(selected_random) < count and attempts < count * 100:
            candidates = rng.integers(0, int(num_nodes), size=max(32, 4 * (count - len(selected_random))))
            attempts += len(candidates)
            for candidate in candidates.tolist():
                if candidate != receiver and candidate not in neighbor_set:
                    selected_random.add(int(candidate))
                    if len(selected_random) == count:
                        break
        if len(selected_random) < count:
            fallback = [node for node in range(int(num_nodes)) if node != receiver and node not in neighbor_set and node not in selected_random]
            selected_random.update(fallback[: count - len(selected_random)])
        if len(selected_random) != count or len(actual) != count:
            raise RuntimeError("could not construct source-count-matched nonneighbor control")
        neighbor_flat.extend(map(int, actual))
        random_flat.extend(sorted(selected_random))
        counts.append(count)
    return ReceiverSources(
        receiver_ids,
        torch.tensor(neighbor_flat, dtype=torch.long),
        torch.tensor(random_flat, dtype=torch.long),
        torch.tensor(counts, dtype=torch.long),
        int(neighbor_cap),
        int(seed),
    )


def _target_and_other(
    h_t: torch.Tensor,
    h_i: torch.Tensor,
    modality: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if modality == "text":
        return h_t, h_i
    if modality == "image":
        return h_i, h_t
    raise ValueError(f"unknown modality {modality!r}")


def _loss_from_hidden_batched(
    host: FeatureOnlyMultimodalDiagnosticHost,
    target: torch.Tensor,
    other: torch.Tensor,
    labels: torch.Tensor,
    modality: str,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    output = []
    with torch.inference_mode():
        for start in range(0, target.size(0), int(batch_size)):
            end = min(start + int(batch_size), target.size(0))
            logits = host.logits_for_target(target[start:end].to(device), other[start:end].to(device), modality)
            output.append(F.cross_entropy(logits, labels[start:end].long().to(device), reduction="none").cpu())
    return torch.cat(output) if output else torch.empty(0)


def _perturbed_loss(
    host: FeatureOnlyMultimodalDiagnosticHost,
    target: torch.Tensor,
    other: torch.Tensor,
    labels: torch.Tensor,
    modality: str,
    alpha: float,
    direction: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    normalized = equal_relative_norm(direction, target)
    return _loss_from_hidden_batched(host, target + float(alpha) * normalized, other, labels, modality, device, batch_size)


def cross_modal_utility(
    host: FeatureOnlyMultimodalDiagnosticHost,
    h_t: torch.Tensor,
    h_i: torch.Tensor,
    aligned_t_to_i: torch.Tensor,
    aligned_i_to_t: torch.Tensor,
    labels: torch.Tensor,
    split: StudySplit,
    sources: ReceiverSources,
    alphas: list[float],
    device: torch.device,
    *,
    batch_size: int = 8192,
    bootstrap_reps: int = 500,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    val = split.original_validation
    pos = {int(node): row for row, node in enumerate(sources.receivers.tolist())}
    rows = torch.tensor([pos[int(node)] for node in val.tolist()], dtype=torch.long)
    counts = sources.counts[rows]
    source_count_match = (
        int(sources.counts.sum()) == int(sources.neighbor_ids.numel())
        and int(sources.counts.sum()) == int(sources.random_ids.numel())
    )
    if not source_count_match:
        raise AssertionError("graph and random source counts differ")
    agg_t2i_graph = sources.aggregate(aligned_t_to_i)[rows]
    agg_t2i_random = sources.aggregate(aligned_t_to_i, random_control=True)[rows]
    agg_i2t_graph = sources.aggregate(aligned_i_to_t)[rows]
    agg_i2t_random = sources.aggregate(aligned_i_to_t, random_control=True)[rows]
    target_t, other_t = _target_and_other(h_t, h_i, "text")
    target_i, other_i = _target_and_other(h_t, h_i, "image")
    target_t, other_t = target_t[val], other_t[val]
    target_i, other_i = target_i[val], other_i[val]
    y_val = labels[val]
    result_rows = []
    raw: dict[str, torch.Tensor] = {"receiver_nodes": val.clone(), "source_count": counts.clone()}

    base_t = _loss_from_hidden_batched(host, target_t, other_t, y_val, "text", device, batch_size)
    base_i = _loss_from_hidden_batched(host, target_i, other_i, y_val, "image", device, batch_size)
    for direction, target, other, graph_state, random_state, base in (
        ("T2I", target_i, other_i, agg_t2i_graph, agg_t2i_random, base_i),
        ("I2T", target_t, other_t, agg_i2t_graph, agg_i2t_random, base_t),
    ):
        graph_delta_by_alpha, random_delta_by_alpha, gain_by_alpha = [], [], []
        for alpha in alphas:
            graph_delta = _perturbed_loss(
                host, target, other, y_val, "image" if direction == "T2I" else "text", alpha,
                graph_state - target, device, batch_size,
            ) - base
            random_delta = _perturbed_loss(
                host, target, other, y_val, "image" if direction == "T2I" else "text", alpha,
                random_state - target, device, batch_size,
            ) - base
            gain = random_delta - graph_delta
            graph_delta_by_alpha.append(graph_delta)
            random_delta_by_alpha.append(random_delta)
            gain_by_alpha.append(gain)
            ci = block_bootstrap_mean_ci(gain, val, reps=bootstrap_reps, seed=seed + int(alpha * 10000))
            summary = finite_summary(gain)
            result_rows.append({
                "direction": direction,
                "alpha": float(alpha),
                "graph_gain_mean": summary["mean"],
                "graph_gain_median": summary["median"],
                "graph_gain_fraction_positive": float((gain > 0).float().mean()),
                "graph_gain_ci_low": ci["ci_low"],
                "graph_gain_ci_high": ci["ci_high"],
                "delta_ce_graph_mean": float(graph_delta.mean()),
                "delta_ce_random_mean": float(random_delta.mean()),
                "receiver_count": int(val.numel()),
                "isolated_or_zero_control_fraction": float((counts == 0).float().mean()),
                "source_count_matched": source_count_match,
                "bootstrap_unit": "validation receiver node",
            })
        raw[f"{direction}_graph_delta"] = torch.stack(graph_delta_by_alpha)
        raw[f"{direction}_random_delta"] = torch.stack(random_delta_by_alpha)
        raw[f"{direction}_graph_gain"] = torch.stack(gain_by_alpha)
    for alpha in alphas:
        g_t2i = raw["T2I_graph_gain"][alphas.index(alpha)]
        g_i2t = raw["I2T_graph_gain"][alphas.index(alpha)]
        asymmetry = g_t2i - g_i2t
        ci = block_bootstrap_mean_ci(asymmetry, val, reps=bootstrap_reps, seed=seed + 177 + int(alpha * 10000))
        summary = finite_summary(asymmetry)
        result_rows.append({
            "direction": "T2I_minus_I2T",
            "alpha": float(alpha),
            "directional_asymmetry_mean": summary["mean"],
            "directional_asymmetry_median": summary["median"],
            "directional_asymmetry_ci_low": ci["ci_low"],
            "directional_asymmetry_ci_high": ci["ci_high"],
            "receiver_count": int(val.numel()),
            "bootstrap_unit": "validation receiver node",
        })
        raw[f"asymmetry_{alpha:g}"] = asymmetry
    return result_rows, raw


def operator_response(
    host: FeatureOnlyMultimodalDiagnosticHost,
    h_t: torch.Tensor,
    h_i: torch.Tensor,
    aligned_t_to_i: torch.Tensor,
    aligned_i_to_t: torch.Tensor,
    labels: torch.Tensor,
    split: StudySplit,
    sources: ReceiverSources,
    alpha: float,
    device: torch.device,
    *,
    batch_size: int = 8192,
    bootstrap_reps: int = 500,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    result_rows: list[dict[str, Any]] = []
    raw: dict[str, torch.Tensor] = {}
    all_splits = {
        "calib": split.diagnostic_calib,
        "report": split.original_validation,
    }
    source_positions = {int(node): row for row, node in enumerate(sources.receivers.tolist())}

    for modality in ("text", "image"):
        target_all, other_all = _target_and_other(h_t, h_i, modality)
        same_all = sources.aggregate(target_all)
        cross_all = sources.aggregate(aligned_t_to_i if modality == "image" else aligned_i_to_t)
        action_deltas: dict[str, dict[str, torch.Tensor]] = {key: {} for key in all_splits}
        for split_name, indices in all_splits.items():
            rows = torch.tensor([source_positions[int(node)] for node in indices.tolist()], dtype=torch.long)
            target, other = target_all[indices], other_all[indices]
            y = labels[indices]
            same_direction = same_all[rows] - target
            cross_direction = cross_all[rows] - target
            base = _loss_from_hidden_batched(host, target, other, y, modality, device, batch_size)
            smooth = _perturbed_loss(host, target, other, y, modality, alpha, same_direction, device, batch_size) - base
            contrast = _perturbed_loss(host, target, other, y, modality, alpha, -same_direction, device, batch_size) - base
            cross = _perturbed_loss(host, target, other, y, modality, alpha, cross_direction, device, batch_size) - base
            delta = torch.stack([torch.zeros_like(base), smooth, contrast, cross], dim=-1)
            action_deltas[split_name] = {"base": base, "delta": delta}

        calib_delta = action_deltas["calib"]["delta"]
        calib_ce = action_deltas["calib"]["base"][:, None] + calib_delta
        action_names = ["NoOp", "Same-Smooth", "Same-Contrast", "Cross-Neighbor"]
        global_idx = int(calib_ce.mean(0).argmin())
        report = action_deltas["report"]
        report_ce = report["base"][:, None] + report["delta"]
        sorted_ce, sorted_idx = torch.sort(report_ce, dim=1)
        margins = sorted_ce[:, 1] - sorted_ce[:, 0]
        per_node_min = sorted_ce[:, 0]
        global_ce = report_ce[:, global_idx]
        headroom = global_ce - per_node_min
        headroom_ci = block_bootstrap_mean_ci(
            headroom, split.original_validation, reps=bootstrap_reps, seed=seed + (0 if modality == "text" else 1)
        )
        row = {
            "modality": modality,
            "alpha": float(alpha),
            "global_fixed_action_selected_on_calib": action_names[global_idx],
            "calib_mean_ce_by_action": {name: float(calib_ce[:, i].mean()) for i, name in enumerate(action_names)},
            "report_mean_delta_ce_by_action": {name: float(report["delta"][:, i].mean()) for i, name in enumerate(action_names)},
            "report_fraction_best_by_action": {name: float((sorted_idx[:, 0] == i).float().mean()) for i, name in enumerate(action_names)},
            "global_fixed_report_ce": float(global_ce.mean()),
            "adaptive_min_report_ce": float(per_node_min.mean()),
            "response_headroom_mean": float(headroom.mean()),
            "response_headroom_ci_low": headroom_ci["ci_low"],
            "response_headroom_ci_high": headroom_ci["ci_high"],
            "best_second_delta_ce_margin_mean": float(margins.mean()),
            "best_second_delta_ce_margin_median": float(margins.median()),
            "best_second_margin_below_1e-3_fraction": float((margins < 1e-3).float().mean()),
            "report_nodes": int(split.original_validation.numel()),
            "interpretation": "frozen-host operator response; adaptive bound is non-deployable and label-aware",
            "adaptive_bound_caveat": "frozen-host/action-bank-specific; not an oracle for future MoE and not evidence that an end-to-end router realizes this gap",
            "bootstrap_unit": "validation receiver node",
        }
        result_rows.append(row)
        raw[f"{modality}_response_profile"] = report["delta"]
        raw[f"{modality}_report_base_ce"] = report["base"]
        raw[f"{modality}_report_headroom"] = headroom
        raw[f"{modality}_report_margin"] = margins
    raw["validation_nodes"] = split.original_validation.clone()
    return result_rows, raw


def central_difference_edges(
    host: FeatureOnlyMultimodalDiagnosticHost,
    target_states: torch.Tensor,
    other_states: torch.Tensor,
    source_ids: torch.Tensor,
    receiver_ids: torch.Tensor,
    labels: torch.Tensor,
    modality: str,
    alphas: list[float],
    device: torch.device,
    batch_size: int = 4096,
) -> dict[float, torch.Tensor]:
    if source_ids.numel() != receiver_ids.numel():
        raise ValueError("source and receiver indices must align")
    target = target_states[receiver_ids]
    source = target_states[source_ids]
    other = other_states[receiver_ids]
    y = labels[receiver_ids]
    direction = equal_relative_norm(source - target, target)
    out: dict[float, torch.Tensor] = {}
    for alpha in alphas:
        plus = _loss_from_hidden_batched(host, target + float(alpha) * direction, other, y, modality, device, batch_size)
        minus = _loss_from_hidden_batched(host, target - float(alpha) * direction, other, y, modality, device, batch_size)
        out[float(alpha)] = (plus - minus) / (2.0 * float(alpha))
    return out


def central_difference_directions(
    host: FeatureOnlyMultimodalDiagnosticHost,
    target_states: torch.Tensor,
    other_states: torch.Tensor,
    direction: torch.Tensor,
    receiver_ids: torch.Tensor,
    labels: torch.Tensor,
    modality: str,
    alphas: list[float],
    device: torch.device,
    batch_size: int = 4096,
) -> dict[float, torch.Tensor]:
    """Central finite difference for an explicit, already edge-aligned direction."""
    target = target_states[receiver_ids]
    other = other_states[receiver_ids]
    normalized = equal_relative_norm(direction, target)
    y = labels[receiver_ids]
    out: dict[float, torch.Tensor] = {}
    for alpha in alphas:
        plus = _loss_from_hidden_batched(host, target + float(alpha) * normalized, other, y, modality, device, batch_size)
        minus = _loss_from_hidden_batched(host, target - float(alpha) * normalized, other, y, modality, device, batch_size)
        out[float(alpha)] = (plus - minus) / (2.0 * float(alpha))
    return out
