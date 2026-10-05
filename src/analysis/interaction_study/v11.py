"""Corrective and robustness analyses for the post-hoc V1.1 audit.

This module consumes frozen V1 artifacts. It does not train or update a host.
All uncertainty calculations use receiver blocks; host seeds are treated as
robustness runs, not as independent population replicates.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from src.analysis.interaction_study.diagnostic_host import FeatureOnlyMultimodalDiagnosticHost
from src.analysis.interaction_study.perturbation import (
    ReceiverSources,
    _loss_from_hidden_batched,
    _perturbed_loss,
    _target_and_other,
    equal_relative_norm,
    sample_receiver_sources,
)
from src.analysis.interaction_study.splits import StudySplit
from src.analysis.interaction_study.statistics import block_bootstrap_mean_ci, finite_summary
from src.analysis.problem_deep_dive import canonical_pairs


FUNCTIONAL_ACTIONS = ("NoOp", "Same-Smooth", "Same-Contrast", "Cross-Neighbor")
NULL_ACTIONS = ("NoOp", "Rand-Same-Smooth", "Rand-Same-Contrast", "Rand-Cross")


def orient_validation_edges(
    pairs: torch.Tensor,
    validation_nodes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return source/receiver rows so every validation endpoint is a receiver.

    Canonical physical edges produce one interaction when one endpoint is in
    validation and two directed interactions when both endpoints are there.
    """
    rows = pairs.detach().long().cpu()
    if rows.ndim != 2:
        raise ValueError("pairs must be rank two")
    if rows.size(1) != 2 and rows.size(0) == 2:
        rows = rows.T.contiguous()
    if rows.size(1) != 2:
        raise ValueError("pairs must have shape [E,2] or [2,E]")
    val = set(map(int, validation_nodes.detach().long().cpu().tolist()))
    sources: list[int] = []
    receivers: list[int] = []
    for a, b in rows.tolist():
        a_val, b_val = a in val, b in val
        if b_val:
            sources.append(int(a)); receivers.append(int(b))
        if a_val:
            sources.append(int(b)); receivers.append(int(a))
    return torch.tensor(sources, dtype=torch.long), torch.tensor(receivers, dtype=torch.long)


def _receiver_stat(values: torch.Tensor, receivers: torch.Tensor, reps: int, seed: int) -> dict[str, Any]:
    x = values.detach().float().cpu().flatten()
    r = receivers.detach().long().cpu().flatten()
    valid = torch.isfinite(x)
    x, r = x[valid], r[valid]
    summary = finite_summary(x)
    ci = block_bootstrap_mean_ci(x, r, reps=reps, seed=seed)
    return {
        **summary,
        "ci_low": ci["ci_low"],
        "ci_high": ci["ci_high"],
        "receiver_blocks": ci["block_count"],
        "bootstrap_reps": int(reps),
        "fraction_positive": float((x > 0).float().mean()) if x.numel() else float("nan"),
        "fraction_negative": float((x < 0).float().mean()) if x.numel() else float("nan"),
    }


def corrected_p1_summary(
    pairs: torch.Tensor,
    percentile_text: torch.Tensor,
    percentile_image: torch.Tensor,
    validation_nodes: torch.Tensor,
    *,
    reps: int = 1000,
    seed: int = 0,
    v1_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute V1 P1 summaries with corrected receiver orientation.

    V1 stores one percentile per canonical physical edge after averaging both
    endpoint source percentiles. The corrected orientation therefore preserves
    the point estimate while assigning bootstrap blocks to the actual receiver.
    """
    rows = pairs.detach().long().cpu()
    if rows.ndim != 2:
        raise ValueError("pairs must be rank two")
    if rows.size(0) == 2 and rows.size(1) != 2:
        rows = rows.T.contiguous()
    if rows.size(1) != 2:
        raise ValueError("pairs must have shape [E,2] or [2,E]")
    if rows.numel() and not torch.equal(rows, canonical_pairs(rows.T.contiguous())):
        raise ValueError("P1 pairs must be sorted canonical physical edges")
    pt = percentile_text.detach().float().cpu().flatten()
    pi = percentile_image.detach().float().cpu().flatten()
    if pt.numel() != rows.size(0) or pi.numel() != rows.size(0):
        raise ValueError("P1 percentile vectors must align with physical pairs")
    val = set(map(int, validation_nodes.detach().long().cpu().tolist()))
    src: list[int] = []; recv: list[int] = []; pair_idx: list[int] = []
    for idx, (a, b) in enumerate(rows.tolist()):
        a_val, b_val = a in val, b in val
        if b_val:
            src.append(a); recv.append(b); pair_idx.append(idx)
        if a_val:
            src.append(b); recv.append(a); pair_idx.append(idx)
    idx = torch.tensor(pair_idx, dtype=torch.long)
    receivers = torch.tensor(recv, dtype=torch.long)
    p_t = pt[idx] if idx.numel() else torch.empty(0)
    p_i = pi[idx] if idx.numel() else torch.empty(0)
    gap = (p_t - p_i).abs()
    valid = torch.isfinite(gap) & torch.isfinite(p_t) & torch.isfinite(p_i)
    gap, p_t, p_i, receivers = gap[valid], p_t[valid], p_i[valid], receivers[valid]
    mean_stats = _receiver_stat(gap, receivers, reps, seed)
    median = float(gap.median()) if gap.numel() else float("nan")
    rho = float(spearmanr(p_t.numpy(), p_i.numpy()).statistic) if p_t.numel() > 1 else float("nan")
    v1_mean = None if v1_summary is None else v1_summary.get("mean_absolute_percentile_gap")
    v1_median = None if v1_summary is None else v1_summary.get("median_absolute_percentile_gap")
    return {
        "orientation": "validation endpoints are receivers; both-validation physical edges yield both directions",
        "directed_interactions": int(gap.numel()),
        "receiver_count": int(receivers.unique().numel()),
        "mean_abs_percentile_gap": mean_stats["mean"],
        "median_abs_percentile_gap": median,
        "spearman": rho,
        "mean_gap_ci_low": mean_stats["ci_low"],
        "mean_gap_ci_high": mean_stats["ci_high"],
        "bootstrap_unit": "validation receiver node",
        "bootstrap_reps": int(reps),
        "v1_mean_gap": v1_mean,
        "v1_mean_gap_ci_low": None if v1_summary is None else v1_summary.get("mean_absolute_percentile_gap_ci_low"),
        "v1_mean_gap_ci_high": None if v1_summary is None else v1_summary.get("mean_absolute_percentile_gap_ci_high"),
        "v1_median_gap": v1_median,
        "mean_gap_change_vs_v1": None if v1_mean is None else float(mean_stats["mean"] - float(v1_mean)),
        "median_gap_change_vs_v1": None if v1_median is None else float(median - float(v1_median)),
        "point_estimate_invariant_to_orientation": bool(v1_mean is not None and abs(float(mean_stats["mean"]) - float(v1_mean)) <= 1e-7),
    }


@dataclass(frozen=True)
class SourceReferenceSample:
    active_sources: torch.Tensor
    references: torch.Tensor
    seed: int
    nonedges_per_source: int


def sample_source_matched_references(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    seed: int,
    max_sources: int = 250000,
    nonedges_per_source: int = 32,
) -> SourceReferenceSample:
    """Match V1's deterministic source and nonedge draw procedure exactly."""
    pairs = canonical_pairs(edge_index)
    adj: list[set[int]] = [set() for _ in range(int(num_nodes))]
    for a, b in pairs.tolist():
        adj[a].add(b); adj[b].add(a)
    active = torch.tensor([i for i, row in enumerate(adj) if row], dtype=torch.long)
    generator = torch.Generator().manual_seed(int(seed))
    if active.numel() > int(max_sources):
        active = active[torch.randperm(active.numel(), generator=generator)[:int(max_sources)]].sort().values
    references = torch.empty((active.numel(), int(nonedges_per_source)), dtype=torch.long)
    for row, source in enumerate(active.tolist()):
        selected: list[int] = []
        tries = 0
        while len(selected) < int(nonedges_per_source) and tries < int(nonedges_per_source) * 100:
            batch = max(32, (int(nonedges_per_source) - len(selected)) * 4)
            candidates = torch.randint(int(num_nodes), (batch,), generator=generator).tolist()
            tries += len(candidates)
            for candidate in candidates:
                if candidate != source and candidate not in adj[source] and candidate not in selected:
                    selected.append(int(candidate))
                    if len(selected) == int(nonedges_per_source):
                        break
        if len(selected) != int(nonedges_per_source):
            raise RuntimeError(f"source {source} could not draw {nonedges_per_source} nonedges")
        references[row] = torch.tensor(selected, dtype=torch.long)
    return SourceReferenceSample(active, references, int(seed), int(nonedges_per_source))


def directional_reference_percentiles(
    states: torch.Tensor,
    source_ids: torch.Tensor,
    receiver_ids: torch.Tensor,
    sample: SourceReferenceSample,
    *,
    device: torch.device,
    batch_sources: int = 2048,
) -> torch.Tensor:
    """Compute directed empirical percentile ranks against sampled nonedges."""
    src = source_ids.detach().long().cpu().flatten()
    recv = receiver_ids.detach().long().cpu().flatten()
    active = sample.active_sources
    source_row = torch.full((states.size(0),), -1, dtype=torch.long)
    source_row[active] = torch.arange(active.numel())
    rows = source_row[src]
    if (rows < 0).any():
        raise RuntimeError("a validation interaction source was omitted by max_sources")
    x = F.normalize(states.detach().float().to(device), dim=-1, eps=1e-8)
    out = torch.empty(src.numel(), dtype=torch.float32)
    # Batch by unique source: each source's reference scores are shared by all
    # validation interactions originating from it.
    unique_sources, inverse = torch.unique(src, sorted=True, return_inverse=True)
    unique_rows = source_row[unique_sources]
    query_order = torch.argsort(inverse, stable=True)
    counts = torch.bincount(inverse, minlength=unique_sources.numel())
    query_starts = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)[:-1]])
    for start in range(0, unique_sources.numel(), int(batch_sources)):
        end = min(start + int(batch_sources), unique_sources.numel())
        src_batch = unique_sources[start:end].to(device)
        ref_rows = unique_rows[start:end]
        ref_ids = sample.references[ref_rows].to(device)
        ref_scores = (x[src_batch, None, :] * x[ref_ids]).sum(-1).sort(dim=1).values
        q0 = int(query_starts[start])
        q1 = int(query_starts[end - 1] + counts[end - 1])
        selected = query_order[q0:q1]
        local_source_rows = (inverse[selected] - start).to(device)
        edge_source = src[selected].to(device)
        edge_receiver = recv[selected].to(device)
        edge_scores = (x[edge_source] * x[edge_receiver]).sum(-1)
        refs_by_edge = ref_scores[local_source_rows]
        rank = (refs_by_edge <= edge_scores[:, None]).sum(-1).float() / sample.nonedges_per_source
        out[selected] = rank.detach().cpu()
    return out


def summarize_reference_noise(
    text_percentiles: torch.Tensor,
    image_percentiles: torch.Tensor,
    receivers: torch.Tensor,
    *,
    reps: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Summarize repeated source-matched reference draws and their noise floor."""
    pt = text_percentiles.detach().float().cpu()
    pi = image_percentiles.detach().float().cpu()
    block = receivers.detach().long().cpu().flatten()
    if pt.ndim != 2 or pi.ndim != 2 or pt.shape != pi.shape or pt.size(0) < 2:
        raise ValueError("percentiles must be aligned [reference_seed, interaction] matrices")
    if pt.size(1) != block.numel():
        raise ValueError("receivers must align with percentile interaction columns")
    valid = torch.isfinite(pt).all(0) & torch.isfinite(pi).all(0)
    pt, pi, block = pt[:, valid], pi[:, valid], block[valid]
    mean_t, mean_i = pt.mean(0), pi.mean(0)
    cross = (mean_t - mean_i).abs()
    instability_t = (pt - mean_t).abs().mean(0)
    instability_i = (pi - mean_i).abs().mean(0)
    adjusted = cross - instability_t - instability_i
    vectors = {
        "cross_signal": cross,
        "text_reference_instability": instability_t,
        "image_reference_instability": instability_i,
        "conservative_adjusted_gap": adjusted,
    }
    summaries = {}
    for offset, (name, values) in enumerate(vectors.items()):
        stat = _receiver_stat(values, block, reps, seed + offset)
        summaries[name] = stat
    cross_mean = summaries["cross_signal"]["mean"]
    noise_mean = summaries["text_reference_instability"]["mean"] + summaries["image_reference_instability"]["mean"]
    return {
        "interaction_count": int(valid.sum()),
        "receiver_count": int(block.unique().numel()),
        "reference_seed_count": int(pt.size(0)),
        "cross_signal_median": float(cross.median()) if cross.numel() else float("nan"),
        "descriptive_ratio": float(cross_mean / (noise_mean + 1e-12)),
        "metrics": summaries,
    }


def factorial_p2_metrics(
    L0: torch.Tensor,
    LS: torch.Tensor,
    LC: torch.Tensor,
    LR: torch.Tensor,
    LSC: torch.Tensor,
    LSR: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return the directional graph and matched-random factorial contrasts."""
    return {
        "cross_only_graph_gain": LR - LC,
        "conditional_cross_delta_graph": LSC - LS,
        "conditional_cross_delta_random": LSR - LS,
        "conditional_graph_gain": LSR - LSC,
        "same_cross_perturbation_interaction_term": LSC - LS - LC + L0,
    }


def _indexed_receiver_rows(sources: ReceiverSources, receivers: torch.Tensor) -> torch.Tensor:
    positions = {int(node): row for row, node in enumerate(sources.receivers.tolist())}
    try:
        return torch.tensor([positions[int(node)] for node in receivers.tolist()], dtype=torch.long)
    except KeyError as exc:
        raise ValueError(f"receiver {exc.args[0]} is missing from the frozen source bank") from exc


def compute_p2_factorial(
    host: FeatureOnlyMultimodalDiagnosticHost,
    h_t: torch.Tensor,
    h_i: torch.Tensor,
    aligned_t_to_i: torch.Tensor,
    aligned_i_to_t: torch.Tensor,
    labels: torch.Tensor,
    split: StudySplit,
    sources: ReceiverSources,
    alphas: Sequence[float],
    *,
    device: torch.device,
    batch_size: int = 4096,
) -> dict[str, dict[float, dict[str, torch.Tensor]]]:
    """Compute P2's same/cross/random factorial perturbations per receiver."""
    validation = split.original_validation.detach().long().cpu()
    rows = _indexed_receiver_rows(sources, validation)
    counts = sources.counts[rows]
    if int(sources.counts.sum()) != int(sources.neighbor_ids.numel()) or int(sources.counts.sum()) != int(sources.random_ids.numel()):
        raise AssertionError("graph and nonneighbor source banks have different source counts")
    y = labels[validation].long().cpu()
    result: dict[str, dict[float, dict[str, torch.Tensor]]] = {}
    direction_specs = (
        ("T2I", "image", h_i, h_t, aligned_t_to_i),
        ("I2T", "text", h_t, h_i, aligned_i_to_t),
    )
    for direction_name, modality, target_all, other_all, aligned_other in direction_specs:
        target = target_all[validation].float().cpu()
        other = other_all[validation].float().cpu()
        same_state = sources.aggregate(target_all)[rows]
        cross_state = sources.aggregate(aligned_other)[rows]
        random_cross_state = sources.aggregate(aligned_other, random_control=True)[rows]
        d_same = equal_relative_norm(same_state - target, target)
        d_cross = equal_relative_norm(cross_state - target, target)
        d_random = equal_relative_norm(random_cross_state - target, target)
        base = _loss_from_hidden_batched(host, target, other, y, modality, device, batch_size)
        by_alpha: dict[float, dict[str, torch.Tensor]] = {}
        for alpha_value in alphas:
            alpha = float(alpha_value)
            L0 = base
            LS = _loss_from_hidden_batched(host, target + alpha * d_same, other, y, modality, device, batch_size)
            LC = _loss_from_hidden_batched(host, target + alpha * d_cross, other, y, modality, device, batch_size)
            LR = _loss_from_hidden_batched(host, target + alpha * d_random, other, y, modality, device, batch_size)
            LSC = _loss_from_hidden_batched(host, target + alpha * d_same + alpha * d_cross, other, y, modality, device, batch_size)
            LSR = _loss_from_hidden_batched(host, target + alpha * d_same + alpha * d_random, other, y, modality, device, batch_size)
            by_alpha[alpha] = {
                **factorial_p2_metrics(L0, LS, LC, LR, LSC, LSR),
                "L0": L0, "LS": LS, "LC": LC, "LR": LR, "LSC": LSC, "LSR": LSR,
            }
        result[direction_name] = by_alpha
    for alpha_value in alphas:
        alpha = float(alpha_value)
        for metric in (
            "cross_only_graph_gain",
            "conditional_graph_gain",
            "conditional_cross_delta_graph",
            "conditional_cross_delta_random",
            "same_cross_perturbation_interaction_term",
        ):
            result.setdefault("T2I_minus_I2T", {}).setdefault(alpha, {})[metric] = (
                result["T2I"][alpha][metric] - result["I2T"][alpha][metric]
            )
    return result


def summarize_seed_receiver_vectors(
    per_seed: Mapping[int, torch.Tensor],
    receivers: torch.Tensor,
    *,
    reps: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Average receiver values across host seeds, then bootstrap receivers."""
    if not per_seed:
        raise ValueError("at least one host seed vector is required")
    ordered = sorted(per_seed)
    vectors = [per_seed[s].detach().float().cpu().flatten() for s in ordered]
    if any(v.shape != vectors[0].shape for v in vectors):
        raise ValueError("host seeds must share the same receiver universe")
    block = receivers.detach().long().cpu().flatten()
    if block.numel() != vectors[0].numel():
        raise ValueError("receiver IDs and per-seed metric vectors must align")
    matrix = torch.stack(vectors)
    valid = torch.isfinite(matrix).all(0)
    averaged = matrix[:, valid].mean(0)
    valid_receivers = block[valid]
    return {
        "host_seed_averaged_receiver_values": averaged,
        "seed_means": {int(s): float(v.mean()) for s, v in zip(ordered, vectors, strict=True)},
        "seed_signs": {int(s): int(torch.sign(v.mean()).item()) for s, v in zip(ordered, vectors, strict=True)},
        "summary": _receiver_stat(averaged, valid_receivers, reps, seed),
        "seed_count": len(ordered),
        "aggregation": "average the same receiver across host seeds, then resample validation receiver blocks",
    }


def classify_p2_evidence(seed_rows: Sequence[Mapping[str, Any]], averaged: Mapping[str, Any]) -> str:
    """Apply the V1.1 Strong / Consistent-Weak / Mixed / Unavailable rubric."""
    if not seed_rows:
        return "Unavailable"
    means = [float(row.get("mean", float("nan"))) for row in seed_rows]
    if not all(np.isfinite(means)):
        return "Unavailable"
    same_positive = all(value > 0 for value in means)
    same_negative = all(value < 0 for value in means)
    if not (same_positive or same_negative):
        return "Mixed"
    ci = averaged.get("summary", {})
    ci_low, ci_high = float(ci.get("ci_low", float("nan"))), float(ci.get("ci_high", float("nan")))
    excludes_same_direction = ci_low > 0 if same_positive else ci_high < 0
    if not excludes_same_direction:
        return "Mixed"
    individual = sum(
        (float(row.get("ci_low", float("nan"))) > 0 if same_positive else float(row.get("ci_high", float("nan"))) < 0)
        for row in seed_rows
    )
    return "Strong" if individual >= 2 else "Consistent-Weak"


def _matched_group_totals(
    values: np.ndarray,
    semantic_bin: np.ndarray,
    structure_bin: np.ndarray,
    source_degree_bin: np.ndarray,
    receiver_degree_bin: np.ndarray,
) -> tuple[float, int, int, np.ndarray, np.ndarray]:
    valid = (
        (semantic_bin >= 0) & (semantic_bin < 4)
        & (structure_bin >= 0) & (structure_bin < 4)
        & (source_degree_bin >= 0) & (source_degree_bin < 4)
        & (receiver_degree_bin >= 0) & (receiver_degree_bin < 4)
        & np.isfinite(values)
    )
    group = semantic_bin * 16 + source_degree_bin * 4 + receiver_degree_bin
    low = valid & (structure_bin == 0)
    high = valid & (structure_bin == 3)
    low_count = np.bincount(group[low], minlength=64).astype(np.int64)
    high_count = np.bincount(group[high], minlength=64).astype(np.int64)
    low_sum = np.bincount(group[low], weights=values[low], minlength=64)
    high_sum = np.bincount(group[high], weights=values[high], minlength=64)
    weights = np.minimum(low_count, high_count)
    legal = weights > 0
    if not legal.any():
        return float("nan"), 0, 0, np.stack([low_count, high_count], 1), np.stack([low_sum, high_sum], 1)
    delta = high_sum[legal] / high_count[legal] - low_sum[legal] / low_count[legal]
    effect = float(np.dot(weights[legal], delta) / weights[legal].sum())
    return effect, int(legal.sum()), int(weights.sum()), np.stack([low_count, high_count], 1), np.stack([low_sum, high_sum], 1)


def deterministic_matched_effect(
    values: torch.Tensor | np.ndarray,
    semantic_bin: torch.Tensor | np.ndarray,
    structure_bin: torch.Tensor | np.ndarray,
    source_degree_bin: torch.Tensor | np.ndarray,
    receiver_degree_bin: torch.Tensor | np.ndarray,
) -> dict[str, Any]:
    """Deterministic min(n_high,n_low)-weighted high-minus-low point estimate."""
    arrays = [np.asarray(torch.as_tensor(x).detach().cpu()) for x in (values, semantic_bin, structure_bin, source_degree_bin, receiver_degree_bin)]
    vals = arrays[0].astype(np.float64, copy=False).reshape(-1)
    sem, struct, src_deg, recv_deg = [x.astype(np.int64, copy=False).reshape(-1) for x in arrays[1:]]
    if any(x.size != vals.size for x in (sem, struct, src_deg, recv_deg)):
        raise ValueError("all matched-effect inputs must have the same number of rows")
    effect, strata, interactions, _, _ = _matched_group_totals(vals, sem, struct, src_deg, recv_deg)
    return {
        "matched_effect": effect,
        "matched_strata": strata,
        "matched_interactions_per_arm": interactions,
        "status": "MATCHED" if strata else "UNAVAILABLE_NO_COMMON_HIGH_LOW_STRATA",
        "weighting": "min(n_high,n_low) within semantic_bin × source_degree_bin × receiver_degree_bin",
    }


def matched_effect_receiver_bootstrap(
    values: torch.Tensor | np.ndarray,
    semantic_bin: torch.Tensor | np.ndarray,
    structure_bin: torch.Tensor | np.ndarray,
    source_degree_bin: torch.Tensor | np.ndarray,
    receiver_degree_bin: torch.Tensor | np.ndarray,
    receivers: torch.Tensor | np.ndarray,
    *,
    reps: int = 1000,
    seed: int = 0,
    receiver_universe: torch.Tensor | np.ndarray | None = None,
) -> dict[str, Any]:
    """Receiver-block bootstrap that re-forms matched strata on each draw.

    Receiver multiplicity scales every interaction in that receiver block. The
    point estimate remains the deterministic weighted estimator above.
    """
    arrays = [np.asarray(torch.as_tensor(x).detach().cpu()) for x in (values, semantic_bin, structure_bin, source_degree_bin, receiver_degree_bin, receivers)]
    vals = arrays[0].astype(np.float64, copy=False).reshape(-1)
    sem, struct, src_deg, recv_deg, receiver = [x.astype(np.int64, copy=False).reshape(-1) for x in arrays[1:]]
    if any(x.size != vals.size for x in (sem, struct, src_deg, recv_deg, receiver)):
        raise ValueError("all receiver-bootstrap inputs must have the same number of rows")
    point = deterministic_matched_effect(vals, sem, struct, src_deg, recv_deg)
    valid = (
        (sem >= 0) & (sem < 4) & (struct >= 0) & (struct < 4)
        & (src_deg >= 0) & (src_deg < 4) & (recv_deg >= 0) & (recv_deg < 4)
        & np.isfinite(vals)
    )
    observed_receivers = (
        np.unique(receiver[valid]) if receiver_universe is None
        else np.unique(np.asarray(torch.as_tensor(receiver_universe).detach().cpu(), dtype=np.int64).reshape(-1))
    )
    if receiver_universe is not None and np.any(~np.isin(receiver[valid], observed_receivers)):
        raise ValueError("an interaction receiver is outside the supplied validation universe")
    if not point["matched_strata"] or observed_receivers.size == 0 or reps < 1:
        return {
            **point,
            "ci_low": float("nan"), "ci_high": float("nan"),
            "valid_bootstrap_count": 0, "requested_bootstrap_reps": int(reps),
            "bootstrap_unit": "validation receiver node; all interaction rows retained as blocks",
        }
    receiver_pos = np.searchsorted(observed_receivers, receiver)
    group = sem * 16 + src_deg * 4 + recv_deg
    arm = (struct == 3).astype(np.int64)
    valid_rows = valid & ((struct == 0) | (struct == 3))
    n_blocks = observed_receivers.size
    block_counts = np.zeros((n_blocks, 64, 2), dtype=np.float64)
    block_sums = np.zeros((n_blocks, 64, 2), dtype=np.float64)
    np.add.at(block_counts, (receiver_pos[valid_rows], group[valid_rows], arm[valid_rows]), 1.0)
    np.add.at(block_sums, (receiver_pos[valid_rows], group[valid_rows], arm[valid_rows]), vals[valid_rows])
    rng = np.random.default_rng(int(seed))
    effects: list[float] = []
    for _ in range(int(reps)):
        selected = rng.integers(0, n_blocks, size=n_blocks)
        multiplicity = np.bincount(selected, minlength=n_blocks).astype(np.float64)
        counts = np.tensordot(multiplicity, block_counts, axes=(0, 0))
        sums = np.tensordot(multiplicity, block_sums, axes=(0, 0))
        weights = np.minimum(counts[:, 0], counts[:, 1])
        legal = weights > 0
        if not legal.any():
            continue
        delta = sums[legal, 1] / counts[legal, 1] - sums[legal, 0] / counts[legal, 0]
        effects.append(float(np.dot(weights[legal], delta) / weights[legal].sum()))
    if effects:
        low, high = np.quantile(effects, [0.025, 0.975])
    else:
        low = high = float("nan")
    return {
        **point,
        "ci_low": float(low), "ci_high": float(high),
        "valid_bootstrap_count": len(effects), "requested_bootstrap_reps": int(reps),
        "receiver_block_count": int(n_blocks),
        "bootstrap_unit": "validation receiver node; all interaction rows retained as blocks",
        "interval_method": "percentile 95% receiver-block bootstrap with re-matching per draw",
    }


def classify_p3_modality(seed_rows: Sequence[Mapping[str, Any]]) -> str:
    """Grade one dataset × modality over its three host robustness seeds."""
    if len(seed_rows) != 3 or any(row.get("matched_strata", 0) == 0 for row in seed_rows):
        return "Unavailable"
    effects = [float(row.get("matched_effect", float("nan"))) for row in seed_rows]
    if not all(np.isfinite(effects)):
        return "Unavailable"
    signs = [int(np.sign(value)) for value in effects]
    if len(set(signs)) != 1:
        return "Mixed"
    direction = signs[0]
    if direction == 0:
        return "Weak"
    concordant = sum(
        float(row.get("ci_low", float("nan"))) > 0 if direction > 0
        else float(row.get("ci_high", float("nan"))) < 0
        for row in seed_rows
    )
    if concordant == 3:
        return "Strong"
    if concordant >= 2:
        return "Consistent"
    return "Weak"


def p3_dataset_description(text_grade: str, image_grade: str) -> str:
    if text_grade == "Unavailable" and image_grade == "Unavailable":
        return "unavailable"
    consistent = {"Strong", "Consistent"}
    if text_grade in consistent and image_grade in consistent:
        return "both modalities consistent"
    if text_grade in consistent or image_grade in consistent:
        return "one modality consistent"
    return "mixed modalities"


def classify_p3_dataset(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Always grade text and image independently and return both detail sets."""
    details = {}
    for modality in ("text", "image"):
        selected = [row for row in rows if row.get("modality") == modality]
        details[modality] = {
            "grade": classify_p3_modality(selected),
            "seed_details": [
                {
                    "host_seed": row.get("host_seed"),
                    "matched_effect": row.get("matched_effect"),
                    "ci_low": row.get("ci_low"),
                    "ci_high": row.get("ci_high"),
                    "matched_strata": row.get("matched_strata"),
                }
                for row in selected
            ],
        }
    return {
        "text": details["text"],
        "image": details["image"],
        "dataset_description": p3_dataset_description(details["text"]["grade"], details["image"]["grade"]),
    }


def _profile_for_bank(
    host: FeatureOnlyMultimodalDiagnosticHost,
    h_t: torch.Tensor,
    h_i: torch.Tensor,
    aligned_t_to_i: torch.Tensor,
    aligned_i_to_t: torch.Tensor,
    labels: torch.Tensor,
    indices: torch.Tensor,
    sources: ReceiverSources,
    modality: str,
    bank: str,
    alpha: float,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_all, other_all = _target_and_other(h_t, h_i, modality)
    aligned = aligned_t_to_i if modality == "image" else aligned_i_to_t
    source_rows = _indexed_receiver_rows(sources, indices)
    target = target_all[indices].float().cpu()
    other = other_all[indices].float().cpu()
    y = labels[indices].long().cpu()
    base = _loss_from_hidden_batched(host, target, other, y, modality, device, batch_size)
    is_random = bank == "null"
    if bank not in {"functional", "null"}:
        raise ValueError("bank must be functional or null")
    same_state = sources.aggregate(target_all, random_control=is_random)[source_rows]
    cross_state = sources.aggregate(aligned, random_control=is_random)[source_rows]
    same_direction = same_state - target
    cross_direction = cross_state - target
    same_smooth = _perturbed_loss(host, target, other, y, modality, alpha, same_direction, device, batch_size) - base
    same_contrast = _perturbed_loss(host, target, other, y, modality, alpha, -same_direction, device, batch_size) - base
    cross = _perturbed_loss(host, target, other, y, modality, alpha, cross_direction, device, batch_size) - base
    deltas = torch.stack([torch.zeros_like(base), same_smooth, same_contrast, cross], dim=-1)
    return base, deltas


def compute_null_action_profiles(
    host: FeatureOnlyMultimodalDiagnosticHost,
    h_t: torch.Tensor,
    h_i: torch.Tensor,
    aligned_t_to_i: torch.Tensor,
    aligned_i_to_t: torch.Tensor,
    labels: torch.Tensor,
    split: StudySplit,
    sources: ReceiverSources,
    *,
    alpha: float = 0.1,
    device: torch.device,
    batch_size: int = 4096,
) -> dict[str, dict[str, torch.Tensor]]:
    """Compute the matched random-nonneighbor action bank on calib and report."""
    output: dict[str, dict[str, torch.Tensor]] = {}
    for modality in ("text", "image"):
        output[modality] = {}
        for split_name, indices in (("calib", split.diagnostic_calib), ("report", split.original_validation)):
            base, profile = _profile_for_bank(
                host, h_t, h_i, aligned_t_to_i, aligned_i_to_t, labels,
                indices.detach().long().cpu(), sources, modality, "null", float(alpha),
                device=device, batch_size=batch_size,
            )
            output[modality][f"{split_name}_base_ce"] = base
            output[modality][f"{split_name}_response_profile"] = profile
            output[modality][f"{split_name}_global_action_index"] = profile.mean(0).argmin().reshape(1)
    output["validation_nodes"] = {"ids": split.original_validation.detach().long().cpu().clone()}
    return output


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    if np.allclose(x, y, equal_nan=True):
        return 1.0
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    if np.allclose(x, y, equal_nan=True):
        return 1.0
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(spearmanr(x, y).statistic)


def _cohen_kappa(left: np.ndarray, right: np.ndarray, action_count: int) -> float:
    if left.size != right.size or left.size == 0:
        return float("nan")
    observed = float(np.mean(left == right))
    if observed == 1.0:
        return 1.0
    left_freq = np.bincount(left, minlength=action_count) / left.size
    right_freq = np.bincount(right, minlength=action_count) / right.size
    expected = float(np.dot(left_freq, right_freq))
    return float((observed - expected) / (1.0 - expected)) if expected < 1.0 else float("nan")


def top_action_agreement(left: np.ndarray, right: np.ndarray, action_count: int) -> dict[str, Any]:
    if left.shape != right.shape:
        raise ValueError("top-action arrays must align node by node")
    return {
        "raw_agreement": float(np.mean(left == right)) if left.size else float("nan"),
        "cohen_kappa": _cohen_kappa(left, right, action_count),
        "receiver_count": int(left.size),
    }


def margin_conditioned_pair_agreement(
    left_actions: np.ndarray,
    right_actions: np.ndarray,
    left_margins: np.ndarray,
    right_margins: np.ndarray,
    *,
    minimum_margin: float = 1e-3,
) -> dict[str, Any]:
    arrays = [np.asarray(x).reshape(-1) for x in (left_actions, right_actions, left_margins, right_margins)]
    if any(x.shape != arrays[0].shape for x in arrays[1:]):
        raise ValueError("actions and margins must share node order and shape")
    la, ra, lm, rm = arrays
    strict = (lm >= minimum_margin) & (rm >= minimum_margin)
    top_left = lm >= np.median(lm)
    top_right = rm >= np.median(rm)
    top_half = top_left & top_right
    def pack(mask: np.ndarray) -> dict[str, Any]:
        return {
            "receiver_count": int(mask.sum()),
            "raw_agreement": float(np.mean(la[mask] == ra[mask])) if mask.any() else float("nan"),
            "cohen_kappa": _cohen_kappa(la[mask], ra[mask], int(max(la.max(initial=0), ra.max(initial=0)) + 1)) if mask.any() else float("nan"),
        }
    return {
        "minimum_margin": float(minimum_margin),
        "both_margin_at_least_threshold": pack(strict),
        "both_in_seed_specific_top_half_margin": pack(top_half),
    }


def operator_stability_summary(
    profiles_by_seed: Mapping[int, np.ndarray],
    action_names: Sequence[str],
    *,
    minimum_margin: float = 1e-3,
) -> dict[str, Any]:
    seeds = sorted(profiles_by_seed)
    if len(seeds) != 3:
        raise ValueError("operator stability requires exactly three host seeds")
    profiles = [np.asarray(profiles_by_seed[s], dtype=np.float64) for s in seeds]
    if any(p.ndim != 2 or p.shape[1] != len(action_names) for p in profiles):
        raise ValueError("each response profile must have shape [validation_nodes, action_count]")
    if any(p.shape != profiles[0].shape for p in profiles):
        raise ValueError("all host seeds must share identical validation node order and profile shape")
    best = [p.argmin(1) for p in profiles]
    margins = [np.sort(p, axis=1)[:, 1] - np.sort(p, axis=1)[:, 0] for p in profiles]
    pair_rows = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        agreement = top_action_agreement(best[i], best[j], len(action_names))
        conditioned = margin_conditioned_pair_agreement(best[i], best[j], margins[i], margins[j], minimum_margin=minimum_margin)
        per_action = {}
        for action_index, action in enumerate(action_names):
            left = profiles[i][:, action_index]
            right = profiles[j][:, action_index]
            per_action[action] = {"pearson": _pearson(left, right), "spearman": _spearman(left, right)}
        centered_left = profiles[i] - profiles[i].mean(1, keepdims=True)
        centered_right = profiles[j] - profiles[j].mean(1, keepdims=True)
        norm_left = np.linalg.norm(centered_left, axis=1)
        norm_right = np.linalg.norm(centered_right, axis=1)
        cos = np.full(norm_left.shape, np.nan, dtype=np.float64)
        both_zero = (norm_left <= 1e-12) & (norm_right <= 1e-12)
        cos[both_zero] = 1.0
        valid = (norm_left > 1e-12) & (norm_right > 1e-12)
        cos[valid] = (centered_left[valid] * centered_right[valid]).sum(1) / (norm_left[valid] * norm_right[valid])
        pair_rows.append({
            "seed_pair": f"{seeds[i]}-{seeds[j]}",
            **agreement,
            "margin_conditioned": conditioned,
            "response_profile_by_action": per_action,
            "centered_profile_cosine_mean": float(np.nanmean(cos)) if np.isfinite(cos).any() else float("nan"),
            "centered_profile_cosine_median": float(np.nanmedian(cos)) if np.isfinite(cos).any() else float("nan"),
        })
    votes = np.stack(best)
    counts = np.stack([np.sum(votes == action, axis=0) for action in range(len(action_names))])
    return {
        "action_names": list(action_names),
        "seed_pairs": pair_rows,
        "unanimous_top_action_fraction": float((counts.max(0) == 3).mean()),
        "at_least_two_of_three_same_action_fraction": float((counts.max(0) >= 2).mean()),
        "top_action_fraction_by_seed": {
            int(seed): {action: float(np.mean(best[pos] == idx)) for idx, action in enumerate(action_names)}
            for pos, seed in enumerate(seeds)
        },
        "margin_definition": "second-best minus best per-node CE response",
    }


def functional_null_paired_metrics(
    functional_profile: torch.Tensor | np.ndarray,
    null_profile: torch.Tensor | np.ndarray,
    functional_global_index: int,
    null_global_index: int,
) -> dict[str, torch.Tensor]:
    function = torch.as_tensor(functional_profile, dtype=torch.float32)
    null = torch.as_tensor(null_profile, dtype=torch.float32)
    if function.shape != null.shape or function.ndim != 2 or function.size(1) != 4:
        raise ValueError("functional and null profiles must both be [nodes,4]")
    h_func = function[:, int(functional_global_index)] - function.min(1).values
    h_null = null[:, int(null_global_index)] - null.min(1).values
    return {
        "functional_headroom": h_func,
        "null_headroom": h_null,
        "excess_headroom": h_func - h_null,
        "best_action_advantage": null.min(1).values - function.min(1).values,
        "global_fixed_advantage": null[:, int(null_global_index)] - function[:, int(functional_global_index)],
    }


def build_null_action_directions(
    target: torch.Tensor,
    random_same_mean: torch.Tensor,
    random_cross_mean: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build the 4-action matched null bank with equal-relative-norm vectors."""
    if target.shape != random_same_mean.shape or target.shape != random_cross_mean.shape:
        raise ValueError("null bank target, same-source, and cross-source tensors must align")
    same = random_same_mean - target
    cross = random_cross_mean - target
    return {
        "NoOp": torch.zeros_like(target),
        "Rand-Same-Smooth": equal_relative_norm(same, target),
        "Rand-Same-Contrast": equal_relative_norm(-same, target),
        "Rand-Cross": equal_relative_norm(cross, target),
    }
