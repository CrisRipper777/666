from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr, t as student_t
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.analysis.mechanism_discovery import safe_spearman
from src.analysis.problem_deep_dive import (
    canonical_pairs,
    context_novelty,
    source_matched_compatibility,
    structural_edge_descriptors,
    train_edge_quartile_thresholds,
)
from .diagnostic_host import FeatureOnlyMultimodalDiagnosticHost
from .perturbation import central_difference_directions, central_difference_edges, equal_relative_norm
from .splits import StudySplit
from .statistics import block_bootstrap_mean_ci, bins_from_thresholds, finite_summary, node_quantile_bins, stratified_sample


def compute_compatibility(
    states: torch.Tensor,
    edge_index: torch.Tensor,
    seed: int,
    max_sources: int,
    nonedges_per_source: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    """Reuse problem_deep_dive's vectorized source-matched percentile definition."""
    return source_matched_compatibility(
        states, edge_index, seed=seed, max_sources=max_sources,
        nonedges_per_source=nonedges_per_source,
    )


def _pair_keys(rows: torch.Tensor, num_nodes: int) -> torch.Tensor:
    return rows[:, 0].long() * int(num_nodes) + rows[:, 1].long()


def _aligned_pair_values(
    pairs_a: torch.Tensor,
    values_a: torch.Tensor,
    pairs_b: torch.Tensor,
    values_b: torch.Tensor,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows_a = pairs_a.detach().long().cpu().T.contiguous()
    rows_b = pairs_b.detach().long().cpu().T.contiguous()
    keys_a, keys_b = _pair_keys(rows_a, num_nodes), _pair_keys(rows_b, num_nodes)
    if not torch.equal(keys_a, keys_b):
        positions = torch.searchsorted(keys_b, keys_a)
        valid = positions < keys_b.numel()
        valid &= keys_b[positions.clamp(max=max(keys_b.numel() - 1, 0))] == keys_a
        rows_a, values_a = rows_a[valid], values_a[valid]
        positions = positions[valid]
        keys_a = keys_a[valid]
        rows_b, values_b = rows_b[positions], values_b[positions]
    return rows_a, values_a.detach().float().cpu(), values_b.detach().float().cpu()


def p1_modality_heterogeneity(
    pairs_t: torch.Tensor,
    percentile_t: torch.Tensor,
    pairs_i: torch.Tensor,
    percentile_i: torch.Tensor,
    original_train: torch.Tensor,
    validation: torch.Tensor,
    labels: torch.Tensor,
    num_nodes: int,
    *,
    bootstrap_reps: int = 500,
    seed: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, torch.Tensor]]:
    pairs, p_t, p_i = _aligned_pair_values(pairs_t, percentile_t, pairs_i, percentile_i, num_nodes)
    thresholds_t = train_edge_quartile_thresholds(p_t, pairs, original_train)
    thresholds_i = train_edge_quartile_thresholds(p_i, pairs, original_train)
    bins_t = bins_from_thresholds(p_t, thresholds_t)
    bins_i = bins_from_thresholds(p_i, thresholds_i)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask[validation] = True
    src, dst = pairs[:, 0], pairs[:, 1]
    directed_src = torch.cat([src[val_mask[src]], dst[val_mask[dst]]])
    directed_dst = torch.cat([dst[val_mask[src]], src[val_mask[dst]]])
    pair_indices = torch.cat([torch.where(val_mask[src])[0], torch.where(val_mask[dst])[0]])
    recv = directed_dst
    if recv.numel():
        pt = p_t[pair_indices]
        pi = p_i[pair_indices]
        bt = bins_t[pair_indices]
        bi = bins_i[pair_indices]
        valid = (bt >= 0) & (bi >= 0)
        src_v, recv_v = directed_src[valid], recv[valid]
        pt_v, pi_v, bt_v, bi_v = pt[valid], pi[valid], bt[valid], bi[valid]
    else:
        src_v = recv_v = torch.empty(0, dtype=torch.long)
        pt_v = pi_v = torch.empty(0)
        bt_v = bi_v = torch.empty(0, dtype=torch.long)

    abs_gap = (pt_v - pi_v).abs()
    gap_summary = finite_summary(abs_gap)
    mean_ci = block_bootstrap_mean_ci(abs_gap, recv_v, reps=bootstrap_reps, seed=seed)
    hi_t_lo_i = ((bt_v == 3) & (bi_v == 0)).float()
    lo_t_hi_i = ((bt_v == 0) & (bi_v == 3)).float()
    hi_ci = block_bootstrap_mean_ci(hi_t_lo_i, recv_v, reps=bootstrap_reps, seed=seed + 1)
    lo_ci = block_bootstrap_mean_ci(lo_t_hi_i, recv_v, reps=bootstrap_reps, seed=seed + 2)
    cross_table = torch.bincount(bt_v * 4 + bi_v, minlength=16).reshape(4, 4)

    descriptive = torch.full((4, 4), float("nan"))
    both_validation = val_mask[src_v] & val_mask[recv_v]
    both_ids = torch.where(both_validation)[0]
    allowed_positions = both_ids[(labels[src_v[both_ids]] >= 0) & (labels[recv_v[both_ids]] >= 0)]
    allowed_label = torch.zeros_like(both_validation)
    allowed_label[allowed_positions] = True
    quadrant_values: dict[str, Any] = {}
    for t_bin in range(4):
        for i_bin in range(4):
            cell = (bt_v == t_bin) & (bi_v == i_bin) & allowed_label
            if cell.any():
                same = (labels[src_v[cell]] == labels[recv_v[cell]]).float()
                descriptive[t_bin, i_bin] = same.mean()
                quadrant_values[f"T{t_bin}_I{i_bin}"] = {
                    "count": int(cell.sum()),
                    "same_label_rate": float(same.mean()),
                    "descriptive_only": True,
                }
            else:
                quadrant_values[f"T{t_bin}_I{i_bin}"] = {
                    "count": 0, "same_label_rate": float("nan"), "descriptive_only": True,
                }
    record = {
        "spearman": safe_spearman(pt_v, pi_v),
        "mean_absolute_percentile_gap": gap_summary["mean"],
        "median_absolute_percentile_gap": gap_summary["median"],
        "mean_absolute_percentile_gap_ci_low": mean_ci["ci_low"],
        "mean_absolute_percentile_gap_ci_high": mean_ci["ci_high"],
        "high_text_low_image_fraction": float(hi_t_lo_i.mean()) if hi_t_lo_i.numel() else float("nan"),
        "high_text_low_image_ci_low": hi_ci["ci_low"],
        "high_text_low_image_ci_high": hi_ci["ci_high"],
        "low_text_high_image_fraction": float(lo_t_hi_i.mean()) if lo_t_hi_i.numel() else float("nan"),
        "low_text_high_image_ci_low": lo_ci["ci_low"],
        "low_text_high_image_ci_high": lo_ci["ci_high"],
        "quartile_cross_table": cross_table.tolist(),
        "train_threshold_text": thresholds_t,
        "train_threshold_image": thresholds_i,
        "same_label_by_quadrant": quadrant_values,
        "same_label_descriptive_only": True,
        "validation_receiver_directed_interactions": int(recv_v.numel()),
        "both_endpoints_original_validation_interactions": int(allowed_label.sum()),
        "bootstrap_unit": "validation receiver node",
        "edge_reference": "source-matched non-edge cosine percentile; no raw cosine comparison",
    }
    table = []
    for t_bin in range(4):
        for i_bin in range(4):
            table.append({
                "text_quartile": t_bin,
                "image_quartile": i_bin,
                "count": int(cross_table[t_bin, i_bin]),
                "same_label_rate": quadrant_values[f"T{t_bin}_I{i_bin}"]["same_label_rate"],
                "descriptive_only": True,
            })
    raw = {
        "pairs": pairs,
        "percentile_text": p_t,
        "percentile_image": p_i,
        "validation_source": src_v,
        "validation_receiver": recv_v,
        "validation_text_percentile": pt_v,
        "validation_image_percentile": pi_v,
        "validation_text_bin": bt_v,
        "validation_image_bin": bi_v,
    }
    return record, table, raw


def _edge_context_features(
    pairs: torch.Tensor,
    values: torch.Tensor,
    source_ids: torch.Tensor,
    receiver_ids: torch.Tensor,
    pair_rows: torch.Tensor,
    descriptors: dict[str, torch.Tensor],
    context: torch.Tensor,
    node_degree_bins: torch.Tensor,
    num_nodes: int,
) -> dict[str, torch.Tensor]:
    pair_keys = _pair_keys(pair_rows, num_nodes)
    query = torch.minimum(source_ids, receiver_ids) * num_nodes + torch.maximum(source_ids, receiver_ids)
    pos = torch.searchsorted(pair_keys, query)
    if pair_keys.numel() == 0 or not torch.equal(pair_keys[pos], query):
        raise RuntimeError("directed interactions could not be mapped to physical edges")
    out = {name: value[pos] for name, value in descriptors.items() if name not in {"pairs", "degree", "degree_quantile", "same_label_edge"}}
    out["compatibility"] = values[pos]
    out["context_novelty"] = context[pos]
    out["source_degree_bin"] = node_degree_bins[source_ids]
    out["receiver_degree_bin"] = node_degree_bins[receiver_ids]
    out["pair_index"] = pos
    return out


def _matched_structure_effect(
    values: torch.Tensor,
    semantic_bin: torch.Tensor,
    structure_bin: torch.Tensor,
    source_degree_bin: torch.Tensor,
    receiver_degree_bin: torch.Tensor,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    valid = (semantic_bin >= 0) & (structure_bin >= 0) & (source_degree_bin >= 0) & (receiver_degree_bin >= 0) & torch.isfinite(values)
    groups: dict[tuple[int, int, int], dict[str, list[int]]] = defaultdict(lambda: {"low": [], "high": []})
    for idx in torch.where(valid)[0].tolist():
        key = (int(semantic_bin[idx]), int(source_degree_bin[idx]), int(receiver_degree_bin[idx]))
        if int(structure_bin[idx]) == 0:
            groups[key]["low"].append(idx)
        elif int(structure_bin[idx]) == 3:
            groups[key]["high"].append(idx)
    matched = {key: min(len(value["low"]), len(value["high"])) for key, value in groups.items()}
    matched = {key: count for key, count in matched.items() if count > 0}
    if not matched:
        return {
            "matched_strata": 0, "matched_interaction_count_per_arm": 0,
            "mean_high_minus_low": float("nan"), "variance_across_repeats": float("nan"),
            "ci_low": float("nan"), "ci_high": float("nan"), "repeat_q025": float("nan"), "repeat_q975": float("nan"),
            "repeats": int(repeats), "status": "NO_COMMON_HIGH_LOW_STRATA",
        }
    rng = np.random.default_rng(int(seed))
    values_np = values.detach().float().cpu().numpy()
    effects = []
    for _ in range(int(repeats)):
        weighted_delta = 0.0
        total = 0
        for key in sorted(matched):
            count = matched[key]
            row = groups[key]
            low_idx = rng.choice(row["low"], size=count, replace=False)
            high_idx = rng.choice(row["high"], size=count, replace=False)
            weighted_delta += float(values_np[high_idx].mean() - values_np[low_idx].mean()) * count
            total += count
        effects.append(weighted_delta / max(total, 1))
    array = np.asarray(effects, dtype=np.float64)
    mean = float(array.mean())
    sd = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    radius = float(student_t.ppf(0.975, df=len(array) - 1) * sd / math.sqrt(len(array))) if len(array) > 1 else float("nan")
    return {
        "matched_strata": int(len(matched)),
        "matched_interaction_count_per_arm": int(sum(matched.values())),
        "mean_high_minus_low": mean,
        "variance_across_repeats": float(array.var(ddof=1)) if len(array) > 1 else 0.0,
        "ci_low": mean - radius,
        "ci_high": mean + radius,
        "repeat_q025": float(np.quantile(array, 0.025)),
        "repeat_q975": float(np.quantile(array, 0.975)),
        "repeats": int(repeats),
        "status": "MATCHED",
        "matching_strata": "semantic_bin × source_degree_quartile × receiver_degree_quartile",
        "interval_method": "t interval across deterministic balanced subsampling repetitions",
    }


def _regression_comparison(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray) -> dict[str, Any]:
    valid_train = np.isfinite(train_y) & np.isfinite(train_x).any(axis=1)
    valid_val = np.isfinite(val_y) & np.isfinite(val_x).any(axis=1)
    train_x, train_y = train_x[valid_train], train_y[valid_train]
    val_x, val_y = val_x[valid_val], val_y[valid_val]
    if train_y.size < 2 or val_y.size < 2:
        return {"train_count": int(train_y.size), "validation_count": int(val_y.size), "r2": float("nan"), "mae": float("nan")}
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    model.fit(train_x, train_y)
    pred = model.predict(val_x)
    return {
        "train_count": int(train_y.size),
        "validation_count": int(val_y.size),
        "r2": float(r2_score(val_y, pred)),
        "mae": float(mean_absolute_error(val_y, pred)),
        "ridge_alpha": 1.0,
    }


def p3_structure_response(
    host: FeatureOnlyMultimodalDiagnosticHost,
    edge_index: torch.Tensor,
    h_t: torch.Tensor,
    h_i: torch.Tensor,
    aligned_t_to_i: torch.Tensor,
    aligned_i_to_t: torch.Tensor,
    labels: torch.Tensor,
    split: StudySplit,
    p1_values: dict[str, tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    *,
    semantic_bins_thresholds: dict[str, dict[str, float]],
    max_interactions: int = 50000,
    alphas: list[float] | None = None,
    bootstrap_reps: int = 500,
    matched_repeats: int = 20,
    seed: int = 0,
    batch_size: int = 4096,
    descriptors: dict[str, torch.Tensor] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, torch.Tensor]]:
    alphas = list(alphas or [0.05, 0.1, 0.2])
    physical = canonical_pairs(edge_index)
    descriptors = descriptors or structural_edge_descriptors(edge_index, h_t.size(0))
    # Canonical pair rows in both helpers are sorted; retain an explicit key check.
    desc_pairs = descriptors["pairs"].long().cpu()
    n = int(h_t.size(0))
    desc_keys = _pair_keys(desc_pairs, n)
    raw_rows: list[dict[str, Any]] = []
    regression_rows: list[dict[str, Any]] = []
    raw: dict[str, torch.Tensor] = {"physical_pairs": physical}
    degree_bins, degree_cuts = node_quantile_bins(descriptors["degree"], split.original_train)
    raw["node_degree_quartile"] = degree_bins
    table_by_modality: dict[str, dict[str, torch.Tensor]] = {}

    for modality_index, modality in enumerate(("text", "image")):
        target_states, other_states = (h_t, h_i) if modality == "text" else (h_i, h_t)
        cross_source_states = aligned_i_to_t if modality == "text" else aligned_t_to_i
        p_pairs, compatibility = p1_values[modality]
        p_rows = p_pairs.detach().long().cpu().T.contiguous()
        p_keys = _pair_keys(p_rows, n)
        positions = torch.searchsorted(p_keys, desc_keys)
        p_aligned = compatibility[positions]
        sem_thresholds = semantic_bins_thresholds[modality]
        sem_bins_pair = bins_from_thresholds(p_aligned, sem_thresholds)
        embedded = descriptors["edge_embeddedness"].float()
        structure_thresholds = train_edge_quartile_thresholds(embedded, desc_pairs, split.original_train)
        structure_bins_pair = bins_from_thresholds(embedded, structure_thresholds)
        context = context_novelty(target_states, desc_pairs)

        directed_source = torch.cat([desc_pairs[:, 0], desc_pairs[:, 1]])
        directed_receiver = torch.cat([desc_pairs[:, 1], desc_pairs[:, 0]])
        directed_pair_idx = torch.cat([torch.arange(desc_pairs.size(0)), torch.arange(desc_pairs.size(0))])
        mask_fit = torch.zeros(n, dtype=torch.bool); mask_fit[split.diagnostic_fit] = True
        mask_val = torch.zeros(n, dtype=torch.bool); mask_val[split.original_validation] = True
        node_sem = sem_bins_pair[directed_pair_idx]
        node_struct = structure_bins_pair[directed_pair_idx]
        receiver_q = degree_bins[directed_receiver]
        eligible = (
            ((mask_fit[directed_receiver]) | (mask_val[directed_receiver]))
            & (node_sem >= 0) & (node_struct >= 0)
            & (degree_bins[directed_source] >= 0) & (receiver_q >= 0)
            & torch.isfinite(context[directed_pair_idx])
        )
        eligible_ids = torch.where(eligible)[0]
        strata = [(int(node_sem[i]), int(receiver_q[i])) for i in eligible_ids.tolist()]
        sampled_fit_local = torch.where(mask_fit[directed_receiver[eligible_ids]])[0]
        sampled_val_local = torch.where(mask_val[directed_receiver[eligible_ids]])[0]
        fit_selected, fit_allocation = stratified_sample(
            sampled_fit_local, [strata[i] for i in sampled_fit_local.tolist()], max_interactions, seed + 100 + modality_index
        )
        val_selected, val_allocation = stratified_sample(
            sampled_val_local, [strata[i] for i in sampled_val_local.tolist()], max_interactions, seed + 200 + modality_index
        )
        fit_directed_ids = eligible_ids[fit_selected]
        val_directed_ids = eligible_ids[val_selected]
        chosen = torch.cat([fit_directed_ids, val_directed_ids])
        src_ids, recv_ids = directed_source[chosen], directed_receiver[chosen]
        if chosen.numel() == 0:
            raw_rows.append({"modality": modality, "row_type": "failure", "status": "NO_ELIGIBLE_INTERACTIONS"})
            continue

        same_response = central_difference_edges(
            host, target_states, other_states, src_ids, recv_ids, labels, modality, alphas, device, batch_size
        )
        cross_source = cross_source_states[src_ids]
        cross_direction = cross_source - target_states[recv_ids]
        cross_response = central_difference_directions(
            host, target_states, other_states, cross_direction, recv_ids, labels, modality, alphas, device, batch_size
        )
        pair_idx = directed_pair_idx[chosen]
        local_features = {
            "compatibility": p_aligned[pair_idx],
            "context_novelty": context[pair_idx],
            "embeddedness": embedded[pair_idx],
            "common_neighbors": descriptors["common_neighbors"][pair_idx],
            "neighbor_jaccard": descriptors["neighbor_jaccard"][pair_idx],
            "adamic_adar": descriptors["adamic_adar"][pair_idx],
            "source_degree": descriptors["degree"][src_ids],
            "receiver_degree": descriptors["degree"][recv_ids],
            "semantic_bin": sem_bins_pair[pair_idx],
            "structure_bin": structure_bins_pair[pair_idx],
            "source_degree_bin": degree_bins[src_ids],
            "receiver_degree_bin": degree_bins[recv_ids],
        }
        n_fit = fit_directed_ids.numel()
        is_report = torch.cat([torch.zeros(n_fit, dtype=torch.bool), torch.ones(val_directed_ids.numel(), dtype=torch.bool)])
        validation_receiver = recv_ids[is_report]
        for alpha in alphas:
            d_same, d_cross = same_response[alpha], cross_response[alpha]
            for semantic_bin in range(4):
                for structure_bin in range(4):
                    cell = (local_features["semantic_bin"] == semantic_bin) & (local_features["structure_bin"] == structure_bin) & is_report
                    values = d_same[cell]
                    blocks = recv_ids[cell]
                    summary = finite_summary(values)
                    ci = block_bootstrap_mean_ci(values, blocks, reps=bootstrap_reps, seed=seed + modality_index * 1000 + int(alpha * 10000) + semantic_bin * 10 + structure_bin)
                    cross_cell = d_cross[cell]
                    raw_rows.append({
                        "row_type": "semantic_structure_cell",
                        "modality": modality,
                        "alpha": float(alpha),
                        "semantic_bin": semantic_bin,
                        "structure_bin": structure_bin,
                        "count": summary["count"],
                        "mean_D_same": summary["mean"],
                        "median_D_same": summary["median"],
                        "fraction_D_same_below_zero": float((values < 0).float().mean()) if values.numel() else float("nan"),
                        "D_same_ci_low": ci["ci_low"],
                        "D_same_ci_high": ci["ci_high"],
                        "mean_D_cross_secondary": float(cross_cell.mean()) if cross_cell.numel() else float("nan"),
                        "receiver_blocks": ci["block_count"],
                        "bootstrap_unit": "validation receiver node",
                        "semantic_thresholds_train_edges": sem_thresholds,
                        "structure_thresholds_train_edges": structure_thresholds,
                    })

        primary = same_response[0.1] if 0.1 in same_response else same_response[alphas[0]]
        report_slice = slice(n_fit, None)
        matched = _matched_structure_effect(
            primary[report_slice],
            local_features["semantic_bin"][report_slice],
            local_features["structure_bin"][report_slice],
            local_features["source_degree_bin"][report_slice],
            local_features["receiver_degree_bin"][report_slice],
            matched_repeats,
            seed + 500 + modality_index,
        )
        raw_rows.append({
            "row_type": "matched_high_low_structure",
            "modality": modality,
            "alpha": 0.1,
            **matched,
            "comparison": "high embeddedness quartile minus low embeddedness quartile",
        })

        feature_names_sem = ["compatibility", "context_novelty"]
        feature_names_full = feature_names_sem + [
            "embeddedness", "common_neighbors", "neighbor_jaccard", "adamic_adar", "source_degree", "receiver_degree"
        ]
        features_sem = torch.stack([local_features[name] for name in feature_names_sem], dim=1).numpy()
        features_full = torch.stack([local_features[name] for name in feature_names_full], dim=1).numpy()
        train_mask = ~is_report
        val_mask_sample = is_report
        model_sem = _regression_comparison(features_sem[train_mask], primary[train_mask].numpy(), features_sem[val_mask_sample], primary[val_mask_sample].numpy())
        model_full = _regression_comparison(features_full[train_mask], primary[train_mask].numpy(), features_full[val_mask_sample], primary[val_mask_sample].numpy())
        regression_rows.append({
            "modality": modality,
            "target": "continuous D_same at alpha=0.1",
            "semantic_only_features": "compatibility percentile + context novelty",
            "semantic_only": model_sem,
            "semantic_plus_structure_features": "semantic-only + embeddedness/common-neighbors/Jaccard/Adamic-Adar/source-degree/receiver-degree",
            "semantic_plus_structure": model_full,
            "incremental_explanatory_signal_delta_R2": model_full["r2"] - model_sem["r2"],
            "interpretation": "incremental explanatory signal; not router learnability",
        })
        raw[f"{modality}_source"] = src_ids
        raw[f"{modality}_receiver"] = recv_ids
        raw[f"{modality}_is_report"] = is_report
        raw[f"{modality}_D_same"] = torch.stack([same_response[a] for a in alphas])
        raw[f"{modality}_D_cross"] = torch.stack([cross_response[a] for a in alphas])
        raw[f"{modality}_semantic_bin"] = local_features["semantic_bin"]
        raw[f"{modality}_structure_bin"] = local_features["structure_bin"]
        raw[f"{modality}_source_degree_bin"] = local_features["source_degree_bin"]
        raw[f"{modality}_receiver_degree_bin"] = local_features["receiver_degree_bin"]
        raw[f"{modality}_compatibility"] = local_features["compatibility"]
        raw[f"{modality}_context_novelty"] = local_features["context_novelty"]
        table_by_modality[modality] = {"compatibility": p_aligned, "semantic_bins": sem_bins_pair}
        raw_rows.append({
            "row_type": "sampling_metadata",
            "modality": modality,
            "total_fit_eligible": int(sampled_fit_local.numel()),
            "sampled_fit": int(fit_selected.numel()),
            "total_validation_eligible": int(sampled_val_local.numel()),
            "sampled_validation": int(val_selected.numel()),
            "fit_strata_counts": fit_allocation,
            "validation_strata_counts": val_allocation,
            "degree_thresholds_from_original_train_nodes": degree_cuts,
            "semantic_thresholds_train_edges": sem_thresholds,
            "embeddedness_thresholds_train_edges": structure_thresholds,
        })
    raw["p3_alpha_values"] = torch.tensor(alphas)
    return raw_rows, regression_rows, raw
