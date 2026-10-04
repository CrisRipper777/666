from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch


def finite_summary(values: torch.Tensor | np.ndarray | list[float]) -> dict[str, float | int]:
    x = torch.as_tensor(values, dtype=torch.float64).flatten()
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return {"count": 0, "mean": float("nan"), "median": float("nan"), "std": float("nan")}
    return {
        "count": int(x.numel()),
        "mean": float(x.mean()),
        "median": float(x.median()),
        "std": float(x.std(unbiased=False)) if x.numel() > 1 else 0.0,
    }


def block_bootstrap_mean_ci(
    values: torch.Tensor,
    blocks: torch.Tensor,
    *,
    reps: int = 500,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Percentile CI from resampling receiver blocks, keeping all rows per block."""
    x = values.detach().double().cpu().flatten()
    b = blocks.detach().long().cpu().flatten()
    if x.numel() != b.numel():
        raise ValueError("values and block IDs must have the same length")
    valid = torch.isfinite(x)
    x, b = x[valid].numpy(), b[valid].numpy()
    unique = np.unique(b)
    if not len(x):
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "block_count": 0}
    point = float(x.mean())
    if len(unique) < 2 or reps < 1:
        return {"mean": point, "ci_low": float("nan"), "ci_high": float("nan"), "block_count": int(len(unique))}
    sums = np.zeros(len(unique), dtype=np.float64)
    counts = np.zeros(len(unique), dtype=np.float64)
    block_to_pos = {int(v): i for i, v in enumerate(unique)}
    for value, block in zip(x, b, strict=True):
        pos = block_to_pos[int(block)]
        sums[pos] += value
        counts[pos] += 1.0
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, len(unique), size=(int(reps), len(unique)))
    sampled_sums = sums[draws].sum(axis=1)
    sampled_counts = counts[draws].sum(axis=1)
    means = sampled_sums / np.maximum(sampled_counts, 1.0)
    tail = (1.0 - confidence) / 2.0
    return {
        "mean": point,
        "ci_low": float(np.quantile(means, tail)),
        "ci_high": float(np.quantile(means, 1.0 - tail)),
        "block_count": int(len(unique)),
        "bootstrap_reps": int(reps),
        "bootstrap_unit": "validation receiver node",
        "confidence": float(confidence),
    }


def bins_from_thresholds(values: torch.Tensor, thresholds: dict[str, float]) -> torch.Tensor:
    out = torch.full(values.shape, -1, dtype=torch.long)
    cuts = torch.tensor(
        [thresholds.get("q25", float("nan")), thresholds.get("q50", float("nan")), thresholds.get("q75", float("nan"))],
        dtype=values.dtype,
    )
    valid = torch.isfinite(values) & torch.isfinite(cuts).all()
    if valid.any():
        out[valid] = torch.bucketize(values[valid].contiguous(), cuts.contiguous(), right=False)
    return out


def node_quantile_bins(values: torch.Tensor, reference: torch.Tensor) -> tuple[torch.Tensor, list[float]]:
    """Quartiles using only the supplied reference-node values."""
    ref = values[reference.detach().long().cpu()].float()
    ref = ref[torch.isfinite(ref)]
    out = torch.full(values.shape, -1, dtype=torch.long)
    if ref.numel() == 0:
        return out, [float("nan")] * 3
    cuts = torch.quantile(ref, torch.tensor([0.25, 0.5, 0.75]))
    valid = torch.isfinite(values)
    out[valid] = torch.bucketize(values[valid].float().contiguous(), cuts.contiguous(), right=False)
    return out, [float(v) for v in cuts]


def stratified_sample(
    indices: torch.Tensor,
    strata: list[tuple[int, ...]] | torch.Tensor,
    max_items: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Deterministic proportional sampling with at least one row per nonempty stratum."""
    ids = indices.detach().long().cpu().flatten()
    if isinstance(strata, torch.Tensor):
        strata_list = [tuple(int(x) for x in row) if row.ndim else (int(row),) for row in strata.detach().cpu()]
    else:
        strata_list = [tuple(map(int, row)) for row in strata]
    if len(strata_list) != ids.numel():
        raise ValueError("one stratum key is required per index")
    if max_items <= 0 or ids.numel() <= max_items:
        return ids.sort().values, {str(key): count for key, count in _group_counts(strata_list).items()}

    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for idx, key in zip(ids.tolist(), strata_list, strict=True):
        groups[key].append(int(idx))
    keys = sorted(groups)
    budget = min(int(max_items), len(ids))
    allocations = {key: 0 for key in keys}
    if budget < len(keys):
        rng = np.random.default_rng(int(seed))
        chosen_keys = sorted(rng.choice(len(keys), size=budget, replace=False).tolist())
        for pos in chosen_keys:
            allocations[keys[pos]] = 1
    else:
        for key in keys:
            allocations[key] = 1
        remaining = budget - len(keys)
        capacities = {key: len(groups[key]) - 1 for key in keys}
        while remaining > 0:
            active = [key for key in keys if capacities[key] > 0]
            if not active:
                break
            active_total = sum(len(groups[key]) for key in active)
            quotas = {key: remaining * len(groups[key]) / active_total for key in active}
            progressed = 0
            for key in active:
                add = min(capacities[key], int(quotas[key]))
                allocations[key] += add
                capacities[key] -= add
                progressed += add
            remaining -= progressed
            if remaining <= 0:
                break
            order = sorted(active, key=lambda key: (-(quotas[key] - int(quotas[key])), key))
            for key in order:
                if remaining == 0:
                    break
                if capacities[key] > 0:
                    allocations[key] += 1
                    capacities[key] -= 1
                    remaining -= 1
            if progressed == 0 and remaining > 0 and not any(capacities.values()):
                break

    rng = np.random.default_rng(int(seed))
    selected: list[int] = []
    actual: dict[str, int] = {}
    for key in keys:
        count = allocations[key]
        if not count:
            continue
        candidates = np.asarray(groups[key], dtype=np.int64)
        chosen = rng.choice(candidates, size=count, replace=False)
        selected.extend(map(int, chosen))
        actual[str(key)] = int(count)
    return torch.tensor(sorted(selected), dtype=torch.long), actual


def _group_counts(keys: list[tuple[int, ...]]) -> dict[tuple[int, ...], int]:
    result: dict[tuple[int, ...], int] = defaultdict(int)
    for key in keys:
        result[key] += 1
    return dict(result)
