from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .splits import StudySplit


@dataclass
class RidgeAligner:
    weight: torch.Tensor
    source_mean: torch.Tensor
    target_mean: torch.Tensor
    regularization: float
    source_name: str
    target_name: str

    def apply(self, source: torch.Tensor) -> torch.Tensor:
        return (source.float() - self.source_mean) @ self.weight + self.target_mean


def fit_ridge_aligner(
    source: torch.Tensor,
    target: torch.Tensor,
    fit_idx: torch.Tensor,
    regularization: float,
    source_name: str,
    target_name: str,
) -> RidgeAligner:
    if source.size(1) != target.size(1):
        raise ValueError("ridge alignment requires equal hidden dimensions")
    ids = fit_idx.detach().long().cpu()
    x = source[ids].float().cpu()
    y = target[ids].float().cpu()
    x_mean, y_mean = x.mean(0, keepdim=True), y.mean(0, keepdim=True)
    xc, yc = x - x_mean, y - y_mean
    gram = xc.T @ xc
    rhs = xc.T @ yc
    eye = torch.eye(gram.size(0), dtype=gram.dtype)
    weight = torch.linalg.solve(gram + float(regularization) * eye, rhs)
    return RidgeAligner(weight, x_mean, y_mean, float(regularization), source_name, target_name)


def alignment_metrics(
    aligner: RidgeAligner,
    source: torch.Tensor,
    target: torch.Tensor,
    indices: torch.Tensor,
) -> dict[str, float | int]:
    ids = indices.detach().long().cpu()
    prediction = aligner.apply(source[ids])
    truth = target[ids].float().cpu()
    cosine = F.cosine_similarity(prediction, truth, dim=-1, eps=1e-8)
    mse = (prediction - truth).square().mean()
    mean_baseline = truth.mean(0, keepdim=True).expand_as(truth)
    baseline_mse = (mean_baseline - truth).square().mean()
    return {
        "count": int(ids.numel()),
        "cosine": float(cosine.mean()),
        "cosine_median": float(cosine.median()),
        "mse": float(mse),
        "mean_baseline_mse": float(baseline_mse),
        "mse_skill_vs_constant": float(1.0 - mse / baseline_mse.clamp_min(1e-12)),
    }


def alignment_record(
    t_to_i: RidgeAligner,
    i_to_t: RidgeAligner,
    h_t: torch.Tensor,
    h_i: torch.Tensor,
    split: StudySplit,
    min_validation_cosine: float,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    metrics = {}
    for name, aligner, src, tgt in (
        ("T2I", t_to_i, h_t, h_i),
        ("I2T", i_to_t, h_i, h_t),
    ):
        metrics[name] = {
            "calib": alignment_metrics(aligner, src, tgt, split.diagnostic_calib),
            "validation": alignment_metrics(aligner, src, tgt, split.original_validation),
            "regularization": float(aligner.regularization),
            "fit_split": "diagnostic_fit",
        }
    low = any(
        metrics[direction]["validation"]["cosine"] < float(min_validation_cosine)
        or metrics[direction]["validation"]["mse_skill_vs_constant"] <= 0.0
        for direction in ("T2I", "I2T")
    )
    record = {
        "directions": metrics,
        "validation_cosine_asymmetry_T2I_minus_I2T": metrics["T2I"]["validation"]["cosine"] - metrics["I2T"]["validation"]["cosine"],
        "validation_mse_asymmetry_T2I_minus_I2T": metrics["T2I"]["validation"]["mse"] - metrics["I2T"]["validation"]["mse"],
        "alignment_low_quality": bool(low),
        "low_quality_rule": f"either validation cosine < {min_validation_cosine} or MSE skill vs constant <= 0",
    }
    return record, t_to_i.apply(h_t), i_to_t.apply(h_i)
