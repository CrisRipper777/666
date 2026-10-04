from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from .splits import StudySplit


class FeatureOnlyMultimodalDiagnosticHost(nn.Module):
    """Graph-free text/image reference host used only by analysis code."""

    def __init__(self, text_dim: int, image_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.text_projector = nn.Sequential(
            nn.LayerNorm(int(text_dim)),
            nn.Linear(int(text_dim), self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.image_projector = nn.Sequential(
            nn.LayerNorm(int(image_dim)),
            nn.Linear(int(image_dim), self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.classifier = nn.Linear(self.hidden_dim, int(num_classes))

    def project(self, x_t: torch.Tensor, x_i: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.text_projector(x_t), self.image_projector(x_i)

    def logits_from_hidden(self, h_t: torch.Tensor, h_i: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.fusion(torch.cat([h_t, h_i], dim=-1)))

    def logits_for_target(
        self,
        target: torch.Tensor,
        other: torch.Tensor,
        target_modality: str,
    ) -> torch.Tensor:
        if target_modality == "text":
            return self.logits_from_hidden(target, other)
        if target_modality == "image":
            return self.logits_from_hidden(other, target)
        raise ValueError(f"target_modality must be text or image, got {target_modality!r}")

    def forward(self, x_t: torch.Tensor, x_i: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_t, h_i = self.project(x_t, x_i)
        return self.logits_from_hidden(h_t, h_i), h_t, h_i


@dataclass(frozen=True)
class HostTrainingConfig:
    hidden_dim: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 200
    patience: int = 20
    min_delta: float = 1e-5
    batch_size: int = 8192
    train_node_cap: int | None = None
    low_quality_accuracy_gain: float = 0.02


def _classification_metrics(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> dict[str, float]:
    pred = logits.argmax(-1).detach().cpu().numpy()
    target = labels.detach().long().cpu().numpy()
    return {
        "accuracy": float(accuracy_score(target, pred)),
        "macro_f1": float(f1_score(target, pred, labels=list(range(num_classes)), average="macro", zero_division=0)),
    }


def _stratified_cap(index: torch.Tensor, labels: torch.Tensor, cap: int, seed: int) -> torch.Tensor:
    index = index.detach().long().cpu()
    if cap <= 0 or index.numel() <= cap:
        return index
    y = labels[index].detach().long().cpu().numpy()
    ids = index.numpy()
    classes = np.unique(y)
    rng = np.random.default_rng(int(seed))
    groups = {int(cls): rng.permutation(ids[y == cls]) for cls in classes}
    allocations = {cls: 0 for cls in groups}
    active = sorted(groups)
    budget = int(cap)
    if budget < len(active):
        selected_classes = sorted(rng.choice(active, size=budget, replace=False).tolist())
        for cls in selected_classes:
            allocations[cls] = 1
    else:
        for cls in active:
            allocations[cls] = 1
        remaining = budget - len(active)
        while remaining:
            candidates = [cls for cls in active if allocations[cls] < len(groups[cls])]
            if not candidates:
                break
            cls = max(candidates, key=lambda c: (len(groups[c]) - allocations[c], -c))
            allocations[cls] += 1
            remaining -= 1
    picked = np.concatenate([groups[cls][:allocations[cls]] for cls in active if allocations[cls]])
    return torch.from_numpy(np.sort(picked).copy())


def _run_logits(
    model: FeatureOnlyMultimodalDiagnosticHost,
    x_t: torch.Tensor,
    x_i: torch.Tensor,
    indices: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, indices.numel(), batch_size):
            ids = indices[start : start + batch_size]
            logits, _, _ = model(x_t[ids].to(device), x_i[ids].to(device))
            outputs.append(logits.detach().cpu())
    return torch.cat(outputs, 0) if outputs else torch.empty((0, model.classifier.out_features))


def encode_all_modalities(
    model: FeatureOnlyMultimodalDiagnosticHost,
    x_t: torch.Tensor,
    x_i: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    text_out, image_out = [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, x_t.size(0), batch_size):
            end = min(start + batch_size, x_t.size(0))
            h_t, h_i = model.project(x_t[start:end].to(device), x_i[start:end].to(device))
            text_out.append(h_t.detach().float().cpu())
            image_out.append(h_i.detach().float().cpu())
    return torch.cat(text_out), torch.cat(image_out)


def train_diagnostic_host(
    x_t: torch.Tensor,
    x_i: torch.Tensor,
    labels: torch.Tensor,
    split: StudySplit,
    num_classes: int,
    seed: int,
    device: torch.device,
    config: HostTrainingConfig,
) -> dict[str, Any]:
    if config.epochs < 1 or config.patience < 1:
        raise ValueError("host epochs and patience must be positive")
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    model = FeatureOnlyMultimodalDiagnosticHost(
        int(x_t.size(1)), int(x_i.size(1)), config.hidden_dim, num_classes
    ).to(device)
    fit_used = split.diagnostic_fit
    if config.train_node_cap is not None:
        fit_used = _stratified_cap(fit_used, labels, int(config.train_node_cap), seed + 991)
    fit_used_device = fit_used.to(device)
    calib_device = split.diagnostic_calib.to(device)
    x_t_device, x_i_device = x_t.float().to(device), x_i.float().to(device)
    y_device = labels.long().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_loss = float("inf")
    best_epoch = 0
    patience_left = int(config.patience)
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, int(config.epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, _, _ = model(x_t_device[fit_used_device], x_i_device[fit_used_device])
        loss = nn.functional.cross_entropy(logits, y_device[fit_used_device])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"diagnostic host seed={seed} diverged at epoch={epoch}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, error_if_nonfinite=True)
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            calib_logits, _, _ = model(x_t_device[calib_device], x_i_device[calib_device])
            calib_loss = float(nn.functional.cross_entropy(calib_logits, y_device[calib_device]).item())
        if calib_loss < best_loss - config.min_delta:
            best_loss = calib_loss
            best_epoch = epoch
            patience_left = int(config.patience)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            patience_left -= 1
        if patience_left <= 0:
            break
    if best_state is None:
        raise RuntimeError("diagnostic host did not produce a calibration-selected checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    logits_by_split: dict[str, torch.Tensor] = {}
    indices_by_split = {
        "fit": split.diagnostic_fit,
        "calib": split.diagnostic_calib,
        "report": split.original_validation,
    }
    metrics: dict[str, dict[str, float]] = {}
    for name, indices in indices_by_split.items():
        logits = _run_logits(model, x_t, x_i, indices, device, config.batch_size)
        logits_by_split[name] = logits
        metrics[name] = _classification_metrics(logits, labels[indices], num_classes)

    fit_labels = labels[split.diagnostic_fit].long()
    counts = torch.bincount(fit_labels, minlength=num_classes)
    majority_class = int(counts.argmax())
    majority_metrics = {}
    for name, indices in indices_by_split.items():
        pred = torch.full((indices.numel(),), majority_class, dtype=torch.long)
        majority_metrics[name] = _classification_metrics(
            torch.nn.functional.one_hot(pred, num_classes).float(), labels[indices], num_classes
        )

    h_t, h_i = encode_all_modalities(model, x_t, x_i, device, config.batch_size)
    low_quality = metrics["report"]["accuracy"] - majority_metrics["report"]["accuracy"] < float(config.low_quality_accuracy_gain)
    return {
        "model": model,
        "h_t": h_t,
        "h_i": h_i,
        "metrics": metrics,
        "majority_baseline": majority_metrics,
        "majority_class_from_fit": majority_class,
        "low_host_quality": bool(low_quality),
        "low_host_quality_definition": f"report accuracy gain over fit majority baseline < {config.low_quality_accuracy_gain}",
        "best_epoch": int(best_epoch),
        "best_calibration_ce": float(best_loss),
        "train_nodes_used": int(fit_used.numel()),
        "train_nodes_available": int(split.diagnostic_fit.numel()),
        "config": asdict(config),
        "seed": int(seed),
        "test_label_access": False,
        "report_split": "original_validation",
    }
