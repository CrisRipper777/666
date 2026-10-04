from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import numpy as np
import torch


def _index_sha256(index: torch.Tensor) -> str:
    values = index.detach().long().cpu().contiguous().numpy().astype("<i8", copy=False)
    return hashlib.sha256(values.tobytes()).hexdigest()


@dataclass(frozen=True)
class StudySplit:
    """Label-use boundary for the interaction study.

    ``diagnostic_fit`` and ``diagnostic_calib`` partition the original train
    indices. ``original_validation`` is the only report split. Test indices are
    retained for audit and masking; no study computation indexes their labels.
    """

    diagnostic_fit: torch.Tensor
    diagnostic_calib: torch.Tensor
    original_train: torch.Tensor
    original_validation: torch.Tensor
    original_test: torch.Tensor
    seed: int
    index_sha256: dict[str, str]
    stratification: str = "per-class deterministic 80/20 split of original train"

    def metadata(self) -> dict:
        return {
            "seed": int(self.seed),
            "fit_split": "diagnostic_fit",
            "calib_split": "diagnostic_calib",
            "report_split": "original_validation",
            "test_label_access": False,
            "test_metric_access": False,
            "transductive_topology": True,
            "stratification": self.stratification,
            "split_counts": {
                "diagnostic_fit": int(self.diagnostic_fit.numel()),
                "diagnostic_calib": int(self.diagnostic_calib.numel()),
                "original_train": int(self.original_train.numel()),
                "original_validation": int(self.original_validation.numel()),
                "original_test": int(self.original_test.numel()),
            },
            "index_sha256": dict(self.index_sha256),
        }


def make_study_split(
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    validation_idx: torch.Tensor,
    test_idx: torch.Tensor,
    seed: int = 0,
    fit_fraction: float = 0.8,
) -> StudySplit:
    if not 0.0 < fit_fraction < 1.0:
        raise ValueError("fit_fraction must be between 0 and 1")
    train = train_idx.detach().long().cpu().flatten()
    validation = validation_idx.detach().long().cpu().flatten()
    test = test_idx.detach().long().cpu().flatten()
    for name, value in (("train", train), ("validation", validation), ("test", test)):
        if value.unique().numel() != value.numel():
            raise ValueError(f"{name} split contains duplicate indices")
    if torch.isin(train, validation).any() or torch.isin(train, test).any() or torch.isin(validation, test).any():
        raise ValueError("original train/validation/test splits must be disjoint")
    if train.numel() == 0 or validation.numel() == 0 or test.numel() == 0:
        raise ValueError("all original NC splits must be nonempty")

    # This is the only label read in split construction: labels at original-train
    # indices only. Validation and test labels are not consulted for stratifying.
    y_train = labels.detach().long().cpu()[train]
    if (y_train < 0).any():
        raise ValueError("original train labels must be nonnegative for stratification")
    rng = np.random.default_rng(int(seed))
    fit_parts: list[np.ndarray] = []
    calib_parts: list[np.ndarray] = []
    train_np = train.numpy()
    y_np = y_train.numpy()
    for class_id in np.unique(y_np):
        class_ids = train_np[y_np == class_id]
        class_ids = rng.permutation(class_ids)
        if len(class_ids) < 2:
            n_calib = 0
        else:
            n_calib = int(round(len(class_ids) * (1.0 - fit_fraction)))
            n_calib = min(max(n_calib, 1), len(class_ids) - 1)
        calib_parts.append(class_ids[:n_calib])
        fit_parts.append(class_ids[n_calib:])

    fit = np.concatenate(fit_parts).astype(np.int64, copy=False)
    calib = np.concatenate(calib_parts).astype(np.int64, copy=False)
    if fit.size == 0 or calib.size == 0:
        raise ValueError("stratified split produced an empty fit or calibration split")
    fit = rng.permutation(fit)
    calib = rng.permutation(calib)
    fit_t = torch.from_numpy(fit.copy())
    calib_t = torch.from_numpy(calib.copy())
    if torch.isin(fit_t, validation).any() or torch.isin(fit_t, test).any():
        raise AssertionError("diagnostic fit leaked outside original train")
    if torch.isin(calib_t, validation).any() or torch.isin(calib_t, test).any():
        raise AssertionError("diagnostic calibration leaked outside original train")
    if fit_t.numel() + calib_t.numel() != train.numel():
        raise AssertionError("diagnostic fit/calibration do not partition original train")

    index_sets = {
        "diagnostic_fit": fit_t,
        "diagnostic_calib": calib_t,
        "original_train": train,
        "original_validation": validation,
        "original_test": test,
    }
    return StudySplit(
        diagnostic_fit=fit_t,
        diagnostic_calib=calib_t,
        original_train=train,
        original_validation=validation,
        original_test=test,
        seed=int(seed),
        index_sha256={key: _index_sha256(value) for key, value in index_sets.items()},
    )


def mask_test_labels(labels: torch.Tensor, split: StudySplit) -> torch.Tensor:
    """Return labels visible to the study, replacing all test entries by -1."""
    visible = labels.detach().long().cpu().clone()
    # The operation writes the masked positions without reading their label values.
    visible[split.original_test] = -1
    return visible


def split_metadata_json(split: StudySplit) -> dict:
    return asdict(split) | split.metadata()
