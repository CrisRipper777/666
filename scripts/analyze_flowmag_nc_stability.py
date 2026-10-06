#!/usr/bin/env python
"""Run frozen FlowMAG NC representation probes and aggregate the V1 study."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = ["Movies", "Toys", "Grocery", "Reddit-S", "ele-fashion"]
SEEDS = [42, 43, 44]
PROBE_SEED = 2026
PROBE_EPOCHS = 100
PROBE_PATIENCE = 15
PROBE_LR = 1e-2
ROUTING_METRICS = {
    "pi_smooth_given_same_valid": "pi_smooth_given_same_valid",
    "pi_contrast_given_same_valid": "pi_contrast_given_same_valid",
    "same_admission_given_valid": "same_admission_given_valid",
    "cross_admission_given_valid": "cross_admission_given_valid",
    "same_valid_rate": "same_valid_rate",
    "cross_valid_rate": "cross_valid_rate",
    "null_mass_given_cross_valid": "null_mass_given_cross_valid",
    "normalized_cross_attention_entropy": "normalized_cross_attention_entropy_given_valid",
    "same_update_ratio": "same_update_ratio",
    "cross_update_ratio": "cross_update_ratio",
    "graph_update_ratio": "graph_update_ratio",
    "ffn_update_ratio": "ffn_update_ratio",
    "smooth_contrast_cosine": "smooth_contrast_cosine_given_valid",
}
GRADIENT_COMPONENTS = {"contrast", "cross_attention", "controller", "same_relation", "ffn"}


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _parse_logged_int(pattern: str, text: str) -> int | None:
    match = re.search(pattern, text)
    return int(match.group(1)) if match else None


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return statistics.mean(values), statistics.pstdev(values)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _load_dataset_config(dataset: str):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                f"dataset={dataset}", "task=nc", "model=flowmag", "seed=42",
                "num_runs=3", "task.evaluate_test=false", "model.collect_diagnostics=false",
            ],
        )
    return cfg


def _dataset_info(data) -> dict[str, int | None]:
    return {
        "input_dim": int(data.input_dim),
        "num_nodes": int(data.num_nodes),
        "num_classes": int(data.num_classes),
        "text_dim": int(data.x_t.shape[1]) if data.x_t is not None else 0,
        "visual_dim": int(data.x_i.shape[1]) if data.x_i is not None else 0,
    }


def _macro_f1(target: torch.Tensor, prediction: torch.Tensor, labels: list[int]) -> float:
    return float(f1_score(
        target.detach().cpu().numpy(),
        prediction.detach().cpu().numpy(),
        labels=labels,
        average="macro",
        zero_division=0,
    ))


def _fit_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    labels: list[int],
    num_classes: int,
    device: torch.device,
) -> dict[str, float | int]:
    torch.manual_seed(PROBE_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(PROBE_SEED)
    classifier = nn.Linear(int(train_x.size(1)), int(num_classes)).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=PROBE_LR)
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    val_x = val_x.to(device)
    val_y = val_y.to(device)
    best_acc = -1.0
    best_epoch = 0
    best_state = None
    patience = PROBE_PATIENCE
    for epoch in range(1, PROBE_EPOCHS + 1):
        classifier.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(classifier(train_x), train_y)
        loss.backward()
        optimizer.step()
        classifier.eval()
        with torch.no_grad():
            val_prediction = classifier(val_x).argmax(dim=-1)
            accuracy = float((val_prediction == val_y).float().mean().item())
        if accuracy > best_acc + 1e-12:
            best_acc = accuracy
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in classifier.state_dict().items()}
            patience = PROBE_PATIENCE
        else:
            patience -= 1
            if patience <= 0:
                break
    if best_state is None:
        raise RuntimeError("linear probe did not produce a validation checkpoint")
    classifier.load_state_dict(best_state)
    classifier.eval()
    with torch.no_grad():
        val_prediction = classifier(val_x).argmax(dim=-1)
    return {
        "val_acc": best_acc,
        "val_macro_f1": _macro_f1(val_y, val_prediction, labels),
        "probe_best_epoch": best_epoch,
    }


def run_probes(dataset: str, output_root: Path, report_root: Path, device: torch.device) -> list[dict[str, Any]]:
    from src.data import load_mag_data
    from src.models import build_model
    from src.tasks.common import restore_checkpoint_schedule

    cfg = _load_dataset_config(dataset)
    data = load_mag_data(cfg, "nc", 42)
    rows: list[dict[str, Any]] = []
    class_count = int(data.num_classes)
    train_idx = data.train_idx.long().cpu()
    val_idx = data.val_idx.long().cpu()
    train_y = data.y[train_idx].long()
    val_y = data.y[val_idx].long()
    labels = sorted(set(train_y.tolist()) | set(val_y.tolist()))
    majority_class = int(torch.bincount(train_y, minlength=class_count).argmax().item())
    majority_prediction = torch.full_like(val_y, majority_class)
    rows.append({
        "dataset": dataset,
        "seed": "all",
        "block": "majority_baseline",
        "val_acc": float((majority_prediction == val_y).float().mean().item()),
        "val_macro_f1": _macro_f1(val_y, majority_prediction, labels),
        "checkpoint_best_epoch": None,
        "probe_best_epoch": None,
        "probe_seed": PROBE_SEED,
        "status": "complete",
    })

    data_info = _dataset_info(data)
    for seed in SEEDS:
        checkpoint = output_root / dataset / f"flowmag_best_run{SEEDS.index(seed) + 1}.pt"
        if not checkpoint.exists():
            rows.append({
                "dataset": dataset, "seed": seed, "block": "all",
                "val_acc": None, "val_macro_f1": None,
                "checkpoint_best_epoch": None, "probe_best_epoch": None,
                "probe_seed": PROBE_SEED, "status": "missing_checkpoint",
            })
            continue
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = build_model(cfg, data_info)
        model.load_state_dict(payload["model_state"], strict=True)
        restore_checkpoint_schedule(model, int(payload["epoch"]))
        model.to(device).eval()
        z = model.inference(
            data.x, data.edge_index, device=device,
            batch_size=int(cfg.task.inference_batch_size),
        )
        prior_dim = int(cfg.model.prior_dim_per_modality)
        graph_dim = int(cfg.model.graph_block_dim)
        expected_width = 2 * prior_dim + graph_dim
        if z.size(1) != expected_width:
            raise RuntimeError(f"unexpected FlowMAG representation width: {z.size(1)} != {expected_width}")
        blocks = {
            "prior_only": z[:, : 2 * prior_dim],
            "graph_only": z[:, 2 * prior_dim :],
            "prior_plus_graph": z,
        }
        train_features = {name: value[train_idx] for name, value in blocks.items()}
        val_features = {name: value[val_idx] for name, value in blocks.items()}
        for block_name, feature_block in blocks.items():
            metrics = _fit_probe(
                train_features[block_name], train_y,
                val_features[block_name], val_y, labels, class_count, device,
            )
            rows.append({
                "dataset": dataset, "seed": seed, "block": block_name,
                **metrics,
                "checkpoint_best_epoch": int(payload["epoch"]),
                "probe_seed": PROBE_SEED, "status": "complete",
            })
        del model, z, payload, blocks, train_features, val_features
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def _run_rows(output_root: Path, dataset: str) -> tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    metrics_payload = _read_json(output_root / dataset / "run_metrics.json") or {}
    runs: dict[int, dict[str, Any]] = {}
    for run in metrics_payload.get("runs", []):
        seed = int(run.get("seed", -1))
        runs[seed] = run
    diagnostics = {
        seed: _read_jsonl(output_root / dataset / f"epoch_diag_run{index}.jsonl")
        for index, seed in enumerate(SEEDS, start=1)
    }
    return runs, diagnostics


def aggregate_study(datasets: list[str], output_root: Path, report_root: Path, probe_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary_rows: list[dict[str, Any]] = []
    efficiency_rows: list[dict[str, Any]] = []
    routing_by_key: dict[tuple, dict[int, list[float]]] = {}
    gradient_by_key: dict[tuple, dict[int, list[float]]] = {}
    study: dict[str, Any] = {}
    oom_notes: dict[str, str] = {}

    for dataset in datasets:
        runs, diag_by_seed = _run_rows(output_root, dataset)
        dataset_run_details: dict[str, Any] = {}
        val_acc_values: list[float] = []
        val_f1_values: list[float] = []
        for seed in SEEDS:
            run = runs.get(seed)
            if run is None:
                log_path = output_root / dataset / "formal.log"
                log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
                nodes = _parse_logged_int(r"num_nodes=(\d+)", log_text)
                directed_edges = _parse_logged_int(r"num_edges=(\d+)", log_text)
                model_parameters = _parse_logged_int(r"Model parameters=(\d+)", log_text)
                classifier_parameters = _parse_logged_int(r"Classifier parameters=(\d+)", log_text)
                oom = "out of memory" in log_text.lower() or "cuda error: out of memory" in log_text.lower()
                run_index = SEEDS.index(seed) + 1
                was_started = (
                    (output_root / dataset / f"epoch_diag_run{run_index}.jsonl").exists()
                    or (output_root / dataset / f"flowmag_best_run{run_index}.pt").exists()
                )
                if oom and was_started:
                    oom_line = next((
                        line.strip() for line in log_text.splitlines()
                        if "outofmemoryerror" in line.lower() or "out of memory" in line.lower()
                    ), "CUDA OOM reported; model dimensions were unchanged.")
                    detail = f"Seed {seed} was unavailable after CUDA OOM: {oom_line}"
                    initial_log = output_root / dataset / "formal_initial_diagnostics_oom.log"
                    if initial_log.exists():
                        initial_text = initial_log.read_text(errors="replace")
                        initial_line = next((
                            line.strip() for line in initial_text.splitlines()
                            if "outofmemoryerror" in line.lower() or "out of memory" in line.lower()
                        ), None)
                        if initial_line:
                            detail += f" An earlier attempt also OOMed in diagnostic RMS; chunked RMS reduced scratch memory, after which epochs 1–2 completed before the backward-pass OOM. Earlier detail: {initial_line}"
                    oom_notes[dataset] = detail
                    status = "oom"
                elif oom:
                    status = "not_run_after_oom"
                else:
                    status = "missing_or_incomplete"
                dataset_run_details[str(seed)] = {"status": status}
                partial_epochs = len(diag_by_seed.get(seed, []))
                summary_rows.append({
                    "dataset": dataset, "seed": seed, "row_type": "status",
                    "status": status, "val_acc": None, "val_macro_f1": None,
                    "best_epoch": None, "epochs_executed": partial_epochs or None,
                    "model_parameters": model_parameters,
                    "classifier_parameters": classifier_parameters,
                    "total_wall_clock_training_seconds": None,
                })
                efficiency_rows.append({
                    "dataset": dataset, "model": "FlowMAG", "seed": seed,
                    "status": status,
                    "warning": "see run_manifest.json for PyTorch memory details" if status == "oom"
                    else ("not started after the preceding seed OOM" if status == "not_run_after_oom" else None),
                    "model_parameters": model_parameters,
                    "classifier_parameters": classifier_parameters,
                    "nodes": nodes, "directed_edges": directed_edges,
                })
                continue
            metrics = run.get("metrics", {})
            metadata = run.get("metadata", {})
            best_epoch = int(metadata.get("best_epoch", 0))
            val_acc = _float_or_none(metrics.get("val_acc"))
            val_f1 = _float_or_none(metrics.get("val_macro_f1"))
            if val_acc is not None:
                val_acc_values.append(val_acc)
            if val_f1 is not None:
                val_f1_values.append(val_f1)
            summary_rows.append({
                "dataset": dataset, "seed": seed, "row_type": "seed",
                "status": "complete",
                "val_acc": val_acc, "val_macro_f1": val_f1,
                "best_epoch": best_epoch,
                "epochs_executed": metadata.get("epochs_executed"),
                "model_parameters": metadata.get("model_parameters"),
                "classifier_parameters": metadata.get("classifier_parameters"),
                "total_wall_clock_training_seconds": metadata.get("total_wall_clock_training_seconds"),
            })
            diag_rows = diag_by_seed.get(seed, [])
            best_row = next((row for row in diag_rows if int(row.get("epoch", -1)) == best_epoch), None)
            eligible_rows = [row for row in diag_rows if row.get("checkpoint_eligible") is True]
            last5 = eligible_rows[-5:]
            for window, selected in (("best_epoch", [best_row] if best_row else []), ("last5_eligible_median", last5)):
                selected_layers = (
                    (selected[0] or {}).get("layer_diagnostics", [])
                    if window == "best_epoch" and selected
                    else (last5[-1].get("layer_diagnostics", []) if last5 else [])
                )
                for layer_index, layer_diag in enumerate(selected_layers):
                    for modality in ("text", "image"):
                        modality_diag = layer_diag.get(modality, {})
                        for metric, source_key in ROUTING_METRICS.items():
                            if window == "best_epoch":
                                value = _float_or_none(modality_diag.get(source_key))
                            else:
                                samples = []
                                for epoch_row in selected:
                                    layers = epoch_row.get("layer_diagnostics", [])
                                    if layer_index < len(layers):
                                        sample = _float_or_none(layers[layer_index].get(modality, {}).get(source_key))
                                        if sample is not None:
                                            samples.append(sample)
                                value = statistics.median(samples) if samples else None
                            if value is not None:
                                key = (dataset, layer_index, modality, metric, window)
                                routing_by_key.setdefault(key, {}).setdefault(seed, []).append(value)
                gradient_keys = sorted({
                    key for epoch_row in selected
                    for key in epoch_row.get("gradient_diagnostics", {})
                    if key.startswith("layers.") and key.rsplit(".", 1)[-1] in GRADIENT_COMPONENTS
                })
                for group in gradient_keys:
                    if window == "best_epoch":
                        values = [_float_or_none(best_row.get("gradient_diagnostics", {}).get(group))] if best_row else []
                    else:
                        values = [
                            _float_or_none(epoch_row.get("gradient_diagnostics", {}).get(group))
                            for epoch_row in selected
                        ]
                    values = [value for value in values if value is not None]
                    if values:
                        layer_number, component = group.split(".")[1:]
                        key = (dataset, int(layer_number), component, window)
                        gradient_by_key.setdefault(key, {}).setdefault(seed, []).append(
                            values[0] if window == "best_epoch" else statistics.median(values)
                        )

            timed_eligible = [
                row for row in eligible_rows
                if _float_or_none(row.get("train_step_seconds")) is not None
            ]
            timed_after_warmup = timed_eligible[3:]
            train_times = [float(row["train_step_seconds"]) for row in timed_after_warmup]
            validation_times = [
                float(row["validation_seconds"]) for row in timed_after_warmup
                if _float_or_none(row.get("validation_seconds")) is not None
            ]
            all_train_peaks = [_float_or_none(row.get("peak_train_allocated_mb")) for row in diag_rows]
            all_reserved_peaks = [_float_or_none(row.get("peak_train_reserved_mb")) for row in diag_rows]
            all_eval_peaks = [_float_or_none(row.get("peak_eval_allocated_mb")) for row in diag_rows]
            all_eval_reserved = [_float_or_none(row.get("peak_eval_reserved_mb")) for row in diag_rows]
            median_train = statistics.median(train_times) if train_times else None
            median_val = statistics.median(validation_times) if validation_times else None
            efficiency_rows.append({
                "dataset": dataset, "model": "FlowMAG", "seed": seed,
                "model_parameters": metadata.get("model_parameters"),
                "classifier_parameters": metadata.get("classifier_parameters"),
                "best_epoch": best_epoch,
                "epochs_executed": metadata.get("epochs_executed"),
                "total_wall_clock_training_seconds": metadata.get("total_wall_clock_training_seconds"),
                "median_train_step_seconds_after_warmup": median_train,
                "median_validation_seconds_after_warmup": median_val,
                "peak_train_allocated_mb": max((v for v in all_train_peaks if v is not None), default=None),
                "peak_train_reserved_mb": max((v for v in all_reserved_peaks if v is not None), default=None),
                "peak_eval_allocated_mb": max((v for v in all_eval_peaks if v is not None), default=None),
                "peak_eval_reserved_mb": max((v for v in all_eval_reserved if v is not None), default=None),
                "nodes": metadata.get("nodes"),
                "directed_edges": metadata.get("directed_edges"),
                "edges_per_train_second": (metadata.get("directed_edges") / median_train)
                if median_train else None,
                "timed_eligible_epochs_after_warmup": len(train_times),
            })
            dataset_run_details[str(seed)] = {
                "status": "complete", "best_epoch": best_epoch,
                "epochs_executed": metadata.get("epochs_executed"),
                "val_acc": val_acc, "val_macro_f1": val_f1,
                "model_parameters": metadata.get("model_parameters"),
            }

        mean_acc, std_acc = _mean_std(val_acc_values)
        mean_f1, std_f1 = _mean_std(val_f1_values)
        if val_acc_values:
            summary_rows.append({
                "dataset": dataset, "seed": "mean_std", "row_type": "mean_std",
                "status": "complete",
                "val_acc": mean_acc, "val_acc_std": std_acc,
                "val_macro_f1": mean_f1, "val_macro_f1_std": std_f1,
                "best_epoch": None, "epochs_executed": None,
                "model_parameters": None, "classifier_parameters": None,
                "total_wall_clock_training_seconds": None,
            })
        study[dataset] = {
            "runs": dataset_run_details,
            "completed_seeds": sorted(int(seed) for seed in dataset_run_details if dataset_run_details[seed].get("status") == "complete"),
            "val_acc_mean": mean_acc, "val_acc_std": std_acc,
            "val_macro_f1_mean": mean_f1, "val_macro_f1_std": std_f1,
        }

    routing_rows: list[dict[str, Any]] = []
    for key, per_seed in sorted(routing_by_key.items()):
        dataset, layer, modality, metric, window = key
        seed_values = {seed: statistics.median(values) for seed, values in per_seed.items()}
        mean, std = _mean_std(list(seed_values.values()))
        for seed, value in sorted(seed_values.items()):
            routing_rows.append({
                "dataset": dataset, "layer": layer, "modality": modality,
                "window": window, "metric": metric, "seed": seed,
                "value": value, "mean_across_seeds": mean,
                "std_across_seeds": std, "n_seeds": len(seed_values),
            })
        routing_rows.append({
            "dataset": dataset, "layer": layer, "modality": modality,
            "window": window, "metric": metric, "seed": "mean_std",
            "value": mean, "mean_across_seeds": mean,
            "std_across_seeds": std, "n_seeds": len(seed_values),
        })

    gradient_rows: list[dict[str, Any]] = []
    for key, per_seed in sorted(gradient_by_key.items()):
        dataset, layer, component, window = key
        seed_values = {seed: statistics.median(values) for seed, values in per_seed.items()}
        mean, std = _mean_std(list(seed_values.values()))
        for seed, value in sorted(seed_values.items()):
            gradient_rows.append({
                "dataset": dataset, "layer": layer, "component": component,
                "window": window, "seed": seed, "gradient_l2_norm": value,
                "mean_across_seeds": mean, "std_across_seeds": std,
                "n_seeds": len(seed_values),
            })
        gradient_rows.append({
            "dataset": dataset, "layer": layer, "component": component,
            "window": window, "seed": "mean_std", "gradient_l2_norm": mean,
            "mean_across_seeds": mean, "std_across_seeds": std,
            "n_seeds": len(seed_values),
        })

    _write_csv(report_root / "summary.csv", summary_rows, [
        "dataset", "seed", "row_type", "status", "val_acc", "val_acc_std", "val_macro_f1",
        "val_macro_f1_std", "best_epoch", "epochs_executed", "model_parameters",
        "classifier_parameters", "total_wall_clock_training_seconds",
    ])
    _write_csv(report_root / "routing_summary.csv", routing_rows, [
        "dataset", "layer", "modality", "window", "metric", "seed", "value",
        "mean_across_seeds", "std_across_seeds", "n_seeds",
    ])
    _write_csv(report_root / "gradient_summary.csv", gradient_rows, [
        "dataset", "layer", "component", "window", "seed", "gradient_l2_norm",
        "mean_across_seeds", "std_across_seeds", "n_seeds",
    ])
    sage_payload = _read_json(output_root / "sage_reference.json") or {}
    sage_references = sage_payload.get("references", [])
    for reference in sage_references:
        if reference.get("status") == "complete":
            median_train = _float_or_none(reference.get("median_train_step_seconds"))
            efficiency_rows.append({
                "dataset": reference.get("dataset"), "model": "SAGE_reference",
                "seed": "reference", "status": "complete", "model_parameters": reference.get("parameters_including_classifier"),
                "classifier_parameters": None, "best_epoch": None,
                "epochs_executed": reference.get("epochs"),
                "total_wall_clock_training_seconds": reference.get("total_wall_clock_seconds"),
                "median_train_step_seconds_after_warmup": median_train,
                "median_validation_seconds_after_warmup": None,
                "peak_train_allocated_mb": reference.get("median_peak_allocated_mb"),
                "peak_train_reserved_mb": reference.get("median_peak_reserved_mb"),
                "peak_eval_allocated_mb": None, "peak_eval_reserved_mb": None,
                "nodes": reference.get("nodes"), "directed_edges": reference.get("directed_edges"),
                "edges_per_train_second": reference.get("directed_edges") / median_train if median_train else None,
                "timed_eligible_epochs_after_warmup": reference.get("measured_epochs"),
            })
        else:
            efficiency_rows.append({
                "dataset": reference.get("dataset"), "model": "SAGE_reference",
                "seed": "reference", "status": reference.get("status"),
                "warning": reference.get("error"),
            })
    _write_csv(report_root / "efficiency.csv", efficiency_rows, [
        "dataset", "model", "seed", "status", "warning", "model_parameters", "classifier_parameters", "best_epoch",
        "epochs_executed", "total_wall_clock_training_seconds",
        "median_train_step_seconds_after_warmup", "median_validation_seconds_after_warmup",
        "peak_train_allocated_mb", "peak_train_reserved_mb", "peak_eval_allocated_mb",
        "peak_eval_reserved_mb", "nodes", "directed_edges", "edges_per_train_second",
        "timed_eligible_epochs_after_warmup",
    ])
    _write_csv(report_root / "probe_summary.csv", probe_rows, [
        "dataset", "seed", "block", "val_acc", "val_macro_f1",
        "checkpoint_best_epoch", "probe_best_epoch", "probe_seed", "status",
    ])

    return {
        "dataset_study": study,
        "routing_rows": routing_rows,
        "gradient_rows": gradient_rows,
        "efficiency_rows": efficiency_rows,
        "probe_rows": probe_rows,
        "oom_notes": oom_notes,
        "sage_references": sage_references,
    }


def _fmt(value: Any, scale: float = 1.0, digits: int = 3) -> str:
    val = _float_or_none(value)
    return "NA" if val is None else f"{val * scale:.{digits}f}"


def write_report(report_root: Path, aggregate: dict[str, Any], output_root: Path) -> None:
    study = aggregate["dataset_study"]
    probe_rows = aggregate["probe_rows"]
    probe_index = {(row.get("dataset"), str(row.get("seed")), row.get("block")): row for row in probe_rows}
    routing_rows = aggregate["routing_rows"]
    gradient_rows = aggregate["gradient_rows"]
    efficiency_rows = aggregate["efficiency_rows"]
    routing_mean = [row for row in routing_rows if row["seed"] == "mean_std" and row["window"] == "last5_eligible_median"]
    gradient_mean = [row for row in gradient_rows if row["seed"] == "mean_std" and row["window"] == "last5_eligible_median"]
    assessments: dict[str, dict[str, str]] = {}

    lines = [
        "# FlowMAG Full-Model NC Stability & Efficiency Study V1",
        "",
        "## Study goal",
        "",
        "Audit the V0.2.1 implementation fixes and describe full-model NC validation stability, internal routing, gradient health, representation probes, runtime, and GPU memory. This is a descriptive robustness study, not an ablation or SOTA comparison.",
        "",
        "## Code fixes and protocol",
        "",
        "- Removed the Contrast base bias so a zero residual (`hbar - mu`) maps to zero.",
        "- Conditional routing statistics are computed only over nodes with a valid same-modal or cross-modal source; source availability is reported separately.",
        "- Cross-attention entropy includes real sources and Null, is computed per target and head then averaged across heads, and normalized by `log(D_i + 1)` for targets with `D_i > 0`. Entropy is interpreted as attention diffuseness.",
        "- Added graph-flow versus FFN update magnitudes and pre-clipping component gradient norms. Diagnostic-off forwards avoid summary reductions and scalar CPU copies.",
        "- RMS diagnostics use chunked reductions to cap temporary memory on larger full graphs without changing model outputs or training gradients.",
        "- Five datasets use the existing full-graph NC protocol: 300 epoch cap, AdamW, learning rate 1e-3, weight decay 1e-4, patience 30, validation every epoch, early-stop minimum epoch 30, gradient clip 1.0, no scheduler, default FlowMAG architecture and schedules.",
        "- FlowMAG dimensions are fixed across datasets: modality priors 64 + 64, graph block 128, hidden size 256, two layers, relation/router dimensions 64, adapter rank 16, two cross-attention heads, FFN multiplier 2, dropout 0.2, and attention dropout 0.0.",
        "- Routing defaults are smooth probability 0.75, same/cross admission 0.80/0.20, temperatures 2.0 to 1.0 annealed by 0.60 progress, warmup/ramp 0.05/0.10, update scales 0.1/0.1, and auxiliary weights 0.05/0.10/0.001 (prior/graph-start/router-Z).",
        "- Seeds 42, 43, and 44 are optimization/model seeds on one dataset-provided split loaded once per dataset; MAGB split files use seed 42, while ele-fashion uses its supplied fixed split. They are not independent data splits.",
        "- Best checkpoints are selected by validation accuracy. Representation probes freeze each best checkpoint, fit a fresh linear classifier on train nodes, and select probe stopping by validation accuracy only.",
        "- Probe seed is fixed at 2026; probes run for at most 100 epochs with patience 15 and Adam learning rate 0.01.",
        "- All metric analysis uses train and validation indices only. Test metrics were not evaluated, read for model selection, or reported. No architecture ablation or hyperparameter search was performed.",
        "",
        "## Training and validation results",
        "",
        "| Dataset | Status | Completed seeds | Validation accuracy (mean ± SD) | Validation macro-F1 (mean ± SD) | Best epochs |",
        "|---|---|---:|---:|---:|---|",
    ]

    for dataset in DEFAULT_DATASETS:
        detail = study.get(dataset, {})
        completed = detail.get("completed_seeds", [])
        run_map = detail.get("runs", {})
        if len(completed) != 3:
            status = "failed / incomplete"
            reason = aggregate["oom_notes"].get(dataset, "one or more formal runs or checkpoints are missing")
            unattempted = [seed for seed in SEEDS if run_map.get(str(seed), {}).get("status") == "not_run_after_oom"]
            if unattempted:
                if dataset == "ele-fashion":
                    reason = (
                        "Seed 42 reached epoch 2, then the backward pass OOMed requesting 192 MiB with 193.69 MiB free; "
                        "the initial 96 MiB diagnostic scratch OOM was reduced by chunked RMS, but training still exceeded "
                        "the GPU limit. Seeds 43 and 44 were not attempted."
                    )
                else:
                    reason += f" Seeds {', '.join(map(str, unattempted))} were not attempted after the OOM."
        else:
            # Flag strong graph bypass only when the joint evidence agrees.
            graph_rows = [row for row in routing_mean if row["dataset"] == dataset and row["metric"] == "graph_update_ratio"]
            ffn_rows = [row for row in routing_mean if row["dataset"] == dataset and row["metric"] == "ffn_update_ratio"]
            probe_full = [row for row in probe_rows if row.get("dataset") == dataset and row.get("block") == "prior_plus_graph" and row.get("status") == "complete"]
            probe_prior = [row for row in probe_rows if row.get("dataset") == dataset and row.get("block") == "prior_only" and row.get("status") == "complete"]
            bypass_signal = False
            if graph_rows and ffn_rows and probe_full and probe_prior:
                graph_values = [float(row["value"]) for row in graph_rows if row.get("value") is not None]
                ffn_values = [float(row["value"]) for row in ffn_rows if row.get("value") is not None]
                paired_diff = [
                    float(full["val_acc"]) - float(prior["val_acc"])
                    for full in probe_full for prior in probe_prior
                    if str(full.get("seed")) == str(prior.get("seed"))
                    and full.get("val_acc") is not None and prior.get("val_acc") is not None
                ]
                bypass_signal = bool(graph_values and ffn_values and paired_diff)
                bypass_signal = bypass_signal and statistics.median(graph_values) < 1e-3
                bypass_signal = bypass_signal and statistics.median(ffn_values) > 0.1
                bypass_signal = bypass_signal and statistics.median(paired_diff) <= 0.01
            status = "suspicious" if bypass_signal else "healthy"
            reason = "graph-bypass indicators jointly present" if bypass_signal else "three runs and required summaries completed; no joint graph-bypass trigger"
        assessments[dataset] = {"status": status, "reason": reason}
        best_epochs = ", ".join(str(run_map[str(seed)].get("best_epoch", "NA")) for seed in completed if str(seed) in run_map) or "NA"
        val_acc = _fmt(detail.get("val_acc_mean"), 100, 2)
        val_acc_std = _fmt(detail.get("val_acc_std"), 100, 2)
        val_f1 = _fmt(detail.get("val_macro_f1_mean"), 100, 2)
        val_f1_std = _fmt(detail.get("val_macro_f1_std"), 100, 2)
        lines.append(
            f"| {dataset} | {status} | {len(completed)}/3 | {val_acc} ± {val_acc_std}% | {val_f1} ± {val_f1_std}% | {best_epochs} |"
        )
        if status.startswith("failed"):
            lines.append(f"\n{dataset}: {reason.rstrip('.')}.")

    lines += [
        "",
        "## Runtime and memory",
        "",
        "FlowMAG medians use checkpoint-eligible epochs after excluding the first three timed eligible epochs. The optional SAGE reference uses ten full-graph train epochs, excludes the first three, and reports medians over the final seven. GPU memory is measured through PyTorch peak allocated/reserved counters. `edges_per_train_second` is descriptive system throughput, not evidence of algorithmic complexity.",
        "",
        "| Dataset | Model | Seed | Train median (s) | Validation median (s) | Peak train allocated (MB) | Peak train reserved (MB) | Nodes | Directed edges | Edges / train-second |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in efficiency_rows:
        lines.append(
            f"| {row['dataset']} | {row.get('model', 'FlowMAG')} | {row['seed']} | {_fmt(row.get('median_train_step_seconds_after_warmup'))} | {_fmt(row.get('median_validation_seconds_after_warmup'))} | {_fmt(row.get('peak_train_allocated_mb'), digits=1)} | {_fmt(row.get('peak_train_reserved_mb'), digits=1)} | {row.get('nodes', 'NA')} | {row.get('directed_edges', 'NA')} | {_fmt(row.get('edges_per_train_second'), digits=1)} |"
        )
    if not efficiency_rows:
        lines.append("| No complete profiled runs | | | | | | | | |")

    lines += ["", "## Internal routing and update behavior", ""]
    lines.append("Last-five-eligible-epoch medians are aggregated per seed, then summarized across the three optimization/model seeds with population SD. Availability and conditional routing remain distinct in the CSV.")
    lines.append("")
    lines.append("| Dataset | Metric | Mean across layer × modality summaries | Across-seed SD |")
    lines.append("|---|---|---:|---:|")
    for dataset in DEFAULT_DATASETS:
        for metric in ("same_valid_rate", "cross_valid_rate", "pi_smooth_given_same_valid", "pi_contrast_given_same_valid", "same_admission_given_valid", "cross_admission_given_valid", "null_mass_given_cross_valid", "normalized_cross_attention_entropy", "graph_update_ratio", "ffn_update_ratio", "smooth_contrast_cosine"):
            vals = [row for row in routing_mean if row["dataset"] == dataset and row["metric"] == metric and row.get("value") is not None]
            if vals:
                seed_rows = [row for row in routing_rows if row["dataset"] == dataset and row["metric"] == metric and row["window"] == "last5_eligible_median" and row["seed"] != "mean_std"]
                seed_values: dict[int, list[float]] = {}
                for row in seed_rows:
                    seed_values.setdefault(int(row["seed"]), []).append(float(row["value"]))
                per_seed_average = [statistics.mean(values) for values in seed_values.values() if values]
                mean, sd = _mean_std(per_seed_average)
                lines.append(f"| {dataset} | {metric} | {_fmt(mean)} | {_fmt(sd)} |")

    lines += ["", "## Gradient health", "", "Gradient norms below are raw pre-clipping L2 norms. The study summarizes Contrast, cross-attention, controller, same-relation, and FFN groups per layer.", ""]
    lines.append("| Dataset | Component | Mean layer × seed norm (last-five median) |")
    lines.append("|---|---|---:|")
    for dataset in DEFAULT_DATASETS:
        for component in sorted(GRADIENT_COMPONENTS):
            vals = [row["gradient_l2_norm"] for row in gradient_mean if row["dataset"] == dataset and row["component"] == component and row.get("gradient_l2_norm") is not None]
            if vals:
                lines.append(f"| {dataset} | {component} | {_fmt(statistics.mean(float(v) for v in vals), digits=6)} |")

    lines += ["", "## Frozen-checkpoint representation probes", "", "The main NC classifier metrics above and linear-probe metrics below come from separate training procedures. Prior-only is `concat(r_text, r_image)`, graph-only is the graph block, and Prior+Graph is their concatenation. Probe rows use train-fit / validation-evaluate only; probe validation scores do not select FlowMAG checkpoints.", ""]
    lines.append("| Dataset | Seed | Prior-only val acc / F1 | Graph-only val acc / F1 | Prior+Graph val acc / F1 | Majority val acc / F1 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for dataset in DEFAULT_DATASETS:
        majority = probe_index.get((dataset, "all", "majority_baseline"), {})
        for seed in SEEDS:
            items = [probe_index.get((dataset, str(seed), block), {}) for block in ("prior_only", "graph_only", "prior_plus_graph")]
            def metric_pair(item):
                if item.get("val_acc") is None:
                    return "NA"
                return f"{_fmt(item['val_acc'], 100, 2)} / {_fmt(item.get('val_macro_f1'), 100, 2)}"
            majority_pair = metric_pair(majority)
            lines.append(f"| {dataset} | {seed} | {metric_pair(items[0])} | {metric_pair(items[1])} | {metric_pair(items[2])} | {majority_pair} |")

    lines += ["", "## Cross-dataset consistency", ""]
    lines.append("Consistency is described from per-seed values and mean/SD summaries. Three seeds are treated as a robustness sample only; no seed-level p-values or t-tests are used.")
    lines.append("")
    lines.append("| Dataset | Status | Evidence note |")
    lines.append("|---|---|---|")
    for dataset in DEFAULT_DATASETS:
        item = assessments[dataset]
        lines.append(f"| {dataset} | {item['status']} | {item['reason']} |")

    lines += [
        "",
        "## Failures, threats to validity, and scope",
        "",
        "- Formal test metrics were not evaluated, read for checkpoint selection, or reported. Only train and validation labels were copied into the runner's working label tensor.",
        "- No architecture ablation, B0–B4 study, hyperparameter search, dataset-specific tuning, LP campaign, or books-nc run was performed.",
        "- The three optimization/model seeds share a single loaded split, so this does not measure split-to-split uncertainty.",
        "- The validation split is used for both FlowMAG checkpoint selection and probe early stopping; probe outcomes are descriptive and do not feed back into FlowMAG selection.",
        "- Routing and update summaries are layer/modality averages; they do not establish causal component necessity.",
        "- GPU timing depends on the shared device and runtime environment. PyTorch allocator reserved memory may include cached blocks.",
        "- SAGE rows are fixed-budget system references only; no SAGE validation or test accuracy is calculated or compared.",
        "- Low flow usage alone is not interpreted as evidence to remove a component. Health labels are descriptive operational/optimization flags only.",
        "",
    ]
    for dataset, note in aggregate["oom_notes"].items():
        lines.append(f"- OOM evidence ({dataset}): {note}")
    lines += ["", "## Factual assessment", ""]
    for dataset in DEFAULT_DATASETS:
        item = assessments[dataset]
        lines.append(f"- **{dataset}: {item['status']}** — {item['reason'].rstrip('.')}.")
    lines += [
        "",
        "No claim is made that FlowMAG is better than state of the art, that the proposed factorization has been causally validated, or that any flow is necessary.",
        "",
        "## Artifacts",
        "",
        "- `summary.csv` — per-seed and across-seed validation results.",
        "- `routing_summary.csv` — per-seed and mean/SD conditional routing and update summaries at best epoch and over the final five eligible epochs.",
        "- `gradient_summary.csv` — pre-clipping gradient norms at the same windows.",
        "- `efficiency.csv` — runtime, PyTorch memory peaks, parameters, graph size, and descriptive throughput.",
        "- `probe_summary.csv` — frozen checkpoint linear probes and majority baselines.",
        "- `run_manifest.json` — exact protocol, seeds, completion state, and warnings.",
        "",
        f"Raw logs, epoch diagnostics, and checkpoints are under `{output_root}` and are ignored by Git.",
    ]
    (report_root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "study": "FlowMAG Full-Model NC Stability & Efficiency Study V1",
        "branch": "research/flowmag-nc-stability-v1",
        "base_sha": "77ab1f66260d318d25ea26643a19d7c02de4e819",
        "datasets": DEFAULT_DATASETS,
        "status_by_dataset": assessments,
        "runs_by_dataset": study,
        "optimization_model_seeds": SEEDS,
        "probe_seed": PROBE_SEED,
        "data_split": "dataset-provided NC task split loaded once and shared by optimization/model seeds 42, 43, 44; MAGB split paths use seed 42, ele-fashion uses its supplied fixed split",
        "train_validation_only": True,
        "test_metrics_evaluated": False,
        "test_metrics_reported": False,
        "test_labels_used_for_analysis_or_selection": False,
        "architecture_ablation": False,
        "hyperparameter_search": False,
        "formal_protocol": {
            "task": "nc", "model": "flowmag", "training_mode": "full_graph",
            "num_runs": 3, "split_seed": 42,
            "epochs": 300, "lr": 0.001, "weight_decay": 0.0001,
            "optimizer": "adamw", "patience": 30, "eval_every": 1,
            "early_stop_min_epoch": 30, "early_stop_min_delta": 0.0001,
            "grad_clip": 1.0, "scheduler": None,
            "hidden_dim": 256, "num_layers": 2, "relation_dim": 64,
            "router_dim": 64, "adapter_rank": 16, "cross_heads": 2,
            "prior_dim_per_modality": 64, "graph_block_dim": 128,
            "ffn_multiplier": 2, "dropout": 0.2, "attention_dropout": 0.0,
            "detach_candidate_descriptor": True, "null_cardinality_correction": True,
            "same_update_scale": 0.1,
            "cross_update_scale": 0.1, "warmup_ratio": 0.05,
            "controller_ramp_ratio": 0.10, "lambda_prior": 0.05,
            "lambda_graph_start": 0.10, "graph_aux_decay_end_ratio": 0.40,
            "lambda_router_z": 0.001, "collect_diagnostics": True,
            "router_temp_start": 2.0, "router_temp_end": 1.0,
            "router_temp_anneal_end_ratio": 0.60,
            "init_smooth_prob": 0.75, "init_same_admission": 0.80,
            "init_cross_admission": 0.20, "eps": 1e-8,
            "aux_weight": 1.0, "inference_mode": "full",
            "collect_model_diagnostics": True, "profile_runtime": True,
            "evaluate_test": False,
        },
        "probe_protocol": {
            "fit_indices": "train_idx", "evaluation_indices": "val_idx",
            "epochs_max": PROBE_EPOCHS, "patience": PROBE_PATIENCE,
            "optimizer": "Adam", "lr": PROBE_LR,
            "checkpoint_selection": "FlowMAG best validation checkpoint; probe early stopping uses validation accuracy only",
        },
        "sage_efficiency_reference": aggregate.get("sage_references", []),
        "warnings": list(aggregate["oom_notes"].values()),
        "spec_deviations": [],
    }
    (report_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs/flowmag_nc_stability_v1")
    parser.add_argument("--report-root", type=Path, default=REPO_ROOT / "reports/flowmag_nc_stability_v1")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    sys.path.insert(0, str(REPO_ROOT))
    args.output_root = args.output_root.expanduser().resolve()
    args.report_root = args.report_root.expanduser().resolve()
    args.report_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for representation probes but is unavailable")

    probe_rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        print(f"Analyzing frozen FlowMAG checkpoints: {dataset}", flush=True)
        try:
            probe_rows.extend(run_probes(dataset, args.output_root, args.report_root, device))
        except Exception as exc:
            print(f"Probe analysis failed for {dataset}: {type(exc).__name__}: {exc}", flush=True)
            probe_rows.append({
                "dataset": dataset, "seed": "all", "block": "all",
                "val_acc": None, "val_macro_f1": None,
                "checkpoint_best_epoch": None, "probe_best_epoch": None,
                "probe_seed": PROBE_SEED,
                "status": f"analysis_error:{type(exc).__name__}:{exc}",
            })
    aggregate = aggregate_study(args.datasets, args.output_root, args.report_root, probe_rows)
    write_report(args.report_root, aggregate, args.output_root)
    print(f"Wrote study reports to {args.report_root}", flush=True)


if __name__ == "__main__":
    main()
