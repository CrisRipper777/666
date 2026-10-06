#!/usr/bin/env python
"""Run the optional fixed-budget SAGE NC system benchmark without test metrics."""
from __future__ import annotations

import json
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["Movies", "ele-fashion"]
EPOCHS = 10
WARMUP_EPOCHS = 3


def _compose(dataset: str):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        return compose(
            config_name="config",
            overrides=[
                f"dataset={dataset}", "task=nc", "model=sage", "seed=42", "num_runs=1",
                "device=cuda:0", "task.epochs=10", "task.evaluate_test=false",
                "task.eval_every=10", "model.hidden_dim=256",
            ],
        )


def _run_one(dataset: str, device: torch.device) -> dict:
    from src.data import load_mag_data
    from src.models import build_model
    from src.tasks.common import build_optimizer
    from src.utils.seeds import set_seed

    cfg = _compose(dataset)
    data = load_mag_data(cfg, "nc", 42)
    set_seed(42)
    info = {
        "input_dim": int(data.input_dim), "num_nodes": int(data.num_nodes),
        "num_classes": int(data.num_classes),
        "text_dim": int(data.x_t.shape[1]) if data.x_t is not None else 0,
        "visual_dim": int(data.x_i.shape[1]) if data.x_i is not None else 0,
    }
    model = build_model(cfg, info).to(device)
    classifier = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    optimizer = build_optimizer(
        list(model.parameters()) + list(classifier.parameters()), cfg, model=model
    )
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    train_idx = data.train_idx.to(device)
    train_y = data.y[data.train_idx.cpu()].to(device)
    timings: list[float] = []
    allocated_peaks: list[float] = []
    reserved_peaks: list[float] = []
    start_wall = time.perf_counter()
    try:
        for epoch in range(1, EPOCHS + 1):
            model.train()
            classifier.train()
            torch.cuda.reset_peak_memory_stats(device)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            z, _, _, _, _ = model(x, edge_index)
            loss = nn.functional.cross_entropy(classifier(z[train_idx]), train_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(classifier.parameters()),
                max_norm=1.0,
                error_if_nonfinite=True,
            )
            optimizer.step()
            torch.cuda.synchronize(device)
            timings.append(time.perf_counter() - start)
            allocated_peaks.append(float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)))
            reserved_peaks.append(float(torch.cuda.max_memory_reserved(device) / (1024 ** 2)))
            del z, loss
        measured_times = timings[WARMUP_EPOCHS:]
        measured_allocated = allocated_peaks[WARMUP_EPOCHS:]
        measured_reserved = reserved_peaks[WARMUP_EPOCHS:]
        return {
            "dataset": dataset, "model": "SAGE", "status": "complete",
            "seed": 42, "epochs": EPOCHS, "warmup_epochs": WARMUP_EPOCHS,
            "measured_epochs": len(measured_times),
            "hidden_dim": 256, "num_layers": int(cfg.model.num_layers),
            "parameters_including_classifier": sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in classifier.parameters()),
            "nodes": int(data.num_nodes), "directed_edges": int(data.num_edges),
            "median_train_step_seconds": statistics.median(measured_times),
            "median_peak_allocated_mb": statistics.median(measured_allocated),
            "median_peak_reserved_mb": statistics.median(measured_reserved),
            "total_wall_clock_seconds": time.perf_counter() - start_wall,
            "validation_evaluated": False, "test_evaluated": False,
        }
    except torch.OutOfMemoryError as exc:
        torch.cuda.synchronize(device)
        return {
            "dataset": dataset, "model": "SAGE", "status": "oom",
            "seed": 42, "epochs_requested": EPOCHS, "epochs_completed": len(timings),
            "hidden_dim": 256, "num_layers": int(cfg.model.num_layers),
            "nodes": int(data.num_nodes), "directed_edges": int(data.num_edges),
            "torch_memory_allocated_mb_at_failure": float(torch.cuda.memory_allocated(device) / (1024 ** 2)),
            "torch_memory_reserved_mb_at_failure": float(torch.cuda.memory_reserved(device) / (1024 ** 2)),
            "error": str(exc), "validation_evaluated": False, "test_evaluated": False,
        }
    finally:
        del model, classifier, optimizer, x, edge_index, train_idx, train_y
        torch.cuda.empty_cache()


def main() -> None:
    sys.path.insert(0, str(REPO_ROOT))
    if not torch.cuda.is_available():
        raise RuntimeError("The SAGE efficiency reference requires the authorized CUDA environment")
    device = torch.device("cuda:0")
    output_path = REPO_ROOT / "outputs/flowmag_nc_stability_v1/sage_reference.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for dataset in DATASETS:
        print(f"Running fixed 10-epoch SAGE system reference: {dataset}", flush=True)
        try:
            result = _run_one(dataset, device)
        except Exception as exc:
            result = {
                "dataset": dataset, "model": "SAGE", "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "validation_evaluated": False, "test_evaluated": False,
            }
        results.append(result)
        output_path.write_text(json.dumps({"references": results}, indent=2) + "\n", encoding="utf-8")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"Wrote SAGE reference metrics to {output_path}", flush=True)


if __name__ == "__main__":
    main()
