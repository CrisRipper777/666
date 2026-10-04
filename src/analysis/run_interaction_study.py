from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import platform
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.analysis.interaction_study.alignment import alignment_record, fit_ridge_aligner
from src.analysis.interaction_study.diagnostic_host import HostTrainingConfig, train_diagnostic_host
from src.analysis.interaction_study.perturbation import operator_response, sample_receiver_sources, cross_modal_utility
from src.analysis.interaction_study.probes import compute_compatibility, p1_modality_heterogeneity, p3_structure_response
from src.analysis.interaction_study.reporting import finalize_reports, json_safe, write_json
from src.analysis.interaction_study.splits import make_study_split, mask_test_labels
from src.analysis.interaction_study.statistics import finite_summary
from src.analysis.problem_deep_dive import structural_edge_descriptors
from src.data.graph_utils import ensure_edge_index, preprocess_edge_index
from src.data.loaders import load_mag_data, resolve_path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "analysis" / "interaction_study.yaml"


def _compose_dataset(name: str):
    with initialize_config_dir(version_base=None, config_dir=str(PROJECT_ROOT / "configs")):
        cfg = compose(config_name="config", overrides=[f"dataset={name}", "task=nc"])
    OmegaConf.resolve(cfg)
    return cfg


def _load_nc_split(path: Path) -> dict[str, torch.Tensor] | None:
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def audit_dataset(name: str, hidden_dim: int, gpu_info: dict[str, Any]) -> dict[str, Any]:
    cfg = _compose_dataset(name)
    ds = cfg.dataset
    source = str(ds.source).lower()
    if source == "magb":
        import dgl
        graph_path = resolve_path(str(ds.graph_path))
        graphs, _ = dgl.load_graphs(str(graph_path))
        graph = graphs[0]
        src, dst = graph.edges()
        raw_edges = torch.stack([src.long(), dst.long()], 0)
        n = int(graph.num_nodes())
        text_shape = np.load(resolve_path(str(ds.text_feat_path)), mmap_mode="r", allow_pickle=False).shape
        image_shape = np.load(resolve_path(str(ds.image_feat_path)), mmap_mode="r", allow_pickle=False).shape
        text_dim, image_dim = int(text_shape[1]), int(image_shape[1])
        split_path = resolve_path(str(ds.nc_split_path))
        split = _load_nc_split(split_path)
        file_sizes = {
            "graph_bytes": graph_path.stat().st_size,
            "text_feature_bytes": resolve_path(str(ds.text_feat_path)).stat().st_size,
            "image_feature_bytes": resolve_path(str(ds.image_feat_path)).stat().st_size,
        }
    elif source == "mmgraph":
        feature_path = resolve_path(str(ds.joint_feat_path))
        x = torch.load(feature_path, map_location="cpu", weights_only=False, mmap=True)
        n = int(x.size(0))
        text_dim, image_dim = int(ds.text_dim), int(ds.visual_dim)
        edge_path = resolve_path(str(ds.edge_path))
        edge_payload = torch.load(edge_path, map_location="cpu", weights_only=False, mmap=True)
        raw_edges = ensure_edge_index(edge_payload)
        split_path = resolve_path(str(ds.node_split_path))
        split = _load_nc_split(split_path)
        file_sizes = {
            "joint_feature_bytes": feature_path.stat().st_size,
            "edge_bytes": edge_path.stat().st_size,
            "label_bytes": resolve_path(str(ds.label_path)).stat().st_size,
        }
    else:
        raise ValueError(f"unsupported NC dataset source {source!r}")

    edges = preprocess_edge_index(raw_edges, n, make_undirected=bool(ds.get("make_undirected", True)), with_self_loops=False)
    degree = torch.bincount(edges[0].cpu(), minlength=n).float()
    quantiles = torch.quantile(degree, torch.tensor([0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]))
    counts = None
    if split is not None:
        counts = {
            "train": int(torch.as_tensor(split["train_idx"]).numel()),
            "validation": int(torch.as_tensor(split["val_idx"]).numel()),
            "test": int(torch.as_tensor(split["test_idx"]).numel()),
        }
    raw_feature_bytes = n * (text_dim + image_dim) * 4
    fit_count = int((counts or {}).get("train", round(0.6 * n)) * 0.8)
    # Conservative estimate: full feature matrices on device plus full-fit autograd activations.
    fit_working = fit_count * (2 * (text_dim + image_dim) + 12 * hidden_dim + int(ds.num_classes)) * 4
    estimated_peak = raw_feature_bytes + fit_working
    return {
        "dataset": name,
        "source": source,
        "num_nodes": n,
        "num_edges_directed": int(edges.size(1)),
        "text_dim": text_dim,
        "image_dim": image_dim,
        "num_classes_configured": int(ds.num_classes),
        "split_counts": counts,
        "split_path": str(split_path),
        "split_available": split is not None,
        "degree": {
            "min": float(quantiles[0]), "q25": float(quantiles[1]), "median": float(quantiles[2]),
            "q75": float(quantiles[3]), "q90": float(quantiles[4]), "q95": float(quantiles[5]),
            "q99": float(quantiles[6]), "max": float(quantiles[7]),
            "mean": float(degree.mean()), "zero_degree_nodes": int((degree == 0).sum()),
        },
        "raw_feature_bytes": int(raw_feature_bytes),
        "raw_feature_gb": float(raw_feature_bytes / 1e9),
        "estimated_diagnostic_gpu_peak_bytes": int(estimated_peak),
        "estimated_diagnostic_gpu_peak_gb": float(estimated_peak / 1e9),
        "gpu": gpu_info,
        "files": file_sizes,
        "books_full_study": "audit-only by design" if name == "books-nc" else None,
        "label_access": "split counts only; no label values loaded for audit",
    }


def run_audit(config: dict[str, Any], raw_root: Path) -> dict[str, Any]:
    device_info = {"cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        device_info["device_name"] = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info(0)
        device_info["cuda_free_bytes_at_audit"] = int(free)
        device_info["cuda_total_bytes"] = int(total)
    datasets, failures = {}, []
    for name in config["audit_datasets"]:
        try:
            datasets[name] = audit_dataset(name, int(config["diagnostic_host"]["hidden_dim"]), device_info)
            print(f"AUDIT {name}: nodes={datasets[name]['num_nodes']} directed_edges={datasets[name]['num_edges_directed']} dims={datasets[name]['text_dim']}+{datasets[name]['image_dim']} gpu_est={datasets[name]['estimated_diagnostic_gpu_peak_gb']:.2f}GB")
        except Exception as exc:
            failures.append({"dataset": name, "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(limit=8)})
            print(f"AUDIT FAILED {name}: {type(exc).__name__}: {exc}")
    audit = {
        "dataset_audit": "node count, directed edge count, feature dimensions, split sizes, degree distribution, raw feature bytes, estimated graph-free host GPU memory",
        "gpu": device_info,
        "datasets": datasets,
        "failures": failures,
    }
    write_json(raw_root / "audit.json", audit)
    return audit


def _formal_data(name: str, split_seed: int):
    cfg = _compose_dataset(name)
    if str(cfg.dataset.source).lower() == "magb":
        cfg.dataset.auto_generate_splits = False
    data = load_mag_data(cfg, "nc", int(cfg.seed))
    if data.x_t is None or data.x_i is None or data.y is None:
        raise ValueError(f"{name} requires text, image and NC labels")
    split = make_study_split(data.y, data.train_idx, data.val_idx, data.test_idx, seed=split_seed, fit_fraction=0.8)
    labels = mask_test_labels(data.y, split)
    # Drop the original unmasked label tensor before response calculations.
    data.y = None
    return cfg, data, labels, split


def _serialize_p1(record: dict[str, Any]) -> dict[str, Any]:
    return record


def _dataset_raw_p1(name: str, data, labels: torch.Tensor, split, cfg: dict[str, Any], dataset_root: Path):
    summary_path = dataset_root / "p1_raw.json"
    tensor_path = dataset_root / "p1_raw.pt"
    if summary_path.exists() and tensor_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        tensors = torch.load(tensor_path, map_location="cpu", weights_only=False)
        return summary, tensors
    max_sources = int(cfg["compatibility"]["max_sources"])
    ref_count = int(cfg["compatibility"]["nonedges_per_source"])
    comp_t = compute_compatibility(data.x_t, data.edge_index, int(cfg["compatibility"]["seed"]), max_sources, ref_count)
    comp_i = compute_compatibility(data.x_i, data.edge_index, int(cfg["compatibility"]["seed"]), max_sources, ref_count)
    record, table, raw = p1_modality_heterogeneity(
        comp_t[0], comp_t[1], comp_i[0], comp_i[1], split.original_train,
        split.original_validation, labels, data.num_nodes,
        bootstrap_reps=int(cfg["interaction"]["p1_bootstrap_reps"]), seed=int(cfg["study_seed"]),
    )
    record["raw_compatibility_stats"] = {"text": comp_t[2], "image": comp_i[2]}
    raw["table"] = table
    raw["thresholds_text"] = record["train_threshold_text"]
    raw["thresholds_image"] = record["train_threshold_image"]
    dataset_root.mkdir(parents=True, exist_ok=True)
    torch.save(raw, tensor_path)
    write_json(summary_path, record)
    return record, raw


def _device_for_run(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def _seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _run_seed(
    name: str,
    host_seed: int,
    data,
    labels: torch.Tensor,
    split,
    cfg: dict[str, Any],
    raw_p1_summary: dict[str, Any],
    raw_p1: dict[str, Any],
    descriptors: dict[str, torch.Tensor],
    device: torch.device,
    output_dir: Path,
    smoke: bool,
) -> dict[str, Any]:
    _seed_everything(host_seed + int(cfg["study_seed"]))
    host_values = dict(cfg["diagnostic_host"])
    interaction_values = dict(cfg["interaction"])
    compat_values = dict(cfg["compatibility"])
    if smoke:
        smoke_cfg = cfg["smoke"]
        host_values.update({
            "epochs": int(smoke_cfg["host_epochs"]), "patience": int(smoke_cfg["host_patience"]),
            "train_node_cap": int(smoke_cfg["train_node_cap"]), "hidden_dim": int(smoke_cfg["hidden_dim"]),
            "batch_size": int(smoke_cfg["batch_size"]),
        })
        interaction_values.update({
            "max_validation_directed_edges": int(smoke_cfg["interaction_cap"]),
            "max_fit_directed_edges": int(smoke_cfg["interaction_cap"]),
            "batch_size": int(smoke_cfg["batch_size"]),
            "p1_bootstrap_reps": int(smoke_cfg["bootstrap_reps"]),
            "p2_bootstrap_reps": int(smoke_cfg["bootstrap_reps"]),
            "p3_bootstrap_reps": int(smoke_cfg["bootstrap_reps"]),
            "operator_bootstrap_reps": int(smoke_cfg["bootstrap_reps"]),
            "matched_subsampling_repeats": int(smoke_cfg["matched_subsampling_repeats"]),
        })
        compat_values.update({
            "max_sources": int(smoke_cfg["compatibility_max_sources"]),
            "nonedges_per_source": int(smoke_cfg["compatibility_nonedges_per_source"]),
        })
    host_config = HostTrainingConfig(**host_values)
    host_result = train_diagnostic_host(
        data.x_t.float().cpu(), data.x_i.float().cpu(), labels, split,
        int(data.num_classes), int(host_seed + int(cfg["study_seed"])), device, host_config,
    )
    model = host_result.pop("model")
    h_t, h_i = host_result["h_t"], host_result["h_i"]
    hidden_dim = int(host_config.hidden_dim)
    if h_t.size(1) != hidden_dim or h_i.size(1) != hidden_dim:
        raise AssertionError("diagnostic host hidden dimension mismatch")
    ridge = float(cfg["alignment"]["ridge_regularization"])
    align_t2i = fit_ridge_aligner(h_t, h_i, split.diagnostic_fit, ridge, "text", "image")
    align_i2t = fit_ridge_aligner(h_i, h_t, split.diagnostic_fit, ridge, "image", "text")
    alignment, aligned_t2i, aligned_i2t = alignment_record(
        align_t2i, align_i2t, h_t, h_i, split,
        float(cfg["alignment"]["min_validation_cosine"]),
    )

    projected_comp_t = compute_compatibility(
        h_t, data.edge_index, int(cfg["compatibility"]["seed"]) + host_seed,
        int(compat_values["max_sources"]), int(compat_values["nonedges_per_source"]),
    )
    projected_comp_i = compute_compatibility(
        h_i, data.edge_index, int(cfg["compatibility"]["seed"]) + host_seed,
        int(compat_values["max_sources"]), int(compat_values["nonedges_per_source"]),
    )
    p1_projected, p1_projected_table, p1_projected_raw = p1_modality_heterogeneity(
        projected_comp_t[0], projected_comp_t[1], projected_comp_i[0], projected_comp_i[1],
        split.original_train, split.original_validation, labels, data.num_nodes,
        bootstrap_reps=int(interaction_values["p1_bootstrap_reps"]), seed=int(cfg["study_seed"]) + host_seed,
    )
    p1_projected["projected_compatibility_stats"] = {"text": projected_comp_t[2], "image": projected_comp_i[2]}
    p1_projected["table"] = p1_projected_table

    receivers = torch.cat([split.diagnostic_calib, split.original_validation]).unique(sorted=True)
    sources = sample_receiver_sources(
        data.edge_index, receivers, data.num_nodes, int(interaction_values["neighbor_cap"]),
        seed=int(cfg["study_seed"]) + 4703,
    )
    p2_rows, p2_raw = cross_modal_utility(
        model, h_t, h_i, aligned_t2i, aligned_i2t, labels, split, sources,
        list(map(float, interaction_values["p2_alphas"])), device,
        batch_size=int(interaction_values["batch_size"]),
        bootstrap_reps=int(interaction_values["p2_bootstrap_reps"]),
        seed=int(cfg["study_seed"]) + host_seed,
    )
    p3_values = {
        "text": (raw_p1["pairs"], raw_p1["percentile_text"]),
        "image": (raw_p1["pairs"], raw_p1["percentile_image"]),
    }
    actual_p3_cap = int(interaction_values["max_validation_directed_edges"])
    actual_p3_batch = int(interaction_values["batch_size"])
    p3_retries = []
    while True:
        try:
            p3_rows, regression_rows, p3_raw = p3_structure_response(
                model, data.edge_index, h_t, h_i, aligned_t2i, aligned_i2t, labels, split, p3_values, device,
                semantic_bins_thresholds={"text": raw_p1["thresholds_text"], "image": raw_p1["thresholds_image"]},
                max_interactions=actual_p3_cap,
                alphas=list(map(float, interaction_values["p3_alphas"])),
                bootstrap_reps=int(interaction_values["p3_bootstrap_reps"]),
                matched_repeats=int(interaction_values["matched_subsampling_repeats"]),
                seed=int(cfg["study_seed"]) + host_seed,
                batch_size=actual_p3_batch,
                descriptors=descriptors,
            )
            break
        except torch.cuda.OutOfMemoryError:
            if actual_p3_cap <= 1000 and actual_p3_batch <= 256:
                raise
            torch.cuda.empty_cache()
            gc.collect()
            actual_p3_cap = max(1000, actual_p3_cap // 2)
            actual_p3_batch = max(256, actual_p3_batch // 2)
            p3_retries.append({"reason": "CUDA OOM", "new_interaction_cap": actual_p3_cap, "new_batch_size": actual_p3_batch})

    op_rows, op_raw = operator_response(
        model, h_t, h_i, aligned_t2i, aligned_i2t, labels, split, sources,
        float(interaction_values["operator_alpha"]), device,
        batch_size=int(interaction_values["batch_size"]),
        bootstrap_reps=int(interaction_values["operator_bootstrap_reps"]),
        seed=int(cfg["study_seed"]) + host_seed,
    )
    for row in p3_rows:
        if row.get("row_type") == "sampling_metadata":
            row["actual_interaction_cap_per_split"] = actual_p3_cap
            row["actual_perturbation_batch_size"] = actual_p3_batch
            row["oom_retries"] = p3_retries

    raw_tensor_path = output_dir / f"{name}_seed_{host_seed}_raw.pt"
    raw_blob = {
        "hidden_text": h_t,
        "hidden_image": h_i,
        "aligned_text_to_image": aligned_t2i,
        "aligned_image_to_text": aligned_i2t,
        "p1_raw": raw_p1,
        "p1_projected": p1_projected_raw,
        "p2": p2_raw,
        "p3": p3_raw,
        "operator": op_raw,
        "host_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(raw_blob, raw_tensor_path)
    finite_counts = {
        "hidden_text_nonfinite": int((~torch.isfinite(h_t)).sum()),
        "hidden_image_nonfinite": int((~torch.isfinite(h_i)).sum()),
        "p2_nonfinite": int(sum((~torch.isfinite(v)).sum() for v in p2_raw.values() if isinstance(v, torch.Tensor) and v.is_floating_point())),
        "p3_nonfinite": int(sum((~torch.isfinite(v)).sum() for v in p3_raw.values() if isinstance(v, torch.Tensor) and v.is_floating_point())),
    }
    if any(value > 0 for key, value in finite_counts.items() if key.endswith("nonfinite")):
        raise FloatingPointError(f"nonfinite study tensors: {finite_counts}")
    result = {
        "status": "success",
        "dataset": name,
        "host_seed": int(host_seed),
        "protocol": {
            "study_seed": int(cfg["study_seed"]),
            **split.metadata(),
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "python_version": platform.python_version(),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "resolved_dataset_config": OmegaConf.to_container(_compose_dataset(name), resolve=True),
            "test_label_access": False,
            "test_metric_access": False,
            "transductive_topology": True,
        },
        "host": {key: value for key, value in host_result.items() if key not in {"h_t", "h_i"}},
        "alignment": alignment,
        "p1": {"raw": raw_p1_summary, "projected": p1_projected},
        "p2": p2_rows,
        "p3": p3_rows,
        "p3_regression": regression_rows,
        "operator": op_rows,
        "actual_p3_cap_per_receiver_split": actual_p3_cap,
        "p3_oom_retries": p3_retries,
        "finite_value_audit": finite_counts,
        "raw_interaction_tensors": str(raw_tensor_path),
    }
    return result


def _failure(name: str, seed: int, exc: Exception) -> dict[str, Any]:
    return {
        "status": "failed", "dataset": name, "host_seed": int(seed),
        "error_type": type(exc).__name__, "error": str(exc),
        "traceback": traceback.format_exc(limit=12),
    }


def run_study(args: argparse.Namespace) -> int:
    config = OmegaConf.to_container(OmegaConf.load(CONFIG_PATH), resolve=True)
    raw_root = (PROJECT_ROOT / config["outputs"]["raw_root"]).resolve()
    report_root = (PROJECT_ROOT / config["outputs"]["report_root"]).resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    if args.finalize_only:
        audit_path = raw_root / "audit.json"
        if not audit_path.exists():
            raise FileNotFoundError(f"dataset audit is missing: {audit_path}")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        summary = finalize_reports(raw_root, report_root, config, audit)
        print(f"REPORT {report_root / 'REPORT.md'} complete={summary['complete']} runs={summary['successful_seed_runs']}/{summary['expected_dataset_seed_runs']}")
        return 0 if summary["complete"] else 1
    audit = run_audit(config, raw_root)
    if args.audit_only:
        print(f"Audit saved: {raw_root / 'audit.json'}")
        return 0 if not audit["failures"] else 1
    if audit["failures"]:
        print("WARNING: one or more NC dataset audits failed; formal runs will continue where data are loadable.")
    if args.smoke:
        datasets = ["Movies"]
        seeds = [0]
        target_root = raw_root / "smoke"
    else:
        datasets = args.datasets or list(config["primary_datasets"])
        seeds = args.host_seeds or list(config["formal_host_seeds"])
        target_root = raw_root
    unknown = [name for name in datasets if name not in config["primary_datasets"]]
    if unknown:
        raise ValueError(f"full interaction study is restricted to configured primary datasets: {unknown}")
    if args.smoke:
        print("Running Movies smoke: one host seed, capped diagnostic fit/interactions; this is not a formal result.")
    device = _device_for_run(args.device or "cuda:0")
    for name in datasets:
        dataset_root = target_root / name if args.smoke else raw_root / name
        try:
            cfg_obj, data, labels, split = _formal_data(name, int(config["study_seed"]))
            if not data.x_t.is_floating_point() or not data.x_i.is_floating_point():
                raise TypeError("interaction study expects floating point modality features")
            raw_p1_summary, raw_p1 = _dataset_raw_p1(name, data, labels, split, config, dataset_root)
            descriptors = structural_edge_descriptors(data.edge_index, data.num_nodes)
            for host_seed in seeds:
                result_path = dataset_root / f"seed_{host_seed}.json"
                if args.resume and not args.smoke and result_path.exists():
                    old = json.loads(result_path.read_text(encoding="utf-8"))
                    if old.get("status") == "success":
                        print(f"SKIP {name} seed={host_seed}: successful checkpoint record exists")
                        continue
                try:
                    result = _run_seed(
                        name, int(host_seed), data, labels, split, config,
                        raw_p1_summary, raw_p1, descriptors, device,
                        dataset_root, bool(args.smoke),
                    )
                    write_json(result_path, result)
                    print(f"SUCCESS {name} seed={host_seed}: host_val_acc={result['host']['metrics']['report']['accuracy']:.4f} low_host_quality={result['host']['low_host_quality']} p3_cap={result['actual_p3_cap_per_receiver_split']}")
                except Exception as exc:
                    if isinstance(exc, torch.cuda.OutOfMemoryError):
                        torch.cuda.empty_cache()
                    write_json(result_path, _failure(name, int(host_seed), exc))
                    print(f"FAILED {name} seed={host_seed}: {type(exc).__name__}: {exc}")
                    if args.fail_fast:
                        raise
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            del data, labels, split, descriptors
            gc.collect()
        except Exception as exc:
            print(f"DATASET FAILED {name}: {type(exc).__name__}: {exc}")
            if args.fail_fast:
                raise

    if not args.smoke:
        expected = [raw_root / dataset / f"seed_{seed}.json" for dataset in config["primary_datasets"] for seed in config["formal_host_seeds"]]
        if all(path.exists() for path in expected):
            result = finalize_reports(raw_root, report_root, config, audit)
            print(f"REPORT {report_root / 'REPORT.md'} complete={result['complete']} runs={result['successful_seed_runs']}/{result['expected_dataset_seed_runs']}")
        else:
            missing = sum(not path.exists() for path in expected)
            print(f"REPORT not finalized: {missing} formal dataset-seed records still missing.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the isolated MAG interaction-study V1 analysis pipeline.")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--host-seeds", nargs="*", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true", help="Run the small Movies end-to-end smoke pipeline.")
    parser.add_argument("--audit-only", action="store_true", help="Audit all six NC datasets without fitting a host.")
    parser.add_argument("--resume", action="store_true", help="Skip successful formal seed records.")
    parser.add_argument("--finalize-only", action="store_true", help="Build compact reports from existing formal run records.")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main() -> None:
    sys.path.insert(0, str(PROJECT_ROOT))
    args = build_parser().parse_args()
    raise SystemExit(run_study(args))


if __name__ == "__main__":
    main()
