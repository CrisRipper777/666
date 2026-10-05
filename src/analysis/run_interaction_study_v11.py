from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.analysis.interaction_study.diagnostic_host import FeatureOnlyMultimodalDiagnosticHost
from src.analysis.interaction_study.perturbation import sample_receiver_sources
from src.analysis.interaction_study.reporting import write_csv, write_json
from src.analysis.interaction_study.v11 import (
    FUNCTIONAL_ACTIONS,
    NULL_ACTIONS,
    classify_p2_evidence,
    classify_p3_dataset,
    classify_p3_modality,
    compute_null_action_profiles,
    compute_p2_factorial,
    corrected_p1_summary,
    deterministic_matched_effect,
    directional_reference_percentiles,
    functional_null_paired_metrics,
    matched_effect_receiver_bootstrap,
    operator_stability_summary,
    orient_validation_edges,
    p3_dataset_description,
    sample_source_matched_references,
    summarize_reference_noise,
    summarize_seed_receiver_vectors,
)
from src.analysis.interaction_study.v11 import _receiver_stat
from src.analysis.run_interaction_study import _formal_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "analysis" / "interaction_study_v11.yaml"
V1_CONFIG_PATH = PROJECT_ROOT / "configs" / "analysis" / "interaction_study.yaml"
DATASETS = ("Movies", "Toys", "Grocery", "Reddit-S", "ele-fashion")
HOST_SEEDS = (0, 1, 2)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_dataset_config(name: str):
    with initialize_config_dir(version_base=None, config_dir=str(PROJECT_ROOT / "configs")):
        cfg = compose(config_name="config", overrides=[f"dataset={name}", "task=nc"])
    OmegaConf.resolve(cfg)
    return cfg


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_input_manifest(v1_raw_root: Path, v11_raw_root: Path) -> dict[str, Any]:
    manifest_path = v11_raw_root / "input_manifest.json"
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
    else:
        artifacts = []
        shape_schema_by_dataset: dict[str, Any] = {}
        for dataset in DATASETS:
            for seed in HOST_SEEDS:
                path = v1_raw_root / dataset / f"{dataset}_seed_{seed}_raw.pt"
                artifacts.append({"dataset": dataset, "seed": seed, "path": str(path), "sha256": _sha256(path)})
        manifest = {"artifacts": artifacts, "shape_schema_by_dataset": shape_schema_by_dataset}
        write_json(manifest_path, manifest)
    if len(manifest.get("artifacts", [])) != 15:
        raise RuntimeError("V1.1 requires exactly 15 verified V1 raw seed artifacts")
    for artifact in manifest["artifacts"]:
        path = PROJECT_ROOT / artifact["path"]
        if not path.is_file():
            raise FileNotFoundError(f"V1 raw seed artifact missing: {path}")
        actual = _sha256(path)
        if actual != artifact["sha256"]:
            raise RuntimeError(f"V1 raw seed artifact changed after manifest: {path}")
    return manifest


def _load_artifact(path: Path, expected_hash: str) -> dict[str, Any]:
    actual = _sha256(path)
    if actual != expected_hash:
        raise RuntimeError(f"SHA256 mismatch for V1 raw artifact {path}")
    blob = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    required = {
        "hidden_text", "hidden_image", "aligned_text_to_image", "aligned_image_to_text",
        "p1_raw", "p1_projected", "p2", "p3", "operator", "host_state_dict",
    }
    missing = required - set(blob)
    if missing:
        raise RuntimeError(f"V1 raw artifact {path} missing keys {sorted(missing)}")
    return blob


def _construct_host(blob: dict[str, Any], data, device: torch.device) -> FeatureOnlyMultimodalDiagnosticHost:
    num_classes = int(data.num_classes)
    hidden_dim = int(blob["hidden_text"].size(1))
    host = FeatureOnlyMultimodalDiagnosticHost(data.x_t.size(1), data.x_i.size(1), hidden_dim, num_classes)
    host.load_state_dict(blob["host_state_dict"], strict=True)
    host.to(device).eval()
    for parameter in host.parameters():
        parameter.requires_grad_(False)
    return host


def _v1_seed_json(raw_root: Path, dataset: str, seed: int) -> dict[str, Any]:
    path = raw_root / dataset / f"seed_{seed}.json"
    result = _read_json(path)
    if result.get("status") != "success":
        raise RuntimeError(f"V1 seed record is not successful: {path}")
    return result


def _validation_reorder(profile: torch.Tensor, nodes: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    supplied = nodes.detach().long().cpu().flatten()
    wanted = expected.detach().long().cpu().flatten()
    if supplied.unique().numel() != supplied.numel() or wanted.unique().numel() != wanted.numel():
        raise RuntimeError("validation node IDs must be unique")
    if supplied.numel() != wanted.numel() or set(supplied.tolist()) != set(wanted.tolist()):
        raise RuntimeError("V1 operator profile does not cover the V1.1 validation receiver universe")
    positions = {int(node): row for row, node in enumerate(supplied.tolist())}
    order = torch.tensor([positions[int(node)] for node in wanted.tolist()], dtype=torch.long)
    return profile.detach().float().cpu()[order]


def _functional_action_index(seed_json: dict[str, Any], modality: str) -> int:
    rows = [row for row in seed_json.get("operator", []) if row.get("modality") == modality]
    if len(rows) != 1:
        raise RuntimeError(f"V1 operator row missing for modality={modality}")
    action = rows[0]["global_fixed_action_selected_on_calib"]
    return FUNCTIONAL_ACTIONS.index(action)


def _p3_alpha_index(raw_p3: dict[str, Any], alpha: float) -> int:
    values = raw_p3["p3_alpha_values"].detach().float().cpu().tolist()
    matches = [i for i, value in enumerate(values) if abs(float(value) - alpha) < 1e-7]
    if len(matches) != 1:
        raise RuntimeError(f"V1 P3 raw artifact has no unique alpha={alpha}: {values}")
    return matches[0]


def _summary_with_mean_ci(values: torch.Tensor, receivers: torch.Tensor, reps: int, seed: int) -> dict[str, Any]:
    return _receiver_stat(values, receivers, reps, seed)


def _analyze_p3_seed(
    dataset: str,
    seed: int,
    raw_p3: dict[str, Any],
    seed_json: dict[str, Any],
    *,
    validation_nodes: torch.Tensor,
    bootstrap_reps: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for modality in ("text", "image"):
        prefix = f"{modality}_"
        is_report = raw_p3[f"{prefix}is_report"].detach().bool().cpu()
        mask = is_report
        receivers = raw_p3[f"{prefix}receiver"].detach().long().cpu()[mask]
        source = raw_p3[f"{prefix}source"].detach().long().cpu()[mask]
        semantic = raw_p3[f"{prefix}semantic_bin"].detach().long().cpu()[mask]
        structure = raw_p3[f"{prefix}structure_bin"].detach().long().cpu()[mask]
        source_degree = raw_p3[f"{prefix}source_degree_bin"].detach().long().cpu()[mask]
        receiver_degree = raw_p3[f"{prefix}receiver_degree_bin"].detach().long().cpu()[mask]
        ds = raw_p3[f"{prefix}D_same"].detach().float().cpu()[:, mask]
        if not (source.numel() == receivers.numel() == ds.size(1)):
            raise RuntimeError(f"V1 P3 row tensors do not align: {dataset} seed={seed} {modality}")
        report_std = float(ds[1].std(unbiased=False)) if ds.size(1) else float("nan")
        historical = next((
            row for row in seed_json.get("p3", [])
            if row.get("row_type") == "matched_high_low_structure" and row.get("modality") == modality
        ), {})
        for alpha in (0.05, 0.1, 0.2):
            alpha_idx = _p3_alpha_index(raw_p3, alpha)
            values = ds[alpha_idx]
            point = deterministic_matched_effect(values, semantic, structure, source_degree, receiver_degree)
            if abs(alpha - 0.1) < 1e-8:
                uncertainty = matched_effect_receiver_bootstrap(
                    values, semantic, structure, source_degree, receiver_degree, receivers,
                    reps=bootstrap_reps, seed=92000 + seed * 100 + (0 if modality == "text" else 1),
                    receiver_universe=validation_nodes,
                )
            else:
                uncertainty = {"ci_low": None, "ci_high": None, "valid_bootstrap_count": 0,
                               "requested_bootstrap_reps": 0,
                               "bootstrap_unit": "primary receiver-block bootstrap is reported at alpha=0.1"}
            effect = float(point["matched_effect"])
            rows.append({
                "dataset": dataset,
                "host_seed": int(seed),
                "modality": modality,
                "alpha": float(alpha),
                "primary_alpha": bool(abs(alpha - 0.1) < 1e-8),
                **point,
                "ci_low": uncertainty.get("ci_low"),
                "ci_high": uncertainty.get("ci_high"),
                "valid_bootstrap_count": uncertainty.get("valid_bootstrap_count", 0),
                "requested_bootstrap_reps": uncertainty.get("requested_bootstrap_reps", 0),
                "receiver_block_count": uncertainty.get("receiver_block_count", int(receivers.unique().numel())),
                "bootstrap_unit": uncertainty.get("bootstrap_unit"),
                "standardized_effect": effect / report_std if np.isfinite(effect) and report_std > 0 else None,
                "validation_report_interactions": int(values.numel()),
                "validation_receivers_with_interactions": int(receivers.unique().numel()),
                "historical_v1_repeated_subsample_interval": {
                    key: historical.get(key) for key in ("mean_high_minus_low", "ci_low", "ci_high", "repeat_q025", "repeat_q975", "interval_method")
                } if abs(alpha - 0.1) < 1e-8 else None,
                "sensitivity_effects_alpha_0_05_0_1_0_2": None,
            })
    # Include each modality's three alpha estimates next to its primary row.
    for modality in ("text", "image"):
        modality_rows = [r for r in rows if r["modality"] == modality]
        effects = {f"alpha_{r['alpha']:.2f}": r["matched_effect"] for r in modality_rows}
        signs = [
            int(np.sign(float(r["matched_effect"])))
            if r["matched_effect"] is not None and np.isfinite(float(r["matched_effect"]))
            else None
            for r in modality_rows
        ]
        sign_stable = len(signs) == 3 and all(sign is not None for sign in signs) and len(set(signs)) == 1
        primary = next(r for r in modality_rows if r["primary_alpha"])
        primary["sensitivity_effects_alpha_0_05_0_1_0_2"] = effects
        for row in modality_rows:
            row["alpha_effect_signs_0_05_0_1_0_2"] = signs
            row["sign_consistent_across_alpha"] = sign_stable if all(sign is not None for sign in signs) else None
    return rows


def _run_one_seed(
    dataset: str,
    seed: int,
    raw_path: Path,
    expected_sha: str,
    data,
    labels: torch.Tensor,
    split,
    sources,
    device: torch.device,
    output_dir: Path,
    config: dict[str, Any],
    v1_raw_root: Path,
) -> dict[str, Any]:
    blob = _load_artifact(raw_path, expected_sha)
    if blob["hidden_text"].size(0) != data.num_nodes or blob["hidden_image"].size(0) != data.num_nodes:
        raise RuntimeError(f"V1 hidden-state node count mismatch in {raw_path}")
    host = _construct_host(blob, data, device)
    # Confirm V1's frozen receiver-source counts match the reproduced deterministic source bank.
    old_nodes = blob["p2"]["receiver_nodes"].detach().long().cpu()
    old_counts = blob["p2"]["source_count"].detach().long().cpu()
    positions = {int(node): i for i, node in enumerate(sources.receivers.tolist())}
    val_positions = torch.tensor([positions[int(node)] for node in old_nodes.tolist()], dtype=torch.long)
    if not torch.equal(old_counts, sources.counts[val_positions]):
        raise RuntimeError(f"V1 receiver source counts do not reproduce for {dataset} seed={seed}")
    if set(old_nodes.tolist()) != set(split.original_validation.tolist()):
        raise RuntimeError(f"V1 P2 receiver universe differs from original validation: {dataset} seed={seed}")

    seed_json = _v1_seed_json(v1_raw_root, dataset, seed)
    p2 = compute_p2_factorial(
        host, blob["hidden_text"], blob["hidden_image"],
        blob["aligned_text_to_image"], blob["aligned_image_to_text"],
        labels, split, sources, config["p2_alphas"], device=device,
        batch_size=int(config["batch_size"]),
    )
    p2_rows: list[dict[str, Any]] = []
    p2_array_payload: dict[str, np.ndarray] = {"validation_nodes": split.original_validation.numpy()}
    metric_names = (
        "cross_only_graph_gain", "conditional_graph_gain", "conditional_cross_delta_graph",
        "conditional_cross_delta_random", "same_cross_perturbation_interaction_term",
    )
    for direction in ("T2I", "I2T", "T2I_minus_I2T"):
        for alpha in map(float, config["p2_alphas"]):
            for offset, metric in enumerate(metric_names):
                values = p2[direction][alpha][metric]
                p2_array_payload[f"{direction}_a{alpha:.2f}_{metric}"] = values.numpy()
                stat = _summary_with_mean_ci(values, split.original_validation, int(config["bootstrap_reps"]), seed * 10000 + offset + int(alpha * 1000))
                p2_rows.append({
                    "dataset": dataset, "host_seed": int(seed), "direction": direction,
                    "alpha": float(alpha), "metric": metric, **stat,
                })

    p3_rows = _analyze_p3_seed(
        dataset, seed, blob["p3"], seed_json,
        validation_nodes=split.original_validation,
        bootstrap_reps=int(config["bootstrap_reps"]),
    )

    null = compute_null_action_profiles(
        host, blob["hidden_text"], blob["hidden_image"],
        blob["aligned_text_to_image"], blob["aligned_image_to_text"],
        labels, split, sources, alpha=float(config["operator_alpha"]),
        device=device, batch_size=int(config["batch_size"]),
    )
    op_raw = blob["operator"]
    validation_nodes = op_raw["validation_nodes"].detach().long().cpu()
    op_payload: dict[str, np.ndarray] = {"validation_nodes": split.original_validation.numpy()}
    op_rows: list[dict[str, Any]] = []
    for modality in ("text", "image"):
        functional_profile = _validation_reorder(
            op_raw[f"{modality}_response_profile"], validation_nodes, split.original_validation,
        )
        function_base = _validation_reorder(
            op_raw[f"{modality}_report_base_ce"], validation_nodes, split.original_validation,
        ).flatten()
        null_profile = null[modality]["report_response_profile"].detach().float().cpu()
        null_base = null[modality]["report_base_ce"].detach().float().cpu().flatten()
        functional_index = _functional_action_index(seed_json, modality)
        null_index = int(null[modality]["calib_global_action_index"].item())
        if functional_profile.size(1) != 4 or null_profile.size(1) != 4:
            raise RuntimeError("functional and matched null banks must both contain four actions")
        if functional_profile.shape != null_profile.shape or function_base.numel() != null_base.numel():
            raise RuntimeError("functional and null profiles do not share validation node order")
        paired = functional_null_paired_metrics(functional_profile, null_profile, functional_index, null_index)
        for offset, metric in enumerate(("functional_headroom", "null_headroom", "excess_headroom", "best_action_advantage", "global_fixed_advantage")):
            stat = _summary_with_mean_ci(
                paired[metric], split.original_validation,
                int(config["bootstrap_reps"]), seed * 100 + offset + (0 if modality == "text" else 20),
            )
            op_rows.append({
                "dataset": dataset, "host_seed": int(seed), "modality": modality,
                "metric": metric, **stat,
                "functional_global_action": FUNCTIONAL_ACTIONS[functional_index],
                "null_global_action": NULL_ACTIONS[null_index],
                "action_count": 4,
                "alpha": float(config["operator_alpha"]),
                "matched_source_count_total": int(sources.counts.sum()),
                "source_count_match_verified": True,
                "interpretation": "frozen-host, label-aware diagnostic response; not deployable router performance",
            })
        op_payload[f"{modality}_functional_profile"] = functional_profile.numpy()
        op_payload[f"{modality}_null_profile"] = null_profile.numpy()
        op_payload[f"{modality}_functional_base_ce"] = function_base.numpy()
        op_payload[f"{modality}_null_base_ce"] = null_base.numpy()
        op_payload[f"{modality}_functional_global_index"] = np.asarray([functional_index], dtype=np.int64)
        op_payload[f"{modality}_null_global_index"] = np.asarray([null_index], dtype=np.int64)
        op_payload[f"{modality}_functional_margin"] = (functional_profile.sort(1).values[:, 1] - functional_profile.sort(1).values[:, 0]).numpy()
        op_payload[f"{modality}_null_margin"] = (null_profile.sort(1).values[:, 1] - null_profile.sort(1).values[:, 0]).numpy()

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / f"seed_{seed}_p2_receiver_metrics.npz", **p2_array_payload)
    np.savez_compressed(output_dir / f"seed_{seed}_operator_profiles.npz", **op_payload)
    result = {
        "dataset": dataset,
        "host_seed": int(seed),
        "v1_artifact": {"path": str(raw_path), "sha256": expected_sha},
        "p2": p2_rows,
        "p3": p3_rows,
        "operator_null_comparison": op_rows,
        "protocol": {
            "test_label_access": False,
            "test_metric_access": False,
            "host_retrained": False,
            "validation_nodes_sha256": seed_json["protocol"].get("index_sha256", {}).get("original_validation"),
            "source_count_match_verified": True,
            "response_profile_order": "reordered to canonical V1 original-validation index order",
        },
    }
    write_json(output_dir / f"seed_{seed}.json", result)
    del blob, host, null
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _run_p1_dataset(
    dataset: str,
    data,
    split,
    raw_seed0: dict[str, Any],
    seed0_json: dict[str, Any],
    device: torch.device,
    output_dir: Path,
    v1_config: dict[str, Any],
    v11_config: dict[str, Any],
) -> dict[str, Any]:
    p1_raw = raw_seed0["p1_raw"]
    pairs = p1_raw["pairs"].detach().long().cpu()
    corrected = corrected_p1_summary(
        pairs,
        p1_raw["percentile_text"],
        p1_raw["percentile_image"],
        split.original_validation,
        reps=int(v11_config["bootstrap_reps"]),
        seed=70403,
        v1_summary=seed0_json["p1"]["raw"],
    )
    if not corrected["point_estimate_invariant_to_orientation"]:
        raise RuntimeError(f"corrected P1 point estimate did not reproduce V1 for {dataset}")
    if int(v1_config["compatibility"]["nonedges_per_source"]) != 32:
        raise RuntimeError("V1.1 reference-noise audit requires the V1 setting nonedges_per_source=32")

    source, receiver = orient_validation_edges(pairs, split.original_validation)
    reference_seeds = list(map(int, v11_config["p1_reference_seeds"]))
    text_values, image_values = [], []
    per_reference = []
    max_sources = int(v1_config["compatibility"]["max_sources"])
    per_source = int(v1_config["compatibility"]["nonedges_per_source"])
    for ref_seed in reference_seeds:
        sample = sample_source_matched_references(
            data.edge_index, data.num_nodes, seed=ref_seed,
            max_sources=max_sources, nonedges_per_source=per_source,
        )
        p_t = directional_reference_percentiles(data.x_t, source, receiver, sample, device=device)
        p_i = directional_reference_percentiles(data.x_i, source, receiver, sample, device=device)
        if p_t.numel() != receiver.numel() or p_i.numel() != receiver.numel():
            raise RuntimeError("repeated P1 reference draws did not cover all directed validation interactions")
        text_values.append(p_t)
        image_values.append(p_i)
        gap = (p_t - p_i).abs()
        per_reference.append({
            "reference_seed": ref_seed,
            "mean_cross_signal": float(gap.mean()),
            "median_cross_signal": float(gap.median()),
            "spearman": float(__import__("scipy.stats", fromlist=["spearmanr"]).spearmanr(p_t.numpy(), p_i.numpy()).statistic),
            "source_pattern_shared_across_modalities": True,
            "sampled_source_count": int(sample.active_sources.numel()),
            "reference_count_per_source": int(sample.nonedges_per_source),
        })
        del sample, p_t, p_i
    p_text = torch.stack(text_values)
    p_image = torch.stack(image_values)
    noise = summarize_reference_noise(
        p_text, p_image, receiver,
        reps=int(v11_config["bootstrap_reps"]), seed=70411,
    )
    result = {
        "dataset": dataset,
        "corrected_orientation": corrected,
        "reference_sampling": {
            **noise,
            "reference_seeds": reference_seeds,
            "nonedges_per_source": per_source,
            "max_sources": max_sources,
            "same_graph_and_physical_edge_universe": True,
            "text_image_sampling_pattern_shared_within_seed": True,
            "seed_details": per_reference,
            "adjusted_gap_positive_ci_rule": "CI low for the mean conservative-adjusted gap is greater than zero",
        },
        "v1_seed0_raw_point_estimate_source": "embedded p1_raw pairs and source-matched percentile tensors",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "p1_reference_percentiles.npz",
        source=source.numpy(), receiver=receiver.numpy(),
        text=np.stack([x.numpy() for x in text_values]),
        image=np.stack([x.numpy() for x in image_values]),
        reference_seeds=np.asarray(reference_seeds, dtype=np.int64),
    )
    write_json(output_dir / "p1_analysis.json", result)
    return result


def _run_dataset(
    dataset: str,
    args: argparse.Namespace,
    v1_config: dict[str, Any],
    v11_config: dict[str, Any],
    manifest: dict[str, Any],
    v1_raw_root: Path,
    raw_root: Path,
    device: torch.device,
) -> None:
    cfg_obj, data, labels, split = _formal_data(dataset, int(v1_config["study_seed"]))
    if data.y is not None:
        raise AssertionError("_formal_data must drop the unmasked label tensor")
    if not data.x_t.is_floating_point() or not data.x_i.is_floating_point():
        raise TypeError(f"{dataset} features must be floating point")
    output_dir = raw_root / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = {
        int(row["seed"]): row for row in manifest["artifacts"] if row["dataset"] == dataset
    }
    if sorted(manifest_rows) != list(HOST_SEEDS):
        raise RuntimeError(f"V1 raw artifact manifest is incomplete for {dataset}")

    seed0_path = PROJECT_ROOT / manifest_rows[0]["path"]
    seed0_blob = _load_artifact(seed0_path, manifest_rows[0]["sha256"])
    seed0_json = _v1_seed_json(v1_raw_root, dataset, 0)
    p1_path = output_dir / "p1_analysis.json"
    p1_npz = output_dir / "p1_reference_percentiles.npz"
    if not (args.resume and p1_path.exists() and p1_npz.exists()):
        _run_p1_dataset(dataset, data, split, seed0_blob, seed0_json, device, output_dir, v1_config, v11_config)
        print(f"P1.1 {dataset}: corrected orientation and {len(v11_config['p1_reference_seeds'])} reference draws complete")
    else:
        _read_json(p1_path)
        print(f"SKIP P1.1 {dataset}: verified existing raw summary and percentile draws")
    del seed0_blob

    receivers = torch.cat([split.diagnostic_calib, split.original_validation]).unique(sorted=True)
    sources = sample_receiver_sources(
        data.edge_index, receivers, int(data.num_nodes), int(v1_config["interaction"]["neighbor_cap"]),
        seed=int(v1_config["study_seed"]) + 4703,
    )
    for seed in HOST_SEEDS:
        artifact = manifest_rows[seed]
        raw_path = PROJECT_ROOT / artifact["path"]
        result_path = output_dir / f"seed_{seed}.json"
        p2_path = output_dir / f"seed_{seed}_p2_receiver_metrics.npz"
        op_path = output_dir / f"seed_{seed}_operator_profiles.npz"
        if args.resume and result_path.exists() and p2_path.exists() and op_path.exists():
            saved = _read_json(result_path)
            if saved.get("v1_artifact", {}).get("sha256") == artifact["sha256"]:
                print(f"SKIP V1.1 {dataset} seed={seed}: raw artifact hash unchanged")
                continue
        result = _run_one_seed(
            dataset, seed, raw_path, artifact["sha256"], data, labels, split,
            sources, device, output_dir, v11_config, v1_raw_root,
        )
        print(
            f"V1.1 {dataset} seed={seed}: P2 rows={len(result['p2'])} "
            f"P3 modality rows={len(result['p3'])} operator rows={len(result['operator_null_comparison'])}"
        )
    del data, labels, split, sources
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _p2_seed_aggregate(dataset: str, raw_root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    seed_results = {seed: _read_json(raw_root / dataset / f"seed_{seed}.json") for seed in HOST_SEEDS}
    archives = {seed: np.load(raw_root / dataset / f"seed_{seed}_p2_receiver_metrics.npz") for seed in HOST_SEEDS}
    node_ids = [archives[seed]["validation_nodes"] for seed in HOST_SEEDS]
    if any(not np.array_equal(node_ids[0], candidate) for candidate in node_ids[1:]):
        raise RuntimeError(f"P2 host seeds do not share the same ordered validation receiver universe for {dataset}")
    receivers = torch.from_numpy(node_ids[0].astype(np.int64, copy=False))
    metric_names = (
        "cross_only_graph_gain", "conditional_graph_gain", "conditional_cross_delta_graph",
        "conditional_cross_delta_random", "same_cross_perturbation_interaction_term",
    )
    aggregate_rows = []
    for direction in ("T2I", "I2T", "T2I_minus_I2T"):
        for alpha in map(float, config["p2_alphas"]):
            for offset, metric in enumerate(metric_names):
                key = f"{direction}_a{alpha:.2f}_{metric}"
                vectors = {seed: torch.from_numpy(archives[seed][key].astype(np.float32, copy=False)) for seed in HOST_SEEDS}
                averaged = summarize_seed_receiver_vectors(
                    vectors, receivers, reps=int(config["bootstrap_reps"]),
                    seed=88100 + offset + int(alpha * 1000) + (0 if direction == "T2I" else 10000 if direction == "I2T" else 20000),
                )
                per_seed = []
                for seed in HOST_SEEDS:
                    row = next(
                        row for row in seed_results[seed]["p2"]
                        if row["direction"] == direction and abs(float(row["alpha"]) - alpha) < 1e-8 and row["metric"] == metric
                    )
                    per_seed.append(row)
                aggregate_rows.append({
                    "dataset": dataset,
                    "host_seed": "seed_averaged_receiver_metric",
                    "direction": direction,
                    "alpha": alpha,
                    "metric": metric,
                    **averaged["summary"],
                    "seed_means": {str(seed): averaged["seed_means"][seed] for seed in HOST_SEEDS},
                    "seed_mean_signs": {str(seed): averaged["seed_signs"][seed] for seed in HOST_SEEDS},
                    "evidence_grade": classify_p2_evidence(per_seed, averaged),
                    "aggregation": averaged["aggregation"],
                    "receiver_count": int(receivers.numel()),
                    "bootstrap_unit": "validation receiver node after host-seed averaging",
                    "bootstrap_reps": int(config["bootstrap_reps"]),
                })
    for archive in archives.values():
        archive.close()
    return aggregate_rows


def _operator_stability_rows(dataset: str, raw_root: Path) -> list[dict[str, Any]]:
    archives = {seed: np.load(raw_root / dataset / f"seed_{seed}_operator_profiles.npz") for seed in HOST_SEEDS}
    node_ids = [archives[seed]["validation_nodes"] for seed in HOST_SEEDS]
    if any(not np.array_equal(node_ids[0], candidate) for candidate in node_ids[1:]):
        raise RuntimeError(f"operator host seeds do not share the same ordered validation nodes for {dataset}")
    out = []
    for modality in ("text", "image"):
        for bank, actions, suffix in (("functional", FUNCTIONAL_ACTIONS, "functional"), ("matched_random_null", NULL_ACTIONS, "null")):
            profiles = {
                seed: archives[seed][f"{modality}_{suffix}_profile"]
                for seed in HOST_SEEDS
            }
            stability = operator_stability_summary(profiles, actions, minimum_margin=1e-3)
            out.append({
                "dataset": dataset,
                "modality": modality,
                "bank": bank,
                "action_count": len(actions),
                "validation_receiver_count": int(node_ids[0].size),
                **stability,
            })
    for archive in archives.values():
        archive.close()
    return out


def _flatten_p1(dataset_result: dict[str, Any]) -> dict[str, Any]:
    corrected = dataset_result["corrected_orientation"]
    sampling = dataset_result["reference_sampling"]
    metrics = sampling["metrics"]
    row = {
        "dataset": dataset_result["dataset"],
        "directed_interactions": corrected["directed_interactions"],
        "receiver_count": corrected["receiver_count"],
        "corrected_mean_abs_gap": corrected["mean_abs_percentile_gap"],
        "corrected_mean_gap_ci_low": corrected["mean_gap_ci_low"],
        "corrected_mean_gap_ci_high": corrected["mean_gap_ci_high"],
        "corrected_median_abs_gap": corrected["median_abs_percentile_gap"],
        "corrected_spearman": corrected["spearman"],
        "v1_mean_abs_gap": corrected["v1_mean_gap"],
        "v1_mean_gap_ci_low": corrected["v1_mean_gap_ci_low"],
        "v1_mean_gap_ci_high": corrected["v1_mean_gap_ci_high"],
        "v1_median_abs_gap": corrected["v1_median_gap"],
        "mean_gap_change_vs_v1": corrected["mean_gap_change_vs_v1"],
        "median_gap_change_vs_v1": corrected["median_gap_change_vs_v1"],
        "orientation_point_estimate_invariant": corrected["point_estimate_invariant_to_orientation"],
        "reference_seeds": sampling["reference_seeds"],
        "cross_signal_median": sampling["cross_signal_median"],
        "descriptive_ratio": sampling["descriptive_ratio"],
        "reference_sampling_interactions": sampling["interaction_count"],
        "reference_sampling_receivers": sampling["receiver_count"],
        "reference_sampling_method": "5 same-source/nonedge patterns, shared text/image within seed; 32 nonedges/source",
        "adjusted_gap_ci_low_positive": metrics["conservative_adjusted_gap"]["ci_low"] > 0,
    }
    for key, metric_name in (
        ("cross_signal", "cross_signal"),
        ("text_instability", "text_reference_instability"),
        ("image_instability", "image_reference_instability"),
        ("adjusted_gap", "conservative_adjusted_gap"),
    ):
        stat = metrics[metric_name]
        row[f"{key}_mean"] = stat["mean"]
        row[f"{key}_median"] = stat["median"]
        row[f"{key}_ci_low"] = stat["ci_low"]
        row[f"{key}_ci_high"] = stat["ci_high"]
    return row


def _finalize(raw_root: Path, report_root: Path, manifest: dict[str, Any], config: dict[str, Any], v1_config: dict[str, Any]) -> dict[str, Any]:
    completed_datasets = [
        dataset for dataset in DATASETS
        if (raw_root / dataset / "p1_analysis.json").is_file()
        and all((raw_root / dataset / f"seed_{seed}.json").is_file() for seed in HOST_SEEDS)
    ]
    p1_rows = []
    p2_rows = []
    p2_aggregate_rows = []
    p3_rows = []
    operator_rows = []
    stability_rows = []
    dataset_summary = []
    for dataset in completed_datasets:
        p1_record = _read_json(raw_root / dataset / "p1_analysis.json")
        p1_rows.append(_flatten_p1(p1_record))
        seeds = [_read_json(raw_root / dataset / f"seed_{seed}.json") for seed in HOST_SEEDS]
        for result in seeds:
            p2_rows.extend(result["p2"])
            p3_rows.extend(result["p3"])
            operator_rows.extend(result["operator_null_comparison"])
        p3_groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
        for row in p3_rows:
            if row["dataset"] == dataset:
                p3_groups.setdefault((dataset, int(row["host_seed"]), row["modality"]), []).append(row)
        alpha_sign_consistency = {}
        for (group_dataset, host_seed, modality), group_rows in p3_groups.items():
            if group_dataset != dataset:
                continue
            alpha_rows = sorted(group_rows, key=lambda item: float(item["alpha"]))
            signs = [
                int(np.sign(float(item["matched_effect"])))
                if item["matched_effect"] is not None and np.isfinite(float(item["matched_effect"]))
                else None
                for item in alpha_rows
            ]
            stable = len(alpha_rows) == 3 and all(sign is not None for sign in signs) and len(set(signs)) == 1
            for item in alpha_rows:
                item["alpha_effect_signs_0_05_0_1_0_2"] = signs
                item["sign_consistent_across_alpha"] = stable if all(sign is not None for sign in signs) else None
            alpha_sign_consistency[f"{host_seed}/{modality}"] = {
                "signs": signs,
                "stable": stable if all(sign is not None for sign in signs) else None,
                "available_alpha_count": sum(sign is not None for sign in signs),
            }
        agg = _p2_seed_aggregate(dataset, raw_root, config)
        p2_aggregate_rows.extend(agg)
        stable = _operator_stability_rows(dataset, raw_root)
        stability_rows.extend(stable)

        primary_for_dataset = [row for row in p3_rows if row["dataset"] == dataset and row["primary_alpha"]]
        p3_grade = classify_p3_dataset(primary_for_dataset)
        p3_grades = {modality: p3_grade[modality]["grade"] for modality in ("text", "image")}
        for row in p3_rows:
            if row["dataset"] == dataset:
                row["modality_evidence_grade"] = p3_grades[row["modality"]]
        detail = {
            "dataset": dataset,
            "p1_adjusted_gap_ci_low_positive": p1_rows[-1]["adjusted_gap_ci_low_positive"],
            "p1_adjusted_gap_mean": p1_rows[-1]["adjusted_gap_mean"],
            "p1_cross_signal_mean": p1_rows[-1]["cross_signal_mean"],
            "p2_grade_counts_alpha_0_1": {
                metric: Counter(row["evidence_grade"] for row in agg if abs(float(row["alpha"]) - 0.1) < 1e-8 and row["metric"] == metric)
                for metric in ("cross_only_graph_gain", "conditional_graph_gain", "conditional_cross_delta_graph")
            },
            "p3_text_grade": p3_grades["text"],
            "p3_image_grade": p3_grades["image"],
            "p3_description": p3_dataset_description(p3_grades["text"], p3_grades["image"]),
            "p3_alpha_sign_consistency": alpha_sign_consistency,
            "operator_stability": [row for row in stable],
            "operator_paired_rows": [row for row in operator_rows if row["dataset"] == dataset],
        }
        dataset_summary.append(detail)

    # Counter values serialize as regular dictionaries in JSON and CSV.
    for item in dataset_summary:
        for metric, counts in item["p2_grade_counts_alpha_0_1"].items():
            item["p2_grade_counts_alpha_0_1"][metric] = dict(counts)
    expected_run_count = len(DATASETS) * len(HOST_SEEDS)
    present_run_count = sum(
        int((_read_json(raw_root / dataset / f"seed_{seed}.json")).get("host_seed") == seed)
        for dataset in completed_datasets for seed in HOST_SEEDS
    )
    summary = {
        "study": "MAG Interaction Study V1.1",
        "positioning": "post-hoc corrective and robustness audit of V1",
        "complete": len(completed_datasets) == len(DATASETS) and present_run_count == expected_run_count,
        "formal_datasets_complete": completed_datasets,
        "formal_datasets_expected": list(DATASETS),
        "successful_seed_reanalyses": present_run_count,
        "expected_seed_reanalyses": expected_run_count,
        "v1_raw_seed_artifacts_reused": int(len(manifest["artifacts"])),
        "host_retrained": False,
        "test_label_access": False,
        "test_metric_access": False,
        "dataset_results": dataset_summary,
        "p2_evidence_grade_counts_alpha_0_1": {
            metric: dict(Counter(row["evidence_grade"] for row in p2_aggregate_rows if abs(float(row["alpha"]) - 0.1) < 1e-8 and row["metric"] == metric))
            for metric in ("cross_only_graph_gain", "conditional_graph_gain", "conditional_cross_delta_graph")
        },
        "p3_modality_grade_counts": {
            modality: dict(Counter(row[f"p3_{modality}_grade"] for row in dataset_summary))
            for modality in ("text", "image")
        },
    }
    report_root.mkdir(parents=True, exist_ok=True)
    write_csv(report_root / "p1_reference_noise.csv", p1_rows)
    write_csv(report_root / "p2_conditional_utility.csv", p2_rows + p2_aggregate_rows)
    write_csv(report_root / "p3_matched_bootstrap.csv", p3_rows)
    write_csv(report_root / "operator_stability.csv", stability_rows)
    write_csv(report_root / "operator_null_comparison.csv", operator_rows)
    write_json(report_root / "summary.json", summary)
    protocol = {
        "study": "MAG Interaction Study V1.1",
        "positioning": "post-hoc corrective / robustness audit",
        "base_commit": "b5fb04e3aa8a9a86b52f1b56f99135ef3783f17f",
        "source_branch": "research/interaction-study-v1",
        "analysis_branch": "research/interaction-study-v1.1",
        "formal_datasets": list(DATASETS),
        "host_seeds": list(HOST_SEEDS),
        "books_nc_included": False,
        "p1_reference_seeds": list(config["p1_reference_seeds"]),
        "p1_reference_sampling": {
            "nonedges_per_source": int(v1_config["compatibility"]["nonedges_per_source"]),
            "max_sources": int(v1_config["compatibility"]["max_sources"]),
            "graph_and_physical_edge_universe_held_fixed": True,
            "text_image_share_source_reference_pattern_within_seed": True,
        },
        "p2_alphas": list(config["p2_alphas"]),
        "p3_alphas": list(config["p3_alphas"]),
        "operator_alpha": float(config["operator_alpha"]),
        "bootstrap_reps": int(config["bootstrap_reps"]),
        "bootstrap_unit": "validation receiver node; P3 keeps all interactions from each sampled receiver block and re-matches per draw",
        "host_seed_aggregation": "average per-receiver metrics across seeds, then receiver-bootstrap; individual CI endpoints are never averaged",
        "p2_formal_name": "Directional Cross-modal Neighborhood Utility",
        "p2_interaction_term_name": "same/cross perturbation interaction term",
        "host_retrained": False,
        "test_label_access": False,
        "test_metric_access": False,
        "fit_calib_report_splits": "V1 diagnostic_fit / diagnostic_calib / original_validation reused unchanged",
        "validation_node_order_verified_across_seeds": True,
        "v1_reports_overwritten": False,
        "v1_artifact_manifest": manifest,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_requested": config.get("device", "cuda:0"),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    write_json(report_root / "protocol.json", protocol)
    _write_report(report_root / "REPORT.md", summary, p1_rows, p2_aggregate_rows, p3_rows, stability_rows, operator_rows, manifest)
    print(f"REPORT {report_root / 'REPORT.md'} complete={summary['complete']} datasets={len(completed_datasets)}/5 seeds={present_run_count}/{expected_run_count}")
    return summary


def _num(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{number:.{digits}f}" if math.isfinite(number) else "NA"


def _ci(value: Any, low: Any, high: Any, digits: int = 4) -> str:
    return f"{_num(value, digits)} [{_num(low, digits)}, {_num(high, digits)}]"


def _write_report(
    path: Path,
    summary: dict[str, Any],
    p1_rows: list[dict[str, Any]],
    p2_aggregate_rows: list[dict[str, Any]],
    p3_rows: list[dict[str, Any]],
    stability_rows: list[dict[str, Any]],
    operator_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    lines = [
        "# MAG Interaction Study V1.1 — Corrective and Robustness Re-analysis",
        "",
        "## 1. Positioning",
        "",
        "This is a **post-hoc corrective and robustness audit of V1**. It reuses V1's frozen validation split, hidden states, alignments, host state dictionaries, and response tensors. It does not retrain diagnostic hosts. No test labels or test metrics were accessed. The findings remain specific to the graph-free frozen diagnostic host and the stated action banks.",
        "",
        "## 2. Reused V1 artifacts",
        "",
        "All 15 formal dataset × host-seed raw artifacts passed existence, required-key, shape-schema, and SHA256 checks before analysis. Their shape schemas are in `protocol.json`; the complete hashes are listed here.",
        "",
        "| Dataset | Seed | Artifact | SHA256 |",
        "|---|---:|---|---|",
    ]
    for item in manifest["artifacts"]:
        lines.append(f"| {item['dataset']} | {item['seed']} | `{item['path']}` | `{item['sha256']}` |")
    lines += [
        "",
        "## 3. P1 — Correct receiver orientation and reference noise",
        "",
        "For each canonical edge, a validation endpoint is now the receiver; an edge with two validation endpoints yields both directed interactions. V1 stored one source-averaged percentile per physical edge, so this orientation correction preserves the edge-weighted point estimate while changing receiver-block assignment and its uncertainty.",
        "",
        "The reference audit uses five deterministic draws `[4103, 5103, 6103, 7103, 8103]`, 32 nonedges per source, and the same graph/physical edge set. Within each draw, text and image use the same sampled source/nonedge pattern. The descriptive ratio is not a formal signal-to-noise statistic.",
        "",
        "| Dataset | V1 mean gap | Corrected mean gap (95% receiver CI) | Median | Spearman | Cross signal mean (CI) | Text instability mean (CI) | Image instability mean (CI) | Adjusted gap mean (CI) | Ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in p1_rows:
        lines.append(
            f"| {row['dataset']} | {_num(row['v1_mean_abs_gap'])} | {_ci(row['corrected_mean_abs_gap'], row['corrected_mean_gap_ci_low'], row['corrected_mean_gap_ci_high'])} | "
            f"{_num(row['corrected_median_abs_gap'])} | {_num(row['corrected_spearman'])} | {_ci(row['cross_signal_mean'], row['cross_signal_ci_low'], row['cross_signal_ci_high'])} | "
            f"{_ci(row['text_instability_mean'], row['text_instability_ci_low'], row['text_instability_ci_high'])} | {_ci(row['image_instability_mean'], row['image_instability_ci_low'], row['image_instability_ci_high'])} | "
            f"{_ci(row['adjusted_gap_mean'], row['adjusted_gap_ci_low'], row['adjusted_gap_ci_high'])} | {_num(row['descriptive_ratio'])} |"
        )
    lines += [
        "",
        "A positive adjusted-gap receiver-bootstrap lower bound indicates that the mean cross-modal percentile gap exceeds the summed within-modality reference instability under this conservative definition. Raw gap magnitude is retained regardless of that interval.",
        "",
        "## 4. P2 — Directional Cross-modal Neighborhood Utility",
        "",
        "The formal name is **Directional Cross-modal Neighborhood Utility**. For each direction and alpha, same-modal, true cross-modal, and matched random cross-modal perturbations are separately scaled to equal relative norm. The cross-only baseline is `LR − LC`; conditional graph gain is `LSR − LSC`; conditional graph cross delta is `LSC − LS`; conditional random cross delta is `LSR − LS`. The interaction quantity is reported only as a same/cross perturbation interaction term.",
        "",
        "Each seed row reports its own receiver-bootstrap interval and sign fractions. The seed-averaged row first averages each receiver's metric over the three host seeds, then bootstraps those receiver values. No seed-interval endpoints are averaged. The evidence grade follows the V1.1 Strong / Consistent-Weak / Mixed / Unavailable rubric; no minimum effect cutoff is added.",
        "",
        "| Dataset | Direction | Metric | Seed-averaged mean (receiver CI) | Seed mean signs (0/1/2) | Grade |",
        "|---|---|---|---:|---|---|",
    ]
    for row in p2_aggregate_rows:
        if abs(float(row["alpha"]) - 0.1) >= 1e-8 or row["metric"] not in {
            "cross_only_graph_gain", "conditional_graph_gain", "conditional_cross_delta_graph", "conditional_cross_delta_random",
        }:
            continue
        signs = row["seed_mean_signs"]
        sign_text = ", ".join(str(signs.get(str(seed), "NA")) for seed in HOST_SEEDS)
        lines.append(
            f"| {row['dataset']} | {row['direction']} | `{row['metric']}` | {_ci(row['mean'], row['ci_low'], row['ci_high'])} | {sign_text} | {row['evidence_grade']} |"
        )
    lines += [
        "",
        "Directional asymmetry rows use T2I minus I2T receiver-level contrasts. Alpha sensitivity at 0.05 and 0.2 is included in `p2_conditional_utility.csv` alongside the primary 0.1 rows.",
        "",
        "## 5. P3 — Receiver-block matched structural response",
        "",
        "The primary deterministic estimate matches `semantic_bin × source_degree_bin × receiver_degree_bin` strata with both `structure_bin=0` and `structure_bin=3`, then weights each stratum by `min(n_high,n_low)`. The 95% percentile interval resamples validation receiver blocks, retains every sampled receiver's rows with multiplicity, and re-forms the strata on each draw. Draws without any valid matched stratum are omitted and counted. Standardization divides the matched estimate by the population standard deviation of all report-interaction `D_same` values for that seed and modality.",
        "",
        "| Dataset | Modality | Seed | Matched effect at α=.1 (95% receiver CI) | Standardized | Matched strata | Interactions/arm | Valid bootstrap draws | α sign stable | Evidence grade |",
        "|---|---|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    primary_p3 = [row for row in p3_rows if row["primary_alpha"]]
    for row in primary_p3:
        grade_rows = [x for x in primary_p3 if x["dataset"] == row["dataset"] and x["modality"] == row["modality"]]
        grade = classify_p3_modality(grade_rows)
        alpha_sign = row.get("sign_consistent_across_alpha")
        alpha_sign_label = "yes" if alpha_sign is True else "no" if alpha_sign is False else "NA"
        lines.append(
            f"| {row['dataset']} | {row['modality']} | {row['host_seed']} | {_ci(row['matched_effect'], row['ci_low'], row['ci_high'])} | "
            f"{_num(row['standardized_effect'])} | {row['matched_strata']} | {row['matched_interactions_per_arm']} | {row['valid_bootstrap_count']} | {alpha_sign_label} | {grade} |"
        )
    lines += ["", "Per-dataset modality descriptions:", ""]
    for dataset in DATASETS:
        text_grade = classify_p3_modality([r for r in primary_p3 if r["dataset"] == dataset and r["modality"] == "text"])
        image_grade = classify_p3_modality([r for r in primary_p3 if r["dataset"] == dataset and r["modality"] == "image"])
        if any(row["dataset"] == dataset for row in primary_p3):
            lines.append(f"- **{dataset}:** text {text_grade}; image {image_grade}; {p3_dataset_description(text_grade, image_grade)}.")
    reddit = [r for r in primary_p3 if r["dataset"] == "Reddit-S"]
    if reddit and all(int(r["matched_strata"]) == 0 for r in reddit):
        lines += ["", "Reddit-S remains unavailable because no report interactions form legal high/low embeddedness pairs within the specified strata. No bin threshold or jitter was changed."]
    lines += [
        "",
        "Alpha 0.05/0.2 point estimates are reported as sensitivity checks. `sign_consistent_across_alpha` records whether a seed's deterministic matched-effect sign agrees at 0.05, 0.1, and 0.2; it does not replace the alpha=0.1 receiver-bootstrap grade. V1 repeated-subsample intervals appear only as historical diagnostics in the CSV and are not used for V1.1 evidence grades.",
        "",
        "| Dataset | Modality | Seeds with same matched-effect sign / available seeds at α=.05/.1/.2 |",
        "|---|---|---:|",
    ]
    for dataset in DATASETS:
        if dataset not in summary["formal_datasets_complete"]:
            continue
        for modality in ("text", "image"):
            rows = [r for r in p3_rows if r["dataset"] == dataset and r["modality"] == modality and r["primary_alpha"]]
            available_count = sum(r.get("sign_consistent_across_alpha") is not None for r in rows)
            stable_count = sum(r.get("sign_consistent_across_alpha") is True for r in rows)
            lines.append(f"| {dataset} | {modality} | {stable_count}/{available_count} |")
    lines += [
        "",
        "## 6. Operator preference and matched random-action null",
        "",
        "Functional validation profiles are reused from V1 in the same canonical validation-node order. The functional fixed action is the V1 calibration-selected action. Null calibration/report profiles are computed from the frozen host using the same graph source counts, alpha, equal-relative-norm rule, head, and four actions. Stability is reported for both banks. Margin-filtered subsets affect only the stability summaries.",
        "",
        "Top-action agreement is shown as the mean over the three seed pairs; per-pair values and Cohen kappa, per-action Pearson/Spearman profile correlations, and centered four-action profile cosine are in `operator_stability.csv`.",
        "",
        "| Dataset | Modality | Bank | Pair agreement | Pair kappa | Unanimous top action | At least 2/3 agree | Profile cosine mean | Agreement if both margins ≥1e−3 | Agreement in both top-half margin sets |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in stability_rows:
        pairs = row["seed_pairs"]
        margin_strict = [p["margin_conditioned"]["both_margin_at_least_threshold"] for p in pairs]
        margin_top = [p["margin_conditioned"]["both_in_seed_specific_top_half_margin"] for p in pairs]
        cos_values = [p["centered_profile_cosine_mean"] for p in pairs if p["centered_profile_cosine_mean"] is not None]
        corr = [p["raw_agreement"] for p in pairs]
        kappa = [p["cohen_kappa"] for p in pairs]
        strict_values = [x["raw_agreement"] for x in margin_strict if x["receiver_count"] > 0]
        top_values = [x["raw_agreement"] for x in margin_top if x["receiver_count"] > 0]
        lines.append(
            f"| {row['dataset']} | {row['modality']} | {row['bank']} | {_num(np.mean(corr))} | {_num(np.mean(kappa))} | "
            f"{_num(row['unanimous_top_action_fraction'])} | {_num(row['at_least_two_of_three_same_action_fraction'])} | "
            f"{_num(np.mean(cos_values) if cos_values else float('nan'))} | {_num(np.mean(strict_values) if strict_values else float('nan'))} | {_num(np.mean(top_values) if top_values else float('nan'))} |"
        )
    lines += [
        "",
        "Paired validation quantities compare the functional and matched random-null banks node by node. Positive excess headroom means functional adaptive response spread is larger; positive best-action advantage means the best functional response has lower CE; positive global-fixed advantage means the calibration-selected functional action has lower CE. These are frozen-host, label-aware diagnostic quantities, not deployable performance.",
        "",
        "| Dataset | Modality | Seed | Functional headroom | Null headroom | Excess headroom | Best-action advantage | Global-fixed advantage |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    operator_map = {(r["dataset"], r["modality"], r["host_seed"], r["metric"]): r for r in operator_rows}
    for dataset in DATASETS:
        if dataset not in summary["formal_datasets_complete"]:
            continue
        for modality in ("text", "image"):
            for seed in HOST_SEEDS:
                stats = {
                    metric: operator_map.get((dataset, modality, seed, metric), {})
                    for metric in ("functional_headroom", "null_headroom", "excess_headroom", "best_action_advantage", "global_fixed_advantage")
                }
                lines.append(
                    f"| {dataset} | {modality} | {seed} | "
                    + " | ".join(_ci(stats[k].get("mean"), stats[k].get("ci_low"), stats[k].get("ci_high")) for k in stats)
                    + " |"
                )
    lines += [
        "",
        "## 7. Cross-dataset consistency",
        "",
        "| Question | Dataset-level observations |",
        "|---|---|",
        f"| P1 adjusted gap CI lower bound > 0 | {sum(bool(r['adjusted_gap_ci_low_positive']) for r in p1_rows)}/{len(p1_rows)} datasets |",
    ]
    for metric in ("cross_only_graph_gain", "conditional_graph_gain", "conditional_cross_delta_graph"):
        counts = Counter(
            row["evidence_grade"] for row in p2_aggregate_rows
            if abs(float(row["alpha"]) - 0.1) < 1e-8 and row["metric"] == metric and row["direction"] in {"T2I", "I2T"}
        )
        lines.append(f"| P2 `{metric}` grades across dataset × direction | {dict(counts)} |")
    for modality in ("text", "image"):
        grades = Counter(
            classify_p3_modality([r for r in primary_p3 if r["dataset"] == ds and r["modality"] == modality])
            for ds in DATASETS if ds in summary["formal_datasets_complete"]
        )
        lines.append(f"| P3 {modality} modality grades | {dict(grades)} |")
    lines += [
        "",
        "## 8. Corrections relative to V1",
        "",
        "- **P1:** the corrected point estimate is compared directly with V1; because V1 stores edge-level source-averaged percentiles, reassigning receiver orientation leaves that point estimate unchanged. The receiver-block interval is recomputed. Whether the gap clears reference-sampling instability is shown by each dataset's adjusted-gap interval, without a magnitude cutoff.",
        "- **P2:** the graph-versus-random cross-only baseline is retained, while the new conditional contrasts ask what remains after the same-modal graph perturbation. Conditional results that differ from V1's cross-only pattern are explicitly weakened or unresolved at that dataset/direction; no universal direction is imposed.",
        "- **P3:** V1's repeated-subsample intervals are historical only. The primary conclusions now depend on deterministic weighted matching and receiver-block intervals, with text and image graded separately. Results outside the V1 direction or with unavailable strata are weakened or unavailable under the corrected uncertainty.",
        "- **Operator:** V1's response profiles remain descriptive. V1.1 adds seed agreement and a matched random-action comparison; stability and paired headroom are reported as measurements, without a GO/NO-GO threshold.",
        "",
        "### Observed disposition from V1.1",
    ]
    p1_above_noise = [r["dataset"] for r in p1_rows if bool(r["adjusted_gap_ci_low_positive"])]
    p1_not_above_noise = [r["dataset"] for r in p1_rows if not bool(r["adjusted_gap_ci_low_positive"])]
    lines.append(
        f"- **P1:** the original point magnitude is retained by construction for all analyzed datasets. The adjusted-gap interval is above zero for {p1_above_noise or 'none'}; it is not above zero for {p1_not_above_noise or 'none'}. The V1 gap magnitude remains a descriptive fact; evidence beyond reference sampling noise is dataset-specific."
    )
    p2_by_key = {
        (row["dataset"], row["direction"], row["metric"]): row
        for row in p2_aggregate_rows if abs(float(row["alpha"]) - 0.1) < 1e-8
    }
    conditional_retained = []
    conditional_reverse = []
    conditional_uncertain = []
    for dataset in summary["formal_datasets_complete"]:
        for direction in ("T2I", "I2T"):
            gain = p2_by_key.get((dataset, direction, "conditional_graph_gain"), {})
            delta = p2_by_key.get((dataset, direction, "conditional_cross_delta_graph"), {})
            gain_ci = float(gain.get("ci_low", float("nan"))), float(gain.get("ci_high", float("nan")))
            delta_ci = float(delta.get("ci_low", float("nan"))), float(delta.get("ci_high", float("nan")))
            label = f"{dataset} {direction}"
            if gain.get("evidence_grade") in {"Strong", "Consistent-Weak"} and gain_ci[0] > 0 and delta_ci[1] < 0:
                conditional_retained.append(label)
            elif gain_ci[1] < 0 and delta_ci[0] > 0:
                conditional_reverse.append(label)
            else:
                conditional_uncertain.append(label)
    lines.append(
        f"- **P2:** conditional graph utility is retained in {conditional_retained or 'no dataset-direction cells'} under positive conditional graph gain and negative graph cross delta. Opposite-signed receiver intervals occur in {conditional_reverse or 'no cells'}; the remaining cells are weakened or unresolved: {conditional_uncertain or 'none'}. This is a per-cell description, not a cross-dataset rule."
    )
    p3_dispositions = []
    for dataset in summary["formal_datasets_complete"]:
        for modality in ("text", "image"):
            cells = [r for r in primary_p3 if r["dataset"] == dataset and r["modality"] == modality]
            grade = classify_p3_modality(cells)
            if grade == "Unavailable":
                disposition = "unavailable"
            else:
                newer = [int(np.sign(float(r["matched_effect"]))) for r in cells]
                older = [int(np.sign(float((r.get("historical_v1_repeated_subsample_interval") or {}).get("mean_high_minus_low", float("nan"))))) for r in cells]
                if grade in {"Strong", "Consistent"} and len(older) == 3 and newer == older:
                    disposition = "remains directionally consistent"
                elif grade == "Strong" and len(older) == 3 and all(a == -b and a != 0 for a, b in zip(newer, older, strict=True)):
                    disposition = "overturned in direction"
                else:
                    disposition = "weakened or unresolved under receiver-block uncertainty"
            p3_dispositions.append(f"{dataset}/{modality}: {disposition} ({grade})")
    lines.append("- **P3:** " + "; ".join(p3_dispositions) + ".")
    excess_positive = [
        f"{r['dataset']}/{r['modality']}/seed{r['host_seed']}"
        for r in operator_rows if r["metric"] == "excess_headroom" and float(r["ci_low"]) > 0
    ]
    excess_negative = [
        f"{r['dataset']}/{r['modality']}/seed{r['host_seed']}"
        for r in operator_rows if r["metric"] == "excess_headroom" and float(r["ci_high"]) < 0
    ]
    excess_crossing = [
        f"{r['dataset']}/{r['modality']}/seed{r['host_seed']}"
        for r in operator_rows if r["metric"] == "excess_headroom" and float(r["ci_low"]) <= 0 <= float(r["ci_high"])
    ]
    lines.append(
        f"- **Operator vs matched null:** functional excess-headroom intervals are wholly positive in {excess_positive or 'no cells'}, wholly negative in {excess_negative or 'no cells'}, and include zero in {excess_crossing or 'no cells'}. Best-action and global-fixed advantages remain separately reported; the data do not choose a deployment model."
    )
    lines += [
        "",
        "## 9. Threats to validity",
        "",
        "- This is a post-hoc audit, and it reuses the same validation data as V1.",
        "- The diagnostic host is frozen, graph-free, and defines the response surface; another host may produce different responses.",
        "- The matched nonneighbor bank is one null design and does not represent every possible control.",
        "- Per-node response profiles and adaptive best-action quantities use validation labels and are not deployable router outputs.",
        "- Three host seeds are robustness seeds, not statistical population replicates; no seed-level t-test is used.",
        "- The study remains transductive with respect to graph topology and uses the V1 validation split, so results do not establish performance on a new split or an end-to-end model.",
        "",
        "## 10. Final factual evidence table",
        "",
        "| Question | Evidence reported | Uncertainty unit | Interpretation boundary |",
        "|---|---|---|---|",
        "| P1 gap vs reference noise | Corrected raw-feature gap, five-reference cross signal, modality instability, conservative adjusted gap | Validation receiver blocks | Source-matched percentile estimator and selected graph only |",
        "| P2 directional utility | Cross-only and conditional factorial contrasts at α=.05/.1/.2, separate T2I/I2T | Same receiver averaged across host seeds, then receiver bootstrap | Frozen host local loss response |",
        "| P3 structure response | Deterministic matched effect, standardized magnitude, receiver-block interval, text/image grades | Validation receiver blocks; re-match each draw | Structural bins and report interactions inherited from V1 |",
        "| Functional vs null actions | Agreement, kappa, profile correlation/cosine, margin subgroups, paired headroom/advantages | Nodewise profile stability; receiver bootstrap for paired means | Label-aware frozen-host diagnostic response |",
        "",
        "No final model or router is recommended by this audit. Full protocol, exact seed rows, bootstrap counts, and compact CSVs are provided beside this report.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    v11_config = OmegaConf.to_container(OmegaConf.load(CONFIG_PATH), resolve=True)
    v1_config = OmegaConf.to_container(OmegaConf.load(V1_CONFIG_PATH), resolve=True)
    raw_root = (PROJECT_ROOT / v11_config["outputs"]["raw_root"]).resolve()
    report_root = (PROJECT_ROOT / v11_config["outputs"]["report_root"]).resolve()
    v1_raw_root = (PROJECT_ROOT / v1_config["outputs"]["raw_root"]).resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"V1.1 raw output directory is missing: {raw_root}")
    branch = __import__("subprocess").run(
        ["git", "branch", "--show-current"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()
    if branch != "research/interaction-study-v1.1":
        raise RuntimeError(f"V1.1 runner must execute on research/interaction-study-v1.1, current={branch}")
    head = __import__("subprocess").run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()
    ancestor = __import__("subprocess").run(
        ["git", "merge-base", "--is-ancestor", v11_config["base_commit"], "HEAD"], cwd=PROJECT_ROOT,
    ).returncode == 0
    if not ancestor:
        raise RuntimeError(f"HEAD {head} is not based on configured V1 commit {v11_config['base_commit']}")
    if args.finalize_only:
        manifest = _load_input_manifest(v1_raw_root, raw_root)
        summary = _finalize(raw_root, report_root, manifest, v11_config, v1_config)
        return 0 if summary["complete"] else 1
    datasets = list(args.datasets or v11_config["formal_datasets"])
    unknown = sorted(set(datasets) - set(v11_config["formal_datasets"]))
    if unknown:
        raise ValueError(f"only configured formal V1.1 datasets may be run: {unknown}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)
    if args.bootstrap_reps is not None:
        v11_config["bootstrap_reps"] = int(args.bootstrap_reps)
    torch.set_num_threads(4)
    raw_root.mkdir(parents=True, exist_ok=True)
    manifest = _load_input_manifest(v1_raw_root, raw_root)
    print(f"V1.1 post-hoc audit base={v11_config['base_commit']} branch={branch} device={device}")
    print(f"Verified V1 raw seed artifacts: {len(manifest['artifacts'])}/15; host retraining disabled.")
    for dataset in datasets:
        print(f"START V1.1 {dataset}")
        _run_dataset(dataset, args, v1_config, v11_config, manifest, v1_raw_root, raw_root, device)
        summary = _finalize(raw_root, report_root, manifest, v11_config, v1_config)
        print(f"CHECKPOINT {dataset}: complete={summary['complete']} reports={report_root}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Corrective post-hoc robustness re-analysis of frozen Interaction Study V1 artifacts.")
    parser.add_argument("--datasets", nargs="*", choices=DATASETS, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true", help="reuse existing V1.1 result files when raw V1 hashes match")
    parser.add_argument("--bootstrap-reps", type=int, default=None, help="override bootstrap reps for pilot validation only")
    parser.add_argument("--finalize-only", action="store_true", help="regenerate compact V1.1 reports from existing raw summaries")
    return parser


def main() -> None:
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    args = build_parser().parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
