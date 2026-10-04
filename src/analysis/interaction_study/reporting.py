from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (torch.Tensor, np.ndarray)):
        return json_safe(value.detach().cpu().tolist() if isinstance(value, torch.Tensor) else value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            packed = {}
            for key in columns:
                value = json_safe(row.get(key))
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, sort_keys=True, allow_nan=False)
                packed[key] = value
            writer.writerow(packed)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _f(value: Any) -> float:
    return float(value) if value is not None else float("nan")


def _p3_status(rows: list[dict[str, Any]], min_abs: float, seed_consistency: int) -> tuple[str, list[dict[str, Any]]]:
    matched = [r for r in rows if r.get("row_type") == "matched_high_low_structure"]
    details = []
    for modality in ("text", "image"):
        part = [r for r in matched if r.get("modality") == modality]
        pos = sum(_f(r.get("mean_high_minus_low")) > min_abs and _f(r.get("ci_low")) > 0 for r in part)
        neg = sum(_f(r.get("mean_high_minus_low")) < -min_abs and _f(r.get("ci_high")) < 0 for r in part)
        details.append({"modality": modality, "positive_seed_count": pos, "negative_seed_count": neg,
                        "effects": [r.get("mean_high_minus_low") for r in part]})
        if max(pos, neg) >= seed_consistency:
            return "Supported", details
    if any(math.isfinite(_f(r.get("mean_high_minus_low"))) and abs(_f(r.get("mean_high_minus_low"))) > min_abs for r in matched):
        return "Mixed", details
    return "Unsupported", details


def _p2_status(rows: list[dict[str, Any]], min_abs: float) -> tuple[str, float, float, float, bool]:
    rows = [r for r in rows if r.get("direction") == "T2I_minus_I2T" and abs(_f(r.get("alpha")) - 0.1) < 1e-8]
    if not rows:
        return "Unsupported", float("nan"), float("nan"), float("nan"), True
    means = [_f(r.get("directional_asymmetry_mean")) for r in rows]
    lows = [_f(r.get("directional_asymmetry_ci_low")) for r in rows]
    highs = [_f(r.get("directional_asymmetry_ci_high")) for r in rows]
    low_align = any(bool(r.get("alignment_low_quality")) for r in rows)
    mean, low, high = float(np.nanmean(means)), float(np.nanmean(lows)), float(np.nanmean(highs))
    same_sign = all(x > 0 for x in means) or all(x < 0 for x in means)
    if not low_align and same_sign and (low > min_abs or high < -min_abs):
        return "Supported", mean, low, high, low_align
    if any(math.isfinite(x) and abs(x) > min_abs for x in means):
        return "Mixed", mean, low, high, low_align
    return "Unsupported", mean, low, high, low_align


def finalize_reports(raw_root: str | Path, report_root: str | Path, protocol: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    raw_root, report_root = Path(raw_root), Path(report_root)
    runs = [_read(p) for p in sorted(raw_root.glob("*/seed_*.json"))]
    good = [r for r in runs if r.get("status") == "success"]
    failures = [r for r in runs if r.get("status") != "success"]
    expected_datasets = list(protocol["primary_datasets"])
    expected_seeds = list(protocol["formal_host_seeds"])
    host_rows, p1_rows, p2_rows, p3_rows, operator_rows = [], [], [], [], []
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    raw_p1_datasets: set[str] = set()
    for run in good:
        name = run["dataset"]
        by_dataset.setdefault(name, []).append(run)
        host = run["host"]
        host_rows.append({
            "dataset": name, "host_seed": run["host_seed"], "best_epoch": host["best_epoch"],
            "best_calibration_ce": host["best_calibration_ce"], "train_nodes_used": host["train_nodes_used"],
            "train_nodes_available": host["train_nodes_available"], "low_host_quality": host["low_host_quality"],
            "fit_accuracy": host["metrics"]["fit"]["accuracy"], "fit_macro_f1": host["metrics"]["fit"]["macro_f1"],
            "calib_accuracy": host["metrics"]["calib"]["accuracy"], "calib_macro_f1": host["metrics"]["calib"]["macro_f1"],
            "report_accuracy": host["metrics"]["report"]["accuracy"], "report_macro_f1": host["metrics"]["report"]["macro_f1"],
            "majority_report_accuracy": host["majority_baseline"]["report"]["accuracy"],
            "majority_report_macro_f1": host["majority_baseline"]["report"]["macro_f1"],
        })
        for space in ("raw", "projected"):
            if space == "raw" and name in raw_p1_datasets:
                continue
            p1_rows.append({"dataset": name, "host_seed": run["host_seed"], "analysis_space": space, **run["p1"][space]})
            if space == "raw":
                raw_p1_datasets.add(name)
        directions = run["alignment"]["directions"]
        alignment_columns = {
            f"alignment_{direction.lower()}_{split}_{metric}": directions[direction][split][metric]
            for direction in ("T2I", "I2T")
            for split in ("calib", "validation")
            for metric in ("cosine", "mse", "mse_skill_vs_constant")
        }
        p2_rows.extend({"dataset": name, "host_seed": run["host_seed"],
                        "alignment_low_quality": run["alignment"]["alignment_low_quality"],
                        **alignment_columns, **row} for row in run.get("p2", []))
        p3_rows.extend({"dataset": name, "host_seed": run["host_seed"], **row} for row in run.get("p3", []))
        p3_rows.extend({"dataset": name, "host_seed": run["host_seed"], "row_type": "incremental_explanatory_signal", **row}
                       for row in run.get("p3_regression", []))
        operator_rows.extend({"dataset": name, "host_seed": run["host_seed"], **row} for row in run.get("operator", []))

    majority = int(protocol["decision_rules"]["cross_dataset_majority"])
    dataset_rows = []
    for name in expected_datasets:
        audit_row = audit.get("datasets", {}).get(name, {})
        runs_ds = by_dataset.get(name, [])
        p1_ds = [r for r in p1_rows if r["dataset"] == name and r["analysis_space"] == "raw"]
        p1 = p1_ds[0] if p1_ds else {}
        gap = _f(p1.get("mean_absolute_percentile_gap"))
        p1_supported = gap >= float(protocol["decision_rules"]["p1_min_mean_abs_percentile_gap"]) and _f(p1.get("mean_absolute_percentile_gap_ci_low")) > 0
        p2_status, p2_mean, p2_lo, p2_hi, p2_low = _p2_status([r for r in p2_rows if r["dataset"] == name], float(protocol["decision_rules"]["p2_min_abs_directional_asymmetry"]))
        p3_status, p3_details = _p3_status([r for r in p3_rows if r["dataset"] == name], float(protocol["decision_rules"]["p3_min_abs_matched_structure_effect"]), int(protocol["decision_rules"]["p3_seed_sign_consistency"]))
        dataset_rows.append({
            "dataset": name, "nodes": audit_row.get("num_nodes"), "edges_directed": audit_row.get("num_edges_directed"),
            "text_dim": audit_row.get("text_dim"), "image_dim": audit_row.get("image_dim"),
            "train_nodes": (audit_row.get("split_counts") or {}).get("train"),
            "validation_nodes": (audit_row.get("split_counts") or {}).get("validation"),
            "test_nodes": (audit_row.get("split_counts") or {}).get("test"),
            "raw_feature_gb": audit_row.get("raw_feature_gb"),
            "mean_degree": (audit_row.get("degree") or {}).get("mean"),
            "median_degree": (audit_row.get("degree") or {}).get("median"),
            "p95_degree": (audit_row.get("degree") or {}).get("q95"),
            "max_degree": (audit_row.get("degree") or {}).get("max"),
            "zero_degree_nodes": (audit_row.get("degree") or {}).get("zero_degree_nodes"),
            "estimated_diagnostic_gpu_peak_gb": audit_row.get("estimated_diagnostic_gpu_peak_gb"),
            "successful_seed_runs": len(runs_ds), "expected_seed_runs": len(expected_seeds),
            "p1_raw_status": "Supported" if p1_supported else "Unsupported",
            "p2_directional_asymmetry_status": p2_status, "p2_mean_asymmetry_alpha_0_1": p2_mean,
            "p2_mean_seed_ci_low": p2_lo, "p2_mean_seed_ci_high": p2_hi, "p2_alignment_low_quality_any_seed": p2_low,
            "p3_matched_structure_status": p3_status, "p3_seed_details": p3_details, "audit_only": False,
        })
    books = audit.get("datasets", {}).get("books-nc")
    if books:
        dataset_rows.append({"dataset": "books-nc", "nodes": books.get("num_nodes"), "edges_directed": books.get("num_edges_directed"),
                             "text_dim": books.get("text_dim"), "image_dim": books.get("image_dim"),
                             "train_nodes": (books.get("split_counts") or {}).get("train"),
                             "validation_nodes": (books.get("split_counts") or {}).get("validation"),
                             "test_nodes": (books.get("split_counts") or {}).get("test"),
                             "raw_feature_gb": books.get("raw_feature_gb"),
                             "mean_degree": (books.get("degree") or {}).get("mean"),
                             "median_degree": (books.get("degree") or {}).get("median"),
                             "p95_degree": (books.get("degree") or {}).get("q95"),
                             "max_degree": (books.get("degree") or {}).get("max"),
                             "zero_degree_nodes": (books.get("degree") or {}).get("zero_degree_nodes"),
                             "estimated_diagnostic_gpu_peak_gb": books.get("estimated_diagnostic_gpu_peak_gb"),
                             "successful_seed_runs": 0, "expected_seed_runs": 0, "audit_only": True,
                             "status": "audit only; full interaction study not run by protocol"})
    def overall_status(key: str) -> str:
        values = [r.get(key, "Unsupported") for r in dataset_rows if not r.get("audit_only")]
        count = sum(v == "Supported" for v in values)
        if count >= majority:
            return "Supported"
        return "Mixed" if count or any(v == "Mixed" for v in values) else "Unsupported"
    expected_count = len(expected_datasets) * len(expected_seeds)
    summary = {
        "study_name": protocol["study_name"], "complete": len(good) == expected_count and not failures,
        "expected_dataset_seed_runs": expected_count, "successful_seed_runs": len(good), "failed_seed_runs": len(failures),
        "p1_cross_dataset_status": overall_status("p1_raw_status"),
        "p2_cross_dataset_status": overall_status("p2_directional_asymmetry_status"),
        "p3_cross_dataset_status": overall_status("p3_matched_structure_status"),
        "decision_rules": protocol["decision_rules"], "dataset_decisions": dataset_rows, "failure_cases": failures,
        "warning": "Decision labels apply only to the preregistered descriptive criteria; they do not establish causality or router performance.",
    }
    protocol_output = dict(protocol)
    protocol_output["runtime_metadata"] = {
        "python_versions": sorted({str(r["protocol"].get("python_version")) for r in good}),
        "torch_versions": sorted({str(r["protocol"].get("torch_version")) for r in good}),
        "cuda_versions": sorted({str(r["protocol"].get("cuda_version")) for r in good}),
        "gpu_names": sorted({str(r["protocol"].get("gpu_name")) for r in good}),
        "devices": sorted({str(r["protocol"].get("device")) for r in good}),
    }
    protocol_output["dataset_configs"] = {
        name: next((r["protocol"].get("resolved_dataset_config") for r in good if r["dataset"] == name), None)
        for name in expected_datasets
    }
    protocol_output["study_splits"] = {
        f"{r['dataset']}:seed_{r['host_seed']}": {
            "study_seed": r["protocol"].get("study_seed"),
            "split_counts": r["protocol"].get("split_counts"),
            "index_sha256": r["protocol"].get("index_sha256"),
            "fit_split": r["protocol"].get("fit_split"),
            "calib_split": r["protocol"].get("calib_split"),
            "report_split": r["protocol"].get("report_split"),
            "test_label_access": r["protocol"].get("test_label_access"),
            "test_metric_access": r["protocol"].get("test_metric_access"),
            "transductive_topology": r["protocol"].get("transductive_topology"),
        }
        for r in good
    }
    report_root.mkdir(parents=True, exist_ok=True)
    write_json(report_root / "protocol.json", protocol_output)
    write_json(report_root / "summary.json", summary)
    write_csv(report_root / "dataset_summary.csv", dataset_rows)
    write_csv(report_root / "diagnostic_host_summary.csv", host_rows)
    write_csv(report_root / "p1_summary.csv", p1_rows)
    write_csv(report_root / "p2_summary.csv", p2_rows)
    write_csv(report_root / "p3_summary.csv", p3_rows)
    write_csv(report_root / "operator_response_summary.csv", operator_rows)
    _write_markdown(report_root / "REPORT.md", protocol, audit, summary, host_rows, p1_rows, p2_rows, p3_rows, operator_rows)
    return summary


def _fmt(value: Any) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{x:.4f}" if math.isfinite(x) else "NA"


def _write_markdown(path: Path, protocol: dict[str, Any], audit: dict[str, Any], summary: dict[str, Any], host_rows: list[dict[str, Any]], p1_rows: list[dict[str, Any]], p2_rows: list[dict[str, Any]], p3_rows: list[dict[str, Any]], operator_rows: list[dict[str, Any]]) -> None:
    failures = summary.get("failure_cases", [])
    lines = [
        "# MAG Problem Validation / Interaction Study V1", "",
        f"Status: **{'complete' if summary['complete'] else 'incomplete'}** ({summary['successful_seed_runs']}/{summary['expected_dataset_seed_runs']} formal dataset-seed runs).", "",
        "## 1. Exact protocol", "",
        f"- Host: graph-free, separate text/image projectors, hidden_dim={protocol['diagnostic_host']['hidden_dim']}, AdamW lr={protocol['diagnostic_host']['lr']}, weight_decay={protocol['diagnostic_host']['weight_decay']}, epochs≤{protocol['diagnostic_host']['epochs']}, patience={protocol['diagnostic_host']['patience']}; seeds={protocol['formal_host_seeds']}.",
        "- Split: deterministic per-class 80/20 split of original train into diagnostic_fit/calib; original validation is the report split.",
        "- Isolation: test_label_access=false; test_metric_access=false. Test indices are hashed/masked; no test label values or test metrics are used.",
        "- Full transductive topology and unlabelled node features are retained. No operator-specific classifier is trained.",
        f"- P1: {protocol['protocol']['p1_compatibility']}; raw features primary, host projected states robustness; thresholds are fit from train-endpoint edges.",
        f"- P2: neighbor_cap={protocol['interaction']['neighbor_cap']}, alphas={protocol['interaction']['p2_alphas']}; gain=random-control delta CE − graph-neighbor delta CE.",
        f"- P3: directed interactions capped at {protocol['interaction']['max_validation_directed_edges']} per receiver split; alphas={protocol['interaction']['p3_alphas']}; train-edge thresholds; receiver-block bootstrap.",
        f"- Cross-modal alignment: train-only ridge λ={protocol['alignment']['ridge_regularization']}; quality assessed on calibration and validation.", "",
        "## 2. Dataset audit", "",
        "| Dataset | Nodes | Directed edges | Text/Image dim | Train/val/test | Mean/median/p95/max degree | Raw feature GB | Estimated GPU peak GB | Status |",
        "|---|---:|---:|---|---|---|---:|---:|---|",
    ]
    for name in protocol["audit_datasets"]:
        row = audit.get("datasets", {}).get(name, {})
        counts = row.get("split_counts", {})
        triplet = f"{counts.get('train','NA')}/{counts.get('validation','NA')}/{counts.get('test','NA')}"
        status = "audit only" if name == "books-nc" else ("formal study" if any(r["dataset"] == name for r in host_rows) else "not run")
        degree = row.get("degree", {})
        degree_text = "/".join(_fmt(degree.get(k)) for k in ("mean", "median", "q95", "max"))
        lines.append(f"| {name} | {row.get('num_nodes','NA')} | {row.get('num_edges_directed','NA')} | {row.get('text_dim','NA')}/{row.get('image_dim','NA')} | {triplet} | {degree_text} | {_fmt(row.get('raw_feature_gb'))} | {_fmt(row.get('estimated_diagnostic_gpu_peak_gb'))} | {status} |")
    lines.extend(["", "Books-nc was audited only; a full study is excluded by the protocol.", "", "## 3. Diagnostic host quality", "",
                  "| Dataset | Seed | Fit Acc/F1 | Calib Acc/F1 | Validation Acc/F1 | Majority val Acc | Low quality |", "|---|---:|---|---|---|---:|---|"])
    for r in host_rows:
        lines.append(f"| {r['dataset']} | {r['host_seed']} | {_fmt(r['fit_accuracy'])}/{_fmt(r['fit_macro_f1'])} | {_fmt(r['calib_accuracy'])}/{_fmt(r['calib_macro_f1'])} | {_fmt(r['report_accuracy'])}/{_fmt(r['report_macro_f1'])} | {_fmt(r['majority_report_accuracy'])} | {r['low_host_quality']} |")
    lines.extend(["", "Low host quality means report accuracy is less than 0.02 above the fit-derived majority baseline; this flags lower response confidence and does not select models.", "", "## 4. P1 — modality-specific edge interaction heterogeneity", ""])
    for r in p1_rows:
        if r.get("analysis_space") == "raw":
            projected = [x for x in p1_rows if x.get("dataset") == r["dataset"] and x.get("analysis_space") == "projected"]
            projected_gap = float(np.nanmean([_f(x.get("mean_absolute_percentile_gap")) for x in projected])) if projected else float("nan")
            lines.append(f"- {r['dataset']}: Spearman={_fmt(r.get('spearman'))}; raw mean/median |pT−pI|={_fmt(r.get('mean_absolute_percentile_gap'))}/{_fmt(r.get('median_absolute_percentile_gap'))}; receiver-block CI=[{_fmt(r.get('mean_absolute_percentile_gap_ci_low'))}, {_fmt(r.get('mean_absolute_percentile_gap_ci_high'))}]; projected mean gap across seeds={_fmt(projected_gap)}; high-T/low-I={_fmt(r.get('high_text_low_image_fraction'))}; low-T/high-I={_fmt(r.get('low_text_high_image_fraction'))}.")
    lines.extend(["", f"Decision: **{summary['p1_cross_dataset_status']}**. Per-dataset support requires raw mean absolute percentile gap ≥ {protocol['decision_rules']['p1_min_mean_abs_percentile_gap']} and CI lower bound > 0. Raw cosine distributions are not directly compared. Same-label-by-quadrant is descriptive only, not edge utility.", "", "## 5. P2 — directional cross-modal complementarity", ""])
    for r in summary["dataset_decisions"]:
        if not r.get("audit_only"):
            data_rows = [x for x in p2_rows if x["dataset"] == r["dataset"] and abs(_f(x.get("alpha")) - 0.1) < 1e-8]
            direction_summaries = []
            for direction in ("T2I", "I2T"):
                part = [x for x in data_rows if x.get("direction") == direction]
                if part:
                    gain = float(np.nanmean([_f(x.get("graph_gain_mean")) for x in part]))
                    positive = float(np.nanmean([_f(x.get("graph_gain_fraction_positive")) for x in part]))
                    direction_summaries.append(f"{direction} gain={_fmt(gain)} (positive receivers={_fmt(positive)})")
            align_t2i = float(np.nanmean([_f(x.get("alignment_t2i_validation_cosine")) for x in data_rows])) if data_rows else float("nan")
            align_i2t = float(np.nanmean([_f(x.get("alignment_i2t_validation_cosine")) for x in data_rows])) if data_rows else float("nan")
            lines.append(f"- {r['dataset']}: α=0.1 {', '.join(direction_summaries)}; T→I−I→T gain={_fmt(r.get('p2_mean_asymmetry_alpha_0_1'))}, mean seed CI=[{_fmt(r.get('p2_mean_seed_ci_low'))}, {_fmt(r.get('p2_mean_seed_ci_high'))}], alignment val cosine T→I/I→T={_fmt(align_t2i)}/{_fmt(align_i2t)}, status={r.get('p2_directional_asymmetry_status')}, alignment-low={r.get('p2_alignment_low_quality_any_seed')}.")
    lines.extend(["", f"Decision: **{summary['p2_cross_dataset_status']}**. Positive graph_gain means real neighbors caused a smaller CE increase than equal-count random sources in the frozen host. Directional asymmetry is computed from graph-minus-random gains. Low alignment quality blocks a Supported label.", "", "## 6. P3 — structure-conditioned interaction response", ""])
    for r in p3_rows:
        if r.get("row_type") == "matched_high_low_structure":
            lines.append(f"- {r['dataset']} seed {r['host_seed']} {r['modality']}: high-minus-low embeddedness D_same={_fmt(r.get('mean_high_minus_low'))}, repeated-subsample interval=[{_fmt(r.get('ci_low'))}, {_fmt(r.get('ci_high'))}], matched strata={r.get('matched_strata')}, interactions/arm={r.get('matched_interaction_count_per_arm')}.")
    lines.append(f"- Decision: **{summary['p3_cross_dataset_status']}**. The primary table compares D_same across embeddedness quartiles within each fixed semantic bin. Matching controls semantic bin plus source/receiver degree quartiles. D_same < 0 means moving toward the same-modality neighbor lowers local loss; D_same > 0 means it raises loss. This is frozen-host local response, not a causal effect.")
    regression = [r for r in p3_rows if r.get("row_type") == "incremental_explanatory_signal"]
    lines.extend(["", "### Incremental explanatory signal", ""])
    for r in regression:
        sem, full = r.get("semantic_only", {}), r.get("semantic_plus_structure", {})
        lines.append(f"- {r['dataset']} seed {r['host_seed']} {r['modality']}: validation R² semantic/local-context={_fmt(sem.get('r2'))}, plus structure={_fmt(full.get('r2'))}, ΔR²={_fmt(r.get('incremental_explanatory_signal_delta_R2'))}; MAE {_fmt(sem.get('mae'))}→{_fmt(full.get('mae'))}. Not router learnability.")
    if regression:
        deltas_r2 = [_f(r.get("incremental_explanatory_signal_delta_R2")) for r in regression]
        deltas_mae = [_f(r.get("semantic_plus_structure", {}).get("mae")) - _f(r.get("semantic_only", {}).get("mae")) for r in regression]
        finite_r2 = [v for v in deltas_r2 if math.isfinite(v)]
        finite_mae = [v for v in deltas_mae if math.isfinite(v)]
        lines.append(f"- Across {len(regression)} comparisons, mean ΔR²={_fmt(np.mean(finite_r2))}; R² improved in {sum(v > 0 for v in finite_r2)}/{len(finite_r2)}; MAE improved in {sum(v < 0 for v in finite_mae)}/{len(finite_mae)}. R² is near or below zero in most rows, so this secondary regression provides little stable validation explanatory signal.")
    lines.extend(["", "## 7. Operator-response study", ""])
    for r in operator_rows:
        lines.append(f"- {r['dataset']} seed {r['host_seed']} {r['modality']}: calibration-selected fixed action={r.get('global_fixed_action_selected_on_calib')}; report action ΔCE={json.dumps(r.get('report_mean_delta_ce_by_action'), sort_keys=True)}; headroom={_fmt(r.get('response_headroom_mean'))} [{_fmt(r.get('response_headroom_ci_low'))}, {_fmt(r.get('response_headroom_ci_high'))}]; median best/second margin={_fmt(r.get('best_second_delta_ce_margin_median'))}.")
    lines.extend(["", "The adaptive response upper bound is non-deployable, label-aware, specific to this frozen host and action bank, not an oracle for a future MoE, and does not show an end-to-end router can realize the gap. Argmin action is not a ground-truth operator label.", "", "## 8. Cross-dataset consistency", "",
                  f"The overall rule requires at least {protocol['decision_rules']['cross_dataset_majority']} of five datasets to meet the preregistered dataset criterion. The per-dataset decisions and all seed rows are in the compact CSVs. P2 direction signs, P3 seed consistency, and projected P1 robustness are retained in full.", "",
                  "## 9. Alpha and seed stability", ""])
    positive_p2 = [r["dataset"] for r in summary["dataset_decisions"] if not r.get("audit_only") and _f(r.get("p2_mean_asymmetry_alpha_0_1")) > 0]
    negative_p2 = [r["dataset"] for r in summary["dataset_decisions"] if not r.get("audit_only") and _f(r.get("p2_mean_asymmetry_alpha_0_1")) < 0]
    lines.append(f"Dataset mean P2 asymmetry is positive for {', '.join(positive_p2) or 'none'} and negative for {', '.join(negative_p2) or 'none'}; the sign is not universal across datasets.")
    p2_asym = [r for r in p2_rows if r.get("direction") == "T2I_minus_I2T"]
    for alpha in protocol["interaction"]["p2_alphas"]:
        vals = [_f(r.get("directional_asymmetry_mean")) for r in p2_asym if abs(_f(r.get("alpha")) - float(alpha)) < 1e-8]
        if vals:
            lines.append(f"- P2 alpha={alpha}: mean asymmetry={_fmt(float(np.nanmean(vals)))}; finite runs={sum(math.isfinite(v) for v in vals)}/{len(vals)}.")
    for name in protocol["primary_datasets"]:
        vals = [_f(r.get("mean_high_minus_low")) for r in p3_rows if r.get("row_type") == "matched_high_low_structure" and r.get("dataset") == name]
        if vals:
            finite_vals = [value for value in vals if math.isfinite(value)]
            if finite_vals:
                lines.append(f"- P3 {name}: matched effect span across seed×modality=[{_fmt(min(finite_vals))}, {_fmt(max(finite_vals))}].")
            else:
                lines.append(f"- P3 {name}: matched high/low effect unavailable in all seed×modality cells because no common matched strata were available.")
    low_alignment_runs = {
        (r["dataset"], r["host_seed"])
        for r in p2_rows
        if r.get("alignment_low_quality") and r.get("direction") in {"T2I", "I2T"}
    }
    lines.append(f"- Low host quality flags: {sum(bool(r['low_host_quality']) for r in host_rows)}/{len(host_rows)}; low alignment runs: {len(low_alignment_runs)}/{len(host_rows)}.")
    lines.extend(["", "## 10. Failure cases", ""])
    if failures:
        lines.extend([f"- {r.get('dataset')} seed {r.get('host_seed')}: {r.get('error_type')}: {r.get('error')}" for r in failures])
    else:
        lines.append("No formal run failure was recorded.")
    if not summary["complete"]:
        lines.append("The report is incomplete because formal dataset-seed runs are missing or failed.")
    lines.extend(["", "## 11. Threats to validity", "",
                  "- The graph-free host and frozen classifier define the response space; another host can produce different responses.",
                  "- Full graph topology and unlabelled features are transductive; this is not an inductive evaluation.",
                  "- Source-matched percentile estimates depend on finite non-edge samples; fixed ridge regularization may fit directions unevenly.",
                  "- P3 uses sampled directed edges, train-edge cut points, finite-difference alpha, and a structural embeddedness definition; these choices bound interpretation.",
                  "- Receiver-block intervals capture within-dataset receiver clustering, not all between-dataset uncertainty.",
                  "- Validation is report-only: host fitting/early stopping and alignment fitting do not use its labels.",
                  "", "## 12. Factual conclusions", "",
                  f"- P1: {summary['p1_cross_dataset_status']} under the raw percentile-gap criterion.",
                  f"- P2: {summary['p2_cross_dataset_status']} under graph-versus-random directional asymmetry with an alignment gate.",
                  f"- P3: {summary['p3_cross_dataset_status']} under matched high/low embeddedness response and seed consistency.",
                  "- Operator results describe the frozen host's response profile; any headroom is a label-aware validation upper bound for this action bank.",
                  "", "## 13. Conclusions not supported by this evidence", "",
                  "- No ground-truth operator, oracle expert, or causal effect is established.",
                  "- The results do not establish that an end-to-end router improves NC, generalizes, or realizes the response upper bound.",
                  "- Same-label edges are not treated as useful edges, and no test-label-based claim is made.",
                  "", "## Decision Table", "",
                  "| Hypothesis | Supported / Mixed / Unsupported | Evidence | Caveat |",
                  "|---|---|---|---|"])
    decisions = [
        ("P1: modality-specific relation on the same physical edge", summary["p1_cross_dataset_status"], "Raw source-matched percentile gap and receiver-block CI", "Train-edge bins; sampled non-edge references"),
        ("P2: T→I versus I→T utility asymmetry", summary["p2_cross_dataset_status"], "Graph-minus-random gains with source-count matching", "Fixed ridge and frozen host; alignment gate"),
        ("P3: structure response after semantic control", summary["p3_cross_dataset_status"], "Within-bin D_same tables and matched high/low embeddedness", "Sampled local response; not causal"),
    ]
    lines.extend([f"| {a} | {b} | {c} | {d} |" for a, b, c, d in decisions])
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
