from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader as TorchDataLoader

from src.data import MAGData
from src.models import build_model
from src.tasks.common import (
    build_optimizer,
    checkpoint_is_eligible,
    clone_state_dict,
    compute_optional_supervised_aux_loss,
    load_state_dict_cpu,
    restore_checkpoint_schedule,
    scheduler_step,
)
from src.tasks.inference import infer_all_embeddings, resolve_inference_mode
from src.utils.metrics import format_pct
from src.utils.seeds import set_seed
from src.utils.summary import count_parameters, mean_std


def _uses_graph_encoder(cfg) -> bool:
    return str(cfg.model.name).lower() != "mlp"


def _resolve_training_mode(cfg, model=None) -> str:
    """Resolve the frozen NC path independently of model capability flags."""
    mode = str(cfg.task.get("training_mode", "full_graph")).strip().lower()
    if mode != "full_graph":
        raise ValueError(
            "unified_full_graph_nc_v1 requires task.training_mode='full_graph'; "
            f"got {mode!r}"
        )
    # MLP has no graph computation; it uses feature-only node minibatches.
    # Graph models always use one full graph forward, even if a model preset
    # advertises full_graph_training=False.
    return "full_graph" if _uses_graph_encoder(cfg) else "feature_only"


def _resolve_nc_eval_labels(data: MAGData) -> list[int]:
    """Use one stable Macro-F1 label set across the supervised task splits."""
    if data.y is None:
        raise ValueError("NC data must contain labels")
    if data.num_classes is None:
        raise ValueError("NC data must define num_classes")
    split_indices = [
        idx for idx in (data.train_idx, data.val_idx, data.test_idx) if idx is not None
    ]
    if not split_indices:
        raise ValueError("NC data must contain at least one supervised split")
    all_indices = torch.cat([idx.reshape(-1) for idx in split_indices]).to(data.y.device)
    observed = data.y[all_indices].detach().cpu()
    valid = observed[(observed >= 0) & (observed < int(data.num_classes))]
    labels = sorted({int(value) for value in valid.tolist()})
    if not labels:
        raise ValueError("NC supervised splits contain no valid class labels")
    return labels


@torch.no_grad()
def _evaluate_split(
    classifier: nn.Module,
    z: torch.Tensor,
    labels: torch.Tensor,
    idx: torch.Tensor,
    device: torch.device,
    batch_size: int,
    eval_labels: list[int] | tuple[int, ...],
) -> dict[str, float]:
    classifier.eval()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for batch_idx in TorchDataLoader(idx.cpu(), batch_size=batch_size, shuffle=False):
        logits = classifier(z[batch_idx].to(device))
        preds.append(logits.argmax(dim=-1).cpu())
        targets.append(labels[batch_idx].cpu())
    pred = torch.cat(preds, dim=0)
    target = torch.cat(targets, dim=0)
    return {
        "acc": float((pred == target).float().mean().item()),
        "macro_f1": float(
            f1_score(
                target.numpy(),
                pred.numpy(),
                labels=list(eval_labels),
                average="macro",
                zero_division=0,
            )
        ),
    }


def _checkpoint_path_for_run(path_like: str | Path, cfg, run_id: int) -> Path:
    path = Path(str(path_like))
    if int(cfg.num_runs) > 1:
        path = path.with_name(f"{path.stem}_run{run_id + 1}{path.suffix}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _run_single_nc(
    cfg,
    data: MAGData,
    device: torch.device,
    logger: logging.Logger,
    run_id: int,
    seed: int,
    eval_labels: list[int],
) -> dict[str, float]:
    inference_mode = resolve_inference_mode(cfg)
    data_info = {
        "input_dim": data.input_dim,
        "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]) if data.x_t is not None else 0,
        "visual_dim": int(data.x_i.shape[1]) if data.x_i is not None else 0,
    }
    model = build_model(cfg, data_info).to(device)
    classifier = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    optimizer = build_optimizer(
        list(model.parameters()) + list(classifier.parameters()), cfg, model=model
    )
    criterion = nn.CrossEntropyLoss()
    training_mode = _resolve_training_mode(cfg, model)
    uses_graph = _uses_graph_encoder(cfg)
    full_graph_training = training_mode == "full_graph"

    # Graph encoders use the entire graph every epoch. Only MLP receives
    # feature-only minibatches.
    x_all = data.x.to(device)
    y_all = data.y.to(device)
    edge_index_all = data.edge_index.to(device) if uses_graph else None
    train_idx_all = data.train_idx.to(device)
    train_loader = None
    if not uses_graph:
        train_loader = TorchDataLoader(
            data.train_idx.cpu(),
            batch_size=int(cfg.task.batch_size),
            shuffle=True,
        )

    logger.info(
        "[Run %d/%d] seed=%d | model params=%d",
        run_id + 1,
        int(cfg.num_runs),
        seed,
        count_parameters(model) + count_parameters(classifier),
    )
    logger.info("Protocol: %s", str(cfg.task.protocol_version))
    logger.info("Training mode: %s", training_mode)
    logger.info("Loader: %s", "FullGraph" if uses_graph else "NodeDataLoader")
    logger.info("Inference mode: %s", inference_mode)
    logger.info(
        "Model parameters=%d | Classifier parameters=%d | optimizer=%s | groups=%s | hidden_dim=%s | num_layers=%s",
        count_parameters(model),
        count_parameters(classifier),
        type(optimizer).__name__,
        [
            {"lr": float(group["lr"]), "weight_decay": float(group["weight_decay"])}
            for group in optimizer.param_groups
        ],
        cfg.model.get("hidden_dim", None),
        cfg.model.get("num_layers", None),
    )

    best_val = -1.0
    best_metrics: dict[str, float] = {}
    best_model_state = None
    best_head_state = None
    best_epoch: int | None = None
    patience_total = int(cfg.task.patience)
    patience_left = patience_total
    min_epoch = int(cfg.task.get("early_stop_min_epoch", 1))
    min_delta = float(cfg.task.get("early_stop_min_delta", 0.0))
    grad_clip = float(cfg.task.get("grad_clip", 1.0))
    aux_weight = float(cfg.task.loss.aux_weight)
    inference_batch_size = int(cfg.task.inference_batch_size)
    eval_every = int(cfg.task.eval_every)
    if eval_every < 1:
        raise ValueError("task.eval_every must be >= 1")

    for epoch in range(1, int(cfg.task.epochs) + 1):
        model.train()
        classifier.train()
        if hasattr(model, "set_epoch"):
            model.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        if full_graph_training:
            if hasattr(model, "_batch_n_id"):
                model._batch_n_id = None
            z, _, _, aux_loss, aux_info = model(x_all, edge_index_all)
            logits = classifier(z[train_idx_all])
            labels = y_all[train_idx_all]
            supervised_aux = compute_optional_supervised_aux_loss(
                model,
                task="nc",
                aux_info=aux_info,
                labels=labels,
                node_index=train_idx_all,
            )
            loss = nn.functional.cross_entropy(logits, labels) + aux_weight * (aux_loss + supervised_aux)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(classifier.parameters()),
                max_norm=grad_clip,
                error_if_nonfinite=True,
            )
            optimizer.step()
            train_loss = float(loss.detach().item())
            del z
        else:
            total_loss = 0.0
            total_examples = 0
            for batch_idx in train_loader:
                batch_idx = batch_idx.to(device)
                labels = y_all[batch_idx]
                z, _, _, aux_loss, aux_info = model(x_all[batch_idx], None)
                logits = classifier(z)
                supervised_aux = compute_optional_supervised_aux_loss(
                    model,
                    task="nc",
                    aux_info=aux_info,
                    labels=labels,
                    node_index=torch.arange(batch_idx.numel(), device=batch_idx.device),
                )
                loss = nn.functional.cross_entropy(logits, labels) + aux_weight * (aux_loss + supervised_aux)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(classifier.parameters()),
                    max_norm=grad_clip,
                    error_if_nonfinite=True,
                )
                optimizer.step()
                total_loss += float(loss.detach().item()) * int(labels.numel())
                total_examples += int(labels.numel())
            train_loss = total_loss / max(total_examples, 1)

        scheduler_step(cfg, optimizer, epoch, int(cfg.task.epochs))
        if epoch % eval_every != 0:
            logger.info("Epoch %05d | Train Loss %.4f", epoch, train_loss)
            continue

        z = infer_all_embeddings(
            model, data, device, uses_graph, inference_batch_size, inference_mode
        )
        val_metrics = _evaluate_split(
            classifier,
            z,
            data.y,
            data.val_idx,
            device,
            inference_batch_size,
            eval_labels,
        )
        logger.info(
            "Epoch %05d | Train Loss %.4f | Val Acc %.2f | Val Macro-F1 %.2f",
            epoch,
            train_loss,
            format_pct(val_metrics["acc"]),
            format_pct(val_metrics["macro_f1"]),
        )

        checkpoint_eligible = checkpoint_is_eligible(model, epoch)
        improved = checkpoint_eligible and val_metrics["acc"] > best_val + min_delta
        stop_early = False
        if not checkpoint_eligible:
            logger.info("Validation checkpoint selection deferred by model schedule at epoch %05d", epoch)
        elif improved:
            best_val = val_metrics["acc"]
            best_epoch = epoch
            best_metrics = {
                "val_acc": val_metrics["acc"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
            best_model_state = clone_state_dict(model)
            best_head_state = clone_state_dict(classifier)
            patience_left = patience_total
        elif epoch >= min_epoch:
            patience_left -= 1
            logger.info(
                "Patience %d/%d | Best Val Acc %.2f",
                patience_total - patience_left,
                patience_total,
                format_pct(best_val),
            )
            stop_early = patience_left <= 0
        del z
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if stop_early:
            logger.info("Early stopping at epoch %03d", epoch)
            break

    if best_model_state is None or best_head_state is None:
        raise RuntimeError("NC training finished without a validation-selected checkpoint")
    load_state_dict_cpu(model, best_model_state)
    load_state_dict_cpu(classifier, best_head_state)
    restore_checkpoint_schedule(model, int(best_epoch))

    if bool(cfg.task.get("evaluate_test", True)):
        z = infer_all_embeddings(
            model, data, device, uses_graph, inference_batch_size, inference_mode
        )
        test_metrics = _evaluate_split(
            classifier,
            z,
            data.y,
            data.test_idx,
            device,
            inference_batch_size,
            eval_labels,
        )
        best_metrics["test_acc"] = test_metrics["acc"]
        best_metrics["test_macro_f1"] = test_metrics["macro_f1"]
        del z

    run_metadata = {
        "best_epoch": int(best_epoch),
        "model_parameters": count_parameters(model),
        "classifier_parameters": count_parameters(classifier),
        "optimizer": type(optimizer).__name__,
        "optimizer_groups": [
            {"lr": float(group["lr"]), "weight_decay": float(group["weight_decay"])}
            for group in optimizer.param_groups
        ],
        "hidden_dim": cfg.model.get("hidden_dim", None),
        "num_layers": cfg.model.get("num_layers", None),
        "other_depth_fields": {
            name: cfg.model.get(name)
            for name in ("d_model", "q_dim", "mp_hops")
            if cfg.model.get(name) is not None
        },
    }
    save_ckpt_path = cfg.task.get("save_ckpt_path")
    if save_ckpt_path:
        path = _checkpoint_path_for_run(save_ckpt_path, cfg, run_id)
        torch.save(
            {
                "task": "nc",
                "protocol_version": str(cfg.task.protocol_version),
                "seed": seed,
                "selection": "best_val_accuracy",
                "epoch": best_epoch,
                "metrics": dict(best_metrics),
                "run_metadata": run_metadata,
                "model_state": clone_state_dict(model),
                "head_state": clone_state_dict(classifier),
                "data_info": data_info,
            },
            path,
        )
        logger.info("Saved checkpoint: %s | best_epoch=%s", path, best_epoch)

    logger.info(
        "[Run %d] Best Val Acc %.2f | Val Macro-F1 %.2f | epoch=%d",
        run_id + 1,
        format_pct(best_metrics["val_acc"]),
        format_pct(best_metrics["val_macro_f1"]),
        int(best_epoch),
    )
    if "test_acc" in best_metrics:
        logger.info(
            "[Run %d] Test Acc %.2f | Test Macro-F1 %.2f",
            run_id + 1,
            format_pct(best_metrics["test_acc"]),
            format_pct(best_metrics["test_macro_f1"]),
        )
    return {**best_metrics, "_run_metadata": run_metadata}


def run_nc(
    cfg,
    data: MAGData,
    device: torch.device,
    logger: logging.Logger,
) -> dict[str, tuple[float, float]]:
    if data.y is None or data.train_idx is None or data.val_idx is None or data.test_idx is None:
        raise ValueError("NC data must contain y/train_idx/val_idx/test_idx")
    _resolve_training_mode(cfg)
    eval_labels = _resolve_nc_eval_labels(data)

    run_results = []
    for run_id in range(int(cfg.num_runs)):
        seed = int(cfg.seed) + run_id
        set_seed(seed)
        run_results.append(
            _run_single_nc(cfg, data, device, logger, run_id, seed, eval_labels)
        )

    run_metrics_path = cfg.task.get("run_metrics_path")
    if run_metrics_path:
        path = Path(str(run_metrics_path)).expanduser()
        if not path.is_absolute():
            path = path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "protocol_version": str(cfg.task.get("protocol_version", "")),
                    "base_seed": int(cfg.seed),
                    "run_seeds": [int(cfg.seed) + run_id for run_id in range(int(cfg.num_runs))],
                    "aggregation": "mean ± population std (ddof=0)",
                    "runs": [
                        {
                            "run_id": run_id,
                            "seed": int(cfg.seed) + run_id,
                            "metrics": {
                                key: float(value)
                                for key, value in item.items()
                                if not key.startswith("_")
                            },
                            "metadata": item.get("_run_metadata", {}),
                        }
                        for run_id, item in enumerate(run_results)
                    ],
                },
                handle,
                indent=2,
            )
    output: dict[str, tuple[float, float]] = {}
    keys = ["val_acc", "val_macro_f1"]
    if bool(cfg.task.get("evaluate_test", True)):
        keys.extend(["test_acc", "test_macro_f1"])
    names = {
        "val_acc": "Val Accuracy",
        "val_macro_f1": "Val Macro-F1",
        "test_acc": "Test Accuracy",
        "test_macro_f1": "Test Macro-F1",
    }
    logger.info("============================================================")
    logger.info("Final Results over %d runs", int(cfg.num_runs))
    for key in keys:
        mean, std = mean_std([item[key] for item in run_results])
        output[key] = (mean, std)
        logger.info("%s: %.2f ± %.2f", names[key], format_pct(mean), format_pct(std))
    logger.info("============================================================")
    return output
