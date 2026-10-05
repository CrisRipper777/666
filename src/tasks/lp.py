from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import logging
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import TensorDataset
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader

from src.data import EdgeSplit, MAGData
from src.data.graph_utils import edge_dict_to_index
from src.models import LinkPredictor, build_model
from omegaconf import ListConfig

from src.tasks.common import (
    build_optimizer,
    checkpoint_is_eligible,
    clone_state_dict,
    compute_optional_supervised_aux_loss,
    load_state_dict_cpu,
    resolve_num_neighbors,
    restore_checkpoint_schedule,
    scheduler_step,
)
from src.tasks.inference import infer_all_embeddings, resolve_inference_mode
from src.utils.metrics import format_pct
from src.utils.seeds import set_seed
from src.utils.summary import count_parameters, mean_std


def _uses_graph_encoder(cfg) -> bool:
    return str(cfg.model.name).lower() != "mlp"


def _resolve_lp_training_mode(cfg) -> str:
    mode = str(cfg.task.get("training_mode", "sampled")).strip().lower()
    if mode != "sampled":
        raise ValueError(
            "unified_sampled_lp_v1 requires task.training_mode='sampled'; "
            f"got {mode!r}"
        )
    return mode


def _edge_keys(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    return edge_index[0].long() * int(num_nodes) + edge_index[1].long()


def _all_positive_edge_index(edge_split: EdgeSplit) -> torch.Tensor:
    return torch.cat(
        [
            edge_dict_to_index(edge_split.train),
            edge_dict_to_index(edge_split.valid),
            edge_dict_to_index(edge_split.test),
        ],
        dim=1,
    ).cpu()


def _build_forbidden_edge_keys(
    edge_split: EdgeSplit, num_nodes: int, undirected: bool
) -> torch.Tensor:
    edge_index = _all_positive_edge_index(edge_split)
    if undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return torch.unique(_edge_keys(edge_index, num_nodes).contiguous(), sorted=True)


def _is_forbidden_edge(
    src: torch.Tensor,
    dst: torch.Tensor,
    num_nodes: int,
    forbidden_keys: torch.Tensor,
) -> torch.Tensor:
    keys = src.long() * int(num_nodes) + dst.long()
    positions = torch.searchsorted(forbidden_keys, keys)
    in_bounds = positions < forbidden_keys.numel()
    matches = torch.zeros_like(in_bounds, dtype=torch.bool)
    if bool(in_bounds.any()):
        matches[in_bounds] = forbidden_keys[positions[in_bounds]] == keys[in_bounds]
    return matches


def _sample_filtered_negative_targets(
    src: torch.Tensor,
    num_nodes: int,
    num_neg: int,
    forbidden_keys: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    if num_neg <= 0:
        raise ValueError(f"num_neg must be positive, got {num_neg}")
    src = src.cpu().long().contiguous()
    forbidden_keys = forbidden_keys.cpu().long().contiguous()
    negatives = torch.empty((src.numel(), num_neg), dtype=torch.long)
    row_index = torch.arange(src.numel(), dtype=torch.long)

    for neg_col in range(num_neg):
        pending = row_index
        attempts = 0
        while pending.numel() > 0:
            attempts += 1
            if attempts > 1000:
                raise RuntimeError(
                    "Unable to sample filtered train negatives after 1000 attempts; "
                    "the graph may be too dense for the requested num_train_neg."
                )
            pending_src = src[pending]
            candidate = torch.randint(
                0, num_nodes, (pending.numel(),), generator=generator
            )
            ok = candidate != pending_src
            ok &= ~_is_forbidden_edge(
                pending_src, candidate, num_nodes, forbidden_keys
            )
            if neg_col > 0:
                ok &= ~(
                    negatives[pending, :neg_col] == candidate.view(-1, 1)
                ).any(dim=1)
            accepted = pending[ok]
            if accepted.numel() > 0:
                negatives[accepted, neg_col] = candidate[ok]
            pending = pending[~ok]
    return negatives


def _build_epoch_train_labels(
    train_pos_edge_index: torch.Tensor | EdgeSplit,
    num_nodes: int,
    num_neg: int,
    forbidden_keys: torch.Tensor,
    generator: torch.Generator,
    train_pos_per_epoch: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(train_pos_edge_index, EdgeSplit):
        pos_edge_index = edge_dict_to_index(train_pos_edge_index.train).cpu()
    else:
        pos_edge_index = train_pos_edge_index.cpu().long().contiguous()
    if train_pos_per_epoch is not None and train_pos_per_epoch < pos_edge_index.size(1):
        perm = torch.randperm(
            pos_edge_index.size(1), generator=generator
        )[:train_pos_per_epoch]
        pos_edge_index = pos_edge_index[:, perm]
    pos_src = pos_edge_index[0]
    neg_dst = _sample_filtered_negative_targets(
        pos_src, num_nodes, num_neg, forbidden_keys, generator
    )
    neg_edge_index = torch.stack(
        [pos_src.repeat_interleave(num_neg), neg_dst.reshape(-1)], dim=0
    )
    edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1).contiguous()
    edge_label = torch.cat(
        [
            torch.ones(pos_edge_index.size(1), dtype=torch.float32),
            torch.zeros(neg_edge_index.size(1), dtype=torch.float32),
        ],
        dim=0,
    ).contiguous()
    return edge_label_index, edge_label


def _exclude_positive_label_edges_from_message_graph(
    edge_index: torch.Tensor,
    edge_label_index: torch.Tensor,
    edge_label: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Local-key implementation retained for equivalence audits."""
    positive_mask = edge_label > 0.5
    if not bool(positive_mask.any()):
        return edge_index.contiguous()
    positive_edges = edge_label_index[:, positive_mask].long()
    forbidden_edges = torch.cat([positive_edges, positive_edges.flip(0)], dim=1)
    forbidden_keys = torch.unique(_edge_keys(forbidden_edges, num_nodes), sorted=True)
    edge_keys = _edge_keys(edge_index, num_nodes)
    positions = torch.searchsorted(forbidden_keys, edge_keys)
    in_bounds = positions < forbidden_keys.numel()
    drop_mask = torch.zeros(edge_index.size(1), dtype=torch.bool, device=edge_index.device)
    if bool(in_bounds.any()):
        drop_mask[in_bounds] = forbidden_keys[positions[in_bounds]] == edge_keys[in_bounds]
    return edge_index[:, ~drop_mask].contiguous()


@dataclass
class _MessageEdgeLookup:
    """CPU lookup from a directed global edge code to every matching edge ID."""

    num_nodes: int
    sorted_codes: torch.Tensor
    sorted_edge_ids: torch.Tensor
    removal_scratch: torch.Tensor


def _build_message_edge_lookup(
    message_edge_index: torch.Tensor, num_nodes: int
) -> _MessageEdgeLookup:
    """Build a duplicate-safe lookup used by the official global_eid mask."""
    edge_index = message_edge_index.detach().cpu().long().contiguous()
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(
            "message_edge_index must have shape [2, num_edges], "
            f"got {tuple(edge_index.shape)}"
        )
    if edge_index.numel() > 0 and (
        int(edge_index.min()) < 0 or int(edge_index.max()) >= int(num_nodes)
    ):
        raise ValueError("message_edge_index contains an out-of-range node ID")
    sorted_codes, sorted_edge_ids = _edge_keys(edge_index, num_nodes).sort()
    return _MessageEdgeLookup(
        num_nodes=int(num_nodes),
        sorted_codes=sorted_codes,
        sorted_edge_ids=sorted_edge_ids,
        removal_scratch=torch.zeros(edge_index.size(1), dtype=torch.bool),
    )


def _lookup_all_message_edge_ids(
    lookup: _MessageEdgeLookup, query_codes: torch.Tensor
) -> torch.Tensor:
    query_codes = torch.unique(query_codes.detach().cpu().long())
    if query_codes.numel() == 0 or lookup.sorted_codes.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    starts = torch.searchsorted(lookup.sorted_codes, query_codes, right=False)
    ends = torch.searchsorted(lookup.sorted_codes, query_codes, right=True)
    lengths = ends - starts
    present = lengths > 0
    starts = starts[present]
    lengths = lengths[present]
    if starts.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    group_ids = torch.repeat_interleave(torch.arange(starts.numel()), lengths)
    group_starts = torch.repeat_interleave(lengths.cumsum(dim=0) - lengths, lengths)
    positions = (
        starts[group_ids]
        + torch.arange(int(lengths.sum().item()), dtype=torch.long)
        - group_starts
    )
    return lookup.sorted_edge_ids[positions]


def _exclude_positive_label_edges_by_global_eid(
    batch: Data, message_lookup: _MessageEdgeLookup
) -> int:
    """Remove both orientations of positive supervision edges by global e_id."""
    positive_mask = batch.edge_label > 0.5
    if not bool(positive_mask.any()) or batch.edge_index.numel() == 0:
        return 0
    if not hasattr(batch, "n_id") or not hasattr(batch, "e_id"):
        raise ValueError("global_eid edge masking requires sampled batch n_id and e_id")
    if batch.edge_index.device.type != "cpu" or batch.e_id.device.type != "cpu":
        raise ValueError("global_eid edge masking must run on CPU before batch.to(device)")
    if batch.e_id.dim() != 1 or batch.e_id.numel() != batch.edge_index.size(1):
        raise ValueError(
            "batch.e_id must align one-to-one with batch.edge_index columns; "
            f"got {batch.e_id.numel()} IDs for {batch.edge_index.size(1)} edges"
        )
    edge_ids = batch.e_id.long()
    if edge_ids.numel() > 0 and (
        int(edge_ids.min()) < 0
        or int(edge_ids.max()) >= message_lookup.removal_scratch.numel()
    ):
        raise ValueError("batch.e_id contains an ID outside the global message graph")

    supervision = batch.edge_label_index[:, positive_mask].long()
    global_source = batch.n_id[supervision[0]]
    global_target = batch.n_id[supervision[1]]
    query_codes = torch.cat(
        [
            global_source * message_lookup.num_nodes + global_target,
            global_target * message_lookup.num_nodes + global_source,
        ]
    )
    removal_ids = _lookup_all_message_edge_ids(message_lookup, query_codes)
    if removal_ids.numel() == 0:
        return 0
    message_lookup.removal_scratch[removal_ids] = True
    try:
        keep = ~message_lookup.removal_scratch[edge_ids]
        removed = int((~keep).sum())
        batch.edge_index = batch.edge_index[:, keep].contiguous()
        batch.e_id = batch.e_id[keep].contiguous()
    finally:
        message_lookup.removal_scratch[removal_ids] = False
    return removed


def _seed_neighbor_worker(worker_id: int, base_seed: int) -> None:
    worker_seed = (int(base_seed) + int(worker_id)) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _resolve_lp_num_neighbors(cfg, model: nn.Module | None = None) -> list[int]:
    """Resolve the frozen sampler fanouts, independent of unrelated model iterations.

    The formal LP protocol fixes three hops. Models that explicitly require
    sampler depth to equal encoder depth opt in through the
    requires_full_lp_sampler_depth attribute.
    """
    if bool(getattr(model, "requires_full_lp_sampler_depth", False)):
        return resolve_num_neighbors(cfg)
    raw = cfg.task.get("num_neighbors", [5, 5, 5])
    if isinstance(raw, str):
        raw = raw.strip()
        if raw.startswith("[") and raw.endswith("]"):
            values = [int(item.strip()) for item in raw[1:-1].split(",") if item.strip()]
        else:
            values = [int(raw)]
    elif isinstance(raw, (list, tuple, ListConfig)):
        values = [int(value) for value in raw]
    else:
        values = [int(raw)]
    if len(values) != 3:
        raise ValueError("unified_sampled_lp_v1 fixes three num_neighbors fanouts")
    return values


def _build_link_loader(
    cfg,
    pyg_data: Data,
    edge_label_index: torch.Tensor,
    edge_label: torch.Tensor,
    batch_generator: torch.Generator,
    neighbor_seed: int,
    model: nn.Module | None = None,
) -> LinkNeighborLoader:
    num_workers = int(cfg.task.get("loader_num_workers", 0))
    kwargs = {}
    if num_workers > 0:
        kwargs["worker_init_fn"] = partial(
            _seed_neighbor_worker, base_seed=int(neighbor_seed)
        )
        kwargs["prefetch_factor"] = int(cfg.task.get("loader_prefetch_factor", 2))
    return LinkNeighborLoader(
        pyg_data,
        num_neighbors=_resolve_lp_num_neighbors(cfg, model),
        batch_size=int(cfg.task.batch_size),
        shuffle=True,
        subgraph_type=str(cfg.task.get("subgraph_type", "bidirectional")),
        num_workers=num_workers,
        generator=batch_generator,
        edge_label_index=edge_label_index,
        edge_label=edge_label,
        **kwargs,
    )


def _build_edge_loader(
    cfg,
    edge_label_index: torch.Tensor,
    edge_label: torch.Tensor,
    generator: torch.Generator,
) -> TorchDataLoader:
    return TorchDataLoader(
        TensorDataset(edge_label_index.t().contiguous(), edge_label.contiguous()),
        batch_size=int(cfg.task.batch_size),
        shuffle=True,
        generator=generator,
    )


def _raise_if_nonfinite(value: torch.Tensor, label: str) -> None:
    if not bool(torch.isfinite(value).all()):
        finite_count = int(torch.isfinite(value).sum().item())
        raise FloatingPointError(
            f"Non-finite {label}: {finite_count}/{value.numel()} values are finite"
        )


def _prepare_eval_embeddings(
    z: torch.Tensor, device: torch.device, preload: bool, logger: logging.Logger
) -> torch.Tensor:
    z_eval = z.to(device, non_blocking=True) if preload else z
    logger.info(
        "Eval node embeddings: preload=%s | device=%s | shape=%s",
        preload,
        z_eval.device,
        tuple(z_eval.shape),
    )
    return z_eval


@torch.no_grad()
def _evaluate_split(
    z: torch.Tensor,
    predictor: LinkPredictor,
    split: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    predictor.eval()
    src_all = split["source_node"]
    dst_all = split["target_node"]
    neg_all = split["target_node_neg"]
    total = int(src_all.numel())
    mrr_sum = hits1_sum = hits3_sum = hits10_sum = 0.0

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        src = src_all[start:end].to(z.device, non_blocking=True).long()
        dst = dst_all[start:end].to(z.device, non_blocking=True).long()
        neg = neg_all[start:end].to(z.device, non_blocking=True).long()
        pos_src = z[src]
        pos_dst = z[dst]
        neg_dst = z[neg]
        pos_score = predictor.score_pairs(pos_src.to(device), pos_dst.to(device))
        neg_pair_feature = pos_src.unsqueeze(1) * neg_dst
        neg_score = predictor(neg_pair_feature.to(device)).view(neg.size(0), neg.size(1))
        _raise_if_nonfinite(pos_score, "LP evaluation positive scores")
        _raise_if_nonfinite(neg_score, "LP evaluation negative scores")
        # V3 uses pessimistic ties: an equal-scoring negative ranks before positive.
        ranks = 1.0 + (neg_score >= pos_score.view(-1, 1)).sum(dim=1).float()
        mrr_sum += float((1.0 / ranks).sum().item())
        hits1_sum += float((ranks <= 1).float().sum().item())
        hits3_sum += float((ranks <= 3).float().sum().item())
        hits10_sum += float((ranks <= 10).float().sum().item())

    denom = max(total, 1)
    return {
        "mrr": mrr_sum / denom,
        "hits@1": hits1_sum / denom,
        "hits@3": hits3_sum / denom,
        "hits@10": hits10_sum / denom,
    }



def _epoch_random_seeds(cfg, run_seed: int, epoch: int) -> tuple[int, int, int]:
    """Return independent negative, shuffle, and neighbor-worker seed streams."""
    return (
        int(run_seed) + int(cfg.task.get("negative_seed_offset", 10_000)) + int(epoch),
        int(run_seed) + int(cfg.task.get("shuffle_seed_offset", 20_000)) + int(epoch),
        int(run_seed) + int(cfg.task.get("neighbor_seed_offset", 30_000)) + int(epoch),
    )


def _checkpoint_path_for_run(path_like: str | Path, cfg, run_id: int) -> Path:
    path = Path(str(path_like))
    if int(cfg.num_runs) > 1:
        path = path.with_name(f"{path.stem}_run{run_id + 1}{path.suffix}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _run_single_lp(
    cfg,
    data: MAGData,
    device: torch.device,
    logger: logging.Logger,
    run_id: int,
    seed: int,
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
    proj_dim = int(cfg.task.decoder.get("proj_dim", 0) or 0)
    projection = nn.Linear(model.out_dim, proj_dim).to(device) if proj_dim > 0 else None
    predictor_in_dim = proj_dim if projection is not None else model.out_dim
    predictor = LinkPredictor(
        in_dim=predictor_in_dim,
        hidden_dim=int(cfg.task.decoder.hidden_dim),
        num_layers=int(cfg.task.decoder.num_layers),
        dropout=float(cfg.task.decoder.dropout),
    ).to(device)
    optimizer_parameters = list(model.parameters()) + list(predictor.parameters())
    if projection is not None:
        optimizer_parameters += list(projection.parameters())
    optimizer = build_optimizer(optimizer_parameters, cfg, model=model)
    criterion = nn.BCEWithLogitsLoss()
    uses_graph = _uses_graph_encoder(cfg)
    _resolve_lp_training_mode(cfg)
    x_all = data.x.to(device) if not uses_graph else None
    pyg_data = Data(x=data.x, edge_index=data.edge_index) if uses_graph else None
    undirected = bool(
        data.edge_split.metadata.get(
            "undirected", cfg.dataset.get("make_undirected", True)
        )
    )
    forbidden_keys = _build_forbidden_edge_keys(
        data.edge_split, data.num_nodes, undirected=undirected
    )
    train_pos = edge_dict_to_index(data.edge_split.train).cpu()
    train_pos_per_epoch = cfg.task.get("train_pos_per_epoch")
    if train_pos_per_epoch is not None:
        train_pos_per_epoch = int(train_pos_per_epoch)
    mask_backend = str(
        cfg.task.get("positive_edge_mask_backend", "global_eid")
    ).strip().lower()
    if mask_backend not in {"global_eid", "local_keys"}:
        raise ValueError(
            "task.positive_edge_mask_backend must be 'global_eid' or 'local_keys'; "
            f"got {mask_backend!r}"
        )
    if uses_graph and mask_backend == "global_eid":
        message_lookup = _build_message_edge_lookup(data.edge_index, data.num_nodes)
    else:
        message_lookup = None

    logger.info(
        "[Run %d/%d] seed=%d | model+decoder params=%d",
        run_id + 1,
        int(cfg.num_runs),
        seed,
        count_parameters(model)
        + count_parameters(predictor)
        + (count_parameters(projection) if projection is not None else 0),
    )
    logger.info("Protocol: %s", str(cfg.task.protocol_version))
    logger.info("LP training mode: sampled")
    logger.info("Loader: %s", "LinkNeighborLoader" if uses_graph else "EdgeLabelDataLoader")
    if uses_graph:
        logger.info("Train neighbor sampling fanouts: %s", _resolve_lp_num_neighbors(cfg, model))
        logger.info("Positive message-edge masking: %s", mask_backend)
    logger.info("Inference mode: %s", inference_mode)

    best_val = -1.0
    best_metrics: dict[str, float] = {}
    best_model_state = None
    best_predictor_state = None
    best_projection_state = None
    best_epoch: int | None = None
    patience_total = int(cfg.task.patience)
    patience_left = patience_total
    min_epoch = int(cfg.task.get("early_stop_min_epoch", 1))
    min_delta = float(cfg.task.get("early_stop_min_delta", 0.0))
    grad_clip = float(cfg.task.get("grad_clip", 1.0))
    aux_weight = float(cfg.task.loss.aux_weight)
    max_train_batches = cfg.task.get("max_train_batches")
    inference_batch_size = int(cfg.task.inference_batch_size)
    eval_preload = bool(cfg.task.get("eval_preload_node_emb", False))
    eval_every = int(cfg.task.eval_every)
    if eval_every < 1:
        raise ValueError("task.eval_every must be >= 1")

    for epoch in range(1, int(cfg.task.epochs) + 1):
        model.train()
        predictor.train()
        if projection is not None:
            projection.train()
        if hasattr(model, "set_epoch"):
            model.set_epoch(epoch)
        total_loss = 0.0
        total_examples = 0
        removed_edges = 0
        neg_seed, shuffle_seed, neighbor_seed = _epoch_random_seeds(
            cfg, seed, epoch
        )
        neg_generator = torch.Generator().manual_seed(neg_seed)
        batch_generator = torch.Generator().manual_seed(shuffle_seed)
        edge_label_index, edge_label = _build_epoch_train_labels(
            train_pos,
            data.num_nodes,
            int(cfg.task.num_train_neg),
            forbidden_keys,
            neg_generator,
            train_pos_per_epoch=train_pos_per_epoch,
        )
        if uses_graph:
            loader = _build_link_loader(
                cfg,
                pyg_data,
                edge_label_index,
                edge_label,
                batch_generator,
                neighbor_seed,
                model=model,
            )
        else:
            loader = _build_edge_loader(
                cfg, edge_label_index, edge_label, batch_generator
            )

        for step, batch in enumerate(loader):
            if max_train_batches is not None and step >= int(max_train_batches):
                break
            optimizer.zero_grad(set_to_none=True)
            if uses_graph:
                if message_lookup is not None:
                    removed_edges += _exclude_positive_label_edges_by_global_eid(
                        batch, message_lookup
                    )
                else:
                    old_edge_count = int(batch.edge_index.size(1))
                    batch.edge_index = _exclude_positive_label_edges_from_message_graph(
                        batch.edge_index,
                        batch.edge_label_index,
                        batch.edge_label,
                        num_nodes=int(batch.x.size(0)),
                    )
                    removed_edges += old_edge_count - int(batch.edge_index.size(1))
                batch = batch.to(device)
                if hasattr(model, "_batch_n_id"):
                    model._batch_n_id = batch.n_id
                z, _, _, aux_loss, aux_info = model(batch.x, batch.edge_index)
                if projection is not None:
                    z = projection(z)
                src, dst = batch.edge_label_index
                logits = predictor.score_pairs(z[src], z[dst])
                labels = batch.edge_label.float()
                supervised_aux = compute_optional_supervised_aux_loss(
                    model,
                    task="lp",
                    aux_info=aux_info,
                    edge_label_index=batch.edge_label_index,
                    edge_label=labels,
                )
            else:
                edges, labels = batch
                edges = edges.to(device)
                labels = labels.to(device)
                z_all, _, _, aux_loss, aux_info = model(x_all[edges.reshape(-1)], None)
                if projection is not None:
                    z_all = projection(z_all)
                z_pairs = z_all.view(edges.size(0), 2, -1)
                logits = predictor.score_pairs(z_pairs[:, 0], z_pairs[:, 1])
                supervised_aux = compute_optional_supervised_aux_loss(
                    model,
                    task="lp",
                    aux_info=aux_info,
                    edge_label_index=torch.arange(edges.size(0) * 2, device=edges.device).reshape(-1, 2).T,
                    edge_label=labels,
                )

            _raise_if_nonfinite(logits, f"LP logits at epoch {epoch} batch {step}")
            loss = criterion(logits, labels) + aux_weight * (aux_loss + supervised_aux)
            _raise_if_nonfinite(loss, f"LP loss at epoch {epoch} batch {step}")
            loss.backward()
            clip_parameters = list(model.parameters()) + list(predictor.parameters())
            if projection is not None:
                clip_parameters += list(projection.parameters())
            torch.nn.utils.clip_grad_norm_(
                clip_parameters, max_norm=grad_clip, error_if_nonfinite=True
            )
            optimizer.step()
            total_loss += float(loss.detach().item()) * int(labels.numel())
            total_examples += int(labels.numel())
            if hasattr(model, "_batch_n_id"):
                model._batch_n_id = None

        train_loss = total_loss / max(total_examples, 1)
        scheduler_step(cfg, optimizer, epoch, int(cfg.task.epochs))
        if epoch % eval_every != 0:
            logger.info(
                "Epoch %05d | Train Loss %.4f | Positive Message Edges Removed %d",
                epoch,
                train_loss,
                removed_edges,
            )
            continue

        z = infer_all_embeddings(
            model, data, device, uses_graph, inference_batch_size, inference_mode
        )
        if projection is not None:
            projection.eval()
            z = projection(z.to(device)).detach().cpu()
        z_eval = _prepare_eval_embeddings(z, device, eval_preload, logger)
        val_metrics = _evaluate_split(
            z_eval,
            predictor,
            data.edge_split.valid,
            device,
            int(cfg.task.eval_edge_batch_size),
        )
        logger.info(
            "Epoch %05d | Train Loss %.4f | Positive Message Edges Removed %d | "
            "Val MRR %.2f | H@1 %.2f | H@3 %.2f | H@10 %.2f",
            epoch,
            train_loss,
            removed_edges,
            format_pct(val_metrics["mrr"]),
            format_pct(val_metrics["hits@1"]),
            format_pct(val_metrics["hits@3"]),
            format_pct(val_metrics["hits@10"]),
        )

        checkpoint_eligible = checkpoint_is_eligible(model, epoch)
        improved = checkpoint_eligible and val_metrics["mrr"] > best_val + min_delta
        stop_early = False
        if not checkpoint_eligible:
            logger.info("Validation checkpoint selection deferred by model schedule at epoch %05d", epoch)
        elif improved:
            best_val = val_metrics["mrr"]
            best_epoch = epoch
            best_metrics = {"val_mrr": val_metrics["mrr"]}
            best_model_state = clone_state_dict(model)
            best_predictor_state = clone_state_dict(predictor)
            best_projection_state = (
                clone_state_dict(projection) if projection is not None else None
            )
            patience_left = patience_total
        elif epoch >= min_epoch:
            patience_left -= 1
            logger.info(
                "Patience %d/%d | Best Val MRR %.2f",
                patience_total - patience_left,
                patience_total,
                format_pct(best_val),
            )
            stop_early = patience_left <= 0
        del z_eval
        del z
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if stop_early:
            logger.info("Early stopping at epoch %03d", epoch)
            break

    if best_model_state is None or best_predictor_state is None:
        raise RuntimeError(
            "LP training finished without a validation checkpoint; "
            "ensure at least one epoch is evaluated"
        )
    load_state_dict_cpu(model, best_model_state)
    load_state_dict_cpu(predictor, best_predictor_state)
    if projection is not None and best_projection_state is not None:
        load_state_dict_cpu(projection, best_projection_state)
    restore_checkpoint_schedule(model, int(best_epoch))

    if bool(cfg.task.get("evaluate_test", True)):
        z = infer_all_embeddings(
            model, data, device, uses_graph, inference_batch_size, inference_mode
        )
        if projection is not None:
            projection.eval()
            z = projection(z.to(device)).detach().cpu()
        z_eval = _prepare_eval_embeddings(z, device, eval_preload, logger)
        test_metrics = _evaluate_split(
            z_eval,
            predictor,
            data.edge_split.test,
            device,
            int(cfg.task.eval_edge_batch_size),
        )
        best_metrics.update(
            {
                "test_mrr": test_metrics["mrr"],
                "test_hits@1": test_metrics["hits@1"],
                "test_hits@3": test_metrics["hits@3"],
                "test_hits@10": test_metrics["hits@10"],
            }
        )
        del z_eval
        del z

    save_ckpt_path = cfg.task.get("save_ckpt_path")
    if save_ckpt_path:
        path = _checkpoint_path_for_run(save_ckpt_path, cfg, run_id)
        torch.save(
            {
                "task": "lp",
                "protocol_version": str(cfg.task.protocol_version),
                "seed": seed,
                "selection": "best_val_mrr",
                "epoch": best_epoch,
                "metrics": dict(best_metrics),
                "model_state": clone_state_dict(model),
                "head_state": clone_state_dict(predictor),
                "proj_state": (
                    clone_state_dict(projection) if projection is not None else None
                ),
                "data_info": data_info,
            },
            path,
        )
        logger.info("Saved checkpoint: %s | best_epoch=%s", path, best_epoch)

    logger.info(
        "[Run %d] Best Val MRR %.2f | epoch=%d",
        run_id + 1,
        format_pct(best_metrics["val_mrr"]),
        int(best_epoch),
    )
    if "test_mrr" in best_metrics:
        logger.info(
            "[Run %d] Test MRR %.2f | H@1 %.2f | H@3 %.2f | H@10 %.2f",
            run_id + 1,
            format_pct(best_metrics["test_mrr"]),
            format_pct(best_metrics["test_hits@1"]),
            format_pct(best_metrics["test_hits@3"]),
            format_pct(best_metrics["test_hits@10"]),
        )
    return best_metrics


def run_lp(
    cfg,
    data: MAGData,
    device: torch.device,
    logger: logging.Logger,
) -> dict[str, tuple[float, float]]:
    if data.edge_split is None:
        raise ValueError("LP data must contain edge_split")
    _resolve_lp_training_mode(cfg)
    run_results = []
    for run_id in range(int(cfg.num_runs)):
        seed = int(cfg.seed) + run_id
        set_seed(seed)
        run_results.append(
            _run_single_lp(cfg, data, device, logger, run_id, seed)
        )

    output: dict[str, tuple[float, float]] = {}
    names = {
        "val_mrr": "Val MRR",
        "test_mrr": "Test MRR",
        "test_hits@1": "Hits@1",
        "test_hits@3": "Hits@3",
        "test_hits@10": "Hits@10",
    }
    keys = ["val_mrr"]
    if bool(cfg.task.get("evaluate_test", True)):
        keys.extend(["test_mrr", "test_hits@1", "test_hits@3", "test_hits@10"])
    logger.info("============================================================")
    logger.info("Final Results over %d runs", int(cfg.num_runs))
    for key in keys:
        mean, std = mean_std([item[key] for item in run_results])
        output[key] = (mean, std)
        logger.info("%s: %.2f ± %.2f", names[key], format_pct(mean), format_pct(std))
    logger.info("============================================================")
    return output
