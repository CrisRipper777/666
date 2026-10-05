from __future__ import annotations

import copy
import math

import torch
from omegaconf import ListConfig


def clone_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def load_state_dict_cpu(module: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    module.load_state_dict(copy.deepcopy(state_dict))


def compute_optional_supervised_aux_loss(
    model: torch.nn.Module,
    *,
    task: str,
    aux_info: dict,
    **kwargs,
) -> torch.Tensor:
    """Run a model's optional task-supervised hook, defaulting to exact zero."""
    hook = getattr(model, "compute_supervised_aux_loss", None)
    if not callable(hook):
        parameter = next(model.parameters(), None)
        return parameter.new_zeros(()) if parameter is not None else torch.zeros(())
    result = hook(task=task, aux_info=aux_info, **kwargs)
    loss = result.get("loss") if isinstance(result, dict) else result
    if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
        raise TypeError("compute_supervised_aux_loss must return a scalar tensor or dict with scalar 'loss'")
    return loss.reshape(())


def checkpoint_is_eligible(model: torch.nn.Module, epoch: int) -> bool:
    """Preserve baseline checkpoint behavior unless a model opts into a schedule."""
    hook = getattr(model, "checkpoint_eligible", None)
    return bool(hook(int(epoch))) if callable(hook) else True


def restore_checkpoint_schedule(model: torch.nn.Module, epoch: int) -> None:
    """Restore model-specific non-parameter schedule state after best-weight load."""
    hook = getattr(model, "restore_checkpoint_epoch", None)
    if callable(hook):
        hook(int(epoch))


def build_optimizer(parameters, cfg, model=None) -> torch.optim.Optimizer:
    """Build a task optimizer while honoring an imported model preset."""
    name = str(cfg.task.get("optimizer", "adamw")).strip().lower()
    lr = float(cfg.model.get("lr", cfg.task.lr))
    weight_decay = float(cfg.model.get("weight_decay", cfg.task.weight_decay))
    parameter_list = list(parameters)
    optimizer_parameters: list[torch.nn.Parameter] | list[dict] = parameter_list

    no_decay_names = frozenset(
        str(parameter_name)
        for parameter_name in getattr(model, "no_weight_decay_parameter_names", ())
    )
    if no_decay_names:
        named_model_parameters = dict(model.named_parameters())
        missing_names = no_decay_names - named_model_parameters.keys()
        if missing_names:
            raise ValueError(
                "Model declared unknown no-weight-decay parameters: "
                + ", ".join(sorted(missing_names))
            )
        no_decay_ids = {id(named_model_parameters[key]) for key in no_decay_names}
        included_ids = {id(parameter) for parameter in parameter_list}
        if no_decay_ids - included_ids:
            raise ValueError("Model no-weight-decay parameters are absent from the optimizer input")
        optimizer_parameters = [
            {"params": [p for p in parameter_list if id(p) not in no_decay_ids]},
            {
                "params": [p for p in parameter_list if id(p) in no_decay_ids],
                "weight_decay": 0.0,
            },
        ]

    if name == "adamw":
        return torch.optim.AdamW(optimizer_parameters, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(optimizer_parameters, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"task.optimizer must be adam|adamw, got {name!r}")


def scheduler_step(cfg, optimizer: torch.optim.Optimizer, epoch: int, total_epochs: int) -> None:
    """Apply the optional generic warmup-cosine schedule; the frozen default is null."""
    name = str(cfg.task.get("scheduler", "null")).strip().lower()
    if name in {"null", "", "none"}:
        return
    if name != "warmup_cosine":
        raise ValueError(f"task.scheduler must be null|warmup_cosine, got {name!r}")
    warmup = max(int(cfg.task.get("scheduler_warmup_epochs", 10)), 1)
    base_lr = float(cfg.model.get("lr", cfg.task.lr))
    final_lr = float(cfg.task.get("scheduler_min_lr", 1e-5))
    if epoch <= warmup:
        lr = base_lr * epoch / warmup
    else:
        progress = (epoch - warmup) / max(total_epochs - warmup, 1)
        lr = final_lr + 0.5 * (base_lr - final_lr) * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = lr


def resolve_num_neighbors(cfg) -> list[int]:
    """Resolve sampler fanouts to exactly the configured encoder depth."""
    num_layers = int(cfg.model.get("num_layers", 1))
    if num_layers < 1:
        raise ValueError(f"model.num_layers must be >= 1, got {num_layers}")

    raw_neighbors = cfg.task.get("num_neighbors", -1)
    if isinstance(raw_neighbors, str):
        raw_neighbors = raw_neighbors.strip()
        if raw_neighbors.startswith("[") and raw_neighbors.endswith("]"):
            values = [int(item.strip()) for item in raw_neighbors[1:-1].split(",") if item.strip()]
        else:
            values = [int(raw_neighbors)]
    elif isinstance(raw_neighbors, (list, tuple, ListConfig)):
        values = [int(value) for value in raw_neighbors]
    else:
        values = [int(raw_neighbors)]

    if not values:
        raise ValueError("task.num_neighbors must contain at least one value")
    if len(values) >= num_layers:
        return values[:num_layers]
    return values + [values[-1]] * (num_layers - len(values))
