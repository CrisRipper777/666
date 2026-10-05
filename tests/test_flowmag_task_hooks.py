from __future__ import annotations

import torch
import pytest
from omegaconf import OmegaConf

from src.models.flowmag import Model as FlowMAG
from src.tasks.common import (
    checkpoint_is_eligible,
    compute_optional_supervised_aux_loss,
    restore_checkpoint_schedule,
)
from src.tasks.lp import _resolve_lp_num_neighbors


class _LegacyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))


class _AuxModel(_LegacyModel):
    def compute_supervised_aux_loss(self, *, task, aux_info, **kwargs):
        del task, aux_info, kwargs
        return {"loss": self.weight.square()}


def test_optional_hooks_leave_existing_models_eligible_with_zero_aux():
    model = _LegacyModel()
    aux = compute_optional_supervised_aux_loss(
        model, task="nc", aux_info={}, labels=torch.tensor([1])
    )
    assert aux.ndim == 0 and aux.item() == 0
    assert checkpoint_is_eligible(model, 1)
    restore_checkpoint_schedule(model, 1)  # legacy models have no schedule hook


def test_optional_supervised_hook_accepts_loss_mapping():
    model = _AuxModel()
    loss = compute_optional_supervised_aux_loss(model, task="nc", aux_info={})
    assert loss.item() == 1
    loss.backward()
    assert model.weight.grad.item() == 2


def test_checkpoint_hook_defers_flowmag_until_ramp_is_complete():
    cfg = OmegaConf.create({
        "model": {
            "hidden_dim": 8,
            "num_layers": 2,
            "prior_dim_per_modality": 2,
            "graph_block_dim": 4,
            "relation_dim": 4,
            "router_dim": 4,
            "adapter_rank": 2,
            "cross_heads": 2,
            "ffn_multiplier": 2,
            "warmup_ratio": 0.1,
            "controller_ramp_ratio": 0.2,
        },
        "task": {"name": "lp", "epochs": 10},
    })
    model = FlowMAG(cfg, {"input_dim": 6, "text_dim": 3, "visual_dim": 3, "num_classes": None})
    assert not checkpoint_is_eligible(model, 2)
    assert checkpoint_is_eligible(model, 3)
    model.set_epoch(10)
    restore_checkpoint_schedule(model, 2)
    assert model.current_epoch == 2
    assert abs(model.rho - 0.5) < 1e-9


def test_flowmag_sampler_depth_matches_encoder_without_changing_legacy_models():
    cfg = OmegaConf.create({
        "model": {"num_layers": 2},
        "task": {"num_neighbors": [7, 5, 3]},
    })
    flowmag = type("DepthModel", (), {"requires_full_lp_sampler_depth": True})()
    legacy = _LegacyModel()
    assert _resolve_lp_num_neighbors(cfg, flowmag) == [7, 5]
    assert _resolve_lp_num_neighbors(cfg, legacy) == [7, 5, 3]


def test_flowmag_rejects_incompatible_dimensions_and_attention_dropout():
    cfg = OmegaConf.create({
        "model": {"hidden_dim": 10, "cross_heads": 3},
        "task": {"name": "nc", "epochs": 2},
    })
    with pytest.raises(ValueError, match="cross_heads must divide hidden_dim"):
        FlowMAG(cfg, {"input_dim": 6, "text_dim": 3, "visual_dim": 3, "num_classes": 2})

    cfg.model.hidden_dim = 12
    cfg.model.cross_heads = 3
    cfg.model.attention_dropout = 0.1
    with pytest.raises(ValueError, match="attention_dropout=0"):
        FlowMAG(cfg, {"input_dim": 6, "text_dim": 3, "visual_dim": 3, "num_classes": 2})
