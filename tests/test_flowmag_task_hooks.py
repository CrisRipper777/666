from __future__ import annotations

import torch
import pytest
from omegaconf import OmegaConf

from src.models.flowmag import Model as FlowMAG
from src.tasks.common import (
    checkpoint_is_eligible,
    collect_optional_gradient_diagnostics,
    compute_optional_supervised_aux_loss,
    restore_checkpoint_schedule,
)
from src.tasks.lp import _resolve_lp_num_neighbors
from src.tasks.nc import _diagnostics_path_for_run, _resolve_nc_eval_labels


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
    assert collect_optional_gradient_diagnostics(model, True) is None


def test_optional_gradient_hook_is_opt_in_and_accepts_model_mapping():
    class _GradientModel(_LegacyModel):
        def __init__(self):
            super().__init__()
            self.called = False

        def gradient_diagnostics(self):
            self.called = True
            return {"weight": 3.0}

    model = _GradientModel()
    assert collect_optional_gradient_diagnostics(model, False) is None
    assert not model.called
    assert collect_optional_gradient_diagnostics(model, True) == {"weight": 3.0}
    assert model.called


def test_nc_label_resolution_can_avoid_test_split_entirely():
    class _NoTestAccess:
        y = torch.tensor([0, 1, 2, 99])
        train_idx = torch.tensor([0, 1])
        val_idx = torch.tensor([2])
        num_classes = 3

        @property
        def test_idx(self):
            raise AssertionError("test indices were accessed while evaluate_test=false")

    assert _resolve_nc_eval_labels(_NoTestAccess(), include_test=False) == [0, 1, 2]


def test_run_diagnostics_paths_are_distinct_across_runs(tmp_path):
    cfg = OmegaConf.create({"num_runs": 3})
    base = tmp_path / "dataset" / "epoch_diag.jsonl"
    paths = [_diagnostics_path_for_run(base, cfg, run_id) for run_id in range(3)]
    assert [path.name for path in paths] == [
        "epoch_diag_run1.jsonl", "epoch_diag_run2.jsonl", "epoch_diag_run3.jsonl"
    ]
    assert len(set(paths)) == 3
    assert all(path.parent.is_dir() for path in paths)


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
