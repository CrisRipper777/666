from __future__ import annotations

import torch
from omegaconf import OmegaConf

from src.models.flowmag import Model


def _cfg(task: str = "nc", **overrides):
    model = {
        "hidden_dim": 16,
        "num_layers": 2,
        "prior_dim_per_modality": 4,
        "graph_block_dim": 8,
        "relation_dim": 8,
        "router_dim": 8,
        "adapter_rank": 2,
        "cross_heads": 2,
        "ffn_multiplier": 2,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "detach_candidate_descriptor": True,
        "null_cardinality_correction": True,
        "warmup_ratio": 0.05,
        "controller_ramp_ratio": 0.10,
        "router_temp_start": 2.0,
        "router_temp_end": 1.0,
        "router_temp_anneal_end_ratio": 0.60,
        "init_smooth_prob": 0.75,
        "init_same_admission": 0.80,
        "init_cross_admission": 0.20,
        "lambda_prior": 0.05,
        "lambda_graph_start": 0.10,
        "graph_aux_decay_end_ratio": 0.40,
        "lambda_router_z": 1e-3,
        "eps": 1e-8,
    }
    model.update(overrides)
    return OmegaConf.create({"model": model, "task": {"name": task, "epochs": 100}})


def _data_info(task: str = "nc"):
    return {"input_dim": 7, "text_dim": 4, "visual_dim": 3, "num_classes": 3 if task == "nc" else None}


def _toy_graph():
    torch.manual_seed(8)
    x = torch.randn(6, 7)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 1, 3], [1, 2, 3, 4, 5, 0, 4, 0]], dtype=torch.long
    )
    return x, edge_index


def test_model_contract_and_output_width():
    model = Model(_cfg(), _data_info())
    x, edge_index = _toy_graph()
    output = model(x, edge_index)
    assert isinstance(output, tuple) and len(output) == 5
    assert output[0].shape == (x.size(0), 16)
    assert model.out_dim == 2 * 4 + 8
    assert output[1] is None and output[2] is None
    assert output[3].ndim == 0


def test_final_prior_blocks_are_independent_of_graph_state():
    model = Model(_cfg(), _data_info()).eval()
    torch.manual_seed(4)
    u_text, u_image = torch.randn(5, 16), torch.randn(5, 16)
    text_valid = torch.tensor([True, False, True, True, False])
    image_valid = torch.tensor([False, True, True, False, True])
    h_text_a, h_image_a = torch.randn(5, 16), torch.randn(5, 16)
    h_text_b, h_image_b = h_text_a + 10, h_image_a - 7
    z_a, _, _, _ = model._compose_output(u_text, u_image, h_text_a, h_image_a, text_valid, image_valid)
    z_b, _, _, _ = model._compose_output(u_text, u_image, h_text_b, h_image_b, text_valid, image_valid)
    assert torch.equal(z_a[:, :8], z_b[:, :8])
    assert not torch.allclose(z_a[:, 8:], z_b[:, 8:])


def test_schedule_phases_checkpoint_eligibility_and_router_z():
    model = Model(_cfg(warmup_ratio=0.2, controller_ramp_ratio=0.4), _data_info())
    model.set_epoch(1)  # progress .01, phase A
    assert model.phase == "A" and model.rho == 0
    assert model.lambda_router_z_effective == 0
    assert not model.checkpoint_eligible(29)
    model.set_epoch(40)  # progress .4, midpoint of ramp
    assert model.phase == "B"
    assert abs(model.rho - 0.5) < 1e-9
    assert abs(model.lambda_router_z_effective - 0.5 * model.lambda_router_z) < 1e-9
    assert not model.checkpoint_eligible(59)
    model.set_epoch(60)  # progress .6, ramp complete
    assert model.phase == "C" and model.rho == 1
    assert model.checkpoint_eligible(60)
    assert model.lambda_router_z_effective == model.lambda_router_z

    x, _ = _toy_graph()
    no_edges = torch.empty((2, 0), dtype=torch.long)
    model.set_epoch(1)
    _, _, _, loss_a, info_a = model(x, no_edges)
    assert loss_a.item() == 0
    assert info_a["router_z_raw"] == 0
    model.set_epoch(60)
    _, _, _, loss_b, info_b = model(x, no_edges)
    assert info_b["router_z_raw"] == 0
    assert loss_b.item() == 0


def test_missing_modalities_stay_finite_and_can_receive_neighbor_updates():
    model = Model(_cfg(warmup_ratio=0, controller_ramp_ratio=0), _data_info()).eval()
    x, _ = _toy_graph()
    x[0, :4] = 0.0  # missing text target
    x[0, 4:] = 0.0  # missing image target
    x[1, 4:] = float("nan")  # missing image source
    edge_index = torch.tensor([[1, 2, 3], [0, 0, 1]], dtype=torch.long)
    z, _, _, _, aux = model(x, edge_index)
    assert torch.isfinite(z).all()
    _, _, text_valid, image_valid, _, _ = model._split_and_stem(x)
    assert not text_valid[0] and text_valid[1]
    assert not image_valid[1]
    cross_text, null_mass, _, cross_valid = model.layers[0].cross_attention(
        torch.randn(6, 16), torch.randn(6, 16), edge_index[:, :1], text_valid, "t2i"
    )
    assert cross_valid[0]  # image target 0 can receive valid text source 1
    assert null_mass[0] < 1
    assert cross_text[0].abs().sum() > 0
    masked_cross, masked_null, _, masked_valid = model.layers[0].cross_attention(
        torch.randn(6, 16), torch.randn(6, 16), edge_index[:, :1], image_valid, "i2t"
    )
    assert not masked_valid[0]
    assert masked_null[0] == 1
    assert torch.equal(masked_cross[0], torch.zeros_like(masked_cross[0]))
    assert len(aux["layers"]) == 2


def test_full_forward_matches_exact_layerwise_inference():
    model = Model(_cfg(warmup_ratio=0, controller_ramp_ratio=0), _data_info()).eval()
    x, edge_index = _toy_graph()
    x[0, :4] = 0
    x[2, 4:] = float("nan")
    with torch.no_grad():
        expected = model(x, edge_index)[0].cpu()
        actual = model.inference(x, edge_index, device="cpu", batch_size=2)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_nc_auxiliary_uses_only_passed_training_node_labels():
    model = Model(_cfg(), _data_info())
    x, edge_index = _toy_graph()
    _, _, _, _, aux_info = model(x, edge_index)
    train_idx = torch.tensor([0, 2, 3])
    labels_a = torch.tensor([0, 1, 2, 1, 0, 2])
    labels_b = labels_a.clone()
    labels_b[[1, 4, 5]] = torch.tensor([2, 2, 1])
    loss_a = model.compute_supervised_aux_loss(
        task="nc", aux_info=aux_info, labels=labels_a, node_index=train_idx
    )
    loss_b = model.compute_supervised_aux_loss(
        task="nc", aux_info=aux_info, labels=labels_b, node_index=train_idx
    )
    torch.testing.assert_close(loss_a["loss"], loss_b["loss"], rtol=0, atol=0)
    assert set(loss_a) == {"loss", "prior_text", "prior_image", "graph"}
    assert torch.isfinite(loss_a["loss"])


def test_lp_auxiliary_uses_current_training_batch_pairs_and_labels():
    model = Model(_cfg(task="lp"), _data_info(task="lp"))
    x, edge_index = _toy_graph()
    _, _, _, _, aux_info = model(x, edge_index)
    pair_index = torch.tensor([[0, 1, 2], [1, 3, 5]])
    batch_labels = torch.tensor([1.0, 0.0, 0.0])
    aux_a = model.compute_supervised_aux_loss(
        task="lp", aux_info=aux_info, edge_label_index=pair_index, edge_label=batch_labels
    )
    aux_b = model.compute_supervised_aux_loss(
        task="lp", aux_info=aux_info, edge_label_index=pair_index, edge_label=batch_labels.flip(0)
    )
    assert torch.isfinite(aux_a["loss"]) and torch.isfinite(aux_b["loss"])
    assert not torch.allclose(aux_a["loss"], aux_b["loss"])
    aux_a["loss"].backward()
    assert model.supervised_auxiliary.graph_pair.weight.grad is not None


def test_end_to_end_forward_backward_has_finite_gradients_in_major_components():
    model = Model(_cfg(warmup_ratio=0, controller_ramp_ratio=0), _data_info())
    model.set_epoch(100)
    x, edge_index = _toy_graph()
    z, _, _, aux_loss, aux_info = model(x, edge_index)
    train_idx = torch.tensor([0, 1, 2, 3])
    labels = torch.tensor([0, 1, 2, 1])
    sup = model.compute_supervised_aux_loss(
        task="nc", aux_info=aux_info, labels=labels, node_index=train_idx
    )["loss"]
    (z.square().mean() + aux_loss + sup).backward()
    groups = (
        "stem_text", "stem_image", "layers.0.same_relation", "layers.0.smooth",
        "layers.0.contrast", "layers.0.cross_attention", "layers.0.context",
        "layers.0.controller", "layers.0.ffn", "prior_text_projection",
        "prior_image_projection", "graph_fuse", "supervised_auxiliary",
    )
    named = dict(model.named_parameters())
    for prefix in groups:
        grads = [p.grad for name, p in named.items() if name.startswith(prefix)]
        assert grads and any(grad is not None for grad in grads), prefix
        assert all(grad is None or torch.isfinite(grad).all() for grad in grads), prefix
