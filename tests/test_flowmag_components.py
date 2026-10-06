from __future__ import annotations

import torch
import math

from src.models.flowmag_components.adapters import AdaptedLinear
from src.models.flowmag_components.attention import CrossModalAttention, SameModalRelation
from src.models.flowmag_components.controller import FactorizedController, candidate_descriptor
from src.models.flowmag_components.layer import FlowMAGLayer


def _layer() -> FlowMAGLayer:
    return FlowMAGLayer(
        hidden_dim=8,
        relation_dim=4,
        router_dim=4,
        adapter_rank=2,
        cross_heads=2,
        ffn_multiplier=2,
        dropout=0.0,
        attention_dropout=0.0,
        null_cardinality_correction=True,
        tau_start=2.0,
        init_smooth_prob=0.75,
        init_same_admission=0.8,
        init_cross_admission=0.2,
        eps=1e-8,
        same_update_scale=0.1,
        cross_update_scale=0.1,
    )


def test_adapted_linear_zero_delta_has_live_zero_factor_gradient():
    torch.manual_seed(1)
    module = AdaptedLinear(5, 3, 2, ("text",))
    value = torch.randn(4, 5)
    torch.testing.assert_close(module(value, "text"), module.base(value), rtol=0, atol=0)
    module(value, "text").square().sum().backward()
    assert torch.count_nonzero(module.a["text"].weight) > 0
    assert module.b["text"].weight.grad is not None
    assert torch.isfinite(module.b["text"].weight.grad).all()
    assert module.b["text"].weight.grad.abs().sum() > 0


def test_empty_same_neighborhood_has_zero_flows_and_admission():
    module = _layer().eval()
    h_text = torch.randn(3, 8)
    h_image = torch.randn(3, 8)
    valid = torch.zeros(3, dtype=torch.bool)
    edge_index = torch.tensor([[0, 1], [2, 2]])
    mu, same_valid, alpha = module.same_relation(h_text, edge_index, valid, "text")
    assert torch.equal(mu, torch.zeros_like(mu))
    assert not same_valid.any()
    assert alpha.numel() == 0
    _, _, diag, _ = module(
        h_text, h_image, valid, valid, edge_index, rho=1.0, tau=1.0
    )
    assert diag["text"]["smooth_rms"].item() == 0
    assert diag["text"]["contrast_rms"].item() == 0
    assert diag["text"]["same_admission"].item() == 0


def test_cross_without_real_source_is_exact_zero_and_null_only():
    module = CrossModalAttention(8, 2, 2, null_cardinality_correction=True).eval()
    target = torch.randn(4, 8)
    source = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    correction, null_mass, entropy, cross_valid = module(
        target, source, edge_index, torch.zeros(4, dtype=torch.bool), "t2i"
    )
    assert torch.equal(correction, torch.zeros_like(correction))
    torch.testing.assert_close(null_mass, torch.ones_like(null_mass))
    assert entropy.item() == 0
    assert not cross_valid.any()


def test_null_cardinality_correction_keeps_equal_logit_mass_at_half():
    module = CrossModalAttention(8, 2, 2, null_cardinality_correction=True).eval()
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.zero_()
    target = torch.zeros(5, 8)
    source = torch.zeros(5, 8)
    edge_index = torch.tensor([[0, 0, 1, 2], [3, 4, 4, 4]])
    _, null_mass, _, valid = module(
        target, source, edge_index, torch.ones(5, dtype=torch.bool), "i2t"
    )
    assert valid[3] and valid[4]
    torch.testing.assert_close(null_mass[3], torch.tensor(0.5), atol=1e-6, rtol=0)
    torch.testing.assert_close(null_mass[4], torch.tensor(0.5), atol=1e-6, rtol=0)


def test_candidate_descriptor_detaches_direction_and_magnitude():
    candidate = torch.randn(4, 8, requires_grad=True)
    desc = candidate_descriptor(candidate, 1e-8)
    nn = torch.nn.Linear(desc.size(-1), 1)
    nn(desc).sum().backward()
    assert candidate.grad is None


def test_controller_warmup_uses_exact_priors_even_if_learned_heads_change():
    controller = FactorizedController(8, 4, 2.0, 0.75, 0.8, 0.2)
    with torch.no_grad():
        controller.candidate_scorer[-1].weight.fill_(0.5)
        controller.same_admission[-1].weight.fill_(0.5)
        controller.cross_admission[-1].weight.fill_(0.5)
    q = torch.randn(3, 4)
    smooth, contrast, cross = torch.randn(3, 8), torch.randn(3, 8), torch.randn(3, 8)
    valid = torch.ones(3, dtype=torch.bool)
    logits, pi = controller.frequency(q, smooth, contrast, tau=1.5, rho=0.0)
    same, cross_gate = controller.admissions(q, smooth, cross, torch.full((3,), 0.3), valid, valid, rho=0.0)
    torch.testing.assert_close(pi, torch.tensor([[0.75, 0.25]]).expand_as(pi), rtol=0, atol=1e-7)
    torch.testing.assert_close(same, torch.full((3,), 0.8), rtol=0, atol=1e-7)
    torch.testing.assert_close(cross_gate, torch.full((3,), 0.2), rtol=0, atol=1e-7)
    assert torch.isfinite(logits).all()


def test_controller_ramp_is_linear_for_frequency_and_admissions():
    controller = FactorizedController(8, 4, 2.0, 0.75, 0.8, 0.2)
    with torch.no_grad():
        controller.candidate_scorer[-1].weight.fill_(0.03)
        controller.same_admission[-1].weight.fill_(0.02)
        controller.cross_admission[-1].weight.fill_(-0.02)
    q, smooth, contrast, cross = torch.randn(2, 4), torch.randn(2, 8), torch.randn(2, 8), torch.randn(2, 8)
    valid = torch.ones(2, dtype=torch.bool)
    _, p0 = controller.frequency(q, smooth, contrast, tau=1.3, rho=0.0)
    _, p_half = controller.frequency(q, smooth, contrast, tau=1.3, rho=0.5)
    _, p1 = controller.frequency(q, smooth, contrast, tau=1.3, rho=1.0)
    torch.testing.assert_close(p_half, 0.5 * p0 + 0.5 * p1)
    a0 = controller.admissions(q, smooth, cross, torch.zeros(2), valid, valid, rho=0.0)
    ah = controller.admissions(q, smooth, cross, torch.zeros(2), valid, valid, rho=0.5)
    a1 = controller.admissions(q, smooth, cross, torch.zeros(2), valid, valid, rho=1.0)
    for half, start, end in zip(ah, a0, a1, strict=True):
        torch.testing.assert_close(half, 0.5 * start + 0.5 * end)


def test_same_relation_masks_invalid_sources_and_missing_target_can_receive():
    relation = SameModalRelation(8, 4, 2)
    h = torch.randn(3, 8)
    edges = torch.tensor([[0, 1], [2, 2]])
    valid = torch.tensor([False, True, False])
    mu, same_valid, alpha = relation(h, edges, valid, "text")
    assert same_valid.tolist() == [False, False, True]
    assert alpha.numel() == 1 and alpha.item() == 1.0
    torch.testing.assert_close(mu[2], h[1])

    cross = CrossModalAttention(8, 2, 2)
    target = torch.randn(3, 8)
    correction, null_mass, _, cross_valid = cross(
        target, h, edges[:, 1:], valid, "i2t"
    )
    assert cross_valid[2]
    assert null_mass[2] < 1
    assert correction[2].abs().sum() > 0


def test_contrast_has_no_bias_and_zero_difference_stays_zero():
    module = _layer()
    assert module.contrast.base.bias is None
    difference = torch.zeros(5, 8)
    with torch.no_grad():
        for parameter in module.contrast.parameters():
            parameter.normal_()
    output = module.contrast(difference, "text")
    torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)


def test_conditional_diagnostics_exclude_nodes_without_corresponding_sources():
    torch.manual_seed(9)
    module = _layer().eval()
    h_text = torch.randn(4, 8)
    h_image = torch.randn(4, 8)
    text_valid = torch.tensor([True, False, False, False])
    image_valid = torch.tensor([False, True, False, False])
    edges = torch.tensor([[0, 1], [2, 3]])
    _, _, diagnostics, _ = module(
        h_text, h_image, text_valid, image_valid, edges, rho=1.0, tau=1.0
    )
    text_diag = diagnostics["text"]
    assert text_diag["same_valid_rate"].item() == 0.25
    assert text_diag["no_same_source_rate"].item() == 0.75
    assert text_diag["cross_valid_rate"].item() == 0.25
    assert text_diag["no_cross_source_rate"].item() == 0.75
    assert text_diag["smooth_rms_given_valid"] > text_diag["smooth_rms"]
    _, expected_null, _, _, _, _ = module.cross_attention(
        module.pre_norm(h_text), module.pre_norm(h_image), edges, image_valid, "i2t",
        return_attention_diagnostics=True,
    )
    torch.testing.assert_close(text_diag["null_mass_given_cross_valid"], expected_null[3])
    assert text_diag["null_mass_given_cross_valid"] < expected_null.mean()


def test_attention_entropy_averages_per_head_and_normalizes_over_real_plus_null():
    module = CrossModalAttention(8, 2, 2, null_cardinality_correction=True).eval()
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.zero_()
        module.query.base.weight.copy_(torch.eye(8))
        module.null_key["t2i"][1].fill_(1.0)
    target = torch.ones(3, 8)
    source = torch.zeros(3, 8)
    edges = torch.tensor([[0, 1, 2], [2, 2, 2]])
    _, _, _, valid, entropy_by_target, normalized = module(
        target, source, edges, torch.ones(3, dtype=torch.bool), "t2i",
        return_attention_diagnostics=True,
    )
    p_real_head0 = torch.tensor(1.0 / 6.0)
    p_null_head0 = torch.tensor(0.5)
    denom_head1 = 3.0 + 3.0 * math.exp(2.0)
    p_real_head1 = torch.tensor(1.0 / denom_head1)
    p_null_head1 = torch.tensor(3.0 * math.exp(2.0) / denom_head1)
    h0 = -(3 * p_real_head0 * p_real_head0.log() + p_null_head0 * p_null_head0.log())
    h1 = -(3 * p_real_head1 * p_real_head1.log() + p_null_head1 * p_null_head1.log())
    expected = (h0 + h1) / 2
    assert valid.tolist() == [False, False, True]
    torch.testing.assert_close(entropy_by_target[2], expected, rtol=1e-5, atol=1e-5)
    assert torch.isfinite(normalized).all()
    assert torch.all(normalized >= 0) and torch.all(normalized <= 1 + 1e-6)
    torch.testing.assert_close(normalized[2], expected / math.log(4), rtol=1e-5, atol=1e-5)


def test_update_ratios_use_input_and_middle_state_rms_denominators():
    module = _layer().eval()
    h_text = torch.randn(4, 8)
    h_image = torch.randn(4, 8)
    valid = torch.ones(4, dtype=torch.bool)
    edges = torch.tensor([[0, 1, 2], [1, 2, 3]])
    _, _, diagnostics, _ = module(
        h_text, h_image, valid, valid, edges, rho=1.0, tau=1.0
    )
    d = diagnostics["text"]
    eps = module.controller.eps
    torch.testing.assert_close(
        d["graph_update_ratio"], d["graph_update_rms"] / (d["state_input_rms"] + eps)
    )
    torch.testing.assert_close(
        d["ffn_update_ratio"], d["ffn_update_rms"] / (d["middle_state_rms"] + eps)
    )
