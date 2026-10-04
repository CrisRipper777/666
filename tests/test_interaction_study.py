from __future__ import annotations

import torch
import torch.nn.functional as F

from src.analysis.interaction_study.diagnostic_host import FeatureOnlyMultimodalDiagnosticHost
from src.analysis.interaction_study.perturbation import (
    finite_difference_loss,
    operator_response,
    sample_receiver_sources,
    equal_relative_norm,
)
from src.analysis.interaction_study.splits import make_study_split, mask_test_labels
from src.analysis.interaction_study.statistics import stratified_sample
from src.analysis.problem_deep_dive import train_edge_quartile_thresholds


def _split_fixture():
    labels = torch.tensor([0] * 20 + [1] * 20 + [2] * 20 + [0, 1, 2] * 4 + [0, 1, 2] * 4)
    train = torch.arange(60)
    validation = torch.arange(60, 72)
    test = torch.arange(72, 84)
    return labels, train, validation, test


def test_diagnostic_split_is_deterministic_stratified_and_validation_is_report_only():
    labels, train, val, test = _split_fixture()
    a = make_study_split(labels, train, val, test, seed=31)
    b = make_study_split(labels, train, val, test, seed=31)
    assert torch.equal(a.diagnostic_fit, b.diagnostic_fit)
    assert torch.equal(a.diagnostic_calib, b.diagnostic_calib)
    assert not torch.isin(a.diagnostic_fit, val).any()
    assert not torch.isin(a.diagnostic_fit, test).any()
    assert not torch.isin(a.diagnostic_calib, val).any()
    assert not torch.isin(a.diagnostic_calib, test).any()
    assert a.diagnostic_fit.numel() + a.diagnostic_calib.numel() == train.numel()
    fit_counts = torch.bincount(labels[a.diagnostic_fit], minlength=3)
    calib_counts = torch.bincount(labels[a.diagnostic_calib], minlength=3)
    assert fit_counts.tolist() == [16, 16, 16]
    assert calib_counts.tolist() == [4, 4, 4]
    assert a.metadata()["report_split"] == "original_validation"


def _label_only_summary(labels, train, val, test):
    split = make_study_split(labels, train, val, test, seed=7)
    visible = mask_test_labels(labels, split)
    return {
        "fit_hist": torch.bincount(visible[split.diagnostic_fit], minlength=3).tolist(),
        "calib_hist": torch.bincount(visible[split.diagnostic_calib], minlength=3).tolist(),
        "report_hist": torch.bincount(visible[split.original_validation], minlength=3).tolist(),
        "split_hashes": split.index_sha256,
        "test_label_access": split.metadata()["test_label_access"],
    }


def test_test_label_mutation_does_not_change_study_summary():
    labels, train, val, test = _split_fixture()
    changed = labels.clone()
    changed[test] = 99 - torch.arange(test.numel())
    assert _label_only_summary(labels, train, val, test) == _label_only_summary(changed, train, val, test)


def test_equal_relative_norm_matches_target_norm_and_preserves_direction():
    target = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    direction = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    normalized = equal_relative_norm(direction, target)
    assert torch.allclose(normalized.norm(dim=-1), target.norm(dim=-1), atol=1e-6)
    assert torch.allclose(F.normalize(normalized, dim=-1), F.normalize(direction, dim=-1), atol=1e-6)


def test_central_finite_difference_matches_toy_linear_classifier_derivative():
    weight = torch.tensor([[1.2, -0.4], [-0.3, 0.8], [0.2, 0.1]])
    bias = torch.tensor([0.1, -0.2, 0.05])
    base = torch.tensor([[0.3, -0.7]], requires_grad=True)
    direction = torch.tensor([[0.4, 0.2]])
    target = torch.tensor([1])

    def loss_fn(z):
        return F.cross_entropy(z @ weight.T + bias, target, reduction="none")

    exact = torch.autograd.grad(loss_fn(base).sum(), base)[0].mul(direction).sum()
    numerical = finite_difference_loss(loss_fn, base.detach(), direction, alpha=1e-3)
    assert torch.allclose(numerical.squeeze(), exact, atol=2e-4, rtol=2e-4)


def test_train_quantile_thresholds_ignore_validation_and_test_edges():
    pairs = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    train_idx = torch.tensor([0, 1, 2, 3])
    values = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    first = train_edge_quartile_thresholds(values, pairs, train_idx)
    changed = values.clone()
    changed[3:] = torch.tensor([-100.0, 100.0, 500.0])
    second = train_edge_quartile_thresholds(changed, pairs, train_idx)
    assert first == second
    assert first["train_edge_count"] == 3


def test_random_nonneighbor_control_preserves_each_receiver_source_count():
    edge_index = torch.tensor([[0, 0, 1, 2, 3], [1, 2, 2, 3, 4]])
    receivers = torch.tensor([0, 2, 4, 5])
    a = sample_receiver_sources(edge_index, receivers, num_nodes=8, neighbor_cap=3, seed=19)
    b = sample_receiver_sources(edge_index, receivers, num_nodes=8, neighbor_cap=3, seed=19)
    assert torch.equal(a.counts, b.counts)
    assert torch.equal(a.neighbor_ids, b.neighbor_ids)
    assert torch.equal(a.random_ids, b.random_ids)
    assert int(a.counts.sum()) == a.neighbor_ids.numel() == a.random_ids.numel()
    pos = {int(node): row for row, node in enumerate(a.receivers.tolist())}
    for receiver in receivers.tolist():
        row = pos[receiver]
        assert int(a.counts[row]) <= 3


def test_stratified_edge_sampling_is_deterministic():
    ids = torch.arange(100)
    strata = [(int(i % 4), int((i // 4) % 3)) for i in ids.tolist()]
    a, count_a = stratified_sample(ids, strata, max_items=25, seed=5)
    b, count_b = stratified_sample(ids, strata, max_items=25, seed=5)
    assert torch.equal(a, b)
    assert count_a == count_b
    assert a.numel() == 25
    assert len(count_a) == 12


def test_operator_response_profile_shape_and_finite_values():
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    split = make_study_split(labels, torch.arange(6), torch.arange(6, 9), torch.arange(9, 12), seed=4)
    visible = mask_test_labels(labels, split)
    graph = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 11, 6]])
    sources = sample_receiver_sources(graph, torch.cat([split.diagnostic_calib, split.original_validation]), 12, 3, 9)
    host = FeatureOnlyMultimodalDiagnosticHost(4, 4, 8, 3)
    h_t, h_i = torch.randn(12, 8), torch.randn(12, 8)
    rows, raw = operator_response(
        host, h_t, h_i, h_t.clone(), h_i.clone(), visible, split, sources, 0.1,
        torch.device("cpu"), batch_size=4, bootstrap_reps=20, seed=0,
    )
    assert len(rows) == 2
    for modality in ("text", "image"):
        profile = raw[f"{modality}_response_profile"]
        assert profile.shape == (split.original_validation.numel(), 4)
        assert torch.isfinite(profile).all()
