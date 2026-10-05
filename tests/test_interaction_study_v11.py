from __future__ import annotations

import numpy as np
import torch

from src.analysis.interaction_study.perturbation import sample_receiver_sources
from src.analysis.interaction_study.splits import make_study_split, mask_test_labels
from src.analysis.interaction_study.v11 import (
    build_null_action_directions,
    classify_p3_dataset,
    deterministic_matched_effect,
    factorial_p2_metrics,
    functional_null_paired_metrics,
    matched_effect_receiver_bootstrap,
    margin_conditioned_pair_agreement,
    operator_stability_summary,
    orient_validation_edges,
    summarize_reference_noise,
    summarize_seed_receiver_vectors,
)


def test_p1_validation_endpoint_orientation_is_correct_for_one_or_two_validation_ends():
    pairs = torch.tensor([[1, 4], [0, 3], [5, 6]])
    source, receiver = orient_validation_edges(pairs, torch.tensor([0, 3, 4]))
    assert list(zip(source.tolist(), receiver.tolist(), strict=True)) == [(1, 4), (0, 3), (3, 0)]
    assert set(receiver.tolist()) == {0, 3, 4}


def test_p1_identical_modalities_with_sampling_jitter_do_not_create_positive_adjusted_signal():
    values = torch.tensor([[0.1, 0.3, 0.7], [0.2, 0.4, 0.6], [0.15, 0.35, 0.65]])
    result = summarize_reference_noise(values, values.clone(), torch.tensor([10, 20, 30]), reps=200, seed=8)
    assert result["metrics"]["cross_signal"]["mean"] == 0.0
    assert result["metrics"]["text_reference_instability"]["mean"] > 0
    assert result["metrics"]["conservative_adjusted_gap"]["mean"] < 0
    assert result["metrics"]["conservative_adjusted_gap"]["ci_high"] < 0


def test_seed_aggregation_averages_same_receivers_before_bootstrapping():
    receivers = torch.tensor([100, 200, 300])
    per_seed = {
        0: torch.tensor([0.0, 10.0, 20.0]),
        1: torch.tensor([10.0, 0.0, 20.0]),
        2: torch.tensor([20.0, 0.0, 10.0]),
    }
    result = summarize_seed_receiver_vectors(per_seed, receivers, reps=200, seed=3)
    assert torch.allclose(
        result["host_seed_averaged_receiver_values"],
        torch.tensor([10.0, 10.0 / 3.0, 50.0 / 3.0]), atol=1e-6,
    )
    assert np.isclose(result["summary"]["mean"], np.mean([10.0, 10.0 / 3.0, 50.0 / 3.0]))
    assert "then resample" in result["aggregation"]
    assert result["summary"]["receiver_blocks"] == 3


def test_p2_factorial_conditional_graph_gain_equals_lsr_minus_lsc():
    values = factorial_p2_metrics(
        torch.tensor([10.0]), torch.tensor([9.0]), torch.tensor([8.0]),
        torch.tensor([8.5]), torch.tensor([7.0]), torch.tensor([8.0]),
    )
    assert values["conditional_graph_gain"].item() == 1.0
    assert values["conditional_graph_gain"].item() == 8.0 - 7.0


def test_p3_deterministic_weighted_matched_effect_matches_hand_calculation():
    result = deterministic_matched_effect(
        values=torch.tensor([1.0, 3.0, 5.0, 0.0, 2.0, 4.0, 6.0]),
        semantic_bin=torch.tensor([0, 0, 0, 1, 1, 1, 1]),
        structure_bin=torch.tensor([0, 0, 3, 0, 3, 3, 3]),
        source_degree_bin=torch.zeros(7, dtype=torch.long),
        receiver_degree_bin=torch.zeros(7, dtype=torch.long),
    )
    # Stratum 0: min(2,1)=1 times (5-2)=3; stratum 1: min(1,3)=1 times (4-0)=4.
    assert result["matched_effect"] == 3.5
    assert result["matched_strata"] == 2
    assert result["matched_interactions_per_arm"] == 2


def test_p3_bootstrap_resamples_complete_receiver_blocks_and_rematches():
    result = matched_effect_receiver_bootstrap(
        values=torch.tensor([0.0, 2.0, 0.0, 10.0]),
        semantic_bin=torch.zeros(4, dtype=torch.long),
        structure_bin=torch.tensor([0, 3, 0, 3]),
        source_degree_bin=torch.zeros(4, dtype=torch.long),
        receiver_degree_bin=torch.zeros(4, dtype=torch.long),
        receivers=torch.tensor([1, 1, 2, 2]),
        reps=2000,
        seed=29,
    )
    assert result["matched_effect"] == 6.0
    assert result["receiver_block_count"] == 2
    assert result["valid_bootstrap_count"] == 2000
    assert result["ci_low"] <= 2.0
    assert result["ci_high"] >= 10.0
    assert "all interaction rows retained as blocks" in result["bootstrap_unit"]


def test_p3_dataset_grading_returns_text_and_image_even_when_text_is_strong():
    rows = []
    for seed, value in enumerate([1.0, 1.1, 1.2]):
        rows.append({"modality": "text", "host_seed": seed, "matched_effect": value,
                     "ci_low": value - 0.1, "ci_high": value + 0.1, "matched_strata": 2})
        rows.append({"modality": "image", "host_seed": seed, "matched_effect": value,
                     "ci_low": value - 2.0, "ci_high": value + 2.0, "matched_strata": 1})
    result = classify_p3_dataset(rows)
    assert result["text"]["grade"] == "Strong"
    assert result["image"]["grade"] == "Weak"
    assert len(result["text"]["seed_details"]) == 3
    assert len(result["image"]["seed_details"]) == 3


def test_operator_stability_identical_seed_profiles_have_perfect_agreement_and_kappa():
    profile = np.asarray([[0.0, -0.2, 0.3, 0.1], [0.0, 0.2, 0.3, -0.1], [0.0, -0.1, 0.4, 0.2]])
    result = operator_stability_summary({0: profile, 1: profile.copy(), 2: profile.copy()},
                                        ["NoOp", "Smooth", "Contrast", "Cross"])
    assert result["unanimous_top_action_fraction"] == 1.0
    assert result["at_least_two_of_three_same_action_fraction"] == 1.0
    assert all(row["raw_agreement"] == 1.0 and row["cohen_kappa"] == 1.0 for row in result["seed_pairs"])


def test_margin_conditioned_operator_stability_filters_near_ties():
    left_profile = np.asarray([[0.0, 0.0005, 1.0, 2.0], [0.0, 0.2, 1.0, 2.0], [0.0, 0.1, 1.0, 2.0]])
    right_profile = np.asarray([[0.0004, 0.0, 1.0, 2.0], [0.0, 0.2, 1.0, 2.0], [0.0, 0.1, 1.0, 2.0]])
    left_actions = left_profile.argmin(1)
    right_actions = right_profile.argmin(1)
    left_margin = np.sort(left_profile, axis=1)[:, 1] - np.sort(left_profile, axis=1)[:, 0]
    right_margin = np.sort(right_profile, axis=1)[:, 1] - np.sort(right_profile, axis=1)[:, 0]
    result = margin_conditioned_pair_agreement(left_actions, right_actions, left_margin, right_margin)
    strict = result["both_margin_at_least_threshold"]
    assert strict["receiver_count"] == 2
    assert strict["raw_agreement"] == 1.0


def test_null_action_bank_has_four_matched_actions_and_equal_relative_norm():
    edge = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]])
    receivers = torch.tensor([1, 3, 5])
    source_bank = sample_receiver_sources(edge, receivers, num_nodes=8, neighbor_cap=2, seed=55)
    assert source_bank.neighbor_ids.numel() == source_bank.random_ids.numel()
    assert source_bank.counts.sum().item() == source_bank.neighbor_ids.numel()
    target = torch.tensor([[3.0, 4.0], [1.0, 2.0], [2.0, 3.0]])
    same = torch.tensor([[1.0, 0.0], [2.0, 2.0], [1.0, 1.0]])
    cross = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 2.0]])
    actions = build_null_action_directions(target, same, cross)
    assert list(actions) == ["NoOp", "Rand-Same-Smooth", "Rand-Same-Contrast", "Rand-Cross"]
    assert all(direction.shape == target.shape for direction in actions.values())
    for action in ("Rand-Same-Smooth", "Rand-Same-Contrast", "Rand-Cross"):
        assert torch.allclose(actions[action].norm(dim=-1), target.norm(dim=-1), atol=1e-6)


def test_functional_vs_null_paired_metric_formulas():
    functional = torch.tensor([[0.0, -2.0, 1.0, -1.0], [0.0, 1.0, 2.0, -3.0]])
    null = torch.tensor([[0.0, -1.0, 0.5, -0.5], [0.0, -2.0, 1.0, -1.0]])
    result = functional_null_paired_metrics(functional, null, functional_global_index=1, null_global_index=2)
    assert torch.equal(result["functional_headroom"], torch.tensor([0.0, 4.0]))
    assert torch.equal(result["null_headroom"], torch.tensor([1.5, 3.0]))
    assert torch.equal(result["excess_headroom"], torch.tensor([-1.5, 1.0]))
    assert torch.equal(result["best_action_advantage"], torch.tensor([1.0, 1.0]))
    assert torch.equal(result["global_fixed_advantage"], torch.tensor([2.5, 0.0]))


def test_toy_v11_reanalysis_summary_is_invariant_to_test_label_mutation():
    labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0])
    train = torch.arange(0, 6)
    validation = torch.arange(6, 9)
    test = torch.arange(9, 12)

    def toy_summary(y):
        split = make_study_split(y, train, validation, test, seed=5)
        visible = mask_test_labels(y, split)
        counts = torch.bincount(visible[split.diagnostic_fit], minlength=2)
        majority = counts.argmax()
        report_accuracy = float((visible[split.original_validation] == majority).float().mean())
        return {
            "fit_hist": counts.tolist(),
            "report_accuracy": report_accuracy,
            "split_hashes": split.index_sha256,
            "test_label_access": split.metadata()["test_label_access"],
            "test_metric_access": split.metadata()["test_metric_access"],
        }

    changed = labels.clone()
    changed[test] = torch.tensor([99, -10, 12345])
    assert toy_summary(labels) == toy_summary(changed)
