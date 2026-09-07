import math

import pytest
import torch

from temporal.depth_prior_new_seeding import (
    DepthPriorSeedConfig,
    build_depth_prior_new_seeds,
    canonical_metric_depth,
    fit_metric_anchor_depth_scale,
    fit_reference_depth_scale,
    front_depth_sign_evidence,
    metric_anchor_median_relative_error,
    panel7_visible_signed_support,
    q_weighted_upsampled_signed_sam_score,
    reference_scale_anchor_mask,
    retain_top_magnitude_mask,
    threshold_signed_support,
    uncovered_by_learned_gaussian_support,
    unproject_camera_depth,
)


def test_canonical_metric_depth_uses_processed_mean_focal_length():
    raw = torch.full((2, 3), 2.0)
    intrinsics = torch.tensor(
        [[270.0, 0.0, 1.0], [0.0, 330.0, 1.0], [0.0, 0.0, 1.0]]
    )

    metric = canonical_metric_depth(raw, intrinsics)

    torch.testing.assert_close(metric, torch.full((2, 3), 2.0))


def test_canonical_metric_depth_rejects_invalid_focal_length():
    intrinsics = torch.eye(3)
    intrinsics[0, 0] = 0.0
    intrinsics[1, 1] = 0.0

    with pytest.raises(ValueError):
        canonical_metric_depth(torch.ones((2, 2)), intrinsics)


def test_front_depth_evidence_assigns_soft_mass_to_sam_pca_signs():
    predicted = torch.full((4, 4), 2.0)
    reference = torch.full((4, 4), 2.0)
    predicted[:2] = 1.0
    plus = torch.zeros((4, 4), dtype=torch.bool)
    minus = torch.zeros_like(plus)
    plus[0] = True
    minus[1] = True
    cue = torch.zeros((4, 4))
    cue[0] = 0.8
    cue[1] = 0.2

    evidence = front_depth_sign_evidence(
        predicted_depth=predicted,
        confidence=torch.ones((4, 4)),
        reference_depth=reference,
        plus_mask=plus,
        minus_mask=minus,
        cue_strength=cue,
        scale=1.0,
        config=DepthPriorSeedConfig(
            confidence_quantile=0.0,
            min_front_gap=0.01,
            min_front_gap_ratio=0.001,
        ),
        cue_support_threshold=0.05,
    )

    assert evidence.plus_pixels == 4
    assert evidence.minus_pixels == 4
    assert math.isclose(evidence.plus_mass, 3.2, rel_tol=1.0e-6)
    assert math.isclose(evidence.minus_mass, 0.8, rel_tol=1.0e-6)
    assert math.isclose(evidence.plus_fraction, 0.8, rel_tol=1.0e-6)
    assert int(evidence.front_mask.sum()) == 8


def test_front_depth_evidence_rejects_surfaces_behind_reference():
    evidence = front_depth_sign_evidence(
        predicted_depth=torch.full((4, 4), 3.0),
        confidence=torch.ones((4, 4)),
        reference_depth=torch.full((4, 4), 2.0),
        plus_mask=torch.ones((4, 4), dtype=torch.bool),
        minus_mask=torch.zeros((4, 4), dtype=torch.bool),
        cue_strength=torch.ones((4, 4)),
        scale=1.0,
        config=DepthPriorSeedConfig(confidence_quantile=0.0),
    )

    assert evidence.total_pixels == 0
    assert evidence.total_mass == 0.0


def test_top_magnitude_mask_keeps_only_requested_candidate_tail():
    score = torch.tensor([[4.0, 2.0], [-3.0, -1.0]])
    candidate = torch.ones((2, 2), dtype=torch.bool)

    kept, threshold = retain_top_magnitude_mask(
        score, candidate, quantile=0.5
    )

    assert threshold == 2.5
    assert kept.tolist() == [[True, False], [True, False]]


def test_q_weighted_upsampled_sam_score_keeps_every_signed_magnitude():
    score = torch.tensor([[4.0, 0.25], [-3.0, -0.10]])
    cue = torch.tensor([[1.0, 0.5], [0.25, 0.0]])

    weighted, normalization = q_weighted_upsampled_signed_sam_score(
        score,
        cue,
        height=2,
        width=2,
    )

    assert normalization == pytest.approx(4.0)
    torch.testing.assert_close(
        weighted,
        torch.tensor([[1.0, 0.03125], [-0.1875, 0.0]]),
    )
    assert (weighted > 0.0).tolist() == [[True, True], [False, False]]


def test_panel7_visible_support_excludes_values_rendered_as_black():
    score = torch.tensor(
        [[0.0, 0.49 / 255.0, 0.51 / 255.0], [-0.49 / 255.0, -0.51 / 255.0, 1.0]]
    )

    positive, negative = panel7_visible_signed_support(score)

    assert positive.tolist() == [[False, False, True], [False, False, True]]
    assert negative.tolist() == [[False, False, False], [False, True, False]]


def test_threshold_signed_support_requires_strict_signed_magnitude():
    score = torch.tensor([[-0.2, -0.1, 0.0, 0.1, 0.2]])

    positive, negative = threshold_signed_support(score, threshold=0.1)

    assert positive.tolist() == [[False, False, False, False, True]]
    assert negative.tolist() == [[True, False, False, False, False]]


def test_learned_gaussian_coverage_uses_current_position_scale_and_maturity():
    proposals = torch.tensor([[1.05, 0.0, 0.0], [2.0, 0.0, 0.0]])
    current_centers = torch.tensor([[1.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    current_scaling = torch.tensor([[0.04, 0.02, 0.02], [1.0, 1.0, 1.0]])

    uncovered = uncovered_by_learned_gaussian_support(
        proposals,
        current_centers,
        current_scaling,
        learned_update_counts=torch.tensor([4, 3]),
        eligible_learned=torch.tensor([True, True]),
        min_updates=4,
        support_sigma=2.0,
    )

    assert uncovered.tolist() == [False, True]


def test_closed_or_immature_gaussian_does_not_block_a_proposal():
    proposals = torch.tensor([[0.0, 0.0, 0.0]])
    centers = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    scaling = torch.full((2, 3), 0.1)

    uncovered = uncovered_by_learned_gaussian_support(
        proposals,
        centers,
        scaling,
        learned_update_counts=torch.tensor([100, 3]),
        eligible_learned=torch.tensor([False, True]),
        min_updates=4,
        support_sigma=2.0,
    )

    assert uncovered.tolist() == [True]


def test_robust_scale_fit_rejects_large_depth_outliers():
    predicted = torch.full((8, 8), 2.0)
    reference = torch.full((8, 8), 5.0)
    reference[0, :4] = 100.0

    fit = fit_reference_depth_scale(
        predicted, reference, torch.ones_like(predicted, dtype=torch.bool)
    )

    assert math.isclose(fit.scale, 2.5, rel_tol=1.0e-6)
    assert fit.samples == 64
    assert fit.inliers == 60
    assert fit.median_absolute_relative_error < 1.0e-6


def test_metric_anchor_scale_samples_native_depth_bilinearly():
    rows = torch.arange(8, dtype=torch.float32)[:, None]
    cols = torch.arange(8, dtype=torch.float32)[None, :]
    predicted = 1.0 + 0.1 * cols + 0.2 * rows
    pixels = torch.tensor(
        [[x + 0.25, y + 0.50] for y in range(4) for x in range(4)],
        dtype=torch.float32,
    )
    sampled = 1.0 + 0.1 * pixels[:, 0] + 0.2 * pixels[:, 1]

    fit = fit_metric_anchor_depth_scale(
        predicted,
        pixels,
        sampled * 2.75,
    )

    assert fit.samples == 16
    assert fit.inliers == 16
    assert math.isclose(fit.scale, 2.75, rel_tol=1.0e-6)
    assert fit.median_absolute_relative_error < 1.0e-6
    assert metric_anchor_median_relative_error(
        predicted, pixels, sampled * 2.75, scale=fit.scale
    ) < 1.0e-6


def test_metric_anchor_scale_rejects_invalid_coordinates_and_depth_outliers():
    predicted = torch.full((8, 8), 2.0)
    valid_pixels = torch.tensor(
        [[float(x), float(y)] for y in range(4) for x in range(5)]
    )
    pixels = torch.cat((valid_pixels, torch.tensor([[-1.0, 2.0], [9.0, 2.0]])))
    targets = torch.cat((torch.full((20,), 5.0), torch.tensor([5.0, 5.0])))
    targets[:3] = 100.0

    fit = fit_metric_anchor_depth_scale(predicted, pixels, targets)

    assert fit.samples == 20
    assert fit.inliers == 17
    assert math.isclose(fit.scale, 2.5, rel_tol=1.0e-6)


def test_reference_anchors_exclude_change_and_transparent_pixels():
    depth = torch.ones((4, 4))
    alpha = torch.ones((4, 4))
    cue = torch.zeros((4, 4))
    alpha[0, 0] = 0.2
    cue[1, 1] = 0.8
    cue[2, 2] = 0.2

    mask = reference_scale_anchor_mask(
        depth,
        alpha,
        cue,
        config=DepthPriorSeedConfig(erosion_pixels=0),
    )

    assert int(mask.sum()) == 13
    assert not mask[0, 0]
    assert not mask[1, 1]
    assert not mask[2, 2]


def test_depth_seed_requires_new_mask_and_front_of_reference_surface():
    height = width = 12
    predicted = torch.full((height, width), 1.8)
    reference = torch.full((height, width), 4.0)
    confidence = torch.ones((height, width))
    new = torch.zeros((height, width), dtype=torch.bool)
    new[2:10, 2:10] = True
    # This half is behind the reference after the x2 scale alignment.
    predicted[2:10, 6:10] = 2.2
    K = torch.tensor(
        [[10.0, 0.0, 6.0], [0.0, 10.0, 6.0], [0.0, 0.0, 1.0]]
    )
    w2c = torch.eye(4)

    batch = build_depth_prior_new_seeds(
        predicted_depth=predicted,
        confidence=confidence,
        reference_depth=reference,
        new_mask=new,
        K=K,
        w2c=w2c,
        scale=2.0,
        config=DepthPriorSeedConfig(
            erosion_pixels=0,
            confidence_quantile=0.0,
            min_front_gap=0.01,
            min_front_gap_ratio=0.001,
            sampling_stride=1,
            max_seeds=100,
        ),
    )

    assert batch.count == 32
    assert bool((batch.pixels_xy[:, 0] < 6).all())
    torch.testing.assert_close(batch.camera_depth, torch.full((32,), 3.6))
    torch.testing.assert_close(batch.xyz[:, 2], torch.full((32,), 3.6))


def test_depth_seed_can_use_strictly_positive_panel8_residual_without_margin():
    predicted = torch.tensor([[1.9, 2.0, 2.1], [1.9, 2.0, 2.1]])
    reference = torch.full((2, 3), 2.0)

    batch = build_depth_prior_new_seeds(
        predicted_depth=predicted,
        confidence=torch.ones_like(predicted),
        reference_depth=reference,
        new_mask=torch.ones_like(predicted, dtype=torch.bool),
        K=torch.eye(3),
        w2c=torch.eye(4),
        scale=1.0,
        config=DepthPriorSeedConfig(
            erosion_pixels=0,
            confidence_quantile=0.0,
            min_front_gap=0.0,
            min_front_gap_ratio=0.0,
            sampling_stride=1,
            max_seeds=3,
        ),
    )

    assert batch.pixels_xy.tolist() == [[0, 0], [0, 1]]
    torch.testing.assert_close(batch.camera_depth, torch.tensor([1.9, 1.9]))


def test_depth_seed_uses_soft_cue_as_primary_sampling_priority():
    size = 8
    confidence = torch.ones((size, size))
    confidence[1, 1] = 10.0
    priority = torch.zeros((size, size))
    priority[6, 6] = 0.9
    batch = build_depth_prior_new_seeds(
        predicted_depth=torch.ones((size, size)),
        confidence=confidence,
        reference_depth=torch.full((size, size), 2.0),
        new_mask=torch.ones((size, size), dtype=torch.bool),
        K=torch.tensor(
            [[10.0, 0.0, 4.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]]
        ),
        w2c=torch.eye(4),
        scale=1.0,
        config=DepthPriorSeedConfig(
            erosion_pixels=0,
            confidence_quantile=0.0,
            min_front_gap=0.01,
            min_front_gap_ratio=0.001,
            sampling_stride=8,
            max_seeds=1,
        ),
        sampling_priority=priority,
    )

    assert batch.pixels_xy.tolist() == [[6, 6]]


def test_unprojection_round_trip_with_translated_camera():
    K = torch.tensor(
        [[80.0, 0.0, 32.0], [0.0, 80.0, 24.0], [0.0, 0.0, 1.0]]
    )
    w2c = torch.eye(4)
    w2c[0, 3] = -0.5
    pixels = torch.tensor([[32.0, 24.0], [40.0, 20.0]])
    depth = torch.tensor([2.0, 4.0])

    world = unproject_camera_depth(pixels, depth, K, w2c)
    camera = world @ w2c[:3, :3].T + w2c[:3, 3]
    projected = camera @ K.T
    projected = projected[:, :2] / projected[:, 2:]

    torch.testing.assert_close(projected, pixels)
    torch.testing.assert_close(camera[:, 2], depth)
