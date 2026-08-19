import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.render_temporal_confusion_maps import aggregate_rows
from experiments.train_cue_temporal_rchange import (
    camera_json_to_w2c,
    physical_frame_name,
    validate_cue_cache,
)
from experiments.train_real_temporal_rchange import (
    support_metric_map,
    training_target,
)
from temporal import compute_growth_replay_regularization, compute_ssf_loss


def test_ssf_loss_matches_original_oscd_formula_and_gradient():
    cue = torch.tensor([[[0.1, 0.8], [1.2, 0.3]]], dtype=torch.float64)
    rendered = torch.tensor(
        [
            [[0.2, -0.1], [0.4, 0.7]],
            [[0.1, 0.3], [-0.2, 0.5]],
            [[-0.1, 0.2], [0.6, 0.0]],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    expected_rendered = rendered.detach().clone().requires_grad_(True)

    probability = torch.sigmoid(expected_rendered.mean(dim=0, keepdim=True))
    expected = (cue * (1.0 - probability)).mean()
    expected = expected + torch.log(probability.mean() ** 2 + 1.0)
    actual, parts = compute_ssf_loss(cue, rendered)

    expected.backward()
    actual.backward()
    assert torch.equal(actual, expected.detach())
    assert torch.equal(rendered.grad, expected_rendered.grad)
    assert torch.equal(parts["loss"], actual)


def test_local_ssf_regularization_matches_pixelwise_formula_and_gradient():
    cue = torch.tensor([[[0.2, 0.8], [1.4, 0.4]]], dtype=torch.float64)
    support = (cue / 2.0).clamp(0.0, 1.0)
    rendered = torch.tensor(
        [
            [[0.2, -0.1], [0.4, 0.7]],
            [[0.1, 0.3], [-0.2, 0.5]],
            [[-0.1, 0.2], [0.6, 0.0]],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    expected_rendered = rendered.detach().clone().requires_grad_(True)

    probability = torch.sigmoid(expected_rendered.mean(dim=0, keepdim=True))
    expected_detection = (cue * (1.0 - probability)).mean()
    expected_regularization = ((1.0 - support) * probability).mean()
    expected = expected_detection + 0.75 * expected_regularization
    actual, parts = compute_ssf_loss(
        cue,
        rendered,
        regularization_mode="local",
        local_support=support,
        regularization_weight=0.75,
    )

    expected.backward()
    actual.backward()
    assert torch.equal(actual, expected.detach())
    assert torch.equal(rendered.grad, expected_rendered.grad)
    assert torch.equal(parts["detection"], expected_detection.detach())
    assert torch.equal(parts["regularization"], expected_regularization.detach())


def test_local_ssf_regularization_does_not_couple_other_pixel_activation():
    cue = torch.zeros((1, 1, 2), dtype=torch.float64)
    support = torch.zeros_like(cue)

    def first_pixel_gradient(second_pixel_logit: float) -> float:
        rendered = torch.tensor(
            [[[0.0, second_pixel_logit]]] * 3,
            dtype=torch.float64,
            requires_grad=True,
        )
        loss, _ = compute_ssf_loss(
            cue,
            rendered,
            regularization_mode="local",
            local_support=support,
        )
        loss.backward()
        return float(rendered.grad[0, 0, 0])

    assert first_pixel_gradient(-4.0) == pytest.approx(first_pixel_gradient(4.0))


def test_previous_update_global_regularization_scales_original_global_term():
    cue = torch.tensor([[[0.2, 0.8], [1.4, 0.4]]], dtype=torch.float64)
    rendered = torch.tensor(
        [
            [[0.2, -0.1], [0.4, 0.7]],
            [[0.1, 0.3], [-0.2, 0.5]],
            [[-0.1, 0.2], [0.6, 0.0]],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    expected_rendered = rendered.detach().clone().requires_grad_(True)
    previous_growth = 0.125

    probability = torch.sigmoid(expected_rendered.mean(dim=0, keepdim=True))
    expected_detection = (cue * (1.0 - probability)).mean()
    expected_regularization = previous_growth * torch.log(
        probability.mean() ** 2 + 1.0
    )
    expected = expected_detection + expected_regularization
    actual, parts = compute_ssf_loss(
        cue,
        rendered,
        regularization_mode="previous_update_global",
        previous_growth=previous_growth,
    )

    expected.backward()
    actual.backward()
    assert torch.equal(actual, expected.detach())
    assert torch.equal(rendered.grad, expected_rendered.grad)
    assert torch.equal(parts["regularization"], expected_regularization.detach())


def test_local_plus_previous_update_global_matches_combined_formula():
    cue = torch.tensor([[[0.2, 0.8], [1.4, 0.4]]], dtype=torch.float64)
    support = (cue / 2.0).clamp(0.0, 1.0)
    rendered = torch.tensor(
        [
            [[0.2, -0.1], [0.4, 0.7]],
            [[0.1, 0.3], [-0.2, 0.5]],
            [[-0.1, 0.2], [0.6, 0.0]],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    expected_rendered = rendered.detach().clone().requires_grad_(True)
    previous_growth = 0.125

    probability = torch.sigmoid(expected_rendered.mean(dim=0, keepdim=True))
    expected_detection = (cue * (1.0 - probability)).mean()
    expected_local = ((1.0 - support) * probability).mean()
    expected_global = previous_growth * torch.log(probability.mean() ** 2 + 1.0)
    expected = expected_detection + expected_local + expected_global
    actual, parts = compute_ssf_loss(
        cue,
        rendered,
        regularization_mode="local_plus_previous_update_global",
        local_support=support,
        previous_growth=previous_growth,
    )

    expected.backward()
    actual.backward()
    assert torch.equal(actual, expected.detach())
    assert torch.equal(rendered.grad, expected_rendered.grad)
    assert torch.equal(
        parts["regularization"], (expected_local + expected_global).detach()
    )


@pytest.mark.parametrize("previous_growth", [-0.1, 1.1, float("nan")])
def test_previous_update_global_rejects_invalid_growth(previous_growth):
    with pytest.raises(ValueError):
        compute_ssf_loss(
            torch.zeros((1, 2, 2)),
            torch.zeros((3, 2, 2)),
            regularization_mode="previous_update_global",
            previous_growth=previous_growth,
        )


def test_growth_replay_regularization_matches_elementwise_formula_and_gradient():
    support = torch.tensor(
        [[[0.0, 0.25], [0.75, 1.0]]], dtype=torch.float64
    )
    growth = torch.tensor(
        [[[0.0, 0.2], [0.4, 0.8]]], dtype=torch.float64, requires_grad=True
    )
    probability = torch.tensor(
        [[[0.2, 0.4], [0.6, 0.8]]], dtype=torch.float64, requires_grad=True
    )
    expected_probability = probability.detach().clone().requires_grad_(True)
    expected = (growth.detach() * (1.0 - support) * expected_probability).mean()

    actual = compute_growth_replay_regularization(
        support, probability, growth
    )
    expected.backward()
    actual.backward()

    assert torch.equal(actual, expected.detach())
    assert torch.equal(probability.grad, expected_probability.grad)
    assert growth.grad is None


def test_growth_replay_regularization_rejects_mismatched_maps():
    with pytest.raises(ValueError):
        compute_growth_replay_regularization(
            torch.zeros((1, 2, 2)),
            torch.zeros((1, 2, 2)),
            torch.zeros((1, 2, 3)),
        )


@pytest.mark.parametrize(
    ("cue", "rendered"),
    [
        (torch.zeros(2, 2), torch.zeros(3, 2, 2)),
        (torch.zeros(1, 2, 2), torch.zeros(1, 2, 2)),
        (torch.zeros(1, 2, 3), torch.zeros(3, 2, 2)),
    ],
)
def test_ssf_loss_rejects_invalid_shapes(cue, rendered):
    with pytest.raises(ValueError):
        compute_ssf_loss(cue, rendered)


def test_training_and_support_use_explicit_cue_not_oracle_mask():
    cue = torch.tensor([[[0.2, 0.8], [0.6, 0.1]]])
    oracle = torch.ones_like(cue)
    view = SimpleNamespace(
        training_target=cue,
        support_map=cue,
        oracle_gt_mask=oracle,
    )

    assert training_target(view) is cue
    assert support_metric_map(view, 0.5).tolist() == [0, 1, 1, 0]


def test_camera_json_round_trip_matches_oscd_payload():
    angle = np.deg2rad(25.0)
    rotation = np.array(
        [[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0], [-np.sin(angle), 0.0, np.cos(angle)]]
    )
    position = np.array([1.0, -2.0, 3.0])
    w2c = camera_json_to_w2c(
        {"rotation": rotation.tolist(), "position": position.tolist()}
    )

    assert np.allclose(np.linalg.inv(w2c)[:3, :3], rotation)
    assert np.allclose(np.linalg.inv(w2c)[:3, 3], position)


def test_cue_cache_validation_requires_matching_oscd_metadata(tmp_path):
    base = tmp_path / "base.ply"
    base.write_bytes(b"base")
    cache = tmp_path / "cache"
    cache.mkdir()
    import hashlib

    metadata = {
        "resolution": 4,
        "reference_ply_sha256": hashlib.sha256(b"base").hexdigest(),
        "candidate_map_definition": "O-SCD pixel cue + SAM2.1 Hiera Tiny feature cue",
    }
    (cache / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    assert validate_cue_cache(cache, base, 4) == metadata
    with pytest.raises(ValueError, match="resolution"):
        validate_cue_cache(cache, base, 8)


def test_physical_frame_name_removes_only_sequence_order_prefix():
    assert physical_frame_name("order01_scene_change2_frame_000087.png") == "scene_change2_frame_000087"
    assert physical_frame_name("scene_change2_frame_000087.png") == "scene_change2_frame_000087"


def test_confusion_summary_reports_oscd_aligned_frame_mean():
    base = {
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0,
        "pred_positive": 0,
        "gt_positive": 0,
        "pixels": 1,
    }
    rows = [
        {**base, "tp": 9, "fn": 1, "gt_positive": 10, "pixels": 10, "iou": 0.9, "f1": 18 / 19},
        {**base, "tp": 1, "fn": 9, "gt_positive": 10, "pixels": 10, "iou": 0.1, "f1": 2 / 11},
    ]

    summary = aggregate_rows(rows, "all")

    assert summary["mean_frame_iou"] == pytest.approx(0.5)
    assert summary["mean_frame_f1"] == pytest.approx((18 / 19 + 2 / 11) / 2)
