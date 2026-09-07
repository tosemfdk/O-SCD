import numpy as np
import pytest
import torch

from experiments.build_causal_da3_seed_replay import (
    causal_window,
    choose_depth_new_sign,
    localization_depth_anchors,
    summarize_depth_scale_comparison,
)


@pytest.mark.parametrize(
    ("frame", "size", "expected"),
    [
        (1, 8, [1]),
        (3, 8, [1, 2, 3]),
        (9, 8, [2, 3, 4, 5, 6, 7, 8, 9]),
    ],
)
def test_causal_window_never_contains_a_future_frame(frame, size, expected):
    window = causal_window(frame, size)

    assert window == expected
    assert max(window) == frame


def test_depth_sign_waits_until_front_evidence_is_concentrated():
    sign, confidence = choose_depth_new_sign(
        plus_mass=60.0,
        minus_mass=40.0,
        plus_pixels=60,
        minus_pixels=40,
        concentration=0.7,
        min_pixels=32,
    )

    assert sign is None
    assert confidence is None


def test_depth_sign_selects_dominant_polarity():
    sign, confidence = choose_depth_new_sign(
        plus_mass=25.0,
        minus_mass=75.0,
        plus_pixels=20,
        minus_pixels=60,
        concentration=0.7,
        min_pixels=32,
    )

    assert sign == "-"
    assert confidence == pytest.approx(0.75)


def test_localization_anchors_use_fixed_pose_inliers_and_native_da3_grid():
    image_K = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]
    )
    depth_K = torch.tensor(
        [[50.0, 0.0, 25.0], [0.0, 50.0, 20.0], [0.0, 0.0, 1.0]]
    )
    points3d = np.asarray(
        [
            [0.2, 0.0, 2.0],
            [-0.4, 0.2, 4.0],
            [0.2, 0.0, 2.0],  # Same current keypoint/landmark through another ref.
            [0.0, 0.0, 3.0],  # Descriptor mismatch rejected by reprojection.
        ],
        dtype=np.float32,
    )
    points2d = np.asarray(
        [[60.0, 40.0], [40.0, 45.0], [60.0, 40.0], [70.0, 70.0]],
        dtype=np.float32,
    )

    anchors = localization_depth_anchors(
        points2d=points2d,
        points3d=points3d,
        image_K=image_K,
        depth_K=depth_K,
        w2c=torch.eye(4),
        depth_height=40,
        depth_width=50,
        max_reprojection_error_px=2.0,
    )

    assert anchors.raw_matches == 4
    assert anchors.count == 2
    torch.testing.assert_close(
        anchors.pixels_xy, torch.tensor([[30.0, 20.0], [20.0, 22.5]])
    )
    torch.testing.assert_close(anchors.camera_depth, torch.tensor([2.0, 4.0]))
    assert float(anchors.reprojection_error_px.max()) < 1.0e-6


def test_depth_scale_comparison_uses_paired_metric_anchor_errors():
    rows = [
        {
            "localization_xfeat_scale_median_relative_error": 0.02,
            "localization_reference_scale_median_relative_error": 0.04,
        },
        {
            "localization_xfeat_scale_median_relative_error": 0.03,
            "localization_reference_scale_median_relative_error": 0.02,
        },
        {
            "localization_xfeat_scale_median_relative_error": None,
            "localization_reference_scale_median_relative_error": None,
        },
    ]

    summary = summarize_depth_scale_comparison(rows)

    assert summary["frames"] == 2
    assert summary["xfeat_error_median"] == pytest.approx(0.025)
    assert summary["reference_error_median"] == pytest.approx(0.03)
    assert summary["frames_xfeat_better"] == 1
