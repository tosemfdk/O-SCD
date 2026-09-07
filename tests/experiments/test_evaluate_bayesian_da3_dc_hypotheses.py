from argparse import Namespace

import numpy as np
import pytest

from experiments.evaluate_bayesian_da3_dc_hypotheses import (
    CONDITIONS,
    aggregate_binary,
    aggregate_cue_diagnostics,
    binary_metrics,
    cue_diagnostics,
    viewer_arguments,
)


def test_binary_metrics_matches_expected_confusion() -> None:
    metrics = binary_metrics(
        np.array([[True, True], [False, False]]),
        np.array([[True, False], [True, False]]),
    )

    assert metrics == {
        "tp": 1,
        "tn": 1,
        "fp": 1,
        "fn": 1,
        "iou": pytest.approx(1 / 3),
        "f1": pytest.approx(0.5),
    }


def test_cue_diagnostics_separates_learning_ceiling_and_black_occlusion() -> None:
    learned = np.array([[0.4, 0.1], [0.7, 0.0]])
    cue = np.array([[1.0, 0.0], [0.9, 0.1]])
    white_black = np.array([[0.6, 0.0], [0.4, 0.0]])
    white_open = np.array([[0.8, 0.0], [0.9, 0.0]])
    gt = np.array([[True, False], [True, False]])

    row = cue_diagnostics(learned, cue, white_black, white_open, gt)

    assert row["cue_high_pixels"] == 2
    assert row["learned_high_sum"] == pytest.approx(1.1)
    assert row["white_black_high_below_threshold"] == 1
    assert row["black_occlusion_high_sum"] == pytest.approx(0.7)
    assert row["gt_white_black_below_threshold"] == 1


def test_aggregates_report_frame_mean_and_pixel_weighted_diagnostics() -> None:
    binary_rows = [
        {"gt_tp": 1, "gt_tn": 1, "gt_fp": 0, "gt_fn": 0, "gt_iou": 1.0, "gt_f1": 1.0},
        {"gt_tp": 0, "gt_tn": 0, "gt_fp": 1, "gt_fn": 1, "gt_iou": 0.0, "gt_f1": 0.0},
    ]
    aggregate = aggregate_binary(binary_rows, "gt")
    assert aggregate["mean_frame_iou"] == pytest.approx(0.5)
    assert aggregate["aggregate_iou"] == pytest.approx(1 / 3)

    base = {
        "pixels": 4,
        "cue_high_pixels": 2,
        "cue_low_pixels": 2,
        "gt_positive_pixels": 2,
        "learned_high_sum": 1.0,
        "learned_high_positive": 1,
        "learned_low_sum": 0.2,
        "cue_absolute_error_sum": 1.0,
        "white_black_high_sum": 1.5,
        "white_black_high_positive": 2,
        "white_black_high_below_threshold": 0,
        "white_open_only_high_sum": 1.8,
        "black_occlusion_high_sum": 0.3,
        "gt_white_black_below_threshold": 0,
        "representation_updates": 4,
        "dc_current_updates": 2,
        "dc_view_age_sum": 6,
    }
    cue_aggregate = aggregate_cue_diagnostics([base, base])
    assert cue_aggregate["learned_mean_on_high_cue"] == pytest.approx(0.5)
    assert cue_aggregate["actual_dc_current_view_fraction"] == pytest.approx(0.5)
    assert cue_aggregate["mean_dc_view_age"] == pytest.approx(1.5)


def test_condition_arguments_change_only_dc_ablation_axes(tmp_path) -> None:
    boundary = tmp_path / "boundary.json"
    boundary.write_text(
        '{"remap":"sigmoid","edge_probability":0.05,"frames":{"x":{"tau":0.2,"width":0.1}}}'
    )
    seed = tmp_path / "seed.pt"
    seed.touch()
    sam = tmp_path / "sam"
    sam.mkdir()
    args = Namespace(
        max_frames=1,
        updates_per_frame=2,
        boundary_json=boundary,
        da3_seed_checkpoint=seed,
        sam_sign_trace_root=sam,
    )

    baseline = viewer_arguments(args, CONDITIONS["A_baseline"])
    combined = viewer_arguments(args, CONDITIONS["G_all"])

    assert baseline.representation_cue_amplitude == 1.0
    assert baseline.seed_dc_supervision == "joint"
    assert baseline.dc_replay_mode == "sampled"
    assert baseline.da3_detector_cue_source == "part19_binary"
    assert combined.representation_cue_amplitude == 2.0
    assert combined.seed_dc_supervision == "projected_bce"
    assert combined.dc_replay_mode == "current"
    for key in (
        "cue_fusion",
        "product_exponent",
        "cue_mode",
        "cue_scale",
        "cue_remap",
        "da3_detector_cue_source",
        "representation_updates",
        "current_view_probability",
        "train_never_open_geometry",
    ):
        assert getattr(baseline, key) == getattr(combined, key)
