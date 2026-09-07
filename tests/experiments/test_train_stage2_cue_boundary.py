import numpy as np
import pytest

from experiments.train_stage2_cue_boundary import (
    deterministic_train_validation_split,
    hard_metrics_from_histograms,
    parse_args,
)


def test_ordered_split_reserves_every_fifth_frame() -> None:
    training, validation = deterministic_train_validation_split(12, validation_stride=5)

    assert np.flatnonzero(validation).tolist() == [4, 9]
    assert np.flatnonzero(training).tolist() == [0, 1, 2, 3, 5, 6, 7, 8, 10, 11]


def test_hard_histogram_metrics_use_per_frame_tau() -> None:
    counts = np.asarray([[5, 5], [5, 5]], dtype=np.float32)
    gt = np.asarray([[0, 5], [0, 5]], dtype=np.float32)

    result = hard_metrics_from_histograms(counts, gt, np.asarray([0.5, 0.9]))

    assert result["mean_frame_iou"] == 0.5
    assert result["counts"] == {"tp": 5.0, "fp": 0.0, "fn": 5.0}


def test_parser_uses_heuristic_warm_start() -> None:
    args = parse_args([])

    assert args.input_bins == 64
    assert args.teacher_mode == "online_binary_masks"
    assert args.initial_tau == 0.25
    assert args.initial_width == 0.10
    assert args.head_warmup_epochs == 40


def test_parser_rejects_initial_boundary_outside_bounds() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--initial-tau", "0.9"])


def test_parser_rejects_invalid_sigmoid_edge_probability() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--edge-probability", "0.5"])
