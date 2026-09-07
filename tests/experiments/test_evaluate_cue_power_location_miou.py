import torch

from experiments.evaluate_cue_power_location_miou import (
    metrics_from_counts,
    threshold_counts,
)


def test_threshold_counts_uses_strict_binary_boundary() -> None:
    cue = torch.tensor([[0.1, 0.25], [0.3, 0.9]])
    gt = torch.tensor([[False, True], [True, False]])
    thresholds = torch.tensor([0.25, 0.5])

    counts = threshold_counts(cue, gt, thresholds)

    assert counts["tp"].tolist() == [1, 0]
    assert counts["fp"].tolist() == [1, 1]
    assert counts["fn"].tolist() == [1, 2]
    assert counts["tn"].tolist() == [1, 1]


def test_metrics_from_counts_matches_foreground_iou_contract() -> None:
    metrics = metrics_from_counts(tp=3, fp=1, fn=2)

    assert metrics["iou"] == 0.5
    assert metrics["precision"] == 0.75
    assert metrics["recall"] == 0.6
    assert metrics["f1"] == 2 * 0.75 * 0.6 / (0.75 + 0.6)
