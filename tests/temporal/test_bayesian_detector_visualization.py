import math

import pytest
import torch

from temporal.bayesian_detector_visualization import (
    detector_visual_state,
    black_green_yellow_red_heatmap,
    normalized_log_bayes_factor,
)


def test_normalized_log_bayes_factor_uses_commit_threshold() -> None:
    threshold = 30.0
    values = torch.tensor([-2.0, 0.0, math.log(threshold) / 2.0, math.log(threshold)])

    progress = normalized_log_bayes_factor(values, threshold)

    assert torch.allclose(progress, torch.tensor([0.0, 0.0, 0.5, 1.0]))


def test_lifecycle_colors_show_only_committed_state() -> None:
    threshold = 30.0
    state = detector_visual_state(
        current_active=torch.tensor([False, True, False, True]),
        ever_opened=torch.tensor([False, True, True, True]),
        candidate_active=torch.tensor([False, False, False, True]),
        last_log_bayes_factor=torch.tensor(
            [0.0, 0.0, 0.0, math.log(threshold) / 2.0]
        ),
        bayes_factor_threshold=threshold,
    )

    assert torch.equal(state.colors[0], torch.tensor([0.0, 0.0, 0.0]))
    assert torch.equal(state.colors[1], torch.tensor([0.0, 1.0, 0.0]))
    assert torch.equal(state.colors[2], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.equal(state.colors[3], torch.tensor([0.0, 1.0, 0.0]))
    assert state.never_open.tolist() == [True, False, False, False]
    assert state.open.tolist() == [False, True, False, True]
    assert state.closed.tolist() == [False, False, True, False]
    assert state.uncertain.tolist() == [False, False, False, True]
    assert state.instability.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.5])


def test_black_green_yellow_red_heatmap_is_continuous_through_anchors() -> None:
    colors = black_green_yellow_red_heatmap(
        torch.tensor([-1.0, 1.0 / 6.0, 1.0 / 3.0, 0.5, 2.0 / 3.0, 5.0 / 6.0, 2.0])
    )

    assert torch.allclose(
        colors,
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 1.0, 0.0],
                [0.5, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.5, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ),
        atol=1.0e-7,
        rtol=0.0,
    )


def test_open_requires_an_ever_opened_lifecycle() -> None:
    with pytest.raises(ValueError, match="opened at least once"):
        detector_visual_state(
            current_active=torch.tensor([True]),
            ever_opened=torch.tensor([False]),
            candidate_active=torch.tensor([False]),
            last_log_bayes_factor=torch.tensor([0.0]),
            bayes_factor_threshold=30.0,
        )
