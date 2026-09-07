import copy

import pytest
import torch

from experiments.train_stage2_cue_boundary_causal import train_causal_prequential
from temporal.learnable_cue_boundary import HistogramBoundaryMLP


def _statistics(frames: int = 4):
    input_counts = torch.ones((frames, 4))
    loss_counts = torch.ones((frames, 8)) * 10.0
    teacher_sums = torch.zeros_like(loss_counts)
    teacher_sums[:, 5:] = 8.0
    return input_counts, loss_counts, teacher_sums


def test_causal_boundary_starts_fixed_and_exposes_teacher_after_prediction():
    input_counts, loss_counts, teacher_sums = _statistics()
    model = HistogramBoundaryMLP(
        bins=4, hidden=(4, 2), initial_tau=0.25, initial_width=0.10
    )

    result = train_causal_prequential(
        model=model,
        input_counts=input_counts,
        loss_counts=loss_counts,
        teacher_sums=teacher_sums,
        segment_names=("s",) * 4,
    )

    assert result.tau[0] == pytest.approx(0.25, abs=1.0e-7)
    assert result.width[0] == pytest.approx(0.10, abs=1.0e-7)
    assert result.audit["first_frame_uses_fixed_initial_boundary"] is True
    assert result.audit["future_teacher_accesses"] == 0
    assert result.audit["maximum_teacher_index_before_prediction"] == [-1, 0, 1, 2]


def test_future_teacher_cannot_change_any_existing_prequential_prediction():
    torch.manual_seed(3)
    input_counts, loss_counts, teacher_sums = _statistics()
    initial = HistogramBoundaryMLP(
        bins=4, hidden=(4, 2), initial_tau=0.25, initial_width=0.10
    )
    future_changed = teacher_sums.clone()
    future_changed[-1] = loss_counts[-1] - teacher_sums[-1]

    first = train_causal_prequential(
        model=copy.deepcopy(initial),
        input_counts=input_counts,
        loss_counts=loss_counts,
        teacher_sums=teacher_sums,
        segment_names=("s",) * 4,
    )
    second = train_causal_prequential(
        model=copy.deepcopy(initial),
        input_counts=input_counts,
        loss_counts=loss_counts,
        teacher_sums=future_changed,
        segment_names=("s",) * 4,
    )

    # teacher_3 is exposed only after the last prediction, so changing it
    # cannot affect tau/width for frames 0..3.
    assert first.tau.tolist() == second.tau.tolist()
    assert first.width.tolist() == second.width.tolist()
