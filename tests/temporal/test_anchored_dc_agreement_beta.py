from __future__ import annotations

import torch

from temporal.anchored_dc_agreement_beta import (
    AnchoredDCAgreementBetaConfig,
    AnchoredDCAgreementBetaFilter,
)


def make_filter(*, confirmation_views: int = 3):
    return AnchoredDCAgreementBetaFilter(
        1,
        AnchoredDCAgreementBetaConfig(
            stable_flip_prior=1.0,
            stable_keep_prior=10.0,
            reset_flip_prior=1.0,
            reset_keep_prior=1.0,
            bayes_factor_threshold=2.0,
            confirmation_views=confirmation_views,
            directional_margin=0.0,
        ),
    )


def update(
    filter_,
    *,
    cue_positive: float,
    expected: float,
    active: bool,
    timestamp: int,
):
    return filter_.update(
        torch.tensor([cue_positive]),
        torch.tensor([1.0 - cue_positive]),
        expected_change=torch.tensor([expected]),
        current_active=torch.tensor([active]),
        timestamp=timestamp,
    )


def test_candidate_uses_detached_start_anchor_while_current_dc_adapts():
    filter_ = make_filter()

    started = update(
        filter_, cue_positive=1.0, expected=0.0, active=False, timestamp=0
    )
    continued = update(
        filter_, cue_positive=1.0, expected=0.9, active=False, timestamp=1
    )
    committed = update(
        filter_, cue_positive=1.0, expected=1.0, active=False, timestamp=2
    )

    assert started.candidate_started.tolist() == [True]
    assert started.candidate_anchor.tolist() == [0.0]
    assert torch.allclose(
        continued.current_expected_change, torch.tensor([0.9])
    )
    assert continued.candidate_anchor.tolist() == [0.0]
    assert continued.delta_flip.tolist() == [1.0]
    assert continued.delta_keep.tolist() == [0.0]
    assert continued.candidate_committed.tolist() == [False]
    assert committed.candidate_committed.tolist() == [True]
    assert committed.candidate_support_observations.tolist() == [3]
    assert committed.active_after.tolist() == [True]
    assert filter_.candidate_anchor.tolist() == [0.0]


def test_opposite_direction_view_rejects_candidate_even_if_score_stays_positive():
    filter_ = make_filter()

    started = update(
        filter_, cue_positive=1.0, expected=0.0, active=False, timestamp=0
    )
    rejected = update(
        filter_, cue_positive=0.0, expected=0.0, active=False, timestamp=1
    )

    assert started.candidate_started.tolist() == [True]
    assert rejected.directional_support.tolist() == [False]
    assert rejected.candidate_rejected.tolist() == [True]
    assert rejected.candidate_active.tolist() == [False]
    assert filter_.candidate_anchor.tolist() == [0.0]


def test_open_to_close_direction_is_confirmed_against_open_anchor():
    filter_ = make_filter()
    active = True
    result = None

    for timestamp, expected in enumerate((1.0, 0.2, 0.0)):
        result = update(
            filter_,
            cue_positive=0.0,
            expected=expected,
            active=active,
            timestamp=timestamp,
        )

    assert result is not None
    assert result.candidate_anchor.tolist() == [1.0]
    assert result.candidate_support_observations.tolist() == [3]
    assert result.candidate_committed.tolist() == [True]
    assert result.active_after.tolist() == [False]
