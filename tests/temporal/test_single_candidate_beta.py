from __future__ import annotations

import math

import pytest
import torch
from torch import nn
from types import SimpleNamespace

from temporal.bayesian_lifespan_controller import (
    BayesianLifespanController,
    LifespanAction,
)
from temporal.bernoulli_bocd import BernoulliBOCDConfig
from temporal.change_model import TemporalChangeModel
from temporal.single_candidate_beta import (
    SingleCandidateBetaConfig,
    SingleCandidateBetaFilter,
)


def _filter(
    *,
    threshold: float = 100.0,
    rows: int = 1,
    support_views: int = 1,
    strict_support: bool = False,
):
    return SingleCandidateBetaFilter(
        rows,
        SingleCandidateBetaConfig(
            prior_a=1.0,
            prior_b=1.0,
            bayes_factor_threshold=threshold,
            min_evidence_mass=1e-6,
            min_candidate_support_views=support_views,
            require_candidate_support_each_view=strict_support,
        ),
        dtype=torch.float64,
    )


def _update(
    tracker,
    positive,
    negative,
    timestamp,
    *,
    mass=None,
    rows=None,
    support=None,
    log_threshold=None,
):
    positive = torch.as_tensor(positive, dtype=torch.float64).flatten()
    negative = torch.as_tensor(negative, dtype=torch.float64).flatten()
    return tracker.update(
        positive,
        negative,
        total_mass=positive + negative if mass is None else mass,
        candidate_support=support,
        log_bayes_factor_threshold=log_threshold,
        row_indices=rows,
        timestamp=timestamp,
    )


def test_config_uses_raw_bayes_factor_and_logs_it_once():
    config = SingleCandidateBetaConfig(bayes_factor_threshold=30.0)
    assert config.log_bayes_factor_threshold == pytest.approx(math.log(30.0))
    with pytest.raises(ValueError, match="> 1"):
        SingleCandidateBetaConfig(bayes_factor_threshold=1.0)


def test_first_observation_initializes_stable_beta_without_changepoint():
    tracker = _filter()

    result = _update(tracker, [0.8], [0.2], 4)

    assert result.initialized.tolist() == [True]
    assert result.candidate_committed.tolist() == [False]
    assert result.changepoint_probability.tolist() == [0.0]
    assert tracker.stable_a.tolist() == pytest.approx([1.8])
    assert tracker.stable_b.tolist() == pytest.approx([1.2])
    assert tracker.stable_run_start.tolist() == [4]
    assert result.change_probability.tolist() == pytest.approx([0.6])


def test_candidate_freezes_stable_beta_then_rejection_merges_entire_block():
    tracker = _filter()
    _update(tracker, [1.0], [0.0], 0)
    stable_before = (tracker.stable_a.clone(), tracker.stable_b.clone())

    started = _update(tracker, [0.0], [1.0], 1)
    assert started.candidate_started.tolist() == [True]
    assert started.candidate_active.tolist() == [True]
    assert torch.equal(tracker.stable_a, stable_before[0])
    assert torch.equal(tracker.stable_b, stable_before[1])

    rejected = _update(tracker, [1.0], [0.0], 2)
    assert rejected.candidate_rejected.tolist() == [True]
    assert rejected.candidate_active.tolist() == [False]
    assert rejected.candidate_log_bayes_factor.item() == pytest.approx(0.0)
    assert tracker.stable_a.tolist() == pytest.approx([3.0])
    assert tracker.stable_b.tolist() == pytest.approx([2.0])
    assert tracker.stable_visible_observations.tolist() == [3]


def test_two_surprising_failures_commit_reset_at_candidate_start():
    tracker = _filter(threshold=10.0)
    tracker.initialized[0] = True
    tracker.stable_a[0] = 10.0
    tracker.stable_b[0] = 1.0
    tracker.stable_total_evidence[0] = 9.0
    tracker.stable_run_start[0] = 0
    tracker.stable_visible_observations[0] = 9

    first = _update(tracker, [0.0], [1.0], 10)
    assert first.candidate_started.tolist() == [True]
    assert first.candidate_committed.tolist() == [False]
    assert first.candidate_log_bayes_factor.item() == pytest.approx(
        math.log(0.5 / (1.0 / 11.0))
    )

    second = _update(tracker, [0.0], [1.0], 11)
    assert second.candidate_committed.tolist() == [True]
    assert second.changepoint_probability.tolist() == [1.0]
    assert second.estimated_run_start.tolist() == [10]
    assert second.candidate_duration.tolist() == [2]
    assert tracker.stable_a.tolist() == pytest.approx([1.0])
    assert tracker.stable_b.tolist() == pytest.approx([3.0])
    assert tracker.stable_visible_observations.tolist() == [2]
    assert tracker.candidate_active.tolist() == [False]




def test_log_threshold_override_can_commit_only_selected_row_early():
    tracker = _filter(threshold=30.0, rows=2)
    tracker.initialized[:] = True
    tracker.stable_a[:] = 10.0
    tracker.stable_b[:] = 1.0
    tracker.stable_total_evidence[:] = 9.0
    tracker.stable_run_start[:] = 0
    tracker.stable_visible_observations[:] = 9

    _update(
        tracker,
        [0.0, 0.0],
        [1.0, 1.0],
        10,
        log_threshold=torch.log(torch.tensor([10.0, 30.0], dtype=torch.float64)),
    )
    second = _update(
        tracker,
        [0.0, 0.0],
        [1.0, 1.0],
        11,
        log_threshold=torch.log(torch.tensor([10.0, 30.0], dtype=torch.float64)),
    )

    assert second.candidate_committed.tolist() == [True, False]
    assert second.candidate_active.tolist() == [False, True]

    third = _update(
        tracker,
        [0.0],
        [1.0],
        12,
        rows=torch.tensor([1]),
        log_threshold=math.log(30.0),
    )

    assert third.candidate_committed.tolist() == [True]


def test_log_threshold_override_validation_happens_before_mutation():
    tracker = _filter(rows=1)
    _update(tracker, [1.0], [0.0], 0)
    before = tracker.state_dict()

    with pytest.raises(ValueError, match="> 0"):
        _update(tracker, [0.0], [1.0], 1, log_threshold=0.0)

    for name, expected in before.items():
        assert torch.equal(getattr(tracker, name), expected), name


def test_commit_waits_for_requested_number_of_supporting_views():
    tracker = _filter(threshold=2.0, support_views=3, strict_support=True)
    tracker.initialized[0] = True
    tracker.stable_a[0] = 10.0
    tracker.stable_b[0] = 1.0
    tracker.stable_run_start[0] = 0
    tracker.stable_visible_observations[0] = 9

    first = _update(tracker, [0.0], [1.0], 10, support=torch.tensor([True]))
    second = _update(tracker, [0.0], [1.0], 11, support=torch.tensor([True]))
    third = _update(tracker, [0.0], [1.0], 12, support=torch.tensor([True]))

    assert first.candidate_started.tolist() == [True]
    assert second.candidate_continued.tolist() == [True]
    assert first.candidate_log_bayes_factor.item() > math.log(2.0)
    assert second.candidate_log_bayes_factor.item() > math.log(2.0)
    assert first.candidate_committed.tolist() == [False]
    assert second.candidate_committed.tolist() == [False]
    assert third.candidate_committed.tolist() == [True]
    assert third.candidate_support_observations.tolist() == [3]


def test_strict_support_rejects_live_candidate_on_opposite_view():
    tracker = _filter(threshold=2.0, support_views=3, strict_support=True)
    tracker.initialized[0] = True
    tracker.stable_a[0] = 10.0
    tracker.stable_b[0] = 1.0
    tracker.stable_run_start[0] = 0
    tracker.stable_visible_observations[0] = 9

    started = _update(
        tracker, [0.0], [1.0], 10, support=torch.tensor([True])
    )
    rejected = _update(
        tracker, [0.0], [1.0], 11, support=torch.tensor([False])
    )

    assert started.candidate_started.tolist() == [True]
    assert rejected.candidate_rejected.tolist() == [True]
    assert rejected.candidate_committed.tolist() == [False]
    assert tracker.candidate_active.tolist() == [False]
    assert tracker.stable_a.tolist() == pytest.approx([10.0])
    assert tracker.stable_b.tolist() == pytest.approx([3.0])


def test_beta_concentration_affects_repeated_but_not_first_failure():
    da = torch.tensor([0.0], dtype=torch.float64)
    one_failure = torch.tensor([1.0], dtype=torch.float64)
    two_failures = torch.tensor([2.0], dtype=torch.float64)
    weak_a = torch.tensor([10.0], dtype=torch.float64)
    weak_b = torch.tensor([1.0], dtype=torch.float64)
    strong_a = torch.tensor([100.0], dtype=torch.float64)
    strong_b = torch.tensor([10.0], dtype=torch.float64)

    weak_first = SingleCandidateBetaFilter.block_log_bayes_factor(
        da, one_failure, weak_a, weak_b
    )
    strong_first = SingleCandidateBetaFilter.block_log_bayes_factor(
        da, one_failure, strong_a, strong_b
    )
    weak_second = SingleCandidateBetaFilter.block_log_bayes_factor(
        da, two_failures, weak_a, weak_b
    )
    strong_second = SingleCandidateBetaFilter.block_log_bayes_factor(
        da, two_failures, strong_a, strong_b
    )

    assert weak_first.item() == pytest.approx(strong_first.item())
    assert strong_second.item() > weak_second.item()


def test_unobserved_selected_row_is_bitwise_unchanged():
    tracker = _filter(rows=2)
    _update(tracker, [1.0], [0.0], 0, rows=torch.tensor([1]))
    before = tracker.state_dict()

    result = _update(
        tracker,
        [1.0],
        [0.0],
        1,
        mass=torch.tensor([0.0], dtype=torch.float64),
        rows=torch.tensor([1]),
    )

    assert result.observed.tolist() == [False]
    for name, expected in before.items():
        assert torch.equal(getattr(tracker, name), expected), name


def test_state_dict_roundtrip_preserves_live_candidate():
    source = _filter(rows=2)
    _update(source, [1.0], [0.0], 0, rows=torch.tensor([1]))
    _update(source, [0.0], [1.0], 1, rows=torch.tensor([1]))
    target = _filter(rows=2)

    target.load_state_dict(source.state_dict())

    for name, expected in source.state_dict().items():
        assert torch.equal(getattr(target, name), expected), name


def test_same_label_reset_preserves_the_existing_lifespan_slot():
    base = SimpleNamespace(
        _xyz=nn.Parameter(torch.zeros(1, 3)),
        _features_dc=nn.Parameter(torch.ones(1, 1, 3)),
        _features_rest=nn.Parameter(torch.zeros(1, 2, 3)),
        _opacity=nn.Parameter(torch.zeros(1, 1)),
        _scaling=nn.Parameter(torch.zeros(1, 3)),
        _rotation=nn.Parameter(torch.zeros(1, 4)),
    )
    model = TemporalChangeModel.from_gaussians(base, max_states=4)
    model.reset_all_lifespans_closed()
    tracker = _filter(threshold=1.4)
    controller = BayesianLifespanController(
        model,
        BernoulliBOCDConfig(
            prior_a=1.0,
            prior_b=1.0,
            hazard=0.01,
            open_probability=0.6,
            close_probability=0.4,
            changepoint_probability=0.5,
            min_run_evidence=1.0,
            min_visible_observations=1,
        ),
    )

    opened = controller.update(
        _update(tracker, [9.0], [0.0], 0), timestamp=0
    )
    slot = int(model.current_state_index.item())
    reset = controller.update(
        _update(tracker, [7.0], [3.0], 1), timestamp=1
    )

    assert opened.action.tolist() == [int(LifespanAction.OPEN)]
    assert reset.action.tolist() == [int(LifespanAction.KEEP)]
    assert reset.changepoint.tolist() == [True]
    assert int(model.current_state_index.item()) == slot
    assert model.num_states.tolist() == [1]
