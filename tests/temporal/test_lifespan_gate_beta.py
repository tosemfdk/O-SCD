from __future__ import annotations

import torch

from temporal.bayesian_lifespan_controller import LifespanAction
from temporal.lifespan_gate_beta import (
    LifespanGateBetaConfig,
    LifespanGateBetaController,
    LifespanGateBetaFilter,
)


def make_filter() -> LifespanGateBetaFilter:
    return LifespanGateBetaFilter(
        1,
        LifespanGateBetaConfig(
            stable_flip_prior=1.0,
            stable_keep_prior=10.0,
            reset_flip_prior=1.0,
            reset_keep_prior=1.0,
            bayes_factor_threshold=30.0,
        ),
    )


def update(
    filter_,
    *,
    cue_positive: float,
    active: bool,
    timestamp: int,
    first_open=None,
    first_open_bayes_factor_threshold=None,
    rows=None,
):
    return filter_.update(
        torch.tensor([cue_positive]),
        torch.tensor([1.0 - cue_positive]),
        current_active=torch.tensor([active]),
        first_open=first_open,
        first_open_bayes_factor_threshold=first_open_bayes_factor_threshold,
        row_indices=rows,
        timestamp=timestamp,
    )


def test_closed_matching_cue_accumulates_keep_evidence_from_reference_prior():
    filter_ = make_filter()
    result = update(filter_, cue_positive=0.0, active=False, timestamp=0)

    assert not bool(result.candidate_committed.item())
    assert not bool(result.active_after.item())
    assert torch.equal(filter_.stable_a, torch.tensor([1.0]))
    assert torch.equal(filter_.stable_b, torch.tensor([11.0]))


def test_opposite_cue_opens_then_matching_cue_keeps_new_gate_state():
    filter_ = make_filter()
    update(filter_, cue_positive=0.0, active=False, timestamp=0)

    committed = None
    for timestamp in range(1, 8):
        result = update(
            filter_, cue_positive=1.0, active=False, timestamp=timestamp
        )
        if bool(result.candidate_committed.item()):
            committed = result
            break

    assert committed is not None
    assert bool(committed.active_after.item())
    # Candidate FLIP observations become KEEP support after the bit toggles.
    assert float(filter_.stable_b.item()) > float(filter_.stable_a.item())
    before = filter_.stable_b.clone()

    kept = update(filter_, cue_positive=1.0, active=True, timestamp=timestamp + 1)
    assert not bool(kept.candidate_committed.item())
    assert bool(kept.active_after.item())
    assert bool((filter_.stable_b > before).item())


def test_committed_flip_carries_reset_posterior_into_toggled_coordinates():
    filter_ = make_filter()

    committed = None
    for timestamp in range(3):
        result = update(
            filter_, cue_positive=1.0, active=False, timestamp=timestamp
        )
        if bool(result.candidate_committed.item()):
            committed = result
            break

    assert committed is not None
    assert bool(committed.active_after.item())
    # The winning RESET posterior is Beta(4, 1) in the old CLOSED
    # coordinates.  Toggling to OPEN swaps FLIP/KEEP, yielding Beta(1, 4),
    # without re-injecting the initial CLOSED-only Beta(1, 10) prior.
    assert filter_.stable_a.tolist() == [1.0]
    assert filter_.stable_b.tolist() == [4.0]
    assert filter_.stable_total_evidence.tolist() == [3.0]


def test_open_gate_closes_on_sustained_negative_cue_and_can_reopen():
    filter_ = make_filter()
    active = False
    timestamp = 0
    for _ in range(8):
        result = update(
            filter_, cue_positive=1.0, active=active, timestamp=timestamp
        )
        timestamp += 1
        if bool(result.candidate_committed.item()):
            active = bool(result.active_after.item())
            break
    assert active

    for _ in range(8):
        result = update(
            filter_, cue_positive=0.0, active=active, timestamp=timestamp
        )
        timestamp += 1
        if bool(result.candidate_committed.item()):
            active = bool(result.active_after.item())
            break
    assert not active

    for _ in range(8):
        result = update(
            filter_, cue_positive=1.0, active=active, timestamp=timestamp
        )
        timestamp += 1
        if bool(result.candidate_committed.item()):
            active = bool(result.active_after.item())
            break
    assert active




def test_first_open_threshold_override_opens_only_marked_never_open_rows_early():
    filter_ = LifespanGateBetaFilter(2, make_filter().gate_config)

    filter_.update(
        torch.tensor([1.0, 1.0]),
        torch.tensor([0.0, 0.0]),
        current_active=torch.tensor([False, False]),
        first_open=torch.tensor([True, False]),
        first_open_bayes_factor_threshold=10.0,
        timestamp=0,
    )
    second = filter_.update(
        torch.tensor([1.0, 1.0]),
        torch.tensor([0.0, 0.0]),
        current_active=torch.tensor([False, False]),
        first_open=torch.tensor([True, False]),
        first_open_bayes_factor_threshold=10.0,
        timestamp=1,
    )

    assert second.candidate_committed.tolist() == [True, False]
    assert second.active_after.tolist() == [True, False]

    third = filter_.update(
        torch.tensor([1.0]),
        torch.tensor([0.0]),
        current_active=torch.tensor([False]),
        first_open=torch.tensor([False]),
        first_open_bayes_factor_threshold=10.0,
        row_indices=torch.tensor([1]),
        timestamp=2,
    )

    assert third.candidate_committed.tolist() == [True]
    assert third.active_after.tolist() == [True]


def test_first_open_mask_without_threshold_preserves_default_gate():
    filter_ = make_filter()

    first = update(
        filter_,
        cue_positive=1.0,
        active=False,
        first_open=torch.tensor([True]),
        timestamp=0,
    )
    second = update(
        filter_,
        cue_positive=1.0,
        active=False,
        first_open=torch.tensor([True]),
        timestamp=1,
    )
    third = update(
        filter_,
        cue_positive=1.0,
        active=False,
        first_open=torch.tensor([True]),
        timestamp=2,
    )

    assert first.candidate_committed.tolist() == [False]
    assert second.candidate_committed.tolist() == [False]
    assert third.candidate_committed.tolist() == [True]


def test_first_open_threshold_does_not_accelerate_close_or_reopen_when_mask_false():
    accelerated = make_filter()
    defaulted = make_filter()
    active = False
    timestamp = 0
    for _ in range(2):
        result = update(
            accelerated,
            cue_positive=1.0,
            active=active,
            first_open=torch.tensor([True]),
            first_open_bayes_factor_threshold=10.0,
            timestamp=timestamp,
        )
        update(
            defaulted,
            cue_positive=1.0,
            active=active,
            first_open=torch.tensor([True]),
            first_open_bayes_factor_threshold=10.0,
            timestamp=timestamp,
        )
        timestamp += 1
        if bool(result.candidate_committed.item()):
            active = bool(result.active_after.item())
            break
    assert active

    accelerated_close_commits = []
    default_close_commits = []
    for step in range(20):
        accelerated_close = update(
            accelerated,
            cue_positive=0.0,
            active=active,
            first_open=torch.tensor([False]),
            first_open_bayes_factor_threshold=10.0,
            timestamp=timestamp + step,
        )
        default_close = update(
            defaulted,
            cue_positive=0.0,
            active=active,
            first_open=torch.tensor([False]),
            timestamp=timestamp + step,
        )
        accelerated_close_commits.append(
            bool(accelerated_close.candidate_committed.item())
        )
        default_close_commits.append(bool(default_close.candidate_committed.item()))
        if accelerated_close_commits[-1]:
            active = False
            timestamp += step + 1
            break

    assert accelerated_close_commits == default_close_commits
    assert True in accelerated_close_commits

    accelerated_reopen_commits = []
    default_reopen_commits = []
    for step in range(20):
        accelerated_reopen = update(
            accelerated,
            cue_positive=1.0,
            active=active,
            first_open=torch.tensor([False]),
            first_open_bayes_factor_threshold=10.0,
            timestamp=timestamp + step,
        )
        default_reopen = update(
            defaulted,
            cue_positive=1.0,
            active=active,
            first_open=torch.tensor([False]),
            timestamp=timestamp + step,
        )
        accelerated_reopen_commits.append(
            bool(accelerated_reopen.candidate_committed.item())
        )
        default_reopen_commits.append(bool(default_reopen.candidate_committed.item()))
        if accelerated_reopen_commits[-1]:
            break

    assert accelerated_reopen_commits == default_reopen_commits
    assert True in accelerated_reopen_commits


def test_first_open_threshold_validation_happens_before_mutation():
    filter_ = make_filter()
    before = filter_.state_dict()

    invalid_calls = [
        {"first_open_bayes_factor_threshold": 10.0},
        {
            "first_open": torch.tensor([True, False]),
            "first_open_bayes_factor_threshold": 10.0,
        },
        {
            "first_open": torch.tensor([1]),
            "first_open_bayes_factor_threshold": 10.0,
        },
        {
            "first_open": torch.tensor([True]),
            "first_open_bayes_factor_threshold": 1.0,
        },
    ]
    for kwargs in invalid_calls:
        try:
            update(
                filter_,
                cue_positive=1.0,
                active=False,
                timestamp=0,
                **kwargs,
            )
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(f"accepted invalid first-open override {kwargs!r}")

    for name, expected in before.items():
        assert torch.equal(getattr(filter_, name), expected), name


def test_first_open_threshold_override_works_on_grown_row_subset():
    filter_ = make_filter()
    filter_.ensure_capacity(3)

    first = update(
        filter_,
        cue_positive=1.0,
        active=False,
        first_open=torch.tensor([True]),
        first_open_bayes_factor_threshold=10.0,
        rows=torch.tensor([2]),
        timestamp=5,
    )
    second = update(
        filter_,
        cue_positive=1.0,
        active=False,
        first_open=torch.tensor([True]),
        first_open_bayes_factor_threshold=10.0,
        rows=torch.tensor([2]),
        timestamp=6,
    )

    assert first.indices.tolist() == [2]
    assert first.candidate_committed.tolist() == [False]
    assert second.indices.tolist() == [2]
    assert second.candidate_committed.tolist() == [True]
    assert second.active_after.tolist() == [True]


class FakeLifespanModel:
    def __init__(self) -> None:
        self.current_state_index = torch.tensor([-1], dtype=torch.long)
        self.num_states = torch.tensor([0], dtype=torch.long)

    def open_rows(self, rows, timestamp, initialization="preserve", optimizer=None):
        del timestamp, initialization, optimizer
        rows = torch.as_tensor(rows, dtype=torch.long)
        self.current_state_index[rows] = self.num_states[rows]
        self.num_states[rows] += 1

    def close_rows(self, rows, timestamp):
        del timestamp
        self.current_state_index[torch.as_tensor(rows, dtype=torch.long)] = -1


def test_controller_toggles_lifespan_only_on_committed_candidate():
    filter_ = make_filter()
    model = FakeLifespanModel()
    controller = LifespanGateBetaController(model)

    decision = None
    for timestamp in range(8):
        result = update(
            filter_, cue_positive=1.0, active=False, timestamp=timestamp
        )
        decision = controller.update(result, timestamp=timestamp)
        if int(decision.action.item()) == int(LifespanAction.OPEN):
            break

    assert decision is not None
    assert int(decision.action.item()) == int(LifespanAction.OPEN)
    assert int(model.current_state_index.item()) == 0

    for timestamp in range(timestamp + 1, timestamp + 9):
        result = update(
            filter_, cue_positive=0.0, active=True, timestamp=timestamp
        )
        decision = controller.update(result, timestamp=timestamp)
        if int(decision.action.item()) == int(LifespanAction.CLOSE):
            break

    assert int(decision.action.item()) == int(LifespanAction.CLOSE)
    assert int(model.current_state_index.item()) == -1


def test_ensure_capacity_preserves_updated_prefix_bitwise_and_initializes_tail_like_fresh_filter():
    config = LifespanGateBetaConfig(
        stable_flip_prior=2.0,
        stable_keep_prior=7.0,
        reset_flip_prior=3.0,
        reset_keep_prior=5.0,
        bayes_factor_threshold=30.0,
    )
    filter_ = LifespanGateBetaFilter(2, config)
    filter_.update(
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 1.0]),
        current_active=torch.tensor([False, False]),
        timestamp=11,
    )
    for timestamp in range(12, 15):
        filter_.update(
            torch.tensor([1.0]),
            torch.tensor([0.0]),
            current_active=torch.tensor([False]),
            row_indices=torch.tensor([0]),
            timestamp=timestamp,
        )

    prefix = {
        name: getattr(filter_, name).clone()
        for name in filter_.topology_buffer_names
    }
    filter_.ensure_capacity(5)

    assert filter_.num_gaussians == 5
    fresh_tail = LifespanGateBetaFilter(
        3, config, device=filter_.device, dtype=filter_.dtype
    )
    for name in filter_.topology_buffer_names:
        actual = getattr(filter_, name)
        assert torch.equal(actual[:2], prefix[name]), name
        assert torch.equal(actual[2:], getattr(fresh_tail, name)), name


def test_ensure_capacity_noop_preserves_tensor_identity():
    filter_ = make_filter()
    before = {name: getattr(filter_, name) for name in filter_.topology_buffer_names}

    filter_.ensure_capacity(1)

    for name, tensor in before.items():
        assert getattr(filter_, name) is tensor


def test_ensure_capacity_newly_grown_rows_can_update_by_index():
    filter_ = make_filter()
    filter_.ensure_capacity(3)

    result = filter_.update(
        torch.tensor([0.0]),
        torch.tensor([1.0]),
        current_active=torch.tensor([False]),
        row_indices=torch.tensor([2]),
        timestamp=5,
    )

    assert result.indices.tolist() == [2]
    assert int(filter_.last_timestamp[2].item()) == 5
    assert torch.equal(filter_.stable_a[2:3], torch.tensor([1.0]))
    assert torch.equal(filter_.stable_b[2:3], torch.tensor([11.0]))


def test_ensure_capacity_rejects_non_integer_and_negative_requirements():
    from fractions import Fraction

    filter_ = make_filter()
    for invalid in (-1, True, 1.5, Fraction(3, 2)):
        try:
            filter_.ensure_capacity(invalid)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(f"accepted invalid required_rows={invalid!r}")
