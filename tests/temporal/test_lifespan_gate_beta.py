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


def update(filter_, *, cue_positive: float, active: bool, timestamp: int):
    return filter_.update(
        torch.tensor([cue_positive]),
        torch.tensor([1.0 - cue_positive]),
        current_active=torch.tensor([active]),
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
