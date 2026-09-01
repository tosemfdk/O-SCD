from __future__ import annotations

import torch
from torch import nn

from temporal.bayesian_lifespan_controller import LifespanAction
from temporal.learned_dc_agreement_beta import (
    LearnedDCAgreementBetaConfig,
    LearnedDCAgreementBetaController,
    LearnedDCAgreementBetaFilter,
)
from temporal.masked_optimizer import MaskedRowAdam
from utils.sh_utils import SH2RGB


def make_filter() -> LearnedDCAgreementBetaFilter:
    return LearnedDCAgreementBetaFilter(
        1,
        LearnedDCAgreementBetaConfig(
            stable_flip_prior=1.0,
            stable_keep_prior=10.0,
            reset_flip_prior=1.0,
            reset_keep_prior=1.0,
            bayes_factor_threshold=30.0,
        ),
    )


def update(
    filter_, *, cue_positive: float, expected: float, active: bool, timestamp: int
):
    return filter_.update(
        torch.tensor([cue_positive]),
        torch.tensor([1.0 - cue_positive]),
        expected_change=torch.tensor([expected]),
        current_active=torch.tensor([active]),
        timestamp=timestamp,
    )


def test_soft_dc_cue_agreement_maps_to_fractional_flip_and_keep():
    filter_ = make_filter()
    result = update(
        filter_, cue_positive=1.0, expected=0.8, active=True, timestamp=0
    )

    assert torch.allclose(result.delta_flip, torch.tensor([0.2]))
    assert torch.allclose(result.delta_keep, torch.tensor([0.8]))
    assert torch.allclose(result.expected_change, torch.tensor([0.8]))


def test_binary_dc_limit_opens_closes_and_reopens():
    filter_ = make_filter()
    active = False
    timestamp = 0
    for expected, cue_positive, target in (
        (0.0, 1.0, True),
        (1.0, 0.0, False),
        (0.0, 1.0, True),
    ):
        for _ in range(8):
            result = update(
                filter_,
                cue_positive=cue_positive,
                expected=expected,
                active=active,
                timestamp=timestamp,
            )
            timestamp += 1
            if bool(result.candidate_committed.item()):
                active = bool(result.active_after.item())
                break
        assert active is target


class FakeLifespanModel:
    def __init__(self) -> None:
        self.current_state_index = torch.tensor([-1], dtype=torch.long)
        self.num_states = torch.tensor([0], dtype=torch.long)
        self.change_dc = nn.Parameter(torch.zeros(1, 1, 3))

    def open_rows(self, rows, timestamp, initialization="preserve", optimizer=None):
        del timestamp, initialization, optimizer
        rows = torch.as_tensor(rows, dtype=torch.long)
        self.current_state_index[rows] = self.num_states[rows]
        self.num_states[rows] += 1

    def close_rows(self, rows, timestamp):
        del timestamp
        self.current_state_index[torch.as_tensor(rows, dtype=torch.long)] = -1


def test_controller_resets_dc_and_adam_at_both_committed_boundaries():
    filter_ = make_filter()
    model = FakeLifespanModel()
    optimizer = MaskedRowAdam(
        {"dc": model.change_dc}, thaw_names=("dc",), lrs={"dc": 0.01}
    )
    model.change_dc.grad = torch.ones_like(model.change_dc)
    optimizer.step(torch.tensor([True]))
    controller = LearnedDCAgreementBetaController(model)

    active = False
    decision = None
    timestamp = 0
    for _ in range(8):
        result = update(
            filter_,
            cue_positive=1.0,
            expected=0.0,
            active=active,
            timestamp=timestamp,
        )
        decision = controller.update(
            result, timestamp=timestamp, optimizer=optimizer
        )
        timestamp += 1
        if int(decision.action.item()) == int(LifespanAction.OPEN):
            controller.initialize_open_dc(
                decision.indices, optimizer=optimizer
            )
            active = True
            break

    assert decision is not None
    assert int(decision.action.item()) == int(LifespanAction.OPEN)
    assert torch.allclose(SH2RGB(model.change_dc), torch.ones_like(model.change_dc))
    for name in ("step", "exp_avg", "exp_avg_sq"):
        assert torch.count_nonzero(optimizer.state[model.change_dc][name]) == 0

    model.change_dc.grad = torch.ones_like(model.change_dc)
    optimizer.step(torch.tensor([True]))
    for _ in range(8):
        result = update(
            filter_,
            cue_positive=0.0,
            expected=1.0,
            active=active,
            timestamp=timestamp,
        )
        decision = controller.update(
            result, timestamp=timestamp, optimizer=optimizer
        )
        timestamp += 1
        if int(decision.action.item()) == int(LifespanAction.CLOSE):
            active = False
            break

    assert int(decision.action.item()) == int(LifespanAction.CLOSE)
    assert torch.count_nonzero(model.change_dc) == 0
    assert torch.allclose(
        SH2RGB(model.change_dc),
        torch.full_like(model.change_dc, 0.5),
    )
    assert controller.dc_commit_reset_count == 2
    for name in ("step", "exp_avg", "exp_avg_sq"):
        assert torch.count_nonzero(optimizer.state[model.change_dc][name]) == 0
