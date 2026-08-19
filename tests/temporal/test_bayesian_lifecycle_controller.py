from types import SimpleNamespace

import pytest
import torch
from torch import nn

from temporal.bayesian_lifespan_controller import (
    BayesianLifespanController,
    LifespanAction,
)
from temporal.bernoulli_bocd import (
    BOCDUpdate,
    BernoulliBOCDConfig,
    MAPResetBernoulliFilter,
)
from temporal.change_model import TemporalChangeModel


def make_base(n: int = 1):
    return SimpleNamespace(
        _xyz=nn.Parameter(torch.zeros(n, 3)),
        _features_dc=nn.Parameter(torch.ones(n, 1, 3)),
        _features_rest=nn.Parameter(torch.zeros(n, 2, 3)),
        _opacity=nn.Parameter(torch.zeros(n, 1)),
        _scaling=nn.Parameter(torch.zeros(n, 3)),
        _rotation=nn.Parameter(torch.zeros(n, 4)),
    )


def observation(
    probability: float,
    *,
    timestamp: int,
    cp: float = 1.0,
    observed: bool = True,
    visible: int = 3,
    concentration: float = 6.0,
) -> BOCDUpdate:
    p = torch.tensor([probability])
    return BOCDUpdate(
        indices=torch.tensor([0]),
        observed=torch.tensor([observed]),
        changepoint_probability=torch.tensor([cp]),
        map_run_length=torch.tensor([0]),
        estimated_run_start=torch.tensor([timestamp]),
        change_probability=p,
        concentration=torch.tensor([concentration]),
        visible_observations=torch.tensor([visible]),
        a_map=p * concentration,
        b_map=(1.0 - p) * concentration,
    )


def make_controller(max_states: int = 4):
    model = TemporalChangeModel.from_gaussians(make_base(), max_states=max_states)
    model.reset_all_lifespans_closed()
    config = BernoulliBOCDConfig(
        prior_a=1.0,
        prior_b=1.0,
        hazard=0.2,
        open_probability=0.7,
        close_probability=0.3,
        changepoint_probability=0.5,
        min_run_evidence=1.0,
        min_visible_observations=2,
    )
    return model, BayesianLifespanController(model, config)


def test_inactive_to_active_opens_once():
    model, controller = make_controller()

    result = controller.update(observation(0.9, timestamp=3), timestamp=3)

    assert result.action_names == ("OPEN",)
    assert model.current_state_index.tolist() == [0]
    assert model.num_states.tolist() == [1]
    assert model.state_start[0, 0].item() == 3


def test_active_to_inactive_closes_without_backdating():
    model, controller = make_controller()
    controller.update(observation(0.9, timestamp=2), timestamp=2)

    result = controller.update(observation(0.1, timestamp=7), timestamp=7)

    assert result.action_names == ("CLOSE",)
    assert model.current_state_index.tolist() == [-1]
    assert model.state_end[0, 0].item() == 7


def test_open_close_reopen_allocates_two_disjoint_slots():
    model, controller = make_controller()

    actions = [
        controller.update(observation(0.9, timestamp=2), timestamp=2).action_names[0],
        controller.update(observation(0.1, timestamp=5), timestamp=5).action_names[0],
        controller.update(observation(0.9, timestamp=9), timestamp=9).action_names[0],
    ]

    assert actions == ["OPEN", "CLOSE", "OPEN"]
    assert model.num_states.tolist() == [2]
    assert model.current_state_index.tolist() == [1]
    assert model.state_valid[0, :2].tolist() == [True, True]
    assert model.state_start[0, :2].tolist() == [2.0, 9.0]
    assert model.state_end[0, 0].item() == 5.0
    assert torch.isinf(model.state_end[0, 1])


def test_active_a_to_active_b_is_keep_with_one_continuous_slot():
    model, controller = make_controller()
    first = controller.update(observation(0.8, timestamp=2), timestamp=2)
    old_slot = int(model.current_state_index.item())

    reset = controller.update(observation(0.98, timestamp=8), timestamp=8)

    assert first.action_names == ("OPEN",)
    assert reset.action_names == ("KEEP",)
    assert bool(reset.changepoint.item())
    assert int(model.current_state_index.item()) == old_slot
    assert model.num_states.tolist() == [1]
    assert model.state_start[0, 0].item() == 2
    assert torch.isinf(model.state_end[0, 0])
    records = reset.event_records()
    assert records[0]["action"] == "KEEP"
    assert records[0]["old_slot"] == records[0]["new_or_current_slot"] == 0


def test_unobserved_and_uncertain_rows_do_not_flip_lifecycle():
    model, controller = make_controller()
    before = controller.state_dict()

    missing = controller.update(
        observation(0.9, timestamp=4, observed=False), timestamp=4
    )

    assert missing.action_names == ("NONE",)
    assert model.num_states.tolist() == [0]
    for name, value in before.items():
        assert torch.equal(controller.state_dict()[name], value), name

    uncertain = controller.update(observation(0.5, timestamp=5), timestamp=5)
    assert uncertain.action_names == ("UNCERTAIN",)
    assert model.num_states.tolist() == [0]


def test_surprising_label_without_changepoint_is_uncertain():
    model, controller = make_controller()
    controller.update(observation(0.9, timestamp=1), timestamp=1)

    result = controller.update(
        observation(0.1, timestamp=2, cp=0.1), timestamp=2
    )

    assert result.action.tolist() == [int(LifespanAction.UNCERTAIN)]
    assert model.current_state_index.tolist() == [0]


def test_latched_active_reset_is_logged_as_keep_at_original_estimate():
    model, controller = make_controller()
    model.open_rows([0], timestamp=0)
    controller.committed_run_label[0] = 1
    controller.committed_run_start[0] = 0

    uncertain = controller.update(
        observation(0.5, timestamp=4, cp=0.9), timestamp=4
    )
    assert uncertain.action_names == ("UNCERTAIN",)

    committed = controller.update(
        observation(0.9, timestamp=5, cp=0.1), timestamp=5
    )
    assert committed.action_names == ("KEEP",)
    assert bool(committed.changepoint.item())
    assert committed.estimated_changepoint_timestamp.tolist() == [4]
    event = committed.event_records()[0]
    assert event["action"] == "KEEP"
    assert event["bocd_estimated_changepoint_timestamp"] == 4
    assert event["changepoint_probability"] == pytest.approx(0.9)
    assert model.num_states.tolist() == [1]


def test_filter_controller_sequence_reopens_and_never_splits_active_to_active():
    model = TemporalChangeModel.from_gaussians(make_base(), max_states=5)
    model.reset_all_lifespans_closed()
    config = BernoulliBOCDConfig(
        prior_a=1.0,
        prior_b=1.0,
        hazard=0.2,
        open_probability=0.6,
        close_probability=0.4,
        changepoint_probability=0.5,
        min_run_evidence=0.1,
        min_visible_observations=1,
    )
    controller = BayesianLifespanController(model, config)
    tracker = MAPResetBernoulliFilter(1, config, dtype=torch.float64)

    actions: list[str] = []
    sequence = (
        [(0.0, 10.0)] * 3
        + [(6.5, 3.5)] * 5
        # A large active-cue magnitude shift triggers an internal reset, but
        # both runs remain binary-active and therefore keep one slot.
        + [(20.0, 0.0)] * 2
        + [(0.0, 10.0)] * 3
        + [(10.0, 0.0)] * 3
    )
    active_reset_seen = False
    for timestamp, (positive, negative) in enumerate(sequence):
        update = tracker.update(
            torch.tensor([positive], dtype=torch.float64),
            torch.tensor([negative], dtype=torch.float64),
            timestamp=timestamp,
        )
        decision = controller.update(update, timestamp=timestamp)
        action = decision.action_names[0]
        actions.append(action)
        if 8 <= timestamp <= 9 and bool(decision.changepoint.item()):
            active_reset_seen = True
            assert action == "KEEP"
            assert model.num_states.tolist() == [1]

    assert active_reset_seen
    assert actions.count("OPEN") == 2
    assert actions.count("CLOSE") == 1
    assert model.num_states.tolist() == [2]
    assert model.state_start[0, :2].tolist() == [3.0, 13.0]
    assert model.state_end[0, 0].item() == 10.0
    assert torch.isinf(model.state_end[0, 1])
