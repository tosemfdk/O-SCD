from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from temporal.binary_state_filter import BinaryStateFilter, BinaryStateFilterUpdate
from temporal.binary_state_lifespan_controller import BinaryLifespanAction
from temporal.change_model import TemporalChangeModel
from temporal.geometry_change_model import TemporalGeometryChangeModel
from temporal.masked_optimizer import MaskedRowSlotAdam
from temporal.view_consistent_binary_lifespan_controller import (
    ViewConsistentBinaryLifespanController,
    ViewConsistentBinaryLifespanControllerConfig,
)


def make_base(n: int = 1):
    rotation = torch.zeros(n, 4)
    rotation[:, 0] = 1.0
    return SimpleNamespace(
        _xyz=nn.Parameter(torch.zeros(n, 3)),
        _features_dc=nn.Parameter(torch.ones(n, 1, 3)),
        _features_rest=nn.Parameter(torch.zeros(n, 2, 3)),
        _opacity=nn.Parameter(torch.zeros(n, 1)),
        _scaling=nn.Parameter(torch.zeros(n, 3)),
        _rotation=nn.Parameter(rotation),
        opacity_activation=torch.sigmoid,
        scaling_activation=torch.exp,
        rotation_activation=F.normalize,
    )


def make_controller(*, max_states=4, confirmation_views=2, model_kind="dc"):
    if model_kind == "geometry":
        model = TemporalGeometryChangeModel.from_gaussians(make_base(), max_states=max_states)
    else:
        model = TemporalChangeModel.from_gaussians(make_base(), max_states=max_states)
    model.reset_all_lifespans_closed()
    cfg = ViewConsistentBinaryLifespanControllerConfig(
        confirmation_views=confirmation_views,
        min_transition_bayes_factor=3.0,
        min_evidence_strength=1e-6,
    )
    return model, ViewConsistentBinaryLifespanController(model, cfg)


def filter_update(filt, q, *, timestamp, total_mass=1.0, observed=True):
    da = torch.tensor([q * total_mass], dtype=torch.float64)
    db = torch.tensor([(1.0 - q) * total_mass], dtype=torch.float64)
    tm = None if observed else torch.tensor([0.0], dtype=torch.float64)
    return filt.update(da, db, total_mass=tm, timestamp=timestamp)


def fake_update(*, p_active=0.99, p00=0.98, p01=0.01, p10=0.0, p11=0.01, strength=1.0, observed=True, timestamp=0):
    return BinaryStateFilterUpdate(
        indices=torch.tensor([0]),
        observed=torch.tensor([observed]),
        p_active=torch.tensor([p_active], dtype=torch.float64),
        p_00=torch.tensor([p00], dtype=torch.float64),
        p_01=torch.tensor([p01], dtype=torch.float64),
        p_10=torch.tensor([p10], dtype=torch.float64),
        p_11=torch.tensor([p11], dtype=torch.float64),
        p_flip=torch.tensor([p01 + p10], dtype=torch.float64),
        q=torch.tensor([0.9], dtype=torch.float64),
        evidence_strength=torch.tensor([strength], dtype=torch.float64),
        visible_observations=torch.tensor([1], dtype=torch.long),
        last_timestamp=torch.tensor([timestamp], dtype=torch.long),
        timestamp=timestamp,
    )


@pytest.mark.parametrize("k,expected", [(1, ["OPEN"]), (2, ["NONE", "OPEN"]), (3, ["NONE", "NONE", "OPEN"])])
def test_confirmation_views_gate_open(k, expected):
    model, controller = make_controller(confirmation_views=k)
    filt = BinaryStateFilter(1, dtype=torch.float64)

    actions = []
    for t in range(k):
        decision = controller.update(filter_update(filt, 0.9, timestamp=t), timestamp=t)
        actions.append(decision.action_names[0])

    assert actions == expected
    assert model.current_state_index.tolist() == [0]


def test_unobserved_hold_preserves_counter_and_lifecycle_bitwise():
    model, controller = make_controller(confirmation_views=2)
    filt = BinaryStateFilter(1, dtype=torch.float64)
    first = controller.update(filter_update(filt, 0.9, timestamp=0), timestamp=0)
    assert first.action_names == ("NONE",)
    assert first.open_support_count.tolist() == [1]
    before_model = {name: value.clone() for name, value in model.state_dict().items()}
    before_controller = controller.state_dict()

    decision = controller.update(filter_update(filt, 0.9, timestamp=1, observed=False), timestamp=1)

    assert decision.action_names == ("NONE",)
    assert decision.open_support_count.tolist() == [1]
    for name, value in before_model.items():
        assert torch.equal(model.state_dict()[name], value), name
    for name, value in before_controller.items():
        assert torch.equal(controller.state_dict()[name], value), name


def test_observed_contradiction_resets_support_counter():
    model, controller = make_controller(confirmation_views=2)
    filt = BinaryStateFilter(1, dtype=torch.float64)

    assert controller.update(filter_update(filt, 0.9, timestamp=0), timestamp=0).open_support_count.tolist() == [1]
    reset = controller.update(filter_update(filt, 0.55, timestamp=1), timestamp=1)
    assert reset.action_names == ("NONE",)
    assert reset.open_support_count.tolist() == [0]
    again = controller.update(filter_update(filt, 0.9, timestamp=2), timestamp=2)
    assert again.action_names == ("NONE",)
    assert again.open_support_count.tolist() == [1]
    assert model.current_state_index.tolist() == [-1]


def test_low_mass_gate_does_not_confirm_transition():
    model, controller = make_controller(confirmation_views=2)
    filt = BinaryStateFilter(1, dtype=torch.float64)

    first = controller.update(filter_update(filt, 0.9, timestamp=0), timestamp=0)
    decision = controller.update(filter_update(filt, 0.9, timestamp=1, total_mass=1e-8), timestamp=1)

    assert first.open_support_count.tolist() == [1]
    assert decision.action_names == ("NONE",)
    assert decision.open_support.tolist() == [False]
    assert decision.open_support_count.tolist() == [1]
    assert model.current_state_index.tolist() == [-1]
    confirmed = controller.update(filter_update(filt, 0.9, timestamp=2), timestamp=2)
    assert confirmed.action_names == ("OPEN",)


def test_p_active_crossing_alone_cannot_flip_lifecycle():
    model, controller = make_controller(confirmation_views=1)

    decision = controller.update(fake_update(p_active=0.99, p00=0.98, p01=0.01, p10=0.0, p11=0.01), timestamp=0)

    assert decision.action_names == ("NONE",)
    assert decision.open_bayes_factor.item() < 3.0
    assert model.current_state_index.tolist() == [-1]


def test_open_close_reopen_allocates_new_slot():
    model, controller = make_controller(max_states=4, confirmation_views=2)
    filt = BinaryStateFilter(1, dtype=torch.float64)
    actions = []
    qs = [0.9, 0.9, 0.1, 0.1, 0.9, 0.9]
    for t, q in enumerate(qs):
        actions.append(controller.update(filter_update(filt, q, timestamp=t), timestamp=t).action_names[0])

    assert actions == ["NONE", "OPEN", "KEEP", "CLOSE", "NONE", "OPEN"]
    # Event diagnostics report the support count that committed the transition,
    # not the reset persistent counter after commit.
    filt2 = BinaryStateFilter(1, dtype=torch.float64)
    model2, controller2 = make_controller(max_states=4, confirmation_views=2)
    controller2.update(filter_update(filt2, 0.9, timestamp=0), timestamp=0)
    open_decision = controller2.update(filter_update(filt2, 0.9, timestamp=1), timestamp=1)
    assert open_decision.event_records()[0]["open_support_count"] == 2
    assert controller2.state_dict()["open_support_count"].tolist() == [0]
    assert model.current_state_index.tolist() == [1]
    assert model.num_states.tolist() == [2]
    assert model.state_status[0, :2].tolist() == [2, 1]
    assert model.state_end[0, 0] < model.state_start[0, 1]


def test_active_to_active_support_keeps_same_slot_without_split():
    model, controller = make_controller(max_states=3, confirmation_views=1)
    filt = BinaryStateFilter(1, dtype=torch.float64)
    assert controller.update(filter_update(filt, 0.9, timestamp=0), timestamp=0).action_names == ("OPEN",)
    before_slot = model.current_state_index.clone()

    decision = controller.update(filter_update(filt, 0.95, timestamp=1), timestamp=1)

    assert decision.action_names == ("KEEP",)
    assert torch.equal(model.current_state_index, before_slot)
    assert model.num_states.tolist() == [1]


def test_open_zero_initializes_geometry_and_optimizer_pairs():
    model, controller = make_controller(max_states=3, confirmation_views=1, model_kind="geometry")
    optimizer = MaskedRowSlotAdam(dict(model.state_parameter_items()), thaw_names=("dc", "xyz", "opacity", "scaling", "rotation"))
    with torch.no_grad():
        for _name, parameter in model.state_parameter_items():
            parameter[:, 0].fill_(123.0)
        for group in optimizer.param_groups:
            parameter = group["params"][0]
            state = optimizer.state[parameter]
            state["step"][:, 0] = 5
            state["exp_avg"][:, 0].fill_(7.0)
            state["exp_avg_sq"][:, 0].fill_(8.0)

    filt = BinaryStateFilter(1, dtype=torch.float64)
    decision = controller.update(filter_update(filt, 0.9, timestamp=0), timestamp=0, optimizer=optimizer)

    assert decision.action_names == ("OPEN",)
    for _name, parameter in model.state_parameter_items():
        assert torch.equal(parameter[:, 0], torch.zeros_like(parameter[:, 0]))
    for group in optimizer.param_groups:
        parameter = group["params"][0]
        state = optimizer.state[parameter]
        assert torch.equal(state["step"][:, 0], torch.zeros_like(state["step"][:, 0]))
        assert torch.equal(state["exp_avg"][:, 0], torch.zeros_like(state["exp_avg"][:, 0]))
        assert torch.equal(state["exp_avg_sq"][:, 0], torch.zeros_like(state["exp_avg_sq"][:, 0]))


def test_controller_state_dict_round_trip():
    _model, controller = make_controller(confirmation_views=2)
    filt = BinaryStateFilter(1, dtype=torch.float64)
    controller.update(filter_update(filt, 0.9, timestamp=0), timestamp=0)
    saved = controller.state_dict()
    model2, controller2 = make_controller(confirmation_views=2)

    controller2.load_state_dict(saved)

    assert torch.equal(controller2.open_support_count, controller.open_support_count)
    assert torch.equal(controller2.close_support_count, controller.close_support_count)
    with pytest.raises(ValueError, match="shape"):
        controller2.load_state_dict({"open_support_count": torch.zeros(2, dtype=torch.long), "close_support_count": torch.zeros(1, dtype=torch.long)})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"confirmation_views": 0},
        {"min_transition_bayes_factor": 0.0},
        {"min_evidence_strength": -1.0},
        {"inactive_to_active_prior": 0.0},
        {"active_to_inactive_prior": 1.0},
    ],
)
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        ViewConsistentBinaryLifespanControllerConfig(**kwargs)


def test_action_codes_are_shared_with_existing_controller():
    assert int(BinaryLifespanAction.OPEN) == 1
    assert int(BinaryLifespanAction.CLOSE) == 3
