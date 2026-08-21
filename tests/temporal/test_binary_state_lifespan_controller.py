from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from temporal.binary_state_filter import BinaryStateFilter
from temporal.binary_state_lifespan_controller import (
    BinaryLifespanAction,
    BinaryStateLifespanController,
)
from temporal.change_model import TemporalChangeModel
from temporal.geometry_change_model import TemporalGeometryChangeModel
from temporal.masked_optimizer import MaskedRowSlotAdam


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


def make_dc_controller(max_states=4):
    model = TemporalChangeModel.from_gaussians(make_base(), max_states=max_states)
    model.reset_all_lifespans_closed()
    return model, BinaryStateLifespanController(model)


def filter_update(q, *, timestamp, total_mass=10.0, observed=True):
    filt = BinaryStateFilter(1, dtype=torch.float64)
    if observed:
        return filt.update(torch.tensor([q * total_mass]), torch.tensor([(1 - q) * total_mass]), timestamp=timestamp)
    return filt.update(torch.tensor([q * total_mass]), torch.tensor([(1 - q) * total_mass]), total_mass=torch.tensor([0.0]), timestamp=timestamp)


def test_synthetic_repeated_sequence_none_open_close_open_new_slot():
    model, controller = make_dc_controller(max_states=4)
    filt = BinaryStateFilter(1, dtype=torch.float64)
    qs = [0.05, 0.08, 0.10, 0.90, 0.92, 0.88, 0.07, 0.05, 0.10, 0.91, 0.94, 0.90]
    actions = []
    for t, q in enumerate(qs):
        upd = filt.update(torch.tensor([q * 10.0]), torch.tensor([(1.0 - q) * 10.0]), timestamp=t)
        actions.append(controller.update(upd, timestamp=t).action_names[0])

    assert "OPEN" in actions
    assert actions.count("OPEN") == 2
    assert actions.count("CLOSE") == 1
    assert actions[-1] == "KEEP"
    assert model.current_state_index.tolist() == [1]
    assert model.num_states.tolist() == [2]
    assert model.state_status[0, :2].tolist() == [2, 1]
    assert model.state_end[0, 0] < model.state_start[0, 1]


def test_stable_active_opens_once_then_keeps_same_slot():
    model, controller = make_dc_controller()
    filt = BinaryStateFilter(1, dtype=torch.float64)
    actions = []
    slots = []
    for t, q in enumerate([0.9, 0.85, 0.95, 0.88]):
        upd = filt.update(torch.tensor([q * 10.0]), torch.tensor([(1 - q) * 10.0]), timestamp=t)
        decision = controller.update(upd, timestamp=t)
        actions.append(decision.action_names[0])
        slots.append(int(model.current_state_index.item()))
    assert actions[0] == "OPEN"
    assert actions[1:] == ["KEEP", "KEEP", "KEEP"]
    assert slots == [0, 0, 0, 0]
    assert model.num_states.tolist() == [1]


def test_stable_inactive_always_none():
    model, controller = make_dc_controller()
    filt = BinaryStateFilter(1, dtype=torch.float64)
    actions = []
    for t, q in enumerate([0.1, 0.05, 0.12]):
        upd = filt.update(torch.tensor([q * 10.0]), torch.tensor([(1 - q) * 10.0]), timestamp=t)
        actions.append(controller.update(upd, timestamp=t).action_names[0])
    assert actions == ["NONE", "NONE", "NONE"]
    assert model.current_state_index.tolist() == [-1]


def test_ambiguous_is_uncertain_without_unnecessary_flip():
    model, controller = make_dc_controller()
    actions = []
    for t, q in enumerate([0.48, 0.52, 0.49]):
        decision = controller.update(filter_update(q, timestamp=t, total_mass=0.1), timestamp=t)
        actions.append(decision.action_names[0])
    assert actions == ["UNCERTAIN", "UNCERTAIN", "UNCERTAIN"]
    assert model.current_state_index.tolist() == [-1]


def test_unobserved_lifecycle_state_is_none_and_unchanged():
    model, controller = make_dc_controller()
    controller.update(filter_update(0.9, timestamp=0), timestamp=0)
    before = {name: value.clone() for name, value in model.state_dict().items()}

    decision = controller.update(filter_update(0.1, timestamp=1, observed=False), timestamp=1)

    assert decision.action_names == ("NONE",)
    for name, value in before.items():
        assert torch.equal(model.state_dict()[name], value), name


def test_action_codes_are_explicit():
    assert int(BinaryLifespanAction.OPEN) == 1
    assert int(BinaryLifespanAction.CLOSE) == 3
    assert int(BinaryLifespanAction.UNCERTAIN) == 4


def test_open_zero_initializes_all_geometry_and_resets_optimizer_moments():
    model = TemporalGeometryChangeModel.from_gaussians(make_base(), max_states=3)
    model.reset_all_lifespans_closed()
    controller = BinaryStateLifespanController(model)
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

    decision = controller.update(filter_update(0.9, timestamp=4), timestamp=4, optimizer=optimizer)

    assert decision.action_names == ("OPEN",)
    assert model.current_state_index.tolist() == [0]
    for _name, parameter in model.state_parameter_items():
        assert torch.equal(parameter[:, 0], torch.zeros_like(parameter[:, 0]))
    for group in optimizer.param_groups:
        parameter = group["params"][0]
        state = optimizer.state[parameter]
        assert torch.equal(state["step"][:, 0], torch.zeros_like(state["step"][:, 0]))
        assert torch.equal(state["exp_avg"][:, 0], torch.zeros_like(state["exp_avg"][:, 0]))
        assert torch.equal(state["exp_avg_sq"][:, 0], torch.zeros_like(state["exp_avg_sq"][:, 0]))


def test_close_then_reopen_does_not_reuse_closed_slot_or_drift_closed_values():
    model = TemporalGeometryChangeModel.from_gaussians(make_base(), max_states=3)
    model.reset_all_lifespans_closed()
    controller = BinaryStateLifespanController(model)
    filt = BinaryStateFilter(1, dtype=torch.float64)

    controller.update(filt.update(torch.tensor([9.0]), torch.tensor([1.0]), timestamp=0), timestamp=0)
    with torch.no_grad():
        for _name, parameter in model.state_parameter_items():
            parameter[:, 0].fill_(3.0)
    closed_snapshot = {name: parameter[:, 0].detach().clone() for name, parameter in model.state_parameter_items()}
    controller.update(filt.update(torch.tensor([0.0]), torch.tensor([10.0]), timestamp=3), timestamp=3)
    controller.update(filt.update(torch.tensor([10.0]), torch.tensor([0.0]), timestamp=6), timestamp=6)

    assert model.current_state_index.tolist() == [1]
    assert model.state_status[0, :2].tolist() == [2, 1]
    for name, parameter in model.state_parameter_items():
        assert torch.equal(parameter[:, 0], closed_snapshot[name])
        assert torch.equal(parameter[:, 1], torch.zeros_like(parameter[:, 1]))


def test_controller_rejects_duplicate_indices():
    model, controller = make_dc_controller()
    update = filter_update(0.9, timestamp=0)
    dup = type(update)(
        indices=torch.tensor([0, 0]),
        observed=torch.tensor([True, True]),
        p_active=torch.tensor([0.9, 0.9]),
        p_00=torch.zeros(2), p_01=torch.zeros(2), p_10=torch.zeros(2), p_11=torch.zeros(2), p_flip=torch.zeros(2),
        q=torch.zeros(2), evidence_strength=torch.ones(2), visible_observations=torch.ones(2, dtype=torch.long),
        last_timestamp=torch.zeros(2, dtype=torch.long), timestamp=0,
    )
    with pytest.raises(ValueError, match="unique"):
        controller.update(dup, timestamp=0)


def test_active_to_active_keep_ignores_high_flip_diagnostic_probability():
    model, controller = make_dc_controller(max_states=3)
    filt = BinaryStateFilter(1, dtype=torch.float64)
    controller.update(filt.update(torch.tensor([0.9]), torch.tensor([0.1]), timestamp=0), timestamp=0)
    before_slots = model.current_state_index.clone()
    update = filter_update(0.95, timestamp=1, total_mass=10.0)
    high_flip = type(update)(
        indices=update.indices,
        observed=update.observed,
        p_active=torch.tensor([0.95], dtype=update.p_active.dtype),
        p_00=torch.tensor([0.01], dtype=update.p_00.dtype),
        p_01=torch.tensor([0.44], dtype=update.p_01.dtype),
        p_10=torch.tensor([0.44], dtype=update.p_10.dtype),
        p_11=torch.tensor([0.11], dtype=update.p_11.dtype),
        p_flip=torch.tensor([0.88], dtype=update.p_flip.dtype),
        q=update.q,
        evidence_strength=update.evidence_strength,
        visible_observations=update.visible_observations,
        last_timestamp=update.last_timestamp,
        timestamp=1,
    )

    decision = controller.update(high_flip, timestamp=1)

    assert decision.action_names == ("KEEP",)
    assert torch.equal(model.current_state_index, before_slots)
    assert model.num_states.tolist() == [1]
