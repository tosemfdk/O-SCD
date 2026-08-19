from types import SimpleNamespace

import pytest
import torch
from torch import nn

from temporal.bayesian_lifespan_controller import BayesianLifespanController, LifespanAction
from temporal.bernoulli_bocd import BernoulliBOCDConfig
from temporal import TemporalChangeModel, TemporalGeometryChangeModel
from temporal.bernoulli_bocd import BOCDUpdate
from temporal.masked_optimizer import MaskedRowSlotAdam


def make_base(n=1):
    return SimpleNamespace(
        _xyz=nn.Parameter(torch.zeros(n, 3)),
        _features_dc=nn.Parameter(torch.ones(n, 1, 3)),
        _features_rest=nn.Parameter(torch.zeros(n, 2, 3)),
        _opacity=nn.Parameter(torch.zeros(n, 1)),
        _scaling=nn.Parameter(torch.zeros(n, 3)),
        _rotation=nn.Parameter(torch.zeros(n, 4)),
    )


def update(p, obs=3, conc=5.0):
    return BOCDUpdate(
        indices=torch.tensor([0]), observed=torch.tensor([True]), changepoint_probability=torch.tensor([0.9]),
        map_run_length=torch.tensor([1]), estimated_run_start=torch.tensor([0]), change_probability=torch.tensor([p]),
        concentration=torch.tensor([conc]), visible_observations=torch.tensor([obs]), a_map=torch.tensor([p*conc]), b_map=torch.tensor([(1-p)*conc])
    )


def test_dynamic_lifecycle_open_close_reopen_disjoint_intervals():
    model = TemporalChangeModel.from_gaussians(make_base(), max_states=3)
    model.reset_all_lifespans_closed()
    c = BayesianLifespanController(model, BernoulliBOCDConfig(open_probability=0.6, close_probability=0.4, changepoint_probability=0.5, min_run_evidence=1.0))
    assert LifespanAction(int(c.update(update(0.2), timestamp=0).action[0])).name == "NONE"
    d = c.update(update(0.8), timestamp=1); assert LifespanAction(int(d.action[0])).name == "OPEN"
    assert model.current_state_index.tolist() == [0]
    d = c.update(update(0.2), timestamp=2); assert LifespanAction(int(d.action[0])).name == "CLOSE"
    assert model.current_state_index.tolist() == [-1]
    d = c.update(update(0.9), timestamp=3)
    assert model.current_state_index.tolist() == [1]
    assert model.state_start[0, :2].tolist() == [1.0, 3.0]
    assert model.state_end[0, 0].item() == 2.0
    assert torch.isinf(model.state_end[0, 1])
    assert model.validate_lifecycle()


def test_active_to_active_changepoint_is_keep_without_new_slot():
    model = TemporalChangeModel.from_gaussians(make_base(), max_states=3)
    model.reset_all_lifespans_closed()
    model.open_rows(torch.tensor([0]), 0.0)
    c = BayesianLifespanController(model, BernoulliBOCDConfig(open_probability=0.6, close_probability=0.4, changepoint_probability=0.5, min_run_evidence=1.0))
    before = model.num_states.clone()
    d = c.update(update(0.95), timestamp=5)
    assert LifespanAction(int(d.action[0])).name == "KEEP"
    assert torch.equal(model.num_states, before)
    assert model.current_state_index.tolist() == [0]
    assert torch.isinf(model.state_end[0, 0])


def test_no_observation_is_uncertain_and_does_not_advance_lifecycle():
    model = TemporalChangeModel.from_gaussians(make_base(), max_states=2)
    model.reset_all_lifespans_closed()
    u = update(0.9, obs=0, conc=10.0)
    u = BOCDUpdate(u.indices, torch.tensor([False]), u.changepoint_probability, u.map_run_length, u.estimated_run_start, u.change_probability, u.concentration, u.visible_observations, u.a_map, u.b_map)
    d = BayesianLifespanController(model, BernoulliBOCDConfig()).update(u, timestamp=1)
    assert LifespanAction(int(d.action[0])).name == "NONE"
    assert model.current_state_index.tolist() == [-1]


def test_open_resets_only_the_new_row_slot_optimizer_state():
    model = TemporalGeometryChangeModel.from_gaussians(make_base(n=2), max_states=2)
    optimizer = MaskedRowSlotAdam(
        dict(model.state_parameter_items()),
        thaw_names=("dc", "xyz", "opacity", "scaling", "rotation"),
    )
    seeded = torch.zeros(2, 2, dtype=torch.bool)
    seeded[:, 0] = True
    for _name, parameter in model.state_parameter_items():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step(seeded)
    preserved = {
        name: {
            key: value[1, 0].clone()
            for key, value in optimizer.state[parameter].items()
        }
        for name, parameter in model.state_parameter_items()
    }

    model.reset_all_lifespans_closed()
    model.open_rows([0], timestamp=1, optimizer=optimizer)

    for name, parameter in model.state_parameter_items():
        state = optimizer.state[parameter]
        assert state["step"][0, 0].item() == 0
        assert torch.count_nonzero(state["exp_avg"][0, 0]).item() == 0
        assert torch.count_nonzero(state["exp_avg_sq"][0, 0]).item() == 0
        for key, expected in preserved[name].items():
            assert torch.equal(state[key][1, 0], expected)


def test_reopen_capacity_overflow_is_explicit():
    model = TemporalChangeModel.from_gaussians(make_base(), max_states=1)
    model.reset_all_lifespans_closed()
    model.open_rows([0], timestamp=1)
    model.close_rows([0], timestamp=2)
    with pytest.raises(RuntimeError, match="capacity exceeded for 1 Gaussians"):
        model.open_rows([0], timestamp=3)
