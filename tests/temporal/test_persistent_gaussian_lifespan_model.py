from types import SimpleNamespace

import torch
from torch import nn

from temporal import PersistentGaussianLifespanModel
from temporal.binary_state_filter import BinaryStateFilterUpdate
from temporal.binary_state_lifespan_controller import (
    BinaryStateLifespanController,
    BinaryStateLifespanControllerConfig,
)


def make_bank(n: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        _xyz=nn.Parameter(torch.arange(n * 3, dtype=torch.float32).reshape(n, 3)),
        _features_dc=nn.Parameter(torch.zeros(n, 1, 3)),
        _features_rest=nn.Parameter(torch.randn(n, 2, 3)),
        _opacity=nn.Parameter(torch.zeros(n, 1)),
        _scaling=nn.Parameter(torch.zeros(n, 3)),
        _rotation=nn.Parameter(
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32).repeat(n, 1)
        ),
        opacity_activation=torch.sigmoid,
        scaling_activation=torch.exp,
        rotation_activation=lambda value: torch.nn.functional.normalize(value, dim=-1),
    )


def update(p_active: float, timestamp: int) -> BinaryStateFilterUpdate:
    active = p_active >= 0.5
    return BinaryStateFilterUpdate(
        indices=torch.tensor([0]),
        observed=torch.tensor([True]),
        p_active=torch.tensor([p_active]),
        p_00=torch.tensor([0.01 if active else 0.9]),
        p_01=torch.tensor([0.49 if active else 0.01]),
        p_10=torch.tensor([0.49 if not active else 0.01]),
        p_11=torch.tensor([0.01 if not active else 0.9]),
        p_flip=torch.tensor([0.49]),
        q=torch.tensor([p_active]),
        evidence_strength=torch.tensor([1.0]),
        visible_observations=torch.tensor([timestamp + 1]),
        last_timestamp=torch.tensor([timestamp]),
        timestamp=timestamp,
    )


def test_close_and_reopen_allocate_intervals_but_preserve_direct_parameters():
    bank = make_bank(1)
    model = PersistentGaussianLifespanModel(bank, max_states=3)
    model.reset_all_lifespans_closed()
    controller = BinaryStateLifespanController(
        model,
        BinaryStateLifespanControllerConfig(active_threshold=0.6, inactive_threshold=0.4),
    )

    opened = controller.update(update(0.9, 0), timestamp=0)
    assert opened.current_slot.tolist() == [0]
    with torch.no_grad():
        model.change_dc[0].fill_(0.97)
        model.xyz[0].add_(4.0)
        model.opacity[0].fill_(2.0)
    snapshot = {
        name: parameter[0].detach().clone()
        for name, parameter in model.persistent_parameter_items()
    }

    closed = controller.update(update(0.1, 1), timestamp=1)
    assert closed.current_slot.tolist() == [-1]
    hidden = model.get_active_render_attributes(1.0)
    assert not hidden["active"].any()
    assert torch.count_nonzero(hidden["dc"]) == 0
    assert torch.count_nonzero(hidden["opacity"]) == 0

    reopened = controller.update(update(0.9, 2), timestamp=2)
    assert reopened.current_slot.tolist() == [1]
    assert model.num_states.tolist() == [2]
    for name, parameter in model.persistent_parameter_items():
        assert torch.equal(parameter[0], snapshot[name])
    assert torch.all(model.get_active_render_attributes(2.0)["dc"] == 0.97)
    assert model.validate_lifecycle()


def test_direct_parameters_are_the_exact_mutable_bank_storage():
    bank = make_bank(2)
    model = PersistentGaussianLifespanModel(bank, max_states=2)
    assert model.change_dc is bank._features_dc
    assert model.xyz is bank._xyz
    assert model.features_rest is bank._features_rest
    assert model.opacity is bank._opacity
    assert model.scaling is bank._scaling
    assert model.rotation is bank._rotation
    assert not hasattr(model, "state_change_dc")
    assert not hasattr(model, "state_xyz_delta")


def test_inactive_rows_are_gradient_detached_but_active_rows_are_trainable():
    model = PersistentGaussianLifespanModel(make_bank(2), max_states=2)
    model.reset_all_lifespans_closed()
    model.open_rows([0], timestamp=0)
    attrs = model.get_active_render_attributes(0.0)
    loss = (
        attrs["dc"].sum()
        + attrs["xyz"].sum()
        + attrs["features_rest"].sum()
        + attrs["opacity"].sum()
        + attrs["scaling"].sum()
        + attrs["rotation"].sum()
    )
    loss.backward()
    for _name, parameter in model.persistent_parameter_items():
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad[0]) > 0
        assert torch.count_nonzero(parameter.grad[1]) == 0
