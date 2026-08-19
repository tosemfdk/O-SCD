from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from experiments.train_geometry_temporal_rchange import state0_xyz_anchor_loss
from temporal import TemporalGeometryChangeModel


def make_base(n: int = 3) -> SimpleNamespace:
    rotation = torch.zeros(n, 4)
    rotation[:, 0] = 1.0
    return SimpleNamespace(
        _xyz=nn.Parameter(torch.randn(n, 3)),
        _features_dc=nn.Parameter(torch.randn(n, 1, 3)),
        _features_rest=nn.Parameter(torch.randn(n, 2, 3)),
        _opacity=nn.Parameter(torch.randn(n, 1)),
        _scaling=nn.Parameter(torch.randn(n, 3)),
        _rotation=nn.Parameter(rotation),
        opacity_activation=torch.sigmoid,
        scaling_activation=torch.exp,
        rotation_activation=F.normalize,
    )


def configure_three_states(model: TemporalGeometryChangeModel) -> None:
    with torch.no_grad():
        model.state_start[:] = torch.tensor([[0.0, 5.0, 9.0]]).repeat(
            model.state_start.shape[0], 1
        )
        model.state_end[:] = torch.tensor([[5.0, 9.0, float("inf")]]).repeat(
            model.state_end.shape[0], 1
        )
        model.state_valid[:] = True


def test_zero_deltas_reproduce_frozen_base_geometry():
    base = make_base(n=2)
    model = TemporalGeometryChangeModel.from_gaussians(base, max_states=3)
    configure_three_states(model)

    attributes = model.get_active_render_attributes(5.0)

    assert torch.equal(attributes["xyz"], base._xyz)
    assert torch.equal(attributes["opacity"], torch.sigmoid(base._opacity))
    assert torch.equal(attributes["scaling"], torch.exp(base._scaling))
    assert torch.equal(attributes["rotation"], F.normalize(base._rotation))
    assert attributes["indices"].tolist() == [1, 1]


def test_state_geometry_selection_and_invalid_rows_are_isolated():
    model = TemporalGeometryChangeModel.from_gaussians(make_base(n=3), max_states=3)
    configure_three_states(model)
    with torch.no_grad():
        model.state_xyz_delta[:, 1] = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        )
        model.state_valid[0, 1] = False

    attributes = model.get_active_render_attributes(5.0)

    assert attributes["indices"].tolist() == [-1, 1, 1]
    assert torch.equal(attributes["xyz"][0], model.base._xyz[0])
    assert torch.equal(attributes["opacity"][0], torch.zeros(1))
    assert torch.equal(
        attributes["xyz"][1:],
        model.base._xyz[1:] + model.state_xyz_delta[1:, 1],
    )

    sum(attributes[name].sum() for name in ("dc", "xyz", "opacity", "scaling", "rotation")).backward()
    for _name, parameter in model.state_parameter_items():
        expected = torch.zeros_like(parameter.grad)
        expected[1:, 1] = 1.0
        if parameter is model.state_opacity_delta:
            # The sigmoid derivative changes the magnitude, not the support.
            assert torch.equal(parameter.grad != 0, expected != 0)
        elif parameter is model.state_rotation_delta:
            # Quaternion normalization can make individual components zero.
            assert not parameter.grad[0].any()
            assert not parameter.grad[:, 0].any()
            assert not parameter.grad[:, 2].any()
            assert parameter.grad[1:, 1].abs().sum() > 0
        else:
            assert torch.equal(parameter.grad != 0, expected != 0)


def test_geometry_state_dict_and_optimizer_contract_are_explicit():
    model = TemporalGeometryChangeModel.from_gaussians(make_base(n=1), max_states=2)

    assert [name for name, _ in model.state_parameter_items()] == [
        "dc",
        "xyz",
        "opacity",
        "scaling",
        "rotation",
    ]
    assert set(model.state_dict()) == {
        "state_change_dc",
        "state_xyz_delta",
        "state_opacity_delta",
        "state_scaling_delta",
        "state_rotation_delta",
        "state_start",
        "state_end",
        "state_valid",
        "state_status",
        "num_states",
        "current_state_index",
    }


def test_state0_xyz_anchor_is_soft_and_updates_only_the_later_slot():
    model = TemporalGeometryChangeModel.from_gaussians(make_base(n=3), max_states=3)
    configure_three_states(model)
    with torch.no_grad():
        model.state_xyz_delta[:, 0] = 1.0
        model.state_xyz_delta[:, 1] = 0.0

    anchor_indices = torch.tensor([0, 2])
    weighted, count, raw = state0_xyz_anchor_loss(
        model,
        state=1,
        anchor_indices=anchor_indices,
        weight=2.0,
    )
    weighted.backward()

    assert count == 2
    assert raw == 3.0
    assert weighted.item() == 6.0
    assert model.state_xyz_delta.grad[:, 0].abs().sum() == 0
    assert model.state_xyz_delta.grad[1].abs().sum() == 0
    assert model.state_xyz_delta.grad[anchor_indices, 1].abs().sum() > 0
    assert model.state_xyz_delta.grad[:, 2].abs().sum() == 0


def test_state_parameter_inheritance_copies_attributes_but_not_lifespan_metadata():
    model = TemporalGeometryChangeModel.from_gaussians(make_base(n=3), max_states=3)
    configure_three_states(model)
    with torch.no_grad():
        for offset, (_name, parameter) in enumerate(model.state_parameter_items()):
            parameter[:, 0].fill_(float(offset + 1))
            parameter[:, 1].fill_(-1.0)
        model.state_valid[:, 1] = torch.tensor([True, False, True])
    target_valid_before = model.state_valid[:, 1].clone()
    target_start_before = model.state_start[:, 1].clone()
    target_end_before = model.state_end[:, 1].clone()

    model.inherit_state_parameters(0, 1)

    for _name, parameter in model.state_parameter_items():
        assert torch.equal(parameter[:, 1], parameter[:, 0])
    assert torch.equal(model.state_valid[:, 1], target_valid_before)
    assert torch.equal(model.state_start[:, 1], target_start_before)
    assert torch.equal(model.state_end[:, 1], target_end_before)
