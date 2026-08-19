from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from temporal import TemporalSharedGeometryChangeModel
from temporal.geometry_freeze import (
    capture_frozen_rows,
    mask_frozen_row_gradients,
    restore_frozen_rows,
)


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


def configure_states(model: TemporalSharedGeometryChangeModel) -> None:
    with torch.no_grad():
        model.state_start[:] = torch.tensor([[0.0, 5.0]]).repeat(3, 1)
        model.state_end[:] = torch.tensor([[5.0, float("inf")]]).repeat(3, 1)
        model.state_valid[:] = True


def test_shared_geometry_is_reused_when_rendering_every_state():
    model = TemporalSharedGeometryChangeModel.from_gaussians(
        make_base(), max_states=2
    )
    configure_states(model)
    with torch.no_grad():
        model.state_change_dc[:, 0].fill_(1.0)
        model.state_change_dc[:, 1].fill_(2.0)
        model.shared_xyz_delta.fill_(3.0)

    state0 = model.get_active_render_attributes(0.0)
    state1 = model.get_active_render_attributes(5.0)

    assert torch.equal(state0["xyz"], state1["xyz"])
    assert torch.equal(state0["xyz"], model.base._xyz + 3.0)
    assert torch.equal(state0["dc"], torch.ones(3, 1, 3))
    assert torch.equal(state1["dc"], torch.full((3, 1, 3), 2.0))


def test_later_state_gradient_updates_shared_geometry_but_not_earlier_dc():
    model = TemporalSharedGeometryChangeModel.from_gaussians(
        make_base(), max_states=2
    )
    configure_states(model)
    model.state_valid[0, 1] = False

    attributes = model.get_active_render_attributes(5.0)
    loss = attributes["dc"].sum() + attributes["xyz"].sum()
    loss.backward()

    assert not model.state_change_dc.grad[:, 0].any()
    assert not model.state_change_dc.grad[0, 1].any()
    assert model.state_change_dc.grad[1:, 1].abs().sum() > 0
    assert not model.shared_xyz_delta.grad[0].any()
    assert model.shared_xyz_delta.grad[1:].abs().sum() > 0


def test_geometry_frozen_is_persistent_monotonic_per_gaussian_metadata():
    model = TemporalSharedGeometryChangeModel.from_gaussians(
        make_base(), max_states=2
    )

    assert model.geometry_frozen.dtype == torch.bool
    assert not model.geometry_frozen.any()

    model.freeze_geometry_rows(torch.tensor([True, False, False]))
    model.freeze_geometry_rows(torch.tensor([False, True, False]))

    assert model.geometry_frozen.tolist() == [True, True, False]
    assert torch.equal(
        model.state_dict()["geometry_frozen"],
        torch.tensor([True, True, False]),
    )


def test_frozen_metadata_masks_geometry_gradients_but_not_state_dc():
    model = TemporalSharedGeometryChangeModel.from_gaussians(
        make_base(), max_states=2
    )
    configure_states(model)
    model.freeze_geometry_rows(torch.tensor([True, False, True]))

    attributes = model.get_active_render_attributes(5.0)
    loss = (
        attributes["dc"].sum()
        + attributes["xyz"].sum()
        + attributes["opacity"].sum()
        + attributes["scaling"].sum()
        + attributes["rotation"].sum()
    )
    loss.backward()
    state1_dc_before = model.state_change_dc.grad[:, 1].clone()

    pre_mask = model.mask_frozen_geometry_gradients()

    assert all(value > 0.0 for value in pre_mask.values())
    for _name, parameter in model.shared_geometry_parameter_items():
        assert not parameter.grad[model.geometry_frozen].any()
        assert parameter.grad[~model.geometry_frozen].abs().sum() > 0
    assert torch.equal(model.state_change_dc.grad[:, 1], state1_dc_before)


def test_model_metadata_mask_matches_external_mask_projection_for_same_gradients():
    external = TemporalSharedGeometryChangeModel.from_gaussians(
        make_base(), max_states=2
    )
    internal = TemporalSharedGeometryChangeModel.from_gaussians(
        make_base(), max_states=2
    )
    frozen = torch.tensor([True, False, True])
    external_items = external.shared_geometry_parameter_items()
    internal_items = internal.shared_geometry_parameter_items()
    anchors = capture_frozen_rows(external_items, frozen)
    internal.freeze_geometry_rows(frozen)
    external_optimizer = torch.optim.Adam(
        [parameter for _name, parameter in external_items],
        lr=0.01,
        eps=1e-15,
    )
    internal_optimizer = torch.optim.Adam(
        [parameter for _name, parameter in internal_items],
        lr=0.01,
        eps=1e-15,
    )

    for step in range(1, 4):
        for parameter_index, (
            (_external_name, external_parameter),
            (_internal_name, internal_parameter),
        ) in enumerate(zip(external_items, internal_items)):
            gradient = torch.arange(
                external_parameter.numel(), dtype=external_parameter.dtype
            ).reshape_as(external_parameter)
            gradient = gradient.add(1).mul(step + parameter_index)
            external_parameter.grad = gradient.clone()
            internal_parameter.grad = gradient.clone()
        mask_frozen_row_gradients(external_items, frozen)
        internal.mask_frozen_geometry_gradients()
        external_optimizer.step()
        internal_optimizer.step()
        restore_frozen_rows(external_items, frozen, anchors)
        external_optimizer.zero_grad(set_to_none=True)
        internal_optimizer.zero_grad(set_to_none=True)

    for (_name_a, external_parameter), (_name_b, internal_parameter) in zip(
        external_items, internal_items
    ):
        assert torch.equal(external_parameter, internal_parameter)


def test_all_geometry_can_be_frozen_while_state1_dc_still_optimizes():
    model = TemporalSharedGeometryChangeModel.from_gaussians(
        make_base(), max_states=2
    )
    configure_states(model)
    model.freeze_geometry_rows(torch.ones(3, dtype=torch.bool))
    geometry_before = {
        name: parameter.detach().clone()
        for name, parameter in model.shared_geometry_parameter_items()
    }
    state1_dc_before = model.state_change_dc[:, 1].detach().clone()
    optimizer = torch.optim.Adam([model.state_change_dc], lr=0.01, eps=1e-15)

    attributes = model.get_active_render_attributes(5.0)
    loss = attributes["dc"].sum() + attributes["xyz"].sum()
    loss.backward()
    model.mask_frozen_geometry_gradients()
    optimizer.step()

    assert not torch.equal(model.state_change_dc[:, 1], state1_dc_before)
    for name, parameter in model.shared_geometry_parameter_items():
        assert torch.equal(parameter, geometry_before[name])
