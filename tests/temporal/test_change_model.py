from types import SimpleNamespace

import pytest
import torch
from torch import nn

from temporal import TemporalChangeModel


def make_base(n=3, dtype=torch.float32):
    return SimpleNamespace(
        _xyz=nn.Parameter(torch.randn(n, 3, dtype=dtype)),
        _features_dc=nn.Parameter(torch.arange(n * 3, dtype=dtype).reshape(n, 1, 3)),
        _features_rest=nn.Parameter(torch.randn(n, 2, 3, dtype=dtype)),
        _opacity=nn.Parameter(torch.randn(n, 1, dtype=dtype)),
        _scaling=nn.Parameter(torch.randn(n, 3, dtype=dtype)),
        _rotation=nn.Parameter(torch.randn(n, 4, dtype=dtype)),
    )


def test_from_gaussians_initializes_state_and_freezes_base_tensors():
    base = make_base(n=2)
    model = TemporalChangeModel.from_gaussians(base, max_states=4, initial_time=7.0)

    assert model.base is base
    assert model.state_change_dc.shape == (2, 4, 1, 3)
    assert torch.equal(model.state_change_dc[:, 0].detach(), base._features_dc.detach())
    assert torch.equal(model.state_change_dc[:, 1:].detach(), torch.zeros(2, 3, 1, 3))
    assert torch.equal(model.state_start[:, 0], torch.full((2,), 7.0))
    assert torch.isinf(model.state_end).all()
    assert torch.equal(model.state_valid[:, 0], torch.ones(2, dtype=torch.bool))
    assert not model.state_valid[:, 1:].any()
    assert [name for name, _ in model.named_parameters()] == ["state_change_dc"]

    for name in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"):
        assert getattr(base, name).requires_grad is False


def test_state_dict_contains_temporal_parameter_and_buffers():
    model = TemporalChangeModel.from_gaussians(make_base(n=1), max_states=2, initial_time=3.0)

    state = model.state_dict()

    # Status/count buffers are added with the later transition lifecycle.
    assert set(state) == {"state_change_dc", "state_start", "state_end", "state_valid"}
    assert state["state_change_dc"].shape == (1, 2, 1, 3)
    assert state["state_start"].dtype == model.state_change_dc.dtype
    assert state["state_valid"].dtype == torch.bool


@pytest.mark.parametrize(
    "kwargs",
    (
        {"max_states": 0},
        {"max_states": 1.5},
        {"max_states": True},
        {"initial_time": float("nan")},
        {"initial_time": float("inf")},
        {"initial_time": True},
    ),
)
def test_from_gaussians_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        TemporalChangeModel.from_gaussians(make_base(n=1), **kwargs)


def test_get_active_change_dc_selects_timestamped_slot():
    model = TemporalChangeModel.from_gaussians(make_base(n=2), max_states=3)
    with torch.no_grad():
        model.state_change_dc[:, 1] = torch.tensor([[[10.0, 11.0, 12.0]], [[20.0, 21.0, 22.0]]])
        model.state_start[:] = torch.tensor([[0.0, 5.0, 9.0], [0.0, 5.0, 9.0]])
        model.state_end[:] = torch.tensor([[5.0, 9.0, float("inf")], [5.0, 9.0, float("inf")]])
        model.state_valid[:] = True

    assert torch.equal(model.get_active_state_indices(5.0), torch.tensor([1, 1]))
    assert torch.equal(
        model.get_active_change_dc(5.0),
        torch.tensor([[[10.0, 11.0, 12.0]], [[20.0, 21.0, 22.0]]]),
    )


def test_get_active_change_dc_backpropagates_only_active_slots():
    model = TemporalChangeModel.from_gaussians(make_base(n=2), max_states=3)
    with torch.no_grad():
        model.state_start[:] = torch.tensor([[0.0, 5.0, 9.0], [0.0, 5.0, 9.0]])
        model.state_end[:] = torch.tensor([[5.0, 9.0, float("inf")], [5.0, 9.0, float("inf")]])
        model.state_valid[:] = True

    loss = model.get_active_change_dc(5.0).sum()
    loss.backward()

    expected_grad = torch.zeros_like(model.state_change_dc)
    expected_grad[:, 1] = 1.0
    assert torch.equal(model.state_change_dc.grad, expected_grad)


def test_get_active_change_masks_outdated_gaussians_and_their_gradients():
    model = TemporalChangeModel.from_gaussians(make_base(n=3), max_states=2)
    with torch.no_grad():
        model.state_change_dc.fill_(1.0)
        model.state_start[:] = torch.tensor([[0.0, 5.0]]).repeat(3, 1)
        model.state_end[:] = torch.tensor([[5.0, 10.0]]).repeat(3, 1)
        model.state_valid[:] = torch.tensor(
            [[True, False], [True, True], [False, True]]
        )

    dc, active = model.get_active_change(6.0)

    assert torch.equal(model.get_active_state_indices(6.0), torch.tensor([-1, 1, 1]))
    assert torch.equal(active, torch.tensor([False, True, True]))
    assert torch.equal(dc[0], torch.zeros_like(dc[0]))

    dc.sum().backward()
    expected_grad = torch.zeros_like(model.state_change_dc)
    expected_grad[1:, 1] = 1.0
    assert torch.equal(model.state_change_dc.grad, expected_grad)


def test_get_active_change_dc_rejects_independent_sidecar_migration():
    model = TemporalChangeModel.from_gaussians(make_base(n=1)).to(torch.float64)

    with pytest.raises(RuntimeError, match="place the base first"):
        model.get_active_change_dc(0.0)
