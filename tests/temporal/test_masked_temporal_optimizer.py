from pathlib import Path
import importlib.util

import pytest
import torch
from torch import nn

spec = importlib.util.spec_from_file_location(
    "masked_optimizer_local",
    Path(__file__).resolve().parents[1].parent / "temporal" / "masked_optimizer.py",
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
MaskedRowSlotAdam = module.MaskedRowSlotAdam
active_visible_pair_mask = module.active_visible_pair_mask


class _DummyTemporalModel(nn.Module):
    """Minimal model exposing row-slot state parameters for optimizer tests."""

    def __init__(self, n: int = 5, s: int = 3, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.base = nn.Module()
        self.base._xyz = nn.Parameter(torch.randn(n, 3, dtype=dtype), requires_grad=False)
        self.base._features_dc = nn.Parameter(torch.randn(n, 1, 3, dtype=dtype), requires_grad=False)
        self.base._features_rest = nn.Parameter(torch.randn(n, 2, 3, dtype=dtype), requires_grad=False)
        self.base._opacity = nn.Parameter(torch.randn(n, 1, dtype=dtype), requires_grad=False)
        self.base._scaling = nn.Parameter(torch.randn(n, 3, dtype=dtype), requires_grad=False)
        self.base._rotation = nn.Parameter(torch.randn(n, 4, dtype=dtype), requires_grad=False)

        self.state_change_dc = nn.Parameter(torch.randn(n, s, 1, 3, dtype=dtype))
        self.state_xyz_delta = nn.Parameter(torch.zeros(n, s, 3, dtype=dtype))
        self.state_opacity_delta = nn.Parameter(torch.zeros(n, s, 1, dtype=dtype))
        self.state_scaling_delta = nn.Parameter(torch.zeros(n, s, 3, dtype=dtype))
        self.state_rotation_delta = nn.Parameter(torch.zeros(n, s, 4, dtype=dtype))

    def state_parameter_items(self):
        return (
            ("dc", self.state_change_dc),
            ("xyz", self.state_xyz_delta),
            ("opacity", self.state_opacity_delta),
            ("scaling", self.state_scaling_delta),
            ("rotation", self.state_rotation_delta),
        )


def _set_masked_gradients(model: _DummyTemporalModel, mask: torch.Tensor, value: float) -> None:
    for _name, parameter in model.state_parameter_items():
        expanded = mask.reshape(mask.shape + (1,) * (parameter.ndim - 2)).expand_as(parameter)
        parameter.grad = torch.zeros_like(parameter)
        parameter.grad[expanded] = float(value)


def _pair_value(parameter: torch.Tensor, row: int, slot: int) -> torch.Tensor:
    return parameter[row, slot].detach().clone()


def test_row_slot_masked_adam_updates_only_selected_pairs_and_preserves_inactive_state():
    torch.manual_seed(0)
    model = _DummyTemporalModel(n=5, s=3)
    opt = MaskedRowSlotAdam(
        dict(model.state_parameter_items()),
        thaw_names=("dc", "xyz", "opacity", "scaling", "rotation"),
        lrs={
            "dc": 0.02,
            "xyz": 0.02,
            "opacity": 0.02,
            "scaling": 0.02,
            "rotation": 0.02,
        },
    )

    active = torch.zeros(5, 3, dtype=torch.bool)
    active[1, 1] = True

    # Build momentum on a single active pair.
    _set_masked_gradients(model, active, 1.0)
    opt.step(active)
    opt.step(active)

    before_param = {_name: _pair_value(param, 1, 1) for _name, param in model.state_parameter_items()}
    before_state = {_name: opt.state[param]["exp_avg"][1, 1].detach().clone() for _name, param in model.state_parameter_items()}
    before_state_sq = {_name: opt.state[param]["exp_avg_sq"][1, 1].detach().clone() for _name, param in model.state_parameter_items()}
    before_step = {_name: opt.state[param]["step"][1, 1].item() for _name, param in model.state_parameter_items()}

    # Deactivate it and step other active rows; inactive row-slot must not drift.
    deactive = torch.zeros_like(active)
    deactive[2, 0] = True
    deactive[3, 0] = True
    _set_masked_gradients(model, deactive, 0.5)
    for _ in range(3):
        opt.step(deactive)

    for _name, param in model.state_parameter_items():
        assert torch.equal(_pair_value(param, 1, 1), before_param[_name])
        assert torch.equal(opt.state[param]["exp_avg"][1, 1], before_state[_name])
        assert torch.equal(opt.state[param]["exp_avg_sq"][1, 1], before_state_sq[_name])
        assert opt.state[param]["step"][1, 1].item() == before_step[_name]
        # Sanity: another row/slot pair advanced step counter.
        assert opt.state[param]["step"][2, 0].item() == 3

    # Intentionally replay/reactivate the old pair without touching its retained
    # state; it must be optimizable only when explicitly selected again.
    _set_masked_gradients(model, active, 2.0)
    opt.step(active)
    for _name, param in model.state_parameter_items():
        assert not torch.equal(_pair_value(param, 1, 1), before_param[_name])
        assert opt.state[param]["step"][1, 1].item() == before_step[_name] + 1

    # A newly allocated pair can then be reset independently without altering
    # any other row-slot state.
    opt.reset_state_pairs(deactive)
    for _name, param in model.state_parameter_items():
        assert torch.equal(
            opt.state[param]["exp_avg"][2, 0],
            torch.zeros_like(opt.state[param]["exp_avg"][2, 0]),
        )
        assert torch.equal(
            opt.state[param]["exp_avg_sq"][2, 0],
            torch.zeros_like(opt.state[param]["exp_avg_sq"][2, 0]),
        )
        assert opt.state[param]["step"][2, 0].item() == 0


def test_active_but_invisible_pair_preserves_parameters_and_adam_state_exactly():
    model = _DummyTemporalModel(n=3, s=2)
    opt = MaskedRowSlotAdam(
        dict(model.state_parameter_items()),
        thaw_names=("dc", "xyz", "opacity", "scaling", "rotation"),
        lrs={name: 0.02 for name in ("dc", "xyz", "opacity", "scaling", "rotation")},
    )
    active = torch.zeros(3, 2, dtype=torch.bool)
    active[0, 0] = True
    active[1, 1] = True

    # Row 0 is initially visible and accumulates non-zero Adam momentum.
    row0_visible = active_visible_pair_mask(active, torch.tensor([2.0, 0.0, 0.0]))
    assert torch.equal(
        row0_visible,
        torch.tensor([[True, False], [False, False], [False, False]]),
    )
    _set_masked_gradients(model, row0_visible, 1.0)
    opt.step(row0_visible)
    opt.step(row0_visible)

    parameter_before = {
        name: parameter[0, 0].detach().clone()
        for name, parameter in model.state_parameter_items()
    }
    state_before = {
        name: {
            key: opt.state[parameter][key][0, 0].detach().clone()
            for key in ("exp_avg", "exp_avg_sq", "step")
        }
        for name, parameter in model.state_parameter_items()
    }

    # Row 0 remains OPEN but leaves the current view.  Only visible row 1 may
    # advance; row 0 must retain parameters and optimizer moments bitwise.
    row1_visible = active_visible_pair_mask(active, torch.tensor([0.0, 3.0, 0.0]))
    _set_masked_gradients(model, row1_visible, 0.5)
    for _ in range(4):
        opt.step(row1_visible)

    for name, parameter in model.state_parameter_items():
        assert torch.equal(parameter[0, 0], parameter_before[name])
        for key in ("exp_avg", "exp_avg_sq", "step"):
            assert torch.equal(opt.state[parameter][key][0, 0], state_before[name][key])
        assert opt.state[parameter]["step"][1, 1].item() == 4

    # The same active pair becomes intentionally optimizable again when it
    # re-enters a later selected view.
    _set_masked_gradients(model, row0_visible, 0.25)
    opt.step(row0_visible)
    for name, parameter in model.state_parameter_items():
        assert not torch.equal(parameter[0, 0], parameter_before[name])
        assert opt.state[parameter]["step"][0, 0].item() == 3


def test_active_visible_pair_mask_rejects_mismatched_renderer_shapes():
    active = torch.zeros(3, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match="shape \\[N\\]"):
        active_visible_pair_mask(active, torch.ones(2))
    with pytest.raises(ValueError, match="boolean \\[N, S\\]"):
        active_visible_pair_mask(active.float(), torch.ones(3))


def test_optimizer_respects_disabled_thaw_names_and_keeps_them_exact():
    model = _DummyTemporalModel(n=3, s=2)
    opt = MaskedRowSlotAdam(
        dict(model.state_parameter_items()),
        thaw_names=("dc", "xyz", "scaling", "rotation"),
        lrs={"dc": 0.01, "xyz": 0.01, "scaling": 0.01, "rotation": 0.01},
    )

    names = {group["name"] for group in opt.param_groups}
    assert "opacity" not in names

    opacity_before = model.state_opacity_delta.detach().clone()
    active = torch.ones(3, 2, dtype=torch.bool)
    _set_masked_gradients(model, active, 1.0)
    opt.step(active)

    # Disabled thaw component stays exact.
    assert torch.equal(model.state_opacity_delta, opacity_before)
    assert model.state_opacity_delta not in opt.state

    # Enabled components must move.
    assert not torch.equal(model.state_change_dc, torch.zeros_like(model.state_change_dc))
    assert not torch.equal(model.state_xyz_delta, torch.zeros_like(model.state_xyz_delta))
    assert not torch.equal(model.state_scaling_delta, torch.zeros_like(model.state_scaling_delta))
    assert not torch.equal(model.state_rotation_delta, torch.zeros_like(model.state_rotation_delta))


def test_masked_optimizer_keeps_inactive_step_state_and_amsgrad_buffer_exact():
    model = _DummyTemporalModel(n=2, s=2)
    opt = MaskedRowSlotAdam(
        dict(model.state_parameter_items()),
        thaw_names=("dc", "xyz"),
        lrs={"dc": 0.03, "xyz": 0.03},
        amsgrad=True,
    )

    pair_a = torch.tensor([[True, False], [False, False]])
    pair_b = torch.tensor([[False, False], [True, False]])

    _set_masked_gradients(model, pair_a, 0.5)
    opt.step(pair_a)
    step_a = opt.state[model.state_change_dc]["step"][0, 0].item()
    assert step_a == 1

    # Pair-b advances; A must stay as-is including step and AMSGrad state.
    before_max_a_dc = opt.state[model.state_change_dc]["max_exp_avg_sq"][0, 0].detach().clone()
    before_max_a_xyz = opt.state[model.state_xyz_delta]["max_exp_avg_sq"][0, 0].detach().clone()
    before_max_b_dc = opt.state[model.state_change_dc]["max_exp_avg_sq"][1, 0].detach().clone()

    _set_masked_gradients(model, pair_b, 0.7)
    for _ in range(2):
        opt.step(pair_b)

    assert opt.state[model.state_change_dc]["step"][0, 0].item() == step_a
    assert opt.state[model.state_xyz_delta]["step"][0, 0].item() == step_a
    assert torch.equal(opt.state[model.state_change_dc]["max_exp_avg_sq"][0, 0], before_max_a_dc)
    assert torch.equal(opt.state[model.state_xyz_delta]["max_exp_avg_sq"][0, 0], before_max_a_xyz)
    assert not torch.equal(opt.state[model.state_change_dc]["max_exp_avg_sq"][1, 0], before_max_b_dc)

    # Resetting and replaying A uses its own pair-scoped step.
    opt.reset_state_pairs(pair_a)
    assert opt.state[model.state_change_dc]["step"][0, 0].item() == 0
    assert opt.state[model.state_xyz_delta]["step"][0, 0].item() == 0

    _set_masked_gradients(model, pair_a, 0.25)
    opt.step(pair_a)
    assert opt.state[model.state_change_dc]["step"][0, 0].item() == 1


def test_masked_optimizer_rejects_unknown_thaw_names_and_invalid_hyperparameters():
    model = _DummyTemporalModel(n=1, s=1)
    parameters = dict(model.state_parameter_items())
    with pytest.raises(ValueError, match="unknown thaw"):
        MaskedRowSlotAdam(parameters, thaw_names=("dc", "shared"))
    with pytest.raises(ValueError, match="eps"):
        MaskedRowSlotAdam(parameters, thaw_names=("dc",), eps=0.0)
