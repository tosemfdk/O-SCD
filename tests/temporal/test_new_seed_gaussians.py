import hashlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from temporal.new_seed_gaussians import NewSeedGaussianModel, ConcatenatedChangeView
from utils.general_utils import inverse_sigmoid


def _tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class DummyBase:
    def __init__(self, n=2, sh_degree=0, dtype=torch.float32, device="cpu"):
        self.max_sh_degree = sh_degree
        self.active_sh_degree = 0
        rest = (sh_degree + 1) ** 2 - 1
        self._xyz = nn.Parameter(torch.arange(n * 3, dtype=dtype, device=device).reshape(n, 3) / 10)
        self._features_dc = nn.Parameter(torch.arange(n * 3, dtype=dtype, device=device).reshape(n, 1, 3) / 100)
        self._features_rest = nn.Parameter(torch.zeros((n, rest, 3), dtype=dtype, device=device))
        self._opacity = nn.Parameter(inverse_sigmoid(torch.full((n, 1), 0.4, dtype=dtype, device=device)))
        self._scaling = nn.Parameter(torch.zeros((n, 3), dtype=dtype, device=device))
        rotation = torch.zeros((n, 4), dtype=dtype, device=device)
        rotation[:, 0] = 1.0
        self._rotation = nn.Parameter(rotation)
        self.scaling_activation = torch.exp
        self.opacity_activation = torch.sigmoid
        self.rotation_activation = torch.nn.functional.normalize
        self.covariance_activation = lambda scaling, scaling_modifier, rotation: scaling * scaling_modifier

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=1)


def test_only_seed_dc_is_trainable_and_geometry_is_buffered():
    model = NewSeedGaussianModel(sh_degree=1)
    model.append(
        xyz=torch.tensor([[1.0, 2.0, 3.0]]),
        start=5.0,
        scaling=torch.full((1, 3), -2.0),
        metadata={"source_sign": "+"},
    )

    assert list(dict(model.named_parameters()).keys()) == ["seed_dc"]
    assert model.seed_dc.requires_grad
    assert model.seed_dc.shape == (1, 1, 3)
    assert not any(buffer.requires_grad for _, buffer in model.named_buffers())
    assert model._xyz.shape == (1, 3)
    assert model._features_rest.shape == (1, 3, 3)
    assert model.metadata == [{"source_sign": "+"}]

    loss = model.seed_dc.sum() + model.get_xyz.detach().sum()
    loss.backward()
    assert model.seed_dc.grad is not None
    assert all(buffer.grad is None for _, buffer in model.named_buffers())


def test_append_preserves_adam_state_and_zero_initializes_new_rows():
    model = NewSeedGaussianModel()
    model.append(xyz=torch.tensor([[0.0, 0.0, 1.0]]), start=0.0, dc=torch.ones(1, 1, 3))
    opt = torch.optim.Adam(model.optimizer_parameter_groups(lr=0.01))
    (model.seed_dc.square().sum()).backward()
    opt.step()
    opt.zero_grad(set_to_none=True)

    old_param = model.seed_dc
    old_state = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in opt.state[old_param].items()}
    old_value = model.seed_dc.detach().clone()

    model.append(
        xyz=torch.tensor([[1.0, 0.0, 1.0], [2.0, 0.0, 1.0]]),
        start=3.0,
        dc=torch.zeros(2, 1, 3),
        optimizer=opt,
    )

    assert model.seed_dc is opt.param_groups[0]["params"][0]
    assert old_param not in opt.state
    state = opt.state[model.seed_dc]
    assert torch.equal(model.seed_dc.detach()[:1], old_value)
    assert torch.equal(state["exp_avg"][:1], old_state["exp_avg"])
    assert torch.equal(state["exp_avg_sq"][:1], old_state["exp_avg_sq"])
    assert torch.equal(state["exp_avg"][1:], torch.zeros_like(state["exp_avg"][1:]))
    assert torch.equal(state["exp_avg_sq"][1:], torch.zeros_like(state["exp_avg_sq"][1:]))
    # Adam step is scalar state and must survive replacement.
    assert torch.equal(state["step"], old_state["step"])


def test_half_open_lifespans_and_close_active():
    model = NewSeedGaussianModel()
    model.append(
        xyz=torch.zeros(3, 3),
        start=torch.tensor([1.0, 2.0, 2.0]),
        end=torch.tensor([3.0, 4.0, float("inf")]),
        dc=torch.arange(9, dtype=torch.float32).reshape(3, 1, 3),
    )

    assert model.active_mask(1.0).tolist() == [True, False, False]
    assert model.active_mask(2.0).tolist() == [True, True, True]
    assert model.active_mask(3.0).tolist() == [False, True, True]
    attrs = model.get_active_render_attributes(3.0)
    assert attrs["dc"].shape == (2, 1, 3)
    assert attrs["dc"][:, 0, 0].tolist() == [3.0, 6.0]

    closed = model.close_active(3.5)
    assert closed.tolist() == [False, True, True]
    assert model.active_mask(3.5).tolist() == [False, False, False]


def test_checkpoint_roundtrip_preserves_tensors_and_metadata():
    model = NewSeedGaussianModel(sh_degree=1)
    model.append(
        xyz=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        start=7.0,
        end=torch.tensor([8.0, float("inf")]),
        scaling=torch.full((2, 3), -3.0),
        rotation=torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]]),
        opacity=torch.full((2, 1), 0.2),
        dc=torch.randn(2, 1, 3),
        metadata=[{"id": 1}, {"id": 2}],
    )

    restored = NewSeedGaussianModel.from_checkpoint(model.to_checkpoint())
    assert restored.max_sh_degree == model.max_sh_degree
    assert restored.metadata == model.metadata
    for name in ("seed_dc", "_xyz", "_features_rest", "_opacity", "_scaling", "_rotation", "start", "end"):
        assert torch.equal(getattr(restored, name), getattr(model, name)), name


def test_concatenated_change_view_adds_only_active_seed_rows_and_detaches_base():
    base = DummyBase(n=2)
    base_hashes = {name: _tensor_sha(getattr(base, name)) for name in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation")}
    seeds = NewSeedGaussianModel()
    seeds.append(
        xyz=torch.tensor([[10.0, 0.0, 1.0], [20.0, 0.0, 1.0]]),
        start=torch.tensor([0.0, 5.0]),
        end=torch.tensor([5.0, 10.0]),
        dc=torch.ones(2, 1, 3),
    )

    view = ConcatenatedChangeView(base, seeds, timestamp=5.0)
    assert view.get_xyz.shape == (3, 3)
    assert torch.equal(view.get_xyz[:2], base.get_xyz.detach())
    assert torch.equal(view.get_xyz[2:], seeds.get_active_render_attributes(5.0)["xyz"])
    # The concatenated tensor requires grad because active seed rows do, but
    # the base slice is detached from the original base parameter.
    assert view._features_dc.requires_grad
    assert view.get_opacity.shape == (3, 1)
    assert view.get_scaling.shape == (3, 3)
    assert view.get_rotation.shape == (3, 4)

    view._features_dc.sum().backward()
    assert base._features_dc.grad is None
    assert seeds.seed_dc.grad is not None
    assert seeds.seed_dc.grad[0].abs().sum() == 0
    assert seeds.seed_dc.grad[1].abs().sum() > 0
    for name, expected_hash in base_hashes.items():
        assert _tensor_sha(getattr(base, name)) == expected_hash


def test_zero_seed_concatenated_view_matches_base_shapes_and_values():
    base = DummyBase(n=2)
    seeds = NewSeedGaussianModel()
    view = ConcatenatedChangeView(base, seeds, timestamp=0.0)

    assert view.get_xyz.shape == base.get_xyz.shape
    assert view._features_dc.shape == base._features_dc.shape
    assert view._features_rest.shape == base._features_rest.shape
    assert view.get_opacity.shape == base.get_opacity.shape
    assert torch.equal(view.get_xyz, base.get_xyz.detach())
    assert torch.equal(view._features_dc, base._features_dc.detach())
    assert torch.equal(view._features_rest, base._features_rest.detach())


def test_densified_children_can_only_inherit_from_new_seed_bank():
    model = NewSeedGaussianModel()
    model.append(
        xyz=torch.tensor([[0.0, 0.0, 2.0]]),
        start=3.0,
        scaling=torch.full((1, 3), -3.0),
        dc=torch.tensor([[[0.2, 0.3, 0.4]]]),
        metadata={"seed_id": 4, "source_sign": "+"},
    )
    optimizer = torch.optim.Adam(model.optimizer_parameter_groups(lr=0.01))

    rows = model.append_densified_children(
        xyz=torch.tensor([[0.1, 0.0, 2.0], [-0.1, 0.0, 2.0]]),
        parent_rows=torch.tensor([0, 0]),
        start=5.0,
        scaling=torch.full((2, 3), -2.5),
        metadata=[{"support_frames": [1, 3, 5]}, {"support_frames": [2, 4, 5]}],
        optimizer=optimizer,
    )

    assert rows.tolist() == [1, 2]
    torch.testing.assert_close(model.seed_dc.detach()[1:], model.seed_dc.detach()[:1].expand(2, -1, -1))
    assert model.metadata[1]["birth_kind"] == "new_only_densified"
    assert model.metadata[1]["parent_seed_row"] == 0
    assert model.metadata[1]["parent_seed_id"] == 4
    assert model.metadata[1]["seed_id"] is None

    with pytest.raises(IndexError, match="existing NEW seed rows"):
        model.append_densified_children(
            xyz=torch.tensor([[1.0, 0.0, 2.0]]),
            parent_rows=torch.tensor([999]),
            start=6.0,
            scaling=torch.full((1, 3), -2.5),
        )
