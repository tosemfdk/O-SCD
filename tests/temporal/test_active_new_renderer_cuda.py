import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytestmark = pytest.mark.cuda

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for active NEW renderer tests", allow_module_level=True)

pytest.importorskip("diff_gaussian_rasterization_fastgs")
pytest.importorskip("simple_knn._C")

from scene.cameras import MiniCam
from scene.gaussian_model import GaussianModel
from temporal.active_new_density import (
    accumulate_density_statistics,
    new_geometry_coverage_loss,
    render_active_new_coverage,
)
from temporal.active_new_gaussians import (
    ActiveNewGaussianModel,
    ActiveNewGeometryView,
    FrozenBaseActiveNewView,
)
from temporal.new_seed_gaussians import (
    NewSeedGaussianModel,
    build_concatenated_change_view,
)
from experiments.run_online_xfeat_new_seed import render_new_sidecar_alpha_score
from utils.general_utils import inverse_sigmoid
from utils.graphics_utils import getProjectionMatrix


def _camera(size=64):
    fov = math.radians(60.0)
    world = torch.eye(4, device="cuda")
    projection = getProjectionMatrix(0.01, 100.0, fov, fov).t().cuda()
    full = world.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)
    return MiniCam(size, size, fov, fov, 0.01, 100.0, world, full)


def _base():
    model = GaussianModel(sh_degree=0, active_sh_degree=0)
    model._xyz = nn.Parameter(torch.tensor([[0.3, 0.0, 2.4]], device="cuda"))
    model._features_dc = nn.Parameter(torch.zeros((1, 1, 3), device="cuda"))
    model._features_rest = nn.Parameter(torch.zeros((1, 0, 3), device="cuda"))
    model._opacity = nn.Parameter(inverse_sigmoid(torch.full((1, 1), 0.5, device="cuda")))
    model._scaling = nn.Parameter(torch.full((1, 3), math.log(0.1), device="cuda"))
    model._rotation = nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda"))
    return model


def test_active_new_geometry_receives_finite_nonzero_gradients_and_base_is_frozen():
    base = _base()
    base_before = {
        name: getattr(base, name).detach().clone()
        for name in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation")
    }
    model = ActiveNewGaussianModel(device="cuda")
    optimizer = torch.optim.Adam(
        model.optimizer_parameter_groups(
            xyz_lr=1e-3,
            dc_lr=1e-3,
            opacity_lr=1e-3,
            scaling_lr=1e-3,
            rotation_lr=1e-3,
        ),
        eps=1e-15,
    )
    model.append_xfeat_anchors(
        xyz=torch.tensor([[0.15, -0.08, 2.0]], device="cuda"),
        start=0.0,
        scaling=torch.tensor([[math.log(0.28), math.log(0.10), math.log(0.06)]], device="cuda"),
        rotation=torch.tensor([[0.96, 0.0, 0.0, 0.28]], device="cuda"),
        opacity=0.4,
        optimizer=optimizer,
    )
    camera = _camera()
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
    background = torch.zeros(3, device="cuda")
    active = model.active_view(0.0)
    package = render_active_new_coverage(camera, active, pipe, background)
    target = torch.zeros((1, 64, 64), device="cuda")
    target[:, 25:48, 20:36] = 1.0
    loss, _ = new_geometry_coverage_loss(target, package["render"])
    loss.backward()
    accumulate_density_statistics(model, active, package)

    for parameter in (model._xyz, model._scaling, model._rotation, model._opacity):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert float(parameter.grad.abs().sum()) > 0.0
    optimizer.step()
    for name, before in base_before.items():
        assert torch.equal(getattr(base, name).detach(), before), name
        assert getattr(base, name).grad is None
    assert float(model.gradient_denom.sum()) > 0.0


def test_e4a_dc_view_detaches_new_geometry_but_keeps_new_dc_trainable():
    base = _base()
    model = ActiveNewGaussianModel(device="cuda")
    model.append_xfeat_anchors(
        xyz=torch.tensor([[0.0, 0.0, 2.0]], device="cuda"),
        start=0.0,
        scaling=torch.full((1, 3), math.log(0.1), device="cuda"),
    )
    combined = FrozenBaseActiveNewView(
        base,
        model,
        timestamp=0.0,
        base_dc=torch.zeros_like(base._features_dc),
        detach_new_geometry=True,
    )
    combined.get_features[1:].sum().backward()

    assert model.new_dc.grad is not None
    assert float(model.new_dc.grad.abs().sum()) > 0.0
    for parameter in (model._xyz, model._opacity, model._scaling, model._rotation):
        assert parameter.grad is None
    for name in (
        "_xyz",
        "_features_dc",
        "_features_rest",
        "_opacity",
        "_scaling",
        "_rotation",
    ):
        assert getattr(base, name).grad is None


def test_seed_only_dc_view_keeps_dc_trainable_and_detaches_geometry():
    model = ActiveNewGaussianModel(device="cuda")
    model.append_xfeat_anchors(
        xyz=torch.tensor([[0.0, 0.0, 2.0]], device="cuda"),
        start=0.0,
        scaling=torch.full((1, 3), math.log(0.1), device="cuda"),
    )
    active_only = ActiveNewGeometryView(
        model,
        0.0,
        detach_geometry=True,
    )
    active_only.get_features.sum().backward()

    assert model.new_dc.grad is not None
    assert float(model.new_dc.grad.abs().sum()) > 0.0
    for parameter in (model._xyz, model._opacity, model._scaling, model._rotation):
        assert parameter.grad is None


def test_e4d_dc_adapter_matches_e4a_joint_render_state():
    base = _base()
    base_dc = torch.zeros_like(base._features_dc)
    xyz = torch.tensor([[0.0, 0.0, 2.0]], device="cuda")
    scaling = torch.full((1, 3), math.log(0.1), device="cuda")
    dc = torch.full((1, 1, 3), 0.3, device="cuda")

    legacy = NewSeedGaussianModel(device="cuda")
    legacy.append(
        xyz=xyz,
        start=0.0,
        scaling=scaling,
        opacity=0.1,
        dc=dc,
    )
    active = ActiveNewGaussianModel(device="cuda")
    active.append_xfeat_anchors(
        xyz=xyz,
        start=0.0,
        scaling=scaling,
        opacity=0.1,
        dc=dc,
    )
    expected = build_concatenated_change_view(
        base,
        legacy,
        timestamp=0.0,
        base_dc=base_dc,
        detach_base_dc=True,
    )
    actual = FrozenBaseActiveNewView(
        base,
        active,
        timestamp=0.0,
        base_dc=base_dc,
        detach_new_geometry=True,
    )

    for name in ("get_xyz", "get_features", "get_opacity", "get_scaling", "get_rotation"):
        assert torch.equal(getattr(actual, name).detach(), getattr(expected, name).detach())


def test_evaluation_sidecar_alpha_score_does_not_train_dc():
    model = ActiveNewGaussianModel(device="cuda")
    model.append_xfeat_anchors(
        xyz=torch.tensor([[0.0, 0.0, 2.0]], device="cuda"),
        start=0.0,
        scaling=torch.full((1, 3), math.log(0.16), device="cuda"),
        opacity=0.8,
        dc=torch.full((1, 1, 3), -4.0, device="cuda"),
    )
    camera = _camera()
    camera.timestamp = 0.0
    pipe = SimpleNamespace(
        compute_cov3D_python=False, convert_SHs_python=False, debug=False
    )
    score = render_new_sidecar_alpha_score(
        camera, model, pipe, torch.zeros(3, device="cuda")
    )

    assert score.shape == (1, 64, 64)
    assert float(score.detach().max()) > 0.5
    score.sum().backward()
    assert model.new_dc.grad is None
    assert model._opacity.grad is not None
