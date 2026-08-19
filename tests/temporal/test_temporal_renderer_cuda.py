import pytest
import torch
import math
from types import SimpleNamespace
from torch import nn


pytestmark = pytest.mark.cuda

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for temporal renderer tests", allow_module_level=True)

pytest.importorskip("diff_gaussian_rasterization_fastgs")
pytest.importorskip("simple_knn._C")

from experiments.temporal_lifespan_smoke import (  # noqa: E402
    build_synthetic_temporal_scene,
    run_temporal_lifespan_smoke,
)
from gaussian_renderer import render_change, render_change_temporal  # noqa: E402
from scene.cameras import MiniCam  # noqa: E402
from scene.gaussian_model import GaussianModel  # noqa: E402
from temporal import opacity_removal_influence  # noqa: E402
from utils.general_utils import inverse_sigmoid  # noqa: E402
from utils.graphics_utils import getProjectionMatrix  # noqa: E402
from utils.sh_utils import RGB2SH  # noqa: E402
from temporal import (  # noqa: E402
    TemporalGeometryChangeModel,
    TemporalSharedGeometryChangeModel,
)


EXPECTED_TIMESTAMPS = (94.0, 95.0, 199.0)
EXPECTED_MASKS = {
    94.0: [True, True, True, False, False],
    95.0: [False, True, True, True, False],
    199.0: [False, False, True, True, True],
}
EXPECTED_INDICES = {
    94.0: [0, 0, 0, -1, -1],
    95.0: [-1, 1, 1, 1, -1],
    199.0: [-1, -1, 2, 2, 2],
}
BASE_FIELDS = ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation")


def _base_snapshot(base):
    return {name: getattr(base, name).detach().clone() for name in BASE_FIELDS}


def _base_shapes(base):
    return {name: tuple(getattr(base, name).shape) for name in BASE_FIELDS}


def _assert_base_unchanged(base, snapshot, shapes):
    assert _base_shapes(base) == shapes
    for name, expected in snapshot.items():
        assert torch.equal(getattr(base, name).detach(), expected), name


def _expected_pair_mask(timestamp):
    expected = torch.zeros((5, 3), dtype=torch.bool)
    for row, state in enumerate(EXPECTED_INDICES[timestamp]):
        if state >= 0:
            expected[row, state] = True
    return expected.tolist()


def _frame_by_timestamp(frames):
    return {float(frame["timestamp"]): frame for frame in frames}


def test_temporal_renderer_applies_per_gaussian_lifespans_and_isolates_gradients():
    result = run_temporal_lifespan_smoke(image_size=96)
    frames = _frame_by_timestamp(result["frames"])

    assert result["base_override_equivalence_max_error"] <= 1e-6
    assert result["state_dc_identical"]
    assert result["base_unchanged_after_backward"]
    assert result["base_gaussian_count"] == 5
    assert result["temporal_state_shape"] == [5, 3, 1, 3]
    assert set(frames) == set(EXPECTED_TIMESTAMPS)

    for timestamp in EXPECTED_TIMESTAMPS:
        frame = frames[timestamp]
        assert frame["active_state_indices"] == EXPECTED_INDICES[timestamp]
        assert frame["active_gaussians"] == EXPECTED_MASKS[timestamp]
        assert frame["gradient_pair_mask"] == _expected_pair_mask(timestamp)
        assert frame["gradient_isolated"]
        assert frame["opacity_isolated"]
        for roi_mean, active in zip(frame["roi_means"], EXPECTED_MASKS[timestamp]):
            if active:
                assert roi_mean > result["background_mean"] + 0.20
            else:
                assert roi_mean < result["background_mean"] + 0.04

    # Identical temporal DC slots mean the visible support comes from lifespan
    # opacity gating, not from state color differences.
    assert all(frames[timestamp]["render_mean"] > 0.0 for timestamp in EXPECTED_TIMESTAMPS)
    centroids = [frames[t]["centroid_x"] for t in EXPECTED_TIMESTAMPS]
    assert centroids[0] < centroids[1] < centroids[2]


def test_temporal_optimizer_updates_only_active_gaussian_state_pairs():
    result = run_temporal_lifespan_smoke(image_size=64)["optimizer_step"]

    assert result["only_active_pairs_changed"]
    assert result["base_unchanged"]
    assert result["state_parameter_unchanged"]
    assert result["state_shape_unchanged"]
    assert result["changed_pairs"] == _expected_pair_mask(95.0)


def test_temporal_renderer_preserves_base_identity_and_topology():
    model, camera, pipe, background = build_synthetic_temporal_scene(image_size=64)
    base_before = _base_snapshot(model.base)
    shapes_before = _base_shapes(model.base)

    assert torch.equal(
        model.state_change_dc[:, :1].expand_as(model.state_change_dc),
        model.state_change_dc,
    )

    for timestamp in EXPECTED_TIMESTAMPS:
        model.zero_grad(set_to_none=True)
        rendered = render_change_temporal(
            camera,
            model,
            pipe,
            background,
            timestamp=timestamp,
        )["render"]
        rendered.mean().backward()
        assert model.get_active_state_indices(timestamp).tolist() == EXPECTED_INDICES[timestamp]
        _assert_base_unchanged(model.base, base_before, shapes_before)


def test_temporal_renderer_checks_timestamp_and_override_contracts():
    model, camera, pipe, background = build_synthetic_temporal_scene(image_size=64)
    active_dc = model.get_active_change_dc(95.0)
    active_opacity = model.base.get_opacity * torch.tensor(
        EXPECTED_MASKS[95.0], device="cuda", dtype=model.base.get_opacity.dtype
    )[:, None]

    with pytest.raises(ValueError, match="timestamp is required"):
        render_change_temporal(camera, model, pipe, background)
    with pytest.raises(ValueError, match="cannot be used together"):
        render_change(
            camera,
            model.base,
            pipe,
            background,
            override_color=torch.zeros((5, 3), device="cuda"),
            override_dc=active_dc,
        )
    with pytest.raises(ValueError, match="shape"):
        render_change(camera, model.base, pipe, background, override_dc=active_dc[:, 0])
    with pytest.raises(ValueError, match="dtype"):
        render_change(camera, model.base, pipe, background, override_dc=active_dc.double())
    with pytest.raises(ValueError, match="same device"):
        render_change(camera, model.base, pipe, background, override_dc=active_dc.cpu())
    with pytest.raises(ValueError, match="shape"):
        render_change(
            camera,
            model.base,
            pipe,
            background,
            override_dc=active_dc,
            override_opacity=active_opacity[:, 0],
        )
    with pytest.raises(ValueError, match="dtype"):
        render_change(
            camera,
            model.base,
            pipe,
            background,
            override_dc=active_dc,
            override_opacity=active_opacity.double(),
        )
    with pytest.raises(ValueError, match="same device"):
        render_change(
            camera,
            model.base,
            pipe,
            background,
            override_dc=active_dc,
            override_opacity=active_opacity.cpu(),
        )

    camera.timestamp = 95.0
    rendered = render_change_temporal(camera, model, pipe, background)["render"]
    assert rendered.is_cuda

    pipe.convert_SHs_python = True
    python_sh_render = render_change(camera, model.base, pipe, background)["render"]
    override_render = render_change(
        camera,
        model.base,
        pipe,
        background,
        override_dc=model.base._features_dc,
        override_opacity=model.base.get_opacity,
    )["render"]
    assert torch.allclose(python_sh_render, override_render, atol=1e-6)


def test_signed_removal_influence_marks_dark_occluder_and_bright_change():
    device = torch.device("cuda")
    base = GaussianModel(sh_degree=3, active_sh_degree=0)
    base._xyz = nn.Parameter(
        torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 3.0]], device=device)
    )
    rgb = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], device=device)
    base._features_dc = nn.Parameter(RGB2SH(rgb).view(2, 1, 3))
    base._features_rest = nn.Parameter(torch.zeros((2, 15, 3), device=device))
    base._opacity = nn.Parameter(
        inverse_sigmoid(torch.full((2, 1), 0.8, device=device))
    )
    base._scaling = nn.Parameter(
        torch.full((2, 3), math.log(0.3), device=device)
    )
    rotation = torch.zeros((2, 4), device=device)
    rotation[:, 0] = 1.0
    base._rotation = nn.Parameter(rotation)

    fov = math.radians(60.0)
    world_view = torch.eye(4, device=device)
    projection = getProjectionMatrix(0.01, 100.0, fov, fov).t().to(device)
    full_projection = world_view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)
    camera = MiniCam(64, 64, fov, fov, 0.01, 100.0, world_view, full_projection)
    pipe = SimpleNamespace(
        compute_cov3D_python=False,
        convert_SHs_python=False,
        debug=False,
    )
    opacity = base.get_opacity.detach().clone().requires_grad_(True)
    rendered = render_change(
        camera,
        base,
        pipe,
        torch.zeros(3, device=device),
        override_dc=base._features_dc.detach(),
        override_opacity=opacity,
        clamp_output=False,
    )["render"]

    signed = opacity_removal_influence(rendered, opacity)

    assert signed[0] < 0  # dark front Gaussian suppresses the bright one
    assert signed[1] > 0  # bright rear Gaussian adds change brightness


def test_geometry_overrides_validate_and_receive_only_active_state_gradients():
    dc_model, camera, pipe, background = build_synthetic_temporal_scene(image_size=64)
    model = TemporalGeometryChangeModel.from_gaussians(
        dc_model.base,
        max_states=dc_model.max_states,
    )
    with torch.no_grad():
        model.state_change_dc.copy_(dc_model.state_change_dc)
        model.state_start.copy_(dc_model.state_start)
        model.state_end.copy_(dc_model.state_end)
        model.state_valid.copy_(dc_model.state_valid)
        model.state_xyz_delta[:, 1, 0] = 0.01
        model.state_scaling_delta[:, 1] = 0.02
        model.state_rotation_delta[:, 1, 1] = 0.01

    attributes = model.get_active_render_attributes(95.0)
    rendered = render_change_temporal(
        camera,
        model,
        pipe,
        background,
        timestamp=95.0,
    )["render"]
    rendered.sum().backward()

    expected_pairs = torch.tensor(_expected_pair_mask(95.0), device="cuda")
    for _name, parameter in model.state_parameter_items():
        pair_has_gradient = parameter.grad.flatten(start_dim=2).abs().sum(dim=2) > 0
        assert not pair_has_gradient[~expected_pairs].any()
    assert model.state_xyz_delta.grad[:, 1].abs().sum() > 0
    assert model.state_opacity_delta.grad[:, 1].abs().sum() > 0
    assert model.state_scaling_delta.grad[:, 1].abs().sum() > 0

    overrides = {
        "override_xyz": attributes["xyz"],
        "override_scaling": attributes["scaling"],
        "override_rotation": attributes["rotation"],
    }
    for name, value in overrides.items():
        bad_shape = value[:-1]
        with pytest.raises(ValueError, match="shape"):
            render_change(camera, model.base, pipe, background, **{name: bad_shape})
        with pytest.raises(ValueError, match="dtype"):
            render_change(camera, model.base, pipe, background, **{name: value.double()})
        with pytest.raises(ValueError, match="same device"):
            render_change(camera, model.base, pipe, background, **{name: value.cpu()})


def test_geometry_override_python_covariance_path_is_supported():
    dc_model, camera, pipe, background = build_synthetic_temporal_scene(image_size=64)
    model = TemporalGeometryChangeModel.from_gaussians(
        dc_model.base,
        max_states=dc_model.max_states,
    )
    with torch.no_grad():
        model.state_change_dc.copy_(dc_model.state_change_dc)
        model.state_start.copy_(dc_model.state_start)
        model.state_end.copy_(dc_model.state_end)
        model.state_valid.copy_(dc_model.state_valid)

    pipe.compute_cov3D_python = True
    rendered = render_change_temporal(
        camera,
        model,
        pipe,
        background,
        timestamp=95.0,
    )["render"]
    assert rendered.is_cuda
    assert torch.isfinite(rendered).all()


def test_shared_geometry_renderer_updates_only_currently_valid_rows():
    dc_model, camera, pipe, background = build_synthetic_temporal_scene(
        image_size=64
    )
    model = TemporalSharedGeometryChangeModel.from_gaussians(
        dc_model.base,
        max_states=dc_model.max_states,
    )
    with torch.no_grad():
        model.state_change_dc.copy_(dc_model.state_change_dc)
        model.state_start.copy_(dc_model.state_start)
        model.state_end.copy_(dc_model.state_end)
        model.state_valid.copy_(dc_model.state_valid)
        model.shared_xyz_delta[:, 0] = 0.01
        model.shared_scaling_delta[:] = 0.02
        model.shared_rotation_delta[:, 1] = 0.01

    rendered = render_change_temporal(
        camera,
        model,
        pipe,
        background,
        timestamp=95.0,
    )["render"]
    rendered.sum().backward()

    active = torch.tensor(EXPECTED_MASKS[95.0], device="cuda")
    dc_pairs = model.state_change_dc.grad.flatten(start_dim=2).abs().sum(dim=2) > 0
    assert not dc_pairs[:, 0].any()
    assert not dc_pairs[:, 2].any()
    assert not dc_pairs[~active, 1].any()
    for _name, parameter in model.shared_geometry_parameter_items():
        row_has_gradient = parameter.grad.flatten(start_dim=1).abs().sum(dim=1) > 0
        assert not row_has_gradient[~active].any()
    assert model.shared_xyz_delta.grad[active].abs().sum() > 0
    assert model.shared_opacity_delta.grad[active].abs().sum() > 0
    assert model.shared_scaling_delta.grad[active].abs().sum() > 0
