import pytest
import torch


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
from temporal import TemporalGeometryChangeModel  # noqa: E402


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
