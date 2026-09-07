import pytest
import torch

pytestmark = pytest.mark.cuda
if not torch.cuda.is_available():
    pytest.skip("CUDA is required for alpha-T evidence VJP", allow_module_level=True)

pytest.importorskip("diff_gaussian_rasterization_fastgs")

from experiments.temporal_lifespan_smoke import build_synthetic_temporal_scene  # noqa: E402
from gaussian_renderer import render_change, render_change_temporal  # noqa: E402
from temporal.change_evidence import accumulate_change_evidence  # noqa: E402
from temporal.change_model import CLOSED, EMPTY, OPEN  # noqa: E402
from temporal.binary_state_filter import BinaryStateFilter  # noqa: E402
from temporal.dynamic_gaussian_topology import DynamicGaussianTopologyManager  # noqa: E402
from temporal.geometry_change_model import TemporalGeometryChangeModel  # noqa: E402
from temporal.masked_optimizer import MaskedRowAdam  # noqa: E402
from temporal.persistent_gaussian_lifespan_model import PersistentGaussianLifespanModel  # noqa: E402
from temporal.view_consistent_binary_lifespan_controller import (  # noqa: E402
    ViewConsistentBinaryLifespanController,
)

def test_override_color_contracts_cuda():
    model, camera, pipe, background = build_synthetic_temporal_scene(image_size=32)
    with pytest.raises(ValueError, match="shape"):
        render_change(camera, model.base, pipe, background, override_color=torch.zeros((5, 1), device="cuda"))
    with pytest.raises(ValueError, match="dtype"):
        render_change(camera, model.base, pipe, background, override_color=torch.zeros((5, 3), device="cuda", dtype=torch.float64))

def test_alpha_t_vjp_closed_gaussian_remains_observable_and_no_base_grad():
    model, camera, pipe, background = build_synthetic_temporal_scene(image_size=48)
    camera.timestamp = 95.0
    temporal = render_change_temporal(camera, model, pipe, background, timestamp=95.0)["render"]
    cue = torch.ones_like(temporal[:1])
    result = accumulate_change_evidence(camera, model.base, pipe, background, cue, cue_mode="binary", cue_threshold=0.5)
    assert result.total_mass[0] > 0  # row 0 is closed at timestamp 95 in temporal renderer
    for name in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"):
        assert getattr(model.base, name).grad is None

def test_alpha_t_evidence_is_bitwise_independent_of_temporal_geometry_and_lifecycle():
    model, camera, pipe, background = build_synthetic_temporal_scene(image_size=40)
    temporal_geometry = TemporalGeometryChangeModel.from_gaussians(model.base, max_states=4)
    cue = torch.linspace(0.0, 1.0, 40, device="cuda").repeat(40, 1)[None]

    expected_grad = {}
    for offset, name in enumerate(("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation")):
        parameter = getattr(model.base, name)
        parameter.grad = torch.arange(
            parameter.numel(), device=parameter.device, dtype=parameter.dtype
        ).reshape_as(parameter) + float(offset)
        expected_grad[name] = parameter.grad.clone()

    baseline = accumulate_change_evidence(
        camera,
        model.base,
        pipe,
        background,
        cue,
        cue_mode="soft",
        cue_scale=1.0,
        count_mode="capped",
        mass_saturation=0.75,
        min_evidence_mass=0.0,
    )
    baseline_tensors = {
        "delta_a": baseline.delta_a.clone(),
        "delta_b": baseline.delta_b.clone(),
        "total_mass": baseline.total_mass.clone(),
        "observed": baseline.observed.clone(),
    }

    n = temporal_geometry.state_valid.shape[0]
    rows = torch.arange(n, device="cuda")
    with torch.no_grad():
        temporal_geometry.state_change_dc.fill_(123.0)
        temporal_geometry.state_change_dc[:, 1].fill_(-77.0)
        temporal_geometry.state_xyz_delta.copy_(
            torch.linspace(
                -20.0,
                20.0,
                temporal_geometry.state_xyz_delta.numel(),
                device="cuda",
                dtype=temporal_geometry.state_xyz_delta.dtype,
            ).reshape_as(temporal_geometry.state_xyz_delta)
        )
        temporal_geometry.state_opacity_delta.fill_(15.0)
        temporal_geometry.state_scaling_delta.fill_(-12.0)
        temporal_geometry.state_rotation_delta.copy_(
            torch.linspace(
                -5.0,
                5.0,
                temporal_geometry.state_rotation_delta.numel(),
                device="cuda",
                dtype=temporal_geometry.state_rotation_delta.dtype,
            ).reshape_as(temporal_geometry.state_rotation_delta)
        )

        temporal_geometry.state_start.zero_()
        temporal_geometry.state_end.fill_(float("inf"))
        temporal_geometry.state_valid.zero_()
        temporal_geometry.state_status.zero_()
        temporal_geometry.num_states.zero_()
        temporal_geometry.current_state_index.fill_(-1)

        # Mix inactive, closed, and open rows while preserving lifecycle invariants.
        temporal_geometry.state_start[:, 0] = 0.0
        temporal_geometry.state_end[:, 0] = 1.0
        temporal_geometry.state_valid[:, 0] = True
        temporal_geometry.state_status[:, 0] = CLOSED
        temporal_geometry.num_states[:] = 1

        open_rows = rows[rows % 2 == 0]
        if open_rows.numel() > 0:
            temporal_geometry.state_start[open_rows, 1] = 2.0
            temporal_geometry.state_end[open_rows, 1] = float("inf")
            temporal_geometry.state_valid[open_rows, 1] = True
            temporal_geometry.state_status[open_rows, 1] = OPEN
            temporal_geometry.current_state_index[open_rows] = 1
            temporal_geometry.num_states[open_rows] = 2

        empty_rows = rows[rows % 3 == 0]
        if empty_rows.numel() > 0:
            temporal_geometry.state_start[empty_rows, 0] = 0.0
            temporal_geometry.state_end[empty_rows, 0] = float("inf")
            temporal_geometry.state_valid[empty_rows, 0] = False
            temporal_geometry.state_status[empty_rows, 0] = EMPTY
            temporal_geometry.num_states[empty_rows] = temporal_geometry.state_valid[
                empty_rows
            ].sum(dim=1)
            temporal_geometry.current_state_index[empty_rows] = torch.where(
                temporal_geometry.state_status[empty_rows, 1] == OPEN,
                torch.ones_like(empty_rows),
                torch.full_like(empty_rows, -1),
            )

    assert temporal_geometry.validate_lifecycle()

    changed = accumulate_change_evidence(
        camera,
        model.base,
        pipe,
        background,
        cue,
        cue_mode="soft",
        cue_scale=1.0,
        count_mode="capped",
        mass_saturation=0.75,
        min_evidence_mass=0.0,
    )

    assert torch.equal(changed.delta_a, baseline_tensors["delta_a"])
    assert torch.equal(changed.delta_b, baseline_tensors["delta_b"])
    assert torch.equal(changed.total_mass, baseline_tensors["total_mass"])
    assert torch.equal(changed.observed, baseline_tensors["observed"])
    for name, grad in expected_grad.items():
        assert torch.equal(getattr(model.base, name).grad, grad)

def test_alpha_t_soft_vjp_capped_binary_uses_exclusive_gaussian_increments_and_no_base_grad():
    model, camera, pipe, background = build_synthetic_temporal_scene(image_size=40)
    cue = torch.linspace(0.0, 1.0, 40, device="cuda").repeat(40, 1)[None]

    result = accumulate_change_evidence(
        camera,
        model.base,
        pipe,
        background,
        cue,
        cue_mode="soft",
        cue_scale=1.0,
        count_mode="capped_binary",
        mass_saturation=0.75,
        min_evidence_mass=0.0,
    )

    expected_sum = torch.clamp(result.total_mass / 0.75, 0.0, 1.0)
    expected_active = result.e_plus > result.e_minus
    assert torch.equal(result.delta_a > 0, expected_active & (expected_sum > 0))
    assert torch.equal(result.delta_b > 0, (~expected_active) & (expected_sum > 0))
    assert torch.equal(result.delta_a * result.delta_b, torch.zeros_like(result.delta_a))
    assert torch.allclose(result.delta_a + result.delta_b, expected_sum, atol=1e-6)
    assert torch.any((result.e_plus > 0) & (result.e_minus > 0))
    assert torch.any(result.e_plus > result.e_minus)
    assert torch.any(result.e_plus <= result.e_minus)
    for name in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"):
        assert getattr(model.base, name).grad is None

def test_alpha_t_vjp_matches_both_finite_difference_channels():
    model, camera, pipe, background = build_synthetic_temporal_scene(image_size=32)
    cue = torch.linspace(0.0, 1.0, 32, device="cuda").repeat(32, 1)[None]
    result = accumulate_change_evidence(camera, model.base, pipe, background, cue, cue_mode="binary")
    eps = 1e-3
    black = torch.zeros_like(background)
    color0 = torch.zeros((5, 3), device="cuda")
    render_kwargs = dict(
        override_opacity=model.base.get_opacity.detach(),
        override_xyz=model.base.get_xyz.detach(),
        override_scaling=model.base.get_scaling.detach(),
        override_rotation=model.base.get_rotation.detach(),
        clamp_output=False,
    )
    base_render = render_change(
        camera, model.base, pipe, black, override_color=color0, **render_kwargs
    )["render"]
    for channel, weights, evidence in (
        (0, result.cue[0] if result.cue.ndim == 3 else result.cue, result.e_plus),
        (1, 1.0 - (result.cue[0] if result.cue.ndim == 3 else result.cue), result.e_minus),
    ):
        row = int(torch.argmax(evidence).item())
        perturbed = color0.clone()
        perturbed[row, channel] = eps
        changed = render_change(
            camera, model.base, pipe, black, override_color=perturbed, **render_kwargs
        )["render"]
        fd = ((changed[channel] - base_render[channel]) * weights).sum() / eps
        assert torch.isclose(fd, evidence[row], rtol=5e-2, atol=5e-2)

def test_alpha_t_vjp_preserves_existing_base_gradients():
    model, camera, pipe, background = build_synthetic_temporal_scene(image_size=24)
    expected = {}
    for name in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"):
        parameter = getattr(model.base, name)
        parameter.grad = torch.randn_like(parameter)
        expected[name] = parameter.grad.clone()
    cue = torch.ones((1, 24, 24), device="cuda")
    accumulate_change_evidence(
        camera, model.base, pipe, background, cue, cue_mode="binary"
    )
    for name, grad in expected.items():
        assert torch.equal(getattr(model.base, name).grad, grad)

def test_dynamic_child_remains_detector_observable_after_it_becomes_inactive():
    synthetic, camera, pipe, background = build_synthetic_temporal_scene(image_size=32)
    temporal = PersistentGaussianLifespanModel(synthetic.base, max_states=4)
    temporal.reset_all_lifespans_closed()
    temporal.open_rows(torch.tensor([0], device="cuda"), timestamp=0)
    optimizer = MaskedRowAdam(
        dict(temporal.persistent_parameter_items()),
        lrs={name: 1e-3 for name, _ in temporal.persistent_parameter_items()},
    )
    tracker = BinaryStateFilter(temporal.current_state_index.numel(), device="cuda")
    controller = ViewConsistentBinaryLifespanController(temporal)
    topology = DynamicGaussianTopologyManager(
        temporal,
        optimizer,
        tracker,
        controller,
        percent_dense=1000.0,
    )
    topology.xyz_gradient_accum[0] = 1.0
    topology.denom[0] = 1.0
    result = topology.apply_active_oscd_density_control(
        timestamp=1,
        scene_extent=1.0,
        grad_threshold=1e-3,
        min_opacity=0.0,
    )
    assert result.clone_child_count == 1
    child = topology.count - 1
    temporal.close_rows(torch.tensor([child], device="cuda"), timestamp=2)
    attributes = temporal.get_active_render_attributes(timestamp=2)
    assert attributes["opacity"][child].item() == 0.0

    cue = torch.ones((1, 32, 32), device="cuda")
    evidence = accumulate_change_evidence(
        camera,
        temporal.base,
        pipe,
        background,
        cue,
        cue_mode="binary",
        cue_threshold=0.5,
    )
    assert evidence.total_mass[child] > 0
