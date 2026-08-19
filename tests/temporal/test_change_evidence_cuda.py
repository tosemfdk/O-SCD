import pytest
import torch

pytestmark = pytest.mark.cuda
if not torch.cuda.is_available():
    pytest.skip("CUDA is required for alpha-T evidence VJP", allow_module_level=True)

pytest.importorskip("diff_gaussian_rasterization_fastgs")

from experiments.temporal_lifespan_smoke import build_synthetic_temporal_scene  # noqa: E402
from gaussian_renderer import render_change, render_change_temporal  # noqa: E402
from temporal.change_evidence import accumulate_change_evidence  # noqa: E402


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
