# Gate C: target visibility backend A' (metric_map). GPU.
import pytest
import torch

from tests.conftest import make_scene, make_pipe
from tests.test_candidates_util import simple_cam
from target_nbv.config import TargetNBVConfig
from target_nbv.visibility import make_visibility_backend, project_gaussian_ellipse

pytestmark = pytest.mark.gpu

CFG = TargetNBVConfig().validate()
BG = None


def setup_module(module):
    global BG
    BG = torch.zeros(3, device="cuda")


def _eval(model, cam, row=0):
    backend = make_visibility_backend(CFG)
    return backend.evaluate(model, cam, row, make_pipe(), BG)


def test_visible_target():
    model = make_scene(points=[[0.0, 0.0, 0.0]], scales=[[0.05, 0.05, 0.05]])
    vis = _eval(model, simple_cam(distance=2.0))
    assert vis.valid
    assert vis.visible_pixel_count > 10
    assert vis.occlusion_ratio < 0.5
    assert vis.projected_radius_px > 1.0


def test_fully_occluded_target_invalid():
    # opaque occluder between camera (z=-2) and target (origin), same LOS
    model = make_scene(
        points=[[0.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
        scales=[[0.05, 0.05, 0.05], [0.5, 0.5, 0.05]],
        opacities=[[0.9], [0.999]],
    )
    vis = _eval(model, simple_cam(distance=2.0), row=0)
    assert not vis.valid
    assert vis.invalid_reason in ("fully_occluded", "occlusion_above_threshold", "not_rendered")


def test_partial_occlusion_ordering():
    free = make_scene(points=[[0.0, 0.0, 0.0]], scales=[[0.05, 0.05, 0.05]])
    partial = make_scene(
        points=[[0.0, 0.0, 0.0], [0.05, 0.0, -1.0]],
        scales=[[0.05, 0.05, 0.05], [0.03, 0.06, 0.03]],
        opacities=[[0.9], [0.999]],
    )
    v_free = _eval(free, simple_cam(distance=2.0), row=0)
    v_part = _eval(partial, simple_cam(distance=2.0), row=0)
    assert v_free.valid
    assert v_free.responsibility_sum > v_part.responsibility_sum
    assert v_free.occlusion_ratio < v_part.occlusion_ratio + 1e-6


def test_target_behind_camera():
    model = make_scene(points=[[0.0, 0.0, -5.0]], scales=[[0.05, 0.05, 0.05]])
    vis = _eval(model, simple_cam(distance=2.0))  # camera at z=-2 facing +z
    assert not vis.valid
    assert vis.invalid_reason == "behind_camera"


def test_evaluate_does_not_mutate_model():
    model = make_scene(points=[[0.0, 0.0, 0.0], [0.3, 0.0, -1.0]],
                       scales=[[0.05, 0.05, 0.05], [0.05, 0.05, 0.05]])
    snap = {name: getattr(model, name).detach().clone()
            for name in ("_xyz", "_scaling", "_rotation", "_opacity", "_features_dc")}
    _eval(model, simple_cam(distance=2.0), row=0)
    for name, before in snap.items():
        assert torch.equal(getattr(model, name).detach(), before), f"{name} mutated"


def test_responsibility_map_is_alpha_T():
    # single isotropic gaussian, opacity 0.9: peak responsibility ~= 0.9 at center
    model = make_scene(points=[[0.0, 0.0, 0.0]], scales=[[0.05, 0.05, 0.05]],
                       opacities=[[0.9]])
    from target_nbv.visibility import render_target_responsibility
    resp = render_target_responsibility(model, simple_cam(distance=2.0), 0, make_pipe())
    assert float(resp.max()) == pytest.approx(0.9, abs=0.03)
    assert float(resp.min()) >= 0.0


def test_ellipse_projection_matches_radii():
    model = make_scene(points=[[0.0, 0.0, 0.0]], scales=[[0.05, 0.05, 0.05]])
    cam = simple_cam(distance=2.0)
    center, radii_px, depth, bbox = project_gaussian_ellipse(model, 0, cam)
    assert depth == pytest.approx(2.0, rel=1e-4)
    # center pixel should be image center
    assert center[0] == pytest.approx((cam.image_width - 1) / 2, abs=1.0)
    assert center[1] == pytest.approx((cam.image_height - 1) / 2, abs=1.0)
    # isotropic gaussian: both semi-axes equal, = 3*sigma*f/d
    from utils.graphics_utils import fov2focal
    expected = 3.0 * 0.05 * fov2focal(cam.FoVx, cam.image_width) / 2.0
    assert float(radii_px[0]) == pytest.approx(expected, rel=0.05)
    assert float(radii_px[1]) == pytest.approx(expected, rel=0.05)
