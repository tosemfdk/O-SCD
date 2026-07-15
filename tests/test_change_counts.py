# Gate G: adjoint soft counts vs per-Gaussian color-probe ground truth (GPU).
import pytest
import torch

from tests.conftest import make_scene, make_pipe
from tests.test_candidates_util import simple_cam
from target_nbv.change.counts import responsibility_probe_render, soft_counts
from target_nbv.visibility import render_target_responsibility

pytestmark = pytest.mark.gpu


def three_gaussian_scene():
    # target at origin, second off to the side, third BEHIND an occluder
    return make_scene(
        points=[[0.0, 0.0, 0.0], [0.6, 0.4, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.5]],
        scales=[[0.05] * 3, [0.05] * 3, [0.05] * 3, [0.15, 0.15, 0.02]],
        opacities=[[0.9], [0.9], [0.9], [0.99]])


def test_adjoint_counts_match_color_probe():
    model = three_gaussian_scene()
    cam, pipe = simple_cam(distance=2.0), make_pipe()
    H, W = cam.image_height, cam.image_width
    M = torch.zeros((H, W), device="cuda")
    M[:, : W // 2] = 1.0  # left half "changed"

    e1, e0, tau = soft_counts(model, cam, M, pipe)
    for g in range(4):
        r = render_target_responsibility(model, cam, g, pipe)  # exact alpha*T map
        assert float(e1[g]) == pytest.approx(float((r * M).sum()), rel=2e-3, abs=1e-3)
        assert float(tau[g]) == pytest.approx(float(r.sum()), rel=2e-3, abs=1e-3)
        assert float(e0[g]) == pytest.approx(float((r * (1 - M)).sum()), rel=2e-3, abs=1e-3)


def test_occluded_gaussian_gets_no_counts():
    model = three_gaussian_scene()
    cam, pipe = simple_cam(distance=2.0), make_pipe()
    M = torch.ones((cam.image_height, cam.image_width), device="cuda")
    e1, e0, tau = soft_counts(model, cam, M, pipe)
    # gaussian 2 sits behind the opaque plate (index 3) on the camera axis
    assert float(tau[2]) < 0.05 * float(tau[3])
    assert float(e1[2]) < 0.05 * float(e1[3])


def test_full_mask_puts_everything_in_e1():
    model = three_gaussian_scene()
    cam, pipe = simple_cam(distance=2.0), make_pipe()
    M = torch.ones((cam.image_height, cam.image_width), device="cuda")
    e1, e0, tau = soft_counts(model, cam, M, pipe)
    assert torch.allclose(e1, tau, atol=1e-6)
    assert float(e0.abs().max()) < 1e-6


def test_weighted_probe_one_hot_equals_responsibility():
    model = three_gaussian_scene()
    cam, pipe = simple_cam(distance=2.0), make_pipe()
    w = torch.zeros(4, device="cuda")
    w[0] = 1.0
    probe = responsibility_probe_render(model, cam, w, pipe)
    ref = render_target_responsibility(model, cam, 0, pipe)
    assert float((probe - ref).abs().max()) < 2e-3


def test_weighted_probe_sum_identity():
    # pixel-sum of the weighted probe == sum_g w_g * tau_g (the frame score)
    model = three_gaussian_scene()
    cam, pipe = simple_cam(distance=2.0), make_pipe()
    w = torch.tensor([0.9, 0.4, 0.7, 0.2], device="cuda")
    probe_sum = float(responsibility_probe_render(model, cam, w, pipe).sum())
    _, _, tau = soft_counts(model, cam,
                            torch.ones((cam.image_height, cam.image_width),
                                       device="cuda"), pipe)
    ref = float((w.double() * tau).sum())
    assert probe_sum == pytest.approx(ref, rel=5e-3)


def test_model_not_mutated():
    model = three_gaussian_scene()
    cam, pipe = simple_cam(distance=2.0), make_pipe()
    snap = {n: getattr(model, n).detach().clone()
            for n in ("_xyz", "_features_dc", "_opacity")}
    soft_counts(model, cam, torch.ones((cam.image_height, cam.image_width),
                                       device="cuda"), pipe)
    for name, before in snap.items():
        assert torch.equal(getattr(model, name).detach(), before)
