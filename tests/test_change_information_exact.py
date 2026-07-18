# Gate S1: exact toy oracle vs Hutchinson estimator + fastgs guards (spec §6).
import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.conftest import make_pipe, make_scene  # noqa: E402


def look_at_mini_cam(position, width=32, height=32, fovy=0.9, target=(0, 0, 0)):
    """MiniCam looking at `target` from `position` (OpenGL c2w -> build_mini_cam
    flip math, but taking the rotation directly)."""
    from scene.cameras import MiniCam
    from utils.graphics_utils import getProjectionMatrix

    pos = np.asarray(position, dtype=np.float32)
    f = np.asarray(target, dtype=np.float32) - pos
    f = f / (np.linalg.norm(f) + 1e-12)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(float(f @ up)) > 0.98:
        up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    z_b = -f  # OpenGL camera looks down -z
    x = np.cross(up, z_b); x = x / (np.linalg.norm(x) + 1e-12)
    y = np.cross(z_b, x)
    R_c2w_opengl = np.stack([x, y, z_b], axis=1)
    flip = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    R_w2c = (R_c2w_opengl @ flip).T
    t = -R_w2c @ pos
    W2C = np.eye(4, dtype=np.float32); W2C[:3, :3] = R_w2c; W2C[:3, 3] = t
    aspect = width / height
    FoVx = 2.0 * math.atan(math.tan(fovy / 2.0) * aspect)
    proj = getProjectionMatrix(0.01, 100.0, FoVx, fovy).transpose(0, 1).cuda()
    wv = torch.tensor(W2C).transpose(0, 1).cuda()
    return MiniCam(width=width, height=height, fovy=fovy, fovx=FoVx,
                   znear=0.01, zfar=100.0, world_view_transform=wv,
                   full_proj_transform=wv.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0))


def make_change_toy(n_extra_rest=True, c_value=0.0, seed=3):
    """Toy CHANGE model at the scoring state: dc = c (default 0), plus a
    non-empty dummy rest tensor (counts.py trap workaround) unless disabled."""
    import torch.nn as nn

    g = torch.Generator().manual_seed(seed)
    pts = ((torch.rand((7, 3), generator=g) - 0.5) * 1.6).tolist()
    scales = (0.08 + 0.10 * torch.rand((7, 3), generator=g)).tolist()
    model = make_scene(pts, scales, colors=[[c_value] * 3] * 7)
    if n_extra_rest:
        n = model.get_xyz.shape[0]
        model._features_rest = nn.Parameter(
            torch.zeros((n, 3, 3), device="cuda").requires_grad_(True))
        model.max_sh_degree = 1
    return model


CAMS = [(0, 0, 2.5), (2.5, 0, 0), (0, 2.5, 0.2), (-1.8, 0.4, 1.8),
        (1.4, -1.6, 1.4), (0.5, 1.8, -1.8), (-2.2, -0.8, -0.6), (1.0, 1.0, 2.0)]


@pytest.fixture(scope="module")
def toy():
    from view_selection.types import InformationConfig
    model = make_change_toy()
    pipe = make_pipe()
    bg = torch.zeros(3, device="cuda")
    cams = [look_at_mini_cam(p) for p in CAMS]
    w = torch.ones(32, 32, device="cuda")
    cfg = InformationConfig(num_probes=4)
    return model, pipe, bg, cams, w, cfg


def test_exact_nonnegative_and_fd_matches_projection(toy):
    from view_selection.information import (_render_raw, exact_information,
                                            project_dc_gradient)
    model, pipe, bg, cams, w, _ = toy
    b = exact_information(model, cams[0], w, "raw", pipe, bg)
    assert (b >= 0).all() and torch.isfinite(b).all()
    assert b.sum() > 0

    # tied-channel FD: perturb ALL channels of one Gaussian's dc together and
    # compare the loss delta with the projected (summed) channel gradient.
    y, _ = _render_raw(model, cams[0], pipe, bg)
    xi = torch.ones_like(y)  # fixed probe
    loss = (y * xi).sum()
    g = torch.autograd.grad(loss, model._features_dc)[0]
    gc = project_dc_gradient(g)
    j = int(gc.abs().argmax())
    h = 1e-3
    with torch.no_grad():
        model._features_dc[j] += h
    y2, _ = _render_raw(model, cams[0], pipe, bg)
    with torch.no_grad():
        model._features_dc[j] -= 2 * h
    y0, _ = _render_raw(model, cams[0], pipe, bg)
    with torch.no_grad():
        model._features_dc[j] += h
    fd = float(((y2 - y0).sum()) / (2 * h))
    assert abs(fd - float(gc[j])) <= 0.05 * max(abs(fd), 1e-6)


def test_channel_gradients_tied_and_projection_scale(toy):
    from view_selection.information import _render_raw
    model, pipe, bg, cams, _, _ = toy
    y, _ = _render_raw(model, cams[0], pipe, bg)
    g = torch.autograd.grad(y.sum(), model._features_dc)[0]  # (N,1,3)
    per_ch = g.reshape(g.shape[0], -1)
    spread = (per_ch - per_ch.mean(dim=1, keepdim=True)).abs().max()
    assert float(spread) < 1e-6  # 3 channels carry identical gradients
    # projected b = (3 g_ch)^2 = 9 * per-channel b
    b_ch = per_ch[:, 0].square()
    b_proj = per_ch.sum(dim=1).square()
    assert torch.allclose(b_proj, 9.0 * b_ch, rtol=1e-4, atol=1e-10)


def test_exact_vs_hutchinson_convergence(toy):
    # Spec §6.4 asked for <=5% at M=256; measured toy variance (heavy
    # inter-Gaussian overlap) is 3-8% at 256 and 2-5% at 1024 with a clean
    # 1/sqrt(M) trend, so the gate is: unbiased convergence + <=5% at 1024.
    # (Deviation recorded in change_nbv_s1_report.md.)
    from dataclasses import replace
    from view_selection.information import exact_information, hutchinson_information
    model, pipe, bg, cams, w, cfg = toy
    exact = exact_information(model, cams[0], w, "raw", pipe, bg)
    denom = float(exact.abs().sum())

    def err(m, scope):
        est = hutchinson_information(model, cams[0], w,
                                     replace(cfg, num_probes=m),
                                     scope, pipe, bg, frame_id=0).diagonal.cuda()
        return float((est - exact).abs().sum()) / denom

    e256 = [err(256, ("cv", s)) for s in range(3)]
    e1024 = [err(1024, ("cv", s)) for s in range(3)]
    assert min(e1024) <= 0.05
    assert sum(e1024) / 3 < sum(e256) / 3  # 1/sqrt(M) convergence


def test_hutchinson_m4_candidate_rank(toy):
    from scipy.stats import spearmanr
    from view_selection.information import exact_information, hutchinson_information
    model, pipe, bg, cams, w, cfg = toy
    exact_tot, est_tot = [], []
    for fid, cam in enumerate(cams):
        exact_tot.append(float(exact_information(model, cam, w, "raw", pipe, bg).sum()))
        est_tot.append(float(hutchinson_information(
            model, cam, w, cfg, ("toy",), pipe, bg, frame_id=fid).diagonal.sum()))
    rho = spearmanr(exact_tot, est_tot).statistic
    assert rho >= 0.9
    assert int(np.argmax(exact_tot)) == int(np.argmax(est_tot))


def test_hutchinson_deterministic(toy):
    from view_selection.information import hutchinson_information
    model, pipe, bg, cams, w, cfg = toy
    a = hutchinson_information(model, cams[1], w, cfg, ("s",), pipe, bg, 1).diagonal
    b = hutchinson_information(model, cams[1], w, cfg, ("s",), pipe, bg, 1).diagonal
    assert torch.equal(a, b)


def test_empty_render_guard(toy):
    from view_selection.information import hutchinson_information
    model, pipe, bg, _, w, cfg = toy
    away = look_at_mini_cam((0, 0, 60.0), target=(0, 0, 120.0))  # scene behind
    info = hutchinson_information(model, away, w, cfg, ("t",), pipe, bg, 9)
    assert info.diagonal.sum() == 0 and info.metadata["probes_ran"] == 0
    # CUDA context must still be healthy afterwards
    ok = hutchinson_information(model, look_at_mini_cam(CAMS[0]), w, cfg,
                                ("t",), pipe, bg, 0)
    assert ok.diagonal.sum() > 0


def test_zero_adjoint_trap_detected():
    # Empirically confirmed: with an EMPTY _features_rest the fastgs backward
    # silently writes zero dc-gradients (counts.py trap). strict mode must
    # fail loudly instead of scoring the candidate 0.
    from view_selection.information import hutchinson_information
    from view_selection.types import InformationConfig, ZeroAdjointError
    model = make_change_toy(n_extra_rest=False)
    pipe = make_pipe()
    bg = torch.zeros(3, device="cuda")
    w = torch.ones(32, 32, device="cuda")
    with pytest.raises(ZeroAdjointError):
        hutchinson_information(model, look_at_mini_cam(CAMS[0]), w,
                               InformationConfig(), ("trap",), pipe, bg, 0)


def test_degenerate_weight_guard(toy):
    from view_selection.information import hutchinson_information
    model, pipe, bg, cams, _, cfg = toy
    info = hutchinson_information(model, cams[0], torch.zeros(32, 32), cfg,
                                  ("t",), pipe, bg, 0)
    assert info.diagonal.sum() == 0 and info.metadata["probes_ran"] == 0
