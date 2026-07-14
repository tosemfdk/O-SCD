# Target visibility / responsibility backends (docs/target_gaussian_nbv.md §8).
#
# Backend A'' ("color_probe"): render a recolored PROXY of the model (target DC
# color -> 1, all others -> 0, sh_degree 0). The rendered image is then exactly
# the per-pixel compositing responsibility of the target, r_t,p = alpha_t,p *
# T_t,p, with no CUDA changes and no mutation of the real model. A second
# single-row render gives the unoccluded self-responsibility for the occlusion
# ratio. (The earlier metric_map/accum_metric_counts idea was rejected: the
# count condition in forward.cu:387-406 is alpha>=1/255 and T>=1e-4, ignoring
# the alpha*T magnitude, so a strong occluder barely reduces the count.)
#
# Backend B ("exact_alpha_t"): per-pixel alpha*T exported from CUDA; stub —
# color_probe already yields the exact value, so B exists only if we ever need
# it fused into one pass for speed.

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from types import SimpleNamespace

import torch

from utils.general_utils import build_scaling_rotation
from utils.graphics_utils import fov2focal

from target_nbv.config import TargetNBVConfig
from target_nbv.types import TargetVisibility

SH_C0 = 0.28209479177387814  # utils/sh_utils.py


def project_gaussian_ellipse(model, row: int, cam):
    """Analytic EWA projection of one Gaussian.

    Returns (center_uv, radii_px (2,), depth, bbox_xyxy or None). radii are the
    3-sigma semi-axes of the projected 2D covariance, in pixels. bbox is clipped
    to the image and None if the ellipse lies fully outside.
    """
    device = model._xyz.device
    mu = model._xyz[row].detach()
    wvt = cam.world_view_transform  # w2c, transposed (row-vector layout)
    p_cam = (torch.cat([mu, torch.ones(1, device=device)]) @ wvt)[:3]
    x, y, z = float(p_cam[0]), float(p_cam[1]), float(p_cam[2])
    if z <= cam.znear:
        return None, None, z, None

    W, H = cam.image_width, cam.image_height
    fx = fov2focal(cam.FoVx, W)
    fy = fov2focal(cam.FoVy, H)
    cx_px, cy_px = (W - 1) / 2.0, (H - 1) / 2.0
    u = fx * x / z + cx_px
    v = fy * y / z + cy_px

    # Sigma_3D = L L^T in world; rotate into camera: R_w2c Sigma R_w2c^T
    L = build_scaling_rotation(model.get_scaling[row:row + 1].detach(),
                               model._rotation[row:row + 1].detach())[0]
    Sigma_w = L @ L.T
    R_w2c = wvt[:3, :3].T  # undo the transposed storage
    Sigma_c = R_w2c @ Sigma_w @ R_w2c.T

    J = torch.tensor([[fx / z, 0.0, -fx * x / (z * z)],
                      [0.0, fy / z, -fy * y / (z * z)]], device=device)
    Sigma_2d = J @ Sigma_c @ J.T
    Sigma_2d = 0.5 * (Sigma_2d + Sigma_2d.T) + 1e-9 * torch.eye(2, device=device)
    eigvals = torch.linalg.eigvalsh(Sigma_2d).clamp_min(1e-12)
    radii_px = 3.0 * eigvals.sqrt()  # (2,) ascending

    r_max = float(radii_px[1])
    x0, y0 = int(math.floor(u - r_max)), int(math.floor(v - r_max))
    x1, y1 = int(math.ceil(u + r_max)) + 1, int(math.ceil(v + r_max)) + 1
    x0c, y0c, x1c, y1c = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    bbox = (x0c, y0c, x1c, y1c) if (x0c < x1c and y0c < y1c) else None
    return (u, v), radii_px, z, bbox


def _probe_dc(n: int, row: int | None, device) -> torch.Tensor:
    """DC features rendering to color 1 for `row` and 0 elsewhere (sh_degree 0:
    color = SH_C0 * dc + 0.5)."""
    dc = torch.full((n, 1, 3), -0.5 / SH_C0, device=device)
    if row is not None:
        dc[row] = 0.5 / SH_C0
    return dc


class _ProxyModel(SimpleNamespace):
    """Duck-typed stand-in for GaussianModel accepted by gaussian_renderer.render
    (default pipe: compute_cov3D_python=False, convert_SHs_python=False)."""


def _make_probe_proxy(model, target_row: int, only_target: bool) -> _ProxyModel:
    n = model.get_xyz.shape[0]
    device = model.get_xyz.device
    if only_target:
        sl = slice(target_row, target_row + 1)
        return _ProxyModel(
            get_xyz=model.get_xyz[sl].detach(),
            get_opacity=model.get_opacity[sl].detach(),
            get_scaling=model.get_scaling[sl].detach(),
            get_rotation=model.get_rotation[sl].detach(),
            _features_dc=_probe_dc(1, 0, device),
            _features_rest=model._features_rest[sl].detach(),
            active_sh_degree=0,
            max_sh_degree=model.max_sh_degree,
        )
    return _ProxyModel(
        get_xyz=model.get_xyz.detach(),
        get_opacity=model.get_opacity.detach(),
        get_scaling=model.get_scaling.detach(),
        get_rotation=model.get_rotation.detach(),
        _features_dc=_probe_dc(n, target_row, device),
        _features_rest=model._features_rest.detach(),
        active_sh_degree=0,
        max_sh_degree=model.max_sh_degree,
    )


def render_target_responsibility(model, cam, target_row: int, pipe,
                                 only_target: bool = False) -> torch.Tensor:
    """Per-pixel responsibility map r_t,p = alpha_t,p * T_t,p, shape (H, W)."""
    from gaussian_renderer import render  # local import: needs CUDA ext

    proxy = _make_probe_proxy(model, target_row, only_target)
    bg = torch.zeros(3, device="cuda")
    with torch.no_grad():
        pkg = render(cam, proxy, pipe, bg)
    return pkg["render"].mean(dim=0)  # 3 identical channels


class VisibilityBackend(ABC):
    @abstractmethod
    def evaluate(self, model, cam, target_row: int, pipe, background,
                 return_map: bool = False) -> TargetVisibility:
        ...


class ColorProbeVisibilityBackend(VisibilityBackend):
    """Backend A''. Two renders per evaluation: full-scene probe (actual
    responsibility) + target-only probe (unoccluded self-responsibility)."""

    def __init__(self, cfg: TargetNBVConfig):
        self.cfg = cfg

    def evaluate(self, model, cam, target_row: int, pipe, background,
                 return_map: bool = False) -> TargetVisibility:
        center, radii_px, depth, bbox = project_gaussian_ellipse(model, target_row, cam)
        if center is None:
            return TargetVisibility(valid=False, invalid_reason="behind_camera")
        if bbox is None:
            return TargetVisibility(valid=False, invalid_reason="outside_image")

        resp = render_target_responsibility(model, cam, target_row, pipe, only_target=False)
        resp_sum = float(resp.sum())
        unocc = render_target_responsibility(model, cam, target_row, pipe, only_target=True)
        unocc_sum = float(unocc.sum())

        if unocc_sum < 1e-9:
            return TargetVisibility(valid=False, invalid_reason="not_rendered",
                                    bbox_xyxy=bbox, projected_radius_px=float(radii_px[1]))

        occlusion_ratio = min(1.0, max(0.0, 1.0 - resp_sum / unocc_sum))
        visible_px = int((resp > 1.0 / 255.0).sum())
        vis = TargetVisibility(
            valid=True,
            bbox_xyxy=bbox,
            projected_radius_px=float(radii_px[1]),
            responsibility_sum=resp_sum,
            responsibility_mean=resp_sum / max(visible_px, 1),
            visible_pixel_count=visible_px,
            occlusion_ratio=occlusion_ratio,
        )
        if self.cfg.visibility.reject_fully_occluded and resp_sum < self.cfg.visibility.min_responsibility:
            vis.valid = False
            vis.invalid_reason = "fully_occluded"
        elif occlusion_ratio > self.cfg.visibility.max_occlusion_ratio:
            vis.valid = False
            vis.invalid_reason = "occlusion_above_threshold"
        if return_map:
            vis.responsibility_map = resp
        return vis


class ExactAlphaTBackend(VisibilityBackend):
    """Backend B: single-pass per-pixel alpha*T export from CUDA. color_probe
    already returns the exact value in two passes; implement B only if the
    extra pass ever matters."""

    def evaluate(self, model, cam, target_row: int, pipe, background,
                 return_map: bool = False) -> TargetVisibility:
        raise NotImplementedError(
            "exact_alpha_t requires a rasterizer modification; use 'color_probe'")


def make_visibility_backend(cfg: TargetNBVConfig) -> VisibilityBackend:
    if cfg.visibility.backend == "color_probe":
        return ColorProbeVisibilityBackend(cfg)
    if cfg.visibility.backend == "exact_alpha_t":
        return ExactAlphaTBackend()
    raise ValueError(f"unknown visibility backend {cfg.visibility.backend!r}")
