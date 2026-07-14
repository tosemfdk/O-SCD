# Finite-difference render Jacobian for one target Gaussian (stage 6).
# Correctness oracle: any faster backend must match this one on the toy scenes
# (docs/target_gaussian_nbv.md §7). 1 unperturbed + 2*D full renders, pixels
# sliced to a crop frozen from the unperturbed view.

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch

from utils.graphics_utils import fov2focal

from target_nbv.adapter import get_theta, fd_epsilons, perturbed
from target_nbv.config import TargetNBVConfig
from target_nbv.types import TargetParameterSpec
from target_nbv.visibility import project_gaussian_ellipse


@dataclass
class JacobianResult:
    valid: bool
    invalid_reason: Optional[str] = None
    J: Optional[torch.Tensor] = None        # (n_crop_px*3, D) float64 cpu
    w: Optional[torch.Tensor] = None        # (n_crop_px*3,) float64 cpu, obs precision
    crop_xyxy: Optional[tuple] = None
    diagnostics: dict = field(default_factory=dict)


def _render_crop(model, cam, pipe, background, crop):
    from gaussian_renderer import render  # local import: needs CUDA ext
    x0, y0, x1, y1 = crop
    with torch.no_grad():
        img = render(cam, model, pipe, background)["render"]
    return img[:, y0:y1, x0:x1]


def compute_target_jacobian(model, cam, row: int, spec: TargetParameterSpec,
                            pipe, background, cfg: TargetNBVConfig) -> JacobianResult:
    center, radii_px, depth, bbox = project_gaussian_ellipse(model, row, cam)
    if center is None:
        return JacobianResult(valid=False, invalid_reason="behind_camera")
    if bbox is None:
        return JacobianResult(valid=False, invalid_reason="outside_image")

    theta0 = get_theta(model, row, spec)
    eps = fd_epsilons(model, row, spec,
                      cfg.jacobian.mean_epsilon_rel, cfg.jacobian.log_scale_epsilon)

    # Freeze the crop from the unperturbed view: 3-sigma bbox + configured
    # margin + the pixel motion a mean perturbation can cause (risk 3).
    W, H = cam.image_width, cam.image_height
    f_px = fov2focal(cam.FoVx, W)
    eps_motion_px = float(eps.max()) * f_px / max(depth, 1e-6)
    margin = cfg.visibility.crop_margin_px + int(math.ceil(eps_motion_px)) + 1
    x0, y0, x1, y1 = bbox
    crop = (max(0, x0 - margin), max(0, y0 - margin),
            min(W, x1 + margin), min(H, y1 + margin))

    img0 = _render_crop(model, cam, pipe, background, crop)
    n_px = img0.numel()
    if n_px == 0:
        return JacobianResult(valid=False, invalid_reason="empty_crop")

    D = spec.dimension
    J = torch.empty((n_px, D), dtype=torch.float64)
    second_order = torch.empty(D, dtype=torch.float64)
    nonfinite = 0
    for k in range(D):
        e_k = torch.zeros(D, dtype=torch.float64); e_k[k] = eps[k]
        with perturbed(model, row, spec, theta0 + e_k,
                       verify_restoration=cfg.debug.verify_model_restoration):
            img_plus = _render_crop(model, cam, pipe, background, crop)
        with perturbed(model, row, spec, theta0 - e_k,
                       verify_restoration=cfg.debug.verify_model_restoration):
            img_minus = _render_crop(model, cam, pipe, background, crop)
        col = (img_plus.double() - img_minus.double()).reshape(-1) / (2.0 * float(eps[k]))
        nonfinite += int((~torch.isfinite(col)).sum())
        J[:, k] = col.cpu()
        second_order[k] = float(
            (img_plus.double() + img_minus.double() - 2.0 * img0.double()).norm())

    if nonfinite > 0:
        return JacobianResult(valid=False, invalid_reason="non_finite_jacobian",
                              diagnostics={"nonfinite": nonfinite})

    return JacobianResult(
        valid=True, J=J, w=torch.ones(n_px, dtype=torch.float64),
        crop_xyxy=crop,
        diagnostics={
            "crop_px": n_px // 3,
            "eps": eps.tolist(),
            "second_order_residual": second_order.tolist(),
            "col_norms": J.norm(dim=0).tolist(),
        },
    )
