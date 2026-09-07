"""Projected 2-D covariance occupancy for causal DA3 root birth.

The helpers here are deliberately representation-light: they use only seed
geometry, lifecycle/materialization state, and camera intrinsics/extrinsics.
They do not inspect learned DC, opacity, base-GS occlusion, detector evidence,
or ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ProjectedOccupancyResult:
    """Union of projected seed covariance ellipses on one image grid."""

    mask: torch.Tensor
    rows_used: int
    pixels: int
    rows_projected: torch.Tensor


@dataclass(frozen=True)
class ProjectedEllipseParams:
    center_xy: torch.Tensor
    inv_cov2: torch.Tensor
    extent: torch.Tensor
    valid: torch.Tensor


def _as_cpu_float(tensor: Any) -> torch.Tensor:
    return torch.as_tensor(tensor).detach().cpu().to(dtype=torch.float32)


def _rotation_matrix_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    q = _as_cpu_float(quaternion)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError("rotation quaternion must have shape [N,4] in wxyz order")
    norm = q.norm(dim=1, keepdim=True)
    if bool((norm <= 0.0).any()) or not bool(torch.isfinite(q).all()):
        raise ValueError("rotation quaternion must be finite and non-zero")
    q = q / norm.clamp_min(torch.finfo(q.dtype).eps)
    w, x, y, z = q.unbind(dim=1)
    R = torch.empty((q.shape[0], 3, 3), dtype=q.dtype)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def scale_depth_intrinsics_to_image(
    depth_K: torch.Tensor,
    *,
    native_height: int,
    native_width: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Match ``panel10_da3_seed_proposals`` full-resolution K scaling."""

    K = _as_cpu_float(depth_K).clone()
    if K.shape != (3, 3):
        raise ValueError("depth_K must have shape [3,3]")
    if min(native_height, native_width, height, width) <= 0:
        raise ValueError("image sizes must be positive")
    K[0] *= int(width) / int(native_width)
    K[1] *= int(height) / int(native_height)
    return K


def projected_covariance_ellipse_occupancy(
    xyz: torch.Tensor,
    scaling: torch.Tensor,
    rotation_wxyz: torch.Tensor,
    *,
    K: torch.Tensor,
    w2c: torch.Tensor,
    height: int,
    width: int,
    sigma: float = 2.0,
    row_indices: torch.Tensor | None = None,
) -> ProjectedOccupancyResult:
    """Rasterize projected Gaussian covariance ellipses into a boolean mask.

    Pixels are marked when their integer image-center coordinate satisfies the
    covariance Mahalanobis test ``dᵀΣ₂ᴅ⁻¹d <= sigma²``. Rows behind the camera,
    with non-finite geometry, or whose clipped support does not intersect the
    image contribute no pixels.
    """

    if sigma <= 0.0 or not torch.isfinite(torch.tensor(float(sigma))):
        raise ValueError("sigma must be finite and positive")
    H, W = int(height), int(width)
    if H <= 0 or W <= 0:
        raise ValueError("height and width must be positive")
    xyz_cpu = _as_cpu_float(xyz)
    scaling_cpu = _as_cpu_float(scaling)
    if xyz_cpu.ndim != 2 or xyz_cpu.shape[1] != 3:
        raise ValueError("xyz must have shape [N,3]")
    if scaling_cpu.shape != xyz_cpu.shape:
        raise ValueError("scaling must match xyz with shape [N,3]")
    if bool((scaling_cpu <= 0.0).any()) or not bool(torch.isfinite(scaling_cpu).all()):
        raise ValueError("scaling must be finite and positive")
    if not bool(torch.isfinite(xyz_cpu).all()):
        raise ValueError("xyz must be finite")
    params = projected_covariance_ellipse_params(
        xyz_cpu, scaling_cpu, rotation_wxyz, K=K, w2c=w2c, min_depth=1e-6
    )

    mask = torch.zeros((H, W), dtype=torch.bool)
    if xyz_cpu.shape[0] == 0:
        rows = torch.empty(0, dtype=torch.long) if row_indices is None else torch.as_tensor(row_indices, dtype=torch.long)[:0]
        return ProjectedOccupancyResult(mask=mask, rows_used=0, pixels=0, rows_projected=rows)
    rows_in = torch.arange(xyz_cpu.shape[0], dtype=torch.long) if row_indices is None else torch.as_tensor(row_indices, dtype=torch.long).flatten().cpu()
    if rows_in.shape != (xyz_cpu.shape[0],):
        raise ValueError("row_indices must align with xyz")

    projected_rows: list[int] = []
    sigma2 = float(sigma) * float(sigma)

    for i in range(xyz_cpu.shape[0]):
        if not bool(params.valid[i]):
            continue
        u, v = float(params.center_xy[i, 0]), float(params.center_xy[i, 1])
        extent = float(sigma) * float(params.extent[i])
        xmin = max(0, int(torch.floor(torch.tensor(u - extent)).item()))
        xmax = min(W - 1, int(torch.ceil(torch.tensor(u + extent)).item()))
        ymin = max(0, int(torch.floor(torch.tensor(v - extent)).item()))
        ymax = min(H - 1, int(torch.ceil(torch.tensor(v + extent)).item()))
        if xmin > xmax or ymin > ymax:
            continue
        has_support, _new_pixels = _rasterize_param_into_mask(
            mask,
            center_xy=params.center_xy[i],
            inv_cov2=params.inv_cov2[i],
            sigma2=sigma2,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
        )
        if has_support:
            projected_rows.append(int(rows_in[i]))

    return ProjectedOccupancyResult(
        mask=mask,
        rows_used=len(projected_rows),
        pixels=int(mask.sum().item()),
        rows_projected=torch.as_tensor(projected_rows, dtype=torch.long),
    )


def projected_covariance_ellipse_params(
    xyz: torch.Tensor,
    scaling: torch.Tensor,
    rotation_wxyz: torch.Tensor,
    *,
    K: torch.Tensor,
    w2c: torch.Tensor,
    min_depth: float = 1e-6,
) -> ProjectedEllipseParams:
    """Vectorized projection of 3-D Gaussian covariance to 2-D ellipses."""

    # Use float64 for projection/covariance/inversion so tiny but valid SPD
    # ellipses keep their real support.  A fixed or scale-relative determinant
    # floor changes the Mahalanobis region for thin covariances and can inflate
    # support toward the local bounding box; degenerate rows are skipped instead.
    xyz_cpu = torch.as_tensor(xyz).detach().cpu().to(dtype=torch.float64)
    scaling_cpu = torch.as_tensor(scaling).detach().cpu().to(dtype=torch.float64)
    if xyz_cpu.ndim != 2 or xyz_cpu.shape[1] != 3:
        raise ValueError("xyz must have shape [N,3]")
    if scaling_cpu.shape != xyz_cpu.shape:
        raise ValueError("scaling must match xyz with shape [N,3]")
    Rq = _rotation_matrix_wxyz(rotation_wxyz)
    if Rq.shape[0] != xyz_cpu.shape[0]:
        raise ValueError("rotation row count must match xyz")
    Rq = Rq.to(dtype=torch.float64)
    K_cpu = torch.as_tensor(K).detach().cpu().to(dtype=torch.float64)
    w2c_cpu = torch.as_tensor(w2c).detach().cpu().to(dtype=torch.float64)
    if K_cpu.shape != (3, 3) or w2c_cpu.shape != (4, 4):
        raise ValueError("K/w2c must have shape [3,3]/[4,4]")
    Rcam = w2c_cpu[:3, :3]
    tcam = w2c_cpu[:3, 3]
    cam = xyz_cpu @ Rcam.T + tcam
    z = cam[:, 2]
    fx, fy = K_cpu[0, 0], K_cpu[1, 1]
    cx, cy = K_cpu[0, 2], K_cpu[1, 2]
    center = torch.empty((xyz_cpu.shape[0], 2), dtype=torch.float64)
    z_safe = z.clamp_min(float(min_depth))
    center[:, 0] = fx * cam[:, 0] / z_safe + cx
    center[:, 1] = fy * cam[:, 1] / z_safe + cy

    cov_world = Rq @ torch.diag_embed(scaling_cpu.square()) @ Rq.transpose(1, 2)
    Rcam_b = Rcam.expand(xyz_cpu.shape[0], 3, 3)
    cov_cam = Rcam_b @ cov_world @ Rcam_b.transpose(1, 2)
    J = torch.zeros((xyz_cpu.shape[0], 2, 3), dtype=torch.float64)
    J[:, 0, 0] = fx / z_safe
    J[:, 0, 2] = -fx * cam[:, 0] / z_safe.square()
    J[:, 1, 1] = fy / z_safe
    J[:, 1, 2] = -fy * cam[:, 1] / z_safe.square()
    cov2 = J @ cov_cam @ J.transpose(1, 2)
    cov2 = (cov2 + cov2.transpose(1, 2)) * 0.5
    a = cov2[:, 0, 0]
    b = cov2[:, 0, 1]
    c = cov2[:, 1, 1]
    trace = a + c
    disc = (a - c).square() + 4.0 * b.square()
    largest = 0.5 * (trace + torch.sqrt(disc.clamp_min(0.0)))
    det = a * c - b.square()
    det_positive = det > 0.0
    det_safe = torch.where(det_positive, det, torch.ones_like(det))
    inv = torch.empty_like(cov2)
    inv[:, 0, 0] = c / det_safe
    inv[:, 0, 1] = -b / det_safe
    inv[:, 1, 0] = -b / det_safe
    inv[:, 1, 1] = a / det_safe
    finite = torch.isfinite(center).all(dim=1) & torch.isfinite(inv).flatten(1).all(dim=1)
    valid = finite & (z > float(min_depth)) & det_positive & (largest > 0.0)
    return ProjectedEllipseParams(
        center_xy=center.to(dtype=torch.float32),
        inv_cov2=inv.to(dtype=torch.float32),
        extent=torch.sqrt(largest.clamp_min(0.0)).to(dtype=torch.float32),
        valid=valid,
    )


def _rasterize_param_into_mask(
    mask: torch.Tensor,
    *,
    center_xy: torch.Tensor,
    inv_cov2: torch.Tensor,
    sigma2: float,
    xmin: int,
    xmax: int,
    ymin: int,
    ymax: int,
) -> tuple[bool, int]:
    yy, xx = torch.meshgrid(
        torch.arange(ymin, ymax + 1, dtype=torch.float32),
        torch.arange(xmin, xmax + 1, dtype=torch.float32),
        indexing="ij",
    )
    dx = xx - float(center_xy[0])
    dy = yy - float(center_xy[1])
    mahal = (
        inv_cov2[0, 0] * dx.square()
        + (inv_cov2[0, 1] + inv_cov2[1, 0]) * dx * dy
        + inv_cov2[1, 1] * dy.square()
    )
    local = mahal <= float(sigma2)
    if bool(local.any()):
        target = mask[ymin : ymax + 1, xmin : xmax + 1]
        added = local & ~target
        target |= local
        return True, int(added.sum().item())
    return False, 0


def union_projected_gaussian_ellipse_(
    mask: torch.Tensor,
    xyz: torch.Tensor,
    scaling: torch.Tensor,
    rotation_wxyz: torch.Tensor,
    *,
    K: torch.Tensor,
    w2c: torch.Tensor,
    sigma: float = 2.0,
) -> int:
    """Union one projected ellipse into ``mask`` in-place; return newly marked pixels."""

    if mask.ndim != 2 or mask.dtype != torch.bool:
        raise ValueError("mask must be a boolean [H,W] tensor")
    H, W = map(int, mask.shape)
    params = projected_covariance_ellipse_params(xyz, scaling, rotation_wxyz, K=K, w2c=w2c)
    if params.valid.numel() != 1 or not bool(params.valid[0]):
        return 0
    extent = float(sigma) * float(params.extent[0])
    u, v = float(params.center_xy[0, 0]), float(params.center_xy[0, 1])
    xmin = max(0, int(torch.floor(torch.tensor(u - extent)).item()))
    xmax = min(W - 1, int(torch.ceil(torch.tensor(u + extent)).item()))
    ymin = max(0, int(torch.floor(torch.tensor(v - extent)).item()))
    ymax = min(H - 1, int(torch.ceil(torch.tensor(v + extent)).item()))
    if xmin > xmax or ymin > ymax:
        return 0
    _has_support, added = _rasterize_param_into_mask(
        mask,
        center_xy=params.center_xy[0],
        inv_cov2=params.inv_cov2[0],
        sigma2=float(sigma) * float(sigma),
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
    )
    return added


def seed_rows_for_root_birth_2d_occupancy(
    lifecycle: Any,
    *,
    timestamp: int,
    retired: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return existing OPEN and NEVER_OPEN rows eligible for 2-D birth occupancy."""

    active = torch.as_tensor(lifecycle.active_mask(int(timestamp))).detach().cpu().bool()
    never = torch.as_tensor(lifecycle.never_open_mask(int(timestamp))).detach().cpu().bool()
    if retired is None:
        not_retired = torch.ones_like(active)
    else:
        not_retired = ~torch.as_tensor(retired).detach().cpu().bool().flatten()
        if not_retired.shape != active.shape:
            raise ValueError("retired mask must align with lifecycle rows")
    active_rows = torch.nonzero(active & not_retired, as_tuple=False).flatten()
    never_rows = torch.nonzero(never & not_retired, as_tuple=False).flatten()
    return active_rows, never_rows
