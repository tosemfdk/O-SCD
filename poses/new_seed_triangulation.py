"""Known-pose geometry helpers for online NEW Gaussian seeding.

This module intentionally stays independent from the existing CUDA triangulator.
It uses explicit OpenCV-style conventions throughout:

* keypoints are pixels in ``[x, y]`` order;
* ``w2c`` is a 4x4 world-to-camera transform, ``X_c = R X_w + t``;
* ``K`` is a 3x3 pinhole intrinsic matrix and may have ``fx != fy``.

The helpers are small CPU/GPU agnostic PyTorch functions so they can be unit-tested
without XFeat, CUDA graphs, or reference-depth state.  They are meant to validate
inference-to-inference tracks before those tracks become fixed-geometry seed rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


_EPS = 1.0e-9


@dataclass(frozen=True)
class GeometryGateConfig:
    """Hard gates used to decide whether a known-pose match is seed-worthy."""

    min_translation: float = 0.10
    min_ray_angle_deg: float = 1.5
    max_epipolar_error_px: float = 2.0
    max_reprojection_rmse_px: float = 3.0
    min_depth: float = 1.0e-6


@dataclass(frozen=True)
class PairTriangulationDiagnostics:
    """Per-correspondence diagnostics for a two-view triangulation pass."""

    points_world: torch.Tensor  # [N,3]
    depths1: torch.Tensor  # [N]
    depths2: torch.Tensor  # [N]
    ray_angle_deg: torch.Tensor  # [N]
    reprojection_error1_px: torch.Tensor  # [N]
    reprojection_error2_px: torch.Tensor  # [N]
    reprojection_rmse_px: torch.Tensor  # [N]
    sampson_error_px: torch.Tensor  # [N]
    translation_norm: torch.Tensor  # scalar
    valid: torch.Tensor  # [N]
    rejection_reasons: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class MaskedMatchTriangulation:
    """Matched masked keypoints plus geometry diagnostics.

    ``idx1``/``idx2`` always refer to the original, unfiltered keypoint arrays.
    """

    idx1: torch.Tensor
    idx2: torch.Tensor
    kpts1: torch.Tensor
    kpts2: torch.Tensor
    diagnostics: PairTriangulationDiagnostics

    @property
    def valid(self) -> torch.Tensor:
        return self.diagnostics.valid

    @property
    def points_world(self) -> torch.Tensor:
        return self.diagnostics.points_world


# ---------------------------------------------------------------------------
# Validation and camera primitives


def _as_float_tensor(x: torch.Tensor, *, name: str) -> torch.Tensor:
    if not torch.is_tensor(x):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not x.is_floating_point():
        x = x.float()
    return x


def validate_w2c(w2c: torch.Tensor, *, name: str = "w2c", atol: float = 1.0e-4) -> torch.Tensor:
    """Validate and return a floating 4x4 world-to-camera transform."""

    w2c = _as_float_tensor(w2c, name=name)
    if w2c.shape != (4, 4):
        raise ValueError(f"{name} must have shape [4,4], got {tuple(w2c.shape)}")
    if not torch.isfinite(w2c).all():
        raise ValueError(f"{name} contains non-finite values")
    bottom = w2c[3]
    expected_bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=w2c.dtype, device=w2c.device)
    if not torch.allclose(bottom, expected_bottom, atol=atol, rtol=0.0):
        raise ValueError(f"{name} must be a homogeneous W2C matrix with bottom row [0,0,0,1]")

    R = w2c[:3, :3]
    eye = torch.eye(3, dtype=w2c.dtype, device=w2c.device)
    if not torch.allclose(R @ R.T, eye, atol=atol, rtol=0.0):
        raise ValueError(f"{name} rotation is not orthonormal; did you pass C2W or a transposed pose?")
    det = torch.linalg.det(R)
    if not torch.isclose(det, torch.ones((), dtype=w2c.dtype, device=w2c.device), atol=atol, rtol=0.0):
        raise ValueError(f"{name} rotation determinant must be +1, got {det.item():.6g}")
    return w2c


def validate_K(K: torch.Tensor, *, name: str = "K") -> torch.Tensor:
    """Validate and return a floating 3x3 intrinsic matrix."""

    K = _as_float_tensor(K, name=name)
    if K.shape != (3, 3):
        raise ValueError(f"{name} must have shape [3,3], got {tuple(K.shape)}")
    if not torch.isfinite(K).all():
        raise ValueError(f"{name} contains non-finite values")
    if K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError(f"{name} focal lengths must be positive")
    if torch.abs(K[2, 2]) <= _EPS:
        raise ValueError(f"{name}[2,2] must be non-zero")
    expected_last = torch.tensor([0.0, 0.0, 1.0], dtype=K.dtype, device=K.device)
    if not torch.allclose(K[2], expected_last, atol=1.0e-5, rtol=0.0):
        raise ValueError(f"{name} must have last row [0,0,1]")
    return K


def camera_center_from_w2c(w2c: torch.Tensor) -> torch.Tensor:
    """Return the camera center in world coordinates for an explicit W2C pose."""

    w2c = validate_w2c(w2c)
    return -(w2c[:3, :3].T @ w2c[:3, 3])


def camera_centers_from_w2c(w2cs: torch.Tensor) -> torch.Tensor:
    if w2cs.ndim != 3 or w2cs.shape[-2:] != (4, 4):
        raise ValueError(f"w2cs must have shape [V,4,4], got {tuple(w2cs.shape)}")
    return torch.stack([camera_center_from_w2c(w2c) for w2c in w2cs], dim=0)


def _homogeneous_pixels(kpts_xy: torch.Tensor) -> torch.Tensor:
    kpts_xy = _as_float_tensor(kpts_xy, name="kpts_xy")
    if kpts_xy.ndim == 1:
        if kpts_xy.numel() != 2:
            raise ValueError("single keypoint must have shape [2]")
        kpts_xy = kpts_xy[None]
    if kpts_xy.ndim != 2 or kpts_xy.shape[-1] != 2:
        raise ValueError(f"keypoints must have shape [N,2] in [x,y] order, got {tuple(kpts_xy.shape)}")
    ones = torch.ones(kpts_xy.shape[0], 1, dtype=kpts_xy.dtype, device=kpts_xy.device)
    return torch.cat([kpts_xy, ones], dim=-1)


def camera_rays(kpts_xy: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Return normalized camera-frame rays for ``[x,y]`` pixels."""

    K = validate_K(K).to(device=kpts_xy.device, dtype=kpts_xy.dtype)
    rays = torch.linalg.solve(K, _homogeneous_pixels(kpts_xy).T).T
    return rays / torch.linalg.norm(rays, dim=-1, keepdim=True).clamp_min(_EPS)


def world_rays(kpts_xy: torch.Tensor, K: torch.Tensor, w2c: torch.Tensor) -> torch.Tensor:
    """Return normalized world-frame rays for ``[x,y]`` pixels and W2C pose."""

    w2c = validate_w2c(w2c).to(device=kpts_xy.device, dtype=kpts_xy.dtype)
    rays_cam = camera_rays(kpts_xy, K)
    rays_world = rays_cam @ w2c[:3, :3]
    return rays_world / torch.linalg.norm(rays_world, dim=-1, keepdim=True).clamp_min(_EPS)


# ---------------------------------------------------------------------------
# Projection, epipolar geometry, and errors


def skew(v: torch.Tensor) -> torch.Tensor:
    v = _as_float_tensor(v, name="v")
    if v.shape != (3,):
        raise ValueError(f"v must have shape [3], got {tuple(v.shape)}")
    z = torch.zeros((), dtype=v.dtype, device=v.device)
    return torch.stack(
        [
            torch.stack([z, -v[2], v[1]]),
            torch.stack([v[2], z, -v[0]]),
            torch.stack([-v[1], v[0], z]),
        ]
    )


def relative_cam2_from_cam1(w2c1: torch.Tensor, w2c2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``R,t`` such that ``X_cam2 = R @ X_cam1 + t``."""

    w2c1 = validate_w2c(w2c1)
    w2c2 = validate_w2c(w2c2).to(dtype=w2c1.dtype, device=w2c1.device)
    R1, t1 = w2c1[:3, :3], w2c1[:3, 3]
    R2, t2 = w2c2[:3, :3], w2c2[:3, 3]
    R21 = R2 @ R1.T
    t21 = t2 - R21 @ t1
    return R21, t21


def essential_from_w2c(w2c1: torch.Tensor, w2c2: torch.Tensor) -> torch.Tensor:
    """Known-pose essential matrix for ``x2.T @ E @ x1 == 0``."""

    R21, t21 = relative_cam2_from_cam1(w2c1, w2c2)
    return skew(t21) @ R21


def fundamental_from_w2c(K1: torch.Tensor, w2c1: torch.Tensor, K2: torch.Tensor, w2c2: torch.Tensor) -> torch.Tensor:
    """Known-pose fundamental matrix for pixel points in image1 -> image2."""

    K1 = validate_K(K1)
    K2 = validate_K(K2).to(dtype=K1.dtype, device=K1.device)
    E = essential_from_w2c(w2c1.to(dtype=K1.dtype, device=K1.device), w2c2.to(dtype=K1.dtype, device=K1.device))
    F = torch.linalg.inv(K2).T @ E @ torch.linalg.inv(K1)
    denom = torch.linalg.norm(F).clamp_min(_EPS)
    return F / denom


def sampson_error_px(kpts1_xy: torch.Tensor, kpts2_xy: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    """Return square-root Sampson error in pixel units for corresponding pixels."""

    if kpts1_xy.shape != kpts2_xy.shape:
        raise ValueError("kpts1_xy and kpts2_xy must have the same shape")
    dtype = torch.promote_types(kpts1_xy.dtype, F.dtype)
    x1 = _homogeneous_pixels(kpts1_xy.to(dtype=dtype))
    x2 = _homogeneous_pixels(kpts2_xy.to(dtype=dtype)).to(device=x1.device)
    F = F.to(dtype=dtype, device=x1.device)
    Fx1 = x1 @ F.T
    Ftx2 = x2 @ F
    numerator = (x2 * Fx1).sum(dim=-1).square()
    denom = Fx1[:, 0].square() + Fx1[:, 1].square() + Ftx2[:, 0].square() + Ftx2[:, 1].square()
    return torch.sqrt(numerator / denom.clamp_min(_EPS))


def projection_matrix(K: torch.Tensor, w2c: torch.Tensor) -> torch.Tensor:
    K = validate_K(K)
    w2c = validate_w2c(w2c).to(dtype=K.dtype, device=K.device)
    return K @ w2c[:3, :]


def project_points(points_world: torch.Tensor, K: torch.Tensor, w2c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Project world points; returns ``(pixels_xy, depth_z)``."""

    points_world = _as_float_tensor(points_world, name="points_world")
    if points_world.ndim == 1:
        if points_world.numel() != 3:
            raise ValueError("single point must have shape [3]")
        points_world = points_world[None]
    if points_world.ndim != 2 or points_world.shape[-1] != 3:
        raise ValueError(f"points_world must have shape [N,3], got {tuple(points_world.shape)}")
    K = validate_K(K).to(dtype=points_world.dtype, device=points_world.device)
    w2c = validate_w2c(w2c).to(dtype=points_world.dtype, device=points_world.device)
    Xc = points_world @ w2c[:3, :3].T + w2c[:3, 3]
    z = Xc[:, 2:3]
    safe_z = torch.where(z.abs() > _EPS, z, torch.full_like(z, _EPS))
    uv = Xc[:, :2] / safe_z
    uv = uv @ K[:2, :2].T + K[:2, 2]
    return uv, Xc[:, 2]


def reprojection_errors_px(points_world: torch.Tensor, kpts_xy: torch.Tensor, Ks: torch.Tensor, w2cs: torch.Tensor) -> torch.Tensor:
    """Return per-view reprojection errors for one point or one point per view."""

    if Ks.ndim != 3 or w2cs.ndim != 3 or Ks.shape[0] != w2cs.shape[0]:
        raise ValueError("Ks and w2cs must have matching leading view dimension")
    kpts_xy = _as_float_tensor(kpts_xy, name="kpts_xy")
    if kpts_xy.ndim != 2 or kpts_xy.shape != (Ks.shape[0], 2):
        raise ValueError(f"kpts_xy must have shape [V,2], got {tuple(kpts_xy.shape)}")
    if points_world.ndim == 1:
        pts = points_world[None].repeat(Ks.shape[0], 1)
    elif points_world.ndim == 2 and points_world.shape[0] == Ks.shape[0]:
        pts = points_world
    else:
        raise ValueError("points_world must be [3] or [V,3]")
    errors = []
    for point, kpt, K, w2c in zip(pts, kpts_xy, Ks, w2cs):
        uv, _ = project_points(point, K, w2c)
        errors.append(torch.linalg.norm(uv[0] - kpt))
    return torch.stack(errors, dim=0)


# ---------------------------------------------------------------------------
# DLT triangulation and gates


def _triangulation_A(kpts_xy: torch.Tensor, Ks: torch.Tensor, w2cs: torch.Tensor) -> torch.Tensor:
    rows = []
    for kpt, K, w2c in zip(kpts_xy, Ks, w2cs):
        P = projection_matrix(K, w2c).to(dtype=kpts_xy.dtype, device=kpts_xy.device)
        x, y = kpt[0], kpt[1]
        rows.append(x * P[2] - P[0])
        rows.append(y * P[2] - P[1])
    return torch.stack(rows, dim=0)


def triangulate_multiview_dlt(kpts_xy: torch.Tensor, Ks: torch.Tensor, w2cs: torch.Tensor) -> torch.Tensor:
    """Triangulate one world point from ``V>=2`` known-pose observations."""

    kpts_xy = _as_float_tensor(kpts_xy, name="kpts_xy")
    Ks = _as_float_tensor(Ks, name="Ks")
    w2cs = _as_float_tensor(w2cs, name="w2cs")
    if kpts_xy.ndim != 2 or kpts_xy.shape[-1] != 2:
        raise ValueError(f"kpts_xy must have shape [V,2], got {tuple(kpts_xy.shape)}")
    if kpts_xy.shape[0] < 2:
        raise ValueError("at least two views are required for triangulation")
    if Ks.shape != (kpts_xy.shape[0], 3, 3):
        raise ValueError(f"Ks must have shape [V,3,3], got {tuple(Ks.shape)}")
    if w2cs.shape != (kpts_xy.shape[0], 4, 4):
        raise ValueError(f"w2cs must have shape [V,4,4], got {tuple(w2cs.shape)}")
    for i in range(kpts_xy.shape[0]):
        validate_K(Ks[i])
        validate_w2c(w2cs[i])

    A = _triangulation_A(kpts_xy, Ks.to(kpts_xy), w2cs.to(kpts_xy))
    _, _, vh = torch.linalg.svd(A)
    Xh = vh[-1]
    if torch.abs(Xh[-1]) <= _EPS:
        return torch.full((3,), float("nan"), dtype=kpts_xy.dtype, device=kpts_xy.device)
    return Xh[:3] / Xh[3]


def triangulate_pair_dlt(kpts1_xy: torch.Tensor, kpts2_xy: torch.Tensor, K1: torch.Tensor, w2c1: torch.Tensor, K2: torch.Tensor, w2c2: torch.Tensor) -> torch.Tensor:
    """Vectorized two-view DLT triangulation for corresponding pixels."""

    kpts1_xy = _as_float_tensor(kpts1_xy, name="kpts1_xy")
    kpts2_xy = _as_float_tensor(kpts2_xy, name="kpts2_xy").to(device=kpts1_xy.device, dtype=kpts1_xy.dtype)
    if kpts1_xy.shape != kpts2_xy.shape or kpts1_xy.ndim != 2 or kpts1_xy.shape[-1] != 2:
        raise ValueError("kpts1_xy and kpts2_xy must both have shape [N,2]")
    K1 = validate_K(K1).to(device=kpts1_xy.device, dtype=kpts1_xy.dtype)
    K2 = validate_K(K2).to(device=kpts1_xy.device, dtype=kpts1_xy.dtype)
    w2c1 = validate_w2c(w2c1).to(device=kpts1_xy.device, dtype=kpts1_xy.dtype)
    w2c2 = validate_w2c(w2c2).to(device=kpts1_xy.device, dtype=kpts1_xy.dtype)
    if kpts1_xy.shape[0] == 0:
        return torch.empty(0, 3, dtype=kpts1_xy.dtype, device=kpts1_xy.device)
    Ks = torch.stack([K1, K2], dim=0)
    w2cs = torch.stack([w2c1, w2c2], dim=0)
    return torch.stack([triangulate_multiview_dlt(torch.stack([p1, p2], dim=0), Ks, w2cs) for p1, p2 in zip(kpts1_xy, kpts2_xy)], dim=0)


def depths_in_cameras(points_world: torch.Tensor, w2cs: torch.Tensor) -> torch.Tensor:
    points_world = _as_float_tensor(points_world, name="points_world")
    w2cs = _as_float_tensor(w2cs, name="w2cs")
    if points_world.ndim == 1:
        points_world = points_world[None]
    if w2cs.ndim == 2:
        w2cs = w2cs[None]
    depths = []
    for w2c in w2cs:
        w2c = validate_w2c(w2c).to(dtype=points_world.dtype, device=points_world.device)
        depths.append((points_world @ w2c[:3, :3].T + w2c[:3, 3])[:, 2])
    return torch.stack(depths, dim=-1)


def ray_angles_deg(kpts1_xy: torch.Tensor, K1: torch.Tensor, w2c1: torch.Tensor, kpts2_xy: torch.Tensor, K2: torch.Tensor, w2c2: torch.Tensor) -> torch.Tensor:
    rays1 = world_rays(kpts1_xy, K1, w2c1)
    rays2 = world_rays(kpts2_xy, K2, w2c2).to(device=rays1.device, dtype=rays1.dtype)
    cos = (rays1 * rays2).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos))


def _reason_tuple(validity: dict[str, bool]) -> tuple[str, ...]:
    return tuple(name for name, ok in validity.items() if not ok) or ("ok",)


def triangulate_pair_with_diagnostics(
    kpts1_xy: torch.Tensor,
    kpts2_xy: torch.Tensor,
    K1: torch.Tensor,
    w2c1: torch.Tensor,
    K2: torch.Tensor,
    w2c2: torch.Tensor,
    config: GeometryGateConfig = GeometryGateConfig(),
    *,
    mask1: torch.Tensor | None = None,
    mask2: torch.Tensor | None = None,
    image_size1: tuple[int, int] | None = None,
    image_size2: tuple[int, int] | None = None,
) -> PairTriangulationDiagnostics:
    """Triangulate two-view correspondences and apply named hard gates."""

    points = triangulate_pair_dlt(kpts1_xy, kpts2_xy, K1, w2c1, K2, w2c2)
    dtype, device = points.dtype, points.device
    w2c1 = validate_w2c(w2c1).to(dtype=dtype, device=device)
    w2c2 = validate_w2c(w2c2).to(dtype=dtype, device=device)
    K1 = validate_K(K1).to(dtype=dtype, device=device)
    K2 = validate_K(K2).to(dtype=dtype, device=device)
    kpts1_xy = kpts1_xy.to(dtype=dtype, device=device)
    kpts2_xy = kpts2_xy.to(dtype=dtype, device=device)

    depths = depths_in_cameras(points, torch.stack([w2c1, w2c2], dim=0))
    uv1, _ = project_points(points, K1, w2c1)
    uv2, _ = project_points(points, K2, w2c2)
    err1 = torch.linalg.norm(uv1 - kpts1_xy, dim=-1)
    err2 = torch.linalg.norm(uv2 - kpts2_xy, dim=-1)
    rmse = torch.sqrt((err1.square() + err2.square()) / 2.0)
    F = fundamental_from_w2c(K1, w2c1, K2, w2c2)
    sampson = sampson_error_px(kpts1_xy, kpts2_xy, F)
    angles = ray_angles_deg(kpts1_xy, K1, w2c1, kpts2_xy, K2, w2c2)
    translation = torch.linalg.norm(camera_center_from_w2c(w2c2) - camera_center_from_w2c(w2c1))

    finite = torch.isfinite(points).all(dim=-1) & torch.isfinite(rmse) & torch.isfinite(sampson) & torch.isfinite(angles)
    positive_depth = (depths[:, 0] > config.min_depth) & (depths[:, 1] > config.min_depth)
    enough_translation = translation >= config.min_translation
    enough_angle = angles >= config.min_ray_angle_deg
    epipolar_ok = sampson <= config.max_epipolar_error_px
    reprojection_ok = rmse <= config.max_reprojection_rmse_px

    mask1_ok = torch.ones_like(finite)
    mask2_ok = torch.ones_like(finite)
    if mask1 is not None:
        mask1_ok = points_inside_mask(uv1, mask1, image_size=image_size1).to(device=device)
    if mask2 is not None:
        mask2_ok = points_inside_mask(uv2, mask2, image_size=image_size2).to(device=device)

    valid = finite & positive_depth & enough_translation & enough_angle & epipolar_ok & reprojection_ok & mask1_ok & mask2_ok

    reasons = []
    for i in range(points.shape[0]):
        reasons.append(
            _reason_tuple(
                {
                    "finite": bool(finite[i].item()),
                    "positive_depth": bool(positive_depth[i].item()),
                    "min_translation": bool(enough_translation.item()),
                    "min_ray_angle": bool(enough_angle[i].item()),
                    "epipolar": bool(epipolar_ok[i].item()),
                    "reprojection": bool(reprojection_ok[i].item()),
                    "mask1": bool(mask1_ok[i].item()),
                    "mask2": bool(mask2_ok[i].item()),
                }
            )
        )

    return PairTriangulationDiagnostics(
        points_world=points,
        depths1=depths[:, 0],
        depths2=depths[:, 1],
        ray_angle_deg=angles,
        reprojection_error1_px=err1,
        reprojection_error2_px=err2,
        reprojection_rmse_px=rmse,
        sampson_error_px=sampson,
        translation_norm=translation,
        valid=valid,
        rejection_reasons=tuple(reasons),
    )


def triangulate_track_with_diagnostics(
    kpts_xy: torch.Tensor,
    Ks: torch.Tensor,
    w2cs: torch.Tensor,
    config: GeometryGateConfig = GeometryGateConfig(),
    *,
    masks: Sequence[torch.Tensor | None] | None = None,
    image_sizes: Sequence[tuple[int, int] | None] | None = None,
) -> dict[str, object]:
    """Multi-view DLT plus cheirality/reprojection/ray-angle/mask diagnostics."""

    point = triangulate_multiview_dlt(kpts_xy, Ks, w2cs)
    depths = depths_in_cameras(point, w2cs)[0]
    errors = reprojection_errors_px(point, kpts_xy, Ks, w2cs)
    rmse = torch.sqrt(errors.square().mean())

    centers = camera_centers_from_w2c(w2cs).to(dtype=point.dtype, device=point.device)
    rays_to_point = point[None] - centers
    rays_to_point = rays_to_point / torch.linalg.norm(rays_to_point, dim=-1, keepdim=True).clamp_min(_EPS)
    min_angle = torch.tensor(float("inf"), dtype=point.dtype, device=point.device)
    for i in range(w2cs.shape[0]):
        for j in range(i + 1, w2cs.shape[0]):
            angle = torch.rad2deg(torch.acos((rays_to_point[i] * rays_to_point[j]).sum().clamp(-1.0, 1.0)))
            min_angle = torch.minimum(min_angle, angle)

    finite = torch.isfinite(point).all() & torch.isfinite(rmse) & torch.isfinite(min_angle)
    positive_depth = (depths > config.min_depth).all()
    enough_angle = min_angle >= config.min_ray_angle_deg
    reprojection_ok = rmse <= config.max_reprojection_rmse_px
    mask_ok = torch.tensor(True, device=point.device)
    if masks is not None:
        if image_sizes is None:
            image_sizes = [None] * len(masks)
        oks = []
        for K, w2c, mask, image_size in zip(Ks, w2cs, masks, image_sizes):
            if mask is None:
                oks.append(torch.tensor(True, device=point.device))
            else:
                uv, _ = project_points(point, K, w2c)
                oks.append(points_inside_mask(uv, mask, image_size=image_size).to(device=point.device)[0])
        mask_ok = torch.stack(oks).all()
    valid = finite & positive_depth & enough_angle & reprojection_ok & mask_ok
    reasons = _reason_tuple(
        {
            "finite": bool(finite.item()),
            "positive_depth": bool(positive_depth.item()),
            "min_ray_angle": bool(enough_angle.item()),
            "reprojection": bool(reprojection_ok.item()),
            "mask": bool(mask_ok.item()),
        }
    )
    return {
        "point_world": point,
        "depths": depths,
        "reprojection_errors_px": errors,
        "reprojection_rmse_px": rmse,
        "min_ray_angle_deg": min_angle,
        "valid": valid,
        "rejection_reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Mask and descriptor-match helpers


def points_inside_mask(kpts_xy: torch.Tensor, mask: torch.Tensor, *, image_size: tuple[int, int] | None = None) -> torch.Tensor:
    """Nearest-neighbor mask lookup for ``[x,y]`` pixels.

    If ``image_size=(width,height)`` is given, pixels are scaled to the mask grid.
    Otherwise keypoints are assumed to already be in mask-pixel coordinates.
    """

    kpts_xy = _as_float_tensor(kpts_xy, name="kpts_xy")
    if mask.ndim != 2:
        raise ValueError(f"mask must have shape [H,W], got {tuple(mask.shape)}")
    mask_bool = mask.bool().to(device=kpts_xy.device)
    H, W = mask_bool.shape
    xy = kpts_xy.clone()
    if image_size is not None:
        width, height = image_size
        if width <= 0 or height <= 0:
            raise ValueError("image_size must be positive (width,height)")
        xy[:, 0] = xy[:, 0] * (W / float(width))
        xy[:, 1] = xy[:, 1] * (H / float(height))
    ix = torch.floor(xy[:, 0]).long()
    iy = torch.floor(xy[:, 1]).long()
    inside_bounds = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    out = torch.zeros(kpts_xy.shape[0], dtype=torch.bool, device=kpts_xy.device)
    if inside_bounds.any():
        out[inside_bounds] = mask_bool[iy[inside_bounds], ix[inside_bounds]]
    return out


def masked_valid_indices(kpts_xy: torch.Tensor, descriptors: torch.Tensor, mask: torch.Tensor, *, valid: torch.Tensor | None = None, image_size: tuple[int, int] | None = None) -> torch.Tensor:
    """Return original keypoint indices that are valid and inside ``mask``."""

    if descriptors.ndim != 2 or descriptors.shape[0] != kpts_xy.shape[0]:
        raise ValueError("descriptors must have shape [N,D] matching keypoints")
    desc_valid = torch.isfinite(descriptors).all(dim=-1) & (descriptors.abs().sum(dim=-1) > 0)
    if valid is not None:
        if valid.shape != desc_valid.shape:
            raise ValueError("valid must have shape [N]")
        desc_valid = desc_valid & valid.to(device=desc_valid.device, dtype=torch.bool)
    in_mask = points_inside_mask(kpts_xy, mask, image_size=image_size).to(device=desc_valid.device)
    return torch.nonzero(desc_valid & in_mask, as_tuple=False).flatten()


def _mutual_descriptor_match(feats1: torch.Tensor, feats2: torch.Tensor, min_cossim: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CPU-safe equivalent of :func:`poses.matcher.match`.

    The existing matcher module imports optional CUDA/CuPy RANSAC dependencies at
    import time.  Keeping this tiny mutual-NN helper local lets geometry tests run
    on CPU-only environments while preserving the same cosine/mutual semantics.
    """

    cossim = feats1 @ feats2.T
    bestcossim, match12 = cossim.max(dim=1)
    _, match21 = cossim.max(dim=0)
    idx0 = torch.arange(match12.shape[0], device=match12.device)
    keep = match21[match12] == idx0
    if min_cossim > 0:
        keep = keep & (bestcossim > min_cossim)
    return idx0, match12, keep


def match_masked_keypoints(
    kpts1_xy: torch.Tensor,
    desc1: torch.Tensor,
    mask1: torch.Tensor,
    kpts2_xy: torch.Tensor,
    desc2: torch.Tensor,
    mask2: torch.Tensor,
    *,
    valid1: torch.Tensor | None = None,
    valid2: torch.Tensor | None = None,
    image_size1: tuple[int, int] | None = None,
    image_size2: tuple[int, int] | None = None,
    min_cossim: float = 0.82,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Descriptor mutual-NN matching after endpoint mask/valid filtering."""

    idx1_subset = masked_valid_indices(kpts1_xy, desc1, mask1, valid=valid1, image_size=image_size1)
    idx2_subset = masked_valid_indices(kpts2_xy, desc2, mask2, valid=valid2, image_size=image_size2)
    if idx1_subset.numel() == 0 or idx2_subset.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=kpts1_xy.device)
        return empty, empty.to(device=kpts2_xy.device)
    d1 = desc1[idx1_subset].float()
    d2 = desc2[idx2_subset].float().to(device=d1.device)
    rel1, rel2, keep = _mutual_descriptor_match(d1, d2, min_cossim=min_cossim)
    rel1 = rel1[keep].to(device=idx1_subset.device)
    rel2 = rel2[keep].to(device=idx2_subset.device)
    return idx1_subset[rel1], idx2_subset[rel2]


def match_and_triangulate_masked_pair(
    kpts1_xy: torch.Tensor,
    desc1: torch.Tensor,
    mask1: torch.Tensor,
    K1: torch.Tensor,
    w2c1: torch.Tensor,
    kpts2_xy: torch.Tensor,
    desc2: torch.Tensor,
    mask2: torch.Tensor,
    K2: torch.Tensor,
    w2c2: torch.Tensor,
    *,
    valid1: torch.Tensor | None = None,
    valid2: torch.Tensor | None = None,
    image_size1: tuple[int, int] | None = None,
    image_size2: tuple[int, int] | None = None,
    min_cossim: float = 0.82,
    config: GeometryGateConfig = GeometryGateConfig(),
) -> MaskedMatchTriangulation:
    """Masked mutual descriptor matching followed by known-pose pair gates."""

    idx1, idx2 = match_masked_keypoints(
        kpts1_xy,
        desc1,
        mask1,
        kpts2_xy,
        desc2,
        mask2,
        valid1=valid1,
        valid2=valid2,
        image_size1=image_size1,
        image_size2=image_size2,
        min_cossim=min_cossim,
    )
    mkpts1 = kpts1_xy[idx1]
    mkpts2 = kpts2_xy[idx2]
    diagnostics = triangulate_pair_with_diagnostics(
        mkpts1,
        mkpts2,
        K1,
        w2c1,
        K2,
        w2c2,
        config,
        mask1=mask1,
        mask2=mask2,
        image_size1=image_size1,
        image_size2=image_size2,
    )
    return MaskedMatchTriangulation(idx1=idx1, idx2=idx2, kpts1=mkpts1, kpts2=mkpts2, diagnostics=diagnostics)


__all__ = [
    "GeometryGateConfig",
    "PairTriangulationDiagnostics",
    "MaskedMatchTriangulation",
    "validate_w2c",
    "validate_K",
    "camera_center_from_w2c",
    "camera_centers_from_w2c",
    "camera_rays",
    "world_rays",
    "skew",
    "relative_cam2_from_cam1",
    "essential_from_w2c",
    "fundamental_from_w2c",
    "sampson_error_px",
    "projection_matrix",
    "project_points",
    "reprojection_errors_px",
    "triangulate_pair_dlt",
    "triangulate_multiview_dlt",
    "depths_in_cameras",
    "ray_angles_deg",
    "triangulate_pair_with_diagnostics",
    "triangulate_track_with_diagnostics",
    "points_inside_mask",
    "masked_valid_indices",
    "match_masked_keypoints",
    "match_and_triangulate_masked_pair",
]
