"""Causal NEW-only mask densification around triangulated XFeat anchors.

The reference Gaussian bank is never edited here.  Candidate children are
created only after a sign has already been identified as NEW, and only in
under-covered cells of that signed cue.  Their depth is inherited from the
nearest visible NEW seed anchor; recent inference views then carve away points
that are not consistently inside the same signed cue.

This is deliberately an anchor-constrained visual-hull approximation rather
than unconstrained ray filling.  Pure silhouette intersection leaves depth
ambiguous and tends to create floaters along the viewing ray.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from temporal.new_seed_observation import SignedXFeatObservation


_MASK_SIZE = 64
_EPS = 1.0e-8


@dataclass(frozen=True)
class NewSeedDensificationConfig:
    """Conservative gates for NEW-bank-only child seed generation."""

    min_support_views: int = 3
    min_support_ratio: float = 0.65
    min_baseline: float = 0.10
    min_view_angle_deg: float = 1.5
    erosion_cells: int = 1
    coverage_radius_cells: float = 0.80
    min_child_separation_cells: float = 0.90
    max_parent_distance_cells: float = 10.0
    min_world_separation: float = 0.008
    footprint_px: float = 3.0
    max_new_per_frame: int = 64
    max_total_seeds: int = 2500
    history_views: int = 12
    interval_frames: int = 2
    opacity: float = 0.10

    def __post_init__(self) -> None:
        for name in (
            "min_support_views",
            "max_new_per_frame",
            "max_total_seeds",
            "history_views",
            "interval_frames",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_support_views < 2:
            raise ValueError("min_support_views must be at least 2")
        if isinstance(self.erosion_cells, bool) or not isinstance(self.erosion_cells, int) or self.erosion_cells < 0:
            raise ValueError("erosion_cells must be a non-negative integer")
        for name in (
            "min_baseline",
            "min_view_angle_deg",
            "coverage_radius_cells",
            "min_child_separation_cells",
            "max_parent_distance_cells",
            "min_world_separation",
            "footprint_px",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < float(self.min_support_ratio) <= 1.0:
            raise ValueError("min_support_ratio must be in (0,1]")
        if not 0.0 < float(self.opacity) < 1.0:
            raise ValueError("opacity must be in (0,1)")


@dataclass(frozen=True)
class DensifiedSeedBatch:
    """Accepted child rows plus causal support diagnostics."""

    xyz: Tensor  # CPU float32 [N,3]
    log_scaling: Tensor  # CPU float32 [N,3]
    parent_rows: Tensor  # CPU int64 [N], global rows in the seed bank
    support_frames: tuple[tuple[int, ...], ...]
    support_ratios: tuple[float, ...]
    visible_views: tuple[int, ...]
    max_baselines: tuple[float, ...]
    max_view_angles_deg: tuple[float, ...]
    considered_cells: int
    uncovered_cells: int
    rejection_counts: dict[str, int]

    @property
    def count(self) -> int:
        return int(self.xyz.shape[0])


def _empty_result(*, considered: int = 0, uncovered: int = 0, rejection_counts: dict[str, int] | None = None) -> DensifiedSeedBatch:
    return DensifiedSeedBatch(
        xyz=torch.empty((0, 3), dtype=torch.float32),
        log_scaling=torch.empty((0, 3), dtype=torch.float32),
        parent_rows=torch.empty((0,), dtype=torch.long),
        support_frames=(),
        support_ratios=(),
        visible_views=(),
        max_baselines=(),
        max_view_angles_deg=(),
        considered_cells=int(considered),
        uncovered_cells=int(uncovered),
        rejection_counts={} if rejection_counts is None else dict(rejection_counts),
    )


def _mask_for_sign(observation: SignedXFeatObservation, sign: str) -> Tensor:
    if sign == "+":
        return observation.plus_mask64
    if sign == "-":
        return observation.minus_mask64
    raise ValueError("source_sign must be '+' or '-'")


def erode_mask64(mask: Tensor, cells: int) -> Tensor:
    """Binary square erosion without introducing a cv2 dependency."""

    mask = mask.detach().cpu().bool()
    if tuple(mask.shape) != (_MASK_SIZE, _MASK_SIZE):
        raise ValueError("signed cue mask must have shape [64,64]")
    if cells == 0:
        return mask.clone()
    kernel = 2 * cells + 1
    count = F.conv2d(
        mask.float()[None, None],
        torch.ones((1, 1, kernel, kernel), dtype=torch.float32),
        padding=cells,
    )[0, 0]
    return count == float(kernel * kernel)


def project_points(points_world: Tensor, observation: SignedXFeatObservation) -> tuple[Tensor, Tensor]:
    """Project CPU world points into one explicit OpenCV camera."""

    points = points_world.detach().cpu().float().reshape(-1, 3)
    R = observation.w2c[:3, :3].float()
    t = observation.w2c[:3, 3].float()
    camera = points @ R.T + t
    depth = camera[:, 2]
    homogeneous = camera @ observation.K.float().T
    denominator = torch.where(homogeneous[:, 2].abs() > _EPS, homogeneous[:, 2], torch.ones_like(depth))
    uv = homogeneous[:, :2] / denominator[:, None]
    return uv, depth


def unproject_at_camera_depth(pixels_xy: Tensor, depth: Tensor, observation: SignedXFeatObservation) -> Tensor:
    """Unproject image pixels at explicit camera-z depths into world space."""

    pixels = pixels_xy.detach().cpu().float().reshape(-1, 2)
    z = depth.detach().cpu().float().reshape(-1)
    if len(pixels) != len(z):
        raise ValueError("pixels_xy and depth must have matching rows")
    homogeneous = torch.cat([pixels, torch.ones((len(pixels), 1))], dim=1)
    camera = (torch.linalg.inv(observation.K.float()) @ homogeneous.T).T * z[:, None]
    R = observation.w2c[:3, :3].float()
    t = observation.w2c[:3, 3].float()
    return (camera - t) @ R


def camera_center(observation: SignedXFeatObservation) -> Tensor:
    R = observation.w2c[:3, :3].float()
    t = observation.w2c[:3, 3].float()
    return -(R.T @ t)


def _inside_eroded_signed_mask(point_world: Tensor, observation: SignedXFeatObservation, sign: str, erosion: int) -> tuple[bool, bool]:
    uv, depth = project_points(point_world[None], observation)
    h, w = observation.image_size
    x, y = float(uv[0, 0]), float(uv[0, 1])
    visible = float(depth[0]) > 0.0 and 0.0 <= x < float(w) and 0.0 <= y < float(h)
    if not visible:
        return False, False
    col = min(_MASK_SIZE - 1, max(0, int(math.floor(x * _MASK_SIZE / float(w)))))
    row = min(_MASK_SIZE - 1, max(0, int(math.floor(y * _MASK_SIZE / float(h)))))
    mask = erode_mask64(_mask_for_sign(observation, sign), erosion)
    return True, bool(mask[row, col])


def _support_diagnostics(
    point_world: Tensor,
    observations: Sequence[SignedXFeatObservation],
    sign: str,
    config: NewSeedDensificationConfig,
) -> tuple[bool, tuple[int, ...], int, float, float, float, str]:
    visible = 0
    hits: list[SignedXFeatObservation] = []
    for observation in observations:
        is_visible, in_mask = _inside_eroded_signed_mask(
            point_world, observation, sign, config.erosion_cells
        )
        visible += int(is_visible)
        if is_visible and in_mask:
            hits.append(observation)
    ratio = len(hits) / max(visible, 1)
    if len(hits) < config.min_support_views:
        return False, tuple(obs.frame_index for obs in hits), visible, ratio, 0.0, 0.0, "support_views"
    if ratio < config.min_support_ratio:
        return False, tuple(obs.frame_index for obs in hits), visible, ratio, 0.0, 0.0, "support_ratio"

    centers = torch.stack([camera_center(obs) for obs in hits], dim=0)
    max_baseline = float(torch.pdist(centers).max().item()) if len(centers) > 1 else 0.0
    if max_baseline < config.min_baseline:
        return False, tuple(obs.frame_index for obs in hits), visible, ratio, max_baseline, 0.0, "baseline"
    rays = point_world[None] - centers
    rays = rays / torch.linalg.norm(rays, dim=1, keepdim=True).clamp_min(_EPS)
    cosine = (rays @ rays.T).clamp(-1.0, 1.0)
    max_angle = float(torch.rad2deg(torch.acos(cosine.min())).item())
    if max_angle < config.min_view_angle_deg:
        return False, tuple(obs.frame_index for obs in hits), visible, ratio, max_baseline, max_angle, "view_angle"
    return (
        True,
        tuple(obs.frame_index for obs in hits),
        visible,
        ratio,
        max_baseline,
        max_angle,
        "accepted",
    )


def densify_confirmed_new_region(
    *,
    current: SignedXFeatObservation,
    history: Sequence[SignedXFeatObservation],
    source_sign: str,
    active_xyz: Tensor,
    active_rows: Tensor,
    config: NewSeedDensificationConfig,
    max_new: int | None = None,
) -> DensifiedSeedBatch:
    """Generate fixed child seeds in under-covered, multi-view-consistent NEW cue cells.

    ``active_xyz`` contains only the currently active NEW sidecar rows.  No
    reference Gaussian, reference depth, GT mask, or future observation enters
    this function.
    """

    if source_sign not in {"+", "-"}:
        raise ValueError("source_sign must be '+' or '-'")
    xyz = active_xyz.detach().cpu().float().reshape(-1, 3)
    rows = active_rows.detach().cpu().long().reshape(-1)
    if len(xyz) != len(rows):
        raise ValueError("active_xyz and active_rows must have matching rows")
    budget = config.max_new_per_frame if max_new is None else min(config.max_new_per_frame, int(max_new))
    if len(xyz) == 0 or budget <= 0:
        return _empty_result()

    mask = erode_mask64(_mask_for_sign(current, source_sign), config.erosion_cells)
    cells_rc = torch.nonzero(mask, as_tuple=False).float()
    if len(cells_rc) == 0:
        return _empty_result()

    parent_uv, parent_depth = project_points(xyz, current)
    h, w = current.image_size
    parent_visible = (
        (parent_depth > 0.0)
        & (parent_uv[:, 0] >= 0.0)
        & (parent_uv[:, 0] < float(w))
        & (parent_uv[:, 1] >= 0.0)
        & (parent_uv[:, 1] < float(h))
    )
    if not bool(parent_visible.any()):
        return _empty_result(considered=len(cells_rc), rejection_counts={"no_visible_parent": len(cells_rc)})
    visible_indices = torch.nonzero(parent_visible, as_tuple=False).flatten()
    visible_uv = parent_uv[visible_indices]
    parent_cells_xy = torch.stack(
        [
            visible_uv[:, 0] * _MASK_SIZE / float(w),
            visible_uv[:, 1] * _MASK_SIZE / float(h),
        ],
        dim=1,
    )
    candidate_cells_xy = torch.stack([cells_rc[:, 1] + 0.5, cells_rc[:, 0] + 0.5], dim=1)
    distances = torch.cdist(candidate_cells_xy, parent_cells_xy)
    nearest_distance, nearest_visible = distances.min(dim=1)
    uncovered = (nearest_distance > config.coverage_radius_cells) & (
        nearest_distance <= config.max_parent_distance_cells
    )
    candidate_ids = torch.nonzero(uncovered, as_tuple=False).flatten()
    if len(candidate_ids) == 0:
        return _empty_result(considered=len(cells_rc), uncovered=0)
    candidate_ids = candidate_ids[torch.argsort(nearest_distance[candidate_ids], descending=True)]

    causal_history = sorted(
        {obs.frame_index: obs for obs in history if obs.frame_index <= current.frame_index}.values(),
        key=lambda obs: obs.frame_index,
    )[-config.history_views :]
    if current.frame_index not in {obs.frame_index for obs in causal_history}:
        causal_history.append(current)
        causal_history.sort(key=lambda obs: obs.frame_index)

    accepted_xyz: list[Tensor] = []
    accepted_scale: list[Tensor] = []
    accepted_parent_rows: list[int] = []
    support_frames: list[tuple[int, ...]] = []
    support_ratios: list[float] = []
    visible_views: list[int] = []
    max_baselines: list[float] = []
    max_angles: list[float] = []
    selected_cells: list[Tensor] = []
    rejection_counts: dict[str, int] = {}

    for candidate_id in candidate_ids.tolist():
        if len(accepted_xyz) >= budget:
            break
        cell_xy = candidate_cells_xy[candidate_id]
        if selected_cells:
            selected_distance = torch.linalg.norm(torch.stack(selected_cells) - cell_xy[None], dim=1)
            if bool((selected_distance < config.min_child_separation_cells).any()):
                rejection_counts["cell_separation"] = rejection_counts.get("cell_separation", 0) + 1
                continue
        visible_parent_pos = int(nearest_visible[candidate_id])
        parent_local = int(visible_indices[visible_parent_pos])
        pixel_xy = torch.tensor(
            [[cell_xy[0] * float(w) / _MASK_SIZE, cell_xy[1] * float(h) / _MASK_SIZE]],
            dtype=torch.float32,
        )
        point = unproject_at_camera_depth(
            pixel_xy,
            parent_depth[parent_local : parent_local + 1],
            current,
        )[0]
        all_existing = xyz if not accepted_xyz else torch.cat([xyz, torch.stack(accepted_xyz)], dim=0)
        if bool((torch.linalg.norm(all_existing - point[None], dim=1) < config.min_world_separation).any()):
            rejection_counts["world_separation"] = rejection_counts.get("world_separation", 0) + 1
            continue

        passed, frames, n_visible, ratio, baseline, angle, reason = _support_diagnostics(
            point, causal_history, source_sign, config
        )
        if not passed:
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            continue

        focal = math.sqrt(float(current.K[0, 0]) * float(current.K[1, 1]))
        radius_world = float(parent_depth[parent_local]) * config.footprint_px / focal
        log_radius = math.log(float(min(0.10, max(1.0e-4, radius_world))))
        accepted_xyz.append(point)
        accepted_scale.append(torch.full((3,), log_radius, dtype=torch.float32))
        accepted_parent_rows.append(int(rows[parent_local]))
        support_frames.append(frames)
        support_ratios.append(float(ratio))
        visible_views.append(int(n_visible))
        max_baselines.append(float(baseline))
        max_angles.append(float(angle))
        selected_cells.append(cell_xy)

    if not accepted_xyz:
        return _empty_result(
            considered=len(cells_rc),
            uncovered=len(candidate_ids),
            rejection_counts=rejection_counts,
        )
    return DensifiedSeedBatch(
        xyz=torch.stack(accepted_xyz),
        log_scaling=torch.stack(accepted_scale),
        parent_rows=torch.tensor(accepted_parent_rows, dtype=torch.long),
        support_frames=tuple(support_frames),
        support_ratios=tuple(support_ratios),
        visible_views=tuple(visible_views),
        max_baselines=tuple(max_baselines),
        max_view_angles_deg=tuple(max_angles),
        considered_cells=int(len(cells_rc)),
        uncovered_cells=int(len(candidate_ids)),
        rejection_counts=rejection_counts,
    )


__all__ = [
    "DensifiedSeedBatch",
    "NewSeedDensificationConfig",
    "camera_center",
    "densify_confirmed_new_region",
    "erode_mask64",
    "project_points",
    "unproject_at_camera_depth",
]
