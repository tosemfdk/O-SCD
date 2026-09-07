"""Depth-prior point seeding for causally confirmed NEW image regions.

The functions in this module do not infer whether a change is NEW or REMOVED.
They consume an already-confirmed NEW mask, align a monocular/multiview depth
prediction to immutable reference-scene depth, and back-project only pixels
that lie in front of the reference surface.  Ground-truth masks are not part
of this path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


_EPS = 1.0e-8


@dataclass(frozen=True)
class DepthScaleFit:
    """Robust positive scale taking predicted camera-z into scene units."""

    scale: float
    samples: int
    inliers: int
    log_ratio_mad: float
    median_absolute_relative_error: float


@dataclass(frozen=True)
class DepthPriorSeedConfig:
    """Conservative gates for depth-prior NEW seed creation."""

    min_reference_alpha: float = 0.50
    stable_cue_threshold: float = 0.20
    confidence_quantile: float = 0.40
    min_front_gap: float = 0.03
    min_front_gap_ratio: float = 0.015
    erosion_pixels: int = 1
    sampling_stride: int = 4
    max_seeds: int = 2048
    footprint_pixels: float = 2.0
    min_scale: float = 1.0e-4
    max_scale: float = 0.10

    def __post_init__(self) -> None:
        for name in ("min_reference_alpha", "stable_cue_threshold"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if not 0.0 <= float(self.confidence_quantile) < 1.0:
            raise ValueError("confidence_quantile must be in [0,1)")
        for name in ("footprint_pixels", "min_scale", "max_scale"):
            if not math.isfinite(float(getattr(self, name))) or float(
                getattr(self, name)
            ) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("min_front_gap", "min_front_gap_ratio"):
            if not math.isfinite(float(getattr(self, name))) or float(
                getattr(self, name)
            ) < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.min_scale > self.max_scale:
            raise ValueError("min_scale cannot exceed max_scale")
        for name in ("sampling_stride", "max_seeds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.erosion_pixels, bool)
            or not isinstance(self.erosion_pixels, int)
            or self.erosion_pixels < 0
        ):
            raise ValueError("erosion_pixels must be a non-negative integer")


@dataclass(frozen=True)
class DepthPriorSeedBatch:
    """Selected pixels and their fixed initial 3D Gaussian attributes."""

    xyz: Tensor
    pixels_xy: Tensor
    camera_depth: Tensor
    log_scaling: Tensor
    confidence: Tensor
    candidate_mask: Tensor
    confidence_threshold: float

    @property
    def count(self) -> int:
        return int(self.xyz.shape[0])


@dataclass(frozen=True)
class DepthFrontSignEvidence:
    """Soft cue mass where the observed surface lies before the reference."""

    plus_mass: float
    minus_mass: float
    plus_pixels: int
    minus_pixels: int
    front_mask: Tensor
    confidence_threshold: float

    @property
    def total_mass(self) -> float:
        return self.plus_mass + self.minus_mass

    @property
    def total_pixels(self) -> int:
        return self.plus_pixels + self.minus_pixels

    @property
    def plus_fraction(self) -> float:
        return self.plus_mass / max(self.total_mass, _EPS)


def retain_top_magnitude_mask(
    score: Tensor, candidate_mask: Tensor, *, quantile: float
) -> tuple[Tensor, float]:
    """Keep only the largest absolute scores inside a candidate mask."""

    values = score.detach().cpu().float().squeeze()
    candidate = candidate_mask.detach().cpu().bool().squeeze()
    if values.ndim != 2 or candidate.shape != values.shape:
        raise ValueError("score and candidate_mask must share shape [H,W]")
    if not math.isfinite(float(quantile)) or not 0.0 <= float(quantile) < 1.0:
        raise ValueError("quantile must lie in [0,1)")
    finite_candidate = candidate & torch.isfinite(values)
    if not bool(finite_candidate.any()):
        return torch.zeros_like(candidate), math.inf
    threshold = float(
        torch.quantile(values[finite_candidate].abs(), float(quantile)).item()
    )
    return finite_candidate & (values.abs() >= threshold), threshold


def q_weighted_upsampled_signed_sam_score(
    score: Tensor,
    learned_q: Tensor,
    *,
    height: int,
    width: int,
) -> tuple[Tensor, float]:
    """Return panel-7's continuous signed SAM score multiplied by learned Q.

    The native SAM score is normalized once on its 64x64 grid, bilinearly
    upsampled, and multiplied by the continuous learned cue.  No epsilon,
    magnitude-quantile, or hard cue-support gate is applied here.
    """

    values = _as_depth("score", score)
    cue = _as_depth("learned_q", learned_q)
    if height < 1 or width < 1:
        raise ValueError("height and width must be positive")
    normalization_scale = float(values.abs().amax().item())
    if normalization_scale > 0.0:
        values = values / normalization_scale
    else:
        values = torch.zeros_like(values)
    signed_full = F.interpolate(
        values[None, None],
        size=(int(height), int(width)),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    if tuple(cue.shape) != (int(height), int(width)):
        cue = F.interpolate(
            cue[None, None],
            size=(int(height), int(width)),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
    return signed_full * cue.clamp(0.0, 1.0), normalization_scale


def panel7_visible_signed_support(
    weighted_score: Tensor,
    *,
    display_levels: int = 255,
) -> tuple[Tensor, Tensor]:
    """Return positive/negative support that is non-black in panel 7.

    Panel 7 is rendered by rounding ``abs(score) * 255`` to uint8.  Testing
    only ``score != 0`` would therefore admit tiny values that the viewer
    displays as black.  This helper uses the same quantization boundary so
    seed birth and representation supervision never consume a displayed-zero
    pixel.  It is a visualization-consistency rule, not a hard Q threshold.
    """

    values = torch.as_tensor(weighted_score).detach().float()
    if values.ndim != 2:
        raise ValueError("weighted_score must have shape [H,W]")
    if isinstance(display_levels, bool) or int(display_levels) < 1:
        raise ValueError("display_levels must be a positive integer")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("weighted_score must be finite")
    visible = torch.round(
        values.abs().clamp(0.0, 1.0) * int(display_levels)
    ) > 0.0
    return visible & (values > 0.0), visible & (values < 0.0)


def threshold_signed_support(
    weighted_score: Tensor,
    *,
    threshold: float,
) -> tuple[Tensor, Tensor]:
    """Split a signed score into strict positive/negative threshold support."""

    values = torch.as_tensor(weighted_score).detach().float()
    if values.ndim != 2:
        raise ValueError("weighted_score must have shape [H,W]")
    if not math.isfinite(float(threshold)) or float(threshold) < 0.0:
        raise ValueError("threshold must be finite and nonnegative")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("weighted_score must be finite")
    return values > float(threshold), values < -float(threshold)


@torch.no_grad()
def uncovered_by_learned_gaussian_support(
    proposal_xyz: Tensor,
    learned_xyz: Tensor,
    learned_scaling: Tensor,
    learned_update_counts: Tensor,
    eligible_learned: Tensor,
    *,
    min_updates: int,
    support_sigma: float,
    learned_chunk_size: int = 8192,
) -> Tensor:
    """Return proposal rows outside mature learned Gaussian support spheres.

    Coverage follows the *current optimized* center and scale, not the birth
    center or a fixed voxel.  A learned row becomes eligible only after it has
    received ``min_updates`` geometry optimizer steps.  Its conservative
    spherical support radius is ``support_sigma * max(scale_xyz)``.
    """

    proposals = torch.as_tensor(proposal_xyz).detach()
    centers = torch.as_tensor(learned_xyz).detach().to(
        device=proposals.device, dtype=proposals.dtype
    )
    scaling = torch.as_tensor(learned_scaling).detach().to(
        device=proposals.device, dtype=proposals.dtype
    )
    updates = torch.as_tensor(learned_update_counts).detach().to(
        device=proposals.device, dtype=torch.long
    ).flatten()
    eligible = torch.as_tensor(eligible_learned).detach().to(
        device=proposals.device, dtype=torch.bool
    ).flatten()
    if proposals.ndim != 2 or proposals.shape[1] != 3:
        raise ValueError("proposal_xyz must have shape [N,3]")
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("learned_xyz must have shape [M,3]")
    if scaling.shape != centers.shape:
        raise ValueError("learned_scaling must match learned_xyz")
    if updates.shape != (len(centers),) or eligible.shape != (len(centers),):
        raise ValueError("learned row state must have shape [M]")
    if isinstance(min_updates, bool) or int(min_updates) < 1:
        raise ValueError("min_updates must be a positive integer")
    if not math.isfinite(float(support_sigma)) or float(support_sigma) <= 0.0:
        raise ValueError("support_sigma must be finite and positive")
    if isinstance(learned_chunk_size, bool) or int(learned_chunk_size) < 1:
        raise ValueError("learned_chunk_size must be a positive integer")
    if not bool(torch.isfinite(proposals).all()):
        raise ValueError("proposal_xyz must be finite")
    if not bool(torch.isfinite(centers).all()) or not bool(torch.isfinite(scaling).all()):
        raise ValueError("learned geometry must be finite")
    if bool((scaling <= 0.0).any()):
        raise ValueError("learned scaling must be positive")

    mature = eligible & (updates >= int(min_updates))
    mature_rows = torch.nonzero(mature, as_tuple=False).flatten()
    uncovered = torch.ones(len(proposals), device=proposals.device, dtype=torch.bool)
    if proposals.numel() == 0 or mature_rows.numel() == 0:
        return uncovered
    for start in range(0, int(mature_rows.numel()), int(learned_chunk_size)):
        rows = mature_rows[start : start + int(learned_chunk_size)]
        distance = torch.cdist(proposals[uncovered], centers[rows])
        radius = scaling[rows].amax(dim=1) * float(support_sigma)
        newly_covered = (distance <= radius[None]).any(dim=1)
        if bool(newly_covered.any()):
            uncovered_rows = torch.nonzero(uncovered, as_tuple=False).flatten()
            uncovered[uncovered_rows[newly_covered]] = False
        if not bool(uncovered.any()):
            break
    return uncovered


def canonical_metric_depth(
    raw_depth: Tensor,
    processed_intrinsics: Tensor,
    *,
    canonical_focal: float = 300.0,
) -> Tensor:
    """Convert DA3Metric canonical depth to metric camera-z.

    DA3Metric predicts depth normalized to a 300 px canonical focal length.
    The official conversion uses the processed-image focal length in pixels.
    We use the arithmetic mean of ``fx`` and ``fy``, matching the model-card
    guidance for a single scalar focal length.
    """

    depth = _as_depth("raw_depth", raw_depth)
    intrinsics = processed_intrinsics.detach().cpu().float().squeeze()
    if intrinsics.shape != (3, 3):
        raise ValueError("processed_intrinsics must have shape [3,3]")
    focal = float((intrinsics[0, 0] + intrinsics[1, 1]).item() * 0.5)
    if not math.isfinite(focal) or focal <= 0.0:
        raise ValueError("processed focal length must be finite and positive")
    if not math.isfinite(float(canonical_focal)) or float(canonical_focal) <= 0.0:
        raise ValueError("canonical_focal must be finite and positive")
    return depth * (focal / float(canonical_focal))


def front_depth_sign_evidence(
    *,
    predicted_depth: Tensor,
    confidence: Tensor,
    reference_depth: Tensor,
    plus_mask: Tensor,
    minus_mask: Tensor,
    cue_strength: Tensor,
    scale: float,
    config: DepthPriorSeedConfig,
    cue_support_threshold: float = 0.05,
) -> DepthFrontSignEvidence:
    """Measure which SAM/PCA sign contains the front-of-reference depth mass.

    The depth residual is evaluated after the same positive scale-only alignment
    used for seed creation.  Soft cue values weight the evidence; they are not
    thresholded except for a broad support gate that rejects numerical tails.
    """

    if not 0.0 <= float(cue_support_threshold) <= 1.0:
        raise ValueError("cue_support_threshold must lie in [0,1]")
    predicted = _as_depth("predicted_depth", predicted_depth)
    conf = _as_depth("confidence", confidence)
    reference = _as_depth("reference_depth", reference_depth)
    plus = plus_mask.detach().cpu().bool().squeeze()
    minus = minus_mask.detach().cpu().bool().squeeze()
    cue = _as_depth("cue_strength", cue_strength)
    if not (
        predicted.shape
        == conf.shape
        == reference.shape
        == plus.shape
        == minus.shape
        == cue.shape
    ):
        raise ValueError("depth, masks, confidence, and cue must share shape [H,W]")
    if bool((plus & minus).any()):
        raise ValueError("plus_mask and minus_mask must be disjoint")

    aligned = predicted * float(scale)
    finite = (
        torch.isfinite(aligned)
        & torch.isfinite(reference)
        & torch.isfinite(conf)
        & torch.isfinite(cue)
        & (aligned > _EPS)
        & (reference > _EPS)
    )
    broad_support = finite & (cue >= float(cue_support_threshold))
    confidence_values = conf[broad_support]
    confidence_threshold = (
        float(
            torch.quantile(
                confidence_values, float(config.confidence_quantile)
            ).item()
        )
        if confidence_values.numel()
        else math.inf
    )
    required_gap = torch.maximum(
        torch.full_like(reference, float(config.min_front_gap)),
        reference * float(config.min_front_gap_ratio),
    )
    front = broad_support & (conf >= confidence_threshold)
    front &= (reference - aligned) > required_gap
    plus_front = front & plus
    minus_front = front & minus
    return DepthFrontSignEvidence(
        plus_mass=float(cue[plus_front].sum().item()),
        minus_mass=float(cue[minus_front].sum().item()),
        plus_pixels=int(plus_front.sum().item()),
        minus_pixels=int(minus_front.sum().item()),
        front_mask=front,
        confidence_threshold=confidence_threshold,
    )


def _as_depth(name: str, value: Tensor) -> Tensor:
    result = value.detach().cpu().float().squeeze()
    if result.ndim != 2:
        raise ValueError(f"{name} must reduce to shape [H,W]")
    return result


def _robust_positive_scale(
    predicted: Tensor,
    target: Tensor,
    *,
    mad_multiplier: float,
    min_samples: int,
) -> DepthScaleFit:
    """Fit ``target ~= scale * predicted`` from paired positive depths."""

    if not math.isfinite(float(mad_multiplier)) or float(mad_multiplier) <= 0.0:
        raise ValueError("mad_multiplier must be finite and positive")
    if isinstance(min_samples, bool) or not isinstance(min_samples, int):
        raise ValueError("min_samples must be an integer")
    if min_samples < 1:
        raise ValueError("min_samples must be positive")
    predicted = predicted.detach().cpu().float().reshape(-1)
    target = target.detach().cpu().float().reshape(-1)
    if predicted.shape != target.shape:
        raise ValueError("predicted and target depths must have matching rows")
    valid = torch.isfinite(predicted) & torch.isfinite(target)
    valid &= (predicted > _EPS) & (target > _EPS)
    predicted = predicted[valid]
    target = target[valid]
    samples = int(predicted.numel())
    if samples < min_samples:
        raise ValueError(
            f"at least {min_samples} valid depth correspondences are required"
        )
    log_ratio = torch.log(target) - torch.log(predicted)
    center = log_ratio.median()
    absolute = (log_ratio - center).abs()
    mad = absolute.median()
    if float(mad) <= _EPS:
        inlier_mask = absolute <= 1.0e-6
    else:
        robust_sigma = 1.4826 * mad
        inlier_mask = absolute <= float(mad_multiplier) * robust_sigma
    if int(inlier_mask.sum().item()) < min_samples:
        inlier_mask = torch.ones_like(log_ratio, dtype=torch.bool)
    scale = float(torch.exp(log_ratio[inlier_mask].median()).item())
    aligned = predicted[inlier_mask] * scale
    target_inliers = target[inlier_mask]
    relative = (aligned - target_inliers).abs() / target_inliers.clamp_min(_EPS)
    return DepthScaleFit(
        scale=scale,
        samples=samples,
        inliers=int(inlier_mask.sum().item()),
        log_ratio_mad=float(mad.item()),
        median_absolute_relative_error=float(relative.median().item()),
    )


def fit_reference_depth_scale(
    predicted_depth: Tensor,
    reference_depth: Tensor,
    valid_mask: Tensor,
    *,
    mad_multiplier: float = 3.5,
) -> DepthScaleFit:
    """Fit ``reference_depth ~= scale * predicted_depth`` in log-ratio space.

    A positive scale-only fit preserves the pinhole camera origin.  An affine
    depth shift would move points non-rigidly along their rays and is therefore
    intentionally excluded.
    """

    predicted = _as_depth("predicted_depth", predicted_depth)
    reference = _as_depth("reference_depth", reference_depth)
    mask = valid_mask.detach().cpu().bool().squeeze()
    if predicted.shape != reference.shape or mask.shape != predicted.shape:
        raise ValueError("depths and valid_mask must share shape [H,W]")
    mask &= torch.isfinite(predicted) & torch.isfinite(reference)
    mask &= (predicted > _EPS) & (reference > _EPS)
    return _robust_positive_scale(
        predicted[mask],
        reference[mask],
        mad_multiplier=mad_multiplier,
        min_samples=16,
    )


def _sample_metric_anchor_pairs(
    predicted_depth: Tensor,
    anchor_pixels_xy: Tensor,
    anchor_camera_depth: Tensor,
) -> tuple[Tensor, Tensor]:
    predicted = _as_depth("predicted_depth", predicted_depth)
    pixels = anchor_pixels_xy.detach().cpu().float().reshape(-1, 2)
    target = anchor_camera_depth.detach().cpu().float().reshape(-1)
    if len(pixels) != len(target):
        raise ValueError("anchor pixels and camera depths must have matching rows")
    height, width = predicted.shape
    finite = torch.isfinite(pixels).all(dim=1) & torch.isfinite(target)
    finite &= target > _EPS
    finite &= (pixels[:, 0] >= 0.0) & (pixels[:, 0] <= float(width - 1))
    finite &= (pixels[:, 1] >= 0.0) & (pixels[:, 1] <= float(height - 1))
    pixels = pixels[finite]
    target = target[finite]
    if pixels.numel() == 0:
        sampled = torch.empty((0,), dtype=torch.float32)
    else:
        normalizer = torch.tensor(
            [max(width - 1, 1), max(height - 1, 1)], dtype=torch.float32
        )
        grid = (2.0 * pixels / normalizer - 1.0).reshape(1, -1, 1, 2)
        sampled = F.grid_sample(
            predicted[None, None],
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).reshape(-1)
    valid_depth = torch.isfinite(sampled) & (sampled > _EPS)
    return sampled[valid_depth], target[valid_depth]


def fit_metric_anchor_depth_scale(
    predicted_depth: Tensor,
    anchor_pixels_xy: Tensor,
    anchor_camera_depth: Tensor,
    *,
    mad_multiplier: float = 3.5,
    min_samples: int = 16,
) -> DepthScaleFit:
    """Fit monocular depth scale at sparse metric localization anchors.

    ``anchor_pixels_xy`` is expressed directly in the native predicted-depth
    image coordinates. Values are sampled bilinearly without first upsampling
    the dense depth map. ``anchor_camera_depth`` is metric camera-z obtained by
    transforming each matched reference 3D point with the current camera pose.
    """

    sampled, target = _sample_metric_anchor_pairs(
        predicted_depth, anchor_pixels_xy, anchor_camera_depth
    )
    return _robust_positive_scale(
        sampled,
        target,
        mad_multiplier=mad_multiplier,
        min_samples=min_samples,
    )


def metric_anchor_median_relative_error(
    predicted_depth: Tensor,
    anchor_pixels_xy: Tensor,
    anchor_camera_depth: Tensor,
    *,
    scale: float,
) -> float:
    """Evaluate one positive scale on all valid sparse metric anchors."""

    if not math.isfinite(float(scale)) or float(scale) <= 0.0:
        raise ValueError("scale must be finite and positive")
    sampled, target = _sample_metric_anchor_pairs(
        predicted_depth, anchor_pixels_xy, anchor_camera_depth
    )
    if sampled.numel() == 0:
        raise ValueError("at least one valid metric depth anchor is required")
    relative = (sampled * float(scale) - target).abs() / target.clamp_min(_EPS)
    return float(relative.median().item())


def erode_binary_mask(mask: Tensor, pixels: int) -> Tensor:
    result = mask.detach().cpu().bool().squeeze()
    if result.ndim != 2:
        raise ValueError("mask must reduce to shape [H,W]")
    if pixels == 0:
        return result.clone()
    kernel = 2 * pixels + 1
    count = F.conv2d(
        result.float()[None, None],
        torch.ones((1, 1, kernel, kernel), dtype=torch.float32),
        padding=pixels,
    )[0, 0]
    return count == float(kernel * kernel)


def reference_scale_anchor_mask(
    reference_depth: Tensor,
    reference_alpha: Tensor,
    cue_strength: Tensor,
    *,
    config: DepthPriorSeedConfig,
) -> Tensor:
    """Select unchanged opaque reference pixels for scale calibration."""

    depth = _as_depth("reference_depth", reference_depth)
    alpha = _as_depth("reference_alpha", reference_alpha)
    cue = _as_depth("cue_strength", cue_strength)
    if depth.shape != alpha.shape or depth.shape != cue.shape:
        raise ValueError("reference depth, alpha, and cue must share shape")
    return (
        torch.isfinite(depth)
        & (depth > _EPS)
        & (alpha >= float(config.min_reference_alpha))
        & (cue < float(config.stable_cue_threshold))
    )


def unproject_camera_depth(
    pixels_xy: Tensor, depth: Tensor, K: Tensor, w2c: Tensor
) -> Tensor:
    """Back-project OpenCV camera-z depths into world coordinates."""

    pixels = pixels_xy.detach().cpu().float().reshape(-1, 2)
    z = depth.detach().cpu().float().reshape(-1)
    intrinsics = K.detach().cpu().float()
    extrinsics = w2c.detach().cpu().float()
    if intrinsics.shape != (3, 3) or extrinsics.shape != (4, 4):
        raise ValueError("K and w2c must have shapes [3,3] and [4,4]")
    if len(pixels) != len(z):
        raise ValueError("pixels_xy and depth must have matching rows")
    homogeneous = torch.cat((pixels, torch.ones((len(pixels), 1))), dim=1)
    camera = (torch.linalg.inv(intrinsics) @ homogeneous.T).T * z[:, None]
    rotation = extrinsics[:3, :3]
    translation = extrinsics[:3, 3]
    return (camera - translation) @ rotation


def _stratified_top_confidence(
    mask: Tensor,
    confidence: Tensor,
    stride: int,
    maximum: int,
    priority: Tensor | None = None,
) -> Tensor:
    """Choose one pixel per cell, optionally prioritizing an external cue."""

    rows, cols = torch.nonzero(mask, as_tuple=True)
    if rows.numel() == 0:
        return torch.empty((0, 2), dtype=torch.long)
    width = int(mask.shape[1])
    cells_wide = (width + stride - 1) // stride
    cell_ids = (rows // stride) * cells_wide + (cols // stride)
    order = torch.argsort(confidence[rows, cols], descending=True, stable=True)
    if priority is not None:
        cue_order = torch.argsort(
            priority[rows[order], cols[order]], descending=True, stable=True
        )
        order = order[cue_order]
    chosen: list[int] = []
    seen: set[int] = set()
    for index in order.tolist():
        cell = int(cell_ids[index].item())
        if cell in seen:
            continue
        seen.add(cell)
        chosen.append(index)
        if len(chosen) >= maximum:
            break
    selected = torch.tensor(chosen, dtype=torch.long)
    return torch.stack((cols[selected], rows[selected]), dim=1)


def build_depth_prior_new_seeds(
    *,
    predicted_depth: Tensor,
    confidence: Tensor,
    reference_depth: Tensor,
    new_mask: Tensor,
    K: Tensor,
    w2c: Tensor,
    scale: float,
    config: DepthPriorSeedConfig,
    sampling_priority: Tensor | None = None,
) -> DepthPriorSeedBatch:
    """Create fixed 3D seeds only where NEW depth is in front of reference."""

    predicted = _as_depth("predicted_depth", predicted_depth)
    conf = _as_depth("confidence", confidence)
    reference = _as_depth("reference_depth", reference_depth)
    new = erode_binary_mask(new_mask, config.erosion_pixels)
    if predicted.shape != conf.shape or predicted.shape != reference.shape:
        raise ValueError("depth, confidence, and reference depth must share shape")
    if new.shape != predicted.shape:
        raise ValueError("new_mask must share the depth shape")
    priority = None
    if sampling_priority is not None:
        priority = _as_depth("sampling_priority", sampling_priority)
        if priority.shape != predicted.shape:
            raise ValueError("sampling_priority must share the depth shape")
        if not bool(torch.isfinite(priority).all()):
            raise ValueError("sampling_priority must be finite")
    aligned = predicted * float(scale)
    finite = (
        torch.isfinite(aligned)
        & torch.isfinite(reference)
        & torch.isfinite(conf)
        & (aligned > _EPS)
        & (reference > _EPS)
    )
    confidence_values = conf[finite & new]
    threshold = (
        float(
            torch.quantile(confidence_values, float(config.confidence_quantile)).item()
        )
        if confidence_values.numel()
        else math.inf
    )
    required_gap = torch.maximum(
        torch.full_like(reference, float(config.min_front_gap)),
        reference * float(config.min_front_gap_ratio),
    )
    candidate = finite & new & (conf >= threshold)
    candidate &= (reference - aligned) > required_gap
    pixels = _stratified_top_confidence(
        candidate,
        conf,
        int(config.sampling_stride),
        int(config.max_seeds),
        priority,
    )
    if pixels.numel() == 0:
        return DepthPriorSeedBatch(
            xyz=torch.empty((0, 3), dtype=torch.float32),
            pixels_xy=pixels,
            camera_depth=torch.empty((0,), dtype=torch.float32),
            log_scaling=torch.empty((0, 3), dtype=torch.float32),
            confidence=torch.empty((0,), dtype=torch.float32),
            candidate_mask=candidate,
            confidence_threshold=threshold,
        )
    x, y = pixels[:, 0], pixels[:, 1]
    z = aligned[y, x]
    xyz = unproject_camera_depth(pixels.float(), z, K, w2c)
    focal = math.sqrt(float(K[0, 0]) * float(K[1, 1]))
    radius = (z * float(config.footprint_pixels) / focal).clamp(
        float(config.min_scale), float(config.max_scale)
    )
    scaling = torch.log(radius)[:, None].repeat(1, 3)
    return DepthPriorSeedBatch(
        xyz=xyz,
        pixels_xy=pixels,
        camera_depth=z,
        log_scaling=scaling,
        confidence=conf[y, x],
        candidate_mask=candidate,
        confidence_threshold=threshold,
    )


__all__ = [
    "DepthFrontSignEvidence",
    "DepthPriorSeedBatch",
    "DepthPriorSeedConfig",
    "DepthScaleFit",
    "build_depth_prior_new_seeds",
    "erode_binary_mask",
    "fit_metric_anchor_depth_scale",
    "fit_reference_depth_scale",
    "front_depth_sign_evidence",
    "panel7_visible_signed_support",
    "q_weighted_upsampled_signed_sam_score",
    "threshold_signed_support",
    "uncovered_by_learned_gaussian_support",
    "metric_anchor_median_relative_error",
    "reference_scale_anchor_mask",
    "retain_top_magnitude_mask",
    "unproject_camera_depth",
]
