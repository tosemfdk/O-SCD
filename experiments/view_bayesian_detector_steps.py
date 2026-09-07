"""Step through causal detector updates and online R_change optimization.

In ``panel10_new``, each press first proposes current-view DA3 seeds, then
consumes exactly one pre-optimization raw cue with a shared base+seed BF30
detector. Its probes are immutable reference rows and fixed birth seed rows.
The legacy variant retains its separate, next-frame seed detector. It then performs 16 online
representation updates by default.  Every update selects the latest observed
view with probability 0.33 and otherwise uniformly samples an already-observed
view.  A sampled frame ``k`` is rendered with the base and DA3 lifespan
population valid at ``k``; current-frame lifecycle masks are never paired with
historical camera/cue targets.  This reproduces O-SCD's causal replay policy
without future-frame or future-born Gaussian access.

Committed OPEN base Gaussians learn DC only.  Causally born DA3 NEW seeds learn
DC plus xyz, opacity, scale, and rotation while the reference representation is
kept frozen.

The main render follows the committed lifecycle color contract:

* black: reference-consistent and NEVER_OPEN;
* green: committed OPEN;
* red: committed CLOSED after at least one OPEN.

Usage:
    /home/rvl/miniforge3/envs/oscd/bin/python \
        -m experiments.view_bayesian_detector_steps --port 8090
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from io import BytesIO
import json
import math
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
from torch import nn
import viser

from experiments.run_online_bayesian_lifespan_thaw import (
    BASE_PLY_REL,
    build_causal_records,
)
from experiments.train_cue_temporal_rchange import (
    DEFAULT_CUE_CACHE,
    DEFAULT_FIXED_CAMERAS,
    build_fixed_cue_views,
    camera_json_to_w2c,
    load_fixed_camera_index,
    validate_cue_cache,
)
from experiments.train_real_temporal_rchange import DEFAULT_SOURCE
from temporal.bayesian_detector_visualization import (
    DetectorVisualState,
    detector_visual_state,
    black_green_yellow_red_heatmap,
    normalized_log_bayes_factor,
)
from temporal.change_cue_fusion import (
    PowerProductCue,
    fuse_power_product,
    normalized_oscd_pixel_cue_from_terms,
    oscd_pixel_terms,
    power_product_from_cached_sum,
    semantic_from_cached_sum,
    sigmoid_soft_binarize,
    smoothstep_soft_binarize,
)
from temporal.depth_prior_new_seeding import (
    fit_reference_depth_scale,
    panel7_visible_signed_support,
    q_weighted_upsampled_signed_sam_score,
    threshold_signed_support,
    uncovered_by_learned_gaussian_support,
)
from temporal.fusion import compute_ssf_loss
from temporal.masked_optimizer import MaskedRowAdam
from temporal.seed_projected_occupancy import (
    projected_covariance_ellipse_occupancy,
    scale_depth_intrinsics_to_image,
    seed_rows_for_root_birth_2d_occupancy,
    union_projected_gaussian_ellipse_,
)
from utils.sh_utils import RGB2SH


DISPLAY_LIFECYCLE = "Committed lifecycle: black/green/red"
DISPLAY_PROJECTED = "Current cue projected to Gaussians"
DISPLAY_BAYESIAN = "Accumulated Bayesian instability"
DISPLAY_LEARNED = "Learned current R_change prediction"
DISPLAY_OPTIONS = (
    DISPLAY_LIFECYCLE,
    DISPLAY_PROJECTED,
    DISPLAY_BAYESIAN,
    DISPLAY_LEARNED,
)
DEFAULT_DASHBOARD_ASPECT = 16.0 / 9.0
PANEL_LABEL_HEIGHT = 44
GT_CHANGE_COLOR = (255, 255, 255)
DEPTH_ALIGNMENT_MAX_CUE = 0.2
DEPTH_ALIGNMENT_MIN_REFERENCE_ALPHA = 0.5
PANEL7_DEPTH_SUPPORT_THRESHOLD = 0.1
DEPTH_DIFFERENCE_SUPPORT_THRESHOLD = 0.03


@dataclass(frozen=True)
class ReplayStepSummary:
    """Small CPU summary retained by the GUI after one causal detector step."""

    timestamp: int
    frame_name: str
    observed: int
    never_open: int
    uncertain: int
    open: int
    closed: int
    opened_now: int
    closed_now: int
    candidate_started: int
    candidate_continued: int
    candidate_rejected: int
    candidate_committed: int
    positive_pseudocount_mass: float
    negative_pseudocount_mass: float
    cue_tau: float | None = None
    cue_width: float | None = None
    representation_updates: int = 0
    sampled_latest_branch: int = 0
    sampled_oldest_timestamp: int | None = None
    sampled_newest_timestamp: int | None = None
    historical_lifespan_replay_updates: int = 0
    lifespan_render_violations: int = 0
    representation_loss: float | None = None
    trainable_open_rows: int = 0
    active_da3_seed_rows: int = 0
    visible_da3_seed_rows: int = 0
    visible_pending_da3_seed_rows: int = 0
    da3_seed_opened_now: int = 0
    da3_seed_closed_now: int = 0
    da3_seed_never_open: int = 0
    da3_seed_closed: int = 0
    da3_seed_proposed_now: int = 0
    da3_seed_accepted_now: int = 0
    da3_seed_coverage_rejected_now: int = 0
    da3_seed_coverage_3d_rejected_now: int = 0
    da3_seed_coverage_2d_existing_rejected_now: int = 0
    da3_seed_coverage_2d_same_frame_rejected_now: int = 0
    da3_seed_budget_rejected_now: int = 0
    da3_seed_occupancy_2d_pixels: int = 0
    da3_seed_occupancy_2d_rows: int = 0
    base_loss: float = 0.0
    new_loss: float = 0.0
    da3_density_children: int = 0
    da3_cloned_now: int = 0
    da3_split_sources_now: int = 0
    da3_pruned_now: int = 0
    da3_seed_observed_now: int = 0


@dataclass(frozen=True)
class AddRemoveMaskPaths:
    """Evaluation-only object masks partitioned by annotation change state."""

    add: tuple[Path, ...]
    remove: tuple[Path, ...]


@dataclass(frozen=True)
class LearnedSigmoidBoundary:
    """One frame's learned Stage-2 sigmoid parameters."""

    tau: float
    width: float


@dataclass(frozen=True)
class DA3SeedReplay:
    """Fixed DA3 seed geometry indexed by its causal birth frame."""

    xyz: np.ndarray
    birth_global: np.ndarray
    birth_name: tuple[str, ...]
    log_scaling: np.ndarray | None = None
    source_sign: tuple[str, ...] = ()


@dataclass(frozen=True)
class DA3MetricDepthReplay:
    """DA3Metric configuration and caches associated with a seed replay."""

    cache_root: Path
    model_name: str
    process_res: int


@dataclass(frozen=True)
class RepresentationReplayFrame:
    """One already-observed training view and its causal supervision maps."""

    timestamp: int
    view: Any
    cue_target: torch.Tensor
    new_target: torch.Tensor


@dataclass(frozen=True)
class ViewerFrameSnapshot:
    """Lossless display history, deliberately excluding detector/optimizer state."""

    panels: tuple[tuple[str, bytes], ...]
    status: str
    cue_status: str

    @classmethod
    def capture(
        cls, panels: Sequence[tuple[str, np.ndarray]], status: str, cue_status: str
    ) -> ViewerFrameSnapshot:
        encoded = []
        for title, rgb in panels:
            buffer = BytesIO()
            Image.fromarray(rgb).save(buffer, format="PNG")
            encoded.append((title, buffer.getvalue()))
        return cls(tuple(encoded), status, cue_status)

    def decoded_panels(self) -> tuple[tuple[str, np.ndarray], ...]:
        panels = []
        for title, data in self.panels:
            with Image.open(BytesIO(data)) as image:
                panels.append((title, np.asarray(image.convert("RGB")).copy()))
        return tuple(panels)


@dataclass(frozen=True)
class DA3SeedProjectionStats:
    """Counts shown alongside the 2D seed-center overlay."""

    accepted_so_far: int = 0
    visible: int = 0
    born_now: int = 0
    born_now_visible: int = 0


@dataclass(frozen=True)
class SamSignedFeatureFrame:
    """Causal PC1 axis and strong signed thresholds for one online frame."""

    axis: np.ndarray
    epsilon_negative: float
    epsilon_positive: float


@dataclass(frozen=True)
class SamSignedFeatureStats:
    """Signed Q-weighted SAM cue counts and display normalization."""

    positive: int = 0
    negative: int = 0
    neutral: int = 0
    normalization_scale: float = 0.0


@dataclass(frozen=True)
class DepthDifferenceStats:
    """Summary of rendered-GS versus aligned-DA3 depth residuals."""

    valid_pixels: int = 0
    median_absolute_difference: float = 0.0
    normalization_scale: float = 0.0
    alignment_scale: float = 1.0
    alignment_samples: int = 0
    alignment_inliers: int = 0
    alignment_median_relative_error: float = 0.0


@dataclass(frozen=True)
class CueTypeStats:
    """Visible Panel-3 cue pixels partitioned by signed agreement."""

    new: int = 0
    removed: int = 0
    appearance: int = 0
    background: int = 0


def load_learned_sigmoid_boundaries(
    path: Path,
) -> tuple[dict[str, LearnedSigmoidBoundary], float]:
    """Load and validate a Stage-2 per-frame boundary artifact."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("remap") != "sigmoid":
        raise ValueError("learned boundary artifact must declare remap=sigmoid")
    edge_probability = float(payload.get("edge_probability", float("nan")))
    if not math.isfinite(edge_probability) or not 0.0 < edge_probability < 0.5:
        raise ValueError("learned boundary edge_probability must lie in (0,0.5)")
    raw_frames = payload.get("frames")
    if not isinstance(raw_frames, dict) or not raw_frames:
        raise ValueError("learned boundary artifact must contain frame parameters")
    boundaries: dict[str, LearnedSigmoidBoundary] = {}
    for frame_name, raw in raw_frames.items():
        if not isinstance(raw, dict):
            raise ValueError(f"invalid learned boundary for {frame_name}")
        tau = float(raw.get("tau", float("nan")))
        width = float(raw.get("width", float("nan")))
        if not math.isfinite(tau) or not 0.0 < tau < 1.0:
            raise ValueError(f"invalid tau for {frame_name}")
        if not math.isfinite(width) or width <= 0.0:
            raise ValueError(f"invalid width for {frame_name}")
        boundaries[str(frame_name)] = LearnedSigmoidBoundary(tau=tau, width=width)
    return boundaries, edge_probability


def load_causal_sam_sign_trace(
    root: Path, records: Sequence[Any]
) -> dict[str, SamSignedFeatureFrame]:
    """Load scene-local causal PC1 traces and align their sign continuously."""

    raw: dict[str, SamSignedFeatureFrame] = {}
    local_trace = root / "causal_pca_posterior_arrays.npz"
    paths = ([local_trace] if local_trace.is_file() else [
        root / f"scene_change{scene}" / "causal_pca_posterior_arrays.npz"
        for scene in (1, 2, 3)
    ])
    for path in paths:
        if not path.is_file():
            continue
        with np.load(path) as arrays:
            names = arrays["frame_names"].astype(str)
            axes = np.asarray(arrays["pc1_axes"], dtype=np.float32)
            negative = np.asarray(arrays["epsilon_negative"], dtype=np.float64)
            positive = np.asarray(arrays["epsilon_positive"], dtype=np.float64)
        if axes.shape != (len(names), 256):
            raise ValueError(f"invalid SAM PC1 axes in {path}")
        if negative.shape != names.shape or positive.shape != names.shape:
            raise ValueError(f"invalid SAM sign thresholds in {path}")
        for index, name in enumerate(names.tolist()):
            if name in raw:
                raise ValueError(f"duplicate SAM sign trace frame: {name}")
            raw[name] = SamSignedFeatureFrame(
                axis=np.ascontiguousarray(axes[index]),
                epsilon_negative=float(negative[index]),
                epsilon_positive=float(positive[index]),
            )

    aligned: dict[str, SamSignedFeatureFrame] = {}
    previous_axis: np.ndarray | None = None
    for record in records:
        name = str(record.name)
        frame = raw.get(name)
        if frame is None:
            raise KeyError(f"SAM sign trace missing replay frame: {name}")
        axis = frame.axis.copy()
        epsilon_negative = frame.epsilon_negative
        epsilon_positive = frame.epsilon_positive
        if previous_axis is not None and float(np.dot(axis, previous_axis)) < 0.0:
            axis = -axis
            epsilon_negative, epsilon_positive = (
                -epsilon_positive,
                -epsilon_negative,
            )
        aligned[name] = SamSignedFeatureFrame(
            axis=np.ascontiguousarray(axis),
            epsilon_negative=epsilon_negative,
            epsilon_positive=epsilon_positive,
        )
        previous_axis = axis
    return aligned


def load_da3_seed_replay(path: Path) -> DA3SeedReplay:
    """Load accepted DA3 seed centers and audit their past-only birth metadata."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    sidecar = payload.get("seed_sidecar") if isinstance(payload, dict) else None
    if not isinstance(sidecar, dict):
        raise ValueError("DA3 checkpoint must contain a seed_sidecar mapping")
    xyz_tensor = sidecar.get("xyz")
    log_scaling_tensor = sidecar.get("log_scaling")
    metadata = sidecar.get("metadata")
    if (
        not isinstance(xyz_tensor, torch.Tensor)
        or xyz_tensor.ndim != 2
        or xyz_tensor.shape[1] != 3
    ):
        raise ValueError("DA3 seed xyz must have shape [N,3]")
    if not isinstance(metadata, list) or len(metadata) != int(xyz_tensor.shape[0]):
        raise ValueError("DA3 seed metadata must align with xyz rows")
    birth_global: list[int] = []
    birth_name: list[str] = []
    source_sign: list[str] = []
    for index, row in enumerate(metadata):
        if not isinstance(row, dict):
            raise ValueError(f"DA3 seed metadata row {index} must be a mapping")
        global_index = row.get("frame_global")
        local_index = row.get("frame_local")
        frame_name = row.get("frame_name")
        causal_window_global = row.get("causal_window_global")
        causal_window_local = row.get("causal_window")
        causal_window = (
            causal_window_global
            if causal_window_global is not None
            else causal_window_local
        )
        if (
            isinstance(global_index, bool)
            or not isinstance(global_index, int)
            or isinstance(local_index, bool)
            or not isinstance(local_index, int)
            or not isinstance(frame_name, str)
            or not frame_name
            or not isinstance(causal_window, list)
            or not causal_window
        ):
            raise ValueError(f"DA3 seed metadata row {index} is incomplete")
        causal_limit = (
            int(global_index)
            if causal_window_global is not None
            else int(local_index)
        )
        if max(int(value) for value in causal_window) > causal_limit:
            raise ValueError(f"DA3 seed metadata row {index} accesses a future view")
        birth_global.append(int(global_index))
        birth_name.append(frame_name)
        sign = str(row.get("source_sign", "+"))
        if sign not in {"+", "-", "depth_positive"}:
            raise ValueError(f"DA3 seed metadata row {index} has invalid source_sign")
        source_sign.append(sign)
    xyz = xyz_tensor.detach().float().cpu().numpy()
    if not np.isfinite(xyz).all():
        raise ValueError("DA3 seed xyz must be finite")
    if log_scaling_tensor is None:
        # Schema-v1 visualization artifacts did not retain the footprint that
        # was already computed at birth.  Keep them viewable with the original
        # conservative 1 cm isotropic fallback; schema-v2 artifacts preserve
        # the exact DA3 camera-depth footprint.
        log_scaling = np.full_like(xyz, math.log(0.01), dtype=np.float32)
    else:
        if (
            not isinstance(log_scaling_tensor, torch.Tensor)
            or tuple(log_scaling_tensor.shape) != tuple(xyz_tensor.shape)
        ):
            raise ValueError("DA3 seed log_scaling must have shape [N,3]")
        log_scaling = log_scaling_tensor.detach().float().cpu().numpy()
        if not np.isfinite(log_scaling).all():
            raise ValueError("DA3 seed log_scaling must be finite")
    return DA3SeedReplay(
        xyz=np.ascontiguousarray(xyz),
        birth_global=np.asarray(birth_global, dtype=np.int64),
        birth_name=tuple(birth_name),
        log_scaling=np.ascontiguousarray(log_scaling),
        source_sign=tuple(source_sign),
    )


def load_da3_metric_depth_replay(path: Path) -> DA3MetricDepthReplay | None:
    """Load the DA3Metric model settings and causal cache locations."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("DA3 checkpoint must be a mapping")
    configuration = payload.get("configuration")
    if not isinstance(configuration, dict):
        return None
    if configuration.get("depth_scale_source") not in {
        "da3metric_reference_bank_fixed",
        "da3metric_reference_render",
    }:
        return None
    configured_root = configuration.get("metric_cache_root")
    cache_root = (
        Path(configured_root)
        if configured_root not in (None, "None")
        else path.parent / "da3metric_cache"
    )
    if not cache_root.is_dir():
        raise FileNotFoundError(f"DA3Metric depth cache not found: {cache_root}")
    return DA3MetricDepthReplay(
        cache_root=cache_root,
        model_name=str(
            configuration.get("metric_model", "depth-anything/DA3METRIC-LARGE")
        ),
        process_res=int(configuration.get("process_res", 504)),
    )


def causal_training_view_index(
    timestamp: int,
    update_index: int,
    *,
    seed: int,
    current_probability: float = 0.33,
) -> tuple[int, bool]:
    """Sample one observed view with O-SCD's 0.33-current branch.

    The non-current branch samples uniformly from every index in ``[0, t]``;
    consequently it may independently draw the latest frame too.  The boolean
    reports the explicit 0.33 branch rather than whether both paths happened
    to return the same index.
    """

    if timestamp < 0 or update_index < 0:
        raise ValueError("timestamp and update_index must be nonnegative")
    if not 0.0 <= float(current_probability) <= 1.0:
        raise ValueError("current_probability must lie in [0,1]")
    rng = np.random.default_rng(
        int(seed) + 1_000_003 * int(timestamp) + 1009 * int(update_index)
    )
    latest_branch = float(rng.random()) <= float(current_probability)
    if latest_branch:
        return int(timestamp), True
    return int(rng.integers(0, int(timestamp) + 1)), False


def representation_dc_target(
    unit_cue_target: torch.Tensor,
    *,
    amplitude: float,
) -> torch.Tensor:
    """Scale normalized Q for the representation loss without clipping it.

    Detector evidence always consumes the normalized unit cue.  This separate
    amplitude switch makes it possible to test the original O-SCD 0..2 DC
    target while leaving detector observations and seed birth unchanged.
    """

    if unit_cue_target.ndim != 3 or unit_cue_target.shape[0] != 1:
        raise ValueError("unit_cue_target must have shape [1,H,W]")
    if not torch.is_floating_point(unit_cue_target):
        raise TypeError("unit_cue_target must be floating-point")
    if not math.isfinite(float(amplitude)) or float(amplitude) <= 0.0:
        raise ValueError("amplitude must be finite and positive")
    return unit_cue_target * float(amplitude)


def representation_geometry_target(
    unit_cue_target: torch.Tensor,
    *,
    amplitude: float,
) -> torch.Tensor:
    """Scale normalized Q for geometry supervision without clipping it."""

    return representation_dc_target(unit_cue_target, amplitude=amplitude)


def part19_da3_detector_cue(view: Any) -> torch.Tensor:
    """Return untouched cached P+S for explicit historical Part19 reproduction.

    The viewer may replace ``candidate_map`` with a fused and learned-sigmoid
    cue for representation supervision. Part19 keeps the DA3 lifecycle
    detector independent of that path: it consumes the original cached O-SCD
    P+S cue exactly once, before optimization.
    """

    cue = getattr(view, "cached_sum_candidate_map", None)
    if cue is None:
        cue = getattr(view, "pre_remap_candidate_map", None)
    if cue is None:
        cue = view.candidate_map
    if not isinstance(cue, torch.Tensor):
        raise TypeError("Part19 DA3 detector cue must be a tensor")
    if cue.numel() == 0 or not bool(torch.isfinite(cue).all()):
        raise ValueError("Part19 DA3 detector cue must be nonempty and finite")
    return cue


def da3_seed_detector_eligible_rows(
    birth_global: Sequence[int],
    *,
    timestamp: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return seeds born strictly before the current detector observation.

    A DA3 seed is a geometry proposal made from its birth-frame observation.
    Reusing that same observation as BF evidence would validate a selected
    proposal with the data that selected it.  The strict inequality therefore
    makes the birth frame proposal-only and starts detector evidence on the
    next newly observed frame.
    """

    births = torch.as_tensor(birth_global, device=device, dtype=torch.long)
    if births.ndim != 1:
        raise ValueError("DA3 seed birth indices must be one-dimensional")
    if bool((births > int(timestamp)).any()):
        raise ValueError("future DA3 seed proposals cannot enter the detector")
    return torch.nonzero(births < int(timestamp), as_tuple=False).flatten()


@torch.no_grad()
def constrain_da3_seed_geometry_(
    model: Any,
    active_rows: torch.Tensor,
    *,
    max_displacement_ratio: float,
    min_scale_ratio: float,
    max_scale_ratio: float,
    min_opacity: float,
    max_opacity: float,
) -> dict[str, torch.Tensor]:
    """Keep online DA3 geometry inside its birth-relative trust region."""

    if active_rows.dtype != torch.bool or active_rows.ndim != 1:
        raise ValueError("active_rows must be boolean [N]")
    if active_rows.shape[0] != model.num_gaussians:
        raise ValueError("active_rows must align with the DA3 seed bank")
    if not math.isfinite(float(max_displacement_ratio)) or max_displacement_ratio <= 0:
        raise ValueError("max_displacement_ratio must be finite and positive")
    if not 0.0 < float(min_scale_ratio) <= float(max_scale_ratio):
        raise ValueError("scale ratios must satisfy 0 < min <= max")
    if not 0.0 < float(min_opacity) < float(max_opacity) < 1.0:
        raise ValueError("opacity bounds must satisfy 0 < min < max < 1")

    device = model._xyz.device
    selected = active_rows.to(device=device)
    empty = torch.zeros_like(selected)
    if not bool(selected.any()):
        return {
            "xyz": empty.clone(),
            "scaling": empty.clone(),
            "opacity": empty.clone(),
            "rotation": empty.clone(),
        }

    changed: dict[str, torch.Tensor] = {}
    epsilon = torch.finfo(model._xyz.dtype).eps

    displacement = model._xyz - model.root_anchor_xyz
    distance = torch.linalg.vector_norm(displacement, dim=1)
    max_distance = (
        float(max_displacement_ratio) * model.root_anchor_scale
    ).clamp_min(epsilon)
    invalid_xyz = ~torch.isfinite(model._xyz).all(dim=1)
    clipped_xyz = selected & (invalid_xyz | (distance > max_distance))
    if bool(clipped_xyz.any()):
        safe_direction = displacement / distance.clamp_min(epsilon)[:, None]
        bounded = model.root_anchor_xyz + safe_direction * max_distance[:, None]
        bounded[invalid_xyz] = model.root_anchor_xyz[invalid_xyz]
        model._xyz[clipped_xyz] = bounded[clipped_xyz]
    changed["xyz"] = clipped_xyz

    root_scale = model.root_anchor_scale[:, None]
    scale = model.get_scaling
    bounded_scale = scale.clamp(
        min=float(min_scale_ratio) * root_scale,
        max=float(max_scale_ratio) * root_scale,
    )
    invalid_scale = ~torch.isfinite(scale).all(dim=1)
    clipped_scale = selected & (
        invalid_scale | (bounded_scale != scale).any(dim=1)
    )
    if bool(clipped_scale.any()):
        bounded_scale[invalid_scale] = root_scale[invalid_scale]
        model._scaling[clipped_scale] = torch.log(
            bounded_scale[clipped_scale].clamp_min(epsilon)
        )
    changed["scaling"] = clipped_scale

    opacity = model.get_opacity
    bounded_opacity = opacity.clamp(float(min_opacity), float(max_opacity))
    invalid_opacity = ~torch.isfinite(opacity).all(dim=1)
    clipped_opacity = selected & (
        invalid_opacity | (bounded_opacity != opacity).any(dim=1)
    )
    if bool(clipped_opacity.any()):
        bounded_opacity[invalid_opacity] = float(min_opacity)
        model._opacity[clipped_opacity] = model.inverse_opacity_activation(
            bounded_opacity[clipped_opacity]
        )
    changed["opacity"] = clipped_opacity

    rotation = model._rotation
    norm = torch.linalg.vector_norm(rotation, dim=1, keepdim=True)
    invalid_rotation = (~torch.isfinite(rotation).all(dim=1)) | (
        norm[:, 0] <= epsilon
    )
    normalized = rotation / norm.clamp_min(epsilon)
    normalized[invalid_rotation] = 0.0
    normalized[invalid_rotation, 0] = 1.0
    model._rotation[selected] = normalized[selected]
    changed["rotation"] = selected & invalid_rotation
    return changed


@torch.no_grad()
def constrain_base_representation_geometry_(
    *,
    xyz: torch.Tensor,
    scaling: torch.Tensor,
    rotation: torch.Tensor,
    anchor_xyz: torch.Tensor,
    anchor_scaling: torch.Tensor,
    selected_rows: torch.Tensor,
    max_displacement_ratio: float,
    min_scale_ratio: float,
    max_scale_ratio: float,
) -> dict[str, torch.Tensor]:
    """Bound base-only representation geometry around immutable reference rows."""

    count = int(xyz.shape[0])
    if xyz.shape != anchor_xyz.shape or scaling.shape != anchor_scaling.shape:
        raise ValueError("base geometry and immutable anchors must have equal shapes")
    if rotation.shape[0] != count or selected_rows.shape != (count,):
        raise ValueError("base geometry selection must align with row count")
    if selected_rows.dtype != torch.bool:
        raise TypeError("selected_rows must be boolean")
    selected = selected_rows.to(device=xyz.device)
    empty = torch.zeros_like(selected)
    if not bool(selected.any()):
        return {"xyz": empty.clone(), "scaling": empty.clone(), "rotation": empty}

    anchor_scale = torch.exp(anchor_scaling).amax(dim=1).clamp_min(
        torch.finfo(xyz.dtype).eps
    )
    delta = xyz - anchor_xyz
    distance = torch.linalg.vector_norm(delta, dim=1)
    radius = float(max_displacement_ratio) * anchor_scale
    invalid_xyz = ~torch.isfinite(xyz).all(dim=1)
    clipped_xyz = selected & (invalid_xyz | (distance > radius))
    if bool(clipped_xyz.any()):
        direction = delta / distance.clamp_min(torch.finfo(xyz.dtype).eps)[:, None]
        bounded = anchor_xyz + direction * radius[:, None]
        bounded[invalid_xyz] = anchor_xyz[invalid_xyz]
        xyz[clipped_xyz] = bounded[clipped_xyz]

    lower = anchor_scaling + math.log(float(min_scale_ratio))
    upper = anchor_scaling + math.log(float(max_scale_ratio))
    bounded_scaling = torch.maximum(torch.minimum(scaling, upper), lower)
    invalid_scaling = ~torch.isfinite(scaling).all(dim=1)
    clipped_scaling = selected & (
        invalid_scaling | (bounded_scaling != scaling).any(dim=1)
    )
    if bool(clipped_scaling.any()):
        bounded_scaling[invalid_scaling] = anchor_scaling[invalid_scaling]
        scaling[clipped_scaling] = bounded_scaling[clipped_scaling]

    norm = torch.linalg.vector_norm(rotation, dim=1, keepdim=True)
    invalid_rotation = (~torch.isfinite(rotation).all(dim=1)) | (
        norm[:, 0] <= torch.finfo(rotation.dtype).eps
    )
    normalized = rotation / norm.clamp_min(torch.finfo(rotation.dtype).eps)
    normalized[invalid_rotation] = 0.0
    normalized[invalid_rotation, 0] = 1.0
    rotation[selected] = normalized[selected]
    return {
        "xyz": clipped_xyz,
        "scaling": clipped_scaling,
        "rotation": selected & invalid_rotation,
    }



def overlay_da3_seed_centers(
    rgb: np.ndarray,
    seeds: DA3SeedReplay,
    *,
    global_index: int,
    camera: dict[str, Any],
    marker_radius: int = 2,
) -> tuple[np.ndarray, DA3SeedProjectionStats]:
    """Overlay causally available accepted seed centers on the current RGB."""

    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("rgb must be uint8 with shape [H,W,3]")
    if marker_radius < 1:
        raise ValueError("marker_radius must be positive")
    available = seeds.birth_global <= int(global_index)
    if not bool(available.any()):
        return image.copy(), DA3SeedProjectionStats()

    xyz = seeds.xyz[available].astype(np.float64, copy=False)
    birth = seeds.birth_global[available]
    homogeneous = np.concatenate(
        (xyz, np.ones((xyz.shape[0], 1), dtype=np.float64)), axis=1
    )
    w2c = camera_json_to_w2c(camera)
    camera_xyz = homogeneous @ w2c.T
    depth = camera_xyz[:, 2]
    height, width = image.shape[:2]
    source_width = float(camera["width"])
    source_height = float(camera["height"])
    fx = float(camera["fx"]) * width / source_width
    fy = float(camera["fy"]) * height / source_height
    x = fx * camera_xyz[:, 0] / np.where(depth > 1.0e-9, depth, 1.0)
    y = fy * camera_xyz[:, 1] / np.where(depth > 1.0e-9, depth, 1.0)
    x += width / 2.0
    y += height / 2.0
    visible = (
        (depth > 1.0e-9)
        & (x >= 0.0)
        & (x < width)
        & (y >= 0.0)
        & (y < height)
    )
    born_now = birth == int(global_index)

    canvas = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(canvas, "RGBA")
    prior_rows = np.flatnonzero(visible & ~born_now)
    current_rows = np.flatnonzero(visible & born_now)
    for row in prior_rows:
        px, py = float(x[row]), float(y[row])
        draw.ellipse(
            (px - marker_radius, py - marker_radius, px + marker_radius, py + marker_radius),
            fill=(40, 220, 80, 105),
            outline=(40, 255, 90, 210),
            width=1,
        )
    current_radius = marker_radius + 2
    for row in current_rows:
        px, py = float(x[row]), float(y[row])
        draw.ellipse(
            (
                px - current_radius,
                py - current_radius,
                px + current_radius,
                py + current_radius,
            ),
            fill=(255, 40, 30, 150),
            outline=(255, 255, 255, 255),
            width=1,
        )
    stats = DA3SeedProjectionStats(
        accepted_so_far=int(available.sum()),
        visible=int(visible.sum()),
        born_now=int(born_now.sum()),
        born_now_visible=int((visible & born_now).sum()),
    )
    return np.asarray(canvas, dtype=np.uint8).copy(), stats


def build_binary_change_gt_index(source_path: Path, records: Sequence[Any]) -> dict[str, Path]:
    """Index evaluation-only union masks without inventing NEW/REMOVE labels."""
    paths = {str(record.name): source_path / "gt_mask" / f"{Path(record.name).stem}.png" for record in records}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def build_add_remove_gt_index(
    source_path: Path,
    records: Sequence[Any],
    *,
    object_gt_root: Path | None = None,
) -> dict[str, AddRemoveMaskPaths]:
    """Index per-object GT without loading any pixels into the detector path."""

    root = source_path.parent if object_gt_root is None else object_gt_root
    segment_names = tuple(dict.fromkeys(str(record.segment_name) for record in records))
    mutable: dict[str, dict[str, list[Path]]] = {}
    for segment_name in segment_names:
        segment_root = root / segment_name
        annotation_path = segment_root / "object_change_annotations.json"
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        for obj in payload.get("objects", []):
            state = str(obj["attributes"]["change_state"])
            if state == "NEW":
                partition = "add"
            elif state == "REMOVED":
                partition = "remove"
            else:
                raise ValueError(f"unknown GT change_state: {state}")
            for mask in obj["segmentation"]["masks"]:
                frame_name = str(mask["frame_name"])
                frame_paths = mutable.setdefault(frame_name, {"add": [], "remove": []})
                frame_paths[partition].append(segment_root / str(mask["mask_path"]))

    missing = [str(record.name) for record in records if str(record.name) not in mutable]
    if missing:
        raise KeyError(f"object GT is missing {len(missing)} replay frames; first={missing[0]}")
    return {
        frame_name: AddRemoveMaskPaths(
            add=tuple(paths["add"]),
            remove=tuple(paths["remove"]),
        )
        for frame_name, paths in mutable.items()
    }


def load_union_mask(
    paths: Sequence[Path],
    *,
    width: int,
    height: int,
) -> np.ndarray:
    """Load and nearest-resize one union mask for evaluation display."""

    union = np.zeros((height, width), dtype=bool)
    for path in paths:
        with Image.open(path) as image:
            resized = image.convert("L").resize(
                (width, height), resample=Image.Resampling.NEAREST
            )
            union |= np.asarray(resized, dtype=np.uint8) > 0
    return union


class DetectorReplayLifespan:
    """Causal half-open lifespan history used by detector and replay renders.

    Learnable representation tensors remain outside this lightweight object,
    but replay still needs the complete ``[start, end)`` interval history.  A
    historical training view must select exactly the rows that were OPEN (or
    materialized but NEVER_OPEN) at that view's timestamp, rather than using
    the lifecycle state at the latest observation.
    """

    def __init__(
        self,
        count: int,
        *,
        max_states: int,
        device: torch.device,
        materialized_timestamp: int = -1,
        cache_states: bool = False,
    ) -> None:
        if count < 0 or max_states < 1:
            raise ValueError("count must be nonnegative and max_states positive")
        self.max_states = int(max_states)
        from temporal.lifespan_state_snapshots import LifespanStateSnapshots

        self.state_snapshots = LifespanStateSnapshots() if cache_states else None
        self.current_state_index = torch.full(
            (count,), -1, device=device, dtype=torch.long
        )
        self.num_states = torch.zeros(count, device=device, dtype=torch.long)
        self.open_timestamp = torch.full(
            (count,), -1, device=device, dtype=torch.long
        )
        self.materialized_timestamp = torch.full(
            (count,),
            float(materialized_timestamp),
            device=device,
            dtype=torch.float32,
        )
        shape = (count, self.max_states)
        self.state_start = torch.full(
            shape, float("inf"), device=device, dtype=torch.float32
        )
        self.state_end = torch.full(
            shape, float("inf"), device=device, dtype=torch.float32
        )
        self.state_valid = torch.zeros(shape, device=device, dtype=torch.bool)

    @property
    def count(self) -> int:
        return int(self.current_state_index.numel())

    def seal_snapshot(self, timestamp: int | float) -> None:
        if self.state_snapshots is not None:
            self.state_snapshots.seal(self, timestamp)

    def _invalidate_snapshot(self, timestamp: int | float) -> None:
        if self.state_snapshots is not None:
            self.state_snapshots.invalidate(timestamp)

    def _timestamp(self, timestamp: int | float) -> torch.Tensor:
        value = torch.as_tensor(
            timestamp,
            device=self.state_start.device,
            dtype=self.state_start.dtype,
        )
        if value.ndim != 0 or not bool(torch.isfinite(value)):
            raise TypeError("timestamp must be a finite scalar")
        return value

    @torch.no_grad()
    def append_rows(
        self,
        count: int,
        *,
        materialized_timestamp: int | float,
    ) -> torch.Tensor:
        """Append causally materialized rows with no allocated lifespan yet."""

        if isinstance(count, bool) or int(count) < 0:
            raise ValueError("count must be a nonnegative integer")
        count = int(count)
        timestamp = self._timestamp(materialized_timestamp)
        first = self.count
        if count == 0:
            return torch.empty(
                0, device=self.current_state_index.device, dtype=torch.long
            )
        device = self.current_state_index.device
        self.current_state_index = torch.cat(
            (
                self.current_state_index,
                torch.full((count,), -1, device=device, dtype=torch.long),
            )
        )
        self.num_states = torch.cat(
            (self.num_states, torch.zeros(count, device=device, dtype=torch.long))
        )
        self.open_timestamp = torch.cat(
            (
                self.open_timestamp,
                torch.full((count,), -1, device=device, dtype=torch.long),
            )
        )
        self.materialized_timestamp = torch.cat(
            (
                self.materialized_timestamp,
                timestamp.expand(count).clone(),
            )
        )
        interval_shape = (count, self.max_states)
        self.state_start = torch.cat(
            (
                self.state_start,
                torch.full(
                    interval_shape,
                    float("inf"),
                    device=device,
                    dtype=self.state_start.dtype,
                ),
            ),
            dim=0,
        )
        self.state_end = torch.cat(
            (
                self.state_end,
                torch.full(
                    interval_shape,
                    float("inf"),
                    device=device,
                    dtype=self.state_end.dtype,
                ),
            ),
            dim=0,
        )
        self.state_valid = torch.cat(
            (
                self.state_valid,
                torch.zeros(interval_shape, device=device, dtype=torch.bool),
            ),
            dim=0,
        )
        self._invalidate_snapshot(materialized_timestamp)
        return torch.arange(first, first + count, device=device, dtype=torch.long)

    def materialized_mask(self, timestamp: int | float) -> torch.Tensor:
        if self.state_snapshots is not None:
            return self.state_snapshots.masks(self, timestamp)[2].clone()
        current = self._timestamp(timestamp)
        return self.materialized_timestamp <= current

    def get_active_state_indices(self, timestamp: int | float) -> torch.Tensor:
        """Return the slot active at ``timestamp`` or ``-1`` for each row."""

        if self.count == 0:
            return torch.empty(
                0, device=self.current_state_index.device, dtype=torch.long
            )
        current = self._timestamp(timestamp)
        active = (
            self.state_valid
            & (self.state_start <= current)
            & (current < self.state_end)
            & self.materialized_mask(timestamp)[:, None]
        )
        if bool((active.sum(dim=1) > 1).any()):
            raise RuntimeError("lifespan intervals overlap at the replay timestamp")
        slots = torch.arange(
            self.max_states, device=active.device, dtype=torch.long
        ).expand_as(active)
        return torch.where(active, slots, torch.full_like(slots, -1)).amax(dim=1)

    def active_mask(self, timestamp: int | float) -> torch.Tensor:
        if self.state_snapshots is not None:
            return self.state_snapshots.masks(self, timestamp)[0].clone()
        return self.get_active_state_indices(timestamp) >= 0

    def ever_opened_mask(self, timestamp: int | float) -> torch.Tensor:
        if self.state_snapshots is not None:
            _, never, materialized = self.state_snapshots.masks(self, timestamp)
            return materialized & ~never
        if self.count == 0:
            return torch.empty(
                0, device=self.current_state_index.device, dtype=torch.bool
            )
        current = self._timestamp(timestamp)
        return self.materialized_mask(timestamp) & (
            self.state_valid & (self.state_start <= current)
        ).any(dim=1)

    def never_open_mask(self, timestamp: int | float) -> torch.Tensor:
        if self.state_snapshots is not None:
            return self.state_snapshots.masks(self, timestamp)[1].clone()
        materialized = self.materialized_mask(timestamp)
        return materialized & ~self.ever_opened_mask(timestamp)

    def closed_mask(self, timestamp: int | float) -> torch.Tensor:
        if self.state_snapshots is not None:
            active, never, materialized = self.state_snapshots.masks(self, timestamp)
            return materialized & ~(active | never)
        materialized = self.materialized_mask(timestamp)
        ever_opened = self.ever_opened_mask(timestamp)
        return materialized & ever_opened & ~self.active_mask(timestamp)

    def _rows(self, row_mask_or_indices: Any) -> torch.Tensor:
        rows = torch.as_tensor(row_mask_or_indices, device=self.current_state_index.device)
        if rows.dtype == torch.bool:
            if rows.shape != self.current_state_index.shape:
                raise ValueError("row mask must have shape [N]")
            rows = torch.nonzero(rows, as_tuple=False).flatten()
        else:
            rows = rows.to(dtype=torch.long).flatten()
        if rows.numel() and bool(
            ((rows < 0) | (rows >= self.current_state_index.numel())).any()
        ):
            raise IndexError("row index out of range")
        return rows.unique(sorted=True)

    @torch.no_grad()
    def open_rows(
        self,
        row_mask_or_indices: Any,
        timestamp: int,
        initialization: str = "preserve",
        optimizer: Any | None = None,
    ) -> torch.Tensor:
        del optimizer
        if initialization not in {"zero", "preserve"}:
            raise ValueError("initialization must be zero|preserve")
        rows = self._rows(row_mask_or_indices)
        if bool((self.current_state_index[rows] >= 0).any()):
            raise RuntimeError("cannot OPEN an already OPEN Gaussian")
        current = self._timestamp(timestamp)
        if bool((self.materialized_timestamp[rows] > current).any()):
            raise RuntimeError("cannot OPEN a row before it is materialized")
        unused = ~self.state_valid[rows]
        if bool((~unused.any(dim=1)).any()):
            raise RuntimeError("detector replay lifespan capacity exceeded")
        slots = unused.to(torch.int8).argmax(dim=1).long()
        self.state_start[rows, slots] = current
        self.state_end[rows, slots] = float("inf")
        self.state_valid[rows, slots] = True
        self.current_state_index[rows] = slots
        self.num_states[rows] += 1
        self.open_timestamp[rows] = int(timestamp)
        if rows.numel():
            self._invalidate_snapshot(timestamp)
        return slots.clone()

    @torch.no_grad()
    def close_rows(self, row_mask_or_indices: Any, timestamp: int) -> torch.Tensor:
        rows = self._rows(row_mask_or_indices)
        slots = self.current_state_index[rows]
        active = slots >= 0
        rows = rows[active]
        slots = slots[active]
        if rows.numel() and bool((int(timestamp) <= self.open_timestamp[rows]).any()):
            raise RuntimeError("CLOSE timestamp must be after OPEN")
        current = self._timestamp(timestamp)
        self.state_end[rows, slots] = current
        self.current_state_index[rows] = -1
        self.open_timestamp[rows] = -1
        if rows.numel():
            self._invalidate_snapshot(timestamp)
        return slots.clone()

    def validate_lifecycle(self) -> bool:
        shape = (self.count, self.max_states)
        for name in ("state_start", "state_end", "state_valid"):
            if tuple(getattr(self, name).shape) != shape:
                raise RuntimeError(f"{name} shape mismatch")
        for name in (
            "num_states",
            "current_state_index",
            "open_timestamp",
            "materialized_timestamp",
        ):
            if tuple(getattr(self, name).shape) != (self.count,):
                raise RuntimeError(f"{name} shape mismatch")
        if not torch.equal(
            self.num_states, self.state_valid.sum(dim=1, dtype=torch.long)
        ):
            raise RuntimeError("num_states does not match valid interval count")
        valid = self.state_valid
        if bool((valid & ~(self.state_start < self.state_end)).any()):
            raise RuntimeError("valid lifespan intervals must satisfy start < end")
        if bool(
            (
                valid
                & (self.state_start < self.materialized_timestamp[:, None])
            ).any()
        ):
            raise RuntimeError("lifespan starts before row materialization")
        open_intervals = valid & torch.isposinf(self.state_end)
        if bool((open_intervals.sum(dim=1) > 1).any()):
            raise RuntimeError("each row may have at most one OPEN interval")
        has_open = open_intervals.any(dim=1)
        if not torch.equal(self.current_state_index >= 0, has_open):
            raise RuntimeError("current_state_index disagrees with OPEN intervals")
        rows = torch.arange(self.count, device=self.current_state_index.device)
        if bool(has_open.any()):
            slots = self.current_state_index[has_open]
            if not bool(open_intervals[rows[has_open], slots].all()):
                raise RuntimeError("current_state_index does not select the OPEN slot")
        return True


def _rgb_u8(image: torch.Tensor) -> np.ndarray:
    array = (
        image.detach()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .contiguous()
        .cpu()
        .numpy()
    )
    return np.ascontiguousarray(np.rint(array * 255.0).astype(np.uint8))


def _score_heatmap_image(value: torch.Tensor) -> np.ndarray:
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2:
        raise ValueError("heatmap input must have shape [H,W] or [1,H,W]")
    flat = value.detach().float().reshape(-1)
    colors = black_green_yellow_red_heatmap(flat).reshape(*value.shape, 3)
    return np.ascontiguousarray(
        np.rint(colors.cpu().numpy() * 255.0).astype(np.uint8)
    )


def binary_change_mask_image(
    value: torch.Tensor, *, threshold: float = 0.5
) -> np.ndarray:
    """Convert a raw rendered change score to a black/white prediction mask."""

    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2:
        raise ValueError("binary-mask input must have shape [H,W] or [1,H,W]")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("binary-mask input must be finite")
    if not math.isfinite(float(threshold)) or not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("binary-mask threshold must lie in [0,1]")
    mask = value.detach() >= float(threshold)
    rgb = mask[..., None].expand(*mask.shape, 3).to(dtype=torch.uint8) * 255
    return np.ascontiguousarray(rgb.cpu().numpy())


def q_weighted_signed_sam_feature_diff_image(
    score: torch.Tensor,
    learned_q: torch.Tensor,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, SamSignedFeatureStats]:
    """Render ``upsample(normalized signed SAM diff) * learned Q``.

    Positive values are red, negative values are blue, and zero is black. The
    signed SAM score is normalized once on its native 64x64 grid before
    bilinear upsampling; unlike the historical panel, no magnitude quantile or
    hard Q support threshold is applied.
    """

    weighted, normalization_scale = q_weighted_upsampled_signed_sam_score(
        score,
        learned_q,
        height=height,
        width=width,
    )
    positive, negative = panel7_visible_signed_support(weighted)
    red = weighted.clamp(min=0.0)
    blue = (-weighted).clamp(min=0.0)
    image = torch.stack((red, torch.zeros_like(weighted), blue), dim=-1)
    image = np.uint8(
        np.clip(np.rint(image.cpu().numpy() * 255.0), 0.0, 255.0)
    )
    stats = SamSignedFeatureStats(
        positive=int(positive.sum().item()),
        negative=int(negative.sum().item()),
        neutral=int((~positive & ~negative).sum().item()),
        normalization_scale=normalization_scale,
    )
    return np.ascontiguousarray(image), stats


def signed_reference_da3_depth_difference_image(
    reference_depth: torch.Tensor,
    online_depth: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, DepthDifferenceStats]:
    """Render ``reference depth - online depth`` in signed red/blue.

    Red means the online surface lies in front of the reference surface; blue
    means it lies behind. Intensity is normalized by the valid residual's 95th
    absolute percentile so isolated outliers do not flatten the map.
    """

    reference = reference_depth.detach().float().cpu().squeeze()
    online = online_depth.detach().float().cpu().squeeze()
    valid = valid_mask.detach().bool().cpu().squeeze()
    if reference.ndim != 2 or online.shape != reference.shape:
        raise ValueError("reference and online depth must share shape [H,W]")
    if valid.shape != reference.shape:
        raise ValueError("depth validity mask must share shape [H,W]")
    if width < 1 or height < 1:
        raise ValueError("depth-difference image dimensions must be positive")

    valid &= torch.isfinite(reference) & torch.isfinite(online)
    valid &= (reference > 0.0) & (online > 0.0)
    difference = reference - online
    absolute = difference[valid].abs()
    if absolute.numel():
        median_absolute_difference = float(absolute.median().item())
        normalization_scale = float(torch.quantile(absolute, 0.95).item())
    else:
        median_absolute_difference = 0.0
        normalization_scale = 0.0
    if normalization_scale > 0.0:
        normalized = (difference / normalization_scale).clamp(-1.0, 1.0)
    else:
        normalized = torch.zeros_like(difference)
    normalized[~valid] = 0.0
    normalized_full = F.interpolate(
        normalized[None, None],
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    valid_full = F.interpolate(
        valid.float()[None, None],
        size=(height, width),
        mode="nearest",
    )[0, 0].bool()
    normalized_full[~valid_full] = 0.0
    red = normalized_full.clamp(min=0.0)
    blue = (-normalized_full).clamp(min=0.0)
    image = torch.stack((red, torch.zeros_like(red), blue), dim=-1)
    image = np.uint8(
        np.clip(np.rint(image.numpy() * 255.0), 0.0, 255.0)
    )
    stats = DepthDifferenceStats(
        valid_pixels=int(valid.sum().item()),
        median_absolute_difference=median_absolute_difference,
        normalization_scale=normalization_scale,
    )
    return np.ascontiguousarray(image), stats


def rendered_gs_online_da3_depth_difference_image(
    online_da3_depth: torch.Tensor,
    rendered_gs_depth: torch.Tensor,
    alignment_mask: torch.Tensor,
    sam_support_mask: torch.Tensor,
    *,
    width: int,
    height: int,
    depth_difference_threshold: float = DEPTH_DIFFERENCE_SUPPORT_THRESHOLD,
) -> tuple[np.ndarray, DepthDifferenceStats]:
    """Show signed depth residual only inside strong panel-7 support."""

    online_fit = fit_reference_depth_scale(
        online_da3_depth,
        rendered_gs_depth,
        alignment_mask,
    )
    signed_difference = rendered_gs_depth - online_da3_depth * float(
        online_fit.scale
    )
    comparison_mask = sam_support_mask.detach().bool().cpu().squeeze()
    comparison_mask &= signed_difference.detach().float().cpu().abs() > float(
        depth_difference_threshold
    )
    image, residual = signed_reference_da3_depth_difference_image(
        rendered_gs_depth,
        online_da3_depth * float(online_fit.scale),
        comparison_mask,
        width=width,
        height=height,
    )
    return image, DepthDifferenceStats(
        valid_pixels=residual.valid_pixels,
        median_absolute_difference=residual.median_absolute_difference,
        normalization_scale=residual.normalization_scale,
        alignment_scale=float(online_fit.scale),
        alignment_samples=int(online_fit.samples),
        alignment_inliers=int(online_fit.inliers),
        alignment_median_relative_error=float(
            online_fit.median_absolute_relative_error
        ),
    )


def _mask_image(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2:
        raise ValueError("mask must have shape [H,W]")
    output = np.zeros((*array.shape, 3), dtype=np.uint8)
    output[array] = color
    return output


@torch.no_grad()
def typed_change_cue_masks(
    learned_q: torch.Tensor,
    weighted_sam: torch.Tensor,
    signed_depth: torch.Tensor,
    depth_valid: torch.Tensor,
    *,
    sam_threshold: float = PANEL7_DEPTH_SUPPORT_THRESHOLD,
    depth_threshold: float = DEPTH_DIFFERENCE_SUPPORT_THRESHOLD,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Partition Q's grid by strong raw signs, before RGB quantization.

    Red is NEW (+/+), blue REMOVE (-/-), and yellow APPEARANCE/uncertain
    (conflicting, weak, or invalid evidence). Brightness remains Q, with no
    extra hard Q cutoff. Depth support is classified before nearest-neighbor
    resizing, so display interpolation cannot invent strong signed evidence.
    Shared by Panel10 and the optional NEW/base representation partition.
    It never supplies detector evidence.
    """

    cue = learned_q.detach().float().cpu()
    if cue.ndim == 3 and cue.shape[0] == 1:
        cue = cue[0]
    sam = weighted_sam.detach().float().cpu()
    depth = signed_depth.detach().float().cpu()
    valid = depth_valid.detach().cpu()
    if cue.ndim != 2 or cue.numel() == 0:
        raise ValueError("learned_q must have shape [H,W] or [1,H,W]")
    if not bool(torch.isfinite(cue).all()) or bool(((cue < 0) | (cue > 1)).any()):
        raise ValueError("learned_q must be finite and lie in [0,1]")
    if sam.shape != cue.shape:
        raise ValueError("weighted_sam must share the cue shape [H,W]")
    if depth.ndim != 2 or depth.numel() == 0:
        raise ValueError("signed_depth must have shape [h,w]")
    if valid.dtype != torch.bool or valid.shape != depth.shape:
        raise ValueError("depth_valid must be boolean and share the depth shape")
    for name, threshold in (("sam", sam_threshold), ("depth", depth_threshold)):
        if not math.isfinite(float(threshold)) or float(threshold) < 0:
            raise ValueError(f"{name} threshold must be finite and nonnegative")

    finite_depth = valid & torch.isfinite(depth)
    depth_signs = torch.stack(
        (
            finite_depth & (depth > float(depth_threshold)),
            finite_depth & (depth < -float(depth_threshold)),
        )
    )
    depth_signs = F.interpolate(
        depth_signs.float()[None], size=cue.shape, mode="nearest"
    )[0].bool()
    finite_sam = torch.isfinite(sam)
    new = finite_sam & (sam > float(sam_threshold)) & depth_signs[0]
    removed = finite_sam & (sam < -float(sam_threshold)) & depth_signs[1]
    appearance = ~(new | removed)
    return new, removed, appearance


@torch.no_grad()
def typed_change_cue_image(
    learned_q: torch.Tensor,
    weighted_sam: torch.Tensor,
    signed_depth: torch.Tensor,
    depth_valid: torch.Tensor,
    *,
    sam_threshold: float = PANEL7_DEPTH_SUPPORT_THRESHOLD,
    depth_threshold: float = DEPTH_DIFFERENCE_SUPPORT_THRESHOLD,
) -> tuple[np.ndarray, CueTypeStats]:
    """Display shared sign classes with continuous Q brightness."""

    new, removed, appearance = typed_change_cue_masks(
        learned_q, weighted_sam, signed_depth, depth_valid,
        sam_threshold=sam_threshold, depth_threshold=depth_threshold,
    )
    cue = learned_q.detach().float().cpu().reshape(new.shape)
    brightness = torch.round(cue * 255.0).to(torch.uint8)
    visible = brightness > 0
    image = torch.zeros((*cue.shape, 3), dtype=torch.uint8)
    image[..., 0] = brightness * (new | appearance)
    image[..., 1] = brightness * appearance
    image[..., 2] = brightness * removed
    stats = CueTypeStats(
        new=int((new & visible).sum().item()),
        removed=int((removed & visible).sum().item()),
        appearance=int((appearance & visible).sum().item()),
        background=int((~visible).sum().item()),
    )
    return np.ascontiguousarray(image.numpy()), stats


def compose_gt_change_image(
    add_mask: np.ndarray, remove_mask: np.ndarray
) -> np.ndarray:
    """Render the evaluation-only union of NEW and REMOVED ground truth."""

    add = np.asarray(add_mask, dtype=bool)
    remove = np.asarray(remove_mask, dtype=bool)
    if add.ndim != 2 or remove.ndim != 2 or add.shape != remove.shape:
        raise ValueError("ADD and REMOVE masks must have the same [H,W] shape")
    return _mask_image(add | remove, GT_CHANGE_COLOR)


def panel10_da3_seed_proposals(
    cue: torch.Tensor,
    new_mask: torch.Tensor,
    aligned_depth: torch.Tensor,
    reference_depth: torch.Tensor,
    depth_K: torch.Tensor,
    w2c: torch.Tensor,
    *,
    stride: int,
    maximum: int,
) -> Any:
    """Unproject current NEW pixels using the same aligned depth as Panel9.

    No replay-checkpoint seed list, GT, future view, or learned GS parameter is
    used. Native DA3 intrinsics are scaled to the full-resolution cue grid.
    """
    from temporal.depth_prior_new_seeding import (
        DepthPriorSeedConfig,
        build_depth_prior_new_seeds,
    )

    q = cue.detach().float().cpu()
    if q.ndim == 3 and q.shape[0] == 1:
        q = q[0]
    if q.ndim != 2 or new_mask.shape != q.shape or new_mask.dtype != torch.bool:
        raise ValueError("Q and boolean NEW support must share [H,W]")
    if aligned_depth.ndim != 2 or reference_depth.shape != aligned_depth.shape:
        raise ValueError("aligned and reference depths must share [h,w]")
    if depth_K.shape != (3, 3) or w2c.shape != (4, 4):
        raise ValueError("depth intrinsics/extrinsics must be [3,3]/[4,4]")
    native_h, native_w = aligned_depth.shape
    height, width = q.shape
    K = depth_K.detach().float().cpu().clone()
    K[0] *= width / native_w
    K[1] *= height / native_h
    aligned = F.interpolate(
        aligned_depth.detach().float().cpu()[None, None], size=q.shape,
        mode="nearest",
    )[0, 0]
    reference = F.interpolate(
        reference_depth.detach().float().cpu()[None, None], size=q.shape,
        mode="nearest",
    )[0, 0]
    return build_depth_prior_new_seeds(
        predicted_depth=aligned,
        confidence=q,
        reference_depth=reference,
        new_mask=new_mask.detach().cpu(),
        K=K,
        w2c=w2c,
        scale=1.0,
        config=DepthPriorSeedConfig(
            confidence_quantile=0.0, erosion_pixels=0,
            min_front_gap=DEPTH_DIFFERENCE_SUPPORT_THRESHOLD,
            min_front_gap_ratio=0.0,
            sampling_stride=int(stride), max_seeds=int(maximum),
        ),
        sampling_priority=q,
    )


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def letterbox_image(
    image: np.ndarray,
    *,
    width: int,
    height: int,
    background: tuple[int, int, int] = (18, 18, 18),
) -> np.ndarray:
    """Fit one RGB image into a box without changing its aspect ratio."""

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError("image must be uint8 RGB with shape [H,W,3]")
    if width < 1 or height < 1:
        raise ValueError("letterbox dimensions must be positive")
    source_height, source_width = array.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, min(width, int(round(source_width * scale))))
    resized_height = max(1, min(height, int(round(source_height * scale))))
    resized = Image.fromarray(array).resize(
        (resized_width, resized_height), resample=Image.Resampling.BILINEAR
    )
    canvas = Image.new("RGB", (width, height), color=background)
    left = (width - resized_width) // 2
    top = (height - resized_height) // 2
    canvas.paste(resized, (left, top))
    return np.asarray(canvas, dtype=np.uint8).copy()


def compose_horizontal_dashboard(
    panels: Sequence[tuple[str, np.ndarray]],
    *,
    width: int,
    aspect: float,
) -> np.ndarray:
    """Compose equal-width, aspect-preserving panels for one browser canvas."""

    if len(panels) < 1:
        raise ValueError("at least one dashboard panel is required")
    if width < len(panels):
        raise ValueError("dashboard width is too small for its panels")
    if not math.isfinite(float(aspect)) or float(aspect) <= 0.0:
        raise ValueError("dashboard aspect must be finite and positive")
    height = max(PANEL_LABEL_HEIGHT + 1, int(round(width / float(aspect))))
    canvas = Image.new("RGB", (width, height), color=(8, 8, 8))
    draw = ImageDraw.Draw(canvas)
    font = _font(22)
    panel_edges = np.rint(np.linspace(0, width, len(panels) + 1)).astype(int)
    for index, (label, image) in enumerate(panels):
        left = int(panel_edges[index])
        right = int(panel_edges[index + 1])
        panel_width = right - left
        content = letterbox_image(
            image,
            width=panel_width,
            height=height - PANEL_LABEL_HEIGHT,
        )
        canvas.paste(Image.fromarray(content), (left, PANEL_LABEL_HEIGHT))
        draw.rectangle((left, 0, right - 1, PANEL_LABEL_HEIGHT - 1), fill=(28, 28, 28))
        draw.text((left + 12, 9), label, font=font, fill=(245, 245, 245))
        if index:
            draw.line((left, 0, left, height), fill=(90, 90, 90), width=2)
    return np.asarray(canvas, dtype=np.uint8).copy()


def compose_two_row_dashboard(
    panels: Sequence[tuple[str, np.ndarray]],
    *,
    width: int,
    aspect: float,
) -> np.ndarray:
    """Compose panels 1-7 on top and 8-14 on the bottom browser row."""

    columns = 7
    rows = 2
    capacity = columns * rows
    if len(panels) != capacity:
        raise ValueError("two-row dashboard requires exactly 14 panels")
    if width < columns:
        raise ValueError("dashboard width is too small for seven columns")
    if not math.isfinite(float(aspect)) or float(aspect) <= 0.0:
        raise ValueError("dashboard aspect must be finite and positive")
    height = max(rows * (PANEL_LABEL_HEIGHT + 1), int(round(width / float(aspect))))
    canvas = Image.new("RGB", (width, height), color=(8, 8, 8))
    draw = ImageDraw.Draw(canvas)
    font = _font(18)
    column_edges = np.rint(np.linspace(0, width, columns + 1)).astype(int)
    row_edges = np.rint(np.linspace(0, height, rows + 1)).astype(int)
    for index, (label, image) in enumerate(panels):
        row = index // columns
        column = index % columns
        left = int(column_edges[column])
        right = int(column_edges[column + 1])
        top = int(row_edges[row])
        bottom = int(row_edges[row + 1])
        panel_width = right - left
        panel_height = bottom - top
        content = letterbox_image(
            image,
            width=panel_width,
            height=panel_height - PANEL_LABEL_HEIGHT,
        )
        canvas.paste(
            Image.fromarray(content),
            (left, top + PANEL_LABEL_HEIGHT),
        )
        draw.rectangle(
            (left, top, right - 1, top + PANEL_LABEL_HEIGHT - 1),
            fill=(28, 28, 28),
        )
        draw.text((left + 8, top + 11), label, font=font, fill=(245, 245, 245))
        if column:
            draw.line((left, top, left, bottom), fill=(90, 90, 90), width=2)
        if row:
            draw.line((left, top, right, top), fill=(90, 90, 90), width=2)
    return np.asarray(canvas, dtype=np.uint8).copy()


def format_status(
    summary: ReplayStepSummary | None,
    *,
    consumed_frames: int,
    total_frames: int,
    bayes_factor_threshold: float,
    first_open_bayes_factor_threshold: float | None = None,
) -> str:
    """Return a compact dashboard status block."""

    if summary is None:
        return (
            "### Reference initialization\n"
            "아직 detector가 cue를 소비하지 않았습니다. "
            "모든 Gaussian은 검정입니다.\n\n"
            f"- frame: `0 / {total_frames}`\n"
            f"- RESET commit: `BF ≥ {bayes_factor_threshold:g}`"
            + (f" (first OPEN only: `{first_open_bayes_factor_threshold:g}`)"
               if first_open_bayes_factor_threshold is not None else "")
        )
    boundary_line = (
        f"\n- learned cue τ / width: **{summary.cue_tau:.4f} / "
        f"{summary.cue_width:.4f}**"
        if summary.cue_tau is not None and summary.cue_width is not None
        else ""
    )
    training_line = (
        "\n- representation: "
        f"**{summary.representation_updates} updates**, "
        f"latest-0.33 branch **{summary.sampled_latest_branch}**, "
        f"sampled t **{summary.sampled_oldest_timestamp}.."
        f"{summary.sampled_newest_timestamp}**, "
        f"historical lifespan renders **"
        f"{summary.historical_lifespan_replay_updates}**, "
        f"lifespan violations **{summary.lifespan_render_violations}**, "
        f"last loss **{summary.representation_loss:.5f}**\n"
        f"- trainable base OPEN / active DA3 / visible DA3: "
        f"**{summary.trainable_open_rows:,} / "
        f"{summary.active_da3_seed_rows:,} / "
        f"{summary.visible_da3_seed_rows:,}**\n"
        f"- visible NEVER_OPEN geometry rows: "
        f"**{summary.visible_pending_da3_seed_rows:,}**\n"
        f"- DA3 proposals / accepted / learned-coverage rejected now: "
        f"**{summary.da3_seed_proposed_now:,} / "
        f"{summary.da3_seed_accepted_now:,} / "
        f"{summary.da3_seed_coverage_rejected_now:,}**\n"
        f"- DA3 OPEN / CLOSE now; NEVER_OPEN / CLOSED: "
        f"**{summary.da3_seed_opened_now:,} / "
        f"{summary.da3_seed_closed_now:,}; "
        f"{summary.da3_seed_never_open:,} / "
        f"{summary.da3_seed_closed:,}**"
        f"\n- base / NEW loss: **{summary.base_loss:.5f} / {summary.new_loss:.5f}**; "
        f"seed density children / pruned: **{summary.da3_density_children:,} / "
        f"{summary.da3_pruned_now:,}**"
        if summary.representation_updates > 0
        and summary.representation_loss is not None
        else ""
    )
    return (
        f"### Frame {consumed_frames} / {total_frames}\n"
        f"`t={summary.timestamp}` · `{summary.frame_name}`\n\n"
        f"- observed: **{summary.observed:,}**\n"
        f"- black NEVER_OPEN: **{summary.never_open:,}**\n"
        f"- yellow candidate: **{summary.uncertain:,}**\n"
        f"- green OPEN: **{summary.open:,}**\n"
        f"- red CLOSED: **{summary.closed:,}**\n"
        f"- this frame OPEN / CLOSE: **{summary.opened_now:,} / "
        f"{summary.closed_now:,}**\n"
        f"- candidate start / continue / reject / commit: "
        f"**{summary.candidate_started:,} / {summary.candidate_continued:,} / "
        f"{summary.candidate_rejected:,} / {summary.candidate_committed:,}**\n"
        f"- cue + / − pseudo-count: **{summary.positive_pseudocount_mass:.2f} / "
        f"{summary.negative_pseudocount_mass:.2f}**"
        f"{boundary_line}"
        f"{training_line}"
    )


class BayesianDetectorReplay:
    """Causal detector plus online R_change representation replay engine."""

    def __init__(self, args: argparse.Namespace) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required by the Gaussian rasterizer")
        from scene import GaussianModel
        from temporal.change_evidence import accumulate_change_evidence

        self._accumulate_change_evidence = accumulate_change_evidence
        self.args = args
        self.device = torch.device("cuda")
        self.pipe = SimpleNamespace(
            compute_cov3D_python=False,
            convert_SHs_python=False,
            debug=False,
        )
        self.evidence_background = torch.zeros(3, device=self.device)
        self.viewer_background = torch.ones(3, device=self.device)

        self.records, _ = build_causal_records(
            args.source_path, max_frames=args.max_frames
        )
        if not self.records:
            raise RuntimeError("no online frames were found")
        expected_indices = list(range(len(self.records)))
        if [int(record.global_index) for record in self.records] != expected_indices:
            raise RuntimeError("online records are not causally indexed")
        self.gt_index = (
            build_binary_change_gt_index(args.source_path, self.records)
            if getattr(args, "gt_format", "objects") == "binary"
            else build_add_remove_gt_index(
                args.source_path, self.records, object_gt_root=args.object_gt_root
            )
        )

        self.base_ply = (args.source_path / BASE_PLY_REL).resolve()
        validate_cue_cache(args.cue_cache_root, self.base_ply, args.resolution)
        self.cameras = load_fixed_camera_index(args.fixed_cameras_json)
        self.da3_seeds = (
            load_da3_seed_replay(args.da3_seed_checkpoint)
            if args.da3_seed_checkpoint is not None and not self.uses_typed_partition
            else None
        )
        self.da3_metric_depth = (
            load_da3_metric_depth_replay(args.da3_seed_checkpoint)
            if args.da3_seed_checkpoint is not None
            else None
        )
        if self.uses_typed_partition and self.da3_metric_depth is None:
            raise ValueError("panel10_new requires valid online DA3Metric cache metadata")
        self._depth_difference_cache: dict[
            int, tuple[np.ndarray, DepthDifferenceStats]
        ] = {}
        self._cue_type_cache: dict[int, tuple[np.ndarray, CueTypeStats]] = {}
        self._typed_new_mask_cache: dict[int, torch.Tensor] = {}
        self._typed_depth_cache: dict[int, tuple[torch.Tensor, ...]] = {}
        self.da3_metric_model = None
        if args.cue_remap == "learned_sigmoid":
            self.learned_boundaries, self.learned_edge_probability = (
                load_learned_sigmoid_boundaries(args.cue_boundary_json)
            )
        else:
            self.learned_boundaries = {}
            self.learned_edge_probability = 0.05

        print(f"Loading immutable reference probe: {self.base_ply}")
        if args.cue_fusion != "sum":
            # The original RGB coefficients are needed only to reconstruct P.
            # All detector/evidence renders still override color explicitly.
            self.base = GaussianModel(sh_degree=3, active_sh_degree=3)
            self.base.load_ply(str(self.base_ply))
        else:
            self.base = GaussianModel(sh_degree=3, active_sh_degree=0)
            self.base.load_ply_change(str(self.base_ply))
        for raw_name in (
            "_xyz",
            "_features_dc",
            "_features_rest",
            "_opacity",
            "_scaling",
            "_rotation",
        ):
            getattr(self.base, raw_name).requires_grad_(False)
        self.count = int(self.base.get_xyz.shape[0])
        print(f"  -> {self.count:,} reference Gaussians")

        self.sam_sign_trace = (
            load_causal_sam_sign_trace(args.sam_sign_trace_root, self.records)
            if args.sam_sign_trace_root is not None
            else {}
        )
        self.sam_model = None
        self._sam_feature_diff_cache: dict[
            int, tuple[np.ndarray, SamSignedFeatureStats]
        ] = {}
        self._sam_signed_score_cache: dict[int, torch.Tensor] = {}
        if self.sam_sign_trace:
            from transformers import Sam2Model

            print(f"Loading SAM signed-feature probe: {args.sam_model}")
            self.sam_model = (
                Sam2Model.from_pretrained(args.sam_model, local_files_only=True)
                .half()
                .cuda()
                .eval()
            )

        self.current_index = -1
        self.current_view = self._load_view(0)
        self.latest_evidence = None
        self.latest_summary: ReplayStepSummary | None = None
        self.latest_open_rows = torch.zeros(
            self.count, device=self.device, dtype=torch.bool
        )
        self.latest_close_rows = torch.zeros_like(self.latest_open_rows)
        self._reset_detector_state()
        self._initialize_representation_state()

    @property
    def uses_typed_partition(self) -> bool:
        return getattr(self.args, "training_partition", "legacy") == "panel10_new"

    @property
    def total_frames(self) -> int:
        return len(self.records)

    @property
    def has_next(self) -> bool:
        return self.current_index + 1 < self.total_frames

    def _load_view(self, index: int) -> Any:
        view = build_fixed_cue_views(
            [self.records[index]],
            self.cameras,
            self.args.cue_cache_root,
            self.args.resolution,
        )[0][0]
        cached_sum = view.candidate_map
        if self.args.cue_fusion != "sum":
            from gaussian_renderer import render

            with torch.no_grad():
                reference_rgb = render(
                    view,
                    self.base,
                    self.pipe,
                    self.evidence_background,
                )["render"]
                view.reference_rgb = reference_rgb.detach()
                pixel_terms = oscd_pixel_terms(
                    reference_rgb,
                    view.original_image,
                )
                original_pixel_cue = normalized_oscd_pixel_cue_from_terms(
                    pixel_terms,
                    l1_exponent=1.0,
                )
                if self.args.cue_fusion == "power_product":
                    components = power_product_from_cached_sum(
                        cached_sum,
                        original_pixel_cue,
                        exponent=self.args.product_exponent,
                    )
                    supervision_source = "reconstructed_2x_pixel_power_times_sam"
                else:
                    l1_powered_pixel_cue = normalized_oscd_pixel_cue_from_terms(
                        pixel_terms,
                        l1_exponent=self.args.product_exponent,
                    )
                    semantic_cue = semantic_from_cached_sum(
                        cached_sum,
                        original_pixel_cue,
                    )
                    components = PowerProductCue(
                        pixel=l1_powered_pixel_cue,
                        semantic=semantic_cue,
                        fused=fuse_power_product(
                            l1_powered_pixel_cue,
                            semantic_cue,
                            exponent=1.0,
                        ),
                    )
                    supervision_source = "reconstructed_2x_l1_power_pixel_times_sam"
            view.cached_sum_candidate_map = cached_sum
            view.original_pixel_candidate_map = original_pixel_cue
            view.pixel_candidate_map = components.pixel
            view.semantic_candidate_map = components.semantic
            view.candidate_map = components.fused
            view.training_target = components.fused
            view.support_map = components.fused
            view.supervision_source = supervision_source

        if self.args.cue_remap == "smoothstep_band":
            view.pre_remap_candidate_map = view.candidate_map
            normalized_cue = torch.clamp(
                view.candidate_map / self.args.cue_scale,
                0.0,
                1.0,
            )
            sharpened = smoothstep_soft_binarize(
                normalized_cue,
                low=self.args.soft_band_low,
                high=self.args.soft_band_high,
            )
            view.candidate_map = sharpened * self.args.cue_scale
            view.training_target = view.candidate_map
            view.support_map = view.candidate_map
            view.supervision_source += "_smoothstep_soft_binary"
        elif self.args.cue_remap == "learned_sigmoid":
            frame_name = str(self.records[index].name)
            boundary = self.learned_boundaries.get(frame_name)
            if boundary is None:
                raise KeyError(f"learned sigmoid boundary missing frame: {frame_name}")
            view.pre_remap_candidate_map = view.candidate_map
            normalized_cue = torch.clamp(
                view.candidate_map / self.args.cue_scale,
                0.0,
                1.0,
            )
            sharpened = sigmoid_soft_binarize(
                normalized_cue,
                tau=boundary.tau,
                width=boundary.width,
                edge_probability=self.learned_edge_probability,
            )
            view.candidate_map = sharpened * self.args.cue_scale
            view.training_target = view.candidate_map
            view.support_map = view.candidate_map
            view.supervision_source += "_learned_stage2_sigmoid"
            view.learned_cue_tau = boundary.tau
            view.learned_cue_width = boundary.width

        mask_paths = self.gt_index[str(self.records[index].name)]
        if getattr(self.args, "gt_format", "objects") == "binary":
            view.gt_change_mask = load_union_mask(
                (mask_paths,), width=int(view.image_width), height=int(view.image_height)
            )
            return view
        view.gt_add_mask = load_union_mask(
            mask_paths.add,
            width=int(view.image_width),
            height=int(view.image_height),
        )
        view.gt_remove_mask = load_union_mask(
            mask_paths.remove,
            width=int(view.image_width),
            height=int(view.image_height),
        )
        return view

    def _ensure_da3_metric_model(self) -> Any:
        if self.da3_metric_depth is None:
            raise RuntimeError("DA3Metric replay configuration is unavailable")
        if self.da3_metric_model is None:
            from depth_anything_3.api import DepthAnything3

            print(
                "Loading DA3Metric depth probe: "
                f"{self.da3_metric_depth.model_name}"
            )
            self.da3_metric_model = (
                DepthAnything3.from_pretrained(self.da3_metric_depth.model_name)
                .to(self.device)
                .eval()
            )
        return self.da3_metric_model

    def _reset_detector_state(self) -> None:
        from temporal.lifespan_gate_beta import (
            LifespanGateBetaConfig,
            LifespanGateBetaController,
            LifespanGateBetaFilter,
        )

        self.lifecycle = DetectorReplayLifespan(
            self.count,
            max_states=self.args.max_states,
            device=self.device,
            cache_states=(self.uses_typed_partition
                          and getattr(self.args, "lifespan_state_cache", "off") == "snapshot"),
        )
        self.tracker = LifespanGateBetaFilter(
            self.count + (int(self.args.da3_max_rows) if self.uses_typed_partition else 0),
            LifespanGateBetaConfig(
                stable_flip_prior=self.args.stable_flip_prior,
                stable_keep_prior=self.args.stable_keep_prior,
                reset_flip_prior=self.args.reset_flip_prior,
                reset_keep_prior=self.args.reset_keep_prior,
                bayes_factor_threshold=self.args.bayes_factor_threshold,
                min_evidence_mass=self.args.min_evidence_mass,
            ),
            device=self.device,
            dtype=self.base.get_xyz.dtype,
        )
        self.controller = LifespanGateBetaController(self.lifecycle)

    def _make_base_optimizer(self) -> MaskedRowAdam:
        named = {"dc": self.change_dc}
        thaw = ["dc"]
        lrs = {"dc": float(self.args.representation_dc_lr)}
        if self.args.base_geometry_scope != "frozen":
            named.update(
                {
                    "xyz": self.representation_xyz,
                    "scaling": self.representation_scaling,
                    "rotation": self.representation_rotation,
                }
            )
            thaw.extend(("xyz", "scaling", "rotation"))
            lrs.update(
                {
                    "xyz": float(self.args.representation_xyz_lr),
                    "scaling": float(self.args.representation_scale_lr),
                    "rotation": float(self.args.representation_rotation_lr),
                }
            )
        if self.args.train_never_open_base_opacity:
            named["opacity"] = self.representation_opacity
            thaw.append("opacity")
            lrs["opacity"] = float(self.args.representation_opacity_lr)
        return MaskedRowAdam(
            named,
            thaw_names=tuple(thaw),
            lrs=lrs,
            eps=1.0e-15,
        )

    def _make_seed_optimizer(self) -> MaskedRowAdam:
        return MaskedRowAdam(
            {
                "xyz": self.seed_model._xyz,
                "dc": self.seed_model.new_dc,
                "opacity": self.seed_model._opacity,
                "scaling": self.seed_model._scaling,
                "rotation": self.seed_model._rotation,
            },
            thaw_names=("xyz", "dc", "opacity", "scaling", "rotation"),
            lrs={
                "xyz": float(self.args.da3_xyz_lr),
                "dc": float(self.args.da3_dc_lr),
                "opacity": float(self.args.da3_opacity_lr),
                "scaling": float(self.args.da3_scale_lr),
                "rotation": float(self.args.da3_rotation_lr),
            },
            eps=1.0e-15,
        )

    def _initialize_representation_state(self) -> None:
        from temporal.active_new_gaussians import ActiveNewGaussianModel
        from temporal.new_seed_gaussians import NewSeedGaussianModel

        self.change_dc = nn.Parameter(torch.zeros_like(self.base._features_dc))
        self.representation_xyz = nn.Parameter(self.base._xyz.detach().clone())
        self.representation_scaling = nn.Parameter(
            self.base._scaling.detach().clone()
        )
        self.representation_rotation = nn.Parameter(
            self.base._rotation.detach().clone()
        )
        self.representation_opacity = nn.Parameter(
            self.base._opacity.detach().clone()
        )
        self.base_optimizer = self._make_base_optimizer()
        self.seed_model = ActiveNewGaussianModel(
            sh_degree=0,
            device=self.device,
            dtype=self.base.get_xyz.dtype,
        )
        self.seed_detector_probe = NewSeedGaussianModel(
            sh_degree=0,
            device=self.device,
            dtype=self.base.get_xyz.dtype,
        )
        self.seed_detector_probe.seed_dc.requires_grad_(False)
        self.seed_lifecycle = DetectorReplayLifespan(
            0,
            max_states=self.args.max_states,
            device=self.device,
            cache_states=(self.uses_typed_partition
                          and getattr(self.args, "lifespan_state_cache", "off") == "snapshot"),
        )
        self.da3_new_sign: str | None = self.args.sam_new_sign
        self.seed_optimizer = self._make_seed_optimizer()
        self.seed_tracker = self._make_seed_tracker()
        self.accepted_da3_source_rows: list[int] = []
        self.accepted_da3_birth_global: list[int] = []
        self.seed_geometry_update_counts = torch.empty(
            0, device=self.device, dtype=torch.long
        )
        self.seed_retired = torch.empty(0, device=self.device, dtype=torch.bool)
        self.seed_last_visible_rows = torch.empty(0, device=self.device, dtype=torch.bool)
        self.representation_replay: list[RepresentationReplayFrame] = []

    def _make_seed_tracker(self) -> Any | None:
        # Current typed training uses one BF bank: [base | seed archive capacity].
        # A separate seed bank exists only for the historical legacy variant.
        if self.uses_typed_partition or self.da3_seeds is None or len(self.da3_seeds.xyz) == 0:
            return None
        from temporal.lifespan_gate_beta import (
            LifespanGateBetaConfig,
            LifespanGateBetaFilter,
        )

        return LifespanGateBetaFilter(
            len(self.da3_seeds.xyz),
            LifespanGateBetaConfig(
                stable_flip_prior=self.args.stable_flip_prior,
                stable_keep_prior=self.args.stable_keep_prior,
                reset_flip_prior=self.args.reset_flip_prior,
                reset_keep_prior=self.args.reset_keep_prior,
                bayes_factor_threshold=self.args.bayes_factor_threshold,
                min_evidence_mass=self.args.min_evidence_mass,
            ),
            device=self.device,
            dtype=self.base.get_xyz.dtype,
        )

    @torch.no_grad()
    def _reset_representation_state(self) -> None:
        from temporal.active_new_gaussians import ActiveNewGaussianModel
        from temporal.new_seed_gaussians import NewSeedGaussianModel

        self.change_dc.zero_()
        self.representation_xyz.copy_(self.base._xyz.detach())
        self.representation_scaling.copy_(self.base._scaling.detach())
        self.representation_rotation.copy_(self.base._rotation.detach())
        self.representation_opacity.copy_(self.base._opacity.detach())
        self.base_optimizer = self._make_base_optimizer()
        self.seed_model = ActiveNewGaussianModel(
            sh_degree=0,
            device=self.device,
            dtype=self.base.get_xyz.dtype,
        )
        self.seed_detector_probe = NewSeedGaussianModel(
            sh_degree=0,
            device=self.device,
            dtype=self.base.get_xyz.dtype,
        )
        self.seed_detector_probe.seed_dc.requires_grad_(False)
        self.seed_lifecycle = DetectorReplayLifespan(
            0,
            max_states=self.args.max_states,
            device=self.device,
            cache_states=(self.uses_typed_partition
                          and getattr(self.args, "lifespan_state_cache", "off") == "snapshot"),
        )
        self.seed_optimizer = self._make_seed_optimizer()
        self.seed_tracker = self._make_seed_tracker()
        self.accepted_da3_source_rows.clear()
        self.accepted_da3_birth_global.clear()
        self.seed_geometry_update_counts = torch.empty(
            0, device=self.device, dtype=torch.long
        )
        self.seed_retired = torch.empty(0, device=self.device, dtype=torch.bool)
        self.seed_last_visible_rows = torch.empty(0, device=self.device, dtype=torch.bool)
        self.da3_new_sign = self.args.sam_new_sign
        self.representation_replay.clear()

    def reset(self) -> None:
        self.current_index = -1
        self.current_view = self._load_view(0)
        self.latest_evidence = None
        self.latest_summary = None
        self.latest_open_rows.zero_()
        self.latest_close_rows.zero_()
        self._reset_detector_state()
        self._reset_representation_state()

    def _append_current_da3_seeds(self, timestamp: int) -> dict[str, int]:
        """Accept current proposals outside mature learned Gaussian support."""

        if self.uses_typed_partition:
            return self._append_panel10_da3_seeds(timestamp)
        if self.da3_seeds is None:
            return {"proposed": 0, "accepted": 0, "coverage_rejected": 0}
        selected = np.flatnonzero(self.da3_seeds.birth_global == int(timestamp))
        if selected.size == 0:
            return {"proposed": 0, "accepted": 0, "coverage_rejected": 0}
        proposed = int(selected.size)
        if self.da3_seeds.log_scaling is None:
            raise RuntimeError("DA3 replay scaling is unavailable")
        signs = {self.da3_seeds.source_sign[int(row)] for row in selected.tolist()}
        if len(signs) != 1:
            raise RuntimeError("one DA3 birth frame contains conflicting NEW signs")
        sign = next(iter(signs))
        if sign in {"+", "-"}:
            if self.da3_new_sign is None:
                self.da3_new_sign = sign
            elif sign != self.da3_new_sign:
                raise RuntimeError("DA3 NEW sign changed after the causal sign lock")
        indices = torch.from_numpy(selected).long()
        proposal_xyz = torch.from_numpy(self.da3_seeds.xyz).index_select(0, indices).to(
            self.device
        )
        if self.seed_model.num_gaussians:
            eligible = self.seed_model.active_mask(float(timestamp))
            eligible |= self.seed_model.never_open_mask()
            uncovered = uncovered_by_learned_gaussian_support(
                proposal_xyz,
                self.seed_model.get_xyz,
                self.seed_model.get_scaling,
                self.seed_geometry_update_counts,
                eligible,
                min_updates=int(self.args.da3_coverage_min_updates),
                support_sigma=float(self.args.da3_coverage_sigma),
            )
            selected = selected[uncovered.detach().cpu().numpy()]
            indices = torch.from_numpy(selected).long()
            proposal_xyz = proposal_xyz[uncovered]
        accepted = int(selected.size)
        coverage_rejected = proposed - accepted
        if accepted == 0:
            return {
                "proposed": proposed,
                "accepted": 0,
                "coverage_rejected": coverage_rejected,
            }
        xyz = proposal_xyz
        scaling = (
            torch.from_numpy(self.da3_seeds.log_scaling)
            .index_select(0, indices)
            .to(self.device)
        )
        metadata = [
            {
                "source": "causal_da3_depth_prior",
                "source_row": int(row),
                "frame_name": self.da3_seeds.birth_name[int(row)],
                "source_sign": sign,
                "coverage_min_updates": int(self.args.da3_coverage_min_updates),
                "coverage_sigma": float(self.args.da3_coverage_sigma),
            }
            for row in selected.tolist()
        ]
        self.seed_model.append_xfeat_anchors(
            xyz=xyz,
            start=float(timestamp),
            scaling=scaling,
            opacity=float(self.args.da3_initial_opacity),
            metadata=metadata,
            optimizer=self.seed_optimizer,
            start_active=False,
        )
        self.seed_detector_probe.append(
            xyz=xyz,
            start=float(timestamp),
            scaling=scaling,
            opacity=float(self.args.da3_initial_opacity),
            metadata=metadata,
        )
        self.seed_detector_probe.seed_dc.requires_grad_(False)
        appended_lifecycle_rows = self.seed_lifecycle.append_rows(
            accepted,
            materialized_timestamp=timestamp,
        )
        self.accepted_da3_source_rows.extend(int(row) for row in selected.tolist())
        self.accepted_da3_birth_global.extend([int(timestamp)] * accepted)
        self.seed_geometry_update_counts = torch.cat(
            (
                self.seed_geometry_update_counts,
                torch.zeros(accepted, device=self.device, dtype=torch.long),
            )
        )
        expected = len(self.accepted_da3_source_rows)
        if (
            self.seed_model.num_gaussians != expected
            or self.seed_detector_probe.num_seeds != expected
            or self.seed_lifecycle.count != expected
            or appended_lifecycle_rows.tolist()
            != list(range(expected - accepted, expected))
        ):
            raise RuntimeError("DA3 sidecar did not materialize in causal birth order")
        return {
            "proposed": proposed,
            "accepted": accepted,
            "coverage_rejected": coverage_rejected,
        }

    @torch.no_grad()
    def _append_panel10_da3_seeds(self, timestamp: int) -> dict[str, int]:
        """Birth proposals from this frame's raw NEW mask, not a saved seed list."""
        from experiments.panel10_seed_topology import append_typed_seeds

        empty = {
            "proposed": 0,
            "accepted": 0,
            "coverage_rejected": 0,
            "coverage_3d_rejected": 0,
            "coverage_2d_existing_rejected": 0,
            "coverage_2d_same_frame_rejected": 0,
            "budget_rejected": 0,
            "occupancy_2d_pixels": 0,
            "occupancy_2d_seed_rows": 0,
        }
        if timestamp != self.current_index:
            raise RuntimeError("NEW seed proposals require the current observation")
        self.rendered_gs_online_da3_depth_difference_rgb()
        new_mask = self._typed_new_mask_cache.get(timestamp)
        depth_data = self._typed_depth_cache.get(timestamp)
        max_rows = int(self.args.da3_max_rows)
        available = (int(self.args.da3_birth_max_per_frame) if max_rows == 0
                     else max_rows - self.seed_model.num_gaussians)
        if new_mask is None or depth_data is None or available <= 0:
            return empty
        batch = panel10_da3_seed_proposals(
            self._normalized_cue_target(self.current_view), new_mask, *depth_data,
            stride=int(self.args.da3_birth_stride),
            # Apply the per-frame birth cap after coverage filtering; otherwise
            # already-covered high-Q cells can permanently starve novel cells.
            maximum=(math.ceil(new_mask.shape[0] / self.args.da3_birth_stride)
                     * math.ceil(new_mask.shape[1] / self.args.da3_birth_stride)),
        )
        proposed = len(batch.xyz)
        if proposed == 0:
            return empty
        xyz = batch.xyz.to(self.device)
        keep = torch.ones(proposed, device=self.device, dtype=torch.bool)
        if self.seed_model.num_gaussians:
            active, pending = self._seed_lifecycle_masks(timestamp)
            eligible = (active | pending) & ~self.seed_retired
            # Pending seeds never optimize, but their fixed initial footprint
            # must still suppress duplicate proposals while BF30 confirms them.
            support_updates = self.seed_geometry_update_counts.clone()
            support_updates[pending] = int(self.args.da3_coverage_min_updates)
            keep = uncovered_by_learned_gaussian_support(
                xyz, self.seed_model.get_xyz, self.seed_model.get_scaling,
                support_updates, eligible,
                min_updates=int(self.args.da3_coverage_min_updates),
                support_sigma=float(self.args.da3_coverage_sigma),
            )
        coverage_3d_rejected = proposed - int(keep.sum())
        occupancy_2d_pixels = 0
        occupancy_2d_seed_rows = 0
        coverage_2d_existing_rejected = 0
        coverage_2d_same_frame_rejected = 0
        coverage_mode = getattr(self.args, "da3_birth_coverage", "3d_only")
        candidates = torch.nonzero(keep, as_tuple=False).flatten()
        selected_list: list[int] = []
        if coverage_mode == "3d_plus_2d" and candidates.numel():
            aligned_depth, _reference_depth, depth_K, w2c = depth_data
            height, width = map(int, new_mask.shape)
            native_height, native_width = map(int, aligned_depth.shape)
            K_full = scale_depth_intrinsics_to_image(
                depth_K,
                native_height=native_height,
                native_width=native_width,
                height=height,
                width=width,
            )
            active_rows, never_rows = seed_rows_for_root_birth_2d_occupancy(
                self.seed_lifecycle, timestamp=timestamp, retired=self.seed_retired,
            )
            occ_parts_xyz: list[torch.Tensor] = []
            occ_parts_scaling: list[torch.Tensor] = []
            occ_parts_rotation: list[torch.Tensor] = []
            occ_part_rows: list[torch.Tensor] = []
            if active_rows.numel():
                rows = active_rows.to(device=self.device, dtype=torch.long)
                occ_parts_xyz.append(self.seed_model.get_xyz[rows])
                occ_parts_scaling.append(self.seed_model.get_scaling[rows])
                occ_parts_rotation.append(self.seed_model.get_rotation[rows])
                occ_part_rows.append(active_rows)
            if never_rows.numel():
                rows = never_rows.to(device=self.device, dtype=torch.long)
                occ_parts_xyz.append(self.seed_detector_probe.get_xyz[rows])
                occ_parts_scaling.append(self.seed_detector_probe.get_scaling[rows])
                occ_parts_rotation.append(self.seed_detector_probe.get_rotation[rows])
                occ_part_rows.append(never_rows)
            if occ_parts_xyz:
                existing_occ = projected_covariance_ellipse_occupancy(
                    torch.cat(occ_parts_xyz, dim=0),
                    torch.cat(occ_parts_scaling, dim=0),
                    torch.cat(occ_parts_rotation, dim=0),
                    K=K_full,
                    w2c=w2c,
                    height=height,
                    width=width,
                    sigma=float(getattr(self.args, "da3_coverage_2d_sigma", 2.0)),
                    row_indices=torch.cat(occ_part_rows, dim=0),
                )
                occupancy = existing_occ.mask.clone()
                occupancy_2d_pixels = int(existing_occ.pixels)
                occupancy_2d_seed_rows = int(existing_occ.rows_used)
                existing_mask = existing_occ.mask
            else:
                occupancy = torch.zeros((height, width), dtype=torch.bool)
                existing_mask = None

            pixels_xy = batch.pixels_xy.detach().cpu().to(dtype=torch.long)
            max_accept = min(int(self.args.da3_birth_max_per_frame), available)
            budget_rejected = 0
            for candidate in candidates.detach().cpu().tolist():
                if len(selected_list) >= max_accept:
                    budget_rejected += 1
                    continue
                x, y = pixels_xy[int(candidate)].tolist()
                center_occupied = (
                    0 <= int(y) < height
                    and 0 <= int(x) < width
                    and bool(occupancy[int(y), int(x)])
                )
                if center_occupied:
                    if len(selected_list) == 0:
                        # If no same-frame seed has been accepted yet, the
                        # occupied center can only come from the existing
                        # causal archive.
                        coverage_2d_existing_rejected += 1
                    else:
                        # Disambiguate against the pre-existing mask so the
                        # count identity remains exact even after reservations.
                        # Existing pixels retain priority over same-frame ones.
                        if existing_mask is not None and bool(existing_mask[int(y), int(x)]):
                            coverage_2d_existing_rejected += 1
                        else:
                            coverage_2d_same_frame_rejected += 1
                    continue
                selected_list.append(int(candidate))
                candidate_scaling = batch.log_scaling[int(candidate) : int(candidate) + 1].exp()
                candidate_rotation = torch.zeros((1, 4), dtype=torch.float32)
                candidate_rotation[:, 0] = 1.0
                reserved = union_projected_gaussian_ellipse_(
                    occupancy,
                    batch.xyz[int(candidate) : int(candidate) + 1],
                    candidate_scaling,
                    candidate_rotation,
                    K=K_full,
                    w2c=w2c,
                    sigma=float(getattr(self.args, "da3_coverage_2d_sigma", 2.0)),
                )
                if reserved > 0:
                    occupancy_2d_pixels += int(reserved)
                    occupancy_2d_seed_rows += 1
            selected = torch.as_tensor(selected_list, device=self.device, dtype=torch.long)
            counted = int(selected.numel()) + coverage_2d_existing_rejected + coverage_2d_same_frame_rejected + budget_rejected
            if counted != int(candidates.numel()):
                raise RuntimeError("DA3 birth rejection accounting lost candidates")
        else:
            selected = candidates[:min(int(self.args.da3_birth_max_per_frame), available)]
            budget_rejected = len(candidates) - len(selected)
        keep.zero_()
        keep[selected] = True
        coverage_rejected = (
            coverage_3d_rejected
            + coverage_2d_existing_rejected
            + coverage_2d_same_frame_rejected
        )
        cpu_keep = keep.cpu()
        metadata = [
            {"source": "panel10_new_online_da3", "source_sign": "+",
             "frame_name": self.records[timestamp].name,
             "pixel_xy": [int(x), int(y)], "birth_q": float(q)}
            for (x, y), q in zip(
                batch.pixels_xy[cpu_keep].tolist(),
                batch.confidence[cpu_keep].tolist(),
            )
        ]
        accepted = append_typed_seeds(
            self, xyz=xyz[keep], scaling=batch.log_scaling[cpu_keep].to(self.device),
            timestamp=timestamp, metadata=metadata,
        )
        if int(accepted) != int(selected.numel()):
            raise RuntimeError("DA3 typed seed append accepted fewer rows than precomputed capacity")
        if proposed != accepted + coverage_rejected + int(budget_rejected):
            raise RuntimeError("DA3 birth accounting identity failed")
        return {"proposed": proposed, "accepted": accepted,
                "coverage_rejected": coverage_rejected,
                "coverage_3d_rejected": coverage_3d_rejected,
                "coverage_2d_existing_rejected": coverage_2d_existing_rejected,
                "coverage_2d_same_frame_rejected": coverage_2d_same_frame_rejected,
                "budget_rejected": budget_rejected,
                "occupancy_2d_pixels": occupancy_2d_pixels,
                "occupancy_2d_seed_rows": occupancy_2d_seed_rows}

    def _constrain_typed_seeds(self, selected_rows: torch.Tensor) -> None:
        """Project only updated OPEN seeds into the existing root trust region."""
        clipped = constrain_da3_seed_geometry_(
            self.seed_model, selected_rows,
            max_displacement_ratio=float(self.args.da3_max_displacement_ratio),
            min_scale_ratio=float(self.args.da3_min_scale_ratio),
            max_scale_ratio=float(self.args.da3_max_scale_ratio),
            min_opacity=float(self.args.da3_min_opacity),
            max_opacity=float(self.args.da3_max_opacity),
        )
        for name, mask in clipped.items():
            if bool(mask.any()):
                self.seed_optimizer.reset_state_rows(mask, names=(name,))

    def _normalized_cue_target(self, view: Any) -> torch.Tensor:
        return torch.clamp(
            view.candidate_map.float() / float(self.args.cue_scale), 0.0, 1.0
        )

    def _detector_cue_inputs(self, view: Any) -> tuple[torch.Tensor, str, float, float]:
        """Optionally binarize normalized pixel Q, not aggregated Gaussian counts.

        The representation, Panel10 partition and seed proposals retain soft Q.
        Thresholding is performed by the evidence converter before alpha-T VJP.
        """
        if getattr(self.args, "detector_pixel_cue", "unchanged") == "binary_q05":
            return self._normalized_cue_target(view), "binary", 0.5, 1.0
        return (view.candidate_map, self.args.cue_mode,
                self.args.cue_threshold, self.args.cue_scale)

    def _update_da3_seed_detector(
        self,
        view: Any,
        *,
        timestamp: int,
    ) -> dict[str, int]:
        """Observe seeds with the base cue, before representation updates.

        The current preset shares post-sigmoid soft Q. The Part19 binary
        observation model is available only as an explicit legacy ablation.
        """

        empty = {
            "observed": 0,
            "opened": 0,
            "closed": 0,
            "active": 0,
            "never_open": 0,
            "closed_total": 0,
        }
        count = self.seed_detector_probe.num_seeds
        if count == 0:
            return empty
        if self.seed_lifecycle.count != count:
            raise RuntimeError("DA3 representation lifespan history is misaligned")
        if self.seed_tracker is None:
            raise RuntimeError("DA3 seed tracker is unavailable")
        if len(self.accepted_da3_source_rows) != count or len(
            self.accepted_da3_birth_global
        ) != count:
            raise RuntimeError("accepted DA3 seed mappings are misaligned")
        rows = da3_seed_detector_eligible_rows(
            self.accepted_da3_birth_global,
            timestamp=timestamp,
            device=self.device,
        )
        if self.uses_typed_partition:
            rows = rows[~self.seed_retired[rows]]
        if rows.numel() == 0:
            active = self.seed_lifecycle.active_mask(timestamp)
            return {
                **empty,
                "active": int(active.sum().item()),
                "never_open": int(
                    self.seed_lifecycle.never_open_mask(timestamp).sum().item()
                ),
                "closed_total": int(
                    self.seed_lifecycle.closed_mask(timestamp).sum().item()
                ),
            }
        from temporal.new_seed_gaussians import build_concatenated_change_view

        black_dc = RGB2SH(torch.zeros_like(self.base._features_dc))
        probe = build_concatenated_change_view(
            self.base,
            self.seed_detector_probe,
            timestamp=float(timestamp),
            base_dc=black_dc,
            detach_base_dc=True,
            seed_attribute_mode="all",
        )
        if self.args.da3_detector_cue_source == "part19_binary":
            detector_cue = part19_da3_detector_cue(view)
            cue_mode = "binary"
            cue_threshold = self.args.da3_detector_cue_threshold
            cue_scale = 1.0
        else:
            detector_cue, cue_mode, cue_threshold, cue_scale = self._detector_cue_inputs(view)
        evidence = self._accumulate_change_evidence(
            view,
            probe,
            self.pipe,
            self.evidence_background,
            detector_cue,
            cue_mode=cue_mode,
            cue_threshold=cue_threshold,
            cue_scale=cue_scale,
            count_mode=("capped_binary" if getattr(self.args, "detector_gaussian_cue", "unchanged") == "binary_q05" else "capped"),
            mass_saturation=self.args.evidence_mass_saturation,
            min_evidence_mass=self.args.min_evidence_mass,
            probe_scaling_mode=self.args.probe_scaling,
        )
        offset = self.count
        delta_a = evidence.delta_a[offset : offset + count].index_select(0, rows)
        delta_b = evidence.delta_b[offset : offset + count].index_select(0, rows)
        total_mass = evidence.total_mass[offset : offset + count].index_select(
            0, rows
        )
        all_source_rows = torch.as_tensor(
            self.accepted_da3_source_rows,
            device=self.device,
            dtype=torch.long,
        )
        source_rows = all_source_rows.index_select(0, rows)
        active_before = self.seed_lifecycle.active_mask(timestamp).index_select(
            0, rows
        )
        update = self.seed_tracker.update(
            delta_a,
            delta_b,
            total_mass=total_mass,
            current_active=active_before,
            first_open=self.seed_lifecycle.never_open_mask(timestamp).index_select(0, rows),
            first_open_bayes_factor_threshold=getattr(self.args, "first_open_bayes_factor_threshold", None),
            row_indices=source_rows,
            timestamp=int(timestamp),
        )
        committed = update.candidate_committed.bool()
        open_rows = rows[committed & ~active_before]
        close_rows = rows[committed & active_before]
        if open_rows.numel():
            self.seed_lifecycle.open_rows(open_rows, timestamp)
            self.seed_model.open_rows(open_rows, timestamp)
        if close_rows.numel():
            self.seed_lifecycle.close_rows(close_rows, timestamp)
            self.seed_model.close_rows(close_rows, timestamp)
        active_after = self.seed_lifecycle.active_mask(timestamp)
        if not torch.equal(
            active_after, self.seed_model.active_mask(float(timestamp))
        ):
            raise RuntimeError("DA3 current lifecycle and interval history diverged")
        return {
            "observed": int(update.observed.sum().item()),
            "opened": int(open_rows.numel()),
            "closed": int(close_rows.numel()),
            "active": int(active_after.sum().item()),
            "never_open": int(
                self.seed_lifecycle.never_open_mask(timestamp).sum().item()
            ),
            "closed_total": int(
                self.seed_lifecycle.closed_mask(timestamp).sum().item()
            ),
        }

    def _da3_seed_lifecycle_counts(self, timestamp: int) -> dict[str, int]:
        active = self.seed_lifecycle.active_mask(timestamp)
        return {
            "active": int(active.sum().item()),
            "never_open": int(
                self.seed_lifecycle.never_open_mask(timestamp).sum().item()
            ),
            "closed_total": int(
                self.seed_lifecycle.closed_mask(timestamp).sum().item()
            ),
        }

    def _sam_signed_score(self, index: int, view: Any) -> torch.Tensor | None:
        if self.sam_model is None:
            return None
        cached = self._sam_signed_score_cache.get(int(index))
        if cached is not None:
            return cached.to(device=self.device)

        from experiments.run_online_xfeat_new_seed import extract_delta

        record = self.records[index]
        trace = self.sam_sign_trace[str(record.name)]
        delta, _ = extract_delta(
            view,
            self.base,
            self.pipe,
            self.evidence_background,
            self.sam_model,
        )
        axis = torch.from_numpy(trace.axis).to(
            device=delta.device, dtype=delta.dtype
        )
        score = (delta @ axis).reshape(64, 64).float()
        self._sam_signed_score_cache[int(index)] = score.detach().cpu()
        return score

    def _new_seed_target(
        self, index: int, view: Any, cue_target: torch.Tensor
    ) -> torch.Tensor:
        """Use positive signed-SAM and depth support as the causal NEW target."""

        height = int(view.image_height)
        width = int(view.image_width)
        if self.da3_metric_depth is None:
            return torch.zeros(
                (1, height, width), device=self.device, dtype=cue_target.dtype
            )
        if index != self.current_index or view is not self.current_view:
            raise RuntimeError("panel-8 NEW target must be built for the current view")
        if self.uses_typed_partition:
            self.rendered_gs_online_da3_depth_difference_rgb()
            new_mask = self._typed_new_mask_cache.get(index)
            if new_mask is None:
                return torch.zeros_like(cue_target)
            return cue_target * new_mask.to(
                device=cue_target.device, dtype=cue_target.dtype
            )[None]
        depth_panel, _ = self.rendered_gs_online_da3_depth_difference_rgb()
        positive_depth = torch.from_numpy(depth_panel[..., 0] > 0).to(
            device=self.device
        )
        score = self._sam_signed_score(index, view)
        if score is None:
            return torch.zeros(
                (1, height, width), device=self.device, dtype=cue_target.dtype
            )
        weighted_sam, _ = q_weighted_upsampled_signed_sam_score(
            score,
            cue_target[0],
            height=height,
            width=width,
        )
        positive_sam, _ = threshold_signed_support(
            weighted_sam,
            threshold=PANEL7_DEPTH_SUPPORT_THRESHOLD,
        )
        seed_support = positive_depth & positive_sam.to(device=self.device)
        return seed_support.to(dtype=cue_target.dtype)[None] * cue_target

    def _seed_lifecycle_masks(
        self,
        timestamp: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return historical OPEN and NEVER_OPEN seed rows at ``timestamp``."""

        if self.seed_lifecycle.count != self.seed_model.num_gaussians:
            raise RuntimeError("DA3 lifecycle history is not aligned with seed rows")
        active = self.seed_lifecycle.active_mask(timestamp)
        never_open = self.seed_lifecycle.never_open_mask(timestamp)
        if bool((active & never_open).any()):
            raise RuntimeError("DA3 historical OPEN/NEVER_OPEN masks overlap")
        if timestamp == self.current_index and not torch.equal(
            active, self.seed_model.active_mask(float(timestamp))
        ):
            raise RuntimeError("DA3 current model lifecycle disagrees with history")
        return active, never_open

    def _effective_base_change_attributes(
        self,
        *,
        timestamp: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Render base rows according to the requested replay timestamp."""

        render_timestamp = self.current_index if timestamp is None else int(timestamp)
        opened = self.lifecycle.active_mask(render_timestamp)
        never_open = self.lifecycle.never_open_mask(render_timestamp)
        closed = self.lifecycle.closed_mask(render_timestamp)
        materialized = self.lifecycle.materialized_mask(render_timestamp)
        # FastGS evaluates degree-zero SH as C0 * DC + 0.5.  A raw DC of zero
        # is therefore mid-gray, not black, and would paint the entire frozen
        # NEVER_OPEN occluder layer with change score 0.5.  Use the exact SH
        # coefficient whose rendered RGB value is zero instead.
        black_dc = RGB2SH(self.change_dc.new_zeros(()))
        dc = torch.where(
            opened[:, None, None], self.change_dc, black_dc
        )
        opacity = self.base.get_opacity.detach().clone()
        if self.args.train_never_open_base_opacity:
            learned_opacity = self.base.opacity_activation(
                self.representation_opacity
            )
            opacity = torch.where(never_open[:, None], learned_opacity, opacity)
        opacity[closed | ~materialized] = 0.0
        return dc, opacity, opened

    def _train_representation_update(
        self,
        item: RepresentationReplayFrame,
        *,
        current_timestamp: int,
        dc_item: RepresentationReplayFrame | None = None,
    ) -> dict[str, float | int]:
        """Update the Gaussian population valid at each replay item's timestamp."""

        if self.uses_typed_partition:
            from experiments.panel10_split_training import train_partition_update

            return train_partition_update(
                self, item, current_timestamp=current_timestamp, dc_item=dc_item,
            )
        from gaussian_renderer import render_change
        from temporal.active_new_density import (
            new_geometry_coverage_loss,
            render_active_new_coverage,
        )
        from temporal.active_new_gaussians import ActiveNewGeometryView
        self.base_optimizer.zero_grad(set_to_none=True)
        self.seed_optimizer.zero_grad(set_to_none=True)
        dc_item = item if dc_item is None else dc_item
        geometry_timestamp = int(item.timestamp)
        dc_timestamp = int(dc_item.timestamp)
        for name, replay_timestamp in (
            ("geometry", geometry_timestamp),
            ("DC", dc_timestamp),
        ):
            if replay_timestamp < 0 or replay_timestamp > int(current_timestamp):
                raise RuntimeError(
                    f"{name} replay timestamp {replay_timestamp} is outside "
                    f"the causal range [0,{current_timestamp}]"
                )
        total_loss = 0.0
        trainable_open = int(
            self.lifecycle.active_mask(current_timestamp).sum().item()
        )
        visible_seed_rows = 0
        visible_pending_seed_rows = 0

        base_dc, base_opacity, dc_open_rows = (
            self._effective_base_change_attributes(timestamp=dc_timestamp)
        )
        dc_active_seed_rows, dc_never_open_seed_rows = (
            self._seed_lifecycle_masks(dc_timestamp)
        )
        dc_future_seed_rows = (
            (dc_active_seed_rows | dc_never_open_seed_rows)
            & ~self.seed_lifecycle.materialized_mask(dc_timestamp)
        )
        if bool(dc_future_seed_rows.any()):
            raise RuntimeError("future DA3 rows entered a historical DC render")
        current_active_seed_rows = self.seed_lifecycle.active_mask(current_timestamp)
        visible_base = torch.zeros_like(dc_open_rows)
        visible_dc_seed_rows = torch.zeros_like(dc_active_seed_rows)
        visible_seed_dc_rows = torch.zeros_like(dc_active_seed_rows)
        base_geometry_rows = torch.zeros_like(dc_open_rows)
        never_open_opacity_rows = torch.zeros_like(dc_open_rows)
        if bool(dc_open_rows.any()) or bool(dc_active_seed_rows.any()):
            model: Any = self.base
            override_dc: torch.Tensor | None = base_dc
            override_opacity: torch.Tensor | None = base_opacity
            override_xyz: torch.Tensor | None = self.representation_xyz.detach()
            override_scaling: torch.Tensor | None = self.base.scaling_activation(
                self.representation_scaling.detach()
            )
            override_rotation: torch.Tensor | None = self.base.rotation_activation(
                self.representation_rotation.detach()
            )
            rendered_seed_rows = torch.empty(
                0, device=self.device, dtype=torch.long
            )
            if self.seed_model.num_gaussians:
                from temporal.new_seed_gaussians import build_concatenated_change_view

                seed_render_mask = dc_active_seed_rows | dc_never_open_seed_rows
                rendered_seed_rows = torch.nonzero(
                    seed_render_mask, as_tuple=False
                ).flatten()
                model = build_concatenated_change_view(
                    self.base,
                    self.seed_model,
                    timestamp=float(dc_timestamp),
                    base_dc=base_dc,
                    base_opacity=base_opacity,
                    detach_base_dc=False,
                    seed_attribute_mode="lifecycle",
                    seed_lifecycle_active=dc_active_seed_rows,
                    seed_lifecycle_never_open=dc_never_open_seed_rows,
                )
                override_dc = None
                override_opacity = None
                override_xyz = None
                override_scaling = None
                override_rotation = None
            package = render_change(
                dc_item.view,
                model,
                self.pipe,
                self.evidence_background,
                override_dc=override_dc,
                override_opacity=override_opacity,
                override_xyz=override_xyz,
                override_scaling=override_scaling,
                override_rotation=override_rotation,
                clamp_output=False,
            )
            radii = package["radii"].detach()
            visible_base = dc_open_rows & (radii[: self.count] > 0)
            if rendered_seed_rows.numel():
                visible_rendered_seeds = radii[self.count :] > 0
                visible_dc_seed_rows[rendered_seed_rows] = (
                    dc_active_seed_rows[rendered_seed_rows]
                    & visible_rendered_seeds
                )
            dc_target = representation_dc_target(
                dc_item.cue_target,
                amplitude=float(self.args.representation_cue_amplitude),
            )
            oscd_loss, _ = compute_ssf_loss(dc_target, package["render"])
            total_loss += float(oscd_loss.detach().item())
            if bool(visible_base.any()) or bool(visible_dc_seed_rows.any()):
                oscd_loss.backward()
            if self.args.seed_dc_supervision == "joint":
                visible_seed_dc_rows |= visible_dc_seed_rows
            elif self.seed_model.new_dc.grad is not None:
                # Keep the joint forward population and base gradient exactly
                # unchanged, but replace the seed DC gradient below with a
                # base-free rendering of the same active seed rows.
                self.seed_model.new_dc.grad.zero_()

        if (
            self.args.seed_dc_supervision == "seed_only_ssf"
            and bool(dc_active_seed_rows.any())
        ):
            seed_dc_view = ActiveNewGeometryView(
                self.seed_model,
                float(dc_timestamp),
                row_mask=dc_active_seed_rows,
                detach_geometry=True,
                detach_opacity=True,
            )
            seed_dc_package = render_change(
                dc_item.view,
                seed_dc_view,
                self.pipe,
                self.evidence_background,
                clamp_output=False,
            )
            seed_dc_visible_local = seed_dc_package["radii"].detach() > 0
            if bool(seed_dc_visible_local.any()):
                visible_seed_dc_rows[
                    seed_dc_view.global_rows[seed_dc_visible_local]
                ] = True
                seed_dc_target = representation_dc_target(
                    dc_item.cue_target,
                    amplitude=float(self.args.representation_cue_amplitude),
                )
                seed_dc_loss, _ = compute_ssf_loss(
                    seed_dc_target,
                    seed_dc_package["render"],
                )
                weighted_seed_dc_loss = (
                    float(self.args.seed_dc_loss_weight) * seed_dc_loss
                )
                total_loss += float(weighted_seed_dc_loss.detach().item())
                weighted_seed_dc_loss.backward()
        elif self.args.seed_dc_supervision == "projected_bce" and bool(
            dc_active_seed_rows.any()
        ):
            from experiments.run_online_xfeat_new_seed import (
                new_sidecar_projected_dc_loss,
            )

            seed_dc_loss, _ = new_sidecar_projected_dc_loss(
                self.seed_model,
                dc_item.view,
                dc_item.cue_target,
                timestamp=dc_timestamp,
                active_row_mask=dc_active_seed_rows,
            )
            weighted_seed_dc_loss = (
                float(self.args.seed_dc_loss_weight) * seed_dc_loss
            )
            total_loss += float(weighted_seed_dc_loss.detach().item())
            weighted_seed_dc_loss.backward()
            visible_seed_dc_rows |= dc_active_seed_rows

        if self.args.base_geometry_scope != "frozen":
            geometry_open_rows = self.lifecycle.active_mask(geometry_timestamp)
            geometry_never_open_rows = self.lifecycle.never_open_mask(
                geometry_timestamp
            )
            if self.args.base_geometry_scope == "open":
                geometry_support = geometry_open_rows
            else:
                geometry_support = geometry_open_rows | geometry_never_open_rows
            geometry_opacity = self.base.get_opacity.detach()
            if self.args.train_never_open_base_opacity:
                geometry_opacity = torch.where(
                    geometry_never_open_rows[:, None],
                    self.base.opacity_activation(self.representation_opacity),
                    geometry_opacity,
                )
            geometry_opacity = geometry_opacity * geometry_support[:, None]
            geometry_package = render_change(
                item.view,
                self.base,
                self.pipe,
                self.evidence_background,
                override_color=torch.ones(
                    (self.count, 3),
                    device=self.device,
                    dtype=self.base.get_xyz.dtype,
                ),
                override_opacity=geometry_opacity,
                override_xyz=self.representation_xyz,
                override_scaling=self.base.scaling_activation(
                    self.representation_scaling
                ),
                override_rotation=self.base.rotation_activation(
                    self.representation_rotation
                ),
                clamp_output=False,
            )
            unit_base_geometry_target = (
                item.new_target
                if self.args.base_geometry_target == "signed_new"
                else item.cue_target
            )
            base_geometry_target = representation_geometry_target(
                unit_base_geometry_target,
                amplitude=float(self.args.geometry_cue_amplitude),
            )
            base_geometry_loss, _ = new_geometry_coverage_loss(
                base_geometry_target,
                geometry_package["render"],
                inside_weight=float(self.args.base_geometry_inside_weight),
                outside_weight=float(self.args.base_geometry_outside_weight),
            )
            total_loss += float(base_geometry_loss.detach().item())
            base_geometry_rows = geometry_support & (
                geometry_package["radii"].detach() > 0
            )
            if self.args.train_never_open_base_opacity:
                never_open_opacity_rows = (
                    geometry_never_open_rows
                    & (geometry_package["radii"].detach() > 0)
                )
            if bool(base_geometry_rows.any()):
                base_geometry_loss.backward()

        if bool(visible_base.any()) or bool(base_geometry_rows.any()):
            if self.args.base_geometry_scope == "frozen":
                self.base_optimizer.step(visible_base)
            else:
                optimizer_masks = {
                    "dc": visible_base,
                    "xyz": base_geometry_rows,
                    "scaling": base_geometry_rows,
                    "rotation": base_geometry_rows,
                }
                if self.args.train_never_open_base_opacity:
                    optimizer_masks["opacity"] = never_open_opacity_rows
                self.base_optimizer.step(optimizer_masks)
                clipped_base = constrain_base_representation_geometry_(
                    xyz=self.representation_xyz,
                    scaling=self.representation_scaling,
                    rotation=self.representation_rotation,
                    anchor_xyz=self.base._xyz.detach(),
                    anchor_scaling=self.base._scaling.detach(),
                    selected_rows=base_geometry_rows,
                    max_displacement_ratio=float(
                        self.args.base_max_displacement_ratio
                    ),
                    min_scale_ratio=float(self.args.base_min_scale_ratio),
                    max_scale_ratio=float(self.args.base_max_scale_ratio),
                )
                for name, mask in clipped_base.items():
                    if bool(mask.any()):
                        self.base_optimizer.reset_state_rows(mask, names=(name,))

        geometry_active_seed_rows, geometry_never_open_seed_rows = (
            self._seed_lifecycle_masks(geometry_timestamp)
        )
        geometry_future_seed_rows = (
            (geometry_active_seed_rows | geometry_never_open_seed_rows)
            & ~self.seed_lifecycle.materialized_mask(geometry_timestamp)
        )
        if bool(geometry_future_seed_rows.any()):
            raise RuntimeError("future DA3 rows entered a historical geometry render")
        pending_seed_rows = (
            geometry_never_open_seed_rows
            if self.args.train_never_open_geometry
            else torch.zeros_like(geometry_active_seed_rows)
        )
        active_geometry_rows = torch.zeros_like(geometry_active_seed_rows)
        pending_geometry_rows = torch.zeros_like(geometry_active_seed_rows)
        if bool(geometry_active_seed_rows.any()):
            active_view = ActiveNewGeometryView(
                self.seed_model,
                float(geometry_timestamp),
                row_mask=geometry_active_seed_rows,
            )
            geometry_package = render_active_new_coverage(
                item.view,
                active_view,
                self.pipe,
                self.evidence_background,
            )
            geometry_target = representation_geometry_target(
                item.cue_target,
                amplitude=float(self.args.geometry_cue_amplitude),
            )
            geometry_loss, _ = new_geometry_coverage_loss(
                geometry_target,
                geometry_package["render"],
                inside_weight=float(self.args.da3_geometry_inside_weight),
                outside_weight=float(self.args.da3_geometry_outside_weight),
            )
            total_loss += float(geometry_loss.detach().item())
            visible_local = geometry_package["radii"].detach() > 0
            if bool(visible_local.any()):
                active_geometry_rows[active_view.global_rows[visible_local]] = True
                geometry_loss.backward()
            visible_seed_rows = int(active_geometry_rows.sum().item())

        if bool(pending_seed_rows.any()):
            pending_view = ActiveNewGeometryView(
                self.seed_model,
                float(geometry_timestamp),
                row_mask=pending_seed_rows,
                detach_dc=True,
                detach_opacity=True,
            )
            pending_package = render_active_new_coverage(
                item.view,
                pending_view,
                self.pipe,
                self.evidence_background,
            )
            pending_target = representation_geometry_target(
                item.cue_target,
                amplitude=float(self.args.geometry_cue_amplitude),
            )
            pending_loss, _ = new_geometry_coverage_loss(
                pending_target,
                pending_package["render"],
                inside_weight=float(self.args.da3_geometry_inside_weight),
                outside_weight=float(self.args.da3_geometry_outside_weight),
            )
            total_loss += float(pending_loss.detach().item())
            pending_visible_local = pending_package["radii"].detach() > 0
            if bool(pending_visible_local.any()):
                pending_geometry_rows[
                    pending_view.global_rows[pending_visible_local]
                ] = True
                pending_loss.backward()
            visible_pending_seed_rows = int(pending_geometry_rows.sum().item())

        dc_rows = torch.zeros_like(dc_active_seed_rows)
        if self.seed_model.new_dc.grad is not None:
            dc_rows = visible_seed_dc_rows & (
                self.seed_model.new_dc.grad.detach()
                .reshape(self.seed_model.num_gaussians, -1)
                .abs()
                .sum(dim=1)
                > 0
            )
        geometry_rows = active_geometry_rows | pending_geometry_rows
        if bool(geometry_rows.any()) or bool(dc_rows.any()):
            self.seed_optimizer.step(
                {
                    "xyz": geometry_rows,
                    "dc": dc_rows,
                    "opacity": active_geometry_rows,
                    "scaling": geometry_rows,
                    "rotation": geometry_rows,
                }
            )
        if self.seed_geometry_update_counts.shape != geometry_rows.shape:
            raise RuntimeError("DA3 geometry update counts are misaligned")
        self.seed_geometry_update_counts[geometry_rows] += 1
        constrained_rows = geometry_active_seed_rows | pending_seed_rows
        if bool(constrained_rows.any()):
            clipped = constrain_da3_seed_geometry_(
                self.seed_model,
                constrained_rows,
                max_displacement_ratio=float(
                    self.args.da3_max_displacement_ratio
                ),
                min_scale_ratio=float(self.args.da3_min_scale_ratio),
                max_scale_ratio=float(self.args.da3_max_scale_ratio),
                min_opacity=float(self.args.da3_min_opacity),
                max_opacity=float(self.args.da3_max_opacity),
            )
            for name, mask in clipped.items():
                if bool(mask.any()):
                    self.seed_optimizer.reset_state_rows(mask, names=(name,))

        return {
            "loss": total_loss,
            "trainable_open_rows": trainable_open,
            "active_seed_rows": int(current_active_seed_rows.sum().item()),
            "visible_seed_rows": visible_seed_rows,
            "visible_pending_seed_rows": visible_pending_seed_rows,
            "lifespan_render_violations": int(dc_future_seed_rows.sum().item())
            + int(geometry_future_seed_rows.sum().item()),
        }

    def _train_representation(self, timestamp: int) -> dict[str, Any]:
        if len(self.representation_replay) != timestamp + 1:
            raise RuntimeError("representation replay is not causally aligned")
        sampled: list[int] = []
        latest_branch_count = 0
        max_visible_seed_rows = 0
        density_children = 0
        clone_count = 0
        split_source_count = 0
        pruned_rows = 0
        lifespan_render_violations = 0
        last: dict[str, float | int] = {
            "loss": 0.0,
            "trainable_open_rows": 0,
            "active_seed_rows": self.seed_model.num_gaussians,
            "visible_seed_rows": 0,
            "lifespan_render_violations": 0,
        }
        for update_index in range(int(self.args.representation_updates)):
            sampled_index, latest_branch = causal_training_view_index(
                timestamp,
                update_index,
                seed=int(self.args.representation_seed),
                current_probability=float(self.args.current_view_probability),
            )
            if self.uses_typed_partition and self.args.dc_replay_mode == "current":
                # Partitioned DC+geometry share a view; audit the actual view,
                # not the discarded legacy geometry-sampler choice.
                sampled_index, latest_branch = timestamp, True
            if sampled_index > timestamp:
                raise RuntimeError("representation replay accessed a future frame")
            if (
                int(self.representation_replay[sampled_index].timestamp)
                != sampled_index
            ):
                raise RuntimeError("representation replay timestamp/index mismatch")
            sampled.append(sampled_index)
            latest_branch_count += int(latest_branch)
            last = self._train_representation_update(
                self.representation_replay[sampled_index],
                current_timestamp=timestamp,
                dc_item=(
                    self.representation_replay[timestamp]
                    if self.args.dc_replay_mode == "current"
                    else None
                ),
            )
            max_visible_seed_rows = max(
                max_visible_seed_rows, int(last["visible_seed_rows"])
            )
            lifespan_render_violations += int(last["lifespan_render_violations"])
            if self.uses_typed_partition and update_index + 1 == int(
                self.args.da3_density_update
            ):
                from experiments.panel10_seed_topology import apply_typed_density

                density = apply_typed_density(
                    self, timestamp=timestamp,
                    random_seed=int(self.args.representation_seed) + timestamp,
                )
                density_children += int(density["children"])
                clone_count += int(density["clone_count"])
                split_source_count += int(density["split_source_count"])
        if self.uses_typed_partition:
            from experiments.panel10_seed_topology import prune_typed_seeds

            pruning = prune_typed_seeds(self, timestamp=timestamp)
            pruned_rows = int(pruning["pruned"])
            self.seed_lifecycle.validate_lifecycle()
            self.seed_model.validate()
        return {
            **last,
            "updates": len(sampled),
            "latest_branch_count": latest_branch_count,
            "sampled_oldest": min(sampled) if sampled else None,
            "sampled_newest": max(sampled) if sampled else None,
            "sampled_indices": tuple(sampled),
            "future_view_accesses": sum(index > timestamp for index in sampled),
            "historical_lifespan_replay_updates": sum(
                index < timestamp for index in sampled
            ),
            "lifespan_render_violations": lifespan_render_violations,
            "visible_seed_rows": max_visible_seed_rows,
            "density_children": density_children,
            "clone_count": clone_count,
            "split_source_count": split_source_count,
            "pruned_rows": pruned_rows,
        }

    def _visual_state(self) -> DetectorVisualState:
        return detector_visual_state(
            current_active=self.lifecycle.current_state_index >= 0,
            ever_opened=self.lifecycle.num_states > 0,
            candidate_active=self.tracker.candidate_active[:self.count],
            last_log_bayes_factor=self.tracker.last_log_bayes_factor[:self.count],
            bayes_factor_threshold=self.args.bayes_factor_threshold,
        )

    def _seed_visual_state(self) -> DetectorVisualState:
        """Translate every causally born DA3 row to committed lifecycle colors."""

        count = self.seed_model.num_gaussians
        if count == 0:
            raise RuntimeError("DA3 seed visual state requires at least one born seed")
        tracker = self.tracker if self.uses_typed_partition else self.seed_tracker
        if tracker is None:
            raise RuntimeError("DA3 seed tracker is unavailable")
        source_rows = torch.as_tensor(
            self.accepted_da3_source_rows,
            device=self.device,
            dtype=torch.long,
        )
        if source_rows.shape != (count,):
            raise RuntimeError("accepted DA3 source-row mapping is misaligned")
        if self.uses_typed_partition:
            source_rows = source_rows + self.count
        active, never_open = self._seed_lifecycle_masks(self.current_index)
        return detector_visual_state(
            current_active=active,
            ever_opened=~never_open,
            candidate_active=tracker.candidate_active[source_rows],
            last_log_bayes_factor=tracker.last_log_bayes_factor[
                source_rows
            ],
            bayes_factor_threshold=self.args.bayes_factor_threshold,
        )

    def _joint_detector_probe(self, *, timestamp: int) -> tuple[Any, torch.Tensor]:
        """One fixed base+seed field, including roots proposed this frame.

        Lifecycle CLOSED probes stay observable for REOPEN; permanently retired
        density rows are excluded from both evidence and occlusion. The returned
        mapping preserves archive IDs when those retired rows leave holes.
        No learned DC, seed geometry, or lifecycle opacity gate enters this field.
        """
        from temporal.new_seed_gaussians import build_concatenated_change_view

        count = self.seed_detector_probe.num_seeds
        births = torch.as_tensor(self.accepted_da3_birth_global, device=self.device)
        if births.shape != (count,) or self.seed_lifecycle.count != count:
            raise RuntimeError("joint detector seed archive is misaligned")
        if bool((births > timestamp).any()):
            raise ValueError("future DA3 seed proposals cannot enter the detector")
        selected = ~self.seed_retired
        probe = build_concatenated_change_view(
            self.base, self.seed_detector_probe, timestamp=float(timestamp),
            base_dc=RGB2SH(torch.zeros_like(self.base._features_dc)),
            seed_lifecycle_active=selected,
            seed_lifecycle_never_open=torch.zeros_like(selected),
        )
        rows = torch.cat((
            torch.arange(self.count, device=self.device),
            self.count + torch.nonzero(selected, as_tuple=False).flatten(),
        ))
        return probe, rows

    def step(self) -> ReplayStepSummary:
        """Consume one view exactly once and apply detector lifecycle commits."""

        if not self.has_next:
            raise StopIteration("all online frames have been consumed")
        from temporal.bayesian_lifespan_controller import LifespanAction

        index = self.current_index + 1
        view = self._load_view(index)
        probe = self.base
        detector_rows = torch.arange(self.count, device=self.device)
        active_before = self.lifecycle.current_state_index >= 0
        first_open = self.lifecycle.never_open_mask(index)
        if self.uses_typed_partition:
            self.current_index = index
            self.current_view = view
            cue_target = self._normalized_cue_target(view)
            new_target = self._new_seed_target(index, view, cue_target)
            seed_birth = self._append_current_da3_seeds(index)
            probe, detector_rows = self._joint_detector_probe(timestamp=index)
            active_before = torch.cat((active_before, self.seed_lifecycle.active_mask(index)))
            first_open = torch.cat((first_open, self.seed_lifecycle.never_open_mask(index)))
        seed_detection = {"observed": 0, "opened": 0, "closed": 0}
        detector_cue, cue_mode, cue_threshold, cue_scale = self._detector_cue_inputs(view)
        evidence = self._accumulate_change_evidence(
            view,
            probe,
            self.pipe,
            self.evidence_background,
            detector_cue,
            cue_mode=cue_mode,
            cue_threshold=cue_threshold,
            cue_scale=cue_scale,
            count_mode=("capped_binary" if getattr(self.args, "detector_gaussian_cue", "unchanged") == "binary_q05" else "capped"),
            mass_saturation=self.args.evidence_mass_saturation,
            min_evidence_mass=self.args.min_evidence_mass,
            probe_scaling_mode=self.args.probe_scaling,
        )
        observed = (
            (evidence.total_mass > 0)
            & (evidence.total_mass >= self.args.min_evidence_mass)
            & ((evidence.delta_a + evidence.delta_b) > 0)
        )
        rows_all = torch.nonzero(observed, as_tuple=False).flatten()
        action_counts = {action.name: 0 for action in LifespanAction}
        candidate_counts = {name: 0 for name in ("started", "continued", "rejected", "committed")}
        self.latest_open_rows.zero_()
        self.latest_close_rows.zero_()

        for start in range(0, int(rows_all.numel()), self.args.filter_chunk_size):
            rows = rows_all[start : start + self.args.filter_chunk_size]
            bank_rows = detector_rows[rows]
            update = self.tracker.update(
                evidence.delta_a[rows],
                evidence.delta_b[rows],
                total_mass=evidence.total_mass[rows],
                current_active=active_before[bank_rows],
                first_open=first_open[bank_rows],
                first_open_bayes_factor_threshold=getattr(self.args, "first_open_bayes_factor_threshold", None),
                row_indices=bank_rows,
                timestamp=index,
            )
            base_positions = bank_rows < self.count
            # One inference bank; only the lifecycle storage/representation is
            # split by ownership. Slice the controller's row fields, not evidence.
            base_update = SimpleNamespace(**{
                name: getattr(update, name)[base_positions]
                for name in ("indices", "observed", "candidate_committed", "active_before",
                             "concentration", "estimated_run_start", "visible_observations")
            })
            decision = self.controller.update(base_update, timestamp=index, optimizer=None)
            counts = torch.bincount(
                decision.action.to(torch.long), minlength=len(LifespanAction)
            )
            for action in LifespanAction:
                action_counts[action.name] += int(counts[int(action)].item())
            for name in candidate_counts:
                candidate_counts[name] += int(getattr(update, f"candidate_{name}")[base_positions].sum().item())
            open_positions = decision.action == int(LifespanAction.OPEN)
            close_positions = decision.action == int(LifespanAction.CLOSE)
            self.latest_open_rows[decision.indices[open_positions]] = True
            self.latest_close_rows[decision.indices[close_positions]] = True

            if self.uses_typed_partition:
                seed_positions = ~base_positions
                seed_rows = bank_rows[seed_positions] - self.count
                committed = update.candidate_committed[seed_positions].bool()
                was_active = update.active_before[seed_positions]
                opens = seed_rows[committed & ~was_active]
                closes = seed_rows[committed & was_active]
                if opens.numel():
                    self.seed_lifecycle.open_rows(opens, index)
                    self.seed_model.open_rows(opens, index)
                if closes.numel():
                    self.seed_lifecycle.close_rows(closes, index)
                    self.seed_model.close_rows(closes, index)
                seed_detection["observed"] += int(update.observed[seed_positions].sum().item())
                seed_detection["opened"] += int(opens.numel())
                seed_detection["closed"] += int(closes.numel())

        self.current_index = index
        self.current_view = view
        self.latest_evidence = evidence
        if self.uses_typed_partition:
            if not torch.equal(self.seed_lifecycle.active_mask(index), self.seed_model.active_mask(float(index))):
                raise RuntimeError("DA3 current lifecycle and interval history diverged")
        else:
            # Preserve historical separate-detector experiments explicitly.
            cue_target = self._normalized_cue_target(view)
            seed_detection = self._update_da3_seed_detector(view, timestamp=index)
            new_target = self._new_seed_target(index, view, cue_target)
            seed_birth = self._append_current_da3_seeds(index)
        seed_detection.update(self._da3_seed_lifecycle_counts(index))
        self.representation_replay.append(
            RepresentationReplayFrame(
                timestamp=index,
                view=view,
                cue_target=cue_target.detach(),
                new_target=new_target.detach(),
            )
        )
        training = self._train_representation(index)
        # Finalize only after same-timestamp child birth and retirement/pruning.
        self.lifecycle.seal_snapshot(index)
        self.seed_lifecycle.seal_snapshot(index)
        seed_detection.update(self._da3_seed_lifecycle_counts(index))
        if int(training["future_view_accesses"]) != 0:
            raise RuntimeError("future representation view access audit failed")
        visual = self._visual_state()
        summary = ReplayStepSummary(
            timestamp=index,
            frame_name=self.records[index].name,
            observed=int(observed[:self.count].sum().item()),
            never_open=int(visual.never_open.sum().item()),
            uncertain=int(visual.uncertain.sum().item()),
            open=int(visual.open.sum().item()),
            closed=int(visual.closed.sum().item()),
            opened_now=action_counts["OPEN"],
            closed_now=action_counts["CLOSE"],
            candidate_started=candidate_counts["started"],
            candidate_continued=candidate_counts["continued"],
            candidate_rejected=candidate_counts["rejected"],
            candidate_committed=candidate_counts["committed"],
            positive_pseudocount_mass=float(evidence.delta_a[:self.count].sum().item()),
            negative_pseudocount_mass=float(evidence.delta_b[:self.count].sum().item()),
            cue_tau=getattr(view, "learned_cue_tau", None),
            cue_width=getattr(view, "learned_cue_width", None),
            representation_updates=int(training["updates"]),
            sampled_latest_branch=int(training["latest_branch_count"]),
            sampled_oldest_timestamp=training["sampled_oldest"],
            sampled_newest_timestamp=training["sampled_newest"],
            historical_lifespan_replay_updates=int(
                training["historical_lifespan_replay_updates"]
            ),
            lifespan_render_violations=int(
                training["lifespan_render_violations"]
            ),
            representation_loss=float(training["loss"]),
            trainable_open_rows=int(training["trainable_open_rows"]),
            active_da3_seed_rows=int(seed_detection["active"]),
            visible_da3_seed_rows=int(training["visible_seed_rows"]),
            visible_pending_da3_seed_rows=int(
                training["visible_pending_seed_rows"]
            ),
            da3_seed_opened_now=int(seed_detection["opened"]),
            da3_seed_closed_now=int(seed_detection["closed"]),
            da3_seed_never_open=int(seed_detection["never_open"]),
            da3_seed_closed=int(seed_detection["closed_total"]),
            da3_seed_proposed_now=int(seed_birth["proposed"]),
            da3_seed_accepted_now=int(seed_birth["accepted"]),
            da3_seed_coverage_rejected_now=int(seed_birth["coverage_rejected"]),
            da3_seed_coverage_3d_rejected_now=int(seed_birth.get("coverage_3d_rejected", 0)),
            da3_seed_coverage_2d_existing_rejected_now=int(seed_birth.get("coverage_2d_existing_rejected", 0)),
            da3_seed_coverage_2d_same_frame_rejected_now=int(seed_birth.get("coverage_2d_same_frame_rejected", 0)),
            da3_seed_budget_rejected_now=int(seed_birth.get("budget_rejected", 0)),
            da3_seed_occupancy_2d_pixels=int(seed_birth.get("occupancy_2d_pixels", 0)),
            da3_seed_occupancy_2d_rows=int(seed_birth.get("occupancy_2d_seed_rows", 0)),
            base_loss=float(training.get("base_loss", 0.0)),
            new_loss=float(training.get("new_loss", 0.0)),
            da3_density_children=int(training.get("density_children", 0)),
            da3_cloned_now=int(training.get("clone_count", 0)),
            da3_split_sources_now=int(training.get("split_source_count", 0)),
            da3_pruned_now=int(training.get("pruned_rows", 0)),
            da3_seed_observed_now=int(seed_detection["observed"]),
        )
        self.latest_summary = summary
        return summary

    def input_rgb(self) -> np.ndarray:
        return _rgb_u8(self.current_view.original_image)

    def input_with_da3_seeds_rgb(
        self,
    ) -> tuple[np.ndarray, DA3SeedProjectionStats]:
        rgb = self.input_rgb()
        if (self.da3_seeds is None and not self.uses_typed_partition) or self.current_index < 0:
            return rgb, DA3SeedProjectionStats()
        if not self.accepted_da3_source_rows:
            return rgb, DA3SeedProjectionStats()
        record = self.records[self.current_index]
        camera = self.cameras[Path(record.name).stem]
        accepted = np.asarray(self.accepted_da3_source_rows, dtype=np.int64)
        current_seeds = DA3SeedReplay(
            xyz=np.ascontiguousarray(
                self.seed_model.get_xyz.detach().cpu().numpy().astype(np.float32)
            ),
            birth_global=np.asarray(self.accepted_da3_birth_global, dtype=np.int64),
            birth_name=(
                tuple(self.records[t].name for t in self.accepted_da3_birth_global)
                if self.uses_typed_partition
                else tuple(self.da3_seeds.birth_name[row] for row in accepted)
            ),
            log_scaling=np.ascontiguousarray(
                self.seed_model._scaling.detach().cpu().numpy().astype(np.float32)
            ),
            source_sign=(
                ("+",) * len(accepted) if self.uses_typed_partition
                else tuple(self.da3_seeds.source_sign[row] for row in accepted)
            ),
        )
        return overlay_da3_seed_centers(
            rgb,
            current_seeds,
            global_index=int(record.global_index),
            camera=camera,
            marker_radius=int(self.args.da3_seed_marker_radius),
        )

    def cue_rgb(self) -> np.ndarray:
        if self.args.cue_mode == "binary":
            cue = (self.current_view.candidate_map > self.args.cue_threshold).float()
        else:
            cue = torch.clamp(
                self.current_view.candidate_map / self.args.cue_scale, 0.0, 1.0
            )
        return _score_heatmap_image(cue)

    def sam_feature_diff_rgb(
        self,
    ) -> tuple[np.ndarray, SamSignedFeatureStats]:
        """Compute upsampled signed SAM difference weighted by learned Q."""

        height = int(self.current_view.image_height)
        width = int(self.current_view.image_width)
        if self.sam_model is None or self.current_index < 0:
            return np.zeros((height, width, 3), dtype=np.uint8), SamSignedFeatureStats(
                neutral=height * width
            )
        cached = self._sam_feature_diff_cache.get(self.current_index)
        if cached is not None:
            return cached[0].copy(), cached[1]

        score = self._sam_signed_score(self.current_index, self.current_view)
        if score is None:
            raise RuntimeError("SAM signed score is unavailable")
        learned_q = torch.clamp(
            self.current_view.candidate_map / self.args.cue_scale,
            0.0,
            1.0,
        )[0].detach().cpu()
        image, stats = q_weighted_signed_sam_feature_diff_image(
            score,
            learned_q,
            width=width,
            height=height,
        )
        self._sam_feature_diff_cache[self.current_index] = (image, stats)
        return image.copy(), stats

    def rendered_gs_online_da3_depth_difference_rgb(
        self,
    ) -> tuple[np.ndarray, DepthDifferenceStats]:
        """Compare rendered GS depth with per-frame aligned online DA3Metric."""

        height = int(self.current_view.image_height)
        width = int(self.current_view.image_width)
        if self.da3_metric_depth is None or self.current_index < 0:
            return np.zeros((height, width, 3), dtype=np.uint8), DepthDifferenceStats()
        cached = self._depth_difference_cache.get(self.current_index)
        if cached is not None:
            return cached[0].copy(), cached[1]

        record = self.records[self.current_index]
        online_cache_path = (
            self.da3_metric_depth.cache_root
            / f"frame_{int(record.global_index):06d}.npz"
        )
        from experiments.build_causal_da3_seed_replay import (
            _cached_or_infer_metric_depth,
        )
        from experiments.run_online_xfeat_new_seed import fixed_camera_matrices

        w2c, image_K_tensor = fixed_camera_matrices(record, self.cameras)
        image_K = image_K_tensor.numpy()
        model = (
            self._ensure_da3_metric_model()
            if not online_cache_path.is_file()
            else None
        )
        online_depth_meters, online_K = _cached_or_infer_metric_depth(
            da3_metric=model,
            cache_path=online_cache_path,
            image=_rgb_u8(self.current_view.original_image[:3]),
            image_K=image_K,
            process_res=self.da3_metric_depth.process_res,
        )
        if online_K.shape != (3, 3) or not np.isfinite(online_K).all():
            raise RuntimeError("online DA3Metric processed intrinsics are invalid")
        online_depth = torch.from_numpy(
            np.asarray(online_depth_meters, dtype=np.float32).copy()
        )
        target_shape = tuple(int(value) for value in online_depth.shape)
        learned_q_full = torch.clamp(
            self.current_view.candidate_map / self.args.cue_scale,
            0.0,
            1.0,
        )[0].detach().cpu()
        learned_q = F.interpolate(
            learned_q_full[None, None],
            size=target_shape,
            mode="area",
        )[0, 0]

        from experiments.analyze_da3_depth_prior_new_seeds import (
            render_reference_depth,
        )

        rendered_full, alpha_full = render_reference_depth(
            self.current_view,
            self.base,
            w2c,
            self.pipe,
            self.evidence_background,
        )
        rendered_numerator = F.interpolate(
            (rendered_full * alpha_full)[None, None],
            size=target_shape,
            mode="area",
        )[0, 0]
        reference_alpha = F.interpolate(
            alpha_full[None, None],
            size=target_shape,
            mode="area",
        )[0, 0]
        rendered_depth = rendered_numerator / reference_alpha.clamp_min(1.0e-6)
        alignment_mask = (
            reference_alpha >= DEPTH_ALIGNMENT_MIN_REFERENCE_ALPHA
        ) & (learned_q < DEPTH_ALIGNMENT_MAX_CUE)
        score = self._sam_signed_score(self.current_index, self.current_view)
        if score is None:
            weighted_sam_full = torch.zeros((height, width), dtype=torch.float32)
            sam_support = torch.zeros(target_shape, dtype=torch.bool)
        else:
            weighted_sam_full, _ = q_weighted_upsampled_signed_sam_score(
                score,
                torch.clamp(
                    self.current_view.candidate_map / self.args.cue_scale,
                    0.0,
                    1.0,
                )[0].detach().cpu(),
                height=height,
                width=width,
            )
            positive_sam, negative_sam = threshold_signed_support(
                weighted_sam_full,
                threshold=PANEL7_DEPTH_SUPPORT_THRESHOLD,
            )
            sam_support_full = positive_sam | negative_sam
            sam_support = F.interpolate(
                sam_support_full.float()[None, None],
                size=target_shape,
                mode="nearest",
            )[0, 0].bool()
        sam_support &= reference_alpha >= DEPTH_ALIGNMENT_MIN_REFERENCE_ALPHA
        image, stats = rendered_gs_online_da3_depth_difference_image(
            online_depth,
            rendered_depth,
            alignment_mask,
            sam_support,
            width=width,
            height=height,
            depth_difference_threshold=DEPTH_DIFFERENCE_SUPPORT_THRESHOLD,
        )
        aligned_depth = online_depth * stats.alignment_scale
        depth_valid = (
            sam_support
            & torch.isfinite(rendered_depth)
            & torch.isfinite(aligned_depth)
            & (rendered_depth > 0)
            & (aligned_depth > 0)
        )
        new_mask, _, _ = typed_change_cue_masks(
            learned_q_full, weighted_sam_full,
            rendered_depth - aligned_depth, depth_valid,
        )
        self._typed_new_mask_cache[self.current_index] = new_mask
        self._typed_depth_cache[self.current_index] = (
            aligned_depth.detach().cpu(), rendered_depth.detach().cpu(),
            torch.from_numpy(online_K.copy()).float(), w2c.detach().cpu(),
        )
        self._cue_type_cache[self.current_index] = typed_change_cue_image(
            learned_q_full,
            weighted_sam_full,
            rendered_depth - aligned_depth,
            depth_valid,
        )
        self._depth_difference_cache[self.current_index] = (image, stats)
        return image.copy(), stats

    def cue_type_rgb(self) -> tuple[np.ndarray, CueTypeStats]:
        """Display-only NEW/REMOVE/APPEARANCE partition of the current soft Q."""

        cue = self._normalized_cue_target(self.current_view)[0].detach().cpu()
        if self.current_index < 0:
            return np.zeros((*cue.shape, 3), dtype=np.uint8), CueTypeStats(
                background=cue.numel()
            )
        self.rendered_gs_online_da3_depth_difference_rgb()
        cached = self._cue_type_cache.get(self.current_index)
        if cached is not None:
            return cached[0].copy(), cached[1]
        # Missing depth analysis cannot assert NEW/REMOVE; show Q as uncertain.
        return typed_change_cue_image(
            cue, torch.zeros_like(cue), torch.zeros_like(cue),
            torch.zeros_like(cue, dtype=torch.bool),
        )

    def gt_change_rgb(self) -> np.ndarray:
        if getattr(self.args, "gt_format", "objects") == "binary":
            return _mask_image(self.current_view.gt_change_mask, GT_CHANGE_COLOR)
        return compose_gt_change_image(
            self.current_view.gt_add_mask, self.current_view.gt_remove_mask
        )

    def _projected_evidence_score(self) -> torch.Tensor:
        if self.latest_evidence is None:
            return torch.zeros(self.count, device=self.device, dtype=self.base.get_xyz.dtype)
        return self.latest_evidence.delta_a[:self.count].clamp(0.0, 1.0)

    def _bayes_factor_progress_score(self) -> torch.Tensor:
        progress = normalized_log_bayes_factor(
            self.tracker.last_log_bayes_factor[:self.count],
            self.args.bayes_factor_threshold,
        )
        first_bf = getattr(self.args, "first_open_bayes_factor_threshold", None)
        if first_bf is not None:
            first_progress = normalized_log_bayes_factor(
                self.tracker.last_log_bayes_factor[:self.count], first_bf
            )
            progress = torch.where(self.lifecycle.never_open_mask(self.current_index),
                                   first_progress, progress)
        return torch.where(
            self.tracker.candidate_active[:self.count], progress, torch.zeros_like(progress)
        )

    @torch.no_grad()
    def learned_change_score(self) -> torch.Tensor:
        """Render current valid base DC together with born DA3 seed state."""

        from gaussian_renderer import render_change
        from temporal.new_seed_gaussians import build_concatenated_change_view

        if self.uses_typed_partition and self.current_index >= 0:
            from experiments.panel10_split_training import build_partition_view

            model = build_partition_view(
                self, timestamp=self.current_index, branch="joint", train=False,
            )
            package = render_change(
                self.current_view, model, self.pipe, self.evidence_background,
                clamp_output=False,
            )
            return package["render"].mean(dim=0, keepdim=True).clamp(0.0, 1.0)
        base_dc, base_opacity, _ = self._effective_base_change_attributes()
        model: Any = self.base
        override_dc: torch.Tensor | None = base_dc.detach()
        override_opacity: torch.Tensor | None = base_opacity.detach()
        override_xyz: torch.Tensor | None = self.representation_xyz.detach()
        override_scaling: torch.Tensor | None = self.base.scaling_activation(
            self.representation_scaling.detach()
        )
        override_rotation: torch.Tensor | None = self.base.rotation_activation(
            self.representation_rotation.detach()
        )
        if self.current_index >= 0 and self.seed_model.num_gaussians:
            seed_active, seed_never_open = self._seed_lifecycle_masks(
                self.current_index
            )
            model = build_concatenated_change_view(
                self.base,
                self.seed_model,
                timestamp=float(self.current_index),
                base_dc=base_dc,
                base_opacity=base_opacity,
                detach_base_dc=True,
                seed_attribute_mode="lifecycle",
                seed_lifecycle_active=seed_active,
                seed_lifecycle_never_open=seed_never_open,
            )
            override_dc = None
            override_opacity = None
            override_xyz = None
            override_scaling = None
            override_rotation = None
        package = render_change(
            self.current_view,
            model,
            self.pipe,
            self.evidence_background,
            override_dc=override_dc,
            override_opacity=override_opacity,
            override_xyz=override_xyz,
            override_scaling=override_scaling,
            override_rotation=override_rotation,
            clamp_output=False,
        )
        return package["render"].mean(dim=0, keepdim=True).clamp(0.0, 1.0)

    def colors_for_display(self, display: str) -> torch.Tensor:
        if display == DISPLAY_LIFECYCLE:
            return self._visual_state().colors
        if display == DISPLAY_PROJECTED:
            return black_green_yellow_red_heatmap(self._projected_evidence_score())
        if display == DISPLAY_BAYESIAN:
            return black_green_yellow_red_heatmap(
                self._bayes_factor_progress_score()
            )
        raise ValueError(f"unknown display layer: {display}")

    @torch.no_grad()
    def render_display(self, display: str) -> np.ndarray:
        from gaussian_renderer import render_change
        from temporal.new_seed_gaussians import build_concatenated_change_view

        if display == DISPLAY_LEARNED:
            return _score_heatmap_image(self.learned_change_score())
        colors = self.colors_for_display(display)
        model: Any = self.base
        override_opacity = self.base.get_opacity.detach()
        override_xyz = self.base.get_xyz.detach()
        override_scaling = self.base.get_scaling.detach()
        override_rotation = self.base.get_rotation.detach()
        if (
            display == DISPLAY_LIFECYCLE
            and self.current_index >= 0
            and self.seed_model.num_gaussians
        ):
            # Panel 1 must cover the same base+DA3 population as panel 5.
            # Include all born seed rows so CLOSED rows remain visible in red.
            model = build_concatenated_change_view(
                self.base,
                self.seed_model,
                timestamp=float(self.current_index),
                seed_attribute_mode="all",
            )
            colors = torch.cat((colors, self._seed_visual_state().colors), dim=0)
            override_opacity = model.get_opacity.detach()
            override_xyz = model.get_xyz.detach()
            override_scaling = model.get_scaling.detach()
            override_rotation = model.get_rotation.detach()
        package = render_change(
            self.current_view,
            model,
            self.pipe,
            self.viewer_background,
            override_color=colors,
            override_opacity=override_opacity,
            override_xyz=override_xyz,
            override_scaling=override_scaling,
            override_rotation=override_rotation,
            clamp_output=True,
        )
        return _rgb_u8(package["render"])


class BayesianDetectorViewer:
    """Viser GUI around ``BayesianDetectorReplay``."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.replay = BayesianDetectorReplay(args)
        self.lock = threading.Lock()
        self.display = DISPLAY_LIFECYCLE
        self.main_image = self.replay.render_display(self.display)
        self.detector_state_image = self.replay.render_display(DISPLAY_BAYESIAN)
        learned_score = self.replay.learned_change_score()
        self.learned_change_image = _score_heatmap_image(learned_score)
        self.predicted_change_mask_image = binary_change_mask_image(
            learned_score, threshold=0.5
        )
        self.input_image, self.da3_seed_stats = (
            self.replay.input_with_da3_seeds_rgb()
        )
        self.cue_image = self.replay.cue_rgb()
        self.gt_change_image = self.replay.gt_change_rgb()
        self.sam_feature_diff_image, self.sam_feature_diff_stats = (
            self.replay.sam_feature_diff_rgb()
        )
        self.depth_difference_image, self.depth_difference_stats = (
            self.replay.rendered_gs_online_da3_depth_difference_rgb()
        )
        self.cue_type_image, self.cue_type_stats = self.replay.cue_type_rgb()
        self._client_aspects: dict[int, float] = {}
        self._frame_history: dict[int, ViewerFrameSnapshot] = {}
        self._viewed_index = self.replay.current_index
        self._history_panels: tuple[tuple[str, np.ndarray], ...] | None = None

        self.server = viser.ViserServer(host=args.host, port=args.port)
        self._setup_gui()
        self._remember_current_frame()
        self._setup_callbacks()
        print("\nBayesian detector viewer")
        print(f"  http://localhost:{args.port}")
        print(
            "  Press 'Next cue →' for one detector observation plus "
            f"{args.representation_updates} causal representation updates.\n"
        )

    def _setup_gui(self) -> None:
        with self.server.gui.add_folder("Causal detector replay", expand_by_default=True):
            self.gui_previous = self.server.gui.add_button(
                "← Previous", disabled=True,
                hint="Show the previous saved frame without rewinding training or detector state",
            )
            self.gui_next = self.server.gui.add_button(
                "Next cue →",
                color="green",
                hint=(
                    "Browse the next saved frame, or at the latest frame consume "
                    "one new cue and train R_change for "
                    f"{self.args.representation_updates} causal random updates"
                ),
            )
            self.gui_reset = self.server.gui.add_button(
                "Reset to reference initialization", color="red"
            )
            self.gui_display = self.server.gui.add_dropdown(
                "Main Gaussian layer",
                options=DISPLAY_OPTIONS,
                initial_value=DISPLAY_LIFECYCLE,
            )
            self.gui_status = self.server.gui.add_markdown(
                format_status(
                    None,
                    consumed_frames=0,
                    total_frames=self.replay.total_frames,
                    bayes_factor_threshold=self.args.bayes_factor_threshold,
                    first_open_bayes_factor_threshold=self.args.first_open_bayes_factor_threshold,
                )
            )
            cue_contract = (
                f"binary: raw cue > {self.args.cue_threshold:g}"
                if self.args.cue_mode == "binary"
                else f"soft: clamp(raw cue / {self.args.cue_scale:g}, 0, 1)"
            )
            if self.args.detector_pixel_cue == "binary_q05":
                cue_contract = "binary 1[normalized Q > 0.5]; training/Panel3 retain soft Q"
            if self.args.detector_gaussian_cue == "binary_q05":
                cue_contract = "soft pixels; Gaussian 1[E+ > E-] after alpha-T; capped mass retained"
            if self.args.cue_remap == "identity":
                remap_contract = "identity"
            elif self.args.cue_remap == "smoothstep_band":
                remap_contract = (
                    f"q≤{self.args.soft_band_low:g}→0; "
                    f"q≥{self.args.soft_band_high:g}→1; smoothstep between"
                )
            else:
                remap_contract = (
                    "Stage-2 histogram MLP sigmoid: per-frame learned τ,width; "
                    f"artifact={self.args.cue_boundary_json}"
                )
            fusion_contract = (
                "P + S"
                if self.args.cue_fusion == "sum"
                else (
                    f"2 · P^{self.args.product_exponent:g} · S"
                    if self.args.cue_fusion == "power_product"
                    else (
                        "2 · norm(0.8·L1^"
                        f"{self.args.product_exponent:g} + 0.2·(1−SSIM)) · S"
                    )
                )
            )
            self.gui_cue_mode = self.server.gui.add_markdown(
                f"**Cue fusion:** `{fusion_contract}`  \n"
                f"**Cue sharpening:** `{remap_contract}`  \n"
                f"**Detector cue mode:** `{cue_contract}`  \n"
                f"**First OPEN / CLOSE,REOPEN BF:** `"
                f"{self.args.first_open_bayes_factor_threshold or self.args.bayes_factor_threshold:g} / "
                f"{self.args.bayes_factor_threshold:g}`  \n"
                f"**Representation replay:** `{self.args.representation_updates} updates; "
                f"latest branch p={self.args.current_view_probability:g}, otherwise "
                "uniform observed-frame sampling`  \n"
                f"**Training partition:** `{self.args.training_partition}`  \n"
                f"**SSF amplitude / legacy coverage amplitude:** `"
                f"{self.args.representation_cue_amplitude:g}Q / "
                f"{self.args.geometry_cue_amplitude:g}Q`  \n"
                f"**NEVER_OPEN geometry:** `"
                f"{'train' if self.args.train_never_open_geometry else 'frozen'}`"
            )
            self.gui_cue_types = self.server.gui.add_markdown(
                self._cue_type_status()
            )
            if self.replay.uses_typed_partition:
                self.server.gui.add_markdown(
                    "**Joint BF30:** current DA3 birth → one fixed base+seed "
                    "alpha-T field + full Q → one tracker → training.  \n"
                    "**Partitioned SSF:** `Q_NEW=M_NEW*Q; Q_BASE=Q-Q_NEW`.  \n"
                    "Seed DC + geometry share NEW SSF; base DC uses remaining SSF. "
                    "Both branches include all frozen black NEVER_OPEN occluders. "
                    "Final mask is one joint base+seed alpha-composited render.  \n"
                    f"Seed clone/split at update {self.args.da3_density_update}; "
                    f"max {self.args.da3_density_max_children} children/event, "
                    f"archive cap: {self.args.da3_max_rows or 'unlimited'}. "
                    "Low-opacity OPEN seed pruning retires history, not memory."
                )

        with self.server.gui.add_folder("Legend", expand_by_default=True):
            seed_cue_description = (
                f"the same post-remap {self.args.cue_mode} cue and normalization "
                "as the base detector"
                if self.args.da3_detector_cue_source == "shared"
                else "legacy untouched cached P+S > "
                f"{self.args.da3_detector_cue_threshold:g} binary evidence"
            )
            self.server.gui.add_markdown(
                "- **black**: committed NEVER_OPEN\n"
                "- **green**: committed OPEN\n"
                "- **red**: committed CLOSED after a previous OPEN\n\n"
                "Cue and BF heatmaps: black → green → yellow → red as the normalized "
                "value approaches 1.\n\n"
                "DA3 overlay: green centers are previously accepted seeds; red centers "
                "with a white rim were born at the current frame. Seed birth uses only "
                "the current/past causal window. NEW support intersects "
                "panel 8 > +0.1 with panel 9's depth residual > +0.03. "
                "It has no separate hard Q cutoff; viewer "
                "typed birth uses current aligned DA3 depth, with per-frame/total "
                "budgets and mature or pending seed coverage suppression.\n\n"
                "SAM feature diff × Q: the signed 64x64 PC1 response is normalized, "
                "bilinearly upsampled, then multiplied by panel 3's learned Q. "
                "Positive values are red, negative values are blue, zero is black; "
                "there is no top-magnitude cutoff or hard Q gate.\n\n"
                "Depth difference: panel 9 positive-scale aligns the current online "
                "DA3Metric depth directly to immutable rendered GS camera-z using "
                "only pixels where reference alpha is at least "
                f"{DEPTH_ALIGNMENT_MIN_REFERENCE_ALPHA:g} and learned Q is below "
                f"{DEPTH_ALIGNMENT_MAX_CUE:g}. Rendered GS depth minus the aligned "
                "online depth is displayed only where |panel 8| > "
                f"{PANEL7_DEPTH_SUPPORT_THRESHOLD:g} and |depth residual| > "
                f"{DEPTH_DIFFERENCE_SUPPORT_THRESHOLD:g}, with signed q95 "
                "normalization recomputed over that overlap every frame. "
                "Red means the online surface is in front, blue means behind, and "
                "black means zero or invalid.\n\n"
                "Panel 10 partitions panel 3's soft Q (panel10_new uses this for training): "
                "red NEW requires both signed SAM and depth to be strongly positive; "
                "blue REMOVE requires both strongly negative; yellow "
                "APPEARANCE/uncertain covers conflicting, weak, or invalid signals. "
                "Thresholds are |SAM x Q| > 0.1 and |depth residual| > 0.03; "
                "brightness is Q, not another hard Q mask.\n\n"
                "GT Change is the evaluation-only union ADD ∪ REMOVE and never enters "
                "the detector. Current alpha-T cue projection is the observation; the "
                "dedicated detector-state panel shows stable p(flip), or normalized "
                "RESET-vs-KEEP log-BF while a candidate is live.\n\n"
                "Learned R_change: current OPEN base rows learn DC only. DA3 rows "
                "are materialized as NEVER_OPEN only on their causal birth frame; "
                f"their fixed birth geometry receives {seed_cue_description} "
                "for BF30 OPEN/CLOSE "
                "updates. "
                "OPEN DA3 rows learn DC, xyz, opacity, scale, and rotation. "
                "In panel10_new every NEVER_OPEN attribute stays frozen; legacy "
                "geometry ablations are opt-in. The detector probe remains fixed "
                "after each row's birth and consumes only new raw-Q observations. "
                "CLOSED rows are omitted. Panel 5 shows the continuous raw score; "
                "panel 6 applies the unchanged >=0.5 final-mask threshold exactly."
            )

    def _setup_callbacks(self) -> None:
        @self.server.on_client_connect
        def _on_connect(client: viser.ClientHandle) -> None:
            self._push_background_to_client(client, force=True)

            @client.camera.on_update
            def _on_camera_update(camera: viser.CameraHandle) -> None:
                del camera
                self._push_background_to_client(client, force=False)

        @self.gui_display.on_update
        def _on_display(event: viser.GuiEvent) -> None:
            del event
            with self.lock:
                if self._viewed_index != self.replay.current_index:
                    return  # A historical snapshot must not render current GS state.
                self.display = self.gui_display.value
                self.main_image = self.replay.render_display(self.display)
                self._remember_current_frame()
                self._push_background()

        @self.gui_previous.on_click
        def _on_previous(event: viser.GuiEvent) -> None:
            del event
            if not self.lock.acquire(blocking=False):
                return
            try:
                self._show_saved_frame(self._viewed_index - 1)
            finally:
                self._sync_navigation_buttons()
                self.lock.release()

        @self.gui_next.on_click
        def _on_next(event: viser.GuiEvent) -> None:
            del event
            if not self.lock.acquire(blocking=False):
                return
            self.gui_next.disabled = True
            self.gui_previous.disabled = True
            try:
                if self._viewed_index < self.replay.current_index:
                    self._show_saved_frame(self._viewed_index + 1)
                    return
                self.gui_status.content = (
                    "### Processing next cue and "
                    f"{self.args.representation_updates} representation updates..."
                )
                self.replay.step()
                self._refresh_images()
            except StopIteration:
                self.gui_status.content = "### Replay complete"
            except Exception as error:
                self.gui_status.content = f"### Error\n`{type(error).__name__}: {error}`"
                raise
            finally:
                self._sync_navigation_buttons()
                self.lock.release()

        @self.gui_reset.on_click
        def _on_reset(event: viser.GuiEvent) -> None:
            del event
            if not self.lock.acquire(blocking=False):
                return
            self.gui_next.disabled = True
            self.gui_previous.disabled = True
            self.gui_status.content = "### Resetting detector..."
            try:
                self.replay.reset()
                self._frame_history.clear()
                self._refresh_images()
            finally:
                self._sync_navigation_buttons()
                self.lock.release()

    def _remember_current_frame(self) -> None:
        """Freeze displayed pixels; navigation must never rerun an observation."""
        self._viewed_index = self.replay.current_index
        self._history_panels = None
        self._frame_history[self._viewed_index] = ViewerFrameSnapshot.capture(
            self._dashboard_panels(), self.gui_status.content,
            self.gui_cue_types.content,
        )
        self._sync_navigation_buttons()

    def _sync_navigation_buttons(self) -> None:
        historical = self._viewed_index < self.replay.current_index
        self.gui_previous.disabled = self._viewed_index - 1 not in self._frame_history
        self.gui_next.disabled = not (historical or self.replay.has_next)
        # Only the saved main-layer image is available for older frames.
        self.gui_display.disabled = historical

    def _show_saved_frame(self, index: int) -> bool:
        snapshot = self._frame_history.get(index)
        if snapshot is None:
            return False
        self._viewed_index = index
        self._history_panels = (
            snapshot.decoded_panels() if index < self.replay.current_index else None
        )
        self.gui_status.content = snapshot.status
        if index < self.replay.current_index:
            self.gui_status.content += (
                f"\n\n**Saved view t={index}; processed through "
                f"t={self.replay.current_index}.** Detector/training state unchanged."
            )
        self.gui_cue_types.content = snapshot.cue_status
        self._sync_navigation_buttons()
        self._push_background()
        return True

    def _push_background(self) -> None:
        for client in self.server.get_clients().values():
            self._push_background_to_client(client, force=True)

    def _cue_type_status(self) -> str:
        stats = self.cue_type_stats
        purpose = (
            "NEW/base training partition"
            if getattr(self.args, "training_partition", "legacy") == "panel10_new"
            else "display only"
        )
        return (
            f"**Panel 10 — cue types ({purpose})**  \n"
            f"🔴 NEW (+/+): **{stats.new:,}**  \n"
            f"🔵 REMOVE (−/−): **{stats.removed:,}**  \n"
            f"🟡 APPEARANCE / uncertain: **{stats.appearance:,}**  \n"
            "Strong evidence: `|SAM×Q| > 0.1`, `|GS−sDA3| > 0.03`.  \n"
            "Brightness = soft Q; weak/conflicting/invalid signals stay yellow."
        )

    def _dashboard_panels(self) -> tuple[tuple[str, np.ndarray], ...]:
        if getattr(self, "_history_panels", None) is not None:
            return self._history_panels
        display_label = {
            DISPLAY_LIFECYCLE: "1. Gaussian lifecycle (base + DA3)",
            DISPLAY_PROJECTED: "1. Projected cue layer",
            DISPLAY_BAYESIAN: "1. Bayesian instability",
            DISPLAY_LEARNED: "1. Learned R_change prediction",
        }[self.display]
        cue_label = (
            "3. Binary 2D change cue"
            if self.args.cue_mode == "binary"
            else (
                "3. Smoothstep sharp-soft cue"
                if self.args.cue_remap == "smoothstep_band"
                else "3. Learned sigmoid cue"
                if self.args.cue_remap == "learned_sigmoid"
                else (
                    "3. Soft P+S cue"
                    if self.args.cue_fusion == "sum"
                    else (
                        f"3. Soft P^{self.args.product_exponent:g} x SAM"
                        if self.args.cue_fusion == "power_product"
                        else f"3. Soft L1^{self.args.product_exponent:g} pixel x SAM"
                    )
                )
            )
        )
        panels = [
            (display_label, self.main_image),
            (
                (
                    "2. Online RGB"
                    if self.replay.da3_seeds is None
                    else (
                        "2. Online RGB + DA3 seeds "
                        f"(all {self.da3_seed_stats.accepted_so_far:,}; "
                        f"new +{self.da3_seed_stats.born_now:,})"
                    )
                ),
                self.input_image,
            ),
            (cue_label, self.cue_image),
            ("4. BF (Bayes Factor)", self.detector_state_image),
            (
                "5. Current-valid learned R_change (continuous raw score)",
                self.learned_change_image,
            ),
            (
                "6. Predicted change mask (raw R_change >= 0.5)",
                self.predicted_change_mask_image,
            ),
            (("7. GT Change (binary union)" if getattr(self.args, "gt_format", "objects") == "binary"
              else "7. GT Change (ADD union REMOVE)"), self.gt_change_image),
        ]
        if self.replay.sam_model is not None:
            panels.append(
                (
                    "8. Upsampled signed SAM diff x learned Q "
                    "(red + / blue -; black excluded from seed support) "
                    f"(norm={self.sam_feature_diff_stats.normalization_scale:.3g})",
                    self.sam_feature_diff_image,
                )
            )
        else:
            panels.append(
                ("8. Signed SAM diff x learned Q (unavailable)", np.zeros_like(self.input_image))
            )
        panels.append(
            (
                "9. GS-sDA3 on |panel8|>0.1 and |depth diff|>0.03 "
                f"s={self.depth_difference_stats.alignment_scale:.2f} R+/B-; "
                "per-frame q95 norm; red front / blue behind; "
                f"inliers={self.depth_difference_stats.alignment_inliers:,}/"
                f"{self.depth_difference_stats.alignment_samples:,}; "
                f"med|d|={self.depth_difference_stats.median_absolute_difference:.3g}; "
                f"q95={self.depth_difference_stats.normalization_scale:.3g})",
                self.depth_difference_image,
            )
        )
        panels.append(("10. Cue types (N/R/A)", self.cue_type_image))
        for panel_number in range(11, 15):
            panels.append((f"{panel_number}. Reserved", np.zeros_like(self.input_image)))
        return tuple(panels)

    def _push_background_to_client(
        self, client: viser.ClientHandle, *, force: bool
    ) -> None:
        aspect = float(client.camera.aspect)
        if not math.isfinite(aspect) or aspect <= 0.0:
            aspect = DEFAULT_DASHBOARD_ASPECT
        key = id(client)
        previous = self._client_aspects.get(key)
        if not force and previous is not None and abs(previous - aspect) < 1e-3:
            return
        dashboard = compose_two_row_dashboard(
            self._dashboard_panels(),
            width=self.args.dashboard_width,
            aspect=aspect,
        )
        client.scene.set_background_image(dashboard, format="jpeg", jpeg_quality=90)
        self._client_aspects[key] = aspect

    def _refresh_images(self) -> None:
        self._viewed_index = self.replay.current_index
        self._history_panels = None
        self.main_image = self.replay.render_display(self.display)
        self.detector_state_image = self.replay.render_display(DISPLAY_BAYESIAN)
        learned_score = self.replay.learned_change_score()
        self.learned_change_image = _score_heatmap_image(learned_score)
        self.predicted_change_mask_image = binary_change_mask_image(
            learned_score, threshold=0.5
        )
        self.input_image, self.da3_seed_stats = (
            self.replay.input_with_da3_seeds_rgb()
        )
        self.cue_image = self.replay.cue_rgb()
        self.gt_change_image = self.replay.gt_change_rgb()
        self.sam_feature_diff_image, self.sam_feature_diff_stats = (
            self.replay.sam_feature_diff_rgb()
        )
        self.depth_difference_image, self.depth_difference_stats = (
            self.replay.rendered_gs_online_da3_depth_difference_rgb()
        )
        self.cue_type_image, self.cue_type_stats = self.replay.cue_type_rgb()
        self.gui_cue_types.content = self._cue_type_status()
        self.gui_status.content = format_status(
            self.replay.latest_summary,
            consumed_frames=self.replay.current_index + 1,
            total_frames=self.replay.total_frames,
            bayes_factor_threshold=self.args.bayes_factor_threshold,
            first_open_bayes_factor_threshold=self.args.first_open_bayes_factor_threshold,
        )
        self._remember_current_frame()
        self._push_background()
        self._capture_current_frame()

    def _capture_current_frame(self) -> None:
        if self.args.capture_dir is None:
            return
        self.args.capture_dir.mkdir(parents=True, exist_ok=True)
        stem = (
            "initialization"
            if self.replay.current_index < 0
            else f"{self.replay.current_index:06d}"
        )
        Image.fromarray(self.main_image, mode="RGB").save(
            self.args.capture_dir / f"{stem}_main.png"
        )
        Image.fromarray(self.detector_state_image, mode="RGB").save(
            self.args.capture_dir / f"{stem}_detector_state.png"
        )
        Image.fromarray(self.learned_change_image, mode="RGB").save(
            self.args.capture_dir / f"{stem}_learned_rchange.png"
        )
        Image.fromarray(self.predicted_change_mask_image, mode="RGB").save(
            self.args.capture_dir / f"{stem}_prediction_mask.png"
        )
        if self.replay.sam_model is not None:
            Image.fromarray(self.sam_feature_diff_image, mode="RGB").save(
                self.args.capture_dir / f"{stem}_sam_feature_diff.png"
            )
        Image.fromarray(self.depth_difference_image, mode="RGB").save(
            self.args.capture_dir / f"{stem}_depth_difference.png"
        )
        Image.fromarray(self.cue_type_image, mode="RGB").save(
            self.args.capture_dir / f"{stem}_cue_types.png"
        )
        dashboard = compose_two_row_dashboard(
            self._dashboard_panels(),
            width=self.args.dashboard_width,
            aspect=DEFAULT_DASHBOARD_ASPECT,
        )
        Image.fromarray(dashboard, mode="RGB").save(
            self.args.capture_dir / f"{stem}_dashboard.png"
        )
        if self.replay.latest_summary is not None:
            from dataclasses import asdict

            (self.args.capture_dir / f"{stem}_summary.json").write_text(
                json.dumps(asdict(self.replay.latest_summary), indent=2) + "\n"
            )

    def run(self) -> None:
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nStopping Bayesian detector viewer")
        finally:
            self.server.stop()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and nonnegative")
    return parsed


def probability(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be in [0,1]")
    return parsed


def bayes_factor(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be finite and greater than one")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", type=Path, default=Path(DEFAULT_SOURCE))
    parser.add_argument("--gt-format", choices=("objects", "binary"), default="objects",
                        help="Evaluation/display only: per-object ESCD annotations or PASLCD binary gt_mask.")
    parser.add_argument(
        "--fixed-cameras-json", type=Path, default=Path(DEFAULT_FIXED_CAMERAS)
    )
    parser.add_argument("--cue-cache-root", type=Path, default=Path(DEFAULT_CUE_CACHE))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=positive_int, default=8090)
    parser.add_argument("--resolution", type=positive_float, default=4.0)
    parser.add_argument("--max-frames", type=positive_int, default=None)
    parser.add_argument("--max-states", type=positive_int, default=16)
    parser.add_argument(
        "--lifespan-state-cache", choices=("off", "snapshot"), default="off",
        help="Panel10 ablation: compact finalized states and cache decoded replay masks.",
    )
    parser.add_argument("--cue-mode", choices=("binary", "soft"), default="binary")
    parser.add_argument(
        "--cue-fusion",
        choices=("sum", "power_product", "l1_power_product"),
        default="sum",
        help=(
            "Use cached P+S, historical 2*P^alpha*S, or apply alpha only to "
            "L1 before the pixel mix and multiply by S"
        ),
    )
    parser.add_argument(
        "--product-exponent",
        type=positive_float,
        default=0.3,
        help="Pixel-cue exponent for --cue-fusion power_product",
    )
    parser.add_argument("--cue-threshold", type=probability, default=0.5)
    parser.add_argument(
        "--cue-scale",
        type=positive_float,
        default=2.0,
        help="Soft mode divisor mapping the cached O-SCD 0..2 cue into [0,1]",
    )
    parser.add_argument(
        "--cue-remap",
        choices=("identity", "smoothstep_band", "learned_sigmoid"),
        default="identity",
        help="Optional post-fusion soft binarization before alpha-T evidence",
    )
    parser.add_argument("--soft-band-low", type=probability, default=0.15)
    parser.add_argument("--soft-band-high", type=probability, default=0.35)
    parser.add_argument(
        "--cue-boundary-json",
        type=Path,
        default=None,
        help="Per-frame Stage-2 boundary artifact for --cue-remap learned_sigmoid",
    )
    parser.add_argument("--evidence-mass-saturation", type=positive_float, default=1.0)
    parser.add_argument("--min-evidence-mass", type=nonnegative_float, default=1e-6)
    parser.add_argument("--bayes-factor-threshold", type=bayes_factor, default=30.0)
    parser.add_argument("--first-open-bayes-factor-threshold", type=bayes_factor, default=None,
                        help="Optional BF for NEVER_OPEN only; CLOSE/REOPEN retain --bayes-factor-threshold.")
    parser.add_argument("--detector-pixel-cue", choices=("unchanged", "binary_q05"), default="unchanged",
                        help="Detector-only Q > 0.5 before alpha-T aggregation; training/birth keep soft Q.")
    parser.add_argument("--detector-gaussian-cue", choices=("unchanged", "binary_q05"), default="unchanged",
                        help="Threshold current-frame soft alpha-T Gaussian ratio > 0.5, then assign capped mass to one side.")
    parser.add_argument("--stable-flip-prior", type=positive_float, default=1.0)
    parser.add_argument("--stable-keep-prior", type=positive_float, default=10.0)
    parser.add_argument("--reset-flip-prior", type=positive_float, default=1.0)
    parser.add_argument("--reset-keep-prior", type=positive_float, default=1.0)
    parser.add_argument("--filter-chunk-size", type=positive_int, default=65536)
    parser.add_argument("--dashboard-width", type=positive_int, default=1920)
    parser.add_argument(
        "--representation-updates",
        type=positive_int,
        default=16,
        help="Online R_change optimizer steps after each detector observation",
    )
    parser.add_argument("--representation-seed", type=int, default=0)
    parser.add_argument(
        "--current-view-probability",
        type=probability,
        default=0.33,
        help=(
            "Per-update probability of explicitly selecting the latest view; "
            "the other branch is uniform over every observed view"
        ),
    )
    parser.add_argument(
        "--representation-dc-lr", type=nonnegative_float, default=2.5e-3
    )
    parser.add_argument(
        "--representation-cue-amplitude",
        type=positive_float,
        default=1.0,
        help=(
            "Multiplier applied only to normalized Q inside the representation "
            "DC SSF loss. Detector evidence and geometry supervision remain unit-Q."
        ),
    )
    parser.add_argument(
        "--geometry-cue-amplitude",
        type=positive_float,
        default=1.0,
        help=(
            "Multiplier applied to normalized Q inside base/DA3 geometry "
            "coverage losses. Detector evidence and seed birth remain unit-Q."
        ),
    )
    parser.add_argument(
        "--seed-dc-supervision",
        choices=("joint", "seed_only_ssf", "projected_bce"),
        default="joint",
        help=(
            "Train OPEN DA3 DC through the joint base+seed SSF render, or replace "
            "only its DC gradient with a base-free seed-only SSF render or the "
            "validated projected per-seed BCE surrogate."
        ),
    )
    parser.add_argument(
        "--seed-dc-loss-weight",
        type=nonnegative_float,
        default=1.0,
        help=(
            "Weight of the replacement DA3 DC loss when --seed-dc-supervision "
            "is seed_only_ssf or projected_bce."
        ),
    )
    parser.add_argument(
        "--dc-replay-mode",
        choices=("sampled", "current"),
        default="sampled",
        help=(
            "Use each sampled historical view together with the Gaussian "
            "lifespans valid at that view timestamp, or keep the identical "
            "causal geometry replay schedule while supervising DC only on the "
            "latest view."
        ),
    )
    parser.add_argument(
        "--base-geometry-scope",
        choices=("frozen", "open", "open_and_never_open"),
        default="frozen",
        help=(
            "Base-only geometry coverage refinement scope. The detector keeps "
            "using immutable reference geometry."
        ),
    )
    parser.add_argument(
        "--base-geometry-target",
        choices=("change", "signed_new"),
        default="change",
        help=(
            "Coverage target for base geometry refinement: the complete learned-Q "
            "change cue or only the globally locked signed SAM/PCA NEW support."
        ),
    )
    parser.add_argument(
        "--representation-xyz-lr", type=nonnegative_float, default=1.6e-4
    )
    parser.add_argument(
        "--representation-scale-lr", type=nonnegative_float, default=5.0e-3
    )
    parser.add_argument(
        "--representation-rotation-lr", type=nonnegative_float, default=1.0e-3
    )
    parser.add_argument(
        "--representation-opacity-lr", type=nonnegative_float, default=2.5e-2
    )
    parser.add_argument(
        "--train-never-open-base-opacity",
        action="store_true",
        help=(
            "In the base-only geometry ablation, also optimize visible "
            "NEVER_OPEN opacity while keeping its change DC fixed black."
        ),
    )
    parser.add_argument(
        "--base-geometry-inside-weight", type=nonnegative_float, default=1.0
    )
    parser.add_argument(
        "--base-geometry-outside-weight", type=nonnegative_float, default=0.1
    )
    parser.add_argument(
        "--base-max-displacement-ratio", type=positive_float, default=4.0
    )
    parser.add_argument("--base-min-scale-ratio", type=positive_float, default=0.25)
    parser.add_argument("--base-max-scale-ratio", type=positive_float, default=4.0)
    parser.add_argument(
        "--probe-scaling", choices=("native", "isotropic_min"), default="native"
    )
    parser.add_argument("--capture-dir", type=Path, default=None)
    parser.add_argument(
        "--da3-seed-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional causal DA3 seed checkpoint; accepted centers are overlaid on "
            "the online RGB according to metadata.frame_global"
        ),
    )
    parser.add_argument(
        "--da3-seed-marker-radius",
        type=positive_int,
        default=2,
        help="Radius in pixels for previously accepted DA3 seed centers",
    )
    parser.add_argument(
        "--da3-detector-cue-source",
        choices=("shared", "part19_binary"),
        default="shared",
        help=(
            "Share the base detector's post-remap cue/mode/scale (learned soft Q "
            "in the current preset), or explicitly reproduce historical Part19 "
            "untouched P+S binary evidence."
        ),
    )
    parser.add_argument(
        "--da3-detector-cue-threshold",
        type=probability,
        default=0.5,
        help=(
            "Legacy threshold used only with --da3-detector-cue-source "
            "part19_binary; shared mode uses the base detector settings"
        ),
    )
    parser.add_argument(
        "--train-never-open-geometry",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Refine NEVER_OPEN xyz, scale, and rotation with the causal NEW "
            "coverage loss while keeping DC, opacity, and detector geometry fixed"
        ),
    )
    parser.add_argument("--da3-initial-opacity", type=probability, default=0.10)
    parser.add_argument(
        "--da3-coverage-min-updates",
        type=positive_int,
        default=4,
        help=(
            "Geometry optimizer steps required before an existing DA3 seed can "
            "suppress a nearby new proposal"
        ),
    )
    parser.add_argument(
        "--da3-coverage-sigma",
        type=positive_float,
        default=2.0,
        help=(
            "Current learned max-axis Gaussian scale multiplier used as the "
            "adaptive proposal-suppression radius"
        ),
    )
    parser.add_argument(
        "--da3-birth-coverage",
        choices=("3d_only", "3d_plus_2d"),
        default="3d_only",
        help=(
            "Root-birth duplicate suppression: historical 3-D mature/pending "
            "sphere support only, or that same check followed by projected "
            "2-D covariance ellipse occupancy for DA3 roots."
        ),
    )
    parser.add_argument(
        "--da3-coverage-2d-sigma",
        type=positive_float,
        default=2.0,
        help="Projected covariance ellipse sigma used by --da3-birth-coverage 3d_plus_2d",
    )
    parser.add_argument("--da3-xyz-lr", type=nonnegative_float, default=1.6e-4)
    parser.add_argument("--da3-dc-lr", type=nonnegative_float, default=2.5e-3)
    parser.add_argument("--da3-opacity-lr", type=nonnegative_float, default=2.5e-2)
    parser.add_argument("--da3-scale-lr", type=nonnegative_float, default=5.0e-3)
    parser.add_argument("--da3-rotation-lr", type=nonnegative_float, default=1.0e-3)
    parser.add_argument(
        "--da3-geometry-inside-weight", type=nonnegative_float, default=1.0
    )
    parser.add_argument(
        "--da3-geometry-outside-weight", type=nonnegative_float, default=0.1
    )
    parser.add_argument(
        "--da3-max-displacement-ratio", type=positive_float, default=4.0
    )
    parser.add_argument("--da3-min-scale-ratio", type=positive_float, default=0.25)
    parser.add_argument("--da3-max-scale-ratio", type=positive_float, default=4.0)
    parser.add_argument("--da3-min-opacity", type=probability, default=0.01)
    parser.add_argument("--da3-max-opacity", type=probability, default=0.99)
    parser.add_argument(
        "--sam-sign-trace-root",
        type=Path,
        default=None,
        help=(
            "Optional scene_change*/causal_pca_posterior_arrays.npz root; "
            "enables the Part-9-style strong signed SAM cue overlay"
        ),
    )
    parser.add_argument(
        "--sam-new-sign",
        choices=("+", "-"),
        default=None,
        help=(
            "Optional causal global SAM/PCA sign lock for NEW. This permits "
            "signed-NEW supervision without materializing a DA3 seed sidecar."
        ),
    )
    parser.add_argument(
        "--sam-cue-threshold",
        type=probability,
        default=0.5,
        help="Historical raw O-SCD P+S support threshold for the SAM overlay",
    )
    parser.add_argument(
        "--sam-model",
        default="facebook/sam2.1-hiera-tiny",
        help="Local Hugging Face SAM2 model used by the signed-feature panel",
    )
    parser.add_argument(
        "--object-gt-root",
        type=Path,
        default=None,
        help="Directory containing scene_change*/object_change_annotations.json",
    )
    parser.add_argument(
        "--training-partition", choices=("legacy", "panel10_new"), default="legacy",
        help=("panel10_new splits Q into NEW seed/all-geometry SSF and remaining "
              "base/DC SSF, with online NEW depth birth and seed-only topology. "
              "Legacy coverage-loss ablations remain explicitly available."),
    )
    parser.add_argument(
        "--panel10-render-mode", choices=("split", "joint_channels"), default="split",
        help="Opt-in one-scene NEW/BASE channel rendering with shared occlusion; SSF targets stay separate.",
    )
    parser.add_argument("--da3-max-rows", type=nonnegative_int, default=20000,
                        help="Total archived seed rows (0: unlimited); per-frame and lineage limits still apply.")
    parser.add_argument("--da3-birth-max-per-frame", type=positive_int, default=1024)
    parser.add_argument("--da3-birth-stride", type=positive_int, default=4)
    parser.add_argument("--da3-density-update", type=positive_int, default=4)
    parser.add_argument("--da3-density-max-children", type=positive_int, default=128)
    parser.add_argument("--da3-density-grad-threshold", type=nonnegative_float, default=2e-4)
    parser.add_argument("--da3-density-abs-grad-threshold", type=nonnegative_float, default=1.2e-3)
    parser.add_argument("--da3-split-scale-ratio", type=positive_float, default=1.0)
    parser.add_argument("--da3-max-generation", type=positive_int, default=2)
    parser.add_argument("--da3-max-root-children", type=positive_int, default=32)
    parser.add_argument("--da3-prune-opacity", type=probability, default=0.02)
    parser.add_argument("--da3-prune-grace-frames", type=positive_int, default=3)
    parser.add_argument("--da3-prune-min-updates", type=positive_int, default=4)
    args = parser.parse_args(argv)
    if args.panel10_render_mode != "split" and args.training_partition != "panel10_new":
        parser.error("joint_channels requires --training-partition panel10_new")
    if args.detector_gaussian_cue == "binary_q05":
        if args.cue_mode != "soft" or args.detector_pixel_cue != "unchanged":
            parser.error("Gaussian binarization requires soft pixels, without pixel binarization")
        if args.da3_detector_cue_source != "shared":
            parser.error("Gaussian binarization requires the shared soft-cue detector")
    if args.training_partition == "panel10_new":
        if args.da3_seed_checkpoint is None or args.sam_sign_trace_root is None:
            parser.error("panel10_new requires DA3 depth-cache metadata and SAM sign trace")
        if args.train_never_open_geometry or args.train_never_open_base_opacity:
            parser.error("panel10_new keeps every NEVER_OPEN attribute frozen")
        if args.base_geometry_scope != "frozen":
            parser.error("panel10_new keeps base geometry frozen and trains base DC")
        if args.seed_dc_supervision != "joint":
            parser.error("panel10_new replaces legacy seed DC supervision with partitioned SSF")
        if args.da3_detector_cue_source != "shared":
            parser.error("panel10_new requires the shared raw-cue BF30 detector")
        if args.da3_prune_opacity < args.da3_min_opacity:
            parser.error("pruning opacity must be at least the seed opacity lower bound")
    if args.soft_band_low >= args.soft_band_high:
        parser.error("--soft-band-low must be smaller than --soft-band-high")
    if args.cue_remap == "learned_sigmoid":
        if args.cue_boundary_json is None:
            parser.error("--cue-boundary-json is required for learned_sigmoid")
        if not args.cue_boundary_json.is_file():
            parser.error(f"learned boundary artifact not found: {args.cue_boundary_json}")
    if args.da3_seed_checkpoint is not None and not args.da3_seed_checkpoint.is_file():
        parser.error(f"DA3 seed checkpoint not found: {args.da3_seed_checkpoint}")
    if args.base_geometry_scope != "frozen" and args.da3_seed_checkpoint is not None:
        parser.error(
            "base geometry refinement is a base-only ablation; omit the DA3 seed checkpoint"
        )
    if (
        args.train_never_open_base_opacity
        and args.base_geometry_scope != "open_and_never_open"
    ):
        parser.error(
            "--train-never-open-base-opacity requires "
            "--base-geometry-scope open_and_never_open"
        )
    if args.sam_sign_trace_root is not None and not args.sam_sign_trace_root.is_dir():
        parser.error(f"SAM sign trace root not found: {args.sam_sign_trace_root}")
    if args.base_geometry_target == "signed_new":
        if args.sam_sign_trace_root is None:
            parser.error(
                "--sam-sign-trace-root is required for --base-geometry-target signed_new"
            )
        if args.sam_new_sign is None and args.da3_seed_checkpoint is None:
            parser.error(
                "--sam-new-sign is required for signed_new without a DA3 seed checkpoint"
            )
    if not 0.0 < args.da3_initial_opacity < 1.0:
        parser.error("--da3-initial-opacity must lie in (0,1)")
    if args.da3_min_scale_ratio > args.da3_max_scale_ratio:
        parser.error("--da3-min-scale-ratio cannot exceed --da3-max-scale-ratio")
    if args.base_min_scale_ratio > args.base_max_scale_ratio:
        parser.error("--base-min-scale-ratio cannot exceed --base-max-scale-ratio")
    if not 0.0 < args.da3_min_opacity < args.da3_max_opacity < 1.0:
        parser.error("DA3 opacity bounds must satisfy 0 < min < max < 1")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    BayesianDetectorViewer(args).run()


if __name__ == "__main__":
    main()
