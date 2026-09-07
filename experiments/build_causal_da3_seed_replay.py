#!/usr/bin/env python3
"""Build one causal DA3 seed-birth replay across SC1, SC2, and SC3.

This utility is intentionally representation-training free.  It applies the
causal prequential Stage-2 cue, runs DA3 on one continuous current/past-only
stream, and assigns NEW to the SAM/PCA sign containing most positive
``reference depth - aligned DA3 depth`` evidence.  The first sufficiently
concentrated sign is locked for SC1 through SC3; scene boundaries never reset
or re-estimate it.  Only accepted fixed 3D seed geometry and causal birth
metadata are stored for the interactive viewer.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Sam2Model

from experiments.analyze_da3_depth_prior_new_seeds import (
    _read_resized_rgb,
    _resize_tensor,
    load_trace,
    render_reference_depth,
)
from experiments.run_online_da3_new_seed_dc_only import (
    _calibrate_stage2_cue,
    _load_learned_cue_artifact,
)
from experiments.run_online_xfeat_new_seed import (
    RUN,
    build_fixed_cue_views,
    extract_delta,
    fixed_camera_matrices,
    load_feature_cache,
    load_fixed_camera_index,
    observation_cache_key,
    scene_records,
    validate_feature_cache,
)
from experiments.train_real_temporal_rchange import (
    load_reference_frames,
    match_query_to_reference,
)
from gaussian_renderer import render
from poses.feature_detector import DescribedKeypoints, Detector
from scene import GaussianModel
from temporal.depth_prior_new_seeding import (
    DepthPriorSeedConfig,
    DepthScaleFit,
    build_depth_prior_new_seeds,
    canonical_metric_depth,
    fit_metric_anchor_depth_scale,
    fit_reference_depth_scale,
    front_depth_sign_evidence,
    metric_anchor_median_relative_error,
    q_weighted_upsampled_signed_sam_score,
    reference_scale_anchor_mask,
    threshold_signed_support,
)


DEFAULT_OUTPUT = Path("outputs/causal_da3_seed_replay_scene123_20260903")
DEFAULT_TRACE_ROOT = Path(
    "outputs/e4e_learned_dc_object_matched_metrics_20260902_164614"
)
DEFAULT_BOUNDARY = Path(
    "outputs/stage2_l1_power_sigmoid_boundary_causal_prequential/"
    "learned_boundaries_causal.json"
)
DEFAULT_XFEAT_CACHE = Path(
    "outputs/e4_online_xfeat_new_seed_scene123_120_final_corrected_20260814/"
    "xfeat_features.pt"
)


@dataclass(frozen=True)
class LocalizationDepthAnchors:
    """Geometrically valid metric camera-z anchors for one DA3 depth map."""

    pixels_xy: torch.Tensor
    camera_depth: torch.Tensor
    reprojection_error_px: torch.Tensor
    raw_matches: int

    @property
    def count(self) -> int:
        return int(self.camera_depth.numel())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--cue-boundary-json", type=Path, default=DEFAULT_BOUNDARY)
    parser.add_argument("--scenes", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument("--model", default="depth-anything/DA3-SMALL")
    parser.add_argument(
        "--metric-model", default="depth-anything/DA3METRIC-LARGE"
    )
    parser.add_argument("--sam-model", default="facebook/sam2.1-hiera-tiny")
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--sampling-stride", type=int, default=4)
    parser.add_argument("--max-new-seeds-per-frame", type=int, default=2048)
    parser.add_argument("--max-total-seeds", type=int, default=1_000_000)
    parser.add_argument("--confidence-quantile", type=float, default=0.40)
    parser.add_argument("--sam-support-threshold", type=float, default=0.10)
    parser.add_argument("--min-front-gap", type=float, default=0.03)
    parser.add_argument("--min-front-gap-ratio", type=float, default=0.0)
    parser.add_argument("--depth-sign-concentration", type=float, default=0.7)
    parser.add_argument("--depth-sign-min-pixels", type=int, default=32)
    parser.add_argument(
        "--depth-scale-source",
        choices=(
            "xfeat_localization",
            "reference_render",
            "da3metric_reference_render",
            "da3metric_first_lock",
            "da3metric_reference_bank_fixed",
        ),
        default="xfeat_localization",
        help=(
            "Fit DA3-SMALL per frame from localization XFeat/reference depth, "
            "or initialize one locked DA3-SMALL scale from DA3Metric + XFeat"
        ),
    )
    parser.add_argument("--xfeat-feature-cache", type=Path, default=DEFAULT_XFEAT_CACHE)
    parser.add_argument("--xfeat-top-k", type=int, default=512)
    parser.add_argument("--localization-reference-frames", type=int, default=16)
    parser.add_argument(
        "--localization-reference-association-px", type=float, default=3.0
    )
    parser.add_argument("--localization-max-matches-per-reference", type=int, default=256)
    parser.add_argument("--localization-reprojection-error-px", type=float, default=8.0)
    parser.add_argument("--localization-min-anchors", type=int, default=16)
    parser.add_argument(
        "--da3-cache-root",
        type=Path,
        help="Optional read-only DA3 cache root; a missing entry is an error",
    )
    parser.add_argument(
        "--metric-cache-root",
        type=Path,
        help=(
            "Optional read-only DA3Metric cache root; a missing entry is an "
            "error"
        ),
    )
    parser.add_argument(
        "--metric-fixed-scene-scale",
        type=float,
        help=(
            "Fixed scene-units-per-metre calibrated from immutable reference "
            "views; required by da3metric_reference_bank_fixed"
        ),
    )
    args = parser.parse_args()
    args.scenes = tuple(dict.fromkeys(int(scene) for scene in args.scenes))
    if not args.scenes or any(scene not in (1, 2, 3) for scene in args.scenes):
        parser.error("--scenes must contain values from 1,2,3")
    for name in (
        "window",
        "process_res",
        "sampling_stride",
        "max_new_seeds_per_frame",
        "max_total_seeds",
        "xfeat_top_k",
        "localization_reference_frames",
        "localization_max_matches_per_reference",
        "localization_min_anchors",
    ):
        if int(getattr(args, name)) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.window < 2:
        parser.error("--window must be at least 2")
    if args.min_front_gap < 0.0 or args.min_front_gap_ratio < 0.0:
        parser.error("front-gap thresholds must be nonnegative")
    if not 0.0 < args.sam_support_threshold < 1.0:
        parser.error("--sam-support-threshold must lie in (0,1)")
    if not 0.5 < args.depth_sign_concentration <= 1.0:
        parser.error("--depth-sign-concentration must lie in (0.5,1]")
    if args.depth_sign_min_pixels < 1:
        parser.error("--depth-sign-min-pixels must be positive")
    if args.localization_reference_association_px <= 0.0:
        parser.error("--localization-reference-association-px must be positive")
    if args.localization_reprojection_error_px <= 0.0:
        parser.error("--localization-reprojection-error-px must be positive")
    if not args.cue_boundary_json.is_file():
        parser.error(f"cue boundary artifact not found: {args.cue_boundary_json}")
    if (
        args.depth_scale_source in {"xfeat_localization", "da3metric_first_lock"}
        and not args.xfeat_feature_cache.is_file()
    ):
        parser.error(f"XFeat feature cache not found: {args.xfeat_feature_cache}")
    if args.depth_scale_source == "da3metric_reference_bank_fixed":
        if (
            args.metric_fixed_scene_scale is None
            or not np.isfinite(float(args.metric_fixed_scene_scale))
            or float(args.metric_fixed_scene_scale) <= 0.0
        ):
            parser.error(
                "--metric-fixed-scene-scale must be positive for "
                "da3metric_reference_bank_fixed"
            )
    return args


def causal_window(local_frame: int, size: int) -> list[int]:
    """Return a 1-based current/past-only frame window."""

    if local_frame < 1 or size < 1:
        raise ValueError("local_frame and size must be positive")
    return list(range(max(1, local_frame - size + 1), local_frame + 1))


def choose_depth_new_sign(
    *,
    plus_mass: float,
    minus_mass: float,
    plus_pixels: int,
    minus_pixels: int,
    concentration: float,
    min_pixels: int,
) -> tuple[str | None, float | None]:
    """Choose a NEW sign once cumulative front-depth evidence is concentrated."""

    if not 0.5 < float(concentration) <= 1.0:
        raise ValueError("concentration must lie in (0.5,1]")
    if min_pixels < 1:
        raise ValueError("min_pixels must be positive")
    total_pixels = int(plus_pixels) + int(minus_pixels)
    total_mass = float(plus_mass) + float(minus_mass)
    if total_pixels < int(min_pixels) or total_mass <= 0.0:
        return None, None
    plus_fraction = float(plus_mass) / total_mass
    winning_fraction = max(plus_fraction, 1.0 - plus_fraction)
    if winning_fraction < float(concentration):
        return None, None
    return ("+" if plus_fraction >= 0.5 else "-"), winning_fraction


def localization_depth_anchors(
    *,
    points2d: np.ndarray,
    points3d: np.ndarray,
    image_K: torch.Tensor,
    depth_K: torch.Tensor,
    w2c: torch.Tensor,
    depth_height: int,
    depth_width: int,
    max_reprojection_error_px: float,
) -> LocalizationDepthAnchors:
    """Filter XFeat 2D-3D matches by the camera pose used for DA3.

    The fixed-pose reprojection gate is the replay equivalent of keeping PnP
    geometric inliers. The accepted query keypoint is mapped into the native
    DA3 grid through the two intrinsic matrices, so the depth image itself is
    never upsampled for scale fitting.
    """

    pixels = torch.as_tensor(points2d, dtype=torch.float32).reshape(-1, 2)
    world = torch.as_tensor(points3d, dtype=torch.float32).reshape(-1, 3)
    raw_matches = len(pixels)
    if len(pixels) != len(world):
        raise ValueError("points2d and points3d must have matching rows")
    if depth_height < 1 or depth_width < 1:
        raise ValueError("depth dimensions must be positive")
    if max_reprojection_error_px <= 0.0:
        raise ValueError("max_reprojection_error_px must be positive")
    K_image = image_K.detach().cpu().float()
    K_depth = depth_K.detach().cpu().float()
    extrinsics = w2c.detach().cpu().float()
    if K_image.shape != (3, 3) or K_depth.shape != (3, 3):
        raise ValueError("image_K and depth_K must have shape [3,3]")
    if extrinsics.shape != (4, 4):
        raise ValueError("w2c must have shape [4,4]")
    if raw_matches == 0:
        empty = torch.empty((0,), dtype=torch.float32)
        return LocalizationDepthAnchors(
            pixels_xy=torch.empty((0, 2), dtype=torch.float32),
            camera_depth=empty,
            reprojection_error_px=empty,
            raw_matches=0,
        )

    camera = world @ extrinsics[:3, :3].T + extrinsics[:3, 3]
    projected_h = camera @ K_image.T
    projected = projected_h[:, :2] / projected_h[:, 2:].clamp_min(1.0e-8)
    error = torch.linalg.vector_norm(projected - pixels, dim=1)

    image_h = torch.cat((pixels, torch.ones((raw_matches, 1))), dim=1)
    normalized_ray = image_h @ torch.linalg.inv(K_image).T
    depth_h = normalized_ray @ K_depth.T
    depth_pixels = depth_h[:, :2] / depth_h[:, 2:].clamp_min(1.0e-8)

    valid = torch.isfinite(camera).all(dim=1)
    valid &= torch.isfinite(projected).all(dim=1)
    valid &= torch.isfinite(depth_pixels).all(dim=1)
    valid &= torch.isfinite(error)
    valid &= camera[:, 2] > 1.0e-8
    valid &= error <= float(max_reprojection_error_px)
    valid &= (depth_pixels[:, 0] >= 0.0) & (
        depth_pixels[:, 0] <= float(depth_width - 1)
    )
    valid &= (depth_pixels[:, 1] >= 0.0) & (
        depth_pixels[:, 1] <= float(depth_height - 1)
    )
    valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        empty = torch.empty((0,), dtype=torch.float32)
        return LocalizationDepthAnchors(
            pixels_xy=torch.empty((0, 2), dtype=torch.float32),
            camera_depth=empty,
            reprojection_error_px=empty,
            raw_matches=raw_matches,
        )

    # One current keypoint and one metric 3D landmark should each contribute at
    # most once even when the landmark is visible in several reference frames.
    order = valid_indices[torch.argsort(error[valid_indices], stable=True)]
    kept: list[int] = []
    seen_query: set[tuple[int, int]] = set()
    seen_world: set[tuple[int, int, int]] = set()
    for index in order.tolist():
        query_key = tuple(torch.round(pixels[index] * 4.0).int().tolist())
        world_key = tuple(torch.round(world[index] * 100000.0).int().tolist())
        if query_key in seen_query or world_key in seen_world:
            continue
        seen_query.add(query_key)
        seen_world.add(world_key)
        kept.append(index)
    keep = torch.tensor(kept, dtype=torch.long)
    return LocalizationDepthAnchors(
        pixels_xy=depth_pixels[keep],
        camera_depth=camera[keep, 2],
        reprojection_error_px=error[keep],
        raw_matches=raw_matches,
    )


def cached_query_descriptors(
    feature_cache: dict[str, Any], *, scene: int, record: Any
) -> DescribedKeypoints:
    """Restore one current-frame XFeat observation without re-inference."""

    key = observation_cache_key(scene, record)
    if key not in feature_cache:
        raise KeyError(f"XFeat feature cache is missing {key}")
    row = feature_cache[key]
    query = DescribedKeypoints(
        row["keypoints"].cuda(non_blocking=True),
        row["descriptors"].cuda(non_blocking=True),
    )
    query.valid = row["valid"].cuda(non_blocking=True).bool()
    return query


def match_all_localization_references(
    query: DescribedKeypoints, reference_frames: list[Any], max_matches: int
) -> tuple[np.ndarray, np.ndarray]:
    """Collect descriptor matches from every localization reference frame."""

    points2d: list[np.ndarray] = []
    points3d: list[np.ndarray] = []
    for reference in reference_frames:
        matched_2d, matched_3d = match_query_to_reference(
            query, reference, max_matches=max_matches
        )
        if len(matched_2d):
            points2d.append(matched_2d)
            points3d.append(matched_3d)
    if not points2d:
        return np.empty((0, 2), np.float32), np.empty((0, 3), np.float32)
    return np.concatenate(points2d), np.concatenate(points3d)


def summarize_depth_scale_comparison(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare both scales on the exact same sparse metric anchors."""

    paired = [
        row
        for row in rows
        if row.get("localization_xfeat_scale_median_relative_error") is not None
        and row.get("localization_reference_scale_median_relative_error")
        is not None
    ]
    if not paired:
        return {"frames": 0}
    xfeat = np.sort(
        np.asarray(
            [
                row["localization_xfeat_scale_median_relative_error"]
                for row in paired
            ],
            dtype=np.float64,
        )
    )
    reference = np.sort(
        np.asarray(
            [
                row["localization_reference_scale_median_relative_error"]
                for row in paired
            ],
            dtype=np.float64,
        )
    )
    differences = np.asarray(
        [
            row["localization_reference_scale_median_relative_error"]
            - row["localization_xfeat_scale_median_relative_error"]
            for row in paired
        ],
        dtype=np.float64,
    )
    xfeat_median = float(np.median(xfeat))
    reference_median = float(np.median(reference))
    p90_index = int(0.9 * (len(paired) - 1))
    result = {
        "frames": len(paired),
        "xfeat_error_median": xfeat_median,
        "reference_error_median": reference_median,
        "absolute_point_improvement": reference_median - xfeat_median,
        "relative_median_reduction": 1.0 - xfeat_median / reference_median,
        "xfeat_error_p90": float(xfeat[p90_index]),
        "reference_error_p90": float(reference[p90_index]),
        "frames_xfeat_better": int((differences > 0.0).sum()),
        "frames_equal": int((np.abs(differences) < 1.0e-12).sum()),
        "paired_difference_median": float(np.median(differences)),
    }
    locked = np.sort(
        np.asarray(
            [
                row["metric_locked_scale_median_relative_error"]
                for row in paired
                if row.get("metric_locked_scale_median_relative_error")
                is not None
            ],
            dtype=np.float64,
        )
    )
    if locked.size:
        locked_p90_index = int(0.9 * (locked.size - 1))
        result.update(
            {
                "metric_locked_error_frames": int(locked.size),
                "metric_locked_error_median": float(np.median(locked)),
                "metric_locked_error_p90": float(locked[locked_p90_index]),
            }
        )
    return result


def _cached_or_infer_depth(
    *,
    da3: Any | None,
    cache_path: Path,
    images: list[np.ndarray],
    extrinsics: list[np.ndarray],
    intrinsics: list[np.ndarray],
    process_res: int,
    window_frames: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cache_path.exists():
        with np.load(cache_path) as cache:
            if "window_frames" in cache:
                cached_window = np.asarray(cache["window_frames"], dtype=np.int64)
                expected_window = np.asarray(window_frames, dtype=np.int64)
                if not np.array_equal(cached_window, expected_window):
                    raise ValueError(
                        f"DA3 cache window mismatch for {cache_path}: "
                        f"{cached_window.tolist()} != {expected_window.tolist()}"
                    )
            return (
                np.asarray(cache["depth"], dtype=np.float32),
                np.asarray(cache["confidence"], dtype=np.float32),
                np.asarray(cache["K"], dtype=np.float32),
            )
    if da3 is None:
        raise FileNotFoundError(
            f"DA3 cache entry not found: {cache_path}; remove --da3-cache-root "
            "to enable inference"
        )
    prediction = da3.inference(
        images,
        extrinsics=np.stack(extrinsics),
        intrinsics=np.stack(intrinsics),
        align_to_input_ext_scale=True,
        process_res=int(process_res),
        process_res_method="upper_bound_resize",
    )
    depth = np.asarray(prediction.depth[-1], dtype=np.float32)
    confidence = np.asarray(prediction.conf[-1], dtype=np.float32)
    K = np.asarray(prediction.intrinsics[-1], dtype=np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        depth=depth,
        confidence=confidence,
        K=K,
        window_frames=np.asarray(window_frames, dtype=np.int64),
    )
    return depth, confidence, K


def _resize_intrinsics_to_depth_grid(
    image_K: np.ndarray,
    *,
    image_height: int,
    image_width: int,
    depth_height: int,
    depth_width: int,
) -> np.ndarray:
    """Map image intrinsics through DA3's resize-only preprocessing."""

    if min(image_height, image_width, depth_height, depth_width) < 1:
        raise ValueError("image and depth dimensions must be positive")
    K = np.asarray(image_K, dtype=np.float32).copy()
    if K.shape != (3, 3):
        raise ValueError("image_K must have shape [3,3]")
    K[0] *= float(depth_width) / float(image_width)
    K[1] *= float(depth_height) / float(image_height)
    return K


def _cached_or_infer_metric_depth(
    *,
    da3_metric: Any | None,
    cache_path: Path,
    image: np.ndarray,
    image_K: np.ndarray,
    process_res: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return DA3Metric monocular camera-z in metres and its processed K."""

    if cache_path.exists():
        with np.load(cache_path) as cache:
            return (
                np.asarray(cache["depth_meters"], dtype=np.float32),
                np.asarray(cache["K"], dtype=np.float32),
            )
    if da3_metric is None:
        raise FileNotFoundError(
            f"DA3Metric cache entry not found: {cache_path}; remove "
            "--metric-cache-root to enable inference"
        )
    prediction = da3_metric.inference(
        [image],
        intrinsics=np.asarray(image_K, dtype=np.float32)[None],
        align_to_input_ext_scale=False,
        process_res=int(process_res),
        process_res_method="upper_bound_resize",
    )
    raw_depth = np.asarray(prediction.depth[-1], dtype=np.float32)
    depth_height, depth_width = raw_depth.shape
    image_height, image_width = image.shape[:2]
    K = _resize_intrinsics_to_depth_grid(
        image_K,
        image_height=image_height,
        image_width=image_width,
        depth_height=depth_height,
        depth_width=depth_width,
    )
    depth_meters = canonical_metric_depth(
        torch.from_numpy(raw_depth), torch.from_numpy(K)
    ).numpy()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        depth_meters=depth_meters,
        raw_depth=raw_depth,
        K=K,
        canonical_focal=np.asarray(300.0, dtype=np.float32),
    )
    return depth_meters, K


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
    cameras = load_fixed_camera_index(Path(summary["fixed_cameras_json"]))
    learned_cue = _load_learned_cue_artifact(args.cue_boundary_json)
    checkpoint = torch.load(
        RUN / "temporal_rchange_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    base = GaussianModel(sh_degree=3, active_sh_degree=0)
    base.load_ply(str(checkpoint["base_ply"]))
    for parameter in (
        base._xyz,
        base._features_dc,
        base._features_rest,
        base._opacity,
        base._scaling,
        base._rotation,
    ):
        parameter.requires_grad_(False)
    pipe = SimpleNamespace(
        convert_SHs_python=False, compute_cov3D_python=False, debug=False
    )
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    sam = (
        Sam2Model.from_pretrained(args.sam_model, local_files_only=True)
        .half()
        .cuda()
        .eval()
    )
    da3 = None
    if (
        args.depth_scale_source
        not in {"da3metric_reference_render", "da3metric_reference_bank_fixed"}
        and args.da3_cache_root is None
    ):
        from depth_anything_3.api import DepthAnything3

        da3 = DepthAnything3.from_pretrained(args.model).to("cuda").eval()
    da3_metric = None
    if (
        args.depth_scale_source
        in {
            "da3metric_reference_render",
            "da3metric_first_lock",
            "da3metric_reference_bank_fixed",
        }
        and args.metric_cache_root is None
    ):
        from depth_anything_3.api import DepthAnything3

        da3_metric = (
            DepthAnything3.from_pretrained(args.metric_model).to("cuda").eval()
        )
    feature_cache: dict[str, Any] = {}
    reference_frames: list[Any] = []
    if args.depth_scale_source in {
        "xfeat_localization",
        "da3metric_first_lock",
    }:
        first_camera = next(iter(cameras.values()))
        image_width = int(first_camera["width"])
        image_height = int(first_camera["height"])
        feature_cache = load_feature_cache(args.xfeat_feature_cache)
        validate_feature_cache(
            feature_cache,
            top_k=int(args.xfeat_top_k),
            width=image_width,
            height=image_height,
        )
        xfeat_detector = Detector(
            top_k=int(args.xfeat_top_k), width=image_width, height=image_height
        )
        reference_frames, _ = load_reference_frames(
            Path(summary["source_path"]),
            float(summary["resolution"]),
            int(args.localization_reference_frames),
            xfeat_detector,
            float(args.localization_reference_association_px),
            image_width,
            image_height,
        )
        del xfeat_detector
    seed_config = DepthPriorSeedConfig(
        confidence_quantile=float(args.confidence_quantile),
        min_front_gap=float(args.min_front_gap),
        min_front_gap_ratio=float(args.min_front_gap_ratio),
        erosion_pixels=0,
        sampling_stride=int(args.sampling_stride),
        max_seeds=int(args.max_new_seeds_per_frame),
    )
    xyz_parts: list[torch.Tensor] = []
    scaling_parts: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    future_view_accesses = 0
    locked_new_sign: str | None = None
    locked_mapping_confidence: float | None = None
    first_sign_lock: dict[str, Any] | None = None
    cumulative_plus_mass = 0.0
    cumulative_minus_mass = 0.0
    cumulative_plus_pixels = 0
    cumulative_minus_pixels = 0
    depth_sign_observations = 0
    scale_source_counts = {
        "xfeat_localization": 0,
        "reference_render": 0,
        "da3metric_reference_render": 0,
        "da3metric_first_lock": 0,
        "da3metric_reference_bank_fixed": 0,
    }
    scale_fallback_frames = 0
    locked_metric_scene_scale: float | None = None
    locked_metric_scale_fit: DepthScaleFit | None = None
    metric_scale_lock: dict[str, Any] | None = None
    if args.depth_scale_source == "da3metric_reference_bank_fixed":
        locked_metric_scene_scale = float(args.metric_fixed_scene_scale)
        locked_metric_scale_fit = DepthScaleFit(
            scale=locked_metric_scene_scale,
            samples=0,
            inliers=0,
            log_ratio_mad=0.0,
            median_absolute_relative_error=0.0,
        )
        metric_scale_lock = {
            "source": "immutable_reference_bank_median",
            "scene_units_per_meter": locked_metric_scene_scale,
            "reference_views": 16,
            "artifact": (
                "outputs/reference_da3metric_large_vs_small_xfeat_20260904/"
                "summary.json"
            ),
        }
    past_stream: list[tuple[Any, Any]] = []
    previous_axis: torch.Tensor | None = None

    for scene in args.scenes:
        records = scene_records(summary, scene, None)
        views, _, _ = build_fixed_cue_views(
            records,
            cameras,
            Path(summary["run_arguments"]["cue_cache_root"]),
            float(summary["resolution"]),
        )
        _, trace_arrays = load_trace(
            args.trace_root / f"scene_change{scene}"
        )
        trace_index = {
            str(name): index
            for index, name in enumerate(trace_arrays["frame_names"].astype(str))
        }
        accepted_scene = 0
        for local_frame, (record, view) in enumerate(zip(records, views), start=1):
            accepted_count = 0
            candidate_count = 0
            rejected_capacity = 0
            frame_plus_mass = 0.0
            frame_minus_mass = 0.0
            frame_plus_pixels = 0
            frame_minus_pixels = 0
            depth_scale: float | None = None
            depth_scale_source: str | None = None
            depth_scale_samples = 0
            depth_scale_inliers = 0
            depth_scale_median_relative_error: float | None = None
            reference_depth_scale: float | None = None
            reference_scale_median_relative_error: float | None = None
            localization_raw_matches = 0
            localization_geometric_inliers = 0
            localization_reprojection_median_px: float | None = None
            localization_xfeat_scale_median_relative_error: float | None = None
            localization_reference_scale_median_relative_error: float | None = None
            metric_locked_scale_median_relative_error: float | None = None
            scale_fallback_reason: str | None = None
            birth_sign_cells = 0
            past_stream.append((record, view))
            window_pairs = past_stream[-int(args.window) :]
            window_global = [int(item[0].global_index) for item in window_pairs]
            future_view_accesses += sum(
                int(index > int(record.global_index)) for index in window_global
            )

            boundary = learned_cue.boundaries[record.name]
            previous_degree = int(base.active_sh_degree)
            base.active_sh_degree = int(base.max_sh_degree)
            try:
                with torch.no_grad():
                    reference_rgb = render(view, base, pipe, background)["render"]
                    _, cue = _calibrate_stage2_cue(
                        cached_sum=view.candidate_map,
                        reference_rgb=reference_rgb,
                        online_rgb=view.original_image,
                        boundary=boundary,
                        l1_exponent=0.3,
                        edge_probability=learned_cue.edge_probability,
                    )
            finally:
                base.active_sh_degree = previous_degree
            delta, _ = extract_delta(view, base, pipe, background, sam)
            axis_index = trace_index[record.name]
            axis = torch.from_numpy(trace_arrays["pc1_axes"][axis_index]).to(
                device=delta.device, dtype=delta.dtype
            )
            if previous_axis is not None and float(torch.dot(axis, previous_axis)) < 0.0:
                axis = -axis
            previous_axis = axis.detach()
            score64 = (delta @ axis).reshape(64, 64)
            weighted_sam_full, _ = q_weighted_upsampled_signed_sam_score(
                score64,
                cue,
                height=int(view.image_height),
                width=int(view.image_width),
            )
            plus_full, minus_full = threshold_signed_support(
                weighted_sam_full,
                threshold=float(args.sam_support_threshold),
            )

            # DA3 pose alignment is underdetermined for a tiny camera prefix.
            # Use one full current/past window before inferring the sign.
            if len(window_pairs) >= int(args.window):
                images: list[np.ndarray] = []
                extrinsics: list[np.ndarray] = []
                intrinsics: list[np.ndarray] = []
                for window_record, window_view in window_pairs:
                    images.append(
                        _read_resized_rgb(
                            Path(window_record.image_path),
                            int(window_view.image_height),
                            int(window_view.image_width),
                        )
                    )
                    window_w2c, window_K = fixed_camera_matrices(
                        window_record, cameras
                    )
                    extrinsics.append(window_w2c.numpy())
                    intrinsics.append(window_K.numpy())
                w2c, image_K = fixed_camera_matrices(record, cameras)
                metric_depth_meters: np.ndarray | None = None
                metric_output_K: np.ndarray | None = None
                if args.depth_scale_source in {
                    "da3metric_reference_render",
                    "da3metric_reference_bank_fixed",
                }:
                    depth, output_K = _cached_or_infer_metric_depth(
                        da3_metric=da3_metric,
                        cache_path=(
                            (
                                args.metric_cache_root
                                if args.metric_cache_root is not None
                                else args.output_dir / "da3metric_cache"
                            )
                            / f"frame_{int(record.global_index):06d}.npz"
                        ),
                        image=images[-1],
                        image_K=image_K.numpy(),
                        process_res=int(args.process_res),
                    )
                    confidence = np.ones_like(depth, dtype=np.float32)
                else:
                    small_depth, confidence, small_output_K = (
                        _cached_or_infer_depth(
                            da3=da3,
                            cache_path=(
                                (
                                    args.da3_cache_root
                                    if args.da3_cache_root is not None
                                    else args.output_dir / "da3_cache"
                                )
                                / f"frame_{int(record.global_index):06d}.npz"
                            ),
                            images=images,
                            extrinsics=extrinsics,
                            intrinsics=intrinsics,
                            process_res=int(args.process_res),
                            window_frames=window_global,
                        )
                    )
                    depth, output_K = small_depth, small_output_K
                if args.depth_scale_source == "da3metric_first_lock" and (
                    locked_metric_scene_scale is None
                ):
                    metric_depth_meters, metric_output_K = (
                        _cached_or_infer_metric_depth(
                            da3_metric=da3_metric,
                            cache_path=(
                                (
                                    args.metric_cache_root
                                    if args.metric_cache_root is not None
                                    else args.output_dir / "da3metric_cache"
                                )
                                / f"frame_{int(record.global_index):06d}.npz"
                            ),
                            image=images[-1],
                            image_K=image_K.numpy(),
                            process_res=int(args.process_res),
                        )
                    )
                    if metric_depth_meters.shape != depth.shape:
                        metric_depth_meters = (
                            _resize_tensor(
                                torch.from_numpy(metric_depth_meters),
                                int(depth.shape[0]),
                                int(depth.shape[1]),
                                "bilinear",
                            ).numpy().astype(np.float32)
                        )
                        metric_output_K = _resize_intrinsics_to_depth_grid(
                            image_K.numpy(),
                            image_height=int(images[-1].shape[0]),
                            image_width=int(images[-1].shape[1]),
                            depth_height=int(depth.shape[0]),
                            depth_width=int(depth.shape[1]),
                        )
                height, width = depth.shape
                cue_small = _resize_tensor(cue, height, width, "area")
                weighted_sam_small = _resize_tensor(
                    weighted_sam_full, height, width, "bilinear"
                )
                # Preserve the exact panel-7 non-black support through the
                # DA3 processing-grid resize. Bilinear resizing the signed
                # values themselves would leak tiny nonzero support into
                # pixels that panel 7 displays as zero.
                plus_small = _resize_tensor(
                    plus_full.float(), height, width, "nearest"
                ).bool()
                minus_small = _resize_tensor(
                    minus_full.float(), height, width, "nearest"
                ).bool()
                ref_depth_full, ref_alpha_full = render_reference_depth(
                    view, base, w2c, pipe, background
                )
                numerator = _resize_tensor(
                    ref_depth_full * ref_alpha_full, height, width, "area"
                )
                alpha = _resize_tensor(ref_alpha_full, height, width, "area")
                reference_depth = numerator / alpha.clamp_min(1.0e-6)
                reference_depth[alpha <= 1.0e-4] = 0.0
                anchors = reference_scale_anchor_mask(
                    reference_depth, alpha, cue_small, config=seed_config
                )
                reference_fit = fit_reference_depth_scale(
                    torch.from_numpy(depth), reference_depth, anchors
                )
                reference_depth_scale = float(reference_fit.scale)
                reference_scale_median_relative_error = float(
                    reference_fit.median_absolute_relative_error
                )
                fit = reference_fit
                depth_scale_source = "reference_render"
                if args.depth_scale_source == "da3metric_reference_render":
                    depth_scale_source = "da3metric_reference_render"
                if args.depth_scale_source == "da3metric_reference_bank_fixed":
                    if locked_metric_scale_fit is None:
                        raise RuntimeError(
                            "fixed reference-bank metric scale is not initialized"
                        )
                    fit = locked_metric_scale_fit
                    depth_scale_source = "da3metric_reference_bank_fixed"
                if args.depth_scale_source in {
                    "xfeat_localization",
                    "da3metric_first_lock",
                }:
                    query = cached_query_descriptors(
                        feature_cache, scene=int(scene), record=record
                    )
                    matched_2d, matched_3d = match_all_localization_references(
                        query,
                        reference_frames,
                        int(args.localization_max_matches_per_reference),
                    )
                    localization = localization_depth_anchors(
                        points2d=matched_2d,
                        points3d=matched_3d,
                        image_K=image_K,
                        depth_K=torch.from_numpy(output_K),
                        w2c=w2c,
                        depth_height=height,
                        depth_width=width,
                        max_reprojection_error_px=float(
                            args.localization_reprojection_error_px
                        ),
                    )
                    localization_raw_matches = int(localization.raw_matches)
                    localization_geometric_inliers = int(localization.count)
                    if localization.count:
                        localization_reprojection_median_px = float(
                            localization.reprojection_error_px.median().item()
                        )
                    try:
                        localization_fit = fit_metric_anchor_depth_scale(
                            torch.from_numpy(depth),
                            localization.pixels_xy,
                            localization.camera_depth,
                            min_samples=int(args.localization_min_anchors),
                        )
                        localization_xfeat_scale_median_relative_error = (
                            metric_anchor_median_relative_error(
                                torch.from_numpy(depth),
                                localization.pixels_xy,
                                localization.camera_depth,
                                scale=localization_fit.scale,
                            )
                        )
                        localization_reference_scale_median_relative_error = (
                            metric_anchor_median_relative_error(
                                torch.from_numpy(depth),
                                localization.pixels_xy,
                                localization.camera_depth,
                                scale=reference_fit.scale,
                            )
                        )
                        if args.depth_scale_source == "xfeat_localization":
                            fit = localization_fit
                            depth_scale_source = "xfeat_localization"
                        else:
                            if locked_metric_scene_scale is None:
                                if (
                                    metric_depth_meters is None
                                    or metric_output_K is None
                                ):
                                    raise RuntimeError(
                                        "DA3Metric initialization depth is missing"
                                    )
                                metric_localization = localization_depth_anchors(
                                    points2d=matched_2d,
                                    points3d=matched_3d,
                                    image_K=image_K,
                                    depth_K=torch.from_numpy(metric_output_K),
                                    w2c=w2c,
                                    depth_height=int(
                                        metric_depth_meters.shape[0]
                                    ),
                                    depth_width=int(
                                        metric_depth_meters.shape[1]
                                    ),
                                    max_reprojection_error_px=float(
                                        args.localization_reprojection_error_px
                                    ),
                                )
                                metric_scene_fit = (
                                    fit_metric_anchor_depth_scale(
                                        torch.from_numpy(metric_depth_meters),
                                        metric_localization.pixels_xy,
                                        metric_localization.camera_depth,
                                        min_samples=int(
                                            args.localization_min_anchors
                                        ),
                                    )
                                )
                                metric_scene_depth = (
                                    torch.from_numpy(metric_depth_meters)
                                    * float(metric_scene_fit.scale)
                                )
                                locked_metric_scale_fit = (
                                    fit_reference_depth_scale(
                                        torch.from_numpy(depth),
                                        metric_scene_depth,
                                        anchors,
                                    )
                                )
                                locked_metric_scene_scale = float(
                                    locked_metric_scale_fit.scale
                                )
                                metric_scale_lock = {
                                    "scene": int(scene),
                                    "frame_local": int(local_frame),
                                    "frame_global": int(record.global_index),
                                    "frame_name": record.name,
                                    "metric_scene_units_per_meter": float(
                                        metric_scene_fit.scale
                                    ),
                                    "small_depth_units_to_meters": float(
                                        locked_metric_scale_fit.scale
                                        / metric_scene_fit.scale
                                    ),
                                    "small_depth_to_scene_units": float(
                                        locked_metric_scene_scale
                                    ),
                                    "metric_anchor_samples": int(
                                        metric_scene_fit.samples
                                    ),
                                    "metric_anchor_inliers": int(
                                        metric_scene_fit.inliers
                                    ),
                                    "metric_anchor_median_relative_error": float(
                                        metric_scene_fit.median_absolute_relative_error
                                    ),
                                    "dense_cross_model_samples": int(
                                        locked_metric_scale_fit.samples
                                    ),
                                    "dense_cross_model_inliers": int(
                                        locked_metric_scale_fit.inliers
                                    ),
                                    "dense_cross_model_median_relative_error": float(
                                        locked_metric_scale_fit.median_absolute_relative_error
                                    ),
                                }
                            if locked_metric_scale_fit is None:
                                raise RuntimeError(
                                    "DA3Metric scale lock lost its fit metadata"
                                )
                            fit = locked_metric_scale_fit
                            depth_scale_source = "da3metric_first_lock"
                            metric_locked_scale_median_relative_error = (
                                metric_anchor_median_relative_error(
                                    torch.from_numpy(depth),
                                    localization.pixels_xy,
                                    localization.camera_depth,
                                    scale=locked_metric_scene_scale,
                                )
                            )
                    except ValueError as error:
                        scale_fallback_reason = str(error)
                        if args.depth_scale_source == "xfeat_localization":
                            scale_fallback_frames += 1
                        elif locked_metric_scale_fit is not None:
                            fit = locked_metric_scale_fit
                            depth_scale_source = "da3metric_first_lock"
                        else:
                            scale_fallback_frames += 1
                depth_scale = float(fit.scale)
                depth_scale_samples = int(fit.samples)
                depth_scale_inliers = int(fit.inliers)
                depth_scale_median_relative_error = (
                    None
                    if args.depth_scale_source
                    == "da3metric_reference_bank_fixed"
                    else (
                        metric_locked_scale_median_relative_error
                        if metric_locked_scale_median_relative_error is not None
                        else float(fit.median_absolute_relative_error)
                    )
                )
                scale_source_counts[depth_scale_source] += 1
                scale_ready = not (
                    args.depth_scale_source == "da3metric_first_lock"
                    and locked_metric_scene_scale is None
                )
                if locked_new_sign is None and scale_ready:
                    evidence = front_depth_sign_evidence(
                        predicted_depth=torch.from_numpy(depth),
                        confidence=torch.from_numpy(confidence),
                        reference_depth=reference_depth,
                        plus_mask=plus_small,
                        minus_mask=minus_small,
                        cue_strength=cue_small,
                        scale=fit.scale,
                        config=seed_config,
                        cue_support_threshold=0.0,
                    )
                    depth_sign_observations += 1
                    frame_plus_mass = evidence.plus_mass
                    frame_minus_mass = evidence.minus_mass
                    frame_plus_pixels = evidence.plus_pixels
                    frame_minus_pixels = evidence.minus_pixels
                    cumulative_plus_mass += evidence.plus_mass
                    cumulative_minus_mass += evidence.minus_mass
                    cumulative_plus_pixels += evidence.plus_pixels
                    cumulative_minus_pixels += evidence.minus_pixels
                    locked_new_sign, locked_mapping_confidence = (
                        choose_depth_new_sign(
                            plus_mass=cumulative_plus_mass,
                            minus_mass=cumulative_minus_mass,
                            plus_pixels=cumulative_plus_pixels,
                            minus_pixels=cumulative_minus_pixels,
                            concentration=float(args.depth_sign_concentration),
                            min_pixels=int(args.depth_sign_min_pixels),
                        )
                    )
                    if locked_new_sign is not None:
                        first_sign_lock = {
                            "scene": int(scene),
                            "frame_local": int(local_frame),
                            "frame_global": int(record.global_index),
                            "frame_name": record.name,
                            "new_sign": locked_new_sign,
                            "confidence": locked_mapping_confidence,
                            "plus_mass": cumulative_plus_mass,
                            "minus_mass": cumulative_minus_mass,
                            "plus_pixels": cumulative_plus_pixels,
                            "minus_pixels": cumulative_minus_pixels,
                        }

                new_sign = "depth_positive"
                if scale_ready and len(metadata) < int(args.max_total_seeds):
                    # Match the visible panel-8 domain exactly before the
                    # depth-positive gate inside build_depth_prior_new_seeds.
                    new_mask = plus_small & (
                        alpha >= float(seed_config.min_reference_alpha)
                    )
                    birth_sign_cells = int(plus_full.sum().item())
                    batch = build_depth_prior_new_seeds(
                        predicted_depth=torch.from_numpy(depth),
                        confidence=torch.from_numpy(confidence),
                        reference_depth=reference_depth,
                        new_mask=new_mask,
                        K=torch.from_numpy(output_K),
                        w2c=w2c,
                        scale=fit.scale,
                        config=seed_config,
                        sampling_priority=weighted_sam_small.clamp_min(0.0),
                    )
                    capacity = int(args.max_total_seeds) - len(metadata)
                    accepted = torch.arange(
                        min(batch.count, capacity), dtype=torch.long
                    )
                    candidate_count = batch.count
                    accepted_count = int(accepted.numel())
                    rejected_capacity = candidate_count - accepted_count
                    if accepted_count:
                        xyz_parts.append(batch.xyz[accepted].detach().cpu())
                        scaling_parts.append(
                            batch.log_scaling[accepted].detach().cpu()
                        )
                        for index in accepted.tolist():
                            metadata.append(
                                {
                                    "birth_kind": "da3_depth_prior",
                                    "scene": int(scene),
                                    "frame_local": int(local_frame),
                                    "frame_global": int(record.global_index),
                                    "frame_name": record.name,
                                    "source_sign": "+",
                                    "seed_rule": (
                                        "panel7>sam_support_threshold AND "
                                        "rendered_depth-scaled_da3_depth>min_front_gap"
                                    ),
                                    "mapping_confidence": locked_mapping_confidence,
                                    "depth_scale": float(fit.scale),
                                    "depth_scale_source": depth_scale_source,
                                    "depth_prediction_source": (
                                        args.metric_model
                                        if args.depth_scale_source
                                        in {
                                            "da3metric_reference_render",
                                            "da3metric_reference_bank_fixed",
                                        }
                                        else args.model
                                    ),
                                    "confidence_source": (
                                        "uniform_no_metric_confidence"
                                        if args.depth_scale_source
                                        in {
                                            "da3metric_reference_render",
                                            "da3metric_reference_bank_fixed",
                                        }
                                        else args.model
                                    ),
                                    "scale_initialization_model": (
                                        args.metric_model
                                        if args.depth_scale_source
                                        in {
                                            "da3metric_first_lock",
                                            "da3metric_reference_bank_fixed",
                                        }
                                        else None
                                    ),
                                    "depth_scale_samples": int(fit.samples),
                                    "depth_scale_inliers": int(fit.inliers),
                                    "sam_diff_magnitude_gate": (
                                        f"panel7_gt_{float(args.sam_support_threshold):g}"
                                    ),
                                    "sam_support_source": (
                                        "panel7_q_weighted_visible_signed_sam"
                                    ),
                                    "spatial_acceptance": (
                                        "deferred_viewer_learned_gaussian_support"
                                    ),
                                    "da3_confidence": float(
                                        batch.confidence[index].item()
                                    ),
                                    "learned_cue_priority": float(
                                        cue_small[
                                            int(batch.pixels_xy[index, 1]),
                                            int(batch.pixels_xy[index, 0]),
                                        ].item()
                                    ),
                                    "pixel_xy": batch.pixels_xy[index].tolist(),
                                    "causal_window_global": window_global,
                                }
                            )
                        accepted_scene += accepted_count
            new_sign = locked_new_sign
            rows.append(
                {
                    "scene": int(scene),
                    "frame_local": int(local_frame),
                    "frame_global": int(record.global_index),
                    "frame_name": record.name,
                    "new_sign": new_sign,
                    "depth_sign_plus_mass": frame_plus_mass,
                    "depth_sign_minus_mass": frame_minus_mass,
                    "depth_sign_plus_pixels": frame_plus_pixels,
                    "depth_sign_minus_pixels": frame_minus_pixels,
                    "depth_scale": depth_scale,
                    "depth_scale_source": depth_scale_source,
                    "depth_scale_samples": depth_scale_samples,
                    "depth_scale_inliers": depth_scale_inliers,
                    "depth_scale_median_relative_error": (
                        depth_scale_median_relative_error
                    ),
                    "reference_depth_scale": reference_depth_scale,
                    "reference_scale_median_relative_error": (
                        reference_scale_median_relative_error
                    ),
                    "localization_raw_matches": localization_raw_matches,
                    "localization_geometric_inliers": (
                        localization_geometric_inliers
                    ),
                    "localization_reprojection_median_px": (
                        localization_reprojection_median_px
                    ),
                    "localization_xfeat_scale_median_relative_error": (
                        localization_xfeat_scale_median_relative_error
                    ),
                    "localization_reference_scale_median_relative_error": (
                        localization_reference_scale_median_relative_error
                    ),
                    "metric_locked_scale_median_relative_error": (
                        metric_locked_scale_median_relative_error
                    ),
                    "metric_initialized_depth_scale": locked_metric_scene_scale,
                    "depth_scale_fallback_reason": scale_fallback_reason,
                    "birth_sign_cells": int(birth_sign_cells),
                    "candidate": int(candidate_count),
                    "accepted": int(accepted_count),
                    "rejected_capacity": int(rejected_capacity),
                    "accepted_scene": int(accepted_scene),
                    "accepted_total": len(metadata),
                }
            )
            print(
                f"[SC{scene} {local_frame:03d}/{len(records):03d}] "
                f"sign={new_sign if new_sign is not None else 'none'} "
                f"scale={depth_scale_source or 'none'} "
                f"candidates={candidate_count} "
                f"birth={accepted_count} scene_total={accepted_scene}"
            )

    xyz = (
        torch.cat(xyz_parts, dim=0)
        if xyz_parts
        else torch.empty((0, 3), dtype=torch.float32)
    )
    log_scaling = (
        torch.cat(scaling_parts, dim=0)
        if scaling_parts
        else torch.empty((0, 3), dtype=torch.float32)
    )
    if log_scaling.shape != xyz.shape:
        raise RuntimeError("DA3 seed xyz and log-scaling rows diverged")
    replay_path = args.output_dir / "da3_seed_replay.pt"
    torch.save(
        {
            "schema_version": 2,
            "seed_sidecar": {
                "xyz": xyz,
                "log_scaling": log_scaling,
                "metadata": metadata,
            },
            "configuration": vars(args),
            "audit": {
                "future_view_accesses": int(future_view_accesses),
                "ground_truth_birth_accesses": 0,
                "reference_geometry_mutated": False,
                "scenes": list(args.scenes),
                "sign_inference_resets": 0,
                "sign_locked_once": locked_new_sign is not None,
                "sign_source": "front_depth_soft_cue_concentration",
                "posterior_sign_accesses": 0,
                "scene_boundary_depth_window_resets": 0,
                "depth_scale_source_counts": scale_source_counts,
                "depth_scale_fallback_frames": int(scale_fallback_frames),
                "metric_scale_lock": metric_scale_lock,
                "metric_scale_updates_after_lock": 0,
                "metric_scale_initializer_model": (
                    args.metric_model
                    if args.depth_scale_source
                    in {
                        "da3metric_first_lock",
                        "da3metric_reference_bank_fixed",
                    }
                    else None
                ),
                "depth_prediction_model": (
                    args.metric_model
                    if args.depth_scale_source
                    in {
                        "da3metric_reference_render",
                        "da3metric_reference_bank_fixed",
                    }
                    else args.model
                ),
                "confidence_model": (
                    None
                    if args.depth_scale_source
                    in {
                        "da3metric_reference_render",
                        "da3metric_reference_bank_fixed",
                    }
                    else args.model
                ),
                "depth_scale_uses_change_cue": args.depth_scale_source
                in {"reference_render", "da3metric_reference_render"},
                "depth_scale_uses_ground_truth": False,
            },
        },
        replay_path,
    )
    depth_scale_comparison = {
        "scope": (
            "same XFeat localization anchors; median absolute relative "
            "camera-z error"
        ),
        "all": summarize_depth_scale_comparison(rows),
        "by_scene": {
            str(scene): summarize_depth_scale_comparison(
                [row for row in rows if int(row["scene"]) == scene]
            )
            for scene in args.scenes
        },
    }
    depth_scale_comparison_path = args.output_dir / "depth_scale_comparison.json"
    depth_scale_comparison_path.write_text(
        json.dumps(depth_scale_comparison, indent=2) + "\n", encoding="utf-8"
    )
    summary_out = {
        "experiment": "causal panel7-supported positive-depth DA3 seed replay",
        "configuration": {key: str(value) for key, value in vars(args).items()},
        "total_accepted": int(xyz.shape[0]),
        "accepted_by_scene": {
            str(scene): sum(
                int(row["accepted"])
                for row in rows
                if int(row["scene"]) == scene
            )
            for scene in args.scenes
        },
        "first_birth_by_scene": {
            str(scene): next(
                (
                    row
                    for row in rows
                    if int(row["scene"]) == scene and int(row["accepted"]) > 0
                ),
                None,
            )
            for scene in args.scenes
        },
        "first_sign_lock": first_sign_lock,
        "depth_sign_observations_before_lock": int(depth_sign_observations),
        "depth_scale_source_counts": scale_source_counts,
        "depth_scale_fallback_frames": int(scale_fallback_frames),
        "metric_scale_lock": metric_scale_lock,
        "audit": {
            "future_view_accesses": int(future_view_accesses),
            "ground_truth_birth_accesses": 0,
            "reference_geometry_mutated": False,
            "sign_inference_resets": 0,
            "sign_locked_once": locked_new_sign is not None,
            "sign_source": "front_depth_soft_cue_concentration",
            "seed_rule": (
                "panel7>sam_support_threshold AND "
                "rendered_depth-scaled_da3_depth>min_front_gap"
            ),
            "posterior_sign_accesses": 0,
            "scene_boundary_depth_window_resets": 0,
            "depth_scale_source_counts": scale_source_counts,
            "depth_scale_fallback_frames": int(scale_fallback_frames),
            "metric_scale_lock": metric_scale_lock,
            "metric_scale_updates_after_lock": 0,
            "metric_scale_initializer_model": (
                args.metric_model
                if args.depth_scale_source
                in {
                    "da3metric_first_lock",
                    "da3metric_reference_bank_fixed",
                }
                else None
            ),
            "depth_prediction_model": (
                args.metric_model
                if args.depth_scale_source
                in {
                    "da3metric_reference_render",
                    "da3metric_reference_bank_fixed",
                }
                else args.model
            ),
            "confidence_model": (
                None
                if args.depth_scale_source
                in {
                    "da3metric_reference_render",
                    "da3metric_reference_bank_fixed",
                }
                else args.model
            ),
            "depth_scale_uses_change_cue": args.depth_scale_source
            in {"reference_render", "da3metric_reference_render"},
            "depth_scale_uses_ground_truth": False,
        },
        "artifact": str(replay_path),
        "depth_scale_comparison_artifact": str(depth_scale_comparison_path),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary_out, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (args.output_dir / "frame_rows.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary_out, indent=2, default=str))


if __name__ == "__main__":
    main()
