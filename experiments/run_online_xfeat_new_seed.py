#!/usr/bin/env python3
"""Controlled online comparison: two-sign DC-only baseline vs XFeat NEW-seed DC-only branch.

This is an experiment-only runner.  It never changes repository files and it
never optimizes the reference Gaussian geometry/topology.  SceneChange1,
SceneChange2, and SceneChange3 are replayed independently.  At frame t the ordering is:

  1. extract Delta_t = SAM(R_ref_t) - SAM(I_t),
  2. update exact prefix PCA statistics using Delta_1..Delta_t,
  3. sign-align PC1 to PC1_(t-1) and make strong +/- cue masks,
  4. render the +/- DC memories learned only through frame t-1,
  5. update P(+ is ADD) from same-sign follow evidence,
  6. optimize only the two DC tensors on the current signed masks.

Ground-truth object masks are loaded only after the causal evidence update and
are used solely for reporting/visualization.  They never affect PCA, masks,
memory optimization, evidence validity, or the sign posterior.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from plyfile import PlyData, PlyElement
from torch import nn
from transformers import Sam2Model


REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "outputs/instance1_scene_change1_2_3_temporal_rchange_oscd_cues_allframes_120"
DATA = REPO / "data/Instance_1"
DEFAULT_OUT = Path("/tmp/oscd_online_xfeat_new_seed_20260813")
BASE_FIELDS = ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation")
SIGN_NAMES = ("plus", "minus")
FEATURE_CACHE_META_KEY = "__metadata__"

sys.path.insert(0, str(REPO))
from experiments.train_cue_temporal_rchange import (  # noqa: E402
    build_fixed_cue_views,
    camera_json_to_w2c,
    load_fixed_camera_index,
)
from experiments.train_real_temporal_rchange import FrameRecord  # noqa: E402
from gaussian_renderer import render, render_change  # noqa: E402
from scene import GaussianModel  # noqa: E402
from poses.feature_detector import Detector  # noqa: E402
from temporal.new_seed_observation import SignedXFeatObservation, XFeatObservationBuffer, masked_xfeat_subset  # noqa: E402
from temporal.new_seed_manager import (  # noqa: E402
    NewSeedManager, NewSeedManagerConfig, SeedGeometryResult, SeedMatchEdge, SeedObservationId
)
from temporal.new_seed_gaussians import NewSeedGaussianModel, build_concatenated_change_view  # noqa: E402
from temporal.new_seed_densification import (  # noqa: E402
    NewSeedDensificationConfig,
    densify_confirmed_new_region,
)
from poses.new_seed_triangulation import (  # noqa: E402
    GeometryGateConfig, match_and_triangulate_masked_pair, triangulate_track_with_diagnostics
)
from temporal.sign_mapping import (  # noqa: E402
    CausalPC1,
    FollowEvidence,
    SignMappingConfig,
    SignPosterior,
    component_balanced_follow as sign_component_balanced_follow,
    evidence_skip_reason as sign_evidence_skip_reason,
    mapping_from_probability,
    NewSignGate,
    normalized_memory,
    pooled_follow as sign_pooled_follow,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--scenes", type=int, nargs="+", default=(1, 2), choices=(1, 2, 3))
    parser.add_argument("--updates-per-frame", type=int, default=120)
    parser.add_argument("--lr", type=float, default=0.0025)
    parser.add_argument("--eps-sigma", type=float, default=2.5)
    parser.add_argument("--cue-threshold", type=float, default=0.5)
    parser.add_argument("--stable-threshold", type=float, default=0.2)
    parser.add_argument("--stable-bank-size", type=int, default=65536)
    parser.add_argument("--axis-stability-min", type=float, default=0.90)
    parser.add_argument("--min-camera-translation", type=float, default=0.10)
    parser.add_argument("--min-sign-cells", type=int, default=4)
    parser.add_argument("--min-memory-cells", type=int, default=4)
    parser.add_argument("--memory-weight-floor", type=float, default=0.05)
    parser.add_argument("--component-threshold", type=float, default=0.15)
    parser.add_argument("--min-component-cells", type=int, default=3)
    parser.add_argument("--component-area-cap", type=int, default=32)
    parser.add_argument("--confirm-probability", type=float, default=0.90)
    parser.add_argument("--confirm-valid-views", type=int, default=2)
    parser.add_argument("--sam-model", default="facebook/sam2.1-hiera-tiny")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-gif", action="store_true")
    parser.add_argument("--save-frame-pngs", action="store_true")
    parser.add_argument("--feature-cache", type=Path, help="Optional torch cache for pose-time XFeat observations")
    parser.add_argument("--xfeat-top-k", type=int, default=512)
    parser.add_argument("--seed-buffer-frames", type=int, default=32)
    parser.add_argument("--seed-prior-views", type=int, default=8)
    parser.add_argument("--seed-gate-probability", type=float, default=0.8)
    parser.add_argument("--seed-min-inmask-keypoints", type=int, default=8)
    parser.add_argument("--seed-cosine", type=float, default=0.82)
    parser.add_argument("--seed-min-ray-angle-deg", type=float, default=1.5)
    parser.add_argument("--seed-max-epipolar-error", type=float, default=2.0)
    parser.add_argument("--seed-max-reprojection-rmse", type=float, default=3.0)
    parser.add_argument("--seed-candidate-ttl", type=int, default=20)
    parser.add_argument("--seed-lr", type=float, default=None, help="Seed DC LR; defaults to --lr")
    parser.add_argument("--disable-seeds", action="store_true", help="Run comparison harness but pause seed birth")
    parser.add_argument("--new-only-densify", action="store_true", help="Densify only the confirmed-NEW seed sidecar")
    parser.add_argument("--densify-min-support-views", type=int, default=3)
    parser.add_argument("--densify-min-support-ratio", type=float, default=0.65)
    parser.add_argument("--densify-interval", type=int, default=2)
    parser.add_argument("--densify-max-new-per-frame", type=int, default=64)
    parser.add_argument("--densify-max-total-seeds", type=int, default=2500)
    parser.add_argument("--densify-coverage-radius-cells", type=float, default=0.8)
    parser.add_argument("--densify-max-parent-distance-cells", type=float, default=10.0)
    parser.add_argument("--densify-min-child-separation-cells", type=float, default=0.9)
    parser.add_argument("--densify-min-world-separation", type=float, default=0.008)
    parser.add_argument("--densify-footprint-px", type=float, default=3.0)
    parser.add_argument("--densify-opacity", type=float, default=0.1)
    parser.add_argument("--densify-history-views", type=int, default=12)
    parser.add_argument(
        "--seed-coverage-loss-weight",
        type=float,
        default=None,
        help="Base-independent seed-only coverage weight; defaults to 1 for NEW-only densify, else 0",
    )
    args = parser.parse_args()
    if args.updates_per_frame <= 0:
        parser.error("--updates-per-frame must be positive")
    if args.stable_bank_size < 4096:
        parser.error("--stable-bank-size must be at least 4096")
    if not 0.5 < args.confirm_probability < 1.0:
        parser.error("--confirm-probability must be in (0.5, 1)")
    if not 0.5 < args.seed_gate_probability < 1.0:
        parser.error("--seed-gate-probability must be in (0.5, 1)")
    if args.xfeat_top_k <= 0:
        parser.error("--xfeat-top-k must be positive")
    if args.seed_buffer_frames < 3:
        parser.error("--seed-buffer-frames must be at least 3")
    if args.seed_prior_views <= 0:
        parser.error("--seed-prior-views must be positive")
    if args.new_only_densify and args.disable_seeds:
        parser.error("--new-only-densify requires NEW seeds")
    if args.densify_min_support_views < 2:
        parser.error("--densify-min-support-views must be at least 2")
    if not 0.0 < args.densify_min_support_ratio <= 1.0:
        parser.error("--densify-min-support-ratio must be in (0,1]")
    for name in (
        "densify_interval",
        "densify_max_new_per_frame",
        "densify_max_total_seeds",
        "densify_history_views",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.seed_coverage_loss_weight is None:
        args.seed_coverage_loss_weight = 1.0 if args.new_only_densify else 0.0
    if args.seed_coverage_loss_weight < 0.0:
        parser.error("--seed-coverage-loss-weight must be non-negative")
    return args




def active_seed_render_sign(
    gate_sign: str | None,
    current_seed_sign: str | None,
    active_seed_count: int,
) -> str | None:
    """Choose the sign-memory render that owns existing active NEW seeds.

    A neutral posterior pauses *birth/training* only.  Already-promoted rows keep
    their open lifespan and therefore must remain in the current-state render
    under the last agreed NEW sign.  An opposite-sign consensus is handled by
    the manager/model lifespan flip before this helper is called.
    """

    if active_seed_count <= 0:
        return None
    sign = gate_sign if gate_sign in {"+", "-"} else current_seed_sign
    if sign not in {"+", "-"}:
        raise RuntimeError("active NEW seeds exist without an owning PCA sign")
    return sign


def fixed_camera_matrices(record: FrameRecord, cameras: dict[str, dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    stem = Path(record.name).stem
    cam = cameras[stem]
    # Reuse the exact fixed-camera convention used to build the renderer view.
    w2c = camera_json_to_w2c(cam)
    K = np.asarray(
        [[float(cam["fx"]), 0.0, float(cam["width"]) / 2.0], [0.0, float(cam["fy"]), float(cam["height"]) / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return torch.from_numpy(w2c.astype(np.float32)), torch.from_numpy(K)


def observation_cache_key(scene: int, record: FrameRecord) -> str:
    return f"scene{scene}:{record.name}"


def load_feature_cache(path: Path | None) -> dict[str, dict[str, torch.Tensor]]:
    if path is None or not path.exists():
        return {}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload if isinstance(payload, dict) else {}


def validate_feature_cache(
    cache: dict[str, Any], *, top_k: int, width: int, height: int
) -> None:
    expected = {
        "schema_version": 1,
        "detector": "poses.feature_detector.Detector/XFeat",
        "top_k": int(top_k),
        "width": int(width),
        "height": int(height),
    }
    metadata = cache.get(FEATURE_CACHE_META_KEY)
    data_keys = [key for key in cache if key != FEATURE_CACHE_META_KEY]
    if metadata is None:
        if data_keys:
            raise ValueError(
                "XFeat cache has no configuration metadata; use a fresh --feature-cache path"
            )
        cache[FEATURE_CACHE_META_KEY] = expected
        return
    if metadata != expected:
        raise ValueError(f"XFeat cache configuration mismatch: {metadata} != {expected}")


def save_feature_cache(path: Path | None, cache: dict[str, dict[str, torch.Tensor]]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(cache, temporary)
    temporary.replace(path)


@torch.inference_mode()
def extract_or_load_xfeat(
    *,
    scene: int,
    record: FrameRecord,
    view: Any,
    detector: Detector | None,
    cache: dict[str, dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    key = observation_cache_key(scene, record)
    cached = cache.get(key)
    if cached is not None:
        return cached
    if detector is None:
        raise RuntimeError("XFeat detector is required when --feature-cache misses")
    desc = detector(view.original_image[:3].detach())
    row = {
        "keypoints": desc.kpts.detach().cpu().float(),
        "descriptors": desc.feats.detach().cpu().half(),
        "valid": desc.valid.detach().cpu().bool(),
    }
    cache[key] = row
    return row


def make_signed_observation(
    *,
    scene: int,
    record: FrameRecord,
    view: Any,
    cameras: dict[str, dict[str, Any]],
    features: dict[str, torch.Tensor],
    plus64: torch.Tensor,
    minus64: torch.Tensor,
    cue64: torch.Tensor,
    score64: torch.Tensor,
) -> SignedXFeatObservation:
    w2c, K = fixed_camera_matrices(record, cameras)
    return SignedXFeatObservation(
        frame_index=int(record.global_index),
        timestamp=float(record.global_index),
        frame_name=record.name,
        w2c=w2c,
        K=K,
        image_size=(int(view.image_height), int(view.image_width)),
        keypoints=features["keypoints"],
        descriptors=features["descriptors"],
        valid=features["valid"],
        plus_mask64=plus64.detach().cpu(),
        minus_mask64=minus64.detach().cpu(),
        cue_strength64=cue64.detach().cpu().float(),
        pca_margin64=score64.detach().cpu().float(),
    )


def project_point(xyz: torch.Tensor, obs: SignedXFeatObservation) -> tuple[torch.Tensor, float]:
    X = torch.cat([xyz.float(), torch.ones(1)], dim=0)
    cam = (obs.w2c @ X)[:3]
    if float(cam[2]) <= 1e-6:
        return torch.zeros(2), float(cam[2])
    uvh = obs.K @ cam
    return uvh[:2] / uvh[2], float(cam[2])


def geometry_gate_config(args: argparse.Namespace) -> GeometryGateConfig:
    return GeometryGateConfig(
        min_translation=float(args.min_camera_translation),
        min_ray_angle_deg=float(args.seed_min_ray_angle_deg),
        max_epipolar_error_px=float(args.seed_max_epipolar_error),
        max_reprojection_rmse_px=float(args.seed_max_reprojection_rmse),
    )


def densification_config(args: argparse.Namespace) -> NewSeedDensificationConfig:
    return NewSeedDensificationConfig(
        min_support_views=int(args.densify_min_support_views),
        min_support_ratio=float(args.densify_min_support_ratio),
        min_baseline=float(args.min_camera_translation),
        min_view_angle_deg=float(args.seed_min_ray_angle_deg),
        coverage_radius_cells=float(args.densify_coverage_radius_cells),
        min_child_separation_cells=float(args.densify_min_child_separation_cells),
        max_parent_distance_cells=float(args.densify_max_parent_distance_cells),
        min_world_separation=float(args.densify_min_world_separation),
        footprint_px=float(args.densify_footprint_px),
        max_new_per_frame=int(args.densify_max_new_per_frame),
        max_total_seeds=int(args.densify_max_total_seeds),
        history_views=int(args.densify_history_views),
        interval_frames=int(args.densify_interval),
        opacity=float(args.densify_opacity),
    )


def mask_for_sign_np(obs: SignedXFeatObservation, sign: str) -> torch.Tensor:
    return obs.plus_mask64 if sign == "+" else obs.minus_mask64


def geometry_for_track(
    obs_ids: list[SeedObservationId] | tuple[SeedObservationId, ...],
    observation_index: dict[int, SignedXFeatObservation],
    *,
    max_reprojection_rmse: float,
    args: argparse.Namespace | None = None,
    sign: str | None = None,
) -> SeedGeometryResult:
    observations = []
    kpts = []
    Ks = []
    w2cs = []
    masks = []
    image_sizes = []
    for oid in obs_ids:
        obs = observation_index.get(int(oid.frame_index))
        if obs is None or int(oid.keypoint_index) >= int(obs.keypoints.shape[0]):
            return SeedGeometryResult((0.0, 0.0, 0.0), passed=False, diagnostics={"reason": "missing_observation"})
        observations.append(obs)
        kpts.append(obs.keypoints[int(oid.keypoint_index)])
        Ks.append(obs.K)
        w2cs.append(obs.w2c)
        masks.append(mask_for_sign_np(obs, sign) if sign in {"+", "-"} else None)
        h, w = obs.image_size
        image_sizes.append((w, h))
    if len(kpts) < 2:
        return SeedGeometryResult((0.0, 0.0, 0.0), passed=False, diagnostics={"reason": "insufficient_views"})
    cfg = geometry_gate_config(args) if args is not None else GeometryGateConfig(max_reprojection_rmse_px=max_reprojection_rmse)
    diag = triangulate_track_with_diagnostics(
        torch.stack(kpts).float(),
        torch.stack(Ks).float(),
        torch.stack(w2cs).float(),
        cfg,
        masks=masks,
        image_sizes=image_sizes,
    )
    point = diag["point_world"].detach().cpu().float().flatten()
    passed = bool(diag["valid"].item())
    return SeedGeometryResult(
        tuple(float(v) for v in point.tolist()),
        passed=passed,
        diagnostics={
            "reprojection_rmse": float(diag["reprojection_rmse_px"].detach().cpu().item()),
            "min_ray_angle_deg": float(diag["min_ray_angle_deg"].detach().cpu().item()),
            "rejection_reasons": tuple(diag["rejection_reasons"]),
            "views": len(kpts),
        },
    )


def seed_log_scale_from_support(
    xyz: torch.Tensor,
    support: tuple[SeedObservationId, ...],
    observation_index: dict[int, SignedXFeatObservation],
    *,
    footprint_px: float = 1.5,
) -> float:
    """Return an isotropic log scale matching a small supporting-view footprint."""
    radii: list[float] = []
    xyz_cpu = xyz.detach().cpu().float()
    for oid in support:
        obs = observation_index.get(int(oid.frame_index))
        if obs is None:
            continue
        _, depth = project_point(xyz_cpu, obs)
        focal = math.sqrt(float(obs.K[0, 0]) * float(obs.K[1, 1]))
        if depth > 0.0 and focal > 0.0:
            radii.append(depth * float(footprint_px) / focal)
    radius = float(np.median(radii)) if radii else 0.02
    return math.log(float(np.clip(radius, 1.0e-4, 0.10)))


def known_pose_edges(
    current: SignedXFeatObservation,
    previous: SignedXFeatObservation,
    sign: str,
    args: argparse.Namespace,
) -> list[SeedMatchEdge]:
    cur_mask = mask_for_sign_np(current, sign)
    prev_mask = mask_for_sign_np(previous, sign)
    cur_subset = masked_xfeat_subset(current, sign, min_keypoints=args.seed_min_inmask_keypoints)
    prev_subset = masked_xfeat_subset(previous, sign, min_keypoints=args.seed_min_inmask_keypoints)
    if cur_subset.count == 0 or prev_subset.count == 0:
        return []
    h1, w1 = current.image_size
    h2, w2 = previous.image_size
    matched = match_and_triangulate_masked_pair(
        cur_subset.keypoints.float(),
        cur_subset.descriptors.float(),
        cur_mask,
        current.K.float(),
        current.w2c.float(),
        prev_subset.keypoints.float(),
        prev_subset.descriptors.float(),
        prev_mask,
        previous.K.float(),
        previous.w2c.float(),
        image_size1=(w1, h1),
        image_size2=(w2, h2),
        min_cossim=args.seed_cosine,
        config=geometry_gate_config(args),
    )
    edges: list[SeedMatchEdge] = []
    valid_ids = torch.nonzero(matched.valid.detach().cpu(), as_tuple=False).flatten().tolist()
    for row in valid_ids:
        k_cur = int(cur_subset.keypoint_indices[int(matched.idx1[row])].item())
        k_prev = int(prev_subset.keypoint_indices[int(matched.idx2[row])].item())
        score = float((current.descriptors[k_cur].float() * previous.descriptors[k_prev].float()).sum().item())
        diagnostics = {
            "reprojection_rmse": float(matched.diagnostics.reprojection_rmse_px[row].detach().cpu().item()),
            "sampson_error_px": float(matched.diagnostics.sampson_error_px[row].detach().cpu().item()),
            "ray_angle_deg": float(matched.diagnostics.ray_angle_deg[row].detach().cpu().item()),
            "translation_norm": float(matched.diagnostics.translation_norm.detach().cpu().item()),
        }
        edges.append(SeedMatchEdge(SeedObservationId(current.frame_index, k_cur), SeedObservationId(previous.frame_index, k_prev), score, diagnostics))
    return edges

def binary_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float | int]:
    pred = np.asarray(pred, dtype=bool); gt = np.asarray(gt, dtype=bool)
    tp = int(np.count_nonzero(pred & gt)); fp = int(np.count_nonzero(pred & ~gt)); fn = int(np.count_nonzero(~pred & gt)); tn = int(np.count_nonzero(~pred & ~gt))
    iou = tp / max(tp + fp + fn, 1)
    f1 = (2 * tp) / max(2 * tp + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "pred_positive": int(pred.sum()),
        "gt_positive": int(gt.sum()),
        "pixels": int(pred.size),
        "precision": float(precision),
        "recall": float(recall),
        "iou": float(iou),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().contiguous().cpu().numpy()
    return hashlib.sha256(memoryview(array)).hexdigest()


def snapshot_base(base: GaussianModel) -> dict[str, torch.Tensor]:
    return {name: getattr(base, name).detach().clone() for name in BASE_FIELDS}


def audit_base(base: GaussianModel, before: dict[str, torch.Tensor]) -> dict[str, Any]:
    fields = {}
    for name, old in before.items():
        new = getattr(base, name).detach()
        fields[name] = {
            "shape_before": list(old.shape),
            "shape_after": list(new.shape),
            "bitwise_equal": bool(torch.equal(old, new)),
            "max_abs_difference": float((old - new).abs().max().item()) if old.numel() else 0.0,
            "sha256_before": tensor_sha256(old),
            "sha256_after": tensor_sha256(new),
        }
    return {
        "all_fields_bitwise_equal": all(row["bitwise_equal"] for row in fields.values()),
        "gaussian_count_before": int(before["_xyz"].shape[0]),
        "gaussian_count_after": int(base._xyz.shape[0]),
        "topology_unchanged": int(before["_xyz"].shape[0]) == int(base._xyz.shape[0]),
        "fields": fields,
    }


def scene_records(summary: dict[str, Any], scene: int, max_frames: int | None) -> list[FrameRecord]:
    segment = scene - 1
    rows = [row for row in summary["train_frames"] if int(row["segment_id"]) == segment]
    if max_frames is not None:
        rows = rows[:max_frames]
    return [
        FrameRecord(
            global_index=int(row["global_index"]),
            segment_id=segment,
            name=str(row["name"]),
            image_path=str(row["image_path"]),
            mask_path="",
        )
        for row in rows
    ]


def load_object_annotations(scene: int) -> list[dict[str, Any]]:
    root = DATA / f"scene_change{scene}"
    payload = json.loads((root / "object_change_annotations.json").read_text())
    objects = []
    for index, obj in enumerate(payload["objects"], 1):
        paths = {m["frame_name"]: root / m["mask_path"] for m in obj["segmentation"]["masks"]}
        objects.append(
            {
                "index": index,
                "id": obj["object_mask_id"],
                "change_state": obj["attributes"]["change_state"],
                "change_type": obj["attributes"]["change_type"],
                "paths": paths,
            }
        )
    return objects


def annotation_masks64(
    frame_name: str,
    objects: list[dict[str, Any]],
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    object_masks: list[np.ndarray] = []
    added = np.zeros((64, 64), dtype=bool)
    removed = np.zeros((64, 64), dtype=bool)
    appearance = np.zeros((64, 64), dtype=bool)
    for obj in objects:
        raw = cv2.imread(str(obj["paths"][frame_name]), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise FileNotFoundError(obj["paths"][frame_name])
        mask = cv2.resize((raw > 0).astype(np.float32), (64, 64), interpolation=cv2.INTER_AREA) > 0
        object_masks.append(mask)
        if obj["change_type"] == "APPEARANCE":
            appearance |= mask
        elif obj["change_state"] == "NEW":
            added |= mask
        elif obj["change_state"] == "REMOVED":
            removed |= mask
        else:
            raise ValueError(f"Unknown geometry state: {obj['change_state']}")
    return object_masks, added, removed, appearance


def annotation_masks_at_size(
    frame_name: str,
    objects: list[dict[str, Any]],
    *,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return evaluation-only NEW/REMOVE/APPEARANCE masks at renderer size."""
    added = np.zeros((height, width), dtype=bool)
    removed = np.zeros((height, width), dtype=bool)
    appearance = np.zeros((height, width), dtype=bool)
    for obj in objects:
        raw = cv2.imread(str(obj["paths"][frame_name]), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise FileNotFoundError(obj["paths"][frame_name])
        mask = cv2.resize(
            (raw > 0).astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        if obj["change_type"] == "APPEARANCE":
            appearance |= mask
        elif obj["change_state"] == "NEW":
            added |= mask
        elif obj["change_state"] == "REMOVED":
            removed |= mask
        else:
            raise ValueError(f"Unknown geometry state: {obj['change_state']}")
    return added, removed, appearance


def ssf_loss(target: torch.Tensor, rendered: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    """Exact original O-SCD positive-plus-global-sparsity objective."""
    probability = torch.sigmoid(rendered.mean(dim=0, keepdim=True))
    detection = (target * (1.0 - probability)).mean()
    regularization = torch.log(probability.mean() ** 2 + 1.0)
    loss = detection + regularization
    return loss, {
        "loss": float(loss.detach().item()),
        "detection": float(detection.detach().item()),
        "regularization": float(regularization.detach().item()),
        "probability_mean": float(probability.detach().mean().item()),
    }


def seed_only_coverage_loss(
    target: torch.Tensor,
    rendered_seed: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Direct NEW-seed objective that cannot be explained away by base GS.

    A joint base+seed render attenuates the child gradient after background
    reference Gaussians have already learned a high change score.  Rendering
    the NEW sidecar alone removes that occluding/explaining path while keeping
    the standard joint branch for final evaluation.
    """

    if target.ndim != 3 or target.shape[0] != 1:
        raise ValueError("target must have shape [1,H,W]")
    if rendered_seed.ndim != 3:
        raise ValueError("rendered_seed must have shape [C,H,W]")
    probability = torch.sigmoid(rendered_seed.mean(dim=0, keepdim=True))
    positive = target.sum().clamp_min(1.0)
    negative = (1.0 - target).sum().clamp_min(1.0)
    positive_loss = (target * (1.0 - probability)).sum() / positive
    negative_loss = ((1.0 - target) * probability).sum() / negative
    loss = positive_loss + 0.10 * negative_loss
    return loss, {
        "loss": float(loss.detach().item()),
        "positive": float(positive_loss.detach().item()),
        "negative": float(negative_loss.detach().item()),
        "probability_mean": float(probability.detach().mean().item()),
    }


def train_seed_dc_from_projected_coverage(
    *,
    view: Any,
    seeds: NewSeedGaussianModel,
    optimizer: torch.optim.Optimizer,
    target: torch.Tensor,
    updates: int,
    loss_weight: float,
) -> dict[str, Any]:
    """Train seed DC from a differentiable projected footprint surrogate.

    The local FastGS build can render the full branch but its custom DC/color
    backward is not reliable for this dynamically growing sidecar.  Geometry is
    fixed, so splatting projected isotropic kernels in PyTorch gives the needed
    base-independent DC gradient without touching reference rows.
    """

    active = seeds.active_mask(float(view.timestamp))
    active_indices = torch.nonzero(active, as_tuple=False).flatten()
    if active_indices.numel() == 0:
        return {
            "active_seed_rows": 0,
            "first": None,
            "last": None,
            "target_pixels": int(target.sum().item()),
            "joint_gradient_norm": None,
            "coverage_gradient_norm": None,
        }
    xyz = seeds._xyz[active].detach()
    view_matrix = view.world_view_transform
    homogeneous = torch.cat([xyz, torch.ones((len(xyz), 1), device=xyz.device, dtype=xyz.dtype)], dim=1)
    # Renderer matrices store the transposed transform; row-vector multiply is
    # the established convention used by the camera implementation.
    camera = homogeneous @ view_matrix
    depth = camera[:, 2]
    focal_x = float(view.image_width) / (2.0 * math.tan(float(view.FoVx) * 0.5))
    focal_y = float(view.image_height) / (2.0 * math.tan(float(view.FoVy) * 0.5))
    x = focal_x * camera[:, 0] / depth.clamp_min(1.0e-6) + float(view.image_width) * 0.5
    y = focal_y * camera[:, 1] / depth.clamp_min(1.0e-6) + float(view.image_height) * 0.5
    radius = (
        math.sqrt(focal_x * focal_y)
        * seeds.get_scaling[active].detach().mean(dim=1)
        / depth.clamp_min(1.0e-6)
    ).clamp(0.75, 12.0)
    height, width = int(view.image_height), int(view.image_width)
    visible = (depth > 0.0) & (x >= -12.0) & (x < width + 12.0) & (y >= -12.0) & (y < height + 12.0)
    if not bool(visible.any()):
        return {
            "active_seed_rows": int(active_indices.numel()),
            "first": None,
            "last": None,
            "target_pixels": int(target.sum().item()),
            "joint_gradient_norm": None,
            "coverage_gradient_norm": 0.0,
        }
    selected_rows = active_indices[visible]
    x = x[visible]
    y = y[visible]
    radius = radius[visible]
    opacity = seeds.get_opacity[active][visible, 0].detach()
    row_all: list[torch.Tensor] = []
    col_all: list[torch.Tensor] = []
    seed_all: list[torch.Tensor] = []
    weight_all: list[torch.Tensor] = []
    for local_index in range(len(x)):
        support = int(math.ceil(3.0 * float(radius[local_index])))
        col0 = max(0, int(math.floor(float(x[local_index]))) - support)
        col1 = min(width - 1, int(math.floor(float(x[local_index]))) + support)
        row0 = max(0, int(math.floor(float(y[local_index]))) - support)
        row1 = min(height - 1, int(math.floor(float(y[local_index]))) + support)
        if col1 < col0 or row1 < row0:
            continue
        rows = torch.arange(row0, row1 + 1, device=xyz.device)
        cols = torch.arange(col0, col1 + 1, device=xyz.device)
        grid_y, grid_x = torch.meshgrid(rows, cols, indexing="ij")
        d2 = (grid_x.float() - x[local_index]).square() + (grid_y.float() - y[local_index]).square()
        weight = opacity[local_index] * torch.exp(-0.5 * d2 / radius[local_index].square())
        keep = weight > 1.0e-4
        row_all.append(grid_y[keep])
        col_all.append(grid_x[keep])
        seed_all.append(torch.full_like(grid_y[keep], local_index, dtype=torch.long))
        weight_all.append(weight[keep])
    if not weight_all:
        return {
            "active_seed_rows": int(active_indices.numel()),
            "first": None,
            "last": None,
            "target_pixels": int(target.sum().item()),
            "joint_gradient_norm": None,
            "coverage_gradient_norm": 0.0,
        }
    rows = torch.cat(row_all)
    cols = torch.cat(col_all)
    local_seeds = torch.cat(seed_all)
    weights = torch.cat(weight_all).detach()
    first = last = None
    first_gradient = None
    for step in range(updates):
        optimizer.zero_grad(set_to_none=True)
        dc_scalar = seeds.seed_dc[selected_rows, 0].mean(dim=1)
        flat = torch.zeros(height * width, device=xyz.device, dtype=xyz.dtype)
        flat.index_add_(0, rows * width + cols, weights * dc_scalar[local_seeds])
        rendered_seed = flat.view(1, height, width)
        loss, parts = seed_only_coverage_loss(target, rendered_seed)
        weighted = float(loss_weight) * loss
        weighted.backward()
        if step == 0 and seeds.seed_dc.grad is not None:
            first_gradient = float(seeds.seed_dc.grad.detach().norm().item())
        optimizer.step()
        row = {
            "step": step + 1,
            "loss": float(weighted.detach().item()),
            "joint": None,
            "seed_only_coverage": parts,
            "coverage_weight": float(loss_weight),
            "visible_seed_rows": int(len(selected_rows)),
            "splat_samples": int(len(weights)),
        }
        if first is None:
            first = row
        last = row
    return {
        "active_seed_rows": int(active_indices.numel()),
        "first": first,
        "last": last,
        "target_pixels": int(target.sum().item()),
        "joint_gradient_norm": None,
        "coverage_gradient_norm": first_gradient,
    }


@torch.inference_mode()
def extract_delta(
    view: Any,
    base: GaussianModel,
    pipe: SimpleNamespace,
    background: torch.Tensor,
    sam: Sam2Model,
) -> tuple[torch.Tensor, torch.Tensor]:
    reference = render(view, base, pipe, background)["render"].detach()
    inference = view.original_image[:3].detach()
    pair = torch.stack(
        [
            F.interpolate(reference[None], (1024, 1024), mode="bilinear", align_corners=False)[0],
            F.interpolate(inference[None], (1024, 1024), mode="bilinear", align_corners=False)[0],
        ]
    ).half()
    embedding = sam.get_image_embeddings(pair)[-1].float()
    delta = (embedding[0] - embedding[1]).permute(1, 2, 0).reshape(-1, 256).contiguous()
    return delta, reference


@torch.inference_mode()
def render_previous_sign_memories(
    view: Any,
    base: GaussianModel,
    dc_plus: nn.Parameter,
    dc_minus: nn.Parameter,
    pipe: SimpleNamespace,
    background: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    plus_render = render_change(
        view,
        base,
        pipe,
        background,
        override_dc=dc_plus.detach(),
        override_opacity=base.get_opacity.detach(),
    )["render"].mean(dim=0)
    minus_render = render_change(
        view,
        base,
        pipe,
        background,
        override_dc=dc_minus.detach(),
        override_opacity=base.get_opacity.detach(),
    )["render"].mean(dim=0)
    contrast = plus_render - minus_render
    plus_memory = F.interpolate(F.relu(contrast)[None, None], (64, 64), mode="area")[0, 0]
    minus_memory = F.interpolate(F.relu(-contrast)[None, None], (64, 64), mode="area")[0, 0]
    return (
        plus_memory.cpu().numpy(),
        minus_memory.cpu().numpy(),
        plus_render.cpu().numpy(),
        minus_render.cpu().numpy(),
    )




def train_seed_dc_on_new_mask(
    view: Any,
    base: GaussianModel,
    base_new_dc: nn.Parameter,
    seeds: NewSeedGaussianModel,
    optimizer: torch.optim.Optimizer,
    new64: torch.Tensor,
    pipe: SimpleNamespace,
    background: torch.Tensor,
    updates: int,
    coverage_loss_weight: float = 1.0,
) -> dict[str, Any]:
    if seeds.num_seeds == 0 or not bool(seeds.active_mask(float(view.timestamp)).any().item()):
        return {
            "active_seed_rows": int(seeds.active_mask(float(view.timestamp)).sum().item()) if seeds.num_seeds else 0,
            "first": None,
            "last": None,
            "target_pixels": 0,
            "joint_gradient_norm": None,
            "coverage_gradient_norm": None,
        }
    size = (int(view.image_height), int(view.image_width))
    cue_binary = (view.candidate_map >= 0.5).float()
    target = F.interpolate(new64.float()[None, None], size, mode="nearest")[0] * cue_binary
    if float(coverage_loss_weight) > 0.0:
        return train_seed_dc_from_projected_coverage(
            view=view,
            seeds=seeds,
            optimizer=optimizer,
            target=target,
            updates=updates,
            loss_weight=float(coverage_loss_weight),
        )
    first = last = None
    joint_gradient_norm = None
    coverage_gradient_norm = None
    for step in range(updates):
        optimizer.zero_grad(set_to_none=True)
        # FastGS's custom color backward can exceed its launch configuration
        # when a joint 1.28M-base-row render and a seed-only render both enter
        # one backward graph.  Under the explicit coverage objective, train the
        # seed sidecar only.  Sample the joint gradient once solely as the
        # starvation audit; it never contributes to the optimizer step.
        should_audit_gradient = (
            step == 0
            and seeds.num_seeds <= 256
            and not getattr(seeds, "_gradient_starvation_audited", False)
        )
        joint_parts = None
        if should_audit_gradient:
            concat = build_concatenated_change_view(
                base,
                seeds,
                timestamp=float(view.timestamp),
                base_dc=base_new_dc,
                detach_base_dc=True,
            )
            rendered_joint = render_change(view, concat, pipe, background)["render"]
            joint_loss, joint_parts = ssf_loss(target, rendered_joint)
            joint_gradient = torch.autograd.grad(
                joint_loss,
                seeds.seed_dc,
                allow_unused=True,
            )[0]
            if joint_gradient is not None:
                joint_gradient_norm = float(joint_gradient.detach().norm().item())
            del concat, rendered_joint, joint_loss
        if float(coverage_loss_weight) > 0.0:
            active_only = seeds.active_view(float(view.timestamp))
            rendered_seed = render_change(
                view, active_only, pipe, background, clamp_output=False
            )["render"]
            coverage_loss, coverage_parts = seed_only_coverage_loss(target, rendered_seed)
            loss = float(coverage_loss_weight) * coverage_loss
            if should_audit_gradient:
                direct_gradient = torch.autograd.grad(
                    coverage_loss,
                    seeds.seed_dc,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                if direct_gradient is not None:
                    coverage_gradient_norm = float(direct_gradient.detach().norm().item())
                seeds._gradient_starvation_audited = True
        else:
            concat = build_concatenated_change_view(
                base,
                seeds,
                timestamp=float(view.timestamp),
                base_dc=base_new_dc,
                detach_base_dc=True,
            )
            rendered_joint = render_change(view, concat, pipe, background)["render"]
            loss, joint_parts = ssf_loss(target, rendered_joint)
            coverage_parts = None
        loss.backward()
        optimizer.step()
        row = {
            "step": step + 1,
            "loss": float(loss.detach().item()),
            "joint": joint_parts,
            "seed_only_coverage": coverage_parts,
            "coverage_weight": float(coverage_loss_weight),
        }
        if first is None:
            first = row
        last = row
    return {
        "active_seed_rows": int(seeds.active_mask(float(view.timestamp)).sum().item()),
        "first": first,
        "last": last,
        "target_pixels": int(target.sum().item()),
        "joint_gradient_norm": joint_gradient_norm,
        "coverage_gradient_norm": coverage_gradient_norm,
    }


@torch.inference_mode()
def evaluate_branch_change(
    view: Any,
    base: GaussianModel,
    plus_dc: torch.Tensor,
    minus_dc: torch.Tensor,
    pipe: SimpleNamespace,
    background: torch.Tensor,
    gt_masks64: dict[str, np.ndarray],
    *,
    gt_full: np.ndarray | None = None,
    seeds: NewSeedGaussianModel | None = None,
    new_sign: str | None = None,
) -> dict[str, Any]:
    plus_model: Any = base
    minus_model: Any = base
    plus_override = plus_dc.detach()
    minus_override = minus_dc.detach()
    if seeds is not None and seeds.num_seeds > 0 and new_sign in {"+", "-"}:
        if new_sign == "+":
            plus_model = build_concatenated_change_view(base, seeds, timestamp=float(view.timestamp), base_dc=plus_dc, detach_base_dc=True)
            plus_override = None
        else:
            minus_model = build_concatenated_change_view(base, seeds, timestamp=float(view.timestamp), base_dc=minus_dc, detach_base_dc=True)
            minus_override = None
    plus_render = render_change(
        view, plus_model, pipe, background,
        override_dc=plus_override,
        override_opacity=base.get_opacity.detach() if plus_override is not None else None,
    )["render"].mean(dim=0, keepdim=True)
    minus_render = render_change(
        view, minus_model, pipe, background,
        override_dc=minus_override,
        override_opacity=base.get_opacity.detach() if minus_override is not None else None,
    )["render"].mean(dim=0, keepdim=True)
    # render_change already returns the O-SCD change score.  Applying sigmoid a
    # second time would map every nonnegative pixel to >=0.5 and make the whole
    # image positive.  Match the repository evaluator exactly: mean RGB, clamp,
    # then threshold at 0.5.
    score = torch.maximum(plus_render, minus_render).clamp(0.0, 1.0)
    pred_full = score[0].detach().cpu().numpy() >= 0.5
    score64 = F.interpolate(score[None], (64, 64), mode="area")[0, 0]
    pred64 = score64.detach().cpu().numpy() >= 0.5
    metrics: dict[str, Any] = {
        "predicted_pixels_full": int(np.count_nonzero(pred_full)),
        "score_mean_full": float(score.mean().item()),
        "predicted_cells64": int(np.count_nonzero(pred64)),
        "score_mean64": float(score64.mean().item()),
    }
    if gt_full is not None:
        full = binary_metrics(pred_full, np.asarray(gt_full, dtype=bool))
        for key, value in full.items():
            metrics[f"full_{key}"] = value
    for scope, gt64 in gt_masks64.items():
        scoped = binary_metrics(pred64, np.asarray(gt64, dtype=bool))
        for key, value in scoped.items():
            metrics[f"{scope}_{key}"] = value
    return metrics


def train_current_signed_masks(
    view: Any,
    base: GaussianModel,
    dc_plus: nn.Parameter,
    dc_minus: nn.Parameter,
    optimizer: torch.optim.Optimizer,
    plus64: torch.Tensor,
    minus64: torch.Tensor,
    pipe: SimpleNamespace,
    background: torch.Tensor,
    updates: int,
) -> dict[str, Any]:
    size = (int(view.image_height), int(view.image_width))
    cue_binary = (view.candidate_map >= 0.5).float()
    target_plus = F.interpolate(plus64.float()[None, None], size, mode="nearest")[0] * cue_binary
    target_minus = F.interpolate(minus64.float()[None, None], size, mode="nearest")[0] * cue_binary
    first = last = None
    for step in range(updates):
        optimizer.zero_grad(set_to_none=True)
        rendered_plus = render_change(
            view,
            base,
            pipe,
            background,
            override_dc=dc_plus,
            override_opacity=base.get_opacity.detach(),
        )["render"]
        loss_plus, parts_plus = ssf_loss(target_plus, rendered_plus)
        rendered_minus = render_change(
            view,
            base,
            pipe,
            background,
            override_dc=dc_minus,
            override_opacity=base.get_opacity.detach(),
        )["render"]
        loss_minus, parts_minus = ssf_loss(target_minus, rendered_minus)
        combined = loss_plus + loss_minus
        combined.backward()
        optimizer.step()
        row = {
            "combined": float(combined.detach().item()),
            "plus": parts_plus,
            "minus": parts_minus,
            "step": step + 1,
        }
        if first is None:
            first = row
        last = row
    return {
        "target_plus_pixels": int(target_plus.sum().item()),
        "target_minus_pixels": int(target_minus.sum().item()),
        "first": first,
        "last": last,
    }


def camera_translation(previous: Any | None, current: Any) -> float | None:
    if previous is None:
        return None
    return float(torch.linalg.vector_norm(current.camera_center - previous.camera_center).item())


def distribution(values: list[float]) -> dict[str, Any]:
    array = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=float)
    if not len(array):
        return {"count": 0}
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def aggregate_metric_scope(
    rows: list[dict[str, Any]], branch: str, scope: str
) -> dict[str, Any]:
    count_names = ("tp", "tn", "fp", "fn", "pred_positive", "gt_positive", "pixels")
    counts = {
        name: int(sum(int(row[f"{branch}_{scope}_{name}"]) for row in rows))
        for name in count_names
    }
    tp, tn, fp, fn = counts["tp"], counts["tn"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {
        "branch": branch,
        "scope": scope,
        "frames": len(rows),
        **counts,
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "f1": f1,
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "mean_frame_iou": float(
            np.mean([row[f"{branch}_{scope}_iou"] for row in rows])
        ),
        "mean_frame_f1": float(
            np.mean([row[f"{branch}_{scope}_f1"] for row in rows])
        ),
    }


def aggregate_across_scenes(
    scene_reports: list[dict[str, Any]], branch: str, scope: str
) -> dict[str, Any]:
    entries = [report["metric_scopes"][branch][scope] for report in scene_reports]
    counts = {
        name: int(sum(int(entry[name]) for entry in entries))
        for name in ("tp", "tn", "fp", "fn", "pred_positive", "gt_positive", "pixels")
    }
    tp, tn, fp, fn = counts["tp"], counts["tn"], counts["fp"], counts["fn"]
    return {
        "branch": branch,
        "scope": scope,
        "frames": int(sum(int(entry["frames"]) for entry in entries)),
        **counts,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "mean_frame_iou": float(
            np.average(
                [entry["mean_frame_iou"] for entry in entries],
                weights=[entry["frames"] for entry in entries],
            )
        ),
        "mean_frame_f1": float(
            np.average(
                [entry["mean_frame_f1"] for entry in entries],
                weights=[entry["frames"] for entry in entries],
            )
        ),
    }


def rgb8(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == np.uint8:
        return array
    scale = 255.0 if array.max(initial=0) <= 1.5 else 1.0
    return np.uint8(np.clip(np.round(array * scale), 0, 255))


def resize_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    return cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST) > 0.5


def cue_overlay(rgb: np.ndarray, plus: np.ndarray, minus: np.ndarray) -> np.ndarray:
    base = rgb8(rgb).astype(float)
    h, w = base.shape[:2]
    pos = resize_mask(plus, h, w)
    neg = resize_mask(minus, h, w)
    base[pos] = 0.25 * base[pos] + 0.75 * np.array([235, 45, 45], float)
    base[neg] = 0.25 * base[neg] + 0.75 * np.array([35, 105, 230], float)
    return np.uint8(np.clip(np.round(base), 0, 255))


def memory_overlay(rgb: np.ndarray, plus_memory: np.ndarray, minus_memory: np.ndarray, floor: float) -> np.ndarray:
    base = 0.25 * rgb8(rgb).astype(float)
    h, w = base.shape[:2]
    plus, _ = normalized_memory(plus_memory, floor)
    minus, _ = normalized_memory(minus_memory, floor)
    plus = cv2.resize(plus.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    minus = cv2.resize(minus.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    base[..., 0] += 220.0 * plus
    base[..., 1] += 35.0 * np.minimum(plus, minus)
    base[..., 2] += 220.0 * minus
    return np.uint8(np.clip(np.round(base), 0, 255))


def tracking_overlay(
    rgb: np.ndarray,
    plus_memory: np.ndarray,
    minus_memory: np.ndarray,
    plus_cue: np.ndarray,
    minus_cue: np.ndarray,
    floor: float,
) -> np.ndarray:
    base = 0.25 * rgb8(rgb).astype(float)
    h, w = base.shape[:2]
    wp, _ = normalized_memory(plus_memory, floor)
    wm, _ = normalized_memory(minus_memory, floor)
    same = wp * plus_cue + wm * minus_cue
    conflict = wp * minus_cue + wm * plus_cue
    miss = np.maximum(wp + wm - same - conflict, 0.0)
    same = cv2.resize(same.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    conflict = cv2.resize(conflict.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    miss = cv2.resize(miss.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    base[..., 0] += 225.0 * miss + 180.0 * conflict
    base[..., 1] += 220.0 * same
    base[..., 2] += 220.0 * conflict
    return np.uint8(np.clip(np.round(base), 0, 255))


def gt_overlay(rgb: np.ndarray, added: np.ndarray, removed: np.ndarray, appearance: np.ndarray) -> np.ndarray:
    base = rgb8(rgb).astype(float)
    h, w = base.shape[:2]
    masks = [
        (resize_mask(added, h, w), np.array([35, 190, 220], float)),
        (resize_mask(removed, h, w), np.array([245, 190, 35], float)),
        (resize_mask(appearance, h, w), np.array([145, 145, 145], float)),
    ]
    for mask, color in masks:
        base[mask] = 0.28 * base[mask] + 0.72 * color
    return np.uint8(np.clip(np.round(base), 0, 255))


def panel(array: np.ndarray, title: str, subtitle: str, width: int = 280, height: int = 498) -> Image.Image:
    body = Image.fromarray(rgb8(array)).resize((width, height), Image.Resampling.BILINEAR)
    result = Image.new("RGB", (width, height + 62), "white")
    result.paste(body, (0, 62))
    draw = ImageDraw.Draw(result)
    for y, text, size, color in ((7, title, 15, "black"), (35, subtitle, 9, (65, 65, 65))):
        f = font(size)
        box = draw.textbbox((0, 0), text, font=f)
        draw.text(((width - (box[2] - box[0])) / 2, y), text, font=f, fill=color)
    return result


def posterior_panel(
    rows: list[dict[str, Any]],
    total_frames: int,
    width: int = 280,
    height: int = 560,
) -> Image.Image:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, right, top, bottom = 38, width - 12, 105, height - 90
    draw.text((12, 10), "Online sign posterior", font=font(17), fill="black")
    current = rows[-1]
    draw.text((12, 40), f"Global P(+ = ADD): {current['global_p_plus_is_add']:.3f}", font=font(12), fill=(205, 45, 45))
    draw.text((12, 62), f"Balanced P(+ = ADD): {current['balanced_p_plus_is_add']:.3f}", font=font(12), fill=(35, 95, 210))
    draw.rectangle((left, top, right, bottom), outline=(90, 90, 90), width=1)
    for probability, label in ((0.9, "0.9"), (0.5, "0.5"), (0.1, "0.1")):
        y = int(bottom - probability * (bottom - top))
        draw.line((left, y, right, y), fill=(190, 190, 190), width=1)
        draw.text((5, y - 7), label, font=font(9), fill=(80, 80, 80))

    def point(index: int, value: float) -> tuple[int, int]:
        x = left + int(index / max(total_frames - 1, 1) * (right - left))
        y = bottom - int(np.clip(value, 0, 1) * (bottom - top))
        return x, y

    for field, color in (("global_p_plus_is_add", (220, 45, 45)), ("balanced_p_plus_is_add", (35, 95, 220))):
        points = [point(i, float(row[field])) for i, row in enumerate(rows)]
        if len(points) >= 2:
            draw.line(points, fill=color, width=3)
        elif points:
            x, y = points[0]
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
    status_global = current["global_confirmed_mapping"] or "unresolved"
    status_balanced = current["balanced_confirmed_mapping"] or "unresolved"
    y0 = height - 77
    draw.text((12, y0), f"Global: {status_global}", font=font(11), fill=(205, 45, 45))
    draw.text((12, y0 + 20), f"Balanced: {status_balanced}", font=font(11), fill=(35, 95, 210))
    reason = current.get("global_skip_reason") or "valid evidence"
    draw.text((12, y0 + 42), f"Frame: {reason[:38]}", font=font(9), fill=(70, 70, 70))
    return image


def make_gif_frame(
    scene: int,
    frame_index: int,
    total_frames: int,
    frame_name: str,
    rgb: np.ndarray,
    plus: np.ndarray,
    minus: np.ndarray,
    plus_memory: np.ndarray,
    minus_memory: np.ndarray,
    added: np.ndarray,
    removed: np.ndarray,
    appearance: np.ndarray,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> Image.Image:
    row = rows[-1]
    items = [
        panel(rgb, "Inference RGB", frame_name),
        panel(cue_overlay(rgb, plus, minus), "Strong signed change cue", f"red += {int(plus.sum())} cells; blue -= {int(minus.sum())}"),
        panel(memory_overlay(rgb, plus_memory, minus_memory, args.memory_weight_floor), "Previous DC memory contrast", "red=D+>D-; blue=D->D+; pre-update"),
        panel(tracking_overlay(rgb, plus_memory, minus_memory, plus, minus, args.memory_weight_floor), "Same-sign follow", "green=same; orange=miss; purple=opposite"),
        panel(gt_overlay(rgb, added, removed, appearance), "Geometry annotation (evaluation only)", "cyan=ADD; yellow=REMOVE; gray=appearance"),
        posterior_panel(rows, total_frames),
    ]
    board = Image.new("RGB", (3 * 280, 82 + 2 * 560), "white")
    draw = ImageDraw.Draw(board)
    title = f"SceneChange{scene} causal DC-only sign mapping — frame {frame_index:03d}/{total_frames}"
    subtitle = (
        f"axis cos={row['axis_stability']:.3f}; F+={row['global_follow_plus_text']}; "
        f"F-={row['global_follow_minus_text']}; PCA uses frames 1..{frame_index} only"
    )
    for y, text, size, color in ((10, title, 21, "black"), (47, subtitle, 11, (65, 65, 65))):
        f = font(size)
        box = draw.textbbox((0, 0), text, font=f)
        draw.text(((board.width - (box[2] - box[0])) / 2, y), text, font=f, fill=color)
    for index, item in enumerate(items):
        board.paste(item, ((index % 3) * 280, 82 + (index // 3) * 560))
    return board


def save_gif(frames: list[Image.Image], path: Path) -> None:
    paletted = [
        frame.quantize(colors=96, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        for frame in frames
    ]
    sequence = paletted + [paletted[-1]] * 4
    sequence[0].save(
        path,
        save_all=True,
        append_images=sequence[1:],
        duration=190,
        loop=0,
        optimize=False,
        disposal=2,
    )


def mapping_accuracy(mapping: str, overlap: dict[str, int]) -> dict[str, Any]:
    plus_is_add = mapping.startswith("+=ADD")
    if plus_is_add:
        correct = overlap["plus_add"] + overlap["minus_remove"]
        incorrect = overlap["plus_remove"] + overlap["minus_add"]
    else:
        correct = overlap["plus_remove"] + overlap["minus_add"]
        incorrect = overlap["plus_add"] + overlap["minus_remove"]
    return {
        "mapping": mapping,
        "correct_signed_geometry_cells": correct,
        "incorrect_signed_geometry_cells": incorrect,
        "accuracy": correct / max(correct + incorrect, 1),
    }


def sign_mapping_config(args: argparse.Namespace) -> SignMappingConfig:
    """Build the tested reusable sign-mapping config from runner CLI values."""
    return SignMappingConfig(
        eps_sigma=args.eps_sigma,
        stable_bank_size=args.stable_bank_size,
        axis_stability_min=args.axis_stability_min,
        min_camera_translation=args.min_camera_translation,
        min_sign_cells=args.min_sign_cells,
        min_memory_cells=args.min_memory_cells,
        memory_weight_floor=args.memory_weight_floor,
        component_threshold=args.component_threshold,
        min_component_cells=args.min_component_cells,
        component_area_cap=args.component_area_cap,
        confirm_probability=args.confirm_probability,
        confirm_valid_views=args.confirm_valid_views,
        consensus_probability=args.seed_gate_probability,
    )


def finalize_scene_report(
    scene: int,
    rows: list[dict[str, Any]],
    objects: list[dict[str, Any]],
    object_evidence: dict[int, dict[str, int]],
    overlap: dict[str, int],
    trackers: dict[str, SignPosterior],
    geometry_audit: dict[str, Any],
    runtime_seconds: float,
    outputs: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    score_plus_add = overlap["plus_add"] + overlap["minus_remove"]
    score_plus_remove = overlap["plus_remove"] + overlap["minus_add"]
    geometry_overlap_total = score_plus_add + score_plus_remove
    oracle_mapping = None
    if geometry_overlap_total > 0:
        oracle_mapping = "+=ADD,-=REMOVE" if score_plus_add >= score_plus_remove else "+=REMOVE,-=ADD"
    object_rows = []
    for obj in objects:
        if obj["change_type"] != "GEOMETRY":
            continue
        evidence = object_evidence[obj["index"]]
        sign = "+" if evidence["plus"] >= evidence["minus"] else "-"
        has_evidence = evidence["plus"] + evidence["minus"] > 0
        oracle_plus_add = None if oracle_mapping is None else oracle_mapping.startswith("+=ADD")
        predicted = None
        if has_evidence and oracle_plus_add is not None:
            predicted = "NEW" if ((sign == "+") == oracle_plus_add) else "REMOVED"
        object_rows.append(
            {
                "object_index": obj["index"],
                "object_id": obj["id"],
                "ground_truth_state": obj["change_state"],
                "positive_signed_cells": evidence["plus"],
                "negative_signed_cells": evidence["minus"],
                "dominant_sign": sign if has_evidence else None,
                "label_under_oracle_mapping": predicted,
                "correct_under_oracle_mapping": None if predicted is None else predicted == obj["change_state"],
            }
        )

    tracker_reports = {}
    for name, tracker in trackers.items():
        final_mapping = tracker.confirmed_mapping or mapping_from_probability(tracker.probability_plus_is_add)
        first_correct = None
        first_incorrect = None
        add_follow: list[float] = []
        remove_follow: list[float] = []
        if oracle_mapping is not None:
            first_correct = next(
                (
                    row["frame_name"]
                    for row in rows
                    if row[f"{name}_confirmed_mapping"] == oracle_mapping
                ),
                None,
            )
            first_incorrect = next(
                (
                    row["frame_name"]
                    for row in rows
                    if row[f"{name}_confirmed_mapping"] is not None
                    and row[f"{name}_confirmed_mapping"] != oracle_mapping
                ),
                None,
            )
            if oracle_mapping.startswith("+=ADD"):
                add_follow = [row[f"{name}_follow_plus"] for row in rows if row[f"{name}_valid"]]
                remove_follow = [row[f"{name}_follow_minus"] for row in rows if row[f"{name}_valid"]]
            else:
                add_follow = [row[f"{name}_follow_minus"] for row in rows if row[f"{name}_valid"]]
                remove_follow = [row[f"{name}_follow_plus"] for row in rows if row[f"{name}_valid"]]
        tracker_reports[name] = {
            **tracker.snapshot().as_dict(),
            "final_mapping_for_evaluation": final_mapping,
            "matches_oracle_mapping": None if oracle_mapping is None else final_mapping == oracle_mapping,
            "first_correct_confirmation_frame": first_correct,
            "first_incorrect_confirmation_frame": first_incorrect,
            "mapping_accuracy": None if oracle_mapping is None else mapping_accuracy(final_mapping, overlap),
            "true_add_sign_follow_distribution": distribution(add_follow),
            "true_remove_sign_follow_distribution": distribution(remove_follow),
        }

    valid_geometry_objects = [row for row in object_rows if row["correct_under_oracle_mapping"] is not None]
    return {
        "scene": scene,
        "frames": len(rows),
        "runtime_seconds": runtime_seconds,
        "causal_contract": {
            "pca_prefix_only": True,
            "pre_update_memory_render": True,
            "posterior_updated_before_current_frame_dc_training": True,
            "ground_truth_used_in_causal_pipeline": False,
            "ground_truth_loaded_after_posterior_update_for_reporting_only": True,
        },
        "dc_only_contract": {
            "baseline_optimized_parameters": ["dc_plus", "dc_minus"],
            "seed_branch_optimized_parameters": ["dc_plus", "dc_minus", "seed_dc"],
            "reference_geometry_optimized": False,
            "seed_geometry_optimized": False,
            "reference_topology_edited": False,
            "xyz_opacity_scale_rotation_bitwise_unchanged": all(
                geometry_audit["fields"][name]["bitwise_equal"]
                for name in ("_xyz", "_opacity", "_scaling", "_rotation")
            ),
            "full_base_audit": geometry_audit,
        },
        "ground_truth_sign_overlap": overlap,
        "oracle_mapping_from_geometry_annotations": oracle_mapping,
        "oracle_mapping_available": oracle_mapping is not None,
        "oracle_mapping_scores": {
            "plus_is_add": score_plus_add,
            "plus_is_remove": score_plus_remove,
        },
        "appearance_annotation_policy": (
            "excluded from all mapping accuracy/oracle calculations; not masked from causal inputs "
            "because doing so would leak ground truth"
        ),
        "posterior_methods": tracker_reports,
        "geometry_object_sign_assignment": {
            "objects": object_rows,
            "classified_objects": len(valid_geometry_objects),
            "correct_objects_under_oracle_mapping": sum(
                bool(row["correct_under_oracle_mapping"]) for row in valid_geometry_objects
            ),
            "accuracy_under_oracle_mapping": (None if oracle_mapping is None else
                sum(bool(row["correct_under_oracle_mapping"]) for row in valid_geometry_objects)
                / max(len(valid_geometry_objects), 1)
            ),
        },
        "axis_stability": distribution([row["axis_stability"] for row in rows[1:]]),
        "global_skip_reasons": dict(Counter(row["global_skip_reason"] or "valid" for row in rows)),
        "balanced_skip_reasons": dict(Counter(row["balanced_skip_reason"] or "valid" for row in rows)),
        "configuration": vars(args),
        "outputs": outputs,
    }


def write_comparison_artifacts(
    scene_reports: list[dict[str, Any]], output_dir: Path
) -> dict[str, str]:
    """Write the requested global-vs-component-balanced comparison table."""
    rows: list[dict[str, Any]] = []
    for scene_report in scene_reports:
        for method in ("global", "balanced"):
            posterior = scene_report["posterior_methods"][method]
            mapping_accuracy_row = posterior["mapping_accuracy"]
            rows.append(
                {
                    "scene": f"SceneChange{scene_report['scene']}",
                    "method": "Global pooled" if method == "global" else "Component-balanced",
                    "valid_updates": posterior["valid_updates"],
                    "final_p_plus_is_add": posterior["p_plus_is_add"],
                    "confirmed_mapping": posterior["confirmed_mapping"] or "unresolved",
                    "confirmation_frame": posterior["confirmed_at_frame"] or "--",
                    "mapping_flips": posterior["mapping_flips"],
                    "matches_annotation_mapping": posterior["matches_oracle_mapping"],
                    "signed_geometry_cell_accuracy": (
                        None if mapping_accuracy_row is None else mapping_accuracy_row["accuracy"]
                    ),
                    "geometry_object_accuracy": scene_report["geometry_object_sign_assignment"][
                        "accuracy_under_oracle_mapping"
                    ],
                    "geometry_bitwise_unchanged": scene_report["dc_only_contract"][
                        "xyz_opacity_scale_rotation_bitwise_unchanged"
                    ],
                }
            )

    csv_path = output_dir / "global_vs_component_balanced_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    markdown_path = output_dir / "global_vs_component_balanced_comparison.md"
    lines = [
        "# Online PCA sign mapping: global vs component-balanced",
        "",
        "| Scene | Method | Valid views | Final P(+ = ADD) | Decision | First confirmation | Cell accuracy | Object accuracy |",
        "|---|---:|---:|---:|---|---|---:|---:|",
    ]
    for row in rows:
        cell_accuracy = row["signed_geometry_cell_accuracy"]
        object_accuracy = row["geometry_object_accuracy"]
        lines.append(
            "| {scene} | {method} | {valid_updates} | {probability:.4f} | {decision} | {frame} | {cell} | {obj} |".format(
                scene=row["scene"],
                method=row["method"],
                valid_updates=row["valid_updates"],
                probability=row["final_p_plus_is_add"],
                decision=row["confirmed_mapping"],
                frame=row["confirmation_frame"],
                cell="--" if cell_accuracy is None else f"{100 * cell_accuracy:.2f}%",
                obj="--" if object_accuracy is None else f"{100 * object_accuracy:.2f}%",
            )
        )
    lines.extend(
        [
            "",
            "- `Cell accuracy` and `Object accuracy` use geometry annotations for evaluation only.",
            "- Appearance annotations are excluded from these metrics and never enter the causal posterior.",
            "- Gaussian xyz/opacity/scale/rotation are bitwise unchanged in every row.",
        ]
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    headers = ("Scene", "Method", "Valid", "Final P(+ADD)", "Decision", "Confirm frame", "Cell acc.", "Object acc.")
    widths = (155, 215, 90, 155, 215, 285, 130, 130)
    row_height = 58
    image = Image.new("RGB", (sum(widths), 105 + row_height * (len(rows) + 1)), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 12), "Causal DC-only PCA-sign mapping comparison", font=font(25), fill="black")
    draw.text(
        (20, 53),
        "One scene-level sign mapping; component-balanced changes evidence pooling only",
        font=font(14),
        fill=(65, 65, 65),
    )
    y = 105
    x = 0
    for header, width in zip(headers, widths):
        draw.rectangle((x, y, x + width, y + row_height), fill=(225, 230, 238), outline=(120, 120, 120))
        draw.text((x + 10, y + 17), header, font=font(13), fill="black")
        x += width
    for index, row in enumerate(rows, 1):
        y = 105 + index * row_height
        cell_accuracy = row["signed_geometry_cell_accuracy"]
        object_accuracy = row["geometry_object_accuracy"]
        values = (
            row["scene"],
            row["method"],
            str(row["valid_updates"]),
            f"{row['final_p_plus_is_add']:.4f}",
            row["confirmed_mapping"],
            Path(row["confirmation_frame"]).stem.replace("scene_change", "SC") if row["confirmation_frame"] != "--" else "--",
            "--" if cell_accuracy is None else f"{100 * cell_accuracy:.2f}%",
            "--" if object_accuracy is None else f"{100 * object_accuracy:.2f}%",
        )
        x = 0
        fill = (247, 249, 252) if index % 2 else (235, 240, 247)
        for value, width in zip(values, widths):
            draw.rectangle((x, y, x + width, y + row_height), fill=fill, outline=(150, 150, 150))
            draw.text((x + 10, y + 18), value, font=font(12), fill="black")
            x += width
    png_path = output_dir / "global_vs_component_balanced_comparison.png"
    image.save(png_path)
    return {"csv": str(csv_path), "markdown": str(markdown_path), "visual_table": str(png_path)}




def write_seed_comparison_artifacts(scene_reports: list[dict[str, Any]], output_dir: Path) -> dict[str, str]:
    rows = []
    for report in scene_reports:
        for scope in ("full", "all", "geometry", "new", "remove"):
            baseline = report["metric_scopes"]["baseline"][scope]
            seeded = report["metric_scopes"]["seed"][scope]
            rows.append(
                {
                    "scene": f"SceneChange{report['scene']}",
                    "scope": scope,
                    "frames": report["frames"],
                    "seed_count": report.get("seed_branch_contract", {}).get("seed_count", 0),
                    "baseline_iou": baseline["iou"],
                    "seed_iou": seeded["iou"],
                    "seed_minus_baseline_iou": seeded["iou"] - baseline["iou"],
                    "baseline_f1": baseline["f1"],
                    "seed_f1": seeded["f1"],
                    "seed_minus_baseline_f1": seeded["f1"] - baseline["f1"],
                    "baseline_precision": baseline["precision"],
                    "seed_precision": seeded["precision"],
                    "baseline_recall": baseline["recall"],
                    "seed_recall": seeded["recall"],
                }
            )
    csv_path = output_dir / "baseline_vs_xfeat_new_seed_comparison.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader(); writer.writerows(rows)
    md_path = output_dir / "baseline_vs_xfeat_new_seed_comparison.md"
    lines = [
        "# Baseline DC-only vs XFeat NEW-seed DC-only", "",
        "| Scene | Scope | Frames | Seeds | Baseline IoU | Seed IoU | ΔIoU | Baseline F1 | Seed F1 | ΔF1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    def fmt(v):
        return "--" if v is None else f"{float(v):.4f}"
    for row in rows:
        lines.append(
            f"| {row['scene']} | {row['scope']} | {row['frames']} | {row['seed_count']} | "
            f"{fmt(row['baseline_iou'])} | {fmt(row['seed_iou'])} | {fmt(row['seed_minus_baseline_iou'])} | "
            f"{fmt(row['baseline_f1'])} | {fmt(row['seed_f1'])} | {fmt(row['seed_minus_baseline_f1'])} |"
        )
    lines.extend(
        [
            "",
            "- `full` is the official full renderer resolution; `all` and the object-type scopes are 64×64 diagnostics.",
            "- GT is loaded only after each causal update and is used only for evaluation.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # A compact visual artifact makes the controlled effect and its scale
    # explicit without requiring a long per-frame GIF.
    full_rows = [row for row in rows if row["scope"] == "full"]
    overall_baseline = aggregate_across_scenes(scene_reports, "baseline", "full")
    overall_seed = aggregate_across_scenes(scene_reports, "seed", "full")
    visual_rows = full_rows + [
        {
            "scene": "Overall",
            "scope": "full",
            "frames": overall_baseline["frames"],
            "seed_count": sum(
                report.get("seed_branch_contract", {}).get("seed_count", 0)
                for report in scene_reports
            ),
            "baseline_iou": overall_baseline["iou"],
            "seed_iou": overall_seed["iou"],
            "seed_minus_baseline_iou": overall_seed["iou"] - overall_baseline["iou"],
            "baseline_f1": overall_baseline["f1"],
            "seed_f1": overall_seed["f1"],
            "seed_minus_baseline_f1": overall_seed["f1"] - overall_baseline["f1"],
        }
    ]
    widths = (190, 105, 120, 145, 145, 140, 145, 145, 140)
    headers = ("Scene", "Frames", "Seeds", "Base IoU", "Seed IoU", "Delta IoU", "Base F1", "Seed F1", "Delta F1")
    row_height = 58
    image = Image.new("RGB", (sum(widths), 120 + row_height * (len(visual_rows) + 1)), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 14), "Fixed-geometry DC-only vs XFeat NEW-seed DC-only", font=font(25), fill="black")
    draw.text((20, 54), "Official 941x528 GT; the seed branch adds fixed xyz/scale/rotation/opacity rows and learns seed DC only", font=font(14), fill=(65, 65, 65))
    y = 120
    x = 0
    for header, width in zip(headers, widths):
        draw.rectangle((x, y, x + width, y + row_height), fill=(225, 230, 238), outline=(120, 120, 120))
        draw.text((x + 9, y + 17), header, font=font(13), fill="black")
        x += width
    for index, row in enumerate(visual_rows, 1):
        y = 120 + index * row_height
        values = (
            row["scene"], str(row["frames"]), str(row["seed_count"]),
            f"{row['baseline_iou']:.6f}", f"{row['seed_iou']:.6f}", f"{row['seed_minus_baseline_iou']:+.6f}",
            f"{row['baseline_f1']:.6f}", f"{row['seed_f1']:.6f}", f"{row['seed_minus_baseline_f1']:+.6f}",
        )
        fill = (233, 242, 250) if row["scene"] == "Overall" else ((247, 249, 252) if index % 2 else (238, 242, 247))
        x = 0
        for value, width in zip(values, widths):
            draw.rectangle((x, y, x + width, y + row_height), fill=fill, outline=(150, 150, 150))
            color = (0, 105, 45) if value.startswith("+") else (150, 35, 35) if value.startswith("-") else (0, 0, 0)
            draw.text((x + 9, y + 18), value, font=font(12), fill=color)
            x += width
    png_path = output_dir / "baseline_vs_xfeat_new_seed_comparison.png"
    image.save(png_path)
    return {"csv": str(csv_path), "markdown": str(md_path), "visual_table": str(png_path)}


def assert_seed_acceptance_invariants(
    seed_model: NewSeedGaussianModel,
    seed_manager: NewSeedManager,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Audit hard causal/geometry invariants for every promoted seed."""
    failures: list[dict[str, Any]] = []
    for row, metadata in enumerate(seed_model.metadata):
        birth_kind = metadata.get("birth_kind", "xfeat_triangulated")
        support = list(metadata.get("support", []))
        if birth_kind == "new_only_densified":
            unique_frames = {int(item) for item in metadata.get("support_frames", [])}
            diagnostics = {
                "reprojection_rmse": 0.0,
                "min_ray_angle_deg": metadata.get("max_view_angle_deg", float("-inf")),
            }
        else:
            unique_frames = {int(item[0]) for item in support}
            diagnostics = dict(metadata.get("diagnostics", {}))
        start = float(seed_model.start[row].item())
        checks = {
            "three_unique_views": len(unique_frames) >= 3,
            "support_is_causal": all(frame <= start for frame in unique_frames),
            "reprojection_rmse": (
                birth_kind == "new_only_densified"
                or float(diagnostics.get("reprojection_rmse", float("inf")))
                <= float(args.seed_max_reprojection_rmse)
            ),
            "ray_angle": float(diagnostics.get("min_ray_angle_deg", float("-inf")))
            >= float(args.seed_min_ray_angle_deg),
            "finite_xyz": bool(torch.isfinite(seed_model._xyz[row]).all().item()),
            "valid_lifespan": float(seed_model.start[row].item())
            < float(seed_model.end[row].item()),
            "new_bank_only_parent": (
                birth_kind != "new_only_densified"
                or 0 <= int(metadata.get("parent_seed_row", -1)) < row
            ),
        }
        if not all(checks.values()):
            failures.append({"seed_row": row, "checks": checks})
    result = {
        "passed": not failures,
        "promoted_seed_rows": seed_model.num_seeds,
        "manager_active_seed_count": seed_manager.active_seed_count,
        "active_sidecar_rows": int(torch.isinf(seed_model.end).sum().item()) if seed_model.num_seeds else 0,
        "failures": failures,
    }
    if failures:
        raise RuntimeError(f"NEW-seed acceptance invariant failed: {failures[:3]}")
    return result


def save_active_seed_ply(
    seed_model: NewSeedGaussianModel,
    timestamp: float,
    path: Path,
) -> None:
    """Export active fixed seed rows for 3D inspection without changing state."""
    mask = seed_model.active_mask(timestamp).detach().cpu()
    xyz = seed_model._xyz.detach().cpu()[mask].numpy().astype(np.float32)
    dc = seed_model.seed_dc.detach().cpu()[mask, 0].numpy().astype(np.float32)
    opacity = seed_model.get_opacity.detach().cpu()[mask, 0].numpy().astype(np.float32)
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ("opacity", "f4"),
    ]
    vertices = np.empty(len(xyz), dtype=dtype)
    if len(xyz):
        vertices["x"], vertices["y"], vertices["z"] = xyz.T
        vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"] = dc.T
        vertices["opacity"] = opacity
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertices, "vertex")]).write(str(path))


def run_scene(
    scene: int,
    summary: dict[str, Any],
    cameras: dict[str, dict[str, Any]],
    checkpoint: dict[str, Any],
    sam: Sam2Model,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.time()
    scene_out = args.output_dir / f"scene_change{scene}"
    scene_out.mkdir(parents=True, exist_ok=True)
    frame_dir = scene_out / "gif_frames"
    if args.save_frame_pngs:
        frame_dir.mkdir(exist_ok=True)

    records = scene_records(summary, scene, args.max_frames)
    views, _, _ = build_fixed_cue_views(
        records,
        cameras,
        Path(summary["run_arguments"]["cue_cache_root"]),
        float(summary["resolution"]),
    )
    objects = load_object_annotations(scene)

    # Keep the RGB reference intact for SAM delta extraction.  Change rendering
    # always supplies explicit zero-initialized DC overrides, so the learned
    # change fields do not inherit the reference RGB SH coefficients.
    base = GaussianModel(sh_degree=3, active_sh_degree=0)
    base.load_ply(str(checkpoint["base_ply"]))
    for name in BASE_FIELDS:
        getattr(base, name).requires_grad_(False)
    base_before = snapshot_base(base)

    # Two fair branches start from the same fixed reference topology and DC state.
    # Baseline: original two-sign DC-only memory.
    dc_plus = nn.Parameter(torch.zeros_like(base._features_dc), requires_grad=True)
    dc_minus = nn.Parameter(torch.zeros_like(base._features_dc), requires_grad=True)
    optimizer = torch.optim.Adam([dc_plus, dc_minus], lr=args.lr, eps=1e-15)
    # The paired branches share the exact same sign memories.  The seeded branch
    # differs only by the extra fixed-geometry sidecar, avoiding redundant DC
    # training and guaranteeing a truly controlled comparison.
    seed_model = NewSeedGaussianModel(sh_degree=base.max_sh_degree, device=base._features_dc.device, dtype=base._features_dc.dtype)
    seed_optimizer = torch.optim.Adam(seed_model.optimizer_parameter_groups(args.seed_lr or args.lr), lr=args.seed_lr or args.lr, eps=1e-15)
    seed_manager = NewSeedManager(NewSeedManagerConfig(candidate_ttl_frames=args.seed_candidate_ttl))
    observation_buffer = XFeatObservationBuffer(max_frames=args.seed_buffer_frames)
    observation_index: dict[int, SignedXFeatObservation] = {}
    feature_cache = load_feature_cache(args.feature_cache)
    detector: Detector | None = None
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    pca = CausalPC1(channels=256, stable_bank_size=args.stable_bank_size, seed=args.seed + scene)
    seed_sign_gate = NewSignGate(threshold=args.seed_gate_probability)
    sign_config = sign_mapping_config(args)
    trackers = {
        "global": SignPosterior("global", args.confirm_probability, args.confirm_valid_views),
        "balanced": SignPosterior("balanced", args.confirm_probability, args.confirm_valid_views),
    }

    rows: list[dict[str, Any]] = []
    axes: list[np.ndarray] = []
    gif_frames: list[Image.Image] = []
    seed_match_log: list[dict[str, Any]] = []
    seed_candidate_log: list[dict[str, Any]] = []
    seed_rejection_counts: Counter[str] = Counter()
    masked_keypoint_totals = {"plus": 0, "minus": 0}
    densification_totals: Counter[str] = Counter()
    seed_reprojection_rows: list[dict[str, Any]] = []
    overlap = {"plus_add": 0, "plus_remove": 0, "minus_add": 0, "minus_remove": 0}
    object_evidence = {obj["index"]: {"plus": 0, "minus": 0} for obj in objects}
    previous_view = None
    seed_gate_opened = False
    height0, width0 = int(views[0].image_height), int(views[0].image_width)
    if not args.disable_seeds:
        validate_feature_cache(
            feature_cache,
            top_k=args.xfeat_top_k,
            width=width0,
            height=height0,
        )
        detector = Detector(top_k=args.xfeat_top_k, width=width0, height=height0)

    for frame_index, (record, view) in enumerate(zip(records, views), 1):
        frame_started = time.time()
        delta, reference = extract_delta(view, base, pipe, background, sam)
        cue64 = F.interpolate(view.candidate_map[None], (64, 64), mode="area")[0, 0]
        stable64 = cue64 < args.stable_threshold
        pca_row = pca.update(delta, stable64, args.eps_sigma).as_dict()
        pc = pca_row.pop("pc")
        pca_row.pop("mean")
        axes.append(pc.cpu().numpy())
        score64 = (delta @ pc).reshape(64, 64)
        cue_active = cue64 >= args.cue_threshold
        plus64 = cue_active & (score64 > pca_row["epsilon_positive"])
        minus64 = cue_active & (score64 < pca_row["epsilon_negative"])
        plus_np = plus64.cpu().numpy()
        minus_np = minus64.cpu().numpy()

        if frame_index == 1:
            plus_memory = np.zeros((64, 64), dtype=np.float32)
            minus_memory = np.zeros((64, 64), dtype=np.float32)
            plus_render = np.zeros((view.image_height, view.image_width), dtype=np.float32)
            minus_render = np.zeros_like(plus_render)
        else:
            plus_memory, minus_memory, plus_render, minus_render = render_previous_sign_memories(
                view, base, dc_plus, dc_minus, pipe, background
            )

        seed_features = None
        observation = None
        if not args.disable_seeds:
            seed_features = extract_or_load_xfeat(scene=scene, record=record, view=view, detector=detector, cache=feature_cache)
            observation = make_signed_observation(
                scene=scene, record=record, view=view, cameras=cameras, features=seed_features,
                plus64=plus64, minus64=minus64, cue64=cue64, score64=score64,
            )
            evicted = observation_buffer.add(observation)
            for old in evicted:
                observation_index.pop(old.frame_index, None)
            observation_index[observation.frame_index] = observation
            masked_keypoint_totals["plus"] += masked_xfeat_subset(
                observation, "+", min_keypoints=0
            ).count
            masked_keypoint_totals["minus"] += masked_xfeat_subset(
                observation, "-", min_keypoints=0
            ).count
            if args.feature_cache is not None and frame_index % 16 == 0:
                save_feature_cache(args.feature_cache, feature_cache)

        translation = camera_translation(previous_view, view)
        evidence_by_method: dict[str, tuple[FollowEvidence, FollowEvidence, str | None]] = {}
        for method in ("global", "balanced"):
            evidence_fn = sign_pooled_follow if method == "global" else sign_component_balanced_follow
            plus_evidence = evidence_fn(plus_memory, plus_np, minus_np, sign_config)
            minus_evidence = evidence_fn(minus_memory, minus_np, plus_np, sign_config)
            skip = sign_evidence_skip_reason(
                frame_index,
                pca_row["axis_stability"],
                translation,
                int(plus_np.sum()),
                int(minus_np.sum()),
                plus_evidence,
                minus_evidence,
                sign_config,
            )
            if skip is None:
                assert plus_evidence.follow is not None and minus_evidence.follow is not None
                trackers[method].update(plus_evidence.follow, minus_evidence.follow, record.name)
            evidence_by_method[method] = (plus_evidence, minus_evidence, skip)

        # This is the causal boundary: current-frame DC optimization starts only
        # after both posterior variants consumed their pre-update memory render.
        training = train_current_signed_masks(
            view, base, dc_plus, dc_minus, optimizer, plus64, minus64, pipe, background, args.updates_per_frame
        )

        global_state = trackers["global"].snapshot().as_dict()
        balanced_state = trackers["balanced"].snapshot().as_dict()
        gate_decision = seed_sign_gate.update(
            float(global_state["p_plus_is_add"]),
            float(balanced_state["p_plus_is_add"]),
            record.name,
        )
        new_sign = gate_decision.new_sign
        new_confidence = 0.0 if gate_decision.confidence is None else gate_decision.confidence
        seed_edges: list[SeedMatchEdge] = []
        seed_update = None
        seed_training = {
            "active_seed_rows": seed_model.num_seeds,
            "first": None,
            "last": None,
            "target_pixels": 0,
            "joint_gradient_norm": None,
            "coverage_gradient_norm": None,
        }
        densification_row = {
            "attempted": False,
            "considered_cells": 0,
            "uncovered_cells": 0,
            "children": 0,
            "rejections": {},
        }
        new64 = plus64 if new_sign == "+" else minus64 if new_sign == "-" else None
        if args.disable_seeds or observation is None:
            seed_manager.pause_gate(float(record.global_index))
        elif new_sign is None:
            seed_manager.pause_gate(float(record.global_index))
        else:
            if seed_manager.current_sign not in (None, new_sign):
                # Keep the sidecar lifespan synchronized with the manager's
                # half-open close on a posterior sign flip. Rows are retained.
                seed_model.close_active(float(record.global_index))
            seed_manager.set_gate(new_sign, float(record.global_index))
            history = observation_buffer.backfill(record.global_index, args.seed_buffer_frames) if not seed_gate_opened else observation_buffer.latest(args.seed_prior_views + 1)
            seed_gate_opened = True
            priors = [obs for obs in history if obs.frame_index != observation.frame_index][-args.seed_prior_views:]
            for prior in priors:
                pair_edges = known_pose_edges(observation, prior, new_sign, args)
                seed_edges.extend(pair_edges)
                for edge in pair_edges:
                    seed_match_log.append(
                        {
                            "source_sign": new_sign,
                            "mapping_confidence": new_confidence,
                            "current_frame": observation.frame_index,
                            "prior_frame": prior.frame_index,
                            "current_keypoint": edge.obs_a.keypoint_index
                            if edge.obs_a.frame_index == observation.frame_index
                            else edge.obs_b.keypoint_index,
                            "prior_keypoint": edge.obs_b.keypoint_index
                            if edge.obs_b.frame_index == prior.frame_index
                            else edge.obs_a.keypoint_index,
                            "descriptor_cosine": edge.score,
                            **dict(edge.diagnostics),
                        }
                    )
            seed_update = seed_manager.ingest_edges(
                frame_index=int(record.global_index),
                timestamp=float(record.global_index),
                source_sign=new_sign,
                edges=seed_edges,
                geometry_resolver=lambda obs_ids: geometry_for_track(obs_ids, observation_index, max_reprojection_rmse=args.seed_max_reprojection_rmse, args=args, sign=new_sign),
            )
            for _, reason in seed_update.rejected_edges:
                seed_rejection_counts[reason] += 1
            for candidate in seed_update.candidates:
                seed_candidate_log.append(
                    {
                        "event": "candidate",
                        "frame_index": int(record.global_index),
                        "track_id": candidate.track_id,
                        "source_sign": candidate.source_sign,
                        "unique_views": candidate.unique_frame_count,
                        "xyz": candidate.xyz,
                        "first_observed_time": candidate.first_observed_time,
                        "created_time": candidate.created_time,
                        "expires_after_frame": candidate.expires_after_frame,
                        "diagnostics": dict(candidate.diagnostics),
                    }
                )
            for promotion in seed_update.promotions:
                seed_candidate_log.append(
                    {
                        "event": "promotion",
                        "frame_index": int(record.global_index),
                        "seed_id": promotion.seed_id,
                        "track_id": promotion.track_id,
                        "source_sign": promotion.source_sign,
                        "unique_views": promotion.unique_frame_count,
                        "xyz": promotion.xyz,
                        "first_observed_time": promotion.first_observed_time,
                        "promotion_time": promotion.promotion_time,
                        "start_time": promotion.start_time,
                        "support": [
                            (item.frame_index, item.keypoint_index)
                            for item in promotion.support_observations
                        ],
                        "diagnostics": dict(promotion.diagnostics),
                    }
                )
                xyz = torch.tensor([promotion.xyz], device=base._features_dc.device, dtype=base._features_dc.dtype)
                log_scale = seed_log_scale_from_support(
                    xyz[0], promotion.support_observations, observation_index
                )
                seed_model.append(
                    xyz=xyz,
                    start=float(promotion.start_time),
                    scaling=torch.full((1, 3), log_scale, device=xyz.device, dtype=xyz.dtype),
                    opacity=0.1,
                    metadata={
                        "scene": scene, "seed_id": promotion.seed_id, "track_id": promotion.track_id,
                        "source_sign": promotion.source_sign, "mapping_confidence": new_confidence,
                        "support": [(o.frame_index, o.keypoint_index) for o in promotion.support_observations],
                        "diagnostics": dict(promotion.diagnostics),
                    },
                    optimizer=seed_optimizer,
                )
            if (
                args.new_only_densify
                and seed_model.num_seeds > 0
                and int(record.global_index) % int(args.densify_interval) == 0
                and seed_model.num_seeds < int(args.densify_max_total_seeds)
            ):
                cfg = densification_config(args)
                active_mask = seed_model.active_mask(float(record.global_index))
                active_rows = torch.nonzero(active_mask, as_tuple=False).flatten()
                history_for_densify = observation_buffer.latest(args.densify_history_views)
                batch = densify_confirmed_new_region(
                    current=observation,
                    history=history_for_densify,
                    source_sign=new_sign,
                    active_xyz=seed_model._xyz[active_mask],
                    active_rows=active_rows,
                    config=cfg,
                    max_new=int(args.densify_max_total_seeds) - seed_model.num_seeds,
                )
                densification_row = {
                    "attempted": True,
                    "considered_cells": batch.considered_cells,
                    "uncovered_cells": batch.uncovered_cells,
                    "children": batch.count,
                    "rejections": dict(batch.rejection_counts),
                }
                densification_totals["calls"] += 1
                densification_totals["considered_cells"] += batch.considered_cells
                densification_totals["uncovered_cells"] += batch.uncovered_cells
                densification_totals["children"] += batch.count
                for reason, count in batch.rejection_counts.items():
                    densification_totals[f"rejected_{reason}"] += count
                if batch.count:
                    child_metadata = []
                    for idx in range(batch.count):
                        child_metadata.append(
                            {
                                "scene": scene,
                                "source_sign": new_sign,
                                "mapping_confidence": new_confidence,
                                "promotion_time": float(record.global_index),
                                "support_frames": list(batch.support_frames[idx]),
                                "support_ratio": batch.support_ratios[idx],
                                "visible_views": batch.visible_views[idx],
                                "max_baseline": batch.max_baselines[idx],
                                "max_view_angle_deg": batch.max_view_angles_deg[idx],
                                "geometry_source": "parent_depth_plus_multiview_signed_cue_carving",
                            }
                        )
                    seed_model.append_densified_children(
                        xyz=batch.xyz.to(device=base._features_dc.device, dtype=base._features_dc.dtype),
                        parent_rows=batch.parent_rows.to(device=base._features_dc.device),
                        start=float(record.global_index),
                        scaling=batch.log_scaling.to(device=base._features_dc.device, dtype=base._features_dc.dtype),
                        opacity=float(args.densify_opacity),
                        metadata=child_metadata,
                        optimizer=seed_optimizer,
                    )
            if new64 is not None:
                base_new_dc = dc_plus if new_sign == "+" else dc_minus
                seed_training = train_seed_dc_on_new_mask(
                    view,
                    base,
                    base_new_dc,
                    seed_model,
                    seed_optimizer,
                    new64,
                    pipe,
                    background,
                    args.updates_per_frame,
                    coverage_loss_weight=args.seed_coverage_loss_weight,
                )

        # Evaluation-only annotations are intentionally loaded after posterior
        # and training decisions so they cannot enter the online path.
        object_masks, added64, removed64, appearance64 = annotation_masks64(record.name, objects)
        overlap["plus_add"] += int(np.count_nonzero(plus_np & added64))
        overlap["plus_remove"] += int(np.count_nonzero(plus_np & removed64))
        overlap["minus_add"] += int(np.count_nonzero(minus_np & added64))
        overlap["minus_remove"] += int(np.count_nonzero(minus_np & removed64))
        for obj, mask in zip(objects, object_masks):
            if obj["change_type"] != "GEOMETRY":
                continue
            object_evidence[obj["index"]]["plus"] += int(np.count_nonzero(plus_np & mask))
            object_evidence[obj["index"]]["minus"] += int(np.count_nonzero(minus_np & mask))
        gt_path = DATA / f"scene_change{scene}" / "gt_mask" / record.name
        gt_raw = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if gt_raw is None:
            raise FileNotFoundError(gt_path)
        gt_all64 = cv2.resize(
            (gt_raw > 127).astype(np.uint8),
            (64, 64),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        gt_full = cv2.resize(
            (gt_raw > 127).astype(np.uint8),
            (int(view.image_width), int(view.image_height)),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        gt_masks64 = {
            "all": gt_all64,
            "geometry": added64 | removed64,
            "new": added64,
            "remove": removed64,
        }
        baseline_eval = evaluate_branch_change(
            view, base, dc_plus, dc_minus, pipe, background, gt_masks64,
            gt_full=gt_full,
        )
        active_seed_rows_for_render = (
            int(seed_model.active_mask(float(view.timestamp)).sum().item())
            if seed_model.num_seeds
            else 0
        )
        seed_render_sign = active_seed_render_sign(
            new_sign,
            seed_manager.current_sign,
            active_seed_rows_for_render,
        )
        seed_eval = evaluate_branch_change(
            view,
            base,
            dc_plus,
            dc_minus,
            pipe,
            background,
            gt_masks64,
            gt_full=gt_full,
            seeds=seed_model,
            new_sign=seed_render_sign,
        )
        added_full, removed_full, appearance_full = annotation_masks_at_size(
            record.name,
            objects,
            height=int(view.image_height),
            width=int(view.image_width),
        )
        active_attributes = seed_model.get_active_render_attributes(float(view.timestamp))
        active_indices = torch.nonzero(
            active_attributes["mask"], as_tuple=False
        ).flatten().detach().cpu().tolist()
        for seed_row in active_indices:
            xyz_world = seed_model._xyz[seed_row].detach().cpu()
            uv, depth = project_point(xyz_world, observation)
            x, y = float(uv[0]), float(uv[1])
            xi, yi = int(math.floor(x)), int(math.floor(y))
            inside = (
                depth > 0.0
                and 0 <= xi < int(view.image_width)
                and 0 <= yi < int(view.image_height)
            )
            metadata = seed_model.metadata[seed_row]
            seed_reprojection_rows.append(
                {
                    "frame_index": int(record.global_index),
                    "frame_name": record.name,
                    "seed_row": seed_row,
                    "seed_id": metadata.get("seed_id"),
                    "source_sign": metadata.get("source_sign"),
                    "x": x,
                    "y": y,
                    "depth": depth,
                    "inside_image": inside,
                    "inside_new_gt": bool(inside and added_full[yi, xi]),
                    "inside_remove_gt": bool(inside and removed_full[yi, xi]),
                    "inside_appearance_gt": bool(inside and appearance_full[yi, xi]),
                    "support_frame_max": max(
                        (
                            [int(item[0]) for item in metadata.get("support", [])]
                            + [int(item) for item in metadata.get("support_frames", [])]
                        ),
                        default=None,
                    ),
                    "promotion_start": float(seed_model.start[seed_row].item()),
                    "is_post_promotion_view": float(record.global_index)
                    > float(seed_model.start[seed_row].item()),
                }
            )

        global_plus, global_minus, global_skip = evidence_by_method["global"]
        balanced_plus, balanced_minus, balanced_skip = evidence_by_method["balanced"]
        global_state = trackers["global"].snapshot().as_dict()
        balanced_state = trackers["balanced"].snapshot().as_dict()
        row = {
            "scene": scene,
            "frame_index": frame_index,
            "frame_name": record.name,
            **pca_row,
            "camera_translation": translation,
            "positive_cue_cells": int(plus_np.sum()),
            "negative_cue_cells": int(minus_np.sum()),
            "global_valid": global_skip is None,
            "global_skip_reason": global_skip,
            "global_follow_plus": global_plus.follow,
            "global_follow_minus": global_minus.follow,
            "global_conflict_plus": global_plus.conflict,
            "global_conflict_minus": global_minus.conflict,
            "global_memory_plus_mass": global_plus.mass,
            "global_memory_minus_mass": global_minus.mass,
            "global_memory_plus_cells": global_plus.support_cells,
            "global_memory_minus_cells": global_minus.support_cells,
            "global_p_plus_is_add": global_state["p_plus_is_add"],
            "global_confirmed_mapping": global_state["confirmed_mapping"],
            "global_valid_updates": global_state["valid_updates"],
            "balanced_valid": balanced_skip is None,
            "balanced_skip_reason": balanced_skip,
            "balanced_follow_plus": balanced_plus.follow,
            "balanced_follow_minus": balanced_minus.follow,
            "balanced_conflict_plus": balanced_plus.conflict,
            "balanced_conflict_minus": balanced_minus.conflict,
            "balanced_plus_components": len(balanced_plus.components),
            "balanced_minus_components": len(balanced_minus.components),
            "balanced_p_plus_is_add": balanced_state["p_plus_is_add"],
            "balanced_confirmed_mapping": balanced_state["confirmed_mapping"],
            "balanced_valid_updates": balanced_state["valid_updates"],
            "seed_new_sign": new_sign,
            "seed_render_sign": seed_render_sign,
            "seed_mapping_confidence": new_confidence,
            "seed_gate_open": bool(new_sign is not None and not args.disable_seeds),
            "seed_pair_edges": len(seed_edges),
            "seed_new_candidates": 0 if seed_update is None else len(seed_update.candidates),
            "seed_new_promotions": 0 if seed_update is None else len(seed_update.promotions),
            "seed_candidate_count": seed_manager.candidate_count,
            "seed_active_count": seed_model.num_seeds,
            "seed_active_rows_current": seed_training["active_seed_rows"],
            "new_only_densify_attempted": densification_row["attempted"],
            "new_only_densify_considered_cells": densification_row["considered_cells"],
            "new_only_densify_uncovered_cells": densification_row["uncovered_cells"],
            "new_only_densify_children": densification_row["children"],
            "baseline_iou": baseline_eval["full_iou"],
            "baseline_f1": baseline_eval["full_f1"],
            "baseline_predicted_cells64": baseline_eval["predicted_cells64"],
            "baseline_predicted_pixels_full": baseline_eval["predicted_pixels_full"],
            "seed_iou": seed_eval["full_iou"],
            "seed_f1": seed_eval["full_f1"],
            "seed_predicted_cells64": seed_eval["predicted_cells64"],
            "seed_predicted_pixels_full": seed_eval["predicted_pixels_full"],
            "eval_gt_cells64": baseline_eval["all_gt_positive"],
            "eval_gt_pixels_full": baseline_eval["full_gt_positive"],
            "training_first_combined_loss": training["first"]["combined"],
            "training_last_combined_loss": training["last"]["combined"],
            "training_target_plus_pixels": training["target_plus_pixels"],
            "training_target_minus_pixels": training["target_minus_pixels"],
            "seed_branch_training_last_combined_loss": training["last"]["combined"],
            "seed_dc_training_last_loss": None if seed_training["last"] is None else seed_training["last"]["loss"],
            "seed_dc_training_last_joint_loss": (
                None
                if seed_training["last"] is None
                or seed_training["last"]["joint"] is None
                else seed_training["last"]["joint"]["loss"]
            ),
            "seed_dc_training_last_coverage_loss": (
                None
                if seed_training["last"] is None
                or seed_training["last"]["seed_only_coverage"] is None
                else seed_training["last"]["seed_only_coverage"]["loss"]
            ),
            "seed_dc_training_first_gradient_norm": seed_training["coverage_gradient_norm"],
            "seed_dc_training_first_joint_gradient_norm": seed_training["joint_gradient_norm"],
            "seed_dc_training_target_pixels": seed_training["target_pixels"],
            "gt_plus_add_cells": int(np.count_nonzero(plus_np & added64)),
            "gt_plus_remove_cells": int(np.count_nonzero(plus_np & removed64)),
            "gt_minus_add_cells": int(np.count_nonzero(minus_np & added64)),
            "gt_minus_remove_cells": int(np.count_nonzero(minus_np & removed64)),
            "frame_runtime_seconds": time.time() - frame_started,
        }
        for branch_name, values in (("baseline", baseline_eval), ("seed", seed_eval)):
            for scope in ("full", "all", "geometry", "new", "remove"):
                for metric_name in (
                    "tp",
                    "tn",
                    "fp",
                    "fn",
                    "pred_positive",
                    "gt_positive",
                    "pixels",
                    "precision",
                    "recall",
                    "iou",
                    "f1",
                    "accuracy",
                ):
                    row[f"{branch_name}_{scope}_{metric_name}"] = values[
                        f"{scope}_{metric_name}"
                    ]
        row["global_follow_plus_text"] = "--" if global_plus.follow is None else f"{global_plus.follow:.2f}"
        row["global_follow_minus_text"] = "--" if global_minus.follow is None else f"{global_minus.follow:.2f}"
        rows.append(row)

        if not args.skip_gif:
            rgb = view.original_image[:3].permute(1, 2, 0).detach().cpu().numpy()
            gif_frame = make_gif_frame(
                scene,
                frame_index,
                len(records),
                record.name,
                rgb,
                plus_np,
                minus_np,
                plus_memory,
                minus_memory,
                added64,
                removed64,
                appearance64,
                rows,
                args,
            )
            if args.save_frame_pngs:
                gif_frame.save(frame_dir / f"{frame_index:04d}.png")
            gif_frames.append(gif_frame)

        print(
            f"Scene{scene} {frame_index:03d}/{len(records)} axis={pca_row['axis_stability']:.3f} "
            f"cells=+{int(plus_np.sum())}/-{int(minus_np.sum())} "
            f"F={row['global_follow_plus_text']}/{row['global_follow_minus_text']} "
            f"Pglobal={global_state['p_plus_is_add']:.3f} "
            f"Pbalanced={balanced_state['p_plus_is_add']:.3f} "
            f"skip={global_skip or 'no'} train={row['training_last_combined_loss']:.4f} "
            f"seed={row['seed_active_count']} IoU(base/seed)={row['baseline_iou']:.3f}/{row['seed_iou']:.3f}",
            flush=True,
        )
        previous_view = view
        del delta, reference, score64

    save_feature_cache(args.feature_cache, feature_cache)

    csv_path = scene_out / "frame_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    matches_path = scene_out / "xfeat_seed_matches.jsonl"
    matches_path.write_text(
        "".join(json.dumps(json_safe(item), ensure_ascii=False) + "\n" for item in seed_match_log),
        encoding="utf-8",
    )
    candidates_path = scene_out / "seed_candidates.jsonl"
    candidates_path.write_text(
        "".join(json.dumps(json_safe(item), ensure_ascii=False) + "\n" for item in seed_candidate_log),
        encoding="utf-8",
    )
    reprojection_path = scene_out / "active_seed_reprojections.csv"
    reprojection_fields = [
        "frame_index", "frame_name", "seed_row", "seed_id", "source_sign",
        "x", "y", "depth", "inside_image", "inside_new_gt",
        "inside_remove_gt", "inside_appearance_gt", "support_frame_max",
        "promotion_start", "is_post_promotion_view",
    ]
    with reprojection_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=reprojection_fields)
        writer.writeheader()
        writer.writerows(seed_reprojection_rows)

    arrays_path = scene_out / "causal_pca_posterior_arrays.npz"
    np.savez_compressed(
        arrays_path,
        frame_names=np.asarray([row["frame_name"] for row in rows]),
        pc1_axes=np.stack(axes),
        axis_stability=np.asarray([row["axis_stability"] for row in rows]),
        epsilon_negative=np.asarray([row["epsilon_negative"] for row in rows]),
        epsilon_positive=np.asarray([row["epsilon_positive"] for row in rows]),
        global_p_plus_is_add=np.asarray([row["global_p_plus_is_add"] for row in rows]),
        balanced_p_plus_is_add=np.asarray([row["balanced_p_plus_is_add"] for row in rows]),
    )
    dc_path = scene_out / "final_sign_dc_memories.pt"
    torch.save(
        {
            "baseline_dc_plus": dc_plus.detach().cpu(),
            "baseline_dc_minus": dc_minus.detach().cpu(),
            "seed_sidecar": seed_model.to_checkpoint(),
            "optimizer_parameters": ["dc_plus", "dc_minus", "seed_dc"],
            "geometry_fixed": True,
            "seed_geometry_fixed": True,
            "gaussian_count": int(base.get_xyz.shape[0]),
            "seed_count": int(seed_model.num_seeds),
        },
        dc_path,
    )
    seed_checkpoint_path = scene_out / "new_seed_checkpoint.pt"
    torch.save(
        {
            "contract": "fixed_geometry_xfeat_new_seed_dc_only",
            "complete_render_state": str(dc_path),
            "load_contract": "load baseline_dc_plus, baseline_dc_minus, and seed_sidecar from complete_render_state",
            "seed_sidecar": seed_model.to_checkpoint(),
            "posterior": {
                name: tracker.snapshot().as_dict() for name, tracker in trackers.items()
            },
            "optimizer_parameters": ["dc_plus", "dc_minus", "seed_dc"],
            "densification_calls": int(densification_totals.get("calls", 0)),
            "pruning_calls": 0,
        },
        seed_checkpoint_path,
    )
    active_seed_ply_path = scene_out / "active_new_seeds.ply"
    save_active_seed_ply(
        seed_model,
        timestamp=float(records[-1].global_index),
        path=active_seed_ply_path,
    )

    gif_path = scene_out / f"scene_change{scene}_online_sign_mapping_dc_only.gif"
    if not args.skip_gif:
        save_gif(gif_frames, gif_path)
    geometry_audit = audit_base(base, base_before)
    if not geometry_audit["all_fields_bitwise_equal"]:
        raise RuntimeError(f"DC-only invariant failed in SceneChange{scene}: {geometry_audit}")
    seed_acceptance_audit = assert_seed_acceptance_invariants(seed_model, seed_manager, args)

    outputs = {
        "frame_metrics_csv": str(csv_path),
        "causal_arrays": str(arrays_path),
        "final_dc_memories": str(dc_path),
        "new_seed_checkpoint": str(seed_checkpoint_path),
        "active_new_seeds_ply": str(active_seed_ply_path),
        "xfeat_seed_matches_jsonl": str(matches_path),
        "seed_candidates_jsonl": str(candidates_path),
        "active_seed_reprojections_csv": str(reprojection_path),
        "gif": str(gif_path) if not args.skip_gif else "not_generated",
        "gif_frames": str(frame_dir) if args.save_frame_pngs else "not_saved",
    }
    report = finalize_scene_report(
        scene, rows, objects, object_evidence, overlap, trackers, geometry_audit, time.time() - started, outputs, args
    )
    report["experiment"] = "online_xfeat_new_seed_vs_dc_only"
    report["seed_branch_contract"] = {
        "pose_time_xfeat_once_per_frame": True,
        "known_pose_triangulation": True,
        "posterior_gate": seed_sign_gate.snapshot(),
        "candidate_support_views": 2,
        "promotion_support_views": 3,
        "seed_geometry_trainable": False,
        "seed_dc_trainable": True,
        "seed_count": int(seed_model.num_seeds),
        "xfeat_seed_count": int(
            sum(
                metadata.get("birth_kind", "xfeat_triangulated")
                != "new_only_densified"
                for metadata in seed_model.metadata
            )
        ),
        "densified_seed_count": int(
            sum(
                metadata.get("birth_kind") == "new_only_densified"
                for metadata in seed_model.metadata
            )
        ),
        "manager_active_promotions": seed_manager.active_seed_count,
        "all_promotions_including_closed": len(seed_manager.active_promotions),
        "rejection_histogram": dict(seed_manager.rejection_histogram),
        "pair_and_track_rejection_histogram": {
            **dict(seed_rejection_counts),
            **dict(seed_manager.rejection_histogram),
        },
        "validated_pair_edge_count": len(seed_match_log),
        "candidate_and_promotion_event_count": len(seed_candidate_log),
        "masked_keypoints_seen": masked_keypoint_totals,
        "causality_audit": {
            "all_support_frames_not_after_promotion": all(
                row["support_frame_max"] is None
                or row["support_frame_max"] <= row["promotion_start"]
                for row in seed_reprojection_rows
            ),
            "future_frames_used": False,
            "gt_used_for_seed_birth_or_training": False,
        },
        "held_out_future_view_reprojection": {
            "samples": int(sum(bool(row["is_post_promotion_view"]) for row in seed_reprojection_rows)),
            "inside_image": int(sum(bool(row["inside_image"]) and bool(row["is_post_promotion_view"]) for row in seed_reprojection_rows)),
            "new_gt_hits": int(sum(bool(row["inside_new_gt"]) and bool(row["is_post_promotion_view"]) for row in seed_reprojection_rows)),
            "remove_gt_hits": int(sum(bool(row["inside_remove_gt"]) and bool(row["is_post_promotion_view"]) for row in seed_reprojection_rows)),
            "appearance_gt_hits": int(sum(bool(row["inside_appearance_gt"]) and bool(row["is_post_promotion_view"]) for row in seed_reprojection_rows)),
            "new_gt_precision_given_inside": (
                sum(bool(row["inside_new_gt"]) and bool(row["is_post_promotion_view"]) for row in seed_reprojection_rows)
                / max(sum(bool(row["inside_image"]) and bool(row["is_post_promotion_view"]) for row in seed_reprojection_rows), 1)
            ),
            "remove_gt_leakage_given_inside": (
                sum(bool(row["inside_remove_gt"]) and bool(row["is_post_promotion_view"]) for row in seed_reprojection_rows)
                / max(sum(bool(row["inside_image"]) and bool(row["is_post_promotion_view"]) for row in seed_reprojection_rows), 1)
            ),
        },
        "new_only_densification": {
            "enabled": bool(args.new_only_densify),
            "statistics": dict(densification_totals),
            "reference_bank_edited": False,
            "geometry_trainable": False,
            "depth_source": "nearest visible NEW XFeat seed anchor",
            "carving_source": "causal multi-view signed cue masks",
            "ground_truth_used": False,
            "seed_only_coverage_loss_weight": float(args.seed_coverage_loss_weight),
            "coverage_gradient_diagnostic": distribution(
                [row["seed_dc_training_first_gradient_norm"] for row in rows]
            ),
        },
        "generic_densification_calls": 0,
        "generic_pruning_calls": 0,
        "optimizer_parameters": ["dc_plus", "dc_minus", "seed_dc"],
        "metric_resolution": {
            "full": [int(views[0].image_height), int(views[0].image_width)],
            "all_geometry_new_remove": [64, 64],
        },
        "acceptance_invariant_audit": seed_acceptance_audit,
    }
    report["per_frame_metric_summary"] = {
        "baseline_iou": distribution([row["baseline_iou"] for row in rows]),
        "seed_iou": distribution([row["seed_iou"] for row in rows]),
        "baseline_f1": distribution([row["baseline_f1"] for row in rows]),
        "seed_f1": distribution([row["seed_f1"] for row in rows]),
        "seed_minus_baseline_iou_mean": float(np.mean([row["seed_iou"] - row["baseline_iou"] for row in rows])),
        "seed_minus_baseline_f1_mean": float(np.mean([row["seed_f1"] - row["baseline_f1"] for row in rows])),
    }
    report["metric_scopes"] = {
        branch: {
            scope: aggregate_metric_scope(rows, branch, scope)
            for scope in ("full", "all", "geometry", "new", "remove")
        }
        for branch in ("baseline", "seed")
    }
    report["metric_deltas_seed_minus_baseline"] = {
        scope: {
            metric: report["metric_scopes"]["seed"][scope][metric]
            - report["metric_scopes"]["baseline"][scope][metric]
            for metric in ("precision", "recall", "iou", "f1", "mean_frame_iou", "mean_frame_f1")
        }
        for scope in ("full", "all", "geometry", "new", "remove")
    }
    summary_path = scene_out / "summary.json"
    report["outputs"]["summary"] = str(summary_path)
    summary_path.write_text(json.dumps(json_safe(report), indent=2, ensure_ascii=False) + "\n")

    del base, dc_plus, dc_minus, optimizer, seed_optimizer, seed_model, views, gif_frames
    torch.cuda.empty_cache()
    return report


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((RUN / "summary.json").read_text())
    cameras = load_fixed_camera_index(Path(summary["fixed_cameras_json"]))
    checkpoint = torch.load(RUN / "temporal_rchange_checkpoint.pt", map_location="cpu", weights_only=False)
    sam = Sam2Model.from_pretrained(args.sam_model, local_files_only=True).half().cuda().eval()

    started = time.time()
    scene_reports = []
    for scene in args.scenes:
        scene_reports.append(run_scene(scene, summary, cameras, checkpoint, sam, args))
    comparison_outputs = write_comparison_artifacts(scene_reports, args.output_dir)
    seed_comparison_outputs = write_seed_comparison_artifacts(scene_reports, args.output_dir)
    combined = {
        "experiment": "online_xfeat_new_seed_vs_dc_only_controlled_comparison",
        "created_at_unix": time.time(),
        "runtime_seconds": time.time() - started,
        "repository_files_modified_by_runner": False,
        "scene_pca_and_posterior_reset_independently": True,
        "comparison_outputs": comparison_outputs,
        "seed_comparison_outputs": seed_comparison_outputs,
        "scenes": scene_reports,
    }
    combined["overall_metric_scopes"] = {
        branch: {
            scope: aggregate_across_scenes(scene_reports, branch, scope)
            for scope in ("full", "all", "geometry", "new", "remove")
        }
        for branch in ("baseline", "seed")
    }
    combined["overall_metric_deltas_seed_minus_baseline"] = {
        scope: {
            metric: combined["overall_metric_scopes"]["seed"][scope][metric]
            - combined["overall_metric_scopes"]["baseline"][scope][metric]
            for metric in ("precision", "recall", "iou", "f1", "mean_frame_iou", "mean_frame_f1")
        }
        for scope in ("full", "all", "geometry", "new", "remove")
    }
    combined_path = args.output_dir / "summary.json"
    combined_path.write_text(json.dumps(json_safe(combined), indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(json_safe(combined), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
