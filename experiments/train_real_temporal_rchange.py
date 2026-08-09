"""Pilot runner for oracle-supervised temporal R_change on real ESCD frames.

This script is intentionally self-contained: it keeps one fixed Gaussian topology,
adds a TemporalChangeModel sidecar, and optimizes only state_change_dc. It uses
oracle ground-truth masks to validate temporal state separation; it does not run
BOCD, SAM, densification, pruning, or topology edits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from dataloaders.read_write_model import qvec2rotmat, read_model
from gaussian_renderer import render_change, render_change_temporal
from poses.feature_detector import DescribedKeypoints, Detector
from scene import GaussianModel
from scene.cameras import Camera
from temporal import TemporalChangeModel, compute_ssf_loss


DEFAULT_SOURCE = "data/Instance_1/scene_change1_2_3"
DEFAULT_BOUNDARIES = (95, 199)
BASE_PLY_REL = "reference_reconstruction/point_cloud/iteration_30000/point_cloud.ply"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class FrameRecord:
    """A real inference frame selected for training or probing."""

    global_index: int
    segment_id: int
    name: str
    image_path: str
    mask_path: str


@dataclass
class PoseResult:
    """Pose-estimation output and diagnostics for one inference frame."""

    ok: bool
    frame_name: str
    reference_name: str | None
    Rt: list[list[float]] | None
    matches: int
    inliers: int
    reprojection_rmse: float | None
    reason: str | None = None


@dataclass(frozen=True)
class TrainingStep:
    """One deterministic optimization step for a temporal state and frame."""

    state: int
    view_index: int
    epoch: int | None
    local_step: int


@dataclass
class ReferenceFrame:
    """Reference frame with XFeat descriptors and COLMAP-associated 3D points."""

    name: str
    image_path: str
    desc_kpts: DescribedKeypoints
    pts3d: torch.Tensor
    has_pt3d: torch.Tensor
    Rt: np.ndarray
    focal: float
    fovx: float
    fovy: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def list_images(folder: Path) -> list[str]:
    names = [p.name for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    names.sort()
    if not names:
        raise FileNotFoundError(f"No images found in {folder}")
    return names


def load_rgb_tensor(path: Path, resolution: float) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if resolution > 0.0 and resolution != 1.0:
        image = cv2.resize(
            image,
            (0, 0),
            fx=1.0 / resolution,
            fy=1.0 / resolution,
            interpolation=cv2.INTER_AREA,
        )
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA if image.shape[-1] == 4 else cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    return tensor[:3].cuda(non_blocking=True)


def load_mask_tensor(path: Path, height: int, width: int) -> torch.Tensor:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return (torch.from_numpy(mask).float().cuda(non_blocking=True) / 255.0).clamp(0, 1)[None]


def read_manifest(source_path: Path) -> dict[str, Any]:
    manifest_path = source_path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def boundaries_from_manifest(manifest: dict[str, Any], fallback: tuple[int, ...]) -> tuple[int, ...]:
    counts = manifest.get("counts", {}).get("per_source_scene")
    order = manifest.get("sequence_order")
    if isinstance(counts, dict) and isinstance(order, list) and len(order) >= 2:
        running = 0
        boundaries: list[int] = []
        for scene_name in order[:-1]:
            running += int(counts[scene_name])
            boundaries.append(running)
        return tuple(boundaries)
    return tuple(int(v) for v in fallback)


def segment_id(frame_index: int, boundaries: tuple[int, ...]) -> int:
    for sid, boundary in enumerate(boundaries):
        if frame_index < boundary:
            return sid
    return len(boundaries)


def segment_ranges(total_frames: int, boundaries: tuple[int, ...]) -> list[tuple[int, int]]:
    starts = (0, *boundaries)
    ends = (*boundaries, total_frames)
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def validate_boundaries(total_frames: int, boundaries: tuple[int, ...]) -> None:
    if not boundaries:
        raise ValueError("At least one manual boundary is required")
    if tuple(sorted(set(boundaries))) != boundaries:
        raise ValueError("Boundaries must be unique and strictly increasing")
    if any(boundary <= 0 or boundary >= total_frames for boundary in boundaries):
        raise ValueError(f"Boundaries must lie strictly inside [0, {total_frames})")


def uniform_indices(start: int, end: int, count: int) -> list[int]:
    if end <= start:
        return []
    if count <= 0:
        return []
    if end - start <= count:
        return list(range(start, end))
    values = np.linspace(start, end - 1, count)
    return sorted({int(round(v)) for v in values})


def oracle_training_indices(
    mask_dir: Path,
    names: list[str],
    start: int,
    end: int,
    count: int,
) -> list[int]:
    """Select interior, non-empty oracle-mask frames across one state."""
    nonempty: list[int] = []
    for idx in range(start, end):
        mask = cv2.imread(str(mask_dir / f"{Path(names[idx]).stem}.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(mask_dir / f"{Path(names[idx]).stem}.png")
        if bool((mask > 127).any()):
            nonempty.append(idx)
    if not nonempty:
        return uniform_indices(start, end, count)
    if len(nonempty) <= count:
        return nonempty
    # Avoid endpoint-biased samples, which are often empty or barely visible.
    positions = np.linspace(0, len(nonempty) - 1, count + 2)[1:-1]
    return sorted({nonempty[int(round(pos))] for pos in positions})


def build_frame_records(
    source_path: Path,
    frames_per_state: int,
    probes: list[int],
    boundaries: tuple[int, ...],
    all_training_frames: bool = False,
) -> tuple[list[FrameRecord], list[FrameRecord], list[str]]:
    image_dir = source_path / "inference_scene" / "images"
    mask_dir = source_path / "gt_mask"
    names = list_images(image_dir)
    total = len(names)
    validate_boundaries(total, boundaries)
    if all_training_frames:
        train_indices = list(range(total))
    else:
        train_indices: list[int] = []
        for start, end in segment_ranges(total, boundaries):
            train_indices.extend(
                oracle_training_indices(mask_dir, names, start, end, frames_per_state)
            )
        train_indices = sorted(set(train_indices))
    probe_indices = sorted({idx for idx in probes if 0 <= idx < total})

    def make_record(idx: int) -> FrameRecord:
        name = names[idx]
        mask_path = mask_dir / (Path(name).stem + ".png")
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing oracle GT mask for {name}: {mask_path}")
        return FrameRecord(
            global_index=idx,
            segment_id=segment_id(idx, boundaries),
            name=name,
            image_path=str(image_dir / name),
            mask_path=str(mask_path),
        )

    return [make_record(i) for i in train_indices], [make_record(i) for i in probe_indices], names


def focal2fov(focal: float, pixels: int) -> float:
    return 2.0 * math.atan(pixels / (2.0 * focal))


def camera_intrinsics(camera: Any, actual_width: int, actual_height: int) -> tuple[np.ndarray, float, float, float]:
    if camera.model == "SIMPLE_PINHOLE":
        fx = fy = float(camera.params[0])
        cx = float(camera.params[1])
        cy = float(camera.params[2])
    elif camera.model == "PINHOLE":
        fx = float(camera.params[0])
        fy = float(camera.params[1])
        cx = float(camera.params[2])
        cy = float(camera.params[3])
    else:
        raise ValueError(f"Unsupported COLMAP camera model for PnP: {camera.model}")
    scale_x = actual_width / float(camera.width)
    scale_y = actual_height / float(camera.height)
    fx *= scale_x
    fy *= scale_y
    cx *= scale_x
    cy *= scale_y
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return K, (fx + fy) * 0.5, focal2fov(fx, actual_width), focal2fov(fy, actual_height)


def load_reference_frames(source_path: Path, resolution: float, refs: int, detector: Detector, nn_px: float, actual_width: int, actual_height: int) -> tuple[list[ReferenceFrame], np.ndarray]:
    sparse_dir = source_path / "reference_scene" / "sparse" / "0"
    cameras, images, points3d = read_model(str(sparse_dir))
    ref_image_dir = source_path / "reference_scene" / "images"
    image_names = set(list_images(ref_image_dir))
    colmap_images = [img for img in images.values() if Path(img.name).name in image_names]
    colmap_images.sort(key=lambda img: Path(img.name).name)
    if not colmap_images:
        raise RuntimeError(f"No COLMAP images matched files in {ref_image_dir}")
    selected_ids = uniform_indices(0, len(colmap_images), refs)
    selected = [colmap_images[i] for i in selected_ids]

    ref_frames: list[ReferenceFrame] = []
    K0: np.ndarray | None = None
    for image in selected:
        name = Path(image.name).name
        cam = cameras[image.camera_id]
        tensor = load_rgb_tensor(ref_image_dir / name, resolution)
        if int(tensor.shape[2]) != actual_width or int(tensor.shape[1]) != actual_height:
            raise RuntimeError(
                f"Reference frame {name} resized to {tensor.shape[2]}x{tensor.shape[1]}, "
                f"expected {actual_width}x{actual_height}"
            )
        K, focal, fovx, fovy = camera_intrinsics(cam, actual_width, actual_height)
        if K0 is None:
            K0 = K
        elif not np.allclose(K, K0, rtol=0.0, atol=1e-6):
            raise ValueError("Selected reference frames do not share one camera intrinsic matrix")
        desc = detector(tensor)
        scale_x = actual_width / float(cam.width)
        scale_y = actual_height / float(cam.height)
        colmap_xy = np.asarray(image.xys, dtype=np.float32).copy()
        colmap_xy[:, 0] *= scale_x
        colmap_xy[:, 1] *= scale_y
        point_ids = np.asarray(image.point3D_ids)
        valid_obs = point_ids >= 0
        pts3d = torch.zeros((desc.kpts.shape[0], 3), device="cuda", dtype=torch.float32)
        has_pt3d = torch.zeros(desc.kpts.shape[0], device="cuda", dtype=torch.bool)
        if valid_obs.any():
            detector_xy = desc.kpts.detach().float().cpu().numpy()
            obs_xy = colmap_xy[valid_obs]
            obs_ids = point_ids[valid_obs]
            for k0 in range(0, detector_xy.shape[0], 256):
                block = detector_xy[k0 : k0 + 256]
                distances = np.linalg.norm(block[:, None, :] - obs_xy[None, :, :], axis=2)
                nearest = distances.argmin(axis=1)
                nearest_dist = distances[np.arange(block.shape[0]), nearest]
                for local_idx, dist in enumerate(nearest_dist):
                    if dist <= nn_px:
                        pid = int(obs_ids[int(nearest[local_idx])])
                        if pid in points3d:
                            has_pt3d[k0 + local_idx] = True
                            pts3d[k0 + local_idx] = torch.tensor(points3d[pid].xyz, device="cuda", dtype=torch.float32)
        Rt = np.eye(4, dtype=np.float64)
        Rt[:3, :3] = qvec2rotmat(image.qvec)
        Rt[:3, 3] = image.tvec
        ref_frames.append(ReferenceFrame(name, str(ref_image_dir / name), desc, pts3d, has_pt3d, Rt, focal, fovx, fovy))
    if K0 is None:
        raise RuntimeError("Failed to load reference intrinsics")
    return ref_frames, K0


def match_query_to_reference(query: DescribedKeypoints, ref: ReferenceFrame, max_matches: int) -> tuple[np.ndarray, np.ndarray]:
    q_valid = query.valid
    r_valid = ref.desc_kpts.valid & ref.has_pt3d
    if int(q_valid.sum()) == 0 or int(r_valid.sum()) == 0:
        return np.empty((0, 2), np.float32), np.empty((0, 3), np.float32)
    q_idx = q_valid.nonzero(as_tuple=False).flatten()
    r_idx = r_valid.nonzero(as_tuple=False).flatten()
    q_feat = query.feats[q_idx].float()
    r_feat = ref.desc_kpts.feats[r_idx].float()
    scores = q_feat @ r_feat.T
    best_score, best_r = scores.max(dim=1)
    best_q_for_r = scores.argmax(dim=0)
    q_order = torch.argsort(best_score, descending=True)
    pts2d: list[np.ndarray] = []
    pts3d: list[np.ndarray] = []
    for q_local in q_order[: max_matches * 2].tolist():
        r_local = int(best_r[q_local].item())
        if int(best_q_for_r[r_local].item()) != q_local:
            continue
        q_abs = int(q_idx[q_local].item())
        r_abs = int(r_idx[r_local].item())
        pts2d.append(query.kpts[q_abs].detach().cpu().numpy().astype(np.float32))
        pts3d.append(ref.pts3d[r_abs].detach().cpu().numpy().astype(np.float32))
        if len(pts2d) >= max_matches:
            break
    if not pts2d:
        return np.empty((0, 2), np.float32), np.empty((0, 3), np.float32)
    return np.stack(pts2d), np.stack(pts3d)


def solve_pose_for_frame(record: FrameRecord, resolution: float, detector: Detector, ref_frames: list[ReferenceFrame], K: np.ndarray, min_inliers: int, max_matches: int) -> PoseResult:
    image = load_rgb_tensor(Path(record.image_path), resolution)
    query_desc = detector(image)
    best: PoseResult | None = None
    dist = np.zeros((4, 1), dtype=np.float64)
    for ref in ref_frames:
        pts2d, pts3d = match_query_to_reference(query_desc, ref, max_matches=max_matches)
        if len(pts2d) < 6:
            candidate = PoseResult(False, record.name, ref.name, None, len(pts2d), 0, None, "too_few_matches")
        else:
            ok, rvec, tvec, inlier_idx = cv2.solvePnPRansac(
                pts3d.astype(np.float64),
                pts2d.astype(np.float64),
                K,
                dist,
                iterationsCount=200,
                reprojectionError=8.0,
                confidence=0.999,
                flags=cv2.SOLVEPNP_EPNP,
            )
            inliers = 0 if inlier_idx is None else int(len(inlier_idx))
            if ok and inliers >= min_inliers:
                R, _ = cv2.Rodrigues(rvec)
                Rt = np.eye(4, dtype=np.float64)
                Rt[:3, :3] = R
                Rt[:3, 3] = tvec.reshape(3)
                projected, _ = cv2.projectPoints(pts3d[inlier_idx[:, 0]], rvec, tvec, K, dist)
                residual = projected.reshape(-1, 2) - pts2d[inlier_idx[:, 0]]
                rmse = float(np.sqrt((residual**2).sum(axis=1).mean()))
                candidate = PoseResult(True, record.name, ref.name, Rt.tolist(), len(pts2d), inliers, rmse)
            else:
                candidate = PoseResult(False, record.name, ref.name, None, len(pts2d), inliers, None, "pnp_failed_or_low_inliers")
        if best is None or candidate.inliers > best.inliers:
            best = candidate
    return best or PoseResult(False, record.name, None, None, 0, 0, None, "no_reference_frames")


def pose_reference_signature(
    source_path: Path,
    ref_frames: list[ReferenceFrame],
    K: np.ndarray,
) -> str:
    sparse_dir = source_path / "reference_scene" / "sparse" / "0"
    payload = {
        "source": str(source_path.resolve()),
        "references": [ref.name for ref in ref_frames],
        "K": K.tolist(),
        "sparse_files": {
            name: {
                "size": (sparse_dir / name).stat().st_size,
                "mtime_ns": (sparse_dir / name).stat().st_mtime_ns,
            }
            for name in ("cameras.bin", "images.bin", "points3D.bin")
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def pose_cache_key(
    record: FrameRecord,
    args: argparse.Namespace,
    reference_signature: str,
) -> str:
    fields = {
        "frame_index": record.global_index,
        "frame_name": record.name,
        "resolution": args.resolution,
        "refs": args.refs,
        "kpts": args.kpts,
        "pnp_matches": args.pnp_matches,
        "min_pnp_inliers": args.min_pnp_inliers,
        "reference_association_px": args.ref_association_px,
        "reference_signature": reference_signature,
        "pose_cache_version": 2,
    }
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def estimate_or_load_poses(records: list[FrameRecord], args: argparse.Namespace, detector: Detector, ref_frames: list[ReferenceFrame], K: np.ndarray, cache_path: Path) -> dict[str, PoseResult]:
    cache: dict[str, Any] = {}
    if cache_path.exists() and not args.no_pose_cache:
        with cache_path.open("r", encoding="utf-8") as f:
            cache = json.load(f)
    out: dict[str, PoseResult] = {}
    dirty = False
    reference_signature = pose_reference_signature(Path(args.source_path), ref_frames, K)
    for record in records:
        key = pose_cache_key(record, args, reference_signature)
        cached = cache.get(key)
        if cached is not None:
            cached_result = PoseResult(**cached)
            if not cached_result.ok or cached_result.inliers >= args.min_pnp_inliers:
                out[record.name] = cached_result
                continue
        result = solve_pose_for_frame(
            record,
            args.resolution,
            detector,
            ref_frames,
            K,
            min_inliers=args.min_pnp_inliers,
            max_matches=args.pnp_matches,
        )
        out[record.name] = result
        cache[key] = asdict(result)
        dirty = True
    if dirty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    return out


def make_camera(record: FrameRecord, pose: PoseResult, image: torch.Tensor, focal: float, fovx: float, fovy: float) -> Camera:
    if not pose.ok or pose.Rt is None:
        raise ValueError(f"Cannot build camera for failed pose: {record.name}: {pose.reason}")
    Rt = np.asarray(pose.Rt, dtype=np.float64)
    R = Rt[:3, :3]
    t = Rt[:3, 3]
    uid = f"real_temporal_{record.global_index:06d}"
    view = Camera(
        colmap_id=uid,
        R=np.transpose(R),
        T=t,
        FoVx=fovx,
        FoVy=fovy,
        image=image,
        gt_alpha_mask=None,
        image_name=Path(record.name).stem,
        uid=uid,
    )
    view.timestamp = float(record.global_index)
    view.segment_id = record.segment_id
    return view


def build_temporal_model(base: GaussianModel, boundaries: tuple[int, ...], total_frames: int) -> TemporalChangeModel:
    model = TemporalChangeModel.from_gaussians(base, max_states=len(boundaries) + 1, initial_time=0.0)
    ranges = segment_ranges(total_frames, boundaries)
    with torch.no_grad():
        model.state_change_dc.zero_()
        for state, (start, end) in enumerate(ranges):
            model.state_change_dc[:, state].copy_(base._features_dc.detach())
            model.state_start[:, state].fill_(float(start))
            model.state_end[:, state].fill_(float("inf") if state == len(ranges) - 1 else float(end))
            model.state_valid[:, state].fill_(True)
    return model


def set_trainable_state_only(model: TemporalChangeModel) -> None:
    for name in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"):
        tensor = getattr(model.base, name, None)
        if isinstance(tensor, torch.Tensor):
            tensor.requires_grad_(False)
    model.state_change_dc.requires_grad_(True)


def oscd_positive_sparsity_loss(
    target: torch.Tensor, rendered_change: torch.Tensor
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compatibility wrapper around the original O-SCD SSF objective."""
    loss, values = compute_ssf_loss(target, rendered_change)
    return loss, {
        "loss": float(loss.detach().item()),
        "positive": float(values["detection"].detach().item()),
        "sparsity": float(values["regularization"].detach().item()),
        "pred_mean": float(values["change_probability"].detach().mean().item()),
    }


def training_target(view: Camera) -> torch.Tensor:
    """Return the explicitly attached training signal for one view."""
    target = getattr(view, "training_target", None)
    if not isinstance(target, torch.Tensor):
        raise AttributeError("view.training_target must be a tensor")
    return target


def support_metric_map(view: Camera, threshold: float) -> torch.Tensor:
    """Binarize the explicitly attached support signal for Gaussian lookup."""
    support_map = getattr(view, "support_map", None)
    if not isinstance(support_map, torch.Tensor):
        raise AttributeError("view.support_map must be a tensor")
    if support_map.ndim != 3 or support_map.shape[0] != 1:
        raise ValueError("view.support_map must have shape [1, H, W]")
    return (support_map[0] > float(threshold)).flatten().int()


def build_views(records: list[FrameRecord], poses: dict[str, PoseResult], args: argparse.Namespace, focal: float, fovx: float, fovy: float, split: str) -> tuple[list[Camera], dict[str, torch.Tensor]]:
    views: list[Camera] = []
    gt_masks: dict[str, torch.Tensor] = {}
    for record in records:
        pose = poses[record.name]
        if not pose.ok:
            continue
        image = load_rgb_tensor(Path(record.image_path), args.resolution)
        view = make_camera(record, pose, image, focal, fovx, fovy)
        mask = load_mask_tensor(Path(record.mask_path), view.image_height, view.image_width)
        view.oracle_gt_mask = mask
        view.training_target = mask
        view.support_map = mask
        view.supervision_source = "oracle_gt_mask"
        view.evaluation_split = split
        views.append(view)
        gt_masks[record.name] = mask
    return views, gt_masks


def compute_state_support(
    model: TemporalChangeModel,
    train_views: list[Camera],
    background: torch.Tensor,
    pipe: SimpleNamespace,
    threshold: int,
    map_threshold: float = 0.5,
    support_view_counts_out: torch.Tensor | None = None,
) -> dict[str, Any]:
    support_counts = torch.zeros_like(model.state_valid, dtype=torch.int32)
    support_view_counts = torch.zeros_like(model.state_valid, dtype=torch.int16)
    per_state_frames: dict[int, int] = {sid: 0 for sid in range(model.max_states)}
    for view in train_views:
        metric_map = support_metric_map(view, map_threshold)
        # render_change_temporal does not expose get_flag; use equivalent active slot render through render_change.
        state = int(view.segment_id)
        override_dc = model.state_change_dc[:, state].detach()
        override_opacity = model.base.get_opacity.detach()
        with torch.no_grad():
            pkg = render_change(
                view,
                model.base,
                pipe,
                background,
                get_flag=True,
                metric_map=metric_map,
                override_dc=override_dc,
                override_opacity=override_opacity,
            )
        frame_counts = pkg["accum_metric_counts"].to(torch.int32)
        support_counts[:, state] += frame_counts
        support_view_counts[:, state] += (frame_counts > 0).to(torch.int16)
        per_state_frames[state] += 1
    with torch.no_grad():
        model.state_valid.zero_()
        for state in range(model.max_states):
            valid = support_counts[:, state] >= int(threshold)
            if not bool(valid.any()) and per_state_frames[state] > 0:
                valid = support_counts[:, state] > 0
            model.state_valid[:, state].copy_(valid)
        if support_view_counts_out is not None:
            if support_view_counts_out.shape != support_view_counts.shape:
                raise ValueError("support_view_counts_out has the wrong shape")
            if support_view_counts_out.dtype != support_view_counts.dtype:
                raise ValueError("support_view_counts_out has the wrong dtype")
            if support_view_counts_out.device != support_view_counts.device:
                raise ValueError("support_view_counts_out is on the wrong device")
            support_view_counts_out.copy_(support_view_counts)
    view_histograms = {}
    for state in range(model.max_states):
        view_histograms[str(state)] = {
            str(view_count): int(
                (support_view_counts[:, state] == view_count).sum().item()
            )
            for view_count in range(per_state_frames[state] + 1)
        }
    return {
        "support_threshold": threshold,
        "support_map_threshold": float(map_threshold),
        "support_source": getattr(train_views[0], "supervision_source", "unknown")
        if train_views
        else "unknown",
        "per_state_frames": per_state_frames,
        "valid_gaussians_per_state": model.state_valid.sum(dim=0).detach().cpu().tolist(),
        "supporting_view_count_histogram_per_state": view_histograms,
        "support_counts_checksum": tensor_checksum(support_counts),
    }


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def require_valid_training_poses(records: list[FrameRecord], poses: dict[str, PoseResult]) -> None:
    """Fail exact all-frame mode before silently dropping unsolved training frames."""
    invalid = []
    for record in records:
        pose = poses.get(record.name)
        if pose is None or not pose.ok or pose.Rt is None:
            invalid.append({"frame": record.name, "index": record.global_index, "reason": None if pose is None else pose.reason})
    if invalid:
        raise RuntimeError(
            "Exact all-frame training requires a valid pose for every training record; "
            f"invalid poses: {json.dumps(invalid, indent=2)}"
        )


def manifest_expected_frame_total(manifest: dict[str, Any]) -> int | None:
    counts = manifest.get("counts", {}).get("per_source_scene")
    order = manifest.get("sequence_order")
    if isinstance(counts, dict) and isinstance(order, list) and order:
        try:
            return sum(int(counts[name]) for name in order)
        except (KeyError, TypeError, ValueError):
            return None
    return None


def exact_dataset_contract_audit(
    manifest: dict[str, Any],
    all_names: list[str],
    train_records: list[FrameRecord],
    train_views: list[Any],
    updates_per_frame: int,
) -> dict[str, Any]:
    expected_frames = manifest_expected_frame_total(manifest)
    actual_frames = len(all_names)
    train_record_count = len(train_records)
    train_view_count = len(train_views)
    errors: list[str] = []
    if expected_frames is None:
        errors.append("manifest_missing_expected_frame_total")
    elif expected_frames != actual_frames:
        errors.append(f"manifest_expected_{expected_frames}_but_discovered_{actual_frames}")
    if actual_frames != train_record_count:
        errors.append(f"discovered_{actual_frames}_but_train_records_{train_record_count}")
    if train_record_count != train_view_count:
        errors.append(f"train_records_{train_record_count}_but_train_views_{train_view_count}")
    checks_passed = not errors
    return {
        "expected_dataset_frames": expected_frames,
        "actual_dataset_frames": actual_frames,
        "actual_train_records": train_record_count,
        "actual_train_views": train_view_count,
        "expected_exact_total_updates": (expected_frames * updates_per_frame) if expected_frames is not None else None,
        "dataset_contract_errors": errors,
        "dataset_contract_passed": checks_passed,
    }


def validate_exact_dataset_contract(
    manifest: dict[str, Any],
    all_names: list[str],
    train_records: list[FrameRecord],
    train_views: list[Any],
    updates_per_frame: int,
) -> dict[str, Any]:
    audit = exact_dataset_contract_audit(manifest, all_names, train_records, train_views, updates_per_frame)
    if not audit["dataset_contract_passed"]:
        raise RuntimeError(f"Exact all-frame dataset contract failed: {json.dumps(audit, indent=2)}")
    return audit


def make_state_optimizer(parameter: torch.Tensor, args: argparse.Namespace) -> torch.optim.Adam:
    return torch.optim.Adam([parameter], lr=args.lr, eps=1e-15)


def completed_state_drift_max(before: torch.Tensor | None, after: torch.Tensor, state: int) -> float:
    if before is None or state <= 0:
        return 0.0
    return float((after[:, :state].detach() - before).abs().max().item())


def make_training_schedule(views_by_state: dict[int, list[Any]], max_states: int, args: argparse.Namespace) -> list[TrainingStep]:
    """Build a deterministic legacy or exact balanced training schedule."""
    schedule: list[TrainingStep] = []
    if args.updates_per_frame is not None:
        for state in range(max_states):
            for epoch in range(args.updates_per_frame):
                for view_index, _view in enumerate(views_by_state.get(state, [])):
                    schedule.append(TrainingStep(state, view_index, epoch, epoch))
    else:
        for state in range(max_states):
            state_views = views_by_state.get(state, [])
            if not state_views:
                continue
            for local_step in range(args.steps_state):
                schedule.append(TrainingStep(state, local_step % len(state_views), None, local_step))
    return schedule


def training_schedule_audit(
    train_views: list[Any],
    schedule: list[TrainingStep],
    args: argparse.Namespace,
    dataset_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    per_frame_counts = {getattr(view, "image_name"): 0 for view in train_views}
    state_ids = sorted({int(getattr(view, "segment_id")) for view in train_views})
    views_by_state: dict[int, list[Any]] = {sid: [] for sid in state_ids}
    for view in train_views:
        views_by_state[int(getattr(view, "segment_id"))].append(view)
    for step in schedule:
        view = views_by_state[step.state][step.view_index]
        per_frame_counts[getattr(view, "image_name")] += 1
    counts = list(per_frame_counts.values())
    exact = args.updates_per_frame is not None
    expected_total = (len(train_views) * args.updates_per_frame) if exact else sum(
        args.steps_state for state_views in views_by_state.values() if state_views
    )
    serialized_counts = json.dumps(per_frame_counts, sort_keys=True).encode()
    expected_dataset_frames = None if dataset_audit is None else dataset_audit.get("expected_dataset_frames")
    actual_dataset_frames = len(train_views) if dataset_audit is None else dataset_audit.get("actual_dataset_frames")
    expected_exact_total_updates = (
        None
        if not exact or expected_dataset_frames is None
        else int(expected_dataset_frames) * int(args.updates_per_frame)
    )
    exact_counts_ok = bool(exact and counts and min(counts) == max(counts) == args.updates_per_frame)
    exact_total_ok = bool(not exact or expected_exact_total_updates == len(schedule))
    dataset_ok = bool(dataset_audit and dataset_audit.get("dataset_contract_passed")) if exact else False
    audit = {
        "mode": "updates_per_frame_exact" if exact else "legacy_steps_per_state",
        "requested_updates_per_frame": args.updates_per_frame,
        "expected_total_updates": int(expected_total),
        "actual_total_updates": int(len(schedule)),
        "expected_dataset_frames": expected_dataset_frames,
        "actual_dataset_frames": actual_dataset_frames,
        "expected_exact_total_updates": expected_exact_total_updates,
        "exact_all_images_guarantee": bool(exact and dataset_ok and exact_counts_ok and exact_total_ok),
        "per_frame_update_counts": per_frame_counts,
        "per_frame_update_counts_checksum": hashlib.sha256(serialized_counts).hexdigest(),
        "min_updates_per_frame": int(min(counts)) if counts else 0,
        "max_updates_per_frame": int(max(counts)) if counts else 0,
    }
    if dataset_audit is not None:
        audit.update(dataset_audit)
    return audit


def should_audit_gradient(step_index: int, total_steps: int, interval: int | None) -> bool:
    if total_steps <= 0:
        return False
    if step_index in {1, total_steps}:
        return True
    if interval is None:
        return True
    return interval > 0 and step_index % interval == 0


def classify_slot_gradients(slot_grad_l1: list[float], active_state: int) -> tuple[bool, bool, float]:
    """Separate inactive-slot isolation from active-slot signal availability."""
    inactive_grad_l1 = max(
        (gradient for slot, gradient in enumerate(slot_grad_l1) if slot != active_state),
        default=0.0,
    )
    return inactive_grad_l1 == 0.0, slot_grad_l1[active_state] > 0.0, inactive_grad_l1


def summarize_losses(model: TemporalChangeModel, views: list[Camera], background: torch.Tensor, pipe: SimpleNamespace) -> dict[str, Any]:
    by_state: dict[int, list[float]] = {sid: [] for sid in range(model.max_states)}
    with torch.no_grad():
        for view in views:
            pkg = render_change_temporal(view, model, pipe, background, timestamp=float(view.timestamp))
            loss, parts = oscd_positive_sparsity_loss(training_target(view), pkg["render"])
            del loss
            by_state[int(view.segment_id)].append(parts["loss"])
    return {
        str(state): {
            "frames": len(values),
            "mean_loss": float(np.mean(values)) if values else None,
        }
        for state, values in by_state.items()
    }


def train_temporal_slots(model: TemporalChangeModel, train_views: list[Camera], background: torch.Tensor, pipe: SimpleNamespace, args: argparse.Namespace, dataset_audit: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    optimizer: torch.optim.Adam | None = None
    optimizer_state = None
    logs: list[dict[str, Any]] = []
    views_by_state: dict[int, list[Camera]] = {sid: [] for sid in range(model.max_states)}
    for view in train_views:
        views_by_state[int(view.segment_id)].append(view)
    schedule = make_training_schedule(views_by_state, model.max_states, args)
    schedule_audit = training_schedule_audit(train_views, schedule, args, dataset_audit)
    if schedule_audit["expected_total_updates"] != schedule_audit["actual_total_updates"]:
        raise RuntimeError(f"Training schedule mismatch: {schedule_audit}")
    if args.updates_per_frame is not None and not schedule_audit["exact_all_images_guarantee"]:
        raise RuntimeError(f"Exact per-frame training guarantee failed: {schedule_audit}")

    audit_interval = args.gradient_audit_interval
    if audit_interval is None and args.updates_per_frame is not None:
        audit_interval = 100
    total_steps = len(schedule)
    max_inactive_grad_l1 = 0.0
    isolation_violations = 0
    zero_active_gradient_audits = 0
    audited_steps = 0
    completed_state_drift_checks = 0
    completed_state_drift_violations = 0
    max_completed_state_drift = 0.0
    progress_interval = max(1, int(args.progress_interval))
    start_time = time.time()
    for state, state_views in views_by_state.items():
        if not state_views:
            logs.append({"state": state, "skipped": True, "reason": "no_pose_solved_training_views"})

    for total_step, step in enumerate(schedule, start=1):
        state_changed = optimizer_state != step.state
        if state_changed:
            optimizer = make_state_optimizer(model.state_change_dc, args)
            optimizer_state = step.state
        state_views = views_by_state[step.state]
        view = state_views[step.view_index]
        assert optimizer is not None
        optimizer.zero_grad(set_to_none=True)
        pkg = render_change_temporal(view, model, pipe, background, timestamp=float(view.timestamp))
        loss, parts = oscd_positive_sparsity_loss(training_target(view), pkg["render"])
        loss.backward()
        audit_this_step = should_audit_gradient(total_step, total_steps, audit_interval)
        audit_drift_this_step = step.state > 0 and (audit_this_step or state_changed)
        completed_before = model.state_change_dc[:, : step.state].detach().clone() if audit_drift_this_step else None
        slot_grad_l1 = None
        gradient_isolated = None
        active_gradient_present = None
        if audit_this_step:
            slot_grad_l1 = model.state_change_dc.grad.detach().abs().sum(dim=(0, 2, 3)).cpu().tolist()
            gradient_isolated, active_gradient_present, inactive_grad_l1 = classify_slot_gradients(
                slot_grad_l1, step.state
            )
            max_inactive_grad_l1 = max(max_inactive_grad_l1, inactive_grad_l1)
            isolation_violations += int(not gradient_isolated)
            zero_active_gradient_audits += int(not active_gradient_present)
            audited_steps += 1
        optimizer.step()
        completed_drift = completed_state_drift_max(completed_before, model.state_change_dc, step.state)
        if audit_drift_this_step:
            completed_state_drift_checks += 1
            max_completed_state_drift = max(max_completed_state_drift, completed_drift)
            completed_state_drift_violations += int(completed_drift != 0.0)

        log_step = total_step in {1, total_steps} or (audit_this_step and len(logs) < 20)
        if log_step:
            row = {
                "state": step.state,
                "local_step": step.local_step,
                "epoch": step.epoch,
                "global_step": total_step,
                "frame": view.image_name,
                "audited_gradient": audit_this_step,
                "audited_completed_state_drift": audit_drift_this_step,
                **parts,
            }
            if slot_grad_l1 is not None:
                row.update({
                    "slot_grad_l1": slot_grad_l1,
                    "gradient_isolated_to_active_slot": gradient_isolated,
                    "active_slot_received_gradient": active_gradient_present,
                })
            if audit_drift_this_step:
                row["completed_state_drift_max_abs"] = completed_drift
            logs.append(row)
        if total_step == 1 or total_step == total_steps or total_step % progress_interval == 0:
            elapsed = time.time() - start_time
            print(
                f"[train] step {total_step}/{total_steps} "
                f"state={step.state} frame={view.image_name} loss={parts['loss']:.6f} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
    return logs, {
        "optimized_steps": total_steps,
        "audited_steps": audited_steps,
        "audit_interval": audit_interval,
        "checked_steps": audited_steps,
        "violations": isolation_violations,
        "zero_active_gradient_audits": zero_active_gradient_audits,
        "max_inactive_slot_grad_l1": max_inactive_grad_l1,
        "all_audited_steps_isolated": isolation_violations == 0,
        "all_steps_isolated": (isolation_violations == 0) if audited_steps == total_steps else None,
        "optimizer_reset_on_state_boundary": True,
        "completed_state_drift_checks": completed_state_drift_checks,
        "completed_state_drift_max_abs": max_completed_state_drift,
        "completed_state_drift_passed": completed_state_drift_violations == 0,
    }, schedule_audit

def evaluate_views(model: TemporalChangeModel, views: list[Camera], background: torch.Tensor, pipe: SimpleNamespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for view in views:
            pkg = render_change_temporal(view, model, pipe, background, timestamp=float(view.timestamp))
            pred = pkg["render"].mean(dim=0, keepdim=True)
            gt = view.oracle_gt_mask > 0.5
            binary = pred > 0.5
            inter = (binary & gt).sum().float()
            union = (binary | gt).sum().float()
            pred_area = binary.sum().float()
            gt_area = gt.sum().float()
            iou = torch.where(union > 0, inter / union, torch.ones_like(union))
            denominator = pred_area + gt_area
            f1 = torch.where(
                denominator > 0,
                2.0 * inter / denominator,
                torch.ones_like(denominator),
            )
            rows.append(
                {
                    "frame": view.image_name,
                    "timestamp": float(view.timestamp),
                    "segment_id": int(view.segment_id),
                    "split": view.evaluation_split,
                    "pred_mean": float(pred.mean().item()),
                    "gt_mean": float(view.oracle_gt_mask.mean().item()),
                    "iou_at_0_5": float(iou.item()),
                    "f1_at_0_5": float(f1.item()),
                }
            )
    return rows


def tensor_checksum(tensor: torch.Tensor) -> str:
    data = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def file_checksum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def image_panel(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.detach().float().cpu().clamp(0, 1)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr.repeat(3, 1, 1)
    if arr.ndim == 2:
        arr = arr[None].repeat(3, 1, 1)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = arr.permute(1, 2, 0)
    return Image.fromarray((arr.numpy() * 255.0).astype(np.uint8))


def render_forced_slot(view: Camera, model: TemporalChangeModel, pipe: SimpleNamespace, background: torch.Tensor, slot: int) -> torch.Tensor:
    with torch.no_grad():
        opacity = model.base.get_opacity * model.state_valid[:, slot : slot + 1].float()
        pkg = render_change(view, model.base, pipe, background, override_dc=model.state_change_dc[:, slot], override_opacity=opacity)
        return pkg["render"].mean(dim=0, keepdim=True)


def save_probe_grid(
    model: TemporalChangeModel,
    probe_views: list[Camera],
    background: torch.Tensor,
    pipe: SimpleNamespace,
    output_path: Path,
) -> tuple[list[str], list[dict[str, Any]]]:
    output_path.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    saved: list[str] = []
    diagnostics: list[dict[str, Any]] = []
    for view in probe_views:
        with torch.no_grad():
            rgb = view.original_image[:3]
            current = render_change_temporal(
                view,
                model,
                pipe,
                background,
                timestamp=float(view.timestamp),
            )["render"].mean(dim=0, keepdim=True)
            forced = [render_forced_slot(view, model, pipe, background, slot) for slot in range(model.max_states)]
            active_slot = int(view.segment_id)
            inactive = [image for slot, image in enumerate(forced) if slot != active_slot]
            inactive_union = torch.stack(inactive, dim=0).amax(dim=0)
            current_binary = current > 0.5
            inactive_binary = inactive_union > 0.5
            diagnostics.append(
                {
                    "frame": view.image_name,
                    "timestamp": float(view.timestamp),
                    "active_slot": active_slot,
                    "split": view.evaluation_split,
                    "current_vs_active_slot_max_abs": float(
                        (current - forced[active_slot]).abs().max().item()
                    ),
                    "current_pixels_at_0_5": int(current_binary.sum().item()),
                    "inactive_slot_union_pixels_at_0_5": int(
                        inactive_binary.sum().item()
                    ),
                    "inactive_pixels_suppressed_from_current": int(
                        (inactive_binary & ~current_binary).sum().item()
                    ),
                }
            )
        panels = [("RGB", rgb), ("GT oracle", view.oracle_gt_mask), ("current", current)]
        panels.extend((f"forced slot {slot}", img) for slot, img in enumerate(forced))
        panels.append(("inactive-slot union", inactive_union))
        pil_panels = [image_panel(img) for _, img in panels]
        thumb_w = max(96, min(220, pil_panels[0].width))
        thumb_h = int(round(pil_panels[0].height * (thumb_w / pil_panels[0].width)))
        label_h = 24
        canvas = Image.new("RGB", (thumb_w * len(pil_panels), thumb_h + label_h), "white")
        draw = ImageDraw.Draw(canvas)
        for col, ((label, _), pil_img) in enumerate(zip(panels, pil_panels)):
            pil_img = pil_img.resize((thumb_w, thumb_h), Image.Resampling.BILINEAR)
            x = col * thumb_w
            canvas.paste(pil_img, (x, label_h))
            draw.text((x + 4, 4), label, fill=(0, 0, 0), font=font)
        out_file = output_path / f"probe_{int(view.timestamp):06d}_{view.image_name}.png"
        canvas.save(out_file)
        saved.append(str(out_file))
    return saved, diagnostics


def save_checkpoint(model: TemporalChangeModel, args: argparse.Namespace, output_dir: Path, manifest: dict[str, Any], train_records: list[FrameRecord], probe_records: list[FrameRecord], pose_results: dict[str, PoseResult], support: dict[str, Any], loss_summary: dict[str, Any], train_log: list[dict[str, Any]], gradient_audit: dict[str, Any], schedule_audit: dict[str, Any], metrics: list[dict[str, Any]], visual_paths: list[str], probe_diagnostics: list[dict[str, Any]], K: np.ndarray) -> tuple[Path, Path]:
    ckpt_path = output_dir / "temporal_rchange_checkpoint.pt"
    summary_path = output_dir / "summary.json"
    state_dict = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
    }
    run_arguments = {
        name: value for name, value in vars(args).items() if name != "started_at"
    }
    script_sha256 = file_checksum(Path(__file__))
    base_ply_path = (Path(args.source_path) / BASE_PLY_REL).resolve()
    base_ply_sha256 = file_checksum(base_ply_path)
    state_valid_counts = model.state_valid.sum(dim=0).detach().cpu().tolist()
    checkpoint_metadata = {
        "schema_version": 1,
        "contract": "fixed_topology_temporal_sidecar_state_change_dc_only_oracle_supervision",
        "source_path": str(Path(args.source_path).resolve()),
        "base_ply_sha256": base_ply_sha256,
        "script_sha256": script_sha256,
        "run_arguments": run_arguments,
        "boundaries": list(args.boundaries),
        "camera_intrinsics": K.tolist(),
        "manifest_counts": manifest.get("counts"),
        "support": support,
        "schedule": schedule_audit,
        "gradient_isolation_audit": gradient_audit,
        "state_valid_counts": state_valid_counts,
    }
    torch.save(
        {
            "state_dict": state_dict,
            "base_ply": str(base_ply_path),
            "boundaries": list(args.boundaries),
            "contract": checkpoint_metadata["contract"],
            "metadata": checkpoint_metadata,
        },
        ckpt_path,
    )
    summary = {
        "script": "experiments/train_real_temporal_rchange.py",
        "created_at_unix": time.time(),
        "runtime_seconds": time.time() - args.started_at,
        "source_path": args.source_path,
        "oracle_supervision": True,
        "bocd": False,
        "sam": False,
        "fixed_gaussian_topology": True,
        "optimized_parameters": ["TemporalChangeModel.state_change_dc"],
        "densify_prune": False,
        "resolution": args.resolution,
        "boundaries": list(args.boundaries),
        "run_arguments": run_arguments,
        "config": {
            "refs": args.refs,
            "kpts": args.kpts,
            "training_mode": schedule_audit["mode"],
            "all_training_frames": args.updates_per_frame is not None,
            "frames_per_state": args.frames_state if args.updates_per_frame is None else None,
            "steps_per_state": args.steps_state if args.updates_per_frame is None else None,
            "updates_per_frame": args.updates_per_frame,
            "gradient_audit_interval": args.gradient_audit_interval,
            "progress_interval": args.progress_interval,
            "learning_rate": args.lr,
            "support_threshold": args.support_threshold,
            "reference_association_px": args.ref_association_px,
            "minimum_pnp_inliers": args.min_pnp_inliers,
            "pnp_matches": args.pnp_matches,
            "seed": args.seed,
        },
        "manifest_counts": manifest.get("counts"),
        "train_frames": [asdict(r) for r in train_records],
        "probe_frames": [asdict(r) for r in probe_records],
        "probe_training_overlap": sorted(
            {record.name for record in train_records}
            & {record.name for record in probe_records}
        ),
        "probe_metrics_are_held_out": not bool(
            {record.name for record in train_records}
            & {record.name for record in probe_records}
        ),
        "pose_results": {name: asdict(result) for name, result in pose_results.items()},
        "support": support,
        "loss_summary": loss_summary,
        "train_log": train_log,
        "training_schedule": schedule_audit,
        "gradient_isolation_audit": gradient_audit,
        "metrics": metrics,
        "visual_paths": visual_paths,
        "probe_diagnostics": probe_diagnostics,
        "gaussian_count": int(model.state_change_dc.shape[0]),
        "state_change_dc_shape": list(model.state_change_dc.shape),
        "trainable_parameter_names": [
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        ],
        "all_gradient_isolation_checks_passed": gradient_audit["all_audited_steps_isolated"],
        "camera_intrinsics": K.tolist(),
        "state_valid_counts": state_valid_counts,
        "states_per_gaussian_histogram": {
            str(count): int((model.state_valid.sum(dim=1) == count).sum().item())
            for count in range(model.max_states + 1)
        },
        "state_change_dc_checksum": tensor_checksum(model.state_change_dc),
        "script_sha256": script_sha256,
        "base_ply_sha256": base_ply_sha256,
        "checkpoint_sha256": file_checksum(ckpt_path),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return ckpt_path, summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Oracle-supervised real-frame temporal R_change pilot")
    parser.add_argument("--source-path", default=DEFAULT_SOURCE, help="Combined ESCD/O-SCD source scene")
    parser.add_argument("--output-dir", default="outputs/real_temporal_rchange_pilot")
    parser.add_argument("--resolution", type=float, default=8.0, help="Image downsampling factor")
    parser.add_argument("--refs", type=int, default=16, help="Uniform reference COLMAP frames")
    parser.add_argument("--kpts", type=int, default=512, help="XFeat keypoints per frame")
    parser.add_argument("--frames-state", type=int, default=3, help="Representative training frames per oracle state")
    parser.add_argument("--steps-state", type=int, default=30, help="Optimization steps per oracle state")
    parser.add_argument("--updates-per-frame", type=positive_int, default=None, metavar="K", help="Exact balanced mode: train every inference frame K times")
    parser.add_argument("--gradient-audit-interval", type=positive_int, default=None, help="Audit gradient isolation every N optimized steps; default is all legacy steps and every 100 exact-mode steps")
    parser.add_argument("--progress-interval", type=positive_int, default=100, help="Print compact training progress every N optimized steps")
    parser.add_argument("--probes", nargs="+", type=int, default=[94, 95, 198, 199], help="Global frame indices for visual probes")
    parser.add_argument(
        "--boundaries",
        nargs="+",
        type=int,
        default=None,
        help="Oracle global boundaries; derived from manifest when omitted",
    )
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--support-threshold", type=int, default=1)
    parser.add_argument("--ref-association-px", type=float, default=3.0)
    parser.add_argument("--min-pnp-inliers", type=int, default=12)
    parser.add_argument("--pnp-matches", type=int, default=256)
    parser.add_argument("--no-pose-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.started_at = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required because the existing renderer and Camera allocate CUDA tensors")
    seed_everything(args.seed)
    source_path = Path(args.source_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(source_path)
    manifest_boundaries = boundaries_from_manifest(manifest, DEFAULT_BOUNDARIES)
    args.boundaries = (
        tuple(args.boundaries) if args.boundaries is not None else manifest_boundaries
    )

    exact_training = args.updates_per_frame is not None
    train_records, probe_records, all_names = build_frame_records(
        source_path,
        args.frames_state,
        args.probes,
        tuple(args.boundaries),
        all_training_frames=exact_training,
    )
    first_image = load_rgb_tensor(Path(train_records[0].image_path), args.resolution)
    height, width = int(first_image.shape[1]), int(first_image.shape[2])
    detector = Detector(args.kpts, width, height)
    ref_frames, K = load_reference_frames(source_path, args.resolution, args.refs, detector, args.ref_association_px, width, height)
    focal = ref_frames[0].focal
    fovx = ref_frames[0].fovx
    fovy = ref_frames[0].fovy

    needed_records = sorted({r.name: r for r in [*train_records, *probe_records]}.values(), key=lambda r: r.global_index)
    pose_cache_path = output_dir / "pose_cache.json"
    poses = estimate_or_load_poses(needed_records, args, detector, ref_frames, K, pose_cache_path)
    if exact_training:
        require_valid_training_poses(train_records, poses)

    train_views, _ = build_views(
        train_records, poses, args, focal, fovx, fovy, split="train"
    )
    probe_views, _ = build_views(
        probe_records, poses, args, focal, fovx, fovy, split="probe"
    )
    exact_dataset_audit = None
    if exact_training:
        exact_dataset_audit = validate_exact_dataset_contract(
            manifest, all_names, train_records, train_views, args.updates_per_frame
        )
    if not train_views:
        failed = {r.name: asdict(poses[r.name]) for r in train_records if r.name in poses}
        raise RuntimeError(f"No training frame had a valid PnP pose; diagnostics: {json.dumps(failed, indent=2)}")

    gaussians_change = GaussianModel(sh_degree=3, active_sh_degree=0)
    gaussians_change.load_ply_change(str(source_path / BASE_PLY_REL))
    temporal_model = build_temporal_model(gaussians_change, tuple(args.boundaries), len(all_names))
    set_trainable_state_only(temporal_model)
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
    background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")

    support = compute_state_support(
        temporal_model,
        train_views,
        background,
        pipe,
        args.support_threshold,
    )
    pre_loss_summary = summarize_losses(temporal_model, train_views, background, pipe)
    train_log, gradient_audit, schedule_audit = train_temporal_slots(
        temporal_model, train_views, background, pipe, args, exact_dataset_audit
    )
    post_loss_summary = summarize_losses(temporal_model, train_views, background, pipe)
    loss_summary = {"pre_train": pre_loss_summary, "post_train": post_loss_summary}
    metrics = evaluate_views(temporal_model, [*train_views, *probe_views], background, pipe)
    if probe_views:
        visual_paths, probe_diagnostics = save_probe_grid(
            temporal_model,
            probe_views,
            background,
            pipe,
            output_dir / "probe_grids",
        )
    else:
        visual_paths, probe_diagnostics = [], []
    ckpt_path, summary_path = save_checkpoint(
        temporal_model,
        args,
        output_dir,
        manifest,
        train_records,
        probe_records,
        poses,
        support,
        loss_summary,
        train_log,
        gradient_audit,
        schedule_audit,
        metrics,
        visual_paths,
        probe_diagnostics,
        K,
    )
    print(
        json.dumps(
            {
                "checkpoint": str(ckpt_path),
                "summary": str(summary_path),
                "train_views": len(train_views),
                "probe_views": len(probe_views),
                "schedule_mode": schedule_audit["mode"],
                "total_updates": schedule_audit["actual_total_updates"],
                "updates_per_frame": schedule_audit["min_updates_per_frame"],
                "exact_all_images_guarantee": schedule_audit[
                    "exact_all_images_guarantee"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
