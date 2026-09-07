#!/usr/bin/env python3
"""SC3 feasibility audit for Depth Anything 3 NEW-object point seeding.

This experiment consumes the causal PCA/sign trace from the completed E4e run.
At each requested frame it runs DA3 on a past-only camera window, robustly
aligns the current predicted depth to immutable reference-render depth using
reference-consistent pixels, and back-projects only the confirmed NEW mask.
Object masks are loaded after seed construction and are evaluation-only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from plyfile import PlyData, PlyElement
from transformers import Sam2Model


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.run_online_xfeat_new_seed import (  # noqa: E402
    DATA,
    RUN,
    annotation_masks_at_size,
    build_fixed_cue_views,
    camera_json_to_w2c,
    extract_delta,
    fixed_camera_matrices,
    load_fixed_camera_index,
    load_object_annotations,
    scene_records,
)
from gaussian_renderer import render_change  # noqa: E402
from scene import GaussianModel  # noqa: E402
from temporal.depth_prior_new_seeding import (  # noqa: E402
    DepthPriorSeedBatch,
    DepthPriorSeedConfig,
    build_depth_prior_new_seeds,
    fit_reference_depth_scale,
    reference_scale_anchor_mask,
)


DEFAULT_TRACE = REPO / "outputs/e4e_learned_dc_object_matched_metrics_20260902_164614/scene_change3"
DEFAULT_OUTPUT = REPO / "outputs/ref_sc3_da3_depth_prior_new_seed_feasibility_20260902"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames", type=int, nargs="+", default=(48, 52, 64, 80, 84))
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--model", default="depth-anything/DA3-SMALL")
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--sampling-stride", type=int, default=4)
    parser.add_argument("--max-seeds-per-frame", type=int, default=2048)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--confidence-quantile", type=float, default=0.40)
    parser.add_argument("--sam-model", default="facebook/sam2.1-hiera-tiny")
    args = parser.parse_args()
    if args.window < 2:
        parser.error("--window must be at least 2")
    if any(frame <= 0 or frame > 105 for frame in args.frames):
        parser.error("--frames must be within SC3 frames 1..105")
    if args.voxel_size <= 0.0:
        parser.error("--voxel-size must be positive")
    args.frames = tuple(sorted(set(args.frames)))
    return args


def _resize_tensor(value: torch.Tensor, height: int, width: int, mode: str) -> torch.Tensor:
    source = value.detach().float().squeeze()
    kwargs = {"mode": mode}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    return F.interpolate(source[None, None], (height, width), **kwargs)[0, 0].cpu()


@torch.inference_mode()
def render_reference_depth(
    view: Any,
    base: GaussianModel,
    w2c: torch.Tensor,
    pipe: SimpleNamespace,
    background: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render alpha-weighted immutable-reference camera-z and alpha."""

    xyz = base.get_xyz.detach()
    rotation = w2c[:3, :3].to(device=xyz.device, dtype=xyz.dtype)
    translation = w2c[:3, 3].to(device=xyz.device, dtype=xyz.dtype)
    camera_z = (xyz @ rotation.T + translation)[:, 2].clamp_min(0.0)
    depth_color = camera_z[:, None].repeat(1, 3)
    white = torch.ones_like(depth_color)
    numerator = render_change(
        view,
        base,
        pipe,
        background,
        override_color=depth_color,
        clamp_output=False,
    )["render"].mean(dim=0)
    alpha = render_change(
        view,
        base,
        pipe,
        background,
        override_color=white,
        clamp_output=False,
    )["render"].mean(dim=0)
    depth = numerator / alpha.clamp_min(1.0e-6)
    depth = torch.where(alpha > 1.0e-4, depth, torch.zeros_like(depth))
    return depth.detach().cpu(), alpha.detach().cpu()


def _read_resized_rgb(path: Path, height: int, width: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def load_trace(trace_dir: Path) -> tuple[dict[str, dict[str, str]], dict[str, np.ndarray]]:
    with (trace_dir / "frame_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = {row["frame_name"]: row for row in csv.DictReader(handle)}
    arrays_npz = np.load(trace_dir / "causal_pca_posterior_arrays.npz")
    arrays = {key: arrays_npz[key] for key in arrays_npz.files}
    return rows, arrays


def current_signed_new_mask(
    *,
    delta: torch.Tensor,
    cue_map: torch.Tensor,
    trace_row: dict[str, str],
    pc1: np.ndarray,
) -> torch.Tensor:
    axis = torch.from_numpy(pc1).to(device=delta.device, dtype=delta.dtype)
    score = (delta @ axis).reshape(64, 64)
    cue64 = F.interpolate(cue_map[None], (64, 64), mode="area")[0, 0]
    active = cue64 >= 0.5
    positive = active & (score > float(trace_row["epsilon_positive"]))
    negative = active & (score < float(trace_row["epsilon_negative"]))
    sign = trace_row["seed_new_sign"]
    if sign == "+":
        return positive
    if sign == "-":
        return negative
    return torch.zeros_like(positive)


def _mask_at(path: Path, height: int, width: int) -> np.ndarray:
    raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(path)
    return cv2.resize(raw, (width, height), interpolation=cv2.INTER_NEAREST) > 0


def _seed_pixel_mask(batch: DepthPriorSeedBatch, height: int, width: int, radius: int) -> np.ndarray:
    result = np.zeros((height, width), dtype=np.uint8)
    for x, y in batch.pixels_xy.tolist():
        cv2.circle(result, (int(x), int(y)), int(radius), 1, thickness=-1)
    return result.astype(bool)


def _depth_visual(depth: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    mask = np.isfinite(depth) & (depth > 0)
    if valid is not None:
        mask &= valid
    result = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if not mask.any():
        return result
    low, high = np.quantile(depth[mask], (0.02, 0.98))
    normalized = np.clip((depth - low) / max(float(high - low), 1.0e-6), 0.0, 1.0)
    color = cv2.applyColorMap(np.uint8(np.round(255.0 * (1.0 - normalized))), cv2.COLORMAP_TURBO)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    result[mask] = color[mask]
    return result


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _panel(array: np.ndarray, title: str, subtitle: str) -> Image.Image:
    height, width = array.shape[:2]
    target_width = 300
    target_height = round(height * target_width / width)
    body = Image.fromarray(array).resize((target_width, target_height), Image.Resampling.BILINEAR)
    panel = Image.new("RGB", (target_width, target_height + 58), "white")
    panel.paste(body, (0, 58))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 5), title, font=_font(15), fill="black")
    draw.text((8, 31), subtitle[:55], font=_font(10), fill=(65, 65, 65))
    return panel


def save_frame_visual(
    path: Path,
    rgb: np.ndarray,
    new_mask: np.ndarray,
    object004: np.ndarray,
    object010: np.ndarray,
    predicted_depth: np.ndarray,
    aligned_depth: np.ndarray,
    reference_depth: np.ndarray,
    batch: DepthPriorSeedBatch,
    row: dict[str, Any],
) -> None:
    overlay = rgb.astype(np.float32)
    overlay[new_mask] = 0.25 * overlay[new_mask] + 0.75 * np.array([20, 210, 220])
    seed_overlay = rgb.copy()
    for x, y in batch.pixels_xy.tolist():
        inside = object004[int(y), int(x)]
        cv2.circle(seed_overlay, (int(x), int(y)), 2, (20, 230, 60) if inside else (240, 50, 50), -1)
    gt_overlay = rgb.astype(np.float32)
    gt_overlay[object004] = 0.2 * gt_overlay[object004] + 0.8 * np.array([40, 220, 80])
    gt_overlay[object010] = 0.2 * gt_overlay[object010] + 0.8 * np.array([245, 190, 35])
    gap = reference_depth - aligned_depth
    positive_gap = np.where(gap > 0, gap, 0.0)
    items = [
        _panel(np.uint8(np.clip(overlay, 0, 255)), "Causal NEW posterior", f"pixels={int(new_mask.sum())}"),
        _panel(_depth_visual(predicted_depth), "DA3 raw camera-z", "pose-conditioned, past-only window"),
        _panel(_depth_visual(reference_depth), "Immutable reference depth", f"alpha anchors={row['scale_samples']}"),
        _panel(_depth_visual(positive_gap, new_mask), "Front-depth gap in NEW", f"median object004={row['object004_front_gap_median']:.3f}"),
        _panel(seed_overlay, "Depth-prior seeds", f"green=object004; red=outside; n={batch.count}"),
        _panel(np.uint8(np.clip(gt_overlay, 0, 255)), "Evaluation-only objects", "green=NEW object004; yellow=REMOVE object010"),
    ]
    board = Image.new("RGB", (900, items[0].height * 2 + 72), "white")
    title = f"SC3 frame {row['frame_local']:03d}: DA3 depth-prior NEW seeding"
    ImageDraw.Draw(board).text((18, 20), title, font=_font(24), fill="black")
    for index, item in enumerate(items):
        board.paste(item, ((index % 3) * 300, 72 + (index // 3) * item.height))
    board.save(path)


def _write_ply(path: Path, xyz: np.ndarray, frame_ids: np.ndarray) -> None:
    vertices = np.empty(
        len(xyz),
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("frame", "i4")],
    )
    if len(xyz):
        vertices["x"], vertices["y"], vertices["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        vertices["red"], vertices["green"], vertices["blue"] = 30, 220, 90
        vertices["frame"] = frame_ids
    PlyData([PlyElement.describe(vertices, "vertex")]).write(str(path))


def _causal_voxel_merge(
    xyz: np.ndarray, frame_ids: np.ndarray, voxel_size: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep the earliest point entering each world-space voxel."""

    kept: list[int] = []
    occupied: set[tuple[int, int, int]] = set()
    order = np.argsort(frame_ids, kind="stable")
    for index in order.tolist():
        key = tuple(np.floor(xyz[index] / float(voxel_size)).astype(np.int64).tolist())
        if key in occupied:
            continue
        occupied.add(key)
        kept.append(index)
    selected = np.asarray(kept, dtype=np.int64)
    return xyz[selected], frame_ids[selected], selected


def _nearest_distance_summary(first: np.ndarray, second: np.ndarray) -> dict[str, float | int]:
    if not len(first) or not len(second):
        return {"first_points": len(first), "second_points": len(second)}
    distances = torch.cdist(torch.from_numpy(first), torch.from_numpy(second))
    directed = torch.cat((distances.min(dim=1).values, distances.min(dim=0).values))
    return {
        "first_points": len(first),
        "second_points": len(second),
        "symmetric_median": float(directed.median().item()),
        "symmetric_p90": float(torch.quantile(directed, 0.90).item()),
        "symmetric_mean": float(directed.mean().item()),
    }


def _save_object004_geometry_plot(
    path: Path, xyz: np.ndarray, frame_ids: np.ndarray
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    pairs = ((0, 1, "world x", "world y"), (0, 2, "world x", "world z"), (1, 2, "world y", "world z"))
    for axis, (first, second, x_label, y_label) in zip(axes, pairs):
        scatter = axis.scatter(
            xyz[:, first],
            xyz[:, second],
            c=frame_ids,
            cmap="viridis",
            s=5,
            alpha=0.75,
            linewidths=0,
        )
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=0.2)
    figure.colorbar(scatter, ax=axes, label="causal birth frame", shrink=0.75)
    figure.suptitle("Evaluation-only object004 subset of DA3 depth-prior seeds")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _project_world(xyz: np.ndarray, w2c: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    camera = xyz @ w2c[:3, :3].T + w2c[:3, 3]
    depth = camera[:, 2]
    homogeneous = camera @ K.T
    pixels = homogeneous[:, :2] / np.maximum(homogeneous[:, 2:3], 1.0e-8)
    return pixels, depth


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "da3_cache"
    cache_dir.mkdir(exist_ok=True)
    summary = json.loads((RUN / "summary.json").read_text())
    cameras = load_fixed_camera_index(Path(summary["fixed_cameras_json"]))
    records = scene_records(summary, 3, None)
    views, _, _ = build_fixed_cue_views(
        records,
        cameras,
        Path(summary["run_arguments"]["cue_cache_root"]),
        float(summary["resolution"]),
    )
    record_by_local = {int(Path(record.name).stem[-6:]): (record, view) for record, view in zip(records, views)}
    trace_rows, trace_arrays = load_trace(args.trace_dir)
    trace_index = {str(name): i for i, name in enumerate(trace_arrays["frame_names"].astype(str))}
    checkpoint = torch.load(RUN / "temporal_rchange_checkpoint.pt", map_location="cpu", weights_only=False)
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
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    sam = Sam2Model.from_pretrained(args.sam_model, local_files_only=True).half().cuda().eval()
    from depth_anything_3.api import DepthAnything3

    da3 = DepthAnything3.from_pretrained(args.model).to("cuda").eval()
    objects = load_object_annotations(3)
    object004 = next(obj for obj in objects if obj["id"] == "inference_base__object_004")
    object010 = next(obj for obj in objects if obj["id"] == "reference_render_base__object_010")
    config = DepthPriorSeedConfig(
        confidence_quantile=float(args.confidence_quantile),
        sampling_stride=int(args.sampling_stride),
        max_seeds=int(args.max_seeds_per_frame),
    )
    report_rows: list[dict[str, Any]] = []
    all_xyz: list[np.ndarray] = []
    all_frame_ids: list[np.ndarray] = []
    all_object004_labels: list[np.ndarray] = []
    object004_xyz_by_frame: dict[int, np.ndarray] = {}
    per_frame_geometry: dict[int, dict[str, np.ndarray]] = {}
    started = time.time()

    for local_frame in args.frames:
        record, view = record_by_local[int(local_frame)]
        trace_row = trace_rows[record.name]
        axis_row = trace_index[record.name]
        window_frames = [frame for frame in range(max(1, local_frame - args.window + 1), local_frame + 1)]
        images: list[np.ndarray] = []
        extrinsics: list[np.ndarray] = []
        intrinsics: list[np.ndarray] = []
        for frame in window_frames:
            window_record, window_view = record_by_local[frame]
            images.append(_read_resized_rgb(REPO / window_record.image_path, int(window_view.image_height), int(window_view.image_width)))
            w2c, K = fixed_camera_matrices(window_record, cameras)
            extrinsics.append(w2c.numpy())
            intrinsics.append(K.numpy())
        cache_path = cache_dir / f"frame_{local_frame:06d}_window_{window_frames[0]:06d}_{window_frames[-1]:06d}.npz"
        if cache_path.exists():
            cache = np.load(cache_path)
            predicted = cache["depth"]
            confidence = cache["confidence"]
            output_K = cache["K"]
        else:
            prediction = da3.inference(
                images,
                extrinsics=np.stack(extrinsics),
                intrinsics=np.stack(intrinsics),
                align_to_input_ext_scale=True,
                process_res=int(args.process_res),
                process_res_method="upper_bound_resize",
            )
            predicted = np.asarray(prediction.depth[-1], dtype=np.float32)
            confidence = np.asarray(prediction.conf[-1], dtype=np.float32)
            output_K = np.asarray(prediction.intrinsics[-1], dtype=np.float32)
            np.savez_compressed(
                cache_path,
                depth=predicted,
                confidence=confidence,
                K=output_K,
                window_frames=np.asarray(window_frames),
            )
        height, width = predicted.shape
        delta, _ = extract_delta(view, base, pipe, background, sam)
        new64 = current_signed_new_mask(
            delta=delta,
            cue_map=view.candidate_map,
            trace_row=trace_row,
            pc1=trace_arrays["pc1_axes"][axis_row],
        )
        new_mask = _resize_tensor(new64.float(), height, width, "nearest") > 0.5
        cue = _resize_tensor(view.candidate_map, height, width, "area")
        new_mask &= cue >= 0.5
        w2c, _ = fixed_camera_matrices(record, cameras)
        reference_depth_full, reference_alpha_full = render_reference_depth(
            view, base, w2c, pipe, background
        )
        # Resize weighted numerator and alpha separately to avoid depth bleeding
        # across empty pixels.
        reference_numerator = reference_depth_full * reference_alpha_full
        alpha = _resize_tensor(reference_alpha_full, height, width, "area")
        numerator = _resize_tensor(reference_numerator, height, width, "area")
        reference_depth = numerator / alpha.clamp_min(1.0e-6)
        reference_depth[alpha <= 1.0e-4] = 0.0
        anchors = reference_scale_anchor_mask(
            reference_depth,
            alpha,
            cue,
            config=config,
        )
        scale_fit = fit_reference_depth_scale(
            torch.from_numpy(predicted), reference_depth, anchors
        )
        batch = build_depth_prior_new_seeds(
            predicted_depth=torch.from_numpy(predicted),
            confidence=torch.from_numpy(confidence),
            reference_depth=reference_depth,
            new_mask=new_mask,
            K=torch.from_numpy(output_K),
            w2c=w2c,
            scale=scale_fit.scale,
            config=config,
        )
        aligned = predicted * scale_fit.scale

        # Evaluation begins here.  These masks do not affect scale, depth, or seeds.
        obj004 = _mask_at(object004["paths"][record.name], height, width)
        obj010 = _mask_at(object010["paths"][record.name], height, width)
        any_new, any_remove, _ = annotation_masks_at_size(
            record.name, objects, height=height, width=width
        )
        pixels = batch.pixels_xy.numpy()
        if len(pixels):
            x, y = pixels[:, 0], pixels[:, 1]
            inside004 = obj004[y, x]
            inside010 = obj010[y, x]
            inside_new = any_new[y, x]
            inside_remove = any_remove[y, x]
        else:
            inside004 = inside010 = inside_new = inside_remove = np.zeros((0,), dtype=bool)
        coverage = _seed_pixel_mask(batch, height, width, max(2, args.sampling_stride // 2))
        object004_valid = obj004 & new_mask.numpy()
        gap004 = reference_depth.numpy()[obj004] - aligned[obj004]
        gap004 = gap004[np.isfinite(gap004)]
        row = {
            "frame_local": int(local_frame),
            "frame_name": record.name,
            "window_first": int(window_frames[0]),
            "window_last": int(window_frames[-1]),
            "future_frames_used": False,
            "new_sign": trace_row["seed_new_sign"],
            "new_posterior_pixels": int(new_mask.sum().item()),
            "scale": scale_fit.scale,
            "scale_samples": scale_fit.samples,
            "scale_inliers": scale_fit.inliers,
            "scale_log_ratio_mad": scale_fit.log_ratio_mad,
            "scale_median_absolute_relative_error": scale_fit.median_absolute_relative_error,
            "seed_candidates": int(batch.candidate_mask.sum().item()),
            "seeds": batch.count,
            "seed_object004_count": int(inside004.sum()),
            "seed_object004_precision": float(inside004.mean()) if len(inside004) else 0.0,
            "seed_any_new_precision": float(inside_new.mean()) if len(inside_new) else 0.0,
            "seed_object010_remove_leakage": float(inside010.mean()) if len(inside010) else 0.0,
            "seed_any_remove_leakage": float(inside_remove.mean()) if len(inside_remove) else 0.0,
            "object004_causal_new_recall": float(object004_valid.sum() / max(obj004.sum(), 1)),
            "object004_seed_coverage": float((coverage & obj004).sum() / max(obj004.sum(), 1)),
            "object004_front_gap_median": float(np.median(gap004)) if len(gap004) else math.nan,
            "confidence_threshold": batch.confidence_threshold,
        }
        report_rows.append(row)
        all_xyz.append(batch.xyz.numpy())
        all_frame_ids.append(np.full((batch.count,), local_frame, dtype=np.int32))
        all_object004_labels.append(inside004)
        object004_xyz_by_frame[local_frame] = batch.xyz.numpy()[inside004]
        per_frame_geometry[local_frame] = {
            "w2c": w2c.numpy(),
            "K": output_K,
            "object004": obj004,
            "object010": obj010,
            "new_mask": new_mask.numpy(),
        }
        rgb_current = cv2.resize(images[-1], (width, height), interpolation=cv2.INTER_AREA)
        save_frame_visual(
            args.output_dir / f"frame_{local_frame:06d}_depth_seed_audit.png",
            rgb_current,
            new_mask.numpy(),
            obj004,
            obj010,
            predicted,
            aligned,
            reference_depth.numpy(),
            batch,
            row,
        )
        print(json.dumps(row, ensure_ascii=False), flush=True)

    xyz_unmerged = np.concatenate(all_xyz, axis=0) if all_xyz else np.empty((0, 3), np.float32)
    frame_ids_unmerged = np.concatenate(all_frame_ids, axis=0) if all_frame_ids else np.empty((0,), np.int32)
    object004_labels_unmerged = (
        np.concatenate(all_object004_labels, axis=0)
        if all_object004_labels
        else np.empty((0,), bool)
    )
    xyz, frame_ids, merge_indices = _causal_voxel_merge(
        xyz_unmerged, frame_ids_unmerged, float(args.voxel_size)
    )
    _write_ply(args.output_dir / "da3_depth_prior_new_seeds.ply", xyz, frame_ids)
    object004_labels = object004_labels_unmerged[merge_indices]
    _write_ply(
        args.output_dir / "da3_depth_prior_new_seeds_object004_evaluation_only.ply",
        xyz[object004_labels],
        frame_ids[object004_labels],
    )
    _save_object004_geometry_plot(
        args.output_dir / "object004_depth_seed_geometry.png",
        xyz[object004_labels],
        frame_ids[object004_labels],
    )
    consistency = []
    for first, second in zip(args.frames, args.frames[1:]):
        consistency.append(
            {
                "first_frame": int(first),
                "second_frame": int(second),
                **_nearest_distance_summary(
                    object004_xyz_by_frame[first], object004_xyz_by_frame[second]
                ),
            }
        )

    # Compare aggregate DA3 density against the 220 fixed XFeat anchors using
    # evaluation-only object004/object010 reprojection masks.
    xfeat_checkpoint = torch.load(
        args.trace_dir / "new_seed_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )["seed_sidecar"]
    xfeat_xyz = xfeat_checkpoint["xyz"].detach().cpu().numpy().astype(np.float32)
    xfeat_start = xfeat_checkpoint["start"].detach().cpu().numpy()
    xfeat_end = xfeat_checkpoint["end"].detach().cpu().numpy()
    comparison: list[dict[str, Any]] = []
    for frame, geometry in per_frame_geometry.items():
        _, current_view = record_by_local[frame]
        current_timestamp = float(current_view.timestamp)
        causal_da3 = xyz[frame_ids <= int(frame)]
        active_xfeat = xfeat_xyz[
            (xfeat_start <= current_timestamp) & (current_timestamp < xfeat_end)
        ]
        for method, points in (
            ("DA3 causal voxel prefix", causal_da3),
            ("XFeat causal anchors", active_xfeat),
        ):
            pixels, depth = _project_world(points, geometry["w2c"], geometry["K"])
            x = np.floor(pixels[:, 0]).astype(int)
            y = np.floor(pixels[:, 1]).astype(int)
            inside = (
                (depth > 0)
                & (x >= 0)
                & (x < geometry["object004"].shape[1])
                & (y >= 0)
                & (y < geometry["object004"].shape[0])
            )
            x_valid, y_valid = x[inside], y[inside]
            object004_hits = int(geometry["object004"][y_valid, x_valid].sum())
            object010_hits = int(geometry["object010"][y_valid, x_valid].sum())
            new_hits = int(geometry["new_mask"][y_valid, x_valid].sum())
            comparison.append(
                {
                    "frame_local": frame,
                    "method": method,
                    "total_points": len(points),
                    "visible_points": int(inside.sum()),
                    "object004_hits": object004_hits,
                    "causal_new_hits": new_hits,
                    "object010_hits": object010_hits,
                    "object004_visible_precision": object004_hits / max(int(inside.sum()), 1),
                    "object010_visible_leakage": object010_hits / max(int(inside.sum()), 1),
                }
            )
    with (args.output_dir / "per_frame_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report_rows[0]))
        writer.writeheader()
        writer.writerows(report_rows)
    with (args.output_dir / "da3_vs_xfeat_reprojection.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    summary_out = {
        "experiment": "SC3 DA3 pose-conditioned depth-prior NEW seed feasibility",
        "model": args.model,
        "model_repository_revision": "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4",
        "frames": list(args.frames),
        "causal_window": args.window,
        "causal_contract": {
            "each_window_ends_at_current_frame": True,
            "future_frames_used": False,
            "new_mask_from_saved_causal_pca_sign_trace": True,
            "reference_depth_from_immutable_reference_gs": True,
            "ground_truth_used_after_seed_construction_only": True,
        },
        "depth_alignment": "positive scale-only median log-ratio with MAD rejection on stable opaque reference pixels",
        "seed_gate": "confirmed NEW AND high DA3 confidence AND aligned current depth in front of reference depth",
        "configuration": vars(args),
        "per_frame": report_rows,
        "da3_vs_xfeat_reprojection": comparison,
        "object004_cross_view_nearest_distance": consistency,
        "aggregate": {
            "total_da3_seeds_before_voxel_merge": int(len(xyz_unmerged)),
            "total_da3_seeds_after_voxel_merge": int(len(xyz)),
            "xfeat_anchor_count": int(len(xfeat_xyz)),
            "mean_seed_object004_precision": float(np.mean([row["seed_object004_precision"] for row in report_rows])),
            "mean_seed_any_new_precision": float(np.mean([row["seed_any_new_precision"] for row in report_rows])),
            "mean_object010_leakage": float(np.mean([row["seed_object010_remove_leakage"] for row in report_rows])),
            "mean_object004_causal_new_recall": float(np.mean([row["object004_causal_new_recall"] for row in report_rows])),
            "mean_object004_seed_coverage": float(np.mean([row["object004_seed_coverage"] for row in report_rows])),
            "object004_voxel_merged_seed_count": int(object004_labels.sum()),
        },
        "runtime_seconds": time.time() - started,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary_out, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary_out["aggregate"], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
