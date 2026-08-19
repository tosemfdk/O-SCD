"""Visualize one selected estimated pose across a manual temporal boundary.

This script reloads the existing temporal R_change pilot, chooses the cached PnP
pose for one requested real frame, and renders S0/S1 masks from that same fixed
estimated camera. It also shows the exact same-image oracle GT mask for the S1
observation. No S0 GT is fabricated for this pose.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from experiments.visualize_temporal_state_switch import (
    DEFAULT_RUN_DIR,
    add_label,
    frame_by_timestamp,
    frame_state,
    heatmap_fixed,
    hstack,
    load_font,
    make_fixed_camera,
    metrics,
    overlay_change,
    read_json,
    render_array,
    render_timestamps,
    resize_panel,
    tensor_to_pil,
    total_frames_from_summary,
    vstack,
)
from temporal import load_temporal_model

DEFAULT_CAMERA_TIMESTAMP = 181
DEFAULT_LEFT_TIMESTAMP = 94
DEFAULT_RIGHT_TIMESTAMP = 95
DEFAULT_PREFIX = "scene2_frame087_scene1_to_2"


def pose_rmse(pose: dict[str, Any]) -> float:
    """Read cached reprojection RMSE, treating missing values as worst."""
    value = pose.get("reprojection_rmse", pose.get("rmse"))
    return float("inf") if value is None else float(value)


def pose_diagnostics(
    cache_key: str,
    pose: dict[str, Any],
    min_inliers: int,
) -> dict[str, Any]:
    """Keep the pose-selection evidence compact and JSON-safe."""
    return {
        "cache_key": cache_key,
        "ok": bool(pose.get("ok", False)),
        "frame_name": pose.get("frame_name"),
        "reference_name": pose.get("reference_name"),
        "matches": None if pose.get("matches") is None else int(pose.get("matches")),
        "inliers": None if pose.get("inliers") is None else int(pose.get("inliers")),
        "reprojection_rmse": None if pose.get("reprojection_rmse", pose.get("rmse")) is None else pose_rmse(pose),
        "eligible": bool(pose.get("ok", False)) and int(pose.get("inliers") or 0) >= min_inliers and pose.get("Rt") is not None,
        "reason": pose.get("reason"),
    }


def select_cached_pose(pose_cache: dict[str, Any], frame_name: str, min_inliers: int = 8) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Choose highest-inlier PnP pose with deterministic tie-breaks."""
    candidates: list[tuple[str, dict[str, Any]]] = [
        (key, value) for key, value in pose_cache.items() if value.get("frame_name") == frame_name
    ]
    if not candidates:
        raise KeyError(f"No cached pose candidates found for {frame_name}")
    diagnostics = [pose_diagnostics(key, pose, min_inliers) for key, pose in candidates]
    eligible = [
        (key, pose)
        for key, pose in candidates
        if bool(pose.get("ok", False)) and int(pose.get("inliers") or 0) >= min_inliers and pose.get("Rt") is not None
    ]
    if not eligible:
        raise ValueError(f"No valid pose candidate for {frame_name} has at least {min_inliers} inliers")
    chosen_key, chosen = max(
        eligible,
        key=lambda item: (
            int(item[1].get("inliers") or 0),
            -pose_rmse(item[1]),
            item[0],
        ),
    )
    chosen = dict(chosen)
    chosen["cache_key"] = chosen_key
    return chosen, diagnostics


def load_gt_mask(path: Path, size: tuple[int, int]) -> torch.Tensor:
    """Load oracle mask as [1,H,W] float in [0,1] at render resolution."""
    image = Image.open(path).convert("L")
    if image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    arr = np.asarray(image, dtype=np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return torch.from_numpy(arr).unsqueeze(0).clamp(0, 1)


def mask_panel(mask: torch.Tensor, color: tuple[int, int, int] = (255, 220, 30)) -> Image.Image:
    """Convert a scalar mask to a simple colored panel."""
    x = render_array(mask)
    rgb = np.zeros((*x.shape, 3), dtype=np.float32)
    for c, value in enumerate(color):
        rgb[..., c] = x * (value / 255.0)
    return Image.fromarray((np.clip(rgb, 0, 1) * 255.0).astype(np.uint8))


def overlay_mask(rgb: torch.Tensor, mask: torch.Tensor, color: tuple[float, float, float], alpha_scale: float = 0.72) -> Image.Image:
    """Overlay any scalar mask on RGB with a fixed color."""
    base = rgb.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    m = render_array(mask)[..., None]
    color_arr = np.zeros_like(base)
    color_arr[..., 0] = color[0]
    color_arr[..., 1] = color[1]
    color_arr[..., 2] = color[2]
    out = base * (1.0 - alpha_scale * m) + color_arr * (alpha_scale * m)
    return Image.fromarray((np.clip(out, 0, 1) * 255.0).astype(np.uint8))


def binary_metrics(render: torch.Tensor, gt: torch.Tensor, threshold: float = 0.5) -> dict[str, Any]:
    """Report thresholded agreement against oracle GT."""
    pred = render.detach().float().cpu() >= threshold
    target = gt.detach().float().cpu() >= threshold
    tp = int((pred & target).sum().item())
    fp = int((pred & ~target).sum().item())
    fn = int((~pred & target).sum().item())
    tn = int((~pred & ~target).sum().item())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    values = render.detach().float().cpu()
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "iou": float(iou),
        "f1": float(f1),
        "pred_positive_pixels": int(pred.sum().item()),
        "gt_positive_pixels": int(target.sum().item()),
        "mean_render_inside_gt": float(values[target].mean().item()) if target.any() else 0.0,
        "mean_render_outside_gt": float(values[~target].mean().item()) if (~target).any() else 0.0,
    }


def text_box(lines: list[str], width: int, font: ImageFont.ImageFont, pad: int = 10) -> Image.Image:
    """Make a readable white note panel."""
    line_h = max(22, font.size + 5 if hasattr(font, "size") else 22)
    out = Image.new("RGB", (width, pad * 2 + line_h * len(lines)), "white")
    draw = ImageDraw.Draw(out)
    y = pad
    for line in lines:
        draw.text((pad, y), line, fill=(0, 0, 0), font=font)
        y += line_h
    draw.rectangle((0, 0, width - 1, out.height - 1), outline=(180, 180, 180), width=1)
    return out


def build_static_png(
    rgb: torch.Tensor,
    gt: torch.Tensor,
    renders: dict[int, torch.Tensor],
    left_timestamp: int,
    right_timestamp: int,
    camera_timestamp: int,
    chosen_pose: dict[str, Any],
    output_path: Path,
) -> None:
    """Create the focused selected-pose transition figure."""
    title_font = load_font(28)
    label_font = load_font(17)
    note_font = load_font(16)
    panel_w = 230

    diff = (renders[right_timestamp] - renders[left_timestamp]).abs().clamp(0, 1)
    rgb_panel = resize_panel(tensor_to_pil(rgb), panel_w)
    gt_panel = resize_panel(mask_panel(gt, color=(255, 255, 255)), panel_w)

    row1 = hstack(
        [
            add_label(rgb_panel, f"Input RGB t{camera_timestamp}", label_font),
            add_label(gt_panel, "Exact same-image S1 GT", label_font),
            add_label(resize_panel(overlay_mask(rgb, gt, (0.0, 1.0, 0.0)), panel_w), "RGB + S1 GT", label_font),
            add_label(resize_panel(overlay_change(rgb, renders[right_timestamp]), panel_w), f"RGB + render S1 t{right_timestamp}", label_font),
        ],
        gap=12,
    )
    row2 = hstack(
        [
            add_label(resize_panel(mask_panel(renders[left_timestamp], color=(255, 220, 30)), panel_w), f"Rendered S0 mask t{left_timestamp}", label_font),
            add_label(resize_panel(mask_panel(renders[right_timestamp], color=(255, 220, 30)), panel_w), f"Rendered S1 mask t{right_timestamp}", label_font),
            add_label(resize_panel(heatmap_fixed(diff), panel_w), "|S1 render - S0 render|", label_font),
            add_label(resize_panel(overlay_mask(rgb, diff, (1.0, 0.25, 0.0)), panel_w), "RGB + render change", label_font),
        ],
        gap=12,
    )
    width = max(row1.width, row2.width)
    title = Image.new("RGB", (width, 54), "white")
    ImageDraw.Draw(title).text((8, 10), "Selected estimated pose: scene_change1 -> scene_change2", fill=(0, 0, 0), font=title_font)
    pose_note = text_box(
        [
            f"Fixed estimated PnP pose from t{camera_timestamp}: {chosen_pose.get('reference_name')} | inliers={chosen_pose.get('inliers')} | RMSE={pose_rmse(chosen_pose):.3f}px",
            "Only the query timestamp changes the TemporalChangeModel state slot.",
            "No synchronized S0 observation exists at this exact pose, so S0 GT is not fabricated.",
        ],
        width,
        note_font,
    )
    vstack([title, pose_note, row1, row2], gap=14).save(output_path)


def build_transition_gif(
    rgb: torch.Tensor,
    gt: torch.Tensor,
    renders: dict[int, torch.Tensor],
    left_timestamp: int,
    right_timestamp: int,
    camera_timestamp: int,
    output_path: Path,
) -> None:
    """Create a compact two-frame GIF for the S0/S1 mask switch."""
    font = load_font(18)
    panel_w = 280
    frames: list[Image.Image] = []
    for timestamp, label in [(left_timestamp, "S0 before boundary"), (right_timestamp, "S1 after boundary")]:
        row = hstack(
            [
                add_label(resize_panel(tensor_to_pil(rgb), panel_w), f"Input RGB t{camera_timestamp}", font),
                add_label(resize_panel(overlay_mask(rgb, gt, (0.0, 1.0, 0.0)), panel_w), "same-image S1 GT", font),
                add_label(resize_panel(overlay_change(rgb, renders[timestamp]), panel_w), f"render {label} t{timestamp}", font),
            ],
            gap=10,
        )
        note = text_box(["Same estimated camera; only temporal query changes. No S0 GT is fabricated."], row.width, font)
        frames.append(vstack([row, note], gap=8))
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=950, loop=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize S0->S1 temporal render change at one selected estimated pose")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--camera-timestamp", type=int, default=DEFAULT_CAMERA_TIMESTAMP)
    parser.add_argument("--left-timestamp", type=int, default=DEFAULT_LEFT_TIMESTAMP)
    parser.add_argument("--right-timestamp", type=int, default=DEFAULT_RIGHT_TIMESTAMP)
    parser.add_argument("--output-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--min-inliers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the existing Gaussian renderer and Camera implementation")

    run_dir = args.run_dir
    summary = read_json(run_dir / "summary.json")
    pose_cache = read_json(run_dir / "pose_cache.json")
    checkpoint_path = run_dir / "temporal_rchange_checkpoint.pt"
    boundaries = [int(x) for x in summary["boundaries"]]
    total_frames = total_frames_from_summary(summary)

    camera_record = frame_by_timestamp(summary, args.camera_timestamp)
    chosen_pose, candidate_diagnostics = select_cached_pose(pose_cache, camera_record["name"], args.min_inliers)
    view = make_fixed_camera(camera_record, chosen_pose, summary)
    model = load_temporal_model(checkpoint_path)

    timestamps = sorted({int(args.left_timestamp), int(args.right_timestamp), int(args.camera_timestamp)})
    renders = render_timestamps(view, model, timestamps)
    invariant = metrics(renders[int(args.right_timestamp)], renders[int(args.camera_timestamp)])
    if invariant["max_abs"] > 1e-8:
        raise AssertionError(f"Expected same-state render equality, got max_abs={invariant['max_abs']}")

    source_path = Path(summary["source_path"])
    gt_path = source_path / "gt_mask" / camera_record["name"]
    if not gt_path.exists():
        raise FileNotFoundError(gt_path)
    fixed_rgb = view.original_image[:3].detach()
    gt = load_gt_mask(gt_path, (int(view.image_width), int(view.image_height)))

    static_path = run_dir / f"{args.output_prefix}.png"
    gif_path = run_dir / f"{args.output_prefix}.gif"
    manifest_path = run_dir / f"{args.output_prefix}_manifest.json"
    build_static_png(
        fixed_rgb,
        gt,
        renders,
        int(args.left_timestamp),
        int(args.right_timestamp),
        int(args.camera_timestamp),
        chosen_pose,
        static_path,
    )
    build_transition_gif(fixed_rgb, gt, renders, int(args.left_timestamp), int(args.right_timestamp), int(args.camera_timestamp), gif_path)

    render_diff = metrics(renders[int(args.left_timestamp)], renders[int(args.right_timestamp)])
    gt_metrics = {
        f"render_t{args.left_timestamp}_vs_same_image_s1_gt": binary_metrics(renders[int(args.left_timestamp)], gt, threshold=0.5),
        f"render_t{args.right_timestamp}_vs_same_image_s1_gt": binary_metrics(renders[int(args.right_timestamp)], gt, threshold=0.5),
        f"render_t{args.camera_timestamp}_vs_same_image_s1_gt": binary_metrics(renders[int(args.camera_timestamp)], gt, threshold=0.5),
    }
    manifest = {
        "script": "experiments/visualize_selected_pose_transition.py",
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "selected_frame": {
            "global_timestamp": int(camera_record["global_index"]),
            "frame": camera_record["name"],
            "image_path": camera_record["image_path"],
            "gt_mask_path": str(gt_path),
            "state": frame_state(int(args.camera_timestamp), boundaries),
        },
        "pose_selection": {
            "criterion": "eligible ok pose with highest inlier count; tie-break by lowest reprojection RMSE, then cache key",
            "min_inliers": int(args.min_inliers),
            "chosen_cache_key": chosen_pose.get("cache_key"),
            "chosen_reference_name": chosen_pose.get("reference_name"),
            "chosen_inliers": int(chosen_pose.get("inliers") or 0),
            "chosen_reprojection_rmse": pose_rmse(chosen_pose),
            "candidate_diagnostics": candidate_diagnostics,
            "note": "This is an estimated PnP pose, not ground truth camera pose.",
        },
        "selected_camera": {
            "R": view.R.tolist() if hasattr(view.R, "tolist") else view.R,
            "T": np.asarray(view.T).tolist(),
            "FoVx": float(view.FoVx),
            "FoVy": float(view.FoVy),
        },
        "dimensions": {"width": int(view.image_width), "height": int(view.image_height)},
        "total_frames": int(total_frames),
        "boundaries": boundaries,
        "rendered_timestamps": timestamps,
        "timestamp_to_state": {str(t): frame_state(t, boundaries) for t in timestamps},
        "same_state_invariance_assertion": {
            "right_timestamp": int(args.right_timestamp),
            "camera_timestamp": int(args.camera_timestamp),
            "max_abs": invariant["max_abs"],
            "passed": invariant["max_abs"] <= 1e-8,
        },
        "render_difference_metrics": {f"{args.left_timestamp}->{args.right_timestamp}": render_diff},
        "gt_vs_render_metrics_at_threshold_0_5": gt_metrics,
        "scientific_note": "Only S1 GT is shown because the selected RGB is a scene_change2 observation. No synchronized S0 observation exists at this exact estimated pose.",
        "outputs": {"static_png": str(static_path), "animated_gif": str(gif_path)},
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps({"static_png": str(static_path), "animated_gif": str(gif_path), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
