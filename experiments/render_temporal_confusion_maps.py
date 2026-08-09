"""Render temporal R_change masks and compare them with oracle GT masks.

The output is intentionally simple: one raw TP/TN/FP/FN color map and one
labeled RGB/GT/pred/confusion panel per inference frame with a solved pose.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from experiments.train_real_temporal_rchange import (
    DEFAULT_BOUNDARIES,
    DEFAULT_SOURCE,
    FrameRecord,
    PoseResult,
    boundaries_from_manifest,
    build_views,
    estimate_or_load_poses,
    list_images,
    load_rgb_tensor,
    load_reference_frames,
    read_manifest,
    seed_everything,
    segment_id,
    validate_boundaries,
)
from experiments.visualize_temporal_state_switch import (
    DEFAULT_RUN_DIR,
    load_font,
    load_temporal_model,
    resize_panel,
    tensor_to_pil,
)
from gaussian_renderer import render_change_temporal
from poses.feature_detector import Detector


DEFAULT_OUTPUT_DIR = Path("outputs/instance1_scene_change1_2_3_temporal_confusion")
PALETTE = {
    "tp": (0, 200, 0),      # green
    "tn": (0, 0, 0),        # uncolored background
    "fp": (255, 105, 180),  # pink
    "fn": (0, 90, 255),     # blue
}


def source_scene_name(frame_name: str) -> str:
    """Return scene_change1/2/3 from combined-frame file names."""
    stem = Path(frame_name).stem
    if "_frame_" in stem:
        return stem.split("_frame_", 1)[0]
    return "unknown_scene"


def build_all_records(source_path: Path, boundaries: tuple[int, ...], max_frames: int | None) -> list[FrameRecord]:
    """Enumerate combined inference frames with global timestamp and state."""
    image_dir = source_path / "inference_scene" / "images"
    mask_dir = source_path / "gt_mask"
    names = list_images(image_dir)
    validate_boundaries(len(names), boundaries)
    if max_frames is not None:
        names = names[: max(0, int(max_frames))]
    records: list[FrameRecord] = []
    for idx, name in enumerate(names):
        mask_path = mask_dir / f"{Path(name).stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing GT mask for {name}: {mask_path}")
        records.append(
            FrameRecord(
                global_index=idx,
                segment_id=segment_id(idx, boundaries),
                name=name,
                image_path=str(image_dir / name),
                mask_path=str(mask_path),
            )
        )
    return records


def seed_pose_cache(run_cache: Path, output_cache: Path) -> None:
    """Copy the pilot pose cache once, keeping this evaluation cache separate."""
    if output_cache.exists() or not run_cache.exists():
        return
    output_cache.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_cache, output_cache)


def mask_to_pil(mask: torch.Tensor, color: tuple[int, int, int]) -> Image.Image:
    arr = mask.detach().float().cpu().squeeze().clamp(0, 1).numpy()
    rgb = np.zeros((*arr.shape, 3), dtype=np.uint8)
    for channel, value in enumerate(color):
        rgb[..., channel] = (arr * value).astype(np.uint8)
    return Image.fromarray(rgb)


def confusion_arrays(pred: torch.Tensor, gt: torch.Tensor, threshold: float) -> tuple[np.ndarray, dict[str, int]]:
    """Compute boolean confusion regions at one threshold."""
    pred_b = pred.detach().float().cpu().squeeze().numpy() >= float(threshold)
    gt_b = gt.detach().float().cpu().squeeze().numpy() >= 0.5
    tp = pred_b & gt_b
    tn = ~pred_b & ~gt_b
    fp = pred_b & ~gt_b
    fn = ~pred_b & gt_b
    out = np.zeros((*pred_b.shape, 3), dtype=np.uint8)
    out[tp] = PALETTE["tp"]
    out[fp] = PALETTE["fp"]
    out[fn] = PALETTE["fn"]
    counts = {
        "tp": int(tp.sum()),
        "tn": int(tn.sum()),
        "fp": int(fp.sum()),
        "fn": int(fn.sum()),
        "pred_positive": int(pred_b.sum()),
        "gt_positive": int(gt_b.sum()),
        "pixels": int(pred_b.size),
    }
    return out, counts


def metrics_from_counts(counts: dict[str, int]) -> dict[str, float]:
    tp, tn, fp, fn = counts["tp"], counts["tn"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn else 0.0
    return {"precision": precision, "recall": recall, "iou": iou, "f1": f1, "accuracy": acc}


def add_label(image: Image.Image, label: str, font: ImageFont.ImageFont, label_h: int = 30) -> Image.Image:
    out = Image.new("RGB", (image.width, image.height + label_h), "white")
    out.paste(image.convert("RGB"), (0, label_h))
    draw = ImageDraw.Draw(out)
    draw.text((6, 7), label, fill=(0, 0, 0), font=font)
    return out


def hstack(images: list[Image.Image], gap: int = 8) -> Image.Image:
    width = sum(im.width for im in images) + gap * (len(images) - 1)
    height = max(im.height for im in images)
    out = Image.new("RGB", (width, height), "white")
    x = 0
    for image in images:
        out.paste(image.convert("RGB"), (x, 0))
        x += image.width + gap
    return out


def save_panel(rgb: torch.Tensor, gt: torch.Tensor, pred_binary: torch.Tensor, confusion: np.ndarray, path: Path, title: str, metrics: dict[str, float], width: int) -> None:
    font = load_font(15)
    small = load_font(13)
    panels = [
        add_label(resize_panel(tensor_to_pil(rgb), width), "RGB", font),
        add_label(resize_panel(mask_to_pil(gt, (255, 255, 255)), width), "GT mask", font),
        add_label(resize_panel(mask_to_pil(pred_binary, (255, 220, 0)), width), "rendered mask", font),
        add_label(resize_panel(Image.fromarray(confusion), width), "TP/TN/FP/FN", font),
    ]
    row = hstack(panels, gap=10)
    header = Image.new("RGB", (row.width, 62), "white")
    draw = ImageDraw.Draw(header)
    draw.text((6, 5), title, fill=(0, 0, 0), font=font)
    draw.text(
        (6, 34),
        "TP=green  TN=uncolored  FP=pink  FN=blue   "
        f"IoU={metrics['iou']:.3f} F1={metrics['f1']:.3f} P={metrics['precision']:.3f} R={metrics['recall']:.3f}",
        fill=(0, 0, 0),
        font=small,
    )
    out = Image.new("RGB", (row.width, header.height + row.height), "white")
    out.paste(header, (0, 0))
    out.paste(row, (0, header.height))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)


def aggregate_rows(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    counts = {k: int(sum(int(r[k]) for r in rows)) for k in ["tp", "tn", "fp", "fn", "pred_positive", "gt_positive", "pixels"]}
    out: dict[str, Any] = {"scope": label, "frames": len(rows), **counts, **metrics_from_counts(counts)}
    out["mean_frame_iou"] = float(np.mean([r["iou"] for r in rows])) if rows else 0.0
    out["mean_frame_f1"] = float(np.mean([r["f1"] for r in rows])) if rows else 0.0
    return out


def save_contact_sheet(paths: list[Path], output_path: Path, columns: int = 3, thumb_width: int = 360) -> None:
    if not paths:
        return
    thumbs = [resize_panel(Image.open(p).convert("RGB"), thumb_width) for p in paths]
    rows: list[Image.Image] = []
    for start in range(0, len(thumbs), columns):
        rows.append(hstack(thumbs[start : start + columns], gap=8))
    width = max(r.width for r in rows)
    height = sum(r.height for r in rows) + 8 * (len(rows) - 1)
    out = Image.new("RGB", (width, height), "white")
    y = 0
    for row in rows:
        out.paste(row, (0, y))
        y += row.height + 8
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path)


def save_gif(paths: list[Path], output_path: Path, width: int, duration_ms: int) -> None:
    """Save ordered comparison panels as a looping GIF."""
    if not paths:
        return
    frames: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as image:
            resized = resize_panel(image.convert("RGB"), width)
        frames.append(resized.quantize(colors=128, method=Image.Quantize.MEDIANCUT))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
        optimize=False,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render temporal confusion maps for every inference frame")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--source-path", type=Path, default=Path(DEFAULT_SOURCE))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--resolution", type=float, default=None, help="Defaults to the saved run resolution")
    parser.add_argument("--refs", type=int, default=None, help="Defaults to the saved run reference count")
    parser.add_argument("--kpts", type=int, default=None, help="Defaults to the saved run keypoint count")
    parser.add_argument("--ref-association-px", type=float, default=None)
    parser.add_argument("--pnp-matches", type=int, default=None)
    parser.add_argument("--min-pnp-inliers", type=int, default=8)
    parser.add_argument("--boundaries", nargs="+", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None, help="Smoke-test prefix length")
    parser.add_argument("--panel-width", type=int, default=260)
    parser.add_argument("--contact-samples", type=int, default=12)
    parser.add_argument("--gif-width", type=int, default=720)
    parser.add_argument("--gif-duration-ms", type=int, default=160)
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-pose-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the existing Gaussian renderer")
    seed_everything(args.seed)

    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads((args.run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = read_manifest(args.source_path)
    boundaries = tuple(args.boundaries) if args.boundaries is not None else boundaries_from_manifest(manifest, DEFAULT_BOUNDARIES)
    records = build_all_records(args.source_path, boundaries, args.max_frames)
    if not records:
        raise RuntimeError("No frames selected")

    args.resolution = float(args.resolution if args.resolution is not None else summary.get("resolution", 8.0))
    first_image = load_rgb_tensor(Path(records[0].image_path), args.resolution)
    height, width = int(first_image.shape[1]), int(first_image.shape[2])
    saved_poses = summary.get("pose_results")
    if summary.get("pose_policy") and isinstance(saved_poses, dict):
        poses = {name: PoseResult(**value) for name, value in saved_poses.items()}
        k_matrix = np.asarray(summary["camera_intrinsics"], dtype=np.float64)
        fovx = 2.0 * np.arctan(width / (2.0 * float(k_matrix[0, 0])))
        fovy = 2.0 * np.arctan(height / (2.0 * float(k_matrix[1, 1])))
        focal = float((k_matrix[0, 0] + k_matrix[1, 1]) * 0.5)
    else:
        config = summary.get("config", {})
        args.refs = int(args.refs if args.refs is not None else config.get("refs", 16))
        args.kpts = int(args.kpts if args.kpts is not None else config.get("kpts", 512))
        args.ref_association_px = float(args.ref_association_px if args.ref_association_px is not None else config.get("reference_association_px", 3.0))
        args.pnp_matches = int(args.pnp_matches if args.pnp_matches is not None else config.get("pnp_matches", 256))
        detector = Detector(args.kpts, width, height)
        ref_frames, k_matrix = load_reference_frames(args.source_path, args.resolution, args.refs, detector, args.ref_association_px, width, height)
        focal, fovx, fovy = ref_frames[0].focal, ref_frames[0].fovx, ref_frames[0].fovy
        pose_cache_path = args.output_dir / "pose_cache.json"
        seed_pose_cache(args.run_dir / "pose_cache.json", pose_cache_path)
        poses = estimate_or_load_poses(records, args, detector, ref_frames, k_matrix, pose_cache_path)

    views, gt_masks = build_views(records, poses, args, focal, fovx, fovy, split="all_inference")
    model = load_temporal_model(args.run_dir / "temporal_rchange_checkpoint.pt")
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
    background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")

    rows: list[dict[str, Any]] = []
    panel_paths_by_scene: dict[str, list[Path]] = {}
    record_by_stem = {Path(r.name).stem: r for r in records}

    with torch.no_grad():
        for view in views:
            record = record_by_stem[view.image_name]
            scene = source_scene_name(record.name)
            rendered = render_change_temporal(view, model, pipe, background, timestamp=float(record.global_index))["render"].mean(dim=0, keepdim=True).clamp(0, 1)
            gt = gt_masks[record.name].detach().cpu()
            confusion, counts = confusion_arrays(rendered, gt, args.threshold)
            metric_vals = metrics_from_counts(counts)
            stem = Path(record.name).stem
            raw_path = args.output_dir / scene / "raw_confusion" / f"{record.global_index:06d}_{stem}_confusion.png"
            panel_path = args.output_dir / scene / "panels" / f"{record.global_index:06d}_{stem}_panel.png"
            pred_path = args.output_dir / "pred_binary" / f"{stem}.png"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(confusion).save(raw_path)
            pred_u8 = (rendered >= args.threshold).to(torch.uint8).mul(255).squeeze().cpu().numpy()
            Image.fromarray(pred_u8).save(pred_path)
            save_panel(
                view.original_image[:3].detach().cpu(),
                gt,
                (rendered >= args.threshold).float().detach().cpu(),
                confusion,
                panel_path,
                f"{scene} | global t={record.global_index} | state S{record.segment_id} | {record.name}",
                metric_vals,
                args.panel_width,
            )
            panel_paths_by_scene.setdefault(scene, []).append(panel_path)
            pose = poses[record.name]
            rows.append(
                {
                    "frame": record.name,
                    "source_scene": scene,
                    "global_index": record.global_index,
                    "state": record.segment_id,
                    "pose_ok": True,
                    "pose_reference": pose.reference_name,
                    "pose_matches": pose.matches,
                    "pose_inliers": pose.inliers,
                    "pose_reprojection_rmse": pose.reprojection_rmse,
                    "raw_confusion_png": str(raw_path),
                    "panel_png": str(panel_path),
                    "pred_binary_png": str(pred_path),
                    **counts,
                    **metric_vals,
                }
            )

    failed_pose_records = []
    for record in records:
        pose = poses.get(record.name)
        if pose is None or not pose.ok:
            failed_pose_records.append({"frame": record.name, "global_index": record.global_index, "source_scene": source_scene_name(record.name), "pose": None if pose is None else asdict(pose)})

    scene_gifs: dict[str, str] = {}
    for scene, paths in panel_paths_by_scene.items():
        if len(paths) > args.contact_samples:
            idx = np.linspace(0, len(paths) - 1, args.contact_samples).round().astype(int).tolist()
            selected = [paths[i] for i in sorted(set(idx))]
        else:
            selected = paths
        save_contact_sheet(selected, args.output_dir / scene / f"{scene}_contact_sheet.png")
        if not args.no_gif:
            gif_path = args.output_dir / scene / f"{scene}_confusion.gif"
            save_gif(paths, gif_path, args.gif_width, args.gif_duration_ms)
            scene_gifs[scene] = str(gif_path)

    scene_summaries = [aggregate_rows([r for r in rows if r["source_scene"] == scene], scene) for scene in sorted({r["source_scene"] for r in rows})]
    overall = aggregate_rows(rows, "overall") if rows else aggregate_rows([], "overall")
    summary_out = {
        "script": "experiments/render_temporal_confusion_maps.py",
        "run_dir": str(args.run_dir),
        "source_path": str(args.source_path),
        "output_dir": str(args.output_dir),
        "threshold": args.threshold,
        "boundaries": list(boundaries),
        "frames_requested": len(records),
        "pose_successes": len(rows),
        "pose_failures": len(failed_pose_records),
        "overall": overall,
        "per_scene": scene_summaries,
        "failed_pose_records": failed_pose_records,
        "runtime_seconds": time.time() - started,
        "palette": PALETTE,
        "scene_gifs": scene_gifs,
        "gif_duration_ms": args.gif_duration_ms,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary_out, indent=2), encoding="utf-8")
    write_csv(args.output_dir / "frame_metrics.csv", rows)
    write_csv(args.output_dir / "scene_summary.csv", [overall, *scene_summaries])
    print(json.dumps({"output_dir": str(args.output_dir), "pose_successes": len(rows), "pose_failures": len(failed_pose_records), "overall_iou": overall.get("iou"), "overall_f1": overall.get("f1")}, indent=2))


if __name__ == "__main__":
    main()
