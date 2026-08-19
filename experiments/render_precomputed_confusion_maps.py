"""Build confusion panels/GIFs from already-rendered binary masks."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from experiments.render_temporal_confusion_maps import (
    PALETTE,
    add_label,
    aggregate_rows,
    build_all_records,
    confusion_arrays,
    hstack,
    metrics_from_counts,
    save_contact_sheet,
    save_gif,
    save_panel,
    source_scene_name,
)
from experiments.train_real_temporal_rchange import (
    DEFAULT_BOUNDARIES,
    DEFAULT_SOURCE,
    boundaries_from_manifest,
    read_manifest,
)
from experiments.visualize_temporal_state_switch import load_font, resize_panel


def image_tensor(path: Path, size: tuple[int, int]) -> torch.Tensor:
    with Image.open(path) as image:
        resized = image.convert("RGB").resize(size, Image.Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def mask_tensor(path: Path, size: tuple[int, int]) -> torch.Tensor:
    with Image.open(path) as image:
        resized = image.convert("L").resize(size, Image.Resampling.NEAREST)
        array = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_comparison_panel(
    rgb: torch.Tensor,
    gt: torch.Tensor,
    comparison_pred: torch.Tensor,
    primary_pred: torch.Tensor,
    comparison_confusion: np.ndarray,
    primary_confusion: np.ndarray,
    path: Path,
    title: str,
    comparison_label: str,
    primary_label: str,
    comparison_metrics: dict[str, float],
    primary_metrics: dict[str, float],
    width: int,
) -> None:
    font = load_font(15)
    small = load_font(13)

    def mask_image(mask: torch.Tensor, color: tuple[int, int, int]) -> Image.Image:
        values = mask.squeeze().numpy()
        output = np.zeros((*values.shape, 3), dtype=np.uint8)
        for channel, value in enumerate(color):
            output[..., channel] = (values * value).astype(np.uint8)
        return Image.fromarray(output)

    rgb_image = Image.fromarray(
        (rgb.permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
    )
    panels = [
        add_label(resize_panel(rgb_image, width), "RGB", font),
        add_label(
            resize_panel(mask_image(gt, (255, 255, 255)), width),
            "GT mask",
            font,
        ),
        add_label(
            resize_panel(mask_image(comparison_pred, (255, 220, 0)), width),
            f"{comparison_label} mask",
            font,
        ),
        add_label(
            resize_panel(mask_image(primary_pred, (255, 220, 0)), width),
            f"{primary_label} mask",
            font,
        ),
        add_label(
            resize_panel(Image.fromarray(comparison_confusion), width),
            f"{comparison_label} TP/TN/FP/FN",
            font,
        ),
        add_label(
            resize_panel(Image.fromarray(primary_confusion), width),
            f"{primary_label} TP/TN/FP/FN",
            font,
        ),
    ]
    row = hstack(panels, gap=8)
    header = Image.new("RGB", (row.width, 82), "white")
    draw = ImageDraw.Draw(header)
    draw.text((6, 5), title, fill=(0, 0, 0), font=font)
    draw.text(
        (6, 31),
        f"{comparison_label}: IoU={comparison_metrics['iou']:.3f} "
        f"F1={comparison_metrics['f1']:.3f}  |  "
        f"{primary_label}: IoU={primary_metrics['iou']:.3f} "
        f"F1={primary_metrics['f1']:.3f}",
        fill=(0, 0, 0),
        font=small,
    )
    draw.text(
        (6, 55),
        "TP=green  TN=uncolored  FP=pink  FN=blue",
        fill=(0, 0, 0),
        font=small,
    )
    output = Image.new("RGB", (row.width, header.height + row.height), "white")
    output.paste(header, (0, 0))
    output.paste(row, (0, header.height))
    path.parent.mkdir(parents=True, exist_ok=True)
    output.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build confusion GIFs from precomputed binary masks"
    )
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-path", type=Path, default=Path(DEFAULT_SOURCE))
    parser.add_argument("--label", default="persistent")
    parser.add_argument("--comparison-pred-dir", type=Path)
    parser.add_argument("--comparison-label", default="temporal geometry")
    parser.add_argument(
        "--comparison-output-name",
        default="geometry_vs_persistent",
        help="Filename stem used for optional per-scene comparison GIFs",
    )
    parser.add_argument("--boundaries", nargs="+", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--panel-width", type=int, default=260)
    parser.add_argument("--gif-width", type=int, default=720)
    parser.add_argument("--comparison-gif-width", type=int, default=1200)
    parser.add_argument("--gif-duration-ms", type=int, default=160)
    parser.add_argument("--contact-samples", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest(args.source_path)
    boundaries = (
        tuple(args.boundaries)
        if args.boundaries is not None
        else boundaries_from_manifest(manifest, DEFAULT_BOUNDARIES)
    )
    records = build_all_records(args.source_path, boundaries, args.max_frames)
    rows: list[dict[str, Any]] = []
    panels_by_scene: dict[str, list[Path]] = {}
    comparison_panels_by_scene: dict[str, list[Path]] = {}

    for record in records:
        stem = Path(record.name).stem
        pred_path = args.pred_dir / f"{stem}.png"
        if not pred_path.exists():
            raise FileNotFoundError(pred_path)
        with Image.open(pred_path) as image:
            size = image.size
        rgb = image_tensor(Path(record.image_path), size)
        gt = mask_tensor(Path(record.mask_path), size)
        pred = (mask_tensor(pred_path, size) >= 0.5).float()
        confusion, counts = confusion_arrays(pred, gt, threshold=0.5)
        metrics = metrics_from_counts(counts)
        scene = source_scene_name(record.name)
        raw_path = (
            args.output_dir
            / scene
            / "raw_confusion"
            / f"{record.global_index:06d}_{stem}_confusion.png"
        )
        panel_path = (
            args.output_dir
            / scene
            / "panels"
            / f"{record.global_index:06d}_{stem}_panel.png"
        )
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(confusion).save(raw_path)
        save_panel(
            rgb,
            gt,
            pred,
            confusion,
            panel_path,
            f"{scene} | global t={record.global_index} | state S{record.segment_id} | {record.name}",
            metrics,
            args.panel_width,
        )
        panels_by_scene.setdefault(scene, []).append(panel_path)

        row = {
            "frame": record.name,
            "source_scene": scene,
            "global_index": record.global_index,
            "state": record.segment_id,
            "pred_binary_png": str(pred_path),
            "raw_confusion_png": str(raw_path),
            "panel_png": str(panel_path),
            **counts,
            **metrics,
        }
        rows.append(row)

        if args.comparison_pred_dir is not None:
            comparison_path = args.comparison_pred_dir / f"{stem}.png"
            if not comparison_path.exists():
                raise FileNotFoundError(comparison_path)
            comparison_pred = (
                mask_tensor(comparison_path, size) >= 0.5
            ).float()
            comparison_confusion, comparison_counts = confusion_arrays(
                comparison_pred, gt, threshold=0.5
            )
            comparison_metrics = metrics_from_counts(comparison_counts)
            comparison_panel_path = (
                args.output_dir
                / scene
                / "comparison_panels"
                / f"{record.global_index:06d}_{stem}_comparison.png"
            )
            save_comparison_panel(
                rgb,
                gt,
                comparison_pred,
                pred,
                comparison_confusion,
                confusion,
                comparison_panel_path,
                f"{scene} | global t={record.global_index} | state S{record.segment_id} | {record.name}",
                args.comparison_label,
                args.label,
                comparison_metrics,
                metrics,
                max(180, args.panel_width - 40),
            )
            comparison_panels_by_scene.setdefault(scene, []).append(
                comparison_panel_path
            )

    scene_gifs: dict[str, str] = {}
    comparison_gifs: dict[str, str] = {}
    for scene, paths in panels_by_scene.items():
        selected_indices = np.linspace(
            0, len(paths) - 1, min(args.contact_samples, len(paths))
        ).round().astype(int)
        save_contact_sheet(
            [paths[index] for index in sorted(set(selected_indices.tolist()))],
            args.output_dir / scene / f"{scene}_contact_sheet.png",
        )
        gif_path = args.output_dir / scene / f"{scene}_confusion.gif"
        save_gif(paths, gif_path, args.gif_width, args.gif_duration_ms)
        scene_gifs[scene] = str(gif_path)

        comparison_paths = comparison_panels_by_scene.get(scene, [])
        if comparison_paths:
            comparison_gif = (
                args.output_dir
                / scene
                / f"{scene}_{args.comparison_output_name}.gif"
            )
            save_gif(
                comparison_paths,
                comparison_gif,
                args.comparison_gif_width,
                args.gif_duration_ms,
            )
            comparison_gifs[scene] = str(comparison_gif)

    scenes = sorted(panels_by_scene)
    scene_summaries = [
        aggregate_rows(
            [row for row in rows if row["source_scene"] == scene], scene
        )
        for scene in scenes
    ]
    overall = aggregate_rows(rows, "overall")
    output_summary = {
        "script": "experiments/render_precomputed_confusion_maps.py",
        "pred_dir": str(args.pred_dir),
        "comparison_pred_dir": (
            None
            if args.comparison_pred_dir is None
            else str(args.comparison_pred_dir)
        ),
        "source_path": str(args.source_path),
        "output_dir": str(args.output_dir),
        "label": args.label,
        "comparison_label": args.comparison_label,
        "boundaries": list(boundaries),
        "overall": overall,
        "per_scene": scene_summaries,
        "palette": PALETTE,
        "scene_gifs": scene_gifs,
        "comparison_gifs": comparison_gifs,
        "runtime_seconds": time.time() - started,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(output_summary, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(args.output_dir / "frame_metrics.csv", rows)
    write_csv(args.output_dir / "scene_summary.csv", [overall, *scene_summaries])
    print(json.dumps(output_summary, indent=2))


if __name__ == "__main__":
    main()
