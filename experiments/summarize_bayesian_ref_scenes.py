"""Summarize independent ESCD ref->scene Bayesian lifespan runs.

Reads per-scene output directories produced by
experiments.run_online_bayesian_lifespan_thaw and writes a compact comparison
CSV/JSON plus a PNG bar chart.  This is post-hoc only; it never reads training
inputs or modifies checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size: int = 14):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def read_frame_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def summarize_scene(run_dir: Path, label: str | None = None) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    frame_path = run_dir / "frame_metrics.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    frames = read_frame_csv(frame_path)
    metrics = summary.get("metrics", {}) if isinstance(summary, Mapping) else {}
    if not isinstance(metrics, Mapping) or metrics.get("evaluated") is not True:
        raise ValueError(
            f"run has no completed post-inference metrics: {summary_path}"
        )
    active_counts = np.asarray([_float(row.get("active_lifespan_count")) for row in frames])
    open_counts = np.asarray([_float(row.get("open_count")) for row in frames])
    keep_counts = np.asarray([_float(row.get("keep_count")) for row in frames])
    pred_fracs = np.asarray([_float(row.get("predicted_positive_fraction")) for row in frames])
    scene = label or Path(summary.get("run_arguments", {}).get("source_path", run_dir.name)).name
    run_config = summary.get("run_config", {})
    return {
        "scene": scene,
        "run_dir": str(run_dir),
        "frames": int(summary.get("frames", len(frames))),
        "algorithm": summary.get("algorithm", ""),
        "optimized_parameters": ",".join(summary.get("optimized_parameters", [])),
        "updates_per_frame": int(run_config.get("updates_per_frame", 0)),
        "bayes_cue_mode": run_config.get("bayes_cue_mode", ""),
        "evidence_count_mode": run_config.get("evidence_count_mode", ""),
        "aggregate_iou": _float(metrics.get("aggregate_iou")),
        "aggregate_f1": _float(metrics.get("aggregate_f1")),
        "precision": _float(metrics.get("precision")),
        "recall": _float(metrics.get("recall")),
        "mean_frame_iou": _float(metrics.get("mean_frame_iou")),
        "mean_frame_f1": _float(metrics.get("mean_frame_f1")),
        "event_count": int(summary.get("event_count", 0)),
        "reopen_count": int(summary.get("reopen_count", 0)),
        "active_to_active_false_split_count": int(summary.get("active_to_active_false_split_count", 0)),
        "mean_active_lifespan_count": float(active_counts.mean()) if active_counts.size else 0.0,
        "final_active_lifespan_count": float(active_counts[-1]) if active_counts.size else 0.0,
        "total_open_count": float(open_counts.sum()) if open_counts.size else 0.0,
        "total_keep_count": float(keep_counts.sum()) if keep_counts.size else 0.0,
        "mean_predicted_positive_fraction": float(pred_fracs.mean()) if pred_fracs.size else 0.0,
        "runtime_seconds": _float(summary.get("runtime_seconds")),
        "cuda_peak_memory_bytes": int(summary.get("cuda_peak_memory_bytes", 0)),
        "visual_summary": summary.get("visualization"),
    }


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["scene"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_comparison_chart(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    width, height = 1100, 730
    margin_left, margin_top, margin_bottom, margin_right = 90, 104, 130, 40
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(18)
    font = _font(13)
    small = _font(11)
    draw.text((margin_left, 25), "Independent ESCD ref->SC Bayesian lifespan comparison", fill="black", font=title_font)
    first = rows[0] if rows else {}
    method = first.get("optimized_parameters") or "detector-only"
    draw.text(
        (margin_left, 52),
        f"{method} | {first.get('updates_per_frame', 0)} updates/frame | "
        f"{first.get('bayes_cue_mode', '')} cue | "
        f"{first.get('evidence_count_mode', '')} evidence | "
        f"{first.get('algorithm', '')}",
        fill=(60, 60, 60),
        font=small,
    )
    for tick in range(0, 11):
        value = tick / 10.0
        y = margin_top + plot_h - int(value * plot_h)
        draw.line((margin_left, y, width - margin_right, y), fill=(230, 230, 230))
        draw.text((35, y - 7), f"{value:.1f}", fill="black", font=small)
    metrics = [
        ("aggregate_iou", (31, 119, 180), "IoU"),
        ("aggregate_f1", (44, 160, 44), "F1"),
        ("precision", (214, 39, 40), "Precision"),
        ("recall", (148, 103, 189), "Recall"),
    ]
    n = max(1, len(rows))
    group_w = plot_w / n
    bar_w = max(14, int(group_w / (len(metrics) + 2)))
    for i, row in enumerate(rows):
        group_x = margin_left + i * group_w
        for j, (key, color, _label) in enumerate(metrics):
            value = max(0.0, min(1.0, float(row[key])))
            x0 = int(group_x + (j + 1) * bar_w)
            x1 = x0 + bar_w - 3
            y0 = margin_top + plot_h - int(value * plot_h)
            draw.rectangle((x0, y0, x1, margin_top + plot_h), fill=color)
            draw.text((x0, y0 - 16), f"{value:.2f}", fill=color, font=small)
        label = str(row["scene"])
        draw.text((int(group_x + bar_w), margin_top + plot_h + 12), label, fill="black", font=font)
    legend_x, legend_y = margin_left, height - 58
    for _key, color, label in metrics:
        draw.rectangle((legend_x, legend_y, legend_x + 24, legend_y + 14), fill=color)
        draw.text((legend_x + 32, legend_y - 1), label, fill="black", font=font)
        legend_x += 185
    draw.text(
        (margin_left, height - 30),
        "GT is used only for post-inference evaluation; lifecycle decisions remain causal.",
        fill=(50, 50, 50),
        font=font,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    labels = args.labels or [None] * len(args.run_dirs)
    if len(labels) != len(args.run_dirs):
        raise ValueError("--labels count must match run_dirs count")
    rows = [summarize_scene(path, label) for path, label in zip(args.run_dirs, labels)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output_dir / "comparison_metrics.csv")
    write_comparison_chart(rows, args.output_dir / "comparison_metrics.png")
    payload = {
        "schema_version": 1,
        "contract": "independent_escd_ref_scene_comparison",
        "scenes": rows,
        "files": {
            "comparison_metrics_csv": str(args.output_dir / "comparison_metrics.csv"),
            "comparison_metrics_png": str(args.output_dir / "comparison_metrics.png"),
        },
    }
    (args.output_dir / "comparison_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["files"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
