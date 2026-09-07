"""Compare whole-pixel-cue power vs L1-only power against 2D GT masks.

The primary operating point is normalized product cue ``q > 0.25``, matching
the historical raw-cue threshold ``2*q > 0.5``.  A threshold sweep is included
to distinguish cue quality from calibration at one hand-picked threshold.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
from PIL import Image
import torch

from experiments.run_online_bayesian_lifespan_thaw import BASE_PLY_REL, build_causal_records
from experiments.train_cue_temporal_rchange import (
    DEFAULT_CUE_CACHE,
    DEFAULT_FIXED_CAMERAS,
    build_fixed_cue_views,
    load_fixed_camera_index,
    validate_cue_cache,
)
from experiments.train_real_temporal_rchange import DEFAULT_SOURCE
from temporal.change_cue_fusion import (
    fuse_power_product,
    normalized_oscd_pixel_cue_from_terms,
    oscd_pixel_terms,
    semantic_from_cached_sum,
)


CONDITIONS = ("whole_pixel_power", "l1_only_power")


def threshold_counts(
    cue: torch.Tensor,
    ground_truth: torch.Tensor,
    thresholds: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return foreground confusion counts for every threshold."""

    if cue.ndim == 3 and cue.shape[0] == 1:
        cue = cue[0]
    if cue.ndim != 2 or ground_truth.shape != cue.shape:
        raise ValueError("cue and ground_truth must share shape [H,W]")
    if thresholds.ndim != 1 or thresholds.numel() < 1:
        raise ValueError("thresholds must be a nonempty vector")
    target = ground_truth.to(device=cue.device, dtype=torch.bool).flatten()
    prediction = cue.flatten().unsqueeze(0) > thresholds.to(
        device=cue.device, dtype=cue.dtype
    ).unsqueeze(1)
    return {
        "tp": (prediction & target.unsqueeze(0)).sum(dim=1),
        "fp": (prediction & ~target.unsqueeze(0)).sum(dim=1),
        "fn": (~prediction & target.unsqueeze(0)).sum(dim=1),
        "tn": (~prediction & ~target.unsqueeze(0)).sum(dim=1),
    }


def metrics_from_counts(tp: float, fp: float, fn: float) -> dict[str, float]:
    """Return the repository's foreground IoU/F1/precision/recall metrics."""

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
        "f1": 2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "precision": precision,
        "recall": recall,
    }


def load_gt(path: Path, *, width: int, height: int) -> torch.Tensor:
    with Image.open(path) as image:
        mask = np.asarray(
            image.convert("L").resize(
                (width, height), resample=Image.Resampling.NEAREST
            ),
            dtype=np.uint8,
        )
    return torch.from_numpy(mask >= 128)


def initialize_accumulator(
    segment_names: Sequence[str],
    threshold_count: int,
) -> dict[str, Any]:
    def scope() -> dict[str, Any]:
        return {
            condition: {
                "tp": np.zeros(threshold_count, dtype=np.int64),
                "fp": np.zeros(threshold_count, dtype=np.int64),
                "fn": np.zeros(threshold_count, dtype=np.int64),
                "tn": np.zeros(threshold_count, dtype=np.int64),
                "frame_iou_sum": np.zeros(threshold_count, dtype=np.float64),
                "frame_f1_sum": np.zeros(threshold_count, dtype=np.float64),
                "frames": 0,
            }
            for condition in CONDITIONS
        }

    return {
        "overall": scope(),
        "segments": {name: scope() for name in segment_names},
    }


def add_counts(
    accumulator: dict[str, Any],
    condition: str,
    counts: dict[str, torch.Tensor],
) -> None:
    arrays = {name: value.detach().cpu().numpy().astype(np.int64) for name, value in counts.items()}
    row = accumulator[condition]
    for name in ("tp", "fp", "fn", "tn"):
        row[name] += arrays[name]
    tp, fp, fn = arrays["tp"], arrays["fp"], arrays["fn"]
    iou_denominator = tp + fp + fn
    f1_denominator = 2 * tp + fp + fn
    row["frame_iou_sum"] += np.divide(
        tp,
        iou_denominator,
        out=np.zeros_like(tp, dtype=np.float64),
        where=iou_denominator > 0,
    )
    row["frame_f1_sum"] += np.divide(
        2 * tp,
        f1_denominator,
        out=np.zeros_like(tp, dtype=np.float64),
        where=f1_denominator > 0,
    )
    row["frames"] += 1


def summarize_scope(
    accumulator: dict[str, Any],
    thresholds: Sequence[float],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for condition in CONDITIONS:
        row = accumulator[condition]
        curves = []
        for index, threshold in enumerate(thresholds):
            metrics = metrics_from_counts(
                float(row["tp"][index]),
                float(row["fp"][index]),
                float(row["fn"][index]),
            )
            curves.append(
                {
                    "threshold": float(threshold),
                    "mean_frame_iou": float(row["frame_iou_sum"][index] / row["frames"]),
                    "mean_frame_f1": float(row["frame_f1_sum"][index] / row["frames"]),
                    "aggregate_iou": metrics["iou"],
                    "aggregate_f1": metrics["f1"],
                    "aggregate_precision": metrics["precision"],
                    "aggregate_recall": metrics["recall"],
                    "tp": int(row["tp"][index]),
                    "fp": int(row["fp"][index]),
                    "fn": int(row["fn"][index]),
                    "tn": int(row["tn"][index]),
                }
            )
        primary_index = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - 0.25))
        best_index = max(range(len(curves)), key=lambda i: curves[i]["mean_frame_iou"])
        output[condition] = {
            "frames": int(row["frames"]),
            "primary_q_gt_0p25": curves[primary_index],
            "best_mean_frame_iou": curves[best_index],
            "threshold_curve": curves,
        }
    return output


def save_plot(summary: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.style.use("dark_background")
    figure, axis = plt.subplots(figsize=(9.5, 5.8), dpi=180)
    figure.patch.set_facecolor("#0b0d10")
    axis.set_facecolor("#11151a")
    colors = {"whole_pixel_power": "#ffd000", "l1_only_power": "#00e5ff"}
    labels = {
        "whole_pixel_power": "whole pixel cue power: P^0.3 × S",
        "l1_only_power": "L1-only power: P_l1pow × S",
    }
    for condition in CONDITIONS:
        curve = summary["overall"][condition]["threshold_curve"]
        axis.plot(
            [row["threshold"] for row in curve],
            [row["mean_frame_iou"] for row in curve],
            linewidth=3,
            color=colors[condition],
            label=labels[condition],
        )
    axis.axvline(0.25, linestyle="--", linewidth=1.6, color="#ff6075", label="primary q threshold 0.25")
    axis.set_xlabel("normalized cue threshold q", fontsize=12)
    axis.set_ylabel("mean-frame foreground IoU", fontsize=12)
    axis.set_title("Cue-vs-GT threshold sweep, 304 ESCD frames", fontsize=15, fontweight="bold")
    axis.grid(True, alpha=0.28)
    axis.legend(framealpha=0.85)
    figure.tight_layout()
    figure.savefig(path, facecolor=figure.get_facecolor(), bbox_inches="tight")
    plt.close(figure)


def threshold_grid() -> tuple[float, ...]:
    return tuple(float(value) for value in np.arange(0.05, 0.5001, 0.025))


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the Gaussian rasterizer")
    from gaussian_renderer import render
    from scene import GaussianModel

    records, _ = build_causal_records(args.source_path, max_frames=args.max_frames)
    base_ply = (args.source_path / BASE_PLY_REL).resolve()
    validate_cue_cache(args.cue_cache_root, base_ply, args.resolution)
    cameras = load_fixed_camera_index(args.fixed_cameras_json)
    segment_names = tuple(dict.fromkeys(str(record.segment_name) for record in records))
    thresholds = threshold_grid()
    threshold_tensor = torch.tensor(thresholds, device="cuda", dtype=torch.float32)
    accumulator = initialize_accumulator(segment_names, len(thresholds))
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
    background = torch.zeros(3, device="cuda")

    model = GaussianModel(sh_degree=3, active_sh_degree=3)
    model.load_ply(str(base_ply))
    for name in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"):
        getattr(model, name).requires_grad_(False)

    for frame_index, record in enumerate(records):
        view = build_fixed_cue_views(
            [record], cameras, args.cue_cache_root, args.resolution
        )[0][0]
        with torch.no_grad():
            reference_rgb = render(view, model, pipe, background)["render"]
            terms = oscd_pixel_terms(reference_rgb, view.original_image)
            original_pixel = normalized_oscd_pixel_cue_from_terms(terms, l1_exponent=1.0)
            l1_powered_pixel = normalized_oscd_pixel_cue_from_terms(
                terms, l1_exponent=args.exponent
            )
            semantic = semantic_from_cached_sum(view.candidate_map, original_pixel)
            cues = {
                "whole_pixel_power": fuse_power_product(
                    original_pixel, semantic, exponent=args.exponent
                )
                / 2.0,
                "l1_only_power": fuse_power_product(
                    l1_powered_pixel, semantic, exponent=1.0
                )
                / 2.0,
            }
        gt_path = args.source_path / "gt_mask" / record.name
        gt = load_gt(
            gt_path,
            width=int(view.image_width),
            height=int(view.image_height),
        ).cuda(non_blocking=True)
        for condition, cue in cues.items():
            counts = threshold_counts(cue, gt, threshold_tensor)
            add_counts(accumulator["overall"], condition, counts)
            add_counts(accumulator["segments"][str(record.segment_name)], condition, counts)
        if frame_index == 0 or (frame_index + 1) % args.progress_interval == 0:
            print(f"[{frame_index + 1:03d}/{len(records):03d}] {record.name}", flush=True)

    summary = {
        "schema_version": 1,
        "comparison": {
            "whole_pixel_power": "q = P^0.3 * S",
            "l1_only_power": "q = norm(0.8*L1^0.3 + 0.2*(1-SSIM)) * S",
        },
        "metric_contract": (
            "per-frame foreground IoU against resized aggregate GT; primary q>0.25 "
            "matches historical raw 2q>0.5"
        ),
        "frames": len(records),
        "resolution": args.resolution,
        "exponent": args.exponent,
        "thresholds": list(thresholds),
        "overall": summarize_scope(accumulator["overall"], thresholds),
        "segments": {
            name: summarize_scope(scope, thresholds)
            for name, scope in accumulator["segments"].items()
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    plot_path = args.output_dir / "threshold_sweep.png"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    save_plot(summary, plot_path)
    print(json.dumps({"summary": str(summary_path), "plot": str(plot_path)}, indent=2))
    return summary


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", type=Path, default=Path(DEFAULT_SOURCE))
    parser.add_argument("--fixed-cameras-json", type=Path, default=Path(DEFAULT_FIXED_CAMERAS))
    parser.add_argument("--cue-cache-root", type=Path, default=Path(DEFAULT_CUE_CACHE))
    parser.add_argument("--resolution", type=positive_float, default=4.0)
    parser.add_argument("--exponent", type=positive_float, default=0.3)
    parser.add_argument("--max-frames", type=positive_int, default=None)
    parser.add_argument("--progress-interval", type=positive_int, default=25)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/cue_power_location_gt_miou"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
