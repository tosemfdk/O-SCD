#!/usr/bin/env python3
"""Summarize the 2x2 cue-fusion by regularization-locality ablation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


CONDITIONS = (
    "global_sum",
    "global_product",
    "local_sum",
    "local_product",
    "local_raw_product",
)
CONDITION_DIRECTORIES = {
    "global_sum": "fresh_global_sum",
    "global_product": "fresh_global_product",
    "local_sum": "local_sum",
    "local_product": "local_product",
    "local_raw_product": "local_raw_product",
}
PHASES = ("online_at_arrival", "online_final_rerender")
METRICS = ("iou", "f1", "precision", "recall", "predicted_fraction")


def load_binary(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    if shape is not None and image.shape != shape:
        image = cv2.resize(
            image,
            (shape[1], shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return image > 127


def mask_metrics(prediction: np.ndarray, ground_truth: np.ndarray) -> dict[str, float]:
    tp = int(np.logical_and(prediction, ground_truth).sum())
    fp = int(np.logical_and(prediction, ~ground_truth).sum())
    fn = int(np.logical_and(~prediction, ground_truth).sum())
    iou_denominator = tp + fp + fn
    f1_denominator = 2 * tp + fp + fn
    return {
        "iou": tp / iou_denominator if iou_denominator else 0.0,
        "f1": 2 * tp / f1_denominator if f1_denominator else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "predicted_fraction": float(prediction.mean()),
    }


def mean_metrics(rows: list[dict], condition: str) -> dict[str, float]:
    return {
        metric: float(np.mean([row[f"{condition}_{metric}"] for row in rows]))
        for metric in METRICS
    }


def delta_diagnostics(rows: list[dict], left: str, right: str) -> dict:
    gt_fraction = np.asarray([row["gt_fraction"] for row in rows])
    delta_iou = np.asarray(
        [row[f"{left}_iou"] - row[f"{right}_iou"] for row in rows]
    )
    order = np.argsort(gt_fraction)
    groups = np.array_split(order, 3)
    labels = ("small", "medium", "large")
    correlation = 0.0
    if gt_fraction.std() > 0.0 and delta_iou.std() > 0.0:
        correlation = float(np.corrcoef(gt_fraction, delta_iou)[0, 1])
    return {
        "comparison": f"{left} - {right}",
        "pearson_gt_fraction_vs_iou_delta": correlation,
        "mean_delta": {
            metric: float(
                np.mean(
                    [
                        row[f"{left}_{metric}"]
                        - row[f"{right}_{metric}"]
                        for row in rows
                    ]
                )
            )
            for metric in METRICS
        },
        "iou_delta_by_gt_area_tertile": {
            label: {
                "frames": int(len(indices)),
                "mean_gt_fraction": float(gt_fraction[indices].mean()),
                "mean_iou_delta": float(delta_iou[indices].mean()),
            }
            for label, indices in zip(labels, groups, strict=True)
        },
    }


def locality_diagnostics(rows: list[dict], fusion: str) -> dict:
    return delta_diagnostics(rows, f"local_{fusion}", f"global_{fusion}")


def product_scale_diagnostics(rows: list[dict]) -> dict:
    return {
        "raw_minus_scaled_local": delta_diagnostics(
            rows, "local_raw_product", "local_product"
        ),
        "raw_local_minus_global": delta_diagnostics(
            rows, "local_raw_product", "global_product"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/Instance_1"),
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("outputs/part11_loss_locality_ablation_20260813"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_rows: list[dict] = []
    summary = {
        "binary_rule": "rendered score > 0.5",
        "aggregation": "arithmetic mean of per-frame metrics",
        "conditions": list(CONDITIONS),
        "scenes": {},
    }
    for scene_index in (1, 2, 3):
        scene = f"scene_change{scene_index}"
        gt_paths = sorted((args.data_root / scene / "gt_mask").glob("*.png"))
        if not gt_paths:
            raise FileNotFoundError(f"No GT masks for {scene}")
        scene_summary = {}
        for phase in PHASES:
            prediction_paths = {
                condition: {
                    path.stem: path
                    for path in (
                        args.experiment_root
                        / f"scene{scene_index}_{CONDITION_DIRECTORIES[condition]}"
                        / "renders"
                        / phase
                    ).glob("*.png")
                }
                for condition in CONDITIONS
            }
            expected_stems = {path.stem for path in gt_paths}
            for condition, paths in prediction_paths.items():
                if set(paths) != expected_stems:
                    raise ValueError(
                        f"{scene}/{phase}/{condition}: "
                        f"predictions={len(paths)} expected={len(expected_stems)}"
                    )
            rows = []
            for gt_path in gt_paths:
                first_prediction = load_binary(
                    prediction_paths[CONDITIONS[0]][gt_path.stem]
                )
                gt = load_binary(gt_path, first_prediction.shape)
                row = {
                    "scene": scene,
                    "phase": phase,
                    "frame": gt_path.stem,
                    "gt_fraction": float(gt.mean()),
                }
                for condition in CONDITIONS:
                    prediction = load_binary(
                        prediction_paths[condition][gt_path.stem], gt.shape
                    )
                    for metric, value in mask_metrics(prediction, gt).items():
                        row[f"{condition}_{metric}"] = value
                rows.append(row)
                all_rows.append(row)
            scene_summary[phase] = {
                "frames": len(rows),
                "metrics": {
                    condition: mean_metrics(rows, condition)
                    for condition in CONDITIONS
                },
                "locality_diagnostics": {
                    fusion: locality_diagnostics(rows, fusion)
                    for fusion in ("sum", "product")
                },
                "product_scale_diagnostics": product_scale_diagnostics(rows),
            }
        summary["scenes"][scene] = scene_summary

    summary["overall"] = {}
    for phase in PHASES:
        rows = [row for row in all_rows if row["phase"] == phase]
        summary["overall"][phase] = {
            "frames": len(rows),
            "metrics": {
                condition: mean_metrics(rows, condition)
                for condition in CONDITIONS
            },
            "locality_diagnostics": {
                fusion: locality_diagnostics(rows, fusion)
                for fusion in ("sum", "product")
            },
            "product_scale_diagnostics": product_scale_diagnostics(rows),
        }

    output_path = args.experiment_root / "summary.json"
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    csv_path = args.experiment_root / "per_frame_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
