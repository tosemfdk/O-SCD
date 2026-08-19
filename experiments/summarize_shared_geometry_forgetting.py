"""Summarize state retention across shared-geometry boundary checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
import torch

from experiments.visualize_temporal_state_switch import load_font


DEFAULT_RUN = Path(
    "outputs/instance1_scene_change1_2_3_temporal_shared_geometry_"
    "forgetting_oscd_cues_allframes_120"
)


def default_eval_dir(run_dir: Path, state: int) -> Path:
    return run_dir.parent / f"{run_dir.name}_after_s{state}_confusion"


def scene_metrics(summary: dict[str, Any]) -> dict[int, dict[str, Any]]:
    output: dict[int, dict[str, Any]] = {}
    for row in summary["per_scene"]:
        scene = str(row["scope"])
        suffix = scene.removeprefix("scene_change")
        output[int(suffix) - 1] = row
    return output


def distribution(values: torch.Tensor) -> dict[str, float | int | None]:
    if values.numel() == 0:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    values = values.detach().float()
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "median": float(values.median().item()),
        "p95": float(torch.quantile(values, 0.95).item()),
        "max": float(values.max().item()),
    }


def checkpoint_overwrite_audit(run_dir: Path) -> dict[str, Any]:
    final = torch.load(
        run_dir / "state2_complete_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )["state_dict"]
    result: dict[str, Any] = {}
    geometry_names = (
        "shared_xyz_delta",
        "shared_opacity_delta",
        "shared_scaling_delta",
        "shared_rotation_delta",
    )
    for state in (0, 1):
        boundary = torch.load(
            run_dir / f"state{state}_complete_checkpoint.pt",
            map_location="cpu",
            weights_only=False,
        )["state_dict"]
        valid = boundary["state_valid"][:, state]
        dc_drift = float(
            (
                final["state_change_dc"][:, state]
                - boundary["state_change_dc"][:, state]
            )
            .abs()
            .max()
            .item()
        )
        geometry_drift = {}
        for name in geometry_names:
            row_magnitude = (final[name] - boundary[name]).flatten(start_dim=1).norm(dim=1)
            geometry_drift[name] = distribution(row_magnitude[valid])
        result[f"state{state}_boundary_to_final"] = {
            "state_dc_max_abs_drift": dc_drift,
            "valid_gaussian_count": int(valid.sum().item()),
            "shared_geometry_row_drift": geometry_drift,
        }
        del boundary
    return result


def heat_color(value: float) -> tuple[int, int, int]:
    value = max(0.0, min(1.0, float(value)))
    return (
        int(245 - 150 * value),
        int(225 - 25 * value),
        int(225 - 150 * value),
    )


def save_matrix_png(result: dict[str, Any], path: Path) -> None:
    cell_w, cell_h = 180, 78
    left, top = 220, 118
    width = left + 3 * cell_w + 30
    height = top + 3 * cell_h + 115
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(23)
    font = load_font(18)
    small = load_font(14)
    draw.text(
        (20, 16),
        "Shared geometry retention (mean frame IoU)",
        fill=(0, 0, 0),
        font=title_font,
    )
    draw.text(
        (20, 54),
        "Rows: training completed through state | Columns: state re-rendered",
        fill=(45, 45, 45),
        font=small,
    )
    for state in range(3):
        draw.text(
            (left + state * cell_w + 55, top - 35),
            f"S{state}",
            fill=(0, 0, 0),
            font=font,
        )
    matrix = result["mean_frame_iou_matrix"]
    for stage in range(3):
        draw.text(
            (20, top + stage * cell_h + 25),
            f"after training S{stage}",
            fill=(0, 0, 0),
            font=font,
        )
        for state in range(3):
            x0 = left + state * cell_w
            y0 = top + stage * cell_h
            trained = state <= stage
            value = float(matrix[stage][state])
            color = heat_color(value) if trained else (225, 225, 225)
            draw.rectangle(
                (x0, y0, x0 + cell_w - 8, y0 + cell_h - 8),
                fill=color,
                outline=(80, 80, 80),
                width=1,
            )
            label = f"{value:.4f}" if trained else f"{value:.4f}\n(untrained DC)"
            draw.multiline_text(
                (x0 + 40, y0 + (16 if trained else 8)),
                label,
                fill=(0, 0, 0),
                font=font if trained else small,
                spacing=2,
                align="center",
            )
    forgetting = result["forgetting"]
    draw.text(
        (20, top + 3 * cell_h + 18),
        f"S0: after S2 - after S0 = {forgetting['s0_after_s2_minus_after_s0']:+.4f}",
        fill=(150, 0, 0),
        font=font,
    )
    draw.text(
        (20, top + 3 * cell_h + 51),
        f"S1: after S2 - after S1 = {forgetting['s1_after_s2_minus_after_s1']:+.4f}",
        fill=(150, 0, 0),
        font=font,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the shared-geometry forgetting retention matrix"
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--after-s0-dir", type=Path)
    parser.add_argument("--after-s1-dir", type=Path)
    parser.add_argument("--after-s2-dir", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-png", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_dirs = [
        args.after_s0_dir or default_eval_dir(args.run_dir, 0),
        args.after_s1_dir or default_eval_dir(args.run_dir, 1),
        args.after_s2_dir or default_eval_dir(args.run_dir, 2),
    ]
    eval_summaries = [
        json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        for directory in eval_dirs
    ]
    metrics = [scene_metrics(summary) for summary in eval_summaries]
    iou_matrix = [
        [float(metrics[stage][state]["mean_frame_iou"]) for state in range(3)]
        for stage in range(3)
    ]
    f1_matrix = [
        [float(metrics[stage][state]["mean_frame_f1"]) for state in range(3)]
        for stage in range(3)
    ]
    run_summary = json.loads(
        (args.run_dir / "summary.json").read_text(encoding="utf-8")
    )
    result = {
        "script": "experiments/summarize_shared_geometry_forgetting.py",
        "run_dir": str(args.run_dir),
        "evaluation_dirs": [str(directory) for directory in eval_dirs],
        "rows": ["after_s0", "after_s1", "after_s2"],
        "columns": ["state0", "state1", "state2"],
        "mean_frame_iou_matrix": iou_matrix,
        "mean_frame_f1_matrix": f1_matrix,
        "forgetting": {
            "s0_after_s1_minus_after_s0": iou_matrix[1][0] - iou_matrix[0][0],
            "s0_after_s2_minus_after_s0": iou_matrix[2][0] - iou_matrix[0][0],
            "s1_after_s2_minus_after_s1": iou_matrix[2][1] - iou_matrix[1][1],
        },
        "interpretation_contract": {
            "negative_delta_means_forgetting": True,
            "later_state_dc_is_not_trained_in_gray_upper_triangle": True,
            "geometry_is_shared_across_all_cells": True,
            "cross_state_replay": False,
        },
        "shared_geometry_transition_movement": run_summary[
            "gradient_isolation_audit"
        ]["shared_geometry_transition_movement"],
        "checkpoint_overwrite_audit": checkpoint_overwrite_audit(args.run_dir),
    }
    output_json = args.output_json or args.run_dir / "retention_matrix.json"
    output_png = args.output_png or args.run_dir / "retention_matrix.png"
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    save_matrix_png(result, output_png)
    print(json.dumps({"json": str(output_json), "png": str(output_png), **result["forgetting"]}, indent=2))


if __name__ == "__main__":
    main()
