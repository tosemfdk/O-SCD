"""Summarize S0 retention and S1 plasticity for influence-row freezing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw

from experiments.train_geometry_temporal_rchange import distribution
from experiments.train_real_temporal_rchange import file_checksum
from experiments.train_s0_influence_freeze_s1 import DEFAULT_OUTPUT, SOURCE_RUN
from experiments.visualize_temporal_state_switch import load_font


BASELINE_S0_EVAL = Path(f"{SOURCE_RUN}_after_s0_confusion")
BASELINE_S1_EVAL = Path(f"{SOURCE_RUN}_after_s1_confusion")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def scene_metrics(summary: dict[str, Any], scene: str) -> dict[str, Any]:
    for row in summary["per_scene"]:
        if row["scope"] == scene:
            return row
    raise KeyError(f"Missing {scene} metrics")


def geometry_movement(
    state0_checkpoint: Path,
    state1_checkpoint: Path,
    influence_artifact: Path,
) -> tuple[dict[str, Any], dict[str, float]]:
    before = torch.load(state0_checkpoint, map_location="cpu", weights_only=False)
    after = torch.load(state1_checkpoint, map_location="cpu", weights_only=False)
    artifact = torch.load(influence_artifact, map_location="cpu", weights_only=False)
    before_state = before["state_dict"]
    after_state = after["state_dict"]
    frozen = artifact["valid_mask"].bool()
    state1_valid = after_state["state_valid"][:, 1].bool()
    groups = {
        "s0_influence_frozen": frozen,
        "s1_valid_frozen": state1_valid & frozen,
        "s1_valid_unfrozen": state1_valid & ~frozen,
        "all_unfrozen": ~frozen,
    }
    result: dict[str, Any] = {}
    frozen_max_abs: dict[str, float] = {}
    for name in ("xyz", "opacity", "scaling", "rotation"):
        key = f"shared_{name}_delta"
        difference = after_state[key] - before_state[key]
        row_distance = difference.flatten(start_dim=1).norm(dim=1)
        result[name] = {
            group: distribution(row_distance[mask])
            for group, mask in groups.items()
        }
        frozen_difference = difference[frozen]
        frozen_max_abs[name] = (
            0.0
            if frozen_difference.numel() == 0
            else float(frozen_difference.abs().max().item())
        )
    return result, frozen_max_abs


def retention_figure(
    iou: list[list[float | None]],
    f1: list[list[float | None]],
    output_path: Path,
) -> None:
    rows = ["After S0 boundary", "After S1 unfrozen", "After S1 S0-influence frozen"]
    columns = ["Evaluate S0", "Evaluate S1"]
    cell_w, cell_h = 170, 72
    table_w = 210 + len(columns) * cell_w
    gap = 28
    width = 30 + table_w * 2 + gap
    height = 125 + len(rows) * cell_h + 45
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(23)
    font = load_font(17)
    small = load_font(14)
    draw.text(
        (20, 15),
        "S0 signed-influence geometry freeze: retention vs S1 plasticity",
        fill=(0, 0, 0),
        font=title_font,
    )

    def heat_color(value: float) -> tuple[int, int, int]:
        clipped = max(0.0, min(1.0, value))
        return (
            int(245 - 150 * clipped),
            int(225 - 25 * clipped),
            int(225 - 150 * clipped),
        )

    for table_index, (matrix, title) in enumerate(
        ((iou, "Mean per-frame IoU"), (f1, "Mean per-frame F1"))
    ):
        origin_x = 20 + table_index * (table_w + gap)
        draw.text((origin_x, 64), title, fill=(0, 0, 0), font=font)
        for column, label in enumerate(columns):
            draw.text(
                (origin_x + 210 + column * cell_w + 28, 93),
                label,
                fill=(0, 0, 0),
                font=small,
            )
        for row, row_label in enumerate(rows):
            y0 = 122 + row * cell_h
            draw.text((origin_x, y0 + 22), row_label, fill=(0, 0, 0), font=small)
            for column in range(len(columns)):
                x0 = origin_x + 210 + column * cell_w
                value = matrix[row][column]
                color = (225, 225, 225) if value is None else heat_color(value)
                draw.rectangle(
                    (x0, y0, x0 + cell_w - 8, y0 + cell_h - 8),
                    fill=color,
                    outline=(80, 80, 80),
                    width=1,
                )
                label = "—" if value is None else f"{value:.4f}"
                draw.text((x0 + 52, y0 + 22), label, fill=(0, 0, 0), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def movement_figure(movement: dict[str, Any], output_path: Path) -> None:
    width, height = 1100, 760
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(23)
    font = load_font(16)
    small = load_font(13)
    draw.text(
        (20, 15),
        "Frozen rows stay fixed while remaining S1-valid rows adapt",
        fill=(0, 0, 0),
        font=title_font,
    )
    panel_w, panel_h = 510, 310
    origins = ((25, 70), (565, 70), (25, 405), (565, 405))
    for (origin_x, origin_y), name in zip(
        origins, ("xyz", "opacity", "scaling", "rotation")
    ):
        frozen = movement[name]["s0_influence_frozen"]
        free = movement[name]["s1_valid_unfrozen"]
        values = [
            float(frozen["mean"] or 0.0),
            float(frozen["p95"] or 0.0),
            float(free["mean"] or 0.0),
            float(free["p95"] or 0.0),
        ]
        labels = ["frozen\nmean", "frozen\np95", "free\nmean", "free\np95"]
        maximum = max(values) or 1.0
        chart_left = origin_x + 52
        chart_top = origin_y + 46
        chart_bottom = origin_y + panel_h - 52
        chart_height = chart_bottom - chart_top
        draw.rectangle(
            (origin_x, origin_y, origin_x + panel_w, origin_y + panel_h),
            outline=(180, 180, 180),
            width=1,
        )
        draw.text(
            (origin_x + 12, origin_y + 10),
            f"{name} row movement after S1",
            fill=(0, 0, 0),
            font=font,
        )
        draw.line(
            (chart_left, chart_top, chart_left, chart_bottom),
            fill=(100, 100, 100),
            width=1,
        )
        draw.line(
            (chart_left, chart_bottom, origin_x + panel_w - 18, chart_bottom),
            fill=(100, 100, 100),
            width=1,
        )
        bar_w = 70
        for index, (value, label) in enumerate(zip(values, labels)):
            x0 = chart_left + 30 + index * 100
            bar_h = int(chart_height * value / maximum)
            color = (70, 120, 210) if index < 2 else (230, 130, 55)
            draw.rectangle(
                (x0, chart_bottom - bar_h, x0 + bar_w, chart_bottom),
                fill=color,
                outline=(60, 60, 60),
            )
            draw.multiline_text(
                (x0 + 5, chart_bottom + 6),
                label,
                fill=(0, 0, 0),
                font=small,
                spacing=1,
                align="center",
            )
            draw.text(
                (x0, max(chart_top, chart_bottom - bar_h - 21)),
                f"{value:.4g}",
                fill=(0, 0, 0),
                font=small,
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def summarize(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    baseline_s0 = read_json(args.baseline_s0_eval / "summary.json")
    baseline_s1 = read_json(args.baseline_s1_eval / "summary.json")
    frozen_s1 = read_json(args.frozen_s1_eval / "summary.json")
    b0_s0 = scene_metrics(baseline_s0, "scene_change1")
    b1_s0 = scene_metrics(baseline_s1, "scene_change1")
    b1_s1 = scene_metrics(baseline_s1, "scene_change2")
    f1_s0 = scene_metrics(frozen_s1, "scene_change1")
    f1_s1 = scene_metrics(frozen_s1, "scene_change2")

    iou = [
        [float(b0_s0["mean_frame_iou"]), None],
        [float(b1_s0["mean_frame_iou"]), float(b1_s1["mean_frame_iou"])],
        [float(f1_s0["mean_frame_iou"]), float(f1_s1["mean_frame_iou"])],
    ]
    f1 = [
        [float(b0_s0["mean_frame_f1"]), None],
        [float(b1_s0["mean_frame_f1"]), float(b1_s1["mean_frame_f1"])],
        [float(f1_s0["mean_frame_f1"]), float(f1_s1["mean_frame_f1"])],
    ]
    movement, frozen_max_abs = geometry_movement(
        args.state0_checkpoint,
        args.state1_checkpoint,
        args.influence_artifact,
    )
    artifact = torch.load(args.influence_artifact, map_location="cpu", weights_only=False)
    result = {
        "script": "experiments/summarize_s0_influence_freeze_s1.py",
        "contract": "compare_unfrozen_and_s0_influence_frozen_s1_continuations",
        "baseline_s0_eval": str(args.baseline_s0_eval),
        "baseline_s1_eval": str(args.baseline_s1_eval),
        "frozen_s1_eval": str(args.frozen_s1_eval),
        "state0_checkpoint": str(args.state0_checkpoint),
        "state0_checkpoint_sha256": file_checksum(args.state0_checkpoint),
        "state1_checkpoint": str(args.state1_checkpoint),
        "state1_checkpoint_sha256": file_checksum(args.state1_checkpoint),
        "influence_artifact": str(args.influence_artifact),
        "influence_artifact_sha256": file_checksum(args.influence_artifact),
        "frozen_gaussians": int(artifact["valid_mask"].sum().item()),
        "mean_frame_iou_matrix": iou,
        "mean_frame_f1_matrix": f1,
        "rows": ["after_s0_boundary", "after_s1_unfrozen", "after_s1_s0_influence_frozen"],
        "columns": ["state0", "state1"],
        "deltas": {
            "unfrozen_s0_forgetting_after_s1": iou[1][0] - iou[0][0],
            "frozen_s0_forgetting_after_s1": iou[2][0] - iou[0][0],
            "s0_retention_recovery_vs_unfrozen": iou[2][0] - iou[1][0],
            "s1_plasticity_delta_vs_unfrozen": iou[2][1] - iou[1][1],
        },
        "shared_geometry_movement_after_s1": movement,
        "frozen_geometry_max_abs_drift": frozen_max_abs,
        "frozen_geometry_exactly_preserved": max(frozen_max_abs.values(), default=0.0) == 0.0,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "influence_freeze_comparison.json"
    retention_path = args.output_dir / "influence_freeze_retention.png"
    movement_path = args.output_dir / "influence_freeze_geometry_movement.png"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    retention_figure(iou, f1, retention_path)
    movement_figure(movement, movement_path)
    print(json.dumps(result, indent=2))
    return json_path, retention_path, movement_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-s0-eval", type=Path, default=BASELINE_S0_EVAL)
    parser.add_argument("--baseline-s1-eval", type=Path, default=BASELINE_S1_EVAL)
    parser.add_argument("--frozen-s1-eval", type=Path, default=DEFAULT_OUTPUT / "confusion_after_s1")
    parser.add_argument("--state0-checkpoint", type=Path, default=SOURCE_RUN / "state0_complete_checkpoint.pt")
    parser.add_argument("--state1-checkpoint", type=Path, default=DEFAULT_OUTPUT / "state1_complete_checkpoint.pt")
    parser.add_argument("--influence-artifact", type=Path, default=DEFAULT_OUTPUT / "signed_influence/state0_signed_influence.pt")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    summarize(parse_args())


if __name__ == "__main__":
    main()
