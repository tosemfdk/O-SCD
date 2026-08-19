"""Compare S1 geometry-freeze scopes from one shared-geometry S0 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw

from experiments.train_s0_influence_freeze_s1 import (
    ALL_GEOMETRY_OUTPUT,
    MODEL_BUFFER_OUTPUT,
    SOURCE_RUN,
)
from experiments.visualize_temporal_state_switch import load_font


UNFROZEN_S0_EVAL = Path(f"{SOURCE_RUN}_after_s0_confusion")
UNFROZEN_S1_EVAL = Path(f"{SOURCE_RUN}_after_s1_confusion")
DC_ONLY_EVAL = Path(
    "outputs/instance1_scene_change1_2_3_"
    "temporal_confusion_oscd_cues_allframes_120"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def scene_metric(summary: dict[str, Any], scope: str, metric: str) -> float:
    for row in summary["per_scene"]:
        if row["scope"] == scope:
            return float(row[metric])
    raise KeyError(f"Missing {scope} in evaluation summary")


def exact_shared_geometry_drift(
    source_checkpoint: Path,
    result_checkpoint: Path,
) -> dict[str, dict[str, float | bool]]:
    source = torch.load(
        source_checkpoint, map_location="cpu", weights_only=False
    )["state_dict"]
    result = torch.load(
        result_checkpoint, map_location="cpu", weights_only=False
    )["state_dict"]
    audit: dict[str, dict[str, float | bool]] = {}
    for name in (
        "shared_xyz_delta",
        "shared_opacity_delta",
        "shared_scaling_delta",
        "shared_rotation_delta",
    ):
        difference = (result[name] - source[name]).abs()
        audit[name] = {
            "exact": bool(torch.equal(result[name], source[name])),
            "max_abs": float(difference.max().item()),
        }
    audit["state0_change_dc"] = {
        "exact": bool(
            torch.equal(
                result["state_change_dc"][:, 0],
                source["state_change_dc"][:, 0],
            )
        ),
        "max_abs": float(
            (
                result["state_change_dc"][:, 0]
                - source["state_change_dc"][:, 0]
            )
            .abs()
            .max()
            .item()
        ),
    }
    return audit


def save_table(result: dict[str, Any], path: Path) -> None:
    rows = result["rows"]
    cell_w, cell_h = 190, 66
    left, top = 345, 120
    width = left + 2 * cell_w + 35
    height = top + len(rows) * cell_h + 55
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title = load_font(23)
    font = load_font(16)
    small = load_font(14)
    draw.text(
        (20, 15),
        "S0 checkpoint geometry-freeze scope control",
        fill=(0, 0, 0),
        font=title,
    )
    draw.text(
        (20, 52),
        "Mean per-frame IoU | support-valid overlap=542,045 | influence-frozen=14,477",
        fill=(45, 45, 45),
        font=small,
    )
    draw.text(
        (20, 76),
        "Rows marked same-S0 all start from the identical S0 shared-geometry checkpoint",
        fill=(45, 45, 45),
        font=small,
    )
    for column, label in enumerate(("Evaluate S0", "Evaluate S1")):
        draw.text(
            (left + column * cell_w + 38, top - 30),
            label,
            fill=(0, 0, 0),
            font=font,
        )

    def color(value: float) -> tuple[int, int, int]:
        clipped = max(0.0, min(1.0, value))
        return (
            int(245 - 150 * clipped),
            int(225 - 25 * clipped),
            int(225 - 150 * clipped),
        )

    for row_index, row in enumerate(rows):
        y0 = top + row_index * cell_h
        draw.text((20, y0 + 20), row["label"], fill=(0, 0, 0), font=font)
        for column, key in enumerate(("state0", "state1")):
            x0 = left + column * cell_w
            value = row[key]
            draw.rectangle(
                (x0, y0, x0 + cell_w - 8, y0 + cell_h - 8),
                fill=(225, 225, 225) if value is None else color(float(value)),
                outline=(80, 80, 80),
                width=1,
            )
            label = "—" if value is None else f"{float(value):.4f}"
            draw.text((x0 + 58, y0 + 19), label, fill=(0, 0, 0), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def compare(args: argparse.Namespace) -> tuple[Path, Path]:
    boundary = read_json(args.unfrozen_s0_eval / "summary.json")
    unfrozen = read_json(args.unfrozen_s1_eval / "summary.json")
    partial = read_json(args.partial_run / "confusion_after_s1/summary.json")
    all_frozen = read_json(args.all_frozen_run / "confusion_after_s1/summary.json")
    dc_only = read_json(args.dc_only_eval / "summary.json")

    state0_checkpoint = torch.load(
        args.source_checkpoint, map_location="cpu", weights_only=False
    )["state_dict"]
    all_checkpoint = torch.load(
        args.all_frozen_checkpoint, map_location="cpu", weights_only=False
    )["state_dict"]
    influence = torch.load(
        args.influence_artifact, map_location="cpu", weights_only=False
    )["valid_mask"].bool()
    state0_valid = state0_checkpoint["state_valid"][:, 0].bool()
    state1_valid = state0_checkpoint["state_valid"][:, 1].bool()
    overlap = state0_valid & state1_valid
    frozen_metadata = all_checkpoint["geometry_frozen"].bool()

    def metrics(summary: dict[str, Any]) -> tuple[float, float]:
        return (
            scene_metric(summary, "scene_change1", "mean_frame_iou"),
            scene_metric(summary, "scene_change2", "mean_frame_iou"),
        )

    boundary_s0 = scene_metric(
        boundary, "scene_change1", "mean_frame_iou"
    )
    unfrozen_s0, unfrozen_s1 = metrics(unfrozen)
    partial_s0, partial_s1 = metrics(partial)
    all_s0, all_s1 = metrics(all_frozen)
    dc_s0, dc_s1 = metrics(dc_only)
    rows = [
        {"label": "S0 boundary", "state0": boundary_s0, "state1": None},
        {"label": "same-S0: all geometry frozen", "state0": all_s0, "state1": all_s1},
        {"label": "same-S0: 14,477 influence frozen", "state0": partial_s0, "state1": partial_s1},
        {"label": "same-S0: geometry unfrozen", "state0": unfrozen_s0, "state1": unfrozen_s1},
        {"label": "original base geometry DC-only", "state0": dc_s0, "state1": dc_s1},
    ]
    result = {
        "script": "experiments/compare_s0_geometry_freeze_scopes.py",
        "rows": rows,
        "columns": ["state0", "state1"],
        "validity_semantics": {
            "state0_support_valid": int(state0_valid.sum().item()),
            "state1_support_valid": int(state1_valid.sum().item()),
            "state0_state1_support_valid_overlap": int(overlap.sum().item()),
            "state0_signed_influence_valid": int(influence.sum().item()),
            "signed_influence_valid_in_support_overlap": int(
                (influence & overlap).sum().item()
            ),
            "support_valid_definition": (
                "At least one accumulated raster contribution inside the "
                "state's thresholded O-SCD cue support."
            ),
            "signed_influence_valid_definition": (
                "Mean signed opacity removal/insertion influence on the soft "
                "0.5-threshold mask area is at least 1e-4 over S0 views."
            ),
        },
        "all_geometry_control": {
            "persistent_frozen_count": int(frozen_metadata.sum().item()),
            "persistent_frozen_fraction": float(
                frozen_metadata.float().mean().item()
            ),
            "geometry_and_s0_dc_drift": exact_shared_geometry_drift(
                args.source_checkpoint,
                args.all_frozen_checkpoint,
            ),
        },
        "deltas": {
            "all_frozen_minus_s0_boundary_on_s0": all_s0 - boundary_s0,
            "partial_minus_all_frozen_on_s0": partial_s0 - all_s0,
            "partial_minus_all_frozen_on_s1": partial_s1 - all_s1,
            "unfrozen_minus_all_frozen_on_s0": unfrozen_s0 - all_s0,
            "unfrozen_minus_all_frozen_on_s1": unfrozen_s1 - all_s1,
            "all_frozen_same_s0_minus_original_base_dc_only_s0": all_s0 - dc_s0,
            "all_frozen_same_s0_minus_original_base_dc_only_s1": all_s1 - dc_s1,
        },
        "interpretation": (
            "All-geometry freezing exactly retains S0 but underfits S1. Partial "
            "free geometry improves S1 over that same-checkpoint control while "
            "damaging S0; original base-geometry DC-only is a different basis."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "s0_geometry_freeze_scope_comparison.json"
    png_path = args.output_dir / "s0_geometry_freeze_scope_comparison.png"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    save_table(result, png_path)
    print(json.dumps(result, indent=2))
    return json_path, png_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, default=SOURCE_RUN / "state0_complete_checkpoint.pt")
    parser.add_argument("--unfrozen-s0-eval", type=Path, default=UNFROZEN_S0_EVAL)
    parser.add_argument("--unfrozen-s1-eval", type=Path, default=UNFROZEN_S1_EVAL)
    parser.add_argument("--partial-run", type=Path, default=MODEL_BUFFER_OUTPUT)
    parser.add_argument("--all-frozen-run", type=Path, default=ALL_GEOMETRY_OUTPUT)
    parser.add_argument("--dc-only-eval", type=Path, default=DC_ONLY_EVAL)
    parser.add_argument(
        "--all-frozen-checkpoint",
        type=Path,
        default=ALL_GEOMETRY_OUTPUT / "state1_complete_checkpoint.pt",
    )
    parser.add_argument(
        "--influence-artifact",
        type=Path,
        default=MODEL_BUFFER_OUTPUT / "signed_influence/state0_signed_influence.pt",
    )
    parser.add_argument("--output-dir", type=Path, default=ALL_GEOMETRY_OUTPUT)
    return parser.parse_args()


def main() -> None:
    compare(parse_args())


if __name__ == "__main__":
    main()
