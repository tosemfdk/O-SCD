"""Compare external projection and persistent model-buffer geometry freezing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw

from experiments.train_real_temporal_rchange import file_checksum
from experiments.train_s0_influence_freeze_s1 import (
    DEFAULT_OUTPUT as EXTERNAL_OUTPUT,
    MODEL_BUFFER_OUTPUT,
)
from experiments.visualize_temporal_state_switch import load_font


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parameter_difference(
    external_checkpoint: Path,
    model_buffer_checkpoint: Path,
) -> dict[str, Any]:
    external = torch.load(
        external_checkpoint, map_location="cpu", weights_only=False
    )["state_dict"]
    model_buffer = torch.load(
        model_buffer_checkpoint, map_location="cpu", weights_only=False
    )["state_dict"]
    result: dict[str, Any] = {}
    for name in sorted(set(external) & set(model_buffer)):
        left, right = external[name], model_buffer[name]
        if not torch.is_floating_point(left):
            result[name] = {"exact": bool(torch.equal(left, right))}
            continue
        difference = (left - right).abs()
        finite = difference[torch.isfinite(difference)]
        result[name] = {
            "exact": bool(torch.equal(left, right)),
            "max_abs": 0.0 if finite.numel() == 0 else float(finite.max().item()),
            "mean_abs": 0.0
            if finite.numel() == 0
            else float(finite.float().mean().item()),
        }
    return result


def save_table(result: dict[str, Any], path: Path) -> None:
    rows = result["rows"]
    cell_w, cell_h = 190, 70
    left, top = 310, 112
    width = left + 2 * cell_w + 35
    height = top + len(rows) * cell_h + 70
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(23)
    font = load_font(17)
    small = load_font(14)
    draw.text(
        (20, 16),
        "Geometry freeze implementation comparison",
        fill=(0, 0, 0),
        font=title_font,
    )
    draw.text(
        (20, 54),
        "Mean per-frame IoU; both freeze policies use the same 14,477 S0 rows",
        fill=(45, 45, 45),
        font=small,
    )
    for column, label in enumerate(("Evaluate S0", "Evaluate S1")):
        draw.text(
            (left + column * cell_w + 37, top - 34),
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
        draw.text((20, y0 + 23), row["label"], fill=(0, 0, 0), font=font)
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
            draw.text((x0 + 58, y0 + 21), label, fill=(0, 0, 0), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def compare(args: argparse.Namespace) -> tuple[Path, Path]:
    external = read_json(args.external_run / "influence_freeze_comparison.json")
    model_buffer = read_json(
        args.model_buffer_run / "influence_freeze_comparison.json"
    )
    external_artifact = torch.load(
        args.external_artifact, map_location="cpu", weights_only=False
    )
    model_artifact = torch.load(
        args.model_buffer_artifact, map_location="cpu", weights_only=False
    )
    external_iou = external["mean_frame_iou_matrix"]
    model_iou = model_buffer["mean_frame_iou_matrix"]
    rows = [
        {
            "label": "S0 boundary",
            "state0": external_iou[0][0],
            "state1": None,
        },
        {
            "label": "S1 unfrozen shared",
            "state0": external_iou[1][0],
            "state1": external_iou[1][1],
        },
        {
            "label": "S1 external mask + projection",
            "state0": external_iou[2][0],
            "state1": external_iou[2][1],
        },
        {
            "label": "S1 model geometry_frozen mask",
            "state0": model_iou[2][0],
            "state1": model_iou[2][1],
        },
    ]
    checkpoint = torch.load(
        args.model_buffer_checkpoint, map_location="cpu", weights_only=False
    )
    frozen_metadata = checkpoint["state_dict"]["geometry_frozen"]
    result = {
        "script": "experiments/compare_geometry_freeze_implementations.py",
        "rows": rows,
        "columns": ["state0", "state1"],
        "external_run": str(args.external_run),
        "model_buffer_run": str(args.model_buffer_run),
        "external_checkpoint_sha256": file_checksum(args.external_checkpoint),
        "model_buffer_checkpoint_sha256": file_checksum(
            args.model_buffer_checkpoint
        ),
        "same_influence_valid_mask": bool(
            torch.equal(
                external_artifact["valid_mask"], model_artifact["valid_mask"]
            )
        ),
        "persistent_geometry_frozen_count": int(frozen_metadata.sum().item()),
        "model_buffer_minus_external_projection": {
            "state0_mean_frame_iou": float(
                model_iou[2][0] - external_iou[2][0]
            ),
            "state1_mean_frame_iou": float(
                model_iou[2][1] - external_iou[2][1]
            ),
        },
        "both_runs_exactly_preserved_frozen_geometry": bool(
            external["frozen_geometry_exactly_preserved"]
            and model_buffer["frozen_geometry_exactly_preserved"]
        ),
        "common_parameter_difference": parameter_difference(
            args.external_checkpoint, args.model_buffer_checkpoint
        ),
        "interpretation_note": (
            "The two policies are optimizer-equivalent for identical gradients; "
            "separate full FastGS CUDA runs are not bitwise deterministic."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "geometry_freeze_implementation_comparison.json"
    png_path = args.output_dir / "geometry_freeze_implementation_comparison.png"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    save_table(result, png_path)
    print(json.dumps(result, indent=2))
    return json_path, png_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-run", type=Path, default=EXTERNAL_OUTPUT)
    parser.add_argument("--model-buffer-run", type=Path, default=MODEL_BUFFER_OUTPUT)
    parser.add_argument(
        "--external-checkpoint",
        type=Path,
        default=EXTERNAL_OUTPUT / "state1_complete_checkpoint.pt",
    )
    parser.add_argument(
        "--model-buffer-checkpoint",
        type=Path,
        default=MODEL_BUFFER_OUTPUT / "state1_complete_checkpoint.pt",
    )
    parser.add_argument(
        "--external-artifact",
        type=Path,
        default=EXTERNAL_OUTPUT / "signed_influence/state0_signed_influence.pt",
    )
    parser.add_argument(
        "--model-buffer-artifact",
        type=Path,
        default=MODEL_BUFFER_OUTPUT / "signed_influence/state0_signed_influence.pt",
    )
    parser.add_argument("--output-dir", type=Path, default=MODEL_BUFFER_OUTPUT)
    return parser.parse_args()


def main() -> None:
    compare(parse_args())


if __name__ == "__main__":
    main()
