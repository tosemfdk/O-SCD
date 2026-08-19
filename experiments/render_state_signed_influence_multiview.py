"""Render offline influence-valid Gaussians over one state's fixed views."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from experiments.export_state_signed_influence import (
    DEFAULT_RUN_DIR,
    _records_through_state_end,
)
from experiments.train_cue_temporal_rchange import (
    build_fixed_cue_views,
    load_fixed_camera_index,
)
from experiments.train_real_temporal_rchange import file_checksum
from experiments.visualize_temporal_state_switch import load_font, resize_panel
from gaussian_renderer import render_change
from temporal import forced_state_render_attributes, load_temporal_model


PALETTE = {
    "additive": (255, 51, 13),
    "occluding": (13, 115, 255),
    "both": (217, 26, 217),
}


def tensor_image(value: torch.Tensor) -> Image.Image:
    array = (
        value.detach()
        .float()
        .clamp(0, 1)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray((array * 255).astype(np.uint8))


def add_label(image: Image.Image, text: str, width: int) -> Image.Image:
    font = load_font(16)
    resized = resize_panel(image.convert("RGB"), width)
    output = Image.new("RGB", (resized.width, resized.height + 32), "white")
    output.paste(resized, (0, 32))
    ImageDraw.Draw(output).text((7, 8), text, fill=(0, 0, 0), font=font)
    return output


def horizontal(images: list[Image.Image], gap: int = 8) -> Image.Image:
    width = sum(image.width for image in images) + gap * (len(images) - 1)
    height = max(image.height for image in images)
    output = Image.new("RGB", (width, height), "white")
    x = 0
    for image in images:
        output.paste(image, (x, 0))
        x += image.width + gap
    return output


def role_colors(
    additive: torch.Tensor,
    occluding: torch.Tensor,
) -> torch.Tensor:
    both = additive & occluding
    colors = torch.zeros(
        (additive.shape[0], 3),
        dtype=torch.float32,
        device=additive.device,
    )
    colors[additive & ~occluding] = torch.tensor(
        [1.0, 0.20, 0.05], device=additive.device
    )
    colors[occluding & ~additive] = torch.tensor(
        [0.05, 0.45, 1.0], device=additive.device
    )
    colors[both] = torch.tensor([0.85, 0.10, 0.85], device=additive.device)
    return colors


def render_mask(
    view,
    model,
    attributes: dict[str, torch.Tensor],
    colors: torch.Tensor,
    selected: torch.Tensor,
    pipe: SimpleNamespace,
    background: torch.Tensor,
) -> torch.Tensor:
    return render_change(
        view,
        model.base,
        pipe,
        background,
        override_color=colors,
        override_opacity=attributes["opacity"] * selected[:, None],
        override_xyz=attributes["xyz"],
        override_scaling=attributes["scaling"],
        override_rotation=attributes["rotation"],
    )["render"]


def make_panel(
    view,
    all_valid: torch.Tensor,
    additive: torch.Tensor,
    occluding: torch.Tensor,
    timestamp: int,
    panel_width: int,
    counts: dict[str, int],
) -> Image.Image:
    panels = [
        add_label(tensor_image(view.original_image[:3]), "RGB view", panel_width),
        add_label(
            tensor_image(all_valid),
            "All influence-valid GS",
            panel_width,
        ),
        add_label(tensor_image(additive), "Additive GS", panel_width),
        add_label(tensor_image(occluding), "Occluding GS", panel_width),
    ]
    row = horizontal(panels)
    header = Image.new("RGB", (row.width, 70), "white")
    draw = ImageDraw.Draw(header)
    draw.text(
        (8, 7),
        f"State 0 signed-influence valid GS | fixed view t={timestamp}",
        fill=(0, 0, 0),
        font=load_font(18),
    )
    draw.text(
        (8, 39),
        "orange=additive  blue=occluding  magenta=both  |  "
        f"valid={counts['valid']:,} additive={counts['additive']:,} "
        f"occluding={counts['occluding']:,} both={counts['both']:,}",
        fill=(25, 25, 25),
        font=load_font(14),
    )
    output = Image.new("RGB", (row.width, header.height + row.height), "white")
    output.paste(header, (0, 0))
    output.paste(row, (0, header.height))
    return output


def save_gif(
    frame_paths: list[Path],
    path: Path,
    duration_ms: int,
) -> None:
    frames = []
    for frame_path in frame_paths:
        with Image.open(frame_path) as image:
            frame = image.convert("RGB").quantize(
                colors=192,
                method=Image.Quantize.MEDIANCUT,
            )
        frames.append(frame)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
        optimize=False,
    )


def save_contact_sheet(
    frame_paths: list[Path],
    path: Path,
    samples: int,
) -> None:
    indices = np.linspace(0, len(frame_paths) - 1, min(samples, len(frame_paths)))
    selected = [frame_paths[index] for index in sorted(set(indices.round().astype(int)))]
    thumbs = []
    for frame_path in selected:
        with Image.open(frame_path) as image:
            thumbs.append(resize_panel(image.convert("RGB"), 720))
    rows = [horizontal(thumbs[start : start + 2]) for start in range(0, len(thumbs), 2)]
    width = max(row.width for row in rows)
    height = sum(row.height for row in rows) + 8 * (len(rows) - 1)
    output = Image.new("RGB", (width, height), "white")
    y = 0
    for row in rows:
        output.paste(row, (0, y))
        y += row.height + 8
    output.save(path)


def render_multiview(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the FastGS renderer")
    started = time.time()
    checkpoint = Path(args.checkpoint).resolve()
    summary_path = Path(args.summary).resolve()
    artifact_path = Path(args.artifact).resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    if artifact.get("contract") != "offline_signed_opacity_removal_influence_fixed_topology":
        raise ValueError("Unsupported signed-influence artifact contract")
    if artifact.get("checkpoint_sha256") != file_checksum(checkpoint):
        raise ValueError("Artifact and temporal checkpoint do not match")

    state = int(artifact["state"])
    end_timestamp = int(artifact["end_timestamp"])
    records = _records_through_state_end(summary, state, end_timestamp)
    records = records[:: int(args.stride)]
    if args.max_views is not None:
        records = records[: int(args.max_views)]
    cameras = load_fixed_camera_index(Path(summary["fixed_cameras_json"]))
    views, _, _ = build_fixed_cue_views(
        records,
        cameras,
        Path(summary["cue_cache_root"]),
        float(summary["resolution"]),
    )

    model = load_temporal_model(checkpoint)
    attributes = {
        name: value.detach()
        for name, value in forced_state_render_attributes(model, state).items()
    }
    masks = {
        name: artifact[f"{name}_mask"].cuda(non_blocking=True)
        for name in ("valid", "additive", "occluding", "both")
    }
    colors = role_colors(masks["additive"], masks["occluding"])
    counts = {name: int(mask.sum().item()) for name, mask in masks.items()}
    pipe = SimpleNamespace(
        compute_cov3D_python=False,
        convert_SHs_python=False,
        debug=False,
    )
    background = torch.zeros(3, dtype=torch.float32, device="cuda")

    output_dir = Path(args.output_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: list[Path] = []
    frame_manifest: list[dict[str, Any]] = []
    with torch.no_grad():
        for index, view in enumerate(views, start=1):
            all_valid = render_mask(
                view,
                model,
                attributes,
                colors,
                masks["valid"],
                pipe,
                background,
            )
            additive = render_mask(
                view,
                model,
                attributes,
                colors,
                masks["additive"],
                pipe,
                background,
            )
            occluding = render_mask(
                view,
                model,
                attributes,
                colors,
                masks["occluding"],
                pipe,
                background,
            )
            timestamp = int(view.timestamp)
            panel = make_panel(
                view,
                all_valid,
                additive,
                occluding,
                timestamp,
                int(args.panel_width),
                counts,
            )
            frame_path = frames_dir / f"{timestamp:06d}_{view.image_name}.png"
            panel.save(frame_path)
            frame_paths.append(frame_path)
            frame_manifest.append(
                {
                    "timestamp": timestamp,
                    "frame": view.image_name,
                    "panel": str(frame_path),
                }
            )
            if index == 1 or index == len(views) or index % 10 == 0:
                print(
                    f"[influence-multiview] {index}/{len(views)} "
                    f"t={timestamp} elapsed={time.time() - started:.1f}s",
                    flush=True,
                )

    gif_path = output_dir / f"state{state}_influence_valid_multiview.gif"
    contact_path = output_dir / f"state{state}_influence_valid_contact_sheet.png"
    manifest_path = output_dir / f"state{state}_influence_valid_multiview.json"
    save_gif(frame_paths, gif_path, int(args.duration_ms))
    save_contact_sheet(frame_paths, contact_path, int(args.contact_samples))
    manifest = {
        "script": "experiments/render_state_signed_influence_multiview.py",
        "contract": "offline_signed_influence_valid_fixed_view_render",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_checksum(checkpoint),
        "influence_artifact": str(artifact_path),
        "influence_artifact_sha256": file_checksum(artifact_path),
        "state": state,
        "view_count": len(views),
        "timestamps": [int(view.timestamp) for view in views],
        "stride": int(args.stride),
        "counts": counts,
        "thresholds": {
            "mask_threshold": artifact["mask_threshold"],
            "mask_temperature": artifact["mask_temperature"],
            "min_per_view_influence": artifact["min_per_view_influence"],
            "min_mean_influence": artifact["min_mean_influence"],
            "min_views": artifact["min_views"],
        },
        "palette": PALETTE,
        "frames": frame_manifest,
        "gif": str(gif_path),
        "contact_sheet": str(contact_path),
        "duration_ms": int(args.duration_ms),
        "runtime_seconds": time.time() - started,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return gif_path, contact_path, manifest_path


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_RUN_DIR / "temporal_rchange_checkpoint.pt",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_RUN_DIR / "summary.json",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=DEFAULT_RUN_DIR / "signed_influence/state0_signed_influence.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RUN_DIR / "signed_influence/state0_multiview_valid_render",
    )
    parser.add_argument("--stride", type=positive_int, default=1)
    parser.add_argument("--max-views", type=positive_int, default=None)
    parser.add_argument("--panel-width", type=positive_int, default=260)
    parser.add_argument("--duration-ms", type=positive_int, default=160)
    parser.add_argument("--contact-samples", type=positive_int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gif_path, contact_path, manifest_path = render_multiview(args)
    print(
        json.dumps(
            {
                "gif": str(gif_path),
                "contact_sheet": str(contact_path),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
