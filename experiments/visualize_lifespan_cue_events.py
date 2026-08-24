"""Visualize causal R_change, cached cues, and exact-timestamp lifespan events.

The event panel is intentionally sparse: only Gaussian rows whose lifecycle
decision is OPEN or CLOSE at the current global image timestamp are rendered.
OPEN is green, CLOSE is red, and KEEP/NONE rows are omitted.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


OPEN_COLOR = (0, 255, 0)
CLOSE_COLOR = (255, 0, 0)


def _font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def load_transition_rows(
    path: Path,
) -> dict[int, dict[str, tuple[int, ...]]]:
    """Group only OPEN/CLOSE Gaussian rows by exact decision timestamp."""
    grouped: dict[int, dict[str, list[int]]] = defaultdict(
        lambda: {"OPEN": [], "CLOSE": []}
    )
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            event = json.loads(line)
            action = str(event.get("action"))
            if action not in {"OPEN", "CLOSE"}:
                continue
            timestamp = int(event["decision_timestamp"])
            grouped[timestamp][action].append(int(event["gaussian_index"]))
    return {
        timestamp: {
            action: tuple(sorted(set(rows)))
            for action, rows in action_rows.items()
        }
        for timestamp, action_rows in grouped.items()
    }


def event_probe_tensors(
    gaussian_count: int,
    open_rows: Sequence[int],
    close_rows: Sequence[int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-row RGB colors and a boolean event mask."""
    if gaussian_count < 1:
        raise ValueError("gaussian_count must be positive")
    open_tensor = torch.as_tensor(open_rows, device=device, dtype=torch.long).flatten()
    close_tensor = torch.as_tensor(close_rows, device=device, dtype=torch.long).flatten()
    for name, rows in (("open_rows", open_tensor), ("close_rows", close_tensor)):
        if rows.numel() and bool(((rows < 0) | (rows >= gaussian_count)).any()):
            raise IndexError(f"{name} contains an out-of-range Gaussian index")
    if open_tensor.numel() and close_tensor.numel():
        overlap = torch.isin(open_tensor, close_tensor)
        if bool(overlap.any()):
            raise ValueError("one Gaussian cannot OPEN and CLOSE at the same timestamp")
    colors = torch.zeros((gaussian_count, 3), device=device, dtype=dtype)
    selected = torch.zeros(gaussian_count, device=device, dtype=torch.bool)
    if open_tensor.numel():
        colors[open_tensor, 1] = 1.0
        selected[open_tensor] = True
    if close_tensor.numel():
        colors[close_tensor, 0] = 1.0
        selected[close_tensor] = True
    return colors, selected


def turbo_heatmap(cue: np.ndarray) -> np.ndarray:
    cue_u8 = np.round(np.clip(cue, 0.0, 1.0) * 255.0).astype(np.uint8)
    bgr = cv2.applyColorMap(cue_u8, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def raw_render_grayscale(raw: np.ndarray) -> np.ndarray:
    array = raw.astype(np.float32, copy=False)
    if array.ndim == 3:
        array = array.mean(axis=2)
    gray = np.clip(array, 0.0, 255.0).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def resize_rgb(array: np.ndarray, width: int) -> Image.Image:
    image = Image.fromarray(array.astype(np.uint8, copy=False)).convert("RGB")
    height = max(1, int(round(image.height * int(width) / image.width)))
    return image.resize((int(width), height), Image.Resampling.BILINEAR)


def labeled_panel(array: np.ndarray, title: str, width: int) -> Image.Image:
    image = resize_rgb(array, width)
    header = 38
    canvas = Image.new("RGB", (image.width, image.height + header), "white")
    ImageDraw.Draw(canvas).text((7, 7), title, fill="black", font=_font(17))
    canvas.paste(image, (0, header))
    return canvas


def compose_frame_panel(
    rgb: np.ndarray,
    cue: np.ndarray,
    raw_render: np.ndarray,
    event_render: np.ndarray,
    *,
    timestamp: int,
    segment: str,
    open_count: int,
    close_count: int,
    panel_width: int,
) -> Image.Image:
    panels = (
        labeled_panel(rgb, "Inference RGB", panel_width),
        labeled_panel(turbo_heatmap(cue), "Combined change cue", panel_width),
        labeled_panel(raw_render_grayscale(raw_render), "Causal raw R_change", panel_width),
        labeled_panel(event_render, "Lifespan events @ current t", panel_width),
    )
    gap = 8
    body_width = sum(panel.width for panel in panels) + gap * (len(panels) - 1)
    body_height = max(panel.height for panel in panels)
    header_height = 58
    canvas = Image.new("RGB", (body_width, header_height + body_height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (8, 7),
        f"global t={timestamp:03d} | {segment}",
        fill="black",
        font=_font(22),
    )
    legend_x = 470
    draw.rectangle((legend_x, 13, legend_x + 18, 31), fill=OPEN_COLOR)
    draw.text(
        (legend_x + 25, 8),
        f"OPEN@t {open_count:,}",
        fill="black",
        font=_font(18),
    )
    close_x = legend_x + 205
    draw.rectangle((close_x, 13, close_x + 18, 31), fill=CLOSE_COLOR)
    draw.text(
        (close_x + 25, 8),
        f"CLOSE@t {close_count:,}",
        fill="black",
        font=_font(18),
    )
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, header_height))
        x += panel.width + gap
    return canvas


def save_gif(
    paths: Sequence[Path],
    output_path: Path,
    *,
    width: int,
    duration_ms: int,
) -> None:
    if not paths:
        raise ValueError("cannot make a GIF without frames")
    frames: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as image:
            height = max(1, int(round(image.height * width / image.width)))
            resized = image.convert("RGB").resize(
                (width, height), Image.Resampling.BILINEAR
            )
        frames.append(resized.quantize(colors=128, method=Image.Quantize.MEDIANCUT))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(duration_ms),
        loop=0,
        disposal=2,
        optimize=False,
    )


def render_event_rows(
    view: Any,
    base: Any,
    pipe: Any,
    background: torch.Tensor,
    open_rows: Sequence[int],
    close_rows: Sequence[int],
) -> np.ndarray:
    """Render immutable-base footprints for only current-timestamp events."""
    from gaussian_renderer import render_change

    colors, selected = event_probe_tensors(
        int(base.get_xyz.shape[0]),
        open_rows,
        close_rows,
        device=base.get_xyz.device,
        dtype=base._features_dc.dtype,
    )
    height, width = int(view.image_height), int(view.image_width)
    if not bool(selected.any()):
        return np.zeros((height, width, 3), dtype=np.uint8)
    package = render_change(
        view,
        base,
        pipe,
        background,
        override_color=colors,
        override_opacity=base.get_opacity.detach() * selected[:, None],
        override_xyz=base.get_xyz.detach(),
        override_scaling=base.get_scaling.detach(),
        override_rotation=base.get_rotation.detach(),
        clamp_output=True,
    )
    return (
        package["render"]
        .detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--raw-render-dir", type=Path, required=True)
    parser.add_argument("--source-path", type=Path, default=None)
    parser.add_argument("--fixed-cameras-json", type=Path, default=None)
    parser.add_argument("--cue-cache-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--panel-width", type=int, default=300)
    parser.add_argument("--gif-width", type=int, default=1200)
    parser.add_argument("--gif-duration-ms", type=int, default=160)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the Gaussian renderer")
    summary = json.loads((args.run_dir / "summary.json").read_text())
    run_arguments: Mapping[str, Any] = summary.get("run_arguments", {})
    source_path = args.source_path or Path(run_arguments["source_path"])
    fixed_cameras = args.fixed_cameras_json or Path(run_arguments["fixed_cameras_json"])
    cue_cache_root = args.cue_cache_root or Path(run_arguments["cue_cache_root"])
    resolution = float(args.resolution or run_arguments.get("resolution", 4.0))

    from experiments.run_online_bayesian_lifespan_thaw import build_causal_records
    from experiments.train_cue_temporal_rchange import (
        build_fixed_cue_views,
        load_fixed_camera_index,
    )
    from scene import GaussianModel

    records, _names = build_causal_records(source_path, max_frames=args.max_frames)
    cameras = load_fixed_camera_index(fixed_cameras)
    base_ply = (source_path / "reference_reconstruction/point_cloud/iteration_30000/point_cloud.ply").resolve()
    base = GaussianModel(sh_degree=3, active_sh_degree=0)
    base.load_ply_change(str(base_ply))
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
    background = torch.zeros(3, device=base.get_xyz.device, dtype=base.get_xyz.dtype)
    events = load_transition_rows(args.run_dir / "lifecycle_events.jsonl")

    panels_dir = args.output_dir / "panels"
    event_dir = args.output_dir / "event_render"
    panels_dir.mkdir(parents=True, exist_ok=True)
    event_dir.mkdir(parents=True, exist_ok=True)
    panel_paths: list[Path] = []
    by_segment: dict[str, list[Path]] = defaultdict(list)
    manifest_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for record in records:
            timestamp = int(record.global_index)
            view = build_fixed_cue_views(
                [record], cameras, cue_cache_root, resolution
            )[0][0]
            action_rows = events.get(timestamp, {"OPEN": (), "CLOSE": ()})
            event_rgb = render_event_rows(
                view,
                base,
                pipe,
                background,
                action_rows["OPEN"],
                action_rows["CLOSE"],
            )
            stem = Path(record.name).stem
            raw_path = args.raw_render_dir / f"{stem}.png"
            if not raw_path.exists():
                raise FileNotFoundError(raw_path)
            with Image.open(record.image_path) as image:
                rgb = np.asarray(
                    image.convert("RGB").resize(
                        (int(view.image_width), int(view.image_height)),
                        Image.Resampling.BILINEAR,
                    )
                )
            cue = view.candidate_map.detach().float().squeeze().cpu().numpy()
            with Image.open(raw_path) as image:
                raw = np.asarray(image.convert("RGB"))
            panel = compose_frame_panel(
                rgb,
                cue,
                raw,
                event_rgb,
                timestamp=timestamp,
                segment=str(record.segment_name),
                open_count=len(action_rows["OPEN"]),
                close_count=len(action_rows["CLOSE"]),
                panel_width=int(args.panel_width),
            )
            event_path = event_dir / f"{timestamp:06d}_{stem}.png"
            panel_path = panels_dir / f"{timestamp:06d}_{stem}.png"
            Image.fromarray(event_rgb).save(event_path)
            panel.save(panel_path)
            panel_paths.append(panel_path)
            by_segment[str(record.segment_name)].append(panel_path)
            manifest_rows.append(
                {
                    "timestamp": timestamp,
                    "segment": str(record.segment_name),
                    "frame": record.name,
                    "open_count": len(action_rows["OPEN"]),
                    "close_count": len(action_rows["CLOSE"]),
                    "panel": str(panel_path),
                    "event_render": str(event_path),
                    "raw_render": str(raw_path),
                }
            )

    continuous_gif = args.output_dir / "ref_sc1_sc2_sc3_lifespan_events.gif"
    save_gif(
        panel_paths,
        continuous_gif,
        width=int(args.gif_width),
        duration_ms=int(args.gif_duration_ms),
    )
    segment_gifs: dict[str, str] = {}
    for segment, paths in by_segment.items():
        path = args.output_dir / segment / f"{segment}_lifespan_events.gif"
        save_gif(
            paths,
            path,
            width=int(args.gif_width),
            duration_ms=int(args.gif_duration_ms),
        )
        segment_gifs[segment] = str(path)

    payload = {
        "schema_version": 1,
        "contract": "exact-timestamp OPEN/CLOSE event-only immutable-base Gaussian render",
        "run_dir": str(args.run_dir),
        "raw_render_dir": str(args.raw_render_dir),
        "frames": len(panel_paths),
        "continuous_gif": str(continuous_gif),
        "segment_gifs": segment_gifs,
        "event_colors": {"OPEN": "green", "CLOSE": "red"},
        "keep_rows_rendered": False,
        "event_rows_persist_across_frames": False,
        "runtime_seconds": time.time() - started,
        "frames_manifest": manifest_rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: payload[key] for key in ("frames", "continuous_gif", "segment_gifs", "runtime_seconds")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
