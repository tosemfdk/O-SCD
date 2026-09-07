"""Visualize causal R_change, cached cues, and Gaussian lifespan state.

The lifecycle panel keeps every Gaussian that has entered a lifespan visible.
Committed OPEN rows remain half-bright green and committed CLOSED rows remain
half-bright red.  Rows whose lifecycle decision changes at the current image
timestamp are highlighted at full brightness.  NEVER_OPEN rows are omitted.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
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
STATE_INTENSITY = 0.5
OPEN_STATE_COLOR = (0, 128, 0)
CLOSED_STATE_COLOR = (128, 0, 0)


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
    open_state_rows: Sequence[int] = (),
    closed_state_rows: Sequence[int] = (),
    state_intensity: float = STATE_INTENSITY,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return lifecycle colors and a mask for state/event Gaussian rows.

    State rows receive half-bright colors by default.  Exact-timestamp OPEN or
    CLOSE event colors are then written at full brightness, so a current event
    always overrides its persistent state color.
    """
    if gaussian_count < 1:
        raise ValueError("gaussian_count must be positive")
    if not math.isfinite(float(state_intensity)) or not 0.0 <= float(
        state_intensity
    ) <= 1.0:
        raise ValueError("state_intensity must be finite and in [0,1]")
    open_tensor = torch.as_tensor(open_rows, device=device, dtype=torch.long).flatten()
    close_tensor = torch.as_tensor(close_rows, device=device, dtype=torch.long).flatten()
    open_state_tensor = torch.as_tensor(
        open_state_rows, device=device, dtype=torch.long
    ).flatten()
    closed_state_tensor = torch.as_tensor(
        closed_state_rows, device=device, dtype=torch.long
    ).flatten()
    for name, rows in (
        ("open_rows", open_tensor),
        ("close_rows", close_tensor),
        ("open_state_rows", open_state_tensor),
        ("closed_state_rows", closed_state_tensor),
    ):
        if rows.numel() and bool(((rows < 0) | (rows >= gaussian_count)).any()):
            raise IndexError(f"{name} contains an out-of-range Gaussian index")
    if open_tensor.numel() and close_tensor.numel():
        overlap = torch.isin(open_tensor, close_tensor)
        if bool(overlap.any()):
            raise ValueError("one Gaussian cannot OPEN and CLOSE at the same timestamp")
    if open_state_tensor.numel() and closed_state_tensor.numel():
        overlap = torch.isin(open_state_tensor, closed_state_tensor)
        if bool(overlap.any()):
            raise ValueError("one Gaussian cannot be OPEN and CLOSED simultaneously")
    colors = torch.zeros((gaussian_count, 3), device=device, dtype=dtype)
    selected = torch.zeros(gaussian_count, device=device, dtype=torch.bool)
    if open_state_tensor.numel():
        colors[open_state_tensor, 1] = float(state_intensity)
        selected[open_state_tensor] = True
    if closed_state_tensor.numel():
        colors[closed_state_tensor, 0] = float(state_intensity)
        selected[closed_state_tensor] = True
    if open_tensor.numel():
        colors[open_tensor] = 0.0
        colors[open_tensor, 1] = 1.0
        selected[open_tensor] = True
    if close_tensor.numel():
        colors[close_tensor] = 0.0
        colors[close_tensor, 0] = 1.0
        selected[close_tensor] = True
    return colors, selected


def advance_lifecycle_state(
    open_state_rows: set[int],
    closed_state_rows: set[int],
    *,
    open_rows: Sequence[int],
    close_rows: Sequence[int],
) -> tuple[set[int], set[int]]:
    """Apply exact-timestamp events and return the current committed states."""

    opens = {int(row) for row in open_rows}
    closes = {int(row) for row in close_rows}
    if opens & closes:
        raise ValueError("one Gaussian cannot OPEN and CLOSE at the same timestamp")
    next_open = set(open_state_rows)
    next_closed = set(closed_state_rows)
    next_open.difference_update(closes)
    next_closed.update(closes)
    next_closed.difference_update(opens)
    next_open.update(opens)
    if next_open & next_closed:
        raise RuntimeError("lifecycle state sets must remain disjoint")
    return next_open, next_closed


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


def threshold_render_rgb(
    raw: np.ndarray, threshold: float = 0.5
) -> np.ndarray:
    """Threshold the channel-mean R_change render into a white binary mask."""

    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be in [0,1]")
    array = np.asarray(raw)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("raw render must have shape [H,W,3]")
    score = array.astype(np.float32).mean(axis=2)
    if np.issubdtype(array.dtype, np.integer) or (
        score.size and float(score.max()) > 1.0
    ):
        score = score / 255.0
    mask = score >= float(threshold)
    return np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=2)


def load_evaluation_mask_rgb(
    path: Path | None,
    *,
    raw_render: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, str]:
    """Load the exact saved metric mask, falling back to raw-render thresholding."""

    if path is None:
        return threshold_render_rgb(raw_render, threshold=threshold), "recomputed"
    if not path.exists():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L")) >= 128
    if mask.shape != raw_render.shape[:2]:
        raise ValueError(
            "saved evaluation mask and raw R_change render must have matching shapes"
        )
    return (
        np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=2),
        "saved_metric_prediction",
    )


def _nice_axis(maximum: int, target_ticks: int = 5) -> tuple[int, int]:
    maximum = max(1, int(maximum))
    raw_step = maximum / max(1, int(target_ticks))
    exponent = 10 ** math.floor(math.log10(raw_step))
    fraction = raw_step / exponent
    if fraction < 1.5:
        nice_fraction = 1
    elif fraction < 3:
        nice_fraction = 2
    elif fraction < 7:
        nice_fraction = 5
    else:
        nice_fraction = 10
    step = max(1, int(nice_fraction * exponent))
    limit = int(math.ceil(maximum / step) * step)
    return limit, step


def open_close_timeline_chart(
    open_counts: Sequence[int],
    close_counts: Sequence[int],
    *,
    current_timestamp: int,
    width: int,
    height: int = 280,
    boundaries: Sequence[int] = (),
    segment_names: Sequence[str] = (),
) -> Image.Image:
    """Draw causal per-frame OPEN/CLOSE counts on one absolute-count axis."""

    opens = [max(0, int(value)) for value in open_counts]
    closes = [max(0, int(value)) for value in close_counts]
    if not opens or len(opens) != len(closes):
        raise ValueError("OPEN/CLOSE count series must be nonempty and aligned")
    if current_timestamp < 0 or current_timestamp >= len(opens):
        raise IndexError("current timestamp is outside the lifecycle timeline")
    if width < 320 or height < 180:
        raise ValueError("timeline chart is too small")

    left, right, top, bottom = 104, 24, 54, 48
    plot_w = width - left - right
    plot_h = height - top - bottom
    canvas = Image.new("RGB", (int(width), int(height)), "white")
    draw = ImageDraw.Draw(canvas)
    small = _font(14)
    title_font = _font(18)
    n = len(opens)
    boundary_values = sorted({int(value) for value in boundaries if 0 < value < n})
    spans = [0, *boundary_values, n]
    shades = ((239, 247, 253), (247, 251, 232), (255, 248, 230))
    for index, (start, end) in enumerate(zip(spans, spans[1:])):
        x0 = left + int(start * plot_w / max(1, n - 1))
        x1 = left + int(min(end, n - 1) * plot_w / max(1, n - 1))
        draw.rectangle((x0, top, x1, top + plot_h), fill=shades[index % len(shades)])
        if index < len(segment_names):
            label = str(segment_names[index])
            box = draw.textbbox((0, 0), label, font=small)
            draw.text(
                ((x0 + x1 - (box[2] - box[0])) // 2, top + 3),
                label,
                fill=(70, 78, 92),
                font=small,
            )

    y_limit, y_step = _nice_axis(max(max(opens), max(closes)))
    for value in range(0, y_limit + 1, y_step):
        y = top + plot_h - int(value * plot_h / y_limit)
        draw.line((left, y, left + plot_w, y), fill=(213, 220, 226), width=1)
        label = f"{value:,}"
        box = draw.textbbox((0, 0), label, font=small)
        draw.text(
            (left - 10 - (box[2] - box[0]), y - 7),
            label,
            fill=(40, 40, 40),
            font=small,
        )
    for boundary in boundary_values:
        x = left + int(boundary * plot_w / max(1, n - 1))
        draw.line((x, top, x, top + plot_h), fill=(90, 90, 90), width=2)

    x_tick_step = 50 if n > 150 else 20
    x_ticks = list(range(0, n, x_tick_step))
    if n - 1 - x_ticks[-1] >= max(8, x_tick_step // 2):
        x_ticks.append(n - 1)
    for value in x_ticks:
        x = left + int(value * plot_w / max(1, n - 1))
        draw.line((x, top + plot_h, x, top + plot_h + 5), fill="black", width=1)
        label = str(value)
        box = draw.textbbox((0, 0), label, font=small)
        draw.text((x - (box[2] - box[0]) // 2, top + plot_h + 7), label, fill="black", font=small)

    def points(values: Sequence[int]) -> list[tuple[int, int]]:
        return [
            (
                left + int(index * plot_w / max(1, n - 1)),
                top + plot_h - int(int(value) * plot_h / y_limit),
            )
            for index, value in enumerate(values[: current_timestamp + 1])
        ]

    open_color = (18, 163, 81)
    close_color = (232, 45, 71)
    for values, color in ((opens, open_color), (closes, close_color)):
        line = points(values)
        if len(line) > 1:
            draw.line(line, fill=color, width=3)
        x, y = line[-1]
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)

    current_x = left + int(current_timestamp * plot_w / max(1, n - 1))
    draw.line((current_x, top, current_x, top + plot_h), fill=(34, 80, 190), width=2)
    draw.text((left, 12), "Lifecycle mutations by image timestamp", fill="black", font=title_font)
    legend_x = max(left + 420, width - 520)
    draw.line((legend_x, 25, legend_x + 28, 25), fill=open_color, width=4)
    draw.text(
        (legend_x + 36, 15),
        f"OPEN@t {opens[current_timestamp]:,}",
        fill="black",
        font=small,
    )
    close_x = legend_x + 190
    draw.line((close_x, 25, close_x + 28, 25), fill=close_color, width=4)
    draw.text(
        (close_x + 36, 15),
        f"CLOSE@t {closes[current_timestamp]:,}",
        fill="black",
        font=small,
    )
    current_label = f"t={current_timestamp}"
    current_box = draw.textbbox((0, 0), current_label, font=small)
    current_label_width = current_box[2] - current_box[0]
    current_label_x = min(
        current_x + 5,
        left + plot_w - current_label_width - 4,
    )
    draw.text(
        (current_label_x, top + 20),
        current_label,
        fill=(34, 80, 190),
        font=small,
    )
    x_label = "Image timestamp"
    box = draw.textbbox((0, 0), x_label, font=small)
    draw.text(
        (left + (plot_w - (box[2] - box[0])) // 2, height - 20),
        x_label,
        fill="black",
        font=small,
    )
    y_label = Image.new("RGBA", (160, 24), (255, 255, 255, 0))
    ImageDraw.Draw(y_label).text((0, 2), "Gaussian count", fill="black", font=small)
    y_label = y_label.rotate(90, expand=True)
    canvas.paste(y_label, (2, top + max(0, (plot_h - y_label.height) // 2)), y_label)
    draw.line((left, top, left, top + plot_h), fill="black", width=2)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill="black", width=2)
    return canvas


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
    open_state_count: int | None = None,
    closed_state_count: int | None = None,
    thresholded_render: np.ndarray | None = None,
    lifecycle_chart: Image.Image | None = None,
) -> Image.Image:
    panel_items = [
        labeled_panel(rgb, "Inference RGB", panel_width),
        labeled_panel(turbo_heatmap(cue), "Combined change cue", panel_width),
        labeled_panel(raw_render_grayscale(raw_render), "Causal raw R_change", panel_width),
    ]
    if thresholded_render is not None:
        panel_items.append(
            labeled_panel(
                thresholded_render,
                "R_change mask (>= 0.5)",
                panel_width,
            )
        )
    panel_items.append(
        labeled_panel(
            event_render,
            "Lifespan state + events",
            panel_width,
        )
    )
    panels = tuple(panel_items)
    gap = 8
    body_width = sum(panel.width for panel in panels) + gap * (len(panels) - 1)
    body_height = max(panel.height for panel in panels)
    show_state_legend = (
        open_state_count is not None and closed_state_count is not None
    )
    header_height = 82 if show_state_legend else 58
    chart_gap = 8 if lifecycle_chart is not None else 0
    chart_height = lifecycle_chart.height if lifecycle_chart is not None else 0
    canvas = Image.new(
        "RGB",
        (body_width, header_height + body_height + chart_gap + chart_height),
        "white",
    )
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
    if show_state_legend:
        assert open_state_count is not None and closed_state_count is not None
        state_y = 43
        draw.rectangle(
            (legend_x, state_y, legend_x + 18, state_y + 18),
            fill=OPEN_STATE_COLOR,
        )
        draw.text(
            (legend_x + 25, state_y - 5),
            f"OPEN state {open_state_count:,}",
            fill="black",
            font=_font(16),
        )
        draw.rectangle(
            (close_x, state_y, close_x + 18, state_y + 18),
            fill=CLOSED_STATE_COLOR,
        )
        draw.text(
            (close_x + 25, state_y - 5),
            f"CLOSED state {closed_state_count:,}",
            fill="black",
            font=_font(16),
        )
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, header_height))
        x += panel.width + gap
    if lifecycle_chart is not None:
        chart = lifecycle_chart.convert("RGB")
        if chart.width != body_width:
            chart = chart.resize(
                (body_width, chart.height), Image.Resampling.BILINEAR
            )
        canvas.paste(chart, (0, header_height + body_height + chart_gap))
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
    *,
    open_state_rows: Sequence[int] = (),
    closed_state_rows: Sequence[int] = (),
) -> np.ndarray:
    """Render immutable-base footprints for lifecycle states and events."""
    from gaussian_renderer import render_change

    colors, selected = event_probe_tensors(
        int(base.get_xyz.shape[0]),
        open_rows,
        close_rows,
        device=base.get_xyz.device,
        dtype=base._features_dc.dtype,
        open_state_rows=open_state_rows,
        closed_state_rows=closed_state_rows,
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
    parser.add_argument("--raw-render-dir", type=Path, default=None)
    parser.add_argument("--thresholded-render-dir", type=Path, default=None)
    parser.add_argument("--captured-event-render-dir", type=Path, default=None)
    parser.add_argument("--evaluation-threshold", type=float, default=None)
    parser.add_argument("--source-path", type=Path, default=None)
    parser.add_argument("--fixed-cameras-json", type=Path, default=None)
    parser.add_argument("--cue-cache-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--panel-width", type=int, default=300)
    parser.add_argument("--gif-width", type=int, default=1200)
    parser.add_argument("--gif-duration-ms", type=int, default=160)
    parser.add_argument("--timeline-height", type=int, default=280)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the Gaussian renderer")
    summary = json.loads((args.run_dir / "summary.json").read_text())
    run_arguments: Mapping[str, Any] = summary.get("run_arguments", {})
    visual_capture: Mapping[str, Any] = summary.get("visual_capture") or {}
    source_path = args.source_path or Path(run_arguments["source_path"])
    fixed_cameras = args.fixed_cameras_json or Path(run_arguments["fixed_cameras_json"])
    cue_cache_root = args.cue_cache_root or Path(run_arguments["cue_cache_root"])
    resolution = float(args.resolution or run_arguments.get("resolution", 4.0))
    raw_render_dir = args.raw_render_dir
    if raw_render_dir is None and visual_capture.get("raw_render"):
        raw_render_dir = Path(str(visual_capture["raw_render"]))
    if raw_render_dir is None:
        raise ValueError(
            "--raw-render-dir is required when the run summary has no visual capture"
        )
    thresholded_render_dir = args.thresholded_render_dir
    if thresholded_render_dir is None and visual_capture.get("thresholded_render"):
        thresholded_render_dir = Path(str(visual_capture["thresholded_render"]))
    captured_event_render_dir = args.captured_event_render_dir
    if captured_event_render_dir is None and visual_capture.get("event_render"):
        captured_event_render_dir = Path(str(visual_capture["event_render"]))
    evaluation_threshold = float(
        args.evaluation_threshold
        if args.evaluation_threshold is not None
        else visual_capture.get(
            "threshold", run_arguments.get("evaluation_threshold", 0.5)
        )
    )
    if not 0.0 <= evaluation_threshold <= 1.0:
        raise ValueError("evaluation threshold must be in [0,1]")
    max_frames = (
        args.max_frames
        if args.max_frames is not None
        else run_arguments.get("max_frames")
    )

    from experiments.run_online_bayesian_lifespan_thaw import build_causal_records
    from experiments.train_cue_temporal_rchange import (
        build_fixed_cue_views,
        load_fixed_camera_index,
    )
    records, _names = build_causal_records(source_path, max_frames=max_frames)
    cameras = load_fixed_camera_index(fixed_cameras)
    base = None
    pipe = None
    background = None
    if captured_event_render_dir is None:
        from scene import GaussianModel

        base_ply = (
            source_path
            / "reference_reconstruction/point_cloud/iteration_30000/point_cloud.ply"
        ).resolve()
        base = GaussianModel(sh_degree=3, active_sh_degree=0)
        base.load_ply_change(str(base_ply))
        pipe = SimpleNamespace(
            compute_cov3D_python=False,
            convert_SHs_python=False,
            debug=False,
        )
        background = torch.zeros(
            3, device=base.get_xyz.device, dtype=base.get_xyz.dtype
        )
    events = load_transition_rows(args.run_dir / "lifecycle_events.jsonl")
    open_counts = [
        len(events.get(int(record.global_index), {"OPEN": (), "CLOSE": ()})["OPEN"])
        for record in records
    ]
    close_counts = [
        len(events.get(int(record.global_index), {"OPEN": (), "CLOSE": ()})["CLOSE"])
        for record in records
    ]
    segment_names: list[str] = []
    segment_boundaries: list[int] = []
    previous_segment: str | None = None
    for position, record in enumerate(records):
        segment = str(record.segment_name)
        if segment != previous_segment:
            if position:
                segment_boundaries.append(position)
            segment_names.append(segment)
            previous_segment = segment
    panel_count = 5
    panel_gap = 8
    composed_width = int(args.panel_width) * panel_count + panel_gap * (
        panel_count - 1
    )

    panels_dir = args.output_dir / "panels"
    event_dir = args.output_dir / "event_render"
    panels_dir.mkdir(parents=True, exist_ok=True)
    event_dir.mkdir(parents=True, exist_ok=True)
    panel_paths: list[Path] = []
    by_segment: dict[str, list[Path]] = defaultdict(list)
    manifest_rows: list[dict[str, Any]] = []
    open_state_rows: set[int] = set()
    closed_state_rows: set[int] = set()

    with torch.no_grad():
        for position, record in enumerate(records):
            timestamp = int(record.global_index)
            view = build_fixed_cue_views(
                [record], cameras, cue_cache_root, resolution
            )[0][0]
            action_rows = events.get(timestamp, {"OPEN": (), "CLOSE": ()})
            open_state_rows, closed_state_rows = advance_lifecycle_state(
                open_state_rows,
                closed_state_rows,
                open_rows=action_rows["OPEN"],
                close_rows=action_rows["CLOSE"],
            )
            stem = Path(record.name).stem
            captured_event_path = (
                captured_event_render_dir / f"{stem}.png"
                if captured_event_render_dir is not None
                else None
            )
            if captured_event_path is not None:
                if not captured_event_path.exists():
                    raise FileNotFoundError(captured_event_path)
                with Image.open(captured_event_path) as image:
                    event_rgb = np.asarray(image.convert("RGB"))
                event_source = "saved_causal_lifecycle_state_event_render"
            else:
                assert base is not None and pipe is not None and background is not None
                event_rgb = render_event_rows(
                    view,
                    base,
                    pipe,
                    background,
                    action_rows["OPEN"],
                    action_rows["CLOSE"],
                    open_state_rows=sorted(open_state_rows),
                    closed_state_rows=sorted(closed_state_rows),
                )
                event_source = "reconstructed_immutable_base_lifecycle_state"
            raw_path = raw_render_dir / f"{stem}.png"
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
            threshold_path = (
                thresholded_render_dir / f"{stem}.png"
                if thresholded_render_dir is not None
                else None
            )
            thresholded, threshold_source = load_evaluation_mask_rgb(
                threshold_path,
                raw_render=raw,
                threshold=evaluation_threshold,
            )
            chart = open_close_timeline_chart(
                open_counts,
                close_counts,
                current_timestamp=position,
                width=composed_width,
                height=int(args.timeline_height),
                boundaries=segment_boundaries,
                segment_names=segment_names,
            )
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
                open_state_count=len(open_state_rows),
                closed_state_count=len(closed_state_rows),
                thresholded_render=thresholded,
                lifecycle_chart=chart,
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
                    "open_state_count": len(open_state_rows),
                    "closed_state_count": len(closed_state_rows),
                    "panel": str(panel_path),
                    "event_render": str(event_path),
                    "event_render_source": event_source,
                    "raw_render": str(raw_path),
                    "thresholded_render": (
                        str(threshold_path) if threshold_path is not None else None
                    ),
                    "threshold_source": threshold_source,
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
        "schema_version": 2,
        "contract": "persistent half-bright lifecycle state with full-bright exact-timestamp OPEN/CLOSE events",
        "run_dir": str(args.run_dir),
        "raw_render_dir": str(raw_render_dir),
        "thresholded_render_dir": (
            str(thresholded_render_dir)
            if thresholded_render_dir is not None
            else None
        ),
        "captured_event_render_dir": (
            str(captured_event_render_dir)
            if captured_event_render_dir is not None
            else None
        ),
        "evaluation_threshold": evaluation_threshold,
        "threshold_panel_source": (
            "saved post-opt metric prediction"
            if thresholded_render_dir is not None
            else "recomputed from saved raw render"
        ),
        "lifecycle_timeline": "causal line prefix; Gaussian counts on one shared axis",
        "frames": len(panel_paths),
        "continuous_gif": str(continuous_gif),
        "segment_gifs": segment_gifs,
        "event_colors": {
            "OPEN": "full green",
            "CLOSE": "full red",
            "OPEN_state": "half green",
            "CLOSED_state": "half red",
        },
        "state_intensity": STATE_INTENSITY,
        "never_open_rows_rendered": False,
        "persistent_state_rows_rendered": True,
        "event_rows_persist_across_frames_as_state": True,
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
