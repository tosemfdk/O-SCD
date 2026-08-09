"""Visualize discrete temporal R_change state switches from an existing pilot run.

This script does not estimate poses or train anything. It reloads the saved
TemporalChangeModel sidecar, reconstructs one cached camera pose, and renders the
same fixed camera while only the query timestamp changes the active state slot.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from gaussian_renderer import render_change_temporal
from scene import GaussianModel
from scene.cameras import Camera
from temporal import TemporalChangeModel, TemporalGeometryChangeModel


DEFAULT_RUN_DIR = Path("outputs/instance1_scene_change1_2_3_temporal_rchange")
DEFAULT_TIMESTAMPS = (94, 95, 198, 199)
STATE_COLORS = [(66, 133, 244), (251, 140, 0), (52, 168, 83), (156, 39, 176)]


def focal2fov(focal: float, pixels: int) -> float:
    return 2.0 * math.atan(float(pixels) / (2.0 * float(focal)))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_font(size: int) -> ImageFont.ImageFont:
    """Prefer a readable system font, falling back to PIL's default font."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def total_frames_from_summary(summary: dict[str, Any]) -> int:
    """Derive the full sequence length from manifest counts when available."""
    counts = summary.get("manifest_counts", {})
    if isinstance(counts.get("inference_images"), int):
        return int(counts["inference_images"])
    per_scene = counts.get("per_source_scene")
    if isinstance(per_scene, dict):
        return int(sum(int(v) for v in per_scene.values()))
    records = [*summary.get("probe_frames", []), *summary.get("train_frames", [])]
    if records:
        return max(int(r["global_index"]) for r in records) + 1
    return int(max(summary.get("boundaries", [0]))) + 1


def frame_state(timestamp: int, boundaries: list[int]) -> int:
    for state, boundary in enumerate(boundaries):
        if timestamp < int(boundary):
            return state
    return len(boundaries)


def load_rgb_tensor(path: Path, resolution: float) -> torch.Tensor:
    """Load the real image only to give Camera fixed dimensions/device tensors."""
    image = Image.open(path).convert("RGB")
    if resolution > 0.0 and resolution != 1.0:
        width = int(round(image.width / resolution))
        height = int(round(image.height / resolution))
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous().cuda(non_blocking=True)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.detach().float().cpu().clamp(0, 1)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr.repeat(3, 1, 1)
    if arr.ndim == 2:
        arr = arr[None].repeat(3, 1, 1)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = arr.permute(1, 2, 0)
    return Image.fromarray((arr.numpy() * 255.0).astype(np.uint8))


def render_array(render: torch.Tensor) -> np.ndarray:
    arr = render.detach().float().cpu().clamp(0, 1)
    if arr.ndim == 3:
        arr = arr.mean(dim=0)
    return arr.numpy()


def overlay_change(rgb: torch.Tensor, render: torch.Tensor, alpha_scale: float = 0.72) -> Image.Image:
    """Overlay a fixed red/yellow change cue on RGB using render values in [0, 1]."""
    base = rgb.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    cue = render_array(render)[..., None]
    color = np.zeros_like(base)
    color[..., 0] = 1.0
    color[..., 1] = 0.18 + 0.72 * cue[..., 0]
    alpha = alpha_scale * cue
    out = base * (1.0 - alpha) + color * alpha
    return Image.fromarray((np.clip(out, 0, 1) * 255.0).astype(np.uint8))


def heatmap_fixed(diff: torch.Tensor) -> Image.Image:
    """Use a fixed [0, 1] black-red-yellow-white heatmap scale."""
    x = render_array(diff)
    x = np.clip(x, 0.0, 1.0)
    rgb = np.zeros((*x.shape, 3), dtype=np.float32)
    rgb[..., 0] = np.clip(3.0 * x, 0.0, 1.0)
    rgb[..., 1] = np.clip(3.0 * x - 1.0, 0.0, 1.0)
    rgb[..., 2] = np.clip(3.0 * x - 2.0, 0.0, 1.0)
    return Image.fromarray((rgb * 255.0).astype(np.uint8))


def add_label(image: Image.Image, label: str, font: ImageFont.ImageFont, label_h: int = 34) -> Image.Image:
    out = Image.new("RGB", (image.width, image.height + label_h), "white")
    out.paste(image.convert("RGB"), (0, label_h))
    draw = ImageDraw.Draw(out)
    draw.text((8, 7), label, fill=(0, 0, 0), font=font)
    return out


def resize_panel(image: Image.Image, width: int) -> Image.Image:
    height = int(round(image.height * (width / image.width)))
    return image.resize((width, height), Image.Resampling.BILINEAR)


def hstack(images: list[Image.Image], gap: int = 12, bg: str = "white") -> Image.Image:
    width = sum(im.width for im in images) + gap * (len(images) - 1)
    height = max(im.height for im in images)
    out = Image.new("RGB", (width, height), bg)
    x = 0
    for im in images:
        out.paste(im.convert("RGB"), (x, 0))
        x += im.width + gap
    return out


def vstack(images: list[Image.Image], gap: int = 12, bg: str = "white") -> Image.Image:
    width = max(im.width for im in images)
    height = sum(im.height for im in images) + gap * (len(images) - 1)
    out = Image.new("RGB", (width, height), bg)
    y = 0
    for im in images:
        out.paste(im.convert("RGB"), (0, y))
        y += im.height + gap
    return out


def timeline_panel(
    timestamps: list[int],
    boundaries: list[int],
    total_frames: int,
    active_timestamp: int | None,
    width: int,
    height: int = 96,
    show_all_markers: bool = False,
) -> Image.Image:
    """Draw the full sequence timeline with boundary labels kept apart."""
    max_t = max(1, int(total_frames) - 1)
    out = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(out)
    font = load_font(18)
    small = load_font(15)
    left_pad, right_pad = 24, 24
    y = height // 2

    def x_of(t: int) -> int:
        return left_pad + int(round((width - left_pad - right_pad) * int(t) / max_t))

    starts = [0, *boundaries]
    ends = [*boundaries, int(total_frames)]
    for state, (start, end) in enumerate(zip(starts, ends)):
        x0, x1 = x_of(start), x_of(max(start, end - 1))
        draw.rectangle((x0, y - 11, x1, y + 11), fill=STATE_COLORS[state % len(STATE_COLORS)])
        draw.text((x0 + 4, y + 17), f"S{state}", fill=(0, 0, 0), font=small)
    draw.line((left_pad, y, width - right_pad, y), fill=(0, 0, 0), width=1)
    draw.text((left_pad, 6), "full sequence timeline", fill=(0, 0, 0), font=small)
    draw.text((left_pad, height - 22), "0", fill=(0, 0, 0), font=small)
    draw.text((width - right_pad - 38, height - 22), str(total_frames - 1), fill=(0, 0, 0), font=small)
    for idx, boundary in enumerate(boundaries):
        x = x_of(boundary)
        draw.line((x, 16, x, height - 18), fill=(190, 0, 0), width=3)
        label_y = 18 if idx % 2 == 0 else 2
        draw.text((x + 5, label_y), f"boundary {boundary}", fill=(190, 0, 0), font=small)
    marker_timestamps = list(timestamps) if show_all_markers else ([] if active_timestamp is None else [active_timestamp])
    for timestamp in marker_timestamps:
        x = x_of(timestamp)
        fill = (255, 0, 0) if timestamp == active_timestamp else (0, 0, 0)
        draw.ellipse((x - 5, y - 26, x + 5, y - 16), fill=fill)
        draw.text((x - 13, y - 45), str(timestamp), fill=fill, font=small)
    return out


def find_pose(summary: dict[str, Any], pose_cache: dict[str, Any], frame_name: str) -> dict[str, Any]:
    pose = summary.get("pose_results", {}).get(frame_name)
    if pose and pose.get("ok"):
        return pose
    candidates = [
        (key, value)
        for key, value in pose_cache.items()
        if value.get("frame_name") == frame_name and value.get("ok")
    ]
    if candidates:
        # Prefer the most geometrically supported PnP solution.
        _, selected = max(
            candidates,
            key=lambda item: (
                int(item[1].get("inliers", 0)),
                -float(item[1].get("reprojection_rmse") or float("inf")),
                item[0],
            ),
        )
        return selected
    raise KeyError(f"No valid cached pose found for {frame_name}")


def frame_by_timestamp(summary: dict[str, Any], timestamp: int) -> dict[str, Any]:
    records = [*summary.get("probe_frames", []), *summary.get("train_frames", [])]
    for record in records:
        if int(record["global_index"]) == int(timestamp):
            return record
    source = Path(summary["source_path"])
    image_dir = source / "inference_scene" / "images"
    names = sorted(p.name for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    name = names[int(timestamp)]
    return {
        "global_index": int(timestamp),
        "segment_id": frame_state(int(timestamp), list(summary["boundaries"])),
        "name": name,
        "image_path": str(image_dir / name),
    }


def make_fixed_camera(record: dict[str, Any], pose: dict[str, Any], summary: dict[str, Any]) -> Camera:
    image = load_rgb_tensor(Path(record["image_path"]), float(summary.get("resolution", 1.0)))
    height, width = int(image.shape[1]), int(image.shape[2])
    K = summary.get("camera_intrinsics")
    if not K:
        raise KeyError("summary.json is missing camera_intrinsics")
    fovx = focal2fov(float(K[0][0]), width)
    fovy = focal2fov(float(K[1][1]), height)
    Rt = np.asarray(pose["Rt"], dtype=np.float64)
    uid = f"fixed_temporal_switch_{int(record['global_index']):06d}"
    view = Camera(
        colmap_id=uid,
        R=np.transpose(Rt[:3, :3]),
        T=Rt[:3, 3],
        FoVx=fovx,
        FoVy=fovy,
        image=image,
        gt_alpha_mask=None,
        image_name=Path(record["name"]).stem,
        uid=uid,
    )
    view.timestamp = float(record["global_index"])
    view.segment_id = int(record.get("segment_id", frame_state(int(record["global_index"]), list(summary["boundaries"]))))
    return view


def load_temporal_model(
    checkpoint_path: Path,
) -> TemporalChangeModel | TemporalGeometryChangeModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    base_ply = Path(checkpoint["base_ply"])
    if not base_ply.exists():
        raise FileNotFoundError(base_ply)
    state_dict = checkpoint["state_dict"]
    max_states = int(state_dict["state_change_dc"].shape[1])
    base = GaussianModel(sh_degree=3, active_sh_degree=0)
    base.load_ply_change(str(base_ply))
    model_class = (
        TemporalGeometryChangeModel
        if "state_xyz_delta" in state_dict
        else TemporalChangeModel
    )
    model = model_class.from_gaussians(
        base,
        max_states=max_states,
        initial_time=0.0,
    )
    cuda_state = {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()}
    model.load_state_dict(cuda_state, strict=True)
    model.eval()
    return model


def render_timestamps(view: Camera, model: TemporalChangeModel, timestamps: list[int]) -> dict[int, torch.Tensor]:
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
    background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
    out: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for timestamp in timestamps:
            rendered = render_change_temporal(view, model, pipe, background, timestamp=float(timestamp))["render"]
            out[int(timestamp)] = rendered.detach().mean(dim=0, keepdim=True).clamp(0, 1)
    return out


def state_combination_counts(state_valid: torch.Tensor) -> dict[str, int]:
    """Return all exact state_valid combinations, including zero-count combos."""
    valid = state_valid.detach().bool().cpu()
    if valid.ndim != 2:
        raise ValueError("state_valid must be [N, S]")
    powers = (2 ** torch.arange(valid.shape[1], dtype=torch.long)).view(1, -1)
    codes = (valid.long() * powers).sum(dim=1)
    counts = torch.bincount(codes, minlength=2 ** valid.shape[1])
    labels: dict[str, int] = {}
    for code, count in enumerate(counts.tolist()):
        active = [f"S{s}" for s in range(valid.shape[1]) if code & (1 << s)]
        labels["+".join(active) if active else "none"] = int(count)
    return labels


def metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    d = (a - b).abs().detach().float().cpu().numpy()
    return {
        "mean_abs": float(d.mean()),
        "max_abs": float(d.max()),
        "p95_abs": float(np.percentile(d, 95.0)),
        "pixels_gt_0_05": int((d > 0.05).sum()),
        "pixels_gt_0_10": int((d > 0.10).sum()),
    }


def build_static_png(
    rgb: torch.Tensor,
    renders: dict[int, torch.Tensor],
    boundaries: list[int],
    total_frames: int,
    output_path: Path,
) -> None:
    title_font = load_font(28)
    row_font = load_font(23)
    label_font = load_font(18)
    panel_w = 250

    rgb_panel = resize_panel(tensor_to_pil(rgb), panel_w)
    rows: list[Image.Image] = []
    for boundary, left, right, row_title in [
        (95, 94, 95, "Boundary 94->95: S0->S1"),
        (199, 198, 199, "Boundary 198->199: S1->S2"),
    ]:
        diff = (renders[right] - renders[left]).abs().clamp(0, 1)
        panels = [
            add_label(rgb_panel, "fixed RGB", label_font),
            add_label(resize_panel(overlay_change(rgb, renders[left]), panel_w), f"RGB + current t{left} (S{frame_state(left, boundaries)})", label_font),
            add_label(resize_panel(overlay_change(rgb, renders[right]), panel_w), f"RGB + current t{right} (S{frame_state(right, boundaries)})", label_font),
            add_label(resize_panel(heatmap_fixed(diff), panel_w), "fixed-scale abs diff [0,1]", label_font),
        ]
        row = hstack(panels, gap=12)
        header = Image.new("RGB", (row.width, 42), "white")
        ImageDraw.Draw(header).text((8, 7), row_title, fill=STATE_COLORS[frame_state(right, boundaries) % len(STATE_COLORS)], font=row_font)
        rows.append(vstack([header, row], gap=0))

    width = max(row.width for row in rows)
    title = Image.new("RGB", (width, 48), "white")
    ImageDraw.Draw(title).text((8, 8), "Temporal R_change fixed-camera state switch", fill=(0, 0, 0), font=title_font)
    legend = Image.new("RGB", (width, 34), "white")
    legend_draw = ImageDraw.Draw(legend)
    legend_draw.rectangle((8, 9, 28, 27), fill=(255, 220, 30))
    legend_draw.text((36, 8), "yellow = current active change", fill=(0, 0, 0), font=label_font)
    legend_draw.rectangle((360, 9, 380, 27), fill=(255, 80, 0))
    legend_draw.text((388, 8), "red/orange = |after-before|", fill=(0, 0, 0), font=label_font)
    timeline = timeline_panel([], boundaries, total_frames, None, width, height=102)
    vstack([title, legend, *rows, timeline], gap=14).save(output_path)


def build_gif(
    rgb: torch.Tensor,
    renders: dict[int, torch.Tensor],
    boundaries: list[int],
    total_frames: int,
    output_path: Path,
) -> None:
    font = load_font(20)
    frames: list[Image.Image] = []
    rgb_panel = resize_panel(tensor_to_pil(rgb), 290)
    for timestamp in sorted(renders):
        overlay = resize_panel(overlay_change(rgb, renders[timestamp]), 290)
        panels = hstack([
            add_label(rgb_panel, "fixed RGB", font),
            add_label(overlay, f"current t{timestamp} | S{frame_state(timestamp, boundaries)}", font),
        ], gap=10)
        timeline = timeline_panel([], boundaries, total_frames, timestamp, panels.width, height=82)
        frames.append(vstack([panels, timeline], gap=6))
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=800, loop=0)


def combination_label(pattern: str) -> str:
    """Make exact non-empty state_valid pattern labels human-readable."""
    if pattern == "none":
        return "none"
    parts = pattern.split("+")
    if len(parts) == 1:
        return f"{parts[0]} only"
    return " + ".join(parts)


def build_lifespan_counts_png(active_counts: list[int], combination_counts: dict[str, int], output_path: Path) -> None:
    font = load_font(16)
    title_font = load_font(20)
    section_font = load_font(18)
    width, row_h = 780, 28
    nonempty_patterns = [(k, v) for k, v in combination_counts.items() if k != "none" and v > 0]
    none_count = int(combination_counts.get("none", 0))
    active_rows = len(active_counts)
    pattern_rows = len(nonempty_patterns)
    height = 112 + row_h * (active_rows + pattern_rows) + 48
    out = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(out)
    draw.text((8, 8), "Temporal lifespan counts", fill=(0, 0, 0), font=title_font)
    draw.text((8, 38), f"none = {none_count} (not bar-scaled)", fill=(70, 70, 70), font=font)

    y = 72
    draw.text((8, y), "Active counts per state", fill=(0, 0, 0), font=section_font)
    y += 28
    active_max = max(active_counts) if active_counts else 1
    for idx, value in enumerate(active_counts):
        draw.text((18, y + 4), f"S{idx}", fill=(0, 0, 0), font=font)
        bar_w = int(430 * value / active_max) if active_max else 0
        draw.rectangle((150, y + 5, 150 + bar_w, y + 22), fill=STATE_COLORS[idx % len(STATE_COLORS)])
        draw.text((600, y + 4), str(value), fill=(0, 0, 0), font=font)
        y += row_h

    y += 16
    draw.text((8, y), "Exact non-empty state_valid patterns", fill=(0, 0, 0), font=section_font)
    y += 28
    pattern_max = max((value for _, value in nonempty_patterns), default=1)
    for pattern, value in nonempty_patterns:
        draw.text((18, y + 4), combination_label(pattern), fill=(0, 0, 0), font=font)
        bar_w = int(430 * value / pattern_max) if pattern_max else 0
        draw.rectangle((210, y + 5, 210 + bar_w, y + 22), fill=(90, 140, 220))
        draw.text((660, y + 4), str(value), fill=(0, 0, 0), font=font)
        y += row_h
    out.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize fixed-camera temporal state switches from a saved R_change run")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--camera-timestamp", type=int, default=225, help="Cached frame pose to reuse as the one fixed camera")
    parser.add_argument("--timestamps", nargs="+", type=int, default=list(DEFAULT_TIMESTAMPS))
    parser.add_argument("--output-prefix", default="temporal_state_switch")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the existing Gaussian renderer and Camera implementation")

    run_dir = args.run_dir
    summary = read_json(run_dir / "summary.json")
    pose_cache = read_json(run_dir / "pose_cache.json")
    checkpoint_path = run_dir / "temporal_rchange_checkpoint.pt"
    boundaries = [int(x) for x in summary["boundaries"]]
    total_frames = total_frames_from_summary(summary)

    camera_record = frame_by_timestamp(summary, args.camera_timestamp)
    camera_pose = find_pose(summary, pose_cache, camera_record["name"])
    view = make_fixed_camera(camera_record, camera_pose, summary)
    model = load_temporal_model(checkpoint_path)

    timestamps = sorted({int(t) for t in args.timestamps} | {94, 95, 198, 199})
    renders = render_timestamps(view, model, timestamps)
    fixed_rgb = view.original_image[:3].detach()

    active_counts = [int(x) for x in model.state_valid.detach().sum(dim=0).cpu().tolist()]
    combination_counts = state_combination_counts(model.state_valid)
    pair_metrics = {f"{a}->{b}": metrics(renders[a], renders[b]) for a, b in [(94, 95), (198, 199), (95, 198)] if a in renders and b in renders}

    static_path = run_dir / f"{args.output_prefix}.png"
    gif_path = run_dir / f"{args.output_prefix}.gif"
    counts_path = run_dir / f"{args.output_prefix}_lifespan_counts.png"
    manifest_path = run_dir / f"{args.output_prefix}_manifest.json"
    build_static_png(fixed_rgb, renders, boundaries, total_frames, static_path)
    build_gif(fixed_rgb, renders, boundaries, total_frames, gif_path)
    build_lifespan_counts_png(active_counts, combination_counts, counts_path)

    manifest = {
        "script": "experiments/visualize_temporal_state_switch.py",
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "selected_camera": {
            "timestamp": int(camera_record["global_index"]),
            "frame": camera_record["name"],
            "pose_reference_name": camera_pose.get("reference_name"),
            "R": view.R.tolist() if hasattr(view.R, "tolist") else view.R,
            "T": np.asarray(view.T).tolist(),
            "FoVx": float(view.FoVx),
            "FoVy": float(view.FoVy),
        },
        "dimensions": {"width": int(view.image_width), "height": int(view.image_height)},
        "total_frames": int(total_frames),
        "boundaries": boundaries,
        "rendered_timestamps": timestamps,
        "timestamp_to_state": {str(t): frame_state(t, boundaries) for t in timestamps},
        "state_valid_counts": active_counts,
        "state_valid_combination_counts": combination_counts,
        "pairwise_render_difference_metrics": pair_metrics,
        "within_state_95_198_max_abs": pair_metrics.get("95->198", {}).get("max_abs"),
        "outputs": {
            "static_png": str(static_path),
            "animated_gif": str(gif_path),
            "lifespan_counts_png": str(counts_path),
        },
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps({"static_png": str(static_path), "animated_gif": str(gif_path), "lifespan_counts_png": str(counts_path), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
