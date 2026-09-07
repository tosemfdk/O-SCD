#!/usr/bin/env python3
"""Visualize E4 XFeat-triangulated NEW Gaussian seeds in 2D and 3D."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from plyfile import PlyData

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.train_cue_temporal_rchange import camera_json_to_w2c

DATA = REPO / "data/Instance_1"
RUN = REPO / "outputs/instance1_scene_change1_2_3_temporal_rchange_oscd_cues_allframes_120"
DEFAULT_RESULT = REPO / "outputs/e4_online_xfeat_new_seed_scene123_120_final_corrected_20260814"
DEFAULT_OUTPUT = Path("/tmp/e4_xfeat_new_seed_visualization")
COLORS = {
    "new": (35, 220, 85),
    "outside_new": (255, 115, 45),
    "birth": (255, 225, 50),
    "offscreen": (155, 155, 155),
}
SOURCE_COLORS = {
    "xfeat": (40, 170, 255),
    "densified": (255, 55, 210),
    "outside": (255, 135, 35),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scenes", type=int, nargs="+", default=(1, 2, 3), choices=(1, 2, 3))
    parser.add_argument("--gif-width", type=int, default=528)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--marker-radius", type=int, default=5)
    parser.add_argument("--reference-sample", type=int, default=45000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def bool_csv(value: str) -> bool:
    return value.strip().lower() == "true"


def load_cameras() -> dict[str, dict[str, Any]]:
    summary = json.loads((RUN / "summary.json").read_text())
    payload = json.loads(Path(summary["fixed_cameras_json"]).read_text())
    return {str(row["img_name"]): row for row in payload}


def load_scene_seed(scene_dir: Path) -> dict[str, Any]:
    payload = torch.load(scene_dir / "new_seed_checkpoint.pt", map_location="cpu", weights_only=False)
    return payload["seed_sidecar"]


def project_seeds(
    xyz: np.ndarray,
    scaling: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    global_index: int,
    camera: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    active = (start <= float(global_index)) & (float(global_index) < end)
    w2c = camera_json_to_w2c(camera)
    K = np.asarray(
        [
            [float(camera["fx"]), 0.0, float(camera["width"]) / 2.0],
            [0.0, float(camera["fy"]), float(camera["height"]) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    homogeneous = np.concatenate([xyz.astype(np.float64), np.ones((len(xyz), 1))], axis=1)
    cam = homogeneous @ w2c.T
    depth = cam[:, 2]
    projected = cam[:, :3] @ K.T
    uv = projected[:, :2] / np.where(np.abs(projected[:, 2:3]) > 1e-9, projected[:, 2:3], 1.0)
    # The initialized seeds are isotropic.  This is a first-order projected
    # 1-sigma scale radius, not the renderer's full alpha-support ellipse.
    radius_world = np.exp(scaling.astype(np.float64)).mean(axis=1)
    focal = math.sqrt(float(camera["fx"]) * float(camera["fy"]))
    radius_px = np.where(depth > 1e-9, focal * radius_world / depth, np.nan)
    return uv, depth, active, radius_px


def load_new_mask(scene: int, frame_name: str, width: int, height: int) -> np.ndarray:
    root = DATA / f"scene_change{scene}"
    annotation = json.loads((root / "object_change_annotations.json").read_text())
    mask = np.zeros((height, width), dtype=bool)
    for obj in annotation["objects"]:
        attrs = obj["attributes"]
        if attrs["change_type"] != "GEOMETRY" or attrs["change_state"] != "NEW":
            continue
        row = next(item for item in obj["segmentation"]["masks"] if item["frame_name"] == frame_name)
        raw = cv2.imread(str(root / row["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise FileNotFoundError(root / row["mask_path"])
        resized = cv2.resize((raw > 0).astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
        mask |= resized.astype(bool)
    return mask


def read_rgb(scene: int, frame_name: str, width: int, height: int) -> np.ndarray:
    path = DATA / f"scene_change{scene}" / "inference_scene" / "images" / frame_name
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)


def overlay_new(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    green = np.asarray(COLORS["new"], dtype=np.float32)
    out[mask] = 0.73 * out[mask] + 0.27 * green
    return np.uint8(np.clip(out, 0, 255))


def draw_cross(draw: ImageDraw.ImageDraw, x: float, y: float, radius: int, color: tuple[int, int, int], width: int = 2) -> None:
    draw.line((x - radius, y, x + radius, y), fill=color, width=width)
    draw.line((x, y - radius, x, y + radius), fill=color, width=width)


def draw_seed_view(
    rgb: np.ndarray,
    new_mask: np.ndarray,
    uv: np.ndarray,
    depth: np.ndarray,
    active: np.ndarray,
    radius_px: np.ndarray,
    start: np.ndarray,
    global_index: int,
    *,
    marker_radius: int,
    actual_only: bool,
    birth_kind: list[str] | None = None,
) -> tuple[Image.Image, dict[str, int | float]]:
    image = Image.fromarray(overlay_new(rgb, new_mask))
    draw = ImageDraw.Draw(image, "RGBA")
    height, width = new_mask.shape
    visible = 0
    new_hits = 0
    outside_hits = 0
    born = 0
    xfeat_visible = 0
    densified_visible = 0
    visible_radii: list[float] = []
    for row in np.flatnonzero(active):
        x, y = uv[row]
        inside = depth[row] > 0 and 0 <= x < width and 0 <= y < height
        if not inside:
            continue
        visible += 1
        xi = min(max(int(math.floor(x)), 0), width - 1)
        yi = min(max(int(math.floor(y)), 0), height - 1)
        in_new = bool(new_mask[yi, xi])
        new_hits += int(in_new)
        outside_hits += int(not in_new)
        visible_radii.append(float(radius_px[row]))
        is_birth = math.isclose(float(start[row]), float(global_index), rel_tol=0.0, abs_tol=1e-4)
        born += int(is_birth)
        kind = "xfeat" if birth_kind is None else birth_kind[row]
        xfeat_visible += int(kind != "new_only_densified")
        densified_visible += int(kind == "new_only_densified")
        if not in_new:
            color = SOURCE_COLORS["outside"]
        elif kind == "new_only_densified":
            color = SOURCE_COLORS["densified"]
        else:
            color = SOURCE_COLORS["xfeat"]
        if is_birth:
            color = COLORS["birth"]
        if actual_only:
            r = max(0.5, float(radius_px[row]))
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(*color, 150), outline=(*color, 255), width=1)
        else:
            r = marker_radius + (2 if is_birth else 0)
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(*color, 70), outline=(*color, 255), width=2)
            draw_cross(draw, x, y, max(2, r - 2), (*color, 255), width=1)
    median_radius = float(np.median(visible_radii)) if visible_radii else 0.0
    q90_radius = float(np.quantile(visible_radii, 0.9)) if visible_radii else 0.0
    return image, {
        "active": int(active.sum()),
        "visible": visible,
        "new": new_hits,
        "outside": outside_hits,
        "born": born,
        "xfeat_visible": xfeat_visible,
        "densified_visible": densified_visible,
        "median_radius_px": median_radius,
        "q90_radius_px": q90_radius,
    }


def label_panel(panel: Image.Image, title: str, subtitle: str) -> Image.Image:
    header = 72
    canvas = Image.new("RGB", (panel.width, panel.height + header), "white")
    canvas.paste(panel, (0, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), title, font=font(20, bold=True), fill="black")
    draw.text((12, 40), subtitle, font=font(12), fill=(65, 65, 65))
    return canvas


def compose_gif_frame(
    scene: int,
    local_frame: int,
    total_frames: int,
    frame_name: str,
    actual: Image.Image,
    enlarged: Image.Image,
    stats: dict[str, int | float],
) -> Image.Image:
    p1 = label_panel(actual, "Projected seed scale", "First-order 1-sigma radius proxy; green tint = NEW GT")
    p2 = label_panel(enlarged, "Enlarged seed centers", "Blue=XFeat, magenta=NEW-only child, orange=outside GT, yellow=new")
    top = 92
    footer = 62
    canvas = Image.new("RGB", (p1.width + p2.width, top + p1.height + footer), "white")
    canvas.paste(p1, (0, top)); canvas.paste(p2, (p1.width, top))
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 12), f"SceneChange{scene}: XFeat-triangulated NEW Gaussian seeds", font=font(25, bold=True), fill="black")
    draw.text((18, 51), f"Frame {local_frame}/{total_frames}  {frame_name}", font=font(15), fill=(55, 55, 55))
    offscreen = stats["active"] - stats["visible"]
    draw.text(
        (18, top + p1.height + 11),
        f"active={stats['active']}  visible={stats['visible']}  inside NEW={stats['new']}  "
        f"outside NEW={stats['outside']}  offscreen={offscreen}  born now={stats['born']}",
        font=font(15, bold=True),
        fill=(25, 25, 25),
    )
    draw.text(
        (18, top + p1.height + 36),
        f"visible projected radius: median={stats['median_radius_px']:.2f}px, q90={stats['q90_radius_px']:.2f}px",
        font=font(13),
        fill=(55, 55, 55),
    )
    return canvas


def create_scene_gif(
    scene: int,
    result_dir: Path,
    output_dir: Path,
    cameras: dict[str, dict[str, Any]],
    *,
    width: int,
    fps: float,
    marker_radius: int,
) -> dict[str, Any]:
    scene_dir = result_dir / f"scene_change{scene}"
    seed = load_scene_seed(scene_dir)
    xyz = seed["xyz"].numpy()
    scaling = seed["scaling"].numpy()
    start = seed["start"].numpy()
    end = seed["end"].numpy()
    birth_kind = [str(row.get("birth_kind", "xfeat_triangulated")) for row in seed.get("metadata", [{} for _ in range(len(xyz))])]
    frame_names = sorted(path.name for path in (DATA / f"scene_change{scene}" / "inference_scene" / "images").glob("*.png"))
    global_offset = {1: 0, 2: 95, 3: 199}[scene]
    height = int(round(width * 941 / 528))
    frames: list[Image.Image] = []
    frame_summaries: list[dict[str, Any]] = []
    for local_frame, frame_name in enumerate(frame_names, 1):
        global_index = global_offset + local_frame - 1
        camera = cameras[Path(frame_name).stem]
        rgb = read_rgb(scene, frame_name, width, height)
        new_mask = load_new_mask(scene, frame_name, width, height)
        uv, depth, active, radius_px = project_seeds(xyz, scaling, start, end, global_index, camera)
        actual, stats = draw_seed_view(rgb, new_mask, uv, depth, active, radius_px, start, global_index, marker_radius=marker_radius, actual_only=True, birth_kind=birth_kind)
        enlarged, stats2 = draw_seed_view(rgb, new_mask, uv, depth, active, radius_px, start, global_index, marker_radius=marker_radius, actual_only=False, birth_kind=birth_kind)
        assert stats == stats2
        frames.append(compose_gif_frame(scene, local_frame, len(frame_names), frame_name, actual, enlarged, stats))
        frame_summaries.append({"local_frame": local_frame, "global_index": global_index, "frame_name": frame_name, **stats})
    gif_path = output_dir / f"scene_change{scene}_new_seed_projection.gif"
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=int(round(1000 / fps)), loop=0, optimize=False, disposal=2)
    return {"scene": scene, "gif": str(gif_path), "frames": frame_summaries, "seed_count": len(xyz)}


def load_reference_xyz(sample: int, seed: int) -> np.ndarray:
    checkpoint = torch.load(RUN / "temporal_rchange_checkpoint.pt", map_location="cpu", weights_only=False)
    vertices = PlyData.read(str(checkpoint["base_ply"]))["vertex"].data
    xyz = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float32)
    rng = np.random.default_rng(seed)
    if len(xyz) > sample:
        xyz = xyz[rng.choice(len(xyz), sample, replace=False)]
    return xyz


def projection_2d(points: np.ndarray, axes: tuple[int, int]) -> np.ndarray:
    return points[:, axes]


def draw_3d_overview(result_dir: Path, output_dir: Path, scenes: list[int], reference_sample: int, seed: int) -> Path:
    ref = load_reference_xyz(reference_sample, seed)
    seed_xyz = {}
    starts = {}
    source_masks = {}
    for scene in scenes:
        payload = load_scene_seed(result_dir / f"scene_change{scene}")
        seed_xyz[scene] = payload["xyz"].numpy()
        starts[scene] = payload["start"].numpy()
        source_masks[scene] = np.asarray(
            [row.get("birth_kind") == "new_only_densified" for row in payload.get("metadata", [])],
            dtype=bool,
        )
    # Robust crop around all E4 seed rows, then show a local reference context.
    all_seed = np.concatenate(list(seed_xyz.values()), axis=0)
    low = np.quantile(all_seed, 0.01, axis=0) - np.array([0.6, 0.6, 0.5])
    high = np.quantile(all_seed, 0.99, axis=0) + np.array([0.6, 0.6, 0.5])
    ref_local = ref[np.all((ref >= low) & (ref <= high), axis=1)]
    views = [("Top view (x,y)", (0, 1)), ("Front view (x,z)", (0, 2))]
    panel_w, panel_h = 760, 650
    canvas = Image.new("RGB", (panel_w * 2, panel_h + 120), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((22, 14), "3D distribution of XFeat-triangulated NEW Gaussian seeds", font=font(27, bold=True), fill="black")
    draw.text(
        (22, 56),
        "Gray = sampled reference Gaussians; colored points = fixed seed xyz; robust crop hides only labeled outliers",
        font=font(15),
        fill=(65, 65, 65),
    )
    scene_colors = {1: (238, 74, 92), 2: (45, 130, 230), 3: (35, 178, 105)}
    for panel_index, (title, axes) in enumerate(views):
        x0 = panel_index * panel_w
        plot = (x0 + 70, 150, x0 + panel_w - 30, panel_h + 65)
        draw.rectangle(plot, fill=(249, 250, 252), outline=(130, 130, 130), width=1)
        all_local = np.concatenate([ref_local] + [seed_xyz[s] for s in scenes], axis=0)
        pts = projection_2d(all_local, axes)
        lo = np.quantile(pts, 0.005, axis=0); hi = np.quantile(pts, 0.995, axis=0)
        span = np.maximum(hi - lo, 1e-6); lo -= 0.04 * span; hi += 0.04 * span
        def to_px(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            q = (projection_2d(values, axes) - lo) / (hi - lo)
            inside = np.all((q >= 0.0) & (q <= 1.0), axis=1)
            q = np.clip(q, 0.0, 1.0)
            x = plot[0] + q[:, 0] * (plot[2] - plot[0])
            y = plot[3] - q[:, 1] * (plot[3] - plot[1])
            return np.stack([x, y], axis=1), inside
        ref_points, ref_inside = to_px(ref_local)
        for x, y in ref_points[ref_inside]:
            draw.point((float(x), float(y)), fill=(165, 165, 165))
        hidden_seed_count = 0
        for scene in scenes:
            points, inside = to_px(seed_xyz[scene])
            hidden_seed_count += int((~inside).sum())
            order = np.argsort(starts[scene])
            for rank, row in enumerate(order):
                if not inside[row]:
                    continue
                x, y = points[row]
                r = 3
                color = (255, 55, 210) if len(source_masks[scene]) and source_masks[scene][row] else scene_colors[scene]
                draw.ellipse((x-r, y-r, x+r, y+r), fill=color, outline=(255,255,255), width=1)
        draw.text((plot[0], 108), title, font=font(20, bold=True), fill="black")
        draw.text((plot[0], plot[3] + 10), f"axis range: {lo[0]:.2f}..{hi[0]:.2f}, {lo[1]:.2f}..{hi[1]:.2f}", font=font(12), fill=(70,70,70))
        draw.text((plot[0], plot[3] + 29), f"hidden seed outliers in this view: {hidden_seed_count}", font=font(12), fill=(70,70,70))
    legend_x = 1250
    for idx, scene in enumerate(scenes):
        y = 92 + idx * 24
        color = scene_colors[scene]
        draw.ellipse((legend_x, y, legend_x + 12, y + 12), fill=color)
        draw.text((legend_x + 20, y - 3), f"SC{scene}: {len(seed_xyz[scene])} seeds", font=font(13), fill="black")
    if any(mask.any() for mask in source_masks.values()):
        y = 92 + len(scenes) * 24
        draw.ellipse((legend_x, y, legend_x + 12, y + 12), fill=(255, 55, 210))
        draw.text((legend_x + 20, y - 3), "NEW-only densified child", font=font(13), fill="black")
    path = output_dir / "scene123_new_seed_3d_overview.png"
    canvas.save(path)
    return path


def create_contact_sheet(scene_reports: list[dict[str, Any]], output_dir: Path) -> Path:
    representative = []
    for report in scene_reports:
        visible = sorted(report["frames"], key=lambda row: (row["visible"], row["new"]), reverse=True)
        representative.append(visible[0])
    panel_w, panel_h = 528, 941
    panels = []
    cameras = load_cameras()
    result_dir = Path(scene_reports[0]["result_dir"])
    for report, row in zip(scene_reports, representative):
        scene = report["scene"]
        payload = load_scene_seed(result_dir / f"scene_change{scene}")
        rgb = read_rgb(scene, row["frame_name"], panel_w, panel_h)
        mask = load_new_mask(scene, row["frame_name"], panel_w, panel_h)
        uv, depth, active, radius_px = project_seeds(
            payload["xyz"].numpy(),
            payload["scaling"].numpy(),
            payload["start"].numpy(),
            payload["end"].numpy(),
            row["global_index"],
            cameras[Path(row["frame_name"]).stem],
        )
        birth_kind = [str(item.get("birth_kind", "xfeat_triangulated")) for item in payload.get("metadata", [{} for _ in range(payload["xyz"].shape[0])])]
        enlarged, stats = draw_seed_view(
            rgb,
            mask,
            uv,
            depth,
            active,
            radius_px,
            payload["start"].numpy(),
            row["global_index"],
            marker_radius=6,
            actual_only=False,
            birth_kind=birth_kind,
        )
        panels.append(
            label_panel(
                enlarged,
                f"SceneChange{scene} frame {row['local_frame']}",
                f"active={stats['active']}, visible={stats['visible']}, inside NEW={stats['new']}, "
                f"outside={stats['outside']}, median r={stats['median_radius_px']:.2f}px",
            )
        )
    title_h = 100
    canvas = Image.new("RGB", (panel_w * len(panels), title_h + panels[0].height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 12), "Representative views of newly seeded 3D Gaussians", font=font(27, bold=True), fill="black")
    draw.text((18, 55), "Marker centers are actual projections; marker radius is enlarged for visibility. Green tint is evaluation-only NEW GT.", font=font(15), fill=(60,60,60))
    for idx, panel in enumerate(panels): canvas.paste(panel, (idx * panel_w, title_h))
    path = output_dir / "scene123_new_seed_representative_views.png"
    canvas.save(path)
    return path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cameras = load_cameras()
    reports = []
    for scene in args.scenes:
        report = create_scene_gif(scene, args.result_dir, args.output_dir, cameras, width=args.gif_width, fps=args.fps, marker_radius=args.marker_radius)
        report["result_dir"] = str(args.result_dir)
        reports.append(report)
        print(f"SceneChange{scene}: {report['seed_count']} seeds -> {report['gif']}", flush=True)
    contact = create_contact_sheet(reports, args.output_dir)
    overview = draw_3d_overview(args.result_dir, args.output_dir, list(args.scenes), args.reference_sample, args.seed)
    summary = {"result_dir": str(args.result_dir), "output_dir": str(args.output_dir), "scenes": reports, "representative_views": str(contact), "overview_3d": str(overview)}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(contact); print(overview)


if __name__ == "__main__":
    main()
