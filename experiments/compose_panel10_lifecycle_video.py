"""Recompose saved viewer captures as ten panels (5x2) plus an event timeline.

Read-only with respect to the source experiment: no model, detector, or GT
re-evaluation. Panels 2/3/7 are recovered from the saved 7x2 dashboard because
older viewers did not save those layers separately. All other panels use their
original standalone PNGs, including the exact evaluated prediction mask.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from experiments.visualize_lifespan_cue_events import _font, open_close_timeline_chart


PANEL_TITLES = (
    ("1. Gaussian lifecycle", "Base + DA3 seeds"),
    ("2. Online RGB + DA3 seeds", "Saved seed overlay"),
    ("3. Learned sigmoid cue", "Soft Q"),
    ("4. Bayes Factor", "Detector evidence"),
    ("5. Learned R_change", "Current-valid continuous score"),
    ("6. Predicted change mask", "Raw R_change >= 0.5"),
    ("7. Ground-truth change", "ADD union REMOVE; evaluation only"),
    ("8. Signed SAM difference", "SAM difference x learned Q"),
    ("9. Reference - DA3 depth", "Red: front / Blue: behind"),
    ("10. Cue types", "Red: NEW / Blue: REMOVE / Yellow: appearance"),
)
SIDECARS = {
    0: "main", 3: "detector_state", 4: "learned_rchange",
    5: "prediction_mask", 7: "sam_feature_diff",
    8: "depth_difference", 9: "cue_types",
}
EVENT_FIELDS = {
    "base_opened_now": "opened_now", "base_closed_now": "closed_now",
    "seed_opened_now": "da3_seed_opened_now", "seed_closed_now": "da3_seed_closed_now",
}
SOURCE_COLUMNS = 7
SOURCE_LABEL_HEIGHT = 44
DEFAULT_PANEL_ORDER = tuple(range(1, 11))


def ordered_panel_titles(panel_order=DEFAULT_PANEL_ORDER) -> tuple:
    """One-based source panel IDs; displayed titles follow destination numbers."""
    if sorted(panel_order) != list(DEFAULT_PANEL_ORDER):
        raise ValueError("panel order must be a permutation of 1 through 10")
    return tuple(
        (f"{destination}." + PANEL_TITLES[source - 1][0].partition(".")[2],
         PANEL_TITLES[source - 1][1])
        for destination, source in enumerate(panel_order, start=1)
    )


def load_records(run_dir: Path) -> list[dict]:
    with (run_dir / "frame_metrics.csv").open(newline="") as stream:
        records = list(csv.DictReader(stream))
    records.sort(key=lambda row: int(row["timestamp"]))
    if not records or [int(row["timestamp"]) for row in records] != list(range(len(records))):
        raise ValueError("empty or non-contiguous timestamps (missing/duplicate frame)")
    for row in records:
        t = int(row["timestamp"])
        summary = json.loads((run_dir / "captures" / f"{t:06d}_summary.json").read_text())
        if summary["timestamp"] != t or summary["frame_name"] != row["frame_name"]:
            raise ValueError(f"frame/summary mismatch at timestamp {t}")
        for field, summary_field in EVENT_FIELDS.items():
            value = int(row[field])
            if value < 0:
                raise ValueError(f"negative event count {field} at timestamp {t}")
            if value != summary[summary_field]:
                raise ValueError(f"event count mismatch: {field} at timestamp {t}")
        for suffix in ("dashboard", *SIDECARS.values()):
            path = run_dir / "captures" / f"{t:06d}_{suffix}.png"
            if not path.is_file():
                raise FileNotFoundError(path)
    return records


def event_counts(records: list[dict]) -> tuple[list[int], list[int]]:
    """Counts of commits at t, not active stocks, seed births, or pruning."""
    return (
        [int(r["base_opened_now"]) + int(r["seed_opened_now"]) for r in records],
        [int(r["base_closed_now"]) + int(r["seed_closed_now"]) for r in records],
    )


def extract_dashboard_panel(
    dashboard: Image.Image, index: int, source_size: tuple[int, int],
) -> Image.Image:
    """Invert the saved viewer's 7x2 cell placement and letterboxing (no labels)."""
    if not 0 <= index < 10:
        raise ValueError("panel index must be in [0, 9]")
    source_width, source_height = source_size
    if min(source_width, source_height) < 1:
        raise ValueError("source dimensions must be positive")
    xs = np.rint(np.linspace(0, dashboard.width, SOURCE_COLUMNS + 1)).astype(int)
    ys = np.rint(np.linspace(0, dashboard.height, 3)).astype(int)
    row, column = divmod(index, SOURCE_COLUMNS)
    left, right = int(xs[column]), int(xs[column + 1])
    top, bottom = int(ys[row]) + SOURCE_LABEL_HEIGHT, int(ys[row + 1])
    width, height = right - left, bottom - top
    if min(width, height) < 1:
        raise ValueError("dashboard is too small for the saved 7x2 layout")
    scale = min(width / source_width, height / source_height)
    w = max(1, min(width, int(round(source_width * scale))))
    h = max(1, min(height, int(round(source_height * scale))))
    x, y = left + (width - w) // 2, top + (height - h) // 2
    return dashboard.crop((x, y, x + w, y + h)).convert("RGB")


def load_panels(run_dir: Path, timestamp: int) -> list[Image.Image]:
    prefix = run_dir / "captures" / f"{timestamp:06d}"
    with Image.open(f"{prefix}_main.png") as image:
        source_size = image.size
    with Image.open(f"{prefix}_dashboard.png") as source:
        dashboard = source.convert("RGB")
    result = []
    for index in range(10):
        if index in SIDECARS:
            with Image.open(f"{prefix}_{SIDECARS[index]}.png") as image:
                if image.size != source_size:
                    raise ValueError(f"sidecar shape mismatch at t={timestamp}, panel={index + 1}")
                result.append(image.convert("RGB"))
        else:
            result.append(extract_dashboard_panel(dashboard, index, source_size))
    return result


def compose_frame(
    panels: list[Image.Image], record: dict, opens: list[int], closes: list[int], *,
    width: int = 1920, chart_height: int = 320, boundaries=(), segment_names=(),
    run_label: str = "", panel_order=DEFAULT_PANEL_ORDER,
) -> Image.Image:
    if len(panels) != 10:
        raise ValueError("exactly 10 panels are required")
    titles_in_order = ordered_panel_titles(panel_order)
    if width < 1600 or width % 2 or chart_height < 180:
        raise ValueError("use an even width >= 1600 and chart height >= 180")
    t = int(record["timestamp"])
    header, label, gap, footer = 72, 56, 8, 28
    edges = np.rint(np.linspace(0, width, 6)).astype(int)
    content_height = round((width / 5) * panels[0].height / panels[0].width)
    tile_height = label + content_height
    chart_y = header + 2 * tile_height + 2 * gap
    height = chart_y + chart_height + footer
    height += height % 2
    canvas = Image.new("RGB", (width, height), (12, 14, 18))
    draw = ImageDraw.Draw(canvas)
    title_font, subtitle_font = _font(20), _font(13)
    scene = str(record["scene"]).replace("scene_change", "SC")
    draw.text((16, 8), f"{scene} | t={t:03d} | Frame {t + 1}/{len(opens)} | {record['frame_name']}",
              font=_font(23), fill="white")
    draw.text((16, 42), run_label, font=_font(16), fill=(180, 190, 205))
    for index, (source, titles) in enumerate(zip(panel_order, titles_in_order)):
        panel = panels[source - 1]
        row, column = divmod(index, 5)
        left, right = int(edges[column]), int(edges[column + 1])
        top = header + row * (tile_height + gap)
        draw.rectangle((left, top, right - 1, top + label - 1), fill=(30, 34, 42))
        for text, font in zip(titles, (title_font, subtitle_font)):
            if draw.textbbox((0, 0), text, font=font)[2] > right - left - 16:
                raise ValueError(f"panel title does not fit: {text}")
        draw.text((left + 8, top + 5), titles[0], font=title_font, fill="white")
        draw.text((left + 8, top + 33), titles[1], font=subtitle_font, fill=(188, 200, 217))
        # Nearest-neighbor for binary/categorical display maps, no new threshold.
        resample = Image.Resampling.NEAREST if source in (6, 7, 10) else Image.Resampling.BILINEAR
        fitted = ImageOps.contain(panel, (right - left - 2, content_height), method=resample)
        canvas.paste(fitted, (left + (right - left - fitted.width) // 2,
                              top + label + (content_height - fitted.height) // 2))
    chart = open_close_timeline_chart(
        opens, closes, current_timestamp=t, width=width, height=chart_height,
        boundaries=boundaries, segment_names=segment_names,
    )
    canvas.paste(chart, (0, chart_y))
    draw.text((16, chart_y + chart_height + 5),
              "Per-frame base + seed commits | OPEN includes REOPEN | Birth/density/pruning are not lifecycle events | No retraining",
              font=_font(14), fill=(188, 200, 217))
    return canvas


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--chart-height", type=int, default=320)
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--preview-timestamp", type=int, nargs="*")
    parser.add_argument("--panel-order", type=int, nargs=10, default=DEFAULT_PANEL_ORDER,
                        help="One-based source panel IDs in destination order; titles are renumbered")
    args = parser.parse_args()
    try:
        panel_titles = ordered_panel_titles(args.panel_order)
    except ValueError as error:
        parser.error(str(error))
    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("fps must be finite and positive")
    run = args.run_dir.resolve()
    output = args.output_dir.resolve()
    if output == run or run in output.parents:
        parser.error("output must be outside the source run to preserve source artifacts")
    records = load_records(run)
    opens, closes = event_counts(records)
    config = json.loads((run / "configuration.json").read_text())
    bf = config["bayes_factor_threshold"]
    first_bf = config.get("first_open_bayes_factor_threshold") or bf
    gaussian = config.get("detector_gaussian_cue", "unchanged")
    pixel = config.get("detector_pixel_cue", "unchanged")
    label = (f"Saved run: first OPEN BF{first_bf:g} / CLOSE+REOPEN BF{bf:g} | "
             f"pixel={pixel}, Gaussian={gaussian} | u{config['representation_updates']} | "
             f"seed cap={config['da3_max_rows']} (0=uncapped)")
    if config.get("da3_birth_coverage"):
        label += f" | birth={config['da3_birth_coverage']}"
    boundaries = [i for i in range(1, len(records)) if records[i]["scene"] != records[i - 1]["scene"]]
    names = [str(records[i]["scene"]).replace("scene_change", "SC") for i in (0, *boundaries)]
    indices = args.preview_timestamp if args.preview_timestamp is not None else list(range(len(records)))
    if not indices or any(t < 0 or t >= len(records) for t in indices):
        parser.error("preview timestamps must be a nonempty subset of source timestamps")
    video = output / "viewer_10panels_lifecycle_all_frames.mp4"
    if args.preview_timestamp is None and video.exists():
        raise FileExistsError(f"refusing to replace completed video: {video}")
    frame_dir = output / ("previews" if args.preview_timestamp is not None else "frames")
    frame_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for t in indices:
        frame = compose_frame(load_panels(run, t), records[t], opens, closes,
                              width=args.width, chart_height=args.chart_height,
                              boundaries=boundaries, segment_names=names, run_label=label,
                              panel_order=args.panel_order)
        frame.save(frame_dir / f"{t:06d}.png")
        manifest.append({"timestamp": t, "frame_name": records[t]["frame_name"],
                         "open_events": opens[t], "close_events": closes[t],
                         "source_sha256": {suffix: sha256(run / "captures" / f"{t:06d}_{suffix}.png")
                                           for suffix in ("dashboard", *SIDECARS.values())}})
        if t % 25 == 0 or t == indices[-1]:
            print(f"Composed t={t}, size={frame.size}, OPEN={opens[t]}, CLOSE={closes[t]}", flush=True)
    if args.preview_timestamp is not None:
        return
    provenance = {
        "source_run": str(run), "source_configuration": config,
        "source_metadata_sha256": {name: sha256(run / name) for name in ("frame_metrics.csv", "configuration.json")},
        "frames": len(records), "fps": args.fps, "size": list(frame.size),
        "layout": "panels 1-5 / panels 6-10 / OPEN-CLOSE timeline",
        "panel_order_source_ids": list(args.panel_order),
        "panel_titles": panel_titles, "dashboard_recovered_panels": [2, 3, 7],
        "dashboard_recovery": "inverse 7x2 letterbox crop; original saved resolution, not rerendered",
        "event_definition": "base+seed per-frame lifecycle commits; OPEN includes REOPEN; excludes birth/density/pruning",
        "open_events_total": sum(opens), "close_events_total": sum(closes),
        "segment_boundaries": boundaries, "segment_names": names,
        "chart": "only prefix through current t; fixed full-run y scale and segment annotations are posthoc visualization only",
        "no_retraining": True, "manifest": manifest,
    }
    (output / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    subprocess.run([
        "ffmpeg", "-n", "-hide_banner", "-loglevel", "warning", "-framerate", str(args.fps),
        "-start_number", "0", "-i", str(frame_dir / "%06d.png"), "-frames:v", str(len(records)),
        "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        "-threads", "4", "-movflags", "+faststart", str(video),
    ], check=True)
    print(video, flush=True)


if __name__ == "__main__":
    main()
