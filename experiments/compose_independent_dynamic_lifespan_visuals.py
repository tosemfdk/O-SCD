"""Compose captured dynamic-lifespan runs into one SC1/SC2/SC3 video.

The source runs must have been executed with ``--visualization-dir`` so event
footprints are captured before each frame's dynamic-topology mutation.  This is
necessary for split/clone rows that have no row in the immutable reference PLY.

Either three independent runs under ``--run-root`` or one state-preserving
``continuous`` run under ``--continuous-run-dir`` may be visualized.  Only the
latter can show parameters and lifespan state surviving SC1/SC2/SC3 boundaries.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from experiments.run_ref_sc1_change_cue_density import (
    select_scope_records,
)
from experiments.visualize_lifespan_cue_events import (
    compose_frame_panel,
    open_close_timeline_chart,
    save_gif,
    threshold_render_rgb,
)


SCOPES = ("scene_change1", "scene_change2", "scene_change3")


def _frame_metrics(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    indexed = {int(row["timestamp"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"duplicate timestamps in {path}")
    return indexed


def _visual_dir(summary: Mapping[str, Any], run_dir: Path) -> Path:
    capture = summary.get("visual_capture")
    if not isinstance(capture, Mapping) or not capture.get("root"):
        raise ValueError(f"{run_dir} has no online visual capture")
    root = Path(str(capture["root"]))
    if not root.is_absolute() and not root.exists():
        root = run_dir / root
    return root


def _encode_mp4(gif_path: Path, mp4_path: Path) -> None:
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(gif_path),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-vf",
            "fps=25,pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(mp4_path),
        ],
        check=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-root", type=Path)
    source.add_argument("--continuous-run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--panel-width", type=int, default=300)
    parser.add_argument("--video-width", type=int, default=1200)
    parser.add_argument("--frame-duration-ms", type=int, default=160)
    parser.add_argument("--timeline-height", type=int, default=280)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from experiments.run_online_bayesian_lifespan_thaw import build_causal_records
    from experiments.train_cue_temporal_rchange import (
        build_fixed_cue_views,
        load_fixed_camera_index,
    )

    panel_dir = args.output_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    continuous_paths: list[Path] = []
    segment_paths: dict[str, list[Path]] = {}
    manifest: list[dict[str, Any]] = []
    source_summaries: dict[str, dict[str, Any]] = {}
    global_offset = 0
    if args.continuous_run_dir is not None:
        run_specs = (("continuous", args.continuous_run_dir),)
    else:
        run_specs = tuple((scope, args.run_root / scope) for scope in SCOPES)

    timeline_open_counts: list[int] = []
    timeline_close_counts: list[int] = []
    timeline_segments: list[str] = []
    for _scope, run_dir in run_specs:
        rows = _frame_metrics(run_dir / "frame_metrics.csv")
        if sorted(rows) != list(range(len(rows))):
            raise ValueError(f"{run_dir} frame metrics are not causally contiguous")
        for timestamp in range(len(rows)):
            row = rows[timestamp]
            timeline_open_counts.append(int(row["open_count"]))
            timeline_close_counts.append(int(row["close_count"]))
            stem = Path(row["frame"]).stem
            timeline_segments.append(stem.rsplit("_frame_", 1)[0])
    timeline_boundaries = [
        index
        for index in range(1, len(timeline_segments))
        if timeline_segments[index] != timeline_segments[index - 1]
    ]
    timeline_segment_names: list[str] = []
    for segment in timeline_segments:
        label = segment.replace("scene_change", "SC")
        if not timeline_segment_names or timeline_segment_names[-1] != label:
            timeline_segment_names.append(label)
    thresholded_dir = args.output_dir / "thresholded_render"
    thresholded_dir.mkdir(parents=True, exist_ok=True)
    timeline_width = int(args.panel_width) * 5 + 8 * 4

    for scope, run_dir in run_specs:
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        if summary.get("scope_key") != scope:
            raise ValueError(f"{run_dir} is not the expected {scope} run")
        source_summaries[scope] = summary
        run_arguments: Mapping[str, Any] = summary["run_arguments"]
        source_path = Path(str(run_arguments["source_path"]))
        cameras = load_fixed_camera_index(Path(str(run_arguments["fixed_cameras_json"])))
        cue_root = Path(str(run_arguments["cue_cache_root"]))
        resolution = float(run_arguments.get("resolution", 4.0))
        all_records, _ = build_causal_records(source_path)
        records = select_scope_records(
            all_records,
            scope=scope,
            max_frames=int(summary["frames"]),
        )
        metrics = _frame_metrics(run_dir / "frame_metrics.csv")
        visual_dir = _visual_dir(summary, run_dir)

        for record in records:
            local_timestamp = int(record.global_index)
            row = metrics[local_timestamp]
            view = build_fixed_cue_views(
                [record], cameras, cue_root, resolution
            )[0][0]
            stem = Path(record.name).stem
            raw_path = visual_dir / "raw_render" / f"{stem}.png"
            event_path = visual_dir / "event_render" / f"{stem}.png"
            if not raw_path.exists() or not event_path.exists():
                missing = raw_path if not raw_path.exists() else event_path
                raise FileNotFoundError(missing)
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
            with Image.open(event_path) as image:
                event = np.asarray(image.convert("RGB"))
            global_timestamp = (
                local_timestamp
                if scope == "continuous"
                else global_offset + local_timestamp
            )
            segment = str(record.segment_name)
            exact_thresholded_path = visual_dir / "thresholded_render" / f"{stem}.png"
            if exact_thresholded_path.exists():
                with Image.open(exact_thresholded_path) as image:
                    thresholded = np.asarray(image.convert("RGB"))
                threshold_source = "online_float_render"
            else:
                thresholded = threshold_render_rgb(raw, threshold=0.5)
                threshold_source = "derived_from_captured_rgb8"
            thresholded_path = (
                thresholded_dir / f"{global_timestamp:06d}_{stem}.png"
            )
            Image.fromarray(thresholded, mode="RGB").save(thresholded_path)
            timeline = open_close_timeline_chart(
                timeline_open_counts,
                timeline_close_counts,
                current_timestamp=global_timestamp,
                width=timeline_width,
                height=int(args.timeline_height),
                boundaries=timeline_boundaries,
                segment_names=timeline_segment_names,
            )
            panel = compose_frame_panel(
                rgb,
                cue,
                raw,
                event,
                timestamp=global_timestamp,
                segment=segment,
                open_count=int(row["open_count"]),
                close_count=int(row["close_count"]),
                panel_width=int(args.panel_width),
                thresholded_render=thresholded,
                lifecycle_chart=timeline,
            )
            panel_path = panel_dir / f"{global_timestamp:06d}_{stem}.png"
            panel.save(panel_path)
            segment_paths.setdefault(segment, []).append(panel_path)
            continuous_paths.append(panel_path)
            manifest.append(
                {
                    "global_timestamp": global_timestamp,
                    "local_timestamp": local_timestamp,
                    "scope": scope,
                    "segment": segment,
                    "frame": record.name,
                    "open_count": int(row["open_count"]),
                    "close_count": int(row["close_count"]),
                    "panel": str(panel_path),
                    "raw_render": str(raw_path),
                    "thresholded_render": str(thresholded_path),
                    "thresholded_render_source": threshold_source,
                    "event_render": str(event_path),
                }
            )
        if scope != "continuous":
            global_offset += len(records)

    render_support_modes = sorted(
        {
            str(
                summary.get("representation", {}).get(
                    "render_support_mode",
                    summary.get("run_arguments", {}).get(
                        "render_support_mode", "unknown"
                    ),
                )
            )
            for summary in source_summaries.values()
        }
    )
    render_slug = (
        render_support_modes[0]
        if len(render_support_modes) == 1
        else "mixed_render_support"
    )
    continuous_gif = (
        args.output_dir
        / f"ref_sc1_sc2_sc3_{render_slug}_lifespan_events_threshold_timeline.gif"
    )
    continuous_mp4 = (
        args.output_dir
        / f"ref_sc1_sc2_sc3_{render_slug}_lifespan_events_threshold_timeline.mp4"
    )
    save_gif(
        continuous_paths,
        continuous_gif,
        width=int(args.video_width),
        duration_ms=int(args.frame_duration_ms),
    )
    _encode_mp4(continuous_gif, continuous_mp4)

    segment_outputs: dict[str, dict[str, str]] = {}
    for scope, paths in segment_paths.items():
        directory = args.output_dir / scope
        gif_path = (
            directory
            / f"{scope}_{render_slug}_lifespan_events_threshold_timeline.gif"
        )
        mp4_path = (
            directory
            / f"{scope}_{render_slug}_lifespan_events_threshold_timeline.mp4"
        )
        save_gif(
            paths,
            gif_path,
            width=int(args.video_width),
            duration_ms=int(args.frame_duration_ms),
        )
        _encode_mp4(gif_path, mp4_path)
        segment_outputs[scope] = {"gif": str(gif_path), "mp4": str(mp4_path)}

    payload = {
        "schema_version": 2,
        "script": "experiments/compose_independent_dynamic_lifespan_visuals.py",
        "contract": "online-captured dynamic OPEN/CLOSE footprints with post-update R_change",
        "render_support_modes": render_support_modes,
        "input_mode": (
            "state_preserving_continuous"
            if args.continuous_run_dir is not None
            else "independent_concatenation"
        ),
        "run_root": str(args.run_root) if args.run_root is not None else None,
        "continuous_run_dir": (
            str(args.continuous_run_dir)
            if args.continuous_run_dir is not None
            else None
        ),
        "frames": len(continuous_paths),
        "continuous_gif": str(continuous_gif),
        "continuous_mp4": str(continuous_mp4),
        "segment_outputs": segment_outputs,
        "source_metrics": {
            scope: summary["metrics"] for scope, summary in source_summaries.items()
        },
        "event_colors": {"OPEN": "green", "CLOSE": "red"},
        "thresholded_render": {
            "threshold": 0.5,
            "score": "mean of the three rendered R_change channels",
            "rule": "score >= threshold",
            "fallback": "captured RGB8 when an exact online binary capture is unavailable",
        },
        "lifecycle_timeline": {
            "x_axis": "global image timestamp",
            "y_axis": "Gaussian event count",
            "series": ["OPEN", "CLOSE"],
            "causal_reveal": True,
            "scene_boundaries": timeline_boundaries,
        },
        "frame_duration_ms": int(args.frame_duration_ms),
        "runtime_seconds": time.time() - started,
        "frames_manifest": manifest,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "frames": payload["frames"],
                "continuous_mp4": payload["continuous_mp4"],
                "runtime_seconds": payload["runtime_seconds"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
