"""Summarize the continuous single-candidate Beta threshold sweep.

The script compares ``BF in {10,30,100,300}`` runs against an existing direct
binary continuous baseline and writes a machine-readable JSON, a flat CSV, and
one compact comparison plot.  Ground-truth metrics are read only from completed
run summaries; this script never participates in the causal experiment loop.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_BASELINE = Path(
    "outputs/escd_open_never_dc_opacity_oscd_densify_continuous_u16_seed0_20260825"
)
DEFAULT_THRESHOLDS = (10.0, 30.0, 100.0, 300.0)


def _load_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("scope_key") != "continuous" or int(summary.get("frames", 0)) != 304:
        raise ValueError(f"{path} is not a complete 304-frame continuous run")
    return summary


def _segment_map(metrics: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row["segment"]): row for row in metrics.get("segments", [])}


def _invariants(summary: Mapping[str, Any]) -> dict[str, Any]:
    audit = summary.get("closed_row_persistence_audit", {})
    return {
        "closed_drift_passed": audit.get("passed"),
        "closed_drift_max_abs": audit.get("max_abs"),
        "inactive_gradient_violations": int(
            summary.get("inactive_gradient_violations", 0)
        ),
        "topology_integrity": bool(summary.get("topology_integrity", False)),
        "future_view_access_count": int(
            summary.get("density_control", {}).get("future_view_access_count", 0)
        ),
        "gt_used_in_causal_loop": bool(summary.get("gt_used_in_causal_loop", True)),
    }


def _representation_repeated_events(
    run_dir: Path, boundaries: Sequence[int] = (95, 199)
) -> int:
    """Count only repeated OPEN/CLOSE mutations, excluding same-label resets."""

    path = run_dir / "lifecycle_events.jsonl"
    per_gaussian_segment: dict[tuple[int, int], int] = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            event = json.loads(line)
            if event.get("action") not in {"OPEN", "CLOSE"}:
                continue
            timestamp = int(event["decision_timestamp"])
            segment = sum(timestamp >= int(boundary) for boundary in boundaries)
            key = (int(event["gaussian_index"]), segment)
            per_gaussian_segment[key] = per_gaussian_segment.get(key, 0) + 1
    return int(sum(max(0, count - 1) for count in per_gaussian_segment.values()))


def _row(label: str, run_dir: Path, summary: Mapping[str, Any]) -> dict[str, Any]:
    metrics = summary["metrics"]
    segments = _segment_map(metrics)
    candidate = summary.get("candidate_diagnostics", {})
    detector_config = summary.get("detector", {}).get("single_candidate_beta")
    threshold = (
        None
        if not isinstance(detector_config, Mapping)
        else float(detector_config["bayes_factor_threshold"])
    )
    row: dict[str, Any] = {
        "label": label,
        "run_dir": str(run_dir),
        "detector": summary.get("detector", {}).get("algorithm"),
        "bayes_factor_threshold": threshold,
        "log_bayes_factor_threshold": (
            None if threshold is None else math.log(threshold)
        ),
        "mean_frame_iou": float(metrics["mean_frame_iou"]),
        "mean_frame_f1": float(metrics["mean_frame_f1"]),
        "aggregate_iou": float(metrics["aggregate_iou"]),
        "aggregate_f1": float(metrics["aggregate_f1"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "open_count": int(summary.get("open_count", 0)),
        "close_count": int(summary.get("close_count", 0)),
        "reopen_count": int(summary.get("reopen_count", 0)),
        "same_scene_repeated_transition_event_count": int(
            summary.get("same_scene_repeated_transition_event_count", 0)
        ),
        "same_scene_representation_repeated_transition_event_count": (
            _representation_repeated_events(run_dir)
        ),
        "candidate_started_count": int(candidate.get("candidate_started_count", 0)),
        "candidate_rejected_count": int(candidate.get("candidate_rejected_count", 0)),
        "candidate_committed_count": int(candidate.get("candidate_committed_count", 0)),
        "max_live_candidate_count": int(candidate.get("max_live_candidate_count", 0)),
        "final_live_candidate_count": int(candidate.get("final_live_candidate_count", 0)),
        "initial_gaussian_count": int(
            summary.get("density_control", {}).get("initial_gaussian_count", 0)
        ),
        "final_gaussian_count": int(
            summary.get("density_control", {}).get("final_gaussian_count", 0)
        ),
        "runtime_seconds": float(summary.get("runtime_seconds", 0.0)),
        "peak_cuda_memory_bytes": int(summary.get("peak_cuda_memory_bytes", 0)),
        **_invariants(summary),
    }
    for segment in ("scene_change1", "scene_change2", "scene_change3"):
        values = segments.get(segment)
        row[f"{segment}_mean_frame_iou"] = (
            None if values is None else float(values["mean_frame_iou"])
        )
        row[f"{segment}_mean_frame_f1"] = (
            None if values is None else float(values["mean_frame_f1"])
        )
    return row


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    from PIL import Image, ImageDraw, ImageFont

    canvas = Image.new("RGB", (1500, 920), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 26)
        panel_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:  # pragma: no cover - font availability is host-specific.
        title_font = panel_font = font = ImageFont.load_default()
    draw.text(
        (750, 12),
        "Single-candidate Beta changepoint threshold sweep",
        fill="black",
        font=title_font,
        anchor="ma",
    )

    panels = ((35, 65, 735, 450), (765, 65, 1465, 450), (35, 500, 735, 885), (765, 500, 1465, 885))
    palette = ("#2878b5", "#e07a1f", "#3a9d5d", "#b34789")

    def axes(box, title: str, labels: Sequence[str], maximum: float, *, log=False):
        x0, y0, x1, y1 = box
        draw.rounded_rectangle(box, radius=8, outline="#8a949e", width=2)
        draw.text(((x0 + x1) // 2, y0 + 10), title, fill="black", font=panel_font, anchor="ma")
        plot = (x0 + 72, y0 + 52, x1 - 20, y1 - 48)
        px0, py0, px1, py1 = plot
        for tick in range(5):
            fraction = tick / 4
            y = int(py1 - fraction * (py1 - py0))
            draw.line((px0, y, px1, y), fill="#e0e4e8", width=1)
            value = (10 ** (fraction * math.log10(maximum))) if log else fraction * maximum
            label = f"{value:,.0f}" if log else f"{value:.2f}"
            draw.text((px0 - 8, y), label, fill="#48515a", font=font, anchor="rm")
        draw.line((px0, py0, px0, py1), fill="black", width=2)
        draw.line((px0, py1, px1, py1), fill="black", width=2)
        step = (px1 - px0) / max(1, len(labels))
        centers = [px0 + step * (index + 0.5) for index in range(len(labels))]
        for center, label in zip(centers, labels):
            draw.text((center, py1 + 12), label, fill="black", font=font, anchor="ma")
        return plot, centers

    labels = [str(row["label"]) for row in rows]
    plot, centers = axes(panels[0], "Segment mean-frame mIoU", labels, 0.75)
    px0, py0, px1, py1 = plot
    group_width = (px1 - px0) / len(labels) * 0.72
    bar_width = group_width / 3
    segments = (("scene_change1", "SC1"), ("scene_change2", "SC2"), ("scene_change3", "SC3"))
    for segment_index, (segment, legend) in enumerate(segments):
        color = palette[segment_index]
        for center, row in zip(centers, rows):
            value = float(row[f"{segment}_mean_frame_iou"])
            left = center - group_width / 2 + segment_index * bar_width
            top = py1 - value / 0.75 * (py1 - py0)
            draw.rectangle((left, top, left + bar_width - 2, py1), fill=color)
        draw.rectangle((px0 + segment_index * 90, py0 + 5, px0 + segment_index * 90 + 18, py0 + 19), fill=color)
        draw.text((px0 + segment_index * 90 + 24, py0 + 12), legend, fill="black", font=font, anchor="lm")

    plot, centers = axes(panels[1], "Overall mean-frame quality", labels, 0.8)
    px0, py0, px1, py1 = plot
    for series_index, (key, legend) in enumerate((("mean_frame_iou", "mIoU"), ("mean_frame_f1", "F1"))):
        points = [
            (center, py1 - float(row[key]) / 0.8 * (py1 - py0))
            for center, row in zip(centers, rows)
        ]
        draw.line(points, fill=palette[series_index], width=4)
        for point in points:
            draw.ellipse((point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill=palette[series_index])
        draw.text((px0 + series_index * 95, py0 + 12), legend, fill=palette[series_index], font=panel_font, anchor="lm")

    event_max = max(max(int(row[key]) for row in rows) for key in ("open_count", "close_count", "reopen_count"))
    plot, centers = axes(panels[2], "Lifecycle events (log scale)", labels, float(event_max), log=True)
    px0, py0, px1, py1 = plot
    log_max = math.log10(max(1, event_max))
    for series_index, (key, legend) in enumerate((("open_count", "OPEN"), ("close_count", "CLOSE"), ("reopen_count", "REOPEN"))):
        points = [
            (center, py1 - math.log10(max(1, int(row[key]))) / log_max * (py1 - py0))
            for center, row in zip(centers, rows)
        ]
        draw.line(points, fill=palette[series_index], width=4)
        for point in points:
            draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill=palette[series_index])
        draw.text((px0 + series_index * 105, py0 + 12), legend, fill=palette[series_index], font=font, anchor="lm")

    candidate_rows = rows[1:]
    candidate_labels = [str(row["label"]) for row in candidate_rows]
    candidate_max = max(max(int(row[key]) for row in candidate_rows) for key in ("candidate_started_count", "candidate_rejected_count", "candidate_committed_count"))
    plot, centers = axes(panels[3], "Candidate decisions (log scale)", candidate_labels, float(candidate_max), log=True)
    px0, py0, px1, py1 = plot
    log_max = math.log10(max(1, candidate_max))
    for series_index, (key, legend) in enumerate((("candidate_started_count", "START"), ("candidate_rejected_count", "REJECT"), ("candidate_committed_count", "COMMIT"))):
        points = [
            (center, py1 - math.log10(max(1, int(row[key]))) / log_max * (py1 - py0))
            for center, row in zip(centers, candidate_rows)
        ]
        draw.line(points, fill=palette[series_index], width=4)
        for point in points:
            draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill=palette[series_index])
        draw.text((px0 + series_index * 105, py0 + 12), legend, fill=palette[series_index], font=font, anchor="lm")

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--baseline-run-dir", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    baseline_summary = _load_summary(args.baseline_run_dir)
    rows = [_row("direct_binary", args.baseline_run_dir, baseline_summary)]
    for threshold in DEFAULT_THRESHOLDS:
        label = f"bf{int(threshold)}"
        run_dir = args.sweep_root / label
        summary = _load_summary(run_dir)
        actual = float(
            summary["detector"]["single_candidate_beta"]["bayes_factor_threshold"]
        )
        if actual != threshold:
            raise ValueError(f"{run_dir} has BF={actual}, expected {threshold}")
        rows.append(_row(label, run_dir, summary))

    baseline_iou = float(rows[0]["mean_frame_iou"])
    baseline_f1 = float(rows[0]["mean_frame_f1"])
    comparisons = []
    for row in rows[1:]:
        comparisons.append(
            {
                **row,
                "delta_mean_frame_iou_vs_direct": float(row["mean_frame_iou"])
                - baseline_iou,
                "delta_mean_frame_f1_vs_direct": float(row["mean_frame_f1"])
                - baseline_f1,
            }
        )
    best = max(comparisons, key=lambda row: float(row["mean_frame_iou"]))
    payload = {
        "schema_version": 1,
        "script": "experiments/summarize_single_candidate_beta_threshold_sweep.py",
        "baseline": rows[0],
        "sweep": comparisons,
        "best_by_mean_frame_iou": best["label"],
        "all_invariants_passed": all(
            bool(row["closed_drift_passed"])
            and int(row["inactive_gradient_violations"]) == 0
            and bool(row["topology_integrity"])
            and int(row["future_view_access_count"]) == 0
            and not bool(row["gt_used_in_causal_loop"])
            for row in rows
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(args.output_dir / "comparison.csv", rows)
    _plot(args.output_dir / "comparison.png", rows)
    print(
        json.dumps(
            {
                "comparison": str(args.output_dir / "comparison.json"),
                "plot": str(args.output_dir / "comparison.png"),
                "best": payload["best_by_mean_frame_iou"],
                "all_invariants_passed": payload["all_invariants_passed"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
