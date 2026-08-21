"""Plot and compare direct binary-state lifespan experiment outputs.

The runner writes only textual/machine-readable artifacts.  This utility turns
those artifacts into the required diagnostic PNGs without using ground truth for
anything beyond already-post-inference metrics stored in the output directory.
Manual boundaries are optional plot-only vertical guide lines.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:  # matplotlib is intentionally optional in the oscd env.
    from PIL import Image, ImageDraw, ImageFont
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Pillow is required for binary-state diagnostic PNGs") from exc

PER_RUN_PLOTS = (
    "binary_state_filter_metrics.png",
    "binary_state_lifecycle_counts.png",
    "binary_state_belief_histogram.png",
    "binary_state_transition_probability_timeline.png",
    "pre_vs_post_open_render.png",
)
COMPARISON_FILES = (
    "dc_only_vs_all_geometry.png",
    "comparison.json",
    "comparison.md",
)
STAT_COLUMNS = ("count", "mean", "q05", "q50", "q95")


def _read_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_cell(value: str) -> Any:
    value = value.strip()
    if value == "":
        return None
    if value in {"None", "nan", "NaN"}:
        return None
    if value[0:1] in {"{", "["}:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    try:
        if any(c in value for c in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _read_frame_metrics(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "frame_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as f:
        return [{k: _parse_cell(v) for k, v in row.items()} for row in csv.DictReader(f)]


def _load_npz(run_dir: Path) -> dict[str, np.ndarray]:
    path = run_dir / "per_frame_binary_state_stats.npz"
    if not path.exists():
        return {}
    data = np.load(path)
    return {name: data[name] for name in data.files}


def _same_scene_repeated_event_count(
    run_dir: Path,
    boundaries: Sequence[int] = (95, 199),
) -> int:
    path = run_dir / "lifecycle_events.jsonl"
    if not path.exists():
        return 0
    ordered_boundaries = tuple(sorted(int(value) for value in boundaries))
    counts: dict[tuple[int, int], int] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            event = json.loads(line)
            timestamp = int(event["decision_timestamp"])
            segment = sum(timestamp >= boundary for boundary in ordered_boundaries)
            key = (int(event["gaussian_index"]), int(segment))
            counts[key] = counts.get(key, 0) + 1
    return int(sum(max(0, count - 1) for count in counts.values()))


def _timestamps(rows: Sequence[Mapping[str, Any]], stats: Mapping[str, np.ndarray]) -> np.ndarray:
    if "timestamp" in stats:
        return np.asarray(stats["timestamp"], dtype=float)
    return np.asarray([float(row.get("timestamp", i)) for i, row in enumerate(rows)], dtype=float)


def _series(rows: Sequence[Mapping[str, Any]], name: str, *, default=np.nan) -> np.ndarray:
    vals = []
    for row in rows:
        value = row.get(name, default)
        vals.append(np.nan if value is None else float(value))
    return np.asarray(vals, dtype=float)


def _stat_series(rows: Sequence[Mapping[str, Any]], stats: Mapping[str, np.ndarray], name: str, column: str) -> np.ndarray:
    if name in stats:
        arr = np.asarray(stats[name], dtype=float)
        return arr[:, STAT_COLUMNS.index(column)]
    values = []
    for row in rows:
        stat = row.get(f"{name}_stats", {}) or {}
        value = stat.get(column)
        values.append(np.nan if value is None else float(value))
    return np.asarray(values, dtype=float)


def _font(size: int = 14, *, bold: bool = False):
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _color(index: int) -> tuple[int, int, int]:
    palette = [
        (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
        (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
        (188, 189, 34), (23, 190, 207),
    ]
    return palette[index % len(palette)]


def _finite_minmax(series: Sequence[np.ndarray], *, ymin: float | None = None, ymax: float | None = None) -> tuple[float, float]:
    vals = np.concatenate([np.asarray(v, dtype=float).ravel() for v in series if np.asarray(v).size]) if series else np.asarray([])
    vals = vals[np.isfinite(vals)]
    lo = float(np.min(vals)) if vals.size else 0.0
    hi = float(np.max(vals)) if vals.size else 1.0
    if ymin is not None:
        lo = float(ymin)
    if ymax is not None:
        hi = float(ymax)
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _draw_axes(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, xlabel: str, ylabel: str) -> None:
    left, top, right, bottom = box
    draw.rectangle(box, outline=(40, 40, 40), width=1)
    draw.text((left, top - 28), title, fill=(0, 0, 0), font=_font(18, bold=True))
    draw.text(((left + right) // 2 - 55, bottom + 28), xlabel, fill=(0, 0, 0), font=_font(12))
    draw.text((8, (top + bottom) // 2 - 8), ylabel, fill=(0, 0, 0), font=_font(12))


def _line_plot(path: Path, title: str, x: np.ndarray, series: Sequence[tuple[str, np.ndarray]], *, boundaries: Iterable[int] | None = None, ymin: float | None = 0.0, ymax: float | None = 1.0, second_series: Sequence[tuple[str, np.ndarray]] = ()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = 1100, 520
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    box = (78, 62, 920, 430)
    _draw_axes(draw, box, title, "global timestamp", "value")
    x = np.asarray(x, dtype=float)
    if not x.size:
        x = np.arange(1, dtype=float)
    xmin, xmax = _finite_minmax([x], ymin=None, ymax=None)
    if xmax <= xmin:
        xmax = xmin + 1.0
    all_series = [v for _, v in list(series) + list(second_series)]
    lo, hi = _finite_minmax(all_series, ymin=ymin, ymax=ymax)
    left, top, right, bottom = box
    def px(v): return left + int(round((float(v) - xmin) / (xmax - xmin) * (right - left)))
    def py(v): return bottom - int(round((float(v) - lo) / (hi - lo) * (bottom - top)))
    # grid
    for frac in np.linspace(0, 1, 5):
        y = int(round(bottom - frac * (bottom - top)))
        draw.line((left, y, right, y), fill=(230, 230, 230))
        draw.text((right + 6, y - 7), f"{lo + frac*(hi-lo):.2g}", fill=(80, 80, 80), font=_font(10))
    for boundary in boundaries or ():
        bx = px(boundary)
        draw.line((bx, top, bx, bottom), fill=(100, 100, 100), width=1)
    legend_x, legend_y = 940, 70
    for i, (label, vals) in enumerate(list(series) + list(second_series)):
        vals = np.asarray(vals, dtype=float)
        color = _color(i)
        pts = [(px(xx), py(vv)) for xx, vv in zip(x[: len(vals)], vals) if np.isfinite(vv)]
        if len(pts) >= 2:
            draw.line(pts, fill=color, width=3 if i < len(series) else 2)
        elif len(pts) == 1:
            draw.ellipse((pts[0][0] - 2, pts[0][1] - 2, pts[0][0] + 2, pts[0][1] + 2), fill=color)
        draw.rectangle((legend_x, legend_y + i * 22, legend_x + 14, legend_y + 14 + i * 22), fill=color)
        draw.text((legend_x + 20, legend_y - 2 + i * 22), label, fill=(0, 0, 0), font=_font(12))
    img.save(path)


def _hist_plot(path: Path, title: str, values: np.ndarray, *, thresholds: Sequence[tuple[float, str]] = ()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = 900, 500
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    box = (80, 62, 720, 410)
    _draw_axes(draw, box, title, "P(active)", "frames")
    left, top, right, bottom = box
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    counts, edges = np.histogram(vals, bins=np.linspace(0, 1, 41)) if vals.size else (np.zeros(40), np.linspace(0, 1, 41))
    maxc = max(1, int(counts.max()))
    bw = (right - left) / len(counts)
    for i, c in enumerate(counts):
        x0 = int(left + i * bw)
        x1 = int(left + (i + 1) * bw - 1)
        y0 = bottom - int((c / maxc) * (bottom - top))
        draw.rectangle((x0, y0, x1, bottom), fill=(76, 120, 168))
    for j, (thr, label) in enumerate(thresholds):
        x = int(left + float(thr) * (right - left))
        draw.line((x, top, x, bottom), fill=_color(j + 3), width=2)
        draw.text((735, 80 + j * 22), f"{label}: {thr:.2f}", fill=_color(j + 3), font=_font(12))
    img.save(path)


def _bar_compare(path: Path, title: str, labels: Sequence[str], groups: Sequence[tuple[str, Sequence[float]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = 1100, 540
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    draw.text((40, 24), title, fill=(0, 0, 0), font=_font(20, bold=True))
    box = (80, 70, 1000, 420)
    draw.rectangle(box, outline=(40, 40, 40))
    left, top, right, bottom = box
    vals = np.asarray([v for _, arr in groups for v in arr], dtype=float)
    vals = vals[np.isfinite(vals)]
    ymax = max(1.0, float(vals.max()) if vals.size else 1.0)
    n = len(labels); g = len(groups)
    slot_w = (right - left) / max(n, 1)
    bar_w = slot_w / (g + 1)
    for frac in np.linspace(0, 1, 5):
        y = int(bottom - frac * (bottom - top))
        draw.line((left, y, right, y), fill=(230, 230, 230))
        draw.text((right + 5, y - 7), f"{frac*ymax:.2g}", fill=(80, 80, 80), font=_font(10))
    for i, label in enumerate(labels):
        cx = left + i * slot_w
        draw.text((int(cx + 5), bottom + 12), label[:28], fill=(0, 0, 0), font=_font(10))
        for j, (_name, arr) in enumerate(groups):
            v = float(arr[i]) if i < len(arr) and np.isfinite(arr[i]) else 0.0
            x0 = int(cx + (j + 0.5) * bar_w)
            x1 = int(x0 + bar_w * 0.8)
            y0 = int(bottom - (v / ymax) * (bottom - top))
            draw.rectangle((x0, y0, x1, bottom), fill=_color(j))
    for j, (name, _arr) in enumerate(groups):
        draw.rectangle((80 + j * 150, 440, 94 + j * 150, 454), fill=_color(j))
        draw.text((100 + j * 150, 438), name, fill=(0, 0, 0), font=_font(12))
    img.save(path)

def _metric(summary: Mapping[str, Any], key: str, default: float | None = None) -> float | None:
    metrics = summary.get("metrics", {}) or {}
    if key in metrics and metrics[key] is not None:
        return float(metrics[key])
    post = metrics.get("post_opt", {}) or {}
    if key in post and post[key] is not None:
        return float(post[key])
    aliases = {
        "mIoU": "mean_frame_iou",
        "F1": "mean_frame_f1",
        "Precision": "precision",
        "Recall": "recall",
    }
    alias = aliases.get(key, key)
    if alias in metrics and metrics[alias] is not None:
        return float(metrics[alias])
    if alias in post and post[alias] is not None:
        return float(post[alias])
    return default


def plot_run(run_dir: Path, *, output_dir: Path | None = None, boundaries: Sequence[int] = (95, 199)) -> dict[str, Any]:
    run_dir = Path(run_dir)
    output_dir = Path(output_dir) if output_dir is not None else run_dir
    summary = _read_summary(run_dir)
    rows = _read_frame_metrics(run_dir)
    stats = _load_npz(run_dir)
    t = _timestamps(rows, stats)

    metric_series = []
    for name, label in (("iou", "post IoU"), ("f1", "post F1"), ("pre_iou", "pre IoU"), ("pre_f1", "pre F1")):
        vals = _series(rows, name)
        if np.isfinite(vals).any():
            metric_series.append((label, vals))
    if not metric_series:
        metric_series = [
            ("q median", _stat_series(rows, stats, "q", "q50")),
            ("P(active) q05", _stat_series(rows, stats, "p_active", "q05")),
            ("P(active) median", _stat_series(rows, stats, "p_active", "q50")),
            ("P(active) q95", _stat_series(rows, stats, "p_active", "q95")),
        ]
    _line_plot(
        output_dir / PER_RUN_PLOTS[0],
        "Binary-state filter metrics",
        t,
        metric_series,
        boundaries=boundaries,
        ymin=0.0,
        ymax=1.0,
    )

    _line_plot(
        output_dir / PER_RUN_PLOTS[1],
        "Binary-state lifecycle counts (log10(1 + count))",
        t,
        [
            ("OPEN", np.log10(1.0 + _series(rows, "open_count", default=0.0))),
            ("CLOSE", np.log10(1.0 + _series(rows, "close_count", default=0.0))),
            ("KEEP", np.log10(1.0 + _series(rows, "keep_count", default=0.0))),
            ("UNCERTAIN", np.log10(1.0 + _series(rows, "uncertain_count", default=0.0))),
            ("REOPEN", np.log10(1.0 + _series(rows, "reopen_count", default=0.0))),
            ("active GS", np.log10(1.0 + _series(rows, "active_lifespan_count", default=0.0))),
        ],
        boundaries=boundaries,
        ymin=0.0,
        ymax=None,
    )

    p_active_median = _stat_series(rows, stats, "p_active", "q50")
    _hist_plot(
        output_dir / PER_RUN_PLOTS[2],
        "Observed-row p_active median histogram",
        p_active_median,
        thresholds=(
            (float(summary.get("run_config", {}).get("close_probability", 0.4)), "close"),
            (float(summary.get("run_config", {}).get("open_probability", 0.6)), "open"),
        ),
    )

    _line_plot(
        output_dir / PER_RUN_PLOTS[3],
        "Binary-state transition probabilities",
        t,
        [
            ("P(active) median", _stat_series(rows, stats, "p_active", "q50")),
            ("P(flip) median", _stat_series(rows, stats, "p_flip", "q50")),
            ("P(0→1) q95", _stat_series(rows, stats, "p_01", "q95")),
            ("P(1→0) q95", _stat_series(rows, stats, "p_10", "q95")),
        ],
        boundaries=boundaries,
        ymin=0.0,
        ymax=1.0,
    )

    _line_plot(
        output_dir / PER_RUN_PLOTS[4],
        "Pre-vs-post optimization at OPEN frames",
        t,
        [
            ("pre IoU", _series(rows, "pre_iou")),
            ("post IoU", _series(rows, "iou")),
            ("pre positive frac", _series(rows, "pre_predicted_positive_fraction")),
            ("post positive frac", _series(rows, "post_predicted_positive_fraction")),
            ("OPEN frame", (_series(rows, "open_count", default=0.0) > 0).astype(float)),
        ],
        boundaries=boundaries,
        ymin=0.0,
        ymax=1.0,
    )

    return {"run_dir": str(run_dir), "output_dir": str(output_dir), "plots": [str(output_dir / name) for name in PER_RUN_PLOTS]}


def _condition_summary(label: str, run_dir: Path) -> dict[str, Any]:
    summary = _read_summary(run_dir)
    rows = _read_frame_metrics(run_dir)
    metrics = summary.get("metrics", {}) or {}
    open_delta = metrics.get("open_frame_pre_post", {}) or {}
    return {
        "label": label,
        "run_dir": str(run_dir),
        "mIoU": _metric(summary, "mean_frame_iou"),
        "F1": _metric(summary, "mean_frame_f1"),
        "precision": _metric(summary, "precision"),
        "recall": _metric(summary, "recall"),
        "pre_opt_mIoU": ((metrics.get("pre_opt", {}) or {}).get("mean_frame_iou") if isinstance(metrics.get("pre_opt"), dict) else None),
        "post_opt_mIoU": ((metrics.get("post_opt", {}) or {}).get("mean_frame_iou") if isinstance(metrics.get("post_opt"), dict) else _metric(summary, "mean_frame_iou")),
        "open_count": int(summary.get("open_count", sum(int(r.get("open_count", 0) or 0) for r in rows))),
        "close_count": int(summary.get("close_count", sum(int(r.get("close_count", 0) or 0) for r in rows))),
        "keep_count": int(sum(int(r.get("keep_count", 0) or 0) for r in rows)),
        "uncertain_count": int(sum(int(r.get("uncertain_count", 0) or 0) for r in rows)),
        "reopen_count": int(summary.get("reopen_count", sum(int(r.get("reopen_count", 0) or 0) for r in rows))),
        "same_scene_repeated_transition_event_count": int(
            summary.get(
                "same_scene_repeated_transition_event_count",
                _same_scene_repeated_event_count(run_dir),
            )
        ),
        "final_active_gs": int(summary.get("final_active_gs", rows[-1].get("final_active_gs", rows[-1].get("active_lifespan_count", 0)) if rows else 0)),
        "p_active_q05_median_q95": [float(np.nanmedian(_stat_series(rows, _load_npz(run_dir), "p_active", c))) for c in ("q05", "q50", "q95")],
        "p_flip_q05_median_q95": [float(np.nanmedian(_stat_series(rows, _load_npz(run_dir), "p_flip", c))) for c in ("q05", "q50", "q95")],
        "p_01_mean_q95": [float(np.nanmean(_stat_series(rows, _load_npz(run_dir), "p_01", "mean"))), float(np.nanmedian(_stat_series(rows, _load_npz(run_dir), "p_01", "q95")))],
        "p_10_mean_q95": [float(np.nanmean(_stat_series(rows, _load_npz(run_dir), "p_10", "mean"))), float(np.nanmedian(_stat_series(rows, _load_npz(run_dir), "p_10", "q95")))],
        "open_frame_pre_post_delta_iou": open_delta.get("mean_delta_iou"),
        "base_max_drift": float(summary.get("base_max_drift", (summary.get("base_tensor_drift", {}) or {}).get("max_abs", 0.0))),
        "closed_slot_max_drift": float(summary.get("closed_slot_max_drift", 0.0)),
        "inactive_gradient_first_step_violations": float(
            summary.get(
                "inactive_gradient_first_step_violations",
                summary.get("inactive_gradient_violations", 0.0),
            )
        ),
        "false_active_to_active_splits": int(summary.get("active_to_active_false_split_count", 0)),
        "reused_slot_violations": int(summary.get("reused_slot_violations", 0)),
        "runtime_seconds": float(summary.get("runtime_seconds", np.nan)),
        "peak_cuda_memory_bytes": int(summary.get("cuda_peak_memory_bytes", 0)),
    }


def compare_runs(dc_only_dir: Path, all_geometry_dir: Path, *, output_dir: Path, beam2_summary: Path | None = None) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    conditions = [
        _condition_summary("direct-binary DC-only", Path(dc_only_dir)),
        _condition_summary("direct-binary all-geometry", Path(all_geometry_dir)),
    ]
    if beam2_summary is not None:
        beam_summary_path = Path(beam2_summary)
        beam = json.loads(beam_summary_path.read_text(encoding="utf-8"))
        beam_dir = beam_summary_path.parent
        beam_rows = _read_frame_metrics(beam_dir)
        beam_open_count = int(
            beam.get(
                "open_count",
                sum(int(row.get("open_count", 0) or 0) for row in beam_rows),
            )
        )
        beam_close_count = int(
            beam.get(
                "close_count",
                sum(int(row.get("close_count", 0) or 0) for row in beam_rows),
            )
        )
        beam_final_active = int(
            beam.get(
                "final_active_gs",
                beam_rows[-1].get("active_lifespan_count", 0) if beam_rows else 0,
            )
        )
        conditions.append({
            "label": "Beam-2 all-geometry baseline",
            "run_dir": str(beam_dir),
            "mIoU": _metric(beam, "mean_frame_iou"),
            "F1": _metric(beam, "mean_frame_f1"),
            "precision": _metric(beam, "precision"),
            "recall": _metric(beam, "recall"),
            "open_count": beam_open_count,
            "close_count": beam_close_count,
            "reopen_count": int(beam.get("reopen_count", 0)),
            "same_scene_repeated_transition_event_count": int(
                beam.get(
                    "same_scene_repeated_transition_event_count",
                    _same_scene_repeated_event_count(beam_dir),
                )
            ),
            "final_active_gs": beam_final_active,
            "runtime_seconds": float(beam.get("runtime_seconds", np.nan)),
            "peak_cuda_memory_bytes": int(beam.get("cuda_peak_memory_bytes", 0)),
        })

    labels = [c["label"] for c in conditions]
    quality = [(m, [np.nan if c.get(m) is None else float(c.get(m)) for c in conditions]) for m in ("mIoU", "F1", "precision", "recall")]
    # Keep the quality bars on one interpretable [0,1] scale. Lifecycle counts
    # remain available without lossy rescaling in comparison.json/.md and in
    # each run's lifecycle timeline.
    _bar_compare(
        output_dir / "dc_only_vs_all_geometry.png",
        "DC-only vs all-geometry binary-state ablation",
        labels,
        quality,
    )

    comparison = {"conditions": conditions, "files": list(COMPARISON_FILES)}
    (output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Direct binary state lifespan comparison", "", "| Condition | mIoU | F1 | Precision | Recall | OPEN | CLOSE | REOPEN | runtime(s) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in conditions:
        def fmt(v): return "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{float(v):.4f}"
        lines.append(f"| {c['label']} | {fmt(c.get('mIoU'))} | {fmt(c.get('F1'))} | {fmt(c.get('precision'))} | {fmt(c.get('recall'))} | {c.get('open_count', '')} | {c.get('close_count', '')} | {c.get('reopen_count', '')} | {fmt(c.get('runtime_seconds'))} |")
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return comparison


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None, help="single run output directory for per-run plots")
    parser.add_argument("--dc-only-dir", type=Path, default=None, help="DC-only run output directory for comparison")
    parser.add_argument("--all-geometry-dir", type=Path, default=None, help="all-geometry run output directory for comparison")
    parser.add_argument("--beam2-summary", type=Path, default=None, help="optional Beam-2 summary.json baseline")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--boundaries", nargs="*", type=int, default=[95, 199])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results: dict[str, Any] = {}
    if args.run_dir is not None:
        results["run"] = plot_run(args.run_dir, output_dir=args.output_dir, boundaries=tuple(args.boundaries))
    if args.dc_only_dir is not None or args.all_geometry_dir is not None:
        if args.dc_only_dir is None or args.all_geometry_dir is None:
            raise SystemExit("--dc-only-dir and --all-geometry-dir must be provided together")
        output_dir = args.output_dir or (Path(args.all_geometry_dir).parent / "binary_state_comparison")
        results["comparison"] = compare_runs(args.dc_only_dir, args.all_geometry_dir, output_dir=output_dir, beam2_summary=args.beam2_summary)
    if not results:
        raise SystemExit("provide --run-dir or both --dc-only-dir/--all-geometry-dir")
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
