"""Summarize A0--A4 oracle-boundary runs and generate requested diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

MODES = ("dc_only", "dc_opacity", "geo_adam", "geo_sgld", "geo_mcmc")
LABELS = {
    "dc_only": "A0 DC",
    "dc_opacity": "A1 DC+opacity",
    "geo_adam": "A2 geometry Adam",
    "geo_sgld": "A3 +SGLD",
    "geo_mcmc": "A4 +relocation",
}
COLORS = dict(zip(MODES, plt.cm.tab10.colors[: len(MODES)]))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def iter_csv(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def boundary_offset(timestamp: int, boundaries: Iterable[int]) -> int:
    start = 0
    for boundary in boundaries:
        if timestamp >= int(boundary):
            start = int(boundary)
        else:
            break
    return timestamp - start


def mean_by_x(rows: list[dict[str, str]], x_key: str, y_key: str) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        x = float_or_none(row.get(x_key))
        y = float_or_none(row.get(y_key))
        if x is not None and y is not None:
            grouped[int(x)].append(y)
    xs = np.asarray(sorted(grouped), dtype=np.float64)
    ys = np.asarray([np.mean(grouped[int(x)]) for x in xs], dtype=np.float64)
    return xs, ys


def save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), q, method="higher"))


def build_comparison(root: Path, modes: tuple[str, ...]) -> dict[str, Any]:
    runs: dict[str, Any] = {}
    for mode in modes:
        run = root / mode
        train = read_json(run / "summary.json")
        evaluation = read_json(run / "evaluation" / "summary.json")
        counts = read_csv(run / "gaussian_counts.csv")
        relative = [
            abs(value)
            for row in iter_csv(run / "relocation_audit.csv")
            if row.get("event_type") == "aggregate"
            and (value := float_or_none(row.get("relative_delta_total_energy"))) is not None
        ]
        n_values = sorted({int(float(row["N_t"])) for row in counts if row.get("N_t")})
        replay = train.get("training_audit", {}).get("archive_replay_audit") or []
        dynamics = train.get("training_audit", {}).get("mcmc_dynamics", {})
        runs[mode] = {
            "label": LABELS[mode],
            "run_dir": str(run),
            "mIoU": evaluation.get("full_state_posthoc_mIoU"),
            "mF1": evaluation.get("full_state_posthoc_mF1"),
            "seen_prefix_mIoU": evaluation.get("seen_prefix_mIoU"),
            "seen_prefix_mF1": evaluation.get("seen_prefix_mF1"),
            "per_state": evaluation.get("per_state", []),
            "runtime_seconds": train.get("runtime_seconds"),
            "runtime_per_frame_seconds": train.get("runtime_per_frame_seconds"),
            "peak_cuda_memory_bytes": train.get("peak_cuda_memory_bytes"),
            "gaussian_count_values": n_values,
            "gaussian_count_invariant": len(n_values) == 1 and n_values[0] == int(train.get("gaussian_count", n_values[0])) if n_values else False,
            "archive_parameter_drift_zero": all(bool(item.get("parameter_drift_zero")) for item in replay) if replay else mode == "dc_only",
            "archive_render_drift_zero": all(bool(item.get("render_drift_zero")) for item in replay) if replay else mode == "dc_only",
            "relocation_event_count": dynamics.get("relocation_applied_count", 0),
            "relocation_audit_count": len(relative),
            "relocation_relative_energy_abs_p95": quantile(relative, 0.95),
            "relocation_relative_energy_abs_max": max(relative) if relative else None,
            "topology_ops": train.get("topology_ops", {}),
            "input_hash": train.get("input_hash"),
        }
    hashes = {value.get("input_hash") for value in runs.values()}
    comparison = {
        "schema_version": 1,
        "protocol_root": str(root),
        "headline_metric": "arithmetic mean of per-frame IoU/F1",
        "modes": runs,
        "input_hash_match": len(hashes) == 1,
        "deltas": {},
    }
    pairs = (("geo_adam", "dc_only"), ("geo_sgld", "geo_adam"), ("geo_mcmc", "geo_sgld"))
    for newer, older in pairs:
        comparison["deltas"][f"{newer}_minus_{older}"] = {
            metric: (float(runs[newer][metric]) - float(runs[older][metric]))
            if runs[newer][metric] is not None and runs[older][metric] is not None
            else None
            for metric in ("mIoU", "mF1", "seen_prefix_mIoU", "seen_prefix_mF1")
        }
    return comparison


def plot_adaptation(root: Path, out: Path, modes: tuple[str, ...]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True)
    for mode in modes:
        rows = read_csv(root / mode / "evaluation" / "online_arrival_metrics.csv")
        for axis, metric, title in zip(axes, ("iou", "f1"), ("Arrival-time IoU", "Arrival-time F1")):
            xs, ys = mean_by_x(rows, "online_frames_since_boundary", metric)
            if xs.size:
                axis.plot(xs, ys, label=LABELS[mode], color=COLORS[mode])
            axis.set_title(title)
            axis.set_xlabel("Frames since oracle boundary")
            axis.set_ylabel(metric.upper())
            axis.grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    save_figure(out / "adaptation_curve.png")


def plot_energy(root: Path, out: Path, modes: tuple[str, ...], boundaries: tuple[int, ...]) -> None:
    plt.figure(figsize=(8, 4.5))
    for mode in modes:
        rows = read_csv(root / mode / "training_energy.csv")
        collapsed: dict[int, list[float]] = defaultdict(list)
        for row in rows:
            timestamp = float_or_none(row.get("timestamp"))
            energy = float_or_none(row.get("energy"))
            if timestamp is not None and energy is not None:
                collapsed[boundary_offset(int(timestamp), boundaries)].append(energy)
        xs = sorted(collapsed)
        if xs:
            plt.plot(xs, [np.mean(collapsed[x]) for x in xs], label=LABELS[mode], color=COLORS[mode])
    plt.xlabel("Frames since oracle boundary")
    plt.ylabel("Target energy")
    plt.title("Target energy adaptation")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    save_figure(out / "energy_adaptation_curve.png")


def plot_coverage(root: Path, out: Path, modes: tuple[str, ...]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True)
    for mode in modes:
        rows = read_csv(root / mode / "evaluation" / "per_frame_metrics.csv")
        for axis, metric, title in zip(
            axes,
            ("cue_positive_mass_coverage", "rendered_alpha_coverage"),
            ("Cue-positive mass coverage", "Rendered alpha coverage"),
        ):
            xs, ys = mean_by_x(rows, "frames_since_boundary", metric)
            if xs.size:
                axis.plot(xs, ys, label=LABELS[mode], color=COLORS[mode])
            axis.set_xlabel("Frames since oracle boundary")
            axis.set_ylabel("Coverage")
            axis.set_title(title)
            axis.grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    save_figure(out / "cue_alpha_coverage_curve.png")


def plot_capacity(root: Path, out: Path, modes: tuple[str, ...]) -> None:
    fig, axes = plt.subplots(len(modes), 1, figsize=(10, 2.0 * len(modes)), sharex=True)
    for axis, mode in zip(np.atleast_1d(axes), modes):
        rows = read_csv(root / mode / "gaussian_counts.csv")
        x = [int(float(r["global_step"])) for r in rows]
        live = [float(r["live_count"]) for r in rows]
        dead = [float(r["dead_count"]) for r in rows]
        axis.plot(x, live, label="live", color="#2ca02c", linewidth=1)
        axis.plot(x, dead, label="dead", color="#d62728", linewidth=1)
        axis.set_ylabel(LABELS[mode], fontsize=8)
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8, ncol=2)
    axes[-1].set_xlabel("Global update")
    fig.suptitle("Fixed-capacity live/dead counts", y=1.01)
    save_figure(out / "alive_dead_gaussians.png")


def plot_relocation(root: Path, out: Path) -> None:
    grouped: dict[int, list[float]] = defaultdict(list)
    events_path = root / "geo_mcmc" / "relocation_events.jsonl"
    if events_path.exists():
        with events_path.open(encoding="utf-8") as handle:
            for line in handle:
                event = json.loads(line)
                if event.get("event_type") == "source":
                    grouped[int(event["global_iteration"])].append(float(event["source_target_distance"]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    xs = sorted(grouped)
    axes[0].bar(xs, [len(grouped[x]) for x in xs], width=max(1, (xs[1] - xs[0]) * 0.7) if len(xs) > 1 else 1)
    axes[0].set_title("Relocations per event")
    axes[0].set_xlabel("Global update")
    axes[0].set_ylabel("Source count")
    distances = [distance for values in grouped.values() for distance in values]
    axes[1].hist(distances, bins=50, color=COLORS["geo_mcmc"], alpha=0.85)
    axes[1].set_title("Relocation source-target distance")
    axes[1].set_xlabel("3D distance")
    axes[1].set_ylabel("Count")
    save_figure(out / "relocation_count_distance.png")


def plot_relocation_energy(root: Path, out: Path) -> None:
    values = [
        value
        for row in iter_csv(root / "geo_mcmc" / "relocation_audit.csv")
        if row.get("event_type") == "aggregate"
        and (value := float_or_none(row.get("relative_delta_total_energy"))) is not None
    ]
    plt.figure(figsize=(7, 4.2))
    plt.hist(values, bins=min(40, max(5, len(values))), color=COLORS["geo_mcmc"], alpha=0.85)
    plt.axvline(0.05, color="red", linestyle="--", label="+5% stop gate")
    plt.axvline(-0.05, color="red", linestyle="--")
    plt.xlabel("Relative total-energy change")
    plt.ylabel("Relocation audits")
    plt.title("Pure relocation energy perturbation")
    plt.legend()
    save_figure(out / "relocation_delta_energy_histogram.png")


def plot_displacement(root: Path, out: Path, modes: tuple[str, ...]) -> None:
    plt.figure(figsize=(8, 4.5))
    for mode in modes:
        train = read_json(root / mode / "summary.json")
        diagnostics = train.get("training_audit", {}).get("geometry_diagnostics", {})
        if not diagnostics:
            continue
        final_key = max(diagnostics, key=lambda key: int(key))
        data = diagnostics[final_key].get("xyz_displacement_vs_base", {})
        counts = np.asarray(data.get("histogram_counts", []), dtype=np.float64)
        edges = np.asarray(data.get("histogram_edges", []), dtype=np.float64)
        if counts.size and edges.size == counts.size + 1:
            centers = 0.5 * (edges[:-1] + edges[1:])
            plt.plot(centers, counts / max(counts.sum(), 1.0), label=f"{LABELS[mode]} S2", color=COLORS[mode])
    plt.xlabel("XYZ displacement from reference")
    plt.ylabel("Mean state histogram mass")
    plt.title("Gaussian displacement distribution")
    plt.yscale("log")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    save_figure(out / "xyz_displacement_histogram.png")


def combine_contact_sheets(root: Path, out: Path, modes: tuple[str, ...]) -> None:
    items = []
    for mode in modes:
        path = root / mode / "evaluation" / "contact_sheet.png"
        if path.exists():
            items.append((mode, Image.open(path).convert("RGB")))
    if not items:
        return
    width = max(image.width for _mode, image in items)
    label_height = 34
    resized = []
    for mode, image in items:
        if image.width != width:
            height = round(image.height * width / image.width)
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        resized.append((mode, image))
    canvas = Image.new("RGB", (width, sum(image.height + label_height for _mode, image in resized)), "white")
    draw = ImageDraw.Draw(canvas)
    y = 0
    for mode, image in resized:
        draw.text((10, y + 8), LABELS[mode], fill="black")
        y += label_height
        canvas.paste(image, (0, y))
        y += image.height
    canvas.save(out / "state_confusion_contact_sheet.png")


def plot_comparison(comparison: dict[str, Any], out: Path, modes: tuple[str, ...]) -> None:
    x = np.arange(len(modes))
    miou = [comparison["modes"][m]["mIoU"] or 0.0 for m in modes]
    mf1 = [comparison["modes"][m]["mF1"] or 0.0 for m in modes]
    width = 0.38
    plt.figure(figsize=(9, 4.5))
    plt.bar(x - width / 2, miou, width, label="mIoU")
    plt.bar(x + width / 2, mf1, width, label="mF1")
    plt.xticks(x, [LABELS[m] for m in modes], rotation=15, ha="right")
    plt.ylabel("Score")
    plt.ylim(0, max([0.05, *miou, *mf1]) * 1.15)
    plt.title("A0--A4 final archive comparison")
    plt.legend()
    save_figure(out / "ablation_final_comparison.png")


def plot_runtime_memory(comparison: dict[str, Any], out: Path, modes: tuple[str, ...]) -> None:
    x = np.arange(len(modes))
    runtime = [float(comparison["modes"][m]["runtime_seconds"] or 0.0) for m in modes]
    memory = [float(comparison["modes"][m]["peak_cuda_memory_bytes"] or 0.0) / (1024**3) for m in modes]
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax2 = ax1.twinx()
    ax1.bar(x - 0.2, runtime, 0.4, label="Runtime", color="#4c78a8")
    ax2.bar(x + 0.2, memory, 0.4, label="Peak GPU", color="#f58518")
    ax1.set_xticks(x, [LABELS[m] for m in modes], rotation=15, ha="right")
    ax1.set_ylabel("Runtime (s)", color="#4c78a8")
    ax2.set_ylabel("Peak allocated GPU memory (GiB)", color="#f58518")
    ax1.set_title("Runtime and memory comparison")
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
    save_figure(out / "runtime_memory_comparison.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/oracle_boundary_mcmc/oracle_stream"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--comparison", type=Path, default=None)
    parser.add_argument("--boundaries", nargs="+", type=int, default=[95, 199])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modes = tuple(args.modes)
    out = args.output_dir or args.root.parent / "diagnostics"
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    comparison_path = args.comparison or args.root.parent / "comparison.json"
    comparison = build_comparison(args.root, modes)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    plot_adaptation(args.root, out, modes)
    plot_energy(args.root, out, modes, tuple(args.boundaries))
    plot_coverage(args.root, out, modes)
    plot_capacity(args.root, out, modes)
    plot_relocation(args.root, out)
    plot_relocation_energy(args.root, out)
    plot_displacement(args.root, out, tuple(m for m in modes if m != "dc_only"))
    combine_contact_sheets(args.root, out, modes)
    plot_comparison(comparison, out, modes)
    plot_runtime_memory(comparison, out, modes)
    print(json.dumps({"comparison": str(comparison_path), "diagnostics": str(out), "plots": sorted(path.name for path in out.glob("*.png"))}, indent=2))


if __name__ == "__main__":
    main()
