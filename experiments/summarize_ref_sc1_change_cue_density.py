"""Aggregate repeated ref -> SC1 change-cue density-control runs.

The input directory is expected to contain ``seed*/<condition>/summary.json``
files produced by :mod:`experiments.run_ref_sc1_change_cue_density`.  Generated
tables and plots are experiment artifacts and must not be committed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
from typing import Any, Callable, Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont


Metric = tuple[str, Callable[[dict[str, Any]], float]]

METRICS: tuple[Metric, ...] = (
    ("mIoU", lambda row: row["metrics"]["mean_frame_iou"]),
    ("F1", lambda row: row["metrics"]["mean_frame_f1"]),
    ("precision", lambda row: row["metrics"]["precision"]),
    ("recall", lambda row: row["metrics"]["recall"]),
    ("final_gaussians", lambda row: row["final_gaussian_count"]),
    ("gaussian_growth", lambda row: row["final_gaussian_count"] - row["initial_gaussian_count"]),
    ("clones", lambda row: row["total_clones"]),
    ("splits", lambda row: row["total_splits"]),
    ("pruned", lambda row: row["total_pruned"]),
    ("runtime_seconds", lambda row: row["runtime_seconds"]),
    ("peak_cuda_gib", lambda row: row["peak_cuda_memory_bytes"] / (1024**3)),
)


def condition_key(row: dict[str, Any]) -> str:
    condition = str(row["condition"])
    k = row.get("k_views")
    return f"cue_vcd_k{int(k)}" if condition == "cue_vcd" else condition


def load_summaries(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("seed*/*/summary.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("scope") != "ref -> scene_change1 only":
            raise ValueError(f"non-SC1 summary found: {path}")
        if row.get("gt_used_in_causal_loop") is not False:
            raise ValueError(f"causal GT contract failed: {path}")
        if row.get("future_view_access_count") != 0:
            raise ValueError(f"future-view access found: {path}")
        if not row.get("immutable_reference_unchanged"):
            raise ValueError(f"immutable reference audit failed: {path}")
        if not row.get("topology_integrity", {}).get("passed"):
            raise ValueError(f"topology audit failed: {path}")
        row["_summary_path"] = str(path)
        row["_seed"] = int(path.parents[1].name.removeprefix("seed"))
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"no repeated summaries found below {root}")
    return rows


def _mean_std(values: Iterable[float]) -> dict[str, float]:
    normalized = [float(value) for value in values]
    return {
        "mean": statistics.fmean(normalized),
        "std": statistics.stdev(normalized) if len(normalized) > 1 else 0.0,
    }


def aggregate_summaries(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(condition_key(row), []).append(row)
    conditions: list[dict[str, Any]] = []
    for key in sorted(grouped, key=_sort_key):
        members = sorted(grouped[key], key=lambda row: row["_seed"])
        metrics = {
            name: _mean_std(extractor(row) for row in members)
            for name, extractor in METRICS
        }
        conditions.append(
            {
                "condition": key,
                "k_views": members[0].get("k_views"),
                "repeats": len(members),
                "seeds": [row["_seed"] for row in members],
                "metrics": metrics,
            }
        )
    lookup = {row["condition"]: row for row in conditions}
    paired: dict[str, dict[str, float]] = {}
    fastgs = lookup.get("fastgs_gradient_only")
    if fastgs is not None:
        base_miou = fastgs["metrics"]["mIoU"]["mean"]
        base_splits = fastgs["metrics"]["splits"]["mean"]
        for row in conditions:
            if not row["condition"].startswith("cue_vcd_k"):
                continue
            splits = row["metrics"]["splits"]["mean"]
            paired[row["condition"]] = {
                "mean_mIoU_delta_vs_fastgs_gradient_only": row["metrics"]["mIoU"]["mean"] - base_miou,
                "mean_split_reduction_vs_fastgs_gradient_only": base_splits - splits,
                "mean_split_reduction_fraction_vs_fastgs_gradient_only": (
                    (base_splits - splits) / base_splits if base_splits else 0.0
                ),
            }
    return {
        "schema_version": 1,
        "scope": "ref -> scene_change1 only",
        "summary_count": len(rows),
        "conditions": conditions,
        "comparisons": paired,
        "all_causal_audits_passed": True,
    }


def _sort_key(key: str) -> tuple[int, int]:
    if key == "baseline":
        return (0, 0)
    if key == "fastgs_gradient_only":
        return (1, 0)
    if key.startswith("cue_vcd_k"):
        return (2, int(key.removeprefix("cue_vcd_k")))
    return (3, 0)


def write_csv(path: Path, comparison: dict[str, Any]) -> None:
    fieldnames = ["condition", "k_views", "repeats", "seeds"]
    for name, _extractor in METRICS:
        fieldnames.extend((f"{name}_mean", f"{name}_std"))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for condition in comparison["conditions"]:
            row: dict[str, Any] = {
                "condition": condition["condition"],
                "k_views": condition["k_views"],
                "repeats": condition["repeats"],
                "seeds": ",".join(map(str, condition["seeds"])),
            }
            for name, _extractor in METRICS:
                row[f"{name}_mean"] = condition["metrics"][name]["mean"]
                row[f"{name}_std"] = condition["metrics"][name]["std"]
            writer.writerow(row)


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _bar_plot(
    path: Path,
    conditions: Sequence[dict[str, Any]],
    *,
    metric: str,
    title: str,
    y_label: str,
) -> None:
    width, height = 1400, 760
    left, right, top, bottom = 130, 1340, 115, 620
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    values = [row["metrics"][metric]["mean"] for row in conditions]
    errors = [row["metrics"][metric]["std"] for row in conditions]
    maximum = max((value + error for value, error in zip(values, errors)), default=1.0)
    minimum = 0.0 if metric != "mIoU" else max(0.0, min(values) - 0.05)
    span = max(maximum - minimum, 1e-9) * 1.18
    draw.line((left, top, left, bottom), fill="black", width=3)
    draw.line((left, bottom, right, bottom), fill="black", width=3)
    slot = (right - left) / max(len(conditions), 1)
    palette = ((85, 85, 85), (230, 125, 35), (30, 120, 190), (45, 160, 90), (145, 90, 180), (220, 65, 70))
    for index, (row, value, error) in enumerate(zip(conditions, values, errors)):
        x0 = left + index * slot + slot * 0.18
        x1 = left + (index + 1) * slot - slot * 0.18
        y = bottom - (value - minimum) / span * (bottom - top)
        draw.rectangle((x0, y, x1, bottom), fill=palette[index % len(palette)])
        center = (x0 + x1) / 2
        err_top = bottom - (value + error - minimum) / span * (bottom - top)
        err_bottom = bottom - (value - error - minimum) / span * (bottom - top)
        draw.line((center, err_top, center, err_bottom), fill="black", width=3)
        draw.line((center - 12, err_top, center + 12, err_top), fill="black", width=3)
        draw.line((center - 12, err_bottom, center + 12, err_bottom), fill="black", width=3)
        label = row["condition"].replace("fastgs_gradient_only", "FastGS grad").replace("cue_vcd_", "cue ")
        draw.text((center, bottom + 18), label, fill="black", font=_font(18), anchor="ma")
        value_label = (
            f"{value:.0f}\n±{error:.0f}"
            if metric == "gaussian_growth"
            else f"{value:.4f}\n±{error:.4f}"
        )
        draw.text((center, err_top - 6), value_label, fill="black", font=_font(16), anchor="ms")
    draw.text((left, 24), title, fill="black", font=_font(30))
    draw.text((left, 78), y_label, fill=(55, 55, 55), font=_font(18))
    draw.text((left, height - 55), "mean ± sample std, seeds 0/1/2", fill=(70, 70, 70), font=_font(17))
    image.save(path)


def write_markdown(path: Path, comparison: dict[str, Any]) -> None:
    lines = [
        "# ref → SC1 soft alpha-T cue-VCD 반복 실험",
        "",
        "| Condition | K | mIoU | F1 | Precision | Recall | Final GS | Splits | Runtime(s) | Peak GiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison["conditions"]:
        metric = row["metrics"]
        fmt = lambda name, digits=4: f"{metric[name]['mean']:.{digits}f} ± {metric[name]['std']:.{digits}f}"
        lines.append(
            "| {condition} | {k} | {miou} | {f1} | {precision} | {recall} | {final} | {splits} | {runtime} | {peak} |".format(
                condition=row["condition"],
                k=row["k_views"] if row["k_views"] is not None else "-",
                miou=fmt("mIoU"),
                f1=fmt("F1"),
                precision=fmt("precision"),
                recall=fmt("recall"),
                final=fmt("final_gaussians", 1),
                splits=fmt("splits", 1),
                runtime=fmt("runtime_seconds", 2),
                peak=fmt("peak_cuda_gib", 3),
            )
        )
    lines.extend(("", "모든 값은 seed 0/1/2의 mean ± sample std이다.", ""))
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(root: Path, comparison: dict[str, Any]) -> list[str]:
    json_path = root / "comparison.json"
    csv_path = root / "comparison.csv"
    md_path = root / "comparison.md"
    json_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(csv_path, comparison)
    write_markdown(md_path, comparison)
    conditions = comparison["conditions"]
    miou_plot = root / "ref_sc1_miou_comparison.png"
    growth_plot = root / "ref_sc1_gaussian_growth_comparison.png"
    _bar_plot(miou_plot, conditions, metric="mIoU", title="ref → SC1 mean-frame mIoU", y_label="mIoU")
    _bar_plot(growth_plot, conditions, metric="gaussian_growth", title="ref → SC1 mutable R_change growth", y_label="new GS")
    return [str(json_path), str(csv_path), str(md_path), str(miou_plot), str(growth_plot)]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = load_summaries(args.root)
    comparison = aggregate_summaries(rows)
    outputs = write_outputs(args.root, comparison)
    print(json.dumps({"summaries": len(rows), "outputs": outputs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
