"""Run direct binary-state lifespan ablations on PASLCD and compare with O-SCD.

This is an adapter around `run_online_binary_state_lifespan_thaw.py`; it does not
change detector/training logic.  It uses original O-SCD fixed poses and newly
prepared immutable-reference O-SCD cue caches, then aggregates scene metrics
against the saved O-SCD PASLCD baseline artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.prepare_paslcd_fixed_pose_cues import (
    CUE_VALUE_STORAGE,
    DEFAULT_DATASET_ROOT,
    DEFAULT_INSTANCES,
    DEFAULT_OSCD_OUTPUT_ROOT,
    DEFAULT_OSCD_FALLBACK_OUTPUT_ROOT,
    DEFAULT_SCENES,
    SceneSpec,
    discover_scenes,
    parse_csv,
    sha256_file,
)

DEFAULT_CUE_ROOT = Path("outputs/paslcd_fixed_pose_cues_res4_v1")
DEFAULT_OUTPUT_ROOT = Path("outputs/paslcd_direct_binary_state_benchmark_res4")
DEFAULT_BASELINE_CSV = Path("/home/rvl/workspace/github/O-SCD/artifacts/baseline/metrics_per_scene.csv")
DEFAULT_BASELINE_SUMMARY = Path("/home/rvl/workspace/github/O-SCD/artifacts/baseline/metrics_summary.json")

CONDITIONS = {
    "B0_binary_dc": ("binary", "dc"),
    "B1_binary_all_geometry": ("binary", "dc,xyz,opacity,scaling,rotation"),
}
EVENT_STRUCTURE_FIELDS = (
    "gaussian_index",
    "decision_timestamp",
    "old_binary_label",
    "new_binary_label",
    "action",
    "old_slot",
    "new_current_slot",
)


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    cue_mode: str
    thaw_parameters: str


def load_baseline_rows(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    rows: dict[tuple[str, str], dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["instance"], row["scene"])
            rows[key] = {
                "oscd_online_miou": float(row["miou_online"]),
                "oscd_online_f1": float(row["f1_online"]),
                "oscd_refined_miou": float(row["miou_refined"]),
                "oscd_refined_f1": float(row["f1_refined"]),
            }
    return rows


def parse_conditions(value: str | None) -> list[ConditionSpec]:
    names = tuple(CONDITIONS) if not value else tuple(part.strip() for part in value.split(",") if part.strip())
    specs: list[ConditionSpec] = []
    for name in names:
        if name not in CONDITIONS:
            raise ValueError(f"unknown condition {name}; choose from {sorted(CONDITIONS)}")
        cue_mode, thaw = CONDITIONS[name]
        specs.append(ConditionSpec(name=name, cue_mode=cue_mode, thaw_parameters=thaw))
    return specs


def command_for_scene(
    spec: SceneSpec,
    condition: ConditionSpec,
    *,
    output_root: Path,
    resolution: float,
    updates_per_frame: int,
    max_frames: int | None,
    skip_checkpoint: bool,
    extra_args: Sequence[str],
) -> tuple[list[str], Path]:
    budget_name = f"u{int(updates_per_frame)}"
    output_dir = output_root / budget_name / condition.name / spec.instance / spec.scene
    cue_cache_root = spec.output_dir
    cmd = [
        sys.executable,
        "-m",
        "experiments.run_online_binary_state_lifespan_thaw",
        "--source-path",
        str(spec.source_path),
        "--fixed-cameras-json",
        str(spec.cameras_json),
        "--cue-cache-root",
        str(cue_cache_root),
        "--output-dir",
        str(output_dir),
        "--resolution",
        str(float(resolution)),
        "--disable-boundary-diagnostics",
        "--bayes-cue-mode",
        condition.cue_mode,
        "--evidence-count-mode",
        "capped",
        "--updates-per-frame",
        str(int(updates_per_frame)),
        "--seed",
        "0",
        "--thaw-parameters",
        condition.thaw_parameters,
    ]
    if max_frames is not None:
        cmd.extend(["--max-frames", str(int(max_frames))])
    if skip_checkpoint:
        cmd.append("--skip-checkpoint")
    cmd.extend(extra_args)
    return cmd, output_dir


def validate_scene_summary(
    summary: Mapping[str, Any],
    spec: SceneSpec,
    condition: ConditionSpec,
    *,
    updates_per_frame: int,
    max_frames: int | None,
    checkpoint_expected: bool,
    expected_command: Sequence[str] | None = None,
) -> None:
    config = summary.get("run_config", {})
    arguments = summary.get("run_arguments", {})
    cue_metadata = summary.get("cue_cache_metadata", {})
    expected_thaw = tuple(part for part in condition.thaw_parameters.split(",") if part)
    checks = {
        "algorithm": summary.get("algorithm") == "direct_binary_state_filter",
        "cue_mode": config.get("bayes_cue_mode") == condition.cue_mode,
        "thaw_parameters": tuple(config.get("thaw_parameters", ())) == expected_thaw,
        "updates_per_frame": int(config.get("updates_per_frame", -1))
        == int(updates_per_frame),
        "evidence_count_mode": config.get("evidence_count_mode") == "capped",
        "raw_cue_cache": cue_metadata.get("cue_value_storage")
        == CUE_VALUE_STORAGE,
        "cue_camera_checksum": cue_metadata.get("fixed_cameras_sha256")
        == sha256_file(spec.cameras_json),
        "seed": int(config.get("seed", -1)) == 0,
        "source_path": Path(str(arguments.get("source_path", ""))).resolve()
        == spec.source_path.resolve(),
        "fixed_cameras_json": Path(
            str(arguments.get("fixed_cameras_json", ""))
        ).resolve()
        == spec.cameras_json.resolve(),
        "cue_cache_root": Path(str(arguments.get("cue_cache_root", ""))).resolve()
        == spec.output_dir.resolve(),
        "post_inference_evaluation": bool(summary.get("metrics", {}).get("evaluated")),
        "gt_not_used_for_training": not bool(summary.get("gt_used_for_training", True)),
        "manual_boundaries_not_used": not bool(
            summary.get("manual_boundaries_used_for_inference", True)
        ),
        "checkpoint_policy": bool(summary.get("checkpoint_saved", True))
        == bool(checkpoint_expected),
        "max_frames_argument": arguments.get("max_frames") == max_frames,
        "frame_count": int(summary.get("frames", -1))
        == int(max_frames if max_frames is not None else 25),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if expected_command is not None:
        if len(expected_command) < 4 or expected_command[1:3] != [
            "-m",
            "experiments.run_online_binary_state_lifespan_thaw",
        ]:
            raise ValueError(f"unexpected direct-runner command: {expected_command}")
        from experiments.run_online_binary_state_lifespan_thaw import (
            parse_args as parse_direct_args,
            serializable_arguments,
        )

        expected_arguments = serializable_arguments(
            parse_direct_args(list(expected_command[3:]))
        )
        expected_arguments = json.loads(json.dumps(expected_arguments))
        actual_arguments = json.loads(json.dumps(arguments))
        # Result directories may be relocated after a completed run; the path
        # does not affect inference or optimization semantics.
        expected_arguments.pop("output_dir", None)
        actual_arguments.pop("output_dir", None)
        if actual_arguments != expected_arguments:
            failed.append("run_arguments")
    if failed:
        raise ValueError(
            f"incompatible scene summary for {spec.instance}/{spec.scene}: {failed}"
        )


def summarize_scene(summary: Mapping[str, Any], spec: SceneSpec, condition: ConditionSpec, baseline: Mapping[str, float]) -> dict[str, Any]:
    metrics = summary.get("metrics", {})
    pre_opt = metrics.get("pre_opt", {})
    open_frame = metrics.get("open_frame_pre_post", {})
    return {
        "condition": condition.name,
        "instance": spec.instance,
        "scene": spec.scene,
        "frames": int(summary.get("frames", 0)),
        "miou": float(metrics.get("mean_frame_iou", 0.0)),
        "f1": float(metrics.get("mean_frame_f1", 0.0)),
        "precision": float(metrics.get("precision", 0.0)),
        "recall": float(metrics.get("recall", 0.0)),
        "aggregate_iou": float(metrics.get("aggregate_iou", 0.0)),
        "aggregate_f1": float(metrics.get("aggregate_f1", 0.0)),
        "pre_opt_miou": float(pre_opt.get("mean_frame_iou", 0.0)),
        "pre_opt_f1": float(pre_opt.get("mean_frame_f1", 0.0)),
        "open_frame_count": int(open_frame.get("count", 0)),
        "open_frame_miou_delta": float(open_frame.get("mean_delta_iou", 0.0)),
        "open_frame_f1_delta": float(open_frame.get("mean_delta_f1", 0.0)),
        "open": int(summary.get("open_count", 0)),
        "close": int(summary.get("close_count", 0)),
        "reopen": int(summary.get("reopen_count", 0)),
        "keep": int(summary.get("keep_count", 0)),
        "uncertain": int(summary.get("uncertain_count", 0)),
        "same_scene_repeated_transition_events": int(
            summary.get("same_scene_repeated_transition_event_count", 0)
        ),
        "same_scene_repeated_gaussians": int(
            summary.get("same_scene_repeated_gaussian_count", 0)
        ),
        "active_to_active_false_splits": int(
            summary.get("active_to_active_false_split_count", 0)
        ),
        "reused_slot_violations": int(summary.get("reused_slot_violations", 0)),
        "final_active_gs": int(summary.get("final_active_gs", 0)),
        "base_max_drift": float(summary.get("base_max_drift", 0.0)),
        "closed_slot_max_drift": float(summary.get("closed_slot_max_drift", 0.0)),
        "inactive_gradient_violations": int(summary.get("inactive_gradient_first_step_violations", 0)),
        "runtime_seconds": float(summary.get("runtime_seconds", 0.0)),
        "cuda_peak_memory_bytes": int(summary.get("cuda_peak_memory_bytes", 0)),
        "fixed_cameras_json": str(spec.cameras_json),
        "used_fallback_camera": spec.camera_source == "fallback",
        "gt_used_for_training": bool(summary.get("gt_used_for_training", True)),
        "manual_boundaries_used_for_inference": bool(
            summary.get("manual_boundaries_used_for_inference", True)
        ),
        "checkpoint_saved": bool(summary.get("checkpoint_saved", True)),
        **baseline,
        "delta_vs_oscd_online_miou": float(metrics.get("mean_frame_iou", 0.0)) - float(baseline.get("oscd_online_miou", 0.0)),
        "delta_vs_oscd_online_f1": float(metrics.get("mean_frame_f1", 0.0)) - float(baseline.get("oscd_online_f1", 0.0)),
        "delta_vs_oscd_refined_miou": float(metrics.get("mean_frame_iou", 0.0)) - float(baseline.get("oscd_refined_miou", 0.0)),
        "delta_vs_oscd_refined_f1": float(metrics.get("mean_frame_f1", 0.0)) - float(baseline.get("oscd_refined_f1", 0.0)),
    }


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def weighted_mean(
    rows: Sequence[Mapping[str, Any]], value_key: str, weight_key: str
) -> float:
    weight = sum(int(row[weight_key]) for row in rows)
    if weight <= 0:
        return 0.0
    return float(
        sum(float(row[value_key]) * int(row[weight_key]) for row in rows) / weight
    )


def lifecycle_event_structure_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open(encoding="utf-8") as f:
        for line in f:
            event = json.loads(line)
            structure = [event.get(field) for field in EVENT_STRUCTURE_FIELDS]
            digest.update(
                json.dumps(structure, separators=(",", ":")).encode("utf-8")
            )
            digest.update(b"\n")
    return digest.hexdigest()


def scene_outputs_complete(output_dir: Path, *, checkpoint_expected: bool) -> bool:
    required = (
        "summary.json",
        "frame_metrics.csv",
        "lifecycle_events.jsonl",
        "per_frame_binary_state_stats.npz",
    )
    if not all((output_dir / name).is_file() for name in required):
        return False
    checkpoint_exists = (output_dir / "checkpoint.pt").is_file()
    return checkpoint_exists == bool(checkpoint_expected)


def detector_consistency(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    budgets = sorted({int(row["updates_per_frame"]) for row in rows})
    for budget in budgets:
        budget_rows = [row for row in rows if int(row["updates_per_frame"]) == budget]
        by_scene: dict[tuple[str, str], dict[str, str]] = {}
        for row in budget_rows:
            key = (str(row["instance"]), str(row["scene"]))
            by_scene.setdefault(key, {})[str(row["condition_base"])] = str(
                row["lifecycle_event_structure_sha256"]
            )
        comparable = [
            values
            for values in by_scene.values()
            if set(CONDITIONS).issubset(values)
        ]
        mismatches = sum(
            values["B0_binary_dc"] != values["B1_binary_all_geometry"]
            for values in comparable
        )
        result[f"u{budget}"] = {
            "comparable_scene_count": len(comparable),
            "event_structure_mismatch_scene_count": int(mismatches),
            "passed": bool(comparable) and mismatches == 0,
        }
    if len(budgets) > 1:
        by_condition_scene: dict[tuple[str, str, str], dict[int, str]] = {}
        for row in rows:
            key = (
                str(row["condition_base"]),
                str(row["instance"]),
                str(row["scene"]),
            )
            by_condition_scene.setdefault(key, {})[
                int(row["updates_per_frame"])
            ] = str(row["lifecycle_event_structure_sha256"])
        cross_budget = [
            values
            for values in by_condition_scene.values()
            if set(budgets).issubset(values)
        ]
        cross_mismatches = sum(
            len({values[budget] for budget in budgets}) != 1
            for values in cross_budget
        )
        result["cross_budget"] = {
            "comparable_condition_scene_count": len(cross_budget),
            "event_structure_mismatch_count": int(cross_mismatches),
            "passed": bool(cross_budget) and cross_mismatches == 0,
        }
    return result


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_condition.setdefault(str(row["condition"]), []).append(row)
    summary: dict[str, Any] = {}
    for condition, cond_rows in sorted(by_condition.items()):
        summary[condition] = {
            "scene_count": len(cond_rows),
            "miou": mean([float(r["miou"]) for r in cond_rows]),
            "f1": mean([float(r["f1"]) for r in cond_rows]),
            "precision": mean([float(r["precision"]) for r in cond_rows]),
            "recall": mean([float(r["recall"]) for r in cond_rows]),
            "pre_opt_miou": mean([float(r["pre_opt_miou"]) for r in cond_rows]),
            "pre_opt_f1": mean([float(r["pre_opt_f1"]) for r in cond_rows]),
            "open_frame_count": int(sum(int(r["open_frame_count"]) for r in cond_rows)),
            "open_frame_miou_delta": weighted_mean(
                cond_rows, "open_frame_miou_delta", "open_frame_count"
            ),
            "open_frame_f1_delta": weighted_mean(
                cond_rows, "open_frame_f1_delta", "open_frame_count"
            ),
            "oscd_online_miou": mean([float(r["oscd_online_miou"]) for r in cond_rows]),
            "oscd_online_f1": mean([float(r["oscd_online_f1"]) for r in cond_rows]),
            "oscd_refined_miou": mean([float(r["oscd_refined_miou"]) for r in cond_rows]),
            "oscd_refined_f1": mean([float(r["oscd_refined_f1"]) for r in cond_rows]),
            "delta_vs_oscd_online_miou": mean([float(r["delta_vs_oscd_online_miou"]) for r in cond_rows]),
            "delta_vs_oscd_online_f1": mean([float(r["delta_vs_oscd_online_f1"]) for r in cond_rows]),
            "delta_vs_oscd_refined_miou": mean([float(r["delta_vs_oscd_refined_miou"]) for r in cond_rows]),
            "delta_vs_oscd_refined_f1": mean([float(r["delta_vs_oscd_refined_f1"]) for r in cond_rows]),
            "open": int(sum(int(r["open"]) for r in cond_rows)),
            "close": int(sum(int(r["close"]) for r in cond_rows)),
            "reopen": int(sum(int(r["reopen"]) for r in cond_rows)),
            "keep": int(sum(int(r["keep"]) for r in cond_rows)),
            "uncertain": int(sum(int(r["uncertain"]) for r in cond_rows)),
            "same_scene_repeated_transition_events": int(
                sum(int(r["same_scene_repeated_transition_events"]) for r in cond_rows)
            ),
            "same_scene_repeated_gaussians": int(
                sum(int(r["same_scene_repeated_gaussians"]) for r in cond_rows)
            ),
            "active_to_active_false_splits": int(
                sum(int(r["active_to_active_false_splits"]) for r in cond_rows)
            ),
            "reused_slot_violations": int(
                sum(int(r["reused_slot_violations"]) for r in cond_rows)
            ),
            "final_active_gs": int(sum(int(r["final_active_gs"]) for r in cond_rows)),
            "scene_wins_vs_oscd_online_miou": int(
                sum(float(r["delta_vs_oscd_online_miou"]) > 0.0 for r in cond_rows)
            ),
            "scene_wins_vs_oscd_refined_miou": int(
                sum(float(r["delta_vs_oscd_refined_miou"]) > 0.0 for r in cond_rows)
            ),
            "fallback_camera_scene_count": int(
                sum(bool(r["used_fallback_camera"]) for r in cond_rows)
            ),
            "gt_or_boundary_contract_violations": int(
                sum(
                    bool(r["gt_used_for_training"])
                    or bool(r["manual_boundaries_used_for_inference"])
                    for r in cond_rows
                )
            ),
            "saved_checkpoint_scene_count": int(
                sum(bool(r["checkpoint_saved"]) for r in cond_rows)
            ),
            "runtime_seconds": float(sum(float(r["runtime_seconds"]) for r in cond_rows)),
            "peak_cuda_memory_bytes": int(max(int(r["cuda_peak_memory_bytes"]) for r in cond_rows)) if cond_rows else 0,
            "max_base_drift": max(float(r["base_max_drift"]) for r in cond_rows) if cond_rows else 0.0,
            "max_closed_slot_drift": max(float(r["closed_slot_max_drift"]) for r in cond_rows) if cond_rows else 0.0,
            "inactive_gradient_violations": int(sum(int(r["inactive_gradient_violations"]) for r in cond_rows)),
        }
    return summary


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, aggregate: Mapping[str, Any]) -> None:
    lines = [
        "# PASLCD direct binary-state benchmark",
        "",
        "| condition | mIoU | F1 | precision | recall | O-SCD online mIoU/F1 | O-SCD refined mIoU/F1 | Δ online mIoU/F1 | wins | OPEN/CLOSE/REOPEN | repeated events | runtime(s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in aggregate.items():
        lines.append(
            f"| {name} | {row['miou']:.4f} | {row['f1']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['oscd_online_miou']:.4f}/{row['oscd_online_f1']:.4f} | "
            f"{row['oscd_refined_miou']:.4f}/{row['oscd_refined_f1']:.4f} | "
            f"{row['delta_vs_oscd_online_miou']:+.4f}/{row['delta_vs_oscd_online_f1']:+.4f} | "
            f"{row['scene_wins_vs_oscd_online_miou']}/{row['scene_count']} | "
            f"{row['open']}/{row['close']}/{row['reopen']} | "
            f"{row['same_scene_repeated_transition_events']} | {row['runtime_seconds']:.1f} |"
        )
    lines.extend(
        [
            "",
            "O-SCD online uses 16 updates/frame. The 120-update direct conditions are the existing ESCD study budget, not a compute-matched comparison.",
            "PASLCD contains one static post-change episode per scene, so repeated CLOSE/REOPEN events diagnose view/cue-induced lifecycle chattering rather than true scene evolution.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--oscd-output-root", type=Path, default=DEFAULT_OSCD_OUTPUT_ROOT)
    parser.add_argument("--oscd-fallback-output-root", type=Path, default=DEFAULT_OSCD_FALLBACK_OUTPUT_ROOT)
    parser.add_argument("--cue-root", type=Path, default=DEFAULT_CUE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--baseline-csv", type=Path, default=DEFAULT_BASELINE_CSV)
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE_SUMMARY)
    parser.add_argument("--instances", type=str, default=None)
    parser.add_argument("--scenes", type=str, default=None)
    parser.add_argument("--conditions", type=str, default=None)
    parser.add_argument("--resolution", type=float, default=4.0)
    parser.add_argument(
        "--update-budgets",
        type=str,
        default="120",
        help="comma-separated update budgets. 120 is the primary direct-study setting; 16 is the original O-SCD online budget.",
    )
    parser.add_argument(
        "--save-checkpoint",
        action="store_true",
        help="Save per-scene checkpoints. Disabled by default to keep PASLCD benchmark storage small.",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def parse_update_budgets(value: str) -> tuple[int, ...]:
    budgets = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not budgets or any(v <= 0 for v in budgets):
        raise ValueError("update budgets must be positive integers")
    return budgets


def sanitize_extra_args(values: Sequence[str]) -> list[str]:
    values = list(values)
    if values and values[0] == "--":
        values = values[1:]
    owned = {
        "--source-path",
        "--fixed-cameras-json",
        "--cue-cache-root",
        "--output-dir",
        "--resolution",
        "--disable-boundary-diagnostics",
        "--bayes-cue-mode",
        "--evidence-count-mode",
        "--updates-per-frame",
        "--max-frames",
        "--seed",
        "--thaw-parameters",
        "--skip-checkpoint",
        "--skip-post-inference-evaluation",
        "--detector-only",
        "--detector-only-smoke",
    }
    for value in values:
        flag = value.split("=", 1)[0]
        if flag in owned:
            raise ValueError(f"benchmark owns runner flag {flag}")
    return values


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    specs = discover_scenes(
        args.dataset_root,
        args.oscd_output_root,
        args.cue_root,
        instances=parse_csv(args.instances, DEFAULT_INSTANCES),
        scenes=parse_csv(args.scenes, DEFAULT_SCENES),
        oscd_fallback_output_root=args.oscd_fallback_output_root,
        resolution=float(args.resolution),
    )
    conditions = parse_conditions(args.conditions)
    update_budgets = parse_update_budgets(args.update_budgets)
    baseline_rows = load_baseline_rows(args.baseline_csv)
    baseline_summary = json.loads(args.baseline_summary.read_text(encoding="utf-8")) if args.baseline_summary.exists() else {}
    extra_args = sanitize_extra_args(args.runner_args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rows: list[dict[str, Any]] = []
    for updates_per_frame in update_budgets:
        for condition in conditions:
            for spec in specs:
                cmd, output_dir = command_for_scene(
                    spec,
                    condition,
                    output_root=args.output_root,
                    resolution=float(args.resolution),
                    updates_per_frame=int(updates_per_frame),
                    max_frames=args.max_frames,
                    skip_checkpoint=not bool(args.save_checkpoint),
                    extra_args=extra_args,
                )
                summary_path = output_dir / "summary.json"
                summary = None
                if args.resume and scene_outputs_complete(
                    output_dir,
                    checkpoint_expected=bool(args.save_checkpoint),
                ):
                    try:
                        summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    except (json.JSONDecodeError, OSError):
                        summary = None
                if summary is None:
                    summary_path.unlink(missing_ok=True)
                    print("RUN", f"u{updates_per_frame}", condition.name, spec.instance, spec.scene, flush=True)
                    subprocess.run(cmd, check=True)
                    if not scene_outputs_complete(
                        output_dir,
                        checkpoint_expected=bool(args.save_checkpoint),
                    ):
                        raise RuntimeError(f"incomplete scene output: {output_dir}")
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                validate_scene_summary(
                    summary,
                    spec,
                    condition,
                    updates_per_frame=int(updates_per_frame),
                    max_frames=args.max_frames,
                    checkpoint_expected=bool(args.save_checkpoint),
                    expected_command=cmd,
                )
                key = (spec.instance, spec.scene)
                if key not in baseline_rows:
                    raise KeyError(f"missing O-SCD baseline row for {key}")
                row = summarize_scene(summary, spec, condition, baseline_rows[key])
                row["updates_per_frame"] = int(updates_per_frame)
                row["budget_label"] = "primary_direct_study" if int(updates_per_frame) == 120 else "oscd_budget_matched" if int(updates_per_frame) == 16 else "custom"
                row["condition_base"] = condition.name
                row["lifecycle_event_structure_sha256"] = (
                    lifecycle_event_structure_sha256(
                        output_dir / "lifecycle_events.jsonl"
                    )
                )
                row["condition"] = f"u{updates_per_frame}_{row['condition']}"
                rows.append(row)
                write_csv(args.output_root / "paslcd_binary_state_scene_metrics.csv", rows)
    aggregate = aggregate_rows(rows)
    report = {
        "schema_version": 1,
        "script": "experiments/run_paslcd_binary_state_benchmark.py",
        "dataset_root": str(args.dataset_root),
        "cue_root": str(args.cue_root),
        "scene_count": len(specs),
        "frame_count": int(sum(int(row["frames"]) for row in rows) / max(1, len(conditions) * len(update_budgets))),
        "condition_definition_count": len(conditions),
        "condition_count": len(aggregate),
        "update_budgets": list(update_budgets),
        "checkpoints_saved": bool(args.save_checkpoint),
        "runtime_seconds": time.time() - started,
        "baseline_summary": baseline_summary,
        "baseline_artifacts": {
            "scene_metrics_csv": str(args.baseline_csv),
            "scene_metrics_csv_sha256": sha256_file(args.baseline_csv),
            "summary_json": str(args.baseline_summary),
            "summary_json_sha256": (
                sha256_file(args.baseline_summary)
                if args.baseline_summary.exists()
                else None
            ),
        },
        "detector_consistency": detector_consistency(rows),
        "comparison_contract": {
            "resolution": float(args.resolution),
            "cue": "raw O-SCD pixel + SAM2.1 Hiera Tiny candidate map",
            "detector_evidence": "immutable-reference alpha-T",
            "gt_during_causal_loop": False,
            "manual_boundaries_during_inference": False,
            "direct_topology": "fixed; no densification or pruning",
            "oscd_online_updates_per_frame": 16,
            "oscd_refined_total_updates_per_scene": 3000,
            "pose_note": (
                "Saved O-SCD cameras are reused. Instance_1/Lounge uses the "
                "compatible seed-0 resolution-4 ablation camera file because "
                "the primary camera file was overwritten at another resolution. "
                "The frozen baseline artifact did not record camera hashes, so "
                "bitwise pose identity with the July baseline cannot be proven."
            ),
        },
        "conditions": aggregate,
    }
    (args.output_root / "comparison.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.output_root / "comparison.md", aggregate)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
