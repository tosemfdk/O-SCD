"""Evaluate base-only NEVER_OPEN geometry refinement with no NEW sidecar.

This is the controlled counterpart of the interactive detector viewer.  It
never loads a DA3 seed checkpoint, uses the causal prequential learned-sigmoid
cue, and optionally restricts the geometry target to the globally locked
signed-SAM/PCA NEW support.  It reports the final 0.5-threshold mask on the
evaluation-only ADD union REMOVE ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
import torch

from experiments.view_bayesian_detector_steps import (
    BayesianDetectorReplay,
    parse_args as parse_viewer_args,
)


DEFAULT_BOUNDARIES = Path(
    "outputs/stage2_l1_power_sigmoid_boundary_causal_prequential/"
    "learned_boundaries_causal.json"
)
DEFAULT_SAM_SIGN_TRACE = Path(
    "outputs/e4_online_xfeat_new_seed_scene123_120_final_corrected_20260814"
)


def _binary_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(prediction, dtype=bool)
    gt = np.asarray(target, dtype=bool)
    if pred.shape != gt.shape:
        raise ValueError(f"prediction/GT shape mismatch: {pred.shape} vs {gt.shape}")
    tp = int(np.logical_and(pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    denominator = tp + fp + fn
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "iou": tp / denominator if denominator else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(int(row["tp"]) for row in rows)
    tn = sum(int(row["tn"]) for row in rows)
    fp = sum(int(row["fp"]) for row in rows)
    fn = sum(int(row["fn"]) for row in rows)
    return {
        "frames": len(rows),
        "mean_frame_iou": float(np.mean([row["iou"] for row in rows])),
        "mean_frame_f1": float(np.mean([row["f1"] for row in rows])),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "aggregate_iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
        "aggregate_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--base-geometry-scope",
        choices=("frozen", "open", "open_and_never_open"),
        required=True,
    )
    parser.add_argument(
        "--base-geometry-target",
        choices=("change", "signed_new"),
        default="change",
    )
    parser.add_argument(
        "--sam-sign-trace-root", type=Path, default=DEFAULT_SAM_SIGN_TRACE
    )
    parser.add_argument("--sam-new-sign", choices=("+", "-"), default="+")
    parser.add_argument("--train-never-open-opacity", action="store_true")
    parser.add_argument("--max-frames", type=int, default=304)
    parser.add_argument("--updates-per-frame", type=int, default=16)
    parser.add_argument("--boundary-json", type=Path, default=DEFAULT_BOUNDARIES)
    args = parser.parse_args(argv)
    if args.max_frames <= 0 or args.max_frames > 304:
        parser.error("--max-frames must lie in [1,304]")
    if args.updates_per_frame <= 0:
        parser.error("--updates-per-frame must be positive")
    if not args.boundary_json.is_file():
        parser.error(f"boundary artifact not found: {args.boundary_json}")
    if (
        args.base_geometry_target == "signed_new"
        and not args.sam_sign_trace_root.is_dir()
    ):
        parser.error(f"SAM sign trace root not found: {args.sam_sign_trace_root}")
    if (
        args.train_never_open_opacity
        and args.base_geometry_scope != "open_and_never_open"
    ):
        parser.error(
            "--train-never-open-opacity requires open_and_never_open geometry"
        )
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    viewer_argv = [
            "--max-frames",
            str(args.max_frames),
            "--cue-fusion",
            "l1_power_product",
            "--product-exponent",
            "0.3",
            "--cue-mode",
            "soft",
            "--cue-scale",
            "2",
            "--cue-remap",
            "learned_sigmoid",
            "--cue-boundary-json",
            str(args.boundary_json),
            "--representation-updates",
            str(args.updates_per_frame),
            "--current-view-probability",
            "0.33",
            "--base-geometry-scope",
            args.base_geometry_scope,
            "--base-geometry-target",
            args.base_geometry_target,
        ]
    if args.base_geometry_target == "signed_new":
        viewer_argv.extend(
            [
                "--sam-sign-trace-root",
                str(args.sam_sign_trace_root),
                "--sam-new-sign",
                args.sam_new_sign,
            ]
        )
    if args.train_never_open_opacity:
        viewer_argv.append("--train-never-open-base-opacity")
    viewer_args = parse_viewer_args(viewer_argv)
    replay = BayesianDetectorReplay(viewer_args)
    if replay.da3_seeds is not None:
        raise RuntimeError("base-only ablation unexpectedly loaded a DA3 sidecar")
    if args.base_geometry_target == "change" and replay.sam_model is not None:
        raise RuntimeError("change-target control unexpectedly loaded signed SAM state")
    if args.base_geometry_target == "signed_new" and replay.sam_model is None:
        raise RuntimeError("signed-NEW ablation failed to load signed SAM state")

    started = time.time()
    rows: list[dict[str, Any]] = []
    while replay.has_next:
        summary = replay.step()
        prediction = (
            replay.learned_change_score()[0].detach().cpu().numpy() >= 0.5
        )
        target = np.asarray(
            replay.current_view.gt_add_mask | replay.current_view.gt_remove_mask,
            dtype=bool,
        )
        metrics = _binary_metrics(prediction, target)
        frame_name = str(replay.records[replay.current_index].name)
        scene = next(
            (name for name in ("scene_change1", "scene_change2", "scene_change3") if name in frame_name),
            "unknown",
        )
        rows.append(
            {
                "timestamp": replay.current_index,
                "frame_name": frame_name,
                "scene": scene,
                **metrics,
                "open": summary.open,
                "never_open": summary.never_open,
                "closed": summary.closed,
                "signed_new_pixels": int(
                    (replay.representation_replay[-1].new_target > 0).sum().item()
                ),
            }
        )
        print(
            f"[{replay.current_index + 1:03d}/{replay.total_frames}] "
            f"IoU={metrics['iou']:.4f} OPEN={summary.open:,}",
            flush=True,
        )

    xyz_delta = torch.linalg.vector_norm(
        replay.representation_xyz.detach() - replay.base._xyz.detach(), dim=1
    )
    scale_delta = (
        replay.representation_scaling.detach() - replay.base._scaling.detach()
    ).abs().amax(dim=1)
    rotation_delta = (
        replay.representation_rotation.detach() - replay.base._rotation.detach()
    ).abs().amax(dim=1)
    opacity_delta = (
        replay.representation_opacity.detach() - replay.base._opacity.detach()
    ).abs().amax(dim=1)
    learned_opacity = replay.base.opacity_activation(
        replay.representation_opacity.detach()
    )[:, 0]
    final_never = replay.lifecycle.num_states == 0
    final_open = replay.lifecycle.current_state_index >= 0
    final_closed = (replay.lifecycle.num_states > 0) & ~final_open

    def drift(mask: torch.Tensor) -> dict[str, Any]:
        if not bool(mask.any()):
            return {
                "rows": 0,
                "xyz_changed": 0,
                "scale_changed": 0,
                "rotation_changed": 0,
                "opacity_changed": 0,
            }
        return {
            "rows": int(mask.sum().item()),
            "xyz_changed": int((xyz_delta[mask] > 0).sum().item()),
            "scale_changed": int((scale_delta[mask] > 0).sum().item()),
            "rotation_changed": int((rotation_delta[mask] > 0).sum().item()),
            "opacity_changed": int((opacity_delta[mask] > 0).sum().item()),
            "xyz_max": float(xyz_delta[mask].max().item()),
            "scale_raw_max": float(scale_delta[mask].max().item()),
            "rotation_raw_max": float(rotation_delta[mask].max().item()),
            "opacity_raw_max": float(opacity_delta[mask].max().item()),
            "opacity_min": float(learned_opacity[mask].min().item()),
            "opacity_max": float(learned_opacity[mask].max().item()),
        }

    by_scene = {
        scene: _aggregate([row for row in rows if row["scene"] == scene])
        for scene in ("scene_change1", "scene_change2", "scene_change3")
        if any(row["scene"] == scene for row in rows)
    }
    result = {
        "contract": "base_only_learned_q_never_open_geometry",
        "base_geometry_scope": args.base_geometry_scope,
        "base_geometry_target": args.base_geometry_target,
        "train_never_open_opacity": bool(args.train_never_open_opacity),
        "da3_seed_checkpoint": None,
        "signed_sam_new_trace": (
            str(args.sam_sign_trace_root)
            if args.base_geometry_target == "signed_new"
            else None
        ),
        "sam_new_sign": (
            args.sam_new_sign
            if args.base_geometry_target == "signed_new"
            else None
        ),
        "cue": "causal prequential learned sigmoid Q",
        "detector": "soft-Q alpha-T BF30",
        "updates_per_frame": args.updates_per_frame,
        "evaluation_threshold": 0.5,
        "evaluation_gt": "ADD union REMOVE",
        "metrics": _aggregate(rows),
        "per_scene": by_scene,
        "geometry_drift": {
            "never_open": drift(final_never),
            "open": drift(final_open),
            "closed": drift(final_closed),
        },
        "runtime_seconds": time.time() - started,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "frame_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def main(argv: Sequence[str] | None = None) -> None:
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
