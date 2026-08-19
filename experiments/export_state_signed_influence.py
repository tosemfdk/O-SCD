"""Export offline signed removal influence for one temporal R_change state.

The checkpoint's current ``state_valid`` render is the baseline. Active rows
are evaluated by removal influence, while inactive rows are evaluated by their
individual insertion influence. This recovers a dark occluder without turning
every inactive reference Gaussian on at the same time.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from experiments.train_cue_temporal_rchange import (
    build_fixed_cue_views,
    load_fixed_camera_index,
)
from experiments.train_real_temporal_rchange import FrameRecord, file_checksum
from gaussian_renderer import render_change
from temporal import (
    classify_influence,
    forced_state_render_attributes,
    opacity_removal_influence,
    split_signed_influence,
    load_temporal_model,
)


DEFAULT_RUN_DIR = Path(
    "outputs/instance1_scene_change1_2_3_temporal_rchange_oscd_cues_allframes_120"
)


def _records_through_state_end(
    summary: dict[str, Any],
    state: int,
    end_timestamp: int,
) -> list[FrameRecord]:
    records = []
    for item in summary.get("train_frames", []):
        if int(item["segment_id"]) != state:
            continue
        if int(item["global_index"]) > end_timestamp:
            continue
        records.append(
            FrameRecord(
                global_index=int(item["global_index"]),
                segment_id=int(item["segment_id"]),
                name=str(item["name"]),
                image_path=str(item["image_path"]),
                mask_path="",
            )
        )
    records.sort(key=lambda record: record.global_index)
    if not records:
        raise ValueError(
            f"No state {state} training frames at or before timestamp {end_timestamp}"
        )
    return records


def _default_end_timestamp(summary: dict[str, Any], state: int) -> int:
    boundaries = [int(value) for value in summary.get("boundaries", [])]
    if state < len(boundaries):
        return boundaries[state] - 1
    frames = [
        int(item["global_index"])
        for item in summary.get("train_frames", [])
        if int(item["segment_id"]) == state
    ]
    if not frames:
        raise ValueError(f"Cannot infer the final timestamp for state {state}")
    return max(frames)


def _distribution(values: torch.Tensor) -> dict[str, float | int | None]:
    nonzero = values[values > 0].float()
    if nonzero.numel() == 0:
        return {
            "nonzero": 0,
            "min": None,
            "median": None,
            "p90": None,
            "p99": None,
            "max": None,
        }
    return {
        "nonzero": int(nonzero.numel()),
        "min": float(nonzero.min().item()),
        "median": float(nonzero.median().item()),
        "p90": float(torch.quantile(nonzero, 0.90).item()),
        "p99": float(torch.quantile(nonzero, 0.99).item()),
        "max": float(nonzero.max().item()),
    }


def export_signed_influence(args: argparse.Namespace) -> tuple[Path, Path]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the FastGS renderer")

    checkpoint_path = Path(args.checkpoint).resolve()
    summary_path = Path(args.summary).resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    model = load_temporal_model(checkpoint_path)
    if args.state < 0 or args.state >= model.max_states:
        raise ValueError(f"state must be in [0, {model.max_states})")

    end_timestamp = (
        int(args.end_timestamp)
        if args.end_timestamp is not None
        else _default_end_timestamp(summary, args.state)
    )
    records = _records_through_state_end(summary, args.state, end_timestamp)
    if args.max_views is not None:
        records = records[: int(args.max_views)]

    cameras_path = Path(args.fixed_cameras_json or summary["fixed_cameras_json"])
    cue_root = Path(args.cue_cache_root or summary["cue_cache_root"])
    cameras = load_fixed_camera_index(cameras_path)
    views, _, _ = build_fixed_cue_views(
        records,
        cameras,
        cue_root,
        float(summary["resolution"]),
    )
    attributes = {
        name: value.detach()
        for name, value in forced_state_render_attributes(model, args.state).items()
    }
    baseline_valid = model.state_valid[:, args.state].detach()
    gaussian_count = int(attributes["opacity"].shape[0])
    device = attributes["opacity"].device
    additive_sum = torch.zeros(gaussian_count, dtype=torch.float32, device=device)
    occluding_sum = torch.zeros_like(additive_sum)
    signed_sum = torch.zeros_like(additive_sum)
    contributing_views = torch.zeros(
        gaussian_count, dtype=torch.int32, device=device
    )
    pipe = SimpleNamespace(
        compute_cov3D_python=False,
        convert_SHs_python=False,
        debug=False,
    )
    background = torch.zeros(3, dtype=torch.float32, device=device)

    started = time.time()
    for view_index, view in enumerate(views, start=1):
        full_opacity = attributes["opacity"].detach()
        effective_opacity = (
            (full_opacity * baseline_valid[:, None])
            .clone()
            .requires_grad_(True)
        )
        package = render_change(
            view,
            model.base,
            pipe,
            background,
            override_dc=attributes["dc"],
            override_opacity=effective_opacity,
            override_xyz=attributes["xyz"],
            override_scaling=attributes["scaling"],
            override_rotation=attributes["rotation"],
            clamp_output=False,
        )
        signed = opacity_removal_influence(
            package["render"],
            effective_opacity,
            influence_opacity=full_opacity,
            mask_threshold=float(args.mask_threshold),
            mask_temperature=float(args.mask_temperature),
        ).float()
        if not bool(torch.isfinite(signed).all()):
            raise FloatingPointError(
                f"Non-finite influence at timestamp {view.timestamp}"
            )
        additive, occluding = split_signed_influence(signed)
        additive_sum.add_(additive)
        occluding_sum.add_(occluding)
        signed_sum.add_(signed)
        contributing_views.add_(
            signed.abs() >= float(args.min_per_view_influence)
        )
        if (
            view_index == 1
            or view_index == len(views)
            or view_index % int(args.progress_interval) == 0
        ):
            print(
                f"[signed-influence] view {view_index}/{len(views)} "
                f"t={int(view.timestamp)} elapsed={time.time() - started:.1f}s",
                flush=True,
            )

    classified = classify_influence(
        additive_sum,
        occluding_sum,
        contributing_views,
        min_mean_influence=float(args.min_mean_influence),
        min_views=int(args.min_views),
        view_count=len(views),
    )
    artifact = {
        "schema_version": 1,
        "contract": "offline_signed_opacity_removal_influence_fixed_topology",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_checksum(checkpoint_path),
        "state": int(args.state),
        "end_timestamp": end_timestamp,
        "view_count": len(views),
        "frame_names": [view.image_name for view in views],
        "gaussian_count": gaussian_count,
        "existing_state_valid_used_as_baseline": True,
        "valid_rows_measure_removal": True,
        "invalid_rows_measure_individual_insertion": True,
        "influence_objective": "soft_binary_change_mask_area",
        "mask_threshold": float(args.mask_threshold),
        "mask_temperature": float(args.mask_temperature),
        "min_per_view_influence": float(args.min_per_view_influence),
        "min_mean_influence": float(args.min_mean_influence),
        "min_views": int(args.min_views),
        "signed_sum": signed_sum.detach().cpu(),
        "mean_additive": classified["mean_additive"].detach().cpu(),
        "mean_occluding": classified["mean_occluding"].detach().cpu(),
        "mean_absolute": classified["mean_absolute"].detach().cpu(),
        "contributing_views": contributing_views.detach().cpu(),
        "additive_mask": classified["additive_mask"].detach().cpu(),
        "occluding_mask": classified["occluding_mask"].detach().cpu(),
        "both_mask": classified["both_mask"].detach().cpu(),
        "valid_mask": classified["valid_mask"].detach().cpu(),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / f"state{args.state}_signed_influence.pt"
    report_path = output_dir / f"state{args.state}_signed_influence_summary.json"
    torch.save(artifact, artifact_path)

    valid = artifact["valid_mask"]
    additive_mask = artifact["additive_mask"]
    occluding_mask = artifact["occluding_mask"]
    both_mask = artifact["both_mask"]
    report = {
        key: value
        for key, value in artifact.items()
        if not isinstance(value, torch.Tensor)
    }
    report.update(
        {
            "runtime_seconds": time.time() - started,
            "valid_gaussians": int(valid.sum().item()),
            "additive_gaussians": int(additive_mask.sum().item()),
            "occluding_gaussians": int(occluding_mask.sum().item()),
            "both_gaussians": int(both_mask.sum().item()),
            "additive_only_gaussians": int(
                (additive_mask & ~occluding_mask).sum().item()
            ),
            "occluding_only_gaussians": int(
                (occluding_mask & ~additive_mask).sum().item()
            ),
            "mean_absolute_distribution": _distribution(
                artifact["mean_absolute"]
            ),
            "mean_additive_distribution": _distribution(
                artifact["mean_additive"]
            ),
            "mean_occluding_distribution": _distribution(
                artifact["mean_occluding"]
            ),
            "artifact": str(artifact_path.resolve()),
            "artifact_sha256": file_checksum(artifact_path),
        }
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return artifact_path, report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_RUN_DIR / "temporal_rchange_checkpoint.pt",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_RUN_DIR / "summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RUN_DIR / "signed_influence",
    )
    parser.add_argument("--state", type=int, default=0)
    parser.add_argument("--end-timestamp", type=int, default=None)
    parser.add_argument("--fixed-cameras-json", type=Path, default=None)
    parser.add_argument("--cue-cache-root", type=Path, default=None)
    parser.add_argument("--min-per-view-influence", type=float, default=1e-6)
    parser.add_argument("--min-mean-influence", type=float, default=1e-4)
    parser.add_argument("--min-views", type=int, default=1)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--mask-temperature", type=float, default=0.05)
    parser.add_argument("--max-views", type=int, default=None)
    parser.add_argument("--progress-interval", type=int, default=10)
    args = parser.parse_args()
    if args.min_per_view_influence < 0 or args.min_mean_influence < 0:
        parser.error("influence thresholds must be non-negative")
    if not 0.0 <= args.mask_threshold <= 1.0:
        parser.error("mask-threshold must be in [0, 1]")
    if args.mask_temperature <= 0.0:
        parser.error("mask-temperature must be positive")
    if args.min_views < 1 or args.progress_interval < 1:
        parser.error("min-views and progress-interval must be positive")
    if args.max_views is not None and args.max_views < 1:
        parser.error("max-views must be positive")
    return args


def main() -> None:
    artifact, report = export_signed_influence(parse_args())
    print(json.dumps({"artifact": str(artifact), "summary": str(report)}, indent=2))


if __name__ == "__main__":
    main()
