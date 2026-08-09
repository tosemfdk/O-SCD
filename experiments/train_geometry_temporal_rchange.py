"""Train state-specific R_change geometry with fixed Gaussian topology.

The experiment uses the same fixed poses, cached O-SCD pixel+feature cues,
manual boundaries, and exact per-frame schedule as the DC-only lifespan run.
Densification and pruning are intentionally disabled.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from experiments.train_cue_temporal_rchange import (
    DEFAULT_CUE_CACHE,
    DEFAULT_FIXED_CAMERAS,
    build_fixed_cue_views,
    load_fixed_camera_index,
    serializable_arguments,
    validate_cue_cache,
)
from experiments.train_real_temporal_rchange import (
    BASE_PLY_REL,
    DEFAULT_BOUNDARIES,
    DEFAULT_SOURCE,
    FrameRecord,
    PoseResult,
    boundaries_from_manifest,
    build_frame_records,
    compute_state_support,
    file_checksum,
    make_training_schedule,
    oscd_positive_sparsity_loss,
    read_manifest,
    seed_everything,
    segment_ranges,
    should_audit_gradient,
    summarize_losses,
    tensor_checksum,
    training_schedule_audit,
    training_target,
    validate_exact_dataset_contract,
)
from gaussian_renderer import render_change_temporal
from scene import GaussianModel
from scene.cameras import Camera
from temporal import TemporalGeometryChangeModel


DEFAULT_OUTPUT = Path(
    "outputs/instance1_scene_change1_2_3_temporal_geometry_oscd_cues_allframes_120"
)


def build_geometry_temporal_model(
    base: GaussianModel,
    boundaries: tuple[int, ...],
    total_frames: int,
) -> TemporalGeometryChangeModel:
    """Initialize every temporal slot from the same fixed base Gaussian set."""
    model = TemporalGeometryChangeModel.from_gaussians(
        base,
        max_states=len(boundaries) + 1,
        initial_time=0.0,
    )
    ranges = segment_ranges(total_frames, boundaries)
    with torch.no_grad():
        model.state_change_dc.zero_()
        for state, (start, end) in enumerate(ranges):
            model.state_change_dc[:, state].copy_(base._features_dc.detach())
            model.state_start[:, state].fill_(float(start))
            model.state_end[:, state].fill_(
                float("inf") if state == len(ranges) - 1 else float(end)
            )
            model.state_valid[:, state].fill_(True)
    return model


def make_geometry_optimizer(
    model: TemporalGeometryChangeModel,
    args: argparse.Namespace,
) -> torch.optim.Adam:
    """Use original O-SCD learning rates without topology-changing operations."""
    learning_rates = {
        "dc": args.dc_lr,
        "xyz": args.xyz_lr,
        "opacity": args.opacity_lr,
        "scaling": args.scaling_lr,
        "rotation": args.rotation_lr,
    }
    groups = [
        {"params": [parameter], "lr": learning_rates[name], "name": name}
        for name, parameter in model.state_parameter_items()
    ]
    return torch.optim.Adam(groups, lr=0.0, eps=1e-15)


def state0_xyz_anchor_loss(
    model: TemporalGeometryChangeModel,
    state: int,
    anchor_indices: torch.Tensor,
    weight: float,
) -> tuple[torch.Tensor, int, float]:
    """Softly retain S0 positions while allowing later cue evidence to move them."""
    zero = model.state_xyz_delta.new_zeros(())
    if state == 0 or weight == 0.0:
        return zero, 0, 0.0
    count = int(anchor_indices.numel())
    if count == 0:
        return zero, 0, 0.0
    difference = (
        model.state_xyz_delta[anchor_indices, state]
        - model.state_xyz_delta[anchor_indices, 0].detach()
    )
    raw = difference.square().sum(dim=1).mean()
    return raw * float(weight), count, float(raw.detach().item())


def slot_gradient_l1(parameter: torch.Tensor) -> list[float]:
    if parameter.grad is None:
        return [0.0] * int(parameter.shape[1])
    dimensions = (0, *range(2, parameter.grad.ndim))
    return parameter.grad.detach().abs().sum(dim=dimensions).cpu().tolist()


def invalid_active_gradient_max(
    parameter: torch.Tensor,
    state: int,
    valid: torch.Tensor,
) -> float:
    if parameter.grad is None or bool(valid.all()):
        return 0.0
    return float(parameter.grad[:, state][~valid].detach().abs().max().item())


def snapshot_state(
    model: TemporalGeometryChangeModel,
    state: int,
) -> dict[str, torch.Tensor]:
    return {
        name: parameter[:, state].detach().cpu().clone()
        for name, parameter in model.state_parameter_items()
    }


def completed_state_drift(
    model: TemporalGeometryChangeModel,
    snapshots: dict[int, dict[str, torch.Tensor]],
) -> dict[str, Any]:
    per_state: dict[str, dict[str, float]] = {}
    maximum = 0.0
    for state, state_snapshot in snapshots.items():
        values: dict[str, float] = {}
        for name, parameter in model.state_parameter_items():
            drift = float(
                (parameter[:, state].detach().cpu() - state_snapshot[name])
                .abs()
                .max()
                .item()
            )
            values[name] = drift
            maximum = max(maximum, drift)
        per_state[str(state)] = values
    return {
        "per_completed_state_parameter_max_abs": per_state,
        "max_abs": maximum,
        "passed": maximum == 0.0,
    }


def train_geometry_slots(
    model: TemporalGeometryChangeModel,
    train_views: list[Camera],
    background: torch.Tensor,
    pipe: SimpleNamespace,
    args: argparse.Namespace,
    dataset_audit: dict[str, Any],
    strong_state0: torch.Tensor,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    views_by_state: dict[int, list[Camera]] = {
        state: [] for state in range(model.max_states)
    }
    for view in train_views:
        views_by_state[int(view.segment_id)].append(view)
    schedule = make_training_schedule(views_by_state, model.max_states, args)
    schedule_audit = training_schedule_audit(
        train_views,
        schedule,
        args,
        dataset_audit,
    )
    if not schedule_audit["exact_all_images_guarantee"]:
        raise RuntimeError(f"Exact per-frame schedule failed: {schedule_audit}")

    optimizer: torch.optim.Adam | None = None
    optimizer_state: int | None = None
    snapshots: dict[int, dict[str, torch.Tensor]] = {}
    logs: list[dict[str, Any]] = []
    audited_steps = 0
    isolation_violations = 0
    invalid_gradient_violations = 0
    max_inactive_slot_gradient = 0.0
    max_invalid_active_gradient = 0.0
    anchor_counts = {
        str(state): int((strong_state0 & model.state_valid[:, state]).sum().item())
        if state > 0
        else 0
        for state in range(model.max_states)
    }
    anchor_indices = {
        state: torch.nonzero(
            strong_state0 & model.state_valid[:, state],
            as_tuple=False,
        ).flatten()
        for state in range(1, model.max_states)
    }
    total_steps = len(schedule)
    started = time.time()

    for global_step, step in enumerate(schedule, start=1):
        state_changed = optimizer_state != step.state
        if state_changed:
            if optimizer_state is not None:
                snapshots[optimizer_state] = snapshot_state(model, optimizer_state)
            optimizer = make_geometry_optimizer(model, args)
            optimizer_state = step.state

        view = views_by_state[step.state][step.view_index]
        assert optimizer is not None
        optimizer.zero_grad(set_to_none=True)
        package = render_change_temporal(
            view,
            model,
            pipe,
            background,
            timestamp=float(view.timestamp),
        )
        data_loss, parts = oscd_positive_sparsity_loss(
            training_target(view),
            package["render"],
        )
        anchor_loss, anchor_count, raw_anchor = state0_xyz_anchor_loss(
            model,
            step.state,
            anchor_indices.get(
                step.state,
                torch.empty(0, dtype=torch.long, device=model.state_valid.device),
            ),
            args.xyz_anchor_weight,
        )
        total_loss = data_loss + anchor_loss
        if not bool(torch.isfinite(total_loss)):
            raise FloatingPointError(
                f"Non-finite loss at step {global_step}, state {step.state}"
            )
        total_loss.backward()

        audit = should_audit_gradient(
            global_step,
            total_steps,
            args.gradient_audit_interval,
        )
        parameter_audit: dict[str, Any] = {}
        if audit:
            audited_steps += 1
            for name, parameter in model.state_parameter_items():
                slot_l1 = slot_gradient_l1(parameter)
                inactive = max(
                    (
                        value
                        for state, value in enumerate(slot_l1)
                        if state != step.state
                    ),
                    default=0.0,
                )
                invalid = invalid_active_gradient_max(
                    parameter,
                    step.state,
                    model.state_valid[:, step.state],
                )
                max_inactive_slot_gradient = max(
                    max_inactive_slot_gradient,
                    inactive,
                )
                max_invalid_active_gradient = max(
                    max_invalid_active_gradient,
                    invalid,
                )
                isolation_violations += int(inactive != 0.0)
                invalid_gradient_violations += int(invalid != 0.0)
                parameter_audit[name] = {
                    "slot_grad_l1": slot_l1,
                    "inactive_slot_max_l1": inactive,
                    "invalid_active_row_max_abs": invalid,
                    "active_slot_received_gradient": slot_l1[step.state] > 0.0,
                }

        optimizer.step()
        if audit:
            active_parameters_finite = all(
                bool(torch.isfinite(parameter[:, step.state]).all())
                for _name, parameter in model.state_parameter_items()
            )
            if not active_parameters_finite:
                raise FloatingPointError(
                    f"Non-finite state parameter at step {global_step}, state {step.state}"
                )

        should_log = (
            global_step in {1, total_steps}
            or global_step % max(1, args.progress_interval) == 0
            or (audit and len(logs) < 20)
        )
        if should_log:
            row = {
                "global_step": global_step,
                "state": step.state,
                "epoch": step.epoch,
                "frame": view.image_name,
                "data_loss": parts["loss"],
                "total_loss": float(total_loss.detach().item()),
                "anchor_loss": float(anchor_loss.detach().item()),
                "anchor_raw_mean_squared_distance": raw_anchor,
                "anchor_gaussian_count": anchor_count,
                "audited_gradient": audit,
            }
            if audit:
                row["parameter_gradient_audit"] = parameter_audit
            logs.append(row)

        if (
            global_step in {1, total_steps}
            or global_step % max(1, args.progress_interval) == 0
        ):
            elapsed = time.time() - started
            print(
                f"[geometry] step {global_step}/{total_steps} "
                f"state={step.state} frame={view.image_name} "
                f"data={parts['loss']:.6f} anchor={float(anchor_loss.detach()):.6f} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    drift = completed_state_drift(model, snapshots)
    audit_summary = {
        "optimized_steps": total_steps,
        "audited_steps": audited_steps,
        "audit_interval": args.gradient_audit_interval,
        "inactive_slot_violations": isolation_violations,
        "invalid_active_row_violations": invalid_gradient_violations,
        "max_inactive_slot_grad_l1": max_inactive_slot_gradient,
        "max_invalid_active_row_grad_abs": max_invalid_active_gradient,
        "all_audited_gradients_isolated": (
            isolation_violations == 0 and invalid_gradient_violations == 0
        ),
        "optimizer_reset_on_state_boundary": True,
        "completed_state_drift": drift,
        "state0_anchor_gaussian_counts": anchor_counts,
    }
    return logs, audit_summary, schedule_audit


def distribution(values: torch.Tensor) -> dict[str, float | int | None]:
    if values.numel() == 0:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    values = values.detach().float()
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "median": float(values.median().item()),
        "p95": float(torch.quantile(values, 0.95).item()),
        "max": float(values.max().item()),
    }


def geometry_movement_summary(
    model: TemporalGeometryChangeModel,
    strong_state0: torch.Tensor,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    with torch.no_grad():
        for state in range(model.max_states):
            valid = model.state_valid[:, state]
            from_base = model.state_xyz_delta[:, state].norm(dim=1)
            anchored = strong_state0 & valid
            from_state0 = (
                model.state_xyz_delta[:, state] - model.state_xyz_delta[:, 0]
            ).norm(dim=1)
            summary[str(state)] = {
                "valid_from_base_xyz_distance": distribution(from_base[valid]),
                "strong_s0_overlap_from_s0_xyz_distance": distribution(
                    from_state0[anchored]
                ),
            }
    return summary


def save_run(
    model: TemporalGeometryChangeModel,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    records: list[FrameRecord],
    poses: dict[str, PoseResult],
    intrinsics: np.ndarray,
    cue_metadata: dict[str, Any],
    support: dict[str, Any],
    loss_summary: dict[str, Any],
    train_log: list[dict[str, Any]],
    gradient_audit: dict[str, Any],
    schedule_audit: dict[str, Any],
    strong_state0: torch.Tensor,
) -> tuple[Path, Path]:
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / "temporal_rchange_checkpoint.pt"
    summary_path = output_dir / "summary.json"
    base_ply = (Path(args.source_path) / BASE_PLY_REL).resolve()
    contract = (
        "fixed_topology_temporal_geometry_dc_oscd_pixel_sam_cue_"
        "manual_boundaries_state0_xyz_anchor"
    )
    metadata = {
        "schema_version": 3,
        "contract": contract,
        "source_path": str(Path(args.source_path).resolve()),
        "base_ply_sha256": file_checksum(base_ply),
        "fixed_cameras_sha256": file_checksum(Path(args.fixed_cameras_json)),
        "cue_cache_metadata_sha256": file_checksum(
            Path(args.cue_cache_root) / "metadata.json"
        ),
        "boundaries": list(args.boundaries),
        "camera_intrinsics": intrinsics.tolist(),
        "support": support,
        "schedule": schedule_audit,
        "gradient_isolation_audit": gradient_audit,
        "gt_used_for_training": False,
        "densification": False,
        "pruning": False,
        "state0_anchor_min_support_views": int(
            args.state0_anchor_min_support_views
        ),
        "strong_state0_anchor_mask_checksum": tensor_checksum(strong_state0),
    }
    state_dict = {
        name: value.detach().cpu() for name, value in model.state_dict().items()
    }
    torch.save(
        {
            "state_dict": state_dict,
            "base_ply": str(base_ply),
            "boundaries": list(args.boundaries),
            "contract": contract,
            "metadata": metadata,
            "strong_state0_anchor_mask": strong_state0.detach().cpu(),
        },
        checkpoint_path,
    )

    state_valid = model.state_valid.detach()
    parameter_checksums = {
        name: tensor_checksum(parameter)
        for name, parameter in model.state_parameter_items()
    }
    summary = {
        "script": "experiments/train_geometry_temporal_rchange.py",
        "contract": contract,
        "created_at_unix": time.time(),
        "runtime_seconds": time.time() - args.started_at,
        "source_path": str(args.source_path),
        "supervision": "O-SCD pixel cue + SAM2.1 Hiera Tiny feature cue",
        "oracle_supervision": False,
        "gt_used_for_training": False,
        "gt_mask_pixels_loaded": 0,
        "manual_boundaries": True,
        "bocd": False,
        "fixed_gaussian_topology": True,
        "optimized_parameters": [
            "state_change_dc",
            "state_xyz_delta",
            "state_opacity_delta",
            "state_scaling_delta",
            "state_rotation_delta",
        ],
        "fixed_parameters": ["base._features_rest"],
        "densify_prune": False,
        "resolution": float(args.resolution),
        "boundaries": list(args.boundaries),
        "fixed_cameras_json": str(Path(args.fixed_cameras_json).resolve()),
        "fixed_cameras_sha256": file_checksum(Path(args.fixed_cameras_json)),
        "cue_cache_root": str(Path(args.cue_cache_root).resolve()),
        "cue_cache_metadata_sha256": file_checksum(
            Path(args.cue_cache_root) / "metadata.json"
        ),
        "cue_cache_metadata": cue_metadata,
        "run_arguments": serializable_arguments(args),
        "config": {
            "training_mode": schedule_audit["mode"],
            "updates_per_frame": int(args.updates_per_frame),
            "learning_rates": {
                "dc": float(args.dc_lr),
                "xyz": float(args.xyz_lr),
                "opacity": float(args.opacity_lr),
                "scaling": float(args.scaling_lr),
                "rotation": float(args.rotation_lr),
            },
            "xyz_anchor_weight": float(args.xyz_anchor_weight),
            "state0_anchor_min_support_views": int(
                args.state0_anchor_min_support_views
            ),
            "support_map_threshold": float(args.support_map_threshold),
            "support_count_threshold": int(args.support_threshold),
            "seed": int(args.seed),
        },
        "manifest_counts": manifest.get("counts"),
        "train_frames": [
            {
                "global_index": record.global_index,
                "segment_id": record.segment_id,
                "name": record.name,
                "image_path": record.image_path,
            }
            for record in records
        ],
        "pose_policy": "O-SCD fixed canonical pose reused from fixed-pose protocol",
        "pose_results": {name: asdict(result) for name, result in poses.items()},
        "camera_intrinsics": intrinsics.tolist(),
        "support": support,
        "strong_state0_anchor_gaussians": int(strong_state0.sum().item()),
        "strong_state0_anchor_mask_checksum": tensor_checksum(strong_state0),
        "loss_summary": loss_summary,
        "train_log": train_log,
        "training_schedule": schedule_audit,
        "gradient_isolation_audit": gradient_audit,
        "geometry_movement": geometry_movement_summary(model, strong_state0),
        "gaussian_count": int(model.state_change_dc.shape[0]),
        "state_valid_counts": state_valid.sum(dim=0).cpu().tolist(),
        "states_per_gaussian_histogram": {
            str(count): int((state_valid.sum(dim=1) == count).sum().item())
            for count in range(model.max_states + 1)
        },
        "state_parameter_shapes": {
            name: list(parameter.shape)
            for name, parameter in model.state_parameter_items()
        },
        "state_parameter_checksums": parameter_checksums,
        "checkpoint_sha256": file_checksum(checkpoint_path),
        "base_ply_sha256": file_checksum(base_ply),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return checkpoint_path, summary_path


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train fixed-topology temporal DC and geometry from O-SCD cues"
    )
    parser.add_argument("--source-path", type=Path, default=Path(DEFAULT_SOURCE))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fixed-cameras-json", type=Path, default=DEFAULT_FIXED_CAMERAS)
    parser.add_argument("--cue-cache-root", type=Path, default=DEFAULT_CUE_CACHE)
    parser.add_argument("--resolution", type=float, default=4.0)
    parser.add_argument("--updates-per-frame", type=positive_int, default=120)
    parser.add_argument("--boundaries", nargs="+", type=int, default=None)
    parser.add_argument("--dc-lr", type=nonnegative_float, default=0.0025)
    parser.add_argument("--xyz-lr", type=nonnegative_float, default=0.00016)
    parser.add_argument("--opacity-lr", type=nonnegative_float, default=0.025)
    parser.add_argument("--scaling-lr", type=nonnegative_float, default=0.005)
    parser.add_argument("--rotation-lr", type=nonnegative_float, default=0.001)
    parser.add_argument("--xyz-anchor-weight", type=nonnegative_float, default=1.0)
    parser.add_argument(
        "--state0-anchor-min-support-views",
        type=positive_int,
        default=3,
    )
    parser.add_argument("--support-map-threshold", type=float, default=0.5)
    parser.add_argument("--support-threshold", type=positive_int, default=1)
    parser.add_argument("--gradient-audit-interval", type=positive_int, default=100)
    parser.add_argument("--progress-interval", type=positive_int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.started_at = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the Gaussian renderer")
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest(args.source_path)
    args.boundaries = tuple(
        args.boundaries
        if args.boundaries is not None
        else boundaries_from_manifest(manifest, DEFAULT_BOUNDARIES)
    )
    records, _, all_names = build_frame_records(
        args.source_path,
        frames_per_state=1,
        probes=[],
        boundaries=args.boundaries,
        all_training_frames=True,
    )
    base_ply = (args.source_path / BASE_PLY_REL).resolve()
    cue_metadata = validate_cue_cache(
        args.cue_cache_root,
        base_ply,
        args.resolution,
    )
    cameras = load_fixed_camera_index(args.fixed_cameras_json)
    views, poses, intrinsics = build_fixed_cue_views(
        records,
        cameras,
        args.cue_cache_root,
        args.resolution,
    )
    dataset_audit = validate_exact_dataset_contract(
        manifest,
        all_names,
        records,
        views,
        args.updates_per_frame,
    )

    base = GaussianModel(sh_degree=3, active_sh_degree=0)
    base.load_ply_change(str(base_ply))
    model = build_geometry_temporal_model(base, args.boundaries, len(records))
    pipe = SimpleNamespace(
        compute_cov3D_python=False,
        convert_SHs_python=False,
        debug=False,
    )
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    support_view_counts = torch.zeros_like(model.state_valid, dtype=torch.int16)
    support = compute_state_support(
        model,
        views,
        background,
        pipe,
        args.support_threshold,
        map_threshold=args.support_map_threshold,
        support_view_counts_out=support_view_counts,
    )
    strong_state0 = (
        model.state_valid[:, 0]
        & (
            support_view_counts[:, 0]
            >= int(args.state0_anchor_min_support_views)
        )
    )

    pre_loss = summarize_losses(model, views, background, pipe)
    train_log, gradient_audit, schedule_audit = train_geometry_slots(
        model,
        views,
        background,
        pipe,
        args,
        dataset_audit,
        strong_state0,
    )
    post_loss = summarize_losses(model, views, background, pipe)
    checkpoint_path, summary_path = save_run(
        model,
        args,
        manifest,
        records,
        poses,
        intrinsics,
        cue_metadata,
        support,
        {"pre_train": pre_loss, "post_train": post_loss},
        train_log,
        gradient_audit,
        schedule_audit,
        strong_state0,
    )
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "summary": str(summary_path),
                "frames": len(views),
                "updates": schedule_audit["actual_total_updates"],
                "strong_state0_anchor_gaussians": int(strong_state0.sum().item()),
                "gt_used_for_training": False,
                "densify_prune": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
