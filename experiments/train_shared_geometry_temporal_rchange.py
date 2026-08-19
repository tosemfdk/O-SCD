"""Train temporal DC slots while all states overwrite one shared geometry.

This is an intentional forgetting experiment.  State DC and lifespans remain
separate, but xyz, opacity, scale, and rotation are a single mutable parameter
set.  The states are optimized sequentially without cross-state replay, so a
later state can damage the geometry used to re-render an earlier state.
"""

from __future__ import annotations

import argparse
import json
import os
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
from experiments.train_geometry_temporal_rchange import distribution
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
from temporal import TemporalSharedGeometryChangeModel


DEFAULT_OUTPUT = Path(
    "outputs/instance1_scene_change1_2_3_temporal_shared_geometry_"
    "forgetting_oscd_cues_allframes_120"
)
CONTRACT = (
    "fixed_topology_temporal_dc_shared_mutable_geometry_"
    "sequential_no_cross_state_replay_manual_boundaries"
)


def build_shared_geometry_model(
    base: GaussianModel,
    boundaries: tuple[int, ...],
    total_frames: int,
) -> TemporalSharedGeometryChangeModel:
    model = TemporalSharedGeometryChangeModel.from_gaussians(
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


def make_optimizer(
    model: TemporalSharedGeometryChangeModel,
    args: argparse.Namespace,
) -> torch.optim.Adam:
    rates = {
        "dc": args.dc_lr,
        "xyz": args.xyz_lr,
        "opacity": args.opacity_lr,
        "scaling": args.scaling_lr,
        "rotation": args.rotation_lr,
    }
    groups = [
        {"params": [model.state_change_dc], "lr": rates["dc"], "name": "dc"}
    ]
    groups.extend(
        {
            "params": [parameter],
            "lr": rates[name],
            "name": f"shared_{name}",
        }
        for name, parameter in model.shared_geometry_parameter_items()
    )
    return torch.optim.Adam(groups, lr=0.0, eps=1e-15)


def slot_gradient_l1(parameter: torch.Tensor) -> list[float]:
    if parameter.grad is None:
        return [0.0] * int(parameter.shape[1])
    dimensions = (0, *range(2, parameter.grad.ndim))
    return parameter.grad.detach().abs().sum(dim=dimensions).cpu().tolist()


def invalid_row_gradient_max(
    parameter: torch.Tensor,
    valid: torch.Tensor,
) -> float:
    if parameter.grad is None or bool(valid.all()):
        return 0.0
    return float(parameter.grad[~valid].detach().abs().max().item())


def invalid_dc_gradient_max(
    parameter: torch.Tensor,
    state: int,
    valid: torch.Tensor,
) -> float:
    if parameter.grad is None or bool(valid.all()):
        return 0.0
    return float(parameter.grad[:, state][~valid].detach().abs().max().item())


def checkpoint_payload(
    model: TemporalSharedGeometryChangeModel,
    base_ply: Path,
    args: argparse.Namespace,
    completed_through_state: int,
) -> dict[str, Any]:
    return {
        "state_dict": {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        },
        "base_ply": str(base_ply),
        "boundaries": list(args.boundaries),
        "contract": CONTRACT,
        "metadata": {
            "schema_version": 4,
            "contract": CONTRACT,
            "completed_through_state": int(completed_through_state),
            "shared_mutable_geometry": True,
            "cross_state_replay": False,
            "fixed_topology": True,
            "densification": False,
            "pruning": False,
        },
    }


def save_stage_checkpoint(
    model: TemporalSharedGeometryChangeModel,
    base_ply: Path,
    args: argparse.Namespace,
    state: int,
) -> Path:
    path = Path(args.output_dir) / f"state{state}_complete_checkpoint.pt"
    torch.save(checkpoint_payload(model, base_ply, args, state), path)
    return path


def shared_snapshot(
    model: TemporalSharedGeometryChangeModel,
) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.shared_geometry_parameter_items()
    }


def geometry_transition_summary(
    previous: dict[str, torch.Tensor] | None,
    current: dict[str, torch.Tensor],
    valid: torch.Tensor,
) -> dict[str, Any]:
    valid_cpu = valid.detach().cpu()
    result: dict[str, Any] = {}
    for name, value in current.items():
        baseline = torch.zeros_like(value) if previous is None else previous[name]
        difference = value - baseline
        row_magnitude = difference.flatten(start_dim=1).norm(dim=1)
        result[name] = distribution(row_magnitude[valid_cpu])
    return result


def train_sequentially(
    model: TemporalSharedGeometryChangeModel,
    views: list[Camera],
    background: torch.Tensor,
    pipe: SimpleNamespace,
    args: argparse.Namespace,
    dataset_audit: dict[str, Any],
    base_ply: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, str],
]:
    views_by_state = {state: [] for state in range(model.max_states)}
    for view in views:
        views_by_state[int(view.segment_id)].append(view)
    schedule = make_training_schedule(views_by_state, model.max_states, args)
    schedule_audit = training_schedule_audit(
        views,
        schedule,
        args,
        dataset_audit,
    )
    if not schedule_audit["exact_all_images_guarantee"]:
        raise RuntimeError(f"Exact per-frame schedule failed: {schedule_audit}")

    optimizer: torch.optim.Adam | None = None
    optimizer_state: int | None = None
    completed_dc: dict[int, torch.Tensor] = {}
    previous_geometry: dict[str, torch.Tensor] | None = None
    geometry_transitions: dict[str, Any] = {}
    stage_paths: dict[str, str] = {}
    logs: list[dict[str, Any]] = []
    audited_steps = 0
    inactive_dc_violations = 0
    invalid_dc_violations = 0
    invalid_shared_violations = 0
    max_inactive_dc = 0.0
    max_invalid_dc = 0.0
    max_invalid_shared = 0.0
    total_steps = len(schedule)
    started = time.time()

    def complete_state(state: int) -> None:
        nonlocal previous_geometry
        completed_dc[state] = model.state_change_dc[:, state].detach().cpu().clone()
        current_geometry = shared_snapshot(model)
        geometry_transitions[f"after_state_{state}"] = {
            "relative_to": "base" if previous_geometry is None else f"after_state_{state - 1}",
            "valid_gaussian_movement": geometry_transition_summary(
                previous_geometry,
                current_geometry,
                model.state_valid[:, state],
            ),
            "shared_parameter_checksums": {
                name: tensor_checksum(value)
                for name, value in current_geometry.items()
            },
        }
        previous_geometry = current_geometry
        path = save_stage_checkpoint(model, base_ply, args, state)
        stage_paths[str(state)] = str(path)
        print(
            f"[shared-geometry] saved S{state} boundary checkpoint: {path}",
            flush=True,
        )

    for global_step, step in enumerate(schedule, start=1):
        if optimizer_state != step.state:
            if optimizer_state is not None:
                complete_state(optimizer_state)
            optimizer = make_optimizer(model, args)
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
        loss, parts = oscd_positive_sparsity_loss(
            training_target(view),
            package["render"],
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(
                f"Non-finite loss at step {global_step}, state {step.state}"
            )
        loss.backward()

        audit = should_audit_gradient(
            global_step,
            total_steps,
            args.gradient_audit_interval,
        )
        audit_row: dict[str, Any] = {}
        if audit:
            audited_steps += 1
            dc_slots = slot_gradient_l1(model.state_change_dc)
            inactive_dc = max(
                (
                    value
                    for state, value in enumerate(dc_slots)
                    if state != step.state
                ),
                default=0.0,
            )
            valid = model.state_valid[:, step.state]
            invalid_dc = invalid_dc_gradient_max(
                model.state_change_dc, step.state, valid
            )
            shared_invalid: dict[str, float] = {}
            shared_grad_l1: dict[str, float] = {}
            for name, parameter in model.shared_geometry_parameter_items():
                shared_invalid[name] = invalid_row_gradient_max(parameter, valid)
                shared_grad_l1[name] = (
                    0.0
                    if parameter.grad is None
                    else float(parameter.grad.detach().abs().sum().item())
                )
            invalid_shared = max(shared_invalid.values(), default=0.0)
            max_inactive_dc = max(max_inactive_dc, inactive_dc)
            max_invalid_dc = max(max_invalid_dc, invalid_dc)
            max_invalid_shared = max(max_invalid_shared, invalid_shared)
            inactive_dc_violations += int(inactive_dc != 0.0)
            invalid_dc_violations += int(invalid_dc != 0.0)
            invalid_shared_violations += int(invalid_shared != 0.0)
            audit_row = {
                "dc_slot_grad_l1": dc_slots,
                "inactive_dc_slot_max_l1": inactive_dc,
                "invalid_active_dc_row_max_abs": invalid_dc,
                "shared_geometry_grad_l1": shared_grad_l1,
                "invalid_shared_geometry_row_max_abs": shared_invalid,
            }

        optimizer.step()
        if audit and not all(
            bool(torch.isfinite(parameter).all())
            for parameter in (
                model.state_change_dc,
                *(p for _name, p in model.shared_geometry_parameter_items()),
            )
        ):
            raise FloatingPointError(
                f"Non-finite parameter at step {global_step}, state {step.state}"
            )

        should_log = (
            global_step in {1, total_steps}
            or global_step % max(1, args.progress_interval) == 0
            or (audit and len(logs) < 20)
        )
        if should_log:
            row: dict[str, Any] = {
                "global_step": global_step,
                "state": step.state,
                "epoch": step.epoch,
                "frame": view.image_name,
                **parts,
                "audited_gradient": audit,
            }
            if audit:
                row["gradient_audit"] = audit_row
            logs.append(row)
        if (
            global_step in {1, total_steps}
            or global_step % max(1, args.progress_interval) == 0
        ):
            print(
                f"[shared-geometry] step {global_step}/{total_steps} "
                f"state={step.state} frame={view.image_name} "
                f"loss={parts['loss']:.6f} elapsed={time.time() - started:.1f}s",
                flush=True,
            )

    if optimizer_state is not None:
        complete_state(optimizer_state)

    drift_by_state: dict[str, float] = {}
    max_dc_drift = 0.0
    for state, snapshot in completed_dc.items():
        drift = float(
            (model.state_change_dc[:, state].detach().cpu() - snapshot)
            .abs()
            .max()
            .item()
        )
        drift_by_state[str(state)] = drift
        max_dc_drift = max(max_dc_drift, drift)

    audit_summary = {
        "optimized_steps": total_steps,
        "audited_steps": audited_steps,
        "audit_interval": args.gradient_audit_interval,
        "inactive_dc_slot_violations": inactive_dc_violations,
        "invalid_active_dc_row_violations": invalid_dc_violations,
        "invalid_shared_geometry_row_violations": invalid_shared_violations,
        "max_inactive_dc_slot_grad_l1": max_inactive_dc,
        "max_invalid_active_dc_row_grad_abs": max_invalid_dc,
        "max_invalid_shared_geometry_row_grad_abs": max_invalid_shared,
        "all_audited_gradients_isolated": (
            inactive_dc_violations == 0
            and invalid_dc_violations == 0
            and invalid_shared_violations == 0
        ),
        "optimizer_reset_on_state_boundary": True,
        "completed_state_dc_drift": {
            "per_state_max_abs": drift_by_state,
            "max_abs": max_dc_drift,
            "passed": max_dc_drift == 0.0,
        },
        "shared_geometry_transition_movement": geometry_transitions,
    }
    return logs, audit_summary, schedule_audit, stage_paths


def base_checksums(base: GaussianModel) -> dict[str, str]:
    return {
        name: tensor_checksum(getattr(base, name))
        for name in (
            "_xyz",
            "_features_dc",
            "_features_rest",
            "_opacity",
            "_scaling",
            "_rotation",
        )
    }


def save_summary(
    model: TemporalSharedGeometryChangeModel,
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
    stage_paths: dict[str, str],
    base_before: dict[str, str],
) -> tuple[Path, Path]:
    output_dir = Path(args.output_dir)
    base_ply = (Path(args.source_path) / BASE_PLY_REL).resolve()
    final_stage_path = Path(stage_paths[str(model.max_states - 1)])
    checkpoint_path = output_dir / "temporal_rchange_checkpoint.pt"
    if checkpoint_path.exists() or checkpoint_path.is_symlink():
        checkpoint_path.unlink()
    os.symlink(final_stage_path.name, checkpoint_path)

    base_after = base_checksums(model.base)
    state_valid = model.state_valid.detach()
    stage_checkpoints = {
        state: {
            "path": path,
            "sha256": file_checksum(Path(path)),
            "completed_through_state": int(state),
        }
        for state, path in stage_paths.items()
    }
    summary = {
        "script": "experiments/train_shared_geometry_temporal_rchange.py",
        "contract": CONTRACT,
        "created_at_unix": time.time(),
        "runtime_seconds": time.time() - args.started_at,
        "source_path": str(args.source_path),
        "supervision": "O-SCD pixel cue + SAM2.1 Hiera Tiny feature cue",
        "oracle_supervision": False,
        "gt_used_for_training": False,
        "manual_boundaries": True,
        "bocd": False,
        "fixed_gaussian_topology": True,
        "densify_prune": False,
        "shared_mutable_geometry": True,
        "cross_state_replay": False,
        "state_specific_parameters": ["state_change_dc", "state lifespan"],
        "shared_parameters": [
            "shared_xyz_delta",
            "shared_opacity_delta",
            "shared_scaling_delta",
            "shared_rotation_delta",
        ],
        "fixed_parameters": [
            "base._xyz",
            "base._features_dc",
            "base._features_rest",
            "base._opacity",
            "base._scaling",
            "base._rotation",
        ],
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
        "loss_summary": loss_summary,
        "train_log": train_log,
        "training_schedule": schedule_audit,
        "gradient_isolation_audit": gradient_audit,
        "stage_checkpoints": stage_checkpoints,
        "gaussian_count": int(model.state_change_dc.shape[0]),
        "state_valid_counts": state_valid.sum(dim=0).cpu().tolist(),
        "states_per_gaussian_histogram": {
            str(count): int((state_valid.sum(dim=1) == count).sum().item())
            for count in range(model.max_states + 1)
        },
        "parameter_shapes": {
            "state_change_dc": list(model.state_change_dc.shape),
            **{
                f"shared_{name}_delta": list(parameter.shape)
                for name, parameter in model.shared_geometry_parameter_items()
            },
        },
        "parameter_checksums": {
            "state_change_dc": tensor_checksum(model.state_change_dc),
            **{
                f"shared_{name}_delta": tensor_checksum(parameter)
                for name, parameter in model.shared_geometry_parameter_items()
            },
        },
        "base_parameter_checksums_before": base_before,
        "base_parameter_checksums_after": base_after,
        "base_parameters_unchanged": base_before == base_after,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": file_checksum(checkpoint_path),
        "base_ply_sha256": file_checksum(base_ply),
    }
    summary_path = output_dir / "summary.json"
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
        description=(
            "Train state-specific DC with one shared mutable geometry to expose "
            "cross-state forgetting"
        )
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
    model = build_shared_geometry_model(base, args.boundaries, len(records))
    base_before = base_checksums(base)
    pipe = SimpleNamespace(
        compute_cov3D_python=False,
        convert_SHs_python=False,
        debug=False,
    )
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    support = compute_state_support(
        model,
        views,
        background,
        pipe,
        args.support_threshold,
        map_threshold=args.support_map_threshold,
    )

    pre_loss = summarize_losses(model, views, background, pipe)
    train_log, gradient_audit, schedule_audit, stage_paths = train_sequentially(
        model,
        views,
        background,
        pipe,
        args,
        dataset_audit,
        base_ply,
    )
    post_loss = summarize_losses(model, views, background, pipe)
    checkpoint_path, summary_path = save_summary(
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
        stage_paths,
        base_before,
    )
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "summary": str(summary_path),
                "stage_checkpoints": stage_paths,
                "frames": len(views),
                "updates": schedule_audit["actual_total_updates"],
                "shared_mutable_geometry": True,
                "cross_state_replay": False,
                "gt_used_for_training": False,
                "densify_prune": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
