"""Continue shared-geometry training on S1 while freezing S0 influence-valid rows.

The experiment starts from the exact S0 boundary checkpoint of the shared
geometry forgetting run. Signed additive/occluding influence is computed
offline at that boundary. Those rows cannot move during S1, either through the
legacy external anchor/projection implementation or the persistent per-GS
``geometry_frozen`` gradient mask; all other S1-active rows remain trainable.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from experiments.train_cue_temporal_rchange import (
    build_fixed_cue_views,
    load_fixed_camera_index,
    serializable_arguments,
    validate_cue_cache,
)
from experiments.train_geometry_temporal_rchange import distribution
from experiments.train_real_temporal_rchange import (
    FrameRecord,
    file_checksum,
    oscd_positive_sparsity_loss,
    seed_everything,
    summarize_losses,
    tensor_checksum,
    training_target,
)
from experiments.train_shared_geometry_temporal_rchange import (
    base_checksums,
    make_optimizer,
)
from gaussian_renderer import render_change_temporal
from temporal import TemporalSharedGeometryChangeModel, load_temporal_model
from temporal.geometry_freeze import (
    capture_frozen_rows,
    mask_frozen_row_gradients,
    max_frozen_row_drift,
    restore_frozen_rows,
)


SOURCE_RUN = Path(
    "outputs/instance1_scene_change1_2_3_temporal_shared_geometry_"
    "forgetting_oscd_cues_allframes_120"
)
DEFAULT_OUTPUT = Path(
    "outputs/instance1_scene_change1_2_3_temporal_shared_geometry_"
    "s0_influence_freeze_s1_oscd_cues_allframes_120"
)
MODEL_BUFFER_OUTPUT = Path(
    "outputs/instance1_scene_change1_2_3_temporal_shared_geometry_"
    "s0_geometry_frozen_attr_s1_oscd_cues_allframes_120"
)
ALL_GEOMETRY_OUTPUT = Path(
    "outputs/instance1_scene_change1_2_3_temporal_shared_geometry_"
    "s0_all_geometry_frozen_s1_dc_only_oscd_cues_allframes_120"
)
EXTERNAL_PROJECTION_CONTRACT = (
    "shared_geometry_s0_boundary_signed_influence_first_owner_"
    "hard_freeze_s1_continuation"
)
MODEL_BUFFER_CONTRACT = (
    "shared_geometry_s0_boundary_signed_influence_persistent_frozen_"
    "metadata_gradient_mask_s1_continuation"
)
ALL_GEOMETRY_CONTRACT = (
    "shared_geometry_s0_boundary_all_geometry_persistent_frozen_"
    "s1_dc_only_continuation"
)


def experiment_contract(
    freeze_implementation: str,
    freeze_scope: str = "influence_valid",
) -> str:
    if freeze_scope == "all_geometry":
        if freeze_implementation != "model_buffer_gradient_mask":
            raise ValueError(
                "all_geometry requires model_buffer_gradient_mask"
            )
        return ALL_GEOMETRY_CONTRACT
    if freeze_scope != "influence_valid":
        raise ValueError(f"Unsupported freeze scope: {freeze_scope}")
    if freeze_implementation == "external_projection":
        return EXTERNAL_PROJECTION_CONTRACT
    if freeze_implementation == "model_buffer_gradient_mask":
        return MODEL_BUFFER_CONTRACT
    raise ValueError(f"Unsupported freeze implementation: {freeze_implementation}")


def records_from_summary(
    summary: dict[str, Any],
    states: set[int],
) -> list[FrameRecord]:
    records = [
        FrameRecord(
            global_index=int(item["global_index"]),
            segment_id=int(item["segment_id"]),
            name=str(item["name"]),
            image_path=str(item["image_path"]),
            mask_path="",
        )
        for item in summary["train_frames"]
        if int(item["segment_id"]) in states
    ]
    records.sort(key=lambda record: record.global_index)
    return records


def load_influence_mask(
    artifact_path: Path,
    checkpoint_path: Path,
    gaussian_count: int,
) -> tuple[dict[str, Any], torch.Tensor]:
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    if artifact.get("contract") != "offline_signed_opacity_removal_influence_fixed_topology":
        raise ValueError("Unsupported signed-influence artifact contract")
    if int(artifact.get("state", -1)) != 0:
        raise ValueError("The freeze artifact must describe state 0")
    if artifact.get("checkpoint_sha256") != file_checksum(checkpoint_path):
        raise ValueError("Influence artifact and S0 checkpoint do not match")
    valid = artifact.get("valid_mask")
    if not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool:
        raise TypeError("Influence artifact valid_mask must be a bool tensor")
    if tuple(valid.shape) != (gaussian_count,):
        raise ValueError("Influence valid_mask has the wrong Gaussian count")
    return artifact, valid


def movement_summary(
    before: dict[str, torch.Tensor],
    model: TemporalSharedGeometryChangeModel,
    frozen_cpu: torch.Tensor,
) -> dict[str, Any]:
    state1_valid = model.state_valid[:, 1].detach().cpu()
    groups = {
        "s0_influence_frozen": frozen_cpu,
        "all_unfrozen": ~frozen_cpu,
        "s1_valid_frozen": state1_valid & frozen_cpu,
        "s1_valid_unfrozen": state1_valid & ~frozen_cpu,
    }
    result: dict[str, Any] = {}
    for name, parameter in model.shared_geometry_parameter_items():
        after = parameter.detach().cpu()
        row_distance = (after - before[name]).flatten(start_dim=1).norm(dim=1)
        result[name] = {
            group: distribution(row_distance[mask])
            for group, mask in groups.items()
        }
    return result


@torch.no_grad()
def max_snapshot_frozen_drift(
    before: dict[str, torch.Tensor],
    model: TemporalSharedGeometryChangeModel,
    frozen_cpu: torch.Tensor,
) -> dict[str, float]:
    drift: dict[str, float] = {}
    for name, parameter in model.shared_geometry_parameter_items():
        difference = parameter.detach().cpu()[frozen_cpu] - before[name][frozen_cpu]
        drift[name] = (
            0.0
            if difference.numel() == 0
            else float(difference.abs().max().item())
        )
    return drift


def save_checkpoint(
    model: TemporalSharedGeometryChangeModel,
    args: argparse.Namespace,
    source_summary: dict[str, Any],
    frozen_count: int,
    artifact: dict[str, Any] | None,
) -> Path:
    contract = experiment_contract(
        args.freeze_implementation,
        args.freeze_scope,
    )
    path = args.output_dir / "state1_complete_checkpoint.pt"
    torch.save(
        {
            "state_dict": {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            },
            "base_ply": source_summary["stage_checkpoints"]["0"].get(
                "base_ply", source_summary.get("base_ply", str(
                    Path(source_summary["source_path"])
                    / "reference_reconstruction/point_cloud/iteration_30000/point_cloud.ply"
                ))
            ),
            "boundaries": source_summary["boundaries"],
            "contract": contract,
            "metadata": {
                "schema_version": 5,
                "contract": contract,
                "completed_through_state": 1,
                "source_state0_checkpoint": str(args.source_checkpoint.resolve()),
                "source_state0_checkpoint_sha256": file_checksum(args.source_checkpoint),
                "signed_influence_artifact": (
                    None
                    if artifact is None
                    else str(args.influence_artifact.resolve())
                ),
                "signed_influence_artifact_sha256": (
                    None
                    if artifact is None
                    else file_checksum(args.influence_artifact)
                ),
                "s0_influence_frozen_gaussians": int(frozen_count),
                "shared_mutable_geometry": args.freeze_scope != "all_geometry",
                "geometry_freeze_policy": args.freeze_implementation,
                "geometry_freeze_scope": args.freeze_scope,
                "cross_state_replay": False,
                "fixed_topology": True,
                "densification": False,
                "pruning": False,
            },
        },
        path,
    )
    return path


def resolve_base_ply(checkpoint_path: Path) -> Path:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return Path(checkpoint["base_ply"])


def train(args: argparse.Namespace) -> tuple[Path, Path]:
    started = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the Gaussian renderer")
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_checkpoint = args.output_dir / "state1_complete_checkpoint.pt"
    if output_checkpoint.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output_checkpoint}")

    source_summary = json.loads(args.source_summary.read_text(encoding="utf-8"))
    model = load_temporal_model(args.source_checkpoint)
    if not isinstance(model, TemporalSharedGeometryChangeModel):
        raise TypeError("S0 checkpoint is not a shared-geometry temporal model")
    model.train()
    gaussian_count = int(model.state_change_dc.shape[0])
    if args.freeze_scope == "influence_valid":
        artifact, frozen_cpu = load_influence_mask(
            args.influence_artifact,
            args.source_checkpoint,
            gaussian_count,
        )
    elif args.freeze_scope == "all_geometry":
        artifact = None
        frozen_cpu = torch.ones(gaussian_count, dtype=torch.bool)
    else:
        raise ValueError(f"Unsupported freeze scope: {args.freeze_scope}")
    frozen = frozen_cpu.cuda(non_blocking=True)
    geometry_items = model.shared_geometry_parameter_items()
    if args.freeze_implementation == "external_projection":
        anchors = capture_frozen_rows(geometry_items, frozen)
    elif args.freeze_implementation == "model_buffer_gradient_mask":
        model.freeze_geometry_rows(frozen)
        anchors = None
    else:
        raise ValueError(
            f"Unsupported freeze implementation: {args.freeze_implementation}"
        )
    geometry_before = {
        name: parameter.detach().cpu().clone()
        for name, parameter in geometry_items
    }
    s0_dc_before = model.state_change_dc[:, 0].detach().cpu().clone()
    base_before = base_checksums(model.base)

    records = records_from_summary(source_summary, {0, 1})
    state1_records = [record for record in records if record.segment_id == 1]
    if not state1_records:
        raise RuntimeError("No state 1 records were found")
    fixed_cameras = Path(source_summary["fixed_cameras_json"])
    cue_root = Path(source_summary["cue_cache_root"])
    resolution = float(source_summary["resolution"])
    base_ply = resolve_base_ply(args.source_checkpoint)
    validate_cue_cache(cue_root, base_ply, resolution)
    cameras = load_fixed_camera_index(fixed_cameras)
    views, _, _ = build_fixed_cue_views(records, cameras, cue_root, resolution)
    state1_views = [view for view in views if int(view.segment_id) == 1]
    if len(state1_views) != len(state1_records):
        raise RuntimeError("State 1 record/view count mismatch")

    pipe = SimpleNamespace(
        compute_cov3D_python=False,
        convert_SHs_python=False,
        debug=False,
    )
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    pre_loss = summarize_losses(model, views, background, pipe)
    if args.freeze_scope == "all_geometry":
        optimizer = torch.optim.Adam(
            [
                {
                    "params": [model.state_change_dc],
                    "lr": args.dc_lr,
                    "name": "dc",
                }
            ],
            lr=0.0,
            eps=1e-15,
        )
    else:
        optimizer = make_optimizer(model, args)
    total_steps = len(state1_views) * int(args.updates_per_frame)
    logs: list[dict[str, Any]] = []
    audited_steps = 0
    max_pre_mask_frozen_gradient = {
        name: 0.0 for name, _parameter in geometry_items
    }
    max_post_projection_drift = {
        name: 0.0 for name, _parameter in geometry_items
    }

    global_step = 0
    for epoch in range(int(args.updates_per_frame)):
        for view in state1_views:
            global_step += 1
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
                raise FloatingPointError(f"Non-finite loss at step {global_step}")
            loss.backward()
            if args.freeze_implementation == "external_projection":
                pre_mask = mask_frozen_row_gradients(geometry_items, frozen)
            else:
                pre_mask = model.mask_frozen_geometry_gradients()
            for name, value in pre_mask.items():
                max_pre_mask_frozen_gradient[name] = max(
                    max_pre_mask_frozen_gradient[name], value
                )
            optimizer.step()
            if anchors is not None:
                restore_frozen_rows(geometry_items, frozen, anchors)

            audit = (
                global_step in {1, total_steps}
                or global_step % int(args.gradient_audit_interval) == 0
            )
            if audit:
                audited_steps += 1
                drift = (
                    max_frozen_row_drift(geometry_items, frozen, anchors)
                    if anchors is not None
                    else max_snapshot_frozen_drift(
                        geometry_before, model, frozen_cpu
                    )
                )
                for name, value in drift.items():
                    max_post_projection_drift[name] = max(
                        max_post_projection_drift[name], value
                    )
                logs.append(
                    {
                        "global_step": global_step,
                        "state": 1,
                        "epoch": epoch,
                        "frame": view.image_name,
                        **parts,
                        "frozen_gradient_pre_mask_max_abs": pre_mask,
                        "frozen_anchor_post_projection_max_abs_drift": drift,
                    }
                )
            if (
                global_step in {1, total_steps}
                or global_step % int(args.progress_interval) == 0
            ):
                print(
                    f"[s0-influence-freeze] step {global_step}/{total_steps} "
                    f"frame={view.image_name} loss={parts['loss']:.6f} "
                    f"elapsed={time.time() - started:.1f}s",
                    flush=True,
                )

    final_frozen_drift = (
        max_frozen_row_drift(geometry_items, frozen, anchors)
        if anchors is not None
        else max_snapshot_frozen_drift(geometry_before, model, frozen_cpu)
    )
    if max(final_frozen_drift.values(), default=0.0) != 0.0:
        raise RuntimeError(f"Frozen geometry drifted: {final_frozen_drift}")
    s0_dc_drift = float(
        (model.state_change_dc[:, 0].detach().cpu() - s0_dc_before)
        .abs()
        .max()
        .item()
    )
    if s0_dc_drift != 0.0:
        raise RuntimeError(f"State 0 DC drifted by {s0_dc_drift}")
    post_loss = summarize_losses(model, views, background, pipe)
    movement = movement_summary(geometry_before, model, frozen_cpu)

    checkpoint_path = save_checkpoint(
        model,
        args,
        source_summary,
        int(frozen_cpu.sum().item()),
        artifact,
    )
    final_link = args.output_dir / "temporal_rchange_checkpoint.pt"
    if final_link.exists() or final_link.is_symlink():
        final_link.unlink()
    os.symlink(checkpoint_path.name, final_link)
    source_link = args.output_dir / "state0_complete_checkpoint.pt"
    if source_link.exists() or source_link.is_symlink():
        source_link.unlink()
    os.symlink(os.path.relpath(args.source_checkpoint.resolve(), args.output_dir.resolve()), source_link)

    state1_valid_cpu = model.state_valid[:, 1].detach().cpu()
    summary = dict(source_summary)
    contract = experiment_contract(
        args.freeze_implementation,
        args.freeze_scope,
    )
    summary.update(
        {
            "script": "experiments/train_s0_influence_freeze_s1.py",
            "contract": contract,
            "created_at_unix": time.time(),
            "runtime_seconds": time.time() - started,
            "completed_through_state": 1,
            "shared_mutable_geometry": args.freeze_scope != "all_geometry",
            "source_run": str(args.source_summary.parent.resolve()),
            "source_state0_checkpoint": str(args.source_checkpoint.resolve()),
            "source_state0_checkpoint_sha256": file_checksum(args.source_checkpoint),
            "checkpoint_path": str(final_link),
            "checkpoint_sha256": file_checksum(checkpoint_path),
            "run_arguments": serializable_arguments(args),
            "loss_summary": {"before_s1": pre_loss, "after_s1": post_loss},
            "training_schedule": {
                "mode": "s1_updates_per_frame_exact_continuation",
                "state": 1,
                "frames": len(state1_views),
                "updates_per_frame": int(args.updates_per_frame),
                "actual_total_updates": total_steps,
                "expected_total_updates": len(state1_views)
                * int(args.updates_per_frame),
                "exact": global_step == total_steps,
            },
            "geometry_freeze": {
                "policy": (
                    "all_geometry_immutable"
                    if args.freeze_scope == "all_geometry"
                    else "first_owner_immutable_rows"
                ),
                "scope": args.freeze_scope,
                "implementation": args.freeze_implementation,
                "persistent_model_buffer": (
                    args.freeze_implementation
                    == "model_buffer_gradient_mask"
                ),
                "owner_state": 0,
                "artifact": (
                    None
                    if artifact is None
                    else str(args.influence_artifact.resolve())
                ),
                "artifact_sha256": (
                    None
                    if artifact is None
                    else file_checksum(args.influence_artifact)
                ),
                "artifact_checkpoint_matches": (
                    None if artifact is None else True
                ),
                "influence_contract": (
                    None if artifact is None else artifact["contract"]
                ),
                "influence_thresholds": (
                    None
                    if artifact is None
                    else {
                        "mask_threshold": artifact["mask_threshold"],
                        "mask_temperature": artifact["mask_temperature"],
                        "min_per_view_influence": artifact[
                            "min_per_view_influence"
                        ],
                        "min_mean_influence": artifact[
                            "min_mean_influence"
                        ],
                        "min_views": artifact["min_views"],
                    }
                ),
                "frozen_gaussians": int(frozen_cpu.sum().item()),
                "frozen_fraction": float(frozen_cpu.float().mean().item()),
                "geometry_optimizer_enabled": args.freeze_scope != "all_geometry",
                "s1_valid_frozen_gaussians": int(
                    (state1_valid_cpu & frozen_cpu).sum().item()
                ),
                "s1_valid_unfrozen_gaussians": int(
                    (state1_valid_cpu & ~frozen_cpu).sum().item()
                ),
                "max_pre_mask_frozen_gradient": max_pre_mask_frozen_gradient,
                "max_post_projection_frozen_drift": max_post_projection_drift,
                "final_frozen_drift": final_frozen_drift,
                "s0_dc_max_abs_drift": s0_dc_drift,
            },
            "shared_geometry_transition_movement": movement,
            "gradient_isolation_audit": {
                "audited_steps": audited_steps,
                "audit_interval": int(args.gradient_audit_interval),
                "frozen_geometry_exactly_preserved": max(
                    final_frozen_drift.values(), default=0.0
                )
                == 0.0,
                "s0_dc_exactly_preserved": s0_dc_drift == 0.0,
                "max_pre_mask_frozen_gradient": max_pre_mask_frozen_gradient,
                "max_post_projection_frozen_drift": max_post_projection_drift,
            },
            "train_log": logs,
            "stage_checkpoints": {
                "0": {
                    "path": str(source_link),
                    "sha256": file_checksum(args.source_checkpoint),
                    "completed_through_state": 0,
                },
                "1": {
                    "path": str(checkpoint_path),
                    "sha256": file_checksum(checkpoint_path),
                    "completed_through_state": 1,
                },
            },
            "parameter_checksums": {
                "state_change_dc": tensor_checksum(model.state_change_dc),
                **{
                    f"shared_{name}_delta": tensor_checksum(parameter)
                    for name, parameter in geometry_items
                },
            },
            "base_parameter_checksums_before": base_before,
            "base_parameter_checksums_after": base_checksums(model.base),
            "base_parameters_unchanged": base_before == base_checksums(model.base),
        }
    )
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "summary": str(summary_path),
                "s1_frames": len(state1_views),
                "updates": total_steps,
                "s0_influence_frozen_gaussians": int(frozen_cpu.sum().item()),
                "s1_valid_frozen_gaussians": int(
                    (state1_valid_cpu & frozen_cpu).sum().item()
                ),
                "s1_valid_unfrozen_gaussians": int(
                    (state1_valid_cpu & ~frozen_cpu).sum().item()
                ),
                "final_frozen_drift": final_frozen_drift,
            },
            indent=2,
        )
    )
    return checkpoint_path, summary_path


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def parse_args(
    *,
    default_output: Path = DEFAULT_OUTPUT,
    default_freeze_implementation: str = "external_projection",
    default_freeze_scope: str = "influence_valid",
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-summary",
        type=Path,
        default=SOURCE_RUN / "summary.json",
    )
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        default=SOURCE_RUN / "state0_complete_checkpoint.pt",
    )
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument(
        "--influence-artifact",
        type=Path,
        default=default_output / "signed_influence/state0_signed_influence.pt",
    )
    parser.add_argument(
        "--freeze-implementation",
        choices=("external_projection", "model_buffer_gradient_mask"),
        default=default_freeze_implementation,
    )
    parser.add_argument(
        "--freeze-scope",
        choices=("influence_valid", "all_geometry"),
        default=default_freeze_scope,
    )
    parser.add_argument("--updates-per-frame", type=positive_int, default=120)
    parser.add_argument("--dc-lr", type=nonnegative_float, default=0.0025)
    parser.add_argument("--xyz-lr", type=nonnegative_float, default=0.00016)
    parser.add_argument("--opacity-lr", type=nonnegative_float, default=0.025)
    parser.add_argument("--scaling-lr", type=nonnegative_float, default=0.005)
    parser.add_argument("--rotation-lr", type=nonnegative_float, default=0.001)
    parser.add_argument("--gradient-audit-interval", type=positive_int, default=100)
    parser.add_argument("--progress-interval", type=positive_int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
