"""Train a persistent O-SCD control with the exact temporal-run schedule."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from arguments import OptimizationParams
from dataloaders.read_write_model import qvec2rotmat, read_images_binary
from experiments.train_cue_temporal_rchange import (
    DEFAULT_CUE_CACHE,
    DEFAULT_FIXED_CAMERAS,
    build_fixed_cue_views,
    load_fixed_camera_index,
    validate_cue_cache,
)
from experiments.train_real_temporal_rchange import (
    BASE_PLY_REL,
    DEFAULT_BOUNDARIES,
    DEFAULT_SOURCE,
    boundaries_from_manifest,
    build_frame_records,
    file_checksum,
    make_training_schedule,
    read_manifest,
    seed_everything,
    training_schedule_audit,
    validate_exact_dataset_contract,
)
from gaussian_renderer import render_change
from scene import GaussianModel
from temporal import compute_ssf_loss


DEFAULT_OUTPUT = Path(
    "outputs/instance1_scene_change1_2_3_oscd_persistent_exact_allframes_120"
)


def optimization_defaults(frame_count: int):
    parser = argparse.ArgumentParser(add_help=False)
    group = OptimizationParams(parser)
    options = group.extract(parser.parse_args([]))
    scaled = copy.copy(options)
    ratio = frame_count / 25.0
    schedule = {}
    for name in (
        "densification_interval",
        "opacity_reset_interval",
        "densify_until_iter",
        "position_lr_max_steps",
    ):
        value = max(1, round(getattr(options, name) * ratio))
        setattr(scaled, name, value)
        schedule[name] = value
    schedule["hard_opacity_reset_interval"] = max(1, round(2000 * ratio))
    return scaled, schedule


def reference_camera_centers(source_path: Path) -> torch.Tensor:
    images = read_images_binary(
        str(source_path / "reference_scene" / "sparse" / "0" / "images.bin")
    )
    centers = []
    for image in images.values():
        rotation = qvec2rotmat(image.qvec)
        centers.append(-rotation.T @ image.tvec)
    return torch.tensor(np.asarray(centers), dtype=torch.float32, device="cuda")


def scene_extent(source_path: Path, views) -> float:
    centers = torch.cat(
        (reference_camera_centers(source_path), torch.stack([v.camera_center for v in views])),
        dim=0,
    )
    center = centers.mean(dim=0)
    return float(torch.linalg.norm(centers - center.unsqueeze(0), dim=-1).max() * 1.1)


def train_control(model, source_path, views, views_by_state, schedule, options, event_schedule, pipe, background, progress_interval):
    extent = scene_extent(source_path, views)
    start = time.time()
    last_loss = 0.0
    for global_step, training_step in enumerate(schedule, start=1):
        view = views_by_state[training_step.state][training_step.view_index]
        model.update_learning_rate(global_step)
        package = render_change(view, model, pipe, background)
        loss, _ = compute_ssf_loss(view.candidate_map, package["render"])
        loss.backward()
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)
        last_loss = float(loss.detach().item())

        with torch.no_grad():
            if global_step < options.densify_until_iter:
                visibility = package["visibility_filter"]
                radii = package["radii"]
                model.max_radii2D[visibility] = torch.maximum(
                    model.max_radii2D[visibility], radii[visibility]
                )
                model.add_densification_stats(package["viewspace_points"], visibility)
                if global_step % options.densification_interval == 0:
                    size_threshold = (
                        20 if global_step > options.opacity_reset_interval else None
                    )
                    model.densify_and_prune(
                        options.densify_grad_threshold,
                        0.4,
                        extent,
                        size_threshold,
                    )
                if global_step % event_schedule["hard_opacity_reset_interval"] == 0:
                    model.reset_opacity()
        if global_step == 1 or global_step % progress_interval == 0 or global_step == len(schedule):
            print(
                f"[control] step {global_step}/{len(schedule)} "
                f"state={training_step.state} frame={view.image_name} "
                f"loss={last_loss:.6f} points={model.get_xyz.shape[0]} "
                f"elapsed={time.time() - start:.1f}s",
                flush=True,
            )
    return time.time() - start, last_loss


def render_masks(model, views, pipe, background, output_dir: Path) -> None:
    pred_dir = output_dir / "pred_binary"
    pred_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for view in views:
            rendered = render_change(view, model, pipe, background)["render"]
            binary = rendered.mean(dim=0).gt(0.5).to(torch.uint8).mul_(255).cpu().numpy()
            if not cv2.imwrite(str(pred_dir / f"{view.image_name}.png"), binary):
                raise OSError(f"Failed to save prediction for {view.image_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exact-exposure persistent O-SCD control")
    parser.add_argument("--source-path", type=Path, default=Path(DEFAULT_SOURCE))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fixed-cameras-json", type=Path, default=DEFAULT_FIXED_CAMERAS)
    parser.add_argument("--cue-cache-root", type=Path, default=DEFAULT_CUE_CACHE)
    parser.add_argument("--resolution", type=float, default=4.0)
    parser.add_argument("--updates-per-frame", type=int, default=120)
    parser.add_argument("--progress-interval", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the Gaussian renderer")
    if args.updates_per_frame <= 0:
        raise ValueError("updates_per_frame must be positive")
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest(args.source_path)
    boundaries = boundaries_from_manifest(manifest, DEFAULT_BOUNDARIES)
    records, _, all_names = build_frame_records(
        args.source_path, 1, [], boundaries, all_training_frames=True
    )
    base_ply = (args.source_path / BASE_PLY_REL).resolve()
    cue_metadata = validate_cue_cache(args.cue_cache_root, base_ply, args.resolution)
    views, _, _ = build_fixed_cue_views(
        records,
        load_fixed_camera_index(args.fixed_cameras_json),
        args.cue_cache_root,
        args.resolution,
    )
    dataset_audit = validate_exact_dataset_contract(
        manifest, all_names, records, views, args.updates_per_frame
    )
    views_by_state = {state: [] for state in range(len(boundaries) + 1)}
    for view in views:
        state_views = views_by_state[int(view.segment_id)]
        state_views.append(view)
    schedule_args = SimpleNamespace(
        updates_per_frame=args.updates_per_frame,
        steps_state=0,
    )
    schedule = make_training_schedule(views_by_state, len(views_by_state), schedule_args)
    schedule_audit = training_schedule_audit(
        views, schedule, schedule_args, dataset_audit
    )

    options, event_schedule = optimization_defaults(len(views))
    model = GaussianModel(sh_degree=3, active_sh_degree=0)
    model.load_ply_change(str(base_ply))
    model.training_setup_change(options)
    initial_points = int(model.get_xyz.shape[0])
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    runtime, final_loss = train_control(
        model,
        args.source_path,
        views,
        views_by_state,
        schedule,
        options,
        event_schedule,
        pipe,
        background,
        args.progress_interval,
    )
    render_masks(model, views, pipe, background, args.output_dir)
    model.save_ply_change(str(args.output_dir / "persistent_control_final.ply"))

    summary = {
        "script": "experiments/train_oscd_exact_control.py",
        "control": "persistent O-SCD change field",
        "supervision": cue_metadata["candidate_map_definition"],
        "gt_used_for_training": False,
        "pose_policy": "same O-SCD fixed canonical poses as lifespan run",
        "schedule": schedule_audit,
        "event_schedule": event_schedule,
        "boundaries_used_for_order_only": list(boundaries),
        "initial_points": initial_points,
        "final_points": int(model.get_xyz.shape[0]),
        "final_loss": final_loss,
        "training_runtime_seconds": runtime,
        "runtime_seconds": time.time() - started,
        "base_ply_sha256": file_checksum(base_ply),
        "fixed_cameras_sha256": file_checksum(args.fixed_cameras_json),
        "cue_cache_metadata_sha256": file_checksum(args.cue_cache_root / "metadata.json"),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
