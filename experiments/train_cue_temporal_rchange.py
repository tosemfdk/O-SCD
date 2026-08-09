"""Train lifespan-aware R_change from the original O-SCD change cues.

GT masks are not loaded by this runner. They are reserved for the separate
confusion-map evaluation step.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch

from experiments.train_real_temporal_rchange import (
    BASE_PLY_REL,
    DEFAULT_BOUNDARIES,
    DEFAULT_SOURCE,
    FrameRecord,
    PoseResult,
    boundaries_from_manifest,
    build_frame_records,
    build_temporal_model,
    compute_state_support,
    file_checksum,
    load_rgb_tensor,
    read_manifest,
    seed_everything,
    set_trainable_state_only,
    summarize_losses,
    tensor_checksum,
    train_temporal_slots,
    validate_exact_dataset_contract,
)
from scene import GaussianModel
from scene.cameras import Camera


DEFAULT_FIXED_CAMERAS = Path(
    "/home/rvl/workspace/github/O-SCD/output/ESCD_fixedpose_protocols_res4/"
    "scene_change1_2_3/cameras_fixed.json"
)
DEFAULT_CUE_CACHE = Path(
    "/home/rvl/workspace/github/O-SCD/artifacts/escd_396ref/"
    "fixed_pose_cues_res4_v1"
)
DEFAULT_OUTPUT = Path(
    "outputs/instance1_scene_change1_2_3_temporal_rchange_oscd_cues_allframes_120"
)
ORDER_PREFIX = re.compile(r"^order\d+_(.+)$")


def physical_frame_name(name: str) -> str:
    """Remove only the ordering prefix used by reverse combined sequences."""
    stem = Path(name).stem
    match = ORDER_PREFIX.match(stem)
    return match.group(1) if match else stem


def camera_json_to_w2c(camera: Mapping[str, object]) -> np.ndarray:
    """Convert O-SCD's camera-to-world JSON payload to world-to-camera."""
    rotation = np.asarray(camera["rotation"], dtype=np.float64)
    position = np.asarray(camera["position"], dtype=np.float64)
    if rotation.shape != (3, 3) or position.shape != (3,):
        raise ValueError("Invalid fixed camera rotation or position")
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = rotation
    c2w[:3, 3] = position
    return np.linalg.inv(c2w)


def focal2fov(focal: float, pixels: int) -> float:
    return 2.0 * np.arctan(pixels / (2.0 * focal))


def load_fixed_camera_index(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cameras = {str(camera["img_name"]): camera for camera in payload}
    if len(cameras) != len(payload):
        raise ValueError(f"Duplicate fixed camera names in {path}")
    return cameras


def validate_cue_cache(
    cue_cache_root: Path,
    base_ply: Path,
    resolution: float,
) -> dict[str, Any]:
    metadata_path = cue_cache_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if float(metadata.get("resolution", -1)) != float(resolution):
        raise ValueError("Cue-cache resolution does not match this run")
    if metadata.get("reference_ply_sha256") != file_checksum(base_ply):
        raise ValueError("Cue-cache reference PLY does not match this run")
    expected = "O-SCD pixel cue + SAM2.1 Hiera Tiny feature cue"
    if metadata.get("candidate_map_definition") != expected:
        raise ValueError("Cue cache is not the expected O-SCD pixel+feature cue")
    return metadata


def build_fixed_cue_views(
    records: list[FrameRecord],
    cameras: Mapping[str, Mapping[str, object]],
    cue_cache_root: Path,
    resolution: float,
) -> tuple[list[Camera], dict[str, PoseResult], np.ndarray]:
    """Build the exact fixed-pose views used by the O-SCD protocol."""
    views: list[Camera] = []
    poses: dict[str, PoseResult] = {}
    first_intrinsics: tuple[float, float, int, int] | None = None
    for record in records:
        stem = Path(record.name).stem
        camera_json = cameras.get(stem)
        if camera_json is None:
            raise KeyError(f"Missing fixed camera for {stem}")
        image = load_rgb_tensor(Path(record.image_path), resolution)
        height, width = int(image.shape[1]), int(image.shape[2])
        expected_size = (int(camera_json["height"]), int(camera_json["width"]))
        if (height, width) != expected_size:
            raise ValueError(
                f"Fixed camera size mismatch for {stem}: "
                f"{(height, width)} != {expected_size}"
            )
        fx, fy = float(camera_json["fx"]), float(camera_json["fy"])
        intrinsics = (fx, fy, width, height)
        if first_intrinsics is None:
            first_intrinsics = intrinsics
        elif intrinsics != first_intrinsics:
            raise ValueError("Fixed views do not share one camera intrinsic matrix")

        w2c = camera_json_to_w2c(camera_json)
        view = Camera(
            colmap_id=str(record.global_index),
            R=w2c[:3, :3].T.astype(np.float32),
            T=w2c[:3, 3].astype(np.float32),
            FoVx=float(focal2fov(fx, width)),
            FoVy=float(focal2fov(fy, height)),
            image=image,
            gt_alpha_mask=None,
            image_name=stem,
            uid=str(record.global_index),
        )
        cue_path = cue_cache_root / "cues" / f"{physical_frame_name(stem)}.pt"
        cue = torch.load(cue_path, map_location="cpu", weights_only=True)
        if not isinstance(cue, torch.Tensor) or tuple(cue.shape) != (1, height, width):
            raise ValueError(f"Invalid cached cue shape for {stem}: {getattr(cue, 'shape', None)}")
        cue = cue.float().cuda(non_blocking=True)
        view.timestamp = float(record.global_index)
        view.segment_id = int(record.segment_id)
        view.evaluation_split = "train"
        view.candidate_map = cue
        view.training_target = cue
        view.support_map = cue
        view.supervision_source = "oscd_pixel_sam_feature_cue"
        views.append(view)
        poses[record.name] = PoseResult(
            ok=True,
            frame_name=record.name,
            reference_name="O-SCD fixed canonical pose",
            Rt=w2c.tolist(),
            matches=0,
            inliers=0,
            reprojection_rmse=0.0,
        )

    if first_intrinsics is None:
        raise RuntimeError("No fixed views were built")
    fx, fy, width, height = first_intrinsics
    intrinsics = np.array(
        [[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return views, poses, intrinsics


def serializable_arguments(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key != "started_at"
    }


def save_run(
    model,
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
) -> tuple[Path, Path]:
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / "temporal_rchange_checkpoint.pt"
    summary_path = output_dir / "summary.json"
    base_ply = (Path(args.source_path) / BASE_PLY_REL).resolve()
    state_dict = {
        name: value.detach().cpu() for name, value in model.state_dict().items()
    }
    contract = "fixed_topology_temporal_dc_oscd_pixel_sam_cue_manual_boundaries"
    metadata = {
        "schema_version": 2,
        "contract": contract,
        "source_path": str(Path(args.source_path).resolve()),
        "base_ply_sha256": file_checksum(base_ply),
        "fixed_cameras_sha256": file_checksum(Path(args.fixed_cameras_json)),
        "cue_cache_metadata_sha256": file_checksum(Path(args.cue_cache_root) / "metadata.json"),
        "boundaries": list(args.boundaries),
        "camera_intrinsics": intrinsics.tolist(),
        "support": support,
        "schedule": schedule_audit,
        "gradient_isolation_audit": gradient_audit,
        "gt_used_for_training": False,
    }
    torch.save(
        {
            "state_dict": state_dict,
            "base_ply": str(base_ply),
            "boundaries": list(args.boundaries),
            "contract": contract,
            "metadata": metadata,
        },
        checkpoint_path,
    )

    state_valid = model.state_valid.detach()
    summary = {
        "script": "experiments/train_cue_temporal_rchange.py",
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
        "optimized_parameters": ["TemporalChangeModel.state_change_dc"],
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
            "learning_rate": float(args.lr),
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
        "gaussian_count": int(model.state_change_dc.shape[0]),
        "state_change_dc_shape": list(model.state_change_dc.shape),
        "state_valid_counts": state_valid.sum(dim=0).cpu().tolist(),
        "states_per_gaussian_histogram": {
            str(count): int((state_valid.sum(dim=1) == count).sum().item())
            for count in range(model.max_states + 1)
        },
        "state_change_dc_checksum": tensor_checksum(model.state_change_dc),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train temporal R_change with cached O-SCD pixel+feature cues"
    )
    parser.add_argument("--source-path", type=Path, default=Path(DEFAULT_SOURCE))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fixed-cameras-json", type=Path, default=DEFAULT_FIXED_CAMERAS)
    parser.add_argument("--cue-cache-root", type=Path, default=DEFAULT_CUE_CACHE)
    parser.add_argument("--resolution", type=float, default=4.0)
    parser.add_argument("--updates-per-frame", type=positive_int, default=120)
    parser.add_argument("--boundaries", nargs="+", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.0025)
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
    cue_metadata = validate_cue_cache(args.cue_cache_root, base_ply, args.resolution)
    cameras = load_fixed_camera_index(args.fixed_cameras_json)
    views, poses, intrinsics = build_fixed_cue_views(
        records, cameras, args.cue_cache_root, args.resolution
    )
    dataset_audit = validate_exact_dataset_contract(
        manifest, all_names, records, views, args.updates_per_frame
    )

    base = GaussianModel(sh_degree=3, active_sh_degree=0)
    base.load_ply_change(str(base_ply))
    model = build_temporal_model(base, args.boundaries, len(records))
    set_trainable_state_only(model)
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
    train_log, gradient_audit, schedule_audit = train_temporal_slots(
        model, views, background, pipe, args, dataset_audit
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
    )
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "summary": str(summary_path),
                "frames": len(views),
                "updates": schedule_audit["actual_total_updates"],
                "gt_used_for_training": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
