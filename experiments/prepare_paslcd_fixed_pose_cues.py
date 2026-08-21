"""Prepare fixed-pose O-SCD pixel+SAM cue caches for PASLCD scenes.

The generated cache is consumed by the causal binary-state runner.  It reuses the
original O-SCD pose output (`cameras.json`) and computes cues only from the
immutable reference render and current inference RGB; GT masks are never read.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import cv2
import torch

from experiments.train_cue_temporal_rchange import (
    camera_json_to_w2c,
    focal2fov,
    load_fixed_camera_index,
)
from experiments.train_real_temporal_rchange import BASE_PLY_REL, file_checksum, list_images, load_rgb_tensor
from scene.cameras import Camera

CANDIDATE_MAP_DEFINITION = "O-SCD pixel cue + SAM2.1 Hiera Tiny feature cue"
CUE_VALUE_STORAGE = "raw_generate_candidate_map_float32_not_clamped"
DEFAULT_DATASET_ROOT = Path("/home/rvl/workspace/github/O-SCD/data/PASLCD")
DEFAULT_OSCD_OUTPUT_ROOT = Path("/home/rvl/workspace/github/O-SCD/output")
DEFAULT_OSCD_FALLBACK_OUTPUT_ROOT = Path("/home/rvl/workspace/github/O-SCD/output_ablation")
DEFAULT_CUE_ROOT = Path("outputs/paslcd_fixed_pose_cues_res4_v1")
DEFAULT_SCENES = (
    "Cantina",
    "Garden",
    "Lounge",
    "Lunch_room",
    "Meeting_room",
    "Playground",
    "Porch",
    "Pots",
    "Printing_area",
    "Zen",
)
DEFAULT_INSTANCES = ("Instance_1", "Instance_2")


@dataclass(frozen=True)
class SceneSpec:
    instance: str
    scene: str
    source_path: Path
    cameras_json: Path
    output_dir: Path
    camera_source: str = "primary"


def sha256_file(path: Path) -> str:
    return file_checksum(path)


def parse_csv(value: str | None, default: Sequence[str]) -> tuple[str, ...]:
    if value is None or value == "":
        return tuple(default)
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _camera_json_matches_scene(source_path: Path, cameras_json: Path, *, resolution: float | None) -> bool:
    if not cameras_json.exists():
        return False
    image_dir = source_path / "inference_scene" / "images"
    if resolution is None or not image_dir.exists():
        return True
    names = list_images(image_dir)
    if not names:
        return True
    stem = Path(names[0]).stem
    try:
        camera = load_fixed_camera_index(cameras_json).get(stem)
        if camera is None:
            return False
        image = cv2.imread(str(image_dir / names[0]), cv2.IMREAD_UNCHANGED)
        if image is None:
            return False
        if resolution > 0.0 and resolution != 1.0:
            image = cv2.resize(
                image,
                (0, 0),
                fx=1.0 / resolution,
                fy=1.0 / resolution,
                interpolation=cv2.INTER_AREA,
            )
    except Exception:
        return False
    return (int(image.shape[0]), int(image.shape[1])) == (
        int(camera["height"]),
        int(camera["width"]),
    )


def _fallback_cameras_json(root: Path | None, instance: str, scene: str) -> Path | None:
    if root is None:
        return None
    return root / instance / scene / "seed_0" / "reference" / "cameras.json"


def select_cameras_json(
    source_path: Path,
    primary: Path,
    *,
    fallback_root: Path | None,
    instance: str,
    scene: str,
    resolution: float | None,
) -> Path:
    if _camera_json_matches_scene(source_path, primary, resolution=resolution):
        return primary
    fallback = _fallback_cameras_json(fallback_root, instance, scene)
    if fallback is not None and _camera_json_matches_scene(source_path, fallback, resolution=resolution):
        return fallback
    if primary.exists():
        raise ValueError(
            f"fixed camera/image mismatch for {instance}/{scene}: {primary}; "
            f"no compatible fallback under {fallback_root}"
        )
    raise FileNotFoundError(primary)


def discover_scenes(
    dataset_root: Path,
    oscd_output_root: Path,
    cue_root: Path,
    *,
    instances: Sequence[str] = DEFAULT_INSTANCES,
    scenes: Sequence[str] = DEFAULT_SCENES,
    oscd_fallback_output_root: Path | None = DEFAULT_OSCD_FALLBACK_OUTPUT_ROOT,
    resolution: float | None = None,
) -> list[SceneSpec]:
    specs: list[SceneSpec] = []
    for instance in instances:
        for scene in scenes:
            source_path = dataset_root / instance / scene
            primary_cameras_json = oscd_output_root / instance / scene / "cameras.json"
            if not source_path.exists():
                raise FileNotFoundError(source_path)
            cameras_json = select_cameras_json(
                source_path,
                primary_cameras_json,
                fallback_root=oscd_fallback_output_root,
                instance=instance,
                scene=scene,
                resolution=resolution,
            )
            specs.append(
                SceneSpec(
                    instance=instance,
                    scene=scene,
                    source_path=source_path,
                    cameras_json=cameras_json,
                    output_dir=cue_root / instance / scene,
                    camera_source=(
                        "primary"
                        if cameras_json == primary_cameras_json
                        else "fallback"
                    ),
                )
            )
    return specs


def build_view(image_path: Path, camera_json: dict[str, Any], *, resolution: float, uid: int) -> Camera:
    image = load_rgb_tensor(image_path, resolution)
    height, width = int(image.shape[1]), int(image.shape[2])
    expected_size = (int(camera_json["height"]), int(camera_json["width"]))
    if (height, width) != expected_size:
        raise ValueError(f"camera/image size mismatch for {image_path.name}: {(height, width)} != {expected_size}")
    fx, fy = float(camera_json["fx"]), float(camera_json["fy"])
    w2c = camera_json_to_w2c(camera_json)
    return Camera(
        colmap_id=str(uid),
        R=w2c[:3, :3].T.astype("float32"),
        T=w2c[:3, 3].astype("float32"),
        FoVx=float(focal2fov(fx, width)),
        FoVy=float(focal2fov(fy, height)),
        image=image,
        gt_alpha_mask=None,
        image_name=Path(image_path).stem,
        uid=str(uid),
    )


def write_metadata(spec: SceneSpec, *, resolution: float, frame_count: int, elapsed: float) -> dict[str, Any]:
    base_ply = spec.source_path / BASE_PLY_REL
    metadata = {
        "schema_version": 1,
        "dataset": "PASLCD",
        "instance": spec.instance,
        "scene": spec.scene,
        "source_path": str(spec.source_path),
        "fixed_cameras_json": str(spec.cameras_json),
        "fixed_cameras_sha256": sha256_file(spec.cameras_json),
        "fixed_camera_source": spec.camera_source,
        "resolution": float(resolution),
        "reference_ply": str(base_ply),
        "reference_ply_sha256": sha256_file(base_ply),
        "candidate_map_definition": CANDIDATE_MAP_DEFINITION,
        "candidate_map_storage_dtype": "float32",
        "cue_value_storage": CUE_VALUE_STORAGE,
        "gt_used": False,
        "frame_count": int(frame_count),
        "elapsed_seconds": float(elapsed),
    }
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    (spec.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def _metadata_matches(metadata_path: Path, *, base_ply: Path, cameras_json: Path, resolution: float, frame_count: int) -> bool:
    if not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return (
        metadata.get("candidate_map_definition") == CANDIDATE_MAP_DEFINITION
        and metadata.get("cue_value_storage") == CUE_VALUE_STORAGE
        and float(metadata.get("resolution", -1.0)) == float(resolution)
        and metadata.get("reference_ply_sha256") == sha256_file(base_ply)
        and metadata.get("fixed_cameras_sha256") == sha256_file(cameras_json)
        and int(metadata.get("frame_count", -1)) == int(frame_count)
    )


def _scene_cache_valid(
    spec: SceneSpec,
    *,
    resolution: float,
    max_frames: int | None,
) -> tuple[list[str], bool]:
    image_dir = spec.source_path / "inference_scene" / "images"
    names = list_images(image_dir)
    if max_frames is not None:
        names = names[: int(max_frames)]
    cues_dir = spec.output_dir / "cues"
    expected_files = [cues_dir / f"{Path(name).stem}.pt" for name in names]
    valid = all(path.exists() for path in expected_files) and _metadata_matches(
        spec.output_dir / "metadata.json",
        base_ply=spec.source_path / BASE_PLY_REL,
        cameras_json=spec.cameras_json,
        resolution=resolution,
        frame_count=len(names),
    )
    return names, bool(valid)


def prepare_scene(
    spec: SceneSpec,
    *,
    resolution: float,
    force: bool = False,
    max_frames: int | None = None,
    sam_model: Any | None = None,
) -> dict[str, Any]:
    image_dir = spec.source_path / "inference_scene" / "images"
    names, cache_valid = _scene_cache_valid(
        spec,
        resolution=resolution,
        max_frames=max_frames,
    )
    cameras = load_fixed_camera_index(spec.cameras_json)
    missing = [Path(name).stem for name in names if Path(name).stem not in cameras]
    if missing:
        raise KeyError(f"missing fixed cameras for {spec.instance}/{spec.scene}: {missing[:5]}")
    base_ply = spec.source_path / BASE_PLY_REL
    cues_dir = spec.output_dir / "cues"
    cues_dir.mkdir(parents=True, exist_ok=True)
    if cache_valid and not force:
        return {
            "instance": spec.instance,
            "scene": spec.scene,
            "frame_count": len(names),
            "cache_hit": True,
            "output_dir": str(spec.output_dir),
        }

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to render PASLCD fixed-pose cues")
    from gaussian_renderer import render
    from oscd import generate_candidate_map
    from scene import GaussianModel
    from transformers import Sam2Model

    model = sam_model
    if model is None:
        model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-tiny").half().cuda().eval()
    gaussians_rgb = GaussianModel(3, 3)
    gaussians_rgb.load_ply(str(base_ply))
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    started = time.time()
    written = 0
    for uid, name in enumerate(names):
        stem = Path(name).stem
        cue_path = cues_dir / f"{stem}.pt"
        if cue_path.exists() and cache_valid and not force:
            written += 1
            continue
        view = build_view(image_dir / name, cameras[stem], resolution=resolution, uid=uid)
        with torch.no_grad():
            reference_render = render(view, gaussians_rgb, pipe, background)["render"].detach().clamp(0, 1)
            cue = generate_candidate_map(
                view.original_image[:3].cuda(non_blocking=True),
                reference_render,
                model,
                14,
                int(view.image_height),
                int(view.image_width),
            ).detach().float().cpu()
        if tuple(cue.shape) != (1, int(view.image_height), int(view.image_width)):
            raise ValueError(f"invalid cue shape for {stem}: {tuple(cue.shape)}")
        torch.save(cue, cue_path)
        written += 1
    metadata = write_metadata(spec, resolution=resolution, frame_count=len(names), elapsed=time.time() - started)
    return {
        "instance": spec.instance,
        "scene": spec.scene,
        "frame_count": len(names),
        "cache_hit": False,
        "written": written,
        "output_dir": str(spec.output_dir),
        "metadata": metadata,
    }


def prepare_all(
    specs: Iterable[SceneSpec],
    *,
    resolution: float,
    force: bool = False,
    max_frames: int | None = None,
) -> list[dict[str, Any]]:
    scene_specs = list(specs)
    needs_model = force or any(
        not _scene_cache_valid(
            spec,
            resolution=resolution,
            max_frames=max_frames,
        )[1]
        for spec in scene_specs
    )
    sam_model = None
    if needs_model:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to render PASLCD fixed-pose cues")
        from transformers import Sam2Model

        sam_model = (
            Sam2Model.from_pretrained("facebook/sam2.1-hiera-tiny")
            .half()
            .cuda()
            .eval()
        )
    return [
        prepare_scene(
            spec,
            resolution=resolution,
            force=force,
            max_frames=max_frames,
            sam_model=sam_model,
        )
        for spec in scene_specs
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--oscd-output-root", type=Path, default=DEFAULT_OSCD_OUTPUT_ROOT)
    parser.add_argument("--oscd-fallback-output-root", type=Path, default=DEFAULT_OSCD_FALLBACK_OUTPUT_ROOT)
    parser.add_argument("--cue-root", type=Path, default=DEFAULT_CUE_ROOT)
    parser.add_argument("--instances", type=str, default=None, help="comma-separated instance names")
    parser.add_argument("--scenes", type=str, default=None, help="comma-separated scene names")
    parser.add_argument("--resolution", type=float, default=4.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


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
    results = prepare_all(specs, resolution=float(args.resolution), force=bool(args.force), max_frames=args.max_frames)
    print(json.dumps({"scene_count": len(results), "results": results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
