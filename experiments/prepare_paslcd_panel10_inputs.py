#!/usr/bin/env python3
"""Prepare causal Panel10 input artifacts for PASLCD scene-level evaluation.

This script is input-preparation only.  It reconstructs the raw powered
pixel×SAM cue from fixed-pose PASLCD cue caches, learns a strict prequential
histogram boundary from the original O-SCD online-at-arrival teacher masks,
exports the causal SAM-delta PC1 trace, and materializes DA3Metric-Large
current-frame depth caches.  It never reads PASLCD GT masks and never changes a
representation/model checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.build_causal_da3_seed_replay import _cached_or_infer_metric_depth
from experiments.prepare_paslcd_fixed_pose_cues import (
    DEFAULT_CUE_ROOT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_INSTANCES,
    DEFAULT_OSCD_FALLBACK_OUTPUT_ROOT,
    DEFAULT_OSCD_OUTPUT_ROOT,
    DEFAULT_SCENES,
    SceneSpec,
    discover_scenes,
    parse_csv,
    sha256_file,
)
from experiments.run_online_bayesian_lifespan_thaw import BASE_PLY_REL, build_causal_records
from experiments.run_online_xfeat_new_seed import extract_delta, fixed_camera_matrices
from experiments.train_cue_temporal_rchange import (
    build_fixed_cue_views,
    load_fixed_camera_index,
    validate_cue_cache,
)
from experiments.train_real_temporal_rchange import file_checksum, seed_everything
from experiments.train_stage2_cue_boundary import _binned_sum
from experiments.train_stage2_cue_boundary_causal import train_causal_prequential
from temporal.change_cue_fusion import (
    fuse_power_product,
    normalized_oscd_pixel_cue_from_terms,
    oscd_pixel_terms,
    semantic_from_cached_sum,
)
from temporal.learnable_cue_boundary import HistogramBoundaryMLP
from temporal.sign_mapping import CausalPC1

SCHEMA_VERSION = 1
DEFAULT_OUTPUT_ROOT = Path("outputs/paslcd_panel10_inputs_20260907")
DEFAULT_RESOLUTION = 4.0
DEFAULT_INPUT_BINS = 64
DEFAULT_LOSS_BINS = 512
DEFAULT_L1_EXPONENT = 0.3
DEFAULT_CUE_PRODUCT_EXPONENT = 1.0
DEFAULT_PROCESS_RES = 504
DEFAULT_METRIC_MODEL = "depth-anything/DA3METRIC-LARGE"
DEFAULT_SAM_MODEL = "facebook/sam2.1-hiera-tiny"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _teacher_mask_path(teacher_dir: Path, record: Any) -> Path:
    """Resolve PASLCD O-SCD online-at-arrival mask as stem + .png."""

    # oscd.py writes rendered change masks as PNG even when PASLCD source
    # frames are JPG, e.g. Inst_1_test_IMG_2863.jpg -> .../Inst_1_test_IMG_2863.png.
    return teacher_dir / f"{Path(record.name).stem}.png"


def _load_binary_mask(path: Path, *, height: int, width: int) -> torch.Tensor:
    raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(path)
    if raw.shape[:2] != (height, width):
        raw = cv2.resize(raw, (width, height), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy((raw > 127).astype(np.float32)).unsqueeze(0)


def _read_rgb(path: Path, *, height: int, width: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image


def reconstruct_panel10_raw_cue(
    *,
    reference_rgb: torch.Tensor,
    online_rgb: torch.Tensor,
    candidate_map: torch.Tensor,
    l1_exponent: float = DEFAULT_L1_EXPONENT,
    product_exponent: float = DEFAULT_CUE_PRODUCT_EXPONENT,
) -> torch.Tensor:
    """Reconstruct raw q in [0,1] from cached O-SCD P+S without GT access."""

    terms = oscd_pixel_terms(reference_rgb, online_rgb)
    original_pixel = normalized_oscd_pixel_cue_from_terms(terms, l1_exponent=1.0)
    powered_pixel = normalized_oscd_pixel_cue_from_terms(terms, l1_exponent=l1_exponent)
    semantic = semantic_from_cached_sum(candidate_map, original_pixel)
    return fuse_power_product(powered_pixel, semantic, exponent=product_exponent) / 2.0


def stable_mask_from_candidate_sum(candidate_map: torch.Tensor, *, threshold: float = 0.2) -> torch.Tensor:
    """Return 64×64 stable SAM-token mask from raw P+S candidate-map area."""

    if candidate_map.ndim == 2:
        candidate_map = candidate_map.unsqueeze(0)
    if candidate_map.ndim != 3 or int(candidate_map.shape[0]) != 1:
        raise ValueError("candidate_map must have shape [1,H,W] or [H,W]")
    cue64 = F.interpolate(candidate_map[None].float(), (64, 64), mode="area")[0, 0]
    return cue64 < float(threshold)


def _expected_contract(
    *,
    spec: SceneSpec,
    output_dir: Path,
    teacher_dir: Path,
    resolution: float,
    input_bins: int,
    loss_bins: int,
    frames: int,
    process_res: int,
    metric_model: str,
) -> dict[str, Any]:
    base_ply = spec.source_path / BASE_PLY_REL
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": "PASLCD",
        "instance": spec.instance,
        "scene": spec.scene,
        "source_path": str(spec.source_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "cue_cache_root": str(spec.output_dir.resolve()),
        "fixed_cameras_json": str(spec.cameras_json.resolve()),
        "fixed_cameras_sha256": sha256_file(spec.cameras_json),
        "reference_ply": str(base_ply.resolve()),
        "reference_ply_sha256": file_checksum(base_ply),
        "teacher_dir": str(teacher_dir.resolve()),
        "teacher_mode": "original_oscd_online_at_arrival_change_mask",
        "gt_used": False,
        "frames": int(frames),
        "resolution": float(resolution),
        "input_bins": int(input_bins),
        "loss_bins": int(loss_bins),
        "l1_exponent": float(DEFAULT_L1_EXPONENT),
        "cue_product_exponent": float(DEFAULT_CUE_PRODUCT_EXPONENT),
        "cue_formula": "q = 0.5 * 2 * norm(0.8*L1^0.3 + 0.2*(1-SSIM)) * SAM",
        "pca": {
            "extract_delta": "experiments.run_online_xfeat_new_seed.extract_delta",
            "channels": 256,
            "stable_bank": 65536,
            "seed": 0,
            "stable_mask": "area(raw P+S candidate_map,64x64) < 0.2",
            "eps_sigma": 2.5,
            "sign_orientation": "CausalPC1 canonical first-axis orientation; no GT/depth orientation",
        },
        "da3_metric": {
            "depth_scale_source": "da3metric_reference_render",
            "metric_cache_root": str((output_dir / "da3metric_cache").resolve()),
            "metric_model": str(metric_model),
            "process_res": int(process_res),
        },
    }


def _complete_cache_hit(output_dir: Path, expected: dict[str, Any], frame_names: Sequence[str]) -> bool:
    metadata_path = output_dir / "input_preparation_metadata.json"
    required = [
        output_dir / "boundary_statistics.npz",
        output_dir / "boundary_statistics.json",
        output_dir / "learned_boundaries_causal.json",
        output_dir / "training_history.json",
        output_dir / "causal_pca_posterior_arrays.npz",
        output_dir / "da3_seed_replay.pt",
        output_dir / "summary.json",
    ]
    required.extend(output_dir / "da3metric_cache" / f"frame_{idx:06d}.npz" for idx, _ in enumerate(frame_names))
    if not metadata_path.exists() or not all(path.exists() for path in required):
        return False
    try:
        observed = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return observed == expected


def _validate_teachers(teacher_dir: Path, records: Sequence[Any]) -> list[str]:
    missing = [str(_teacher_mask_path(teacher_dir, record)) for record in records if not _teacher_mask_path(teacher_dir, record).is_file()]
    if missing:
        raise FileNotFoundError(f"missing O-SCD teacher masks: {missing[:5]}")
    return [file_checksum(_teacher_mask_path(teacher_dir, record)) for record in records]


def _initialize_reference(base_ply: Path) -> Any:
    from scene import GaussianModel

    reference = GaussianModel(sh_degree=3, active_sh_degree=3)
    reference.load_ply(str(base_ply))
    for name in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"):
        getattr(reference, name).requires_grad_(False)
    return reference


def _load_sam_model(model: Any | None, model_name: str) -> Any:
    if model is not None:
        return model
    from transformers import Sam2Model

    return Sam2Model.from_pretrained(model_name).half().cuda().eval()


def _load_da3_metric_model(model: Any | None, model_name: str) -> Any:
    if model is not None:
        return model
    from depth_anything_3.api import DepthAnything3

    return DepthAnything3.from_pretrained(model_name).to("cuda").eval()


def _collect_statistics_and_sidecars(
    *,
    spec: SceneSpec,
    output_dir: Path,
    teacher_dir: Path,
    records: Sequence[Any],
    resolution: float,
    input_bins: int,
    loss_bins: int,
    process_res: int,
    sam_model: Any,
    da3_model: Any,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    from gaussian_renderer import render

    base_ply = (spec.source_path / BASE_PLY_REL).resolve()
    validate_cue_cache(spec.output_dir, base_ply, resolution)
    cameras = load_fixed_camera_index(spec.cameras_json)
    views, _poses, _intrinsics = build_fixed_cue_views(list(records), cameras, spec.output_dir, resolution)
    reference = _initialize_reference(base_ply)
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    pca = CausalPC1(channels=256, stable_bank_size=65536, seed=0, device="cuda")

    input_counts: list[np.ndarray] = []
    loss_counts: list[np.ndarray] = []
    teacher_sums: list[np.ndarray] = []
    frame_names: list[str] = []
    segment_names: list[str] = []
    pc1_axes: list[np.ndarray] = []
    pca_rows: list[dict[str, Any]] = []
    eps_neg: list[float] = []
    eps_pos: list[float] = []
    teacher_sha256: list[str] = []
    da3_cache_rows: list[dict[str, Any]] = []

    cache_root = output_dir / "da3metric_cache"
    started = time.time()
    for index, (record, view) in enumerate(zip(records, views)):
        height, width = int(view.image_height), int(view.image_width)
        teacher_path = _teacher_mask_path(teacher_dir, record)
        with torch.no_grad():
            reference_rgb = render(view, reference, pipe, background)["render"].detach()
            cue = reconstruct_panel10_raw_cue(
                reference_rgb=reference_rgb,
                online_rgb=view.original_image[:3].to(device=reference_rgb.device, dtype=reference_rgb.dtype),
                candidate_map=view.candidate_map.to(device=reference_rgb.device, dtype=reference_rgb.dtype),
            )
            teacher = _load_binary_mask(teacher_path, height=height, width=width).to(device=cue.device, dtype=cue.dtype)
            input_count, _ = _binned_sum(cue, torch.ones_like(cue), bins=input_bins)
            loss_count, teacher_sum = _binned_sum(cue, teacher, bins=loss_bins)
            stable_mask = stable_mask_from_candidate_sum(view.candidate_map)
            delta, _reference_again = extract_delta(view, reference, pipe, background, sam_model)
            update = pca.update(delta, stable_mask, eps_sigma=2.5)

        w2c, image_K = fixed_camera_matrices(record, cameras)
        image = _read_rgb(Path(record.image_path), height=height, width=width)
        depth, depth_K = _cached_or_infer_metric_depth(
            da3_metric=da3_model,
            cache_path=cache_root / f"frame_{index:06d}.npz",
            image=image,
            image_K=image_K.numpy(),
            process_res=process_res,
        )
        del w2c  # fixed pose is validated by fixed_camera_matrices; only K is needed by DA3Metric.

        input_counts.append(input_count.cpu().numpy().astype(np.int64, copy=False))
        loss_counts.append(loss_count.cpu().numpy().astype(np.int64, copy=False))
        teacher_sums.append(teacher_sum.cpu().numpy().astype(np.float32, copy=False))
        frame_names.append(str(record.name))
        segment_names.append(str(record.segment_name))
        pc1_axes.append(update.pc.detach().cpu().numpy().astype(np.float32, copy=False))
        eps_neg.append(float(update.epsilon_negative))
        eps_pos.append(float(update.epsilon_positive))
        pca_rows.append({"frame_index": index, "frame_name": record.name, **update.as_dict(include_tensors=False)})
        teacher_sha256.append(file_checksum(teacher_path))
        da3_cache_rows.append(
            {
                "frame_index": index,
                "frame_name": record.name,
                "cache_path": str((cache_root / f"frame_{index:06d}.npz").resolve()),
                "depth_shape": [int(depth.shape[0]), int(depth.shape[1])],
                "K_shape": [int(depth_K.shape[0]), int(depth_K.shape[1])],
            }
        )
        if index == 0 or (index + 1) % 25 == 0 or index + 1 == len(records):
            print(f"[prep {spec.instance}/{spec.scene} {index + 1:03d}/{len(records):03d}] elapsed={time.time() - started:.1f}s", flush=True)

    arrays = {
        "input_counts": np.stack(input_counts),
        "loss_counts": np.stack(loss_counts),
        "teacher_sums": np.stack(teacher_sums),
        "frame_names": np.asarray(frame_names),
        "segment_names": np.asarray(segment_names),
        "pc1_axes": np.stack(pc1_axes),
        "epsilon_negative": np.asarray(eps_neg, dtype=np.float32),
        "epsilon_positive": np.asarray(eps_pos, dtype=np.float32),
        "teacher_sha256": np.asarray(teacher_sha256),
    }
    return arrays, [{"pca_history": pca_rows, "da3_metric_cache": da3_cache_rows}]


def _train_and_write_boundaries(
    *,
    output_dir: Path,
    arrays: dict[str, np.ndarray],
    updates_per_arrival: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_counts_np = arrays["input_counts"]
    loss_counts_np = arrays["loss_counts"]
    teacher_sums_np = arrays["teacher_sums"]
    frame_names = tuple(str(value) for value in arrays["frame_names"])
    segment_names = tuple(str(value) for value in arrays["segment_names"])
    model = HistogramBoundaryMLP(
        bins=int(input_counts_np.shape[1]),
        initial_tau=0.25,
        initial_width=0.10,
        edge_probability=0.05,
    ).to(device)
    result = train_causal_prequential(
        model=model,
        input_counts=torch.from_numpy(input_counts_np).to(device=device, dtype=torch.float32),
        loss_counts=torch.from_numpy(loss_counts_np).to(device=device, dtype=torch.float32),
        teacher_sums=torch.from_numpy(teacher_sums_np).to(device=device, dtype=torch.float32),
        segment_names=segment_names,
        updates_per_arrival=int(updates_per_arrival),
    )
    artifact = {
        "schema_version": 2,
        "remap": "sigmoid",
        "training_mode": "causal_prequential",
        "prediction_order": "theta_(t-1)+histogram_t -> tau_t,width_t; teacher_t updates theta_t afterwards",
        "width_semantics": "output is 0.05/0.95 at tau-width/tau+width",
        "edge_probability": 0.05,
        "cue_scale": 1.0,
        "cue_formula": "q = 0.5 * 2 * norm(0.8*L1^0.3 + 0.2*(1-SSIM)) * SAM",
        "frames": {name: {"tau": float(tau), "width": float(width)} for name, tau, width in zip(frame_names, result.tau, result.width)},
    }
    (output_dir / "learned_boundaries_causal.json").write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    (output_dir / "training_history.json").write_text(json.dumps(result.history, indent=2) + "\n", encoding="utf-8")
    audit = {
        "model": model.configuration(),
        "causal_audit": result.audit,
        "tau": {"mean": float(result.tau.mean()), "min": float(result.tau.min()), "max": float(result.tau.max())},
        "width": {"mean": float(result.width.mean()), "min": float(result.width.min()), "max": float(result.width.max())},
    }
    return audit, result.history



def scene_preparation_complete(
    spec: SceneSpec,
    output_dir: Path,
    teacher_dir: Path,
    *,
    resolution: float = DEFAULT_RESOLUTION,
    max_frames: int | None = None,
    input_bins: int = DEFAULT_INPUT_BINS,
    loss_bins: int = DEFAULT_LOSS_BINS,
    process_res: int = DEFAULT_PROCESS_RES,
    metric_model: str = DEFAULT_METRIC_MODEL,
) -> bool:
    """Return True only for a complete, metadata-matched prepared scene."""

    records, frame_names = build_causal_records(spec.source_path, max_frames=max_frames)
    if not records:
        return False
    try:
        _validate_teachers(teacher_dir, records)
        expected = _expected_contract(
            spec=spec,
            output_dir=output_dir,
            teacher_dir=teacher_dir,
            resolution=resolution,
            input_bins=input_bins,
            loss_bins=loss_bins,
            frames=len(records),
            process_res=process_res,
            metric_model=metric_model,
        )
    except Exception:
        return False
    return _complete_cache_hit(output_dir, expected, frame_names)

def prepare_scene(
    spec: SceneSpec,
    output_dir: Path,
    teacher_dir: Path,
    *,
    sam_model: Any | None = None,
    da3_model: Any | None = None,
    resolution: float = DEFAULT_RESOLUTION,
    max_frames: int | None = None,
    force: bool = False,
    input_bins: int = DEFAULT_INPUT_BINS,
    loss_bins: int = DEFAULT_LOSS_BINS,
    updates_per_arrival: int = 1,
    seed: int = 0,
    sam_model_name: str = DEFAULT_SAM_MODEL,
    metric_model: str = DEFAULT_METRIC_MODEL,
    process_res: int = DEFAULT_PROCESS_RES,
) -> dict[str, Any]:
    """Prepare one PASLCD scene reset without GT reads or representation changes."""

    output_dir = Path(output_dir)
    teacher_dir = Path(teacher_dir)
    records, frame_names = build_causal_records(spec.source_path, max_frames=max_frames)
    if not records:
        raise RuntimeError(f"no PASLCD frames found for {spec.instance}/{spec.scene}")
    _validate_teachers(teacher_dir, records)
    expected = _expected_contract(
        spec=spec,
        output_dir=output_dir,
        teacher_dir=teacher_dir,
        resolution=resolution,
        input_bins=input_bins,
        loss_bins=loss_bins,
        frames=len(records),
        process_res=process_res,
        metric_model=metric_model,
    )
    if _complete_cache_hit(output_dir, expected, frame_names) and not force:
        summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
        summary["cache_hit"] = True
        return summary
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for PASLCD Panel10 input preparation")

    output_dir.mkdir(parents=True, exist_ok=True)
    sam = _load_sam_model(sam_model, sam_model_name)
    da3_metric = _load_da3_metric_model(da3_model, metric_model)
    started = time.time()
    arrays, sidecar_rows = _collect_statistics_and_sidecars(
        spec=spec,
        output_dir=output_dir,
        teacher_dir=teacher_dir,
        records=records,
        resolution=resolution,
        input_bins=input_bins,
        loss_bins=loss_bins,
        process_res=process_res,
        sam_model=sam,
        da3_model=da3_metric,
    )

    np.savez_compressed(
        output_dir / "boundary_statistics.npz",
        input_counts=arrays["input_counts"],
        loss_counts=arrays["loss_counts"],
        teacher_sums=arrays["teacher_sums"],
        frame_names=arrays["frame_names"],
        segment_names=arrays["segment_names"],
    )
    stats_metadata = {
        **expected,
        "teacher_frame_sha256": arrays["teacher_sha256"].tolist(),
        "statistics_contract": "histograms only; teacher masks are original O-SCD online-at-arrival; no GT arrays stored",
    }
    (output_dir / "boundary_statistics.json").write_text(json.dumps(stats_metadata, indent=2, default=_json_default) + "\n", encoding="utf-8")

    boundary_audit, _history = _train_and_write_boundaries(
        output_dir=output_dir,
        arrays=arrays,
        updates_per_arrival=updates_per_arrival,
        seed=seed,
    )
    np.savez_compressed(
        output_dir / "causal_pca_posterior_arrays.npz",
        frame_names=arrays["frame_names"],
        pc1_axes=arrays["pc1_axes"],
        epsilon_negative=arrays["epsilon_negative"],
        epsilon_positive=arrays["epsilon_positive"],
    )
    history = sidecar_rows[0]
    (output_dir / "causal_pca_history.json").write_text(json.dumps(history["pca_history"], indent=2) + "\n", encoding="utf-8")
    replay = {
        "schema_version": SCHEMA_VERSION,
        "configuration": {
            "depth_scale_source": "da3metric_reference_render",
            "metric_cache_root": str((output_dir / "da3metric_cache").resolve()),
            "metric_model": metric_model,
            "process_res": int(process_res),
        },
        "frame_names": [str(name) for name in arrays["frame_names"]],
        "cache": history["da3_metric_cache"],
        "seed_batches": [],
        "note": "typed viewer reads configuration/cache only; no seed sidecar is generated here",
    }
    torch.save(replay, output_dir / "da3_seed_replay.pt")
    elapsed = time.time() - started
    summary = {
        "schema_version": SCHEMA_VERSION,
        "instance": spec.instance,
        "scene": spec.scene,
        "frames": len(records),
        "output_dir": str(output_dir),
        "cache_hit": False,
        "gt_used": False,
        "runtime_seconds": elapsed,
        "artifacts": {
            "metadata": str(output_dir / "input_preparation_metadata.json"),
            "boundary_statistics_npz": str(output_dir / "boundary_statistics.npz"),
            "boundary_statistics_json": str(output_dir / "boundary_statistics.json"),
            "learned_boundaries_causal": str(output_dir / "learned_boundaries_causal.json"),
            "training_history": str(output_dir / "training_history.json"),
            "causal_pca_arrays": str(output_dir / "causal_pca_posterior_arrays.npz"),
            "causal_pca_history": str(output_dir / "causal_pca_history.json"),
            "da3_seed_replay": str(output_dir / "da3_seed_replay.pt"),
            "da3metric_cache": str(output_dir / "da3metric_cache"),
        },
        "audit": {
            "input_references": expected,
            "teacher_frame_count": len(arrays["teacher_sha256"]),
            "boundary": boundary_audit,
            "pca_future_view_accesses": 0,
            "da3_future_view_accesses": 0,
            "representation_changes": False,
        },
    }
    (output_dir / "input_preparation_metadata.json").write_text(json.dumps(expected, indent=2, default=_json_default) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return summary


def _teacher_dir_for(oscd_output_root: Path, instance: str, scene: str) -> Path:
    return oscd_output_root / instance / scene / "renders" / "change_mask"


def prepare_all(
    specs: Iterable[SceneSpec],
    *,
    output_root: Path,
    oscd_output_root: Path,
    resolution: float,
    max_frames: int | None,
    force: bool,
    input_bins: int,
    loss_bins: int,
    updates_per_arrival: int,
    seed: int,
    process_res: int,
    metric_model: str,
    sam_model_name: str,
) -> list[dict[str, Any]]:
    specs = list(specs)
    sam_model = None
    da3_model = None
    results = []
    for spec in specs:
        teacher_dir = _teacher_dir_for(oscd_output_root, spec.instance, spec.scene)
        if not teacher_dir.is_dir():
            raise FileNotFoundError(f"teacher directory not found: {teacher_dir}")
        scene_output = output_root / spec.instance / spec.scene
        complete = (not force) and scene_preparation_complete(
            spec,
            scene_output,
            teacher_dir,
            resolution=resolution,
            max_frames=max_frames,
            input_bins=input_bins,
            loss_bins=loss_bins,
            process_res=process_res,
            metric_model=metric_model,
        )
        if (not complete) and torch.cuda.is_available() and (sam_model is None or da3_model is None):
            if sam_model is None:
                sam_model = _load_sam_model(None, sam_model_name)
            if da3_model is None:
                da3_model = _load_da3_metric_model(None, metric_model)
        results.append(
            prepare_scene(
                spec,
                scene_output,
                teacher_dir,
                sam_model=sam_model,
                da3_model=da3_model,
                resolution=resolution,
                max_frames=max_frames,
                force=force,
                input_bins=input_bins,
                loss_bins=loss_bins,
                updates_per_arrival=updates_per_arrival,
                seed=seed,
                sam_model_name=sam_model_name,
                metric_model=metric_model,
                process_res=process_res,
            )
        )
    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--oscd-output-root", type=Path, default=DEFAULT_OSCD_OUTPUT_ROOT)
    parser.add_argument("--oscd-fallback-output-root", type=Path, default=DEFAULT_OSCD_FALLBACK_OUTPUT_ROOT)
    parser.add_argument("--cue-root", type=Path, default=DEFAULT_CUE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--instances", type=str, default=None, help="comma-separated instances; default is Instance_1,Instance_2")
    parser.add_argument("--scenes", type=str, default=None, help="comma-separated scenes; default is PASLCD 10 scenes")
    parser.add_argument("--resolution", type=float, default=DEFAULT_RESOLUTION)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--input-bins", type=int, default=DEFAULT_INPUT_BINS)
    parser.add_argument("--loss-bins", type=int, default=DEFAULT_LOSS_BINS)
    parser.add_argument("--updates-per-arrival", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--process-res", type=int, default=DEFAULT_PROCESS_RES)
    parser.add_argument("--metric-model", default=DEFAULT_METRIC_MODEL)
    parser.add_argument("--sam-model", default=DEFAULT_SAM_MODEL)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    for name in ("input_bins", "loss_bins", "updates_per_arrival", "process_res"):
        if int(getattr(args, name)) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not math.isfinite(float(args.resolution)) or float(args.resolution) <= 0.0:
        parser.error("--resolution must be finite and positive")
    return args


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
    results = prepare_all(
        specs,
        output_root=args.output_root,
        oscd_output_root=args.oscd_output_root,
        resolution=float(args.resolution),
        max_frames=args.max_frames,
        force=bool(args.force),
        input_bins=int(args.input_bins),
        loss_bins=int(args.loss_bins),
        updates_per_arrival=int(args.updates_per_arrival),
        seed=int(args.seed),
        process_res=int(args.process_res),
        metric_model=str(args.metric_model),
        sam_model_name=str(args.sam_model),
    )
    aggregate = {"scene_count": len(results), "results": results}
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(
        json.dumps(aggregate, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(aggregate, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
