"""Learn a warm-started per-frame sigmoid cue boundary from a frozen O-SCD teacher.

This is an offline Stage-2 calibration ablation.  It never backpropagates into
the already-trained Gaussian teacher and never uses GT for optimization.  A
small MLP reads the current frame's normalized cue histogram/CDF and predicts
``tau`` and transition half-width for the L1-powered pixel × SAM cue.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch

from experiments.evaluate_cue_power_location_miou import load_gt, metrics_from_counts
from experiments.run_online_bayesian_lifespan_thaw import BASE_PLY_REL, build_causal_records
from experiments.train_cue_temporal_rchange import (
    DEFAULT_CUE_CACHE,
    DEFAULT_FIXED_CAMERAS,
    build_fixed_cue_views,
    load_fixed_camera_index,
    validate_cue_cache,
)
from experiments.train_real_temporal_rchange import DEFAULT_SOURCE, file_checksum, seed_everything
from temporal.change_cue_fusion import (
    fuse_power_product,
    normalized_oscd_pixel_cue_from_terms,
    oscd_pixel_terms,
    semantic_from_cached_sum,
)
from temporal.learnable_cue_boundary import HistogramBoundaryMLP, histogram_stage2_loss


DEFAULT_TEACHER_PLY = Path(
    "outputs/instance1_scene_change1_2_3_oscd_persistent_exact_allframes_120/"
    "persistent_control_final.ply"
)
DEFAULT_TEACHER_MASK_DIR = Path(
    "/home/rvl/workspace/github/O-SCD/output/ESCD_fixedpose_protocols_res4/"
    "scene_change1_2_3/renders/online_at_arrival"
)
DEFAULT_OUTPUT = Path("outputs/stage2_l1_power_sigmoid_boundary")
STATS_SCHEMA_VERSION = 1
BOUNDARY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BoundaryStatistics:
    """Compact sufficient statistics for histogram-conditioned calibration."""

    input_counts: np.ndarray
    loss_counts: np.ndarray
    teacher_sums: np.ndarray
    gt_sums: np.ndarray
    frame_names: tuple[str, ...]
    segment_names: tuple[str, ...]

    @property
    def frames(self) -> int:
        return int(self.input_counts.shape[0])


def _binned_sum(
    cue: torch.Tensor,
    values: torch.Tensor,
    *,
    bins: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return counts and value sums in uniform cue bins over ``[0,1]``."""

    if cue.shape != values.shape:
        raise ValueError("cue and values must share shape")
    indices = torch.floor(cue.clamp(0.0, 1.0) * bins).long().clamp_max(bins - 1)
    counts = torch.bincount(indices.flatten(), minlength=bins)
    sums = torch.bincount(
        indices.flatten(), weights=values.float().flatten(), minlength=bins
    )
    return counts, sums


def deterministic_train_validation_split(
    frames: int,
    *,
    validation_stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reserve every Nth ordered view while keeping both splits nonempty."""

    if frames < 2:
        raise ValueError("at least two frames are required for train/validation")
    if validation_stride < 2:
        raise ValueError("validation_stride must be at least two")
    indices = np.arange(frames)
    validation = (indices + 1) % validation_stride == 0
    if not validation.any():
        validation[-1] = True
    training = ~validation
    if not training.any():
        training[0] = True
        validation[0] = False
    return training, validation


def hard_metrics_from_histograms(
    counts: np.ndarray,
    gt_sums: np.ndarray,
    tau: np.ndarray,
) -> dict[str, Any]:
    """Evaluate per-frame hard ``q>tau`` masks from fine cue histograms."""

    if counts.shape != gt_sums.shape or counts.ndim != 2:
        raise ValueError("counts and gt_sums must share shape [F,K]")
    tau = np.asarray(tau, dtype=np.float64)
    if tau.shape != (counts.shape[0],):
        raise ValueError("tau must have shape [F]")
    centers = (np.arange(counts.shape[1], dtype=np.float64) + 0.5) / counts.shape[1]
    predicted = centers[None, :] > tau[:, None]
    tp = (gt_sums * predicted).sum(axis=1, dtype=np.float64)
    fp = ((counts - gt_sums) * predicted).sum(axis=1, dtype=np.float64)
    total_gt = gt_sums.sum(axis=1, dtype=np.float64)
    fn = total_gt - tp
    frame_metrics = [metrics_from_counts(a, b, c) for a, b, c in zip(tp, fp, fn)]
    aggregate = metrics_from_counts(float(tp.sum()), float(fp.sum()), float(fn.sum()))
    return {
        "mean_frame_iou": float(np.mean([row["iou"] for row in frame_metrics])),
        "mean_frame_f1": float(np.mean([row["f1"] for row in frame_metrics])),
        "aggregate": aggregate,
        "counts": {
            "tp": float(tp.sum()),
            "fp": float(fp.sum()),
            "fn": float(fn.sum()),
        },
    }


def _empty_exact_accumulator(segment_names: Sequence[str]) -> dict[str, Any]:
    def scope() -> dict[str, Any]:
        return {
            name: {"tp": 0.0, "fp": 0.0, "fn": 0.0, "frame_iou": [], "frame_f1": []}
            for name in ("heuristic", "learned")
        }

    return {
        "overall": scope(),
        "segments": {name: scope() for name in dict.fromkeys(segment_names)},
        "splits": {"training": scope(), "validation": scope()},
    }


def _add_exact_prediction(
    scope: dict[str, Any],
    name: str,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> None:
    prediction = prediction.bool()
    target = target.bool()
    tp = float((prediction & target).sum().item())
    fp = float((prediction & ~target).sum().item())
    fn = float((~prediction & target).sum().item())
    metric = metrics_from_counts(tp, fp, fn)
    scope[name]["tp"] += tp
    scope[name]["fp"] += fp
    scope[name]["fn"] += fn
    scope[name]["frame_iou"].append(metric["iou"])
    scope[name]["frame_f1"].append(metric["f1"])


def _summarize_exact_scope(scope: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, row in scope.items():
        output[name] = {
            "mean_frame_iou": float(np.mean(row["frame_iou"])),
            "mean_frame_f1": float(np.mean(row["frame_f1"])),
            "aggregate": metrics_from_counts(row["tp"], row["fp"], row["fn"]),
            "counts": {key: row[key] for key in ("tp", "fp", "fn")},
        }
    return output


def evaluate_exact_gt(
    args: argparse.Namespace,
    records: Sequence[Any],
    learned_tau: np.ndarray,
) -> dict[str, Any]:
    """Recompute raw cues and compare exact pixel masks against GT after training."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the Gaussian rasterizer")
    from gaussian_renderer import render
    from scene import GaussianModel

    if learned_tau.shape != (len(records),):
        raise ValueError("learned_tau must have one value per record")
    base_ply = (args.source_path / BASE_PLY_REL).resolve()
    validate_cue_cache(args.cue_cache_root, base_ply, args.resolution)
    cameras = load_fixed_camera_index(args.fixed_cameras_json)
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    reference = GaussianModel(sh_degree=3, active_sh_degree=3)
    reference.load_ply(str(base_ply))
    for name in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"):
        getattr(reference, name).requires_grad_(False)
    accumulator = _empty_exact_accumulator(
        tuple(str(record.segment_name) for record in records)
    )
    _training_split, validation_split = deterministic_train_validation_split(
        len(records), validation_stride=args.validation_stride
    )
    started = time.time()
    for index, (record, frame_tau) in enumerate(zip(records, learned_tau)):
        view = build_fixed_cue_views(
            [record], cameras, args.cue_cache_root, args.resolution
        )[0][0]
        with torch.no_grad():
            reference_rgb = render(view, reference, pipe, background)["render"]
            terms = oscd_pixel_terms(reference_rgb, view.original_image)
            original_pixel = normalized_oscd_pixel_cue_from_terms(terms, l1_exponent=1.0)
            powered_pixel = normalized_oscd_pixel_cue_from_terms(
                terms, l1_exponent=args.l1_exponent
            )
            semantic = semantic_from_cached_sum(view.candidate_map, original_pixel)
            cue = fuse_power_product(powered_pixel, semantic, exponent=1.0)[0] / 2.0
            gt = load_gt(
                args.source_path / "gt_mask" / record.name,
                width=int(view.image_width),
                height=int(view.image_height),
            ).to(device=cue.device)
        for name, prediction in (
            ("heuristic", cue > args.initial_tau),
            ("learned", cue > float(frame_tau)),
        ):
            _add_exact_prediction(accumulator["overall"], name, prediction, gt)
            _add_exact_prediction(
                accumulator["segments"][str(record.segment_name)], name, prediction, gt
            )
            split_name = "validation" if validation_split[index] else "training"
            _add_exact_prediction(accumulator["splits"][split_name], name, prediction, gt)
        if (
            index == 0
            or (index + 1) % args.progress_interval == 0
            or index + 1 == len(records)
        ):
            print(
                f"[exact GT {index + 1:03d}/{len(records):03d}] {record.name} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    overall = _summarize_exact_scope(accumulator["overall"])
    return {
        "threshold_contract": "exact pixel q>tau; GT is evaluation-only",
        "heuristic_tau": args.initial_tau,
        "overall": overall,
        "delta_mean_frame_iou": (
            overall["learned"]["mean_frame_iou"]
            - overall["heuristic"]["mean_frame_iou"]
        ),
        "segments": {
            name: _summarize_exact_scope(scope)
            for name, scope in accumulator["segments"].items()
        },
        "splits": {
            name: _summarize_exact_scope(scope)
            for name, scope in accumulator["splits"].items()
        },
        "runtime_seconds": time.time() - started,
    }


def _stats_paths(output_dir: Path) -> tuple[Path, Path]:
    return output_dir / "boundary_statistics.npz", output_dir / "boundary_statistics.json"


def _expected_stats_contract(args: argparse.Namespace, frames: int) -> dict[str, Any]:
    teacher_contract = (
        {
            "mode": "online_at_arrival_binary_masks",
            "path": str(args.teacher_mask_dir.resolve()),
            "mask": "saved hard render_change(...).mean(channel)>0.5 after current-frame updates",
        }
        if args.teacher_mode == "online_binary_masks"
        else {
            "mode": "final_ply_soft_rerender",
            "path": str(args.teacher_ply.resolve()),
            "sha256": file_checksum(args.teacher_ply),
            "mask": "clamp(render_change(...).mean(channel),0,1)",
        }
    )
    return {
        "schema_version": STATS_SCHEMA_VERSION,
        "frames": int(frames),
        "resolution": float(args.resolution),
        "input_bins": int(args.input_bins),
        "loss_bins": int(args.loss_bins),
        "l1_exponent": float(args.l1_exponent),
        "source_path": str(args.source_path.resolve()),
        "teacher": teacher_contract,
        "cue_formula": "q = norm(0.8*L1^alpha + 0.2*(1-SSIM)) * SAM",
    }


def _load_cached_statistics(
    args: argparse.Namespace,
    expected_contract: dict[str, Any],
) -> BoundaryStatistics | None:
    arrays_path, metadata_path = _stats_paths(args.output_dir)
    if args.rebuild_statistics or not arrays_path.exists() or not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata != expected_contract:
        return None
    with np.load(arrays_path, allow_pickle=False) as payload:
        return BoundaryStatistics(
            input_counts=payload["input_counts"],
            loss_counts=payload["loss_counts"],
            teacher_sums=payload["teacher_sums"],
            gt_sums=payload["gt_sums"],
            frame_names=tuple(str(value) for value in payload["frame_names"]),
            segment_names=tuple(str(value) for value in payload["segment_names"]),
        )


def build_statistics(
    args: argparse.Namespace,
    records: Sequence[Any],
    contract: dict[str, Any],
) -> BoundaryStatistics:
    """Render the frozen reference/teacher once and retain binned statistics."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the Gaussian rasterizer")
    from gaussian_renderer import render, render_change
    from scene import GaussianModel

    base_ply = (args.source_path / BASE_PLY_REL).resolve()
    validate_cue_cache(args.cue_cache_root, base_ply, args.resolution)
    cameras = load_fixed_camera_index(args.fixed_cameras_json)
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
    background = torch.zeros(3, dtype=torch.float32, device="cuda")

    reference = GaussianModel(sh_degree=3, active_sh_degree=3)
    reference.load_ply(str(base_ply))
    teacher = None
    if args.teacher_mode == "final_ply_soft":
        teacher = GaussianModel(sh_degree=3, active_sh_degree=0)
        teacher.load_ply(str(args.teacher_ply))
    for model in (reference,) if teacher is None else (reference, teacher):
        for name in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"):
            getattr(model, name).requires_grad_(False)

    input_counts: list[np.ndarray] = []
    loss_counts: list[np.ndarray] = []
    teacher_sums: list[np.ndarray] = []
    gt_sums: list[np.ndarray] = []
    started = time.time()
    for index, record in enumerate(records):
        view = build_fixed_cue_views(
            [record], cameras, args.cue_cache_root, args.resolution
        )[0][0]
        with torch.no_grad():
            reference_rgb = render(view, reference, pipe, background)["render"]
            terms = oscd_pixel_terms(reference_rgb, view.original_image)
            original_pixel = normalized_oscd_pixel_cue_from_terms(terms, l1_exponent=1.0)
            powered_pixel = normalized_oscd_pixel_cue_from_terms(
                terms, l1_exponent=args.l1_exponent
            )
            semantic = semantic_from_cached_sum(view.candidate_map, original_pixel)
            cue = fuse_power_product(powered_pixel, semantic, exponent=1.0) / 2.0
            if teacher is None:
                teacher_mask = load_gt(
                    args.teacher_mask_dir / record.name,
                    width=int(view.image_width),
                    height=int(view.image_height),
                ).to(device=cue.device, dtype=cue.dtype).unsqueeze(0)
            else:
                teacher_mask = render_change(view, teacher, pipe, background)["render"].mean(
                    dim=0, keepdim=True
                ).clamp(0.0, 1.0)
            gt = load_gt(
                args.source_path / "gt_mask" / record.name,
                width=int(view.image_width),
                height=int(view.image_height),
            ).to(device=cue.device, dtype=cue.dtype).unsqueeze(0)
            input_count, _ = _binned_sum(cue, torch.ones_like(cue), bins=args.input_bins)
            loss_count, teacher_sum = _binned_sum(cue, teacher_mask, bins=args.loss_bins)
            _, gt_sum = _binned_sum(cue, gt, bins=args.loss_bins)
        input_counts.append(input_count.cpu().numpy().astype(np.int64, copy=False))
        loss_counts.append(loss_count.cpu().numpy().astype(np.int64, copy=False))
        teacher_sums.append(teacher_sum.cpu().numpy().astype(np.float32, copy=False))
        gt_sums.append(gt_sum.cpu().numpy().astype(np.float32, copy=False))
        if index == 0 or (index + 1) % args.progress_interval == 0 or index + 1 == len(records):
            print(
                f"[statistics {index + 1:03d}/{len(records):03d}] {record.name} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )

    result = BoundaryStatistics(
        input_counts=np.stack(input_counts),
        loss_counts=np.stack(loss_counts),
        teacher_sums=np.stack(teacher_sums),
        gt_sums=np.stack(gt_sums),
        frame_names=tuple(str(record.name) for record in records),
        segment_names=tuple(str(record.segment_name) for record in records),
    )
    arrays_path, metadata_path = _stats_paths(args.output_dir)
    np.savez_compressed(
        arrays_path,
        input_counts=result.input_counts,
        loss_counts=result.loss_counts,
        teacher_sums=result.teacher_sums,
        gt_sums=result.gt_sums,
        frame_names=np.asarray(result.frame_names),
        segment_names=np.asarray(result.segment_names),
    )
    metadata_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    return result


def _loss_for_indices(
    model: HistogramBoundaryMLP,
    tau: torch.Tensor,
    width: torch.Tensor,
    counts: torch.Tensor,
    teacher_sums: torch.Tensor,
    indices: torch.Tensor,
    centers: torch.Tensor,
    soft_iou_weight: float,
):
    prediction = model.remap(
        centers.expand(indices.numel(), -1),
        tau[indices, None],
        width[indices, None],
    )
    return histogram_stage2_loss(
        prediction,
        counts[indices],
        teacher_sums[indices],
        soft_iou_weight=soft_iou_weight,
    )


def train_boundary_model(
    args: argparse.Namespace,
    statistics: BoundaryStatistics,
) -> tuple[HistogramBoundaryMLP, list[dict[str, float]], dict[str, Any]]:
    """Optimize Stage 2 with zero-head initialization and head-first warm-up."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HistogramBoundaryMLP(
        bins=args.input_bins,
        hidden=(args.hidden1, args.hidden2),
        initial_tau=args.initial_tau,
        initial_width=args.initial_width,
        tau_bounds=(args.tau_min, args.tau_max),
        width_bounds=(args.width_min, args.width_max),
        edge_probability=args.edge_probability,
    ).to(device)
    input_counts = torch.from_numpy(statistics.input_counts).to(device=device, dtype=torch.float32)
    loss_counts = torch.from_numpy(statistics.loss_counts).to(device=device, dtype=torch.float32)
    teacher_sums = torch.from_numpy(statistics.teacher_sums).to(device=device, dtype=torch.float32)
    training_np, validation_np = deterministic_train_validation_split(
        statistics.frames, validation_stride=args.validation_stride
    )
    training = torch.from_numpy(np.flatnonzero(training_np)).to(device=device, dtype=torch.long)
    validation = torch.from_numpy(np.flatnonzero(validation_np)).to(device=device, dtype=torch.long)
    training_mask = torch.from_numpy(training_np).to(device=device, dtype=torch.bool)
    centers = (
        (torch.arange(args.loss_bins, device=device, dtype=torch.float32) + 0.5)
        / args.loss_bins
    ).unsqueeze(0)
    segment_ids = torch.tensor(
        [list(dict.fromkeys(statistics.segment_names)).index(name) for name in statistics.segment_names],
        device=device,
        dtype=torch.long,
    )
    temporal_training_pairs = (
        (segment_ids[1:] == segment_ids[:-1])
        & training_mask[1:]
        & training_mask[:-1]
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": model.trunk.parameters(), "lr": 0.0, "name": "trunk"},
            {"params": model.head.parameters(), "lr": args.learning_rate * 0.1, "name": "head"},
        ],
        weight_decay=args.weight_decay,
    )
    history: list[dict[str, float]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_validation = float("inf")
    stale_epochs = 0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        head_progress = min(1.0, epoch / max(1, args.head_warmup_epochs))
        optimizer.param_groups[1]["lr"] = args.learning_rate * (0.1 + 0.9 * head_progress)
        optimizer.param_groups[0]["lr"] = (
            0.0 if epoch <= args.head_warmup_epochs else args.learning_rate * args.trunk_lr_scale
        )
        optimizer.zero_grad(set_to_none=True)
        tau, width = model(input_counts)
        stage2 = _loss_for_indices(
            model,
            tau,
            width,
            loss_counts,
            teacher_sums,
            training,
            centers,
            args.soft_iou_weight,
        )
        anchor_progress = min(1.0, epoch / max(1, args.anchor_decay_epochs))
        anchor_weight = args.anchor_weight * 0.5 * (1.0 + math.cos(math.pi * anchor_progress))
        anchor = (
            ((tau[training] - args.initial_tau) / args.initial_width).square()
            + ((width[training] - args.initial_width) / args.initial_width).square()
        ).mean()
        if statistics.frames > 1 and bool(temporal_training_pairs.any()):
            temporal = (
                ((tau[1:] - tau[:-1]) / args.initial_width).square()
                + ((width[1:] - width[:-1]) / args.initial_width).square()
            )[temporal_training_pairs].mean()
        else:
            temporal = tau.new_zeros(())
        objective = stage2.loss + anchor_weight * anchor + args.temporal_weight * temporal
        objective.backward()
        optimizer.step()

        with torch.no_grad():
            tau_eval, width_eval = model(input_counts)
            validation_loss = _loss_for_indices(
                model,
                tau_eval,
                width_eval,
                loss_counts,
                teacher_sums,
                validation,
                centers,
                args.soft_iou_weight,
            )
        row = {
            "epoch": float(epoch),
            "objective": float(objective.detach().item()),
            "train_stage2": float(stage2.loss.detach().item()),
            "train_bce": float(stage2.balanced_bce.detach().item()),
            "train_soft_iou": float(stage2.soft_iou.detach().item()),
            "validation_stage2": float(validation_loss.loss.item()),
            "validation_bce": float(validation_loss.balanced_bce.item()),
            "validation_soft_iou": float(validation_loss.soft_iou.item()),
            "anchor_weight": float(anchor_weight),
            "tau_mean": float(tau_eval.mean().item()),
            "width_mean": float(width_eval.mean().item()),
        }
        history.append(row)
        if row["validation_stage2"] < best_validation - args.minimum_delta:
            best_validation = row["validation_stage2"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % args.training_progress_interval == 0:
            print(
                f"[epoch {epoch:04d}] train={row['train_stage2']:.6f} "
                f"val={row['validation_stage2']:.6f} tau={row['tau_mean']:.4f} "
                f"width={row['width_mean']:.4f}",
                flush=True,
            )
        if epoch > args.anchor_decay_epochs and stale_epochs >= args.patience:
            print(f"[early stop] epoch={epoch} best={best_epoch}", flush=True)
            break

    model.load_state_dict(best_state)
    audit = {
        "best_epoch": best_epoch,
        "best_validation_stage2": best_validation,
        "epochs_completed": len(history),
        "training_frames": int(training.numel()),
        "validation_frames": int(validation.numel()),
        "runtime_seconds": time.time() - started,
        "warm_start": {
            "zero_initialized_output_head": True,
            "initial_tau": args.initial_tau,
            "initial_width": args.initial_width,
            "head_warmup_epochs": args.head_warmup_epochs,
            "trunk_frozen_during_head_warmup": True,
            "anchor_decay_epochs": args.anchor_decay_epochs,
        },
    }
    return model, history, audit


def _summarize_values(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _save_plot(history: list[dict[str, float]], tau: np.ndarray, width: np.ndarray, path: Path) -> None:
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), dpi=160, sharex=False)
    axes[0].plot(epochs, [row["train_stage2"] for row in history], label="train")
    axes[0].plot(epochs, [row["validation_stage2"] for row in history], label="validation")
    axes[0].set_ylabel("Stage-2 loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].plot(tau, label="tau", color="#ff9f1c")
    axes[1].plot(width, label="half-width", color="#2ec4b6")
    axes[1].axhline(0.25, linestyle="--", color="#ff9f1c", alpha=0.5)
    axes[1].axhline(0.10, linestyle="--", color="#2ec4b6", alpha=0.5)
    axes[1].set_xlabel("causal frame index (network input uses current histogram only)")
    axes[1].set_ylabel("learned boundary")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _save_gt_comparison(exact_gt: dict[str, Any], path: Path) -> None:
    """Plot exact heuristic-vs-learned mean-frame IoU by evaluation scope."""

    import matplotlib.pyplot as plt

    labels = ["overall", "train views", "validation views", "SC1", "SC2", "SC3"]
    scopes = [
        exact_gt["overall"],
        exact_gt["splits"]["training"],
        exact_gt["splits"]["validation"],
        *[exact_gt["segments"][f"scene_change{index}"] for index in (1, 2, 3)],
    ]
    heuristic = [scope["heuristic"]["mean_frame_iou"] for scope in scopes]
    learned = [scope["learned"]["mean_frame_iou"] for scope in scopes]
    positions = np.arange(len(labels))
    width = 0.36
    figure, axis = plt.subplots(figsize=(11, 5.5), dpi=170)
    axis.bar(positions - width / 2, heuristic, width, label="fixed tau=0.25", color="#666a73")
    axis.bar(positions + width / 2, learned, width, label="learned tau(frame)", color="#2ec4b6")
    for x, before, after in zip(positions, heuristic, learned):
        axis.text(x, max(before, after) + 0.008, f"{after-before:+.3f}", ha="center", fontsize=9)
    axis.set_xticks(positions, labels)
    axis.set_ylim(0.0, max(learned + heuristic) + 0.10)
    axis.set_ylabel("exact mean-frame foreground IoU")
    axis.set_title("Stage-2 histogram MLP boundary vs fixed heuristic")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.teacher_mode == "online_binary_masks":
        if not args.teacher_mask_dir.is_dir():
            raise FileNotFoundError(
                f"frozen teacher mask directory not found: {args.teacher_mask_dir}"
            )
    elif not args.teacher_ply.is_file():
        raise FileNotFoundError(f"frozen teacher PLY not found: {args.teacher_ply}")
    records, _ = build_causal_records(args.source_path, max_frames=args.max_frames)
    if len(records) < 2:
        raise RuntimeError("Stage 2 requires at least two frames")
    contract = _expected_stats_contract(args, len(records))
    statistics = _load_cached_statistics(args, contract)
    if statistics is None:
        statistics = build_statistics(args, records, contract)
    else:
        print(f"Loaded cached statistics for {statistics.frames} frames", flush=True)

    model, history, training_audit = train_boundary_model(args, statistics)
    device = next(model.parameters()).device
    with torch.no_grad():
        input_counts = torch.from_numpy(statistics.input_counts).to(device=device, dtype=torch.float32)
        tau_tensor, width_tensor = model(input_counts)
    tau = tau_tensor.cpu().numpy()
    width = width_tensor.cpu().numpy()

    baseline_tau = np.full(statistics.frames, args.initial_tau, dtype=np.float64)
    overall_baseline = hard_metrics_from_histograms(
        statistics.loss_counts, statistics.gt_sums, baseline_tau
    )
    overall_learned = hard_metrics_from_histograms(
        statistics.loss_counts, statistics.gt_sums, tau
    )
    segments: dict[str, Any] = {}
    for segment in dict.fromkeys(statistics.segment_names):
        mask = np.asarray([name == segment for name in statistics.segment_names])
        segments[segment] = {
            "heuristic": hard_metrics_from_histograms(
                statistics.loss_counts[mask], statistics.gt_sums[mask], baseline_tau[mask]
            ),
            "learned": hard_metrics_from_histograms(
                statistics.loss_counts[mask], statistics.gt_sums[mask], tau[mask]
            ),
        }
    exact_gt = None
    if not args.skip_exact_gt_evaluation:
        exact_gt = evaluate_exact_gt(args, records, tau)

    boundary_artifact = {
        "schema_version": BOUNDARY_SCHEMA_VERSION,
        "remap": "sigmoid",
        "width_semantics": "output is 0.05/0.95 at tau-width/tau+width",
        "edge_probability": args.edge_probability,
        "cue_scale": 2.0,
        "cue_formula": contract["cue_formula"],
        "frames": {
            name: {"tau": float(frame_tau), "width": float(frame_width)}
            for name, frame_tau, frame_width in zip(statistics.frame_names, tau, width)
        },
    }
    boundary_path = args.output_dir / "learned_boundaries.json"
    boundary_path.write_text(json.dumps(boundary_artifact, indent=2) + "\n", encoding="utf-8")
    checkpoint_path = args.output_dir / "boundary_mlp.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "configuration": model.configuration(),
            "training_audit": training_audit,
        },
        checkpoint_path,
    )
    history_path = args.output_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    plot_path = args.output_dir / "training_and_boundaries.png"
    _save_plot(history, tau, width, plot_path)
    gt_plot_path = args.output_dir / "exact_gt_comparison.png"
    if exact_gt is not None:
        _save_gt_comparison(exact_gt, gt_plot_path)

    summary = {
        "schema_version": 1,
        "script": "experiments/train_stage2_cue_boundary.py",
        "experiment_scope": "offline post-hoc Stage-2 cue calibration ablation",
        "causality_warning": (
            "Each default teacher mask is causal online-at-arrival, but BoundaryNet is "
            "trained post-hoc on this same complete stream. This is a calibration pilot, "
            "not a held-out online detector result. At inference BoundaryNet itself reads "
            "only the current frame cue histogram."
        ),
        "teacher_frozen_and_detached": True,
        "gt_used_for_training": False,
        "teacher": contract["teacher"],
        "frames": statistics.frames,
        "cue_formula": contract["cue_formula"],
        "model": model.configuration(),
        "loss": "frame-balanced soft BCE + lambda_iou*(1-softIoU) + warm-start anchor + temporal smoothness",
        "training": training_audit,
        "learned_tau": _summarize_values(tau),
        "learned_width": _summarize_values(width),
        "gt_evaluation_only": {
            "exact_pixel": exact_gt,
            "histogram_approximation_bins": args.loss_bins,
            "heuristic_tau_0.25": overall_baseline,
            "learned_per_frame_tau": overall_learned,
            "delta_mean_frame_iou": (
                overall_learned["mean_frame_iou"] - overall_baseline["mean_frame_iou"]
            ),
            "segments": segments,
        },
        "artifacts": {
            "boundary_json": str(boundary_path),
            "checkpoint": str(checkpoint_path),
            "history": str(history_path),
            "plot": str(plot_path),
            "gt_comparison_plot": str(gt_plot_path) if exact_gt is not None else None,
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def probability(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("value must lie in (0,1)")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and nonnegative")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", type=Path, default=Path(DEFAULT_SOURCE))
    parser.add_argument(
        "--teacher-mode",
        choices=("online_binary_masks", "final_ply_soft"),
        default="online_binary_masks",
    )
    parser.add_argument(
        "--teacher-mask-dir", type=Path, default=DEFAULT_TEACHER_MASK_DIR
    )
    parser.add_argument("--teacher-ply", type=Path, default=DEFAULT_TEACHER_PLY)
    parser.add_argument("--fixed-cameras-json", type=Path, default=Path(DEFAULT_FIXED_CAMERAS))
    parser.add_argument("--cue-cache-root", type=Path, default=Path(DEFAULT_CUE_CACHE))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resolution", type=positive_float, default=4.0)
    parser.add_argument("--l1-exponent", type=positive_float, default=0.3)
    parser.add_argument("--input-bins", type=positive_int, default=64)
    parser.add_argument("--loss-bins", type=positive_int, default=512)
    parser.add_argument("--hidden1", type=positive_int, default=32)
    parser.add_argument("--hidden2", type=positive_int, default=16)
    parser.add_argument("--initial-tau", type=probability, default=0.25)
    parser.add_argument("--initial-width", type=probability, default=0.10)
    parser.add_argument("--tau-min", type=probability, default=0.05)
    parser.add_argument("--tau-max", type=probability, default=0.75)
    parser.add_argument("--width-min", type=probability, default=0.02)
    parser.add_argument("--width-max", type=probability, default=0.30)
    parser.add_argument("--edge-probability", type=probability, default=0.05)
    parser.add_argument("--epochs", type=positive_int, default=500)
    parser.add_argument("--head-warmup-epochs", type=positive_int, default=40)
    parser.add_argument("--anchor-decay-epochs", type=positive_int, default=100)
    parser.add_argument("--learning-rate", type=positive_float, default=0.002)
    parser.add_argument("--trunk-lr-scale", type=positive_float, default=0.5)
    parser.add_argument("--weight-decay", type=nonnegative_float, default=1e-4)
    parser.add_argument("--soft-iou-weight", type=nonnegative_float, default=1.0)
    parser.add_argument("--anchor-weight", type=nonnegative_float, default=0.25)
    parser.add_argument("--temporal-weight", type=nonnegative_float, default=0.02)
    parser.add_argument("--validation-stride", type=positive_int, default=5)
    parser.add_argument("--patience", type=positive_int, default=100)
    parser.add_argument("--minimum-delta", type=nonnegative_float, default=1e-6)
    parser.add_argument("--max-frames", type=positive_int, default=None)
    parser.add_argument("--progress-interval", type=positive_int, default=25)
    parser.add_argument("--training-progress-interval", type=positive_int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rebuild-statistics", action="store_true")
    parser.add_argument("--skip-exact-gt-evaluation", action="store_true")
    args = parser.parse_args(argv)
    if not args.tau_min < args.initial_tau < args.tau_max:
        parser.error("initial tau must lie strictly inside tau bounds")
    if not args.width_min < args.initial_width < args.width_max:
        parser.error("initial width must lie strictly inside width bounds")
    if args.validation_stride < 2:
        parser.error("validation stride must be at least two")
    if args.input_bins < 2 or args.loss_bins < 2:
        parser.error("histogram bin counts must be at least two")
    if args.edge_probability >= 0.5:
        parser.error("edge probability must be smaller than 0.5")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
