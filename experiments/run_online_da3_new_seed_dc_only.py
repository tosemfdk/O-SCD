#!/usr/bin/env python3
"""SC3 u120: fixed Depth Anything 3 NEW seeds with DC-only learning.

The reference topology and every DA3 seed geometry attribute are immutable.
The original two signed base-DC memories model existing-surface change and
REMOVED regions.  A separate dynamically growing DA3 sidecar learns only its
DC values from the confirmed causal NEW mask.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from transformers import Sam2Model


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.analyze_da3_depth_prior_new_seeds import (  # noqa: E402
    DEFAULT_TRACE,
    _read_resized_rgb,
    _resize_tensor,
    load_trace,
    render_reference_depth,
)
from experiments.run_online_xfeat_new_seed import (  # noqa: E402
    RUN,
    annotation_masks64,
    annotation_masks_at_size,
    audit_base,
    binary_metrics,
    build_concatenated_change_view,
    build_fixed_cue_views,
    evaluate_branch_change,
    extract_delta,
    fixed_camera_matrices,
    load_fixed_camera_index,
    load_object_annotations,
    render_new_sidecar_learned_score,
    save_active_seed_ply,
    scene_records,
    snapshot_base,
    train_current_signed_masks,
    train_seed_dc_from_projected_coverage,
)
from experiments.render_temporal_confusion_maps import (  # noqa: E402
    PALETTE,
    confusion_arrays,
    metrics_from_counts,
    save_gif,
    save_panel,
)
from experiments.run_ref_sc1_change_cue_density import (  # noqa: E402
    deterministic_training_view_index,
)
from gaussian_renderer import render, render_change  # noqa: E402
from scene import GaussianModel  # noqa: E402
from temporal.change_cue_fusion import (  # noqa: E402
    fuse_power_product,
    normalized_oscd_pixel_cue_from_terms,
    oscd_pixel_terms,
    semantic_from_cached_sum,
    sigmoid_soft_binarize,
)
from temporal.change_evidence import accumulate_change_evidence  # noqa: E402
from temporal.depth_prior_new_seeding import (  # noqa: E402
    DepthPriorSeedConfig,
    build_depth_prior_new_seeds,
    fit_reference_depth_scale,
    reference_scale_anchor_mask,
)
from temporal.lifespan_gate_beta import (  # noqa: E402
    LifespanGateBetaConfig,
    LifespanGateBetaFilter,
)
from temporal.new_seed_gaussians import NewSeedGaussianModel  # noqa: E402


DEFAULT_OUTPUT = REPO / "outputs/ref_sc3_da3_new_seed_dc_only_u120_20260902"
VISUAL_FRAMES = {48, 52, 64, 80, 84}


@dataclass(frozen=True)
class LearnedCueBoundary:
    tau: float
    width: float


@dataclass(frozen=True)
class LearnedCueArtifact:
    boundaries: dict[str, LearnedCueBoundary]
    edge_probability: float
    cue_scale: float
    cue_formula: str


def _load_learned_cue_artifact(path: Path) -> LearnedCueArtifact:
    """Load the frozen per-frame Stage-2 sigmoid calibration artifact."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("remap") != "sigmoid":
        raise ValueError("learned cue artifact must declare remap=sigmoid")
    edge_probability = float(payload.get("edge_probability", float("nan")))
    if not math.isfinite(edge_probability) or not 0.0 < edge_probability < 0.5:
        raise ValueError("learned cue edge_probability must lie in (0,0.5)")
    cue_scale = float(payload.get("cue_scale", float("nan")))
    if not math.isfinite(cue_scale) or cue_scale <= 0.0:
        raise ValueError("learned cue_scale must be finite and positive")
    cue_formula = payload.get("cue_formula")
    if not isinstance(cue_formula, str) or not cue_formula.strip():
        raise ValueError("learned cue artifact must declare cue_formula")
    raw_frames = payload.get("frames")
    if not isinstance(raw_frames, dict) or not raw_frames:
        raise ValueError("learned cue artifact must contain frame parameters")
    boundaries: dict[str, LearnedCueBoundary] = {}
    for frame_name, raw in raw_frames.items():
        if not isinstance(raw, dict):
            raise ValueError(f"invalid learned cue boundary for {frame_name}")
        tau = float(raw.get("tau", float("nan")))
        width = float(raw.get("width", float("nan")))
        if not math.isfinite(tau) or not 0.0 < tau < 1.0:
            raise ValueError(f"invalid learned cue tau for {frame_name}")
        if not math.isfinite(width) or width <= 0.0:
            raise ValueError(f"invalid learned cue width for {frame_name}")
        boundaries[str(frame_name)] = LearnedCueBoundary(tau=tau, width=width)
    return LearnedCueArtifact(
        boundaries=boundaries,
        edge_probability=edge_probability,
        cue_scale=cue_scale,
        cue_formula=cue_formula,
    )


def _calibrate_stage2_cue(
    *,
    cached_sum: torch.Tensor,
    reference_rgb: torch.Tensor,
    online_rgb: torch.Tensor,
    boundary: LearnedCueBoundary,
    l1_exponent: float,
    edge_probability: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct normalized Stage-2 q and return its learned soft mask Q."""

    terms = oscd_pixel_terms(reference_rgb, online_rgb)
    original_pixel = normalized_oscd_pixel_cue_from_terms(
        terms, l1_exponent=1.0
    )
    powered_pixel = normalized_oscd_pixel_cue_from_terms(
        terms, l1_exponent=float(l1_exponent)
    )
    semantic = semantic_from_cached_sum(cached_sum, original_pixel)
    normalized = fuse_power_product(
        powered_pixel, semantic, exponent=1.0
    ) / 2.0
    calibrated = sigmoid_soft_binarize(
        normalized,
        tau=boundary.tau,
        width=boundary.width,
        edge_probability=edge_probability,
    )
    return normalized, calibrated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--updates-per-frame", type=int, default=120)
    parser.add_argument("--base-lr", type=float, default=0.0025)
    parser.add_argument("--seed-lr", type=float, default=0.0025)
    parser.add_argument("--model", default="depth-anything/DA3-SMALL")
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--sampling-stride", type=int, default=4)
    parser.add_argument("--max-new-seeds-per-frame", type=int, default=2048)
    parser.add_argument("--max-total-seeds", type=int, default=12000)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--confidence-quantile", type=float, default=0.40)
    parser.add_argument("--seed-opacity", type=float, default=0.10)
    parser.add_argument(
        "--seed-activation",
        choices=("direct_active", "never_open_detector"),
        default="direct_active",
        help=(
            "Either activate every accepted DA3 seed immediately or insert it "
            "as NEVER_OPEN and let the raw-cue BF30 detector decide"
        ),
    )
    parser.add_argument(
        "--occupancy-source",
        choices=("seed_only", "rchange_and_seed"),
        default="seed_only",
        help="Voxel occupancy used to reject DA3 births",
    )
    parser.add_argument("--candidate-bayes-factor-threshold", type=float, default=30.0)
    parser.add_argument("--gate-stable-flip-prior", type=float, default=1.0)
    parser.add_argument("--gate-stable-keep-prior", type=float, default=10.0)
    parser.add_argument("--gate-reset-flip-prior", type=float, default=1.0)
    parser.add_argument("--gate-reset-keep-prior", type=float, default=1.0)
    parser.add_argument("--detector-cue-threshold", type=float, default=0.5)
    parser.add_argument("--detector-mass-saturation", type=float, default=1.0)
    parser.add_argument("--detector-min-evidence-mass", type=float, default=1.0e-6)
    parser.add_argument(
        "--cue-supervision",
        choices=("raw_binary", "learned_stage2_sigmoid_soft"),
        default="raw_binary",
        help=(
            "Keep the Part19 binary cached cue or reconstruct the learned "
            "Stage-2 sigmoid cue and use its soft output for DC supervision"
        ),
    )
    parser.add_argument(
        "--cue-boundary-json",
        type=Path,
        default=None,
        help="Frozen per-frame Stage-2 tau/width artifact",
    )
    parser.add_argument("--cue-l1-exponent", type=float, default=0.3)
    parser.add_argument(
        "--seed-cue-support-threshold",
        type=float,
        default=None,
        help=(
            "Minimum cue support for discrete DA3 birth; defaults to 0.05 for "
            "learned soft Q and to the detector threshold for raw binary cue"
        ),
    )
    parser.add_argument("--sam-model", default="facebook/sam2.1-hiera-tiny")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--training-schedule",
        choices=("current_exact", "oscd_replay"),
        default="current_exact",
        help=(
            "Repeat the current view or use O-SCD's 0.33-current, "
            "0.67-causal-uniform replay schedule"
        ),
    )
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument(
        "--save-confusion-gif",
        action="store_true",
        help="Save per-frame RGB/GT/prediction/confusion panels and a GIF",
    )
    parser.add_argument("--confusion-panel-width", type=int, default=260)
    parser.add_argument("--confusion-gif-width", type=int, default=720)
    parser.add_argument("--confusion-gif-duration-ms", type=int, default=160)
    args = parser.parse_args()
    for name in ("updates_per_frame", "window", "process_res", "sampling_stride", "max_new_seeds_per_frame", "max_total_seeds"):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.window < 2:
        parser.error("--window must be at least 2")
    if args.voxel_size <= 0.0:
        parser.error("--voxel-size must be positive")
    if not 0.0 < args.seed_opacity < 1.0:
        parser.error("--seed-opacity must be in (0,1)")
    if not math.isfinite(args.candidate_bayes_factor_threshold) or args.candidate_bayes_factor_threshold <= 1.0:
        parser.error("--candidate-bayes-factor-threshold must be finite and > 1")
    for name in (
        "gate_stable_flip_prior",
        "gate_stable_keep_prior",
        "gate_reset_flip_prior",
        "gate_reset_keep_prior",
        "detector_mass_saturation",
    ):
        if not math.isfinite(float(getattr(args, name))) or float(getattr(args, name)) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not math.isfinite(args.detector_cue_threshold):
        parser.error("--detector-cue-threshold must be finite")
    if not math.isfinite(args.detector_min_evidence_mass) or args.detector_min_evidence_mass < 0.0:
        parser.error("--detector-min-evidence-mass must be finite and nonnegative")
    if args.cue_supervision == "learned_stage2_sigmoid_soft":
        if args.cue_boundary_json is None:
            parser.error(
                "--cue-boundary-json is required for learned_stage2_sigmoid_soft"
            )
        if not args.cue_boundary_json.is_file():
            parser.error(f"learned cue artifact not found: {args.cue_boundary_json}")
    if not math.isfinite(args.cue_l1_exponent) or args.cue_l1_exponent <= 0.0:
        parser.error("--cue-l1-exponent must be finite and positive")
    if args.seed_cue_support_threshold is None:
        args.seed_cue_support_threshold = (
            0.05
            if args.cue_supervision == "learned_stage2_sigmoid_soft"
            else float(args.detector_cue_threshold)
        )
    if (
        not math.isfinite(float(args.seed_cue_support_threshold))
        or not 0.0 <= float(args.seed_cue_support_threshold) <= 1.0
    ):
        parser.error("--seed-cue-support-threshold must lie in [0,1]")
    for name in (
        "confusion_panel_width",
        "confusion_gif_width",
        "confusion_gif_duration_ms",
    ):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def _sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _seed_geometry_snapshot(
    model: NewSeedGaussianModel, *, include_lifespan: bool = True
) -> dict[str, torch.Tensor]:
    snapshot = {
        "xyz": model._xyz.detach().cpu().clone(),
        "features_rest": model._features_rest.detach().cpu().clone(),
        "opacity": model._opacity.detach().cpu().clone(),
        "scaling": model._scaling.detach().cpu().clone(),
        "rotation": model._rotation.detach().cpu().clone(),
    }
    if include_lifespan:
        snapshot["start"] = model.start.detach().cpu().clone()
        snapshot["end"] = model.end.detach().cpu().clone()
    return snapshot


def _audit_seed_geometry(model: NewSeedGaussianModel, expected: dict[str, torch.Tensor]) -> dict[str, Any]:
    rows = {}
    for name, before in expected.items():
        after = getattr(model, f"_{name}", None)
        if after is None:
            after = getattr(model, name)
        after = after.detach().cpu()
        equal = torch.equal(before, after)
        rows[name] = {
            "shape": list(after.shape),
            "bitwise_equal": equal,
            "sha256": _sha256(after),
            "max_abs_difference": (
                0.0
                if equal or not before.numel()
                else float(
                    torch.nan_to_num(
                        (before.float() - after.float()).abs(),
                        nan=float("inf"),
                        posinf=float("inf"),
                        neginf=float("inf"),
                    )
                    .max()
                    .item()
                )
            ),
        }
    return {"all_fields_bitwise_equal": all(row["bitwise_equal"] for row in rows.values()), "fields": rows}


def _accept_new_voxels(
    xyz: torch.Tensor,
    occupied: set[tuple[int, int, int]],
    voxel_size: float,
    maximum: int,
) -> torch.Tensor:
    accepted: list[int] = []
    for index, point in enumerate(xyz.detach().cpu().numpy()):
        key = tuple(np.floor(point / float(voxel_size)).astype(np.int64).tolist())
        if key in occupied:
            continue
        occupied.add(key)
        accepted.append(index)
        if len(accepted) >= maximum:
            break
    return torch.tensor(accepted, dtype=torch.long)


def _occupied_voxels(
    xyz: torch.Tensor, voxel_size: float
) -> set[tuple[int, int, int]]:
    """Voxelize fixed R_change centers once for causal free-space birth."""

    if not isinstance(xyz, torch.Tensor) or xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape [N,3]")
    if not math.isfinite(float(voxel_size)) or float(voxel_size) <= 0.0:
        raise ValueError("voxel_size must be finite and positive")
    if not bool(torch.isfinite(xyz).all()):
        raise ValueError("xyz must be finite")
    keys = np.floor(
        xyz.detach().cpu().numpy() / float(voxel_size)
    ).astype(np.int64)
    return {tuple(int(value) for value in row) for row in keys}


@torch.no_grad()
def _apply_seed_detector_update(
    *,
    tracker: LifespanGateBetaFilter,
    seeds: NewSeedGaussianModel,
    delta_a: torch.Tensor,
    delta_b: torch.Tensor,
    total_mass: torch.Tensor,
    timestamp: int,
) -> dict[str, Any]:
    """Apply one raw-cue BF30 update to every current fixed seed row."""

    count = seeds.num_seeds
    if not (
        delta_a.shape == delta_b.shape == total_mass.shape == (count,)
    ):
        raise ValueError("seed detector evidence must have shape [num_seeds]")
    if count == 0:
        return {
            "observed": 0,
            "candidate": 0,
            "committed": 0,
            "opened": 0,
            "closed": 0,
            "active": 0,
            "never_open": 0,
            "closed_total": 0,
        }
    rows = torch.arange(count, device=delta_a.device, dtype=torch.long)
    active_before = seeds.active_mask(float(timestamp))
    update = tracker.update(
        delta_a,
        delta_b,
        total_mass,
        current_active=active_before,
        row_indices=rows,
        timestamp=int(timestamp),
    )
    committed = update.candidate_committed.to(dtype=torch.bool)
    open_rows = rows[committed & ~active_before]
    close_rows = rows[committed & active_before]
    if open_rows.numel():
        seeds.open_rows(open_rows, timestamp)
    if close_rows.numel():
        seeds.close_rows(close_rows, timestamp)
    active_after = seeds.active_mask(float(timestamp))
    return {
        "observed": int(update.observed.sum().item()),
        "candidate": int(update.candidate_active.sum().item()),
        "committed": int(committed.sum().item()),
        "opened": int(open_rows.numel()),
        "closed": int(close_rows.numel()),
        "active": int(active_after.sum().item()),
        "never_open": int(seeds.never_open_mask().sum().item()),
        "closed_total": int(seeds.closed_mask(float(timestamp)).sum().item()),
    }


def _aggregate(rows: list[dict[str, Any]], branch: str, scope: str) -> dict[str, Any]:
    names = ("tp", "tn", "fp", "fn", "pred_positive", "gt_positive", "pixels")
    counts = {
        name: int(sum(int(row[f"{branch}_{scope}_{name}"]) for row in rows))
        for name in names
    }
    tp, tn, fp, fn = (counts[name] for name in ("tp", "tn", "fp", "fn"))
    return {
        **counts,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "mean_frame_iou": float(np.mean([row[f"{branch}_{scope}_iou"] for row in rows])),
        "mean_frame_f1": float(np.mean([row[f"{branch}_{scope}_f1"] for row in rows])),
    }


def _summarize_optional(
    rows: list[dict[str, Any]], key: str
) -> dict[str, float | int] | None:
    values = np.asarray(
        [float(row[key]) for row in rows if row.get(key) is not None], dtype=float
    )
    if not values.size:
        return None
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _flatten_metrics(row: dict[str, Any], prefix: str, metrics: dict[str, Any]) -> None:
    for scope in ("full", "new_full", "remove_full"):
        for name in ("tp", "tn", "fp", "fn", "pred_positive", "gt_positive", "pixels", "precision", "recall", "iou", "f1", "accuracy"):
            row[f"{prefix}_{scope}_{name}"] = metrics[f"{scope}_{name}"]


@torch.inference_mode()
def _method_scores(
    view: Any,
    base: GaussianModel,
    plus_dc: torch.Tensor,
    minus_dc: torch.Tensor,
    seeds: NewSeedGaussianModel | None,
    new_sign: str | None,
    pipe: SimpleNamespace,
    background: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    plus_model: Any = base
    minus_model: Any = base
    plus_override: torch.Tensor | None = plus_dc
    minus_override: torch.Tensor | None = minus_dc
    if seeds is not None and seeds.num_seeds and new_sign in {"+", "-"}:
        combined = build_concatenated_change_view(
            base,
            seeds,
            timestamp=float(view.timestamp),
            base_dc=plus_dc if new_sign == "+" else minus_dc,
            detach_base_dc=True,
        )
        if new_sign == "+":
            plus_model, plus_override = combined, None
        else:
            minus_model, minus_override = combined, None
    plus = render_change(
        view,
        plus_model,
        pipe,
        background,
        override_dc=plus_override,
        override_opacity=base.get_opacity.detach() if plus_override is not None else None,
    )["render"].mean(dim=0)
    minus = render_change(
        view,
        minus_model,
        pipe,
        background,
        override_dc=minus_override,
        override_opacity=base.get_opacity.detach() if minus_override is not None else None,
    )["render"].mean(dim=0)
    total = torch.maximum(plus, minus).clamp(0.0, 1.0)
    new = (
        render_new_sidecar_learned_score(view, seeds, pipe, background)[0]
        if seeds is not None
        else torch.zeros_like(total)
    )
    remove = minus if new_sign == "+" else plus if new_sign == "-" else torch.zeros_like(total)
    return total.cpu().numpy(), new.cpu().numpy(), remove.clamp(0.0, 1.0).cpu().numpy()


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def _panel(array: np.ndarray, title: str, subtitle: str) -> Image.Image:
    if array.ndim == 2:
        array = np.repeat(np.uint8(np.clip(array * 255.0, 0, 255))[..., None], 3, axis=2)
    else:
        array = np.uint8(np.clip(array, 0, 255))
    image = Image.fromarray(array).resize((300, 535), Image.Resampling.BILINEAR)
    result = Image.new("RGB", (300, 595), "white")
    result.paste(image, (0, 60))
    draw = ImageDraw.Draw(result)
    draw.text((8, 6), title, font=_font(15), fill="black")
    draw.text((8, 34), subtitle[:58], font=_font(10), fill=(65, 65, 65))
    return result


def _save_visual(
    path: Path,
    rgb: np.ndarray,
    new_mask: np.ndarray,
    baseline: tuple[np.ndarray, np.ndarray, np.ndarray],
    seeded: tuple[np.ndarray, np.ndarray, np.ndarray],
    object004: np.ndarray,
    object010: np.ndarray,
    row: dict[str, Any],
) -> None:
    cue = rgb.astype(np.float32)
    cue[new_mask] = 0.25 * cue[new_mask] + 0.75 * np.array([20, 210, 220])
    gt = rgb.astype(np.float32)
    gt[object004] = 0.2 * gt[object004] + 0.8 * np.array([40, 220, 80])
    gt[object010] = 0.2 * gt[object010] + 0.8 * np.array([245, 190, 35])
    items = [
        _panel(rgb, "Inference RGB", row["frame_name"]),
        _panel(cue, "Causal NEW target", f"pixels={row['new_target_pixels']}"),
        _panel(baseline[0], "Base DC-only total", f"IoU={row['baseline_full_iou']:.3f}"),
        _panel(seeded[0], "Base + DA3 seed DC", f"IoU={row['da3_full_iou']:.3f}"),
        _panel(seeded[1], "DA3 sidecar learned DC", f"NEW IoU={row['da3_new_full_iou']:.3f}"),
        _panel(gt, "Evaluation-only objects", "green=object004; yellow=object010"),
    ]
    board = Image.new("RGB", (900, 1260), "white")
    ImageDraw.Draw(board).text(
        (18, 18),
        f"SC3 frame {row['frame_local']:03d}: fixed DA3 geometry, DC-only u120",
        font=_font(24),
        fill="black",
    )
    for index, item in enumerate(items):
        board.paste(item, ((index % 3) * 300, 70 + (index // 3) * 595))
    board.save(path)


def _save_temporal_confusion_frame(
    output_dir: Path,
    *,
    rgb: torch.Tensor,
    gt: np.ndarray,
    score: np.ndarray,
    row: dict[str, Any],
    panel_width: int,
) -> Path:
    """Save one post-update overall-SCD confusion panel and its raw masks."""

    gt_tensor = torch.from_numpy(np.asarray(gt, dtype=bool))
    pred_tensor = torch.from_numpy(np.asarray(score) >= 0.5)
    confusion, counts = confusion_arrays(pred_tensor, gt_tensor, 0.5)
    for name in ("tp", "tn", "fp", "fn", "pred_positive", "gt_positive", "pixels"):
        expected = int(row[f"da3_full_{name}"])
        if counts[name] != expected:
            raise RuntimeError(
                f"confusion export disagrees with frame metric {name}: "
                f"{counts[name]} != {expected}"
            )

    stem = f"{int(row['frame_local']):06d}_{Path(str(row['frame_name'])).stem}"
    raw_path = output_dir / "raw_confusion" / f"{stem}_confusion.png"
    pred_path = output_dir / "pred_binary" / f"{stem}_pred.png"
    panel_path = output_dir / "panels" / f"{stem}_panel.png"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(confusion).save(raw_path)
    Image.fromarray(pred_tensor.to(torch.uint8).mul(255).numpy()).save(pred_path)
    save_panel(
        rgb.detach().cpu(),
        gt_tensor.float(),
        pred_tensor.float(),
        confusion,
        panel_path,
        (
            f"scene_change3 | local t={int(row['frame_local']):03d} | "
            f"global t={int(row['frame_global'])} | "
            f"seeds={int(row['seed_count'])} "
            f"O/N/C={int(row['seed_active_count'])}/"
            f"{int(row['seed_never_open_count'])}/"
            f"{int(row['seed_closed_count'])}"
        ),
        metrics_from_counts(counts),
        int(panel_width),
    )
    return panel_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "da3_cache"
    cache_dir.mkdir(exist_ok=True)
    summary = json.loads((RUN / "summary.json").read_text())
    cameras = load_fixed_camera_index(Path(summary["fixed_cameras_json"]))
    records = scene_records(summary, 3, args.max_frames)
    views, _, _ = build_fixed_cue_views(
        records,
        cameras,
        Path(summary["run_arguments"]["cue_cache_root"]),
        float(summary["resolution"]),
    )
    learned_cue = (
        _load_learned_cue_artifact(args.cue_boundary_json)
        if args.cue_supervision == "learned_stage2_sigmoid_soft"
        else None
    )
    if learned_cue is not None:
        missing = [record.name for record in records if record.name not in learned_cue.boundaries]
        if missing:
            raise KeyError(
                f"learned cue artifact is missing {len(missing)} frames; first={missing[0]}"
            )
    record_by_local = {
        int(Path(record.name).stem[-6:]): (record, view)
        for record, view in zip(records, views)
    }
    trace_rows, trace_arrays = load_trace(args.trace_dir)
    trace_index = {
        str(name): index
        for index, name in enumerate(trace_arrays["frame_names"].astype(str))
    }
    checkpoint = torch.load(
        RUN / "temporal_rchange_checkpoint.pt", map_location="cpu", weights_only=False
    )
    base = GaussianModel(sh_degree=3, active_sh_degree=0)
    base.load_ply(str(checkpoint["base_ply"]))
    for parameter in (
        base._xyz,
        base._features_dc,
        base._features_rest,
        base._opacity,
        base._scaling,
        base._rotation,
    ):
        parameter.requires_grad_(False)
    base_before = snapshot_base(base)
    dc_plus = nn.Parameter(torch.zeros_like(base._features_dc), requires_grad=True)
    dc_minus = nn.Parameter(torch.zeros_like(base._features_dc), requires_grad=True)
    base_optimizer = torch.optim.Adam(
        (dc_plus, dc_minus), lr=float(args.base_lr), eps=1.0e-15
    )
    seeds = NewSeedGaussianModel(
        sh_degree=base.max_sh_degree,
        device=base._features_dc.device,
        dtype=base._features_dc.dtype,
    )
    detector_enabled = args.seed_activation == "never_open_detector"
    seeds.render_never_open_black = detector_enabled
    seed_optimizer = torch.optim.Adam(
        seeds.optimizer_parameter_groups(float(args.seed_lr)),
        lr=float(args.seed_lr),
        eps=1.0e-15,
    )
    occupied: set[tuple[int, int, int]] = (
        _occupied_voxels(base.get_xyz.detach(), float(args.voxel_size))
        if args.occupancy_source == "rchange_and_seed"
        else set()
    )
    initial_occupied_voxels = len(occupied)
    expected_seed_geometry = _seed_geometry_snapshot(
        seeds, include_lifespan=not detector_enabled
    )
    seed_tracker = (
        LifespanGateBetaFilter(
            int(args.max_total_seeds),
            LifespanGateBetaConfig(
                stable_flip_prior=float(args.gate_stable_flip_prior),
                stable_keep_prior=float(args.gate_stable_keep_prior),
                reset_flip_prior=float(args.gate_reset_flip_prior),
                reset_keep_prior=float(args.gate_reset_keep_prior),
                bayes_factor_threshold=float(
                    args.candidate_bayes_factor_threshold
                ),
                min_evidence_mass=float(args.detector_min_evidence_mass),
            ),
            device=base.get_xyz.device,
            dtype=base.get_xyz.dtype,
        )
        if detector_enabled
        else None
    )
    pipe = SimpleNamespace(
        convert_SHs_python=False, compute_cov3D_python=False, debug=False
    )
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    sam = (
        Sam2Model.from_pretrained(args.sam_model, local_files_only=True)
        .half()
        .cuda()
        .eval()
    )
    from depth_anything_3.api import DepthAnything3

    da3 = DepthAnything3.from_pretrained(args.model).to("cuda").eval()
    objects = load_object_annotations(3)
    object004 = next(
        obj for obj in objects if obj["id"] == "inference_base__object_004"
    )
    object010 = next(
        obj
        for obj in objects
        if obj["id"] == "reference_render_base__object_010"
    )
    seed_config = DepthPriorSeedConfig(
        confidence_quantile=float(args.confidence_quantile),
        sampling_stride=int(args.sampling_stride),
        max_seeds=int(args.max_new_seeds_per_frame),
    )
    rows: list[dict[str, Any]] = []
    confusion_panel_paths: list[Path] = []
    replay_entries: list[dict[str, Any]] = []
    all_sampled_view_indices: list[int] = []
    future_training_view_accesses = 0
    started = time.time()

    for local_frame, (record, view) in enumerate(zip(records, views), 1):
        frame_started = time.time()
        trace_row = trace_rows[record.name]
        axis_index = trace_index[record.name]
        cue_tau = None
        cue_width = None
        normalized_cue = None
        if learned_cue is None:
            detector_cue = view.candidate_map
            training_cue = (detector_cue >= float(args.detector_cue_threshold)).float()
            detector_cue_mode = "binary"
        else:
            boundary = learned_cue.boundaries[record.name]
            previous_degree = int(base.active_sh_degree)
            base.active_sh_degree = int(base.max_sh_degree)
            try:
                with torch.no_grad():
                    reference_rgb = render(view, base, pipe, background)["render"]
                    normalized_cue, detector_cue = _calibrate_stage2_cue(
                        cached_sum=view.candidate_map,
                        reference_rgb=reference_rgb,
                        online_rgb=view.original_image,
                        boundary=boundary,
                        l1_exponent=float(args.cue_l1_exponent),
                        edge_probability=learned_cue.edge_probability,
                    )
            finally:
                base.active_sh_degree = previous_degree
            training_cue = detector_cue
            detector_cue_mode = "soft"
            cue_tau = boundary.tau
            cue_width = boundary.width
        delta, _ = extract_delta(view, base, pipe, background, sam)
        axis = torch.from_numpy(trace_arrays["pc1_axes"][axis_index]).to(
            device=delta.device, dtype=delta.dtype
        )
        score64 = (delta @ axis).reshape(64, 64)
        cue64 = F.interpolate(detector_cue[None], (64, 64), mode="area")[0, 0]
        plus64 = score64 > float(trace_row["epsilon_positive"])
        minus64 = score64 < float(trace_row["epsilon_negative"])
        if learned_cue is None:
            active = cue64 >= float(args.detector_cue_threshold)
            plus64 &= active
            minus64 &= active
        new_sign = trace_row["seed_new_sign"] or None
        if new_sign == "+":
            new64 = plus64
        elif new_sign == "-":
            new64 = minus64
        else:
            new64 = torch.zeros_like(plus64)
        depth_row: dict[str, Any] = {
            "attempted": False,
            "scale": None,
            "scale_medrel": None,
            "candidates": 0,
            "accepted": 0,
            "rejected_voxel": 0,
        }
        if new_sign in {"+", "-"} and seeds.num_seeds < int(args.max_total_seeds):
            window_frames = list(
                range(max(1, local_frame - int(args.window) + 1), local_frame + 1)
            )
            images: list[np.ndarray] = []
            extrinsics: list[np.ndarray] = []
            intrinsics: list[np.ndarray] = []
            for frame in window_frames:
                window_record, window_view = record_by_local[frame]
                images.append(
                    _read_resized_rgb(
                        REPO / window_record.image_path,
                        int(window_view.image_height),
                        int(window_view.image_width),
                    )
                )
                window_w2c, window_K = fixed_camera_matrices(
                    window_record, cameras
                )
                extrinsics.append(window_w2c.numpy())
                intrinsics.append(window_K.numpy())
            cache_path = cache_dir / f"frame_{local_frame:06d}.npz"
            if cache_path.exists():
                cache = np.load(cache_path)
                predicted = cache["depth"]
                confidence = cache["confidence"]
                output_K = cache["K"]
            else:
                prediction = da3.inference(
                    images,
                    extrinsics=np.stack(extrinsics),
                    intrinsics=np.stack(intrinsics),
                    align_to_input_ext_scale=True,
                    process_res=int(args.process_res),
                    process_res_method="upper_bound_resize",
                )
                predicted = np.asarray(prediction.depth[-1], dtype=np.float32)
                confidence = np.asarray(prediction.conf[-1], dtype=np.float32)
                output_K = np.asarray(prediction.intrinsics[-1], dtype=np.float32)
                np.savez_compressed(
                    cache_path,
                    depth=predicted,
                    confidence=confidence,
                    K=output_K,
                    window_frames=np.asarray(window_frames),
                )
            height, width = predicted.shape
            new_mask = _resize_tensor(new64.float(), height, width, "nearest").bool()
            cue = _resize_tensor(detector_cue, height, width, "area")
            new_mask &= cue >= float(args.seed_cue_support_threshold)
            w2c, _ = fixed_camera_matrices(record, cameras)
            ref_depth_full, ref_alpha_full = render_reference_depth(
                view, base, w2c, pipe, background
            )
            numerator = _resize_tensor(
                ref_depth_full * ref_alpha_full, height, width, "area"
            )
            alpha = _resize_tensor(ref_alpha_full, height, width, "area")
            reference_depth = numerator / alpha.clamp_min(1.0e-6)
            reference_depth[alpha <= 1.0e-4] = 0.0
            anchors = reference_scale_anchor_mask(
                reference_depth, alpha, cue, config=seed_config
            )
            fit = fit_reference_depth_scale(
                torch.from_numpy(predicted), reference_depth, anchors
            )
            batch = build_depth_prior_new_seeds(
                predicted_depth=torch.from_numpy(predicted),
                confidence=torch.from_numpy(confidence),
                reference_depth=reference_depth,
                new_mask=new_mask,
                K=torch.from_numpy(output_K),
                w2c=w2c,
                scale=fit.scale,
                config=seed_config,
                sampling_priority=(cue if learned_cue is not None else None),
            )
            capacity = int(args.max_total_seeds) - seeds.num_seeds
            accepted = _accept_new_voxels(
                batch.xyz,
                occupied,
                float(args.voxel_size),
                capacity,
            )
            if accepted.numel():
                xyz = batch.xyz[accepted].to(
                    device=base._features_dc.device,
                    dtype=base._features_dc.dtype,
                )
                scaling = batch.log_scaling[accepted].to(
                    device=xyz.device, dtype=xyz.dtype
                )
                metadata = [
                    {
                        "birth_kind": "da3_depth_prior",
                        "frame_local": int(local_frame),
                        "frame_global": int(record.global_index),
                        "frame_name": record.name,
                        "source_sign": new_sign,
                        "depth_scale": fit.scale,
                        "da3_confidence": float(batch.confidence[index].item()),
                        "learned_cue_priority": (
                            None
                            if learned_cue is None
                            else float(
                                cue[
                                    int(batch.pixels_xy[index, 1]),
                                    int(batch.pixels_xy[index, 0]),
                                ].item()
                            )
                        ),
                        "pixel_xy": batch.pixels_xy[index].tolist(),
                        "causal_window": window_frames,
                    }
                    for index in accepted.tolist()
                ]
                seeds.append(
                    xyz=xyz,
                    start=(
                        float("inf")
                        if detector_enabled
                        else float(record.global_index)
                    ),
                    end=float("inf"),
                    scaling=scaling,
                    opacity=float(args.seed_opacity),
                    metadata=metadata,
                    optimizer=seed_optimizer,
                )
                expected_seed_geometry = _seed_geometry_snapshot(
                    seeds, include_lifespan=not detector_enabled
                )
            depth_row = {
                "attempted": True,
                "scale": fit.scale,
                "scale_medrel": fit.median_absolute_relative_error,
                "candidates": batch.count,
                "accepted": int(accepted.numel()),
                "rejected_voxel": batch.count - int(accepted.numel()),
            }

        detector_row = {
            "observed": seeds.num_seeds,
            "candidate": 0,
            "committed": 0,
            "opened": 0,
            "closed": 0,
            "active": int(
                seeds.active_mask(float(record.global_index)).sum().item()
            ),
            "never_open": 0,
            "closed_total": 0,
        }
        if detector_enabled and seeds.num_seeds:
            if seed_tracker is None:
                raise RuntimeError("NEVER_OPEN seed detector was not initialized")
            detector_probe = build_concatenated_change_view(
                base,
                seeds,
                timestamp=float(record.global_index),
                base_dc=torch.zeros_like(base._features_dc),
                detach_base_dc=True,
                seed_attribute_mode="all",
            )
            detector_evidence = accumulate_change_evidence(
                view,
                detector_probe,
                pipe,
                background,
                detector_cue,
                cue_mode=detector_cue_mode,
                cue_threshold=float(args.detector_cue_threshold),
                cue_scale=1.0,
                count_mode="capped",
                mass_saturation=float(args.detector_mass_saturation),
                min_evidence_mass=float(args.detector_min_evidence_mass),
                probe_scaling_mode="native",
            )
            seed_offset = int(base.get_xyz.shape[0])
            detector_row = _apply_seed_detector_update(
                tracker=seed_tracker,
                seeds=seeds,
                delta_a=detector_evidence.delta_a[seed_offset:],
                delta_b=detector_evidence.delta_b[seed_offset:],
                total_mass=detector_evidence.total_mass[seed_offset:],
                timestamp=int(record.global_index),
            )
            del detector_probe, detector_evidence

        target = (
            F.interpolate(
                new64.float()[None, None],
                (int(view.image_height), int(view.image_width)),
                mode="nearest",
            )[0]
            * training_cue
        )
        replay_entries.append(
            {
                "view": view,
                "plus64": plus64.detach(),
                "minus64": minus64.detach(),
                "new64": new64.detach(),
                "training_cue": training_cue.detach(),
            }
        )

        # Representation optimization happens only after the current cue
        # has proposed geometry and updated every seed's causal detector once.
        # Replay samples only already-observed entries and always renders the
        # current committed seed lifecycle into the sampled historical camera.
        current_training_index = len(replay_entries) - 1
        if args.training_schedule == "current_exact":
            sampled_view_indices = [
                current_training_index
            ] * int(args.updates_per_frame)
            base_training = train_current_signed_masks(
                view,
                base,
                dc_plus,
                dc_minus,
                base_optimizer,
                plus64,
                minus64,
                pipe,
                background,
                int(args.updates_per_frame),
                cue_target=training_cue,
            )
            seed_training = train_seed_dc_from_projected_coverage(
                view=view,
                seeds=seeds,
                optimizer=seed_optimizer,
                target=target,
                updates=int(args.updates_per_frame),
                loss_weight=1.0,
            )
        else:
            sampled_view_indices = []
            base_training = None
            seed_training = None
            first_seed_gradient = None
            for update_index in range(int(args.updates_per_frame)):
                sampled_index = deterministic_training_view_index(
                    current_training_index,
                    update_index,
                    seed=int(args.training_seed),
                )
                if sampled_index > current_training_index:
                    future_training_view_accesses += 1
                    raise RuntimeError("future training view accessed")
                sampled_view_indices.append(sampled_index)
                replay = replay_entries[sampled_index]
                replay_view = replay["view"]
                base_training = train_current_signed_masks(
                    replay_view,
                    base,
                    dc_plus,
                    dc_minus,
                    base_optimizer,
                    replay["plus64"],
                    replay["minus64"],
                    pipe,
                    background,
                    1,
                    cue_target=replay["training_cue"],
                )
                replay_target = (
                    F.interpolate(
                        replay["new64"].float()[None, None],
                        (
                            int(replay_view.image_height),
                            int(replay_view.image_width),
                        ),
                        mode="nearest",
                    )[0]
                    * replay["training_cue"]
                )
                seed_training = train_seed_dc_from_projected_coverage(
                    view=replay_view,
                    seeds=seeds,
                    optimizer=seed_optimizer,
                    target=replay_target,
                    updates=1,
                    loss_weight=1.0,
                    active_timestamp=float(record.global_index),
                )
                if (
                    first_seed_gradient is None
                    and seed_training["coverage_gradient_norm"] is not None
                ):
                    first_seed_gradient = seed_training["coverage_gradient_norm"]
            if base_training is None or seed_training is None:
                raise RuntimeError("empty O-SCD replay update schedule")
            seed_training = dict(seed_training)
            seed_training["coverage_gradient_norm"] = first_seed_gradient
        all_sampled_view_indices.extend(sampled_view_indices)

        # Evaluation-only masks enter after every causal birth and update.
        _, added64, removed64, appearance64 = annotation_masks64(
            record.name, objects
        )
        gt64 = {
            "all": added64 | removed64 | appearance64,
            "new": added64,
            "remove": removed64,
        }
        height_full, width_full = int(view.image_height), int(view.image_width)
        added_full, removed_full, appearance_full = annotation_masks_at_size(
            record.name, objects, height=height_full, width=width_full
        )
        gt_full = added_full | removed_full | appearance_full
        gt_scopes = {"new_full": added_full, "remove_full": removed_full}
        baseline_metrics = evaluate_branch_change(
            view,
            base,
            dc_plus,
            dc_minus,
            pipe,
            background,
            gt64,
            gt_full=gt_full,
            gt_full_scopes=gt_scopes,
            seeds=None,
            new_sign=new_sign,
        )
        da3_metrics = evaluate_branch_change(
            view,
            base,
            dc_plus,
            dc_minus,
            pipe,
            background,
            gt64,
            gt_full=gt_full,
            gt_full_scopes=gt_scopes,
            seeds=seeds,
            new_sign=new_sign,
        )
        new_score = render_new_sidecar_learned_score(
            view, seeds, pipe, background
        )[0].detach().cpu().numpy()
        object004_mask = cv2.resize(
            cv2.imread(
                str(object004["paths"][record.name]), cv2.IMREAD_GRAYSCALE
            ),
            (width_full, height_full),
            interpolation=cv2.INTER_NEAREST,
        ) > 0
        object010_mask = cv2.resize(
            cv2.imread(
                str(object010["paths"][record.name]), cv2.IMREAD_GRAYSCALE
            ),
            (width_full, height_full),
            interpolation=cv2.INTER_NEAREST,
        ) > 0
        object004_metric = binary_metrics(new_score >= 0.5, object004_mask)
        object010_leak = int(((new_score >= 0.5) & object010_mask).sum())
        row: dict[str, Any] = {
            "frame_local": local_frame,
            "frame_global": int(record.global_index),
            "frame_name": record.name,
            "new_sign": new_sign,
            "new_target_pixels": int(target.sum().item()),
            "new_target_mass": float(target.sum().item()),
            "new_target_nonzero_pixels": int((target > 0.0).sum().item()),
            "new_birth_support_pixels": int(
                (
                    F.interpolate(
                        new64.float()[None, None],
                        (int(view.image_height), int(view.image_width)),
                        mode="nearest",
                    )[0]
                    * (detector_cue >= float(args.seed_cue_support_threshold))
                ).sum().item()
            ),
            "cue_tau": cue_tau,
            "cue_width": cue_width,
            "normalized_cue_mean": (
                None if normalized_cue is None else float(normalized_cue.mean().item())
            ),
            "detector_cue_mean": float(detector_cue.mean().item()),
            "detector_cue_support_pixels": int(
                (detector_cue >= float(args.seed_cue_support_threshold)).sum().item()
            ),
            "seed_count": seeds.num_seeds,
            "seed_birth_attempted": depth_row["attempted"],
            "seed_candidates": depth_row["candidates"],
            "seed_accepted": depth_row["accepted"],
            "seed_rejected_voxel": depth_row["rejected_voxel"],
            "occupied_voxels": len(occupied),
            "seed_active_count": detector_row["active"],
            "seed_never_open_count": detector_row["never_open"],
            "seed_closed_count": detector_row["closed_total"],
            "seed_detector_observed": detector_row["observed"],
            "seed_detector_candidate": detector_row["candidate"],
            "seed_detector_committed": detector_row["committed"],
            "seed_opened": detector_row["opened"],
            "seed_closed": detector_row["closed"],
            "training_schedule": args.training_schedule,
            "training_sampled_current": int(
                sum(index == current_training_index for index in sampled_view_indices)
            ),
            "training_sampled_history": int(
                sum(index != current_training_index for index in sampled_view_indices)
            ),
            "training_sampled_view_indices": json.dumps(sampled_view_indices),
            "depth_scale": depth_row["scale"],
            "depth_scale_medrel": depth_row["scale_medrel"],
            "base_training_last_loss": base_training["last"]["combined"],
            "seed_training_last_loss": (
                None
                if seed_training["last"] is None
                else seed_training["last"]["loss"]
            ),
            "seed_dc_gradient_norm": seed_training["coverage_gradient_norm"],
            "object004_new_iou": object004_metric["iou"],
            "object004_new_precision": object004_metric["precision"],
            "object004_new_recall": object004_metric["recall"],
            "object004_gt_pixels": object004_metric["gt_positive"],
            "object004_active_frame": object004_metric["gt_positive"] > 0,
            "object010_new_false_positive_pixels": object010_leak,
            "frame_runtime_seconds": time.time() - frame_started,
        }
        _flatten_metrics(row, "baseline", baseline_metrics)
        _flatten_metrics(row, "da3", da3_metrics)
        rows.append(row)
        method_scores: tuple[
            tuple[np.ndarray, np.ndarray, np.ndarray],
            tuple[np.ndarray, np.ndarray, np.ndarray],
        ] | None = None
        if local_frame in VISUAL_FRAMES or args.save_confusion_gif:
            baseline_scores = _method_scores(
                view,
                base,
                dc_plus,
                dc_minus,
                None,
                new_sign,
                pipe,
                background,
            )
            da3_scores = _method_scores(
                view,
                base,
                dc_plus,
                dc_minus,
                seeds,
                new_sign,
                pipe,
                background,
            )
            method_scores = (baseline_scores, da3_scores)
        if local_frame in VISUAL_FRAMES:
            if method_scores is None:
                raise RuntimeError("selected-frame scores were not rendered")
            baseline_scores, da3_scores = method_scores
            rgb = (
                view.original_image[:3]
                .detach()
                .permute(1, 2, 0)
                .cpu()
                .numpy()
                * 255.0
            )
            new_mask_full = (
                target[0].detach().cpu().numpy()
                >= float(args.seed_cue_support_threshold)
            )
            _save_visual(
                args.output_dir / f"frame_{local_frame:06d}_dc_only.png",
                rgb,
                new_mask_full,
                baseline_scores,
                da3_scores,
                object004_mask,
                object010_mask,
                row,
            )
        if args.save_confusion_gif:
            if method_scores is None:
                raise RuntimeError("confusion scores were not rendered")
            _, da3_scores = method_scores
            confusion_panel_paths.append(
                _save_temporal_confusion_frame(
                    args.output_dir / "temporal_confusion",
                    rgb=view.original_image[:3],
                    gt=gt_full,
                    score=da3_scores[0],
                    row=row,
                    panel_width=int(args.confusion_panel_width),
                )
            )
        print(
            f"[{local_frame:03d}/{len(records):03d}] seeds={seeds.num_seeds} "
            f"active={detector_row['active']} birth={depth_row['accepted']} "
            f"open={detector_row['opened']} close={detector_row['closed']} "
            f"full IoU={row['da3_full_iou']:.4f} "
            f"NEW IoU={row['da3_new_full_iou']:.4f} runtime={row['frame_runtime_seconds']:.2f}s",
            flush=True,
        )

    with (args.output_dir / "frame_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_active_seed_ply(
        seeds, float(records[-1].global_index), args.output_dir / "da3_new_seeds_dc_only.ply"
    )
    confusion_gif_path: Path | None = None
    if confusion_panel_paths:
        confusion_gif_path = (
            args.output_dir
            / "temporal_confusion"
            / "scene_change3_confusion.gif"
        )
        save_gif(
            confusion_panel_paths,
            confusion_gif_path,
            int(args.confusion_gif_width),
            int(args.confusion_gif_duration_ms),
        )
    torch.save(
        {
            "contract": (
                (
                    "fixed_da3_never_open_learned_stage2_sigmoid_soft_cue_seed_dc_only"
                    if learned_cue is not None
                    else "fixed_da3_never_open_raw_cue_detector_seed_dc_only"
                )
                if detector_enabled
                else "fixed_da3_depth_geometry_seed_dc_only"
            ),
            "seed_sidecar": seeds.to_checkpoint(),
            "seed_detector": (
                None
                if seed_tracker is None
                else {
                    name: value.detach().cpu()
                    for name, value in seed_tracker.state_dict().items()
                }
            ),
            "dc_plus": dc_plus.detach().cpu(),
            "dc_minus": dc_minus.detach().cpu(),
            "configuration": vars(args),
        },
        args.output_dir / "checkpoint.pt",
    )
    seed_geometry_audit = _audit_seed_geometry(seeds, expected_seed_geometry)
    base_audit = audit_base(base, base_before)
    metrics = {
        branch: {
            scope: _aggregate(rows, branch, scope)
            for scope in ("full", "new_full", "remove_full")
        }
        for branch in ("baseline", "da3")
    }
    report = {
        "experiment": "SC3 fixed DA3 depth-prior NEW geometry with DC-only learning",
        "frames": len(rows),
        "updates_per_frame": int(args.updates_per_frame),
        "runtime_seconds": time.time() - started,
        "configuration": vars(args),
        "causal_contract": {
            "past_only_da3_window": True,
            "future_frames_used": False,
            "saved_causal_pca_sign_trace_reused": True,
            "ground_truth_used_after_birth_and_training_only": True,
            "seed_detector_uses_pre_optimization_cue_once": detector_enabled,
            "seed_detector_uses_pre_optimization_raw_cue_once": (
                detector_enabled and learned_cue is None
            ),
            "learned_cue_is_raw_derived": learned_cue is not None,
            "boundary_artifact_was_fit_posthoc_on_complete_stream": learned_cue is not None,
            "learned_seed_dc_used_as_detector_input": False,
        },
        "cue_contract": {
            "supervision": args.cue_supervision,
            "boundary_json": (
                None if args.cue_boundary_json is None else str(args.cue_boundary_json)
            ),
            "formula": (
                "cached P+S >= 0.5"
                if learned_cue is None
                else learned_cue.cue_formula
            ),
            "l1_exponent": (
                None if learned_cue is None else float(args.cue_l1_exponent)
            ),
            "detector_mode": "binary" if learned_cue is None else "soft",
            "detector_binary_threshold": (
                float(args.detector_cue_threshold) if learned_cue is None else None
            ),
            "seed_birth_support_threshold": float(
                args.seed_cue_support_threshold
            ),
            "seed_sampling_priority": (
                "DA3 confidence"
                if learned_cue is None
                else "learned Q first, DA3 confidence as stable tie-break"
            ),
            "sign_contract": (
                "hard cue support plus PCA sign"
                if learned_cue is None
                else "PCA sign with continuous learned Q confidence weight"
            ),
            "dc_target": (
                "binary"
                if learned_cue is None
                else "continuous learned sigmoid Q in [0,1]"
            ),
            "edge_probability": (
                None if learned_cue is None else learned_cue.edge_probability
            ),
            "artifact_cue_scale": (
                None if learned_cue is None else learned_cue.cue_scale
            ),
            "used_tau": _summarize_optional(rows, "cue_tau"),
            "used_width": _summarize_optional(rows, "cue_width"),
            "output_mask_threshold": 0.5,
        },
        "training_schedule": {
            "mode": args.training_schedule,
            "updates_per_arrival": int(args.updates_per_frame),
            "seed": int(args.training_seed),
            "current_view_branch_probability": (
                1.0 if args.training_schedule == "current_exact" else 0.33
            ),
            "causal_uniform_branch_probability": (
                0.0 if args.training_schedule == "current_exact" else 0.67
            ),
            "total_updates": len(all_sampled_view_indices),
            "sampled_current": int(
                sum(
                    int(row["training_sampled_current"])
                    for row in rows
                )
            ),
            "sampled_history": int(
                sum(
                    int(row["training_sampled_history"])
                    for row in rows
                )
            ),
            "sampled_indices_sha256": hashlib.sha256(
                np.asarray(all_sampled_view_indices, dtype=np.int64).tobytes()
            ).hexdigest(),
            "future_view_accesses": future_training_view_accesses,
            "seed_lifecycle_render_timestamp": "current arrival timestamp",
        },
        "optimization_contract": {
            "reference_trainable": False,
            "base_trainable_parameters": ["dc_plus", "dc_minus"],
            "seed_trainable_parameters": ["seed_dc"],
            "seed_geometry_trainable": False,
            "seed_optimizer_parameter_groups": [
                group.get("name") for group in seed_optimizer.param_groups
            ],
            "seed_dc_nonzero_rows": int(
                (seeds.seed_dc.detach().abs().amax(dim=(1, 2)) > 0).sum().item()
            ),
        },
        "birth": {
            "final_seed_count": seeds.num_seeds,
            "occupancy_source": args.occupancy_source,
            "initial_rchange_occupied_voxels": initial_occupied_voxels,
            "final_occupied_voxels": len(occupied),
            "total_candidates": int(sum(row["seed_candidates"] for row in rows)),
            "total_accepted": int(sum(row["seed_accepted"] for row in rows)),
            "total_rejected_voxel": int(
                sum(row["seed_rejected_voxel"] for row in rows)
            ),
        },
        "seed_lifecycle": {
            "activation_mode": args.seed_activation,
            "final_active": int(rows[-1]["seed_active_count"]),
            "final_never_open": int(rows[-1]["seed_never_open_count"]),
            "final_closed": int(rows[-1]["seed_closed_count"]),
            "total_open": int(sum(row["seed_opened"] for row in rows)),
            "total_close": int(sum(row["seed_closed"] for row in rows)),
            "total_commits": int(
                sum(row["seed_detector_committed"] for row in rows)
            ),
            "bayes_factor_threshold": (
                float(args.candidate_bayes_factor_threshold)
                if detector_enabled
                else None
            ),
            "never_open_rendering": (
                "black full-opacity occluder"
                if detector_enabled
                else "not applicable"
            ),
        },
        "metrics": metrics,
        "delta_da3_minus_baseline": {
            scope: {
                name: metrics["da3"][scope][name]
                - metrics["baseline"][scope][name]
                for name in ("precision", "recall", "iou", "f1", "mean_frame_iou", "mean_frame_f1")
            }
            for scope in ("full", "new_full", "remove_full")
        },
        "object004": {},
        "object010": {
            "new_sidecar_false_positive_pixels": int(
                sum(row["object010_new_false_positive_pixels"] for row in rows)
            )
        },
        "confusion_visualization": {
            "enabled": bool(args.save_confusion_gif),
            "panels": len(confusion_panel_paths),
            "gif_path": (
                None if confusion_gif_path is None else str(confusion_gif_path)
            ),
            "threshold": 0.5,
            "palette": PALETTE,
        },
        "audits": {
            "reference": base_audit,
            "seed_geometry": seed_geometry_audit,
            "future_view_accesses": 0,
            "future_training_view_accesses": future_training_view_accesses,
            "ground_truth_birth_accesses": 0,
        },
    }
    object004_rows = [row for row in rows if row["object004_active_frame"]]
    report["object004"] = {
        "active_frames": len(object004_rows),
        "first_active_frame": (
            object004_rows[0]["frame_local"] if object004_rows else None
        ),
        "last_active_frame": (
            object004_rows[-1]["frame_local"] if object004_rows else None
        ),
        "active_frame_mean_iou": float(
            np.mean([row["object004_new_iou"] for row in object004_rows])
        )
        if object004_rows
        else 0.0,
        "active_frame_mean_full_image_precision": float(
            np.mean([row["object004_new_precision"] for row in object004_rows])
        )
        if object004_rows
        else 0.0,
        "active_frame_mean_recall": float(
            np.mean([row["object004_new_recall"] for row in object004_rows])
        )
        if object004_rows
        else 0.0,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"birth": report["birth"], "metrics": metrics, "object004": report["object004"], "object010": report["object010"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
