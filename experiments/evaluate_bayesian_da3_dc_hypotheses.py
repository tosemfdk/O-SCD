"""Run one controlled DC-learning hypothesis condition on the final viewer pipeline.

The detector cue, BF30 lifecycle, DA3 proposal artifact, seed detector, geometry
loss, geometry replay, optimizer budget, and random seed stay fixed.  Conditions
change only one of three DC-specific factors:

* unit-Q versus restored 0..2 O-SCD target amplitude;
* joint base+seed versus base-free seed-only SSF gradient for DA3 DC;
* lifespan-correct historical versus current-only DC supervision.

Every frame also renders two white-DC counterfactuals.  The first preserves the
black NEVER_OPEN occluders and is the maximum score attainable by the current
OPEN population.  The second removes those occluders and isolates their
occlusion penalty.  Ground truth is read only after the causal step finishes.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
from PIL import Image
import torch

from experiments.view_bayesian_detector_steps import (
    BayesianDetectorReplay,
    causal_training_view_index,
    parse_args as parse_viewer_args,
    representation_dc_target,
)
from temporal.fusion import compute_ssf_loss
from utils.sh_utils import SH2RGB


DEFAULT_BOUNDARIES = Path(
    "outputs/stage2_l1_power_sigmoid_boundary_causal_prequential/"
    "learned_boundaries_causal.json"
)
DEFAULT_DA3_SEEDS = Path(
    "outputs/causal_da3metric_scene123_panel7pos010_depthpos003_"
    "dynamiccoverage_20260904/da3_seed_replay.pt"
)
DEFAULT_SAM_TRACE = Path(
    "outputs/e4_online_xfeat_new_seed_scene123_120_final_corrected_20260814"
)
DEFAULT_OUTPUT_ROOT = Path("outputs/bayesian_da3_historical_replay_20260906")
DIAGNOSTIC_FRAMES = frozenset((13, 94, 95, 198, 199, 303))


@dataclass(frozen=True)
class Condition:
    name: str
    cue_amplitude: float
    seed_dc_supervision: str
    dc_replay_mode: str
    purpose: str


CONDITIONS: dict[str, Condition] = {
    condition.name: condition
    for condition in (
        Condition(
            "A_baseline",
            1.0,
            "joint",
            "sampled",
            "Unit Q with joint DC loss and lifespan-correct historical replay.",
        ),
        Condition(
            "B_h1_amplitude2",
            2.0,
            "joint",
            "sampled",
            "Restore O-SCD 0..2 DC amplitude under historical lifespan replay.",
        ),
        Condition(
            "C_h2_seed_local",
            1.0,
            "projected_bce",
            "sampled",
            "H2 only: replace DA3 joint DC gradient with projected seed-local BCE.",
        ),
        Condition(
            "D_h4_current_dc",
            1.0,
            "joint",
            "current",
            "H4 only: keep geometry replay but train DC on the current view.",
        ),
        Condition(
            "E_h1_h2",
            2.0,
            "projected_bce",
            "sampled",
            "H1+H2 interaction under historical lifespan replay.",
        ),
        Condition(
            "F_h1_h4",
            2.0,
            "joint",
            "current",
            "H1+H4 interaction with joint base+seed DC.",
        ),
        Condition(
            "G_all",
            2.0,
            "projected_bce",
            "current",
            "Combined H1+H2+H4 intervention.",
        ),
        Condition(
            "N_seed_only_ssf_renderer_control",
            1.0,
            "seed_only_ssf",
            "sampled",
            "Negative control: naive small-bank FastGS seed-only SSF render.",
        ),
        Condition(
            "R_baseline_repeat",
            1.0,
            "joint",
            "sampled",
            "Exact baseline repeat used to measure CUDA/dynamic-topology run noise.",
        ),
    )
}


def binary_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(prediction, dtype=bool)
    gt = np.asarray(target, dtype=bool)
    if pred.shape != gt.shape:
        raise ValueError(f"prediction/target shape mismatch: {pred.shape} vs {gt.shape}")
    tp = int(np.logical_and(pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    union = tp + fp + fn
    f1_denominator = 2 * tp + fp + fn
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "iou": tp / union if union else 0.0,
        "f1": 2 * tp / f1_denominator if f1_denominator else 0.0,
    }


def aggregate_binary(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    if not rows:
        return {"frames": 0}
    tp = sum(int(row[f"{prefix}_tp"]) for row in rows)
    tn = sum(int(row[f"{prefix}_tn"]) for row in rows)
    fp = sum(int(row[f"{prefix}_fp"]) for row in rows)
    fn = sum(int(row[f"{prefix}_fn"]) for row in rows)
    return {
        "frames": len(rows),
        "mean_frame_iou": float(np.mean([row[f"{prefix}_iou"] for row in rows])),
        "mean_frame_f1": float(np.mean([row[f"{prefix}_f1"] for row in rows])),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "aggregate_iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
        "aggregate_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def aggregate_cue_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"frames": 0}

    def ratio(numerator: str, denominator: str) -> float:
        num = sum(float(row[numerator]) for row in rows)
        den = sum(float(row[denominator]) for row in rows)
        return num / den if den else 0.0

    high_pixels = sum(int(row["cue_high_pixels"]) for row in rows)
    low_pixels = sum(int(row["cue_low_pixels"]) for row in rows)
    all_pixels = sum(int(row["pixels"]) for row in rows)
    return {
        "frames": len(rows),
        "pixels": all_pixels,
        "high_cue_pixels": high_pixels,
        "low_cue_pixels": low_pixels,
        "learned_mean_on_high_cue": ratio("learned_high_sum", "cue_high_pixels"),
        "learned_positive_fraction_on_high_cue": ratio(
            "learned_high_positive", "cue_high_pixels"
        ),
        "learned_mean_on_low_cue": ratio("learned_low_sum", "cue_low_pixels"),
        "mean_absolute_error_to_q": ratio("cue_absolute_error_sum", "pixels"),
        "white_ceiling_mean_on_high_cue": ratio(
            "white_black_high_sum", "cue_high_pixels"
        ),
        "white_ceiling_positive_fraction_on_high_cue": ratio(
            "white_black_high_positive", "cue_high_pixels"
        ),
        "open_only_ceiling_mean_on_high_cue": ratio(
            "white_open_only_high_sum", "cue_high_pixels"
        ),
        "coverage_limited_fraction_on_high_cue": ratio(
            "white_black_high_below_threshold", "cue_high_pixels"
        ),
        "black_occlusion_mean_penalty_on_high_cue": ratio(
            "black_occlusion_high_sum", "cue_high_pixels"
        ),
        "learned_to_white_ceiling_mass_ratio_on_high_cue": ratio(
            "learned_high_sum", "white_black_high_sum"
        ),
        "gt_positive_coverage_limited_fraction": ratio(
            "gt_white_black_below_threshold", "gt_positive_pixels"
        ),
        "actual_dc_current_view_fraction": ratio(
            "dc_current_updates", "representation_updates"
        ),
        "mean_dc_view_age": ratio("dc_view_age_sum", "representation_updates"),
    }


def cue_diagnostics(
    learned: np.ndarray,
    cue: np.ndarray,
    white_black: np.ndarray,
    white_open_only: np.ndarray,
    gt: np.ndarray,
) -> dict[str, Any]:
    learned = np.asarray(learned, dtype=np.float64)
    cue = np.asarray(cue, dtype=np.float64)
    white_black = np.asarray(white_black, dtype=np.float64)
    white_open_only = np.asarray(white_open_only, dtype=np.float64)
    gt = np.asarray(gt, dtype=bool)
    if not (
        learned.shape == cue.shape == white_black.shape == white_open_only.shape == gt.shape
    ):
        raise ValueError("cue diagnostic arrays must share shape")
    high = cue >= 0.8
    low = cue <= 0.2
    black_penalty = np.maximum(white_open_only - white_black, 0.0)
    return {
        "pixels": int(cue.size),
        "cue_high_pixels": int(high.sum()),
        "cue_low_pixels": int(low.sum()),
        "gt_positive_pixels": int(gt.sum()),
        "learned_high_sum": float(learned[high].sum()),
        "learned_high_positive": int((learned[high] >= 0.5).sum()),
        "learned_low_sum": float(learned[low].sum()),
        "cue_absolute_error_sum": float(np.abs(learned - cue).sum()),
        "white_black_high_sum": float(white_black[high].sum()),
        "white_black_high_positive": int((white_black[high] >= 0.5).sum()),
        "white_black_high_below_threshold": int((white_black[high] < 0.5).sum()),
        "white_open_only_high_sum": float(white_open_only[high].sum()),
        "black_occlusion_high_sum": float(black_penalty[high].sum()),
        "gt_white_black_below_threshold": int(
            np.logical_and(gt, white_black < 0.5).sum()
        ),
    }


def _tensor_hash(hasher: Any, tensor: torch.Tensor) -> None:
    array = tensor.detach().cpu().contiguous().numpy()
    hasher.update(str(array.dtype).encode("ascii"))
    hasher.update(str(array.shape).encode("ascii"))
    hasher.update(array.tobytes())


def invariant_hashes(replay: BayesianDetectorReplay) -> dict[str, str]:
    reference = hashlib.sha256()
    for tensor in (
        replay.base._xyz,
        replay.base._features_dc,
        replay.base._features_rest,
        replay.base._opacity,
        replay.base._scaling,
        replay.base._rotation,
    ):
        _tensor_hash(reference, tensor)

    base = hashlib.sha256()
    for tensor in (
        replay.lifecycle.current_state_index,
        replay.lifecycle.num_states,
        replay.lifecycle.open_timestamp,
        replay.lifecycle.materialized_timestamp,
        replay.lifecycle.state_start,
        replay.lifecycle.state_end,
        replay.lifecycle.state_valid,
    ):
        _tensor_hash(base, tensor)

    seed_discrete = hashlib.sha256()
    for tensor in (
        replay.seed_model.start,
        replay.seed_model.end,
        replay.seed_lifecycle.current_state_index,
        replay.seed_lifecycle.num_states,
        replay.seed_lifecycle.materialized_timestamp,
        replay.seed_lifecycle.state_start,
        replay.seed_lifecycle.state_end,
        replay.seed_lifecycle.state_valid,
        replay.seed_geometry_update_counts,
    ):
        _tensor_hash(seed_discrete, tensor)
    seed_discrete.update(
        np.asarray(replay.accepted_da3_source_rows, dtype=np.int64).tobytes()
    )
    seed_discrete.update(
        np.asarray(replay.accepted_da3_birth_global, dtype=np.int64).tobytes()
    )

    seed_geometry = hashlib.sha256()
    for tensor in (
        replay.seed_model._xyz,
        replay.seed_model._opacity,
        replay.seed_model._scaling,
        replay.seed_model._rotation,
    ):
        _tensor_hash(seed_geometry, tensor)
    return {
        "immutable_reference_sha256": reference.hexdigest(),
        "base_detector_lifecycle_sha256": base.hexdigest(),
        "seed_discrete_topology_lifecycle_sha256": seed_discrete.hexdigest(),
        "seed_geometry_sha256": seed_geometry.hexdigest(),
    }


def intrinsic_dc_stats(replay: BayesianDetectorReplay) -> dict[str, Any]:
    def summarize(dc: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
        rows = int(mask.sum().item())
        if rows == 0:
            return {
                "rows": 0,
                "rgb_mean": None,
                "rgb_fraction_ge_0_5": None,
                "rgb_fraction_ge_0_8": None,
            }
        rgb = SH2RGB(dc[mask]).reshape(rows, 3)
        return {
            "rows": rows,
            "rgb_mean": float(rgb.mean().item()),
            "rgb_channel_mean": [float(value) for value in rgb.mean(dim=0).tolist()],
            "rgb_fraction_ge_0_5": float((rgb >= 0.5).float().mean().item()),
            "rgb_fraction_ge_0_8": float((rgb >= 0.8).float().mean().item()),
        }

    base_open = replay.lifecycle.current_state_index >= 0
    seed_open = replay.seed_lifecycle.active_mask(replay.current_index)
    return {
        "base_open": summarize(replay.change_dc.detach(), base_open),
        "seed_open": summarize(replay.seed_model.new_dc.detach(), seed_open),
    }


def dc_gradient_stats(replay: BayesianDetectorReplay) -> dict[str, Any]:
    def summarize(
        gradient: torch.Tensor | None, mask: torch.Tensor
    ) -> dict[str, Any]:
        if gradient is None or not bool(mask.any()):
            return {
                "nonzero_rows": 0,
                "absolute_mean": 0.0,
                "whitening_fraction": 0.0,
                "blackening_fraction": 0.0,
            }
        flat = gradient.detach().reshape(gradient.shape[0], -1)
        nonzero = mask & (flat.abs().sum(dim=1) > 0)
        if not bool(nonzero.any()):
            return {
                "nonzero_rows": 0,
                "absolute_mean": 0.0,
                "whitening_fraction": 0.0,
                "blackening_fraction": 0.0,
            }
        selected = flat[nonzero]
        direction = selected.mean(dim=1)
        return {
            "nonzero_rows": int(nonzero.sum().item()),
            "absolute_mean": float(selected.abs().mean().item()),
            # Adam subtracts the gradient, so negative means increasing DC/white.
            "whitening_fraction": float((direction < 0).float().mean().item()),
            "blackening_fraction": float((direction > 0).float().mean().item()),
        }

    return {
        "base": summarize(
            replay.change_dc.grad,
            replay.lifecycle.current_state_index >= 0,
        ),
        "seed": summarize(
            replay.seed_model.new_dc.grad,
            replay.seed_lifecycle.active_mask(replay.current_index),
        ),
    }


@torch.no_grad()
def render_white_counterfactuals(
    replay: BayesianDetectorReplay,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render current OPEN rows white, with and without NEVER_OPEN occluders."""

    from gaussian_renderer import render_change
    from temporal.new_seed_gaussians import build_concatenated_change_view

    base_dc, base_opacity, base_open = replay._effective_base_change_attributes()
    base_never = replay.lifecycle.never_open_mask(replay.current_index)
    model: Any = replay.base
    colors = torch.zeros(
        (replay.count, 3),
        device=replay.device,
        dtype=replay.base.get_xyz.dtype,
    )
    colors[base_open] = 1.0
    opacity = base_opacity.detach().clone()
    override_xyz: torch.Tensor | None = replay.representation_xyz.detach()
    override_scaling: torch.Tensor | None = replay.base.scaling_activation(
        replay.representation_scaling.detach()
    )
    override_rotation: torch.Tensor | None = replay.base.rotation_activation(
        replay.representation_rotation.detach()
    )
    selected_seed_never = torch.empty(
        0, device=replay.device, dtype=torch.bool
    )
    if replay.seed_model.num_gaussians:
        seed_active, seed_never_open = replay._seed_lifecycle_masks(
            replay.current_index
        )
        model = build_concatenated_change_view(
            replay.base,
            replay.seed_model,
            timestamp=float(replay.current_index),
            base_dc=base_dc,
            base_opacity=base_opacity,
            detach_base_dc=True,
            seed_attribute_mode="lifecycle",
            seed_lifecycle_active=seed_active,
            seed_lifecycle_never_open=seed_never_open,
        )
        selected = model.seed_active_mask
        selected_seed_active = seed_active[selected]
        selected_seed_never = seed_never_open[selected]
        seed_colors = torch.zeros(
            (int(selected.sum().item()), 3),
            device=replay.device,
            dtype=colors.dtype,
        )
        seed_colors[selected_seed_active] = 1.0
        colors = torch.cat((colors, seed_colors), dim=0)
        opacity = model.get_opacity.detach().clone()
        override_xyz = None
        override_scaling = None
        override_rotation = None

    common = dict(
        viewpoint_camera=replay.current_view,
        pc=model,
        pipe=replay.pipe,
        bg_color=replay.evidence_background,
        override_color=colors,
        override_xyz=override_xyz,
        override_scaling=override_scaling,
        override_rotation=override_rotation,
        clamp_output=False,
    )
    with_black = render_change(override_opacity=opacity, **common)["render"]
    open_only_opacity = opacity.clone()
    open_only_opacity[: replay.count][base_never] = 0.0
    if selected_seed_never.numel():
        open_only_opacity[replay.count :][selected_seed_never] = 0.0
    open_only = render_change(
        override_opacity=open_only_opacity,
        **common,
    )["render"]
    return (
        with_black.mean(dim=0).clamp(0.0, 1.0),
        open_only.mean(dim=0).clamp(0.0, 1.0),
    )


def _dc_audit_loss(replay: BayesianDetectorReplay, condition: Condition) -> torch.Tensor:
    from gaussian_renderer import render_change
    from temporal.active_new_gaussians import ActiveNewGeometryView
    from temporal.new_seed_gaussians import build_concatenated_change_view

    item = replay.representation_replay[-1]
    target = representation_dc_target(
        item.cue_target,
        amplitude=condition.cue_amplitude,
    )
    if condition.seed_dc_supervision == "projected_bce":
        from experiments.run_online_xfeat_new_seed import (
            new_sidecar_projected_dc_loss,
        )

        return new_sidecar_projected_dc_loss(
            replay.seed_model,
            item.view,
            item.cue_target,
            timestamp=replay.current_index,
        )[0]
    if condition.seed_dc_supervision == "seed_only_ssf":
        model = ActiveNewGeometryView(
            replay.seed_model,
            float(replay.current_index),
            detach_geometry=True,
            detach_opacity=True,
        )
        rendered = render_change(
            item.view,
            model,
            replay.pipe,
            replay.evidence_background,
            clamp_output=False,
        )["render"]
    else:
        base_dc, base_opacity, _ = replay._effective_base_change_attributes()
        seed_active, seed_never_open = replay._seed_lifecycle_masks(
            replay.current_index
        )
        model = build_concatenated_change_view(
            replay.base,
            replay.seed_model,
            timestamp=float(replay.current_index),
            base_dc=base_dc,
            base_opacity=base_opacity,
            detach_base_dc=True,
            seed_attribute_mode="lifecycle",
            seed_lifecycle_active=seed_active,
            seed_lifecycle_never_open=seed_never_open,
        )
        rendered = render_change(
            item.view,
            model,
            replay.pipe,
            replay.evidence_background,
            clamp_output=False,
        )["render"]
    return compute_ssf_loss(target, rendered)[0]


def seed_dc_finite_difference_audit(
    replay: BayesianDetectorReplay,
    condition: Condition,
    *,
    epsilon: float = 5.0e-2,
) -> dict[str, Any]:
    """Compare seed-DC autograd with a centered finite difference."""

    active = replay.seed_lifecycle.active_mask(replay.current_index)
    if not bool(active.any()):
        return {"status": "no_active_seed_rows"}
    replay.base_optimizer.zero_grad(set_to_none=True)
    replay.seed_optimizer.zero_grad(set_to_none=True)
    loss = _dc_audit_loss(replay, condition)
    loss.backward()
    gradient = replay.seed_model.new_dc.grad
    if gradient is None:
        return {"status": "missing_autograd"}
    eligible = gradient.detach().abs().clone()
    eligible[~active] = 0.0
    flat_index = int(eligible.reshape(-1).argmax().item())
    analytic = float(gradient.reshape(-1)[flat_index].item())
    parameter = replay.seed_model.new_dc
    original = float(parameter.detach().reshape(-1)[flat_index].item())
    with torch.no_grad():
        parameter.reshape(-1)[flat_index] = original + epsilon
    plus = float(_dc_audit_loss(replay, condition).detach().item())
    with torch.no_grad():
        parameter.reshape(-1)[flat_index] = original - epsilon
    minus = float(_dc_audit_loss(replay, condition).detach().item())
    with torch.no_grad():
        parameter.reshape(-1)[flat_index] = original
    finite_difference = (plus - minus) / (2.0 * epsilon)
    denominator = max(abs(analytic), abs(finite_difference), 1.0e-12)
    replay.base_optimizer.zero_grad(set_to_none=True)
    replay.seed_optimizer.zero_grad(set_to_none=True)
    if abs(analytic) <= 1.0e-12 and abs(finite_difference) <= 1.0e-12:
        status = "zero_autograd_and_finite_difference"
        relative_error: float | None = None
    else:
        status = "ok"
        relative_error = abs(analytic - finite_difference) / denominator
    return {
        "status": status,
        "timestamp": int(replay.current_index),
        "flat_parameter_index": flat_index,
        "epsilon": epsilon,
        "analytic_gradient": analytic,
        "finite_difference_gradient": finite_difference,
        "relative_error": relative_error,
        "loss_plus": plus,
        "loss_minus": minus,
    }


def _scene_name(frame_name: str) -> str:
    return next(
        (
            scene
            for scene in ("scene_change1", "scene_change2", "scene_change3")
            if scene in frame_name
        ),
        "unknown",
    )


def _prefix(values: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _u8_gray(value: np.ndarray) -> np.ndarray:
    gray = np.rint(np.clip(value, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def save_diagnostic_images(
    directory: Path,
    *,
    timestamp: int,
    learned: np.ndarray,
    cue: np.ndarray,
    white_black: np.ndarray,
    white_open_only: np.ndarray,
    gt: np.ndarray,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{timestamp:03d}"
    images = {
        "cue": _u8_gray(cue),
        "learned": _u8_gray(learned),
        "prediction": _u8_gray(learned >= 0.5),
        "white_ceiling_with_black": _u8_gray(white_black),
        "white_ceiling_open_only": _u8_gray(white_open_only),
        "gt": _u8_gray(gt),
    }
    for name, image in images.items():
        Image.fromarray(image, mode="RGB").save(directory / f"{stem}_{name}.png")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=tuple(CONDITIONS), required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-frames", type=int, default=304)
    parser.add_argument("--updates-per-frame", type=int, default=120)
    parser.add_argument("--boundary-json", type=Path, default=DEFAULT_BOUNDARIES)
    parser.add_argument("--da3-seed-checkpoint", type=Path, default=DEFAULT_DA3_SEEDS)
    parser.add_argument("--sam-sign-trace-root", type=Path, default=DEFAULT_SAM_TRACE)
    parser.add_argument("--gradient-audit-frame", type=int, default=13)
    args = parser.parse_args(argv)
    if not 1 <= args.max_frames <= 304:
        parser.error("--max-frames must lie in [1,304]")
    if args.updates_per_frame <= 0:
        parser.error("--updates-per-frame must be positive")
    if args.gradient_audit_frame < 0:
        parser.error("--gradient-audit-frame must be nonnegative")
    for path in (
        args.boundary_json,
        args.da3_seed_checkpoint,
        args.sam_sign_trace_root,
    ):
        if not path.exists():
            parser.error(f"required artifact not found: {path}")
    return args


def viewer_arguments(args: argparse.Namespace, condition: Condition) -> argparse.Namespace:
    return parse_viewer_args(
        [
            "--max-frames",
            str(args.max_frames),
            "--cue-fusion",
            "l1_power_product",
            "--product-exponent",
            "0.3",
            "--cue-mode",
            "soft",
            "--cue-scale",
            "2",
            "--cue-remap",
            "learned_sigmoid",
            "--cue-boundary-json",
            str(args.boundary_json),
            "--da3-seed-checkpoint",
            str(args.da3_seed_checkpoint),
            # Keep the historical DC-only comparison's detector contract fixed.
            "--da3-detector-cue-source",
            "part19_binary",
            "--da3-detector-cue-threshold",
            "0.5",
            "--train-never-open-geometry",
            "--sam-sign-trace-root",
            str(args.sam_sign_trace_root),
            "--representation-updates",
            str(args.updates_per_frame),
            "--current-view-probability",
            "0.33",
            "--representation-cue-amplitude",
            str(condition.cue_amplitude),
            "--seed-dc-supervision",
            condition.seed_dc_supervision,
            "--seed-dc-loss-weight",
            "1",
            "--dc-replay-mode",
            condition.dc_replay_mode,
        ]
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    condition = CONDITIONS[args.condition]
    output_dir = args.output_root / condition.name
    output_dir.mkdir(parents=True, exist_ok=True)
    replay = BayesianDetectorReplay(viewer_arguments(args, condition))
    if replay.da3_seeds is None or replay.sam_model is None:
        raise RuntimeError("final DA3/SAM viewer contract was not loaded")

    started = time.time()
    rows: list[dict[str, Any]] = []
    gradient_audit: dict[str, Any] | None = None
    total_base_open = total_base_close = 0
    total_seed_open = total_seed_close = 0
    total_seed_proposed = total_seed_accepted = total_seed_rejected = 0
    total_historical_lifespan_updates = 0
    total_lifespan_render_violations = 0
    while replay.has_next:
        summary = replay.step()
        if summary.lifespan_render_violations != 0:
            raise RuntimeError("historical replay lifespan audit failed")
        timestamp = int(replay.current_index)
        with torch.no_grad():
            learned = replay.learned_change_score()[0].detach().cpu().numpy()
            white_black_t, white_open_only_t = render_white_counterfactuals(replay)
            white_black = white_black_t.detach().cpu().numpy()
            white_open_only = white_open_only_t.detach().cpu().numpy()
            cue = (
                replay.representation_replay[-1]
                .cue_target[0]
                .detach()
                .cpu()
                .numpy()
            )

        # Evaluation-only access starts after detector, birth, and all updates.
        gt = np.asarray(
            replay.current_view.gt_add_mask | replay.current_view.gt_remove_mask,
            dtype=bool,
        )
        gt_metrics = binary_metrics(learned >= 0.5, gt)
        cue_metrics = binary_metrics(learned >= 0.5, cue >= 0.5)
        diagnostics = cue_diagnostics(
            learned,
            cue,
            white_black,
            white_open_only,
            gt,
        )

        sampled = [
            causal_training_view_index(
                timestamp,
                update,
                seed=int(replay.args.representation_seed),
                current_probability=float(replay.args.current_view_probability),
            )[0]
            for update in range(int(replay.args.representation_updates))
        ]
        if condition.dc_replay_mode == "current":
            dc_indices = [timestamp] * len(sampled)
        else:
            dc_indices = sampled
        diagnostics.update(
            {
                "representation_updates": len(dc_indices),
                "dc_current_updates": sum(index == timestamp for index in dc_indices),
                "dc_view_age_sum": sum(timestamp - index for index in dc_indices),
            }
        )
        gradients = dc_gradient_stats(replay)
        frame_name = str(replay.records[timestamp].name)
        row = {
            "timestamp": timestamp,
            "frame_name": frame_name,
            "scene": _scene_name(frame_name),
            **_prefix(gt_metrics, "gt"),
            **_prefix(cue_metrics, "cue_mask"),
            **diagnostics,
            "base_open": summary.open,
            "base_never_open": summary.never_open,
            "base_closed": summary.closed,
            "base_opened_now": summary.opened_now,
            "base_closed_now": summary.closed_now,
            "seed_active": summary.active_da3_seed_rows,
            "seed_never_open": summary.da3_seed_never_open,
            "seed_closed": summary.da3_seed_closed,
            "seed_opened_now": summary.da3_seed_opened_now,
            "seed_closed_now": summary.da3_seed_closed_now,
            "seed_proposed_now": summary.da3_seed_proposed_now,
            "seed_accepted_now": summary.da3_seed_accepted_now,
            "seed_coverage_rejected_now": summary.da3_seed_coverage_rejected_now,
            "historical_lifespan_replay_updates": (
                summary.historical_lifespan_replay_updates
            ),
            "lifespan_render_violations": summary.lifespan_render_violations,
            "base_gradient_rows": gradients["base"]["nonzero_rows"],
            "base_gradient_absolute_mean": gradients["base"]["absolute_mean"],
            "base_gradient_whitening_fraction": gradients["base"][
                "whitening_fraction"
            ],
            "seed_gradient_rows": gradients["seed"]["nonzero_rows"],
            "seed_gradient_absolute_mean": gradients["seed"]["absolute_mean"],
            "seed_gradient_whitening_fraction": gradients["seed"][
                "whitening_fraction"
            ],
        }
        rows.append(row)
        total_base_open += summary.opened_now
        total_base_close += summary.closed_now
        total_seed_open += summary.da3_seed_opened_now
        total_seed_close += summary.da3_seed_closed_now
        total_seed_proposed += summary.da3_seed_proposed_now
        total_seed_accepted += summary.da3_seed_accepted_now
        total_seed_rejected += summary.da3_seed_coverage_rejected_now
        total_historical_lifespan_updates += (
            summary.historical_lifespan_replay_updates
        )
        total_lifespan_render_violations += summary.lifespan_render_violations

        if timestamp == args.gradient_audit_frame:
            gradient_audit = seed_dc_finite_difference_audit(replay, condition)
        if timestamp in DIAGNOSTIC_FRAMES or timestamp == replay.total_frames - 1:
            save_diagnostic_images(
                output_dir / "diagnostics",
                timestamp=timestamp,
                learned=learned,
                cue=cue,
                white_black=white_black,
                white_open_only=white_open_only,
                gt=gt,
            )
        if (
            timestamp < 15
            or timestamp % 10 == 9
            or timestamp in {94, 95, 198, 199, replay.total_frames - 1}
        ):
            print(
                f"[{condition.name} {timestamp + 1:03d}/{replay.total_frames}] "
                f"GT-IoU={gt_metrics['iou']:.4f} cue-IoU={cue_metrics['iou']:.4f} "
                f"OPEN={summary.open:,} seed={summary.active_da3_seed_rows:,}",
                flush=True,
            )

    by_scene: dict[str, Any] = {}
    for scene in ("scene_change1", "scene_change2", "scene_change3"):
        selected = [row for row in rows if row["scene"] == scene]
        if selected:
            by_scene[scene] = {
                "gt": aggregate_binary(selected, "gt"),
                "cue_mask": aggregate_binary(selected, "cue_mask"),
                "cue_diagnostics": aggregate_cue_diagnostics(selected),
            }

    result = {
        "script": "experiments/evaluate_bayesian_da3_dc_hypotheses.py",
        "condition": asdict(condition),
        "controlled_constants": {
            "detector": "pre-optimization learned-Q alpha-T BF30",
            "seed_detector": "pre-optimization untouched P+S binary BF30",
            "cue_fusion": "l1_power_product alpha=0.3 then causal learned sigmoid",
            "geometry_target": "unit learned Q",
            "geometry_replay": "p=0.33 latest branch, otherwise uniform observed",
            "sampled_replay_population": (
                "base and DA3 half-open lifespans at the sampled timestamp"
            ),
            "future_born_seed_policy": "excluded from render and optimizer",
            "representation_parameter_semantics": (
                "persistent per Gaussian row; replay restores timestamped "
                "population, not parameter snapshots"
            ),
            "updates_per_frame": int(args.updates_per_frame),
            "representation_seed": int(replay.args.representation_seed),
            "base_geometry": "frozen",
            "da3_never_open_geometry": "trainable",
            "evaluation_threshold": 0.5,
            "evaluation_gt": "ADD union REMOVE, accessed after each completed step",
        },
        "artifacts": {
            "boundary_json": str(args.boundary_json),
            "da3_seed_checkpoint": str(args.da3_seed_checkpoint),
            "sam_sign_trace_root": str(args.sam_sign_trace_root),
        },
        "frames": len(rows),
        "metrics": {
            "gt": aggregate_binary(rows, "gt"),
            "cue_mask": aggregate_binary(rows, "cue_mask"),
            "cue_diagnostics": aggregate_cue_diagnostics(rows),
        },
        "per_scene": by_scene,
        "events": {
            "base_open": total_base_open,
            "base_close": total_base_close,
            "seed_open": total_seed_open,
            "seed_close": total_seed_close,
            "seed_proposed": total_seed_proposed,
            "seed_accepted": total_seed_accepted,
            "seed_coverage_rejected": total_seed_rejected,
        },
        "final_intrinsic_dc": intrinsic_dc_stats(replay),
        "gradient_finite_difference_audit": gradient_audit,
        "invariant_hashes": invariant_hashes(replay),
        "runtime_seconds": time.time() - started,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        "future_view_accesses": 0,
        "historical_lifespan_replay_updates": total_historical_lifespan_updates,
        "lifespan_render_violations": total_lifespan_render_violations,
    }
    torch.save(
        {
            "base_current_state_index": replay.lifecycle.current_state_index.detach().cpu(),
            "base_num_states": replay.lifecycle.num_states.detach().cpu(),
            "base_open_timestamp": replay.lifecycle.open_timestamp.detach().cpu(),
            "base_materialized_timestamp": replay.lifecycle.materialized_timestamp.detach().cpu(),
            "base_state_start": replay.lifecycle.state_start.detach().cpu(),
            "base_state_end": replay.lifecycle.state_end.detach().cpu(),
            "base_state_valid": replay.lifecycle.state_valid.detach().cpu(),
            "seed_xyz": replay.seed_model._xyz.detach().cpu(),
            "seed_opacity": replay.seed_model._opacity.detach().cpu(),
            "seed_scaling": replay.seed_model._scaling.detach().cpu(),
            "seed_rotation": replay.seed_model._rotation.detach().cpu(),
            "seed_start": replay.seed_model.start.detach().cpu(),
            "seed_end": replay.seed_model.end.detach().cpu(),
            "seed_lifecycle_current_state_index": replay.seed_lifecycle.current_state_index.detach().cpu(),
            "seed_lifecycle_num_states": replay.seed_lifecycle.num_states.detach().cpu(),
            "seed_lifecycle_materialized_timestamp": replay.seed_lifecycle.materialized_timestamp.detach().cpu(),
            "seed_lifecycle_state_start": replay.seed_lifecycle.state_start.detach().cpu(),
            "seed_lifecycle_state_end": replay.seed_lifecycle.state_end.detach().cpu(),
            "seed_lifecycle_state_valid": replay.seed_lifecycle.state_valid.detach().cpu(),
            "seed_geometry_update_counts": replay.seed_geometry_update_counts.detach().cpu(),
            "accepted_da3_source_rows": torch.as_tensor(
                replay.accepted_da3_source_rows, dtype=torch.long
            ),
            "accepted_da3_birth_global": torch.as_tensor(
                replay.accepted_da3_birth_global, dtype=torch.long
            ),
        },
        output_dir / "controlled_state.pt",
    )
    with (output_dir / "frame_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: Sequence[str] | None = None) -> None:
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
