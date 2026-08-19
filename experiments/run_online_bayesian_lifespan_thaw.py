"""Causal online Bayesian lifespans with lifespan-gated geometry plasticity.

Inference is strictly frame-major.  Fixed-camera cues are converted to binary or
fractional evidence, immutable reference Gaussians are probed with alpha-T VJPs,
Bayesian state is updated only for observed rows, binary lifecycle decisions are
committed at the current timestamp, and only currently open row-slot pairs may
optimize.  GT masks and diagnostic boundaries are loaded only after the online
loop has completed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from temporal import BayesianLifespanController, TemporalGeometryChangeModel
from temporal.bayesian_lifespan_controller import LifespanAction
from temporal.bernoulli_bocd import (
    BernoulliBOCDConfig,
    BOCDUpdate,
    make_bocd_filter,
)
from temporal.change_evidence import accumulate_change_evidence, evidence_counts
from temporal.masked_optimizer import MaskedRowSlotAdam

DEFAULT_SOURCE = "data/Instance_1/scene_change1_2_3"
DEFAULT_BOUNDARIES = (95, 199)
BASE_PLY_REL = "reference_reconstruction/point_cloud/iteration_30000/point_cloud.ply"
DEFAULT_FIXED_CAMERAS = Path(
    "/home/rvl/workspace/github/O-SCD/output/ESCD_fixedpose_protocols_res4/"
    "scene_change1_2_3/cameras_fixed.json"
)
DEFAULT_CUE_CACHE = Path(
    "/home/rvl/workspace/github/O-SCD/artifacts/escd_396ref/"
    "fixed_pose_cues_res4_v1"
)
DEFAULT_OUTPUT = Path(
    "outputs/instance1_scene_change1_2_3_online_bayesian_lifespan_thaw"
)
OUTPUT_FILES = (
    "summary.json",
    "frame_metrics.csv",
    "lifecycle_events.jsonl",
    "per_frame_bayesian_stats.npz",
    "checkpoint.pt",
)
VISUALIZATION_FILES = (
    "visual_summary.json",
    "timeline_metrics.png",
)
GEOMETRY_NAMES = ("dc", "xyz", "opacity", "scaling", "rotation")
BASE_FIELDS = (
    "_xyz",
    "_features_dc",
    "_features_rest",
    "_opacity",
    "_scaling",
    "_rotation",
)
STAT_KEYS = ("count", "mean", "q05", "q50", "q95")


@dataclass(frozen=True)
class RunConfig:
    bayes_cue_mode: str = "binary"
    bayes_cue_threshold: float = 0.5
    bayes_cue_scale: float = 1.0
    evidence_count_mode: str = "capped"
    evidence_mass_saturation: float = 1.0
    min_evidence_mass: float = 1e-6
    bocd_mode: str = "map_reset"
    prior_a: float = 1.0
    prior_b: float = 1.0
    expected_run_length: float | None = 100.0
    hazard: float | None = None
    max_run_length: int = 128
    bocd_chunk_size: int = 65536
    exact_bocd_memory_limit_gb: float = 8.0
    open_probability: float = 0.6
    close_probability: float = 0.4
    changepoint_probability: float = 0.5
    min_run_evidence: float = 1.0
    min_visible_observations: int = 1
    max_states: int = 8
    thaw_parameters: tuple[str, ...] = ("dc",)
    updates_per_frame: int = 1
    detector_only: bool = False
    evaluation_threshold: float = 0.5
    inactive_audit_max_pairs: int = 4096
    seed: int = 0


@dataclass(frozen=True)
class LifecycleEvent:
    gaussian_index: int
    decision_timestamp: int
    bocd_estimated_changepoint_timestamp: int
    old_binary_label: int
    new_binary_label: int
    action: str
    old_slot: int
    new_current_slot: int
    posterior_probability: float
    changepoint_probability: float
    concentration: float
    visible_observation_count: int


@dataclass
class ClosedPairAudit:
    """Bounded exact audit of formerly active row-slot pairs."""

    max_pairs: int

    def __post_init__(self) -> None:
        self.rows: list[int] = []
        self.slots: list[int] = []
        self.total_closed_pairs = 0

    def add(self, rows: torch.Tensor, slots: torch.Tensor) -> None:
        row_values = rows.detach().cpu().tolist()
        slot_values = slots.detach().cpu().tolist()
        self.total_closed_pairs += len(row_values)
        available = max(0, self.max_pairs - len(self.rows))
        if available:
            self.rows.extend(int(value) for value in row_values[:available])
            self.slots.extend(int(value) for value in slot_values[:available])

    def coordinates(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.rows, device=device, dtype=torch.long),
            torch.tensor(self.slots, device=device, dtype=torch.long),
        )

    @property
    def exhaustive(self) -> bool:
        return self.total_closed_pairs <= self.max_pairs


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("value must be finite and nonnegative")
    return parsed


def parse_visualization_frame_indices(value: str | None) -> tuple[int, ...] | None:
    if value is None or value == "":
        return None
    indices = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if any(index < 0 for index in indices):
        raise argparse.ArgumentTypeError("visualization frame indices must be nonnegative")
    return tuple(dict.fromkeys(indices))


def _font(size: int = 14):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow<10 compatibility.
        return ImageFont.load_default()


def parse_thaw_parameters(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        parts = tuple(part.strip() for part in value.split(",") if part.strip())
    else:
        parts = tuple(value)
    if not parts:
        raise argparse.ArgumentTypeError("at least one thaw parameter is required")
    if len(set(parts)) != len(parts) or any(name not in GEOMETRY_NAMES for name in parts):
        raise argparse.ArgumentTypeError(
            "use unique parameters from dc,xyz,opacity,scaling,rotation"
        )
    if tuple(name for name in GEOMETRY_NAMES if name in parts) != parts:
        raise argparse.ArgumentTypeError(
            "parameters must follow dc,xyz,opacity,scaling,rotation order"
        )
    if parts[0] != "dc":
        raise argparse.ArgumentTypeError("thaw parameters must include dc")
    return parts


def serializable_arguments(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def run_config_from_args(args: argparse.Namespace) -> RunConfig:
    config = RunConfig(
        bayes_cue_mode=args.bayes_cue_mode,
        bayes_cue_threshold=float(args.bayes_cue_threshold),
        bayes_cue_scale=float(args.bayes_cue_scale),
        evidence_count_mode=args.evidence_count_mode,
        evidence_mass_saturation=float(args.evidence_mass_saturation),
        min_evidence_mass=float(args.min_evidence_mass),
        bocd_mode=args.bocd_mode,
        prior_a=float(args.prior_a),
        prior_b=float(args.prior_b),
        expected_run_length=(
            None if args.hazard is not None else float(args.expected_run_length)
        ),
        hazard=None if args.hazard is None else float(args.hazard),
        max_run_length=int(args.max_run_length),
        bocd_chunk_size=int(args.bocd_chunk_size),
        exact_bocd_memory_limit_gb=float(args.exact_bocd_memory_limit_gb),
        open_probability=float(args.open_probability),
        close_probability=float(args.close_probability),
        changepoint_probability=float(args.changepoint_probability),
        min_run_evidence=float(args.min_run_evidence),
        min_visible_observations=int(args.min_visible_observations),
        max_states=int(args.max_states),
        thaw_parameters=parse_thaw_parameters(args.thaw_parameters),
        updates_per_frame=int(args.updates_per_frame),
        detector_only=bool(args.detector_only),
        evaluation_threshold=float(args.evaluation_threshold),
        inactive_audit_max_pairs=int(args.inactive_audit_max_pairs),
        seed=int(args.seed),
    )
    validate_run_config(config)
    return config


def validate_run_config(config: RunConfig) -> None:
    if config.bayes_cue_mode not in {"binary", "soft"}:
        raise ValueError("bayes_cue_mode must be binary|soft")
    if config.bayes_cue_scale <= 0:
        raise ValueError("bayes_cue_scale must be positive")
    if config.evidence_count_mode not in {"raw", "capped"}:
        raise ValueError("evidence_count_mode must be raw|capped")
    if config.evidence_mass_saturation <= 0:
        raise ValueError("evidence_mass_saturation must be positive")
    if config.min_evidence_mass < 0:
        raise ValueError("min_evidence_mass must be nonnegative")
    if config.bocd_mode not in {"exact", "map_reset"}:
        raise ValueError("bocd_mode must be exact|map_reset")
    if min(
        config.max_states,
        config.max_run_length,
        config.bocd_chunk_size,
        config.updates_per_frame,
    ) < 1:
        raise ValueError("state/run/chunk/update sizes must be positive")
    if config.inactive_audit_max_pairs < 0:
        raise ValueError("inactive_audit_max_pairs must be nonnegative")
    if not 0.0 <= config.evaluation_threshold <= 1.0:
        raise ValueError("evaluation_threshold must be in [0,1]")
    parse_thaw_parameters(config.thaw_parameters)
    _ = bocd_config(config)


def bocd_config(config: RunConfig) -> BernoulliBOCDConfig:
    return BernoulliBOCDConfig(
        prior_a=config.prior_a,
        prior_b=config.prior_b,
        expected_run_length=config.expected_run_length,
        hazard=config.hazard,
        max_run_length=config.max_run_length,
        min_evidence_mass=config.min_evidence_mass,
        open_probability=config.open_probability,
        close_probability=config.close_probability,
        changepoint_probability=config.changepoint_probability,
        min_run_evidence=config.min_run_evidence,
        min_visible_observations=config.min_visible_observations,
    )


def estimated_bocd_state_bytes(
    mode: str,
    gaussian_count: int,
    max_run_length: int,
    dtype: torch.dtype = torch.float32,
) -> int:
    float_bytes = torch.empty((), dtype=dtype).element_size()
    long_bytes = torch.empty((), dtype=torch.long).element_size()
    if mode == "exact":
        run_count = max_run_length + 1
        # Four floating and two integer run tensors plus global diagnostics.
        return gaussian_count * (
            run_count * (4 * float_bytes + 2 * long_bytes)
            + float_bytes
            + 3 * long_bytes
        )
    if mode == "map_reset":
        # Four floating and five integer vectors.
        return gaussian_count * (4 * float_bytes + 5 * long_bytes)
    raise ValueError("unknown BOCD mode")


def enforce_bocd_memory_limit(config: RunConfig, gaussian_count: int, dtype: torch.dtype) -> int:
    estimate = estimated_bocd_state_bytes(
        config.bocd_mode, gaussian_count, config.max_run_length, dtype
    )
    limit = int(config.exact_bocd_memory_limit_gb * 1024**3)
    if config.bocd_mode == "exact" and estimate > limit:
        raise MemoryError(
            "exact BOCD persistent state is estimated at "
            f"{estimate / 1024**3:.2f} GiB, above the configured "
            f"{config.exact_bocd_memory_limit_gb:.2f} GiB limit; reduce "
            "--max-run-length/--exact-bocd-memory-limit-gb or explicitly use "
            "--bocd-mode map_reset"
        )
    return estimate


def quantile_summary(values: torch.Tensor) -> dict[str, float | int | None]:
    values = values.detach().flatten().float().cpu()
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {"count": 0, "mean": None, "q05": None, "q50": None, "q95": None}
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "q05": float(torch.quantile(values, 0.05).item()),
        "q50": float(torch.quantile(values, 0.50).item()),
        "q95": float(torch.quantile(values, 0.95).item()),
    }


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def base_snapshots(base: Any) -> dict[str, torch.Tensor]:
    return {
        name: getattr(base, name).detach().cpu().clone()
        for name in BASE_FIELDS
        if isinstance(getattr(base, name, None), torch.Tensor)
    }


def base_checksums(base: Any) -> dict[str, str]:
    return {name: tensor_sha256(value) for name, value in base_snapshots(base).items()}


def combined_checksum(checksums: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(checksums):
        digest.update(name.encode("utf-8"))
        digest.update(checksums[name].encode("ascii"))
    return digest.hexdigest()


def base_drift(before: Mapping[str, torch.Tensor], base: Any) -> dict[str, Any]:
    maximum = 0.0
    equal = True
    per_tensor: dict[str, float] = {}
    for name, expected in before.items():
        actual = getattr(base, name).detach().cpu()
        equal = equal and torch.equal(actual, expected)
        difference = (
            float((actual - expected).abs().max().item()) if expected.numel() else 0.0
        )
        maximum = max(maximum, difference)
        per_tensor[name] = difference
    return {
        "bitwise_equal": bool(equal),
        "max_abs": maximum,
        "per_tensor_max_abs": per_tensor,
    }


def current_pair_mask(model: Any) -> torch.Tensor:
    mask = torch.zeros_like(model.state_valid, dtype=torch.bool)
    rows = torch.arange(mask.shape[0], device=mask.device)
    slots = model.current_state_index.long()
    active = slots >= 0
    if bool(active.any()):
        mask[rows[active], slots[active]] = True
    return mask


def capture_pair_audit(
    model: Any,
    optimizer: MaskedRowSlotAdam | None,
    rows: torch.Tensor,
    slots: torch.Tensor,
) -> dict[str, dict[str, torch.Tensor]]:
    snapshot: dict[str, dict[str, torch.Tensor]] = {}
    for name, parameter in model.state_parameter_items():
        values: dict[str, torch.Tensor] = {
            "parameter": parameter.detach()[rows, slots].cpu().clone()
        }
        if optimizer is not None and parameter in optimizer.state:
            for state_name, state_value in optimizer.state[parameter].items():
                if not isinstance(state_value, torch.Tensor):
                    continue
                if state_value.shape == parameter.shape:
                    values[state_name] = state_value.detach()[rows, slots].cpu().clone()
                elif tuple(state_value.shape) == tuple(parameter.shape[:2]):
                    values[state_name] = state_value.detach()[rows, slots].cpu().clone()
        snapshot[name] = values
    return snapshot


def compare_pair_audit(
    before: Mapping[str, Mapping[str, torch.Tensor]],
    model: Any,
    optimizer: MaskedRowSlotAdam | None,
    rows: torch.Tensor,
    slots: torch.Tensor,
    audit: ClosedPairAudit,
) -> dict[str, Any]:
    maximum = 0.0
    per_state: dict[str, float] = {}
    parameter_map = dict(model.state_parameter_items())
    for name, values in before.items():
        parameter = parameter_map[name]
        for state_name, expected in values.items():
            if state_name == "parameter":
                actual = parameter.detach()[rows, slots].cpu()
            else:
                if optimizer is None or parameter not in optimizer.state:
                    raise RuntimeError("optimizer audit state disappeared")
                actual = optimizer.state[parameter][state_name].detach()[rows, slots].cpu()
            difference = (
                float((actual - expected).abs().max().item())
                if expected.numel()
                else 0.0
            )
            per_state[f"{name}.{state_name}"] = difference
            maximum = max(maximum, difference)
    return {
        "passed": maximum == 0.0,
        "max_abs": maximum,
        "audited_pair_count": len(audit.rows),
        "total_closed_pair_count": audit.total_closed_pairs,
        "exhaustive": audit.exhaustive,
        "per_parameter_and_optimizer_state_max_abs": per_state,
    }


def controller_events(update) -> list[LifecycleEvent]:
    positions = torch.nonzero(update.event_mask, as_tuple=False).flatten().tolist()
    records: list[LifecycleEvent] = []
    for position in positions:
        records.append(
            LifecycleEvent(
                gaussian_index=int(update.indices[position].item()),
                decision_timestamp=int(update.decision_timestamp),
                bocd_estimated_changepoint_timestamp=int(
                    update.estimated_changepoint_timestamp[position].item()
                ),
                old_binary_label=int(update.old_binary_label[position].item()),
                new_binary_label=int(update.new_binary_label[position].item()),
                action=LifespanAction(int(update.action[position].item())).name,
                old_slot=int(update.old_slot[position].item()),
                new_current_slot=int(update.current_slot[position].item()),
                posterior_probability=float(
                    update.posterior_probability[position].item()
                ),
                changepoint_probability=float(
                    update.changepoint_probability[position].item()
                ),
                concentration=float(update.concentration[position].item()),
                visible_observation_count=int(
                    update.visible_observations[position].item()
                ),
            )
        )
    return records


def _concat_or_empty(values: list[torch.Tensor]) -> torch.Tensor:
    if not values:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat([value.detach().float().cpu() for value in values])


def update_bayesian_lifecycle_chunks(
    tracker,
    controller: BayesianLifespanController,
    model: Any,
    optimizer: MaskedRowSlotAdam | None,
    delta_a: torch.Tensor,
    delta_b: torch.Tensor,
    total_mass: torch.Tensor,
    *,
    timestamp: int,
    min_evidence_mass: float,
    chunk_size: int,
    closed_audit: ClosedPairAudit,
) -> dict[str, Any]:
    observed = total_mass >= float(min_evidence_mass)
    observed_rows = torch.nonzero(observed, as_tuple=False).flatten()
    action_counts = {action.name: 0 for action in LifespanAction}
    change_values: list[torch.Tensor] = []
    changepoint_values: list[torch.Tensor] = []
    events: list[LifecycleEvent] = []
    for start in range(0, int(observed_rows.numel()), int(chunk_size)):
        rows = observed_rows[start : start + int(chunk_size)]
        result: BOCDUpdate = tracker.update(
            delta_a[rows],
            delta_b[rows],
            total_mass=total_mass[rows],
            row_indices=rows,
            timestamp=timestamp,
        )
        decision = controller.update(
            result, timestamp=timestamp, optimizer=optimizer
        )
        counts = torch.bincount(
            decision.action.to(torch.long), minlength=len(LifespanAction)
        )
        for action in LifespanAction:
            action_counts[action.name] += int(counts[int(action)].item())
        change_values.append(result.change_probability)
        changepoint_values.append(result.changepoint_probability)
        events.extend(controller_events(decision))

        close_positions = torch.nonzero(
            decision.action == int(LifespanAction.CLOSE), as_tuple=False
        ).flatten()
        if close_positions.numel():
            closed_audit.add(
                decision.indices[close_positions], decision.old_slot[close_positions]
            )

    return {
        "observed": observed,
        "observed_count": int(observed_rows.numel()),
        "action_counts": action_counts,
        "change_probability_stats": quantile_summary(_concat_or_empty(change_values)),
        "changepoint_probability_stats": quantile_summary(
            _concat_or_empty(changepoint_values)
        ),
        "events": events,
    }


def frame_diagnostics(
    *,
    timestamp: int,
    frame_name: str,
    evidence,
    bayesian: Mapping[str, Any],
    model: Any,
    geometry_thawed_rows: int,
    base_checksum: str,
    inactive_audit: Mapping[str, Any],
    frame_runtime_seconds: float,
    cuda_peak_memory_bytes: int,
    predicted_positive_fraction: float,
    algorithm: str,
) -> dict[str, Any]:
    counts = bayesian["action_counts"]
    return {
        "timestamp": int(timestamp),
        "frame": frame_name,
        "bocd_algorithm": algorithm,
        "observed_gaussian_count": int(bayesian["observed_count"]),
        "positive_pseudocount_mass": float(evidence.delta_a.sum().item()),
        "negative_pseudocount_mass": float(evidence.delta_b.sum().item()),
        "raw_alpha_t_mass": float(evidence.total_mass.sum().item()),
        "change_probability_stats": dict(bayesian["change_probability_stats"]),
        "changepoint_probability_stats": dict(
            bayesian["changepoint_probability_stats"]
        ),
        "open_count": int(counts["OPEN"]),
        "keep_count": int(counts["KEEP"]),
        "close_count": int(counts["CLOSE"]),
        "none_count": int(counts["NONE"]),
        "uncertain_count": int(counts["UNCERTAIN"]),
        "active_lifespan_count": int((model.current_state_index >= 0).sum().item()),
        "number_of_geometry_thawed_rows": int(geometry_thawed_rows),
        "base_checksum": base_checksum,
        "inactive_parameter_drift_audit": dict(inactive_audit),
        "predicted_positive_fraction": float(predicted_positive_fraction),
        "frame_runtime_seconds": float(frame_runtime_seconds),
        "cuda_peak_memory_bytes": int(cuda_peak_memory_bytes),
        # Populated only after the causal loop.
        "tp": None,
        "tn": None,
        "fp": None,
        "fn": None,
        "iou": None,
        "f1": None,
        "precision": None,
        "recall": None,
    }


def binary_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, int | float]:
    prediction = prediction.astype(bool, copy=False)
    target = target.astype(bool, copy=False)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    tp = int(np.logical_and(prediction, target).sum())
    tn = int(np.logical_and(~prediction, ~target).sum())
    fp = int(np.logical_and(prediction, ~target).sum())
    fn = int(np.logical_and(~prediction, target).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
        "f1": 2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "precision": precision,
        "recall": recall,
    }


def resize_image_to_array(path: Path, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB").resize((width, height), Image.BILINEAR))


def load_cue_array(cue_cache_root: Path, frame_name: str, shape: tuple[int, int]) -> np.ndarray:
    from experiments.train_cue_temporal_rchange import physical_frame_name

    cue_path = cue_cache_root / "cues" / f"{physical_frame_name(Path(frame_name).stem)}.pt"
    cue = torch.load(cue_path, map_location="cpu", weights_only=True)
    if not isinstance(cue, torch.Tensor):
        raise TypeError(f"cue cache entry must be a tensor: {cue_path}")
    array = cue.detach().float().squeeze().numpy()
    if array.shape != shape:
        array = cv2.resize(array, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.clip(array, 0.0, 1.0)


def mask_rgb(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask.astype(bool, copy=False)] = np.asarray(color, dtype=np.uint8)
    return out


def cue_heatmap(cue: np.ndarray) -> np.ndarray:
    cue = np.clip(cue.astype(np.float32, copy=False), 0.0, 1.0)
    heat = np.zeros((*cue.shape, 3), dtype=np.uint8)
    heat[..., 0] = np.round(255.0 * cue).astype(np.uint8)
    heat[..., 1] = np.round(180.0 * cue).astype(np.uint8)
    heat[..., 2] = np.round(255.0 * (1.0 - cue)).astype(np.uint8)
    return heat


def prediction_overlay(rgb: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    overlay = rgb.copy()
    pred = prediction.astype(bool, copy=False)
    overlay[pred] = np.round(0.45 * overlay[pred] + 0.55 * np.array([255, 220, 0])).astype(np.uint8)
    return overlay


def target_overlay(rgb: np.ndarray, target: np.ndarray) -> np.ndarray:
    overlay = rgb.copy()
    mask = target.astype(bool, copy=False)
    overlay[mask] = np.round(
        0.45 * overlay[mask] + 0.55 * np.array([0, 255, 255])
    ).astype(np.uint8)
    return overlay


def confusion_rgb(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred = prediction.astype(bool, copy=False)
    gt = target.astype(bool, copy=False)
    image = np.zeros((*pred.shape, 3), dtype=np.uint8)
    image[pred & gt] = (0, 200, 0)
    image[pred & ~gt] = (255, 105, 180)
    image[~pred & gt] = (0, 90, 255)
    return image


def labeled_panel(array: np.ndarray, title: str, subtitle: str, width: int) -> Image.Image:
    image = Image.fromarray(array.astype(np.uint8, copy=False)).convert("RGB")
    height = max(1, int(round(image.height * width / image.width)))
    image = image.resize((width, height), Image.BILINEAR)
    header = 58
    canvas = Image.new("RGB", (width, height + header), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 5), title, fill="black", font=_font(14))
    draw.multiline_text(
        (6, 24), subtitle[:150], fill=(70, 70, 70), font=_font(11), spacing=2
    )
    canvas.paste(image, (0, header))
    return canvas


def hstack(images: Sequence[Image.Image], gap: int = 8) -> Image.Image:
    width = sum(image.width for image in images) + gap * (len(images) - 1)
    height = max(image.height for image in images)
    out = Image.new("RGB", (width, height), "white")
    x = 0
    for image in images:
        out.paste(image, (x, 0))
        x += image.width + gap
    return out


def choose_visualization_indices(
    frame_rows: Sequence[Mapping[str, Any]],
    explicit: Sequence[int] | None,
    sample_count: int,
) -> list[int]:
    if explicit is not None:
        if any(index >= len(frame_rows) for index in explicit):
            raise IndexError("visualization frame index outside processed frame range")
        return list(explicit)
    if not frame_rows:
        return []
    budget = max(1, min(int(sample_count), len(frame_rows)))
    anchors = [0, len(frame_rows) // 2, len(frame_rows) - 1]
    scored = [
        (index, float(row["iou"]) if row.get("iou") is not None else float("nan"))
        for index, row in enumerate(frame_rows)
    ]
    finite = [(index, value) for index, value in scored if math.isfinite(value)]
    if finite:
        finite_sorted = sorted(finite, key=lambda item: item[1])
        anchors.extend(
            [
                finite_sorted[0][0],
                finite_sorted[len(finite_sorted) // 2][0],
                finite_sorted[-1][0],
            ]
        )
    chosen: list[int] = []
    for index in anchors:
        if 0 <= index < len(frame_rows) and index not in chosen:
            chosen.append(index)
        if len(chosen) >= budget:
            break
    if len(chosen) < budget:
        for value in np.linspace(0, len(frame_rows) - 1, budget):
            index = int(round(float(value)))
            if index not in chosen:
                chosen.append(index)
            if len(chosen) >= budget:
                break
    return sorted(chosen)


def visualization_reason(
    index: int, frame_rows: Sequence[Mapping[str, Any]]
) -> str:
    tags: list[str] = []
    if index == 0:
        tags.append("early")
    if index == len(frame_rows) - 1:
        tags.append("late")
    finite = [
        (i, float(row["iou"]))
        for i, row in enumerate(frame_rows)
        if row.get("iou") is not None and math.isfinite(float(row["iou"]))
    ]
    if finite:
        ranked = sorted(finite, key=lambda item: item[1])
        labels = {
            ranked[0][0]: "worst IoU",
            ranked[len(ranked) // 2][0]: "median IoU",
            ranked[-1][0]: "best IoU",
        }
        if index in labels:
            tags.append(labels[index])
    if index == len(frame_rows) // 2:
        tags.append("midpoint")
    return " / ".join(dict.fromkeys(tags)) or "coverage sample"


def save_timeline_chart(
    frame_rows: Sequence[Mapping[str, Any]], path: Path, scene_name: str = ""
) -> None:
    width, height = 1400, 760
    margin_left, margin_right = 86, 36
    margin_top, margin_bottom = 96, 92
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = _font(18)
    small = _font(13)
    title = "Causal Bayesian lifespan timeline"
    if scene_name:
        title = f"{scene_name}: {title}"
    draw.text((margin_left, 18), title, fill="black", font=font)
    draw.text(
        (margin_left, 48),
        "Independent ref-to-SC stream: no internal CLOSE/REOPEN boundary is expected.",
        fill=(60, 60, 60),
        font=small,
    )
    for tick in range(0, 11):
        value = tick / 10.0
        y = margin_top + plot_h - int(value * plot_h)
        draw.line((margin_left, y, width - margin_right, y), fill=(225, 225, 225))
        draw.text((34, y - 7), f"{value:.1f}", fill="black", font=small)

    def points(key: str, scale: float = 1.0) -> list[tuple[int, int]]:
        if len(frame_rows) == 1:
            xs = [margin_left]
        else:
            xs = [
                margin_left + int(i * plot_w / (len(frame_rows) - 1))
                for i in range(len(frame_rows))
            ]
        ys = []
        for row in frame_rows:
            value = row.get(key)
            value = 0.0 if value is None else max(0.0, min(1.0, float(value) / scale))
            ys.append(margin_top + plot_h - int(value * plot_h))
        return list(zip(xs, ys))

    series = [
        ("IoU", "iou", (0, 120, 210), 1.0),
        ("F1", "f1", (0, 170, 75), 1.0),
        (
            "Predicted mask area fraction",
            "predicted_positive_fraction",
            (255, 140, 0),
            1.0,
        ),
    ]
    for label, key, color, scale in series:
        pts = points(key, scale)
        if len(pts) > 1:
            draw.line(pts, fill=color, width=3)
        for x, y in pts[:: max(1, len(pts) // 24)]:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
    max_active = max((int(row["active_lifespan_count"]) for row in frame_rows), default=1)
    active_pts = points("active_lifespan_count", max(1.0, float(max_active)))
    if len(active_pts) > 1:
        draw.line(active_pts, fill=(140, 90, 210), width=2)
    frame_label = "frame index"
    label_box = draw.textbbox((0, 0), frame_label, font=small)
    draw.text(
        (margin_left + (plot_w - (label_box[2] - label_box[0])) // 2, margin_top + plot_h + 16),
        frame_label,
        fill="black",
        font=small,
    )
    legend_x = margin_left
    legend_y = height - 42
    legend_series = series + [
        (
            f"Active count / max ({max_active:,})",
            "active_lifespan_count",
            (140, 90, 210),
            1.0,
        )
    ]
    for label, _key, color, _scale in legend_series:
        draw.line((legend_x, legend_y + 8, legend_x + 28, legend_y + 8), fill=color, width=4)
        draw.text((legend_x + 36, legend_y), label, fill="black", font=small)
        legend_x += 305
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def write_visualization_artifacts(
    *,
    source_path: Path,
    cue_cache_root: Path,
    records: Sequence[Any],
    predictions: Sequence[np.ndarray],
    score_maps: Sequence[np.ndarray],
    frame_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    panel_width: int,
    sample_count: int,
    frame_indices: Sequence[int] | None,
) -> dict[str, Any]:
    if len(records) != len(predictions) or len(records) != len(score_maps):
        raise ValueError("visualization inputs have different lengths")
    output_dir.mkdir(parents=True, exist_ok=True)
    panels_dir = output_dir / "panels"
    binary_dir = output_dir / "pred_binary"
    score_dir = output_dir / "pred_score"
    confusion_dir = output_dir / "confusion"
    for directory in (panels_dir, binary_dir, score_dir, confusion_dir):
        directory.mkdir(parents=True, exist_ok=True)

    selected = choose_visualization_indices(frame_rows, frame_indices, sample_count)
    panel_paths: list[str] = []
    selected_reasons: list[str] = []
    for index in selected:
        record = records[index]
        prediction = predictions[index].astype(bool, copy=False)
        shape = prediction.shape
        stem = Path(record.name).stem
        rgb = resize_image_to_array(Path(record.image_path), shape)
        cue = load_cue_array(cue_cache_root, record.name, shape)
        gt_path = source_path / "gt_mask" / f"{stem}.png"
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            raise FileNotFoundError(gt_path)
        if gt.shape != shape:
            gt = cv2.resize(gt, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        gt_mask = gt >= 128
        score_u8 = np.clip(score_maps[index], 0, 255).astype(np.uint8)
        confusion = confusion_rgb(prediction, gt_mask)
        binary = prediction.astype(np.uint8) * 255
        Image.fromarray(binary).save(binary_dir / f"{stem}.png")
        Image.fromarray(score_u8).save(score_dir / f"{stem}.png")
        Image.fromarray(confusion).save(confusion_dir / f"{stem}.png")
        row = frame_rows[index]
        reason = visualization_reason(index, frame_rows)
        selected_reasons.append(reason)
        subtitle = (
            "yellow=prediction\n"
            f"IoU={float(row['iou']):.3f} F1={float(row['f1']):.3f} | "
            f"OPEN={row['open_count']} active={row['active_lifespan_count']}"
        )
        panel = hstack(
            [
                labeled_panel(rgb, f"RGB | {reason}", stem, panel_width),
                labeled_panel(cue_heatmap(cue), "Cue C_t", "blue=low, yellow/red=high", panel_width),
                labeled_panel(prediction_overlay(rgb, prediction), "Prediction overlay", subtitle, panel_width),
                labeled_panel(
                    target_overlay(rgb, gt_mask),
                    "GT overlay",
                    "cyan=GT\npost-inference evaluation only",
                    panel_width,
                ),
                labeled_panel(confusion, "Confusion", "green TP, pink FP, blue FN", panel_width),
            ]
        )
        panel_path = panels_dir / f"{index:06d}_{stem}_panel.png"
        panel.save(panel_path)
        panel_paths.append(str(panel_path))
    timeline_path = output_dir / "timeline_metrics.png"
    save_timeline_chart(frame_rows, timeline_path, source_path.name)
    payload = {
        "schema_version": 1,
        "contract": "online_bayesian_ref_scene_visualization",
        "frames": len(frame_rows),
        "selected_frame_indices": selected,
        "selected_frame_reasons": selected_reasons,
        "panel_paths": panel_paths,
        "timeline_metrics_png": str(timeline_path),
        "confusion_legend": {"tp": "green", "fp": "pink", "fn": "blue", "tn": "black"},
        "gt_loaded_after_inference_only": True,
    }
    (output_dir / "visual_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def select_visualization_indices(
    frame_rows: Sequence[Mapping[str, Any]],
    *,
    max_panels: int,
    explicit_indices: Sequence[int] | None = None,
) -> list[int]:
    """Backward-compatible named wrapper used by tests and ad-hoc scripts."""
    if explicit_indices is not None:
        valid = [int(index) for index in explicit_indices if 0 <= int(index) < len(frame_rows)]
        return valid[: int(max_panels)]
    return choose_visualization_indices(frame_rows, None, int(max_panels))


def write_visualizations(
    visualization_dir: Path,
    *,
    source_path: Path,
    cue_cache_root: Path,
    records: Sequence[Any],
    frame_rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[np.ndarray],
    max_panels: int,
    explicit_indices: Sequence[int] | None,
    run_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Compatibility wrapper that writes panels from binary predictions only."""
    score_maps = [prediction.astype(np.uint8) * 255 for prediction in predictions]
    summary = write_visualization_artifacts(
        source_path=source_path,
        cue_cache_root=cue_cache_root,
        records=records,
        predictions=predictions,
        score_maps=score_maps,
        frame_rows=frame_rows,
        output_dir=visualization_dir,
        panel_width=260,
        sample_count=int(max_panels),
        frame_indices=explicit_indices,
    )
    summary.update(
        {
            "scene": source_path.name,
            "metric_summary": run_summary.get("metrics", {}),
            "algorithm": run_summary.get("algorithm"),
            "detector_only": run_summary.get("detector_only"),
            "optimized_parameters": run_summary.get("optimized_parameters", []),
            "timeline": summary["timeline_metrics_png"],
            "panels": [
                {"frame_index": idx, "path": path}
                for idx, path in zip(summary["selected_frame_indices"], summary["panel_paths"])
            ],
        }
    )
    (visualization_dir / "visual_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def evaluate_after_inference(
    source_path: Path,
    records: Sequence[Any],
    predictions: Sequence[np.ndarray],
    frame_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(records) != len(predictions) or len(records) != len(frame_rows):
        raise ValueError("post-inference evaluation inputs have different lengths")
    aggregate = {key: 0 for key in ("tp", "tn", "fp", "fn")}
    for record, prediction, row in zip(records, predictions, frame_rows):
        mask_path = source_path / "gt_mask" / f"{Path(record.name).stem}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(mask_path)
        if mask.shape != prediction.shape:
            mask = cv2.resize(
                mask,
                (prediction.shape[1], prediction.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        metrics = binary_metrics(prediction, mask >= 128)
        row.update(metrics)
        for key in aggregate:
            aggregate[key] += int(metrics[key])
    precision = (
        aggregate["tp"] / (aggregate["tp"] + aggregate["fp"])
        if aggregate["tp"] + aggregate["fp"]
        else 0.0
    )
    recall = (
        aggregate["tp"] / (aggregate["tp"] + aggregate["fn"])
        if aggregate["tp"] + aggregate["fn"]
        else 0.0
    )
    union = aggregate["tp"] + aggregate["fp"] + aggregate["fn"]
    return {
        "evaluated": True,
        "frames": len(frame_rows),
        **aggregate,
        "precision": precision,
        "recall": recall,
        "aggregate_iou": aggregate["tp"] / union if union else 0.0,
        "aggregate_f1": 2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "mean_frame_iou": float(np.mean([float(row["iou"]) for row in frame_rows])),
        "mean_frame_f1": float(np.mean([float(row["f1"]) for row in frame_rows])),
    }




def event_diagnostics_after_inference(
    events: Sequence[LifecycleEvent], boundaries: Sequence[int]
) -> dict[str, Any]:
    open_times = sorted(event.decision_timestamp for event in events if event.action == "OPEN")
    close_times = sorted(event.decision_timestamp for event in events if event.action == "CLOSE")

    def delays(times: Sequence[int]) -> list[int | None]:
        return [next((time_value - boundary for time_value in times if time_value >= boundary), None) for boundary in boundaries]

    false_splits = sum(
        1
        for event in events
        if event.action in {"OPEN", "CLOSE"}
        and event.old_binary_label == event.new_binary_label == 1
    )
    return {
        "boundaries": [int(value) for value in boundaries],
        "global_open_detection_delay": delays(open_times),
        "global_close_detection_delay": delays(close_times),
        "false_open_rate": None,
        "false_close_rate": None,
        "false_rate_unavailable_reason": (
            "global oracle boundaries do not provide per-Gaussian binary lifecycle labels"
        ),
        "reopen_count": sum(
            1
            for event in events
            if event.action == "OPEN" and event.new_current_slot > 0
        ),
        "active_to_active_false_split_count": int(false_splits),
    }


def compact_npz_stats(frame_rows: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    def stat_matrix(name: str) -> np.ndarray:
        rows: list[list[float]] = []
        for row in frame_rows:
            values = row[name]
            rows.append(
                [
                    float("nan") if values[key] is None else float(values[key])
                    for key in STAT_KEYS
                ]
            )
        return np.asarray(rows, dtype=np.float64)

    return {
        "timestamp": np.asarray([row["timestamp"] for row in frame_rows], dtype=np.int64),
        "observed": np.asarray(
            [row["observed_gaussian_count"] for row in frame_rows], dtype=np.int64
        ),
        "positive_pseudocount_mass": np.asarray(
            [row["positive_pseudocount_mass"] for row in frame_rows], dtype=np.float64
        ),
        "negative_pseudocount_mass": np.asarray(
            [row["negative_pseudocount_mass"] for row in frame_rows], dtype=np.float64
        ),
        "change_probability": stat_matrix("change_probability_stats"),
        "changepoint_probability": stat_matrix("changepoint_probability_stats"),
        "open_count": np.asarray([row["open_count"] for row in frame_rows], dtype=np.int64),
        "keep_count": np.asarray([row["keep_count"] for row in frame_rows], dtype=np.int64),
        "close_count": np.asarray([row["close_count"] for row in frame_rows], dtype=np.int64),
        "uncertain_count": np.asarray(
            [row["uncertain_count"] for row in frame_rows], dtype=np.int64
        ),
        "active_lifespan_count": np.asarray(
            [row["active_lifespan_count"] for row in frame_rows], dtype=np.int64
        ),
        "geometry_thawed_rows": np.asarray(
            [row["number_of_geometry_thawed_rows"] for row in frame_rows],
            dtype=np.int64,
        ),
        "inactive_drift_max_abs": np.asarray(
            [row["inactive_parameter_drift_audit"]["max_abs"] for row in frame_rows],
            dtype=np.float64,
        ),
        "frame_runtime_seconds": np.asarray(
            [row["frame_runtime_seconds"] for row in frame_rows], dtype=np.float64
        ),
        "cuda_peak_memory_bytes": np.asarray(
            [row["cuda_peak_memory_bytes"] for row in frame_rows], dtype=np.int64
        ),
    }


def _json_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def write_outputs(
    output_dir: Path,
    summary: Mapping[str, Any],
    frame_metrics: list[dict[str, Any]],
    events: Sequence[LifecycleEvent],
    checkpoint: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "lifecycle_events.jsonl").open("w", encoding="utf-8") as file:
        for event in events:
            file.write(json.dumps(asdict(event), sort_keys=True) + "\n")
    rows = [{key: _json_cell(value) for key, value in row.items()} for row in frame_metrics]
    with (output_dir / "frame_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        fieldnames = list(rows[0]) if rows else ["timestamp"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(output_dir / "per_frame_bayesian_stats.npz", **compact_npz_stats(frame_metrics))
    torch.save(_cpu_tree(dict(checkpoint)), output_dir / "checkpoint.pt")


def _synthetic_evidence_result(
    delta_a: torch.Tensor, delta_b: torch.Tensor, total_mass: torch.Tensor
):
    return SimpleNamespace(
        delta_a=delta_a,
        delta_b=delta_b,
        e_plus=delta_a,
        e_minus=delta_b,
        total_mass=total_mass,
    )


def run_detector_sequence(
    evidence: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    model: Any,
    config: RunConfig,
) -> dict[str, Any]:
    """CPU-friendly detector/controller path used by causal smoke tests."""
    bayes_config = bocd_config(config)
    tracker = make_bocd_filter(
        config.bocd_mode,
        model.state_valid.shape[0],
        bayes_config,
        device=model.state_valid.device,
        dtype=model.state_start.dtype,
    )
    controller = BayesianLifespanController(model, bayes_config)
    checksums = base_checksums(model.base)
    before_base = base_snapshots(model.base)
    closed_audit = ClosedPairAudit(config.inactive_audit_max_pairs)
    frames: list[dict[str, Any]] = []
    events: list[LifecycleEvent] = []
    for timestamp, (raw_positive, raw_negative, total_mass) in enumerate(evidence):
        positive = raw_positive.to(model.state_start)
        negative = raw_negative.to(model.state_start)
        mass = total_mass.to(model.state_start)
        delta_a, delta_b, _, _ = evidence_counts(
            positive,
            negative,
            mode=config.evidence_count_mode,
            mass_saturation=config.evidence_mass_saturation,
            min_evidence_mass=config.min_evidence_mass,
        )
        bayesian = update_bayesian_lifecycle_chunks(
            tracker,
            controller,
            model,
            None,
            delta_a,
            delta_b,
            mass,
            timestamp=timestamp,
            min_evidence_mass=config.min_evidence_mass,
            chunk_size=config.bocd_chunk_size,
            closed_audit=closed_audit,
        )
        events.extend(bayesian["events"])
        evidence_result = _synthetic_evidence_result(delta_a, delta_b, mass)
        frames.append(
            frame_diagnostics(
                timestamp=timestamp,
                frame_name=f"synthetic_{timestamp:03d}",
                evidence=evidence_result,
                bayesian=bayesian,
                model=model,
                geometry_thawed_rows=0,
                base_checksum=combined_checksum(checksums),
                inactive_audit={
                    "passed": True,
                    "max_abs": 0.0,
                    "audited_pair_count": len(closed_audit.rows),
                    "total_closed_pair_count": closed_audit.total_closed_pairs,
                    "exhaustive": closed_audit.exhaustive,
                    "per_parameter_and_optimizer_state_max_abs": {},
                },
                frame_runtime_seconds=0.0,
                cuda_peak_memory_bytes=0,
                predicted_positive_fraction=0.0,
                algorithm=tracker.algorithm,
            )
        )
    return {
        "frame_metrics": frames,
        "events": events,
        "bayesian_stats": compact_npz_stats(frames),
        "base_drift": base_drift(before_base, model.base),
        "algorithm": tracker.algorithm,
    }


class SyntheticTemporalModel:
    """Small lifecycle model for detector-only smoke without CUDA/rendering."""

    def __init__(self, base: Any, n: int, max_states: int):
        self.base = base
        self.max_states = max_states
        self.state_valid = torch.zeros(n, max_states, dtype=torch.bool)
        self.state_start = torch.zeros(n, max_states)
        self.state_end = torch.full((n, max_states), float("inf"))
        self.state_status = torch.zeros(n, max_states, dtype=torch.int8)
        self.num_states = torch.zeros(n, dtype=torch.long)
        self.current_state_index = torch.full((n,), -1, dtype=torch.long)
        self.state_change_dc = torch.zeros(n, max_states, 1, 3)

    def state_parameter_items(self):
        return (("dc", self.state_change_dc),)

    def open_rows(self, rows, timestamp, initialization="zero"):
        rows = rows.long().flatten()
        if bool((self.current_state_index[rows] >= 0).any()):
            raise RuntimeError("row already open")
        slots = self.num_states[rows].clone()
        if bool((slots >= self.max_states).any()):
            raise RuntimeError("temporal state capacity exceeded")
        self.state_valid[rows, slots] = True
        self.state_status[rows, slots] = 1
        self.state_start[rows, slots] = float(timestamp)
        self.state_end[rows, slots] = float("inf")
        self.current_state_index[rows] = slots
        self.num_states[rows] = slots + 1
        return slots

    def close_rows(self, rows, timestamp):
        rows = rows.long().flatten()
        slots = self.current_state_index[rows].clone()
        active = slots >= 0
        rows, slots = rows[active], slots[active]
        self.state_end[rows, slots] = float(timestamp)
        self.state_status[rows, slots] = 2
        self.current_state_index[rows] = -1
        return slots

    def validate_lifecycle(self):
        return True


def run_detector_only_synthetic_smoke() -> dict[str, Any]:
    base = SimpleNamespace(
        _xyz=torch.zeros(1, 3),
        _features_dc=torch.zeros(1, 1, 3),
        _features_rest=torch.zeros(1, 0, 3),
        _opacity=torch.zeros(1, 1),
        _scaling=torch.zeros(1, 3),
        _rotation=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    )
    model = SyntheticTemporalModel(base, n=1, max_states=4)
    config = RunConfig(
        evidence_count_mode="raw",
        min_evidence_mass=0.1,
        bocd_mode="map_reset",
        hazard=0.2,
        expected_run_length=None,
        open_probability=0.6,
        close_probability=0.4,
        min_run_evidence=0.0,
        max_states=4,
        detector_only=True,
    )
    sequence = [
        (torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([1.0])),
        (torch.tensor([10.0]), torch.tensor([0.0]), torch.tensor([10.0])),
        (torch.tensor([8.0]), torch.tensor([0.0]), torch.tensor([8.0])),
        (torch.tensor([0.0]), torch.tensor([10.0]), torch.tensor([10.0])),
        (torch.tensor([10.0]), torch.tensor([0.0]), torch.tensor([10.0])),
    ]
    result = run_detector_sequence(sequence, model, config)
    result["final_num_states"] = model.num_states.tolist()
    result["final_current_state_index"] = model.current_state_index.tolist()
    return result


def build_causal_records(source_path: Path, *, max_frames: int | None = None) -> tuple[list[Any], list[str]]:
    """Build ordered image records without consulting GT or state boundaries."""
    from experiments.train_real_temporal_rchange import list_images

    image_dir = source_path / "inference_scene" / "images"
    names = list_images(image_dir)
    if max_frames is not None:
        names = names[:max_frames]
    records = [
        SimpleNamespace(
            global_index=index,
            segment_id=0,
            name=name,
            image_path=str(image_dir / name),
            mask_path="",
        )
        for index, name in enumerate(names)
    ]
    return records, names


def make_optimizer(
    model: TemporalGeometryChangeModel,
    config: RunConfig,
    args: argparse.Namespace,
) -> MaskedRowSlotAdam | None:
    if config.detector_only:
        for _name, parameter in model.state_parameter_items():
            parameter.requires_grad_(False)
        return None
    enabled = set(config.thaw_parameters)
    for name, parameter in model.state_parameter_items():
        parameter.requires_grad_(name in enabled)
    return MaskedRowSlotAdam(
        dict(model.state_parameter_items()),
        thaw_names=config.thaw_parameters,
        lrs={
            "dc": args.dc_lr,
            "xyz": args.xyz_lr,
            "opacity": args.opacity_lr,
            "scaling": args.scaling_lr,
            "rotation": args.rotation_lr,
        },
        eps=args.adam_eps,
    )


def run_online(args: argparse.Namespace) -> dict[str, Any]:
    from experiments.train_cue_temporal_rchange import (
        build_fixed_cue_views,
        load_fixed_camera_index,
        validate_cue_cache,
    )
    from experiments.train_real_temporal_rchange import (
        oscd_positive_sparsity_loss,
        seed_everything,
    )
    from gaussian_renderer import render_change_temporal
    from scene import GaussianModel

    config = run_config_from_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the Gaussian renderer")
    seed_everything(config.seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()

    records, names = build_causal_records(args.source_path, max_frames=args.max_frames)
    base_ply = (args.source_path / BASE_PLY_REL).resolve()
    cue_metadata = validate_cue_cache(args.cue_cache_root, base_ply, args.resolution)
    cameras = load_fixed_camera_index(args.fixed_cameras_json)
    timestamps = [int(record.global_index) for record in records]
    if timestamps != sorted(timestamps) or len(timestamps) != len(set(timestamps)):
        raise RuntimeError("views are not in strict global timestamp order")

    base = GaussianModel(sh_degree=3, active_sh_degree=0)
    base.load_ply_change(str(base_ply))
    base_before = base_snapshots(base)
    checksums = base_checksums(base)
    checksum = combined_checksum(checksums)
    model = TemporalGeometryChangeModel.from_gaussians(
        base, max_states=config.max_states
    )
    model.reset_all_lifespans_closed()
    optimizer = make_optimizer(model, config, args)

    estimated_filter_bytes = enforce_bocd_memory_limit(
        config, model.state_valid.shape[0], base._xyz.dtype
    )
    bayes_config = bocd_config(config)
    tracker = make_bocd_filter(
        config.bocd_mode,
        model.state_valid.shape[0],
        bayes_config,
        device=base._xyz.device,
        dtype=base._xyz.dtype,
    )
    controller = BayesianLifespanController(
        model, bayes_config, initialization=args.open_initialization
    )
    pipe = SimpleNamespace(
        compute_cov3D_python=False, convert_SHs_python=False, debug=False
    )
    background = torch.zeros(3, dtype=base._xyz.dtype, device=base._xyz.device)
    closed_audit = ClosedPairAudit(config.inactive_audit_max_pairs)

    frame_rows: list[dict[str, Any]] = []
    lifecycle_events: list[LifecycleEvent] = []
    predictions: list[np.ndarray] = []
    score_maps: list[np.ndarray] = []
    intrinsics: np.ndarray | None = None
    for record in records:
        frame_started = time.time()
        timestamp = int(record.global_index)
        # Load only the current image/cue. Future frame tensors are never
        # materialized before their timestamp, preserving the online contract.
        current_views, _current_pose, current_intrinsics = build_fixed_cue_views(
            [record], cameras, args.cue_cache_root, args.resolution
        )
        view = current_views[0]
        if intrinsics is None:
            intrinsics = current_intrinsics
        elif not np.array_equal(intrinsics, current_intrinsics):
            raise ValueError("fixed views do not share one intrinsic matrix")
        evidence = accumulate_change_evidence(
            view,
            base,
            pipe,
            background,
            view.candidate_map,
            cue_mode=config.bayes_cue_mode,
            cue_threshold=config.bayes_cue_threshold,
            cue_scale=config.bayes_cue_scale,
            count_mode=config.evidence_count_mode,
            mass_saturation=config.evidence_mass_saturation,
            min_evidence_mass=config.min_evidence_mass,
        )
        bayesian = update_bayesian_lifecycle_chunks(
            tracker,
            controller,
            model,
            optimizer,
            evidence.delta_a,
            evidence.delta_b,
            evidence.total_mass,
            timestamp=timestamp,
            min_evidence_mass=config.min_evidence_mass,
            chunk_size=config.bocd_chunk_size,
            closed_audit=closed_audit,
        )
        lifecycle_events.extend(bayesian["events"])

        audit_rows, audit_slots = closed_audit.coordinates(model.state_valid.device)
        inactive_before = capture_pair_audit(
            model, optimizer, audit_rows, audit_slots
        )
        active_pairs = current_pair_mask(model)
        active_row_count = int((model.current_state_index >= 0).sum().item())
        geometry_enabled = optimizer is not None and any(
            name != "dc" for name in config.thaw_parameters
        )
        geometry_thawed_rows = (
            active_row_count if geometry_enabled else 0
        )
        if optimizer is not None:
            # Clear stale gradients even on a frame where every lifespan has
            # just closed and no optimizer step will run.
            optimizer.zero_grad(set_to_none=True)
        package = render_change_temporal(
            view, model, pipe, background, timestamp=float(timestamp)
        )
        if optimizer is not None and active_row_count:
            for _update_index in range(config.updates_per_frame):
                optimizer.zero_grad(set_to_none=True)
                loss, _parts = oscd_positive_sparsity_loss(
                    view.training_target, package["render"]
                )
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(
                        f"non-finite online loss at timestamp {timestamp}"
                    )
                loss.backward()
                optimizer.step(active_pairs)
                package = render_change_temporal(
                    view, model, pipe, background, timestamp=float(timestamp)
                )
        inactive_audit = compare_pair_audit(
            inactive_before,
            model,
            optimizer,
            audit_rows,
            audit_slots,
            closed_audit,
        )
        if not inactive_audit["passed"]:
            raise RuntimeError(
                f"inactive temporal parameter drift at timestamp {timestamp}: "
                f"{inactive_audit}"
            )

        rendered_score = (
            package["render"].detach().mean(dim=0).clamp(0.0, 1.0).cpu()
        )
        prediction = (rendered_score.numpy() >= config.evaluation_threshold)
        predictions.append(prediction)
        if args.visualization_dir is not None:
            score_maps.append(
                rendered_score.mul(255.0).round().to(torch.uint8).numpy()
            )
        frame_rows.append(
            frame_diagnostics(
                timestamp=timestamp,
                frame_name=record.name,
                evidence=evidence,
                bayesian=bayesian,
                model=model,
                geometry_thawed_rows=geometry_thawed_rows,
                base_checksum=checksum,
                inactive_audit=inactive_audit,
                frame_runtime_seconds=time.time() - frame_started,
                cuda_peak_memory_bytes=torch.cuda.max_memory_allocated(),
                predicted_positive_fraction=float(prediction.mean()),
                algorithm=tracker.algorithm,
            )
        )

    # Everything below this line is post-inference diagnostics.  It cannot
    # affect evidence, Bayesian state, lifecycle, DC, or geometry updates.
    diagnostic_boundaries = tuple(
        int(value)
        for value in (
            ()
            if args.disable_boundary_diagnostics
            else args.boundaries
            if args.boundaries is not None
            else DEFAULT_BOUNDARIES
        )
    )
    if args.skip_post_inference_evaluation:
        metric_summary: dict[str, Any] = {
            "evaluated": False,
            "reason": "--skip-post-inference-evaluation",
        }
    else:
        metric_summary = evaluate_after_inference(
            args.source_path, records, predictions, frame_rows
        )
    event_summary = event_diagnostics_after_inference(
        lifecycle_events, diagnostic_boundaries
    )
    if event_summary["active_to_active_false_split_count"] != 0:
        raise RuntimeError("active-to-active false split count must be zero")

    drift = base_drift(base_before, base)
    if not drift["bitwise_equal"]:
        raise RuntimeError(f"immutable base tensor drift detected: {drift}")
    if intrinsics is None:
        raise RuntimeError("no causal frame was processed")
    visualization_summary = None
    if args.visualization_dir is not None:
        visualization_summary = write_visualization_artifacts(
            source_path=args.source_path,
            cue_cache_root=args.cue_cache_root,
            records=records,
            predictions=predictions,
            score_maps=score_maps,
            frame_rows=frame_rows,
            output_dir=args.visualization_dir,
            panel_width=int(args.visualization_panel_width),
            sample_count=int(args.visualization_sample_count),
            frame_indices=args.visualization_frame_indices,
        )

    summary = {
        "schema_version": 1,
        "script": "experiments/run_online_bayesian_lifespan_thaw.py",
        "contract": "causal_online_bayesian_lifespan_active_geometry",
        "runtime_seconds": time.time() - started,
        "run_config": asdict(config),
        "run_arguments": serializable_arguments(args),
        "algorithm": tracker.algorithm,
        "bocd_persistent_state_estimated_bytes": int(estimated_filter_bytes),
        "bocd_persistent_state_estimated_gib": estimated_filter_bytes / 1024**3,
        "gt_used_for_training": False,
        "manual_boundaries_used_for_inference": False,
        "gt_loaded_after_inference_only": not args.skip_post_inference_evaluation,
        "boundary_diagnostics_after_inference": event_summary,
        "metrics": metric_summary,
        "frames": len(frame_rows),
        "processed_frame_count": len(names),
        "event_count": len(lifecycle_events),
        "reopen_count": event_summary["reopen_count"],
        "active_to_active_false_split_count": 0,
        "inactive_geometry_drift_max_abs": max(
            (
                row["inactive_parameter_drift_audit"]["max_abs"]
                for row in frame_rows
            ),
            default=0.0,
        ),
        "inactive_audit_exhaustive": closed_audit.exhaustive,
        "base_tensor_drift": drift,
        "base_checksums": checksums,
        "base_combined_checksum": checksum,
        "fixed_topology": True,
        "optimized_parameters": list(config.thaw_parameters)
        if not config.detector_only
        else [],
        "detector_only": config.detector_only,
        "cue_mode_contract": (
            "source-faithful Beta-Bernoulli binary observations"
            if config.bayes_cue_mode == "binary"
            else "fractional/power-likelihood extension"
        ),
        "output_files": list(OUTPUT_FILES),
        "cue_cache_metadata": cue_metadata,
        "camera_intrinsics": intrinsics.tolist(),
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "visualization": visualization_summary,
    }
    checkpoint = {
        "schema_version": 1,
        "contract": summary["contract"],
        "base_ply": str(base_ply),
        "state_dict": model.state_dict(),
        "bocd_algorithm": tracker.algorithm,
        "bocd_state": tracker.state_dict(),
        "controller_state": controller.state_dict(),
        "optimizer_state": None if optimizer is None else optimizer.state_dict(),
        "metadata": summary,
    }
    write_outputs(
        args.output_dir, summary, frame_rows, lifecycle_events, checkpoint
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", type=Path, default=Path(DEFAULT_SOURCE))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fixed-cameras-json", type=Path, default=DEFAULT_FIXED_CAMERAS)
    parser.add_argument("--cue-cache-root", type=Path, default=DEFAULT_CUE_CACHE)
    parser.add_argument("--resolution", type=float, default=4.0)
    parser.add_argument("--max-frames", type=positive_int, default=None)
    parser.add_argument("--boundaries", nargs="+", type=int, default=None)
    parser.add_argument(
        "--disable-boundary-diagnostics",
        action="store_true",
        help="omit post-inference global boundary delay diagnostics, useful for individual ESCD scenes",
    )
    parser.add_argument("--skip-post-inference-evaluation", action="store_true")
    parser.add_argument("--evaluation-threshold", type=float, default=0.5)
    parser.add_argument(
        "--visualization-dir",
        type=Path,
        default=None,
        help="optional post-inference directory for PNG panels/timeline and visual_summary.json",
    )
    parser.add_argument(
        "--visualization-sample-count",
        "--visualization-max-panels",
        dest="visualization_sample_count",
        type=nonnegative_int,
        default=9,
        help="number of representative frames to write as visual panels",
    )
    parser.add_argument(
        "--visualization-panel-width",
        type=positive_int,
        default=320,
        help="width in pixels for each panel column",
    )
    parser.add_argument(
        "--visualization-frame-indices",
        type=parse_visualization_frame_indices,
        default=None,
        help="comma-separated zero-based frame indices to render as panels",
    )
    parser.add_argument("--bayes-cue-mode", choices=("binary", "soft"), default="binary")
    parser.add_argument("--bayes-cue-threshold", type=float, default=0.5)
    parser.add_argument("--bayes-cue-scale", type=float, default=1.0)
    parser.add_argument("--evidence-count-mode", choices=("raw", "capped"), default="capped")
    parser.add_argument("--evidence-mass-saturation", type=float, default=1.0)
    parser.add_argument("--min-evidence-mass", type=float, default=1e-6)
    parser.add_argument("--bocd-mode", choices=("exact", "map_reset"), default="map_reset")
    parser.add_argument("--prior-a", type=float, default=1.0)
    parser.add_argument("--prior-b", type=float, default=1.0)
    parser.add_argument("--expected-run-length", type=float, default=100.0)
    parser.add_argument("--hazard", type=float, default=None)
    parser.add_argument("--max-run-length", type=positive_int, default=128)
    parser.add_argument("--bocd-chunk-size", type=positive_int, default=65536)
    parser.add_argument("--exact-bocd-memory-limit-gb", type=float, default=8.0)
    parser.add_argument("--open-probability", type=float, default=0.6)
    parser.add_argument("--close-probability", type=float, default=0.4)
    parser.add_argument("--changepoint-probability", type=float, default=0.5)
    parser.add_argument("--min-run-evidence", type=float, default=1.0)
    parser.add_argument("--min-visible-observations", type=nonnegative_int, default=1)
    parser.add_argument("--max-states", type=positive_int, default=8)
    parser.add_argument("--thaw-parameters", type=parse_thaw_parameters, default=("dc",))
    parser.add_argument("--updates-per-frame", type=positive_int, default=1)
    parser.add_argument("--detector-only", action="store_true")
    parser.add_argument("--detector-only-smoke", action="store_true")
    parser.add_argument("--open-initialization", choices=("zero", "base"), default="zero")
    parser.add_argument("--inactive-audit-max-pairs", type=nonnegative_int, default=4096)
    parser.add_argument("--dc-lr", type=nonnegative_float, default=0.0025)
    parser.add_argument("--xyz-lr", type=nonnegative_float, default=0.00016)
    parser.add_argument("--opacity-lr", type=nonnegative_float, default=0.025)
    parser.add_argument("--scaling-lr", type=nonnegative_float, default=0.005)
    parser.add_argument("--rotation-lr", type=nonnegative_float, default=0.001)
    parser.add_argument("--adam-eps", type=float, default=1e-15)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.detector_only_smoke:
        result = run_detector_only_synthetic_smoke()
        summary = {
            "algorithm": result["algorithm"],
            "frames": len(result["frame_metrics"]),
            "event_count": len(result["events"]),
            "active_to_active_false_split_count": 0,
            "base_tensor_drift": result["base_drift"],
            "output_files": list(OUTPUT_FILES),
            "detector_only_smoke": True,
        }
        checkpoint = {"summary": summary}
        write_outputs(
            args.output_dir,
            summary,
            result["frame_metrics"],
            result["events"],
            checkpoint,
        )
        print(json.dumps(summary, indent=2))
        return 0
    summary = run_online(args)
    print(
        json.dumps(
            {
                "summary": str(args.output_dir / "summary.json"),
                "frames": summary["frames"],
                "algorithm": summary["algorithm"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
