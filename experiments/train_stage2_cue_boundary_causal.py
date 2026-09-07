#!/usr/bin/env python3
"""Train the Stage-2 cue boundary in strict prequential stream order.

For frame t, tau_t and width_t are predicted before teacher_t is exposed.  The
current teacher may update the model only after that prediction, so it can
affect frame t+1 and later but never frame t or an earlier frame.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.train_real_temporal_rchange import seed_everything
from experiments.train_stage2_cue_boundary import hard_metrics_from_histograms
from temporal.learnable_cue_boundary import HistogramBoundaryMLP, histogram_stage2_loss


DEFAULT_STATISTICS = Path("outputs/stage2_l1_power_sigmoid_boundary")
DEFAULT_OUTPUT = Path("outputs/stage2_l1_power_sigmoid_boundary_causal_prequential")


@dataclass(frozen=True)
class CausalBoundaryResult:
    tau: np.ndarray
    width: np.ndarray
    history: list[dict[str, Any]]
    audit: dict[str, Any]


def train_causal_prequential(
    *,
    model: HistogramBoundaryMLP,
    input_counts: torch.Tensor,
    loss_counts: torch.Tensor,
    teacher_sums: torch.Tensor,
    segment_names: tuple[str, ...],
    updates_per_arrival: int = 1,
    learning_rate: float = 0.002,
    trunk_lr_scale: float = 0.5,
    weight_decay: float = 1.0e-4,
    soft_iou_weight: float = 1.0,
    anchor_weight: float = 0.25,
    temporal_weight: float = 0.02,
    head_warmup_frames: int = 40,
    anchor_decay_frames: int = 100,
) -> CausalBoundaryResult:
    """Return pre-update boundaries using only teachers from frames < t."""

    frames = int(input_counts.shape[0])
    if frames < 1:
        raise ValueError("at least one frame is required")
    if (
        loss_counts.shape != teacher_sums.shape
        or loss_counts.ndim != 2
        or input_counts.ndim != 2
        or input_counts.shape[0] != frames
        or len(segment_names) != frames
    ):
        raise ValueError("causal boundary statistics have incompatible shapes")
    if input_counts.shape[1] != model.bins:
        raise ValueError("input histogram bins do not match the model")
    if updates_per_arrival < 1:
        raise ValueError("updates_per_arrival must be positive")

    optimizer = torch.optim.AdamW(
        [
            {"params": model.trunk.parameters(), "lr": 0.0, "name": "trunk"},
            {
                "params": model.head.parameters(),
                "lr": learning_rate * 0.1,
                "name": "head",
            },
        ],
        weight_decay=weight_decay,
    )
    centers = (
        torch.arange(
            loss_counts.shape[1],
            device=loss_counts.device,
            dtype=loss_counts.dtype,
        )
        + 0.5
    ) / loss_counts.shape[1]
    tau_before: list[float] = []
    width_before: list[float] = []
    history: list[dict[str, Any]] = []
    maximum_teacher_index_before_prediction: list[int] = []

    for frame_index in range(frames):
        model.eval()
        with torch.no_grad():
            tau_t, width_t = model(input_counts[frame_index : frame_index + 1])
        tau_before.append(float(tau_t.item()))
        width_before.append(float(width_t.item()))
        maximum_teacher_index_before_prediction.append(frame_index - 1)

        # teacher_t becomes available only after Q_t has been frozen.  These
        # updates can therefore influence only frame t+1 and later.
        model.train()
        last_row: dict[str, float] | None = None
        observed_frames = frame_index + 1
        head_progress = min(1.0, observed_frames / max(1, head_warmup_frames))
        optimizer.param_groups[1]["lr"] = learning_rate * (
            0.1 + 0.9 * head_progress
        )
        optimizer.param_groups[0]["lr"] = (
            0.0
            if observed_frames <= head_warmup_frames
            else learning_rate * trunk_lr_scale
        )
        for _ in range(updates_per_arrival):
            optimizer.zero_grad(set_to_none=True)
            current_tau, current_width = model(
                input_counts[frame_index : frame_index + 1]
            )
            prediction = model.remap(
                centers[None], current_tau[:, None], current_width[:, None]
            )
            stage2 = histogram_stage2_loss(
                prediction,
                loss_counts[frame_index : frame_index + 1],
                teacher_sums[frame_index : frame_index + 1],
                soft_iou_weight=soft_iou_weight,
            )
            anchor_progress = min(
                1.0, observed_frames / max(1, anchor_decay_frames)
            )
            current_anchor_weight = anchor_weight * 0.5 * (
                1.0 + math.cos(math.pi * anchor_progress)
            )
            anchor = (
                ((current_tau - model.initial_tau) / model.initial_width).square()
                + (
                    (current_width - model.initial_width) / model.initial_width
                ).square()
            ).mean()
            temporal = current_tau.new_zeros(())
            if (
                frame_index > 0
                and segment_names[frame_index] == segment_names[frame_index - 1]
            ):
                pair_tau, pair_width = model(
                    input_counts[frame_index - 1 : frame_index + 1]
                )
                temporal = (
                    ((pair_tau[1] - pair_tau[0]) / model.initial_width).square()
                    + (
                        (pair_width[1] - pair_width[0]) / model.initial_width
                    ).square()
                )
            objective = (
                stage2.loss
                + current_anchor_weight * anchor
                + temporal_weight * temporal
            )
            objective.backward()
            optimizer.step()
            last_row = {
                "objective": float(objective.detach().item()),
                "stage2": float(stage2.loss.detach().item()),
                "balanced_bce": float(stage2.balanced_bce.detach().item()),
                "soft_iou": float(stage2.soft_iou.detach().item()),
                "anchor_weight": float(current_anchor_weight),
                "temporal": float(temporal.detach().item()),
                "head_lr": float(optimizer.param_groups[1]["lr"]),
                "trunk_lr": float(optimizer.param_groups[0]["lr"]),
            }
        if last_row is None:
            raise RuntimeError("causal update unexpectedly produced no step")
        history.append(
            {
                "frame_index": frame_index,
                "segment": segment_names[frame_index],
                "tau_before_update": tau_before[-1],
                "width_before_update": width_before[-1],
                "teacher_exposed_after_prediction": True,
                **last_row,
            }
        )

    return CausalBoundaryResult(
        tau=np.asarray(tau_before, dtype=np.float64),
        width=np.asarray(width_before, dtype=np.float64),
        history=history,
        audit={
            "prediction_order": "predict Q_t, then expose teacher_t, then update",
            "future_teacher_accesses": 0,
            "current_teacher_used_for_current_prediction": False,
            "maximum_teacher_index_before_prediction": maximum_teacher_index_before_prediction,
            "first_frame_uses_fixed_initial_boundary": bool(
                math.isclose(
                    tau_before[0], model.initial_tau, rel_tol=0.0, abs_tol=1.0e-7
                )
                and math.isclose(
                    width_before[0],
                    model.initial_width,
                    rel_tol=0.0,
                    abs_tol=1.0e-7,
                )
            ),
            "updates_per_arrival": int(updates_per_arrival),
            "head_warmup_frames": int(head_warmup_frames),
            "anchor_decay_frames": int(anchor_decay_frames),
        },
    )


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics-dir", type=Path, default=DEFAULT_STATISTICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--updates-per-arrival", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--trunk-lr-scale", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--soft-iou-weight", type=float, default=1.0)
    parser.add_argument("--anchor-weight", type=float, default=0.25)
    parser.add_argument("--temporal-weight", type=float, default=0.02)
    parser.add_argument("--head-warmup-frames", type=int, default=40)
    parser.add_argument("--anchor-decay-frames", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for name in ("updates_per_arrival", "head_warmup_frames", "anchor_decay_frames"):
        if int(getattr(args, name)) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> None:
    args = parse_args()
    seed_everything(int(args.seed))
    metadata = json.loads(
        (args.statistics_dir / "boundary_statistics.json").read_text(
            encoding="utf-8"
        )
    )
    with np.load(args.statistics_dir / "boundary_statistics.npz") as payload:
        input_counts_np = payload["input_counts"]
        loss_counts_np = payload["loss_counts"]
        teacher_sums_np = payload["teacher_sums"]
        gt_sums_np = payload["gt_sums"]
        frame_names = tuple(str(value) for value in payload["frame_names"])
        segment_names = tuple(str(value) for value in payload["segment_names"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HistogramBoundaryMLP(
        bins=int(input_counts_np.shape[1]),
        initial_tau=0.25,
        initial_width=0.10,
        edge_probability=0.05,
    ).to(device)
    result = train_causal_prequential(
        model=model,
        input_counts=torch.from_numpy(input_counts_np).to(
            device=device, dtype=torch.float32
        ),
        loss_counts=torch.from_numpy(loss_counts_np).to(
            device=device, dtype=torch.float32
        ),
        teacher_sums=torch.from_numpy(teacher_sums_np).to(
            device=device, dtype=torch.float32
        ),
        segment_names=segment_names,
        updates_per_arrival=int(args.updates_per_arrival),
        learning_rate=float(args.learning_rate),
        trunk_lr_scale=float(args.trunk_lr_scale),
        weight_decay=float(args.weight_decay),
        soft_iou_weight=float(args.soft_iou_weight),
        anchor_weight=float(args.anchor_weight),
        temporal_weight=float(args.temporal_weight),
        head_warmup_frames=int(args.head_warmup_frames),
        anchor_decay_frames=int(args.anchor_decay_frames),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema_version": 2,
        "remap": "sigmoid",
        "training_mode": "causal_prequential",
        "prediction_order": "theta_(t-1)+histogram_t -> tau_t,width_t; teacher_t updates theta_t afterwards",
        "width_semantics": "output is 0.05/0.95 at tau-width/tau+width",
        "edge_probability": 0.05,
        "cue_scale": 2.0,
        "cue_formula": metadata["cue_formula"],
        "frames": {
            name: {"tau": float(tau), "width": float(width)}
            for name, tau, width in zip(frame_names, result.tau, result.width)
        },
    }
    boundary_path = args.output_dir / "learned_boundaries_causal.json"
    boundary_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "training_history.json").write_text(
        json.dumps(result.history, indent=2) + "\n", encoding="utf-8"
    )
    heuristic_tau = np.full_like(result.tau, 0.25)
    evaluation = {
        "heuristic": hard_metrics_from_histograms(
            loss_counts_np, gt_sums_np, heuristic_tau
        ),
        "causal_prequential": hard_metrics_from_histograms(
            loss_counts_np, gt_sums_np, result.tau
        ),
    }
    segments = {}
    for segment in dict.fromkeys(segment_names):
        mask = np.asarray([name == segment for name in segment_names])
        segments[segment] = {
            "heuristic": hard_metrics_from_histograms(
                loss_counts_np[mask], gt_sums_np[mask], heuristic_tau[mask]
            ),
            "causal_prequential": hard_metrics_from_histograms(
                loss_counts_np[mask], gt_sums_np[mask], result.tau[mask]
            ),
        }
    summary = {
        "experiment_scope": "strict causal prequential Stage-2 calibration replay",
        "teacher": metadata["teacher"],
        "teacher_availability_warning": (
            "The saved online-at-arrival teacher is exposed only after Q_t is frozen. "
            "A deployment system must still provide an equivalent causal teacher or "
            "self-supervised update signal."
        ),
        "frames": len(frame_names),
        "model": model.configuration(),
        "training": vars(args),
        "causal_audit": result.audit,
        "tau": _summary(result.tau),
        "width": _summary(result.width),
        "gt_evaluation_only_histogram_approximation": {
            "overall": evaluation,
            "segments": segments,
        },
        "artifacts": {
            "boundary_json": str(boundary_path),
            "training_history": str(args.output_dir / "training_history.json"),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps({"boundary": str(boundary_path), "metrics": evaluation}, indent=2))


if __name__ == "__main__":
    main()
