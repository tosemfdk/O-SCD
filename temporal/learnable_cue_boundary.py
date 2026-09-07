"""Histogram-conditioned Stage-2 calibration of a soft change cue."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from temporal.change_cue_fusion import sigmoid_soft_binarize


@dataclass(frozen=True)
class Stage2BoundaryLoss:
    """Components of the frozen-teacher cue calibration objective."""

    loss: torch.Tensor
    balanced_bce: torch.Tensor
    soft_iou_loss: torch.Tensor
    soft_iou: torch.Tensor


class HistogramBoundaryMLP(nn.Module):
    """Predict per-frame sigmoid ``tau`` and ``width`` from a cue histogram.

    The final layer is initialized to zero.  Consequently, every histogram
    starts at the supplied heuristic boundary instead of an arbitrary network
    prediction.  Training can first update only ``head`` before unfreezing the
    histogram-dependent ``trunk``.
    """

    def __init__(
        self,
        bins: int = 64,
        *,
        hidden: tuple[int, int] = (32, 16),
        initial_tau: float = 0.25,
        initial_width: float = 0.10,
        tau_bounds: tuple[float, float] = (0.05, 0.75),
        width_bounds: tuple[float, float] = (0.02, 0.30),
        output_logit_span: float = 4.0,
        edge_probability: float = 0.05,
    ) -> None:
        super().__init__()
        if bins < 2:
            raise ValueError("bins must be at least two")
        if len(hidden) != 2 or min(hidden) < 1:
            raise ValueError("hidden must contain two positive widths")
        self._validate_interval("tau_bounds", tau_bounds)
        self._validate_interval("width_bounds", width_bounds)
        if not tau_bounds[0] < initial_tau < tau_bounds[1]:
            raise ValueError("initial_tau must lie inside tau_bounds")
        if not width_bounds[0] < initial_width < width_bounds[1]:
            raise ValueError("initial_width must lie inside width_bounds")
        if not math.isfinite(output_logit_span) or output_logit_span <= 0.0:
            raise ValueError("output_logit_span must be finite and positive")
        if not 0.0 < edge_probability < 0.5:
            raise ValueError("edge_probability must lie in (0,0.5)")

        self.bins = int(bins)
        self.initial_tau = float(initial_tau)
        self.initial_width = float(initial_width)
        self.tau_bounds = tuple(float(value) for value in tau_bounds)
        self.width_bounds = tuple(float(value) for value in width_bounds)
        self.output_logit_span = float(output_logit_span)
        self.edge_probability = float(edge_probability)
        self.trunk = nn.Sequential(
            nn.Linear(2 * self.bins, hidden[0]),
            nn.SiLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.SiLU(),
        )
        self.head = nn.Linear(hidden[1], 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        self.register_buffer(
            "_initial_logits",
            torch.tensor(
                [
                    self._interval_logit(self.initial_tau, self.tau_bounds),
                    self._interval_logit(self.initial_width, self.width_bounds),
                ],
                dtype=torch.float32,
            ),
        )

    @staticmethod
    def _validate_interval(name: str, bounds: tuple[float, float]) -> None:
        if len(bounds) != 2 or not all(math.isfinite(float(value)) for value in bounds):
            raise ValueError(f"{name} must contain two finite values")
        if not 0.0 < float(bounds[0]) < float(bounds[1]) < 1.0:
            raise ValueError(f"{name} must lie inside (0,1)")

    @staticmethod
    def _interval_logit(value: float, bounds: tuple[float, float]) -> float:
        unit = (float(value) - float(bounds[0])) / (float(bounds[1]) - float(bounds[0]))
        return math.log(unit / (1.0 - unit))

    @staticmethod
    def _decode_interval(
        logit: torch.Tensor,
        bounds: tuple[float, float],
    ) -> torch.Tensor:
        low, high = bounds
        return low + (high - low) * torch.sigmoid(logit)

    def forward(self, histogram_counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if histogram_counts.ndim != 2 or histogram_counts.shape[1] != self.bins:
            raise ValueError(f"histogram_counts must have shape [B,{self.bins}]")
        if not torch.is_floating_point(histogram_counts):
            histogram_counts = histogram_counts.float()
        if not bool(torch.isfinite(histogram_counts).all()):
            raise ValueError("histogram_counts must be finite")
        if bool((histogram_counts < 0.0).any()):
            raise ValueError("histogram_counts must be nonnegative")
        mass = histogram_counts.sum(dim=1, keepdim=True)
        if bool((mass <= 0.0).any()):
            raise ValueError("every histogram must contain positive mass")

        histogram = histogram_counts / mass
        cdf = histogram.cumsum(dim=1)
        raw = self.head(self.trunk(torch.cat((histogram, cdf), dim=1)))
        logits = self._initial_logits.to(dtype=raw.dtype) + self.output_logit_span * torch.tanh(raw)
        tau = self._decode_interval(logits[:, 0], self.tau_bounds)
        width = self._decode_interval(logits[:, 1], self.width_bounds)
        return tau, width

    def remap(
        self,
        cue: torch.Tensor,
        tau: torch.Tensor,
        width: torch.Tensor,
    ) -> torch.Tensor:
        return sigmoid_soft_binarize(
            cue,
            tau=tau,
            width=width,
            edge_probability=self.edge_probability,
        )

    def configuration(self) -> dict[str, object]:
        return {
            "bins": self.bins,
            "hidden": [self.trunk[0].out_features, self.trunk[2].out_features],
            "initial_tau": self.initial_tau,
            "initial_width": self.initial_width,
            "tau_bounds": list(self.tau_bounds),
            "width_bounds": list(self.width_bounds),
            "output_logit_span": self.output_logit_span,
            "edge_probability": self.edge_probability,
        }


def histogram_stage2_loss(
    prediction_by_bin: torch.Tensor,
    bin_counts: torch.Tensor,
    teacher_sums: torch.Tensor,
    *,
    soft_iou_weight: float = 1.0,
    eps: float = 1e-6,
) -> Stage2BoundaryLoss:
    """Compute frame-balanced BCE + soft IoU from binned frozen targets."""

    if prediction_by_bin.shape != bin_counts.shape or teacher_sums.shape != bin_counts.shape:
        raise ValueError("prediction, counts, and teacher_sums must share shape [B,K]")
    if prediction_by_bin.ndim != 2:
        raise ValueError("histogram loss tensors must have shape [B,K]")
    if not all(torch.is_floating_point(value) for value in (prediction_by_bin, bin_counts, teacher_sums)):
        raise TypeError("histogram loss tensors must be floating point")
    if not (
        prediction_by_bin.device == bin_counts.device == teacher_sums.device
        and prediction_by_bin.dtype == bin_counts.dtype == teacher_sums.dtype
    ):
        raise ValueError("histogram loss tensors must share device and dtype")
    if not math.isfinite(float(soft_iou_weight)) or soft_iou_weight < 0.0:
        raise ValueError("soft_iou_weight must be finite and nonnegative")
    if not bool(
        torch.isfinite(prediction_by_bin).all()
        and torch.isfinite(bin_counts).all()
        and torch.isfinite(teacher_sums).all()
    ):
        raise ValueError("histogram loss tensors must be finite")
    if bool((bin_counts < 0.0).any() or (teacher_sums < 0.0).any()):
        raise ValueError("histogram counts and teacher sums must be nonnegative")
    if bool((teacher_sums > bin_counts + 1e-4).any()):
        raise ValueError("teacher_sums cannot exceed bin_counts")

    prediction = prediction_by_bin.clamp(eps, 1.0 - eps)
    positive_mass = teacher_sums.sum(dim=1).clamp_min(eps)
    negative_weights = bin_counts - teacher_sums
    negative_mass = negative_weights.sum(dim=1).clamp_min(eps)
    positive_nll = -(teacher_sums * prediction.log()).sum(dim=1) / positive_mass
    negative_nll = -(negative_weights * torch.log1p(-prediction)).sum(dim=1) / negative_mass
    balanced_bce_per_frame = 0.5 * (positive_nll + negative_nll)

    intersection = (prediction_by_bin * teacher_sums).sum(dim=1)
    prediction_mass = (prediction_by_bin * bin_counts).sum(dim=1)
    union = prediction_mass + positive_mass - intersection
    soft_iou = intersection / union.clamp_min(eps)
    soft_iou_loss_per_frame = 1.0 - soft_iou
    loss_per_frame = balanced_bce_per_frame + float(soft_iou_weight) * soft_iou_loss_per_frame
    return Stage2BoundaryLoss(
        loss=loss_per_frame.mean(),
        balanced_bce=balanced_bce_per_frame.mean(),
        soft_iou_loss=soft_iou_loss_per_frame.mean(),
        soft_iou=soft_iou.mean(),
    )
