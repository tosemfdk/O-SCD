"""Color encodings for interactive Bayesian lifespan-detector inspection.

The detector itself stays authoritative.  This module only translates its
per-Gaussian tensors into two visual layers:

* a committed lifecycle layer where NEVER_OPEN, OPEN, and CLOSED rows are
  black, green, and red respectively; and
* scalar heatmaps for current projected cue evidence and accumulated Bayesian
  instability.

Keeping the translation pure makes the color contract testable without CUDA or
the interactive viewer.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


BLACK = (0.0, 0.0, 0.0)
GREEN = (0.0, 1.0, 0.0)
RED = (1.0, 0.0, 0.0)


@dataclass(frozen=True)
class DetectorVisualState:
    """Per-Gaussian lifecycle colors and the normalized instability score."""

    colors: torch.Tensor
    instability: torch.Tensor
    never_open: torch.Tensor
    uncertain: torch.Tensor
    open: torch.Tensor
    closed: torch.Tensor


def _require_aligned_1d(reference: torch.Tensor, name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.ndim != 1 or value.shape != reference.shape:
        raise ValueError(f"{name} must have shape {tuple(reference.shape)}")
    if value.device != reference.device:
        raise ValueError(f"{name} must be on {reference.device}")


def normalized_log_bayes_factor(
    log_bayes_factor: torch.Tensor,
    bayes_factor_threshold: float,
) -> torch.Tensor:
    """Map RESET-vs-KEEP log Bayes factors to commit progress in ``[0, 1]``."""

    if not isinstance(log_bayes_factor, torch.Tensor):
        raise TypeError("log_bayes_factor must be a tensor")
    if log_bayes_factor.ndim != 1 or not torch.is_floating_point(log_bayes_factor):
        raise ValueError("log_bayes_factor must be a floating rank-1 tensor")
    if not bool(torch.isfinite(log_bayes_factor).all()):
        raise ValueError("log_bayes_factor must be finite")
    threshold = float(bayes_factor_threshold)
    if not math.isfinite(threshold) or threshold <= 1.0:
        raise ValueError("bayes_factor_threshold must be finite and greater than one")
    return torch.clamp(log_bayes_factor / math.log(threshold), 0.0, 1.0)


def black_green_yellow_red_heatmap(values: torch.Tensor) -> torch.Tensor:
    """Encode a normalized scalar as black-green-yellow-red RGB."""

    if not isinstance(values, torch.Tensor):
        raise TypeError("values must be a tensor")
    if values.ndim != 1 or not torch.is_floating_point(values):
        raise ValueError("values must be a floating rank-1 tensor")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("values must be finite")
    value = values.clamp(0.0, 1.0)
    red = torch.clamp(3.0 * value - 1.0, 0.0, 1.0)
    green = torch.where(
        value <= (2.0 / 3.0),
        torch.clamp(3.0 * value, 0.0, 1.0),
        torch.clamp(3.0 * (1.0 - value), 0.0, 1.0),
    )
    blue = torch.zeros_like(value)
    return torch.stack((red, green, blue), dim=1)


def detector_visual_state(
    *,
    current_active: torch.Tensor,
    ever_opened: torch.Tensor,
    candidate_active: torch.Tensor,
    last_log_bayes_factor: torch.Tensor,
    bayes_factor_threshold: float,
) -> DetectorVisualState:
    """Build committed black/green/red lifecycle colors.

    Candidate state remains available through ``uncertain`` and ``instability``
    but deliberately does not override the committed lifecycle color.
    """

    if not isinstance(current_active, torch.Tensor) or current_active.ndim != 1:
        raise ValueError("current_active must be a rank-1 tensor")
    if current_active.dtype != torch.bool:
        raise TypeError("current_active must be boolean")
    for name, value in (
        ("ever_opened", ever_opened),
        ("candidate_active", candidate_active),
        ("last_log_bayes_factor", last_log_bayes_factor),
    ):
        _require_aligned_1d(current_active, name, value)
    if ever_opened.dtype != torch.bool or candidate_active.dtype != torch.bool:
        raise TypeError("ever_opened and candidate_active must be boolean")
    if not torch.is_floating_point(last_log_bayes_factor):
        raise TypeError("last_log_bayes_factor must be floating point")
    if bool((current_active & ~ever_opened).any()):
        raise ValueError("an OPEN Gaussian must have opened at least once")

    progress = normalized_log_bayes_factor(
        last_log_bayes_factor, bayes_factor_threshold
    )
    instability = torch.where(candidate_active, progress, torch.zeros_like(progress))
    never_open = ~ever_opened & ~current_active
    closed = ever_opened & ~current_active
    open_rows = current_active

    colors = torch.zeros(
        (current_active.numel(), 3),
        dtype=last_log_bayes_factor.dtype,
        device=current_active.device,
    )
    colors[open_rows] = colors.new_tensor(GREEN)
    colors[closed] = colors.new_tensor(RED)

    return DetectorVisualState(
        colors=colors,
        instability=instability,
        never_open=never_open,
        uncertain=candidate_active,
        open=open_rows,
        closed=closed,
    )
