"""Row-wise immutable anchors for shared temporal Gaussian geometry."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch


GeometryItems = Iterable[tuple[str, torch.Tensor]]


def _validated_items(
    items: GeometryItems,
    frozen_mask: torch.Tensor,
) -> tuple[tuple[str, torch.Tensor], ...]:
    materialized = tuple(items)
    if frozen_mask.ndim != 1 or frozen_mask.dtype != torch.bool:
        raise ValueError("frozen_mask must be a rank-1 bool tensor")
    for name, parameter in materialized:
        if parameter.ndim < 1:
            raise ValueError(f"{name} must have a Gaussian row dimension")
        if parameter.shape[0] != frozen_mask.shape[0]:
            raise ValueError(f"{name} and frozen_mask have different row counts")
        if parameter.device != frozen_mask.device:
            raise ValueError(f"{name} and frozen_mask must share a device")
    return materialized


@torch.no_grad()
def capture_frozen_rows(
    items: GeometryItems,
    frozen_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Snapshot the rows that become permanently owned by an earlier state."""
    materialized = _validated_items(items, frozen_mask)
    return {
        name: parameter[frozen_mask].detach().clone()
        for name, parameter in materialized
    }


def mask_frozen_row_gradients(
    items: GeometryItems,
    frozen_mask: torch.Tensor,
) -> dict[str, float]:
    """Zero frozen-row gradients and report their pre-mask maximum magnitude."""
    materialized = _validated_items(items, frozen_mask)
    maxima: dict[str, float] = {}
    for name, parameter in materialized:
        gradient = parameter.grad
        if gradient is None or not bool(frozen_mask.any()):
            maxima[name] = 0.0
            continue
        frozen_gradient = gradient[frozen_mask]
        maxima[name] = (
            0.0
            if frozen_gradient.numel() == 0
            else float(frozen_gradient.detach().abs().max().item())
        )
        gradient[frozen_mask] = 0
    return maxima


@torch.no_grad()
def restore_frozen_rows(
    items: GeometryItems,
    frozen_mask: torch.Tensor,
    anchors: Mapping[str, torch.Tensor],
) -> None:
    """Project frozen rows back to their first-state immutable anchors."""
    materialized = _validated_items(items, frozen_mask)
    expected_names = {name for name, _parameter in materialized}
    if set(anchors) != expected_names:
        raise ValueError("anchors do not match the shared geometry parameters")
    for name, parameter in materialized:
        anchor = anchors[name]
        expected_shape = parameter[frozen_mask].shape
        if anchor.shape != expected_shape:
            raise ValueError(f"{name} anchor has the wrong shape")
        if anchor.device != parameter.device or anchor.dtype != parameter.dtype:
            raise ValueError(f"{name} anchor must match parameter device and dtype")
        parameter[frozen_mask] = anchor


@torch.no_grad()
def max_frozen_row_drift(
    items: GeometryItems,
    frozen_mask: torch.Tensor,
    anchors: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    """Return exact per-attribute frozen-row drift from the saved anchors."""
    materialized = _validated_items(items, frozen_mask)
    drift: dict[str, float] = {}
    for name, parameter in materialized:
        difference = parameter[frozen_mask] - anchors[name]
        drift[name] = (
            0.0
            if difference.numel() == 0
            else float(difference.detach().abs().max().item())
        )
    return drift
