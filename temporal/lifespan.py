"""Temporal lifespan utilities for per-Gaussian change states."""

import math
from numbers import Real

import torch


def _validate_timestamp(timestamp: float) -> None:
    if isinstance(timestamp, bool) or not isinstance(timestamp, Real):
        raise TypeError("timestamp must be a finite real non-bool scalar")
    if not math.isfinite(timestamp):
        raise ValueError("timestamp must be finite")


def _validate_state_tensors(
    state_start: torch.Tensor,
    state_end: torch.Tensor,
    state_valid: torch.Tensor,
) -> None:
    if not all(isinstance(t, torch.Tensor) for t in (state_start, state_end, state_valid)):
        raise TypeError("state_start, state_end, and state_valid must be tensors")
    if state_start.ndim != 2 or state_end.ndim != 2 or state_valid.ndim != 2:
        raise ValueError("state tensors must be rank-2")
    if state_start.numel() == 0:
        raise ValueError("state tensors must be nonempty")
    if state_start.shape != state_end.shape or state_start.shape != state_valid.shape:
        raise ValueError("state tensors must have exactly the same shape")
    if not torch.is_floating_point(state_start) or not torch.is_floating_point(state_end):
        raise TypeError("state_start and state_end must be floating tensors")
    if state_start.dtype != state_end.dtype:
        raise TypeError("state_start and state_end must have the same dtype")
    if state_start.device != state_end.device or state_start.device != state_valid.device:
        raise ValueError("state tensors must be on the same device")
    if state_valid.dtype != torch.bool:
        raise TypeError("state_valid must be a bool tensor")
    if torch.any(state_valid & ~(state_start < state_end)):
        raise ValueError("every valid state slot must satisfy state_start < state_end")


def temporal_gate(
    timestamp: float,
    state_start: torch.Tensor,
    state_end: torch.Tensor,
    state_valid: torch.Tensor,
) -> torch.Tensor:
    """Return the raw change-state activation mask at ``timestamp``.

    Use ``get_active_state_indices`` at renderer boundaries that require
    validated metadata and at most one active state per Gaussian.
    """
    # Half-open intervals assign a changepoint frame to the new state.
    return state_valid & (state_start <= timestamp) & (timestamp < state_end)


def get_active_state_indices(
    timestamp: float,
    state_start: torch.Tensor,
    state_end: torch.Tensor,
    state_valid: torch.Tensor,
) -> torch.Tensor:
    """Return the active state index or ``-1`` for each Gaussian.

    Inputs must be exact matching ``[N, S]`` tensors; this function intentionally
    rejects shapes that would rely on broadcasting. A Gaussian may be unborn
    or outdated at a query time, but overlapping states are invalid.
    """
    _validate_state_tensors(state_start, state_end, state_valid)
    _validate_timestamp(timestamp)
    active = temporal_gate(timestamp, state_start, state_end, state_valid)
    active_count = active.sum(dim=1)
    if torch.any(active_count > 1):
        raise ValueError("Each Gaussian may have at most one active state")

    indices = active.to(torch.long).argmax(dim=1)
    return indices.masked_fill(active_count == 0, -1)
