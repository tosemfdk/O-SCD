"""Sidecar temporal change state model for Gaussian change features."""

import math
from numbers import Integral, Real

import torch
from torch import nn

from .lifespan import get_active_state_indices


class TemporalChangeModel(nn.Module):
    """Stores time-indexed change DC features without mutating Gaussian topology.

    Build the sidecar after placing the external base on its final device and
    dtype. Lifecycle status/count buffers are deferred until state transitions
    are introduced.
    """

    def __init__(
        self,
        base_change_gaussians,
        max_states: int = 4,
        initial_time: float = 0.0,
    ):
        super().__init__()
        if (
            isinstance(max_states, bool)
            or not isinstance(max_states, Integral)
            or max_states < 1
        ):
            raise ValueError("max_states must be a positive integer")
        if (
            isinstance(initial_time, bool)
            or not isinstance(initial_time, Real)
            or not math.isfinite(initial_time)
        ):
            raise ValueError("initial_time must be finite")
        if not hasattr(base_change_gaussians, "_features_dc"):
            raise AttributeError("base_change_gaussians must expose _features_dc")

        base_dc = base_change_gaussians._features_dc
        if (
            not isinstance(base_dc, torch.Tensor)
            or base_dc.ndim != 3
            or base_dc.shape[0] == 0
            or base_dc.shape[1:] != (1, 3)
        ):
            raise ValueError("base _features_dc must have shape [N, 1, 3]")
        if not torch.is_floating_point(base_dc):
            raise TypeError("base _features_dc must be floating")

        # Keep a side reference without registering the base as a submodule.
        object.__setattr__(self, "base", base_change_gaussians)
        self.max_states = int(max_states)
        self._freeze_base_tensors(base_change_gaussians)

        n_gaussians = base_dc.shape[0]
        device = base_dc.device
        dtype = base_dc.dtype

        state_change_dc = torch.zeros(
            (n_gaussians, self.max_states, 1, 3), device=device, dtype=dtype
        )
        state_change_dc[:, 0].copy_(base_dc.detach())
        self.state_change_dc = nn.Parameter(state_change_dc)

        shape = (n_gaussians, self.max_states)
        state_start = torch.zeros(shape, device=device, dtype=dtype)
        state_end = torch.full(shape, float("inf"), device=device, dtype=dtype)
        state_valid = torch.zeros(shape, device=device, dtype=torch.bool)
        state_start[:, 0] = torch.as_tensor(initial_time, device=device, dtype=dtype)
        state_valid[:, 0] = True
        self.register_buffer("state_start", state_start)
        self.register_buffer("state_end", state_end)
        self.register_buffer("state_valid", state_valid)

    @classmethod
    def from_gaussians(
        cls,
        base_change_gaussians,
        max_states: int = 4,
        initial_time: float = 0.0,
    ) -> "TemporalChangeModel":
        """Build a temporal sidecar initialized from existing change Gaussians."""
        return cls(base_change_gaussians, max_states=max_states, initial_time=initial_time)

    @staticmethod
    def _freeze_base_tensors(base_change_gaussians) -> None:
        names = (
            "_xyz",
            "_features_dc",
            "_features_rest",
            "_opacity",
            "_scaling",
            "_rotation",
        )
        for name in names:
            value = getattr(base_change_gaussians, name, None)
            if isinstance(value, torch.Tensor):
                value.requires_grad_(False)

    def get_active_state_indices(self, timestamp) -> torch.Tensor:
        """Return active state indices, using ``-1`` for inactive Gaussians."""
        return get_active_state_indices(
            timestamp,
            self.state_start,
            self.state_end,
            self.state_valid,
        )

    def _validate_base_alignment(self) -> None:
        base_dc = self.base._features_dc
        if (
            base_dc.device != self.state_change_dc.device
            or base_dc.dtype != self.state_change_dc.dtype
        ):
            raise RuntimeError(
                "base and temporal state must share a device and dtype; "
                "place the base first, then construct the sidecar"
            )

    def get_active_change(self, timestamp) -> tuple[torch.Tensor, torch.Tensor]:
        """Return active DC values and a per-Gaussian lifespan mask."""
        self._validate_base_alignment()
        indices = self.get_active_state_indices(timestamp)
        active = indices >= 0
        rows = torch.arange(self.state_change_dc.shape[0], device=self.state_change_dc.device)
        dc = self.state_change_dc[rows, indices.clamp_min(0)]
        return dc * active[:, None, None], active

    def get_active_change_dc(self, timestamp) -> torch.Tensor:
        """Return active ``[N, 1, 3]`` DC values, zeroing inactive Gaussians."""
        dc, _ = self.get_active_change(timestamp)
        return dc
