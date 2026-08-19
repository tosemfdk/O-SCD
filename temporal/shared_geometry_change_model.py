"""Temporal DC slots coupled through one mutable shared Gaussian geometry."""

from __future__ import annotations

import torch
from torch import nn

from .change_model import TemporalChangeModel


class TemporalSharedGeometryChangeModel(TemporalChangeModel):
    """Keep state-specific DC/lifespans while every state shares geometry.

    Unfrozen rows can be updated by later states and therefore permit geometric
    forgetting. Rows claimed by an earlier state can instead be marked in the
    persistent ``geometry_frozen`` buffer and masked after backward.
    """

    def __init__(
        self,
        base_change_gaussians,
        max_states: int = 4,
        initial_time: float = 0.0,
    ):
        super().__init__(
            base_change_gaussians,
            max_states=max_states,
            initial_time=initial_time,
        )
        n = self.state_change_dc.shape[0]
        device = self.state_change_dc.device
        dtype = self.state_change_dc.dtype
        self.shared_xyz_delta = nn.Parameter(torch.zeros((n, 3), device=device, dtype=dtype))
        self.shared_opacity_delta = nn.Parameter(torch.zeros((n, 1), device=device, dtype=dtype))
        self.shared_scaling_delta = nn.Parameter(torch.zeros((n, 3), device=device, dtype=dtype))
        self.shared_rotation_delta = nn.Parameter(torch.zeros((n, 4), device=device, dtype=dtype))
        self.register_buffer(
            "geometry_frozen",
            torch.zeros(n, device=device, dtype=torch.bool),
        )

    @classmethod
    def from_gaussians(
        cls,
        base_change_gaussians,
        max_states: int = 4,
        initial_time: float = 0.0,
    ) -> "TemporalSharedGeometryChangeModel":
        return cls(
            base_change_gaussians,
            max_states=max_states,
            initial_time=initial_time,
        )

    def _validate_base_alignment(self) -> None:
        super()._validate_base_alignment()
        for name in ("_xyz", "_opacity", "_scaling", "_rotation"):
            value = getattr(self.base, name)
            if (
                value.device != self.state_change_dc.device
                or value.dtype != self.state_change_dc.dtype
            ):
                raise RuntimeError(
                    "base and shared temporal geometry must share a device and dtype"
                )

    def shared_geometry_parameter_items(self) -> tuple[tuple[str, nn.Parameter], ...]:
        return (
            ("xyz", self.shared_xyz_delta),
            ("opacity", self.shared_opacity_delta),
            ("scaling", self.shared_scaling_delta),
            ("rotation", self.shared_rotation_delta),
        )

    def _validate_geometry_row_mask(self, mask: torch.Tensor) -> None:
        if mask.ndim != 1 or mask.dtype != torch.bool:
            raise ValueError("geometry row mask must be a rank-1 bool tensor")
        if mask.shape != self.geometry_frozen.shape:
            raise ValueError("geometry row mask has the wrong Gaussian count")
        if mask.device != self.geometry_frozen.device:
            raise ValueError("geometry row mask must share the model device")

    @torch.no_grad()
    def freeze_geometry_rows(self, mask: torch.Tensor) -> None:
        """Permanently mark newly owned Gaussian geometry rows as frozen."""
        self._validate_geometry_row_mask(mask)
        self.geometry_frozen.logical_or_(mask)

    def mask_frozen_geometry_gradients(self) -> dict[str, float]:
        """Zero frozen-row geometry gradients and report pre-mask maxima."""
        maxima: dict[str, float] = {}
        for name, parameter in self.shared_geometry_parameter_items():
            gradient = parameter.grad
            if gradient is None:
                maxima[name] = 0.0
                continue
            frozen_gradient = gradient[self.geometry_frozen]
            maxima[name] = (
                0.0
                if frozen_gradient.numel() == 0
                else float(frozen_gradient.detach().abs().max().item())
            )
            gradient[self.geometry_frozen] = 0
        return maxima

    def get_active_render_attributes(self, timestamp) -> dict[str, torch.Tensor]:
        self._validate_base_alignment()
        indices = self.get_active_state_indices(timestamp)
        active = indices >= 0
        rows = torch.arange(self.state_change_dc.shape[0], device=indices.device)
        dc = self.state_change_dc[rows, indices.clamp_min(0)] * active[:, None, None]
        # Keep the shared values identical for every state, but detach rows that
        # have no evidence in the current state.  Their zero opacity already
        # makes them visually inactive; this additionally makes gradient
        # isolation an explicit model contract rather than a rasterizer detail.
        def active_rows(parameter: torch.Tensor) -> torch.Tensor:
            row_mask = active.reshape((-1,) + (1,) * (parameter.ndim - 1))
            return torch.where(row_mask, parameter, parameter.detach())

        return {
            "dc": dc,
            "xyz": self.base._xyz + active_rows(self.shared_xyz_delta),
            "opacity": self.base.opacity_activation(
                self.base._opacity + active_rows(self.shared_opacity_delta)
            )
            * active[:, None],
            "scaling": self.base.scaling_activation(
                self.base._scaling + active_rows(self.shared_scaling_delta)
            ),
            "rotation": self.base.rotation_activation(
                self.base._rotation + active_rows(self.shared_rotation_delta)
            ),
            "active": active,
            "indices": indices,
        }
