"""State-specific Gaussian geometry for fixed-topology temporal change fields."""

from __future__ import annotations

import torch
from torch import nn

from .change_model import TemporalChangeModel


class TemporalGeometryChangeModel(TemporalChangeModel):
    """Adds state-specific geometry deltas without changing Gaussian identity."""

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
        s = self.max_states
        device = self.state_change_dc.device
        dtype = self.state_change_dc.dtype

        self.state_xyz_delta = nn.Parameter(
            torch.zeros((n, s, 3), device=device, dtype=dtype)
        )
        self.state_opacity_delta = nn.Parameter(
            torch.zeros((n, s, 1), device=device, dtype=dtype)
        )
        self.state_scaling_delta = nn.Parameter(
            torch.zeros((n, s, 3), device=device, dtype=dtype)
        )
        self.state_rotation_delta = nn.Parameter(
            torch.zeros((n, s, 4), device=device, dtype=dtype)
        )

    @torch.no_grad()
    def _initialize_state_slot(
        self, rows: torch.Tensor, slots: torch.Tensor, initialization: str
    ) -> None:
        super()._initialize_state_slot(rows, slots, initialization)
        self.state_xyz_delta[rows, slots].zero_()
        self.state_opacity_delta[rows, slots].zero_()
        self.state_scaling_delta[rows, slots].zero_()
        self.state_rotation_delta[rows, slots].zero_()

    @classmethod
    def from_gaussians(
        cls,
        base_change_gaussians,
        max_states: int = 4,
        initial_time: float = 0.0,
    ) -> "TemporalGeometryChangeModel":
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
                    "base and temporal geometry must share a device and dtype; "
                    "place the base first, then construct the sidecar"
                )

    @staticmethod
    def _select_state_tensor(
        tensor: torch.Tensor,
        indices: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        rows = torch.arange(tensor.shape[0], device=tensor.device)
        selected = tensor[rows, indices.clamp_min(0)]
        mask = active.reshape(active.shape[0], *([1] * (selected.ndim - 1)))
        return selected * mask

    def state_parameter_items(self) -> tuple[tuple[str, nn.Parameter], ...]:
        """Return every state-indexed trainable parameter in optimizer order."""
        return (
            ("dc", self.state_change_dc),
            ("xyz", self.state_xyz_delta),
            ("opacity", self.state_opacity_delta),
            ("scaling", self.state_scaling_delta),
            ("rotation", self.state_rotation_delta),
        )

    @torch.no_grad()
    def inherit_state_parameters(self, source: int, target: int) -> None:
        """Initialize one state slot from a completed predecessor slot.

        Lifespan metadata and ``state_valid`` are intentionally not copied:
        the target state keeps its own temporal interval and support gate while
        inheriting the same Gaussian identities and learned attributes.
        """
        for name, value in (("source", source), ("target", target)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0 or value >= self.max_states:
                raise ValueError(f"{name} must be in [0, {self.max_states})")
        if source == target:
            raise ValueError("source and target states must differ")
        for _name, parameter in self.state_parameter_items():
            parameter[:, target].copy_(parameter[:, source])

    def get_active_render_attributes(self, timestamp) -> dict[str, torch.Tensor]:
        """Return activated renderer inputs for the state selected at ``timestamp``."""
        self._validate_base_alignment()
        indices = self.get_active_state_indices(timestamp)
        active = indices >= 0

        dc = self._select_state_tensor(
            self.state_change_dc,
            indices,
            active,
        )
        xyz_delta = self._select_state_tensor(
            self.state_xyz_delta,
            indices,
            active,
        )
        opacity_delta = self._select_state_tensor(
            self.state_opacity_delta,
            indices,
            active,
        )
        scaling_delta = self._select_state_tensor(
            self.state_scaling_delta,
            indices,
            active,
        )
        rotation_delta = self._select_state_tensor(
            self.state_rotation_delta,
            indices,
            active,
        )

        return {
            "dc": dc,
            "xyz": self.base._xyz + xyz_delta,
            "opacity": self.base.opacity_activation(
                self.base._opacity + opacity_delta
            )
            * active[:, None],
            "scaling": self.base.scaling_activation(
                self.base._scaling + scaling_delta
            ),
            "rotation": self.base.rotation_activation(
                self.base._rotation + rotation_delta
            ),
            "active": active,
            "indices": indices,
        }
