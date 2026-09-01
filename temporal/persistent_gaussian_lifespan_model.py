"""Lifespan visibility over one persistent mutable R_change Gaussian bank.

Unlike :class:`TemporalGeometryChangeModel`, this model does not allocate any
state-local learnable tensor.  Lifespan slots retain only interval metadata.
The raw Gaussian parameters are shared across every OPEN/CLOSE episode, so a
row that closes with a learned change DC keeps that value and resumes from it
when a later lifespan opens.

The mutable render bank must be distinct from the immutable reference bank
used by ``change_evidence``.  This class deliberately does not freeze its
``base`` because those raw tensors are the representation being optimized.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any

import torch
from torch import nn
from utils.sh_utils import RGB2SH

from .change_model import CLOSED, EMPTY, OPEN
from .lifespan import get_active_state_indices


class PersistentGaussianLifespanModel(nn.Module):
    """Track lifespan intervals while sharing one mutable parameter set per GS."""

    persistent_parameters_across_lifespans = True

    def __init__(
        self,
        mutable_change_gaussians: Any,
        max_states: int = 4,
        initial_time: float = 0.0,
    ) -> None:
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

        fields = (
            ("change_dc", "_features_dc", 3),
            ("xyz", "_xyz", 2),
            ("features_rest", "_features_rest", 3),
            ("opacity", "_opacity", 2),
            ("scaling", "_scaling", 2),
            ("rotation", "_rotation", 2),
        )
        tensors: dict[str, nn.Parameter] = {}
        n_gaussians: int | None = None
        device: torch.device | None = None
        dtype: torch.dtype | None = None
        for public_name, raw_name, ndim in fields:
            value = getattr(mutable_change_gaussians, raw_name, None)
            if not isinstance(value, nn.Parameter):
                raise TypeError(f"base {raw_name} must be an nn.Parameter")
            if value.ndim != ndim or not torch.is_floating_point(value):
                raise ValueError(
                    f"base {raw_name} must be a floating rank-{ndim} tensor"
                )
            if n_gaussians is None:
                n_gaussians = int(value.shape[0])
                device = value.device
                dtype = value.dtype
            elif (
                int(value.shape[0]) != n_gaussians
                or value.device != device
                or value.dtype != dtype
            ):
                raise ValueError("all mutable Gaussian tensors must align by row/device/dtype")
            tensors[public_name] = value
        assert n_gaussians is not None and device is not None and dtype is not None
        if n_gaussians < 1:
            raise ValueError("mutable Gaussian bank must contain at least one row")
        if tuple(tensors["change_dc"].shape[1:]) != (1, 3):
            raise ValueError("base _features_dc must have shape [N, 1, 3]")
        if tensors["xyz"].shape[1:] != (3,):
            raise ValueError("base _xyz must have shape [N, 3]")
        if tensors["features_rest"].shape[-1:] != (3,):
            raise ValueError("base _features_rest must have shape [N, K, 3]")
        if tensors["opacity"].shape[1:] != (1,):
            raise ValueError("base _opacity must have shape [N, 1]")
        if tensors["scaling"].shape[1:] != (3,):
            raise ValueError("base _scaling must have shape [N, 3]")
        if tensors["rotation"].shape[1:] != (4,):
            raise ValueError("base _rotation must have shape [N, 4]")

        object.__setattr__(self, "base", mutable_change_gaussians)
        self.max_states = int(max_states)

        # Register the exact Parameter objects already owned by the mutable
        # Gaussian bank.  The renderer and optimizer therefore see identical
        # storage rather than a copied delta representation.
        self.change_dc = tensors["change_dc"]
        self.xyz = tensors["xyz"]
        self.features_rest = tensors["features_rest"]
        self.opacity = tensors["opacity"]
        self.scaling = tensors["scaling"]
        self.rotation = tensors["rotation"]

        shape = (n_gaussians, self.max_states)
        state_start = torch.zeros(shape, device=device, dtype=dtype)
        state_end = torch.full(shape, float("inf"), device=device, dtype=dtype)
        state_valid = torch.zeros(shape, device=device, dtype=torch.bool)
        state_status = torch.zeros(shape, device=device, dtype=torch.int8)
        state_start[:, 0] = float(initial_time)
        state_valid[:, 0] = True
        state_status[:, 0] = OPEN
        self.register_buffer("state_start", state_start)
        self.register_buffer("state_end", state_end)
        self.register_buffer("state_valid", state_valid)
        self.register_buffer("state_status", state_status)
        self.register_buffer(
            "num_states", torch.ones(n_gaussians, device=device, dtype=torch.long)
        )
        self.register_buffer(
            "current_state_index",
            torch.zeros(n_gaussians, device=device, dtype=torch.long),
        )

    @classmethod
    def from_gaussians(
        cls,
        mutable_change_gaussians: Any,
        max_states: int = 4,
        initial_time: float = 0.0,
    ) -> "PersistentGaussianLifespanModel":
        return cls(
            mutable_change_gaussians,
            max_states=max_states,
            initial_time=initial_time,
        )

    def persistent_parameter_items(self) -> tuple[tuple[str, nn.Parameter], ...]:
        """Return direct raw Gaussian parameters in optimizer order."""
        return (
            ("dc", self.change_dc),
            ("xyz", self.xyz),
            ("features_rest", self.features_rest),
            ("opacity", self.opacity),
            ("scaling", self.scaling),
            ("rotation", self.rotation),
        )

    def get_active_state_indices(self, timestamp: float) -> torch.Tensor:
        return get_active_state_indices(
            timestamp,
            self.state_start,
            self.state_end,
            self.state_valid,
        )

    @staticmethod
    def _active_rows(parameter: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        mask = active.reshape((-1,) + (1,) * (parameter.ndim - 1))
        return torch.where(mask, parameter, parameter.detach())

    def get_active_render_attributes(self, timestamp: float) -> dict[str, torch.Tensor]:
        """Return direct parameters, gradient-detaching every inactive row."""
        return self._get_render_attributes(timestamp, include_never_open=False)

    def get_open_or_never_open_render_attributes(
        self,
        timestamp: float,
        *,
        train_never_open_dc_opacity: bool = False,
        black_never_open: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Render OPEN rows plus fixed zero-change occluders that never opened.

        A never-opened row still represents the reference surface.  Keeping its
        opacity in the compositor restores alpha/transmittance occlusion. Raw
        SH DC zero renders as the neutral value RGB 0.5 before alpha
        compositing because the rasterizer adds 0.5 after SH evaluation. The
        explicit ``train_never_open_dc_opacity`` ablation lets their stored DC
        and opacity adapt while geometry remains detached.

        ``black_never_open`` instead overrides every never-open row with the
        raw degree-zero SH value for RGB zero.  Geometry and opacity still
        occlude OPEN rows, but the occluder itself contributes no change color.

        Once a row has opened, a later CLOSE hides it completely because its
        persistent geometry/opacity may describe stale changed content.
        """
        return self._get_render_attributes(
            timestamp,
            include_never_open=True,
            train_never_open_dc_opacity=train_never_open_dc_opacity,
            black_never_open=black_never_open,
        )

    def _get_render_attributes(
        self,
        timestamp: float,
        *,
        include_never_open: bool,
        train_never_open_dc_opacity: bool = False,
        black_never_open: bool = False,
    ) -> dict[str, torch.Tensor]:
        if train_never_open_dc_opacity and not include_never_open:
            raise ValueError("never-open appearance training requires render support")
        if black_never_open and not include_never_open:
            raise ValueError("black never-open override requires render support")
        if black_never_open and train_never_open_dc_opacity:
            raise ValueError("black never-open override requires frozen appearance")
        indices = self.get_active_state_indices(timestamp)
        active = indices >= 0
        never_open = self.num_states == 0
        render_support = active | (never_open if include_never_open else False)
        appearance_trainable = active | (
            never_open if train_never_open_dc_opacity else False
        )
        if train_never_open_dc_opacity:
            dc = self._active_rows(self.change_dc, appearance_trainable)
            dc = dc * render_support[:, None, None]
        else:
            dc = self._active_rows(self.change_dc, active) * active[:, None, None]
            if black_never_open:
                black_dc = RGB2SH(dc.new_zeros(()))
                dc = dc + black_dc * never_open[:, None, None]
        opacity_raw = self._active_rows(self.opacity, appearance_trainable)
        return {
            "dc": dc,
            "features_rest": self._active_rows(self.features_rest, active),
            "xyz": self._active_rows(self.xyz, active),
            "opacity": self.base.opacity_activation(opacity_raw)
            * render_support[:, None],
            "scaling": self.base.scaling_activation(
                self._active_rows(self.scaling, active)
            ),
            "rotation": self.base.rotation_activation(
                self._active_rows(self.rotation, active)
            ),
            "active": active,
            "never_open": never_open,
            "appearance_trainable": appearance_trainable,
            "render_support": render_support,
            "indices": indices,
        }

    def get_active_change_dc(self, timestamp: float) -> torch.Tensor:
        return self.get_active_render_attributes(timestamp)["dc"]

    def _normalize_rows(self, row_mask_or_indices: Any) -> torch.Tensor:
        if isinstance(row_mask_or_indices, torch.Tensor):
            rows = row_mask_or_indices.to(device=self.state_valid.device)
        else:
            rows = torch.as_tensor(row_mask_or_indices, device=self.state_valid.device)
        if rows.dtype == torch.bool:
            if rows.ndim != 1 or rows.shape[0] != self.state_valid.shape[0]:
                raise ValueError("row mask must have shape [N]")
            rows = torch.nonzero(rows, as_tuple=False).flatten()
        else:
            rows = rows.long().flatten()
        if rows.numel() and bool(
            ((rows < 0) | (rows >= self.state_valid.shape[0])).any()
        ):
            raise IndexError("row indices out of range")
        return rows.unique(sorted=True)

    @torch.no_grad()
    def reset_all_lifespans_closed(self) -> None:
        """Clear interval metadata without touching learned Gaussian values."""
        self.state_start.zero_()
        self.state_end.fill_(float("inf"))
        self.state_valid.zero_()
        self.state_status.zero_()
        self.num_states.zero_()
        self.current_state_index.fill_(-1)

    @torch.no_grad()
    def open_rows(
        self,
        row_mask_or_indices: Any,
        timestamp: float,
        initialization: str = "zero",
        optimizer: Any | None = None,
    ) -> torch.Tensor:
        """Allocate interval metadata while preserving parameters and moments.

        ``initialization`` and ``optimizer`` are accepted for controller API
        compatibility.  They are intentionally ignored: both first OPEN and
        REOPEN use the one persistent raw Gaussian parameter set.
        """
        del optimizer
        if initialization not in {"zero", "preserve"}:
            raise ValueError("persistent lifespan initialization must be zero|preserve")
        rows = self._normalize_rows(row_mask_or_indices)
        if rows.numel() == 0:
            return torch.empty(0, device=self.state_valid.device, dtype=torch.long)
        already_open = self.current_state_index[rows] >= 0
        if bool(already_open.any()):
            raise RuntimeError(
                f"cannot open {int(already_open.sum().item())} already-OPEN rows"
            )
        unused = ~self.state_valid[rows]
        overflow = ~unused.any(dim=1)
        if bool(overflow.any()):
            raise RuntimeError(
                "temporal state capacity exceeded for "
                f"{int(overflow.sum().item())} Gaussians"
            )
        slots = unused.to(torch.int8).argmax(dim=1).long()
        ts = torch.as_tensor(
            timestamp, device=self.state_start.device, dtype=self.state_start.dtype
        )
        if ts.ndim != 0 or not bool(torch.isfinite(ts)):
            raise TypeError("timestamp must be a finite scalar")
        self.state_start[rows, slots] = ts
        self.state_end[rows, slots] = float("inf")
        self.state_valid[rows, slots] = True
        self.state_status[rows, slots] = OPEN
        self.current_state_index[rows] = slots
        self.num_states[rows] += 1
        return slots.clone()

    @torch.no_grad()
    def close_rows(self, row_mask_or_indices: Any, timestamp: float) -> torch.Tensor:
        """Close interval metadata without modifying persistent parameters."""
        rows = self._normalize_rows(row_mask_or_indices)
        if rows.numel() == 0:
            return torch.empty(0, device=self.state_valid.device, dtype=torch.long)
        slots = self.current_state_index[rows]
        active = slots >= 0
        rows, slots = rows[active], slots[active]
        if rows.numel() == 0:
            return torch.empty(0, device=self.state_valid.device, dtype=torch.long)
        ts = torch.as_tensor(
            timestamp, device=self.state_end.device, dtype=self.state_end.dtype
        )
        if ts.ndim != 0 or not bool(torch.isfinite(ts)):
            raise TypeError("timestamp must be a finite scalar")
        if bool((ts <= self.state_start[rows, slots]).any()):
            raise RuntimeError("close timestamp must be strictly greater than state_start")
        self.state_end[rows, slots] = ts
        self.state_status[rows, slots] = CLOSED
        self.current_state_index[rows] = -1
        return slots.clone()

    @torch.no_grad()
    def sync_lifecycle_from_intervals(self) -> None:
        valid = self.state_valid
        finite = torch.isfinite(self.state_end)
        positive_inf = torch.isposinf(self.state_end)
        if bool((valid & ~(finite | positive_inf)).any()):
            raise RuntimeError("valid state_end must be finite or positive infinity")
        self.state_status.zero_()
        self.state_status[valid & finite] = CLOSED
        self.state_status[valid & positive_inf] = OPEN
        slots = torch.arange(valid.shape[1], device=valid.device).expand_as(valid)
        self.num_states.copy_(valid.sum(dim=1, dtype=torch.long))
        current = torch.where(
            self.state_status == OPEN, slots, torch.full_like(slots, -1)
        ).amax(dim=1)
        self.current_state_index.copy_(current.long())

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        self.sync_lifecycle_from_intervals()
        return super().state_dict(*args, **kwargs)

    def validate_lifecycle(self) -> bool:
        n, s = self.state_valid.shape
        if self.state_status.shape != (n, s):
            raise RuntimeError("state_status shape mismatch")
        if self.num_states.shape != (n,) or self.current_state_index.shape != (n,):
            raise RuntimeError("lifecycle row buffer shape mismatch")
        allowed = (
            (self.state_status == EMPTY)
            | (self.state_status == OPEN)
            | (self.state_status == CLOSED)
        )
        if self.state_status.dtype != torch.int8 or not bool(allowed.all()):
            raise RuntimeError("state_status contains invalid values")
        if not torch.equal(self.state_valid, self.state_status != EMPTY):
            raise RuntimeError("state_valid/status mismatch")
        allocated = (self.state_status != EMPTY).sum(dim=1, dtype=torch.long)
        if not torch.equal(self.num_states, allocated):
            raise RuntimeError("num_states must equal allocated interval count")
        open_count = (self.state_status == OPEN).sum(dim=1)
        if bool((open_count > 1).any()):
            raise RuntimeError("a Gaussian may have at most one OPEN interval")
        has_open = open_count == 1
        if not torch.equal(self.current_state_index >= 0, has_open):
            raise RuntimeError("current_state_index/open status mismatch")
        rows = torch.arange(n, device=self.state_valid.device)
        if bool(has_open.any()):
            slots = self.current_state_index[has_open]
            if bool((self.state_status[rows[has_open], slots] != OPEN).any()):
                raise RuntimeError("current_state_index does not point to OPEN interval")
        if bool(((self.state_status == OPEN) & ~torch.isposinf(self.state_end)).any()):
            raise RuntimeError("OPEN interval must end at +inf")
        if bool(((self.state_status == CLOSED) & ~torch.isfinite(self.state_end)).any()):
            raise RuntimeError("CLOSED interval must have finite end")
        return True
