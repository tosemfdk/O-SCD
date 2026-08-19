"""Sidecar temporal change state model for Gaussian change features."""

import math
from numbers import Integral, Real

import torch
from torch import nn

from .lifespan import get_active_state_indices

EMPTY = 0
OPEN = 1
CLOSED = 2


class TemporalChangeModel(nn.Module):
    """Stores time-indexed change DC features without mutating Gaussian topology.

    Build the sidecar after placing the external base on its final device and
    dtype. Default construction preserves the legacy always-open slot zero;
    automatic lifecycle runners explicitly reset all rows closed.
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
        state_status = torch.zeros(shape, device=device, dtype=torch.int8)
        state_status[:, 0] = OPEN
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
        base_change_gaussians,
        max_states: int = 4,
        initial_time: float = 0.0,
    ) -> "TemporalChangeModel":
        """Build a temporal sidecar initialized from existing change Gaussians."""
        return cls(
            base_change_gaussians,
            max_states=max_states,
            initial_time=initial_time,
        )

    @torch.no_grad()
    def sync_lifecycle_from_intervals(self) -> None:
        """Rebuild lifecycle metadata from authoritative interval buffers."""
        valid = self.state_valid
        finite = torch.isfinite(self.state_end)
        positive_inf = torch.isposinf(self.state_end)
        if bool((valid & ~(finite | positive_inf)).any()):
            raise RuntimeError(
                "valid state_end values must be finite or positive infinity"
            )
        self.state_status.zero_()
        self.state_status[valid & finite] = CLOSED
        self.state_status[valid & positive_inf] = OPEN
        slot_ids = torch.arange(valid.shape[1], device=valid.device).expand_as(valid)
        self.num_states.copy_(valid.sum(dim=1, dtype=torch.long))
        open_slot = torch.where(
            self.state_status == OPEN,
            slot_ids,
            torch.full_like(slot_ids, -1),
        ).amax(dim=1)
        self.current_state_index.copy_(open_slot.long())

    def state_dict(self, *args, **kwargs):
        # Intervals are authoritative because legacy/manual runners may edit
        # state_start/end/valid directly.  Saved checkpoints therefore carry a
        # deterministic lifecycle view of those intervals.
        self.sync_lifecycle_from_intervals()
        return super().state_dict(*args, **kwargs)

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
        rows = torch.arange(
            self.state_change_dc.shape[0], device=self.state_change_dc.device
        )
        dc = self.state_change_dc[rows, indices.clamp_min(0)]
        return dc * active[:, None, None], active

    def get_active_change_dc(self, timestamp) -> torch.Tensor:
        """Return active ``[N, 1, 3]`` DC values, zeroing inactive Gaussians."""
        dc, _ = self.get_active_change(timestamp)
        return dc

    def _normalize_rows(self, row_mask_or_indices) -> torch.Tensor:
        if isinstance(row_mask_or_indices, torch.Tensor):
            rows = row_mask_or_indices.to(device=self.state_valid.device)
        else:
            rows = torch.as_tensor(row_mask_or_indices, device=self.state_valid.device)
        if rows.dtype == torch.bool:
            if rows.ndim != 1 or rows.shape[0] != self.state_valid.shape[0]:
                raise ValueError("row mask must have shape [N]")
            rows = rows.nonzero(as_tuple=False).flatten()
        else:
            rows = rows.long().flatten()
        if rows.numel() and (
            (rows < 0).any() or (rows >= self.state_valid.shape[0]).any()
        ):
            raise IndexError("row indices out of range")
        return rows.unique(sorted=True)

    @torch.no_grad()
    def reset_all_lifespans_closed(self) -> None:
        """Clear all active/manual slots for causal automatic lifecycle runs."""
        self.state_start.zero_()
        self.state_end.fill_(float("inf"))
        self.state_valid.zero_()
        self.state_status.zero_()
        self.num_states.zero_()
        self.current_state_index.fill_(-1)

    @torch.no_grad()
    def _initialize_state_slot(
        self, rows: torch.Tensor, slots: torch.Tensor, initialization: str
    ) -> None:
        if initialization == "zero":
            self.state_change_dc[rows, slots].zero_()
        elif initialization in {"base", "base_change"}:
            self.state_change_dc[rows, slots].copy_(self.base._features_dc.detach()[rows])
        else:
            raise ValueError("initialization must be 'zero' or 'base'")

    @torch.no_grad()
    def _reset_optimizer_pairs(
        self, optimizer, rows: torch.Tensor, slots: torch.Tensor
    ) -> None:
        if optimizer is None:
            return
        if hasattr(optimizer, "reset_pairs"):
            optimizer.reset_pairs(rows, slots)
            return
        if hasattr(optimizer, "reset_state_pairs"):
            pair_mask = torch.zeros_like(self.state_valid, dtype=torch.bool)
            pair_mask[rows, slots] = True
            optimizer.reset_state_pairs(pair_mask)
            return
        raise TypeError(
            "optimizer must expose reset_state_pairs(mask) or reset_pairs(rows, slots)"
        )

    @torch.no_grad()
    def open_rows(
        self,
        row_mask_or_indices,
        timestamp,
        initialization: str = "zero",
        optimizer=None,
    ) -> torch.Tensor:
        """Open a new lifespan slot for each selected row and return its slots."""
        rows = self._normalize_rows(row_mask_or_indices)
        if rows.numel() == 0:
            return torch.empty(0, device=self.state_valid.device, dtype=torch.long)
        already_open = self.current_state_index[rows] >= 0
        if bool(already_open.any()):
            raise RuntimeError(
                f"cannot open {int(already_open.sum().item())} Gaussians with an already OPEN lifespan"
            )
        unused = ~self.state_valid[rows]
        overflow = ~unused.any(dim=1)
        if bool(overflow.any()):
            raise RuntimeError(
                f"temporal state capacity exceeded for {int(overflow.sum().item())} Gaussians"
            )
        slots = unused.to(torch.int8).argmax(dim=1).long()
        if isinstance(timestamp, bool):
            raise TypeError("timestamp must be a finite scalar")
        ts = torch.as_tensor(
            timestamp, device=self.state_start.device, dtype=self.state_start.dtype
        )
        if ts.ndim != 0 or not bool(torch.isfinite(ts)):
            raise TypeError("timestamp must be a finite scalar")
        self._initialize_state_slot(rows, slots, initialization)
        self._reset_optimizer_pairs(optimizer, rows, slots)
        self.state_start[rows, slots] = ts
        self.state_end[rows, slots] = float("inf")
        self.state_valid[rows, slots] = True
        self.state_status[rows, slots] = OPEN
        self.current_state_index[rows] = slots
        self.num_states[rows] += 1
        return slots.clone()

    @torch.no_grad()
    def close_rows(self, row_mask_or_indices, timestamp) -> torch.Tensor:
        """Close currently open lifespan slots for selected rows, preserving history."""
        rows = self._normalize_rows(row_mask_or_indices)
        if rows.numel() == 0:
            return torch.empty(0, device=self.state_valid.device, dtype=torch.long)
        slots = self.current_state_index[rows]
        open_mask = slots >= 0
        rows = rows[open_mask]
        slots = slots[open_mask]
        if rows.numel() == 0:
            return torch.empty(0, device=self.state_valid.device, dtype=torch.long)
        if isinstance(timestamp, bool):
            raise TypeError("timestamp must be a finite scalar")
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

    def validate_lifecycle(self) -> bool:
        """Raise if lifecycle metadata disagrees with temporal intervals."""
        n, s = self.state_valid.shape
        if self.state_status.shape != (n, s):
            raise RuntimeError("state_status shape mismatch")
        if self.num_states.shape != (n,) or self.current_state_index.shape != (n,):
            raise RuntimeError("lifecycle row buffer shape mismatch")
        if self.state_status.dtype != torch.int8:
            raise RuntimeError("state_status must be int8")
        allowed = (
            (self.state_status == EMPTY)
            | (self.state_status == OPEN)
            | (self.state_status == CLOSED)
        )
        if not bool(allowed.all()):
            raise RuntimeError("state_status contains invalid values")
        if not bool(torch.equal(self.state_valid, self.state_status != EMPTY)):
            raise RuntimeError("state_valid must be true exactly for non-empty lifecycle slots")
        if bool((self.num_states < 0).any()) or bool(
            (self.num_states > self.max_states).any()
        ):
            raise RuntimeError("num_states out of range")
        expected_counts = self.state_valid.sum(dim=1, dtype=torch.long)
        if not bool(torch.equal(self.num_states, expected_counts)):
            raise RuntimeError("num_states must equal the number of allocated slots")
        open_counts = (self.state_status == OPEN).sum(dim=1)
        if bool((open_counts > 1).any()):
            raise RuntimeError("at most one OPEN slot is allowed per Gaussian")
        current = self.current_state_index
        has_open = open_counts == 1
        if bool(((current >= 0) != has_open).any()):
            raise RuntimeError("current_state_index must be set iff a row has an OPEN slot")
        rows = torch.arange(n, device=current.device)
        valid_current = current.clamp_min(0)
        if bool(has_open.any()) and not bool(
            (
                self.state_status[rows[has_open], valid_current[has_open]] == OPEN
            ).all()
        ):
            raise RuntimeError("current_state_index must point to OPEN slot")
        if bool(((self.state_status == OPEN) & ~torch.isposinf(self.state_end)).any()):
            raise RuntimeError("OPEN slots must have positive-infinite state_end")
        if bool(((self.state_status == CLOSED) & ~torch.isfinite(self.state_end)).any()):
            raise RuntimeError("CLOSED slots must have finite state_end")
        valid = self.state_valid
        if bool((valid & ~torch.isfinite(self.state_start)).any()):
            raise RuntimeError("valid state_start values must be finite")
        if bool((valid & ~(self.state_start < self.state_end)).any()):
            raise RuntimeError("valid lifecycle intervals must satisfy start < end")
        return True
