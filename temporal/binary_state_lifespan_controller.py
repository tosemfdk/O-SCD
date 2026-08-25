"""Lifespan controller driven only by direct binary state posterior."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import torch

from .binary_state_filter import BinaryStateFilterUpdate


class BinaryLifespanAction(IntEnum):
    NONE = 0
    OPEN = 1
    KEEP = 2
    CLOSE = 3
    UNCERTAIN = 4
    HOLD = 4  # Backward-compatible alias; public diagnostics use UNCERTAIN.


@dataclass(frozen=True)
class BinaryStateLifespanControllerConfig:
    active_threshold: float = 0.6
    inactive_threshold: float = 0.4

    def __post_init__(self) -> None:
        if not (0.0 < float(self.inactive_threshold) < float(self.active_threshold) < 1.0):
            raise ValueError("thresholds must satisfy 0 < inactive < active < 1")


@dataclass(frozen=True)
class BinaryStateLifespanUpdate:
    indices: torch.Tensor
    action: torch.Tensor
    observed: torch.Tensor
    old_binary_label: torch.Tensor
    new_binary_label: torch.Tensor
    old_slot: torch.Tensor
    current_slot: torch.Tensor
    decision_timestamp: int
    p_active: torch.Tensor
    p_00: torch.Tensor
    p_01: torch.Tensor
    p_10: torch.Tensor
    p_11: torch.Tensor
    p_flip: torch.Tensor
    visible_observations: torch.Tensor

    @property
    def action_names(self) -> tuple[str, ...]:
        return tuple(BinaryLifespanAction(int(v)).name for v in self.action.tolist())

    @property
    def event_mask(self) -> torch.Tensor:
        return (self.action == int(BinaryLifespanAction.OPEN)) | (
            self.action == int(BinaryLifespanAction.CLOSE)
        )

    def event_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for pos in torch.nonzero(self.event_mask, as_tuple=False).flatten().tolist():
            old_slot = int(self.old_slot[pos].item())
            current_slot = int(self.current_slot[pos].item())
            records.append(
                {
                    "gaussian_index": int(self.indices[pos].item()),
                    "decision_timestamp": int(self.decision_timestamp),
                    "old_binary_label": int(self.old_binary_label[pos].item()),
                    "new_binary_label": int(self.new_binary_label[pos].item()),
                    "action": BinaryLifespanAction(int(self.action[pos].item())).name,
                    "old_slot": None if old_slot < 0 else old_slot,
                    "new_current_slot": None if current_slot < 0 else current_slot,
                    "new_or_current_slot": None if current_slot < 0 else current_slot,
                    "p_active": float(self.p_active[pos].item()),
                    "p_01": float(self.p_01[pos].item()),
                    "p_10": float(self.p_10[pos].item()),
                    "p_flip": float(self.p_flip[pos].item()),
                    "visible_observation_count": int(self.visible_observations[pos].item()),
                }
            )
        return records


class BinaryStateLifespanController:
    """Apply OPEN/CLOSE/KEEP/NONE from ``p_active`` hysteresis only."""

    def __init__(
        self,
        temporal_model,
        config: BinaryStateLifespanControllerConfig | None = None,
        *,
        initialization: str = "zero",
    ) -> None:
        self.model = temporal_model
        self.config = config or BinaryStateLifespanControllerConfig()
        if initialization != "zero":
            raise ValueError("direct binary ablation only supports zero initialization")
        self.initialization = initialization
        for name in ("state_valid", "current_state_index", "open_rows", "close_rows"):
            if not hasattr(temporal_model, name):
                raise TypeError(f"temporal_model must expose {name}")

    @torch.no_grad()
    def _force_zero_opened_slot_parameters(self, rows: torch.Tensor, slots: torch.Tensor) -> None:
        """Guarantee strict zero-init for new episodes even with advanced indexing.

        Existing temporal models expose initialization hooks, but direct binary
        ablation requires an explicit invariant for every state-specific tensor.
        """
        if rows.numel() == 0:
            return
        if bool(
            getattr(self.model, "persistent_parameters_across_lifespans", False)
        ):
            # Persistent-bank ablations allocate only interval metadata.  The
            # learned DC/geometry and Adam moments intentionally survive CLOSE
            # and are reused by a later REOPEN.
            return
        if hasattr(self.model, "state_parameter_items"):
            items = self.model.state_parameter_items()
        else:
            items = (("dc", self.model.state_change_dc),)
        for _name, parameter in items:
            parameter[rows, slots] = 0

    def _validate_update(self, update: BinaryStateFilterUpdate) -> torch.Tensor:
        indices = update.indices.to(device=self.model.state_valid.device, dtype=torch.long).flatten()
        if indices.numel() != torch.unique(indices).numel():
            raise ValueError("Binary state update indices must be unique")
        if indices.numel() and (
            bool((indices < 0).any()) or bool((indices >= self.model.state_valid.shape[0]).any())
        ):
            raise IndexError("Binary state update indices are out of range")
        for name in (
            "observed",
            "p_active",
            "p_00",
            "p_01",
            "p_10",
            "p_11",
            "p_flip",
            "visible_observations",
        ):
            if getattr(update, name).numel() != indices.numel():
                raise ValueError(f"Binary state update field {name} has the wrong length")
        for name in ("p_active", "p_00", "p_01", "p_10", "p_11", "p_flip"):
            value = getattr(update, name).detach().flatten()
            if not bool(torch.isfinite(value).all()) or bool(((value < 0) | (value > 1)).any()):
                raise ValueError(f"Binary state update field {name} must be finite in [0,1]")
        return indices

    @torch.no_grad()
    def update(
        self,
        update: BinaryStateFilterUpdate,
        *,
        timestamp: int | None = None,
        optimizer=None,
    ) -> BinaryStateLifespanUpdate:
        indices = self._validate_update(update)
        device = self.model.state_valid.device
        if timestamp is None:
            timestamp = int(update.timestamp)
        if isinstance(timestamp, bool) or not isinstance(timestamp, int):
            raise TypeError("timestamp must be an integer")
        observed = update.observed.to(device=device, dtype=torch.bool).flatten()
        p_active = update.p_active.to(device=device).flatten()
        old_slot = self.model.current_state_index[indices].clone()
        old_label = (old_slot >= 0).to(torch.int8)
        new_label = torch.full_like(old_label, -1)
        confident_active = observed & (p_active >= float(self.config.active_threshold))
        confident_inactive = observed & (p_active <= float(self.config.inactive_threshold))
        new_label[confident_active] = 1
        new_label[confident_inactive] = 0

        action = torch.full((indices.numel(),), int(BinaryLifespanAction.NONE), device=device, dtype=torch.int8)
        uncertain = observed & (new_label < 0)
        action[uncertain] = int(BinaryLifespanAction.UNCERTAIN)
        decided = observed & (new_label >= 0)
        if bool(decided.any()):
            positions = torch.nonzero(decided, as_tuple=False).flatten()
            target = new_label[positions]
            previous = old_label[positions]
            open_positions = positions[(previous == 0) & (target == 1)]
            close_positions = positions[(previous == 1) & (target == 0)]
            keep_positions = positions[(previous == 1) & (target == 1)]
            none_positions = positions[(previous == 0) & (target == 0)]
            if open_positions.numel():
                kwargs = {"initialization": self.initialization}
                if optimizer is not None:
                    kwargs["optimizer"] = optimizer
                opened_rows = indices[open_positions]
                self.model.open_rows(opened_rows, timestamp, **kwargs)
                # ``open_rows`` canonicalizes row order internally, so query
                # the authoritative slot per original row before enforcing the
                # strict zero-initialization invariant.
                opened_slots = self.model.current_state_index[opened_rows]
                self._force_zero_opened_slot_parameters(opened_rows, opened_slots)
                action[open_positions] = int(BinaryLifespanAction.OPEN)
            if close_positions.numel():
                self.model.close_rows(indices[close_positions], timestamp)
                action[close_positions] = int(BinaryLifespanAction.CLOSE)
            action[keep_positions] = int(BinaryLifespanAction.KEEP)
            action[none_positions] = int(BinaryLifespanAction.NONE)

        current_slot = self.model.current_state_index[indices].clone()
        active_keep = (
            action == int(BinaryLifespanAction.KEEP)
        ) & (old_label == 1) & (new_label == 1)
        if bool(active_keep.any()) and not torch.equal(current_slot[active_keep], old_slot[active_keep]):
            raise RuntimeError("active-to-active KEEP changed a temporal slot")

        return BinaryStateLifespanUpdate(
            indices=indices.clone(),
            action=action,
            observed=observed.clone(),
            old_binary_label=old_label,
            new_binary_label=new_label,
            old_slot=old_slot,
            current_slot=current_slot,
            decision_timestamp=int(timestamp),
            p_active=p_active.clone(),
            p_00=update.p_00.to(device=device).flatten().clone(),
            p_01=update.p_01.to(device=device).flatten().clone(),
            p_10=update.p_10.to(device=device).flatten().clone(),
            p_11=update.p_11.to(device=device).flatten().clone(),
            p_flip=update.p_flip.to(device=device).flatten().clone(),
            visible_observations=update.visible_observations.to(device=device, dtype=torch.long).flatten().clone(),
        )
