"""View-consistent direct-binary lifespan controller.

This controller preserves the direct binary state filter and temporal model
semantics, but separates marginal state belief from representation lifecycle
transitions.  A lifespan transition is committed only after the transition
branch receives repeated observed support across consecutive views.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import torch

from .binary_state_filter import BinaryStateFilterUpdate
from .binary_state_lifespan_controller import (
    BinaryLifespanAction,
    BinaryStateLifespanController,
    BinaryStateLifespanUpdate,
)


@dataclass(frozen=True)
class ViewConsistentBinaryLifespanControllerConfig:
    """Configuration for consecutive-view transition confirmation.

    ``inactive_to_active_prior`` and ``active_to_inactive_prior`` must match the
    priors used by :class:`temporal.binary_state_filter.BinaryStateFilter` so
    posterior branch odds can be converted to prior-normalized Bayes factors.
    """

    inactive_to_active_prior: float = 0.01
    active_to_inactive_prior: float = 0.01
    min_transition_bayes_factor: float = 3.0
    confirmation_views: int = 2
    min_evidence_strength: float = 1e-6
    eps: float = 1e-12

    def __post_init__(self) -> None:
        for name in ("inactive_to_active_prior", "active_to_inactive_prior"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not (0.0 < value < 1.0):
                raise ValueError(f"{name} must be finite and in (0, 1)")
        if not isinstance(self.confirmation_views, int) or isinstance(self.confirmation_views, bool) or self.confirmation_views < 1:
            raise ValueError("confirmation_views must be a positive integer")
        bf = float(self.min_transition_bayes_factor)
        if not math.isfinite(bf) or bf <= 0.0:
            raise ValueError("min_transition_bayes_factor must be finite and positive")
        mass = float(self.min_evidence_strength)
        if not math.isfinite(mass) or mass < 0.0:
            raise ValueError("min_evidence_strength must be finite and nonnegative")
        eps = float(self.eps)
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be finite and positive")


@dataclass(frozen=True)
class ViewConsistentBinaryLifespanUpdate(BinaryStateLifespanUpdate):
    """Lifecycle update plus transition-odds diagnostics."""

    open_branch_odds: torch.Tensor
    close_branch_odds: torch.Tensor
    open_bayes_factor: torch.Tensor
    close_bayes_factor: torch.Tensor
    open_support: torch.Tensor
    close_support: torch.Tensor
    open_support_count: torch.Tensor
    close_support_count: torch.Tensor

    def event_records(self) -> list[dict[str, Any]]:
        records = super().event_records()
        event_positions = torch.nonzero(self.event_mask, as_tuple=False).flatten().tolist()
        for record, pos in zip(records, event_positions):
            record.update(
                {
                    "open_branch_odds": float(self.open_branch_odds[pos].item()),
                    "close_branch_odds": float(self.close_branch_odds[pos].item()),
                    "open_bayes_factor": float(self.open_bayes_factor[pos].item()),
                    "close_bayes_factor": float(self.close_bayes_factor[pos].item()),
                    "open_support_count": int(self.open_support_count[pos].item()),
                    "close_support_count": int(self.close_support_count[pos].item()),
                }
            )
        return records


class ViewConsistentBinaryLifespanController(BinaryStateLifespanController):
    """Commit OPEN/CLOSE only after consecutive transition-branch support.

    The marginal ``p_active`` remains a belief diagnostic.  It never directly
    triggers lifecycle mutation.  For committed-inactive rows this controller
    compares ``P01`` against ``P00``; for committed-active rows it compares
    ``P10`` against ``P11``.  The resulting branch odds are divided by the
    Markov prior odds, yielding a likelihood Bayes-factor diagnostic.
    """

    topology_buffer_names = ("open_support_count", "close_support_count")

    def __init__(
        self,
        temporal_model,
        config: ViewConsistentBinaryLifespanControllerConfig | None = None,
        *,
        initialization: str = "zero",
    ) -> None:
        if initialization != "zero":
            raise ValueError("view-consistent binary ablation only supports zero initialization")
        self.model = temporal_model
        self.config = config or ViewConsistentBinaryLifespanControllerConfig()
        self.initialization = initialization
        for name in ("state_valid", "current_state_index", "open_rows", "close_rows"):
            if not hasattr(temporal_model, name):
                raise TypeError(f"temporal_model must expose {name}")
        n = int(temporal_model.current_state_index.shape[0])
        device = temporal_model.current_state_index.device
        self.open_support_count = torch.zeros(n, device=device, dtype=torch.long)
        self.close_support_count = torch.zeros(n, device=device, dtype=torch.long)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "open_support_count": self.open_support_count.clone(),
            "close_support_count": self.close_support_count.clone(),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        for name in ("open_support_count", "close_support_count"):
            if name not in state:
                raise KeyError(f"missing controller state tensor {name!r}")
            value = state[name].to(device=getattr(self, name).device, dtype=getattr(self, name).dtype)
            if value.shape != getattr(self, name).shape:
                raise ValueError(
                    f"{name} has shape {tuple(value.shape)}, expected {tuple(getattr(self, name).shape)}"
                )
            getattr(self, name).copy_(value)

    @torch.no_grad()
    def update(
        self,
        update: BinaryStateFilterUpdate,
        *,
        timestamp: int | None = None,
        optimizer=None,
    ) -> ViewConsistentBinaryLifespanUpdate:
        indices = self._validate_update(update)
        device = self.model.state_valid.device
        if timestamp is None:
            timestamp = int(update.timestamp)
        if isinstance(timestamp, bool) or not isinstance(timestamp, int):
            raise TypeError("timestamp must be an integer")

        observed = update.observed.to(device=device, dtype=torch.bool).flatten()
        p_active = update.p_active.to(device=device).flatten()
        p00 = update.p_00.to(device=device).flatten()
        p01 = update.p_01.to(device=device).flatten()
        p10 = update.p_10.to(device=device).flatten()
        p11 = update.p_11.to(device=device).flatten()
        pflip = update.p_flip.to(device=device).flatten()
        strength = update.evidence_strength.to(device=device).flatten()
        if strength.numel() != indices.numel():
            raise ValueError("Binary state update field evidence_strength has the wrong length")
        if not bool(torch.isfinite(strength).all()) or bool((strength < 0).any()):
            raise ValueError("evidence_strength must be finite and nonnegative")

        eps = float(self.config.eps)
        prior_open_odds = float(self.config.inactive_to_active_prior) / (
            1.0 - float(self.config.inactive_to_active_prior)
        )
        prior_close_odds = float(self.config.active_to_inactive_prior) / (
            1.0 - float(self.config.active_to_inactive_prior)
        )
        open_branch_odds = p01 / p00.clamp_min(eps)
        close_branch_odds = p10 / p11.clamp_min(eps)
        open_bayes_factor = open_branch_odds / prior_open_odds
        close_bayes_factor = close_branch_odds / prior_close_odds

        old_slot = self.model.current_state_index[indices].clone()
        old_label = (old_slot >= 0).to(torch.int8)
        new_label = old_label.clone()
        action = torch.full(
            (indices.numel(),), int(BinaryLifespanAction.NONE), device=device, dtype=torch.int8
        )

        quality = observed & (strength >= float(self.config.min_evidence_strength))
        open_support = quality & (open_bayes_factor >= float(self.config.min_transition_bayes_factor))
        close_support = quality & (close_bayes_factor >= float(self.config.min_transition_bayes_factor))

        # Unobserved rows are intentionally not touched: counters and temporal
        # lifecycle state stay bitwise identical.
        inactive = observed & (old_label == 0)
        active = observed & (old_label == 1)
        inactive_quality = inactive & quality
        active_quality = active & quality
        inactive_open_support = inactive & open_support
        inactive_non_support = inactive_quality & ~open_support
        active_close_support = active & close_support
        active_non_support = active_quality & ~close_support

        if bool(inactive_open_support.any()):
            rows = indices[inactive_open_support]
            self.open_support_count[rows] += 1
        if bool(inactive_non_support.any()):
            rows = indices[inactive_non_support]
            self.open_support_count[rows] = 0
        if bool(active_close_support.any()):
            rows = indices[active_close_support]
            self.close_support_count[rows] += 1
        if bool(active_non_support.any()):
            rows = indices[active_non_support]
            self.close_support_count[rows] = 0

        # Report the confirmation counts that caused this decision.  The
        # persistent counters are reset below after a transition is committed.
        reported_open_support_count = self.open_support_count[indices].clone()
        reported_close_support_count = self.close_support_count[indices].clone()

        action[active] = int(BinaryLifespanAction.KEEP)
        confirm = int(self.config.confirmation_views)
        open_ready = inactive_open_support & (self.open_support_count[indices] >= confirm)
        close_ready = active_close_support & (self.close_support_count[indices] >= confirm)

        if bool(open_ready.any()):
            opened_rows = indices[open_ready]
            kwargs = {"initialization": self.initialization}
            if optimizer is not None:
                kwargs["optimizer"] = optimizer
            self.model.open_rows(opened_rows, timestamp, **kwargs)
            opened_slots = self.model.current_state_index[opened_rows]
            self._force_zero_opened_slot_parameters(opened_rows, opened_slots)
            action[open_ready] = int(BinaryLifespanAction.OPEN)
            new_label[open_ready] = 1
            self.open_support_count[opened_rows] = 0
            self.close_support_count[opened_rows] = 0

        if bool(close_ready.any()):
            closed_rows = indices[close_ready]
            self.model.close_rows(closed_rows, timestamp)
            action[close_ready] = int(BinaryLifespanAction.CLOSE)
            new_label[close_ready] = 0
            self.open_support_count[closed_rows] = 0
            self.close_support_count[closed_rows] = 0

        current_slot = self.model.current_state_index[indices].clone()
        active_keep = action == int(BinaryLifespanAction.KEEP)
        if bool(active_keep.any()) and not torch.equal(current_slot[active_keep], old_slot[active_keep]):
            raise RuntimeError("active-to-active KEEP changed a temporal slot")

        return ViewConsistentBinaryLifespanUpdate(
            indices=indices.clone(),
            action=action,
            observed=observed.clone(),
            old_binary_label=old_label,
            new_binary_label=new_label,
            old_slot=old_slot,
            current_slot=current_slot,
            decision_timestamp=int(timestamp),
            p_active=p_active.clone(),
            p_00=p00.clone(),
            p_01=p01.clone(),
            p_10=p10.clone(),
            p_11=p11.clone(),
            p_flip=pflip.clone(),
            visible_observations=update.visible_observations.to(device=device, dtype=torch.long).flatten().clone(),
            open_branch_odds=open_branch_odds.clone(),
            close_branch_odds=close_branch_odds.clone(),
            open_bayes_factor=open_bayes_factor.clone(),
            close_bayes_factor=close_bayes_factor.clone(),
            open_support=open_support.clone(),
            close_support=close_support.clone(),
            open_support_count=reported_open_support_count,
            close_support_count=reported_close_support_count,
        )
