"""Binary lifespan decisions driven by per-Gaussian Bayesian run estimates.

The Bayesian detector may reset its internal run when the evidence distribution
changes.  Representation lifespans are deliberately coarser: only transitions
between ``inactive`` (equal to the immutable reference) and ``active``
(different from the reference) allocate or close temporal slots.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import torch

from .bernoulli_bocd import BOCDUpdate, BernoulliBOCDConfig


class LifespanAction(IntEnum):
    """Vector-friendly lifecycle action codes."""

    NONE = 0
    OPEN = 1
    KEEP = 2
    CLOSE = 3
    UNCERTAIN = 4


@dataclass(frozen=True)
class LifespanControllerUpdate:
    """One controller update over a chunk of Gaussian rows."""

    indices: torch.Tensor
    action: torch.Tensor
    observed: torch.Tensor
    changepoint: torch.Tensor
    changepoint_probability: torch.Tensor
    old_binary_label: torch.Tensor
    new_binary_label: torch.Tensor
    old_slot: torch.Tensor
    current_slot: torch.Tensor
    decision_timestamp: int
    estimated_changepoint_timestamp: torch.Tensor
    posterior_probability: torch.Tensor
    concentration: torch.Tensor
    visible_observations: torch.Tensor

    @property
    def action_names(self) -> tuple[str, ...]:
        return tuple(LifespanAction(int(value)).name for value in self.action.tolist())

    @property
    def event_mask(self) -> torch.Tensor:
        """Rows worth writing to the lifecycle event log.

        OPEN/CLOSE are representation transitions.  A changepoint KEEP is also
        logged because it proves that an active-A -> active-B distribution
        reset did not split the representation lifespan.
        """

        changed = (self.action == int(LifespanAction.OPEN)) | (
            self.action == int(LifespanAction.CLOSE)
        )
        active_reset = self.changepoint & (self.action == int(LifespanAction.KEEP))
        return changed | active_reset

    def event_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for position in torch.nonzero(self.event_mask, as_tuple=False).flatten().tolist():
            old_slot = int(self.old_slot[position].item())
            current_slot = int(self.current_slot[position].item())
            records.append(
                {
                    "gaussian_index": int(self.indices[position].item()),
                    "decision_timestamp": int(self.decision_timestamp),
                    "bocd_estimated_changepoint_timestamp": int(
                        self.estimated_changepoint_timestamp[position].item()
                    ),
                    "old_binary_label": int(self.old_binary_label[position].item()),
                    "new_binary_label": int(self.new_binary_label[position].item()),
                    "action": LifespanAction(int(self.action[position].item())).name,
                    "old_slot": None if old_slot < 0 else old_slot,
                    "new_or_current_slot": None if current_slot < 0 else current_slot,
                    "posterior_probability": float(
                        self.posterior_probability[position].item()
                    ),
                    "changepoint_probability": float(
                        self.changepoint_probability[position].item()
                    ),
                    "concentration": float(self.concentration[position].item()),
                    "visible_observation_count": int(
                        self.visible_observations[position].item()
                    ),
                }
            )
        return records


class BayesianLifespanController:
    """Apply binary OPEN/KEEP/CLOSE semantics to a temporal model.

    A detected BOCD reset is latched while the new run is uncertain.  Once the
    new run becomes confidently active or inactive, the transition table is
    applied.  In particular, active -> active always returns KEEP and never
    calls ``open_rows`` or ``close_rows``.
    """

    def __init__(
        self,
        temporal_model,
        config: BernoulliBOCDConfig | None = None,
        *,
        initialization: str = "zero",
    ) -> None:
        config = config or BernoulliBOCDConfig()
        if config.close_probability > config.open_probability:
            raise ValueError("close_probability must not exceed open_probability")
        if initialization not in {"zero", "base", "base_change"}:
            raise ValueError("initialization must be 'zero' or 'base'")

        self.model = temporal_model
        self.config = config
        self.initialization = initialization
        for name in (
            "state_valid",
            "current_state_index",
            "open_rows",
            "close_rows",
        ):
            if not hasattr(temporal_model, name):
                raise TypeError(f"temporal_model must expose {name}")
        device = temporal_model.state_valid.device
        count = int(temporal_model.state_valid.shape[0])
        initially_active = temporal_model.current_state_index >= 0
        self.committed_run_label = initially_active.to(torch.int8).clone()
        # -1 means that the controller has not yet committed a Bayesian run for
        # this row.  The representation's current open/closed state remains the
        # fallback old binary label for that first decision.
        self.committed_run_start = torch.full(
            (count,), -1, device=device, dtype=torch.long
        )
        self.pending_run_start = torch.full(
            (count,), -1, device=device, dtype=torch.long
        )
        self.pending_changepoint_probability = torch.zeros(
            count, device=device, dtype=temporal_model.state_start.dtype
        )

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "committed_run_label": self.committed_run_label.clone(),
            "committed_run_start": self.committed_run_start.clone(),
            "pending_run_start": self.pending_run_start.clone(),
            "pending_changepoint_probability": (
                self.pending_changepoint_probability.clone()
            ),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        for name, target in self.state_dict().items():
            if name not in state:
                raise KeyError(f"missing controller state tensor {name!r}")
            value = state[name].to(device=target.device, dtype=target.dtype)
            if value.shape != target.shape:
                raise ValueError(
                    f"{name} has shape {tuple(value.shape)}, expected {tuple(target.shape)}"
                )
            getattr(self, name).copy_(value)

    def _validate_update(self, update: BOCDUpdate) -> torch.Tensor:
        indices = update.indices.to(
            device=self.model.state_valid.device, dtype=torch.long
        ).flatten()
        if indices.numel() != torch.unique(indices).numel():
            raise ValueError("BOCD update indices must be unique")
        if indices.numel() and (
            bool((indices < 0).any())
            or bool((indices >= self.model.state_valid.shape[0]).any())
        ):
            raise IndexError("BOCD update indices are out of range")
        for name in (
            "observed",
            "changepoint_probability",
            "estimated_run_start",
            "change_probability",
            "concentration",
            "visible_observations",
        ):
            value = getattr(update, name)
            if value.numel() != indices.numel():
                raise ValueError(f"BOCD update field {name} has the wrong length")
        return indices

    @torch.no_grad()
    def update(
        self,
        update: BOCDUpdate,
        *,
        timestamp: int,
        optimizer=None,
    ) -> LifespanControllerUpdate:
        indices = self._validate_update(update)
        device = self.model.state_valid.device
        observed = update.observed.to(device=device, dtype=torch.bool).flatten()
        probability = update.change_probability.to(device=device).flatten()
        cp_probability = update.changepoint_probability.to(device=device).flatten()
        estimated_start = update.estimated_run_start.to(
            device=device, dtype=torch.long
        ).flatten()
        concentration = update.concentration.to(device=device).flatten()
        visible = update.visible_observations.to(
            device=device, dtype=torch.long
        ).flatten()

        count = indices.numel()
        action = torch.full(
            (count,), int(LifespanAction.NONE), device=device, dtype=torch.int8
        )
        candidate = torch.full((count,), -1, device=device, dtype=torch.int8)
        old_slot = self.model.current_state_index[indices].clone()
        old_label = (old_slot >= 0).to(torch.int8)
        prior_concentration = float(self.config.prior_a + self.config.prior_b)
        run_evidence = (concentration - prior_concentration).clamp_min(0.0)
        enough_support = (
            observed
            & (visible >= int(self.config.min_visible_observations))
            & (run_evidence >= float(self.config.min_run_evidence))
        )
        candidate[enough_support & (probability >= self.config.open_probability)] = 1
        candidate[enough_support & (probability <= self.config.close_probability)] = 0

        cp = observed & (
            cp_probability >= float(self.config.changepoint_probability)
        )
        global_pending = self.pending_run_start[indices].clone()
        global_pending_probability = self.pending_changepoint_probability[
            indices
        ].to(dtype=cp_probability.dtype).clone()
        global_committed_start = self.committed_run_start[indices]
        # The first reliable Bayesian run is allowed to initialize without a
        # changepoint.  Established labels change only through a detected (or
        # previously latched) reset.
        uninitialized = global_committed_start < 0
        global_pending[cp] = estimated_start[cp]
        global_pending_probability[cp] = cp_probability[cp]
        self.pending_run_start[indices] = global_pending
        self.pending_changepoint_probability[indices] = (
            global_pending_probability.to(
                dtype=self.pending_changepoint_probability.dtype
            )
        )
        deciding_new_run = uninitialized | (global_pending >= 0)
        confident = candidate >= 0
        decide = observed & deciding_new_run & confident

        # Stable, confident rows emit the non-transition form of their binary
        # state. A label contradiction without a BOCD reset remains uncertain.
        stable = observed & ~deciding_new_run & confident
        committed_label = self.committed_run_label[indices]
        stable_same = stable & (candidate == committed_label)
        action[stable_same & (candidate == 1)] = int(LifespanAction.KEEP)
        action[stable & ~stable_same] = int(LifespanAction.UNCERTAIN)
        action[observed & ~confident] = int(LifespanAction.UNCERTAIN)

        if bool(decide.any()):
            target = candidate[decide]
            previous = old_label[decide]
            selected_positions = torch.nonzero(decide, as_tuple=False).flatten()
            open_positions = selected_positions[(previous == 0) & (target == 1)]
            close_positions = selected_positions[(previous == 1) & (target == 0)]
            keep_positions = selected_positions[(previous == 1) & (target == 1)]
            none_positions = selected_positions[(previous == 0) & (target == 0)]

            if open_positions.numel():
                open_kwargs = {"initialization": self.initialization}
                if optimizer is not None:
                    open_kwargs["optimizer"] = optimizer
                self.model.open_rows(
                    indices[open_positions], timestamp, **open_kwargs
                )
                action[open_positions] = int(LifespanAction.OPEN)
            if close_positions.numel():
                self.model.close_rows(indices[close_positions], timestamp)
                action[close_positions] = int(LifespanAction.CLOSE)
            action[keep_positions] = int(LifespanAction.KEEP)
            action[none_positions] = int(LifespanAction.NONE)

            committed_positions = selected_positions
            committed_rows = indices[committed_positions]
            self.committed_run_label[committed_rows] = candidate[committed_positions]
            committed_starts = torch.where(
                global_pending[committed_positions] >= 0,
                global_pending[committed_positions],
                estimated_start[committed_positions],
            )
            self.committed_run_start[committed_rows] = committed_starts
            self.pending_run_start[committed_rows] = -1
            self.pending_changepoint_probability[committed_rows] = 0.0

        committed_reset = decide & ~uninitialized & (global_pending >= 0)
        reported_changepoint = cp | committed_reset
        reported_estimated_start = torch.where(
            global_pending >= 0, global_pending, estimated_start
        )
        reported_changepoint_probability = torch.where(
            global_pending >= 0, global_pending_probability, cp_probability
        )

        current_slot = self.model.current_state_index[indices].clone()
        # Hard acceptance criterion: active -> active never changes slots.
        active_keep = (
            action == int(LifespanAction.KEEP)
        ) & (old_label == 1) & (candidate == 1)
        if bool(active_keep.any()) and not torch.equal(
            current_slot[active_keep], old_slot[active_keep]
        ):
            raise RuntimeError("active-to-active KEEP changed a temporal slot")

        return LifespanControllerUpdate(
            indices=indices.clone(),
            action=action,
            observed=observed.clone(),
            changepoint=reported_changepoint.clone(),
            changepoint_probability=reported_changepoint_probability.clone(),
            old_binary_label=old_label,
            new_binary_label=candidate,
            old_slot=old_slot,
            current_slot=current_slot,
            decision_timestamp=int(timestamp),
            estimated_changepoint_timestamp=reported_estimated_start.clone(),
            posterior_probability=probability.clone(),
            concentration=concentration.clone(),
            visible_observations=visible.clone(),
        )


# Backwards-compatible public names used by temporal.__init__ and older tests.
BayesianLifespanControllerConfig = BernoulliBOCDConfig
LifespanDecision = LifespanControllerUpdate
