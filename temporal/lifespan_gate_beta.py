"""O(1) flip/stay detection against a binary lifespan semantic state.

This detector deliberately changes the Beta semantics used by the existing
cue-state filter.  ``a`` counts evidence that the committed lifecycle bit is
wrong (FLIP), while ``b`` counts evidence that it is still valid (KEEP).
The frozen reference state starts from the configured stable prior, while a
reset candidate uses the configured reset prior.  After a committed flip, the
winning reset posterior is reinterpreted in the new state's coordinates:
evidence that was FLIP under the old bit becomes KEEP under the toggled bit.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any

import torch

from .bayesian_lifespan_controller import (
    LifespanAction,
    LifespanControllerUpdate,
)
from .single_candidate_beta import (
    SingleCandidateBetaConfig,
    SingleCandidateBetaFilter,
)


@dataclass(frozen=True)
class LifespanGateBetaConfig:
    """Priors for binary lifecycle validity and its one reset candidate."""

    stable_flip_prior: float = 1.0
    stable_keep_prior: float = 10.0
    reset_flip_prior: float = 1.0
    reset_keep_prior: float = 1.0
    bayes_factor_threshold: float = 30.0
    min_evidence_mass: float = 1e-6

    def __post_init__(self) -> None:
        for name in (
            "stable_flip_prior",
            "stable_keep_prior",
            "reset_flip_prior",
            "reset_keep_prior",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be a positive finite number")
        if (
            isinstance(self.bayes_factor_threshold, bool)
            or not isinstance(self.bayes_factor_threshold, Real)
            or not math.isfinite(float(self.bayes_factor_threshold))
            or float(self.bayes_factor_threshold) <= 1.0
        ):
            raise ValueError("bayes_factor_threshold must be finite and > 1")
        if (
            isinstance(self.min_evidence_mass, bool)
            or not isinstance(self.min_evidence_mass, Real)
            or not math.isfinite(float(self.min_evidence_mass))
            or float(self.min_evidence_mass) < 0.0
        ):
            raise ValueError("min_evidence_mass must be finite and nonnegative")


class LifespanGateBetaFilter(SingleCandidateBetaFilter):
    """Compare cue evidence to the committed OPEN/CLOSED semantic bit.

    Inputs retain the source cue convention: ``delta_a`` is cue-positive mass
    and ``delta_b`` is cue-negative mass.  For a CLOSED row, positive mass is
    FLIP evidence; for an OPEN row, negative mass is FLIP evidence.
    """

    algorithm = "lifespan_gate_flip_stay_single_candidate_beta"

    def __init__(
        self,
        num_gaussians: int,
        config: LifespanGateBetaConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.gate_config = config or LifespanGateBetaConfig()
        # The parent filter's prior is the RESET hypothesis.  The committed
        # lifespan-validity prior is installed explicitly below.
        super().__init__(
            num_gaussians,
            SingleCandidateBetaConfig(
                prior_a=float(self.gate_config.reset_flip_prior),
                prior_b=float(self.gate_config.reset_keep_prior),
                bayes_factor_threshold=float(
                    self.gate_config.bayes_factor_threshold
                ),
                min_evidence_mass=float(self.gate_config.min_evidence_mass),
            ),
            device=device,
            dtype=dtype,
        )
        self.stable_a.fill_(float(self.gate_config.stable_flip_prior))
        self.stable_b.fill_(float(self.gate_config.stable_keep_prior))
        # The frozen reference initialization is already a committed CLOSED
        # state, so frame zero must be tested against the prior rather than
        # silently becoming the first stable run.
        self.initialized.fill_(True)

    @torch.no_grad()
    def ensure_capacity(self, required_rows: int) -> None:
        """Grow row-aligned filter buffers to at least ``required_rows``.

        Existing rows are preserved bitwise.  Newly appended rows are initialized
        exactly like a fresh :class:`LifespanGateBetaFilter` with the same gate
        configuration, device, and dtype.
        """

        if (
            isinstance(required_rows, bool)
            or not isinstance(required_rows, Integral)
            or int(required_rows) < 0
        ):
            raise ValueError("required_rows must be a nonnegative integer")
        required = int(required_rows)
        current = self.num_gaussians
        if required <= current:
            return

        tail_count = required - current
        tail = type(self)(
            tail_count,
            self.gate_config,
            device=self.device,
            dtype=self.dtype,
        )
        replacements = {
            name: torch.cat((getattr(self, name), getattr(tail, name)), dim=0)
            for name in self.topology_buffer_names
        }
        for name, value in replacements.items():
            setattr(self, name, value)

    @torch.no_grad()
    def update(
        self,
        delta_a,
        delta_b,
        total_mass=None,
        *,
        current_active,
        first_open=None,
        first_open_bayes_factor_threshold=None,
        row_indices=None,
        indices=None,
        timestamp: int,
    ):
        cue_positive = torch.as_tensor(
            delta_a, device=self.device, dtype=self.dtype
        ).flatten()
        cue_negative = torch.as_tensor(
            delta_b, device=self.device, dtype=self.dtype
        ).flatten()
        if cue_positive.shape != cue_negative.shape:
            raise ValueError("delta_a and delta_b must have the same shape")
        active = torch.as_tensor(current_active, device=self.device).flatten()
        if active.dtype != torch.bool or active.shape != cue_positive.shape:
            raise ValueError("current_active must be boolean and match evidence")
        if first_open is None:
            first_open_mask = None
            if first_open_bayes_factor_threshold is not None:
                raise ValueError(
                    "first_open must be provided when overriding the first-open threshold"
                )
            log_threshold_override = None
        else:
            first_open_mask = torch.as_tensor(first_open, device=self.device).flatten()
            if (
                first_open_mask.dtype != torch.bool
                or first_open_mask.shape != cue_positive.shape
            ):
                raise ValueError("first_open must be boolean and match evidence")
            if first_open_bayes_factor_threshold is None:
                log_threshold_override = None
            else:
                raw_threshold = first_open_bayes_factor_threshold
                if isinstance(raw_threshold, bool):
                    raise TypeError(
                        "first_open_bayes_factor_threshold must be numeric"
                    )
                if isinstance(raw_threshold, Real):
                    threshold_value = float(raw_threshold)
                    if not math.isfinite(threshold_value) or threshold_value <= 1.0:
                        raise ValueError(
                            "first_open_bayes_factor_threshold must be finite and > 1"
                        )
                    first_open_log_threshold = torch.full(
                        cue_positive.shape,
                        math.log(threshold_value),
                        device=self.device,
                        dtype=self.dtype,
                    )
                else:
                    raw_tensor = torch.as_tensor(raw_threshold, device=self.device)
                    if raw_tensor.dtype == torch.bool or raw_tensor.dtype.is_complex:
                        raise TypeError(
                            "first_open_bayes_factor_threshold must be numeric"
                        )
                    threshold_tensor = raw_tensor.to(dtype=self.dtype).flatten()
                    if threshold_tensor.numel() == 1:
                        threshold_tensor = threshold_tensor.expand_as(cue_positive)
                    elif threshold_tensor.shape != cue_positive.shape:
                        raise ValueError(
                            "first_open_bayes_factor_threshold must be scalar or match evidence"
                        )
                    if not bool(torch.isfinite(threshold_tensor).all()) or bool(
                        (threshold_tensor <= 1.0).any()
                    ):
                        raise ValueError(
                            "first_open_bayes_factor_threshold must be finite and > 1"
                        )
                    first_open_log_threshold = torch.log(threshold_tensor).clone()
                default_log_threshold = torch.full(
                    cue_positive.shape,
                    float(self.config.log_bayes_factor_threshold),
                    device=self.device,
                    dtype=self.dtype,
                )
                log_threshold_override = torch.where(
                    first_open_mask, first_open_log_threshold, default_log_threshold
                )

        flip = torch.where(active, cue_negative, cue_positive)
        keep = torch.where(active, cue_positive, cue_negative)
        result = super().update(
            flip,
            keep,
            total_mass,
            log_bayes_factor_threshold=log_threshold_override,
            row_indices=row_indices,
            indices=indices,
            timestamp=timestamp,
        )
        rows = result.indices
        committed = result.candidate_committed.to(dtype=torch.bool)

        if bool(committed.any()):
            positions = torch.nonzero(committed, as_tuple=False).flatten()
            committed_rows = rows[positions]
            # Parent commit state is RESET-prior + candidate block in the old
            # bit's coordinates.  Toggling the bit swaps FLIP and KEEP.
            old_block_flip = (
                result.a_map[positions]
                - float(self.gate_config.reset_flip_prior)
            ).clamp_min(0.0)
            old_block_keep = (
                result.b_map[positions]
                - float(self.gate_config.reset_keep_prior)
            ).clamp_min(0.0)
            # Carry the winning RESET posterior across the bit toggle.  The
            # prior coordinates swap together with the evidence coordinates;
            # re-applying the initial stable prior here would inject evidence
            # that was absent from the Bayes-factor hypothesis that won.
            self.stable_a[committed_rows] = (
                float(self.gate_config.reset_keep_prior) + old_block_keep
            )
            self.stable_b[committed_rows] = (
                float(self.gate_config.reset_flip_prior) + old_block_flip
            )
            self.stable_total_evidence[committed_rows] = (
                old_block_flip + old_block_keep
            )

        stable_a = self.stable_a[rows].clone()
        stable_b = self.stable_b[rows].clone()
        next_active = torch.logical_xor(active, committed)
        result.a_map = stable_a
        result.b_map = stable_b
        result.beta_a = stable_a
        result.beta_b = stable_b
        result.concentration = stable_a + stable_b
        result.change_probability = next_active.to(dtype=self.dtype)
        result.active_before = active.clone()
        result.active_after = next_active.clone()
        result.flip_probability = stable_a / (stable_a + stable_b)
        result.delta_flip = flip.clone()
        result.delta_keep = keep.clone()
        return result


class LifespanGateBetaController:
    """Toggle representation lifespans only on a committed gate mismatch."""

    topology_buffer_names: tuple[str, ...] = ()

    def __init__(self, temporal_model: Any) -> None:
        self.model = temporal_model
        for name in ("current_state_index", "open_rows", "close_rows"):
            if not hasattr(temporal_model, name):
                raise TypeError(f"temporal_model must expose {name}")

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        if state:
            raise ValueError("lifespan-gate controller has no persistent tensors")

    @torch.no_grad()
    def update(
        self,
        update,
        *,
        timestamp: int,
        optimizer=None,
    ) -> LifespanControllerUpdate:
        indices = update.indices.to(
            device=self.model.current_state_index.device, dtype=torch.long
        ).flatten()
        observed = update.observed.to(device=indices.device, dtype=torch.bool)
        committed = update.candidate_committed.to(
            device=indices.device, dtype=torch.bool
        )
        old_slot = self.model.current_state_index[indices].clone()
        old_label = (old_slot >= 0).to(torch.int8)
        active_before = update.active_before.to(
            device=indices.device, dtype=torch.bool
        )
        if not torch.equal(active_before, old_label.bool()):
            raise RuntimeError("filter lifecycle snapshot disagrees with model state")

        action = torch.full(
            indices.shape,
            int(LifespanAction.NONE),
            device=indices.device,
            dtype=torch.int8,
        )
        action[observed & old_label.bool() & ~committed] = int(
            LifespanAction.KEEP
        )
        target_label = old_label.clone()
        target_label[committed] = 1 - old_label[committed]

        open_positions = torch.nonzero(
            committed & (old_label == 0), as_tuple=False
        ).flatten()
        close_positions = torch.nonzero(
            committed & (old_label == 1), as_tuple=False
        ).flatten()
        if open_positions.numel():
            kwargs = {"initialization": "preserve"}
            if optimizer is not None:
                kwargs["optimizer"] = optimizer
            self.model.open_rows(indices[open_positions], timestamp, **kwargs)
            action[open_positions] = int(LifespanAction.OPEN)
        if close_positions.numel():
            self.model.close_rows(indices[close_positions], timestamp)
            action[close_positions] = int(LifespanAction.CLOSE)

        return LifespanControllerUpdate(
            indices=indices.clone(),
            action=action,
            observed=observed.clone(),
            changepoint=committed.clone(),
            changepoint_probability=committed.to(dtype=update.concentration.dtype),
            old_binary_label=old_label,
            new_binary_label=target_label,
            old_slot=old_slot,
            current_slot=self.model.current_state_index[indices].clone(),
            decision_timestamp=int(timestamp),
            estimated_changepoint_timestamp=update.estimated_run_start.to(
                device=indices.device, dtype=torch.long
            ).clone(),
            posterior_probability=target_label.to(dtype=update.concentration.dtype),
            concentration=update.concentration.to(device=indices.device).clone(),
            visible_observations=update.visible_observations.to(
                device=indices.device, dtype=torch.long
            ).clone(),
        )
