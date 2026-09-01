"""O(1) lifecycle detection from learned-DC/cue agreement.

The detector treats the current intrinsic learned change value ``c_i`` as a
soft Bernoulli prediction and the alpha-T attributed cue ratio ``q_i`` as the
observation.  The runner maps O-SCD raw-zero DC to ``c_i=0`` and rendered
white DC to ``c_i=1`` before calling the filter.  Their expected disagreement
is FLIP evidence::

    flip = q_i (1 - c_i) + (1 - q_i) c_i
    keep = q_i c_i + (1 - q_i) (1 - c_i)

For binary ``c_i`` this is exactly the lifespan-gate detector.  Unlike that
ablation, OPEN rows keep a trainable DC between lifecycle commits.  A committed
OPEN sets rendered DC to white one, while CLOSE restores O-SCD raw DC zero
(neutral rendered 0.5, semantic ``c_i=0``), so the new state starts in
agreement with the candidate block.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any

import torch

from utils.sh_utils import RGB2SH

from .bayesian_lifespan_controller import LifespanAction
from .lifespan_gate_beta import LifespanGateBetaController
from .single_candidate_beta import (
    SingleCandidateBetaConfig,
    SingleCandidateBetaFilter,
)


@dataclass(frozen=True)
class LearnedDCAgreementBetaConfig:
    """Priors for learned-DC agreement and its one reset candidate."""

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
        threshold = self.bayes_factor_threshold
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, Real)
            or not math.isfinite(float(threshold))
            or float(threshold) <= 1.0
        ):
            raise ValueError("bayes_factor_threshold must be finite and > 1")
        mass = self.min_evidence_mass
        if (
            isinstance(mass, bool)
            or not isinstance(mass, Real)
            or not math.isfinite(float(mass))
            or float(mass) < 0.0
        ):
            raise ValueError("min_evidence_mass must be finite and nonnegative")


class LearnedDCAgreementBetaFilter(SingleCandidateBetaFilter):
    """Accumulate soft learned-DC/cue agreement as KEEP/FLIP evidence."""

    algorithm = "learned_dc_cue_agreement_single_candidate_beta"

    def __init__(
        self,
        num_gaussians: int,
        config: LearnedDCAgreementBetaConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.agreement_config = config or LearnedDCAgreementBetaConfig()
        super().__init__(
            num_gaussians,
            SingleCandidateBetaConfig(
                prior_a=float(self.agreement_config.reset_flip_prior),
                prior_b=float(self.agreement_config.reset_keep_prior),
                bayes_factor_threshold=float(
                    self.agreement_config.bayes_factor_threshold
                ),
                min_evidence_mass=float(self.agreement_config.min_evidence_mass),
            ),
            device=device,
            dtype=dtype,
        )
        self.stable_a.fill_(float(self.agreement_config.stable_flip_prior))
        self.stable_b.fill_(float(self.agreement_config.stable_keep_prior))
        # The reference initialization is a committed zero-change state.
        self.initialized.fill_(True)

    @torch.no_grad()
    def update(
        self,
        delta_a,
        delta_b,
        total_mass=None,
        *,
        expected_change,
        current_active,
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

        expected = torch.as_tensor(
            expected_change, device=self.device, dtype=self.dtype
        ).flatten()
        if expected.shape != cue_positive.shape:
            raise ValueError("expected_change must match evidence")
        if not bool(torch.isfinite(expected).all()):
            raise ValueError("expected_change must be finite")
        expected = expected.clamp(0.0, 1.0)

        active = torch.as_tensor(current_active, device=self.device).flatten()
        if active.dtype != torch.bool or active.shape != cue_positive.shape:
            raise ValueError("current_active must be boolean and match evidence")

        flip = cue_positive * (1.0 - expected) + cue_negative * expected
        keep = cue_positive * expected + cue_negative * (1.0 - expected)
        result = super().update(
            flip,
            keep,
            total_mass,
            row_indices=row_indices,
            indices=indices,
            timestamp=timestamp,
        )
        rows = result.indices
        committed = result.candidate_committed.to(dtype=torch.bool)

        if bool(committed.any()):
            positions = torch.nonzero(committed, as_tuple=False).flatten()
            committed_rows = rows[positions]
            old_block_flip = (
                result.a_map[positions]
                - float(self.agreement_config.reset_flip_prior)
            ).clamp_min(0.0)
            old_block_keep = (
                result.b_map[positions]
                - float(self.agreement_config.reset_keep_prior)
            ).clamp_min(0.0)
            # The controller resets DC to the opposite binary semantic value,
            # so old-coordinate FLIP evidence becomes new-coordinate KEEP.
            self.stable_a[committed_rows] = (
                float(self.agreement_config.stable_flip_prior) + old_block_keep
            )
            self.stable_b[committed_rows] = (
                float(self.agreement_config.stable_keep_prior) + old_block_flip
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
        result.expected_change = expected.clone()
        result.delta_flip = flip.clone()
        result.delta_keep = keep.clone()
        return result


class LearnedDCAgreementBetaController(LifespanGateBetaController):
    """Toggle lifespan and reset learned DC to the committed semantic value."""

    def __init__(self, temporal_model: Any) -> None:
        super().__init__(temporal_model)
        if not hasattr(temporal_model, "change_dc"):
            raise TypeError("temporal_model must expose change_dc")
        self.dc_commit_reset_count = 0

    @torch.no_grad()
    def _reset_dc(self, rows: torch.Tensor, *, value: float | None, optimizer) -> None:
        if rows.numel() == 0:
            return
        if value is None:
            # O-SCD initializes raw DC to zero, which the SH renderer maps to
            # neutral RGB 0.5 and the agreement transform maps to C=0.
            self.model.change_dc[rows] = 0.0
        else:
            target_rgb = torch.full_like(
                self.model.change_dc.detach()[rows], value
            )
            self.model.change_dc[rows] = RGB2SH(target_rgb)
        if optimizer is not None:
            mask = torch.zeros(
                self.model.change_dc.shape[0],
                device=self.model.change_dc.device,
                dtype=torch.bool,
            )
            mask[rows] = True
            optimizer.reset_state_rows(mask, names=("dc",))
        self.dc_commit_reset_count += int(rows.numel())

    @torch.no_grad()
    def initialize_open_dc(self, rows: torch.Tensor, *, optimizer=None) -> None:
        """Set OPEN DC after the runner verifies any preceding CLOSED interval."""

        rows = torch.as_tensor(
            rows, device=self.model.change_dc.device, dtype=torch.long
        ).flatten().unique(sorted=True)
        self._reset_dc(rows, value=1.0, optimizer=optimizer)

    @torch.no_grad()
    def update(self, update, *, timestamp: int, optimizer=None):
        decision = super().update(update, timestamp=timestamp, optimizer=optimizer)
        close_rows = decision.indices[
            decision.action == int(LifespanAction.CLOSE)
        ]
        # CLOSE reset precedes the closed-row snapshot.  OPEN reset is deferred
        # until the runner has verified the just-ended CLOSED interval.
        self._reset_dc(close_rows, value=None, optimizer=optimizer)
        return decision
