"""Detached candidate-anchor learned-DC/cue agreement detector.

The representation optimizer remains fully active while a lifecycle candidate
is live.  The detector, however, snapshots the semantic learned change value
at candidate start and evaluates every later candidate observation against that
detached value::

    C_anchor = stop_gradient(C_i at candidate start)
    flip = q (1 - C_anchor) + (1 - q) C_anchor
    keep = q C_anchor + (1 - q) (1 - C_anchor)

This prevents the trainable ``C_i`` from erasing the mismatch that opened the
candidate.  A reset is committed only after the Bayes-factor threshold and a
strict number of consecutive observed views support the lifecycle direction:
``q > C_anchor`` while CLOSED and ``q < C_anchor`` while OPEN.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real

import torch

from .learned_dc_agreement_beta import (
    LearnedDCAgreementBetaConfig,
    LearnedDCAgreementBetaController,
)
from .single_candidate_beta import (
    SingleCandidateBetaConfig,
    SingleCandidateBetaFilter,
)


@dataclass(frozen=True)
class AnchoredDCAgreementBetaConfig(LearnedDCAgreementBetaConfig):
    """Agreement priors plus strict multi-view confirmation settings."""

    confirmation_views: int = 3
    directional_margin: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        views = self.confirmation_views
        if (
            isinstance(views, bool)
            or not isinstance(views, Integral)
            or int(views) < 1
        ):
            raise ValueError("confirmation_views must be a positive integer")
        margin = self.directional_margin
        if (
            isinstance(margin, bool)
            or not isinstance(margin, Real)
            or not math.isfinite(float(margin))
            or not 0.0 <= float(margin) < 1.0
        ):
            raise ValueError("directional_margin must be finite and in [0, 1)")


class AnchoredDCAgreementBetaFilter(SingleCandidateBetaFilter):
    """Use a topology-aligned detached ``C_anchor`` for every live candidate."""

    algorithm = "anchored_learned_dc_cue_agreement_single_candidate_beta"
    topology_buffer_names = SingleCandidateBetaFilter.topology_buffer_names + (
        "candidate_anchor",
    )

    def __init__(
        self,
        num_gaussians: int,
        config: AnchoredDCAgreementBetaConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.agreement_config = config or AnchoredDCAgreementBetaConfig()
        super().__init__(
            num_gaussians,
            SingleCandidateBetaConfig(
                prior_a=float(self.agreement_config.reset_flip_prior),
                prior_b=float(self.agreement_config.reset_keep_prior),
                bayes_factor_threshold=float(
                    self.agreement_config.bayes_factor_threshold
                ),
                min_evidence_mass=float(self.agreement_config.min_evidence_mass),
                min_candidate_support_views=int(
                    self.agreement_config.confirmation_views
                ),
                require_candidate_support_each_view=True,
            ),
            device=device,
            dtype=dtype,
        )
        self.stable_a.fill_(float(self.agreement_config.stable_flip_prior))
        self.stable_b.fill_(float(self.agreement_config.stable_keep_prior))
        self.initialized.fill_(True)
        self.candidate_anchor = torch.zeros(
            self.num_gaussians, device=self.device, dtype=self.dtype
        )

    def _clear_candidate(self, rows: torch.Tensor) -> None:
        super()._clear_candidate(rows)
        self.candidate_anchor[rows] = 0.0

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
        if row_indices is not None and indices is not None:
            raise TypeError("pass only one of row_indices or indices")
        selected_indices = row_indices if indices is None else indices

        cue_positive = torch.as_tensor(
            delta_a, device=self.device, dtype=self.dtype
        ).flatten()
        cue_negative = torch.as_tensor(
            delta_b, device=self.device, dtype=self.dtype
        ).flatten()
        if cue_positive.shape != cue_negative.shape:
            raise ValueError("delta_a and delta_b must have the same shape")
        if not bool(torch.isfinite(cue_positive).all()) or not bool(
            torch.isfinite(cue_negative).all()
        ):
            raise ValueError("evidence counts must be finite")
        if bool((cue_positive < 0).any()) or bool((cue_negative < 0).any()):
            raise ValueError("evidence counts must be nonnegative")

        current_expected = torch.as_tensor(
            expected_change, device=self.device, dtype=self.dtype
        ).flatten()
        if current_expected.shape != cue_positive.shape:
            raise ValueError("expected_change must match evidence")
        if not bool(torch.isfinite(current_expected).all()):
            raise ValueError("expected_change must be finite")
        current_expected = current_expected.clamp(0.0, 1.0)

        active = torch.as_tensor(current_active, device=self.device).flatten()
        if active.dtype != torch.bool or active.shape != cue_positive.shape:
            raise ValueError("current_active must be boolean and match evidence")

        rows = self._normalize_indices(selected_indices, cue_positive.numel())
        was_live = self.candidate_active[rows].clone()
        effective_expected = torch.where(
            was_live,
            self.candidate_anchor[rows],
            current_expected,
        )
        strength = cue_positive + cue_negative
        cue_ratio = cue_positive / strength.clamp_min(
            torch.finfo(self.dtype).eps
        )
        margin = float(self.agreement_config.directional_margin)
        directional_support = torch.where(
            active,
            cue_ratio < (effective_expected - margin),
            cue_ratio > (effective_expected + margin),
        )

        flip = (
            cue_positive * (1.0 - effective_expected)
            + cue_negative * effective_expected
        )
        keep = (
            cue_positive * effective_expected
            + cue_negative * (1.0 - effective_expected)
        )
        result = super().update(
            flip,
            keep,
            total_mass,
            candidate_support=directional_support,
            row_indices=rows,
            timestamp=timestamp,
        )

        started_live = result.candidate_started & result.candidate_active
        if bool(started_live.any()):
            started_rows = rows[started_live]
            self.candidate_anchor[started_rows] = current_expected[started_live]

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
            # The controller toggles the semantic lifecycle value at commit.
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
        result.expected_change = effective_expected.clone()
        result.current_expected_change = current_expected.clone()
        result.candidate_anchor = effective_expected.clone()
        result.candidate_anchor_live_after = self.candidate_anchor[rows].clone()
        result.cue_ratio = cue_ratio.clone()
        result.directional_support = directional_support.clone()
        result.delta_flip = flip.clone()
        result.delta_keep = keep.clone()
        return result


__all__ = [
    "AnchoredDCAgreementBetaConfig",
    "AnchoredDCAgreementBetaFilter",
    "LearnedDCAgreementBetaController",
]
