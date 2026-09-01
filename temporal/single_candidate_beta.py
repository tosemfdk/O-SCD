"""O(1) Beta-Bernoulli changepoint detector with one reset candidate.

The detector retains one committed (``stable``) Beta posterior and, only while
the current observations are better explained by a fresh Beta prior, one
candidate block.  It deliberately has no run-length posterior, hazard, or
state-transition probability.

For candidate pseudo-count totals ``A`` and ``B`` the exact block score is

``log p(A,B | fresh prior) - log p(A,B | committed Beta)``.

The committed Beta is frozen while the candidate is live.  A rejected block is
merged into it; a committed block replaces it.  This prevents the incumbent
from learning away a possible distribution reset before the reset decision is
made.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real

import torch

from .bernoulli_bocd import BOCDUpdate, beta_binomial_log_predictive


@dataclass(frozen=True)
class SingleCandidateBetaConfig:
    """Configuration for :class:`SingleCandidateBetaFilter`.

    ``bayes_factor_threshold`` is supplied in ordinary Bayes-factor units and
    converted to log space exactly once.  Values must exceed one because a
    reset is committed only when the fresh-run hypothesis is positively
    preferred over KEEP.
    """

    prior_a: float = 1.0
    prior_b: float = 1.0
    bayes_factor_threshold: float = 100.0
    min_evidence_mass: float = 1e-6
    min_candidate_support_views: int = 1
    require_candidate_support_each_view: bool = False

    def __post_init__(self) -> None:
        for name in ("prior_a", "prior_b"):
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
        support_views = self.min_candidate_support_views
        if (
            isinstance(support_views, bool)
            or not isinstance(support_views, Integral)
            or int(support_views) < 1
        ):
            raise ValueError("min_candidate_support_views must be a positive integer")
        if not isinstance(self.require_candidate_support_each_view, bool):
            raise TypeError("require_candidate_support_each_view must be boolean")

    @property
    def log_bayes_factor_threshold(self) -> float:
        return math.log(float(self.bayes_factor_threshold))


class SingleCandidateBetaFilter:
    """Constant-state changepoint detector for fractional Beta evidence."""

    algorithm = "single_candidate_beta_log_bayes_factor"
    topology_buffer_names = (
        "stable_a",
        "stable_b",
        "stable_total_evidence",
        "stable_run_start",
        "stable_visible_observations",
        "initialized",
        "candidate_delta_a",
        "candidate_delta_b",
        "candidate_start",
        "candidate_visible_observations",
        "candidate_support_observations",
        "candidate_active",
        "visible_observations",
        "last_timestamp",
        "last_log_bayes_factor",
    )

    def __init__(
        self,
        num_gaussians: int,
        config: SingleCandidateBetaConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if (
            isinstance(num_gaussians, bool)
            or not isinstance(num_gaussians, Integral)
            or int(num_gaussians) < 1
        ):
            raise ValueError("num_gaussians must be a positive integer")
        if not torch.empty((), dtype=dtype).is_floating_point():
            raise TypeError("dtype must be floating")
        self.config = config or SingleCandidateBetaConfig()
        count = int(num_gaussians)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype
        self.stable_a = torch.full(
            (count,), float(self.config.prior_a), device=self.device, dtype=dtype
        )
        self.stable_b = torch.full(
            (count,), float(self.config.prior_b), device=self.device, dtype=dtype
        )
        self.stable_total_evidence = torch.zeros(
            count, device=self.device, dtype=dtype
        )
        self.stable_run_start = torch.full(
            (count,), -1, device=self.device, dtype=torch.long
        )
        self.stable_visible_observations = torch.zeros(
            count, device=self.device, dtype=torch.long
        )
        self.initialized = torch.zeros(count, device=self.device, dtype=torch.bool)
        self.candidate_delta_a = torch.zeros(count, device=self.device, dtype=dtype)
        self.candidate_delta_b = torch.zeros(count, device=self.device, dtype=dtype)
        self.candidate_start = torch.full(
            (count,), -1, device=self.device, dtype=torch.long
        )
        self.candidate_visible_observations = torch.zeros(
            count, device=self.device, dtype=torch.long
        )
        self.candidate_support_observations = torch.zeros(
            count, device=self.device, dtype=torch.long
        )
        self.candidate_active = torch.zeros(count, device=self.device, dtype=torch.bool)
        self.visible_observations = torch.zeros(
            count, device=self.device, dtype=torch.long
        )
        self.last_timestamp = torch.full(
            (count,), -1, device=self.device, dtype=torch.long
        )
        self.last_log_bayes_factor = torch.zeros(
            count, device=self.device, dtype=dtype
        )

    @property
    def num_gaussians(self) -> int:
        return int(self.stable_a.shape[0])

    @property
    def num_rows(self) -> int:
        return self.num_gaussians

    @staticmethod
    def block_log_bayes_factor(
        delta_a: torch.Tensor,
        delta_b: torch.Tensor,
        stable_a: torch.Tensor,
        stable_b: torch.Tensor,
        *,
        prior_a: float = 1.0,
        prior_b: float = 1.0,
    ) -> torch.Tensor:
        """Return exact RESET-vs-KEEP log evidence for one candidate block."""

        output_dtype = delta_a.dtype
        # The score subtracts lgamma differences and its sign decides whether a
        # candidate starts.  Evaluate float32 model state in float64 so a
        # numerically neutral block is not turned into a candidate by avoidable
        # cancellation error.  Only the small selected-row chunk is promoted.
        calculation_dtype = (
            torch.float64 if output_dtype in {torch.float16, torch.bfloat16, torch.float32}
            else output_dtype
        )
        delta_a = delta_a.to(dtype=calculation_dtype)
        delta_b = delta_b.to(dtype=calculation_dtype)
        stable_a = stable_a.to(dtype=calculation_dtype)
        stable_b = stable_b.to(dtype=calculation_dtype)
        prior_a_tensor = torch.as_tensor(
            prior_a, device=delta_a.device, dtype=calculation_dtype
        )
        prior_b_tensor = torch.as_tensor(
            prior_b, device=delta_b.device, dtype=calculation_dtype
        )
        log_keep = beta_binomial_log_predictive(
            delta_a, delta_b, stable_a, stable_b
        )
        log_reset = beta_binomial_log_predictive(
            delta_a, delta_b, prior_a_tensor, prior_b_tensor
        )
        return (log_reset - log_keep).to(dtype=output_dtype)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: getattr(self, name).clone() for name in self.topology_buffer_names
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        for name in self.topology_buffer_names:
            if name not in state:
                raise KeyError(f"missing filter state tensor {name!r}")
            target = getattr(self, name)
            value = state[name].to(device=target.device, dtype=target.dtype)
            if value.shape != target.shape:
                raise ValueError(
                    f"{name} has shape {tuple(value.shape)}, "
                    f"expected {tuple(target.shape)}"
                )
            target.copy_(value)

    def _normalize_indices(
        self, row_indices: torch.Tensor | None, length: int
    ) -> torch.Tensor:
        if row_indices is None:
            if length != self.num_gaussians:
                raise ValueError("full updates require one count per Gaussian")
            indices = torch.arange(length, device=self.device, dtype=torch.long)
        else:
            raw = (
                row_indices
                if isinstance(row_indices, torch.Tensor)
                else torch.as_tensor(row_indices)
            )
            if raw.dtype == torch.bool or raw.dtype.is_floating_point or raw.dtype.is_complex:
                raise TypeError("row_indices must contain integers")
            indices = raw.to(device=self.device, dtype=torch.long).flatten()
        if indices.numel() != length:
            raise ValueError("row_indices length must match evidence length")
        if indices.numel() != torch.unique(indices).numel():
            raise ValueError("update row_indices must be unique")
        if indices.numel() and (
            bool((indices < 0).any())
            or bool((indices >= self.num_gaussians).any())
        ):
            raise IndexError("row_indices are out of range")
        return indices

    def _clear_candidate(self, rows: torch.Tensor) -> None:
        self.candidate_delta_a[rows] = 0.0
        self.candidate_delta_b[rows] = 0.0
        self.candidate_start[rows] = -1
        self.candidate_visible_observations[rows] = 0
        self.candidate_support_observations[rows] = 0
        self.candidate_active[rows] = False

    @torch.no_grad()
    def update(
        self,
        delta_a,
        delta_b,
        total_mass=None,
        *,
        candidate_support=None,
        row_indices=None,
        indices=None,
        timestamp: int,
    ) -> BOCDUpdate:
        """Update selected rows while leaving unobserved rows bitwise unchanged."""

        if row_indices is not None and indices is not None:
            raise TypeError("pass only one of row_indices or indices")
        if indices is not None:
            row_indices = indices
        if isinstance(timestamp, bool) or not isinstance(timestamp, Integral):
            raise TypeError("timestamp must be an integer")
        da = torch.as_tensor(delta_a, device=self.device, dtype=self.dtype).flatten()
        db = torch.as_tensor(delta_b, device=self.device, dtype=self.dtype).flatten()
        if da.shape != db.shape:
            raise ValueError("delta_a and delta_b must have the same flattened shape")
        if not bool(torch.isfinite(da).all()) or not bool(torch.isfinite(db).all()):
            raise ValueError("evidence counts must be finite")
        if bool((da < 0).any()) or bool((db < 0).any()):
            raise ValueError("evidence counts must be nonnegative")
        rows = self._normalize_indices(row_indices, da.numel())
        if candidate_support is None:
            support = torch.ones(da.shape, device=self.device, dtype=torch.bool)
        else:
            support = torch.as_tensor(candidate_support, device=self.device).flatten()
            if support.dtype != torch.bool or support.shape != da.shape:
                raise ValueError("candidate_support must be boolean and match evidence")
        if total_mass is None:
            mass = da + db
        else:
            mass = torch.as_tensor(
                total_mass, device=self.device, dtype=self.dtype
            ).flatten()
            if mass.shape != da.shape:
                raise ValueError("total_mass length must match evidence length")
            if not bool(torch.isfinite(mass).all()) or bool((mass < 0).any()):
                raise ValueError("total_mass must be finite and nonnegative")
        strength = da + db
        observed = (
            (mass > 0)
            & (mass >= float(self.config.min_evidence_mass))
            & (strength > 0)
        )

        count = rows.numel()
        started = torch.zeros(count, device=self.device, dtype=torch.bool)
        continued = torch.zeros_like(started)
        rejected = torch.zeros_like(started)
        committed = torch.zeros_like(started)
        evaluated = torch.zeros_like(started)
        initialized_now = torch.zeros_like(started)
        score = self.last_log_bayes_factor[rows].clone()
        duration = torch.zeros(count, device=self.device, dtype=torch.long)
        support_duration = torch.zeros(count, device=self.device, dtype=torch.long)

        if bool(observed.any()):
            positions = torch.nonzero(observed, as_tuple=False).flatten()
            obs_rows = rows[positions]
            self.visible_observations[obs_rows] += 1
            self.last_timestamp[obs_rows] = int(timestamp)

            first_mask = ~self.initialized[obs_rows]
            if bool(first_mask.any()):
                first_pos = positions[first_mask]
                first_rows = rows[first_pos]
                first_a = da[first_pos]
                first_b = db[first_pos]
                self.stable_a[first_rows] = float(self.config.prior_a) + first_a
                self.stable_b[first_rows] = float(self.config.prior_b) + first_b
                self.stable_total_evidence[first_rows] = first_a + first_b
                self.stable_run_start[first_rows] = int(timestamp)
                self.stable_visible_observations[first_rows] = 1
                self.initialized[first_rows] = True
                self.last_log_bayes_factor[first_rows] = 0.0
                score[first_pos] = 0.0
                initialized_now[first_pos] = True

            eval_mask = ~first_mask
            if bool(eval_mask.any()):
                eval_pos = positions[eval_mask]
                eval_rows = rows[eval_pos]
                eval_a = da[eval_pos]
                eval_b = db[eval_pos]
                eval_support = support[eval_pos]
                was_live = self.candidate_active[eval_rows].clone()

                block_a = torch.where(
                    was_live,
                    self.candidate_delta_a[eval_rows] + eval_a,
                    eval_a,
                )
                block_b = torch.where(
                    was_live,
                    self.candidate_delta_b[eval_rows] + eval_b,
                    eval_b,
                )
                block_start = torch.where(
                    was_live,
                    self.candidate_start[eval_rows],
                    torch.full_like(self.candidate_start[eval_rows], int(timestamp)),
                )
                block_visible = torch.where(
                    was_live,
                    self.candidate_visible_observations[eval_rows] + 1,
                    torch.ones_like(self.candidate_visible_observations[eval_rows]),
                )
                block_support = torch.where(
                    was_live,
                    self.candidate_support_observations[eval_rows]
                    + eval_support.to(dtype=torch.long),
                    eval_support.to(dtype=torch.long),
                )
                block_score = self.block_log_bayes_factor(
                    block_a,
                    block_b,
                    self.stable_a[eval_rows],
                    self.stable_b[eval_rows],
                    prior_a=float(self.config.prior_a),
                    prior_b=float(self.config.prior_b),
                )
                score[eval_pos] = block_score
                duration[eval_pos] = block_visible
                support_duration[eval_pos] = block_support
                evaluated[eval_pos] = True
                self.last_log_bayes_factor[eval_rows] = block_score

                commit_mask = (
                    block_score >= float(self.config.log_bayes_factor_threshold)
                ) & (
                    block_support >= int(self.config.min_candidate_support_views)
                )
                support_failure = (
                    bool(self.config.require_candidate_support_each_view)
                    & ~eval_support
                )
                reject_mask = (~commit_mask) & (
                    (block_score <= 0.0) | support_failure
                )
                live_mask = ~(commit_mask | reject_mask)
                start_mask = live_mask & ~was_live
                continue_mask = live_mask & was_live

                if bool(commit_mask.any()):
                    commit_pos = eval_pos[commit_mask]
                    commit_rows = rows[commit_pos]
                    self.stable_a[commit_rows] = (
                        float(self.config.prior_a) + block_a[commit_mask]
                    )
                    self.stable_b[commit_rows] = (
                        float(self.config.prior_b) + block_b[commit_mask]
                    )
                    self.stable_total_evidence[commit_rows] = (
                        block_a[commit_mask] + block_b[commit_mask]
                    )
                    self.stable_run_start[commit_rows] = block_start[commit_mask]
                    self.stable_visible_observations[commit_rows] = block_visible[
                        commit_mask
                    ]
                    self._clear_candidate(commit_rows)
                    committed[commit_pos] = True
                    started[commit_pos] = ~was_live[commit_mask]

                if bool(reject_mask.any()):
                    reject_pos = eval_pos[reject_mask]
                    reject_rows = rows[reject_pos]
                    # If no candidate was live this is an ordinary KEEP update.
                    # A live candidate rejection merges the whole frozen block.
                    self.stable_a[reject_rows] += block_a[reject_mask]
                    self.stable_b[reject_rows] += block_b[reject_mask]
                    self.stable_total_evidence[reject_rows] += (
                        block_a[reject_mask] + block_b[reject_mask]
                    )
                    self.stable_visible_observations[reject_rows] += block_visible[
                        reject_mask
                    ]
                    live_rejection = was_live[reject_mask]
                    if bool(live_rejection.any()):
                        self._clear_candidate(reject_rows[live_rejection])
                    rejected[reject_pos] = live_rejection

                if bool(live_mask.any()):
                    live_pos = eval_pos[live_mask]
                    live_rows = rows[live_pos]
                    self.candidate_delta_a[live_rows] = block_a[live_mask]
                    self.candidate_delta_b[live_rows] = block_b[live_mask]
                    self.candidate_start[live_rows] = block_start[live_mask]
                    self.candidate_visible_observations[live_rows] = block_visible[
                        live_mask
                    ]
                    self.candidate_support_observations[live_rows] = block_support[
                        live_mask
                    ]
                    self.candidate_active[live_rows] = True
                    started[live_pos] = start_mask[live_mask]
                    continued[live_pos] = continue_mask[live_mask]

        stable_a = self.stable_a[rows].clone()
        stable_b = self.stable_b[rows].clone()
        stable_visible = self.stable_visible_observations[rows].clone()
        stable_start = self.stable_run_start[rows].clone()
        probability = stable_a / (stable_a + stable_b)
        changepoint_probability = committed.to(dtype=self.dtype)
        result = BOCDUpdate(
            indices=rows.clone(),
            observed=observed.clone(),
            changepoint_probability=changepoint_probability,
            map_run_length=(stable_visible - 1).clamp_min(0),
            estimated_run_start=stable_start,
            change_probability=probability,
            concentration=stable_a + stable_b,
            visible_observations=stable_visible,
            a_map=stable_a,
            b_map=stable_b,
            candidate_started=started,
            candidate_continued=continued,
            candidate_rejected=rejected,
            candidate_committed=committed,
            candidate_evaluated=evaluated,
            candidate_active=self.candidate_active[rows].clone(),
            candidate_log_bayes_factor=score.clone(),
            candidate_duration=duration,
            candidate_support=support.clone(),
            candidate_support_observations=support_duration,
            initialized=initialized_now,
            stable_total_evidence=self.stable_total_evidence[rows].clone(),
            total_visible_observations=self.visible_observations[rows].clone(),
            last_timestamp=self.last_timestamp[rows].clone(),
        )
        return result
