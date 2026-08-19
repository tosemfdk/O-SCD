"""Two-branch Beta-Bernoulli changepoint approximation for ESCD ablation."""

from __future__ import annotations

import math
from numbers import Integral

import torch

from .bernoulli_bocd import (
    BOCDUpdate,
    BernoulliBOCDConfig,
    BetaBernoulliBOCD,
    beta_binomial_log_predictive,
)


def beam2_persistent_state_bytes(
    gaussian_count: int, dtype: torch.dtype = torch.float32
) -> int:
    """Return checkpoint-state bytes for the linear two-branch filter."""
    if (
        isinstance(gaussian_count, bool)
        or not isinstance(gaussian_count, Integral)
        or gaussian_count < 1
    ):
        raise ValueError("gaussian_count must be a positive integer")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError("dtype must be floating")
    float_bytes = torch.empty((), dtype=dtype).element_size()
    long_bytes = torch.empty((), dtype=torch.long).element_size()
    # Nine floating and eight int64 vectors are serialized per Gaussian.
    return int(gaussian_count) * (9 * float_bytes + 8 * long_bytes)


class BeamTwoBernoulliFilter:
    """O(N) two-branch BOCD approximation with a persistent reset candidate.

    ``MAPResetBernoulliFilter`` discards a reset branch whenever its one-frame
    posterior is below the threshold. This filter instead retains two branches
    per Gaussian:

    * an incumbent run, and
    * the strongest reset candidate seen so far.

    On each observation the incumbent and existing candidate grow, a fresh reset
    candidate is spawned from the prior, and only the stronger of the two reset
    alternatives is retained. Once the retained candidate posterior reaches
    ``changepoint_probability`` it replaces the incumbent. This preserves the
    repeated weak contradictory evidence that the single-branch MAP filter loses,
    while keeping persistent memory linear in the number of Gaussians.
    """

    algorithm = "beam2"

    def __init__(
        self,
        num_gaussians: int,
        config: BernoulliBOCDConfig | None = None,
        *,
        device=None,
        dtype=torch.float32,
    ):
        if (
            isinstance(num_gaussians, bool)
            or not isinstance(num_gaussians, Integral)
            or num_gaussians < 1
        ):
            raise ValueError("num_gaussians must be a positive integer")
        if not torch.empty((), dtype=dtype).is_floating_point():
            raise TypeError("dtype must be floating")
        self.num_gaussians = int(num_gaussians)
        self.num_rows = self.num_gaussians
        self.config = config or BernoulliBOCDConfig()
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype
        self.max_run_length = int(self.config.max_run_length)
        shape = (self.num_gaussians,)

        # Incumbent branch.
        self.a = torch.full(
            shape, float(self.config.prior_a), device=self.device, dtype=dtype
        )
        self.b = torch.full(
            shape, float(self.config.prior_b), device=self.device, dtype=dtype
        )
        self.total_run_evidence = torch.zeros(shape, device=self.device, dtype=dtype)
        self.run_length = torch.zeros(shape, device=self.device, dtype=torch.long)
        self.run_start = torch.zeros(shape, device=self.device, dtype=torch.long)
        self.run_visible_observations = torch.zeros(
            shape, device=self.device, dtype=torch.long
        )
        self.log_incumbent_weight = torch.zeros(
            shape, device=self.device, dtype=dtype
        )

        # Retained reset candidate. ``-inf`` weight means no live candidate.
        self.candidate_a = torch.full(
            shape, float(self.config.prior_a), device=self.device, dtype=dtype
        )
        self.candidate_b = torch.full(
            shape, float(self.config.prior_b), device=self.device, dtype=dtype
        )
        self.candidate_total_run_evidence = torch.zeros(
            shape, device=self.device, dtype=dtype
        )
        self.candidate_run_length = torch.zeros(
            shape, device=self.device, dtype=torch.long
        )
        self.candidate_run_start = torch.full(
            shape, -1, device=self.device, dtype=torch.long
        )
        self.candidate_visible_observations = torch.zeros(
            shape, device=self.device, dtype=torch.long
        )
        self.log_candidate_weight = torch.full(
            shape, -torch.inf, device=self.device, dtype=dtype
        )

        self.visible_observations = torch.zeros(
            shape, device=self.device, dtype=torch.long
        )
        self.last_changepoint_probability = torch.zeros(
            shape, device=self.device, dtype=dtype
        )
        self.last_candidate_probability = torch.zeros(
            shape, device=self.device, dtype=dtype
        )
        self.last_candidate_run_start = torch.full(
            shape, -1, device=self.device, dtype=torch.long
        )
        self.last_log_continue = torch.full(
            shape, float("nan"), device=self.device, dtype=dtype
        )
        self.last_log_reset = torch.full(
            shape, float("nan"), device=self.device, dtype=dtype
        )
        self.last_log_bayes_factor = torch.full(
            shape, float("nan"), device=self.device, dtype=dtype
        )
        self.last_timestamp = torch.full(
            shape, -1, device=self.device, dtype=torch.long
        )

    @staticmethod
    def integrated_log_predictive(
        s: torch.Tensor, f: torch.Tensor, a: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        return beta_binomial_log_predictive(s, f, a, b)

    def _candidate_is_map(self, indices: torch.Tensor | None = None) -> torch.Tensor:
        incumbent = (
            self.log_incumbent_weight
            if indices is None
            else self.log_incumbent_weight[indices]
        )
        candidate = (
            self.log_candidate_weight
            if indices is None
            else self.log_candidate_weight[indices]
        )
        return candidate > incumbent

    @property
    def map_run_length(self) -> torch.Tensor:
        return torch.where(
            self._candidate_is_map(), self.candidate_run_length, self.run_length
        )

    @property
    def map_run_length_tensor(self) -> torch.Tensor:
        return self.map_run_length

    @property
    def estimated_run_start(self) -> torch.Tensor:
        return torch.where(
            self._candidate_is_map(), self.candidate_run_start, self.run_start
        )

    @property
    def map_beta_a(self) -> torch.Tensor:
        return torch.where(self._candidate_is_map(), self.candidate_a, self.a)

    @property
    def map_beta_b(self) -> torch.Tensor:
        return torch.where(self._candidate_is_map(), self.candidate_b, self.b)

    @property
    def change_probability(self) -> torch.Tensor:
        return self.map_beta_a / (self.map_beta_a + self.map_beta_b)

    @property
    def concentration(self) -> torch.Tensor:
        return self.map_beta_a + self.map_beta_b

    @property
    def p_run_zero(self) -> torch.Tensor:
        return self.run_length_posterior[:, 0]

    @property
    def run_length_posterior(self) -> torch.Tensor:
        posterior = torch.zeros(
            (self.num_gaussians, self.max_run_length + 1),
            device=self.device,
            dtype=self.dtype,
        )
        posterior.scatter_add_(
            1,
            self.run_length.clamp_max(self.max_run_length)[:, None],
            self.log_incumbent_weight.exp()[:, None],
        )
        candidate_mass = torch.where(
            torch.isfinite(self.log_candidate_weight),
            self.log_candidate_weight.exp(),
            torch.zeros_like(self.log_candidate_weight),
        )
        posterior.scatter_add_(
            1,
            self.candidate_run_length.clamp_max(self.max_run_length)[:, None],
            candidate_mass[:, None],
        )
        return posterior

    def posterior(self) -> torch.Tensor:
        return self.run_length_posterior

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "a": self.a.clone(),
            "b": self.b.clone(),
            "total_run_evidence": self.total_run_evidence.clone(),
            "run_length": self.run_length.clone(),
            "run_start": self.run_start.clone(),
            "run_visible_observations": self.run_visible_observations.clone(),
            "log_incumbent_weight": self.log_incumbent_weight.clone(),
            "candidate_a": self.candidate_a.clone(),
            "candidate_b": self.candidate_b.clone(),
            "candidate_total_run_evidence": self.candidate_total_run_evidence.clone(),
            "candidate_run_length": self.candidate_run_length.clone(),
            "candidate_run_start": self.candidate_run_start.clone(),
            "candidate_visible_observations": self.candidate_visible_observations.clone(),
            "log_candidate_weight": self.log_candidate_weight.clone(),
            "visible_observations": self.visible_observations.clone(),
            "last_changepoint_probability": self.last_changepoint_probability.clone(),
            "last_timestamp": self.last_timestamp.clone(),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        for name, target in self.state_dict().items():
            if name not in state:
                raise KeyError(f"missing BOCD state tensor {name!r}")
            value = state[name].to(device=self.device, dtype=target.dtype)
            if value.shape != target.shape:
                raise ValueError(
                    f"{name} has shape {tuple(value.shape)}, expected {tuple(target.shape)}"
                )
            getattr(self, name).copy_(value)

    def _coerce_update(self, *args, **kwargs):
        return BetaBernoulliBOCD._coerce_update(self, *args, **kwargs)

    @torch.no_grad()
    def update(self, *args, **kwargs) -> BOCDUpdate:
        indices, s, f, total_mass, timestamp = self._coerce_update(*args, **kwargs)
        observed = total_mass >= float(self.config.min_evidence_mass)
        if not bool(observed.any()):
            return self._snapshot(indices, observed)

        idx = indices[observed]
        ss = s[observed]
        ff = f[observed]
        first = self.visible_observations[idx] == 0
        if bool(first.any()):
            first_idx = idx[first]
            first_s = ss[first]
            first_f = ff[first]
            self.a[first_idx] = float(self.config.prior_a) + first_s
            self.b[first_idx] = float(self.config.prior_b) + first_f
            self.total_run_evidence[first_idx] = first_s + first_f
            self.run_length[first_idx] = 0
            self.run_start[first_idx] = int(timestamp)
            self.run_visible_observations[first_idx] = 1
            self.log_incumbent_weight[first_idx] = 0.0
            self._clear_candidate(first_idx)
            self.visible_observations[first_idx] = 1
            self.last_changepoint_probability[first_idx] = 0.0
            self.last_candidate_probability[first_idx] = 0.0
            self.last_candidate_run_start[first_idx] = -1
            self.last_log_continue[first_idx] = float("nan")
            self.last_log_reset[first_idx] = float("nan")
            self.last_log_bayes_factor[first_idx] = float("nan")
            self.last_timestamp[first_idx] = int(timestamp)

        continuing = ~first
        if bool(continuing.any()):
            cidx = idx[continuing]
            cs = ss[continuing]
            cf = ff[continuing]
            self._update_existing_rows(cidx, cs, cf, timestamp)

        return self._snapshot(indices, observed)

    def _clear_candidate(self, indices: torch.Tensor) -> None:
        self.candidate_a[indices] = float(self.config.prior_a)
        self.candidate_b[indices] = float(self.config.prior_b)
        self.candidate_total_run_evidence[indices] = 0.0
        self.candidate_run_length[indices] = 0
        self.candidate_run_start[indices] = -1
        self.candidate_visible_observations[indices] = 0
        self.log_candidate_weight[indices] = -torch.inf

    def _update_existing_rows(
        self,
        idx: torch.Tensor,
        s: torch.Tensor,
        f: torch.Tensor,
        timestamp: int,
    ) -> None:
        h = float(self.config.resolved_hazard)
        log_h = math.log(h)
        log_1mh = math.log1p(-h)
        prior_a = torch.full_like(s, float(self.config.prior_a))
        prior_b = torch.full_like(f, float(self.config.prior_b))

        current_predictive = beta_binomial_log_predictive(
            s, f, self.a[idx], self.b[idx]
        )
        current_growth_log = (
            self.log_incumbent_weight[idx] + log_1mh + current_predictive
        )

        candidate_live = torch.isfinite(self.log_candidate_weight[idx])
        candidate_predictive = beta_binomial_log_predictive(
            s, f, self.candidate_a[idx], self.candidate_b[idx]
        )
        candidate_growth_log = (
            self.log_candidate_weight[idx] + log_1mh + candidate_predictive
        )
        candidate_growth_log = torch.where(
            candidate_live,
            candidate_growth_log,
            torch.full_like(candidate_growth_log, -torch.inf),
        )

        old_normalizer = torch.logsumexp(
            torch.stack(
                (self.log_incumbent_weight[idx], self.log_candidate_weight[idx]),
                dim=1,
            ),
            dim=1,
        )
        reset_predictive = beta_binomial_log_predictive(s, f, prior_a, prior_b)
        spawn_log = old_normalizer + log_h + reset_predictive

        use_spawn = spawn_log >= candidate_growth_log
        retained_log = torch.where(use_spawn, spawn_log, candidate_growth_log)
        retained_a = torch.where(use_spawn, prior_a + s, self.candidate_a[idx] + s)
        retained_b = torch.where(use_spawn, prior_b + f, self.candidate_b[idx] + f)
        retained_evidence = torch.where(
            use_spawn,
            s + f,
            self.candidate_total_run_evidence[idx] + s + f,
        )
        retained_length = torch.where(
            use_spawn,
            torch.zeros_like(self.candidate_run_length[idx]),
            (self.candidate_run_length[idx] + 1).clamp_max(self.max_run_length),
        )
        retained_start = torch.where(
            use_spawn,
            torch.full_like(self.candidate_run_start[idx], int(timestamp)),
            self.candidate_run_start[idx],
        )
        retained_visible = torch.where(
            use_spawn,
            torch.ones_like(self.candidate_visible_observations[idx]),
            self.candidate_visible_observations[idx] + 1,
        )

        normalization = torch.logsumexp(
            torch.stack((current_growth_log, retained_log), dim=1), dim=1
        )
        new_current_log = current_growth_log - normalization
        new_candidate_log = retained_log - normalization
        candidate_probability = new_candidate_log.exp()
        commit = candidate_probability >= float(
            self.config.changepoint_probability
        )

        grown_a = self.a[idx] + s
        grown_b = self.b[idx] + f
        grown_evidence = self.total_run_evidence[idx] + s + f
        grown_length = (self.run_length[idx] + 1).clamp_max(self.max_run_length)
        grown_start = self.run_start[idx]
        grown_visible = self.run_visible_observations[idx] + 1

        self.a[idx] = torch.where(commit, retained_a, grown_a)
        self.b[idx] = torch.where(commit, retained_b, grown_b)
        self.total_run_evidence[idx] = torch.where(
            commit, retained_evidence, grown_evidence
        )
        self.run_length[idx] = torch.where(commit, retained_length, grown_length)
        self.run_start[idx] = torch.where(commit, retained_start, grown_start)
        self.run_visible_observations[idx] = torch.where(
            commit, retained_visible, grown_visible
        )
        self.log_incumbent_weight[idx] = torch.where(
            commit, torch.zeros_like(new_current_log), new_current_log
        )

        not_committed = ~commit
        if bool(not_committed.any()):
            keep_idx = idx[not_committed]
            self.candidate_a[keep_idx] = retained_a[not_committed]
            self.candidate_b[keep_idx] = retained_b[not_committed]
            self.candidate_total_run_evidence[keep_idx] = retained_evidence[
                not_committed
            ]
            self.candidate_run_length[keep_idx] = retained_length[not_committed]
            self.candidate_run_start[keep_idx] = retained_start[not_committed]
            self.candidate_visible_observations[keep_idx] = retained_visible[
                not_committed
            ]
            self.log_candidate_weight[keep_idx] = new_candidate_log[not_committed]
        if bool(commit.any()):
            self._clear_candidate(idx[commit])

        self.visible_observations[idx] += 1
        self.last_changepoint_probability[idx] = candidate_probability
        self.last_candidate_probability[idx] = candidate_probability
        self.last_candidate_run_start[idx] = retained_start
        self.last_log_continue[idx] = current_predictive + log_1mh
        self.last_log_reset[idx] = reset_predictive + log_h
        self.last_log_bayes_factor[idx] = (
            self.last_log_reset[idx] - self.last_log_continue[idx]
        )
        self.last_timestamp[idx] = int(timestamp)

    def _snapshot(self, indices: torch.Tensor, observed: torch.Tensor) -> BOCDUpdate:
        candidate_is_map = self._candidate_is_map(indices)
        a_map = torch.where(candidate_is_map, self.candidate_a[indices], self.a[indices])
        b_map = torch.where(candidate_is_map, self.candidate_b[indices], self.b[indices])
        map_run_length = torch.where(
            candidate_is_map,
            self.candidate_run_length[indices],
            self.run_length[indices],
        )
        map_run_start = torch.where(
            candidate_is_map,
            self.candidate_run_start[indices],
            self.run_start[indices],
        )
        map_visible = torch.where(
            candidate_is_map,
            self.candidate_visible_observations[indices],
            self.run_visible_observations[indices],
        )
        branch_a = torch.stack((self.a[indices], self.candidate_a[indices]), dim=1)
        branch_b = torch.stack((self.b[indices], self.candidate_b[indices]), dim=1)
        branch_probability = torch.stack(
            (
                self.log_incumbent_weight[indices].exp(),
                torch.where(
                    torch.isfinite(self.log_candidate_weight[indices]),
                    self.log_candidate_weight[indices].exp(),
                    torch.zeros_like(self.log_candidate_weight[indices]),
                ),
            ),
            dim=1,
        )
        branch_start = torch.stack(
            (self.run_start[indices], self.candidate_run_start[indices]), dim=1
        )
        branch_visible = torch.stack(
            (
                self.run_visible_observations[indices],
                self.candidate_visible_observations[indices],
            ),
            dim=1,
        )
        posterior = torch.zeros(
            (indices.numel(), self.max_run_length + 1),
            device=self.device,
            dtype=self.dtype,
        )
        posterior.scatter_add_(
            1,
            self.run_length[indices].clamp_max(self.max_run_length)[:, None],
            branch_probability[:, :1],
        )
        posterior.scatter_add_(
            1,
            self.candidate_run_length[indices]
            .clamp_max(self.max_run_length)[:, None],
            branch_probability[:, 1:2],
        )
        return BOCDUpdate(
            indices=indices.clone(),
            observed=observed.to(self.device).clone(),
            changepoint_probability=self.last_changepoint_probability[indices].clone(),
            map_run_length=map_run_length.clone(),
            estimated_run_start=map_run_start.clone(),
            change_probability=(a_map / (a_map + b_map)).clone(),
            concentration=(a_map + b_map).clone(),
            visible_observations=map_visible.clone(),
            a_map=a_map.clone(),
            b_map=b_map.clone(),
            run_length_posterior=posterior,
            a=branch_a,
            b=branch_b,
            total_visible_observations=self.visible_observations[indices].clone(),
            branch_probability=branch_probability,
            branch_run_start=branch_start,
            branch_visible_observations=branch_visible,
            log_predictive_continue=self.last_log_continue[indices].clone(),
            log_predictive_reset=self.last_log_reset[indices].clone(),
            log_bayes_factor=self.last_log_bayes_factor[indices].clone(),
            candidate_probability=self.last_candidate_probability[indices].clone(),
            candidate_run_start=self.last_candidate_run_start[indices].clone(),
        )
