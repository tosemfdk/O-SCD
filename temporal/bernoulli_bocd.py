"""Bounded Beta-Bernoulli Bayesian online changepoint detection."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Literal, Optional

import torch

BocdMode = Literal["exact", "adams_mackay", "map_reset"]


@dataclass(frozen=True)
class BernoulliBOCDConfig:
    prior_a: float = 1.0
    prior_b: float = 1.0
    expected_run_length: Optional[float] = 50.0
    hazard: Optional[float] = None
    max_run_length: int = 64
    min_evidence_mass: float = 1e-6
    open_probability: float = 0.6
    close_probability: float = 0.4
    changepoint_probability: float = 0.5
    min_run_evidence: float = 1.0
    min_visible_observations: int = 1

    def __post_init__(self) -> None:
        for name in ("prior_a", "prior_b"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")
        if isinstance(self.max_run_length, bool) or not isinstance(self.max_run_length, Integral) or self.max_run_length < 1:
            raise ValueError("max_run_length must be a positive integer")
        if isinstance(self.min_visible_observations, bool) or not isinstance(self.min_visible_observations, Integral) or self.min_visible_observations < 0:
            raise ValueError("min_visible_observations must be a nonnegative integer")
        for name in ("min_evidence_mass", "min_run_evidence"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("open_probability", "close_probability", "changepoint_probability"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0, 1]")
        _ = self.resolved_hazard

    @property
    def resolved_hazard(self) -> float:
        if self.hazard is None:
            if self.expected_run_length is None or self.expected_run_length <= 1.0:
                raise ValueError("expected_run_length must be > 1 when hazard is omitted")
            hazard = 1.0 / float(self.expected_run_length)
        else:
            hazard = float(self.hazard)
        if not (0.0 < hazard < 1.0):
            raise ValueError("hazard must be in (0, 1)")
        return hazard


class BOCDUpdate:
    def __init__(
        self,
        indices: torch.Tensor,
        observed: torch.Tensor,
        changepoint_probability: torch.Tensor,
        map_run_length: torch.Tensor,
        estimated_run_start: torch.Tensor,
        change_probability: torch.Tensor,
        concentration: torch.Tensor,
        visible_observations: torch.Tensor,
        a_map: torch.Tensor | None = None,
        b_map: torch.Tensor | None = None,
        beta_a: torch.Tensor | None = None,
        beta_b: torch.Tensor | None = None,
        **extras,
    ) -> None:
        self.indices = indices
        self.observed = observed
        self.changepoint_probability = changepoint_probability
        self.map_run_length = map_run_length
        self.estimated_run_start = estimated_run_start
        self.change_probability = change_probability
        self.concentration = concentration
        self.visible_observations = visible_observations
        self.a_map = a_map if a_map is not None else beta_a
        self.b_map = b_map if b_map is not None else beta_b
        self.beta_a = self.a_map
        self.beta_b = self.b_map
        if self.a_map is None or self.b_map is None:
            self.a_map = change_probability * concentration
            self.b_map = (1.0 - change_probability) * concentration
            self.beta_a = self.a_map
            self.beta_b = self.b_map
        for key, value in extras.items():
            setattr(self, key, value)
    run_length_posterior: Optional[torch.Tensor] = None
    a: Optional[torch.Tensor] = None
    b: Optional[torch.Tensor] = None


# Alias requested by the phase spec.
BOCDUpdateResult = BOCDUpdate


def beta_binomial_log_predictive(s: torch.Tensor, f: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return log B(a+s,b+f)-log B(a,b) for fractional pseudo-counts."""
    return torch.lgamma(a + s) + torch.lgamma(b + f) - torch.lgamma(a + b + s + f) - (
        torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
    )


class BetaBernoulliBOCD:
    """Bounded exact/truncated per-row Beta-Bernoulli BOCD.

    Updates are chunkable via ``row_indices``. Rows whose mass is below
    ``min_evidence_mass`` are unobserved and every stored tensor for those rows is
    preserved exactly. The reset branch uses the prior predictive for the current
    observation and then stores that observation in run length 0; this is a
    valid CP-at-current-observation convention, but it is not the literal
    Adams--MacKay Algorithm 1 recurrence. Use
    :class:`AdamsMacKayBetaBernoulliBOCD` for the paper-faithful convention.
    """

    algorithm = "exact"

    def __init__(self, num_gaussians: int, config: BernoulliBOCDConfig | None = None, *, device=None, dtype=torch.float32):
        if isinstance(num_gaussians, bool) or not isinstance(num_gaussians, Integral) or num_gaussians < 1:
            raise ValueError("num_gaussians must be a positive integer")
        if not torch.empty((), dtype=dtype).is_floating_point():
            raise TypeError("dtype must be floating")
        self.num_gaussians = int(num_gaussians)
        self.num_rows = self.num_gaussians
        self.config = config or BernoulliBOCDConfig()
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype
        self.max_run_length = int(self.config.max_run_length)
        r = self.max_run_length + 1
        self.log_run_probs = torch.full((self.num_gaussians, r), -torch.inf, device=self.device, dtype=dtype)
        self.log_run_probs[:, 0] = 0.0
        self.a = torch.full((self.num_gaussians, r), float(self.config.prior_a), device=self.device, dtype=dtype)
        self.b = torch.full((self.num_gaussians, r), float(self.config.prior_b), device=self.device, dtype=dtype)
        self.total_run_evidence = torch.zeros((self.num_gaussians, r), device=self.device, dtype=dtype)
        self.run_start = torch.zeros((self.num_gaussians, r), device=self.device, dtype=torch.long)
        self.run_visible_observations = torch.zeros(
            (self.num_gaussians, r), device=self.device, dtype=torch.long
        )
        self.visible_observations = torch.zeros(self.num_gaussians, device=self.device, dtype=torch.long)
        self.last_changepoint_probability = torch.zeros(self.num_gaussians, device=self.device, dtype=dtype)
        self.last_timestamp = torch.full((self.num_gaussians,), -1, device=self.device, dtype=torch.long)
        self.map_run_start = torch.zeros(self.num_gaussians, device=self.device, dtype=torch.long)

    @staticmethod
    def integrated_log_predictive(s: torch.Tensor, f: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return beta_binomial_log_predictive(s, f, a, b)

    @property
    def run_length_posterior(self) -> torch.Tensor:
        return self.log_run_probs.exp()

    def posterior(self) -> torch.Tensor:
        return self.run_length_posterior

    @property
    def p_run_zero(self) -> torch.Tensor:
        return self.run_length_posterior[:, 0]

    @property
    def map_run_length(self) -> torch.Tensor:
        return self.log_run_probs.argmax(dim=1).to(torch.long)

    @property
    def estimated_run_start(self) -> torch.Tensor:
        rows = torch.arange(self.num_gaussians, device=self.device)
        return self.run_start[rows, self.map_run_length]

    @property
    def map_beta_a(self) -> torch.Tensor:
        rows = torch.arange(self.num_gaussians, device=self.device)
        return self.a[rows, self.map_run_length]

    @property
    def map_beta_b(self) -> torch.Tensor:
        rows = torch.arange(self.num_gaussians, device=self.device)
        return self.b[rows, self.map_run_length]

    @property
    def change_probability(self) -> torch.Tensor:
        return self.map_beta_a / (self.map_beta_a + self.map_beta_b)

    @property
    def concentration(self) -> torch.Tensor:
        return self.map_beta_a + self.map_beta_b

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "log_run_probs": self.log_run_probs.clone(),
            "a": self.a.clone(),
            "b": self.b.clone(),
            "total_run_evidence": self.total_run_evidence.clone(),
            "run_start": self.run_start.clone(),
            "run_visible_observations": self.run_visible_observations.clone(),
            "visible_observations": self.visible_observations.clone(),
            "last_changepoint_probability": self.last_changepoint_probability.clone(),
            "last_timestamp": self.last_timestamp.clone(),
            "map_run_start": self.map_run_start.clone(),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        for name, target in self.state_dict().items():
            if name not in state:
                raise KeyError(f"missing BOCD state tensor {name!r}")
            value = state[name].to(device=self.device, dtype=target.dtype)
            if value.shape != target.shape:
                raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {tuple(target.shape)}")
            getattr(self, name).copy_(value)

    def _coerce_update(self, *args, **kwargs):
        # Supports both update(indices, s, f, total_mass, timestamp) and
        # update(e_plus, e_minus, *, total_mass=None, row_indices=None, timestamp=None).
        if len(args) >= 4:
            row_indices, s, f, total_mass = args[:4]
            timestamp = args[4] if len(args) >= 5 else kwargs.pop("timestamp", None)
        elif len(args) == 3:
            row_indices, s, f = args
            total_mass = kwargs.pop("total_mass", None)
            timestamp = kwargs.pop("timestamp", None)
        elif len(args) == 2:
            s, f = args
            row_indices = kwargs.pop("row_indices", None)
            total_mass = kwargs.pop("total_mass", None)
            timestamp = kwargs.pop("timestamp", None)
        else:
            raise TypeError("update expects (indices,s,f[,total_mass][,timestamp]) or (e_plus,e_minus,...) ")
        if kwargs:
            raise TypeError(f"unexpected keyword arguments: {sorted(kwargs)}")
        s = torch.as_tensor(s, device=self.device, dtype=self.dtype).detach().flatten()
        f = torch.as_tensor(f, device=self.device, dtype=self.dtype).detach().flatten()
        if s.shape != f.shape:
            raise ValueError("positive and negative counts must have the same shape")
        if (s < 0).any() or (f < 0).any() or ~torch.isfinite(s).all() or ~torch.isfinite(f).all():
            raise ValueError("counts must be finite and nonnegative")
        if row_indices is None:
            if s.numel() != self.num_gaussians:
                raise ValueError("full updates require one count per row")
            indices = torch.arange(self.num_gaussians, device=self.device)
        else:
            indices = torch.as_tensor(row_indices, device=self.device, dtype=torch.long).detach().flatten()
            if indices.numel() != s.numel():
                raise ValueError("indices and counts must have the same length")
            if indices.numel() and ((indices < 0).any() or (indices >= self.num_gaussians).any()):
                raise IndexError("indices out of range")
        if total_mass is None:
            mass = s + f
        else:
            mass = torch.as_tensor(total_mass, device=self.device, dtype=self.dtype).detach().flatten()
            if mass.shape != s.shape:
                raise ValueError("total_mass must match counts")
            if (mass < 0).any() or ~torch.isfinite(mass).all():
                raise ValueError("total_mass must be finite and nonnegative")
        if timestamp is None:
            timestamp = 0
        if isinstance(timestamp, bool) or not isinstance(timestamp, (Integral, Real)) or not math.isfinite(timestamp):
            raise TypeError("timestamp must be finite numeric")
        return indices, s, f, mass, int(timestamp)

    @torch.no_grad()
    def update(self, *args, **kwargs) -> BOCDUpdate:
        indices, s, f, total_mass, timestamp = self._coerce_update(*args, **kwargs)
        observed = total_mass >= float(self.config.min_evidence_mass)
        if not bool(observed.any()):
            return self._snapshot(indices, observed, timestamp)
        obs_idx = indices[observed]
        ss = s[observed]
        ff = f[observed]
        h = torch.as_tensor(self.config.resolved_hazard, device=self.device, dtype=self.dtype)
        log_h = torch.log(h)
        log_1mh = torch.log1p(-h)
        old_log = self.log_run_probs[obs_idx]
        old_a = self.a[obs_idx]
        old_b = self.b[obs_idx]
        old_start = self.run_start[obs_idx].clone()
        never_observed = self.visible_observations[obs_idx] == 0
        if bool(never_observed.any()):
            old_start[never_observed] = int(timestamp)
        pred_growth = beta_binomial_log_predictive(ss[:, None], ff[:, None], old_a, old_b)
        new_log = torch.full_like(old_log, -torch.inf)
        prior_a = torch.full((obs_idx.numel(),), float(self.config.prior_a), device=self.device, dtype=self.dtype)
        prior_b = torch.full((obs_idx.numel(),), float(self.config.prior_b), device=self.device, dtype=self.dtype)
        pred_cp = beta_binomial_log_predictive(ss, ff, prior_a, prior_b)
        new_log[:, 0] = torch.logsumexp(old_log + log_h, dim=1) + pred_cp
        new_log[:, 1:] = old_log[:, :-1] + pred_growth[:, :-1] + log_1mh
        new_log = new_log - torch.logsumexp(new_log, dim=1, keepdim=True)
        new_a = torch.full_like(old_a, float(self.config.prior_a))
        new_b = torch.full_like(old_b, float(self.config.prior_b))
        new_a[:, 0] = float(self.config.prior_a) + ss
        new_b[:, 0] = float(self.config.prior_b) + ff
        new_a[:, 1:] = old_a[:, :-1] + ss[:, None]
        new_b[:, 1:] = old_b[:, :-1] + ff[:, None]
        new_e = torch.zeros_like(old_a)
        new_e[:, 0] = ss + ff
        new_e[:, 1:] = self.total_run_evidence[obs_idx, :-1] + (ss + ff)[:, None]
        new_start = torch.zeros_like(old_start)
        new_start[:, 0] = int(timestamp)
        new_start[:, 1:] = old_start[:, :-1]
        new_visible = torch.zeros_like(self.run_visible_observations[obs_idx])
        new_visible[:, 0] = 1
        new_visible[:, 1:] = self.run_visible_observations[obs_idx, :-1] + 1
        self.log_run_probs[obs_idx] = new_log
        self.a[obs_idx] = new_a
        self.b[obs_idx] = new_b
        self.total_run_evidence[obs_idx] = new_e
        self.run_start[obs_idx] = new_start
        self.run_visible_observations[obs_idx] = new_visible
        self.visible_observations[obs_idx] += 1
        self.last_changepoint_probability[obs_idx] = new_log[:, 0].exp()
        self.last_timestamp[obs_idx] = timestamp
        map_r = new_log.argmax(dim=1).to(torch.long)
        rows = torch.arange(obs_idx.numel(), device=self.device)
        self.map_run_start[obs_idx] = self.run_start[obs_idx][rows, map_r]
        return self._snapshot(indices, observed, timestamp)

    def _snapshot(self, indices: torch.Tensor, observed: torch.Tensor, timestamp: int) -> BOCDUpdate:
        # Index in log space first so chunked updates never materialize a full
        # [all_gaussians, run_lengths] posterior just to return one chunk.
        post = self.log_run_probs[indices].exp()
        map_r = post.argmax(dim=1).to(torch.long)
        rows = torch.arange(indices.numel(), device=self.device)
        a_sel = self.a[indices]
        b_sel = self.b[indices]
        a_map = a_sel[rows, map_r]
        b_map = b_sel[rows, map_r]
        return BOCDUpdate(
            indices=indices.clone(),
            observed=observed.to(self.device).clone(),
            changepoint_probability=post[:, 0].clone(),
            map_run_length=map_r.clone(),
            estimated_run_start=self.run_start[indices][rows, map_r].clone(),
            change_probability=(a_map / (a_map + b_map)).clone(),
            concentration=(a_map + b_map).clone(),
            visible_observations=self.run_visible_observations[indices][
                rows, map_r
            ].clone(),
            a_map=a_map.clone(),
            b_map=b_map.clone(),
            run_length_posterior=post.clone(),
            a=a_sel.clone(),
            b=b_sel.clone(),
            total_visible_observations=self.visible_observations[indices].clone(),
        )


class AdamsMacKayBetaBernoulliBOCD(BetaBernoulliBOCD):
    """Paper-faithful Adams--MacKay Algorithm 1 Beta-Bernoulli BOCD.

    This variant uses each previous run's current-observation predictive in
    both the growth and changepoint sums. The changepoint branch is interpreted
    as a CP after the current observation: its post-update run length 0 state is
    reset to the prior with zero observations, and its global ``run_start`` is
    ``timestamp + 1``. Growth branches absorb the current counts and carry their
    previous global run start.

    With a constant hazard and no truncation, ``P(r_t=0)`` equals the hazard;
    it is not an evidence-driven lifecycle commit probability. This variant is
    diagnostic-only until a recent/old start-region posterior policy is wired
    explicitly into the lifecycle controller.
    """

    algorithm = "adams_mackay"

    @torch.no_grad()
    def update(self, *args, **kwargs) -> BOCDUpdate:
        indices, s, f, total_mass, timestamp = self._coerce_update(*args, **kwargs)
        observed = total_mass >= float(self.config.min_evidence_mass)
        if not bool(observed.any()):
            return self._snapshot(indices, observed, timestamp)
        obs_idx = indices[observed]
        ss = s[observed]
        ff = f[observed]
        h = torch.as_tensor(self.config.resolved_hazard, device=self.device, dtype=self.dtype)
        log_h = torch.log(h)
        log_1mh = torch.log1p(-h)
        old_log = self.log_run_probs[obs_idx]
        old_a = self.a[obs_idx]
        old_b = self.b[obs_idx]
        old_e = self.total_run_evidence[obs_idx]
        old_visible = self.run_visible_observations[obs_idx]
        old_start = self.run_start[obs_idx].clone()
        never_observed = self.visible_observations[obs_idx] == 0
        if bool(never_observed.any()):
            old_start[never_observed] = int(timestamp)

        pred = beta_binomial_log_predictive(ss[:, None], ff[:, None], old_a, old_b)
        new_log = torch.full_like(old_log, -torch.inf)
        new_log[:, 0] = torch.logsumexp(old_log + pred + log_h, dim=1)
        new_log[:, 1:] = old_log[:, :-1] + pred[:, :-1] + log_1mh
        new_log = new_log - torch.logsumexp(new_log, dim=1, keepdim=True)

        new_a = torch.full_like(old_a, float(self.config.prior_a))
        new_b = torch.full_like(old_b, float(self.config.prior_b))
        new_a[:, 1:] = old_a[:, :-1] + ss[:, None]
        new_b[:, 1:] = old_b[:, :-1] + ff[:, None]

        new_e = torch.zeros_like(old_e)
        new_e[:, 1:] = old_e[:, :-1] + (ss + ff)[:, None]

        new_start = torch.zeros_like(old_start)
        new_start[:, 0] = int(timestamp) + 1
        new_start[:, 1:] = old_start[:, :-1]

        new_visible = torch.zeros_like(old_visible)
        new_visible[:, 1:] = old_visible[:, :-1] + 1

        self.log_run_probs[obs_idx] = new_log
        self.a[obs_idx] = new_a
        self.b[obs_idx] = new_b
        self.total_run_evidence[obs_idx] = new_e
        self.run_start[obs_idx] = new_start
        self.run_visible_observations[obs_idx] = new_visible
        self.visible_observations[obs_idx] += 1
        self.last_changepoint_probability[obs_idx] = new_log[:, 0].exp()
        self.last_timestamp[obs_idx] = timestamp
        map_r = new_log.argmax(dim=1).to(torch.long)
        rows = torch.arange(obs_idx.numel(), device=self.device)
        self.map_run_start[obs_idx] = self.run_start[obs_idx][rows, map_r]
        return self._snapshot(indices, observed, timestamp)


class MAPResetBernoulliFilter:
    """Explicit O(N) MAP alternative: one Beta run per Gaussian row.

    Unlike exact BOCD, this filter does not store a bounded [N, R] posterior.
    It keeps only the current MAP run sufficient statistics and start timestamp,
    so persistent state is linear in the number of Gaussians.
    """

    algorithm = "map_reset"

    def __init__(self, num_gaussians: int, config: BernoulliBOCDConfig | None = None, *, device=None, dtype=torch.float32):
        if isinstance(num_gaussians, bool) or not isinstance(num_gaussians, Integral) or num_gaussians < 1:
            raise ValueError("num_gaussians must be a positive integer")
        if not torch.empty((), dtype=dtype).is_floating_point():
            raise TypeError("dtype must be floating")
        self.num_gaussians = int(num_gaussians)
        self.num_rows = self.num_gaussians
        self.config = config or BernoulliBOCDConfig()
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype
        self.max_run_length = int(self.config.max_run_length)
        self.a = torch.full((self.num_gaussians,), float(self.config.prior_a), device=self.device, dtype=dtype)
        self.b = torch.full((self.num_gaussians,), float(self.config.prior_b), device=self.device, dtype=dtype)
        self.total_run_evidence = torch.zeros(self.num_gaussians, device=self.device, dtype=dtype)
        self.map_run_length_tensor = torch.zeros(self.num_gaussians, device=self.device, dtype=torch.long)
        self.run_start = torch.zeros(self.num_gaussians, device=self.device, dtype=torch.long)
        self.run_visible_observations = torch.zeros(
            self.num_gaussians, device=self.device, dtype=torch.long
        )
        self.visible_observations = torch.zeros(self.num_gaussians, device=self.device, dtype=torch.long)
        self.last_changepoint_probability = torch.zeros(self.num_gaussians, device=self.device, dtype=dtype)
        self.last_timestamp = torch.full((self.num_gaussians,), -1, device=self.device, dtype=torch.long)
        self.map_run_start = self.run_start

    @staticmethod
    def integrated_log_predictive(s: torch.Tensor, f: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return beta_binomial_log_predictive(s, f, a, b)

    @property
    def map_run_length(self) -> torch.Tensor:
        return self.map_run_length_tensor

    @property
    def estimated_run_start(self) -> torch.Tensor:
        return self.run_start

    @property
    def map_beta_a(self) -> torch.Tensor:
        return self.a

    @property
    def map_beta_b(self) -> torch.Tensor:
        return self.b

    @property
    def change_probability(self) -> torch.Tensor:
        return self.a / (self.a + self.b)

    @property
    def concentration(self) -> torch.Tensor:
        return self.a + self.b

    @property
    def p_run_zero(self) -> torch.Tensor:
        return (self.map_run_length_tensor == 0).to(self.dtype)

    @property
    def run_length_posterior(self) -> torch.Tensor:
        posterior = torch.zeros((self.num_gaussians, self.max_run_length + 1), device=self.device, dtype=self.dtype)
        posterior.scatter_(1, self.map_run_length_tensor.clamp_max(self.max_run_length)[:, None], 1.0)
        return posterior

    def posterior(self) -> torch.Tensor:
        return self.run_length_posterior

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "a": self.a.clone(),
            "b": self.b.clone(),
            "total_run_evidence": self.total_run_evidence.clone(),
            "map_run_length": self.map_run_length_tensor.clone(),
            "run_start": self.run_start.clone(),
            "run_visible_observations": self.run_visible_observations.clone(),
            "visible_observations": self.visible_observations.clone(),
            "last_changepoint_probability": self.last_changepoint_probability.clone(),
            "last_timestamp": self.last_timestamp.clone(),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        name_to_attr = {"map_run_length": "map_run_length_tensor"}
        for name, target in self.state_dict().items():
            if name not in state:
                raise KeyError(f"missing BOCD state tensor {name!r}")
            value = state[name].to(device=self.device, dtype=target.dtype)
            if value.shape != target.shape:
                raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {tuple(target.shape)}")
            getattr(self, name_to_attr.get(name, name)).copy_(value)

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
        prior_a = torch.full_like(ss, float(self.config.prior_a))
        prior_b = torch.full_like(ff, float(self.config.prior_b))
        log_continue = beta_binomial_log_predictive(ss, ff, self.a[idx], self.b[idx]) + math.log1p(-self.config.resolved_hazard)
        log_reset = beta_binomial_log_predictive(ss, ff, prior_a, prior_b) + math.log(self.config.resolved_hazard)
        reset_probability = torch.softmax(
            torch.stack((log_continue, log_reset), dim=1), dim=1
        )[:, 1]
        reset = reset_probability >= float(
            self.config.changepoint_probability
        )
        first_observation = self.visible_observations[idx] == 0
        start_new_run = reset | first_observation
        self.a[idx] = torch.where(reset, prior_a + ss, self.a[idx] + ss)
        self.b[idx] = torch.where(reset, prior_b + ff, self.b[idx] + ff)
        self.total_run_evidence[idx] = torch.where(reset, ss + ff, self.total_run_evidence[idx] + ss + ff)
        self.map_run_length_tensor[idx] = torch.where(
            reset,
            torch.zeros_like(self.map_run_length_tensor[idx]),
            (self.map_run_length_tensor[idx] + 1).clamp_max(self.max_run_length),
        )
        self.run_start[idx] = torch.where(start_new_run, torch.full_like(self.run_start[idx], int(timestamp)), self.run_start[idx])
        self.run_visible_observations[idx] = torch.where(
            start_new_run,
            torch.ones_like(self.run_visible_observations[idx]),
            self.run_visible_observations[idx] + 1,
        )
        self.visible_observations[idx] += 1
        self.last_changepoint_probability[idx] = reset_probability
        self.last_timestamp[idx] = int(timestamp)
        return self._snapshot(indices, observed)

    def _snapshot(self, indices: torch.Tensor, observed: torch.Tensor) -> BOCDUpdate:
        p = self.change_probability[indices]
        concentration = self.concentration[indices]
        posterior = torch.zeros((indices.numel(), self.max_run_length + 1), device=self.device, dtype=self.dtype)
        posterior.scatter_(1, self.map_run_length_tensor[indices].clamp_max(self.max_run_length)[:, None], 1.0)
        return BOCDUpdate(
            indices=indices.clone(),
            observed=observed.to(self.device).clone(),
            changepoint_probability=self.last_changepoint_probability[indices].clone(),
            map_run_length=self.map_run_length_tensor[indices].clone(),
            estimated_run_start=self.run_start[indices].clone(),
            change_probability=p.clone(),
            concentration=concentration.clone(),
            visible_observations=self.run_visible_observations[indices].clone(),
            a_map=self.a[indices].clone(),
            b_map=self.b[indices].clone(),
            run_length_posterior=posterior,
            a=self.a[indices].clone(),
            b=self.b[indices].clone(),
            total_visible_observations=self.visible_observations[indices].clone(),
        )


def make_bocd_filter(mode: BocdMode, num_gaussians: int, config: BernoulliBOCDConfig | None = None, *, device=None, dtype=torch.float32) -> BetaBernoulliBOCD:
    """Construct a BOCD implementation without implying lifecycle compatibility.

    In particular, ``adams_mackay`` exposes literal ``P(r_t=0)``; callers must
    not use that scalar as the existing lifecycle changepoint score.
    """
    if mode == "exact":
        return BetaBernoulliBOCD(num_gaussians, config, device=device, dtype=dtype)
    if mode == "adams_mackay":
        return AdamsMacKayBetaBernoulliBOCD(num_gaussians, config, device=device, dtype=dtype)
    if mode == "map_reset":
        return MAPResetBernoulliFilter(num_gaussians, config, device=device, dtype=dtype)
    raise ValueError("bocd mode must be 'exact', 'adams_mackay', or 'map_reset'")
