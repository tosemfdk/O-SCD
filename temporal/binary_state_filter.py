"""Direct two-state Bayesian filter for causal active/inactive change state.

This module is intentionally independent from the BOCD implementations.  It
tracks only ``P(z_t=1 | D_1:t)`` for each Gaussian and the normalized two-state
transition joint for the current observation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral

import torch


@dataclass(frozen=True)
class BinaryStateFilterConfig:
    """Hyperparameters for :class:`BinaryStateFilter`.

    ``emission_reliability`` is the fixed sensor reliability eta.  The two
    transition priors are the Markov flip probabilities.  ``initial_p_active``
    is deliberately neutral by default.  Rows whose supplied observation mass is
    not greater than ``min_observation_mass`` are treated as unobserved and are
    left bitwise unchanged.
    """

    emission_reliability: float = 0.9
    inactive_to_active_prior: float = 0.01
    active_to_inactive_prior: float = 0.01
    initial_p_active: float = 0.5
    min_observation_mass: float = 0.0
    eps: float = 1e-12

    def __post_init__(self) -> None:
        eta = float(self.emission_reliability)
        if not math.isfinite(eta) or not (0.5 < eta < 1.0):
            raise ValueError("emission_reliability must be finite and in (0.5, 1)")
        for name in ("inactive_to_active_prior", "active_to_inactive_prior"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not (0.0 < value < 1.0):
                raise ValueError(f"{name} must be finite and in (0, 1)")
        initial = float(self.initial_p_active)
        if not math.isfinite(initial) or not (0.0 <= initial <= 1.0):
            raise ValueError("initial_p_active must be finite and in [0, 1]")
        if not math.isfinite(float(self.min_observation_mass)) or self.min_observation_mass < 0:
            raise ValueError("min_observation_mass must be finite and nonnegative")
        if not math.isfinite(float(self.eps)) or self.eps <= 0:
            raise ValueError("eps must be finite and positive")


@dataclass(frozen=True)
class BinaryStateFilterUpdate:
    """Result for one vector/chunk update."""

    indices: torch.Tensor
    observed: torch.Tensor
    p_active: torch.Tensor
    p_00: torch.Tensor
    p_01: torch.Tensor
    p_10: torch.Tensor
    p_11: torch.Tensor
    p_flip: torch.Tensor
    q: torch.Tensor
    evidence_strength: torch.Tensor
    visible_observations: torch.Tensor
    last_timestamp: torch.Tensor
    timestamp: int


class BinaryStateFilter:
    """Causal direct binary filter over Gaussian active/inactive state."""

    algorithm = "direct_binary_state_filter"

    def __init__(
        self,
        num_gaussians: int,
        config: BinaryStateFilterConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if isinstance(num_gaussians, bool) or not isinstance(num_gaussians, Integral) or num_gaussians < 1:
            raise ValueError("num_gaussians must be a positive integer")
        if not torch.is_floating_point(torch.empty((), dtype=dtype)):
            raise TypeError("dtype must be floating")
        self.config = config or BinaryStateFilterConfig()
        self.p_active = torch.full(
            (int(num_gaussians),),
            float(self.config.initial_p_active),
            device=device,
            dtype=dtype,
        )
        self.visible_observations = torch.zeros(
            int(num_gaussians), device=device, dtype=torch.long
        )
        self.last_timestamp = torch.full(
            (int(num_gaussians),), -1, device=device, dtype=torch.long
        )

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "p_active": self.p_active.clone(),
            "visible_observations": self.visible_observations.clone(),
            "last_timestamp": self.last_timestamp.clone(),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        for name, target in self.state_dict().items():
            if name not in state:
                raise KeyError(f"missing filter state tensor {name!r}")
            value = state[name].to(device=target.device, dtype=target.dtype)
            if value.shape != target.shape:
                raise ValueError(
                    f"{name} has shape {tuple(value.shape)}, expected {tuple(target.shape)}"
                )
            getattr(self, name).copy_(value)

    def _normalize_indices(self, indices, length: int) -> torch.Tensor:
        if indices is None:
            out = torch.arange(length, device=self.p_active.device, dtype=torch.long)
        else:
            raw = indices if isinstance(indices, torch.Tensor) else torch.as_tensor(indices)
            if raw.dtype == torch.bool or raw.dtype.is_floating_point or raw.dtype.is_complex:
                raise TypeError("indices must contain integers")
            out = raw.to(device=self.p_active.device, dtype=torch.long).flatten()
        if out.numel() != length:
            raise ValueError("indices length must match evidence length")
        if out.numel() != torch.unique(out).numel():
            raise ValueError("BinaryStateFilter update indices must be unique")
        if out.numel() and (
            bool((out < 0).any()) or bool((out >= self.p_active.shape[0]).any())
        ):
            raise IndexError("BinaryStateFilter update indices are out of range")
        return out

    @torch.no_grad()
    def update(
        self,
        delta_a,
        delta_b,
        total_mass=None,
        *,
        indices=None,
        timestamp: int,
    ) -> BinaryStateFilterUpdate:
        """Update a chunk of rows from immutable-reference evidence counts.

        ``delta_a`` is positive/active evidence and ``delta_b`` is
        negative/inactive evidence.  ``total_mass`` is used only for observation
        presence; the emission strength is always ``delta_a + delta_b`` as in
        the ablation contract.
        """

        if isinstance(timestamp, bool) or not isinstance(timestamp, int):
            raise TypeError("timestamp must be an integer")
        device = self.p_active.device
        dtype = self.p_active.dtype
        da = torch.as_tensor(delta_a, device=device, dtype=dtype).flatten()
        db = torch.as_tensor(delta_b, device=device, dtype=dtype).flatten()
        if da.shape != db.shape:
            raise ValueError("delta_a and delta_b must have the same flattened shape")
        if not bool(torch.isfinite(da).all()) or not bool(torch.isfinite(db).all()):
            raise ValueError("evidence counts must be finite")
        if bool((da < 0).any()) or bool((db < 0).any()):
            raise ValueError("evidence counts must be nonnegative")
        idx = self._normalize_indices(indices, da.numel())
        if total_mass is None:
            tm = da + db
        else:
            tm = torch.as_tensor(total_mass, device=device, dtype=dtype).flatten()
            if tm.shape != da.shape:
                raise ValueError("total_mass length must match evidence length")
            if not bool(torch.isfinite(tm).all()):
                raise ValueError("total_mass must be finite")
            if bool((tm < 0).any()):
                raise ValueError("total_mass must be nonnegative")

        strength = da + db
        observed = (
            (tm > 0)
            & (tm >= float(self.config.min_observation_mass))
            & (strength > 0)
        )
        numeric_eps = max(float(self.config.eps), float(torch.finfo(dtype).eps))
        q = da / (strength + numeric_eps)

        old = self.p_active[idx].clone()
        p00 = torch.zeros_like(old)
        p01 = torch.zeros_like(old)
        p10 = torch.zeros_like(old)
        p11 = torch.zeros_like(old)
        pflip = torch.zeros_like(old)
        new_p = old.clone()

        if bool(observed.any()):
            pos = torch.nonzero(observed, as_tuple=False).flatten()
            b = old[pos].clamp(numeric_eps, 1.0 - numeric_eps)
            q_obs = q[pos].clamp(0.0, 1.0)
            w_obs = strength[pos]
            eta = torch.as_tensor(float(self.config.emission_reliability), device=device, dtype=dtype)
            one_minus_eta = 1.0 - eta
            log_l1 = w_obs * (q_obs * eta.log() + (1.0 - q_obs) * one_minus_eta.log())
            log_l0 = w_obs * (q_obs * one_minus_eta.log() + (1.0 - q_obs) * eta.log())
            prior01 = torch.as_tensor(float(self.config.inactive_to_active_prior), device=device, dtype=dtype)
            prior10 = torch.as_tensor(float(self.config.active_to_inactive_prior), device=device, dtype=dtype)
            logw = torch.stack(
                (
                    (1.0 - b).log() + (1.0 - prior01).log() + log_l0,
                    (1.0 - b).log() + prior01.log() + log_l1,
                    b.log() + prior10.log() + log_l0,
                    b.log() + (1.0 - prior10).log() + log_l1,
                ),
                dim=1,
            )
            joint = torch.softmax(logw, dim=1)
            p00[pos], p01[pos], p10[pos], p11[pos] = joint.unbind(dim=1)
            pflip[pos] = p01[pos] + p10[pos]
            new_p[pos] = p01[pos] + p11[pos]

            self.p_active[idx[pos]] = new_p[pos]
            self.visible_observations[idx[pos]] += 1
            self.last_timestamp[idx[pos]] = int(timestamp)

        return BinaryStateFilterUpdate(
            indices=idx.clone(),
            observed=observed.clone(),
            p_active=new_p.clone(),
            p_00=p00,
            p_01=p01,
            p_10=p10,
            p_11=p11,
            p_flip=pflip,
            q=q.clone(),
            evidence_strength=strength.clone(),
            visible_observations=self.visible_observations[idx].clone(),
            last_timestamp=self.last_timestamp[idx].clone(),
            timestamp=int(timestamp),
        )
