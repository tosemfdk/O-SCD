# Per-Gaussian Beta change state (stage 11 B, docs/target_gaussian_nbv.md §6).
#
# Each Gaussian g carries p_g ~ Beta(a_g, b_g) over "is changed". Observations
# arrive as soft pseudo-counts weighted by compositing responsibility:
#   e1_g = s * sum_p r_g,p * M_p,   e0_g = s * sum_p r_g,p * (1 - M_p)
# (s = pseudo_count_scale). Candidate EIG needs no image:
#   EIG_g(tau) = H[Beta(a,b)] - H[Beta(a + m*tau, b + (1-m)*tau)],  m = a/(a+b).
# Everything is vectorized over all Gaussians (float64 on the model's device).

from __future__ import annotations

import torch
from torch.special import digamma, gammaln


def beta_entropy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Differential entropy of Beta(a,b), elementwise.
    H = ln B(a,b) - (a-1) psi(a) - (b-1) psi(b) + (a+b-2) psi(a+b)."""
    return (gammaln(a) + gammaln(b) - gammaln(a + b)
            - (a - 1.0) * digamma(a) - (b - 1.0) * digamma(b)
            + (a + b - 2.0) * digamma(a + b))


class BetaChangeState:
    def __init__(self, n: int, prior_a: float = 1.0, prior_b: float = 1.0,
                 pseudo_count_scale: float = 1.0, device: str = "cuda"):
        if prior_a <= 0 or prior_b <= 0:
            raise ValueError("Beta priors must be > 0")
        if pseudo_count_scale <= 0:
            raise ValueError("pseudo_count_scale must be > 0")
        self.prior_a, self.prior_b = float(prior_a), float(prior_b)
        self.pseudo_count_scale = float(pseudo_count_scale)
        self.a = torch.full((n,), prior_a, dtype=torch.float64, device=device)
        self.b = torch.full((n,), prior_b, dtype=torch.float64, device=device)
        self.n_updates = 0

    def __len__(self) -> int:
        return self.a.shape[0]

    def posterior_mean(self) -> torch.Tensor:
        return self.a / (self.a + self.b)

    def entropy(self) -> torch.Tensor:
        return beta_entropy(self.a, self.b)

    def expected_eig(self, tau: torch.Tensor | float) -> torch.Tensor:
        """EIG_g for expected responsibility tau (scalar or per-Gaussian).
        Uses the CURRENT posterior mean to split tau into expected counts."""
        tau = torch.as_tensor(tau, dtype=torch.float64, device=self.a.device)
        m = self.posterior_mean()
        return self.entropy() - beta_entropy(self.a + m * tau,
                                             self.b + (1.0 - m) * tau)

    def unit_eig(self) -> torch.Tensor:
        """Per-Gaussian EIG of ONE unit of responsibility (tau=1). Used as the
        weight w_g of the frame scorer: a single weighted probe render then
        yields sum_g w_g * tau_g(c) in one pass. (The exact d EIG/d tau at the
        uniform prior is 0 — EIG is second-order there — so a unit probe, not
        a derivative, is the right linearization.)"""
        return self.expected_eig(1.0).clamp_min(0.0)

    def update(self, e1: torch.Tensor, e0: torch.Tensor) -> None:
        """Commit OBSERVED soft counts (already responsibility-weighted).
        pseudo_count_scale is applied here, exactly once."""
        if e1.shape != self.a.shape or e0.shape != self.b.shape:
            raise ValueError(f"count shape {tuple(e1.shape)} != state {tuple(self.a.shape)}")
        if (e1 < -1e-9).any() or (e0 < -1e-9).any():
            raise ValueError("negative pseudo-counts")
        self.a = self.a + self.pseudo_count_scale * e1.double().clamp_min(0.0)
        self.b = self.b + self.pseudo_count_scale * e0.double().clamp_min(0.0)
        self.n_updates += 1

    def append(self, k: int) -> None:
        """Grow by k Gaussians (densify clone/split); children get the prior."""
        dev = self.a.device
        self.a = torch.cat([self.a, torch.full((k,), self.prior_a,
                                               dtype=torch.float64, device=dev)])
        self.b = torch.cat([self.b, torch.full((k,), self.prior_b,
                                               dtype=torch.float64, device=dev)])

    def save(self, path: str) -> None:
        torch.save({"a": self.a, "b": self.b, "prior_a": self.prior_a,
                    "prior_b": self.prior_b,
                    "pseudo_count_scale": self.pseudo_count_scale,
                    "n_updates": self.n_updates}, path)

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> "BetaChangeState":
        d = torch.load(path, map_location=device, weights_only=False)
        st = cls(d["a"].shape[0], d["prior_a"], d["prior_b"],
                 d["pseudo_count_scale"], device=device)
        st.a, st.b, st.n_updates = d["a"], d["b"], d["n_updates"]
        return st
