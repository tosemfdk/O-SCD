# Per-Gaussian importance / confidence state for the Part 3.1 diagnostic
# (spec §4-§10). Pure tensor math: it takes the responsibility-weighted cue
# (q), the frame saturation (s) and the accumulated 3x3 position blocks (H)
# and returns every scalar the diagnostic renders.
#
# This module NEVER touches ground truth — GT enters only in
# experiments/analyze_rchange_importance.py, which is what keeps the
# "gt_used_for_importance: false" claim structural rather than a promise
# (spec §1.7, test §21.4). It also never imports the renderer, so it is
# CPU-testable with synthetic state.
#
# Shapes: V = number of query frames (25), N = number of Gaussians.
#   q[v, i] in [0,1]  responsibility-weighted cue of Gaussian i in frame v
#   s[v, i] in [0,1]  saturated visible mass of Gaussian i in frame v

from __future__ import annotations

from dataclasses import dataclass, field

import torch

EPS = 1e-8


@dataclass(frozen=True)
class DiagnosticConfig:
    eta: float = 1.0
    tau0_mode: str = "median"          # median | q25 | q75 | fixed
    tau0_value: float | None = None    # used when tau0_mode == "fixed"
    kappa0_sweep: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
    kappa0_default: float = 4.0        # the S used by the required variants
    support_min: float = 0.1           # s >= this counts as an effective view
    conflict_threshold: float = 0.5    # |q - m_-v| >= this is a conflicting view
    lam: float = 1e-6                  # H = lam*I + sum_v B_v
    sweep_epsilon: float = 0.05        # floor in the exponent sweep (§10)

    def key_dict(self) -> dict:
        return {
            "eta": self.eta, "tau0_mode": self.tau0_mode,
            "tau0_value": self.tau0_value,
            "kappa0_sweep": list(self.kappa0_sweep),
            "kappa0_default": self.kappa0_default,
            "support_min": self.support_min,
            "conflict_threshold": self.conflict_threshold,
            "lam": self.lam, "sweep_epsilon": self.sweep_epsilon,
        }


def saturation_scale(tau_all: torch.Tensor, cfg: DiagnosticConfig) -> float:
    """tau_0 from the scene's nonzero visible-mass distribution (§3).

    tau_all: (V, N) raw per-frame visible mass. Only nonzero entries count —
    a Gaussian outside the frustum carries no information about the scale."""
    nz = tau_all[tau_all > 0]
    if cfg.tau0_mode == "fixed":
        if cfg.tau0_value is None or cfg.tau0_value <= 0:
            raise ValueError("tau0_mode='fixed' needs a positive tau0_value")
        return float(cfg.tau0_value)
    if nz.numel() == 0:
        raise ValueError("no nonzero visible mass in the scene")
    q = {"median": 0.5, "q25": 0.25, "q75": 0.75}.get(cfg.tau0_mode)
    if q is None:
        raise ValueError(f"unknown tau0_mode {cfg.tau0_mode!r}")
    return float(torch.quantile(nz.double(), q))


def saturate(tau_all: torch.Tensor, tau0: float) -> torch.Tensor:
    """s = min(tau / tau_0, 1): one frame that happens to own a Gaussian's
    pixels must not outvote the rest (§3)."""
    if tau0 <= 0:
        raise ValueError("tau0 must be positive")
    return (tau_all / tau0).clamp(0.0, 1.0)


@dataclass
class GaussianState:
    """Everything in §4-§8, per Gaussian. All (N,) float64 unless noted."""
    m: torch.Tensor                    # Beta mean change belief
    a: torch.Tensor
    b: torch.Tensor
    kappa: torch.Tensor                # evidence concentration
    e_pos: torch.Tensor
    e_neg: torch.Tensor
    support: torch.Tensor              # S at kappa0_default
    support_by_kappa0: dict[float, torch.Tensor]
    n_view: torch.Tensor               # effective observing views (count)
    agreement: torch.Tensor            # A, LOO consensus
    uncertainty: torch.Tensor          # U, binary entropy of m in bits
    cue_variance: torch.Tensor         # s-weighted variance of q
    pos_neg_ratio: torch.Tensor
    loo_min: torch.Tensor
    loo_median: torch.Tensor
    loo_max: torch.Tensor
    n_agree: torch.Tensor
    n_conflict: torch.Tensor
    n_loo_valid: torch.Tensor
    observability: torch.Tensor | None = None   # O in [0,1]
    g_raw: torch.Tensor | None = None           # logdet(H), unnormalized
    eig_min: torch.Tensor | None = None
    eig_max: torch.Tensor | None = None
    condition: torch.Tensor | None = None
    n_dir_obs: torch.Tensor | None = None
    c_raw: torch.Tensor | None = None           # sigmoid(change_feature)
    extra: dict = field(default_factory=dict)

    def confidence(self) -> torch.Tensor:
        """Conf = S * A * O (§8). Requires observability."""
        return self.support * self.agreement * self._O()

    def verification(self) -> torch.Tensor:
        """Verify = m * (1 - Conf): believed-changed but not yet confirmed."""
        return self.m * (1.0 - self.confidence())

    def _O(self) -> torch.Tensor:
        if self.observability is None:
            raise ValueError("observability was never computed")
        return self.observability


def gaussian_state(q: torch.Tensor, s: torch.Tensor,
                   cfg: DiagnosticConfig) -> GaussianState:
    """§4-§6, §8: Beta state, support, LOO agreement, uncertainty.

    Note kappa = eta * sum_v s[v] exactly (the positive and negative evidence
    of a view sum to its saturated mass), so support measures HOW MUCH the
    Gaussian was looked at, never what the cue said about it."""
    if q.shape != s.shape or q.dim() != 2:
        raise ValueError(f"q/s must both be (V, N); got {tuple(q.shape)}, "
                         f"{tuple(s.shape)}")
    q = q.double().clamp(0.0, 1.0)
    s = s.double().clamp(0.0, 1.0)

    e_pos = (s * q).sum(dim=0)
    e_neg = (s * (1.0 - q)).sum(dim=0)
    a = 1.0 + cfg.eta * e_pos
    b = 1.0 + cfg.eta * e_neg
    m = a / (a + b)
    kappa = cfg.eta * (e_pos + e_neg)

    support_by_k = {float(k): 1.0 - torch.exp(-kappa / float(k))
                    for k in cfg.kappa0_sweep}
    k_def = float(cfg.kappa0_default)
    support = support_by_k.get(k_def, 1.0 - torch.exp(-kappa / k_def))
    n_view = (s >= cfg.support_min).sum(dim=0).double()

    # ---- leave-one-out consensus (§6) --------------------------------------
    # The frame's own cue is already inside the all-25 R_change, so comparing a
    # frame against the full consensus would score its own contribution.
    num, den = (s * q).sum(dim=0), s.sum(dim=0)
    den_loo = den.unsqueeze(0) - s                       # (V, N)
    num_loo = num.unsqueeze(0) - s * q
    # A view is only allowed to vote when a consensus exists WITHOUT it (§6)
    # and when it actually observed the Gaussian: for s ~ 0 the ratio q = e1/tau
    # is 0/0 -> 0, which would otherwise register as a loud disagreement in the
    # per-view diagnostics (the s-weighted aggregate already discounts it).
    valid = (den_loo >= EPS) & (s >= cfg.support_min)
    m_loo = num_loo / (den_loo + EPS)
    dev = (q - m_loo).abs()
    agree_v = torch.where(valid, 1.0 - dev, torch.zeros_like(dev)).clamp(0, 1)

    w = s * valid.double()
    agreement = ((w * agree_v).sum(dim=0) / (w.sum(dim=0) + EPS)).clamp(0, 1)

    # per-Gaussian LOO spread, over valid views only
    big = torch.full_like(agree_v, float("inf"))
    loo_min = torch.where(valid, agree_v, big).min(dim=0).values
    loo_max = torch.where(valid, agree_v, -big).max(dim=0).values
    n_valid = valid.sum(dim=0)
    loo_min = torch.where(n_valid > 0, loo_min, torch.zeros_like(loo_min))
    loo_max = torch.where(n_valid > 0, loo_max, torch.zeros_like(loo_max))
    loo_median = _masked_median(agree_v, valid)

    conflict = valid & (dev >= cfg.conflict_threshold)
    n_conflict = conflict.sum(dim=0).double()
    n_agree = (valid & ~conflict).sum(dim=0).double()

    m_w = num / (den + EPS)
    cue_var = (s * (q - m_w.unsqueeze(0)) ** 2).sum(dim=0) / (den + EPS)

    return GaussianState(
        m=m, a=a, b=b, kappa=kappa, e_pos=e_pos, e_neg=e_neg,
        support=support, support_by_kappa0=support_by_k, n_view=n_view,
        agreement=agreement, uncertainty=binary_entropy(m),
        cue_variance=cue_var, pos_neg_ratio=e_pos / (e_neg + EPS),
        loo_min=loo_min, loo_median=loo_median, loo_max=loo_max,
        n_agree=n_agree, n_conflict=n_conflict, n_loo_valid=n_valid.double(),
    )


def _masked_median(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Column-wise median of x over the rows where valid; 0 where none are."""
    big = torch.full_like(x, float("nan"))
    xv = torch.where(valid, x, big)
    med = xv.nanmedian(dim=0).values
    return torch.nan_to_num(med, nan=0.0)


def binary_entropy(m: torch.Tensor) -> torch.Tensor:
    """Predictive Bernoulli entropy in bits (§8), safe at m in {0, 1}."""
    p = m.double().clamp(EPS, 1.0 - EPS)
    h = -(p * torch.log(p) + (1 - p) * torch.log1p(-p)) / torch.log(
        torch.tensor(2.0, dtype=torch.float64, device=p.device))
    return h.clamp(0.0, 1.0)


def observability(H: torch.Tensor, cfg: DiagnosticConfig) -> dict:
    """§7: geometric observability from the accumulated 3x3 position blocks.

    H: (N, 3, 3) = sum_v B_v WITHOUT the damping — lam*I is added here so the
    caller cannot double-damp. Returns g_raw (logdet, unnormalized — keep it,
    the scene-robust O is only for rendering) plus the eigen diagnostics."""
    if H.dim() != 3 or H.shape[1:] != (3, 3):
        raise ValueError(f"H must be (N,3,3); got {tuple(H.shape)}")
    sym = 0.5 * (H + H.transpose(1, 2)).double()
    lam = float(cfg.lam)
    sym = sym + lam * torch.eye(3, dtype=sym.dtype, device=sym.device)
    ev = torch.linalg.eigvalsh(sym).clamp_min(lam)
    g_raw = torch.log(ev).sum(dim=1)
    return {
        "g_raw": g_raw,
        "O": robust_unit(g_raw, 0.05, 0.95),
        "eig_min": ev[:, 0], "eig_max": ev[:, 2],
        "condition": ev[:, 2] / ev[:, 0].clamp_min(lam),
    }


def robust_unit(x: torch.Tensor, lo_q: float, hi_q: float) -> torch.Tensor:
    """Scene-robust quantile rescale into [0,1] (§7, §11)."""
    xd = x.double()
    lo = torch.quantile(xd, lo_q)
    hi = torch.quantile(xd, hi_q)
    return ((xd - lo) / (hi - lo + EPS)).clamp(0.0, 1.0)


# ---- importance variants ---------------------------------------------------

REQUIRED_VARIANTS = (
    "I0_raw_c", "I1_beta_mean", "I2_mean_support", "I3_mean_agreement",
    "I4_mean_support_agreement", "I5_full_confirmed",
    "I6_rawc_full_confidence", "V0_verification", "U0_uncertainty",
    "S0_support", "A0_agreement", "O0_observability",
)


def required_variants(st: GaussianState) -> dict[str, torch.Tensor]:
    """§9. Every variant lands in [0,1] so it can go straight through the
    responsibility probe render (which clamps colors to [0,1])."""
    if st.c_raw is None:
        raise ValueError("c_raw must be set before building variants")
    m, S, A, O = st.m, st.support, st.agreement, st._O()
    c = st.c_raw.double()
    out = {
        "I0_raw_c": c,
        "I1_beta_mean": m,
        "I2_mean_support": m * S,
        "I3_mean_agreement": m * A,
        "I4_mean_support_agreement": m * S * A,
        "I5_full_confirmed": m * S * A * O,
        "I6_rawc_full_confidence": c * S * A * O,
        "V0_verification": st.verification(),
        "U0_uncertainty": st.uncertainty,
        "S0_support": S,
        "A0_agreement": A,
        "O0_observability": O,
        # not a §9 variant — the state montage (§15 A) renders Conf directly
        "C0_confidence": st.confidence(),
    }
    # kappa0 sensitivity without multiplying every variant by 4 (§5 sweep)
    for k, s_k in st.support_by_kappa0.items():
        out[f"S0_support_k{k:g}"] = s_k
    return {k: v.clamp(0.0, 1.0).float() for k, v in out.items()}


def sweep_grid(gammas=(0.5, 1.0, 2.0), alphas=(0.0, 0.5, 1.0, 2.0),
               betas=(0.0, 0.5, 1.0, 2.0), deltas=(0.0, 0.5, 1.0, 2.0)):
    for g in gammas:
        for al in alphas:
            for be in betas:
                for de in deltas:
                    yield (g, al, be, de)


def sweep_name(g: float, al: float, be: float, de: float) -> str:
    return f"sweep_g{g:g}_a{al:g}_b{be:g}_d{de:g}"


def sweep_variant(st: GaussianState, g: float, al: float, be: float,
                  de: float, cfg: DiagnosticConfig) -> torch.Tensor:
    """§10 general form. The epsilon floor keeps a zero factor from erasing a
    Gaussian outright, so exponent 0 really is 'ignore this axis'."""
    e = cfg.sweep_epsilon

    def fl(x):
        return e + (1.0 - e) * x

    v = st.m.clamp_min(EPS) ** g
    if al:
        v = v * fl(st.support) ** al
    if be:
        v = v * fl(st.agreement) ** be
    if de:
        v = v * fl(st._O()) ** de
    return v.clamp(0.0, 1.0).float()
