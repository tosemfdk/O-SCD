# Shared numerical policy for information-matrix scoring
# (docs/target_gaussian_nbv.md §10): symmetrize, slogdet/Cholesky with jitter
# escalation on a LOCAL COPY, no explicit inverses, negative-gain clamping.

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve

GAIN_TOL = 1e-9


def symmetrize(H: np.ndarray) -> np.ndarray:
    return 0.5 * (H + H.T)


def robust_cholesky(H: np.ndarray, base_jitter: float, max_jitter: float):
    """cho_factor with escalating jitter on a local copy.
    Returns (factor, jitter_used) or (None, None) on failure."""
    jitter = 0.0
    while True:
        try:
            c = cho_factor(H + jitter * np.eye(H.shape[0]), lower=True)
            return c, jitter
        except np.linalg.LinAlgError:
            jitter = base_jitter if jitter == 0.0 else jitter * 10.0
            if jitter > max_jitter:
                return None, None


def logdet_via_slogdet(H: np.ndarray, base_jitter: float, max_jitter: float):
    """(logdet, jitter_used) with jitter escalation; (None, None) on failure."""
    jitter = 0.0
    while True:
        sign, logdet = np.linalg.slogdet(H + jitter * np.eye(H.shape[0]))
        if sign > 0:
            return float(logdet), jitter
        jitter = base_jitter if jitter == 0.0 else jitter * 10.0
        if jitter > max_jitter:
            return None, None


def trace_of_inverse(H: np.ndarray, base_jitter: float, max_jitter: float):
    c, _ = robust_cholesky(H, base_jitter, max_jitter)
    if c is None:
        return None
    return float(np.trace(cho_solve(c, np.eye(H.shape[0]))))


def max_eig_of_inverse(H: np.ndarray) -> float:
    # 6x6: eigvalsh is exact and cheap; lambda_max(H^-1) = 1/lambda_min(H)
    return float(1.0 / max(np.linalg.eigvalsh(H)[0], 1e-300))


def clamp_gain(gain: float, label: str, warnings: list[str]) -> float:
    """Small negative gains are numerical noise; large ones are bugs."""
    if gain < -GAIN_TOL:
        warnings.append(f"large negative {label}: {gain:.3e}")
        return gain
    return max(gain, 0.0)
