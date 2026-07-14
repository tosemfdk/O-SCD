# Geometry FIM proxy scorer (stage 8, docs/target_gaussian_nbv.md §6).
#
# Pre-ranks all candidates with the viewing-geometry position FIM
#   Delta_H_mu(c) = rho_norm / max(d^2, eps) * (I - r r^T),   r = (mu - o)/d,
# against the 3x3 mean block of H_prior. This is the GS-DIFF-style proxy: a
# view constrains the target mean perpendicular to its viewing ray, weakly
# along it, and with 1/d^2 falloff. Only the proxy top-k go to the exact
# (Jacobian) scorer. Responsibility rho is a gating/weight here only — the
# exact scorer never re-multiplies it (docs §8).

from __future__ import annotations

import numpy as np

from target_nbv.config import TargetNBVConfig
from target_nbv.types import (CandidateCamera, CandidateScore,
                              TargetInformationState, TargetVisibility)
from target_nbv.scorers.numerics import symmetrize

_D2_EPS = 1e-12


def normalize_responsibilities(rhos: np.ndarray, method: str) -> np.ndarray:
    """Robust per-candidate-set normalization of responsibility_sum values.
    percentile90: rho_norm = clamp(rho / percentile_90(valid rho), 0, 1)."""
    if method != "percentile90":
        raise ValueError(f"unknown responsibility normalization {method!r}")
    if rhos.size == 0:
        return rhos
    scale = float(np.percentile(rhos, 90.0))
    if scale <= 0.0:
        return np.zeros_like(rhos)
    return np.clip(rhos / scale, 0.0, 1.0)


class GeometryProxyScorer:
    def __init__(self, cfg: TargetNBVConfig):
        self.cfg = cfg

    def rank(self, state: TargetInformationState,
             candidates: list[CandidateCamera],
             visibilities: list[TargetVisibility | None],
             target_mean: np.ndarray) -> list[CandidateScore]:
        """Score every candidate with the proxy FIM and return CandidateScores
        sorted by descending proxy_score (invalid candidates last, by cand_id).
        `visibilities[i]` may be None only when cfg.proxy.uses_visibility is
        False (render-free mode: rho = 1 for every geometrically valid pose)."""
        if len(candidates) != len(visibilities):
            raise ValueError("candidates and visibilities must be parallel lists")

        mean_slice = state.spec.slices().get("mean")
        if mean_slice is None:
            raise ValueError("proxy scorer requires the 'mean' parameter group in the spec")
        H_prior = state.H_prior()
        H_mu = symmetrize(np.array(H_prior[mean_slice, mean_slice], dtype=np.float64))
        sign0, ld_prior = np.linalg.slogdet(H_mu)
        if sign0 <= 0:
            raise ValueError("H_mu_prior is not positive definite; damping missing?")
        # dominant covariance eigenvector = eigenvector of the SMALLEST
        # eigenvalue of H_mu (most uncertain direction), for diagnostics
        eigvals, eigvecs = np.linalg.eigh(H_mu)
        e_max = eigvecs[:, 0]

        mu = np.asarray(target_mean, dtype=np.float64).reshape(3)

        scores: list[CandidateScore] = []
        rho_raw = np.zeros(len(candidates), dtype=np.float64)
        for i, (cand, vis) in enumerate(zip(candidates, visibilities)):
            if vis is None:
                if self.cfg.proxy.uses_visibility:
                    raise ValueError("visibility required when proxy.uses_visibility=true")
                rho_raw[i] = 1.0
            elif vis.valid:
                rho_raw[i] = vis.responsibility_sum
            scores.append(CandidateScore(candidate=cand, visibility=vis,
                                         movement_cost=cand.movement_cost))

        valid_mask = np.array(
            [v is None or v.valid for v in visibilities], dtype=bool)
        rho_norm = np.zeros_like(rho_raw)
        if valid_mask.any():
            rho_norm[valid_mask] = normalize_responsibilities(
                rho_raw[valid_mask], self.cfg.proxy.responsibility_normalization)

        eye3 = np.eye(3, dtype=np.float64)
        for i, (cand, vis, sc) in enumerate(zip(candidates, visibilities, scores)):
            if vis is not None and not vis.valid:
                sc.valid = False
                sc.invalid_reason = vis.invalid_reason
                continue
            diff = mu - np.asarray(cand.position, dtype=np.float64)
            d = float(np.linalg.norm(diff))
            if d < 1e-9:
                sc.valid = False
                sc.invalid_reason = "camera_at_target"
                continue
            r = diff / d
            delta = rho_norm[i] / max(d * d, _D2_EPS) * (eye3 - np.outer(r, r))
            sign, ld_post = np.linalg.slogdet(H_mu + delta)
            if sign <= 0:  # cannot happen for PSD increment, but be explicit
                sc.valid = False
                sc.invalid_reason = "proxy_logdet_failure"
                continue
            d_gain = 0.5 * (ld_post - ld_prior)
            sc.proxy_score = (d_gain
                              - self.cfg.scoring.movement_weight * cand.movement_cost)
            sc.d_gain = d_gain
            sc.candidate.meta["proxy_directional_gain"] = float(
                rho_norm[i] / max(d * d, _D2_EPS) * (1.0 - float(np.dot(e_max, r)) ** 2))
            sc.candidate.meta["proxy_rho_norm"] = float(rho_norm[i])

        # deterministic order: score desc, then cand_id; invalid always last
        scores.sort(key=lambda s: (not s.valid, -s.proxy_score, s.candidate.cand_id))
        return scores


def proxy_top_k(scores: list[CandidateScore], k: int) -> list[CandidateScore]:
    """First k VALID proxy-ranked scores (fewer if fewer are valid)."""
    return [s for s in scores if s.valid][:k]
