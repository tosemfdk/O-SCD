# Exact target-conditioned P-optimal scorer (stage 9, docs/target_gaussian_nbv.md §6).
#
# For each proxy-selected candidate: Delta_H = J^T W J from the FD Jacobian
# backend (the Jacobian already contains alpha*T compositing — responsibility
# is NOT re-multiplied), H_post = H_prior + Delta_H, then
#   d_gain     = 0.5 (logdet H_post - logdet H_prior)          [D-optimality]
#   trace_gain = tr(Sigma_prior) - tr(Sigma_post)              [A-optimality]
#   e_gain     = lmax(Sigma_prior) - lmax(Sigma_post)          [E-optimality]
# combined as  score = w_d d + w_tr tr_norm + w_e e_norm - w_move C_move.
# No candidate RGB is required; H_prior is never mutated here — new views enter
# the state only through commit_observed_view once actually observed.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from target_nbv.config import TargetNBVConfig
from target_nbv.info_builder import TargetInformationBuilder, view_information
from target_nbv.jacobian import compute_target_jacobian
from target_nbv.types import CandidateScore, TargetInformationState
from target_nbv.scorers.numerics import (clamp_gain, logdet_via_slogdet,
                                         max_eig_of_inverse, symmetrize,
                                         trace_of_inverse)

_NORM_EPS = 1e-12


@dataclass
class PriorQuantities:
    logdet: float
    trace_sigma: float
    emax_sigma: float


class TargetPOptimalScorer:
    def __init__(self, cfg: TargetNBVConfig):
        self.cfg = cfg

    # --- pure information-matrix algebra (CPU-testable) ----------------------

    def prior_quantities(self, H_prior: np.ndarray) -> PriorQuantities:
        base, mx = self.cfg.information.absolute_damping, self.cfg.information.max_jitter
        ld, _ = logdet_via_slogdet(H_prior, base, mx)
        tr = trace_of_inverse(H_prior, base, mx)
        if ld is None or tr is None:
            raise ValueError("H_prior is not positive definite even after max jitter")
        return PriorQuantities(logdet=ld, trace_sigma=tr,
                               emax_sigma=max_eig_of_inverse(H_prior))

    def score_delta(self, H_prior: np.ndarray, pq: PriorQuantities,
                    delta_H: np.ndarray, movement_cost: float):
        """Gains for one candidate increment. Returns (gains dict | None,
        invalid_reason | None, warnings). H_prior/delta_H: (D,D) float64."""
        warnings: list[str] = []
        base, mx = self.cfg.information.absolute_damping, self.cfg.information.max_jitter

        delta_H = symmetrize(np.asarray(delta_H, dtype=np.float64))
        if not np.isfinite(delta_H).all():
            return None, "non_finite_delta_H", warnings
        H_post = symmetrize(H_prior + delta_H)

        ld_post, jitter = logdet_via_slogdet(H_post, base, mx)
        if ld_post is None:
            return None, "posterior_logdet_failure", warnings
        if jitter > 0.0:
            warnings.append(f"posterior needed jitter {jitter:.1e}")
        tr_post = trace_of_inverse(H_post, base, mx)
        if tr_post is None:
            return None, "posterior_cholesky_failure", warnings

        d_gain = clamp_gain(0.5 * (ld_post - pq.logdet), "d_gain", warnings)
        trace_gain = clamp_gain(pq.trace_sigma - tr_post, "trace_gain", warnings)
        e_gain = clamp_gain(pq.emax_sigma - max_eig_of_inverse(H_post),
                            "e_gain", warnings)

        trace_gain_norm = trace_gain / (pq.trace_sigma + _NORM_EPS)
        e_gain_norm = e_gain / (pq.emax_sigma + _NORM_EPS)
        s = self.cfg.scoring
        score = (s.d_weight * d_gain
                 + s.trace_weight * trace_gain_norm
                 + s.e_weight * e_gain_norm
                 - s.movement_weight * movement_cost)
        if not np.isfinite(score):
            return None, "non_finite_score", warnings
        return {
            "d_gain": d_gain, "trace_gain": trace_gain, "e_gain": e_gain,
            "trace_gain_norm": trace_gain_norm, "e_gain_norm": e_gain_norm,
            "score": float(score), "jitter": jitter,
        }, None, warnings

    # --- batch API over proxy-selected candidates (GPU) ----------------------

    def score_candidates(self, state: TargetInformationState,
                         candidate_scores: list[CandidateScore],
                         model, target_row: int, pipe, background) -> list[CandidateScore]:
        """Fill exact_* fields of proxy-ranked CandidateScores in place and
        return them re-sorted by exact_score (invalid last). The model is only
        touched through the FD backend, which restores it after every render;
        state.H_prior() is read-only and never mutated."""
        H_prior = state.H_prior()
        pq = self.prior_quantities(H_prior)

        for sc in candidate_scores:
            if not sc.valid:
                continue
            res = compute_target_jacobian(model, sc.candidate.minicam, target_row,
                                          state.spec, pipe, background, self.cfg)
            if not res.valid:
                sc.valid = False
                sc.invalid_reason = f"jacobian:{res.invalid_reason}"
                continue
            delta_H = view_information(res.J, res.w)
            gains, reason, warnings = self.score_delta(
                H_prior, pq, delta_H, sc.movement_cost)
            if gains is None:
                sc.valid = False
                sc.invalid_reason = reason
                continue
            sc.delta_H = delta_H
            sc.d_gain = gains["d_gain"]
            sc.trace_gain = gains["trace_gain"]
            sc.e_gain = gains["e_gain"]
            sc.exact_score = gains["score"]
            sc.candidate.meta["exact_diagnostics"] = {
                **{k: gains[k] for k in ("trace_gain_norm", "e_gain_norm", "jitter")},
                "warnings": warnings, "crop_xyxy": res.crop_xyxy,
            }

        candidate_scores.sort(
            key=lambda s: (not s.valid, -s.exact_score, s.candidate.cand_id))
        return candidate_scores


def commit_observed_view(builder: TargetInformationBuilder, view_id: str,
                         model, cam, target_row: int, pipe, background,
                         cfg: TargetNBVConfig) -> bool:
    """Add an ACTUALLY OBSERVED view to the information state (geometry Fisher
    needs no real RGB — the Jacobian of the current model suffices). Duplicate
    view_ids are ignored; returns whether the state changed."""
    if view_id in builder.state.observed_view_ids:
        return False
    res = compute_target_jacobian(model, cam, target_row, builder.state.spec,
                                  pipe, background, cfg)
    if not res.valid:
        raise ValueError(f"cannot commit view {view_id!r}: {res.invalid_reason}")
    return builder.add_view(view_id, res.J, res.w)
