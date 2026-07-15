# Target-conditioned change Fisher (stage 11 A, docs/target_gaussian_nbv.md §6).
#
# The repo's change representation is a per-Gaussian logit stored in the DC
# features of the change GaussianModel (load_ply_change zeroes them; oscd.py
# optimizes them with sigmoid(rendered mean) in the loss). The target's change
# parameter is therefore scalar c_t = mean(_features_dc[t]); its per-view
# Fisher information under Gaussian pixel noise is
#   delta_h(view) = sum_p (d M_p / d c_t)^2 / sigma^2,  M_p = sigmoid(I_p),
# computed with a central finite difference on c_t. Gain of one candidate:
#   0.5 * log((h_prior + delta_h) / h_prior).      No candidate RGB is needed.

from __future__ import annotations

import math

import torch


class ChangeFisherUnsupported(RuntimeError):
    pass


def _check_change_model(model) -> None:
    if getattr(model, "active_sh_degree", None) != 0:
        raise ChangeFisherUnsupported(
            "change_fisher needs a change model (active_sh_degree == 0 with "
            "logit DC features, as produced by load_ply_change); got "
            f"active_sh_degree={getattr(model, 'active_sh_degree', None)}")


def _render_sigmoid_mask(model, cam, pipe) -> torch.Tensor:
    from gaussian_renderer import render_change  # local import: needs CUDA ext

    bg = torch.zeros(3, device="cuda")
    with torch.no_grad():
        out = render_change(cam, model, pipe, bg)["render"]
    return torch.sigmoid(out.mean(dim=0))


def change_view_information(model, cam, row: int, pipe,
                            eps: float = 1e-2, sigma: float = 1.0) -> float:
    """delta_h of one view for the target's scalar change logit (FD, exact
    restore via try/finally; the model is unchanged on return)."""
    _check_change_model(model)
    dc = model._features_dc
    original = dc.data[row].clone()
    try:
        dc.data[row] = original + eps
        m_plus = _render_sigmoid_mask(model, cam, pipe)
        dc.data[row] = original - eps
        m_minus = _render_sigmoid_mask(model, cam, pipe)
    finally:
        dc.data[row] = original
    dmask = (m_plus - m_minus) / (2.0 * eps)
    if not torch.isfinite(dmask).all():
        raise FloatingPointError("non-finite change-mask derivative")
    return float((dmask.double() ** 2).sum() / (sigma ** 2))


def change_fisher_gain(h_prior: float, delta_h: float) -> float:
    """0.5 * log((h_prior + delta_h) / h_prior). h_prior must include damping."""
    if h_prior <= 0:
        raise ValueError("h_prior must be > 0 (add damping)")
    if delta_h < 0:
        raise ValueError("delta_h must be >= 0")
    return 0.5 * math.log((h_prior + delta_h) / h_prior)
