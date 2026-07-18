# Pixel-weight providers (spec §7). Phase A implements `pose` only; the
# content-using modes land in later commits (spec §29: 14-15) and raise
# loudly until then so a mode typo can never silently fall back.
from __future__ import annotations

import torch


def alpha_map(model_ref, camera, pipe, background) -> torch.Tensor:
    """Accumulated reference alpha per pixel in [0, 1] = sum_g r_g,p, via the
    verified counts.py probe render with unit weights. (The fastgs fork's
    override_color path is broken — dc stays unbound — so it cannot be used.)"""
    from target_nbv.change.counts import responsibility_probe_render

    n = model_ref.get_xyz.shape[0]
    ones = torch.ones(n, device=model_ref.get_xyz.device)
    return responsibility_probe_render(model_ref, camera, ones, pipe).clamp(0.0, 1.0)


def pose_weight(model_ref, camera, pipe, background,
                alpha_threshold: float) -> torch.Tensor:
    """V_v(p) = 1[A_ref,v(p) >= tau]. Uses reference alpha only — no candidate
    image content (claim: active NBV)."""
    a = alpha_map(model_ref, camera, pipe, background)
    return (a >= alpha_threshold).float()


def current_map_weight(model_ref, model_change, camera, pipe, background,
                       alpha_threshold: float, epsilon: float) -> torch.Tensor:
    """V(p) * [eps + (1-eps) * m(p)] with m = 1[z(p) > 0.5], the pipeline's
    own current-change-mask definition (subset_oscd query threshold): a
    re-observation bonus on pixels the CURRENT change scene already flags.
    (sigmoid(z) was tried first and is nearly flat — ~0.59 vs ~0.7 — so it
    left candidate rankings unchanged; the binary mask has 1/eps contrast.)
    Uses only the model state, never candidate content."""
    from gaussian_renderer import render_change

    v = pose_weight(model_ref, camera, pipe, background, alpha_threshold)
    with torch.no_grad():
        z = render_change(camera, model_change, pipe, background)["render"].mean(dim=0)
        m = (z > 0.5).float()
    return (v * (epsilon + (1.0 - epsilon) * m)).detach()


def build_pixel_weight(mode: str, model_ref, camera, pipe, background,
                       config, model_change=None) -> torch.Tensor:
    if mode == "pose":
        return pose_weight(model_ref, camera, pipe, background,
                           config.alpha_threshold).detach()
    if mode == "current_map":
        if model_change is None:
            raise ValueError("current_map weight needs the change model")
        return current_map_weight(model_ref, model_change, camera, pipe,
                                  background, config.alpha_threshold,
                                  config.epsilon)
    if mode in ("cue", "consensus"):
        raise NotImplementedError(
            f"weight mode {mode!r} is a later-phase feature; refusing to "
            f"silently fall back to pose")
    raise ValueError(f"unknown weight mode {mode!r}")
