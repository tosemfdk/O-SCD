# Per-Gaussian soft counts and weighted responsibility probes (stage 11).
#
# Key identity: with a degree-0 probe whose per-Gaussian color is c_g, the
# rendered pixel is I(p) = sum_g r_g,p * c_g (+ nothing, bg=0), where r_g,p =
# alpha*T is the compositing responsibility. Therefore
#   d/d c_g [ sum_p I(p) * M_p ] = sum_p r_g,p * M_p = e1_g
# for ALL Gaussians in ONE render + backward — the rasterizer's dc-gradient is
# the exact adjoint (it is the same code path the change model trains through).
# The weighted probe (c_g = w_g) similarly gives sum_g w_g * tau_g(cam) as the
# pixel sum in one forward pass — used by the NBV frame scorer.

from __future__ import annotations

from types import SimpleNamespace

import torch

SH_C0 = 0.28209479177387814  # utils/sh_utils.py


class _ProbeModel(SimpleNamespace):
    """Duck-typed stand-in accepted by gaussian_renderer.render."""


def _make_probe(model, dc: torch.Tensor) -> _ProbeModel:
    # max_sh_degree=1 with a zero dummy rest tensor: the fastgs CUDA backward
    # silently writes ZERO dc-gradients when the rest tensor is empty
    # (max_sh_degree=0); active degree stays 0 so the rest is never evaluated.
    n = model.get_xyz.shape[0]
    return _ProbeModel(
        get_xyz=model.get_xyz.detach(),
        get_opacity=model.get_opacity.detach(),
        get_scaling=model.get_scaling.detach(),
        get_rotation=model.get_rotation.detach(),
        _features_dc=dc,
        _features_rest=torch.zeros((n, 3, 3), device=dc.device),
        active_sh_degree=0,
        max_sh_degree=1,
    )


def responsibility_probe_render(model, cam, weights: torch.Tensor, pipe) -> torch.Tensor:
    """Render sum_g weights_g * r_g,p per pixel, shape (H, W). weights in [0,1]
    (values outside are clamped by the rasterizer's color clamp)."""
    from gaussian_renderer import render  # local import: needs CUDA ext

    w = weights.detach().float().reshape(-1, 1, 1)
    dc = ((w - 0.5) / SH_C0).expand(-1, 1, 3).contiguous()
    bg = torch.zeros(3, device=dc.device)
    with torch.no_grad():
        out = render(cam, _make_probe(model, dc), pipe, bg)["render"]
    return out.mean(dim=0)


def responsibilities(model, cam, pipe) -> torch.Tensor:
    """Per-Gaussian total responsibility tau_g = sum_p r_g,p at this camera,
    for ALL Gaussians in one render + one backward. float64 (N,)."""
    from gaussian_renderer import render

    n = model.get_xyz.shape[0]
    device = model.get_xyz.device
    dc = torch.zeros((n, 1, 3), device=device, requires_grad=True)
    bg = torch.zeros(3, device=device)
    pkg = render(cam, _make_probe(model, dc), pipe, bg)
    if not (pkg["radii"] > 0).any():
        # nothing rendered: fastgs backward with num_rendered==0 launches an
        # invalid (size-0) kernel and poisons the CUDA context — skip it
        return torch.zeros(n, dtype=torch.float64, device=device)
    g = torch.autograd.grad(pkg["render"].mean(dim=0).sum(), dc)[0]
    return (g.sum(dim=(1, 2)) / SH_C0).double().clamp_min(0.0)


def soft_counts(model, cam, soft_mask: torch.Tensor, pipe):
    """Exact per-Gaussian soft counts against a change mask, via the adjoint.

    soft_mask: (H, W) float in [0,1] (soft change evidence at this view).
    Returns (e1, e0, tau): float64 tensors, shape (N,), where
      e1_g = sum_p r_g,p * M_p,  tau_g = sum_p r_g,p,  e0 = tau - e1.
    The model is not mutated (probe uses detached copies)."""
    from gaussian_renderer import render

    n = model.get_xyz.shape[0]
    device = model.get_xyz.device
    dc = torch.zeros((n, 1, 3), device=device, requires_grad=True)  # color 0.5
    bg = torch.zeros(3, device=device)
    pkg = render(cam, _make_probe(model, dc), pipe, bg)
    out = pkg["render"].mean(dim=0)
    if not (pkg["radii"] > 0).any():  # see responsibilities(): size-0 backward
        z = torch.zeros(n, dtype=torch.float64, device=device)
        return z, z.clone(), z.clone()

    M = soft_mask.detach().to(device=device, dtype=out.dtype).clamp(0.0, 1.0)
    if M.shape != out.shape:
        raise ValueError(f"mask shape {tuple(M.shape)} != render {tuple(out.shape)}")

    g1 = torch.autograd.grad((out * M).sum(), dc, retain_graph=True)[0]
    g_tau = torch.autograd.grad(out.sum(), dc)[0]
    e1 = (g1.sum(dim=(1, 2)) / SH_C0).double().clamp_min(0.0)
    tau = (g_tau.sum(dim=(1, 2)) / SH_C0).double().clamp_min(0.0)
    e0 = (tau - e1).clamp_min(0.0)
    return e1, e0, tau
