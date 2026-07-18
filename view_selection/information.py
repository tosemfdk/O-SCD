# Change-channel information: exact toy reference + guarded Hutchinson VJP
# (spec §6). Measurement: y_v(p) = raw change logit rendered by render_change
# (per-Gaussian color 0.5 + SH_C0 * c; at the reference scoring state c = 0 so
# neither the per-Gaussian nor the image clamp is active and the adjoint is
# exactly SH_C0 * r_vpi). diag(J^T W J) is estimated per candidate.
from __future__ import annotations

import hashlib

import torch

from view_selection.types import InformationConfig, ViewInformation, ZeroAdjointError


def _render_raw(model, camera, pipe, background):
    """One render_change forward with a fresh graph. Returns (y, radii):
    y = per-pixel raw change logit (H, W), channel-mean like the loss path."""
    from gaussian_renderer import render_change

    pkg = render_change(camera, model, pipe, background)
    return pkg["render"].mean(dim=0), pkg["radii"]


def select_output(y_raw: torch.Tensor, output_space: str) -> torch.Tensor:
    if output_space == "raw":
        return y_raw
    if output_space == "sigmoid":
        return torch.sigmoid(y_raw)
    raise ValueError(output_space)


def project_dc_gradient(grad_dc: torch.Tensor) -> torch.Tensor:
    """Tied-channel scalarization (spec §5): the 3 dc channels carry identical
    gradients, the tied scalar's derivative is their sum. (N,1,3)|(N,3,1)->(N,)."""
    return grad_dc.reshape(grad_dc.shape[0], -1).sum(dim=1)


def deterministic_rademacher(shape, seed_key: tuple, device) -> torch.Tensor:
    """Rademacher +-1 noise, reproducible from (scene, frame, probe, cfg) key."""
    seed = int.from_bytes(
        hashlib.sha256(repr(seed_key).encode()).digest()[:8], "little") % (2**63)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    bits = torch.randint(0, 2, shape, generator=gen, dtype=torch.int8)
    return (bits.to(device=device, dtype=torch.float32) * 2.0 - 1.0)


def _guard_inputs(radii: torch.Tensor, weight: torch.Tensor):
    """fastgs guards (spec §6.3): never run backward on an empty render, never
    score with a degenerate weight map. Returns True when scoring can proceed."""
    if not (radii > 0).any():
        return False
    valid = torch.isfinite(weight) & (weight > 0)
    return bool(valid.any())


def hutchinson_information(model, camera, pixel_weight: torch.Tensor,
                           config: InformationConfig, seed_scope: tuple,
                           pipe, background, frame_id: int = -1,
                           strict: bool = True) -> ViewInformation:
    """diag(J^T W J) over the change channel c via Hutchinson probes.
    One fresh render graph per probe (no graph reuse — fastgs stability).
    pixel_weight: (H, W) >= 0, detached."""
    n = model.get_xyz.shape[0]
    device = model.get_xyz.device
    w = pixel_weight.detach().to(device=device, dtype=torch.float32)
    info = torch.zeros(n, dtype=torch.float32, device=device)

    visible = 0
    ran_probes = 0
    for probe_idx in range(config.num_probes):
        y_raw, radii = _render_raw(model, camera, pipe, background)
        if probe_idx == 0:
            visible = int((radii > 0).sum())
        if not _guard_inputs(radii, w):
            break  # empty render or degenerate weight: zero info, no backward
        y = select_output(y_raw, config.output_space)
        if y.shape != w.shape:
            raise ValueError(f"weight {tuple(w.shape)} != render {tuple(y.shape)}")
        xi = deterministic_rademacher(
            y.shape, (*seed_scope, frame_id, probe_idx), device)
        scalar = (y * w.sqrt() * xi).sum()
        grad_dc = torch.autograd.grad(scalar, model._features_dc,
                                      create_graph=False, retain_graph=False,
                                      allow_unused=False)[0]
        grad_c = project_dc_gradient(grad_dc)
        if not torch.isfinite(grad_c).all():
            raise ZeroAdjointError(f"frame {frame_id}: non-finite adjoint")
        info.add_(grad_c.square())
        ran_probes += 1

    if ran_probes:
        info.div_(ran_probes)
        if strict and visible > 0 and not info.any():
            raise ZeroAdjointError(
                f"frame {frame_id}: {visible} visible Gaussians and valid "
                f"pixels but an all-zero dc adjoint (silent fastgs trap)")

    valid = torch.isfinite(w) & (w > 0)
    return ViewInformation(
        frame_id=frame_id,
        diagonal=info.detach().float().cpu(),
        valid_pixels=int(valid.sum()),
        visible_gaussians=visible,
        alpha_coverage=float(valid.float().mean()),
        metadata={"probes_ran": ran_probes,
                  "output_space": config.output_space,
                  "weight_mode": config.weight_mode},
    )


def exact_information(model, camera, pixel_weight: torch.Tensor,
                      output_space: str, pipe, background) -> torch.Tensor:
    """Brute-force b_i = sum_p w_p (dy_p/dc_i)^2 by per-pixel backward on a
    fresh graph each time. Toy fixtures only (correctness oracle, spec §6.4)."""
    device = model.get_xyz.device
    w = pixel_weight.detach().to(device=device, dtype=torch.float32)
    n = model.get_xyz.shape[0]
    b = torch.zeros(n, dtype=torch.float64, device=device)
    nz = (w > 0).nonzero(as_tuple=False)
    for r, c in nz.tolist():
        y_raw, radii = _render_raw(model, camera, pipe, background)
        if not (radii > 0).any():
            return b.float()
        y = select_output(y_raw, output_space)
        g = torch.autograd.grad(y[r, c], model._features_dc)[0]
        b += float(w[r, c]) * project_dc_gradient(g).double().square()
    return b.float()


def derive_relative_lambda(diagonals, lambda_rel: float,
                           lambda_abs: float) -> float:
    """lambda = max(lambda_abs, lambda_rel * median of positive b) (spec §9)."""
    pos = [d[d > 0] for d in diagonals]
    pos = [p for p in pos if p.numel()]
    if not pos:
        return lambda_abs
    med = float(torch.cat(pos).median())
    return max(lambda_abs, lambda_rel * med)
