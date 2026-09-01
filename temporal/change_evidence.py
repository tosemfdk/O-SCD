"""Lifespan-agnostic alpha-transmittance evidence for Bayesian SCD.

This module deliberately probes the immutable reference Gaussian field.  It does
not use temporal lifespan opacity, state validity, or state-local geometry
changes, so closed or never-opened Gaussians remain observable and can reopen.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import SimpleNamespace
from typing import Literal

import torch

from gaussian_renderer import render_change

CueMode = Literal["binary", "soft"]
EvidenceCountMode = Literal["raw", "capped"]
ProbeScalingMode = Literal["native", "isotropic_min"]


@dataclass(frozen=True)
class EvidenceResult:
    """Per-Gaussian pseudo-counts from one view."""

    e_plus: torch.Tensor
    e_minus: torch.Tensor
    total_mass: torch.Tensor
    delta_a: torch.Tensor
    delta_b: torch.Tensor
    observed: torch.Tensor
    cue: torch.Tensor
    mode: CueMode
    count_mode: EvidenceCountMode
    soft_is_fractional_extension: bool


def evidence_probe_scaling(
    scaling: torch.Tensor,
    *,
    mode: ProbeScalingMode = "native",
) -> torch.Tensor:
    """Return detached detector-probe scales without expanding support.

    ``isotropic_min`` replaces every 3D axis with the row's shortest native
    axis.  It therefore removes elongated evidence footprints without growing
    a Gaussian along either of its shorter axes.  This affects only the
    alpha-transmittance evidence probe, not the learned/rendered representation.
    """

    if not isinstance(scaling, torch.Tensor):
        raise TypeError("scaling must be a tensor")
    if scaling.ndim != 2 or scaling.shape[1] != 3:
        raise ValueError("scaling must have shape [N,3]")
    if not torch.is_floating_point(scaling):
        raise TypeError("scaling must be floating point")
    if not bool(torch.isfinite(scaling).all()) or bool((scaling <= 0).any()):
        raise ValueError("scaling must be finite and strictly positive")
    if mode == "native":
        return scaling.detach()
    if mode != "isotropic_min":
        raise ValueError("probe scaling mode must be native|isotropic_min")
    shortest = scaling.detach().amin(dim=1, keepdim=True)
    return shortest.expand_as(scaling).contiguous()


def cue_to_change_probability(
    candidate_map: torch.Tensor,
    *,
    mode: CueMode,
    threshold: float = 0.5,
    scale: float = 1.0,
) -> torch.Tensor:
    """Convert a candidate map to ``C_t(p) in [0, 1]``.

    ``binary`` is the source-faithful Beta-Bernoulli/B3-Seg observation mode.
    ``soft`` is a fractional/power-likelihood extension and is not the exact
    original Bernoulli observation model.
    """
    if not isinstance(candidate_map, torch.Tensor):
        raise TypeError("candidate_map must be a tensor")
    if candidate_map.numel() == 0 or not bool(torch.isfinite(candidate_map).all()):
        raise ValueError("candidate_map must be nonempty and finite")
    if mode not in ("binary", "soft"):
        raise ValueError("mode must be 'binary' or 'soft'")
    if not math.isfinite(float(threshold)):
        raise ValueError("threshold must be finite")
    if mode == "binary":
        return (candidate_map > float(threshold)).to(dtype=candidate_map.dtype)
    if not math.isfinite(float(scale)) or scale <= 0:
        raise ValueError("scale must be finite and positive for soft cue mode")
    return torch.clamp(candidate_map / float(scale), 0.0, 1.0)


def evidence_counts(
    e_plus: torch.Tensor,
    e_minus: torch.Tensor,
    *,
    mode: EvidenceCountMode = "raw",
    mass_saturation: float = 1.0,
    min_evidence_mass: float = 0.0,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map raw alpha-T masses to Beta pseudo-count increments."""
    if not isinstance(e_plus, torch.Tensor) or not isinstance(e_minus, torch.Tensor):
        raise TypeError("e_plus and e_minus must be tensors")
    if e_plus.shape != e_minus.shape:
        raise ValueError("e_plus and e_minus must have the same shape")
    if e_plus.device != e_minus.device or e_plus.dtype != e_minus.dtype:
        raise ValueError("e_plus and e_minus must share device and dtype")
    if not (torch.is_floating_point(e_plus) and torch.is_floating_point(e_minus)):
        raise TypeError("evidence masses must be floating point")
    if not bool(torch.isfinite(e_plus).all() and torch.isfinite(e_minus).all()):
        raise ValueError("evidence masses must be finite")
    if bool((e_plus < 0).any() or (e_minus < 0).any()):
        raise ValueError("evidence masses must be nonnegative")
    if mode not in ("raw", "capped"):
        raise ValueError("evidence count mode must be 'raw' or 'capped'")
    if not math.isfinite(float(mass_saturation)) or mass_saturation <= 0:
        raise ValueError("mass_saturation must be finite and positive")
    if not math.isfinite(float(min_evidence_mass)) or min_evidence_mass < 0:
        raise ValueError("min_evidence_mass must be finite and nonnegative")
    if not math.isfinite(float(eps)) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    total_mass = e_plus + e_minus
    observed = total_mass >= float(min_evidence_mass)
    if mode == "raw":
        delta_a = e_plus.clone()
        delta_b = e_minus.clone()
    else:
        q = e_plus / (total_mass + float(eps))
        w = torch.clamp(total_mass / float(mass_saturation), 0.0, 1.0)
        delta_a = w * q
        delta_b = w * (1.0 - q)
    delta_a = torch.where(observed, delta_a, torch.zeros_like(delta_a))
    delta_b = torch.where(observed, delta_b, torch.zeros_like(delta_b))
    return delta_a, delta_b, total_mass, observed


def alpha_t_evidence_vjp(
    view,
    base,
    pipe: SimpleNamespace,
    background: torch.Tensor,
    cue: torch.Tensor,
    *,
    count_mode: EvidenceCountMode = "raw",
    mass_saturation: float = 1.0,
    min_evidence_mass: float = 0.0,
    eps: float = 1e-8,
    probe_scaling_mode: ProbeScalingMode = "native",
) -> EvidenceResult:
    """Accumulate per-Gaussian ``alpha_i T_i`` evidence with one VJP.

    The render uses detached immutable base geometry/opacity and a differentiable
    probe color.  Gradients are requested only for the probe color, so base and
    temporal model tensors are not left with gradients by this operation.
    """
    if cue.ndim == 2:
        cue_weights = cue
    elif cue.ndim == 3 and cue.shape[0] == 1:
        cue_weights = cue[0]
    else:
        raise ValueError("cue must have shape [H,W] or [1,H,W]")
    n = int(base.get_xyz.shape[0])
    device = base.get_xyz.device
    dtype = base.get_xyz.dtype
    cue_weights = cue_weights.to(device=device, dtype=dtype)
    if not torch.isfinite(cue_weights).all() or float(cue_weights.min().item()) < 0.0 or float(cue_weights.max().item()) > 1.0:
        raise ValueError("cue values must be finite probabilities in [0, 1]")
    if (
        background.shape != (3,)
        or background.device != device
        or background.dtype != dtype
        or not bool(torch.isfinite(background).all())
    ):
        raise ValueError("background must be a [3] tensor on the base device/dtype")
    black_background = torch.zeros_like(background)
    base_grad_before = {
        name: getattr(getattr(base, name, None), "grad", None)
        for name in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation")
    }
    probe_color = torch.zeros((n, 3), device=device, dtype=dtype, requires_grad=True)
    probe_scaling = evidence_probe_scaling(
        base.get_scaling.detach(), mode=probe_scaling_mode
    )
    rendered = render_change(
        view,
        base,
        pipe,
        black_background,
        override_color=probe_color,
        override_opacity=base.get_opacity.detach(),
        override_xyz=base.get_xyz.detach(),
        override_scaling=probe_scaling,
        override_rotation=base.get_rotation.detach(),
        clamp_output=False,
    )["render"]
    if rendered.shape[1:] != cue_weights.shape:
        raise ValueError(
            f"cue spatial shape {tuple(cue_weights.shape)} does not match render {tuple(rendered.shape[1:])}"
        )
    weights = torch.zeros_like(rendered)
    weights[0] = cue_weights
    weights[1] = 1.0 - cue_weights
    grad = torch.autograd.grad(
        outputs=rendered,
        inputs=probe_color,
        grad_outputs=weights,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )[0].detach()
    for name, previous_grad in base_grad_before.items():
        value = getattr(base, name, None)
        if isinstance(value, torch.Tensor) and getattr(value, "grad", None) is not previous_grad:
            raise RuntimeError("alpha-T evidence unexpectedly populated a base tensor gradient")
    if grad.shape != (n, 3) or grad.device != device or grad.dtype != dtype or not torch.isfinite(grad).all():
        raise RuntimeError("override-color VJP returned an invalid gradient")
    tolerance = 1e-6
    if bool((grad[:, :2] < -tolerance).any()):
        raise RuntimeError("override-color VJP returned negative alpha-T responsibility")
    if bool((grad[:, 2].abs() > tolerance).any()):
        raise RuntimeError("override-color VJP violated the zero-weight channel convention")
    # Clamp only sub-tolerance floating noise after rejecting invalid gradients.
    e_plus = grad[:, 0].clamp_min(0.0)
    e_minus = grad[:, 1].clamp_min(0.0)
    if not (torch.isfinite(e_plus).all() and torch.isfinite(e_minus).all()):
        raise RuntimeError("alpha-T evidence contains non-finite values")
    delta_a, delta_b, total_mass, observed = evidence_counts(
        e_plus,
        e_minus,
        mode=count_mode,
        mass_saturation=mass_saturation,
        min_evidence_mass=min_evidence_mass,
        eps=eps,
    )
    return EvidenceResult(
        e_plus=e_plus,
        e_minus=e_minus,
        total_mass=total_mass,
        delta_a=delta_a,
        delta_b=delta_b,
        observed=observed,
        cue=cue_weights.detach(),
        mode="soft",
        count_mode=count_mode,
        soft_is_fractional_extension=True,
    )


def accumulate_change_evidence(
    view,
    base,
    pipe: SimpleNamespace,
    background: torch.Tensor,
    candidate_map: torch.Tensor,
    *,
    cue_mode: CueMode = "binary",
    cue_threshold: float = 0.5,
    cue_scale: float = 1.0,
    count_mode: EvidenceCountMode = "raw",
    mass_saturation: float = 1.0,
    min_evidence_mass: float = 0.0,
    probe_scaling_mode: ProbeScalingMode = "native",
) -> EvidenceResult:
    """Full cue conversion + lifespan-agnostic alpha-T VJP evidence."""
    cue = cue_to_change_probability(
        candidate_map,
        mode=cue_mode,
        threshold=cue_threshold,
        scale=cue_scale,
    )
    result = alpha_t_evidence_vjp(
        view,
        base,
        pipe,
        background,
        cue,
        count_mode=count_mode,
        mass_saturation=mass_saturation,
        min_evidence_mass=min_evidence_mass,
        probe_scaling_mode=probe_scaling_mode,
    )
    return EvidenceResult(
        e_plus=result.e_plus,
        e_minus=result.e_minus,
        total_mass=result.total_mass,
        delta_a=result.delta_a,
        delta_b=result.delta_b,
        observed=result.observed,
        cue=result.cue,
        mode=cue_mode,
        count_mode=count_mode,
        soft_is_fractional_extension=(cue_mode == "soft"),
    )

# Backwards-compatible public names used by temporal.__init__ and experiments.
ChangeEvidence = EvidenceResult
cue_to_probability = cue_to_change_probability
counts_from_evidence = evidence_counts
accumulate_alpha_t_evidence = alpha_t_evidence_vjp
