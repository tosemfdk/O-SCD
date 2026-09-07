"""Signed lifespan-weighted density scores for active mutable Gaussians."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import torch


@dataclass(frozen=True)
class SignedLifespanScore:
    """Per-row diagnostics for score-based active density control."""

    score: torch.Tensor
    sign_support: torch.Tensor
    age: torch.Tensor
    age_weight: torch.Tensor


def _as_flat_tensor(
    name: str,
    value: torch.Tensor | Real,
    reference: torch.Tensor,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device=reference.device, dtype=reference.dtype)
        if tensor.ndim == 0:
            tensor = tensor.expand_as(reference)
        else:
            tensor = tensor.flatten()
    elif isinstance(value, Real) and not isinstance(value, bool):
        tensor = torch.full_like(reference, float(value))
    else:
        raise TypeError(f"{name} must be a tensor or finite real scalar")
    if tensor.shape != reference.shape:
        raise ValueError(f"{name} must have shape {tuple(reference.shape)}")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be finite")
    return tensor


def compute_signed_lifespan_score(
    plus_mass: torch.Tensor,
    minus_mass: torch.Tensor,
    p_plus_is_new: torch.Tensor | Real,
    active: torch.Tensor,
    episode_start: torch.Tensor,
    timestamp: int | float,
) -> SignedLifespanScore:
    """Compute score = active · (2p−1) · (plus−minus) / episode_age.

    ``plus_mass`` and ``minus_mass`` are detached alpha-transmittance masses
    projected onto each Gaussian for the current pre-optimization cue.  The
    ``p_plus_is_new`` posterior aligns the signed cue so that NEW support is
    positive and REMOVE support is negative.  ``episode_start`` must be the
    current lifecycle slot start copied through topology mutation; child
    creation time is deliberately not used.
    """

    if not isinstance(plus_mass, torch.Tensor):
        raise TypeError("plus_mass must be a tensor")
    plus = plus_mass.detach().flatten()
    if plus.ndim != 1:
        raise ValueError("plus_mass must be flattenable to [N]")
    if not torch.is_floating_point(plus):
        plus = plus.to(dtype=torch.float32)
    if not bool(torch.isfinite(plus).all()) or bool((plus < 0.0).any()):
        raise ValueError("plus_mass must be finite and nonnegative")

    minus = _as_flat_tensor("minus_mass", minus_mass, plus)
    if bool((minus < 0.0).any()):
        raise ValueError("minus_mass must be nonnegative")
    posterior = _as_flat_tensor("p_plus_is_new", p_plus_is_new, plus)
    if bool(((posterior < 0.0) | (posterior > 1.0)).any()):
        raise ValueError("p_plus_is_new must be in [0,1]")

    if not isinstance(active, torch.Tensor):
        raise TypeError("active must be a tensor")
    active_flat = active.detach().to(device=plus.device).flatten()
    if active_flat.dtype != torch.bool or active_flat.shape != plus.shape:
        raise ValueError("active must be a boolean tensor with shape [N]")

    start = _as_flat_tensor("episode_start", episode_start, plus)
    if isinstance(timestamp, bool) or not isinstance(timestamp, Real):
        raise TypeError("timestamp must be a finite real value")
    timestamp_tensor = torch.as_tensor(float(timestamp), device=plus.device, dtype=plus.dtype)
    if not bool(torch.isfinite(timestamp_tensor)):
        raise ValueError("timestamp must be finite")

    age = timestamp_tensor - start + 1.0
    if not bool(torch.isfinite(age).all()):
        raise ValueError("episode age must be finite")
    if bool((age[active_flat] <= 0.0).any()):
        raise ValueError("active episode age must be positive")

    active_float = active_flat.to(dtype=plus.dtype)
    safe_age = age.clamp_min(1.0)
    age_weight = active_float / safe_age
    sign_support = (2.0 * posterior - 1.0) * (plus - minus)
    score = sign_support * age_weight
    return SignedLifespanScore(
        score=score,
        sign_support=sign_support * active_float,
        age=age,
        age_weight=age_weight,
    )
