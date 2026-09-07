"""Small, side-effect-free helpers for O-SCD change-cue fusion."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from utils.loss_utils import ssim


@dataclass(frozen=True)
class PowerProductCue:
    """Components of ``2 * P**alpha * S`` reconstructed from cached ``P + S``."""

    pixel: torch.Tensor
    semantic: torch.Tensor
    fused: torch.Tensor


@dataclass(frozen=True)
class OscdPixelTerms:
    """Unnormalized photometric and structural terms of the O-SCD pixel cue."""

    l1: torch.Tensor
    structural: torch.Tensor


def oscd_pixel_terms(
    reference_rgb: torch.Tensor,
    online_rgb: torch.Tensor,
) -> OscdPixelTerms:
    """Return O-SCD's raw ``L1`` and ``1-SSIM`` image-space terms.

    Both inputs use ``[3,H,W]`` RGB values in ``[0,1]``.  The returned tensor
    terms have shape ``[H,W]``.
    """

    if not isinstance(reference_rgb, torch.Tensor) or not isinstance(
        online_rgb, torch.Tensor
    ):
        raise TypeError("reference_rgb and online_rgb must be tensors")
    if reference_rgb.shape != online_rgb.shape:
        raise ValueError("reference_rgb and online_rgb must share shape")
    if reference_rgb.ndim != 3 or reference_rgb.shape[0] != 3:
        raise ValueError("RGB inputs must have shape [3,H,W]")
    if reference_rgb.device != online_rgb.device or reference_rgb.dtype != online_rgb.dtype:
        raise ValueError("RGB inputs must share device and dtype")
    if not torch.is_floating_point(reference_rgb):
        raise TypeError("RGB inputs must be floating point")
    if not bool(torch.isfinite(reference_rgb).all() and torch.isfinite(online_rgb).all()):
        raise ValueError("RGB inputs must be finite")

    with torch.no_grad():
        ssim_map = ssim(
            reference_rgb,
            online_rgb,
            window_size=11,
            map=True,
        ).mean(dim=0)
        l1_image = torch.abs(reference_rgb - online_rgb).mean(dim=0)
    return OscdPixelTerms(l1=l1_image, structural=1.0 - ssim_map)


def normalized_oscd_pixel_cue_from_terms(
    terms: OscdPixelTerms,
    *,
    l1_exponent: float = 1.0,
) -> torch.Tensor:
    """Mix pixel terms and apply O-SCD's per-frame min-max normalization."""

    if terms.l1.shape != terms.structural.shape or terms.l1.ndim != 2:
        raise ValueError("pixel terms must share shape [H,W]")
    if terms.l1.device != terms.structural.device or terms.l1.dtype != terms.structural.dtype:
        raise ValueError("pixel terms must share device and dtype")
    if not torch.is_floating_point(terms.l1):
        raise TypeError("pixel terms must be floating point")
    if not math.isfinite(float(l1_exponent)) or l1_exponent <= 0.0:
        raise ValueError("l1_exponent must be finite and positive")
    if not bool(torch.isfinite(terms.l1).all() and torch.isfinite(terms.structural).all()):
        raise ValueError("pixel terms must be finite")
    if bool((terms.l1 < 0.0).any()):
        raise ValueError("L1 term must be nonnegative")

    with torch.no_grad():
        raw = 0.8 * terms.l1.pow(float(l1_exponent)) + 0.2 * terms.structural
        minimum, maximum = raw.aminmax()
        pixel = (raw - minimum) / (maximum - minimum + 1e-8)
    return pixel.unsqueeze(0)


def normalized_oscd_pixel_cue(
    reference_rgb: torch.Tensor,
    online_rgb: torch.Tensor,
    *,
    l1_exponent: float = 1.0,
) -> torch.Tensor:
    """Reproduce O-SCD's normalized pixel/SSIM cue, optionally powering L1."""

    return normalized_oscd_pixel_cue_from_terms(
        oscd_pixel_terms(reference_rgb, online_rgb),
        l1_exponent=l1_exponent,
    )


def semantic_from_cached_sum(
    cached_sum: torch.Tensor,
    original_pixel_cue: torch.Tensor,
) -> torch.Tensor:
    """Recover cached ``S`` from the historical sum cue ``P+S``."""

    if not isinstance(cached_sum, torch.Tensor) or not isinstance(
        original_pixel_cue, torch.Tensor
    ):
        raise TypeError("cached_sum and original_pixel_cue must be tensors")
    if cached_sum.shape != original_pixel_cue.shape:
        raise ValueError("cached_sum and original_pixel_cue must share shape")
    if (
        cached_sum.device != original_pixel_cue.device
        or cached_sum.dtype != original_pixel_cue.dtype
    ):
        raise ValueError("cached_sum and original_pixel_cue must share device and dtype")
    if not torch.is_floating_point(cached_sum):
        raise TypeError("cues must be floating point")
    if not bool(
        torch.isfinite(cached_sum).all()
        and torch.isfinite(original_pixel_cue).all()
    ):
        raise ValueError("cues must be finite")
    tolerance = 1e-5
    if bool((cached_sum < -tolerance).any() or (cached_sum > 2.0 + tolerance).any()):
        raise ValueError("cached P+S cue must lie in [0,2]")
    if bool(
        (original_pixel_cue < -tolerance).any()
        or (original_pixel_cue > 1.0 + tolerance).any()
    ):
        raise ValueError("original pixel cue must lie in [0,1]")
    return (cached_sum - original_pixel_cue.clamp(0.0, 1.0)).clamp(0.0, 1.0)


def fuse_power_product(
    pixel_cue: torch.Tensor,
    semantic_cue: torch.Tensor,
    *,
    exponent: float = 0.3,
) -> torch.Tensor:
    """Return scale-matched product fusion ``2 * pixel**alpha * semantic``."""

    if pixel_cue.shape != semantic_cue.shape:
        raise ValueError("pixel_cue and semantic_cue must share shape")
    if pixel_cue.device != semantic_cue.device or pixel_cue.dtype != semantic_cue.dtype:
        raise ValueError("pixel_cue and semantic_cue must share device and dtype")
    if not math.isfinite(float(exponent)) or exponent <= 0.0:
        raise ValueError("exponent must be finite and positive")
    if not bool(torch.isfinite(pixel_cue).all() and torch.isfinite(semantic_cue).all()):
        raise ValueError("cues must be finite")
    tolerance = 1e-5
    for name, cue in (("pixel", pixel_cue), ("semantic", semantic_cue)):
        if bool((cue < -tolerance).any() or (cue > 1.0 + tolerance).any()):
            raise ValueError(f"{name} cue must lie in [0,1]")
    return (
        2.0
        * pixel_cue.clamp(0.0, 1.0).pow(float(exponent))
        * semantic_cue.clamp(0.0, 1.0)
    ).clamp(0.0, 2.0)


def smoothstep_soft_binarize(
    cue: torch.Tensor,
    *,
    low: float,
    high: float,
) -> torch.Tensor:
    """Map a narrow uncertainty band smoothly between exact zero and one."""

    if not isinstance(cue, torch.Tensor):
        raise TypeError("cue must be a tensor")
    if not torch.is_floating_point(cue):
        raise TypeError("cue must be floating point")
    if not bool(torch.isfinite(cue).all()):
        raise ValueError("cue must be finite")
    if not (
        math.isfinite(float(low))
        and math.isfinite(float(high))
        and 0.0 <= low < high <= 1.0
    ):
        raise ValueError("smoothstep band must satisfy 0 <= low < high <= 1")
    tolerance = 1e-5
    if bool((cue < -tolerance).any() or (cue > 1.0 + tolerance).any()):
        raise ValueError("cue must lie in [0,1]")
    unit = ((cue.clamp(0.0, 1.0) - float(low)) / (float(high) - float(low))).clamp(
        0.0, 1.0
    )
    return unit.square() * (3.0 - 2.0 * unit)


def sigmoid_soft_binarize(
    cue: torch.Tensor,
    *,
    tau: float | torch.Tensor,
    width: float | torch.Tensor,
    edge_probability: float = 0.05,
) -> torch.Tensor:
    """Sharpen a normalized cue with a learnable sigmoid boundary.

    ``width`` is the half-width of the transition band: the output is
    ``edge_probability`` at ``tau-width`` and ``1-edge_probability`` at
    ``tau+width``.  This keeps the learned width directly comparable with the
    historical smoothstep band ``[tau-width, tau+width]``.
    """

    if not isinstance(cue, torch.Tensor):
        raise TypeError("cue must be a tensor")
    if not torch.is_floating_point(cue):
        raise TypeError("cue must be floating point")
    if not bool(torch.isfinite(cue).all()):
        raise ValueError("cue must be finite")
    tolerance = 1e-5
    if bool((cue < -tolerance).any() or (cue > 1.0 + tolerance).any()):
        raise ValueError("cue must lie in [0,1]")
    if not math.isfinite(float(edge_probability)) or not 0.0 < edge_probability < 0.5:
        raise ValueError("edge_probability must lie in (0,0.5)")

    tau_tensor = torch.as_tensor(tau, dtype=cue.dtype, device=cue.device)
    width_tensor = torch.as_tensor(width, dtype=cue.dtype, device=cue.device)
    if not bool(torch.isfinite(tau_tensor).all() and torch.isfinite(width_tensor).all()):
        raise ValueError("tau and width must be finite")
    if bool(((tau_tensor <= 0.0) | (tau_tensor >= 1.0)).any()):
        raise ValueError("tau must lie in (0,1)")
    if bool((width_tensor <= 0.0).any()):
        raise ValueError("width must be positive")

    edge_logit = math.log((1.0 - edge_probability) / edge_probability)
    return torch.sigmoid(edge_logit * (cue.clamp(0.0, 1.0) - tau_tensor) / width_tensor)


def power_product_from_cached_sum(
    cached_sum: torch.Tensor,
    pixel_cue: torch.Tensor,
    *,
    exponent: float = 0.3,
) -> PowerProductCue:
    """Recover ``S`` from cached ``P+S`` and return ``2 * P**alpha * S``.

    The shipped fixed-pose cache stores only the original sum cue.  Recomputing
    the inexpensive pixel term lets the viewer recover the already-computed SAM
    term without loading SAM2.1 or changing the historical cue definition.
    """

    semantic = semantic_from_cached_sum(cached_sum, pixel_cue)
    pixel = pixel_cue.clamp(0.0, 1.0)
    fused = fuse_power_product(pixel, semantic, exponent=exponent)
    return PowerProductCue(pixel=pixel, semantic=semantic, fused=fused)
