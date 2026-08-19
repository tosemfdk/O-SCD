"""Causal PCA sign mapping utilities for online evolving-scene SCD.

This module factors the sign-mapping logic that was first prototyped in
``/tmp/run_online_pca_sign_mapping_dc_only.py`` into small reusable pieces:

* exact causal prefix PCA with previous-frame sign alignment,
* robust MAD epsilon estimation from a bounded causal stable-token reservoir,
* global and component-balanced same-sign follow evidence,
* the fractional Beta pseudo-posterior ``P(q_plus < q_minus)``, and
* the primary Global+Balanced NEW-sign consensus gate used by E4 seeding.

All state updates are causal: callers must feed frame ``t`` only after all data
from frames ``< t`` that should influence the state has already been applied.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import cv2
import numpy as np
import torch

try:  # Match the prototype exactly when SciPy is available.
    from scipy.stats import beta as _scipy_beta_distribution
except Exception:  # pragma: no cover - exercised only in minimal deployments.
    _scipy_beta_distribution = None

Sign = Literal["+", "-"]
MappingSide = Literal["+=ADD,-=REMOVE", "+=REMOVE,-=ADD"]


@dataclass(frozen=True)
class SignMappingConfig:
    """Thresholds copied from the causal sign-mapping prototype defaults."""

    eps_sigma: float = 2.5
    stable_bank_size: int = 65_536
    axis_stability_min: float = 0.90
    min_camera_translation: float = 0.10
    min_sign_cells: int = 4
    min_memory_cells: int = 4
    memory_weight_floor: float = 0.05
    component_threshold: float = 0.15
    min_component_cells: int = 3
    component_area_cap: int = 32
    confirm_probability: float = 0.90
    confirm_valid_views: int = 2
    consensus_probability: float = 0.80

    def __post_init__(self) -> None:
        if self.eps_sigma < 0:
            raise ValueError("eps_sigma must be non-negative")
        if self.stable_bank_size <= 0:
            raise ValueError("stable_bank_size must be positive")
        if not 0.5 < self.confirm_probability < 1.0:
            raise ValueError("confirm_probability must be in (0.5, 1)")
        if self.confirm_valid_views <= 0:
            raise ValueError("confirm_valid_views must be positive")
        if not 0.5 < self.consensus_probability < 1.0:
            raise ValueError("consensus_probability must be in (0.5, 1)")


@dataclass(frozen=True)
class CausalPC1Update:
    pc: torch.Tensor
    mean: torch.Tensor
    axis_stability: float
    signed_axis_cosine_before_alignment: float | None
    explained_variance_ratio: float
    stable_center: float
    stable_sigma_mad: float
    epsilon_negative: float
    epsilon_positive: float
    prefix_tokens: int
    stable_tokens_seen: int
    stable_bank_tokens: int

    def as_dict(self, *, include_tensors: bool = True) -> dict[str, Any]:
        row: dict[str, Any] = {
            "axis_stability": self.axis_stability,
            "signed_axis_cosine_before_alignment": self.signed_axis_cosine_before_alignment,
            "explained_variance_ratio": self.explained_variance_ratio,
            "stable_center": self.stable_center,
            "stable_sigma_mad": self.stable_sigma_mad,
            "epsilon_negative": self.epsilon_negative,
            "epsilon_positive": self.epsilon_positive,
            "prefix_tokens": self.prefix_tokens,
            "stable_tokens_seen": self.stable_tokens_seen,
            "stable_bank_tokens": self.stable_bank_tokens,
        }
        if include_tensors:
            row.update({"pc": self.pc, "mean": self.mean})
        return row


class CausalPC1:
    """Exact prefix-covariance PC1 with causal sign alignment.

    The running covariance uses every token seen so far.  Epsilon thresholds use
    a bounded reservoir of caller-designated stable tokens, matching the
    prototype's stratified old/new reservoir replacement exactly.
    """

    def __init__(
        self,
        channels: int,
        stable_bank_size: int = 65_536,
        seed: int = 0,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        if isinstance(channels, bool) or channels <= 0:
            raise ValueError("channels must be a positive integer")
        if isinstance(stable_bank_size, bool) or stable_bank_size <= 0:
            raise ValueError("stable_bank_size must be a positive integer")
        self.channels = int(channels)
        self.stable_bank_size = int(stable_bank_size)
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.sum_x = torch.zeros(self.channels, dtype=torch.float64, device=self.device)
        self.sum_xx = torch.zeros((self.channels, self.channels), dtype=torch.float64, device=self.device)
        self.count = 0
        self.prev_pc: torch.Tensor | None = None
        self.stable_bank = torch.empty((0, self.channels), dtype=torch.float16, device=self.device)
        self.stable_seen = 0
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(seed))

    def _update_stable_bank(self, stable: torch.Tensor) -> None:
        stable = stable.detach().to(device=self.device, dtype=torch.float16)
        incoming = int(stable.shape[0])
        if incoming == 0:
            return
        old_seen = self.stable_seen
        self.stable_seen += incoming
        if len(self.stable_bank) + incoming <= self.stable_bank_size:
            self.stable_bank = torch.cat((self.stable_bank, stable), dim=0)
            return

        old_keep = int(round(self.stable_bank_size * old_seen / max(self.stable_seen, 1)))
        old_keep = max(0, min(old_keep, len(self.stable_bank), self.stable_bank_size))
        new_keep = self.stable_bank_size - old_keep
        new_keep = min(new_keep, incoming)
        if old_keep + new_keep < self.stable_bank_size:
            old_keep = min(len(self.stable_bank), self.stable_bank_size - new_keep)
        old_ids = torch.randperm(len(self.stable_bank), generator=self.generator, device=self.device)[:old_keep]
        new_ids = torch.randperm(incoming, generator=self.generator, device=self.device)[:new_keep]
        self.stable_bank = torch.cat((self.stable_bank[old_ids], stable[new_ids]), dim=0)

    @torch.inference_mode()
    def update(
        self,
        delta: torch.Tensor,
        stable_mask: torch.Tensor,
        eps_sigma: float = 2.5,
    ) -> CausalPC1Update:
        """Update prefix PCA from one frame of token deltas."""
        if delta.ndim != 2 or int(delta.shape[1]) != self.channels:
            raise ValueError(f"delta must be [tokens,{self.channels}]")
        stable_flat = stable_mask.detach().to(device=self.device, dtype=torch.bool).flatten()
        if int(stable_flat.numel()) != int(delta.shape[0]):
            raise ValueError("stable_mask must flatten to one entry per delta token")

        delta_device = delta.detach().to(device=self.device)
        x64 = delta_device.double()
        self.sum_x += x64.sum(dim=0)
        self.sum_xx += x64.T @ x64
        self.count += int(delta_device.shape[0])
        self._update_stable_bank(delta_device[stable_flat])
        if int(len(self.stable_bank)) == 0:
            raise ValueError("stable token bank is empty; provide at least one stable token")

        mean = self.sum_x / self.count
        covariance = self.sum_xx / self.count - torch.outer(mean, mean)
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        pc = eigenvectors[:, -1].float()
        if self.prev_pc is None:
            anchor = int(torch.argmax(pc.abs()).item())
            if float(pc[anchor]) < 0:
                pc = -pc
            stability = 1.0
            signed_cosine = None
        else:
            signed_cosine = float(torch.dot(pc, self.prev_pc).item())
            stability = abs(signed_cosine)
            if signed_cosine < 0:
                pc = -pc
        self.prev_pc = pc.detach().clone()

        stable_scores = self.stable_bank.float() @ pc
        center = torch.median(stable_scores)
        sigma = 1.4826 * torch.median(torch.abs(stable_scores - center))
        sigma = sigma.clamp_min(1e-8)
        total_energy = eigenvalues.clamp_min(0).sum().clamp_min(1e-12)
        return CausalPC1Update(
            pc=pc,
            mean=mean.float(),
            axis_stability=stability,
            signed_axis_cosine_before_alignment=signed_cosine,
            explained_variance_ratio=float((eigenvalues[-1] / total_energy).item()),
            stable_center=float(center.item()),
            stable_sigma_mad=float(sigma.item()),
            epsilon_negative=float((center - eps_sigma * sigma).item()),
            epsilon_positive=float((center + eps_sigma * sigma).item()),
            prefix_tokens=self.count,
            stable_tokens_seen=self.stable_seen,
            stable_bank_tokens=int(len(self.stable_bank)),
        )

    @torch.inference_mode()
    def strong_signed_masks(
        self,
        delta: torch.Tensor,
        cue_active: torch.Tensor,
        update: CausalPC1Update,
        *,
        output_shape: tuple[int, int] = (64, 64),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project deltas on PC1 and threshold into strong +/- cue masks."""
        if delta.ndim != 2 or int(delta.shape[1]) != self.channels:
            raise ValueError(f"delta must be [tokens,{self.channels}]")
        if int(np.prod(output_shape)) != int(delta.shape[0]):
            raise ValueError("output_shape must contain exactly one cell per delta token")
        score = (delta.to(device=update.pc.device) @ update.pc).reshape(output_shape)
        cue = cue_active.to(device=score.device, dtype=torch.bool)
        if tuple(cue.shape) != tuple(output_shape):
            raise ValueError("cue_active must match output_shape")
        plus = cue & (score > update.epsilon_positive)
        minus = cue & (score < update.epsilon_negative)
        return plus, minus, score


@dataclass(frozen=True)
class FollowEvidence:
    follow: float | None
    conflict: float | None
    coverage: float | None
    mass: float
    p99: float
    support_cells: int
    components: list[dict[str, Any]] = field(default_factory=list)
    reason: str | None = None


def normalized_memory(raw: np.ndarray, floor: float) -> tuple[np.ndarray, float]:
    """Normalize positive memory contrast by its positive p99 and floor it."""
    raw = np.maximum(np.asarray(raw, dtype=np.float64), 0.0)
    positive = raw[raw > 0]
    if not len(positive):
        return np.zeros_like(raw), 0.0
    p99 = max(float(np.quantile(positive, 0.99)), 1e-12)
    weight = np.clip(raw / p99, 0.0, 1.0)
    weight[weight < floor] = 0.0
    return weight, p99


def pooled_follow(
    raw_memory: np.ndarray,
    same_sign: np.ndarray,
    opposite_sign: np.ndarray,
    config: SignMappingConfig = SignMappingConfig(),
) -> FollowEvidence:
    """Global pooled same-sign follow evidence for one sign memory."""
    weight, p99 = normalized_memory(raw_memory, config.memory_weight_floor)
    same = np.asarray(same_sign, dtype=bool)
    opposite = np.asarray(opposite_sign, dtype=bool)
    mass = float(weight.sum())
    support_cells = int(np.count_nonzero(weight >= config.component_threshold))
    if p99 <= 1e-8:
        return FollowEvidence(None, None, None, mass, p99, support_cells, [], "zero_sign_contrast")
    if support_cells < config.min_memory_cells or mass <= 1e-8:
        return FollowEvidence(None, None, None, mass, p99, support_cells, [], "insufficient_memory_support")
    same_mass = float((weight * same).sum())
    opposite_mass = float((weight * opposite).sum())
    return FollowEvidence(
        follow=same_mass / mass,
        conflict=opposite_mass / mass,
        coverage=same_mass / max(float(same.sum()), 1.0),
        mass=mass,
        p99=p99,
        support_cells=support_cells,
        components=[],
    )


def component_balanced_follow(
    raw_memory: np.ndarray,
    same_sign: np.ndarray,
    opposite_sign: np.ndarray,
    config: SignMappingConfig = SignMappingConfig(),
) -> FollowEvidence:
    """Component-balanced evidence that caps each memory component's influence."""
    weight, p99 = normalized_memory(raw_memory, config.memory_weight_floor)
    same = np.asarray(same_sign, dtype=bool)
    opposite = np.asarray(opposite_sign, dtype=bool)
    support = weight >= config.component_threshold
    count, labels, stats, _ = cv2.connectedComponentsWithStats(support.astype(np.uint8), connectivity=4)
    rows: list[dict[str, Any]] = []
    for component_id in range(1, count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < config.min_component_cells:
            continue
        component = labels == component_id
        local_weight = weight * component
        mass = float(local_weight.sum())
        if mass <= 1e-8:
            continue
        same_mass = float((local_weight * same).sum())
        opposite_mass = float((local_weight * opposite).sum())
        rows.append(
            {
                "area": area,
                "mass": mass,
                "follow": same_mass / mass,
                "conflict": opposite_mass / mass,
                "balance_weight": float(min(area, config.component_area_cap)),
            }
        )
    total_mass = float(weight.sum())
    support_cells = int(support.sum())
    if p99 <= 1e-8:
        return FollowEvidence(None, None, None, total_mass, p99, support_cells, rows, "zero_sign_contrast")
    if not rows:
        return FollowEvidence(None, None, None, total_mass, p99, support_cells, rows, "no_reliable_memory_component")
    balance = np.asarray([row["balance_weight"] for row in rows], dtype=np.float64)
    follow = float(np.average([row["follow"] for row in rows], weights=balance))
    conflict = float(np.average([row["conflict"] for row in rows], weights=balance))
    covered = float(sum(row["mass"] * row["follow"] for row in rows))
    return FollowEvidence(
        follow=follow,
        conflict=conflict,
        coverage=covered / max(float(same.sum()), 1.0),
        mass=total_mass,
        p99=p99,
        support_cells=support_cells,
        components=rows,
    )


_BETA_NODES, _BETA_WEIGHTS = np.polynomial.legendre.leggauss(512)
_BETA_X = 0.5 * (_BETA_NODES + 1.0)
_BETA_W = 0.5 * _BETA_WEIGHTS


def _beta_logpdf_fallback(x: np.ndarray, a: float, b: float) -> np.ndarray:
    log_norm = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    return log_norm + (a - 1.0) * np.log(x) + (b - 1.0) * np.log1p(-x)


def _beta_cdf_quadrature(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    grid = 0.5 * x * (_BETA_NODES + 1.0)
    weights = 0.5 * x * _BETA_WEIGHTS
    values = np.exp(np.clip(_beta_logpdf_fallback(grid, a, b), -745.0, 700.0))
    return float(np.clip(np.sum(weights * values), 0.0, 1.0))


def probability_beta_less(a1: float, b1: float, a2: float, b2: float) -> float:
    """Return ``P(X < Y)``, ``X~Beta(a1,b1)``, ``Y~Beta(a2,b2)``."""
    for value in (a1, b1, a2, b2):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("Beta parameters must be finite and positive")
    if _scipy_beta_distribution is not None:
        log_pdf = _scipy_beta_distribution.logpdf(_BETA_X, a1, b1)
        survival = _scipy_beta_distribution.sf(_BETA_X, a2, b2)
    else:  # pragma: no cover
        log_pdf = _beta_logpdf_fallback(_BETA_X, a1, b1)
        survival = np.asarray([1.0 - _beta_cdf_quadrature(float(x), a2, b2) for x in _BETA_X])
    values = np.exp(np.clip(log_pdf, -745.0, 700.0)) * survival
    probability = float(np.sum(_BETA_W * values))
    return float(np.clip(probability, 0.0, 1.0))


@dataclass(frozen=True)
class PosteriorSnapshot:
    alpha_plus: float
    beta_plus: float
    alpha_minus: float
    beta_minus: float
    p_plus_is_add: float
    pending_side: MappingSide | None
    pending_streak: int
    confirmed_mapping: MappingSide | None
    confirmed_at_frame: str | None
    mapping_flips: int
    valid_updates: int

    @property
    def p_plus_is_new(self) -> float:
        """Alias used by E4 NEW seeding; ADD and NEW are the same sign concept."""
        return self.p_plus_is_add

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class SignPosterior:
    """Fractional Beta pseudo-posterior for one evidence-pooling method."""

    def __init__(self, name: str, confirm_probability: float = 0.90, confirm_views: int = 2):
        if not 0.5 < confirm_probability < 1.0:
            raise ValueError("confirm_probability must be in (0.5, 1)")
        if confirm_views <= 0:
            raise ValueError("confirm_views must be positive")
        self.name = name
        self.alpha_plus = 1.0
        self.beta_plus = 1.0
        self.alpha_minus = 1.0
        self.beta_minus = 1.0
        self.probability_plus_is_add = 0.5
        self.confirm_probability = float(confirm_probability)
        self.confirm_views = int(confirm_views)
        self.pending_side: MappingSide | None = None
        self.pending_streak = 0
        self.confirmed_mapping: MappingSide | None = None
        self.confirmed_at_frame: str | None = None
        self.mapping_flips = 0
        self.valid_updates = 0

    def update(self, follow_plus: float, follow_minus: float, frame_name: str) -> PosteriorSnapshot:
        if not np.isfinite(follow_plus) or not 0.0 <= follow_plus <= 1.0:
            raise ValueError("follow_plus must be in [0, 1]")
        if not np.isfinite(follow_minus) or not 0.0 <= follow_minus <= 1.0:
            raise ValueError("follow_minus must be in [0, 1]")
        self.alpha_plus += float(follow_plus)
        self.beta_plus += 1.0 - float(follow_plus)
        self.alpha_minus += float(follow_minus)
        self.beta_minus += 1.0 - float(follow_minus)
        self.valid_updates += 1
        self.probability_plus_is_add = probability_beta_less(
            self.alpha_plus,
            self.beta_plus,
            self.alpha_minus,
            self.beta_minus,
        )
        p = self.probability_plus_is_add
        side: MappingSide | None = None
        if p >= self.confirm_probability:
            side = "+=ADD,-=REMOVE"
        elif p <= 1.0 - self.confirm_probability:
            side = "+=REMOVE,-=ADD"
        if side is None:
            self.pending_side = None
            self.pending_streak = 0
        elif side == self.pending_side:
            self.pending_streak += 1
        else:
            self.pending_side = side
            self.pending_streak = 1
        if side is not None and self.pending_streak >= self.confirm_views:
            if self.confirmed_mapping is not None and self.confirmed_mapping != side:
                self.mapping_flips += 1
            if self.confirmed_mapping != side:
                self.confirmed_mapping = side
                self.confirmed_at_frame = frame_name
        return self.snapshot()

    def snapshot(self) -> PosteriorSnapshot:
        return PosteriorSnapshot(
            alpha_plus=self.alpha_plus,
            beta_plus=self.beta_plus,
            alpha_minus=self.alpha_minus,
            beta_minus=self.beta_minus,
            p_plus_is_add=self.probability_plus_is_add,
            pending_side=self.pending_side,
            pending_streak=self.pending_streak,
            confirmed_mapping=self.confirmed_mapping,
            confirmed_at_frame=self.confirmed_at_frame,
            mapping_flips=self.mapping_flips,
            valid_updates=self.valid_updates,
        )


def evidence_skip_reason(
    frame_index: int,
    axis_stability: float,
    translation: float | None,
    plus_cells: int,
    minus_cells: int,
    plus_evidence: FollowEvidence,
    minus_evidence: FollowEvidence,
    config: SignMappingConfig = SignMappingConfig(),
) -> str | None:
    """Return the prototype's reason for skipping a posterior update."""
    if frame_index == 1:
        return "bootstrap_frame"
    if axis_stability < config.axis_stability_min:
        return "unstable_pc1_axis"
    if translation is None or translation < config.min_camera_translation:
        return "insufficient_camera_translation"
    if plus_cells < config.min_sign_cells:
        return "insufficient_positive_cue"
    if minus_cells < config.min_sign_cells:
        return "insufficient_negative_cue"
    if plus_evidence.reason is not None:
        return f"plus_{plus_evidence.reason}"
    if minus_evidence.reason is not None:
        return f"minus_{minus_evidence.reason}"
    return None


def mapping_from_probability(probability: float) -> MappingSide:
    return "+=ADD,-=REMOVE" if probability >= 0.5 else "+=REMOVE,-=ADD"


@dataclass(frozen=True)
class NewSignGateDecision:
    new_sign: Sign | None
    confidence: float | None
    p_global: float
    p_balanced: float
    status: Literal["open", "paused"]
    opened_this_update: bool
    flipped_this_update: bool
    frame_name: str | None = None


class NewSignGate:
    """Primary E4 Global+Balanced consensus gate for NEW sign selection."""

    def __init__(self, threshold: float = 0.80) -> None:
        if not 0.5 < threshold < 1.0:
            raise ValueError("threshold must be in (0.5, 1)")
        self.threshold = float(threshold)
        self.current_new_sign: Sign | None = None
        self.current_confidence: float | None = None
        self.first_open_frame: str | None = None
        self.last_frame: str | None = None
        self.flip_count = 0
        self.valid_updates = 0

    def update(self, p_global: float, p_balanced: float, frame_name: str | None = None) -> NewSignGateDecision:
        if not np.isfinite(p_global) or not 0.0 <= p_global <= 1.0:
            raise ValueError("p_global must be in [0, 1]")
        if not np.isfinite(p_balanced) or not 0.0 <= p_balanced <= 1.0:
            raise ValueError("p_balanced must be in [0, 1]")
        self.valid_updates += 1
        low = 1.0 - self.threshold
        if p_global >= self.threshold and p_balanced >= self.threshold:
            new_sign: Sign | None = "+"
            confidence: float | None = min(float(p_global), float(p_balanced))
        elif p_global <= low and p_balanced <= low:
            new_sign = "-"
            confidence = min(1.0 - float(p_global), 1.0 - float(p_balanced))
        else:
            new_sign = None
            confidence = None

        opened = False
        flipped = False
        if new_sign is not None:
            if self.current_new_sign is None:
                opened = True
                if self.first_open_frame is None:
                    self.first_open_frame = frame_name
            elif self.current_new_sign != new_sign:
                flipped = True
                self.flip_count += 1
            self.current_new_sign = new_sign
            self.current_confidence = confidence
        self.last_frame = frame_name
        return NewSignGateDecision(
            new_sign=new_sign,
            confidence=confidence,
            p_global=float(p_global),
            p_balanced=float(p_balanced),
            status="open" if new_sign is not None else "paused",
            opened_this_update=opened,
            flipped_this_update=flipped,
            frame_name=frame_name,
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "current_new_sign": self.current_new_sign,
            "current_confidence": self.current_confidence,
            "first_open_frame": self.first_open_frame,
            "last_frame": self.last_frame,
            "flip_count": self.flip_count,
            "valid_updates": self.valid_updates,
        }


__all__ = [
    "CausalPC1",
    "CausalPC1Update",
    "FollowEvidence",
    "MappingSide",
    "NewSignGate",
    "NewSignGateDecision",
    "PosteriorSnapshot",
    "Sign",
    "SignMappingConfig",
    "SignPosterior",
    "component_balanced_follow",
    "evidence_skip_reason",
    "mapping_from_probability",
    "normalized_memory",
    "pooled_follow",
    "probability_beta_less",
]
