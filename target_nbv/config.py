# Configuration for target-conditioned Gaussian NBV (docs/target_gaussian_nbv.md).
# Plain dataclasses + explicit validate(); JSON round-trip for artifact dumps.

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

from target_nbv.types import ALLOWED_PARAMETER_GROUPS, FORBIDDEN_PARAMETER_GROUPS

SUPPORTED_MODES = (
    "geometry_proxy", "geometry_exact", "geometry_schur",
    "change_fisher", "change_beta", "joint",
)
MVP_MODES = ("geometry_proxy", "geometry_exact")


@dataclass
class CandidateConfig:
    sampling: str = "fibonacci_sphere"
    count: int = 128
    radius_shells: list[float] = field(default_factory=lambda: [0.75, 1.0, 1.5])
    desired_projected_radius_px: float = 24.0
    min_projected_radius_px: float = 4.0
    max_projected_radius_px: float = 96.0
    min_distance: float = 0.1
    max_distance: float = 10.0
    hemisphere_from_normal: bool = False
    jitter: float = 0.0  # relative angular jitter on sphere points


@dataclass
class VisibilityConfig:
    backend: str = "metric_map"   # "metric_map" (A') | "exact_alpha_t" (B, stub)
    min_responsibility: float = 1e-4
    max_occlusion_ratio: float = 0.95
    crop_margin_px: int = 8
    reject_fully_occluded: bool = True


@dataclass
class ProxyConfig:
    enabled: bool = True
    top_k: int = 12
    uses_visibility: bool = True          # False -> render-free sigmoid(opacity) fallback
    responsibility_normalization: str = "percentile90"


@dataclass
class JacobianConfig:
    backend: str = "finite_difference"
    exact_top_k: int = 8
    mean_epsilon_rel: float = 0.01        # eps_mean = rel * r_world(target)
    log_scale_epsilon: float = 0.02
    fixed_crop: bool = True
    separate_rgb_channels: bool = True


@dataclass
class InformationConfig:
    absolute_damping: float = 1e-6
    relative_damping: float = 1e-6
    max_jitter: float = 1e-2


@dataclass
class ScoringConfig:
    d_weight: float = 1.0
    trace_weight: float = 0.0
    e_weight: float = 0.25
    movement_weight: float = 0.05
    translation_scale: float = 1.0
    rotation_weight: float = 1.0


@dataclass
class NeighborConfig:
    enabled: bool = False                 # stage 10, default off
    max_neighbors: int = 4
    selection: str = "covisibility_then_distance"
    neighbor_damping: float = 1e-5
    fallback_to_target_only: bool = True


@dataclass
class ChangeConfig:
    fisher_enabled: bool = False          # stage 11
    beta_enabled: bool = False            # stage 11
    beta_prior_a: float = 1.0
    beta_prior_b: float = 1.0
    pseudo_count_scale: float = 1.0


@dataclass
class StoppingConfig:
    minimum_predicted_gain: float = 1e-4
    max_views: int = 10


@dataclass
class DebugConfig:
    save_renders: bool = True
    save_responsibility_maps: bool = False
    save_candidate_table: bool = True
    verify_model_restoration: bool = True


@dataclass
class TargetNBVConfig:
    mode: str = "geometry_exact"
    seed: int = 7
    parameter_groups: list[str] = field(default_factory=lambda: ["mean", "log_scale"])
    candidates: CandidateConfig = field(default_factory=CandidateConfig)
    visibility: VisibilityConfig = field(default_factory=VisibilityConfig)
    proxy: ProxyConfig = field(default_factory=ProxyConfig)
    jacobian: JacobianConfig = field(default_factory=JacobianConfig)
    information: InformationConfig = field(default_factory=InformationConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    neighbors: NeighborConfig = field(default_factory=NeighborConfig)
    change: ChangeConfig = field(default_factory=ChangeConfig)
    stopping: StoppingConfig = field(default_factory=StoppingConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

    def validate(self) -> "TargetNBVConfig":
        if self.mode not in SUPPORTED_MODES:
            raise ValueError(f"unsupported mode {self.mode!r}; supported: {SUPPORTED_MODES}")
        for g in self.parameter_groups:
            if g in FORBIDDEN_PARAMETER_GROUPS:
                raise ValueError(f"raw quaternion parameter group {g!r} is forbidden")
            if g not in ALLOWED_PARAMETER_GROUPS:
                raise ValueError(f"unknown parameter group {g!r}")
        if self.candidates.count < self.proxy.top_k:
            raise ValueError(
                f"candidate count ({self.candidates.count}) must be >= proxy top_k ({self.proxy.top_k})")
        if self.proxy.top_k < self.jacobian.exact_top_k:
            raise ValueError(
                f"proxy top_k ({self.proxy.top_k}) must be >= exact top_k ({self.jacobian.exact_top_k})")
        if self.information.absolute_damping <= 0:
            raise ValueError("absolute_damping must be > 0")
        if self.information.relative_damping < 0:
            raise ValueError("relative_damping must be >= 0")
        if self.jacobian.mean_epsilon_rel <= 0 or self.jacobian.log_scale_epsilon <= 0:
            raise ValueError("finite-difference epsilons must be > 0")
        if not self.candidates.radius_shells or any(s <= 0 for s in self.candidates.radius_shells):
            raise ValueError("radius_shells must be non-empty and positive")
        if sorted(self.candidates.radius_shells) != list(self.candidates.radius_shells):
            raise ValueError("radius_shells must be sorted ascending")
        if not (0 < self.candidates.min_projected_radius_px
                < self.candidates.desired_projected_radius_px
                < self.candidates.max_projected_radius_px):
            raise ValueError("projected radius bounds must satisfy min < desired < max")
        if self.candidates.min_distance <= 0 or self.candidates.max_distance <= self.candidates.min_distance:
            raise ValueError("require 0 < min_distance < max_distance")
        if self.visibility.backend not in ("metric_map", "exact_alpha_t"):
            raise ValueError(f"unknown visibility backend {self.visibility.backend!r}")
        if self.mode in ("geometry_schur",) and not self.neighbors.enabled:
            raise ValueError("mode geometry_schur requires neighbors.enabled=true")
        if self.mode == "change_fisher" and not self.change.fisher_enabled:
            raise ValueError("mode change_fisher requires change.fisher_enabled=true")
        if self.mode == "change_beta" and not self.change.beta_enabled:
            raise ValueError("mode change_beta requires change.beta_enabled=true")
        if self.change.beta_prior_a <= 0 or self.change.beta_prior_b <= 0:
            raise ValueError("Beta priors must be > 0")
        if self.stopping.max_views < 1:
            raise ValueError("stopping.max_views must be >= 1")
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "TargetNBVConfig":
        d = dict(d)
        nested = {
            "candidates": CandidateConfig, "visibility": VisibilityConfig,
            "proxy": ProxyConfig, "jacobian": JacobianConfig,
            "information": InformationConfig, "scoring": ScoringConfig,
            "neighbors": NeighborConfig, "change": ChangeConfig,
            "stopping": StoppingConfig, "debug": DebugConfig,
        }
        for key, klass in nested.items():
            if key in d and isinstance(d[key], dict):
                d[key] = klass(**d[key])
        return cls(**d).validate()

    @classmethod
    def from_json(cls, s: str) -> "TargetNBVConfig":
        return cls.from_dict(json.loads(s))
