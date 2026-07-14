# Data contracts for target-conditioned Gaussian NBV.
# See docs/target_gaussian_nbv.md §4. Information-matrix algebra is float64
# numpy; renders stay float32 torch. CPU-importable: no CUDA at import time.

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import numpy as np

# Parameter groups that may legally appear in a TargetParameterSpec.
# Raw quaternions are forbidden (normalization gauge -> rank-deficient FIM);
# rotation support will arrive as a 3-dim SO(3) tangent group.
ALLOWED_PARAMETER_GROUPS = {"mean": 3, "log_scale": 3}
FORBIDDEN_PARAMETER_GROUPS = {"rotation", "quaternion", "raw_quaternion"}


@dataclass
class TargetHandle:
    """A stable reference to the target Gaussian(s).

    persistent_id: repo-wide stable int64 ID (survives densify/prune).
    current_indices: current row index/indices into the model tensors; shape [K]
        (list of python ints to stay CPU/JSON friendly). MVP: K == 1.
    """
    persistent_id: int
    current_indices: list[int]
    mode: str = "single"  # "single" | "cluster"
    frozen_during_episode: bool = True

    def __post_init__(self):
        if self.mode not in ("single", "cluster"):
            raise ValueError(f"TargetHandle.mode must be 'single' or 'cluster', got {self.mode!r}")
        if self.mode == "single" and len(self.current_indices) != 1:
            raise ValueError("single-target handle must have exactly one current index")


@dataclass
class TargetParameterSpec:
    """Which parameter groups form theta_t, and their slices into the packed vector."""
    parameter_names: list[str] = field(default_factory=lambda: ["mean", "log_scale"])
    description: str = "theta_t = [mu_xyz, log_s_xyz]; _scaling is stored in log-space already"

    def __post_init__(self):
        for name in self.parameter_names:
            if name in FORBIDDEN_PARAMETER_GROUPS:
                raise ValueError(
                    f"raw quaternion parameter group {name!r} is forbidden; "
                    "use the SO(3) tangent parameterization when rotation support lands"
                )
            if name not in ALLOWED_PARAMETER_GROUPS:
                raise ValueError(f"unsupported parameter group {name!r}; allowed: {sorted(ALLOWED_PARAMETER_GROUPS)}")
        if len(set(self.parameter_names)) != len(self.parameter_names):
            raise ValueError("duplicate parameter groups in spec")

    @property
    def dimension(self) -> int:
        return sum(ALLOWED_PARAMETER_GROUPS[n] for n in self.parameter_names)

    def slices(self) -> dict[str, slice]:
        out, offset = {}, 0
        for name in self.parameter_names:
            d = ALLOWED_PARAMETER_GROUPS[name]
            out[name] = slice(offset, offset + d)
            offset += d
        return out


@dataclass
class CandidateCamera:
    """A synthesized candidate pose. Pose stored as OpenGL c2w quaternion (wxyz,
    real-first) + world position, matching viewer.py's create_mini_cam input
    convention. `minicam` is built lazily on GPU and never serialized."""
    cand_id: int
    position: np.ndarray          # (3,) float64, world
    wxyz: np.ndarray              # (4,) float64, OpenGL c2w quaternion, real part first
    fovx: float
    fovy: float
    width: int
    height: int
    shell_index: int = 0
    movement_cost: float = 0.0
    meta: dict = field(default_factory=dict)
    minicam: Any = None           # scene.cameras.MiniCam, lazy, not serialized

    def to_json_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k != "minicam"}
        d["position"] = np.asarray(self.position, dtype=np.float64).tolist()
        d["wxyz"] = np.asarray(self.wxyz, dtype=np.float64).tolist()
        return d

    @classmethod
    def from_json_dict(cls, d: dict) -> "CandidateCamera":
        d = dict(d)
        d["position"] = np.asarray(d["position"], dtype=np.float64)
        d["wxyz"] = np.asarray(d["wxyz"], dtype=np.float64)
        d.pop("minicam", None)
        return cls(**d)


@dataclass
class TargetVisibility:
    """Result of evaluating whether/how the target is seen from one camera.
    occlusion_ratio in [0,1]; responsibility fields are backend-A' proxies
    (see docs §8) — never multiplied into exact FIMs."""
    valid: bool
    bbox_xyxy: Optional[tuple[int, int, int, int]] = None
    projected_radius_px: float = 0.0
    responsibility_sum: float = 0.0
    responsibility_mean: float = 0.0
    visible_pixel_count: int = 0
    occlusion_ratio: float = 1.0
    invalid_reason: Optional[str] = None
    responsibility_map: Any = None  # optional torch tensor, debug only


@dataclass
class TargetInformationState:
    """Accumulated target-block information from observed views.

    H_data excludes damping. H_prior() applies damping exactly once and returns
    a read-only float64 array; scorers must never add damping again.
    """
    target_pid: int
    spec: TargetParameterSpec
    H_data: np.ndarray                      # (D,D) float64, symmetric, no damping
    absolute_damping: float
    relative_damping: float
    observed_view_ids: list[str] = field(default_factory=list)
    per_view_cache: dict = field(default_factory=dict)  # view_id -> (D,D) float64
    version: int = 0
    model_version: str = ""

    def damping(self) -> float:
        diag_mean = float(np.mean(np.diag(self.H_data))) if self.H_data.size else 0.0
        return max(self.absolute_damping, self.relative_damping * diag_mean)

    def H_prior(self) -> np.ndarray:
        d = self.spec.dimension
        H = self.H_data + self.damping() * np.eye(d, dtype=np.float64)
        H = 0.5 * (H + H.T)
        H.flags.writeable = False
        return H


@dataclass
class CandidateScore:
    """Score breakdown for one candidate. Invalid candidates carry a reason;
    numeric fields for invalid candidates are 0.0, never NaN."""
    candidate: CandidateCamera
    valid: bool = True
    invalid_reason: Optional[str] = None
    proxy_score: float = 0.0
    exact_score: float = 0.0
    d_gain: float = 0.0
    trace_gain: float = 0.0
    e_gain: float = 0.0
    change_eig: float = 0.0
    movement_cost: float = 0.0
    visibility: Optional[TargetVisibility] = None
    delta_H: Optional[np.ndarray] = None    # (D,D) float64, exact only


@dataclass
class SelectionResult:
    best: Optional[CandidateScore]
    scores: list[CandidateScore]
    H_before: np.ndarray
    predicted_H_after: Optional[np.ndarray]
    config_snapshot: dict
    runtime: dict = field(default_factory=dict)  # phase name -> seconds
