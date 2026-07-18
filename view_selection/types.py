# ChangeNBV data contracts (spec §4). Frozen where the spec says frozen.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch


class LeakageError(RuntimeError):
    """Selection code touched candidate content it is not allowed to see."""


class ZeroAdjointError(RuntimeError):
    """Visible Gaussians + valid pixels but an all-zero dc adjoint (silent
    fastgs zero-gradient trap, spec §6.3): the run must fail, not score 0."""


@dataclass(frozen=True)
class InformationConfig:
    output_space: Literal["raw", "sigmoid"] = "raw"
    weight_mode: Literal["pose", "current_map", "cue", "consensus"] = "pose"
    num_probes: int = 4
    alpha_threshold: float = 0.5
    lambda_rel: float = 1e-3
    lambda_abs: float = 1e-8
    epsilon: float = 0.1
    gamma: float = 1.0
    beta: float = 1.0

    def key_dict(self) -> dict[str, Any]:
        return {
            "output_space": self.output_space,
            "weight_mode": self.weight_mode,
            "num_probes": self.num_probes,
            "alpha_threshold": self.alpha_threshold,
            "epsilon": self.epsilon,
            "gamma": self.gamma,
            "beta": self.beta,
        }


@dataclass
class ViewInformation:
    frame_id: int
    diagonal: torch.Tensor  # (N,) float32 CPU
    valid_pixels: int
    visible_gaussians: int
    alpha_coverage: float
    model_revision: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SelectionStep:
    step: int
    selected_frame_id: int
    selected_score: float
    candidate_scores: dict[int, float]
    lambda_value: float
    gaussian_count: int
    model_revision: int = 0


@dataclass
class SelectionResult:
    greedy_order: list[int]
    replay_order: list[int]
    steps: list[SelectionStep]
    claim_scope: Literal[
        "active_pose_only", "active_change_aware", "pool_subset_selection"
    ] = "active_pose_only"


class _ForbiddenFrameContent:
    """Sentinel installed over camera image content during selection; any
    access is a leakage violation."""

    def __init__(self, frame_id):
        self._frame_id = frame_id

    def _raise(self, *_a, **_k):
        raise LeakageError(
            f"selection accessed frame {self._frame_id} image content")

    __getattr__ = _raise
    __getitem__ = _raise
    __call__ = _raise


class FrameAccessGuard:
    """Blocks access to candidate image content (original_image /
    candidate_map) on every camera except `allowed_ids` while active."""

    def __init__(self, cameras, allowed_ids=()):
        self._cameras = list(cameras)
        self._allowed = set(allowed_ids)
        self._saved: list[tuple[Any, Any, Any]] = []

    def __enter__(self):
        for i, cam in enumerate(self._cameras):
            if i in self._allowed:
                continue
            saved_img = cam.__dict__.get("original_image", None)
            saved_map = cam.__dict__.get("candidate_map", None)
            self._saved.append((cam, saved_img, saved_map))
            forb = _ForbiddenFrameContent(i)
            cam.__dict__["original_image"] = forb
            cam.__dict__["candidate_map"] = forb
        return self

    def __exit__(self, *exc):
        for cam, img, cmap in self._saved:
            if img is None:
                cam.__dict__.pop("original_image", None)
            else:
                cam.__dict__["original_image"] = img
            if cmap is None:
                cam.__dict__.pop("candidate_map", None)
            else:
                cam.__dict__["candidate_map"] = cmap
        self._saved.clear()
        return False
