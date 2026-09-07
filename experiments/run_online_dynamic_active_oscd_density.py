"""Equal-status dynamic ``R_change`` with active-only O-SCD density control.

The current mutable Gaussian bank is both detector support and representation:

``mutable alpha-T evidence -> binary ACTIVE/INACTIVE -> selective training``

This experiment branch defaults to the validated BF30 lifespan gate with an
output-only CLOSE-candidate opacity fade.  OPEN candidates remain hidden until
hard commit, while training, density control, and detector evidence continue
to use the committed binary lifespan state.

At local update four of a 16-update online schedule, ACTIVE rows alone may be
cloned/split by the original O-SCD screen-space gradient rule.  The same event
also applies the explicitly requested ACTIVE-only opacity/size pruning.  There
is no immutable prefix, root/residual distinction, FastGS VCD/VCP, or K-view
importance window.  New rows copy the source Bayesian and lifespan state once,
then evolve independently.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from utils.sh_utils import RGB2SH, SH2RGB

from experiments.run_online_bayesian_lifespan_thaw import BASE_PLY_REL, quantile_summary
from experiments.run_online_binary_state_lifespan_thaw import (
    LifecycleEvent,
    RunConfig,
    _prediction_from_package,
    event_diagnostics,
    evaluate_after_inference,
    make_controller,
    make_filter,
    posterior_run_diagnostics,
    same_scene_repeated_transition_diagnostics,
    update_binary_lifecycle_chunks,
)
from experiments.run_online_persistent_gaussian_lifespan import (
    DIRECT_PARAMETER_NAMES,
    _cpu_tree,
    _write_csv,
)
from experiments.run_ref_sc1_change_cue_density import (
    DEFAULT_CUE_CACHE,
    DEFAULT_FIXED_CAMERAS,
    DEFAULT_SOURCE,
    SCOPE_LABELS,
    SCOPE_MAX_FRAMES,
    reference_scene_extent,
    select_scope_records,
)


SCOPES = ("scene_change1", "scene_change2", "scene_change3", "continuous")
DENSITY_POLICIES = (
    "none",
    "active_oscd",
    "active_oscd_cue_mixture",
    "active_oscd_cue_mixture_black_child_prune",
    "active_signed_lifespan_score",
)
_CUE_MIXTURE_DENSITY_POLICIES = {
    "active_oscd_cue_mixture",
    "active_oscd_cue_mixture_black_child_prune",
}
RENDER_SUPPORT_MODES = (
    "open_only",
    "open_or_never_open",
    "open_or_never_open_black",
    "open_or_never_open_dc_opacity",
)
DETECTOR_PROBE_SCALING_MODES = ("native", "isotropic_min")
DETECTOR_MODES = (
    "direct_binary",
    "single_candidate_beta",
    "lifespan_gate_beta",
    "learned_dc_agreement_beta",
    "anchored_dc_agreement_beta",
)
FIRST_OPEN_DC_INITIALIZATIONS = ("preserve", "render_one")
OPTIMIZER_SELECTION_POLICIES = ("active_visible", "all_open")
LOSS_REGULARIZATION_MODES = ("global", "local_growth_replay")
CHANGE_COLOR_MODES = ("learned_dc", "lifespan_gate")
CANDIDATE_RENDER_GATES = (
    "none",
    "log_bf_progress",
    "close_only_log_bf_progress",
)
AGREEMENT_CANDIDATE_DC_POLICIES = ("adapt", "freeze")
LEARNED_DC_AGREEMENT_MODES = (
    "learned_dc_agreement_beta",
    "anchored_dc_agreement_beta",
)


def _includes_never_open(mode: str) -> bool:
    return mode in {
        "open_or_never_open",
        "open_or_never_open_black",
        "open_or_never_open_dc_opacity",
    }


def _trains_never_open_appearance(mode: str) -> bool:
    return mode == "open_or_never_open_dc_opacity"


def _uses_black_never_open_occluders(mode: str) -> bool:
    return mode == "open_or_never_open_black"


def _cue_local_support(target: torch.Tensor, scale: float) -> torch.Tensor:
    """Normalize the cached 0..2 O-SCD cue into fixed local support."""

    if not isinstance(target, torch.Tensor):
        raise TypeError("training target must be a tensor")
    if target.ndim != 3 or target.shape[0] != 1:
        raise ValueError("training target must have shape [1,H,W]")
    if not math.isfinite(float(scale)) or float(scale) <= 0.0:
        raise ValueError("local support scale must be finite and positive")
    return (target / float(scale)).clamp(0.0, 1.0)


def _cue_mixture_score(
    delta_change: torch.Tensor, delta_nonchange: torch.Tensor
) -> torch.Tensor:
    """Measure balanced raw-cue responsibility with observation strength.

    The detector's capped pseudo-counts normally sum to at most one.  This
    implementation also remains bounded for uncapped inputs: observation mass
    is capped at one, while the smaller cue-side fraction determines whether a
    Gaussian footprint genuinely covers both change and non-change pixels.
    """

    if not isinstance(delta_change, torch.Tensor) or not isinstance(
        delta_nonchange, torch.Tensor
    ):
        raise TypeError("cue evidence must be tensors")
    if (
        delta_change.shape != delta_nonchange.shape
        or delta_change.device != delta_nonchange.device
        or delta_change.dtype != delta_nonchange.dtype
    ):
        raise ValueError("cue evidence tensors must share shape/device/dtype")
    if not torch.is_floating_point(delta_change):
        raise TypeError("cue evidence must be floating point")
    if not bool(
        torch.isfinite(delta_change).all()
        and torch.isfinite(delta_nonchange).all()
    ):
        raise ValueError("cue evidence must be finite")
    if bool((delta_change < 0).any() or (delta_nonchange < 0).any()):
        raise ValueError("cue evidence must be nonnegative")
    total = delta_change + delta_nonchange
    ratio = delta_change / total.clamp_min(torch.finfo(total.dtype).eps)
    balanced_fraction = 2.0 * torch.minimum(ratio, 1.0 - ratio)
    return total.clamp(0.0, 1.0) * balanced_fraction


@dataclass(frozen=True)
class SignedScoreArtifact:
    """Causal PC1/sign-posterior arrays used only for density control."""

    frame_names: tuple[str, ...]
    pc1_axes: np.ndarray
    epsilon_negative: np.ndarray
    epsilon_positive: np.ndarray
    p_plus_is_new: np.ndarray
    posterior_key: str
    path: Path


def _load_signed_score_artifact(
    path: Path,
    *,
    expected_frame_names: Sequence[str],
    posterior: str,
) -> SignedScoreArtifact:
    """Load and validate precomputed causal sign arrays for this exact stream."""

    if posterior not in {"balanced", "global"}:
        raise ValueError("signed-score posterior must be balanced or global")
    posterior_key = (
        "balanced_p_plus_is_add"
        if posterior == "balanced"
        else "global_p_plus_is_add"
    )
    with np.load(path, allow_pickle=False) as arrays:
        required = {
            "frame_names",
            "pc1_axes",
            "epsilon_negative",
            "epsilon_positive",
            posterior_key,
        }
        missing = sorted(required.difference(arrays.files))
        if missing:
            raise ValueError(f"signed-score artifact missing keys: {missing}")
        frame_names = tuple(str(item) for item in arrays["frame_names"].tolist())
        expected = tuple(str(item) for item in expected_frame_names)
        if len(frame_names) < len(expected) or frame_names[: len(expected)] != expected:
            raise ValueError(
                "signed-score artifact frame_names do not match selected causal prefix"
            )
        frame_names = frame_names[: len(expected)]
        pc1_axes = np.asarray(arrays["pc1_axes"], dtype=np.float32)
        epsilon_negative = np.asarray(arrays["epsilon_negative"], dtype=np.float32)
        epsilon_positive = np.asarray(arrays["epsilon_positive"], dtype=np.float32)
        p_plus_is_new = np.asarray(arrays[posterior_key], dtype=np.float32)
    count = len(frame_names)
    pc1_axes = pc1_axes[:count]
    epsilon_negative = epsilon_negative[:count]
    epsilon_positive = epsilon_positive[:count]
    p_plus_is_new = p_plus_is_new[:count]
    if pc1_axes.shape != (count, 256) or not np.isfinite(pc1_axes).all():
        raise ValueError("signed-score pc1_axes must be finite with shape [frames,256]")
    for name, values in {
        "epsilon_negative": epsilon_negative,
        "epsilon_positive": epsilon_positive,
        posterior_key: p_plus_is_new,
    }.items():
        if values.shape != (count,) or not np.isfinite(values).all():
            raise ValueError(f"signed-score {name} must be finite [frames]")
    if np.any((p_plus_is_new < 0.0) | (p_plus_is_new > 1.0)):
        raise ValueError("signed-score posterior values must lie in [0,1]")
    if np.any(epsilon_negative >= epsilon_positive):
        raise ValueError(
            "signed-score thresholds require epsilon_negative < epsilon_positive"
        )
    return SignedScoreArtifact(
        frame_names=frame_names,
        pc1_axes=pc1_axes,
        epsilon_negative=epsilon_negative,
        epsilon_positive=epsilon_positive,
        p_plus_is_new=p_plus_is_new,
        posterior_key=posterior_key,
        path=path,
    )


@torch.inference_mode()
def _extract_signed_sam_delta(
    view: Any,
    reference_base: Any,
    pipe: Any,
    background: torch.Tensor,
    sam: Any,
) -> torch.Tensor:
    """Return SAM feature delta ``reference - inference`` on the 64×64 grid."""

    from gaussian_renderer import render

    reference = render(view, reference_base, pipe, background)["render"].detach()
    inference = view.original_image[:3].detach()
    pair = torch.stack(
        [
            F.interpolate(
                reference[None],
                (1024, 1024),
                mode="bilinear",
                align_corners=False,
            )[0],
            F.interpolate(
                inference[None],
                (1024, 1024),
                mode="bilinear",
                align_corners=False,
            )[0],
        ]
    ).half()
    embedding = sam.get_image_embeddings(pair)[-1].float()
    return (embedding[0] - embedding[1]).permute(1, 2, 0).reshape(-1, 256).contiguous()


def _signed_score_masks_from_delta(
    delta: torch.Tensor,
    candidate_map: torch.Tensor,
    pc1_axis: np.ndarray,
    *,
    epsilon_negative: float,
    epsilon_positive: float,
    cue_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project causal PC1 signs to full-resolution NEW/REMOVE candidate masks."""

    if delta.shape != (64 * 64, 256):
        raise ValueError("signed-score SAM delta must have shape [4096,256]")
    if candidate_map.ndim != 3 or candidate_map.shape[0] != 1:
        raise ValueError("candidate_map must have shape [1,H,W]")
    axis = torch.as_tensor(pc1_axis, device=delta.device, dtype=delta.dtype).flatten()
    if axis.shape != (256,):
        raise ValueError("pc1_axis must have shape [256]")
    score64 = (delta @ axis).reshape(64, 64)
    cue64 = F.interpolate(
        candidate_map.detach()[None].to(device=delta.device, dtype=delta.dtype),
        (64, 64),
        mode="area",
    )[0, 0]
    cue_active = cue64 >= float(cue_threshold)
    plus64 = cue_active & (score64 > float(epsilon_positive))
    minus64 = cue_active & (score64 < float(epsilon_negative))
    size = tuple(int(v) for v in candidate_map.shape[-2:])
    plus = F.interpolate(
        plus64.to(dtype=delta.dtype)[None, None], size, mode="nearest"
    )[0]
    minus = F.interpolate(
        minus64.to(dtype=delta.dtype)[None, None], size, mode="nearest"
    )[0]
    return plus, minus


def _current_episode_start(model: Any, timestamp: int) -> torch.Tensor:
    """Return the start frame of the currently OPEN lifespan for each row."""

    slots = model.current_state_index.detach()
    rows = torch.arange(slots.shape[0], device=slots.device, dtype=torch.long)
    starts = torch.full(
        (slots.shape[0],),
        float(timestamp),
        device=model.state_start.device,
        dtype=model.state_start.dtype,
    )
    active = slots >= 0
    starts[active] = model.state_start[rows[active], slots[active]]
    return starts


@torch.no_grad()
def _density_source_rows(
    view: Any,
    xyz: torch.Tensor,
    stable_id: torch.Tensor,
    generation: torch.Tensor,
    masks: Any,
    *,
    selection_signal: torch.Tensor,
    selection_signal_name: str,
    signed: Any | None = None,
) -> list[dict[str, Any]]:
    """Build posthoc density source/projection rows without reading GT masks."""

    selected = masks.clone | masks.split | masks.signed_score_prune
    indices = torch.nonzero(selected, as_tuple=False).flatten()
    if not int(indices.numel()):
        return []
    signal = selection_signal.detach().to(device=xyz.device, dtype=xyz.dtype).flatten()
    if signal.shape != (xyz.shape[0],):
        raise ValueError("selection_signal must align with pre-mutation rows")
    homogeneous = torch.cat(
        (
            xyz.detach(),
            torch.ones((xyz.shape[0], 1), device=xyz.device, dtype=xyz.dtype),
        ),
        dim=1,
    )
    camera = homogeneous @ view.world_view_transform
    depth = camera[:, 2]
    fx = float(view.image_width) / (2.0 * math.tan(float(view.FoVx) * 0.5))
    fy = float(view.image_height) / (2.0 * math.tan(float(view.FoVy) * 0.5))
    x = fx * camera[:, 0] / depth.clamp_min(1.0e-6) + float(view.image_width) * 0.5
    y = fy * camera[:, 1] / depth.clamp_min(1.0e-6) + float(view.image_height) * 0.5
    rows: list[dict[str, Any]] = []
    for row in indices.detach().cpu().tolist():
        action = "clone" if bool(masks.clone[row]) else "split"
        if bool(masks.signed_score_prune[row]):
            action = "negative_prune"
        entry = {
            "row": int(row),
            "stable_id": int(stable_id[row].detach().cpu().item()),
            "generation": int(generation[row].detach().cpu().item()),
            "action": action,
            "selection_signal": selection_signal_name,
            "selection_value": float(signal[row].detach().cpu().item()),
            "selected_for_mutation": True,
            "x": float(x[row].detach().cpu().item()),
            "y": float(y[row].detach().cpu().item()),
            "depth": float(depth[row].detach().cpu().item()),
            "inside_image": bool(
                (
                    (depth[row] > 0.0)
                    & (x[row] >= 0.0)
                    & (x[row] < float(view.image_width))
                    & (y[row] >= 0.0)
                    & (y[row] < float(view.image_height))
                ).detach().cpu().item()
            ),
            "image_width": int(view.image_width),
            "image_height": int(view.image_height),
        }
        if signed is not None:
            entry.update(
                {
                    "score": float(signed.score[row].detach().cpu().item()),
                    "sign_support": float(signed.sign_support[row].detach().cpu().item()),
                    "age": float(signed.age[row].detach().cpu().item()),
                    "age_weight": float(signed.age_weight[row].detach().cpu().item()),
                }
            )
        rows.append(entry)
    return rows


@torch.no_grad()
def _signed_negative_suppression_rows(
    view: Any,
    xyz: torch.Tensor,
    stable_id: torch.Tensor,
    generation: torch.Tensor,
    signed: Any,
    *,
    threshold: float,
    max_rows: int | None,
    already_logged: torch.Tensor,
) -> list[dict[str, Any]]:
    """Log strongest negative signed rows that are intentionally not densified."""

    candidates = (signed.score < -float(threshold)) & ~already_logged
    indices = torch.nonzero(candidates, as_tuple=False).flatten()
    if not int(indices.numel()):
        return []
    order_values = signed.score[indices]
    order = torch.argsort(order_values, descending=False, stable=True)
    indices = indices[order]
    if max_rows is not None:
        indices = indices[: int(max_rows)]
    if not int(indices.numel()):
        return []
    homogeneous = torch.cat(
        (
            xyz.detach(),
            torch.ones((xyz.shape[0], 1), device=xyz.device, dtype=xyz.dtype),
        ),
        dim=1,
    )
    camera = homogeneous @ view.world_view_transform
    depth = camera[:, 2]
    fx = float(view.image_width) / (2.0 * math.tan(float(view.FoVx) * 0.5))
    fy = float(view.image_height) / (2.0 * math.tan(float(view.FoVy) * 0.5))
    x = fx * camera[:, 0] / depth.clamp_min(1.0e-6) + float(view.image_width) * 0.5
    y = fy * camera[:, 1] / depth.clamp_min(1.0e-6) + float(view.image_height) * 0.5
    rows: list[dict[str, Any]] = []
    for row in indices.detach().cpu().tolist():
        rows.append(
            {
                "row": int(row),
                "stable_id": int(stable_id[row].detach().cpu().item()),
                "generation": int(generation[row].detach().cpu().item()),
                "action": "negative_suppress",
                "selection_signal": "signed_lifespan_score",
                "selection_value": float(signed.score[row].detach().cpu().item()),
                "selected_for_mutation": False,
                "score": float(signed.score[row].detach().cpu().item()),
                "sign_support": float(signed.sign_support[row].detach().cpu().item()),
                "age": float(signed.age[row].detach().cpu().item()),
                "age_weight": float(signed.age_weight[row].detach().cpu().item()),
                "x": float(x[row].detach().cpu().item()),
                "y": float(y[row].detach().cpu().item()),
                "depth": float(depth[row].detach().cpu().item()),
                "inside_image": bool(
                    (
                        (depth[row] > 0.0)
                        & (x[row] >= 0.0)
                        & (x[row] < float(view.image_width))
                        & (y[row] >= 0.0)
                        & (y[row] < float(view.image_height))
                    ).detach().cpu().item()
                ),
                "image_width": int(view.image_width),
                "image_height": int(view.image_height),
            }
        )
    return rows


def _render_change_probability(rendered_change: torch.Tensor) -> torch.Tensor:
    """Return the exact sigmoid probability used by the SSF loss."""

    if not isinstance(rendered_change, torch.Tensor):
        raise TypeError("rendered change must be a tensor")
    if rendered_change.ndim != 3 or rendered_change.shape[0] != 3:
        raise ValueError("rendered change must have shape [3,H,W]")
    return torch.sigmoid(rendered_change.mean(dim=0, keepdim=True))


def _positive_growth_map(
    before_probability: torch.Tensor, after_probability: torch.Tensor
) -> torch.Tensor:
    """Detach the positive image-space mask growth made by one frame update."""

    if before_probability.shape != after_probability.shape:
        raise ValueError("before/after probabilities must have the same shape")
    if before_probability.ndim != 3 or before_probability.shape[0] != 1:
        raise ValueError("growth probabilities must have shape [1,H,W]")
    if before_probability.device != after_probability.device:
        raise ValueError("before/after probabilities must share a device")
    return (after_probability - before_probability).clamp_min(0.0).detach()


def _lifespan_gate_semantic_colors(
    active: torch.Tensor, *, dtype: torch.dtype
) -> torch.Tensor:
    """Return precomputed RGB 1 for OPEN rows and RGB 0 otherwise."""

    if active.ndim != 1 or active.dtype != torch.bool:
        raise ValueError("active must be a boolean [N] tensor")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError("semantic color dtype must be floating")
    return active.to(dtype=dtype).unsqueeze(1).expand(-1, 3).contiguous()


def _candidate_transition_progress(
    candidate_active: torch.Tensor,
    log_bayes_factor: torch.Tensor,
    *,
    bayes_factor_threshold: float,
) -> torch.Tensor:
    """Map a live candidate's log Bayes factor to commit-normalized progress.

    The value is deliberately a rendering progress variable rather than a
    posterior probability.  It is zero unless the reset candidate is live,
    reaches one at the configured hard-commit threshold, and is detached from
    both the detector and representation optimizer graphs.
    """

    if not isinstance(candidate_active, torch.Tensor) or not isinstance(
        log_bayes_factor, torch.Tensor
    ):
        raise TypeError("candidate state and log Bayes factor must be tensors")
    if candidate_active.dtype != torch.bool or candidate_active.ndim != 1:
        raise ValueError("candidate_active must be a boolean [N] tensor")
    if (
        log_bayes_factor.ndim != 1
        or log_bayes_factor.shape != candidate_active.shape
        or log_bayes_factor.device != candidate_active.device
        or not torch.is_floating_point(log_bayes_factor)
    ):
        raise ValueError(
            "log_bayes_factor must be a floating [N] tensor aligned with candidates"
        )
    if not bool(torch.isfinite(log_bayes_factor).all()):
        raise ValueError("log_bayes_factor must be finite")
    if (
        not math.isfinite(float(bayes_factor_threshold))
        or float(bayes_factor_threshold) <= 1.0
    ):
        raise ValueError("bayes_factor_threshold must be finite and greater than one")

    progress = torch.zeros_like(log_bayes_factor)
    progress[candidate_active] = (
        log_bayes_factor[candidate_active]
        / math.log(float(bayes_factor_threshold))
    ).clamp(0.0, 1.0)
    return progress.detach()


def _soft_lifespan_change_weight(
    current_active: torch.Tensor,
    transition_progress: torch.Tensor,
) -> torch.Tensor:
    """Cross-fade the committed binary state toward its reset hypothesis."""

    if not isinstance(current_active, torch.Tensor) or not isinstance(
        transition_progress, torch.Tensor
    ):
        raise TypeError("lifespan state and transition progress must be tensors")
    if current_active.dtype != torch.bool or current_active.ndim != 1:
        raise ValueError("current_active must be a boolean [N] tensor")
    if (
        transition_progress.ndim != 1
        or transition_progress.shape != current_active.shape
        or transition_progress.device != current_active.device
        or not torch.is_floating_point(transition_progress)
    ):
        raise ValueError(
            "transition_progress must be a floating [N] tensor aligned with state"
        )
    if not bool(torch.isfinite(transition_progress).all()) or bool(
        ((transition_progress < 0.0) | (transition_progress > 1.0)).any()
    ):
        raise ValueError("transition_progress must be finite and lie in [0,1]")

    active = current_active.to(dtype=transition_progress.dtype)
    return (
        (1.0 - transition_progress) * active
        + transition_progress * (1.0 - active)
    ).detach()


def _close_only_lifespan_change_weight(
    current_active: torch.Tensor,
    transition_progress: torch.Tensor,
) -> torch.Tensor:
    """Fade only a live CLOSE candidate; never preview an OPEN candidate."""

    if not isinstance(current_active, torch.Tensor) or not isinstance(
        transition_progress, torch.Tensor
    ):
        raise TypeError("lifespan state and transition progress must be tensors")
    if current_active.dtype != torch.bool or current_active.ndim != 1:
        raise ValueError("current_active must be a boolean [N] tensor")
    if (
        transition_progress.ndim != 1
        or transition_progress.shape != current_active.shape
        or transition_progress.device != current_active.device
        or not torch.is_floating_point(transition_progress)
    ):
        raise ValueError(
            "transition_progress must be a floating [N] tensor aligned with state"
        )
    if not bool(torch.isfinite(transition_progress).all()) or bool(
        ((transition_progress < 0.0) | (transition_progress > 1.0)).any()
    ):
        raise ValueError("transition_progress must be finite and lie in [0,1]")

    active = current_active.to(dtype=transition_progress.dtype)
    return (active * (1.0 - transition_progress)).detach()


def _single_candidate_reset_change_probability(tracker: Any) -> torch.Tensor:
    """Return the fresh-run change probability for each live reset candidate.

    ``SingleCandidateBetaFilter.change_probability`` describes the frozen
    committed run while a candidate is live. CLOSE direction must therefore
    be read from the candidate block itself: fresh prior plus the candidate's
    accumulated raw cue pseudo-counts.
    """

    required = (
        "candidate_delta_a",
        "candidate_delta_b",
        "candidate_active",
        "config",
    )
    for name in required:
        if not hasattr(tracker, name):
            raise TypeError(f"single-candidate tracker must expose {name}")
    delta_a = tracker.candidate_delta_a
    delta_b = tracker.candidate_delta_b
    candidate_active = tracker.candidate_active
    if not isinstance(delta_a, torch.Tensor) or not isinstance(delta_b, torch.Tensor):
        raise TypeError("candidate pseudo-counts must be tensors")
    if (
        delta_a.ndim != 1
        or delta_b.shape != delta_a.shape
        or delta_b.device != delta_a.device
        or delta_b.dtype != delta_a.dtype
        or not torch.is_floating_point(delta_a)
    ):
        raise ValueError("candidate pseudo-counts must be aligned floating [N] tensors")
    if (
        not isinstance(candidate_active, torch.Tensor)
        or candidate_active.dtype != torch.bool
        or candidate_active.shape != delta_a.shape
        or candidate_active.device != delta_a.device
    ):
        raise ValueError("candidate_active must be a boolean [N] tensor")
    if not bool(torch.isfinite(delta_a).all()) or not bool(torch.isfinite(delta_b).all()):
        raise ValueError("candidate pseudo-counts must be finite")
    if bool((delta_a < 0.0).any()) or bool((delta_b < 0.0).any()):
        raise ValueError("candidate pseudo-counts must be nonnegative")

    prior_a = float(tracker.config.prior_a)
    prior_b = float(tracker.config.prior_b)
    if (
        not math.isfinite(prior_a)
        or not math.isfinite(prior_b)
        or prior_a <= 0.0
        or prior_b <= 0.0
    ):
        raise ValueError("single-candidate priors must be finite and positive")
    candidate_a = delta_a.detach() + prior_a
    candidate_b = delta_b.detach() + prior_b
    return (candidate_a / (candidate_a + candidate_b)).detach()


def _single_candidate_close_candidate_mask(
    tracker: Any,
    current_active: torch.Tensor,
    *,
    close_probability: float,
) -> torch.Tensor:
    """Select live raw-cue reset candidates whose new run implies CLOSE."""

    if (
        not isinstance(current_active, torch.Tensor)
        or current_active.dtype != torch.bool
        or current_active.ndim != 1
    ):
        raise ValueError("current_active must be a boolean [N] tensor")
    threshold = float(close_probability)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("close_probability must be finite and lie in [0,1]")
    probability = _single_candidate_reset_change_probability(tracker)
    if (
        probability.shape != current_active.shape
        or probability.device != current_active.device
    ):
        raise ValueError("tracker state and current_active must be aligned")
    return (
        tracker.candidate_active
        & current_active
        & (probability <= threshold)
    ).detach()


def _candidate_render_gate_state(
    mode: str,
    current_active: torch.Tensor,
    candidate_active: torch.Tensor,
    log_bayes_factor: torch.Tensor,
    *,
    bayes_factor_threshold: float,
    close_candidate_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return progress, output weight, and rows actually softened by ``mode``."""

    if mode not in CANDIDATE_RENDER_GATES or mode == "none":
        raise ValueError("candidate render gate state requires an enabled mode")
    progress = _candidate_transition_progress(
        candidate_active,
        log_bayes_factor,
        bayes_factor_threshold=bayes_factor_threshold,
    )
    if mode == "log_bf_progress":
        if close_candidate_mask is not None:
            raise ValueError("close_candidate_mask is only valid for close-only gating")
        weight = _soft_lifespan_change_weight(current_active, progress)
        gated_rows = candidate_active
    else:
        if close_candidate_mask is None:
            gated_rows = candidate_active & current_active
        else:
            if (
                not isinstance(close_candidate_mask, torch.Tensor)
                or close_candidate_mask.dtype != torch.bool
                or close_candidate_mask.shape != current_active.shape
                or close_candidate_mask.device != current_active.device
            ):
                raise ValueError(
                    "close_candidate_mask must be a boolean [N] tensor aligned with state"
                )
            if bool((close_candidate_mask & ~(candidate_active & current_active)).any()):
                raise ValueError(
                    "close_candidate_mask must select only live candidates on OPEN rows"
                )
            gated_rows = close_candidate_mask
        directional_progress = progress * gated_rows.to(dtype=progress.dtype)
        weight = _close_only_lifespan_change_weight(
            current_active, directional_progress
        )
    return progress, weight, gated_rows


def _soft_lifespan_gate_overrides(
    base_opacity: torch.Tensor,
    never_open: torch.Tensor,
    soft_change_weight: torch.Tensor,
    *,
    include_never_open_occluders: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build semantic color and opacity for output-only soft transitions.

    A stable NEVER_OPEN row remains a black, full-opacity occluder when that
    renderer ablation is enabled.  Once the row has a positive OPEN-candidate
    weight it becomes a white change row whose opacity is scaled by that
    weight.  CLOSED non-candidates remain absent.
    """

    if not isinstance(base_opacity, torch.Tensor):
        raise TypeError("base_opacity must be a tensor")
    if (
        base_opacity.ndim != 2
        or base_opacity.shape[1] != 1
        or not torch.is_floating_point(base_opacity)
    ):
        raise ValueError("base_opacity must be a floating [N,1] tensor")
    if never_open.dtype != torch.bool or never_open.shape != (base_opacity.shape[0],):
        raise ValueError("never_open must be a boolean [N] tensor")
    if (
        soft_change_weight.shape != (base_opacity.shape[0],)
        or soft_change_weight.device != base_opacity.device
        or soft_change_weight.dtype != base_opacity.dtype
    ):
        raise ValueError("soft_change_weight must match base opacity rows/device/dtype")
    if never_open.device != base_opacity.device:
        raise ValueError("never_open must share the base opacity device")
    if not bool(torch.isfinite(base_opacity).all()) or bool(
        ((base_opacity < 0.0) | (base_opacity > 1.0)).any()
    ):
        raise ValueError("base_opacity must be finite and lie in [0,1]")
    if not bool(torch.isfinite(soft_change_weight).all()) or bool(
        ((soft_change_weight < 0.0) | (soft_change_weight > 1.0)).any()
    ):
        raise ValueError("soft_change_weight must be finite and lie in [0,1]")

    change_support = soft_change_weight > 0.0
    occluder_only = (
        never_open & ~change_support
        if include_never_open_occluders
        else torch.zeros_like(never_open)
    )
    opacity_multiplier = soft_change_weight + occluder_only.to(
        dtype=soft_change_weight.dtype
    )
    colors = change_support.to(dtype=base_opacity.dtype).unsqueeze(1).expand(-1, 3)
    opacity = base_opacity.detach() * opacity_multiplier.detach().unsqueeze(1)
    return colors.contiguous(), opacity.contiguous()


def _soft_learned_dc_opacity(
    render_opacity: torch.Tensor,
    active: torch.Tensor,
    soft_change_weight: torch.Tensor,
) -> torch.Tensor:
    """Fade learned-DC opacity only on OPEN rows selected by an output gate.

    NEVER_OPEN rows retain their full black-occluder opacity and CLOSED rows
    remain absent through ``render_opacity``'s existing hard lifespan mask.
    """

    if (
        not isinstance(render_opacity, torch.Tensor)
        or render_opacity.ndim != 2
        or render_opacity.shape[1] != 1
        or not torch.is_floating_point(render_opacity)
    ):
        raise ValueError("render_opacity must be a floating [N,1] tensor")
    row_count = render_opacity.shape[0]
    if (
        not isinstance(active, torch.Tensor)
        or active.dtype != torch.bool
        or active.shape != (row_count,)
        or active.device != render_opacity.device
    ):
        raise ValueError("active must be a boolean [N] tensor aligned with opacity")
    if (
        not isinstance(soft_change_weight, torch.Tensor)
        or soft_change_weight.shape != (row_count,)
        or soft_change_weight.device != render_opacity.device
        or soft_change_weight.dtype != render_opacity.dtype
    ):
        raise ValueError("soft_change_weight must match opacity rows/device/dtype")
    if not bool(torch.isfinite(soft_change_weight).all()) or bool(
        ((soft_change_weight < 0.0) | (soft_change_weight > 1.0)).any()
    ):
        raise ValueError("soft_change_weight must be finite and lie in [0,1]")
    multiplier = torch.where(
        active,
        soft_change_weight.detach(),
        torch.ones_like(soft_change_weight),
    )
    return (render_opacity.detach() * multiplier.unsqueeze(1)).contiguous()


def _active_change_color_value(
    model: Any,
    rows: torch.Tensor,
    change_color_mode: str,
) -> torch.Tensor:
    """Read the effective per-row color used by the selected renderer."""

    if change_color_mode == "learned_dc":
        return _intrinsic_dc_render_value(model, rows)
    if change_color_mode != "lifespan_gate":
        raise ValueError(f"unknown change color mode: {change_color_mode}")
    return model.change_dc.new_ones((rows.numel(),))


def _render_dynamic_change(
    viewpoint_camera: Any,
    model: Any,
    pipe: Any,
    background: torch.Tensor,
    *,
    timestamp: float,
    include_never_open_occluders: bool,
    train_never_open_dc_opacity: bool,
    black_never_open_occluders: bool,
    change_color_mode: str,
    soft_change_weight: torch.Tensor | None = None,
):
    """Render learned DC or a hard/soft lifespan-gate semantic change color."""

    from gaussian_renderer import render_change, render_change_temporal

    if change_color_mode == "learned_dc":
        if soft_change_weight is None:
            return render_change_temporal(
                viewpoint_camera,
                model,
                pipe,
                background,
                timestamp=timestamp,
                include_never_open_occluders=include_never_open_occluders,
                train_never_open_dc_opacity=train_never_open_dc_opacity,
                black_never_open_occluders=black_never_open_occluders,
            )
        if include_never_open_occluders:
            attributes = model.get_open_or_never_open_render_attributes(
                timestamp,
                train_never_open_dc_opacity=train_never_open_dc_opacity,
                black_never_open=black_never_open_occluders,
            )
        else:
            attributes = model.get_active_render_attributes(timestamp)
        opacity = _soft_learned_dc_opacity(
            attributes["opacity"], attributes["active"], soft_change_weight
        )
        return render_change(
            viewpoint_camera,
            model.base,
            pipe,
            background,
            override_dc=attributes["dc"],
            override_features_rest=attributes.get("features_rest"),
            override_opacity=opacity,
            override_xyz=attributes["xyz"],
            override_scaling=attributes["scaling"],
            override_rotation=attributes["rotation"],
        )
    if change_color_mode != "lifespan_gate":
        raise ValueError(f"unknown change color mode: {change_color_mode}")
    if train_never_open_dc_opacity:
        raise ValueError("lifespan-gate color requires frozen NEVER_OPEN appearance")

    if include_never_open_occluders:
        attributes = model.get_open_or_never_open_render_attributes(
            timestamp, train_never_open_dc_opacity=False
        )
    else:
        attributes = model.get_active_render_attributes(timestamp)
    if soft_change_weight is None:
        colors = _lifespan_gate_semantic_colors(
            attributes["active"], dtype=model.xyz.dtype
        )
        opacity = attributes["opacity"]
    else:
        colors, opacity = _soft_lifespan_gate_overrides(
            model.base.get_opacity.detach(),
            attributes["never_open"],
            soft_change_weight,
            include_never_open_occluders=include_never_open_occluders,
        )
    return render_change(
        viewpoint_camera,
        model.base,
        pipe,
        background,
        override_color=colors,
        override_opacity=opacity,
        override_xyz=attributes["xyz"],
        override_scaling=attributes["scaling"],
        override_rotation=attributes["rotation"],
    )


def _optimizer_row_masks(
    mode: str,
    *,
    active_rows: torch.Tensor | None = None,
    active_visible: torch.Tensor,
    never_open_visible: torch.Tensor,
    selection_policy: str = "active_visible",
) -> torch.Tensor | dict[str, torch.Tensor]:
    if selection_policy not in OPTIMIZER_SELECTION_POLICIES:
        raise ValueError(f"unknown optimizer selection policy: {selection_policy}")
    if selection_policy == "all_open":
        if _trains_never_open_appearance(mode):
            raise ValueError(
                "all_open selection requires NEVER_OPEN parameters to remain frozen"
            )
        if active_rows is None:
            raise ValueError("active_rows are required for all_open selection")
        return active_rows
    if not _trains_never_open_appearance(mode):
        return active_visible
    appearance = active_visible | never_open_visible
    return {
        name: appearance if name in {"dc", "opacity"} else active_visible
        for name in DIRECT_PARAMETER_NAMES
    }


def _candidate_dc_optimizer_masks(
    optimizer_masks: torch.Tensor | dict[str, torch.Tensor],
    candidate_active: torch.Tensor,
    *,
    policy: str,
) -> torch.Tensor | dict[str, torch.Tensor]:
    """Optionally freeze DC rows while a mismatch candidate is live."""

    if policy not in AGREEMENT_CANDIDATE_DC_POLICIES:
        raise ValueError(f"unknown agreement candidate DC policy: {policy}")
    if candidate_active.ndim != 1 or candidate_active.dtype != torch.bool:
        raise ValueError("candidate_active must be a boolean [N] tensor")
    if policy == "adapt":
        return optimizer_masks
    if isinstance(optimizer_masks, torch.Tensor):
        if optimizer_masks.shape != candidate_active.shape:
            raise ValueError("candidate mask must match optimizer rows")
        return {
            name: (
                optimizer_masks & ~candidate_active
                if name == "dc"
                else optimizer_masks
            )
            for name in DIRECT_PARAMETER_NAMES
        }
    if set(optimizer_masks) != set(DIRECT_PARAMETER_NAMES):
        raise ValueError("parameter-specific optimizer masks are incomplete")
    if optimizer_masks["dc"].shape != candidate_active.shape:
        raise ValueError("candidate mask must match optimizer rows")
    return {
        name: (
            mask & ~candidate_active if name == "dc" else mask
        )
        for name, mask in optimizer_masks.items()
    }


def _zero_frozen_dc_gradients(
    model: Any, candidate_active: torch.Tensor, *, policy: str
) -> float:
    """Suppress candidate-row DC gradients and return their raw max magnitude."""

    if policy not in AGREEMENT_CANDIDATE_DC_POLICIES:
        raise ValueError(f"unknown agreement candidate DC policy: {policy}")
    gradient = model.change_dc.grad
    if policy == "adapt" or gradient is None or not bool(candidate_active.any()):
        return 0.0
    if candidate_active.shape != (gradient.shape[0],):
        raise ValueError("candidate mask must match DC gradient rows")
    maximum = float(gradient[candidate_active].detach().abs().max().item())
    gradient[candidate_active] = 0.0
    return maximum


def _intrinsic_dc_render_value(model: Any, rows: torch.Tensor) -> torch.Tensor:
    """Return each row's mean RGB value implied by its degree-zero SH DC."""

    if rows.ndim != 1 or rows.dtype != torch.long:
        raise ValueError("rows must be a 1D long tensor")
    if rows.numel() == 0:
        return model.change_dc.new_empty((0,))
    return SH2RGB(model.change_dc.detach()[rows]).mean(dim=(1, 2))


def _learned_dc_change_magnitude(model: Any, rows: torch.Tensor) -> torch.Tensor:
    """Map O-SCD raw-zero DC to semantic 0 and rendered white DC to 1."""

    intrinsic = _intrinsic_dc_render_value(model, rows)
    return ((intrinsic - 0.5) / 0.5).clamp(0.0, 1.0)


@torch.no_grad()
def _initialize_first_open_dc(
    model: Any,
    optimizer: Any,
    events: Sequence[LifecycleEvent],
    stable_ids: torch.Tensor,
    mode: str,
) -> dict[str, Any]:
    """Optionally set first-OPEN rows to intrinsic rendered change value one.

    Only ``OPEN`` events allocating slot zero qualify.  ``REOPEN`` therefore
    preserves the learned persistent DC and its Adam history.  In
    ``render_one`` mode, DC's row-local Adam state is reset so momentum learned
    while the row was NEVER_OPEN cannot push against the explicit jump; every
    non-DC parameter and every unselected row remains untouched.
    """

    if mode not in FIRST_OPEN_DC_INITIALIZATIONS:
        raise ValueError(f"unknown first-OPEN DC initialization: {mode}")
    rows = torch.tensor(
        [
            int(event.gaussian_index)
            for event in events
            if event.action == "OPEN" and int(event.new_current_slot) == 0
        ],
        device=model.change_dc.device,
        dtype=torch.long,
    )
    if rows.numel():
        rows = rows.unique(sorted=True)
    if stable_ids.ndim != 1 or stable_ids.shape[0] != model.change_dc.shape[0]:
        raise ValueError("stable_ids must match the current Gaussian topology")
    selected_stable_ids = stable_ids.to(device=rows.device)[rows].detach().clone()
    before = _intrinsic_dc_render_value(model, rows)
    raw_before = model.change_dc.detach()[rows].clone()

    reset_count = 0
    if mode == "render_one" and rows.numel():
        target = RGB2SH(torch.ones_like(model.change_dc.detach()[rows]))
        model.change_dc[rows] = target
        row_mask = torch.zeros(
            model.change_dc.shape[0], device=rows.device, dtype=torch.bool
        )
        row_mask[rows] = True
        optimizer.reset_state_rows(row_mask, names=("dc",))
        reset_count = int(rows.numel())

    immediate = _intrinsic_dc_render_value(model, rows)
    raw_jump = (
        (model.change_dc.detach()[rows] - raw_before)
        .abs()
        .reshape(rows.numel(), -1)
        .mean(dim=1)
        if rows.numel()
        else model.change_dc.new_empty((0,))
    )
    return {
        "rows": rows,
        "stable_ids": selected_stable_ids,
        "before": before,
        "immediate": immediate,
        "raw_abs_jump": raw_jump,
        "adam_reset_count": reset_count,
    }


def _surviving_intrinsic_dc_render_value(
    topology: Any, stable_ids: torch.Tensor
) -> tuple[torch.Tensor, int]:
    """Read post-mutation DC values for source IDs that still exist."""

    rows, surviving = _surviving_row_positions(topology, stable_ids)
    return _intrinsic_dc_render_value(topology.model, rows), int(surviving.sum().item())


def _surviving_row_positions(
    topology: Any, stable_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map sorted stable IDs to current rows and return a source survival mask."""

    if stable_ids.ndim != 1:
        raise ValueError("stable_ids must be one-dimensional")
    if stable_ids.numel() == 0:
        return stable_ids.new_empty((0,), dtype=torch.long), stable_ids.new_zeros(
            (0,), dtype=torch.bool
        )
    current_ids = topology.stable_id
    if current_ids.numel() > 1 and bool((current_ids[1:] <= current_ids[:-1]).any()):
        raise RuntimeError("dynamic stable IDs must stay strictly increasing")
    positions = torch.searchsorted(current_ids, stable_ids.to(current_ids.device))
    in_bounds = positions < current_ids.numel()
    safe_positions = positions.clamp_max(max(int(current_ids.numel()) - 1, 0))
    surviving = in_bounds & (
        current_ids.index_select(0, safe_positions)
        == stable_ids.to(current_ids.device)
    )
    return positions[surviving], surviving


def first_open_close_latency_diagnostics(
    events: Sequence[LifecycleEvent],
) -> dict[str, Any]:
    """Measure how quickly explicit first OPEN events receive their first CLOSE."""

    first_open: dict[int, int] = {}
    first_close: dict[int, int] = {}
    for event in events:
        stable_id = int(event.gaussian_index)
        timestamp = int(event.decision_timestamp)
        if (
            event.action == "OPEN"
            and int(event.new_current_slot) == 0
            and stable_id not in first_open
        ):
            first_open[stable_id] = timestamp
        elif (
            event.action == "CLOSE"
            and stable_id in first_open
            and stable_id not in first_close
            and timestamp >= first_open[stable_id]
        ):
            first_close[stable_id] = timestamp
    latencies = torch.tensor(
        [first_close[key] - opened for key, opened in first_open.items() if key in first_close],
        dtype=torch.float64,
    )
    return {
        "first_open_count": int(len(first_open)),
        "subsequently_closed_count": int(latencies.numel()),
        "not_closed_by_end_count": int(len(first_open) - latencies.numel()),
        "closed_same_frame_count": int((latencies == 0).sum().item()),
        "closed_within_1_frame_count": int((latencies <= 1).sum().item()),
        "closed_within_3_frames_count": int((latencies <= 3).sum().item()),
        "closed_within_5_frames_count": int((latencies <= 5).sum().item()),
        "close_latency_frames": quantile_summary(latencies),
    }


def _gradient_audit(
    model: Any,
    masks: torch.Tensor | dict[str, torch.Tensor],
) -> dict[str, float | int]:
    violations = 0
    maximum = 0.0
    for name, parameter in model.persistent_parameter_items():
        if parameter.grad is None:
            continue
        selected = masks if isinstance(masks, torch.Tensor) else masks[name]
        gradient = parameter.grad.detach().reshape(parameter.shape[0], -1)
        outside = gradient[~selected]
        nonzero = int(torch.count_nonzero(outside).item())
        violations += nonzero
        if nonzero:
            maximum = max(maximum, float(outside.abs().max().item()))
    return {"count": int(violations), "max_abs": float(maximum)}


def _render_to_rgb_u8(rendered: torch.Tensor) -> np.ndarray:
    """Convert a renderer ``[3,H,W]`` result to a CPU RGB image."""

    if rendered.ndim != 3 or rendered.shape[0] != 3:
        raise ValueError("rendered image must have shape [3,H,W]")
    return (
        rendered.detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )


@torch.no_grad()
def _render_current_lifecycle_events(
    view: Any,
    model: Any,
    pipe: Any,
    background: torch.Tensor,
    events: Sequence[LifecycleEvent],
) -> np.ndarray:
    """Render current lifespan state plus exact-timestamp event highlights.

    Dynamic rows cannot be reconstructed later from stable IDs alone because a
    split child has no immutable-reference row.  Capturing the event footprint
    here preserves the row geometry that actually existed at the decision.
    Existing OPEN/CLOSED rows use half-bright green/red; current OPEN/CLOSE
    events override them with full-bright green/red.  NEVER_OPEN rows are hidden.
    """

    from gaussian_renderer import render_change
    from experiments.visualize_lifespan_cue_events import event_probe_tensors

    base = model.base
    count = int(base.get_xyz.shape[0])
    device = base.get_xyz.device
    dtype = base._features_dc.dtype
    open_rows: list[int] = []
    close_rows: list[int] = []
    for event in events:
        row = int(event.gaussian_index)
        if row < 0 or row >= count:
            raise IndexError(f"lifecycle event row {row} is outside topology {count}")
        if event.action == "OPEN":
            open_rows.append(row)
        elif event.action == "CLOSE":
            close_rows.append(row)
    open_state = model.current_state_index >= 0
    closed_state = (model.num_states > 0) & ~open_state
    colors, selected = event_probe_tensors(
        count,
        open_rows,
        close_rows,
        device=device,
        dtype=dtype,
        open_state_rows=torch.nonzero(open_state, as_tuple=False).flatten(),
        closed_state_rows=torch.nonzero(closed_state, as_tuple=False).flatten(),
    )
    if not bool(selected.any()):
        return np.zeros(
            (int(view.image_height), int(view.image_width), 3), dtype=np.uint8
        )
    package = render_change(
        view,
        base,
        pipe,
        background,
        override_color=colors,
        override_opacity=base.get_opacity.detach() * selected[:, None],
        override_xyz=base.get_xyz.detach(),
        override_scaling=base.get_scaling.detach(),
        override_rotation=base.get_rotation.detach(),
        clamp_output=True,
    )
    return _render_to_rgb_u8(package["render"])


def _save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(path)


def _save_binary_mask(path: Path, mask: np.ndarray) -> None:
    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2:
        raise ValueError("binary mask must have shape [H,W]")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8) * 255, mode="L").save(path)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("value must be finite and nonnegative")
    return parsed


def probability(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be in [0,1]")
    return parsed


def bayes_factor_greater_than_one(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be finite and greater than one")
    return parsed


def _config(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        bayes_cue_mode="binary",
        bayes_cue_threshold=float(args.bayes_cue_threshold),
        bayes_cue_scale=1.0,
        evidence_count_mode="capped",
        evidence_mass_saturation=float(args.evidence_mass_saturation),
        min_evidence_mass=float(args.min_evidence_mass),
        state_emission_reliability=float(args.state_emission_reliability),
        inactive_to_active_prior=float(args.inactive_to_active_prior),
        active_to_inactive_prior=float(args.active_to_inactive_prior),
        initial_active_probability=float(args.initial_active_probability),
        filter_chunk_size=int(args.filter_chunk_size),
        lifecycle_controller="view_consistent",
        open_probability=0.6,
        close_probability=0.4,
        transition_confirmation_views=int(args.transition_confirmation_views),
        min_transition_bayes_factor=float(args.min_transition_bayes_factor),
        min_transition_evidence_strength=float(args.min_transition_evidence_strength),
        max_states=int(args.max_states),
        thaw_parameters=("dc",),
        updates_per_frame=int(args.updates_per_frame),
        detector_only=False,
        evaluation_threshold=float(args.evaluation_threshold),
        seed=int(args.seed),
    )


def _optimizer_row_snapshot(
    model: Any, optimizer: Any, rows: torch.Tensor
) -> dict[str, dict[str, torch.Tensor]]:
    snapshot: dict[str, dict[str, torch.Tensor]] = {}
    for name, parameter in model.persistent_parameter_items():
        values = {"parameter": parameter.detach()[rows].cpu().clone()}
        for state_name, state_value in optimizer.state.get(parameter, {}).items():
            if (
                isinstance(state_value, torch.Tensor)
                and state_value.ndim >= 1
                and state_value.shape[0] == parameter.shape[0]
            ):
                values[state_name] = state_value.detach()[rows].cpu().clone()
        snapshot[name] = values
    return snapshot


class StableClosedRowAudit:
    """Audit CLOSED rows by stable ID across unrelated topology mutations."""

    def __init__(self, manager: Any) -> None:
        self.manager = manager
        self.entries: list[dict[str, Any]] = []
        self.total_closed_rows = 0
        self.reopened_rows = 0
        self.max_abs = 0.0
        self.missing_stable_ids = 0
        self.per_state: dict[str, float] = {}

    def add(
        self,
        model: Any,
        optimizer: Any,
        rows: torch.Tensor,
        slots: torch.Tensor,
    ) -> None:
        del slots
        rows = rows.detach().flatten().long().unique(sorted=True)
        if rows.numel() == 0:
            return
        self.entries.append(
            {
                "stable_ids": self.manager.stable_id[rows].detach().cpu().clone(),
                "snapshot": _optimizer_row_snapshot(model, optimizer, rows),
            }
        )
        self.total_closed_rows += int(rows.numel())

    def _positions(self, stable_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        current = self.manager.stable_id
        query = stable_ids.to(device=current.device)
        positions = torch.searchsorted(current, query)
        valid = positions < current.numel()
        matched = torch.zeros_like(valid)
        matched[valid] = current[positions[valid]] == query[valid]
        return positions, matched

    @torch.no_grad()
    def _compare(
        self,
        entry: dict[str, Any],
        selected: torch.Tensor,
    ) -> None:
        if selected.numel() == 0:
            return
        stable_ids = entry["stable_ids"][selected]
        rows, matched = self._positions(stable_ids)
        self.missing_stable_ids += int((~matched).sum().item())
        if not bool(matched.any()):
            return
        rows = rows[matched]
        selected = selected[matched.cpu()]
        parameters = dict(self.manager.model.persistent_parameter_items())
        for name, values in entry["snapshot"].items():
            parameter = parameters[name]
            for state_name, expected_all in values.items():
                expected = expected_all[selected].cpu()
                if state_name == "parameter":
                    actual = parameter.detach()[rows].cpu()
                else:
                    actual = self.manager.optimizer.state[parameter][state_name].detach()[rows].cpu()
                difference = (
                    float((actual - expected).abs().max().item())
                    if expected.numel()
                    else 0.0
                )
                key = f"{name}.{state_name}"
                self.per_state[key] = max(self.per_state.get(key, 0.0), difference)
                self.max_abs = max(self.max_abs, difference)

    @torch.no_grad()
    def release_reopened_rows(self, rows: torch.Tensor) -> None:
        rows = rows.detach().flatten().long().unique(sorted=True)
        if rows.numel() == 0:
            return
        reopened_ids = self.manager.stable_id[rows].detach().cpu()
        retained: list[dict[str, Any]] = []
        for entry in self.entries:
            selected_mask = torch.isin(entry["stable_ids"], reopened_ids)
            selected = torch.nonzero(selected_mask, as_tuple=False).flatten()
            self._compare(entry, selected)
            self.reopened_rows += int(selected.numel())
            keep = ~selected_mask
            if bool(keep.any()):
                entry["stable_ids"] = entry["stable_ids"][keep]
                for values in entry["snapshot"].values():
                    for state_name in tuple(values):
                        values[state_name] = values[state_name][keep]
                retained.append(entry)
        self.entries = retained

    @torch.no_grad()
    def verify(self) -> dict[str, Any]:
        for entry in self.entries:
            self._compare(
                entry,
                torch.arange(entry["stable_ids"].numel(), dtype=torch.long),
            )
        return {
            "passed": self.max_abs == 0.0 and self.missing_stable_ids == 0,
            "max_abs": float(self.max_abs),
            "total_closed_rows": int(self.total_closed_rows),
            "reopened_rows_verified_before_resume": int(self.reopened_rows),
            "remaining_closed_rows_verified_at_end": int(
                sum(entry["stable_ids"].numel() for entry in self.entries)
            ),
            "missing_stable_ids": int(self.missing_stable_ids),
            "per_parameter_and_optimizer_state_max_abs": dict(self.per_state),
        }


def _stable_lifecycle_events(
    events: Sequence[LifecycleEvent], stable_ids: torch.Tensor
) -> list[LifecycleEvent]:
    return [
        replace(event, gaussian_index=int(stable_ids[event.gaussian_index].item()))
        for event in events
    ]


def _make_detector(model: Any, config: RunConfig, args: argparse.Namespace):
    """Build the selected detector/controller without changing representation code."""

    if args.detector_mode == "direct_binary":
        tracker = make_filter(
            model.state_valid.shape[0],
            config,
            device=model.xyz.device,
            dtype=model.xyz.dtype,
        )
        return tracker, make_controller(model, config)

    if args.detector_mode == "lifespan_gate_beta":
        from temporal.lifespan_gate_beta import (
            LifespanGateBetaConfig,
            LifespanGateBetaController,
            LifespanGateBetaFilter,
        )

        tracker = LifespanGateBetaFilter(
            int(model.state_valid.shape[0]),
            LifespanGateBetaConfig(
                stable_flip_prior=float(args.gate_stable_flip_prior),
                stable_keep_prior=float(args.gate_stable_keep_prior),
                reset_flip_prior=float(args.gate_reset_flip_prior),
                reset_keep_prior=float(args.gate_reset_keep_prior),
                bayes_factor_threshold=float(
                    args.candidate_bayes_factor_threshold
                ),
                min_evidence_mass=float(config.min_evidence_mass),
            ),
            device=model.xyz.device,
            dtype=model.xyz.dtype,
        )
        return tracker, LifespanGateBetaController(model)

    if args.detector_mode == "anchored_dc_agreement_beta":
        from temporal.anchored_dc_agreement_beta import (
            AnchoredDCAgreementBetaConfig,
            AnchoredDCAgreementBetaFilter,
            LearnedDCAgreementBetaController,
        )

        tracker = AnchoredDCAgreementBetaFilter(
            int(model.state_valid.shape[0]),
            AnchoredDCAgreementBetaConfig(
                stable_flip_prior=float(args.agreement_stable_flip_prior),
                stable_keep_prior=float(args.agreement_stable_keep_prior),
                reset_flip_prior=float(args.agreement_reset_flip_prior),
                reset_keep_prior=float(args.agreement_reset_keep_prior),
                bayes_factor_threshold=float(
                    args.candidate_bayes_factor_threshold
                ),
                min_evidence_mass=float(config.min_evidence_mass),
                confirmation_views=int(args.agreement_confirmation_views),
                directional_margin=float(args.agreement_directional_margin),
            ),
            device=model.xyz.device,
            dtype=model.xyz.dtype,
        )
        return tracker, LearnedDCAgreementBetaController(model)

    if args.detector_mode == "learned_dc_agreement_beta":
        from temporal.learned_dc_agreement_beta import (
            LearnedDCAgreementBetaConfig,
            LearnedDCAgreementBetaController,
            LearnedDCAgreementBetaFilter,
        )

        tracker = LearnedDCAgreementBetaFilter(
            int(model.state_valid.shape[0]),
            LearnedDCAgreementBetaConfig(
                stable_flip_prior=float(args.agreement_stable_flip_prior),
                stable_keep_prior=float(args.agreement_stable_keep_prior),
                reset_flip_prior=float(args.agreement_reset_flip_prior),
                reset_keep_prior=float(args.agreement_reset_keep_prior),
                bayes_factor_threshold=float(
                    args.candidate_bayes_factor_threshold
                ),
                min_evidence_mass=float(config.min_evidence_mass),
            ),
            device=model.xyz.device,
            dtype=model.xyz.dtype,
        )
        return tracker, LearnedDCAgreementBetaController(model)

    from temporal.bayesian_lifespan_controller import BayesianLifespanController
    from temporal.bernoulli_bocd import BernoulliBOCDConfig
    from temporal.single_candidate_beta import (
        SingleCandidateBetaConfig,
        SingleCandidateBetaFilter,
    )

    tracker = SingleCandidateBetaFilter(
        int(model.state_valid.shape[0]),
        SingleCandidateBetaConfig(
            prior_a=1.0,
            prior_b=1.0,
            bayes_factor_threshold=float(args.candidate_bayes_factor_threshold),
            min_evidence_mass=float(config.min_evidence_mass),
        ),
        device=model.xyz.device,
        dtype=model.xyz.dtype,
    )
    # BayesianLifespanController only consumes the posterior/run fields in the
    # common BOCDUpdate interface.  The hazard below is a validation-only field
    # of its legacy config and is never used by SingleCandidateBetaFilter.
    controller_config = BernoulliBOCDConfig(
        prior_a=1.0,
        prior_b=1.0,
        hazard=0.01,
        max_run_length=1,
        min_evidence_mass=float(config.min_evidence_mass),
        open_probability=0.6,
        close_probability=0.4,
        changepoint_probability=0.5,
        min_run_evidence=1.0,
        min_visible_observations=1,
    )
    return tracker, BayesianLifespanController(
        model, controller_config, initialization="zero"
    )


def _candidate_controller_events(decision: Any) -> list[LifecycleEvent]:
    """Adapt Bayesian controller records to the runner's stable event schema."""

    from temporal.bayesian_lifespan_controller import LifespanAction

    positions = torch.nonzero(
        decision.event_mask, as_tuple=False
    ).flatten().tolist()
    return [
        LifecycleEvent(
            gaussian_index=int(decision.indices[pos].item()),
            decision_timestamp=int(decision.decision_timestamp),
            old_binary_label=int(decision.old_binary_label[pos].item()),
            new_binary_label=int(decision.new_binary_label[pos].item()),
            action=LifespanAction(int(decision.action[pos].item())).name,
            old_slot=int(decision.old_slot[pos].item()),
            new_current_slot=int(decision.current_slot[pos].item()),
            p_active=float(decision.posterior_probability[pos].item()),
            p_01=0.0,
            p_10=0.0,
            p_flip=float(decision.changepoint_probability[pos].item()),
            visible_observation_count=int(
                decision.visible_observations[pos].item()
            ),
        )
        for pos in positions
    ]


def _concat_cpu(values: list[torch.Tensor]) -> torch.Tensor:
    if not values:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(
        [value.detach().flatten().float().cpu() for value in values], dim=0
    )


def update_single_candidate_lifecycle_chunks(
    tracker: Any,
    controller: Any,
    model: Any,
    optimizer: Any,
    delta_a: torch.Tensor,
    delta_b: torch.Tensor,
    total_mass: torch.Tensor,
    *,
    timestamp: int,
    min_evidence_mass: float,
    chunk_size: int,
    closed_audit: StableClosedRowAudit,
    gate_mismatch: bool = False,
    learned_dc_agreement: bool = False,
) -> dict[str, Any]:
    """Update the O(1) candidate filter and shared lifespan controller in chunks."""

    from temporal.bayesian_lifespan_controller import LifespanAction

    observed = (
        (total_mass > 0)
        & (total_mass >= float(min_evidence_mass))
        & ((delta_a + delta_b) > 0)
    )
    rows_all = torch.nonzero(observed, as_tuple=False).flatten()
    action_counts = {action.name: 0 for action in LifespanAction}
    probability_values: list[torch.Tensor] = []
    cp_values: list[torch.Tensor] = []
    q_values: list[torch.Tensor] = []
    score_values: list[torch.Tensor] = []
    terminal_score_values: list[torch.Tensor] = []
    terminal_duration_values: list[torch.Tensor] = []
    terminal_support_values: list[torch.Tensor] = []
    committed_support_values: list[torch.Tensor] = []
    concentration_values: list[torch.Tensor] = []
    expected_change_values: list[torch.Tensor] = []
    agreement_flip_values: list[torch.Tensor] = []
    candidate_support_observation_values: list[torch.Tensor] = []
    directional_support_values: list[torch.Tensor] = []
    events: list[LifecycleEvent] = []
    candidate_started = 0
    candidate_continued = 0
    candidate_rejected = 0
    candidate_committed = 0
    for start in range(0, int(rows_all.numel()), int(chunk_size)):
        rows = rows_all[start : start + int(chunk_size)]
        update_kwargs = {
            "total_mass": total_mass[rows],
            "row_indices": rows,
            "timestamp": timestamp,
        }
        if gate_mismatch:
            update_kwargs["current_active"] = model.current_state_index[rows] >= 0
        elif learned_dc_agreement:
            update_kwargs["current_active"] = model.current_state_index[rows] >= 0
            update_kwargs["expected_change"] = _learned_dc_change_magnitude(
                model, rows
            )
        update = tracker.update(
            delta_a[rows],
            delta_b[rows],
            **update_kwargs,
        )
        decision = controller.update(update, timestamp=timestamp, optimizer=optimizer)
        counts = torch.bincount(
            decision.action.to(torch.long), minlength=len(LifespanAction)
        )
        for action in LifespanAction:
            action_counts[action.name] += int(counts[int(action)].item())
        probability_values.append(update.change_probability)
        cp_values.append(update.changepoint_probability)
        strength = delta_a[rows] + delta_b[rows]
        q_values.append(delta_a[rows] / strength.clamp_min(torch.finfo(strength.dtype).eps))
        if learned_dc_agreement:
            expected_change_values.append(update.expected_change)
            agreement_flip_values.append(
                update.delta_flip / strength.clamp_min(
                    torch.finfo(strength.dtype).eps
                )
            )
            if hasattr(update, "directional_support"):
                directional_support_values.append(
                    update.directional_support.to(dtype=delta_a.dtype)
                )
        evaluated = update.candidate_evaluated
        if bool(evaluated.any()):
            score_values.append(update.candidate_log_bayes_factor[evaluated])
            if hasattr(update, "candidate_support_observations"):
                candidate_support_observation_values.append(
                    update.candidate_support_observations[evaluated]
                )
        terminal = update.candidate_rejected | update.candidate_committed
        if bool(terminal.any()):
            terminal_score_values.append(update.candidate_log_bayes_factor[terminal])
            terminal_duration_values.append(update.candidate_duration[terminal])
            terminal_support_values.append(
                update.candidate_support_observations[terminal]
            )
        if bool(update.candidate_committed.any()):
            committed_support_values.append(
                update.candidate_support_observations[
                    update.candidate_committed
                ]
            )
        concentration_values.append(update.concentration)
        candidate_started += int(update.candidate_started.sum().item())
        candidate_continued += int(update.candidate_continued.sum().item())
        candidate_rejected += int(update.candidate_rejected.sum().item())
        candidate_committed += int(update.candidate_committed.sum().item())
        events.extend(_candidate_controller_events(decision))
        close_positions = torch.nonzero(
            decision.action == int(LifespanAction.CLOSE), as_tuple=False
        ).flatten()
        if close_positions.numel():
            closed_audit.add(
                model,
                optimizer,
                decision.indices[close_positions],
                decision.old_slot[close_positions],
            )

    empty = quantile_summary(torch.empty(0))
    return {
        "observed": observed,
        "observed_count": int(rows_all.numel()),
        "action_counts": action_counts,
        "p_active_stats": quantile_summary(_concat_cpu(probability_values)),
        "p_flip_stats": quantile_summary(_concat_cpu(cp_values)),
        "p_01_stats": dict(empty),
        "p_10_stats": dict(empty),
        "q_stats": quantile_summary(_concat_cpu(q_values)),
        "expected_change_stats": quantile_summary(
            _concat_cpu(expected_change_values)
        ),
        "agreement_flip_fraction_stats": quantile_summary(
            _concat_cpu(agreement_flip_values)
        ),
        "candidate_support_observation_stats": quantile_summary(
            _concat_cpu(candidate_support_observation_values)
        ),
        "directional_support_fraction_stats": quantile_summary(
            _concat_cpu(directional_support_values)
        ),
        "candidate_started_count": int(candidate_started),
        "candidate_continued_count": int(candidate_continued),
        "candidate_rejected_count": int(candidate_rejected),
        "candidate_committed_count": int(candidate_committed),
        "candidate_live_count": int(tracker.candidate_active.sum().item()),
        "candidate_log_bayes_factor_stats": quantile_summary(
            _concat_cpu(score_values)
        ),
        "terminal_candidate_log_bayes_factor_stats": quantile_summary(
            _concat_cpu(terminal_score_values)
        ),
        "terminal_candidate_duration_stats": quantile_summary(
            _concat_cpu(terminal_duration_values)
        ),
        "terminal_candidate_support_observation_stats": quantile_summary(
            _concat_cpu(terminal_support_values)
        ),
        "committed_candidate_support_observation_stats": quantile_summary(
            _concat_cpu(committed_support_values)
        ),
        "stable_concentration_stats": quantile_summary(
            _concat_cpu(concentration_values)
        ),
        "events": events,
    }


def _empty_candidate_frame_diagnostics() -> dict[str, Any]:
    empty = quantile_summary(torch.empty(0))
    return {
        "candidate_started_count": 0,
        "candidate_continued_count": 0,
        "candidate_rejected_count": 0,
        "candidate_committed_count": 0,
        "candidate_live_count": 0,
        "candidate_log_bayes_factor_stats": dict(empty),
        "terminal_candidate_log_bayes_factor_stats": dict(empty),
        "terminal_candidate_duration_stats": dict(empty),
        "terminal_candidate_support_observation_stats": dict(empty),
        "committed_candidate_support_observation_stats": dict(empty),
        "stable_concentration_stats": dict(empty),
        "expected_change_stats": dict(empty),
        "agreement_flip_fraction_stats": dict(empty),
        "candidate_support_observation_stats": dict(empty),
        "directional_support_fraction_stats": dict(empty),
    }


def candidate_run_diagnostics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize candidate counts and the distribution of per-frame means."""

    def frame_mean_summary(key: str) -> dict[str, Any]:
        values = [
            row[key]["mean"]
            for row in rows
            if row.get(key, {}).get("mean") is not None
        ]
        return quantile_summary(torch.as_tensor(values, dtype=torch.float64))

    return {
        "candidate_started_count": int(
            sum(row.get("candidate_started_count", 0) for row in rows)
        ),
        "candidate_continued_count": int(
            sum(row.get("candidate_continued_count", 0) for row in rows)
        ),
        "candidate_rejected_count": int(
            sum(row.get("candidate_rejected_count", 0) for row in rows)
        ),
        "candidate_committed_count": int(
            sum(row.get("candidate_committed_count", 0) for row in rows)
        ),
        "max_live_candidate_count": int(
            max((row.get("candidate_live_count", 0) for row in rows), default=0)
        ),
        "final_live_candidate_count": int(
            rows[-1].get("candidate_live_count", 0) if rows else 0
        ),
        "candidate_render_gate": (
            rows[-1].get("candidate_render_gate", "none") if rows else "none"
        ),
        "soft_opening_candidate_row_frames": int(
            sum(row.get("candidate_soft_opening_count", 0) for row in rows)
        ),
        "soft_closing_candidate_row_frames": int(
            sum(row.get("candidate_soft_closing_count", 0) for row in rows)
        ),
        "max_soft_opening_candidate_count": int(
            max(
                (row.get("candidate_soft_opening_count", 0) for row in rows),
                default=0,
            )
        ),
        "max_soft_closing_candidate_count": int(
            max(
                (row.get("candidate_soft_closing_count", 0) for row in rows),
                default=0,
            )
        ),
        "candidate_transition_progress_frame_mean_stats": frame_mean_summary(
            "candidate_transition_progress_stats"
        ),
        "candidate_soft_change_weight_frame_mean_stats": frame_mean_summary(
            "candidate_soft_change_weight_stats"
        ),
        "candidate_log_bayes_factor_frame_mean_stats": frame_mean_summary(
            "candidate_log_bayes_factor_stats"
        ),
        "terminal_candidate_log_bayes_factor_frame_mean_stats": frame_mean_summary(
            "terminal_candidate_log_bayes_factor_stats"
        ),
        "terminal_candidate_duration_frame_mean_stats": frame_mean_summary(
            "terminal_candidate_duration_stats"
        ),
        "terminal_candidate_support_frame_mean_stats": frame_mean_summary(
            "terminal_candidate_support_observation_stats"
        ),
        "committed_candidate_support_frame_mean_stats": frame_mean_summary(
            "committed_candidate_support_observation_stats"
        ),
        "stable_concentration_frame_mean_stats": frame_mean_summary(
            "stable_concentration_stats"
        ),
        "candidate_support_observation_frame_mean_stats": frame_mean_summary(
            "candidate_support_observation_stats"
        ),
        "directional_support_fraction_frame_mean_stats": frame_mean_summary(
            "directional_support_fraction_stats"
        ),
    }


def _sample_training_view(
    processed_views: Sequence[Any],
    rng: np.random.Generator,
    current_probability: float,
) -> tuple[Any, int]:
    if not processed_views:
        raise ValueError("processed view bank must be nonempty")
    if float(rng.random()) < float(current_probability):
        index = len(processed_views) - 1
    else:
        index = int(rng.integers(0, len(processed_views)))
    return processed_views[index], index


def _validate_args(args: argparse.Namespace) -> None:
    if args.scope not in SCOPES:
        raise ValueError("unknown independent/continuous ESCD scope")
    if args.density_policy not in DENSITY_POLICIES:
        raise ValueError("unknown density policy")
    if (
        args.density_policy in _CUE_MIXTURE_DENSITY_POLICIES
        and float(args.cue_mixture_threshold) <= 0.0
    ):
        raise ValueError("cue-mixture density requires a positive threshold")
    if args.density_policy == "active_signed_lifespan_score":
        if args.signed_score_artifact is None:
            raise ValueError("active_signed_lifespan_score requires --signed-score-artifact")
        if not math.isfinite(float(args.signed_score_threshold)) or float(
            args.signed_score_threshold
        ) < 0.0:
            raise ValueError("signed score threshold must be finite and nonnegative")
        if (
            args.signed_score_max_sources is not None
            and int(args.signed_score_max_sources) < 0
        ):
            raise ValueError("signed score max sources must be nonnegative")
        if args.signed_score_posterior not in {"balanced", "global"}:
            raise ValueError("signed score posterior must be balanced or global")
        if args.signed_score_prune_threshold is not None and (
            not math.isfinite(float(args.signed_score_prune_threshold))
            or float(args.signed_score_prune_threshold) < 0.0
        ):
            raise ValueError("signed score prune threshold must be finite and nonnegative")
    if args.render_support_mode not in RENDER_SUPPORT_MODES:
        raise ValueError("unknown render support mode")
    if args.detector_mode not in DETECTOR_MODES:
        raise ValueError("unknown detector mode")
    if args.first_open_dc_initialization not in FIRST_OPEN_DC_INITIALIZATIONS:
        raise ValueError("unknown first-OPEN DC initialization")
    if args.optimizer_selection not in OPTIMIZER_SELECTION_POLICIES:
        raise ValueError("unknown optimizer selection policy")
    if args.loss_regularization_mode not in LOSS_REGULARIZATION_MODES:
        raise ValueError("unknown loss regularization mode")
    if args.change_color_mode not in CHANGE_COLOR_MODES:
        raise ValueError("unknown change color mode")
    if args.candidate_render_gate not in CANDIDATE_RENDER_GATES:
        raise ValueError("unknown candidate render gate")
    if (
        args.optimizer_selection == "all_open"
        and _trains_never_open_appearance(args.render_support_mode)
    ):
        raise ValueError(
            "all_open optimizer selection is incompatible with trainable NEVER_OPEN appearance"
        )
    if args.max_frames is not None and args.max_frames > SCOPE_MAX_FRAMES[args.scope]:
        raise ValueError("max_frames exceeds the selected independent scope")
    if not 0 <= args.densify_update_index < args.updates_per_frame:
        raise ValueError("densify_update_index must be inside the update schedule")
    if not math.isfinite(float(args.local_support_scale)) or float(
        args.local_support_scale
    ) <= 0.0:
        raise ValueError("local support scale must be finite and positive")
    if not math.isfinite(float(args.growth_replay_weight)) or float(
        args.growth_replay_weight
    ) < 0.0:
        raise ValueError("growth replay weight must be finite and nonnegative")
    if (
        args.loss_regularization_mode == "local_growth_replay"
        and args.optimizer_selection != "all_open"
    ):
        raise ValueError(
            "local growth replay currently requires all_open optimizer selection"
        )
    gate_detector = args.detector_mode == "lifespan_gate_beta"
    gate_color = args.change_color_mode == "lifespan_gate"
    if gate_detector != gate_color:
        raise ValueError(
            "lifespan_gate_beta detector and lifespan_gate change color must be enabled together"
        )
    learned_dc_close_only_gate = (
        args.candidate_render_gate == "close_only_log_bf_progress"
        and args.detector_mode == "single_candidate_beta"
        and args.change_color_mode == "learned_dc"
    )
    if (
        args.candidate_render_gate != "none"
        and not (gate_detector and gate_color)
        and not learned_dc_close_only_gate
    ):
        raise ValueError(
            "candidate render gating requires lifespan_gate_beta detector and "
            "lifespan_gate change color; close-only gating additionally supports "
            "single_candidate_beta with learned_dc"
        )
    if gate_color and args.first_open_dc_initialization != "preserve":
        raise ValueError("lifespan-gate semantic color requires preserved underlying DC")
    if gate_color and _trains_never_open_appearance(args.render_support_mode):
        raise ValueError("lifespan-gate semantic color requires frozen NEVER_OPEN appearance")
    agreement_detector = args.detector_mode in {
        "learned_dc_agreement_beta",
        "anchored_dc_agreement_beta",
    }
    if agreement_detector and args.change_color_mode != "learned_dc":
        raise ValueError("learned-DC agreement detector requires learned_dc change color")
    if agreement_detector and args.first_open_dc_initialization != "preserve":
        raise ValueError(
            "learned-DC agreement controller owns OPEN/CLOSE DC initialization"
        )
    if agreement_detector and _trains_never_open_appearance(args.render_support_mode):
        raise ValueError(
            "learned-DC agreement requires frozen NEVER_OPEN appearance"
        )
    if args.agreement_candidate_dc_policy not in AGREEMENT_CANDIDATE_DC_POLICIES:
        raise ValueError("unknown agreement candidate DC policy")
    if (
        args.detector_mode == "anchored_dc_agreement_beta"
        and args.agreement_candidate_dc_policy != "adapt"
    ):
        raise ValueError("anchored agreement requires candidate DC optimization")
    if not math.isfinite(float(args.agreement_directional_margin)) or not (
        0.0 <= float(args.agreement_directional_margin) < 1.0
    ):
        raise ValueError("agreement directional margin must be in [0, 1)")
    for name in (
        "gate_stable_flip_prior",
        "gate_stable_keep_prior",
        "gate_reset_flip_prior",
        "gate_reset_keep_prior",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    for name in (
        "agreement_stable_flip_prior",
        "agreement_stable_keep_prior",
        "agreement_reset_flip_prior",
        "agreement_reset_keep_prior",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")


def run(args: argparse.Namespace) -> dict[str, Any]:
    from experiments.run_online_bayesian_lifespan_thaw import build_causal_records
    from experiments.train_cue_temporal_rchange import (
        build_fixed_cue_views,
        load_fixed_camera_index,
        validate_cue_cache,
    )
    from experiments.train_real_temporal_rchange import (
        file_checksum,
        oscd_positive_sparsity_loss,
        seed_everything,
    )
    from scene import GaussianModel
    from temporal import (
        DynamicGaussianTopologyManager,
        PersistentGaussianLifespanModel,
    )
    from temporal.change_evidence import accumulate_change_evidence
    from temporal.fusion import (
        compute_growth_replay_regularization,
        compute_ssf_loss,
    )
    from temporal.masked_optimizer import MaskedRowAdam
    from temporal.signed_lifespan_density import compute_signed_lifespan_score

    _validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    config = _config(args)
    seed_everything(config.seed)
    rng = np.random.default_rng(config.seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_records, _ = build_causal_records(args.source_path)
    frame_limit = int(args.max_frames or SCOPE_MAX_FRAMES[args.scope])
    records = select_scope_records(all_records, scope=args.scope, max_frames=frame_limit)
    if [record.global_index for record in records] != list(range(len(records))):
        raise RuntimeError("independent records must be causally reindexed")
    signed_artifact = (
        _load_signed_score_artifact(
            args.signed_score_artifact,
            expected_frame_names=[record.name for record in records],
            posterior=args.signed_score_posterior,
        )
        if args.density_policy == "active_signed_lifespan_score"
        else None
    )
    base_ply = (args.source_path / BASE_PLY_REL).resolve()
    cue_metadata = validate_cue_cache(args.cue_cache_root, base_ply, args.resolution)
    cameras = load_fixed_camera_index(args.fixed_cameras_json)
    extent = reference_scene_extent(
        args.source_path / "reference_reconstruction/cameras.json"
    )

    mutable = GaussianModel(sh_degree=3, active_sh_degree=0)
    mutable.load_ply_change(str(base_ply))
    signed_reference = None
    signed_sam = None
    if signed_artifact is not None:
        from transformers import Sam2Model

        signed_reference = GaussianModel(sh_degree=3, active_sh_degree=0)
        signed_reference.load_ply(str(base_ply))
        for parameter_name in (
            "_xyz",
            "_features_dc",
            "_features_rest",
            "_opacity",
            "_scaling",
            "_rotation",
        ):
            getattr(signed_reference, parameter_name).requires_grad_(False)
        signed_sam = (
            Sam2Model.from_pretrained(
                args.signed_score_sam_model, local_files_only=True
            )
            .half()
            .cuda()
            .eval()
        )
    model = PersistentGaussianLifespanModel.from_gaussians(
        mutable, max_states=config.max_states
    )
    model.reset_all_lifespans_closed()
    optimizer = MaskedRowAdam(
        dict(model.persistent_parameter_items()),
        thaw_names=DIRECT_PARAMETER_NAMES,
        lrs={
            "dc": float(args.dc_lr),
            "xyz": float(args.xyz_lr),
            "features_rest": float(args.features_rest_lr),
            "opacity": float(args.opacity_lr),
            "scaling": float(args.scaling_lr),
            "rotation": float(args.rotation_lr),
        },
        eps=float(args.adam_eps),
    )
    tracker, controller = _make_detector(model, config, args)
    topology = DynamicGaussianTopologyManager(
        model,
        optimizer,
        tracker,
        controller,
        percent_dense=float(args.percent_dense),
    )
    initial_count = topology.count
    closed_audit = StableClosedRowAudit(topology)
    pipe = SimpleNamespace(
        compute_cov3D_python=False, convert_SHs_python=False, debug=False
    )
    background = torch.zeros(3, device=model.xyz.device, dtype=model.xyz.dtype)
    visual_root = args.visualization_dir
    if visual_root is not None:
        visual_root.mkdir(parents=True, exist_ok=True)

    processed_views: list[Any] = []
    predictions: list[np.ndarray] = []
    pre_predictions: list[np.ndarray] = []
    hard_output_predictions: list[np.ndarray] = []
    hard_pre_output_predictions: list[np.ndarray] = []
    frame_rows: list[dict[str, Any]] = []
    lifecycle_events: list[LifecycleEvent] = []
    density_rows: list[dict[str, Any]] = []
    inactive_gradient_violations = 0
    inactive_gradient_max_abs = 0.0
    max_dc_gradient = 0.0
    max_features_rest_gradient = 0.0
    total_clones = 0
    total_split_sources = 0
    total_split_children = 0
    total_pruned = 0
    total_opacity_pruned = 0
    total_size_pruned = 0
    total_black_child_prune_candidates = 0
    total_black_children_pruned = 0
    total_black_children_retained_for_support = 0
    total_signed_score_candidates = 0
    total_signed_score_clone_sources = 0
    total_signed_score_split_sources = 0
    total_signed_score_prune_candidates = 0
    total_signed_score_pruned = 0
    signed_density_source_rows: list[dict[str, Any]] = []
    first_open_dc_values: dict[str, list[torch.Tensor]] = {
        "before": [],
        "immediate": [],
        "after_first_update": [],
        "before_density": [],
        "post_frame_surviving": [],
        "raw_abs_jump": [],
    }
    total_first_open_dc_adam_resets = 0
    total_first_open_source_survivors = 0
    previous_growth_view: Any | None = None
    previous_growth_map: torch.Tensor | None = None
    growth_replay_values: list[float] = []
    growth_map_means: list[float] = []
    growth_map_maxima: list[float] = []
    growth_map_nonzero_fractions: list[float] = []
    total_growth_replay_joint_steps = 0
    agreement_candidate_gap_before_values: list[torch.Tensor] = []
    agreement_candidate_gap_after_values: list[torch.Tensor] = []
    agreement_candidate_gap_delta_values: list[torch.Tensor] = []
    agreement_candidate_anchor_gap_values: list[torch.Tensor] = []
    agreement_candidate_anchor_drift_values: list[torch.Tensor] = []
    agreement_candidate_source_survivors = 0
    max_suppressed_candidate_dc_gradient = 0.0
    for record in records:
        frame_started = time.time()
        timestamp = int(record.global_index)
        current_view = build_fixed_cue_views(
            [record], cameras, args.cue_cache_root, args.resolution
        )[0][0]
        processed_views.append(current_view)

        # Detector input contract: consume the newly arrived frame exactly once,
        # before any lifecycle mutation or representation optimization at this
        # timestamp.  Raw cue alpha-T evidence is detached below; replay views and
        # post-optimization learned DC are never fed back as observations for the
        # same frame.  Every row remains probeable through raw geometry/opacity,
        # irrespective of lifespan opacity gating.
        evidence = accumulate_change_evidence(
            current_view,
            model.base,
            pipe,
            background,
            current_view.candidate_map,
            cue_mode="binary",
            cue_threshold=config.bayes_cue_threshold,
            count_mode="capped",
            mass_saturation=config.evidence_mass_saturation,
            min_evidence_mass=config.min_evidence_mass,
            probe_scaling_mode=args.detector_probe_scaling,
        )
        signed_plus_mass = None
        signed_minus_mass = None
        signed_frame_delta_nonzero = {"plus": 0, "minus": 0}
        if signed_artifact is not None:
            if signed_reference is None or signed_sam is None:
                raise RuntimeError("signed-score frontend was not initialized")
            delta = _extract_signed_sam_delta(
                current_view, signed_reference, pipe, background, signed_sam
            )
            plus_mask, minus_mask = _signed_score_masks_from_delta(
                delta,
                current_view.candidate_map,
                signed_artifact.pc1_axes[timestamp],
                epsilon_negative=float(signed_artifact.epsilon_negative[timestamp]),
                epsilon_positive=float(signed_artifact.epsilon_positive[timestamp]),
                cue_threshold=float(args.bayes_cue_threshold),
            )
            signed_frame_delta_nonzero = {
                "plus": int((plus_mask > 0.5).sum().item()),
                "minus": int((minus_mask > 0.5).sum().item()),
            }
            plus_evidence = accumulate_change_evidence(
                current_view,
                model.base,
                pipe,
                background,
                plus_mask,
                cue_mode="binary",
                cue_threshold=0.5,
                count_mode="raw",
                mass_saturation=config.evidence_mass_saturation,
                min_evidence_mass=0.0,
                probe_scaling_mode=args.detector_probe_scaling,
            )
            minus_evidence = accumulate_change_evidence(
                current_view,
                model.base,
                pipe,
                background,
                minus_mask,
                cue_mode="binary",
                cue_threshold=0.5,
                count_mode="raw",
                mass_saturation=config.evidence_mass_saturation,
                min_evidence_mass=0.0,
                probe_scaling_mode=args.detector_probe_scaling,
            )
            signed_plus_mass = plus_evidence.e_plus.detach()
            signed_minus_mass = minus_evidence.e_plus.detach()
        stable_before_decision = topology.stable_id.detach().clone()
        if args.detector_mode in {
            "single_candidate_beta",
            "lifespan_gate_beta",
            "learned_dc_agreement_beta",
            "anchored_dc_agreement_beta",
        }:
            binary = update_single_candidate_lifecycle_chunks(
                tracker,
                controller,
                model,
                optimizer,
                evidence.delta_a,
                evidence.delta_b,
                evidence.total_mass,
                timestamp=timestamp,
                min_evidence_mass=config.min_evidence_mass,
                chunk_size=config.filter_chunk_size,
                closed_audit=closed_audit,
                gate_mismatch=(args.detector_mode == "lifespan_gate_beta"),
                learned_dc_agreement=(
                    args.detector_mode in LEARNED_DC_AGREEMENT_MODES
                ),
            )
        else:
            binary = update_binary_lifecycle_chunks(
                tracker,
                controller,
                model,
                optimizer,
                evidence.delta_a,
                evidence.delta_b,
                evidence.total_mass,
                timestamp=timestamp,
                min_evidence_mass=config.min_evidence_mass,
                chunk_size=config.filter_chunk_size,
                closed_audit=closed_audit,
            )
            binary.update(_empty_candidate_frame_diagnostics())
        raw_events = list(binary["events"])
        agreement_candidate_snapshot: dict[str, torch.Tensor] | None = None
        if args.detector_mode in LEARNED_DC_AGREEMENT_MODES:
            strength = evidence.delta_a + evidence.delta_b
            live_open = (
                tracker.candidate_active
                & binary["observed"]
                & topology.active_mask()
            )
            live_rows = torch.nonzero(live_open, as_tuple=False).flatten()
            cue_ratio = evidence.delta_a[live_rows] / strength[live_rows].clamp_min(
                torch.finfo(strength.dtype).eps
            )
            dc_before = _learned_dc_change_magnitude(model, live_rows)
            agreement_candidate_snapshot = {
                "stable_ids": topology.stable_id[live_rows].detach().clone(),
                "cue_ratio": cue_ratio.detach().clone(),
                "dc_before": dc_before.detach().clone(),
                "gap_before": (dc_before - cue_ratio).abs().detach().clone(),
            }
            if args.detector_mode == "anchored_dc_agreement_beta":
                agreement_candidate_snapshot["anchor"] = (
                    tracker.candidate_anchor[live_rows].detach().clone()
                )
        first_open_initialization = _initialize_first_open_dc(
            model,
            optimizer,
            raw_events,
            stable_before_decision,
            args.first_open_dc_initialization,
        )
        for key in ("before", "immediate", "raw_abs_jump"):
            first_open_dc_values[key].append(first_open_initialization[key])
        total_first_open_dc_adam_resets += int(
            first_open_initialization["adam_reset_count"]
        )
        first_open_after_first_update = model.change_dc.new_empty((0,))
        first_open_before_density = model.change_dc.new_empty((0,))
        event_rgb = None
        if visual_root is not None:
            event_rgb = _render_current_lifecycle_events(
                current_view,
                model,
                pipe,
                background,
                raw_events,
            )
        reopened_rows = torch.tensor(
            [
                event.gaussian_index
                for event in raw_events
                if event.action == "OPEN" and event.new_current_slot > 0
            ],
            device=model.xyz.device,
            dtype=torch.long,
        )
        if reopened_rows.numel():
            closed_audit.release_reopened_rows(reopened_rows)
        if args.detector_mode in LEARNED_DC_AGREEMENT_MODES:
            opened_rows = torch.tensor(
                [
                    event.gaussian_index
                    for event in raw_events
                    if event.action == "OPEN"
                ],
                device=model.xyz.device,
                dtype=torch.long,
            )
            controller.initialize_open_dc(opened_rows, optimizer=optimizer)
        frame_events = _stable_lifecycle_events(raw_events, stable_before_decision)
        lifecycle_events.extend(frame_events)
        signed_score = None
        if signed_artifact is not None:
            if signed_plus_mass is None or signed_minus_mass is None:
                raise RuntimeError("signed-score masses were not computed")
            signed_score = compute_signed_lifespan_score(
                signed_plus_mass,
                signed_minus_mass,
                float(signed_artifact.p_plus_is_new[timestamp]),
                topology.active_mask(),
                _current_episode_start(model, timestamp),
                timestamp,
            )

        optimizer.zero_grad(set_to_none=True)
        # Keep the committed-state render as the optimizer/growth baseline.
        # A candidate gate, when enabled, is a separate output-only render.
        pre_package = _render_dynamic_change(
            current_view,
            model,
            pipe,
            background,
            timestamp=float(timestamp),
            include_never_open_occluders=(
                _includes_never_open(args.render_support_mode)
            ),
            train_never_open_dc_opacity=_trains_never_open_appearance(
                args.render_support_mode
            ),
            black_never_open_occluders=_uses_black_never_open_occluders(
                args.render_support_mode
            ),
            change_color_mode=args.change_color_mode,
        )
        pre_output_package = pre_package
        if args.candidate_render_gate != "none":
            pre_close_candidate_mask = (
                _single_candidate_close_candidate_mask(
                    tracker,
                    topology.active_mask(),
                    close_probability=float(controller.config.close_probability),
                )
                if args.detector_mode == "single_candidate_beta"
                else None
            )
            _, pre_soft_change_weight, pre_soft_gated_rows = (
                _candidate_render_gate_state(
                    args.candidate_render_gate,
                    topology.active_mask(),
                    tracker.candidate_active,
                    tracker.last_log_bayes_factor,
                    bayes_factor_threshold=args.candidate_bayes_factor_threshold,
                    close_candidate_mask=pre_close_candidate_mask,
                )
            )
        else:
            pre_soft_change_weight = model.xyz.new_empty((0,))
            pre_soft_gated_rows = torch.zeros_like(topology.active_mask())
        if bool(pre_soft_gated_rows.any()):
            pre_output_package = _render_dynamic_change(
                current_view,
                model,
                pipe,
                background,
                timestamp=float(timestamp),
                include_never_open_occluders=(
                    _includes_never_open(args.render_support_mode)
                ),
                train_never_open_dc_opacity=_trains_never_open_appearance(
                    args.render_support_mode
                ),
                black_never_open_occluders=_uses_black_never_open_occluders(
                    args.render_support_mode
                ),
                change_color_mode=args.change_color_mode,
                soft_change_weight=pre_soft_change_weight,
            )
        pre_prediction, _ = _prediction_from_package(
            pre_output_package, config.evaluation_threshold
        )
        pre_predictions.append(pre_prediction)
        hard_pre_prediction, _ = _prediction_from_package(
            pre_package, config.evaluation_threshold
        )
        hard_pre_output_predictions.append(hard_pre_prediction)
        pre_change_probability = _render_change_probability(
            pre_package["render"].detach()
        )

        selected_stable_ids: list[torch.Tensor] = []
        active_visible_stable_ids: list[torch.Tensor] = []
        selected_never_open_stable_ids: list[torch.Tensor] = []
        sampled_view_indices: list[int] = []
        frame_growth_replay_values: list[float] = []
        frame_growth_replay_joint_steps = 0
        density_result = None
        density_event = {
            "timestamp": timestamp,
            "initial_gaussian_count": topology.count,
            "final_gaussian_count": topology.count,
            "clone_source_count": 0,
            "clone_child_count": 0,
            "split_source_count": 0,
            "split_child_count": 0,
            "gradient_split_source_count": 0,
            "cue_mixture_split_source_count": 0,
            "cue_mixture_only_split_source_count": 0,
            "signed_score_candidate_count": 0,
            "signed_score_clone_source_count": 0,
            "signed_score_split_source_count": 0,
            "signed_score_prune_candidate_count": 0,
            "signed_score_pruned_count": 0,
            "signed_score_positive_count": 0,
            "signed_score_negative_count": 0,
            "signed_score_stats": quantile_summary(model.xyz.new_empty((0,))),
            "signed_score_support_stats": quantile_summary(model.xyz.new_empty((0,))),
            "signed_score_age_stats": quantile_summary(model.xyz.new_empty((0,))),
            "signed_score_plus_mask_pixels": int(signed_frame_delta_nonzero["plus"]),
            "signed_score_minus_mask_pixels": int(signed_frame_delta_nonzero["minus"]),
            "signed_score_p_plus_is_new": (
                float(signed_artifact.p_plus_is_new[timestamp])
                if signed_artifact is not None
                else None
            ),
            "split_source_removed_count": 0,
            "opacity_pruned_count": 0,
            "size_pruned_count": 0,
            "black_child_prune_candidate_count": 0,
            "black_child_pruned_count": 0,
            "black_child_retained_for_support_count": 0,
            "total_removed_count": 0,
        }
        for update_index in range(config.updates_per_frame):
            train_view, sampled_index = _sample_training_view(
                processed_views, rng, float(args.current_view_probability)
            )
            sampled_view_indices.append(sampled_index)
            package = _render_dynamic_change(
                train_view,
                model,
                pipe,
                background,
                timestamp=float(timestamp),
                include_never_open_occluders=(
                    _includes_never_open(args.render_support_mode)
                ),
                train_never_open_dc_opacity=_trains_never_open_appearance(
                    args.render_support_mode
                ),
                black_never_open_occluders=_uses_black_never_open_occluders(
                    args.render_support_mode
                ),
                change_color_mode=args.change_color_mode,
            )
            active = topology.active_mask()
            visible = package["radii"].detach() > 0
            active_visible = active & visible
            selected = (
                active
                if args.optimizer_selection == "all_open"
                else active_visible
            )
            never_open_selected = (model.num_states == 0) & visible
            if bool(active_visible.any()):
                active_visible_stable_ids.append(
                    topology.stable_id[active_visible].detach().clone()
                )
            optimizer_masks = _optimizer_row_masks(
                args.render_support_mode,
                active_rows=active,
                active_visible=active_visible,
                never_open_visible=never_open_selected,
                selection_policy=args.optimizer_selection,
            )
            if args.detector_mode == "learned_dc_agreement_beta":
                optimizer_masks = _candidate_dc_optimizer_masks(
                    optimizer_masks,
                    tracker.candidate_active.to(device=active.device),
                    policy=args.agreement_candidate_dc_policy,
                )
            any_selected = selected | (
                never_open_selected
                if _trains_never_open_appearance(args.render_support_mode)
                else False
            )
            if bool(any_selected.any()):
                if bool(selected.any()):
                    selected_stable_ids.append(
                        topology.stable_id[selected].detach().clone()
                    )
                if (
                    _trains_never_open_appearance(args.render_support_mode)
                    and bool(never_open_selected.any())
                ):
                    selected_never_open_stable_ids.append(
                        topology.stable_id[never_open_selected].detach().clone()
                    )
                optimizer.zero_grad(set_to_none=True)
                if args.loss_regularization_mode == "global":
                    loss, _parts = oscd_positive_sparsity_loss(
                        train_view.training_target, package["render"]
                    )
                else:
                    train_support = _cue_local_support(
                        train_view.training_target,
                        float(args.local_support_scale),
                    )
                    loss, _parts = compute_ssf_loss(
                        train_view.training_target,
                        package["render"],
                        regularization_mode="local",
                        local_support=train_support,
                    )
                    if train_view is current_view and previous_growth_view is not None:
                        if previous_growth_map is None:
                            raise RuntimeError("previous growth map is missing")
                        previous_package = _render_dynamic_change(
                            previous_growth_view,
                            model,
                            pipe,
                            background,
                            timestamp=float(timestamp),
                            include_never_open_occluders=(
                                _includes_never_open(args.render_support_mode)
                            ),
                            train_never_open_dc_opacity=_trains_never_open_appearance(
                                args.render_support_mode
                            ),
                            black_never_open_occluders=_uses_black_never_open_occluders(
                                args.render_support_mode
                            ),
                            change_color_mode=args.change_color_mode,
                        )
                        previous_support = _cue_local_support(
                            previous_growth_view.training_target,
                            float(args.local_support_scale),
                        )
                        previous_loss, previous_parts = compute_ssf_loss(
                            previous_growth_view.training_target,
                            previous_package["render"],
                            regularization_mode="local",
                            local_support=previous_support,
                        )
                        growth_replay = compute_growth_replay_regularization(
                            previous_support,
                            previous_parts["change_probability"],
                            previous_growth_map,
                        )
                        loss = 0.5 * (loss + previous_loss) + float(
                            args.growth_replay_weight
                        ) * growth_replay
                        growth_value = float(growth_replay.detach().item())
                        frame_growth_replay_values.append(growth_value)
                        growth_replay_values.append(growth_value)
                        frame_growth_replay_joint_steps += 1
                        total_growth_replay_joint_steps += 1
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(
                        f"non-finite loss at timestamp {timestamp}, update {update_index}"
                    )
                loss.backward()
                dc_gradient = model.change_dc.grad
                if dc_gradient is not None and dc_gradient.numel():
                    max_dc_gradient = max(
                        max_dc_gradient,
                        float(dc_gradient.detach().abs().max().item()),
                    )
                if args.detector_mode == "learned_dc_agreement_beta":
                    max_suppressed_candidate_dc_gradient = max(
                        max_suppressed_candidate_dc_gradient,
                        _zero_frozen_dc_gradients(
                            model,
                            tracker.candidate_active.to(
                                device=model.change_dc.device
                            ),
                            policy=args.agreement_candidate_dc_policy,
                        ),
                    )
                rest_gradient = model.features_rest.grad
                if rest_gradient is not None and rest_gradient.numel():
                    max_features_rest_gradient = max(
                        max_features_rest_gradient,
                        float(rest_gradient.detach().abs().max().item()),
                    )
                audit = _gradient_audit(model, optimizer_masks)
                inactive_gradient_violations += int(audit["count"])
                inactive_gradient_max_abs = max(
                    inactive_gradient_max_abs, float(audit["max_abs"])
                )
                if args.density_policy != "active_signed_lifespan_score":
                    topology.add_gradient_stats(
                        package["viewspace_points"], package["radii"], active
                    )
                optimizer.step(optimizer_masks)

            if update_index == 0:
                first_open_after_first_update = _intrinsic_dc_render_value(
                    model, first_open_initialization["rows"]
                )
            if update_index == int(args.densify_update_index):
                first_open_before_density = _intrinsic_dc_render_value(
                    model, first_open_initialization["rows"]
                )

            if (
                args.density_policy in {
                    "active_oscd",
                    "active_oscd_cue_mixture",
                    "active_oscd_cue_mixture_black_child_prune",
                    "active_signed_lifespan_score",
                }
                and update_index == int(args.densify_update_index)
            ):
                pre_density_xyz = model.xyz.detach().clone()
                pre_density_stable_id = topology.stable_id.detach().clone()
                pre_density_generation = topology.generation.detach().clone()
                gradient_signal = (
                    torch.linalg.vector_norm(
                        torch.nan_to_num(
                            topology.xyz_gradient_accum
                            / topology.denom.clamp_min(1.0)
                        ),
                        dim=-1,
                    )
                    if topology.xyz_gradient_accum.numel()
                    else model.xyz.new_empty((0,))
                )
                mixture_score = (
                    _cue_mixture_score(evidence.delta_a, evidence.delta_b)
                    if args.density_policy in _CUE_MIXTURE_DENSITY_POLICIES
                    else None
                )
                if args.density_policy == "active_signed_lifespan_score" and signed_score is None:
                    raise RuntimeError("signed lifespan score was not computed")
                signed_density = (
                    signed_score.score
                    if args.density_policy == "active_signed_lifespan_score"
                    else None
                )
                density_result = topology.apply_active_oscd_density_control(
                    timestamp=timestamp,
                    scene_extent=extent,
                    grad_threshold=float(args.oscd_grad_threshold),
                    min_opacity=float(args.min_opacity),
                    max_screen_size=(
                        None
                        if float(args.max_screen_size) == 0.0
                        else float(args.max_screen_size)
                    ),
                    cue_mixture_score=mixture_score,
                    cue_mixture_threshold=float(args.cue_mixture_threshold),
                    signed_density_score=signed_density,
                    signed_density_threshold=float(args.signed_score_threshold),
                    signed_density_max_sources=(
                        None
                        if args.signed_score_max_sources is None
                        or int(args.signed_score_max_sources) == 0
                        else int(args.signed_score_max_sources)
                    ),
                    signed_density_prune_threshold=(
                        float(args.signed_score_prune_threshold)
                        if args.signed_score_prune_threshold is not None
                        else None
                    ),
                    signed_density_prune_min_age_frames=int(
                        args.signed_score_prune_min_age_frames
                    ),
                    black_child_prune_threshold=(
                        float(args.black_child_prune_threshold)
                        if args.density_policy
                        == "active_oscd_cue_mixture_black_child_prune"
                        else None
                    ),
                    black_child_prune_min_age_frames=int(
                        args.black_child_prune_min_age_frames
                    ),
                )
                density_event.update(
                    {
                        "initial_gaussian_count": density_result.initial_count,
                        "final_gaussian_count": density_result.final_count,
                        "clone_source_count": density_result.clone_source_count,
                        "clone_child_count": density_result.clone_child_count,
                        "split_source_count": density_result.split_source_count,
                        "split_child_count": density_result.split_child_count,
                        "gradient_split_source_count": (
                            density_result.gradient_split_source_count
                        ),
                        "cue_mixture_split_source_count": (
                            density_result.cue_mixture_split_source_count
                        ),
                        "cue_mixture_only_split_source_count": (
                            density_result.cue_mixture_only_split_source_count
                        ),
                        "signed_score_candidate_count": (
                            density_result.signed_score_candidate_count
                        ),
                        "signed_score_clone_source_count": (
                            density_result.signed_score_clone_source_count
                        ),
                        "signed_score_split_source_count": (
                            density_result.signed_score_split_source_count
                        ),
                        "signed_score_prune_candidate_count": (
                            density_result.signed_score_prune_candidate_count
                        ),
                        "signed_score_pruned_count": (
                            density_result.signed_score_pruned_count
                        ),
                        "split_source_removed_count": density_result.split_source_removed_count,
                        "opacity_pruned_count": density_result.opacity_pruned_count,
                        "size_pruned_count": density_result.size_pruned_count,
                        "black_child_prune_candidate_count": (
                            density_result.black_child_prune_candidate_count
                        ),
                        "black_child_pruned_count": (
                            density_result.black_child_pruned_count
                        ),
                        "black_child_retained_for_support_count": (
                            density_result.black_child_retained_for_support_count
                        ),
                        "total_removed_count": density_result.total_removed_count,
                    }
                )
                if signed_score is not None:
                    density_event.update(
                        {
                            "signed_score_positive_count": int(
                                (signed_score.score > float(args.signed_score_threshold))
                                .sum()
                                .item()
                            ),
                            "signed_score_negative_count": int(
                                (signed_score.score < -float(args.signed_score_threshold))
                                .sum()
                                .item()
                            ),
                            "signed_score_stats": quantile_summary(signed_score.score),
                            "signed_score_support_stats": quantile_summary(
                                signed_score.sign_support
                            ),
                            "signed_score_age_stats": quantile_summary(signed_score.age),
                        }
                    )
                signal = (
                    signed_score.score
                    if signed_score is not None
                    else gradient_signal
                )
                signal_name = (
                    "signed_lifespan_score"
                    if signed_score is not None
                    else "screen_gradient"
                )
                frame_source_rows = _density_source_rows(
                    current_view,
                    pre_density_xyz,
                    pre_density_stable_id,
                    pre_density_generation,
                    density_result.masks,
                    selection_signal=signal,
                    selection_signal_name=signal_name,
                    signed=signed_score,
                )
                if signed_score is not None:
                    already_logged = (
                        density_result.masks.clone
                        | density_result.masks.split
                        | density_result.masks.signed_score_prune
                    )
                    frame_source_rows.extend(
                        _signed_negative_suppression_rows(
                            current_view,
                            pre_density_xyz,
                            pre_density_stable_id,
                            pre_density_generation,
                            signed_score,
                            threshold=float(args.signed_score_threshold),
                            max_rows=(
                                None
                                if args.signed_score_max_sources is None
                                or int(args.signed_score_max_sources) == 0
                                else int(args.signed_score_max_sources)
                            ),
                            already_logged=already_logged,
                        )
                    )
                for row in frame_source_rows:
                    row["timestamp"] = int(timestamp)
                    row["frame"] = record.name
                    signed_density_source_rows.append(row)
                total_clones += density_result.clone_child_count
                total_split_sources += density_result.split_source_count
                total_split_children += density_result.split_child_count
                total_pruned += density_result.total_removed_count
                total_opacity_pruned += density_result.opacity_pruned_count
                total_size_pruned += density_result.size_pruned_count
                total_black_child_prune_candidates += (
                    density_result.black_child_prune_candidate_count
                )
                total_black_children_pruned += density_result.black_child_pruned_count
                total_black_children_retained_for_support += (
                    density_result.black_child_retained_for_support_count
                )
                total_signed_score_candidates += density_result.signed_score_candidate_count
                total_signed_score_clone_sources += (
                    density_result.signed_score_clone_source_count
                )
                total_signed_score_split_sources += (
                    density_result.signed_score_split_source_count
                )
                total_signed_score_prune_candidates += (
                    density_result.signed_score_prune_candidate_count
                )
                total_signed_score_pruned += density_result.signed_score_pruned_count

        density_event["sampled_training_view_indices"] = sampled_view_indices
        density_event["future_training_view_access_count"] = int(
            sum(index > timestamp for index in sampled_view_indices)
        )
        density_rows.append(density_event)
        if density_event["future_training_view_access_count"]:
            raise RuntimeError("online replay accessed a future view")

        # The hard post render closes the growth-replay loop.  Do not replace
        # it with candidate uncertainty, which must not train the representation.
        post_package = _render_dynamic_change(
            current_view,
            model,
            pipe,
            background,
            timestamp=float(timestamp),
            include_never_open_occluders=(
                _includes_never_open(args.render_support_mode)
            ),
            train_never_open_dc_opacity=_trains_never_open_appearance(
                args.render_support_mode
            ),
            black_never_open_occluders=_uses_black_never_open_occluders(
                args.render_support_mode
            ),
            change_color_mode=args.change_color_mode,
        )
        post_output_package = post_package
        candidate_transition_progress = model.xyz.new_empty((0,))
        candidate_soft_change_weight = model.xyz.new_empty((0,))
        candidate_soft_opening_count = 0
        candidate_soft_closing_count = 0
        if args.candidate_render_gate != "none":
            post_active = topology.active_mask()
            post_close_candidate_mask = (
                _single_candidate_close_candidate_mask(
                    tracker,
                    post_active,
                    close_probability=float(controller.config.close_probability),
                )
                if args.detector_mode == "single_candidate_beta"
                else None
            )
            post_transition_progress, post_soft_change_weight, soft_gated_rows = (
                _candidate_render_gate_state(
                    args.candidate_render_gate,
                    post_active,
                    tracker.candidate_active,
                    tracker.last_log_bayes_factor,
                    bayes_factor_threshold=args.candidate_bayes_factor_threshold,
                    close_candidate_mask=post_close_candidate_mask,
                )
            )
            candidate_transition_progress = post_transition_progress[
                soft_gated_rows
            ]
            candidate_soft_change_weight = post_soft_change_weight[soft_gated_rows]
            candidate_soft_opening_count = int(
                (soft_gated_rows & ~post_active).sum().item()
            )
            candidate_soft_closing_count = int(
                (soft_gated_rows & post_active).sum().item()
            )
            if bool(soft_gated_rows.any()):
                post_output_package = _render_dynamic_change(
                    current_view,
                    model,
                    pipe,
                    background,
                    timestamp=float(timestamp),
                    include_never_open_occluders=(
                        _includes_never_open(args.render_support_mode)
                    ),
                    train_never_open_dc_opacity=_trains_never_open_appearance(
                        args.render_support_mode
                    ),
                    black_never_open_occluders=_uses_black_never_open_occluders(
                        args.render_support_mode
                    ),
                    change_color_mode=args.change_color_mode,
                    soft_change_weight=post_soft_change_weight,
                )
        post_prediction, _ = _prediction_from_package(
            post_output_package, config.evaluation_threshold
        )
        predictions.append(post_prediction)
        hard_post_prediction, _ = _prediction_from_package(
            post_package, config.evaluation_threshold
        )
        hard_output_predictions.append(hard_post_prediction)
        post_change_probability = _render_change_probability(
            post_package["render"].detach()
        )
        current_growth_map = _positive_growth_map(
            pre_change_probability, post_change_probability
        )
        growth_mean = float(current_growth_map.mean().item())
        growth_max = float(current_growth_map.max().item())
        growth_nonzero_fraction = float(
            current_growth_map.gt(0.0).to(dtype=torch.float32).mean().item()
        )
        growth_map_means.append(growth_mean)
        growth_map_maxima.append(growth_max)
        growth_map_nonzero_fractions.append(growth_nonzero_fraction)
        if args.loss_regularization_mode == "local_growth_replay":
            previous_growth_view = current_view
            previous_growth_map = current_growth_map
        first_open_post_frame, first_open_source_survivors = (
            _surviving_intrinsic_dc_render_value(
                topology, first_open_initialization["stable_ids"]
            )
        )
        total_first_open_source_survivors += first_open_source_survivors
        agreement_candidate_post = model.change_dc.new_empty((0,))
        agreement_candidate_gap_before = model.change_dc.new_empty((0,))
        agreement_candidate_gap_after = model.change_dc.new_empty((0,))
        agreement_candidate_gap_delta = model.change_dc.new_empty((0,))
        agreement_candidate_anchor_gap = model.change_dc.new_empty((0,))
        agreement_candidate_anchor_drift = model.change_dc.new_empty((0,))
        if agreement_candidate_snapshot is not None:
            surviving_rows, surviving = _surviving_row_positions(
                topology, agreement_candidate_snapshot["stable_ids"]
            )
            agreement_candidate_post = _learned_dc_change_magnitude(
                model, surviving_rows
            )
            surviving_cue = agreement_candidate_snapshot["cue_ratio"][surviving]
            surviving_before = agreement_candidate_snapshot["dc_before"][surviving]
            agreement_candidate_gap_before = (
                surviving_before - surviving_cue
            ).abs()
            agreement_candidate_gap_after = (
                agreement_candidate_post - surviving_cue
            ).abs()
            agreement_candidate_gap_delta = (
                agreement_candidate_gap_after - agreement_candidate_gap_before
            )
            if "anchor" in agreement_candidate_snapshot:
                surviving_anchor = agreement_candidate_snapshot["anchor"][surviving]
                current_anchor = tracker.candidate_anchor[surviving_rows]
                agreement_candidate_anchor_gap = (
                    surviving_anchor - surviving_cue
                ).abs()
                agreement_candidate_anchor_drift = (
                    current_anchor - surviving_anchor
                ).abs()
                agreement_candidate_anchor_gap_values.append(
                    agreement_candidate_anchor_gap
                )
                agreement_candidate_anchor_drift_values.append(
                    agreement_candidate_anchor_drift
                )
            agreement_candidate_gap_before_values.append(
                agreement_candidate_gap_before
            )
            agreement_candidate_gap_after_values.append(
                agreement_candidate_gap_after
            )
            agreement_candidate_gap_delta_values.append(
                agreement_candidate_gap_delta
            )
            agreement_candidate_source_survivors += int(surviving.sum().item())
        first_open_dc_values["after_first_update"].append(
            first_open_after_first_update
        )
        first_open_dc_values["before_density"].append(first_open_before_density)
        first_open_dc_values["post_frame_surviving"].append(first_open_post_frame)
        if visual_root is not None:
            stem = Path(record.name).stem
            _save_rgb(
                visual_root / "raw_render" / f"{stem}.png",
                _render_to_rgb_u8(post_output_package["render"]),
            )
            _save_binary_mask(
                visual_root / "thresholded_render" / f"{stem}.png",
                post_prediction,
            )
            assert event_rgb is not None
            _save_rgb(
                visual_root / "event_render" / f"{stem}.png",
                event_rgb,
            )
        selected_count = (
            int(torch.unique(torch.cat(selected_stable_ids)).numel())
            if selected_stable_ids
            else 0
        )
        active_visible_count = (
            int(torch.unique(torch.cat(active_visible_stable_ids)).numel())
            if active_visible_stable_ids
            else 0
        )
        never_open_selected_count = (
            int(torch.unique(torch.cat(selected_never_open_stable_ids)).numel())
            if selected_never_open_stable_ids
            else 0
        )
        counts = binary["action_counts"]
        active_rows_post = topology.active_mask()
        active_count = int(active_rows_post.sum().item())
        active_dc_render = _active_change_color_value(
            model,
            torch.nonzero(active_rows_post, as_tuple=False).flatten(),
            args.change_color_mode,
        )
        active_dc_below_half_count = int((active_dc_render < 0.5).sum().item())
        frame_rows.append(
            {
                "timestamp": timestamp,
                "frame": record.name,
                "scope": args.scope,
                "gaussian_count": topology.count,
                "observed_gaussian_count": int(binary["observed_count"]),
                "positive_pseudocount_mass": float(evidence.delta_a.sum().item()),
                "negative_pseudocount_mass": float(evidence.delta_b.sum().item()),
                "open_count": int(counts["OPEN"]),
                "keep_count": int(counts["KEEP"]),
                "close_count": int(counts["CLOSE"]),
                "none_count": int(counts["NONE"]),
                "uncertain_count": int(counts.get("UNCERTAIN", counts.get("HOLD", 0))),
                "reopen_count": int(
                    sum(
                        event.action == "OPEN" and event.new_current_slot > 0
                        for event in raw_events
                    )
                ),
                "p_active_stats": binary["p_active_stats"],
                "p_flip_stats": binary["p_flip_stats"],
                "p_01_stats": binary["p_01_stats"],
                "p_10_stats": binary["p_10_stats"],
                "q_stats": binary["q_stats"],
                "expected_change_stats": binary["expected_change_stats"],
                "agreement_flip_fraction_stats": binary[
                    "agreement_flip_fraction_stats"
                ],
                "candidate_started_count": int(binary["candidate_started_count"]),
                "candidate_continued_count": int(binary["candidate_continued_count"]),
                "candidate_rejected_count": int(binary["candidate_rejected_count"]),
                "candidate_committed_count": int(binary["candidate_committed_count"]),
                "candidate_live_count": int(binary["candidate_live_count"]),
                "candidate_render_gate": args.candidate_render_gate,
                "candidate_soft_opening_count": candidate_soft_opening_count,
                "candidate_soft_closing_count": candidate_soft_closing_count,
                "candidate_transition_progress_stats": quantile_summary(
                    candidate_transition_progress
                ),
                "candidate_soft_change_weight_stats": quantile_summary(
                    candidate_soft_change_weight
                ),
                "agreement_live_open_source_survivor_count": int(
                    agreement_candidate_post.numel()
                ),
                "agreement_live_open_dc_before_stats": quantile_summary(
                    (
                        agreement_candidate_snapshot["dc_before"]
                        if agreement_candidate_snapshot is not None
                        else model.change_dc.new_empty((0,))
                    )
                ),
                "agreement_live_open_dc_post_stats": quantile_summary(
                    agreement_candidate_post
                ),
                "agreement_live_open_gap_before_stats": quantile_summary(
                    agreement_candidate_gap_before
                ),
                "agreement_live_open_gap_after_stats": quantile_summary(
                    agreement_candidate_gap_after
                ),
                "agreement_live_open_gap_delta_stats": quantile_summary(
                    agreement_candidate_gap_delta
                ),
                "agreement_live_open_anchor_gap_stats": quantile_summary(
                    agreement_candidate_anchor_gap
                ),
                "agreement_live_open_anchor_drift_stats": quantile_summary(
                    agreement_candidate_anchor_drift
                ),
                "candidate_support_observation_stats": binary[
                    "candidate_support_observation_stats"
                ],
                "directional_support_fraction_stats": binary[
                    "directional_support_fraction_stats"
                ],
                "candidate_log_bayes_factor_stats": binary[
                    "candidate_log_bayes_factor_stats"
                ],
                "terminal_candidate_log_bayes_factor_stats": binary[
                    "terminal_candidate_log_bayes_factor_stats"
                ],
                "terminal_candidate_duration_stats": binary[
                    "terminal_candidate_duration_stats"
                ],
                "terminal_candidate_support_observation_stats": binary[
                    "terminal_candidate_support_observation_stats"
                ],
                "committed_candidate_support_observation_stats": binary[
                    "committed_candidate_support_observation_stats"
                ],
                "stable_concentration_stats": binary[
                    "stable_concentration_stats"
                ],
                "active_gaussian_count": active_count,
                "optimizer_selected_open_rows": selected_count,
                "optimizer_selected_active_visible_rows": active_visible_count,
                "optimizer_selected_never_open_visible_rows": never_open_selected_count,
                "first_open_dc_initialized_count": int(
                    first_open_initialization["rows"].numel()
                ),
                "first_open_dc_adam_reset_count": int(
                    first_open_initialization["adam_reset_count"]
                ),
                "first_open_dc_render_before_stats": quantile_summary(
                    first_open_initialization["before"]
                ),
                "first_open_dc_render_immediate_stats": quantile_summary(
                    first_open_initialization["immediate"]
                ),
                "first_open_dc_raw_abs_jump_stats": quantile_summary(
                    first_open_initialization["raw_abs_jump"]
                ),
                "first_open_dc_render_after_first_update_stats": quantile_summary(
                    first_open_after_first_update
                ),
                "first_open_dc_render_before_density_stats": quantile_summary(
                    first_open_before_density
                ),
                "first_open_source_surviving_post_frame_count": int(
                    first_open_source_survivors
                ),
                "first_open_dc_render_post_frame_surviving_stats": quantile_summary(
                    first_open_post_frame
                ),
                "clone_count": int(density_event["clone_child_count"]),
                "split_source_count": int(density_event["split_source_count"]),
                "split_child_count": int(density_event["split_child_count"]),
                "cue_mixture_split_source_count": int(
                    density_event["cue_mixture_split_source_count"]
                ),
                "cue_mixture_only_split_source_count": int(
                    density_event["cue_mixture_only_split_source_count"]
                ),
                "signed_score_candidate_count": int(
                    density_event["signed_score_candidate_count"]
                ),
                "signed_score_clone_source_count": int(
                    density_event["signed_score_clone_source_count"]
                ),
                "signed_score_split_source_count": int(
                    density_event["signed_score_split_source_count"]
                ),
                "signed_score_prune_candidate_count": int(
                    density_event["signed_score_prune_candidate_count"]
                ),
                "signed_score_pruned_count": int(
                    density_event["signed_score_pruned_count"]
                ),
                "signed_score_positive_count": int(
                    density_event["signed_score_positive_count"]
                ),
                "signed_score_negative_count": int(
                    density_event["signed_score_negative_count"]
                ),
                "signed_score_p_plus_is_new": (
                    density_event["signed_score_p_plus_is_new"]
                ),
                "signed_score_stats": density_event["signed_score_stats"],
                "signed_score_support_stats": density_event[
                    "signed_score_support_stats"
                ],
                "signed_score_age_stats": density_event["signed_score_age_stats"],
                "pruned_count": int(density_event["total_removed_count"]),
                "black_child_pruned_count": int(
                    density_event["black_child_pruned_count"]
                ),
                "black_child_retained_for_support_count": int(
                    density_event["black_child_retained_for_support_count"]
                ),
                "pre_predicted_positive_fraction": float(pre_prediction.mean()),
                "post_predicted_positive_fraction": float(post_prediction.mean()),
                "loss_regularization_mode": args.loss_regularization_mode,
                "growth_replay_joint_steps": int(
                    frame_growth_replay_joint_steps
                ),
                "growth_replay_raw_loss_mean": (
                    float(np.mean(frame_growth_replay_values))
                    if frame_growth_replay_values
                    else 0.0
                ),
                "positive_growth_map_mean": growth_mean,
                "positive_growth_map_max": growth_max,
                "positive_growth_map_nonzero_fraction": growth_nonzero_fraction,
                "active_dc_intrinsic_render_stats": quantile_summary(
                    active_dc_render
                ),
                "active_dc_below_half_count": active_dc_below_half_count,
                "active_dc_below_half_fraction": (
                    active_dc_below_half_count / active_count
                    if active_count
                    else 0.0
                ),
                "frame_runtime_seconds": time.time() - frame_started,
                "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
                "pre_iou": None,
                "pre_f1": None,
                "iou": None,
                "f1": None,
                "precision": None,
                "recall": None,
            }
        )

    hard_output_frame_rows = [dict(row) for row in frame_rows]
    metrics = evaluate_after_inference(
        args.source_path, records, pre_predictions, predictions, frame_rows
    )
    hard_output_metrics = evaluate_after_inference(
        args.source_path,
        records,
        hard_pre_output_predictions,
        hard_output_predictions,
        hard_output_frame_rows,
    )
    for row, hard_row in zip(frame_rows, hard_output_frame_rows, strict=True):
        row["hard_output_pre_iou"] = hard_row["pre_iou"]
        row["hard_output_pre_f1"] = hard_row["pre_f1"]
        row["hard_output_iou"] = hard_row["iou"]
        row["hard_output_f1"] = hard_row["f1"]
        row["candidate_gate_iou_delta"] = row["iou"] - hard_row["iou"]
        row["candidate_gate_f1_delta"] = row["f1"] - hard_row["f1"]
    frame_iou_deltas = np.asarray(
        [row["candidate_gate_iou_delta"] for row in frame_rows], dtype=np.float64
    )
    frame_f1_deltas = np.asarray(
        [row["candidate_gate_f1_delta"] for row in frame_rows], dtype=np.float64
    )
    candidate_gate_metric_delta = {
        "mean_frame_iou": float(
            metrics["mean_frame_iou"] - hard_output_metrics["mean_frame_iou"]
        ),
        "mean_frame_f1": float(
            metrics["mean_frame_f1"] - hard_output_metrics["mean_frame_f1"]
        ),
        "precision": float(metrics["precision"] - hard_output_metrics["precision"]),
        "recall": float(metrics["recall"] - hard_output_metrics["recall"]),
        "tp": int(metrics["tp"] - hard_output_metrics["tp"]),
        "tn": int(metrics["tn"] - hard_output_metrics["tn"]),
        "fp": int(metrics["fp"] - hard_output_metrics["fp"]),
        "fn": int(metrics["fn"] - hard_output_metrics["fn"]),
        "frame_iou_wins": int((frame_iou_deltas > 0.0).sum()),
        "frame_iou_ties": int((frame_iou_deltas == 0.0).sum()),
        "frame_iou_losses": int((frame_iou_deltas < 0.0).sum()),
        "frame_f1_wins": int((frame_f1_deltas > 0.0).sum()),
        "frame_f1_ties": int((frame_f1_deltas == 0.0).sum()),
        "frame_f1_losses": int((frame_f1_deltas < 0.0).sum()),
    }
    closed_result = closed_audit.verify()
    topology.validate()
    if not closed_result["passed"]:
        raise RuntimeError(f"CLOSED rows drifted under dynamic topology: {closed_result}")
    if inactive_gradient_violations:
        raise RuntimeError(
            f"inactive/off-view parameter gradients detected: {inactive_gradient_violations}"
        )
    event_diag = event_diagnostics(lifecycle_events)
    if event_diag["active_to_active_false_split_count"] or event_diag["reused_slot_violations"]:
        raise RuntimeError(f"lifespan invariant failed: {event_diag}")

    diagnostic_boundaries = (95, 199) if args.scope == "continuous" else ()
    summary = {
        "schema_version": 1,
        "script": "experiments/run_online_dynamic_active_oscd_density.py",
        "contract": (
            "equal_status_mutable_rchange_never_open_appearance"
            if _trains_never_open_appearance(args.render_support_mode)
            else "equal_status_mutable_rchange_active_only_oscd_density"
        ),
        "scope": SCOPE_LABELS[args.scope],
        "scope_key": args.scope,
        "frames": len(records),
        "run_config": asdict(config),
        "detector": {
            "algorithm": getattr(tracker, "algorithm", "direct_binary_state_filter"),
            "mode": args.detector_mode,
            "evidence": (
                "cue-positive/negative alpha-T mass reinterpreted as FLIP/KEEP against the pre-update lifespan bit"
                if args.detector_mode == "lifespan_gate_beta"
                else (
                    (
                        "cue mass compared with candidate-start detached learned-DC anchor as soft FLIP/KEEP evidence"
                        if args.detector_mode == "anchored_dc_agreement_beta"
                        else "cue-positive/negative alpha-T mass compared with clamped intrinsic learned DC as soft FLIP/KEEP evidence"
                    )
                    if args.detector_mode in LEARNED_DC_AGREEMENT_MODES
                    else "current mutable-bank alpha-T capped pseudo-counts without lifespan opacity gating"
                )
            ),
            "probe_scaling": args.detector_probe_scaling,
            "probe_scaling_scope": "detector alpha-T evidence only; representation scales unchanged",
            "observation_timing": "once per newly arrived frame before lifecycle mutation and representation optimization",
            "replay_used_as_detector_evidence": False,
            "post_optimization_render_used_as_detector_evidence": False,
            "learned_dc_used_as_detector_evidence": args.detector_mode
            in LEARNED_DC_AGREEMENT_MODES,
            "representation_feedback": True,
            "dc_feedback": args.detector_mode in LEARNED_DC_AGREEMENT_MODES,
            "geometry_opacity_feedback": True,
            "inactive_rows_observable": True,
            "new_row_state": "one-time copy of source posterior/count/timestamp and controller counters",
            "post_birth_behavior": "independent evidence and posterior updates",
            "single_candidate_beta": (
                {
                    "prior_a": 1.0,
                    "prior_b": 1.0,
                    "bayes_factor_threshold": float(
                        args.candidate_bayes_factor_threshold
                    ),
                    "log_bayes_factor_threshold": math.log(
                        float(args.candidate_bayes_factor_threshold)
                    ),
                    "score": "log marginal RESET minus log marginal KEEP for the exact candidate block",
                    "candidate_policy": "freeze stable; reject merges block; commit replaces stable",
                    "hazard": None,
                    "run_length_posterior": False,
                    "transition_probability": False,
                    "same_label_reset": "Beta run reset only; representation slot preserved",
                }
                if args.detector_mode == "single_candidate_beta"
                else None
            ),
            "lifespan_gate_beta": (
                {
                    "stable_flip_prior": float(args.gate_stable_flip_prior),
                    "stable_keep_prior": float(args.gate_stable_keep_prior),
                    "reset_flip_prior": float(args.gate_reset_flip_prior),
                    "reset_keep_prior": float(args.gate_reset_keep_prior),
                    "bayes_factor_threshold": float(
                        args.candidate_bayes_factor_threshold
                    ),
                    "log_bayes_factor_threshold": math.log(
                        float(args.candidate_bayes_factor_threshold)
                    ),
                    "closed_evidence": "cue-positive is FLIP; cue-negative is KEEP",
                    "open_evidence": "cue-negative is FLIP; cue-positive is KEEP",
                    "commit": "toggle lifespan bit and reinterpret old FLIP block as new KEEP support",
                    "run_length_posterior": False,
                    "transition_probability": False,
                }
                if args.detector_mode == "lifespan_gate_beta"
                else None
            ),
            "learned_dc_agreement_beta": (
                {
                    "stable_flip_prior": float(args.agreement_stable_flip_prior),
                    "stable_keep_prior": float(args.agreement_stable_keep_prior),
                    "reset_flip_prior": float(args.agreement_reset_flip_prior),
                    "reset_keep_prior": float(args.agreement_reset_keep_prior),
                    "bayes_factor_threshold": float(
                        args.candidate_bayes_factor_threshold
                    ),
                    "log_bayes_factor_threshold": math.log(
                        float(args.candidate_bayes_factor_threshold)
                    ),
                    "expected_change": "clamp(2*(mean(SH2RGB(change_dc))-0.5), 0, 1)",
                    "flip_formula": "q*(1-c) + (1-q)*c",
                    "keep_formula": "q*c + (1-q)*(1-c)",
                    "candidate_dc_policy": args.agreement_candidate_dc_policy,
                    "commit_dc_reset": "OPEN->rendered white 1, CLOSE->raw DC zero (rendered neutral 0.5), DC Adam row reset",
                    "run_length_posterior": False,
                    "transition_probability": False,
                }
                if args.detector_mode == "learned_dc_agreement_beta"
                else None
            ),
            "anchored_dc_agreement_beta": (
                {
                    "stable_flip_prior": float(args.agreement_stable_flip_prior),
                    "stable_keep_prior": float(args.agreement_stable_keep_prior),
                    "reset_flip_prior": float(args.agreement_reset_flip_prior),
                    "reset_keep_prior": float(args.agreement_reset_keep_prior),
                    "bayes_factor_threshold": float(
                        args.candidate_bayes_factor_threshold
                    ),
                    "log_bayes_factor_threshold": math.log(
                        float(args.candidate_bayes_factor_threshold)
                    ),
                    "current_change": "clamp(2*(mean(SH2RGB(change_dc))-0.5), 0, 1)",
                    "candidate_anchor": "stop_gradient(current_change) at candidate start; topology-aligned until reject/commit",
                    "flip_formula": "q*(1-C_anchor) + (1-q)*C_anchor",
                    "keep_formula": "q*C_anchor + (1-q)*(1-C_anchor)",
                    "confirmation_views": int(args.agreement_confirmation_views),
                    "directional_margin": float(args.agreement_directional_margin),
                    "directional_support": "CLOSED: q > C_anchor + margin; OPEN: q < C_anchor - margin",
                    "support_policy": "every observed candidate view must support the same lifecycle direction",
                    "candidate_dc_policy": "adapt; optimizer never reads or freezes for C_anchor",
                    "commit_dc_reset": "OPEN->rendered white 1, CLOSE->raw DC zero (rendered neutral 0.5), DC Adam row reset",
                    "run_length_posterior": False,
                    "transition_probability": False,
                }
                if args.detector_mode == "anchored_dc_agreement_beta"
                else None
            ),
        },
        "representation": {
            "gaussian_classes": "none; every row has equal mutable status",
            "render_support_mode": args.render_support_mode,
            "never_open_color_override": (
                "RGB zero via degree-zero SH override"
                if _uses_black_never_open_occluders(args.render_support_mode)
                else None
            ),
            "change_color_mode": args.change_color_mode,
            "candidate_render_gate": {
                "mode": args.candidate_render_gate,
                "enabled": args.candidate_render_gate != "none",
                "progress": (
                    "clamp(live_candidate_log_bf / log(commit_bf), 0, 1)"
                    if args.candidate_render_gate != "none"
                    else None
                ),
                "soft_change_weight": (
                    "(1-progress)*committed_active + progress*(1-committed_active)"
                    if args.candidate_render_gate == "log_bf_progress"
                    else (
                        "committed_active*(1-progress); OPEN candidates stay hidden"
                        if args.candidate_render_gate
                        == "close_only_log_bf_progress"
                        else None
                    )
                ),
                "direction": (
                    "fresh reset-candidate Beta posterior <= controller close threshold"
                    if args.detector_mode == "single_candidate_beta"
                    and args.candidate_render_gate
                    == "close_only_log_bf_progress"
                    else (
                        "opposite of the committed lifespan bit"
                        if args.candidate_render_gate != "none"
                        else None
                    )
                ),
                "scope": "current-frame pre/post metric and visualization renders only",
                "detector_input": "ungated raw mutable-bank alpha-T evidence",
                "training_and_density": "hard committed lifespan gate only",
                "hard_commit": "unchanged BF threshold lifecycle transition",
            },
            "change_color_contract": (
                "precomputed RGB one for every OPEN row; RGB zero for fixed NEVER_OPEN occluders; CLOSED hidden"
                if args.change_color_mode == "lifespan_gate"
                else "learned persistent degree-zero SH DC"
            ),
            "first_open_dc_initialization": {
                "mode": args.first_open_dc_initialization,
                "scope": (
                    "agreement controller resets every committed OPEN to rendered one and every CLOSE to raw DC zero"
                    if args.detector_mode in LEARNED_DC_AGREEMENT_MODES
                    else "explicit first OPEN (slot zero) only; REOPEN preserves DC and moments"
                ),
                "render_one_raw_value": float(
                    RGB2SH(torch.tensor(1.0)).item()
                ),
                "adam_state_reset": (
                    "DC row state only"
                    if args.first_open_dc_initialization == "render_one"
                    else "none"
                ),
                "detector_timing": "current-frame evidence is accumulated before initialization",
            },
            "render_support_contract": (
                "OPEN union NEVER_OPEN; NEVER_OPEN DC/opacity trainable; CLOSED hidden"
                if _trains_never_open_appearance(args.render_support_mode)
                else (
                    (
                        "OPEN semantic-one union fixed semantic-zero NEVER_OPEN; CLOSED hidden"
                        if args.change_color_mode == "lifespan_gate"
                        else (
                            "OPEN learned DC union frozen RGB-zero NEVER_OPEN occluders; CLOSED hidden"
                            if _uses_black_never_open_occluders(
                                args.render_support_mode
                            )
                            else "OPEN union frozen raw-zero-DC (rendered RGB 0.5) NEVER_OPEN occluders; CLOSED hidden"
                        )
                    )
                    if _includes_never_open(args.render_support_mode)
                    else "OPEN only"
                )
            ),
            "lifecycle_render_distinction": (
                "OPEN vs NEVER_OPEN vs CLOSED"
                if _includes_never_open(args.render_support_mode)
                else "OPEN vs non-OPEN"
            ),
            "parameters": list(DIRECT_PARAMETER_NAMES),
            "optimizer_selection": args.optimizer_selection,
            "optimizer_mask": (
                (
                    "all OPEN rows selected independent of sampled-view visibility; DC/SH-rest receive no renderer gradient under semantic override; NEVER_OPEN frozen"
                    if args.change_color_mode == "lifespan_gate"
                    else (
                        "all OPEN rows selected independent of sampled-view visibility; live-candidate DC rows excluded only under agreement freeze policy; NEVER_OPEN frozen"
                        if args.detector_mode == "learned_dc_agreement_beta"
                        else "all OPEN rows for every parameter, independent of sampled-view visibility; NEVER_OPEN frozen"
                    )
                )
                if args.optimizer_selection == "all_open"
                else (
                    "OPEN-visible all attributes; NEVER_OPEN-visible DC/opacity only"
                    if _trains_never_open_appearance(args.render_support_mode)
                    else "currently ACTIVE and visible in sampled causal training view"
                )
            ),
            "closed_behavior": (
                "hidden and exact parameter/Adam preservation between boundaries; intentional raw-zero DC/Adam reset on CLOSE commit"
                if args.detector_mode in LEARNED_DC_AGREEMENT_MODES
                else "hidden and exact parameter/Adam preservation"
            ),
            "features_rest_note": (
                "semantic precomputed color bypasses all SH coefficients"
                if args.change_color_mode == "lifespan_gate"
                else "SH degree 0 keeps features_rest gradient exactly zero"
            ),
        },
        "learned_dc_agreement_diagnostics": {
            "enabled": args.detector_mode in LEARNED_DC_AGREEMENT_MODES,
            "mode": args.detector_mode,
            "candidate_dc_policy": (
                "adapt"
                if args.detector_mode == "anchored_dc_agreement_beta"
                else args.agreement_candidate_dc_policy
            ),
            "dc_commit_reset_count": int(
                getattr(controller, "dc_commit_reset_count", 0)
            ),
            "live_open_source_survivor_count": int(
                agreement_candidate_source_survivors
            ),
            "live_open_gap_before": quantile_summary(
                _concat_cpu(agreement_candidate_gap_before_values)
            ),
            "live_open_gap_after": quantile_summary(
                _concat_cpu(agreement_candidate_gap_after_values)
            ),
            "live_open_gap_delta": quantile_summary(
                _concat_cpu(agreement_candidate_gap_delta_values)
            ),
            "live_open_anchor_cue_gap": quantile_summary(
                _concat_cpu(agreement_candidate_anchor_gap_values)
            ),
            "live_open_anchor_optimizer_drift": quantile_summary(
                _concat_cpu(agreement_candidate_anchor_drift_values)
            ),
            "max_suppressed_candidate_dc_gradient": float(
                max_suppressed_candidate_dc_gradient
            ),
        },
        "loss_regularization": {
            "mode": args.loss_regularization_mode,
            "global_formula": (
                "log(1 + mean(sigmoid(mean_rgb(render)))^2)"
                if args.loss_regularization_mode == "global"
                else None
            ),
            "local_formula": (
                "mean((1 - clamp(cue/local_support_scale,0,1)) * probability)"
                if args.loss_regularization_mode == "local_growth_replay"
                else None
            ),
            "growth_replay_formula": (
                "growth_weight * mean(G_prev * (1 - local_support_prev) * probability_prev_current)"
                if args.loss_regularization_mode == "local_growth_replay"
                else None
            ),
            "local_support_scale": float(args.local_support_scale),
            "growth_replay_weight": float(args.growth_replay_weight),
            "growth_map_definition": "relu(post_frame_probability - pre_frame_probability)",
            "growth_map_excludes_same_frame_lifecycle_and_first_open_jump": True,
            "joint_step_policy": "when the sampled training view is current, jointly rerender t-1 at current lifespan timestamp",
            "total_growth_replay_joint_steps": int(
                total_growth_replay_joint_steps
            ),
            "raw_growth_replay_loss_mean": (
                float(np.mean(growth_replay_values))
                if growth_replay_values
                else 0.0
            ),
            "positive_growth_map_mean": float(np.mean(growth_map_means)),
            "positive_growth_map_max": float(max(growth_map_maxima)),
            "positive_growth_map_nonzero_fraction_mean": float(
                np.mean(growth_map_nonzero_fractions)
            ),
        },
        "density_control": {
            "policy": args.density_policy,
            "updates_per_frame": int(args.updates_per_frame),
            "local_update_index": int(args.densify_update_index),
            "densification": (
                "current-frame signed SAM-diff alpha-T support times inverse active-lifespan age; positive NEW-aligned scores clone/split and negative scores suppress densification"
                if args.density_policy == "active_signed_lifespan_score"
                else (
                    "original online O-SCD ACTIVE-only gradient clone/split plus current-frame pre-optimization raw-cue mixture splitting for large ACTIVE rows"
                    if args.density_policy in _CUE_MIXTURE_DENSITY_POLICIES
                    else "original online O-SCD gradient-only clone/split restricted to ACTIVE rows"
                )
            ),
            "signed_lifespan_score": {
                "enabled": args.density_policy == "active_signed_lifespan_score",
                "formula": "active * (2*p_plus_is_new - 1) * (plus_alphaT - minus_alphaT) / (timestamp - current_lifespan_start + 1)",
                "artifact": str(args.signed_score_artifact)
                if args.signed_score_artifact is not None
                else None,
                "posterior_key": (
                    signed_artifact.posterior_key
                    if signed_artifact is not None
                    else None
                ),
                "sam_model": args.signed_score_sam_model,
                "threshold": float(args.signed_score_threshold),
                "max_sources": (
                    None
                    if args.signed_score_max_sources is None
                    or int(args.signed_score_max_sources) == 0
                    else int(args.signed_score_max_sources)
                ),
                "negative_prune_threshold": (
                    float(args.signed_score_prune_threshold)
                    if args.signed_score_prune_threshold is not None
                    else None
                ),
                "negative_prune_scope": "optional generation>0 ACTIVE descendants only; generation-zero rows are protected",
                "detector_input": "unchanged current-frame pre-optimization raw O-SCD alpha-T evidence; signed score never updates lifecycle posterior",
                "frame_name_policy": "artifact exact causal prefix; unused tail ignored for smoke runs",
                "source_log": str(args.output_dir / "density_source_events.csv"),
                "source_log_contents": "mutation rows and strongest negative_suppress rows without GT masks",
            },
            "cue_mixture": {
                "enabled": args.density_policy in _CUE_MIXTURE_DENSITY_POLICIES,
                "score": "capped_total_mass * 2 * min(change_ratio, nonchange_ratio)",
                "threshold": float(args.cue_mixture_threshold),
                "source": "current newly arrived frame pre-optimization raw alpha-T cue evidence",
                "scope": "large ACTIVE Gaussian split only; no lifecycle or small-row clone effect",
            },
            "pruning_enabled": bool(
                float(args.min_opacity) > 0.0
                or float(args.max_screen_size) > 0.0
                or args.density_policy
                == "active_oscd_cue_mixture_black_child_prune"
                or (
                    args.density_policy == "active_signed_lifespan_score"
                    and args.signed_score_prune_threshold is not None
                )
            ),
            "pruning": (
                "requested extension: O-SCD opacity/size criteria restricted to ACTIVE rows"
                if float(args.min_opacity) > 0.0 or float(args.max_screen_size) > 0.0
                else (
                    "negative signed-score hard-prunes only old enough ACTIVE descendants"
                    if args.density_policy == "active_signed_lifespan_score"
                    and args.signed_score_prune_threshold is not None
                    else (
                        "hard-prune previously densified black ACTIVE children with direct-parent support guard"
                        if args.density_policy
                        == "active_oscd_cue_mixture_black_child_prune"
                        else "disabled; split sources are still replaced by their children"
                    )
                )
            ),
            "black_child_pruning": {
                "enabled": args.density_policy
                == "active_oscd_cue_mixture_black_child_prune",
                "scope": "previously densified ACTIVE children only; generation-zero rows are protected",
                "threshold": float(args.black_child_prune_threshold),
                "min_age_frames": int(args.black_child_prune_min_age_frames),
                "support_guard": "retain the least-black candidate when its direct parent is gone and no sibling survives",
                "source": "learned intrinsic degree-zero SH RGB mean",
            },
            "online_oscd_pruning_source_note": "oscd.py 16-step online loop itself clone/splits but does not prune",
            "fastgs_vcd_vcp": False,
            "k_view_importance": None,
            "replay": "current with configured probability; otherwise uniform already-processed view",
            "future_view_access_count": int(
                sum(row["future_training_view_access_count"] for row in density_rows)
            ),
            "initial_gaussian_count": int(initial_count),
            "final_gaussian_count": int(topology.count),
            "total_clone_children": int(total_clones),
            "total_split_sources": int(total_split_sources),
            "total_split_children": int(total_split_children),
            "total_cue_mixture_split_sources": int(
                sum(row["cue_mixture_split_source_count"] for row in density_rows)
            ),
            "total_cue_mixture_only_split_sources": int(
                sum(
                    row["cue_mixture_only_split_source_count"]
                    for row in density_rows
                )
            ),
            "total_signed_score_candidates": int(total_signed_score_candidates),
            "total_signed_score_clone_sources": int(
                total_signed_score_clone_sources
            ),
            "total_signed_score_split_sources": int(
                total_signed_score_split_sources
            ),
            "total_signed_score_prune_candidates": int(
                total_signed_score_prune_candidates
            ),
            "total_signed_score_pruned": int(total_signed_score_pruned),
            "total_removed": int(total_pruned),
            "total_opacity_pruned": int(total_opacity_pruned),
            "total_size_pruned": int(total_size_pruned),
            "total_black_child_prune_candidates": int(
                total_black_child_prune_candidates
            ),
            "total_black_children_pruned": int(total_black_children_pruned),
            "total_black_children_retained_for_support": int(
                total_black_children_retained_for_support
            ),
        },
        "metrics": metrics,
        "hard_output_control_metrics": hard_output_metrics,
        "candidate_gate_metric_delta": candidate_gate_metric_delta,
        **event_diag,
        **same_scene_repeated_transition_diagnostics(
            lifecycle_events, diagnostic_boundaries
        ),
        "same_scene_representation_transition_diagnostics": (
            same_scene_repeated_transition_diagnostics(
                [
                    event
                    for event in lifecycle_events
                    if event.action in {"OPEN", "CLOSE"}
                ],
                diagnostic_boundaries,
            )
        ),
        "posterior_diagnostics": posterior_run_diagnostics(frame_rows),
        "candidate_diagnostics": candidate_run_diagnostics(frame_rows),
        "optimizer_selection_diagnostics": {
            "policy": args.optimizer_selection,
            "max_selected_open_rows": int(
                max(
                    (row["optimizer_selected_open_rows"] for row in frame_rows),
                    default=0,
                )
            ),
            "max_active_visible_rows": int(
                max(
                    (
                        row["optimizer_selected_active_visible_rows"]
                        for row in frame_rows
                    ),
                    default=0,
                )
            ),
            "max_selected_never_open_rows": int(
                max(
                    (
                        row["optimizer_selected_never_open_visible_rows"]
                        for row in frame_rows
                    ),
                    default=0,
                )
            ),
            "frames_selecting_off_view_open_rows": int(
                sum(
                    row["optimizer_selected_open_rows"]
                    > row["optimizer_selected_active_visible_rows"]
                    for row in frame_rows
                )
            ),
        },
        "active_dc_diagnostics": {
            "black_threshold": 0.5,
            "source": args.change_color_mode,
            "meaning": (
                "effective lifespan-gate semantic color below 0.5"
                if args.change_color_mode == "lifespan_gate"
                else "intrinsic degree-zero SH RGB mean below neutral SH-zero value"
            ),
            "frame_mean_below_half_fraction": float(
                np.mean(
                    [row["active_dc_below_half_fraction"] for row in frame_rows]
                )
            ),
            "frame_max_below_half_fraction": float(
                max(
                    row["active_dc_below_half_fraction"] for row in frame_rows
                )
            ),
            "final_below_half_count": int(
                frame_rows[-1]["active_dc_below_half_count"]
            ),
            "final_below_half_fraction": float(
                frame_rows[-1]["active_dc_below_half_fraction"]
            ),
            "final_intrinsic_render_stats": frame_rows[-1][
                "active_dc_intrinsic_render_stats"
            ],
        },
        "first_open_dc_initialization_diagnostics": {
            "mode": args.first_open_dc_initialization,
            "initialized_count": int(
                sum(row["first_open_dc_initialized_count"] for row in frame_rows)
            ),
            "dc_adam_reset_count": int(total_first_open_dc_adam_resets),
            "source_surviving_post_frame_count": int(
                total_first_open_source_survivors
            ),
            "intrinsic_render_before": quantile_summary(
                _concat_cpu(first_open_dc_values["before"])
            ),
            "intrinsic_render_immediate": quantile_summary(
                _concat_cpu(first_open_dc_values["immediate"])
            ),
            "intrinsic_render_after_first_update": quantile_summary(
                _concat_cpu(first_open_dc_values["after_first_update"])
            ),
            "intrinsic_render_before_density": quantile_summary(
                _concat_cpu(first_open_dc_values["before_density"])
            ),
            "intrinsic_render_post_frame_surviving": quantile_summary(
                _concat_cpu(first_open_dc_values["post_frame_surviving"])
            ),
            "raw_abs_jump": quantile_summary(
                _concat_cpu(first_open_dc_values["raw_abs_jump"])
            ),
        },
        "first_open_close_latency_diagnostics": (
            first_open_close_latency_diagnostics(lifecycle_events)
        ),
        "final_active_gs": int(topology.active_mask().sum().item()),
        "closed_row_persistence_audit": closed_result,
        "inactive_gradient_violations": int(inactive_gradient_violations),
        "inactive_gradient_max_abs": float(inactive_gradient_max_abs),
        "dc_gradient_max_abs": float(max_dc_gradient),
        "features_rest_gradient_max_abs": float(max_features_rest_gradient),
        "topology_integrity": True,
        "gt_used_in_causal_loop": False,
        "gt_loaded_after_inference_only": True,
        "manual_boundaries_used": False,
        "manual_boundaries_used_for_posthoc_diagnostics_only": list(
            diagnostic_boundaries
        ),
        "visual_capture": (
            {
                "root": str(visual_root),
                "raw_render": str(visual_root / "raw_render"),
                "thresholded_render": str(visual_root / "thresholded_render"),
                "threshold": float(config.evaluation_threshold),
                "event_render": str(visual_root / "event_render"),
                "event_geometry": "current dynamic topology before the frame density event",
                "event_render_contract": (
                    "persistent half-bright OPEN/CLOSED state with full-bright "
                    "exact-timestamp OPEN/CLOSE events"
                ),
                "event_state_intensity": 0.5,
                "never_open_rows_rendered": False,
            }
            if visual_root is not None
            else None
        ),
        "runtime_seconds": time.time() - started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "cue_cache_metadata": cue_metadata,
        "fixed_cameras_sha256": file_checksum(args.fixed_cameras_json),
        "source_ply_sha256": file_checksum(base_ply),
        "run_arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    _write_csv(args.output_dir / "frame_metrics.csv", frame_rows)
    _write_csv(args.output_dir / "density_events.csv", density_rows)
    _write_csv(args.output_dir / "density_source_events.csv", signed_density_source_rows)
    with (args.output_dir / "lifecycle_events.jsonl").open(
        "w", encoding="utf-8"
    ) as file:
        for event in lifecycle_events:
            file.write(json.dumps(asdict(event), sort_keys=True) + "\n")
    with (args.output_dir / "topology_events.jsonl").open(
        "w", encoding="utf-8"
    ) as file:
        for event in topology.lineage_events:
            file.write(json.dumps(event, sort_keys=True) + "\n")
    if not args.skip_checkpoint:
        torch.save(
            _cpu_tree(
                {
                    "summary": summary,
                    "model_state": model.state_dict(),
                    "filter_state": tracker.state_dict(),
                    "controller_state": controller.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "topology_state": topology.state_dict(),
                }
            ),
            args.output_dir / "checkpoint.pt",
        )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", type=Path, default=Path(DEFAULT_SOURCE))
    parser.add_argument(
        "--fixed-cameras-json", type=Path, default=Path(DEFAULT_FIXED_CAMERAS)
    )
    parser.add_argument("--cue-cache-root", type=Path, default=Path(DEFAULT_CUE_CACHE))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=SCOPES, required=True)
    parser.add_argument("--max-frames", type=positive_int, default=None)
    parser.add_argument("--resolution", type=float, default=4.0)
    parser.add_argument("--density-policy", choices=DENSITY_POLICIES, default="active_oscd")
    parser.add_argument(
        "--render-support-mode",
        choices=RENDER_SUPPORT_MODES,
        default="open_or_never_open",
    )
    parser.add_argument(
        "--optimizer-selection",
        choices=OPTIMIZER_SELECTION_POLICIES,
        default="all_open",
        help=(
            "Select OPEN rows only when visible, or select every OPEN row on every "
            "update (the latter requires frozen NEVER_OPEN appearance)"
        ),
    )
    parser.add_argument(
        "--first-open-dc-initialization",
        choices=FIRST_OPEN_DC_INITIALIZATIONS,
        default="preserve",
        help=(
            "Preserve persistent DC, or set intrinsic degree-zero SH render value "
            "to one and reset DC Adam state on explicit first OPEN only"
        ),
    )
    parser.add_argument(
        "--change-color-mode",
        choices=CHANGE_COLOR_MODES,
        default="lifespan_gate",
        help=(
            "Render learned persistent DC, or override every OPEN Gaussian with "
            "semantic RGB one and every fixed NEVER_OPEN occluder with RGB zero"
        ),
    )
    parser.add_argument(
        "--candidate-render-gate",
        choices=CANDIDATE_RENDER_GATES,
        default="close_only_log_bf_progress",
        help=(
            "Optionally cross-fade the current-frame evaluation render using "
            "normalized log BF, either bidirectionally or for CLOSE candidates "
            "only; detector, optimizer, and density control remain hard-gated"
        ),
    )
    parser.add_argument(
        "--loss-regularization-mode",
        choices=LOSS_REGULARIZATION_MODES,
        default="local_growth_replay",
        help=(
            "Use original O-SCD global sparsity or local cue support plus the "
            "detached previous-frame positive-growth replay penalty"
        ),
    )
    parser.add_argument(
        "--local-support-scale",
        type=float,
        default=2.0,
        help="Fixed divisor mapping the cached 0..2 cue to local support in [0,1]",
    )
    parser.add_argument(
        "--growth-replay-weight",
        type=float,
        default=7.5,
        help="Weight for element-wise unsupported positive-growth replay",
    )
    parser.add_argument("--updates-per-frame", type=positive_int, default=16)
    parser.add_argument("--densify-update-index", type=int, default=4)
    parser.add_argument("--oscd-grad-threshold", type=nonnegative_float, default=0.001)
    parser.add_argument(
        "--signed-score-artifact",
        type=Path,
        default=None,
        help=(
            "Required for active_signed_lifespan_score: causal PC1/sign posterior "
            "arrays from the signed SAM-diff frontend"
        ),
    )
    parser.add_argument(
        "--signed-score-posterior",
        choices=("balanced", "global"),
        default="balanced",
        help="Choose balanced or global P(+ is NEW/ADD) from the score artifact",
    )
    parser.add_argument(
        "--signed-score-sam-model",
        default="facebook/sam2.1-hiera-tiny",
        help="Local SAM2 model name used to reproduce current-frame signed deltas",
    )
    parser.add_argument(
        "--signed-score-threshold",
        type=nonnegative_float,
        default=0.0,
        help="Positive signed lifespan score threshold for clone/split selection",
    )
    parser.add_argument(
        "--signed-score-max-sources",
        type=int,
        default=2048,
        help="Maximum score-selected source rows per density event; use 0 for no limit",
    )
    parser.add_argument(
        "--signed-score-prune-threshold",
        type=nonnegative_float,
        default=None,
        help=(
            "Optional negative signed score threshold for generation>0 ACTIVE "
            "descendant pruning; omitted disables score pruning"
        ),
    )
    parser.add_argument(
        "--signed-score-prune-min-age-frames",
        type=positive_int,
        default=1,
        help="Minimum OPEN episode age before negative score can prune descendants",
    )
    parser.add_argument(
        "--cue-mixture-threshold",
        type=probability,
        default=0.5,
        help=(
            "For active_oscd_cue_mixture, split a large ACTIVE Gaussian when "
            "its current pre-optimization raw cue mixture score reaches this value"
        ),
    )
    parser.add_argument(
        "--black-child-prune-threshold",
        type=probability,
        default=0.5,
        help=(
            "For active_oscd_cue_mixture_black_child_prune, hard-prune an "
            "eligible densified ACTIVE child below this intrinsic learned-DC value"
        ),
    )
    parser.add_argument(
        "--black-child-prune-min-age-frames",
        type=positive_int,
        default=1,
        help=(
            "Protect newly densified children for at least this many frame "
            "transitions before learned-DC black pruning"
        ),
    )
    parser.add_argument("--percent-dense", type=float, default=0.01)
    parser.add_argument("--min-opacity", type=probability, default=0.0)
    parser.add_argument("--max-screen-size", type=nonnegative_float, default=0.0)
    parser.add_argument("--current-view-probability", type=probability, default=0.33)
    parser.add_argument(
        "--detector-mode",
        choices=DETECTOR_MODES,
        default="lifespan_gate_beta",
        help=(
            "Lifecycle detector; this branch defaults to BF30 FLIP/KEEP evidence "
            "against the committed lifespan bit"
        ),
    )
    parser.add_argument(
        "--candidate-bayes-factor-threshold",
        type=bayes_factor_greater_than_one,
        default=30.0,
        help="RESET-vs-KEEP Bayes factor threshold used by single_candidate_beta",
    )
    parser.add_argument("--gate-stable-flip-prior", type=float, default=1.0)
    parser.add_argument("--gate-stable-keep-prior", type=float, default=10.0)
    parser.add_argument("--gate-reset-flip-prior", type=float, default=1.0)
    parser.add_argument("--gate-reset-keep-prior", type=float, default=1.0)
    parser.add_argument("--agreement-stable-flip-prior", type=float, default=1.0)
    parser.add_argument("--agreement-stable-keep-prior", type=float, default=10.0)
    parser.add_argument("--agreement-reset-flip-prior", type=float, default=1.0)
    parser.add_argument("--agreement-reset-keep-prior", type=float, default=1.0)
    parser.add_argument(
        "--agreement-confirmation-views",
        type=positive_int,
        default=3,
        help=(
            "For anchored_dc_agreement_beta, require this many consecutive "
            "observed views with the same lifecycle-direction mismatch"
        ),
    )
    parser.add_argument(
        "--agreement-directional-margin",
        type=nonnegative_float,
        default=0.0,
        help=(
            "Minimum q-versus-C_anchor directional gap required on every "
            "anchored agreement confirmation view"
        ),
    )
    parser.add_argument(
        "--agreement-candidate-dc-policy",
        choices=AGREEMENT_CANDIDATE_DC_POLICIES,
        default="adapt",
        help=(
            "For learned_dc_agreement_beta, either keep optimizing DC while a "
            "candidate is live or freeze only those candidate DC rows"
        ),
    )
    parser.add_argument(
        "--detector-probe-scaling",
        choices=DETECTOR_PROBE_SCALING_MODES,
        default="native",
        help="Scaling used only by the detached alpha-T detector probe",
    )
    parser.add_argument("--bayes-cue-threshold", type=float, default=0.5)
    parser.add_argument("--evidence-mass-saturation", type=float, default=1.0)
    parser.add_argument("--min-evidence-mass", type=nonnegative_float, default=1e-6)
    parser.add_argument("--state-emission-reliability", type=float, default=0.9)
    parser.add_argument("--inactive-to-active-prior", type=float, default=0.01)
    parser.add_argument("--active-to-inactive-prior", type=float, default=0.01)
    parser.add_argument("--initial-active-probability", type=float, default=0.5)
    parser.add_argument("--filter-chunk-size", type=positive_int, default=65536)
    parser.add_argument("--transition-confirmation-views", type=positive_int, default=3)
    parser.add_argument("--min-transition-bayes-factor", type=nonnegative_float, default=3.0)
    parser.add_argument("--min-transition-evidence-strength", type=nonnegative_float, default=1e-6)
    parser.add_argument("--max-states", type=positive_int, default=16)
    parser.add_argument("--evaluation-threshold", type=float, default=0.5)
    parser.add_argument("--dc-lr", type=nonnegative_float, default=0.0025)
    parser.add_argument("--xyz-lr", type=nonnegative_float, default=0.00016)
    parser.add_argument("--features-rest-lr", type=nonnegative_float, default=0.000125)
    parser.add_argument("--opacity-lr", type=nonnegative_float, default=0.025)
    parser.add_argument("--scaling-lr", type=nonnegative_float, default=0.005)
    parser.add_argument("--rotation-lr", type=nonnegative_float, default=0.001)
    parser.add_argument("--adam-eps", type=float, default=1e-15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-checkpoint", action="store_true")
    parser.add_argument(
        "--visualization-dir",
        type=Path,
        default=None,
        help=(
            "Optionally capture post-update raw renders and current lifecycle "
            "state with exact-timestamp event highlights"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run(args)
    print(
        json.dumps(
            {
                "summary": str(args.output_dir / "summary.json"),
                "scope": summary["scope_key"],
                "density_policy": args.density_policy,
                "mIoU": summary["metrics"]["mean_frame_iou"],
                "F1": summary["metrics"]["mean_frame_f1"],
                "initial_gs": summary["density_control"]["initial_gaussian_count"],
                "final_gs": summary["density_control"]["final_gaussian_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
