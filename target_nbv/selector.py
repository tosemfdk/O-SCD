# End-to-end target-conditioned NBV selector (stage 12, docs/target_gaussian_nbv.md).
#
# Wires registry -> information state -> candidates -> visibility -> proxy ->
# exact P-opt into one dry-run call. select_next_view NEVER mutates the
# information state or the model: new views enter the state only through
# scorers.exact.commit_observed_view once actually observed, and the target is
# frozen (episode) for the duration of the call. MVP modes: geometry_proxy,
# geometry_exact.

from __future__ import annotations

import time

import numpy as np
import torch

from target_nbv import target_registry
from target_nbv.candidates import generate_candidates
from target_nbv.config import TargetNBVConfig
from target_nbv.info_builder import TargetInformationBuilder
from target_nbv.jacobian import compute_target_jacobian
from target_nbv.scorers.exact import TargetPOptimalScorer
from target_nbv.scorers.proxy import GeometryProxyScorer, proxy_top_k
from target_nbv.types import SelectionResult, TargetParameterSpec
from target_nbv.visibility import make_visibility_backend

_SNAPSHOT_TENSORS = ("_xyz", "_scaling", "_rotation", "_opacity",
                     "_features_dc", "_features_rest")


class NoValidCandidateError(RuntimeError):
    pass


def build_information_state(model, target_row: int, target_pid: int,
                            spec: TargetParameterSpec,
                            observed_cameras: list[tuple[str, object]],
                            pipe, background, cfg: TargetNBVConfig,
                            model_version: str = ""):
    """Build the target prior from observed views. Views where the target is
    not visible are skipped (returned as {view_id: reason}), not errors."""
    builder = TargetInformationBuilder(target_pid, spec, cfg, model_version)
    skipped: dict[str, str] = {}
    for view_id, cam in observed_cameras:
        res = compute_target_jacobian(model, cam, target_row, spec, pipe, background, cfg)
        if not res.valid:
            skipped[view_id] = res.invalid_reason
            continue
        builder.add_view(view_id, res.J, res.w)
    return builder, skipped


def select_next_view(model, persistent_id: int,
                     observed_cameras: list[tuple[str, object]],
                     cfg: TargetNBVConfig, pipe, background,
                     current_camera=None,
                     information_builder: TargetInformationBuilder | None = None,
                     width: int | None = None, height: int | None = None,
                     fovy: float | None = None) -> SelectionResult:
    """Pick the next best view for one target Gaussian. Dry-run by contract:
    model and information state are unchanged on return.

    observed_cameras: list of (view_id, MiniCam/Camera); used to build the
    prior when `information_builder` is None, ignored otherwise. Candidate
    image geometry defaults to the first observed camera's."""
    if cfg.mode not in ("geometry_proxy", "geometry_exact"):
        raise NotImplementedError(
            f"mode {cfg.mode!r} is not wired into the selector yet (MVP: geometry_*)")
    if information_builder is None and not observed_cameras:
        raise ValueError("need observed_cameras or a prebuilt information_builder")

    ref_cam = observed_cameras[0][1] if observed_cameras else None
    width = width or ref_cam.image_width
    height = height or ref_cam.image_height
    fovy = fovy or ref_cam.FoVy

    runtime: dict[str, float] = {}
    t_total = time.perf_counter()

    snapshot = None
    if cfg.debug.verify_model_restoration:
        snapshot = {n: getattr(model, n).detach().clone() for n in _SNAPSHOT_TENSORS}

    spec = TargetParameterSpec(parameter_names=list(cfg.parameter_groups))
    with target_registry.target_episode(model, persistent_id) as handle:
        row = handle.current_indices[0]

        t0 = time.perf_counter()
        if information_builder is None:
            information_builder, skipped_views = build_information_state(
                model, row, persistent_id, spec, observed_cameras,
                pipe, background, cfg)
        else:
            if information_builder.state.target_pid != persistent_id:
                raise ValueError(
                    f"information state is for target {information_builder.state.target_pid}, "
                    f"selection is for {persistent_id}")
            skipped_views = {}
        state = information_builder.state
        state_version_before = (state.version, list(state.observed_view_ids))
        runtime["information"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        candidates = generate_candidates(model, row, cfg, width, height, fovy,
                                         current_cam=current_camera)
        runtime["candidates"] = time.perf_counter() - t0
        if not candidates:
            raise NoValidCandidateError(
                "candidate generator produced no feasible poses "
                "(distance/projected-radius bounds too tight?)")

        t0 = time.perf_counter()
        backend = make_visibility_backend(cfg)
        visibilities = [backend.evaluate(model, c.minicam, row, pipe, background)
                        for c in candidates]
        runtime["visibility"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        mu = model._xyz[row].detach().cpu().numpy()
        scores = GeometryProxyScorer(cfg).rank(state, candidates, visibilities, mu)
        runtime["proxy"] = time.perf_counter() - t0

        shortlist = proxy_top_k(scores, cfg.proxy.top_k)
        if not shortlist:
            raise NoValidCandidateError(
                "all candidates invalid: " + _invalid_summary(scores))

        predicted_H_after = None
        if cfg.mode == "geometry_exact":
            t0 = time.perf_counter()
            exact_input = shortlist[:cfg.jacobian.exact_top_k]
            TargetPOptimalScorer(cfg).score_candidates(
                state, exact_input, model, row, pipe, background)
            runtime["exact"] = time.perf_counter() - t0
            survivors = [s for s in exact_input if s.valid]
            if not survivors:
                raise NoValidCandidateError(
                    "no candidate survived exact scoring: " + _invalid_summary(exact_input))
            best = survivors[0]
            predicted_H_after = np.asarray(state.H_prior()) + best.delta_H
            # final report order: exact-scored first, proxy-only next, invalid last
            exact_set = {id(s) for s in exact_input}
            scores.sort(key=lambda s: (
                not s.valid,
                0 if (id(s) in exact_set and s.valid) else 1,
                -(s.exact_score if id(s) in exact_set else s.proxy_score),
                s.candidate.cand_id))
        else:
            best = shortlist[0]

        if (state.version, list(state.observed_view_ids)) != state_version_before:
            raise RuntimeError("selector mutated the information state — bug")

    if snapshot is not None:
        for name, before in snapshot.items():
            if not torch.equal(getattr(model, name).detach(), before):
                raise RuntimeError(f"model tensor {name} not restored after selection")

    runtime["total"] = time.perf_counter() - t_total
    result = SelectionResult(
        best=best, scores=scores,
        H_before=np.asarray(state.H_prior()),
        predicted_H_after=predicted_H_after,
        config_snapshot=cfg.to_dict(), runtime=runtime)
    result.config_snapshot["_selection_meta"] = {
        "target_pid": int(persistent_id),
        "observed_views_used": list(state.observed_view_ids),
        "observed_views_skipped": skipped_views,
        "num_candidates_generated": len(candidates),
        "num_candidates_valid": sum(1 for s in scores if s.valid),
    }
    return result


def _invalid_summary(scores) -> str:
    reasons: dict[str, int] = {}
    for s in scores:
        if not s.valid:
            reasons[s.invalid_reason or "unknown"] = reasons.get(s.invalid_reason or "unknown", 0) + 1
    return ", ".join(f"{k} x{v}" for k, v in sorted(reasons.items()))
