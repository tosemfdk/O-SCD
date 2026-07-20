# Global-local consensus keyframe selection (Part 2 cycle 2).
#
# Claim scope: offline_pool_keyframe_selection — all 25 inference images were
# already spent building R_global, so this is NOT active NBV and never claims
# acquisition savings. The question it answers: given all-view change context
# (R_global) and the current subset's change state (R_local(S)), can 5 frames
# be chosen GT-free that approach the oracle-5 ceiling?
#
# Hard rules enforced here (spec §1):
#   * No GT access of any kind — this module must never import evaluators or
#     touch GT masks; candidate image content is blocked by FrameAccessGuard.
#   * R_global and R_local Gaussian indices are never compared; the two models
#     only meet through image-space soft masks rendered at the same pose.
#   * The exact 3x3 position-Jacobian block machinery is reused unchanged
#     (hutchinson_block_information / block_dopt_gain).
#   * No forced first frame: round 1 scores all candidates on the global
#     component alone (m_local = 0), so frame 0 holds no special position.
from __future__ import annotations

import time

import numpy as np
import torch

from view_selection.information import (block_dopt_gain,
                                        hutchinson_block_information)
from view_selection.types import FrameAccessGuard, InformationConfig
from view_selection.weights import pose_weight

# method -> (components used from round 2 on, direction-aware?)
# Round 1 always scores with the global component alone: with S = empty the
# unresolved mask equals the global mask and the local component is invalid,
# so averaging duplicates would double-count (spec §6) — and this guarantees
# the round-1 ranking of every kf_*_dir method matches kf_g_dir (test 6).
KF_METHODS: dict[str, tuple[tuple[str, ...], bool]] = {
    "kf_g_dir": (("global",), True),
    "kf_l_dir_gseed": (("local4",), True),  # 4th-selector criterion on R_local
    "kf_gu_dir": (("global", "unresolved"), True),
    "kf_gl_dir": (("global", "local"), True),
    "kf_glu_dir": (("global", "local", "unresolved"), True),
    "kf_glu_nodir": (("global", "local", "unresolved"), False),
}

CLAIM_SCOPE = "offline_pool_keyframe_selection"


def local_rebuild_seed(train_seed: int, S) -> int:
    """RNG seed for the clean R_local(S) rebuild. Depends on the SET only
    (sorted), never on greedy order — the determinism contract of spec §2B."""
    import hashlib

    digest = hashlib.sha256(
        repr((train_seed, tuple(sorted(S)))).encode()).digest()
    return int.from_bytes(digest[:4], "little")


def render_soft_mask(model, camera, pipe, background) -> torch.Tensor:
    """m = sigmoid(channel-mean raw change logit), detached (spec §4)."""
    from gaussian_renderer import render_change

    with torch.no_grad():
        z = render_change(camera, model, pipe, background)["render"].mean(dim=0)
        return torch.sigmoid(z).detach()


def component_masks(V: torch.Tensor, m_global: torch.Tensor,
                    m_local: torch.Tensor) -> dict[str, torch.Tensor]:
    """All five spec §4 masks, detached. overlap/local_only are diagnostics
    only — local_only is NOT a penalty in the main method (R_global is not GT,
    so a local-only region may be real change that all-25 fusion washed out)."""
    return {
        "global": (V * m_global).detach(),
        "local": (V * m_local).detach(),
        "unresolved": (V * torch.relu(m_global - m_local)).detach(),
        "overlap": (V * m_global * m_local).detach(),
        "local_only": (V * torch.relu(m_local - m_global)).detach(),
    }


def percentile_rank(scores: dict[int, float]) -> dict[int, float]:
    """Percentile rank in [0, 1] within the given candidate set, ties averaged
    (mask-area scale differences between components must not dominate, §6)."""
    ids = sorted(scores)
    vals = np.array([scores[i] for i in ids], dtype=np.float64)
    n = len(ids)
    if n == 1:
        return {ids[0]: 1.0}
    order = vals.argsort(kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    for v in np.unique(vals):
        tie = vals == v
        if tie.sum() > 1:
            ranks[tie] = ranks[tie].mean()
    ranks /= (n - 1)
    return {i: float(r) for i, r in zip(ids, ranks)}


def _block_lambda(blocks: dict[int, torch.Tensor],
                  config: InformationConfig) -> float:
    """Same trace-median heuristic as the dopt_seq dopt_dir branch."""
    tr = torch.stack([torch.diagonal(b, dim1=1, dim2=2).sum(dim=1)
                      for b in blocks.values()])
    pos = tr[tr > 0]
    return max(config.lambda_abs,
               config.lambda_rel * float(pos.median()) / 3.0
               if pos.numel() else config.lambda_abs)


def _condition_stats(H: torch.Tensor, lam: float) -> dict:
    """min/median/max eigenvalue-condition of the selected-set prior over
    Gaussians that carry any mass (diagnostics for the manifest)."""
    with torch.no_grad():
        tr = torch.diagonal(H, dim1=1, dim2=2).sum(dim=1)
        active = tr > 0
        if not bool(active.any()):
            return {"active_gaussians": 0}
        m = 0.5 * (H[active] + H[active].transpose(1, 2)).double()
        ev = torch.linalg.eigvalsh(m).clamp_min(lam)
        cond = (ev[:, -1] / ev[:, 0])
        return {"active_gaussians": int(active.sum()),
                "cond_min": float(cond.min()),
                "cond_median": float(cond.median()),
                "cond_max": float(cond.max())}


def _component_scores(model_X, all_views, selected: list[int],
                      remaining: list[int], weights: dict[int, torch.Tensor],
                      config: InformationConfig, seed_scope: tuple,
                      pipe, background, direction: bool,
                      gaussian_weights: torch.Tensor | None = None
                      ) -> tuple[dict[int, float], dict]:
    """Score all remaining candidates for ONE component on ONE model topology.

    direction=True : G(v) = sum_i logdet(H_S,i + B_v,i) - logdet(H_S,i), the
        exact 3x3 position-Jacobian D-opt gain — values a NEW viewing direction
        on the regions the component mask marked as worth seeing (§5).
    direction=False: scalar mass tr(sum_i B_v,i) from the same probes — the
        kf_glu_nodir ablation that keeps the information measure and drops
        only the directional structure.
    gaussian_weights: per-Gaussian weights for the logdet sum (ones for the
        spec §5 components; suspicion c for the kf_l_dir_gseed 4th-selector
        replica). Blocks/prior are rebuilt from scratch each round — no append
        or clone inheritance across topologies (§5).
    """
    ids = (selected + remaining) if direction else list(remaining)
    blocks: dict[int, torch.Tensor] = {}
    for i in ids:
        blocks[i] = hutchinson_block_information(
            model_X, all_views[i], weights[i], config, seed_scope,
            pipe, background, frame_id=i)
    lam = _block_lambda(blocks, config)
    diag = {"lambda": lam}
    if direction:
        n = blocks[ids[0]].shape[0]
        gw = (torch.ones(n, device=blocks[ids[0]].device)
              if gaussian_weights is None else gaussian_weights)
        H = torch.zeros_like(blocks[ids[0]])
        for s in selected:
            H = H + blocks[s]
        scores = {i: block_dopt_gain(H, blocks[i], gw, lam)
                  for i in remaining}
        diag.update(_condition_stats(H, lam))
    else:
        scores = {i: float(torch.diagonal(blocks[i], dim1=1, dim2=2).sum())
                  for i in remaining}
    del blocks
    torch.cuda.empty_cache()
    return scores, diag


def _suspicion(model_change) -> torch.Tensor:
    """4th-selector per-Gaussian suspicion: normalized positive change mass."""
    with torch.no_grad():
        c = model_change._features_dc.detach()
        susp = c.reshape(c.shape[0], -1).mean(dim=1).clamp_min(0.0)
        return susp / susp.sum().clamp_min(1e-12)


def select_keyframes_gl(method: str, all_views, model_ref, model_global,
                        budget: int, config: InformationConfig,
                        pipe, background, rebuild_local_fn,
                        scene_id: str, train_seed: int,
                        round_artifact_fn=None) -> dict:
    """Greedy global-local keyframe selection (spec §3).

    rebuild_local_fn(sorted_S) -> fresh R_local(S) built by clean chronological
    replay from the reference checkpoint (never called with S = empty; the
    caller owns fusion, cue computation, and its RNG discipline so that
    R_local depends on the SET only, not on greedy order).

    round_artifact_fn(round_idx, masks) — optional hook (--save_round_models):
    receives the per-candidate component masks before scoring so the caller
    can persist them; must not mutate them.

    Returns a manifest dict: greedy_order/replay_order plus per-round raw
    scores, percentile ranks, mask masses, lambda/condition diagnostics, and
    the frame-0 bookkeeping the spec demands. Never touches GT or candidate
    image content (FrameAccessGuard active while scoring).
    """
    if method not in KF_METHODS:
        raise ValueError(f"unknown keyframe method {method!r}")
    components_after_r1, direction = KF_METHODS[method]
    n = len(all_views)
    assert 0 < budget <= n
    t_start = time.time()

    selected: list[int] = []
    remaining = sorted(range(n))
    rounds: list[dict] = []

    # Pose-only visibility and the FIXED global context masks (R_global never
    # changes during selection; only m_local is refreshed per round).
    with FrameAccessGuard(all_views), torch.no_grad():
        V = {i: pose_weight(model_ref, all_views[i], pipe, background,
                            config.alpha_threshold).detach()
             for i in range(n)}
        m_global = {i: render_soft_mask(model_global, all_views[i], pipe,
                                        background) for i in range(n)}

    for round_idx in range(1, budget + 1):
        t_round = time.time()
        model_local = None
        rebuild_seconds = 0.0
        if selected:
            t_reb = time.time()
            model_local = rebuild_local_fn(sorted(selected))
            rebuild_seconds = time.time() - t_reb

        with FrameAccessGuard(all_views, allowed_ids=set(selected)):
            with torch.no_grad():
                m_local = {i: (render_soft_mask(model_local, all_views[i],
                                                pipe, background)
                               if model_local is not None
                               else torch.zeros_like(m_global[i]))
                           for i in range(n)}
                masks = {i: component_masks(V[i], m_global[i], m_local[i])
                         for i in range(n)}
            if round_artifact_fn is not None:
                round_artifact_fn(round_idx, masks)

            if round_idx == 1:
                components = ("global",)
            else:
                components = components_after_r1

            comp_raw: dict[str, dict[int, float]] = {}
            comp_diag: dict[str, dict] = {}
            for comp in components:
                if comp == "local4":
                    # kf_l_dir_gseed: exact 4th-selector criterion on the clean
                    # R_local — pose pixel weight, suspicion-weighted logdet —
                    # isolating "was frame-0 seeding the only failure cause?"
                    weights = {i: V[i] for i in range(n)}
                    model_X = model_local
                    gw = _suspicion(model_local)
                    comp_dir = True
                else:
                    weights = {i: masks[i][comp] for i in range(n)}
                    model_X = model_global if comp in ("global", "unresolved") \
                        else model_local
                    gw = None
                    comp_dir = direction
                seed_scope = (scene_id, "kf", train_seed, comp, round_idx)
                comp_raw[comp], comp_diag[comp] = _component_scores(
                    model_X, all_views, selected, remaining, weights, config,
                    seed_scope, pipe, background, comp_dir, gw)

            comp_rank = {c: percentile_rank(comp_raw[c]) for c in comp_raw}
            total = {i: float(np.mean([comp_rank[c][i] for c in comp_rank]))
                     for i in remaining}
            # deterministic tie-break: smaller frame ID (spec §3.6)
            pick = min((-v, i) for i, v in total.items())[1]

        rounds.append({
            "round": round_idx,
            "candidates": list(remaining),
            "components": list(components),
            "raw_scores": {c: {str(i): comp_raw[c][i] for i in remaining}
                           for c in comp_raw},
            "percentile_ranks": {c: {str(i): comp_rank[c][i]
                                     for i in remaining} for c in comp_rank},
            "total_score": {str(i): total[i] for i in remaining},
            "mask_mass": {name: {str(i): float(masks[i][name].sum())
                                 for i in remaining}
                          for name in ("global", "local", "unresolved",
                                       "overlap", "local_only")},
            "component_diagnostics": comp_diag,
            "pick": pick,
            "pick_total_score": total[pick],
            "gaussian_count_global": int(model_global.get_xyz.shape[0]),
            "gaussian_count_local": (int(model_local.get_xyz.shape[0])
                                     if model_local is not None else 0),
            "local_rebuild_seconds": round(rebuild_seconds, 2),
            "round_seconds": round(time.time() - t_round, 2),
        })
        selected.append(pick)
        remaining.remove(pick)
        del model_local
        torch.cuda.empty_cache()

    r1 = rounds[0]
    r1_rank = r1["percentile_ranks"]["global"]
    manifest = {
        "schema_version": 1,
        "method": method,
        "claim_scope": CLAIM_SCOPE,
        "global_context_uses_all_25": True,
        "gt_used_for_selection": False,
        "first_frame_forced": False,
        "first_frame_id": selected[0],
        "first_frame_global_rank": r1_rank[str(selected[0])],
        "frame0_global_rank": r1_rank.get("0"),
        "frame0_selected": 0 in selected,
        "greedy_order": list(selected),
        "replay_order": sorted(selected),
        "selected_frame_ids": sorted(selected),
        "budget": budget,
        "train_seed": train_seed,
        "rank_fusion": "percentile_mean",
        "rounds": rounds,
        "selection_seconds": round(time.time() - t_start, 2),
    }
    return manifest
