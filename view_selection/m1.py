# M1 batch-static dopt_pose selection (spec §12): score every candidate pose
# on the REFERENCE scoring state (c = 0, geometry untouched), greedy-select,
# emit a manifest, and hand the replay order back to the standard chronological
# pipeline. Runs entirely under FrameAccessGuard — pose mode may never touch a
# candidate's image content (raises LeakageError if it does).
from __future__ import annotations

import hashlib
import json
import os
import time

import torch

from view_selection.cache import config_hash, load_frame, save_frame
from view_selection.greedy import greedy_static_select
from view_selection.information import (derive_relative_lambda,
                                        hutchinson_information)
from view_selection.types import FrameAccessGuard, InformationConfig
from view_selection.weights import build_pixel_weight


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _camera_hash(view) -> str:
    m = torch.cat([view.world_view_transform.reshape(-1),
                   view.full_proj_transform.reshape(-1)]).detach().cpu()
    return hashlib.sha256(m.numpy().round(6).tobytes()).hexdigest()[:16]


def select_dopt_pose(model_change, model_rgb, views, pipe, background,
                     config: InformationConfig, budget: int, criterion: str,
                     scene_id: str, checkpoint_path: str, cache_root: str,
                     manifest_path: str):
    """Returns (replay_order, manifest_dict). model_change must be at the
    reference scoring state (c = 0, before any fusion)."""
    t0 = time.time()
    ckpt = _file_sha256(checkpoint_path)
    base_key = {
        "scene_id": scene_id, "checkpoint": ckpt,
        "rasterizer": "fastgs", "resolution": f"{views[0].image_height}x{views[0].image_width}",
        "alpha_threshold": config.alpha_threshold,
        "output_space": config.output_space, "weight_mode": config.weight_mode,
        "num_probes": config.num_probes, "probe_seed_version": 1,
        "channel_projection": True,
    }

    infos: dict[int, torch.Tensor] = {}
    stats: dict[int, dict] = {}
    with FrameAccessGuard(views):
        for i, view in enumerate(views):
            key = {**base_key, "camera": _camera_hash(view)}
            hit = load_frame(cache_root, scene_id, key, i)
            if hit is not None:
                infos[i] = hit[0]
                stats[i] = {**hit[1], "cache": "hit"}
                continue
            w = build_pixel_weight(config.weight_mode, model_rgb, view, pipe,
                                   background, config)
            vi = hutchinson_information(
                model_change, view, w, config,
                seed_scope=(scene_id, ckpt), pipe=pipe, background=background,
                frame_id=i, strict=True)
            infos[i] = vi.diagonal
            stats[i] = {"cache": "miss", "alpha_coverage": vi.alpha_coverage,
                        "visible_gaussians": vi.visible_gaussians,
                        "valid_pixels": vi.valid_pixels}
            save_frame(cache_root, scene_id, key, i, vi.diagonal,
                       {k: v for k, v in stats[i].items() if k != "cache"})

    lam = derive_relative_lambda(infos.values(), config.lambda_rel,
                                 config.lambda_abs)
    pos = torch.cat([d[d > 0] for d in infos.values() if (d > 0).any()])
    result = greedy_static_select(infos, budget, criterion, lam)

    manifest = {
        "schema_version": 1, "scene": scene_id, "method": "dopt_pose",
        "selection_mode": "batch", "claim_scope": "active_pose_only",
        "budget": budget, "criterion": criterion,
        "greedy_order": result.greedy_order,
        "replay_order": result.replay_order,
        "output_space": config.output_space, "weight_mode": config.weight_mode,
        "probes": config.num_probes, "lambda_rel": config.lambda_rel,
        "lambda_abs": config.lambda_abs, "lambda_values": [lam],
        "b_positive_median": float(pos.median()) if pos.numel() else 0.0,
        "b_positive_frac": float(sum((d > 0).float().mean() for d in infos.values()) / len(infos)),
        "gaussian_counts": [int(next(iter(infos.values())).shape[0])],
        "candidate_rgb_accessed": False, "candidate_cue_accessed": False,
        "gt_accessed": False, "post_refinement": False,
        "query_count": len(views), "checkpoint_hash": ckpt,
        "config_hash": config_hash(base_key),
        "per_frame": {str(i): {**stats[i],
                               "score_steps": {str(s.step): s.candidate_scores.get(i)
                                               for s in result.steps}}
                      for i in infos},
        "steps": [{"step": s.step, "frame": s.selected_frame_id,
                   "score": s.selected_score} for s in result.steps],
        "selection_seconds": round(time.time() - t0, 2),
    }
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return result.replay_order, manifest
