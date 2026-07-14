# Dry-run target-NBV selection CLI (stage 12).
#
#   python tools/select_target_nbv.py \
#       --ply output/.../point_cloud.ply --sh-degree 3 \
#       --target-gaussian-id 12345 \
#       --observed-cameras cams.json \
#       --mode geometry_exact --output-dir runs/target_nbv/12345
#
# Emits pose + score artifacts only; the information state on disk is NOT
# updated here — run tools/update_target_information.py after the selected
# view has actually been observed. Geometry Fisher needs no candidate RGB.

from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ply", required=True, help="reference point_cloud.ply")
    p.add_argument("--sh-degree", type=int, default=3)
    p.add_argument("--target-gaussian-id", type=int, required=True,
                   help="persistent Gaussian ID (see .pid.pt sidecar)")
    p.add_argument("--observed-cameras", required=True,
                   help="JSON list of observed camera poses (see target_nbv/io_utils.py)")
    p.add_argument("--config", default=None,
                   help="optional TargetNBVConfig JSON; CLI flags below override it")
    p.add_argument("--mode", default=None, choices=["geometry_proxy", "geometry_exact"])
    p.add_argument("--candidate-count", type=int, default=None)
    p.add_argument("--proxy-top-k", type=int, default=None)
    p.add_argument("--exact-top-k", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--fovy-deg", type=float, default=60.0)
    p.add_argument("--white-background", action="store_true")
    p.add_argument("--output-dir", required=True)
    return p.parse_args(argv)


def build_config(args):
    from target_nbv.config import TargetNBVConfig
    if args.config:
        with open(args.config) as f:
            cfg = TargetNBVConfig.from_dict(json.load(f))
    else:
        cfg = TargetNBVConfig()
    if args.mode is not None:
        cfg.mode = args.mode
    if args.candidate_count is not None:
        cfg.candidates.count = args.candidate_count
    if args.proxy_top_k is not None:
        cfg.proxy.top_k = args.proxy_top_k
    if args.exact_top_k is not None:
        cfg.jacobian.exact_top_k = args.exact_top_k
    if args.seed is not None:
        cfg.seed = args.seed
    return cfg.validate()


def score_to_json(s) -> dict:
    d = s.candidate.to_json_dict()
    d.update(valid=s.valid, invalid_reason=s.invalid_reason,
             proxy_score=s.proxy_score, exact_score=s.exact_score,
             d_gain=s.d_gain, trace_gain=s.trace_gain, e_gain=s.e_gain,
             movement_cost=s.movement_cost)
    if s.visibility is not None:
        d.update(responsibility_sum=s.visibility.responsibility_sum,
                 occlusion_ratio=s.visibility.occlusion_ratio,
                 projected_radius_px=s.visibility.projected_radius_px,
                 visible_pixel_count=s.visibility.visible_pixel_count)
    return d


def save_debug_renders(out_dir, model, cfg, best, target_row, pipe, background):
    from torchvision.utils import save_image
    from gaussian_renderer import render
    from target_nbv.visibility import render_target_responsibility

    os.makedirs(out_dir, exist_ok=True)
    cam = best.candidate.minicam
    with torch.no_grad():
        rgb = render(cam, model, pipe, background)["render"].clamp(0, 1)
    save_image(rgb, os.path.join(out_dir, "best_view_rgb.png"))
    if cfg.debug.save_responsibility_maps:
        resp = render_target_responsibility(model, cam, target_row, pipe)
        save_image(resp / max(float(resp.max()), 1e-8),
                   os.path.join(out_dir, "best_view_responsibility.png"))


def main(argv=None) -> str:
    args = parse_args(argv)

    from target_nbv import target_registry
    from target_nbv.io_utils import (default_pipe, load_gaussian_model,
                                     load_observed_cameras, pose_matrices)
    from target_nbv.selector import build_information_state, select_next_view
    from target_nbv.types import TargetParameterSpec

    cfg = build_config(args)
    fovy = math.radians(args.fovy_deg)
    model = load_gaussian_model(args.ply, args.sh_degree)
    observed = load_observed_cameras(args.observed_cameras, fovy,
                                     args.width, args.height)
    pipe = default_pipe()
    background = (torch.ones if args.white_background else torch.zeros)(3, device="cuda")

    row = target_registry.resolve_target_by_persistent_id(model, args.target_gaussian_id)
    spec = TargetParameterSpec(parameter_names=list(cfg.parameter_groups))
    builder, skipped = build_information_state(
        model, row, args.target_gaussian_id, spec, observed, pipe, background, cfg)
    if skipped:
        print(f"skipped observed views (target not visible): {skipped}")

    result = select_next_view(model, args.target_gaussian_id, observed, cfg,
                              pipe, background, information_builder=builder,
                              width=args.width, height=args.height, fovy=fovy)

    out = args.output_dir
    os.makedirs(out, exist_ok=True)
    best = result.best

    best_json = score_to_json(best)
    best_json.update(pose_matrices(best.candidate.wxyz, best.candidate.position))
    best_json["target_persistent_id"] = args.target_gaussian_id
    with open(os.path.join(out, "best_view.json"), "w") as f:
        json.dump(best_json, f, indent=2)
    with open(os.path.join(out, "candidates.json"), "w") as f:
        json.dump([score_to_json(s) for s in result.scores], f, indent=2)
    with open(os.path.join(out, "config_resolved.json"), "w") as f:
        json.dump(result.config_snapshot, f, indent=2)
    with open(os.path.join(out, "runtime.json"), "w") as f:
        json.dump(result.runtime, f, indent=2)
    torch.save(result.H_before, os.path.join(out, "information_before.pt"))
    if result.predicted_H_after is not None:
        torch.save(result.predicted_H_after,
                   os.path.join(out, "predicted_information_after.pt"))
    # prior state for tools/update_target_information.py (selection left it unmodified)
    builder.save(os.path.join(out, "information_state.pt"))

    if cfg.debug.save_renders:
        save_debug_renders(os.path.join(out, "debug"), model, cfg, best,
                           row, pipe, background)

    print(f"best candidate {best.candidate.cand_id}: "
          f"exact={best.exact_score:.6f} proxy={best.proxy_score:.6f} "
          f"d_gain={best.d_gain:.6f} -> {out}/best_view.json")
    return out


if __name__ == "__main__":
    main()
