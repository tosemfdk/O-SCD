# Commit an ACTUALLY OBSERVED view into a target information state (stage 12).
#
#   python tools/update_target_information.py \
#       --ply output/.../point_cloud.ply --sh-degree 3 \
#       --state runs/target_nbv/12345/information_state.pt \
#       --config runs/target_nbv/12345/config_resolved.json \
#       --camera selected_view.json --view-id sel_0
#
# The geometry Fisher update needs no RGB image: the Jacobian of the current
# model at the observed pose is the information the view contributes. Change
# modes (Beta counts from an observed change mask) arrive in a later gate.

from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ply", required=True)
    p.add_argument("--sh-degree", type=int, default=3)
    p.add_argument("--state", required=True, help="information_state.pt from a selection run")
    p.add_argument("--config", required=True, help="config_resolved.json from the same run")
    p.add_argument("--camera", required=True,
                   help="JSON file with ONE observed camera entry (or a best_view.json)")
    p.add_argument("--view-id", default=None,
                   help="unique view id; defaults to the camera entry's view_id")
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--fovy-deg", type=float, default=60.0)
    p.add_argument("--white-background", action="store_true")
    p.add_argument("--output-state", default=None,
                   help="write updated state here (default: overwrite --state)")
    return p.parse_args(argv)


def main(argv=None) -> bool:
    args = parse_args(argv)

    from target_nbv import target_registry
    from target_nbv.config import TargetNBVConfig
    from target_nbv.info_builder import TargetInformationBuilder
    from target_nbv.io_utils import camera_from_json, default_pipe, load_gaussian_model
    from target_nbv.scorers.exact import commit_observed_view

    with open(args.config) as f:
        cfg_dict = json.load(f)
    cfg_dict.pop("_selection_meta", None)
    cfg = TargetNBVConfig.from_dict(cfg_dict)

    builder = TargetInformationBuilder.load(args.state, cfg)
    model = load_gaussian_model(args.ply, args.sh_degree)
    row = target_registry.resolve_target_by_persistent_id(model, builder.state.target_pid)

    with open(args.camera) as f:
        entry = json.load(f)
    if isinstance(entry, list):
        if len(entry) != 1:
            raise ValueError("--camera must contain exactly one camera entry")
        entry = entry[0]
    view_id, cam = camera_from_json(entry, math.radians(args.fovy_deg),
                                    args.width, args.height,
                                    fallback_id=f"view{len(builder.state.observed_view_ids)}")
    view_id = args.view_id or view_id

    pipe = default_pipe()
    background = (torch.ones if args.white_background else torch.zeros)(3, device="cuda")
    logdet_before = float(np.linalg.slogdet(builder.H_prior())[1])
    changed = commit_observed_view(builder, view_id, model, cam, row,
                                   pipe, background, cfg)
    logdet_after = float(np.linalg.slogdet(builder.H_prior())[1])

    out_path = args.output_state or args.state
    builder.save(out_path)
    if changed:
        print(f"committed view {view_id!r}: logdet {logdet_before:.4f} -> "
              f"{logdet_after:.4f} (gain {0.5 * (logdet_after - logdet_before):.6f}), "
              f"{len(builder.state.observed_view_ids)} views -> {out_path}")
    else:
        print(f"view {view_id!r} already in state; nothing to do")
    return changed


if __name__ == "__main__":
    main()
