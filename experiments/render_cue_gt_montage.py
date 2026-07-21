# 4-panel montage for EVERY inference frame: original RGB | 3DGS reference
# render | combined cue C_v | GT change mask. Extends the earlier 3-panel
# cue-vs-GT report (experiments/cue_gt_report.py) with the reference render the
# cue is actually differenced against, so render-failure false alarms are
# visible per frame.
#
# Pulls everything from the cached frame_context.pt the Part 3.1 runner wrote
# (no GPU, no model reload).
#
#   python experiments/render_cue_gt_montage.py                 # all 10 scenes
#   python experiments/render_cue_gt_montage.py --scenes Zen Garden
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CTX = os.path.join(REPO, "outputs", "change_nbv", "diagnostics",
                   "rchange_importance", "Instance_1")
OUT = os.path.join(REPO, "experiments", "figures", "render_cue_gt_instance1")
SCENES = ["Cantina", "Garden", "Lounge", "Lunch_room", "Meeting_room",
          "Playground", "Porch", "Pots", "Printing_area", "Zen"]


def frame_png(scene, i, ctx, out_dir, tau):
    orig = ctx["original"][i].float().numpy() / 255.0
    ref = ctx["reference_render"][i].float().numpy() / 255.0
    cue = ctx["cue"][i].float().numpy()
    gt = ctx["gt"][i].numpy()
    name = ctx["image_names"][i]

    cb = cue > tau
    tp = int((cb & gt).sum())
    prec = tp / max(int(cb.sum()), 1)
    rec = tp / max(int(gt.sum()), 1)
    dark = float(((orig.mean(0) - ref.mean(0)) > 0.10).mean())

    fig, ax = plt.subplots(1, 4, figsize=(15.5, 3.1))
    ax[0].imshow(orig.transpose(1, 2, 0))
    ax[0].set_title("original RGB (query view)", fontsize=10)
    ax[1].imshow(ref.transpose(1, 2, 0))
    ax[1].set_title(f"3DGS reference render "
                    f"({100*dark:.0f}% ≥0.10 too dark)", fontsize=10)
    ax[2].imshow(cue, cmap="inferno", vmin=0, vmax=1.5)
    ax[2].contour(cue, levels=[tau], colors="#3ad6ff", linewidths=1.0)
    ax[2].set_title(f"combined cue C_v (τ={tau})", fontsize=10)
    ax[3].imshow(gt, cmap="gray")
    ax[3].set_title("GT change mask", fontsize=10)
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(f"{scene} / frame {i:02d} ({name})   ·   "
                 f"cue lights {100*cb.mean():.1f}% (GT {100*gt.mean():.2f}%)   "
                 f"·   precision {prec:.3f}   recall {rec:.2f}",
                 fontsize=11, y=1.05)
    fig.savefig(os.path.join(out_dir, f"frame_{i:02d}_{name}.png"),
                dpi=95, bbox_inches="tight")
    plt.close(fig)
    return prec, rec, dark


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["all"])
    ap.add_argument("--tau", type=float, default=0.5)
    args = ap.parse_args()
    scenes = SCENES if args.scenes == ["all"] else args.scenes

    total = 0
    for scene in scenes:
        ctx_path = os.path.join(CTX, scene, "frame_context.pt")
        if not os.path.exists(ctx_path):
            print(f"skip {scene}: no frame_context.pt")
            continue
        ctx = torch.load(ctx_path, weights_only=False)
        out_dir = os.path.join(OUT, scene)
        os.makedirs(out_dir, exist_ok=True)
        n = len(ctx["image_names"])
        precs = []
        for i in range(n):
            p, r, d = frame_png(scene, i, ctx, out_dir, args.tau)
            precs.append(p)
        total += n
        print(f"{scene}: {n} frames -> {os.path.relpath(out_dir, REPO)}  "
              f"(mean cue precision {np.mean(precs):.3f})")
    print(f"\n{total} montages under {os.path.relpath(OUT, REPO)}/<scene>/")


if __name__ == "__main__":
    main()
