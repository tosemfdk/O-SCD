# Why does Playground's frame 14 poison the reconstruction while frame 0 helps?
#
# Playground is the scene where 5 frames beat all 25 by the widest margin
# (oracle-5 0.5213 vs all-25 0.3590). The oracle map's per-frame marginal value
# ranks frame 0 best (+0.035) and frame 14 worst (-0.032). This puts the two
# side by side with the reference render, which is what the cue is actually
# differenced against.
#
#   python experiments/playground_frame_compare.py
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CTX = os.path.join(REPO, "outputs", "change_nbv", "diagnostics",
                   "rchange_importance", "Instance_1")
ORACLE = os.path.join(REPO, "experiments", "oracle_search_results.csv")
INK, INK2, GRID = "#1a1a19", "#5f5e58", "#e6e5e0"
GREEN, RED = "#008300", "#e34948"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "text.color": INK, "font.size": 10,
})


def marginals(scene):
    rows = [r for r in csv.DictReader(open(ORACLE)) if r["scene"] == scene]
    five = [(float(r["miou_query"]), {int(x) for x in r["combo"].split("-")})
            for r in rows if len(r["combo"].split("-")) == 5]
    inc, exc = defaultdict(list), defaultdict(list)
    for m, c in five:
        for f in range(25):
            (inc if f in c else exc)[f].append(m)
    return {f: float(np.mean(inc[f]) - np.mean(exc[f]))
            for f in range(25) if len(inc[f]) >= 3 and len(exc[f]) >= 3}


def stats(orig, ref, cue, gt, tau):
    """Reconstruction fidelity of the reference render + cue-vs-GT at tau."""
    lo, lr = orig.mean(0), ref.mean(0)
    cb = cue > tau
    tp = int((cb & gt).sum())
    return {
        "render_l1": float(np.abs(lo - lr).mean()),
        "render_dark_frac": float(((lo - lr) > 0.10).mean()),
        "cue_area": float(cb.mean()),
        "gt_area": float(gt.mean()),
        "precision": tp / max(int(cb.sum()), 1),
        "recall": tp / max(int(gt.sum()), 1),
        "iou": tp / max(int((cb | gt).sum()), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="Playground")
    ap.add_argument("--frames", type=int, nargs="+", default=[0, 14])
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--out", default=os.path.join(REPO, "experiments",
                                                  "figures"))
    args = ap.parse_args()

    ctx = torch.load(os.path.join(CTX, args.scene, "frame_context.pt"),
                     weights_only=False)
    marg = marginals(args.scene)
    n = len(args.frames)
    fig = plt.figure(figsize=(19.5, 3.5 * n + 0.6))
    gs = GridSpec(n, 5, figure=fig, hspace=0.32, wspace=0.05)

    for r, fr in enumerate(args.frames):
        orig = ctx["original"][fr].float().numpy() / 255.0
        ref = ctx["reference_render"][fr].float().numpy() / 255.0
        cue = ctx["cue"][fr].float().numpy()
        gt = ctx["gt"][fr].numpy()
        s = stats(orig, ref, cue, gt, args.tau)
        m = marg.get(fr, float("nan"))
        colour = GREEN if m > 0 else RED

        panels = [
            (orig.transpose(1, 2, 0), "original RGB (query view)", None),
            (ref.transpose(1, 2, 0), "3DGS reference render", None),
            (np.abs(orig - ref).mean(0), "render error |orig − ref|", "magma"),
            (cue, f"combined cue C_v  (τ={args.tau} contour)", "inferno"),
            (gt, "GT change mask", "gray"),
        ]
        for c, (img, title, cmap) in enumerate(panels):
            ax = fig.add_subplot(gs[r, c])
            kw = {"vmin": 0, "vmax": 1.5} if cmap == "inferno" else {}
            if cmap == "magma":
                kw = {"vmin": 0, "vmax": 0.35}
            ax.imshow(img, cmap=cmap, **kw)
            if c == 3:
                ax.contour(cue, levels=[args.tau], colors="#3ad6ff",
                           linewidths=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=10)
            if c == 0:
                ax.set_ylabel(f"frame {fr}", fontsize=12, color=colour,
                              fontweight="bold")
        fig.text(0.5, 1 - (r + 0.02) / n * 0.96,
                 f"frame {fr} ({ctx['image_names'][fr]})   ·   "
                 f"oracle marginal {m:+.4f} mIoU   ·   "
                 f"render L1 {s['render_l1']:.3f}, "
                 f"{100*s['render_dark_frac']:.1f}% of frame rendered ≥0.10 "
                 f"too dark   ·   cue lights {100*s['cue_area']:.1f}% "
                 f"(GT {100*s['gt_area']:.2f}%), precision {s['precision']:.3f}, "
                 f"recall {s['recall']:.2f}",
                 ha="center", fontsize=11, color=colour, fontweight="bold")
        print(f"frame {fr:2d}  marginal {m:+.4f}  render_L1 {s['render_l1']:.4f}"
              f"  dark {100*s['render_dark_frac']:5.2f}%"
              f"  cue {100*s['cue_area']:5.2f}%  gt {100*s['gt_area']:.2f}%"
              f"  prec {s['precision']:.3f}  rec {s['recall']:.2f}")

    out = os.path.join(args.out, f"{args.scene.lower()}_frame_compare.png")
    fig.savefig(out, dpi=125, bbox_inches="tight")
    plt.close(fig)
    print("saved", os.path.relpath(out, REPO))


if __name__ == "__main__":
    main()
