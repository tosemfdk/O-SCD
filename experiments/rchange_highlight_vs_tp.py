# "What is actually lit, and how much of it is real?" — the picture behind the
# §13 D top-percentile numbers.
#
# For a frame, take the all-25 prediction P_v, keep only its top-k% pixels by
# importance, and colour that surviving highlight by TP/FP. The title carries
# the precision of the highlight, so the drop from lighting everything (the
# prediction's own precision) to lighting the top slice is readable directly.
#
#   python experiments/rchange_highlight_vs_tp.py --scenes Zen Porch Garden Printing_area
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(REPO, "outputs", "change_nbv", "diagnostics",
                    "rchange_importance", "Instance_1")
HIGHLIGHT = {"Zen": [3, 4, 6], "Porch": [0, 7], "Printing_area": [4],
             "Garden": [15, 23]}
PERCENTS = (10, 20, 30, 50, 100)
TP_G, FP_R, FN_B = (0.15, 0.65, 0.15), (0.85, 0.15, 0.15), (0.15, 0.30, 0.85)
INK, INK2, GRID, MUTED = "#1a1a19", "#5f5e58", "#e6e5e0", "#b5b4ac"
BLUE, GREEN, RED = "#2a78d6", "#008300", "#e34948"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "bold",
})


def highlight_rgb(keep, gt):
    """White background, TP green, FP red — only over the kept highlight."""
    img = np.ones((*gt.shape, 3), np.float32)
    img[keep & gt] = TP_G
    img[keep & ~gt] = FP_R
    return img


def error_rgb(pred, gt):
    img = np.ones((*gt.shape, 3), np.float32)
    img[pred & gt] = TP_G
    img[pred & ~gt] = FP_R
    img[~pred & gt] = FN_B
    return img


def topk_mask(score, pred, pct):
    """Top pct% of the PREDICTION's pixels by importance (pct=100 → all of it)."""
    idx = np.flatnonzero(pred.ravel())
    if idx.size == 0:
        return np.zeros_like(pred)
    k = max(1, int(round(idx.size * pct / 100.0)))
    order = idx[np.argsort(-score.ravel()[idx])][:k]
    keep = np.zeros(pred.size, bool)
    keep[order] = True
    return keep.reshape(pred.shape)


def frame_figure(scene, frame, variant, ctx, out_dir):
    heat = torch.load(os.path.join(ROOT, scene, "renders", variant,
                                   f"frame_{frame:02d}.pt"),
                      weights_only=False).float().numpy()
    pred = ctx["pred"][frame].numpy()
    gt = ctx["gt"][frame].numpy()
    rgb = ctx["original"][frame].permute(1, 2, 0).numpy()

    n = 3 + len(PERCENTS)
    fig, axes = plt.subplots(1, n, figsize=(2.9 * n, 2.9))
    axes[0].imshow(rgb)
    axes[0].set_title("original RGB", fontsize=10)
    axes[1].imshow(error_rgb(pred, gt))
    base = (pred & gt).sum() / max(pred.sum(), 1)
    axes[1].set_title(f"all-25 prediction\nTP {100*base:.0f}% of what it lights",
                      fontsize=9)
    lo, hi = np.quantile(heat, 0.01), np.quantile(heat, 0.99)
    axes[2].imshow(heat, cmap="inferno", vmin=lo, vmax=hi)
    axes[2].set_title(f"importance\n{variant}", fontsize=9)

    precs = []
    for ax, pct in zip(axes[3:], PERCENTS):
        keep = topk_mask(heat, pred, pct)
        tp = int((keep & gt).sum())
        prec = tp / max(int(keep.sum()), 1)
        precs.append(prec)
        ax.imshow(highlight_rgb(keep, gt))
        ax.set_title(f"top {pct}% of the prediction\n"
                     f"TP {100*prec:.0f}%  ·  recall {tp/max(int((pred&gt).sum()),1):.2f}",
                     fontsize=9)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    axes[-1].legend(handles=[Patch(color=TP_G, label="TP"),
                             Patch(color=FP_R, label="FP")],
                    loc="upper right", fontsize=8, framealpha=0.9,
                    handlelength=1.0)
    fig.suptitle(f"{scene} / frame {frame} ({ctx['image_names'][frame]}) — "
                 f"how much of the lit region is real change",
                 fontsize=13, y=1.06)
    out = os.path.join(out_dir, f"highlight_{scene}_f{frame:02d}.png")
    fig.savefig(out, dpi=115, bbox_inches="tight")
    plt.close(fig)
    return base, precs


def scene_curve(scene, variant, ctx, out_dir, pcts=(1, 2, 5, 10, 20, 30, 50,
                                                   75, 100)):
    """Precision of the highlight vs how much of the prediction is kept,
    every frame in the scene plus the mean."""
    n_frames = len(ctx["image_names"])
    curves, bases = [], []
    for i in range(n_frames):
        heat = torch.load(os.path.join(ROOT, scene, "renders", variant,
                                       f"frame_{i:02d}.pt"),
                          weights_only=False).float().numpy()
        pred, gt = ctx["pred"][i].numpy(), ctx["gt"][i].numpy()
        if pred.sum() == 0:
            continue
        bases.append((pred & gt).sum() / pred.sum())
        curves.append([(topk_mask(heat, pred, p) & gt).sum()
                       / max(topk_mask(heat, pred, p).sum(), 1) for p in pcts])
    arr = np.array(curves)
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for row in arr:
        ax.plot(pcts, row, color=MUTED, lw=0.9, zorder=1)
    ax.plot(pcts, arr.mean(0), color=BLUE, lw=2.6, marker="o", ms=6,
            zorder=3, label=f"mean over {len(arr)} frames")
    ax.axhline(float(np.mean(bases)), color=RED, lw=1.5, ls="--", zorder=2,
               label=f"lighting the whole prediction = {np.mean(bases):.3f}")
    ax.set_xscale("log")
    ax.set_xticks(list(pcts))
    ax.set_xticklabels([str(p) for p in pcts])
    ax.set_xlabel("% of the all-25 prediction kept, highest importance first")
    ax.set_ylabel("TP fraction of the lit region (precision)")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{scene} — tightening the highlight buys precision\n"
                 f"importance = {variant}   ·   gray = individual frames",
                 loc="left")
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower left")
    fig.savefig(os.path.join(out_dir, f"highlight_curve_{scene}.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    return arr.mean(0), float(np.mean(bases))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+",
                    default=["Zen", "Porch", "Garden", "Printing_area"])
    ap.add_argument("--variant", default="I2_mean_support")
    ap.add_argument("--out", default=os.path.join(REPO, "experiments",
                                                  "figures"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for scene in args.scenes:
        ctx = torch.load(os.path.join(ROOT, scene, "frame_context.pt"),
                         weights_only=False)
        for fr in HIGHLIGHT.get(scene, []):
            base, precs = frame_figure(scene, fr, args.variant, ctx, args.out)
            print(f"{scene}/{fr}: whole prediction {base:.3f} -> " +
                  "  ".join(f"top{p}% {v:.3f}"
                            for p, v in zip(PERCENTS, precs)))
        mean, base = scene_curve(scene, args.variant, ctx, args.out)
        print(f"{scene}: scene mean, whole={base:.3f}, "
              f"top10%={mean[3]:.3f}, top1%={mean[0]:.3f}")


if __name__ == "__main__":
    main()
