# Cue montage: quality-toxic frames vs oracle-best-5 (user request 2026-07-20).
#
# Visual check of the toxicity_decompose finding — quality-toxic frames should
# show large / fragmented / arbitrary cues (SSIM+L1+SAM2 change map) that the
# up-only fusion loss cannot undo, while oracle-best-5 frames should show
# clean, localized cues on the real change. Per scene, two blocks (toxic /
# best), each frame = [original | cue heatmap | GT change mask]. The GT column
# makes "does the cue fire OUTSIDE the real change?" directly visible, and
# in-GT% quantifies it (fraction of lit cue pixels that fall inside GT).
#
# Needs the --dump_cues artifacts at output_subset/cue_montage/<scene>/cues.pt.
#   python experiments/cue_montage.py --scenes Zen Porch Printing_area
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTANCE = "Instance_1"
DUMP_ROOT = os.path.join(REPO, "output_subset", "cue_montage")
FIG_DIR = os.path.join(REPO, "experiments", "figures")
ORACLE_CSV = os.path.join(REPO, "experiments", "oracle_search_results.csv")
DECOMP_CSV = os.path.join(REPO, "experiments", "toxicity_decompose_scores.csv")


def gt_mask(scene: str, image_name: str, hw) -> np.ndarray:
    """Binary GT change mask for one inference frame, resized to (H, W)."""
    p = os.path.join(REPO, "data", "PASLCD", INSTANCE, scene, "gt_mask",
                     image_name + ".png")
    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return np.zeros(hw, np.float32)
    m = cv2.resize(m, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.float32)


def oracle_best5(scene: str) -> list[int]:
    best, best_m = None, -1.0
    for r in csv.DictReader(open(ORACLE_CSV)):
        if r["scene"] != scene or len(r["combo"].split("-")) != 5:
            continue
        m = float(r["miou_query"])
        if m > best_m:
            best_m, best = m, r["combo"]
    return sorted(int(x) for x in best.split("-"))


def quality_toxic(scene: str) -> list[int]:
    """Ceiling-capped toxic frames from the decomposition: marginal < 0 AND
    ceil_gap_z > 1 (no company rescues them = intrinsic quality)."""
    out = []
    for r in csv.DictReader(open(DECOMP_CSV)):
        if r["scene"] != scene:
            continue
        if float(r["marginal"]) < 0 and float(r["ceil_gap_z"]) > 1.0:
            out.append((int(r["frame"]), float(r["marginal"]),
                        float(r["ceil_gap_z"])))
    out.sort(key=lambda t: t[1])  # most toxic first
    return out


def to_hwc(img: torch.Tensor) -> np.ndarray:
    a = img.numpy()
    if a.ndim == 3 and a.shape[0] in (1, 3):
        a = np.transpose(a, (1, 2, 0))
    return np.clip(a, 0, 1)


def marginals(scene: str) -> dict[int, float]:
    combos = defaultdict(list)
    present = defaultdict(list)
    rows = [r for r in csv.DictReader(open(ORACLE_CSV))
            if r["scene"] == scene and len(r["combo"].split("-")) == 5]
    best = {}
    for r in rows:
        c = tuple(sorted(int(x) for x in r["combo"].split("-")))
        best[c] = max(float(r["miou_query"]), best.get(c, -1))
    out = {}
    for f in range(25):
        w = [m for c, m in best.items() if f in c]
        wo = [m for c, m in best.items() if f not in c]
        if w and wo:
            out[f] = float(np.mean(w) - np.mean(wo))
    return out


def _in_gt(cue: np.ndarray, gt: np.ndarray) -> float:
    """Fraction of lit cue pixels (cue>0.5) that fall inside the GT change
    region — cue precision against ground truth. Low = cue fires off-target."""
    lit = cue > 0.5
    n = int(lit.sum())
    return float((lit & (gt > 0.5)).sum()) / n * 100 if n else float("nan")


def montage_scene(scene: str, cue_vmax: float):
    blob = torch.load(os.path.join(DUMP_ROOT, scene, "cues.pt"),
                      map_location="cpu", weights_only=False)
    cues = blob["cues"]
    if cues.ndim == 4:
        cues = cues[:, 0]
    originals = blob["originals"]
    names = blob["image_names"]
    marg = marginals(scene)

    toxic = quality_toxic(scene)
    best = oracle_best5(scene)
    tox_ids = [t[0] for t in toxic]
    groups = [("QUALITY-TOXIC  (ceiling-capped: no company rescues)", "#c0392b",
               [(f, f"marg {m:+.3f}  ·  ceil {z:.1f}σ") for f, m, z in toxic]),
              ("ORACLE-BEST-5  (max-mIoU combo)", "#1e7d34",
               [(f, f"marg {marg.get(f, 0):+.3f}") for f in best])]

    n_frames = sum(len(g[2]) for g in groups)
    # grid: one thin header row per group + one row per frame
    ratios, plan = [], []   # plan: ('hdr', title, color) or ('frame', ...)
    for title, color, frames in groups:
        ratios.append(0.22); plan.append(("hdr", title, color))
        for k, (f, label) in enumerate(frames):
            ratios.append(1.0); plan.append(("frame", f, label, color, k == 0))
    fig_h = 2.55 * n_frames + 0.7 * len(groups) + 0.5
    fig = plt.figure(figsize=(10.5, fig_h))
    gs = GridSpec(len(plan), 3, height_ratios=ratios, figure=fig,
                  hspace=0.28, wspace=0.06,
                  top=0.965, bottom=0.008, left=0.055, right=0.99)
    fig.suptitle(f"{scene} — cue (SSIM+L1+SAM2 change map) vs GT change mask",
                 fontsize=14, y=0.998, fontweight="bold")

    for r, item in enumerate(plan):
        if item[0] == "hdr":
            _, title, color = item
            axh = fig.add_subplot(gs[r, :]); axh.axis("off")
            axh.add_patch(plt.Rectangle((0, 0.05), 1, 0.9, transform=axh.transAxes,
                                        color=color, alpha=0.12, zorder=0))
            axh.text(0.5, 0.5, title, ha="center", va="center", color=color,
                     fontsize=12, fontweight="bold", transform=axh.transAxes)
            continue
        _, f, label, color, first = item
        orig = to_hwc(originals[f])
        cue = cues[f].numpy()
        gt = gt_mask(scene, names[f], cue.shape)
        cue_lit = float((cue > 0.5).mean()) * 100
        in_gt = _in_gt(cue, gt)
        ax0 = fig.add_subplot(gs[r, 0]); ax0.imshow(orig)
        ax1 = fig.add_subplot(gs[r, 1])
        ax1.imshow(cue, cmap="inferno", vmin=0, vmax=cue_vmax)
        ax2 = fig.add_subplot(gs[r, 2]); ax2.imshow(gt, cmap="gray", vmin=0, vmax=1)
        for ax in (ax0, ax1, ax2):
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor(color); s.set_linewidth(2.5)
        ax0.set_ylabel(f"frame {f}", fontsize=11, color=color, fontweight="bold")
        if first:
            ax0.set_title("original", fontsize=10)
            ax1.set_title("cue (SSIM+L1+SAM2)", fontsize=10)
            ax2.set_title("GT change mask", fontsize=10)
        ax0.set_xlabel(label, fontsize=8.5, color=color)
        ax1.set_xlabel(f"cue-lit {cue_lit:.0f}% px", fontsize=8.5)
        ax2.set_xlabel(f"in-GT {in_gt:.0f}% of cue", fontsize=8.5,
                       fontweight="bold")

    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, f"montage_quality_{scene}.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    tox_ingt = np.mean([_in_gt(cues[f].numpy(),
                               gt_mask(scene, names[f], cues[f].shape[-2:]))
                        for f in tox_ids]) if tox_ids else float("nan")
    best_ingt = np.mean([_in_gt(cues[f].numpy(),
                                gt_mask(scene, names[f], cues[f].shape[-2:]))
                         for f in best])
    print(f"{scene:14s} toxic {tox_ids} in-GT {tox_ingt:4.0f}%  |  "
          f"best {best} in-GT {best_ingt:4.0f}%  -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+",
                    default=["Zen", "Porch", "Printing_area"])
    ap.add_argument("--cue_vmax", type=float, default=1.5)
    args = ap.parse_args()
    print("cue-lit = % pixels with cue > 0.5 (the fusion trigger threshold)\n")
    for sc in args.scenes:
        montage_scene(sc, args.cue_vmax)


if __name__ == "__main__":
    main()
