# Cue montage: quality-toxic frames vs oracle-best-5 (user request 2026-07-20).
#
# Visual check of the toxicity_decompose finding — quality-toxic frames should
# show large / fragmented / arbitrary cues (SSIM+L1+SAM2 change map) that the
# up-only fusion loss cannot undo, while oracle-best-5 frames should show
# clean, localized cues on the real change. Per scene, two blocks (toxic /
# best), each frame = [original | cue heatmap | overlay].
#
# Needs the --dump_cues artifacts at output_subset/cue_montage/<scene>/cues.pt.
#   python experiments/cue_montage.py --scenes Zen Porch Printing_area
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DUMP_ROOT = os.path.join(REPO, "output_subset", "cue_montage")
FIG_DIR = os.path.join(REPO, "experiments", "figures")
ORACLE_CSV = os.path.join(REPO, "experiments", "oracle_search_results.csv")
DECOMP_CSV = os.path.join(REPO, "experiments", "toxicity_decompose_scores.csv")


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


def montage_scene(scene: str, cue_vmax: float):
    blob = torch.load(os.path.join(DUMP_ROOT, scene, "cues.pt"),
                      map_location="cpu", weights_only=False)
    cues = blob["cues"]
    if cues.ndim == 4:
        cues = cues[:, 0]
    originals = blob["originals"]
    marg = marginals(scene)

    toxic = quality_toxic(scene)
    best = oracle_best5(scene)
    tox_ids = [t[0] for t in toxic]
    groups = [("QUALITY-TOXIC (capped: no company rescues)", "#c0392b",
               [(f, f"toxic  marg {m:+.3f}  ceil {z:.1f}σ")
                for f, m, z in toxic]),
              ("ORACLE-BEST-5 (max-mIoU combo)", "#1e7d34",
               [(f, f"best   marg {marg.get(f, 0):+.3f}") for f in best])]

    n_rows = sum(len(g[2]) for g in groups)
    fig_h = 2.4 * n_rows + 0.6 * len(groups)
    fig, axes = plt.subplots(n_rows, 3, figsize=(9.5, fig_h),
                             squeeze=False)
    fig.suptitle(f"{scene} — cue (SSIM+L1+SAM2 change map) — "
                 f"toxic vs oracle-best-5", fontsize=13, y=0.997)

    row = 0
    for title, color, frames in groups:
        for k, (f, label) in enumerate(frames):
            orig = to_hwc(originals[f])
            cue = cues[f].numpy()
            cue_area = float((cue > 0.5).mean()) * 100  # % pixels lit
            ax0, ax1, ax2 = axes[row]
            ax0.imshow(orig)
            ax1.imshow(cue, cmap="inferno", vmin=0, vmax=cue_vmax)
            ax2.imshow(orig)
            ax2.imshow(cue, cmap="inferno", vmin=0, vmax=cue_vmax, alpha=0.55)
            for ax in (ax0, ax1, ax2):
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_edgecolor(color); s.set_linewidth(2.5)
            ax0.set_ylabel(f"frame {f}", fontsize=10, color=color,
                           fontweight="bold")
            if k == 0:
                ax0.set_title(f"[{title}]  original", fontsize=10,
                              color=color, loc="left", fontweight="bold")
                ax1.set_title("cue heatmap", fontsize=9)
                ax2.set_title("overlay", fontsize=9)
            ax1.set_xlabel(f"{label}   |  cue>0.5: {cue_area:.0f}% px",
                           fontsize=8, color=color)
            row += 1

    fig.tight_layout(rect=[0, 0, 1, 0.99])
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, f"montage_quality_{scene}.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    # scene-level cue-area contrast (the headline number)
    tox_area = np.mean([float((cues[f].numpy() > 0.5).mean()) * 100
                        for f in tox_ids]) if tox_ids else float("nan")
    best_area = np.mean([float((cues[f].numpy() > 0.5).mean()) * 100
                         for f in best])
    print(f"{scene:14s} toxic {tox_ids} cue-lit {tox_area:5.1f}%  |  "
          f"best {best} cue-lit {best_area:5.1f}%  -> {out}")


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
