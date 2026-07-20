# Case montage for two Garden frames at tau=0.5 vs 0.9 (user request
# 2026-07-20). Shows what the recall/precision trade actually costs on real
# pixels: what the cue drops between the two thresholds is the thing to judge,
# not the aggregate IoU.
#
# One PNG per (frame, tau): original RGB | 3DGS render | cue + tau contour |
# error map (TP green / FP red / FN blue).
#
#   python experiments/cue_tau_case_montage.py
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments"))
from cue_gt_report import DUMP_ROOT, error_rgb, gt_for, to_hwc  # noqa: E402

OUT = os.path.join(REPO, "experiments", "figures")
SCENE = "Garden"
FRAMES = [15, 23]
TAUS = [0.5, 0.9]
INK, INK2, GRID = "#1a1a19", "#5f5e58", "#e6e5e0"
TP_G, FP_R, FN_B = "#26a626", "#d92626", "#264dd9"


def main():
    blob = torch.load(os.path.join(DUMP_ROOT, SCENE, "cues.pt"),
                      map_location="cpu", weights_only=False)
    cues, names = blob["cues"], blob["image_names"]
    if cues.ndim == 4:
        cues = cues[:, 0]

    for i in FRAMES:
        cue = cues[i].numpy()
        name = names[i]
        gt = gt_for(SCENE, name, cue.shape)
        for tau in TAUS:
            cb = cue > tau
            tp = int((cb & gt).sum()); fp = int((cb & ~gt).sum())
            fn = int((~cb & gt).sum())
            prec = tp / (tp + fp) if tp + fp else float("nan")
            rec = tp / (tp + fn) if tp + fn else float("nan")
            iou = tp / (tp + fp + fn) if tp + fp + fn else float("nan")

            fig, ax = plt.subplots(1, 4, figsize=(17.5, 3.4))
            ax[0].imshow(to_hwc(blob["originals"][i]))
            ax[0].set_title("original RGB (query view)", fontsize=11)
            ax[1].imshow(to_hwc(blob["renders"][i]))
            ax[1].set_title("3DGS render (pre-change scene)", fontsize=11)
            im = ax[2].imshow(cue, cmap="inferno", vmin=0, vmax=1.5)
            ax[2].contour(cue, levels=[tau], colors="#3ad6ff", linewidths=1.2)
            ax[2].set_title(f"cue (continuous) + τ={tau} contour", fontsize=11)
            fig.colorbar(im, ax=ax[2], fraction=0.032, pad=0.02)
            ax[3].imshow(error_rgb(cb, gt))
            ax[3].set_title(f"binarized at τ={tau}  vs  GT", fontsize=11)
            ax[3].legend(handles=[Patch(color=TP_G, label="TP  caught"),
                                  Patch(color=FP_R, label="FP  false alarm"),
                                  Patch(color=FN_B, label="FN  missed")],
                         loc="upper right", fontsize=8, framealpha=0.9,
                         handlelength=1.1, borderpad=0.4)
            for a in ax:
                a.set_xticks([]); a.set_yticks([])
            fig.suptitle(
                f"{SCENE} / frame {i} ({name})   ·   τ = {tau}   ·   "
                f"precision {prec:.2f}   recall {rec:.2f}   IoU {iou:.2f}   ·   "
                f"lit {100*(tp+fp)/cue.size:.1f}% of frame, GT {100*(tp+fn)/cue.size:.1f}%",
                fontsize=12.5, y=1.06, color=INK)
            out = os.path.join(OUT, f"cue_case_{SCENE}_f{i:02d}_tau{tau}.png")
            fig.savefig(out, dpi=130, bbox_inches="tight")
            plt.close(fig)
            print(f"saved {os.path.basename(out)}  "
                  f"prec {prec:.2f} rec {rec:.2f} IoU {iou:.2f}")


if __name__ == "__main__":
    main()
