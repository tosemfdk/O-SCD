# Tight 4-panel montages for the 10 render-hole example frames: a title line
# and a one-line subtitle sitting directly on top of the panels, no whitespace
# band. Used for the Notion "reconstruction holes" section.
#
#   python experiments/render_holes_tight.py
from __future__ import annotations

import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

for _p in ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
           "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"):
    if os.path.exists(_p):
        fm.fontManager.addfont(_p)
        plt.rcParams["font.family"] = fm.FontProperties(fname=_p).get_name()
        plt.rcParams["axes.unicode_minus"] = False
        break

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CTX = os.path.join(REPO, "outputs", "change_nbv", "diagnostics",
                   "rchange_importance", "Instance_1")
OUT = os.path.join(REPO, "experiments", "figures", "render_holes_tight")

FRAMES = [
    ("Zen", 24), ("Zen", 13), ("Cantina", 13), ("Pots", 3),
    ("Playground", 14), ("Porch", 6), ("Garden", 23), ("Printing_area", 0),
    ("Meeting_room", 13), ("Playground", 1),
]


def render(scene, i):
    ctx = torch.load(os.path.join(CTX, scene, "frame_context.pt"),
                     weights_only=False)
    orig = ctx["original"][i].float().numpy() / 255.0
    ref = ctx["reference_render"][i].float().numpy() / 255.0
    cue = ctx["cue"][i].float().numpy()
    gt = ctx["gt"][i].numpy()
    name = ctx["image_names"][i]

    bad = np.abs(orig - ref).mean(0) > 0.12
    cb = cue > 0.5
    fp = cb & ~gt
    tp = int((cb & gt).sum())
    prec = tp / max(int(cb.sum()), 1)
    ratio = cue[bad].mean() / (cue[~bad].mean() + 1e-9)
    fp_in_bad = 100 * (fp & bad).sum() / max(fp.sum(), 1)

    # panels flush together: no per-axis titles, a header row of labels instead
    fig, ax = plt.subplots(1, 4, figsize=(15.0, 2.75),
                           gridspec_kw=dict(wspace=0.015, left=0.004,
                                            right=0.996, bottom=0.004,
                                            top=0.80))
    ax[0].imshow(orig.transpose(1, 2, 0))
    ax[1].imshow(ref.transpose(1, 2, 0))
    ax[2].imshow(cue, cmap="inferno", vmin=0, vmax=1.5)
    ax[2].contour(cue, levels=[0.5], colors="#3ad6ff", linewidths=0.9)
    ax[3].imshow(gt, cmap="gray")
    labels = ["original RGB", "3DGS reference render", "cue C_v (τ=0.5)",
              "GT change"]
    for a, lab in zip(ax, labels):
        a.set_xticks([]); a.set_yticks([])
        a.text(0.5, 0.985, lab, transform=a.transAxes, ha="center", va="top",
               fontsize=9, color="white",
               bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none",
                         alpha=0.55))
    fig.suptitle(
        f"{scene} / frame {i:02d} ({name})",
        fontsize=13, fontweight="bold", y=0.985)
    fig.text(0.5, 0.86,
             f"복원 실패 {100*bad.mean():.0f}% · cue 점등 {100*cb.mean():.0f}% "
             f"(GT {100*gt.mean():.1f}%) · precision {prec:.3f} · "
             f"복원실패 영역 cue가 {ratio:.1f}배 · FP의 {fp_in_bad:.0f}%가 복원실패 영역",
             ha="center", va="top", fontsize=10, color="#5f5e58")
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, f"{scene}_f{i:02d}.png")
    fig.savefig(out, dpi=115)
    plt.close(fig)
    return out


def main():
    for sc, fr in FRAMES:
        print("saved", os.path.relpath(render(sc, fr), REPO))


if __name__ == "__main__":
    main()
