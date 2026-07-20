# Tau sweep of the raw fusion cue vs GT, over all 250 Instance_1 inference
# frames (user request 2026-07-20, follow-up to cue_gt_report.py which fixed
# tau=0.5). Re-thresholds the cached cue dumps (output_subset/cue_gt/<scene>/
# cues.pt) at many taus -- no GPU work if the dumps exist.
#
# Question being answered: the cue's false-alarm rate looked like a function of
# how much GT change the scene actually contains, not of the cue itself. The
# sweep tests whether that correlation survives at every operating point.
#
#   python experiments/cue_tau_sweep.py            # sweep + CSV + charts
from __future__ import annotations

import argparse
import csv
import os
import sys

import cv2
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments"))
from oracle_search import INSTANCE, SCENES  # noqa: E402

DUMP_ROOT = os.path.join(REPO, "output_subset", "cue_gt")
FIG = os.path.join(REPO, "experiments", "figures")
CSV_PATH = os.path.join(REPO, "experiments", "cue_tau_sweep_instance1.csv")

# Palette carried over from make_part2_charts.py (fixed entity colors).
BLUE, GREEN, YELLOW, AQUA = "#2a78d6", "#008300", "#eda100", "#1baf7a"
RED = "#e34948"
INK, INK2, GRID, MUTED = "#1a1a19", "#5f5e58", "#e6e5e0", "#b5b4ac"
# Sequential single-hue ramp for tau (magnitude -> one hue, light to dark).
TAU_RAMP = ["#a8c8ee", "#6ba3e0", "#2a78d6", "#1a4e8f"]

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def gt_for(scene, name, hw):
    p = os.path.join(REPO, "data", "PASLCD", INSTANCE, scene, "gt_mask",
                     name + ".png")
    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    return cv2.resize(m, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST) > 127


def sweep(taus):
    """Per (scene, frame, tau) confusion counts, from the cached cue dumps."""
    rows = []
    for scene in SCENES:
        blob = torch.load(os.path.join(DUMP_ROOT, scene, "cues.pt"),
                          map_location="cpu", weights_only=False)
        cues, names = blob["cues"], blob["image_names"]
        if cues.ndim == 4:
            cues = cues[:, 0]
        for i, name in enumerate(names):
            cue = cues[i].numpy()
            gt = gt_for(scene, name, cue.shape)
            if gt is None:
                continue
            npix, gt_pos = cue.size, int(gt.sum())
            for t in taus:
                cb = cue > t
                tp = int((cb & gt).sum())
                pred = int(cb.sum())
                rows.append({
                    "scene": scene, "frame": i, "image": name, "tau": t,
                    "npix": npix, "gt_pos": gt_pos, "pred_pos": pred, "tp": tp,
                })
        print(f"  swept {scene}")
    return rows


def agg(rows):
    """Area-weighted (pixel-pooled) rates -- the honest way to pool frames of
    very different GT area; per-frame means over-weight near-empty frames."""
    gt = sum(r["gt_pos"] for r in rows)
    pred = sum(r["pred_pos"] for r in rows)
    tp = sum(r["tp"] for r in rows)
    npix = sum(r["npix"] for r in rows)
    return {
        "recall": tp / gt if gt else np.nan,
        "precision": tp / pred if pred else np.nan,
        "miss": 1 - tp / gt if gt else np.nan,
        "fp_rate": 1 - tp / pred if pred else np.nan,
        "iou": tp / (pred + gt - tp) if (pred + gt - tp) else np.nan,
        "gt_area": 100 * gt / npix, "cue_area": 100 * pred / npix,
    }


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.corrcoef(x, y)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taus", type=str,
                    default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.2,1.4")
    args = ap.parse_args()
    taus = [float(t) for t in args.taus.split(",")]

    rows = sweep(taus)
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", os.path.relpath(CSV_PATH, REPO))

    by_tau = {t: agg([r for r in rows if r["tau"] == t]) for t in taus}
    by_st = {(s, t): agg([r for r in rows if r["scene"] == s and r["tau"] == t])
             for s in SCENES for t in taus}

    print(f"\n=== tau sweep, Instance_1, n=250 frames (area-weighted) ===")
    print(f"{'tau':>5} {'cueA%':>7} {'recall':>7} {'prec':>7} {'miss%':>7} "
          f"{'FP%':>7} {'IoU':>6}  r(gtArea,FP)")
    corr = {}
    for t in taus:
        a = by_tau[t]
        g = [by_st[(s, t)]["gt_area"] for s in SCENES]
        fp = [by_st[(s, t)]["fp_rate"] for s in SCENES]
        corr[t] = pearson(np.log10(g), fp)
        print(f"{t:>5.1f} {a['cue_area']:>7.2f} {a['recall']:>7.3f} "
              f"{a['precision']:>7.3f} {a['miss']*100:>7.2f} "
              f"{a['fp_rate']*100:>7.2f} {a['iou']:>6.3f}  {corr[t]:>6.2f}")

    # ---- A. operating curves: what tau buys and what it costs ---------------
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for s in SCENES:  # per-scene precision, recessive
        ax.plot(taus, [by_st[(s, t)]["precision"] for t in taus],
                color=MUTED, lw=1.0, zorder=1)
    ax.plot(taus, [by_tau[t]["recall"] for t in taus], color=BLUE, lw=2.2,
            marker="o", ms=5, label="recall  (1 − miss)", zorder=3)
    ax.plot(taus, [by_tau[t]["precision"] for t in taus], color=RED, lw=2.2,
            marker="o", ms=5, label="precision  (1 − FP rate)", zorder=3)
    ax.plot(taus, [by_tau[t]["iou"] for t in taus], color=GREEN, lw=2.2,
            marker="o", ms=5, label="IoU", zorder=3)
    ax.axvline(0.5, color=INK2, lw=1, ls=":", zorder=0)
    ax.text(0.51, 0.02, "τ=0.5 (report default)", color=INK2, fontsize=9)
    for t, lbl, c, dy in [(taus[-1], "recall", BLUE, -9), (taus[-1], "precision", RED, 0),
                          (taus[-1], "IoU", GREEN, 9)]:
        y = by_tau[t][{"recall": "recall", "precision": "precision", "IoU": "iou"}[lbl]]
        ax.annotate(f"{y:.2f}", (t, y), textcoords="offset points",
                    xytext=(8, dy), color=c, fontsize=10, va="center")
    ax.set_xlabel("τ  (cue binarization threshold)")
    ax.set_ylabel("rate, area-weighted over 250 frames")
    ax.set_ylim(0, 1.02)
    best = max(taus, key=lambda t: by_tau[t]["iou"])
    ax.axvline(best, color=GREEN, lw=1, ls=":", zorder=0)
    ax.text(best + 0.01, 0.95, f"IoU peak τ={best}", color=GREEN, fontsize=9)
    ax.set_title("A. τ=0.5 sits far off the cue's own optimum\n"
                 f"IoU 0.23 at τ=0.5 vs {by_tau[best]['iou']:.2f} at τ={best}"
                 "   ·   gray = per-scene precision", loc="left")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(0.02, 0.55))
    save(fig, "cue_tau_a_operating.png")

    # ---- B. THE correlation chart: scene GT area vs false-alarm rate --------
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    show = [t for t in (0.2, 0.5, 0.9) if t in taus] or taus[:3]
    for k, t in enumerate(show):
        g = np.array([by_st[(s, t)]["gt_area"] for s in SCENES])
        fp = np.array([by_st[(s, t)]["fp_rate"] for s in SCENES])
        c = TAU_RAMP[k]
        ax.scatter(g, fp, s=70, color=c, edgecolor="white", linewidth=1.5,
                   zorder=3 + k, label=f"τ = {t}   r = {corr[t]:.2f}")
        b, a0 = np.polyfit(np.log10(g), fp, 1)  # trend in log(GT area)
        xs = np.linspace(np.log10(g.min()), np.log10(g.max()), 50)
        ax.plot(10 ** xs, a0 + b * xs, color=c, lw=1.6, ls="--", zorder=2)
    # scene labels once, on the middle series
    t = show[len(show) // 2]
    for s in SCENES:
        a = by_st[(s, t)]
        ax.annotate(s.replace("_", " "), (a["gt_area"], a["fp_rate"]),
                    textcoords="offset points", xytext=(0, 11), fontsize=8.5,
                    color=INK2, ha="center", zorder=6)
    ax.set_xscale("log")
    ax.set_xlabel("scene GT change area  (% of frame, log scale)")
    ax.set_ylabel("false-alarm rate  FP / (TP+FP)")
    ax.set_ylim(0, 1.02)
    ax.set_title("B. False alarms are a property of the SCENE, not of τ\n"
                 "less real change to find → nearly everything the cue lights is wrong",
                 loc="left")
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              title="threshold", alignment="left")
    save(fig, "cue_tau_b_gtarea_fp_corr.png")

    # ---- C. why: the lit area barely responds to tau -----------------------
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for s in SCENES:
        ax.plot(taus, [by_st[(s, t)]["cue_area"] for t in taus],
                color=MUTED, lw=1.0, zorder=1)
    ax.plot(taus, [by_tau[t]["cue_area"] for t in taus], color=BLUE, lw=2.4,
            marker="o", ms=5, label="cue area lit (all scenes)", zorder=3)
    ax.plot(taus, [by_tau[t]["gt_area"] for t in taus], color=INK2, lw=2.0,
            ls="--", label="real change area (GT)", zorder=3)
    ax.set_xlabel("τ  (cue binarization threshold)")
    ax.set_ylabel("% of frame")
    ax.set_title("C. The gap that has to be closed downstream\n"
                 "gray = per-scene lit area", loc="left")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    save(fig, "cue_tau_c_area_gap.png")


if __name__ == "__main__":
    main()
