# Feasibility-report charts for the Notion page (dataviz-skill compliant:
# reference palette in fixed slot order, thin marks, one axis, direct labels
# selective, legend for >=2 series, English in-chart text).
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, AQUA, YELLOW, GREEN = "#2a78d6", "#1baf7a", "#eda100", "#008300"
RED = "#e34948"          # status: serious (toxic highlight only)
INK, INK2, GRID = "#1a1a19", "#5f5e58", "#e6e5e0"
MUTED = "#b5b4ac"
OUT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=160)
    plt.close(fig)
    print("saved", name)


# ---- A. oracle-5 vs all-25 vs uniform mean (dumbbell) ------------------------
data = [  # scene, best5, all25 mean, uniform stride-5 mean
    ("Playground", 0.4499, 0.3590, 0.3818), ("Garden", 0.5498, 0.4520, 0.4805),
    ("Lunch_room", 0.4177, 0.3714, 0.3234), ("Meeting_room", 0.5661, 0.5151, 0.5036),
    ("Lounge", 0.5868, 0.5345, 0.5388), ("Pots", 0.6520, 0.6116, 0.6134),
    ("Zen", 0.5694, 0.5368, 0.4198), ("Cantina", 0.5900, 0.5655, 0.4374),
    ("Porch", 0.6292, 0.6040, 0.5343), ("Printing_area", 0.7031, 0.6867, 0.5837),
]
data.sort(key=lambda r: r[1] / r[2])  # sort by oracle gain, biggest at top
fig, ax = plt.subplots(figsize=(9, 5.6))
y = np.arange(len(data))
for i, (s, b5, a25, u5) in enumerate(data):
    ax.plot([a25, b5], [i, i], color=GRID, lw=2, zorder=1)
ax.scatter([r[3] for r in data], y, s=64, color=YELLOW, zorder=3, label="uniform-5 (mean of 5 offsets)")
ax.scatter([r[2] for r in data], y, s=64, color=AQUA, zorder=3, label="all-25 (mean of 5 runs)")
ax.scatter([r[1] for r in data], y, s=80, color=BLUE, zorder=4, label="best-5 found (oracle, lower bound)")
for i, (s, b5, a25, u5) in enumerate(data):
    ax.annotate(f"+{(b5/a25-1)*100:.1f}%", (b5, i), xytext=(8, -4),
                textcoords="offset points", fontsize=9.5, color=BLUE, fontweight="bold")
ax.set_yticks(y, [r[0] for r in data])
ax.set_xlabel("query-view mIoU (evaluated on all 25 GT masks)")
ax.set_title("A well-chosen 5-frame set beats all 25 frames — on 10/10 scenes")
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.set_xlim(right=0.76)
ax.legend(loc="lower left", frameon=False, fontsize=9.5)
save(fig, "chart_a_oracle_map.png")

# ---- B. uniform saturation curve + oracle point ------------------------------
K = [2, 3, 5, 8, 10, 15, 25]
pct = [77.8, 87.6, 94.3, 95.8, 97.8, 98.8, 100.0]
fig, ax = plt.subplots(figsize=(9, 5))
ax.axhline(100, color=GRID, lw=1)
ax.axhline(95, color=MUTED, lw=1, ls="--")
ax.annotate("95% of all-25", (24.8, 95), xytext=(0, -13), textcoords="offset points",
            ha="right", fontsize=9.5, color=INK2)
ax.plot(K, pct, color=BLUE, lw=2, marker="o", ms=8, label="uniform selection")
for k, p in zip(K, pct):
    ax.annotate(f"{p:.0f}%", (k, p), xytext=(0, 9), textcoords="offset points",
                ha="center", fontsize=9, color=INK2)
ax.scatter([5], [109.4], s=110, color=AQUA, zorder=5, label="oracle-5 (well-chosen 5)")
ax.annotate("oracle-5: 109% with K=5", (5, 109.4), xytext=(12, 2),
            textcoords="offset points", fontsize=10, color="#0f7a54", fontweight="bold")
ax.set_xlabel("update-frame budget K (out of 25)")
ax.set_ylabel("% of all-25 mIoU (10-scene mean)")
ax.set_title("Blind selection saturates below 100% — good selection goes above it")
ax.set_xticks(K)
ax.set_ylim(70, 115)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.legend(loc="lower right", frameon=False, fontsize=9.5)
save(fig, "chart_b_saturation.png")

# ---- C. target-level information trajectory ----------------------------------
ks = [0, 1, 2, 3, 4, 5]
series = [  # fixed slot order: blue, aqua, yellow, green
    ("exact D-opt", BLUE, [0, 20.50, 25.71, 27.71, 28.74, 29.41]),
    ("geometry proxy", AQUA, [0, 16.35, 19.37, 23.61, 25.38, 26.88]),
    ("uniform", YELLOW, [0, 7.86, 13.66, 22.70, 24.26, 24.88]),
    ("random", GREEN, [0, 6.79, 10.23, 14.67, 16.77, 18.28]),
]
fig, ax = plt.subplots(figsize=(9, 5))
ax.axhline(24.88, color=MUTED, lw=1, ls="--")
for name, c, v in series:
    ax.plot(ks, v, color=c, lw=2, marker="o", ms=7, label=name)
    ax.annotate(name, (ks[-1], v[-1]), xytext=(8, -3), textcoords="offset points",
                fontsize=10, color=c, fontweight="bold")
ax.annotate("uniform's 5-view level,\nreached with 2 views", (2, 25.71), xytext=(1.05, 29.2),
            textcoords="data", fontsize=10, color=BLUE, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=BLUE, lw=1))
ax.set_xlabel("views selected (after shared seed view)")
ax.set_ylabel("target information gain  Δ logdet H  (nats, mean of 10 targets)")
ax.set_title("Target-level D-optimal selection: 2 views buy what uniform buys with 5")
ax.set_xlim(-0.2, 6.4)
ax.set_xticks(ks)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.set_ylim(-1, 33)
ax.legend(loc="lower right", frameon=False, fontsize=9.5)
save(fig, "chart_c_target_info.png")

# ---- D. toxic-frame scatter ---------------------------------------------------
rows = list(csv.DictReader(open(os.path.join(REPO, "experiments/frame_features.csv"))))
by = defaultdict(list)
for r in rows:
    by[r["scene"]].append(r)
xs, ys, margs, names = [], [], [], []
for s, sr in by.items():
    area = np.array([float(r["cue_area"]) for r in sr])
    cons = np.array([float(r["cue_consistency"]) for r in sr])
    for i, r in enumerate(sr):
        xs.append(100 * np.mean(area < area[i]))
        ys.append(100 * np.mean(cons < cons[i]))
        margs.append(float(r["marginal_miou"]))
        names.append(f"{s}/{r['frame']}")
xs, ys, margs = np.array(xs), np.array(ys), np.array(margs)
toxic = margs < -0.05
fig, ax = plt.subplots(figsize=(9, 5.4))
ax.scatter(xs[~toxic], ys[~toxic], s=26, color=MUTED, alpha=0.65, label="other frames")
ax.scatter(xs[toxic], ys[toxic], s=64, color=RED, zorder=4,
           label="toxic frames (marginal mIoU < −0.05)")
for label in ("Zen/3", "Zen/4", "Cantina/11", "Playground/19"):
    i = names.index(label)
    ax.annotate(label, (xs[i], ys[i]), xytext=(7, 4), textcoords="offset points",
                fontsize=9.5, color=RED, fontweight="bold")
ax.add_patch(plt.Rectangle((60, 0), 40, 40, fill=False, ls="--", ec=RED, lw=1.2))
ax.annotate("big cue, low 3D consensus\n= identified toxic type", (98, 42),
            ha="right", fontsize=9.5, color=RED)
ax.set_xlabel("cue size (within-scene percentile)")
ax.set_ylabel("cue 3D-consensus agreement (percentile)")
ax.grid(color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False, fontsize=9.5)
ax.set_title("A GT-free signature catches the worst toxic type", pad=28)
save(fig, "chart_d_toxic.png")

# ---- E. Porch case: diagnose -> fix -------------------------------------------
methods = ["nbv (direction-blind)", "uniform @5", "nbv_dopt (direction-aware)"]
vals = [0.3186, 0.4507, 0.5302]
cols = [MUTED, MUTED, BLUE]
fig, ax = plt.subplots(figsize=(9, 4.2))
bars = ax.barh(methods, vals, height=0.55, color=cols)
ax.axvline(0.6040, color=AQUA, lw=2, ls="--")
ax.text(0.596, 2.52, "all-25 (0.604)", fontsize=10, color="#0f7a54",
        fontweight="bold", ha="right", va="center")
for b, v in zip(bars, vals):
    ax.annotate(f"{v:.3f}", (v, b.get_y() + b.get_height() / 2), xytext=(6, -4),
                textcoords="offset points", fontsize=10, color=INK)
ax.set_xlabel("Porch query-view mIoU (K=5)")
ax.set_title("Diagnose \u2192 fix: direction-awareness flipped Porch (worst scene)")
ax.set_xlim(0, 0.72)
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.set_ylim(2.75, -0.6)
save(fig, "chart_e_porch.png")
print("ALL CHARTS DONE")
