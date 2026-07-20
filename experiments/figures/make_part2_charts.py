# Part-2 report charts (dataviz-skill compliant: dot plot for small mIoU
# deltas instead of truncated bars, diverging blue/red only for polarity,
# fixed entity colors carried over from Part 1: uniform=yellow, all-25=aqua,
# oracle=blue; the new selector entity wears green slot 2).
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, GREEN, YELLOW, AQUA = "#2a78d6", "#008300", "#eda100", "#1baf7a"
RED = "#e34948"
INK, INK2, GRID, MUTED = "#1a1a19", "#5f5e58", "#e6e5e0", "#b5b4ac"
OUT = os.path.dirname(os.path.abspath(__file__))

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


# ---- A. the selector ladder (dot plot; bars would need a truncated axis) ----
variants = [  # bottom-up = build order
    ("1) info mass only\n(candidate_only)", 0.4574, "picks near-duplicate frames"),
    ("2) + revisit suspected\n(binary change mask)", 0.4673, "helps half the scenes"),
    ("3) D-opt discount\n(scalar c-channel)", 0.4778, "avoids re-observation"),
    ("4) revisit + NEW angles\n(3x3 direction blocks)", 0.4883, "first to clear uniform"),
]
fig, ax = plt.subplots(figsize=(9, 4.8))
y = np.arange(len(variants))
for ref, color, label in [(0.4817, YELLOW, "uniform mean 0.482"),
                          (0.5237, AQUA, "all-25  0.524"),
                          (0.5875, BLUE, "oracle-5  0.588")]:
    ax.axvline(ref, color=color, lw=1.6, ls="--", zorder=1)
    ax.annotate(label, (ref, len(variants) - 0.45), xytext=(4, 4),
                textcoords="offset points", fontsize=9.5, color=color,
                fontweight="bold", rotation=0)
for i, (name, v, note) in enumerate(variants):
    c = GREEN if i == len(variants) - 1 else MUTED
    ax.plot([0.43, v], [i, i], color=GRID, lw=1.4, zorder=1)
    ax.scatter([v], [i], s=110, color=c, zorder=3)
    ax.annotate(f"{v:.3f}", (v, i), xytext=(8, -4), textcoords="offset points",
                fontsize=10, fontweight="bold", color=INK)
    ax.annotate(note, (0.432, i), xytext=(0, -13), textcoords="offset points",
                fontsize=8.8, color=INK2, style="italic")
ax.set_yticks(y, [v[0] for v in variants], fontsize=9.5)
ax.set_xlim(0.43, 0.62)
ax.set_ylim(-0.7, len(variants) - 0.2)
ax.set_xlabel("10-scene mean mIoU (clean replay, all 25 query views)")
ax.set_title("Four selector variants: each failure isolated one missing ingredient")
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
save(fig, "part2_chart_a_ladder.png")

# ---- B. why the discount fails: sign inversion once a context exists --------
data = {  # scene -> {method: (rho@S0, rho@S1)}
    "Zen": {"D-opt (discounts overlap)": (0.19, -0.40),
            "info mass (ignores overlap)": (0.19, 0.46)},
    "Porch": {"D-opt (discounts overlap)": (-0.02, -0.68),
              "info mass (ignores overlap)": (0.22, 0.35)},
}
colors = {"D-opt (discounts overlap)": BLUE, "info mass (ignores overlap)": YELLOW}
fig, axes = plt.subplots(1, 2, figsize=(9, 4.6), sharey=True)
for ax, (scene, methods) in zip(axes, data.items()):
    ax.axhline(0, color=MUTED, lw=1.2)
    for name, (a, b) in methods.items():
        ax.plot([0, 1], [a, b], color=colors[name], lw=2, marker="o", ms=8,
                label=name)
        ax.annotate(f"{b:+.2f}", (1, b), xytext=(8, -4),
                    textcoords="offset points", fontsize=10, fontweight="bold",
                    color=colors[name])
    ax.set_xticks([0, 1], ["no context\n(S = {})", "1 frame chosen\n(S = {12})"])
    ax.set_title(scene, fontsize=12)
    ax.set_xlim(-0.25, 1.45)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
axes[0].set_ylabel("rank corr. with TRUE marginal gain (Spearman ρ)")
axes[0].legend(loc="lower left", frameon=False, fontsize=9)
fig.suptitle("One chosen frame flips the discount negative — SCD favors re-observation",
             fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(os.path.join(OUT, "part2_chart_b_inversion.png"), dpi=160)
plt.close(fig)
print("saved part2_chart_b_inversion.png")

# ---- C. dopt_dir per-scene oracle-gap recovery (polarity -> diverging) ------
rec = [("Zen", 57.6), ("Lounge", 42.7), ("Lunch_room", 28.8), ("Garden", 26.0),
       ("Playground", 23.9), ("Pots", 14.6), ("Printing_area", 11.2),
       ("Cantina", -14.0), ("Porch", -82.1), ("Meeting_room", -83.6)]
fig, ax = plt.subplots(figsize=(9, 5.2))
y = np.arange(len(rec))[::-1]
vals = [v for _, v in rec]
ax.barh(y, vals, height=0.58,
        color=[BLUE if v >= 0 else RED for v in vals])
ax.axvline(0, color=INK2, lw=1.2)
ax.axvline(100, color=GRID, lw=1.2, ls="--")
ax.annotate("oracle level (100%)", (100, len(rec) - 0.2), xytext=(-4, 4),
            textcoords="offset points", ha="right", fontsize=9.5, color=INK2)
for yy, (s, v) in zip(y, rec):
    ax.annotate(f"{v:+.0f}%", (v, yy), xytext=(6 if v >= 0 else -6, -4),
                textcoords="offset points", fontsize=10,
                ha="left" if v >= 0 else "right", color=INK)
for yy, (s, v) in zip(y, rec):
    if v < 0:
        ax.annotate("0/5 overlap with oracle set — bad seed", (2, yy),
                    xytext=(0, -4), textcoords="offset points", fontsize=8.8,
                    color=RED, style="italic")
ax.set_yticks(y, [s for s, _ in rec])
ax.set_xlabel("share of the selection headroom recovered  "
              "(selector − uniform) / (oracle − uniform)")
ax.set_title("7 scenes recover 11–58% of the gap; a poisoned seed loses 3")
ax.set_xlim(-100, 112)
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
save(fig, "part2_chart_c_recovery.png")
print("ALL PART2 CHARTS DONE")
