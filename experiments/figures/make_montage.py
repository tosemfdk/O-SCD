# Image-level montage: oracle-5 vs uniform-5 for a scene, 4 rows x 5 cols —
# each frame row is followed by a row with that frame's GT change mask and
# its changed-pixel share, so "which frames actually see the change, and
# how much" is directly visible.
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

BLUE, YELLOW, INK, INK2 = "#2a78d6", "#eda100", "#1a1a19", "#5f5e58"
GRAY, RED = "#c9c8c1", "#e34948"
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.dirname(os.path.abspath(__file__))

# rev3 map (2026-07-17, ~590 evals/scene) vs each scene's BEST stride-5
# uniform offset — the honest per-scene comparison, no cherry-picked offset.
CASES = [
    ("Cantina", (12, 15, 17, 20, 21), 0.5955, (3, 8, 13, 18, 23), 0.4751),
    ("Garden", (1, 4, 14, 16, 19), 0.5515, (0, 5, 10, 15, 20), 0.5112),
    ("Lounge", (0, 1, 3, 11, 20), 0.6037, (0, 5, 10, 15, 20), 0.5722),
    ("Lunch_room", (1, 5, 19, 21, 24), 0.4251, (1, 6, 11, 16, 21), 0.3824),
    ("Meeting_room", (10, 12, 19, 23, 24), 0.5678, (2, 7, 12, 17, 22), 0.5316),
    ("Playground", (0, 4, 19, 20, 21), 0.5199, (1, 6, 11, 16, 21), 0.4258),
    ("Porch", (1, 5, 14, 20, 23), 0.6385, (4, 9, 14, 19, 24), 0.6243),
    ("Pots", (1, 7, 9, 11, 21), 0.6582, (1, 6, 11, 16, 21), 0.6429),
    ("Printing_area", (2, 8, 11, 12, 14), 0.7115, (3, 8, 13, 18, 23), 0.6286),
    ("Zen", (2, 11, 15, 18, 22), 0.5850, (0, 5, 10, 15, 20), 0.5475),
]

for scene, oc, om, uc, um in CASES:
    img_dir = os.path.join(REPO, "data/PASLCD/Instance_1", scene, "inference_scene/images")
    files = sorted(os.listdir(img_dir))
    shared = set(oc) & set(uc)

    fig, axes = plt.subplots(4, 5, figsize=(15, 11.2))
    fig.patch.set_facecolor("white")
    for set_i, (combo, miou, color, name) in enumerate([
            (oc, om, BLUE, "oracle-5 (found by search)"),
            (uc, um, YELLOW, "uniform-5 (best stride-5 offset)")]):
        img_row, mask_row = 2 * set_i, 2 * set_i + 1
        for col, idx in enumerate(combo):
            im = Image.open(os.path.join(img_dir, files[idx])).convert("RGB")
            im.thumbnail((760, 428))
            mask_path = os.path.join(REPO, "data/PASLCD/Instance_1", scene,
                                     "gt_mask", os.path.splitext(files[idx])[0] + ".png")
            m = np.array(Image.open(mask_path).convert("L").resize(im.size)) > 127

            ax = axes[img_row, col]
            ax.imshow(im)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(color); sp.set_linewidth(3.5)
            tag = f"frame {idx}" + ("  (in both sets)" if idx in shared else "")
            ax.set_title(tag, fontsize=10.5,
                         color=INK if idx in shared else INK2,
                         fontweight="bold" if idx in shared else "normal")

            ax = axes[mask_row, col]
            ax.imshow(m, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(GRAY); sp.set_linewidth(1.2)
            ax.text(0.04, 0.88, f"{100*m.mean():.1f}% changed",
                    transform=ax.transAxes, fontsize=10.5, color=RED,
                    fontweight="bold")
        axes[img_row, 0].set_ylabel(f"{'oracle-5' if set_i == 0 else 'uniform-5'}"
                                    f"\nmIoU {miou:.3f}", fontsize=12,
                                    color=color, fontweight="bold", labelpad=12)
        axes[mask_row, 0].set_ylabel("GT change mask", fontsize=10.5,
                                     color=INK2, labelpad=12)
    fig.suptitle(f"{scene}: what the winning 5 frames actually look like "
                 f"(+{(om/um-1)*100:.0f}% over this uniform set)",
                 fontsize=15, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out = os.path.join(OUT, f"montage_{scene.lower()}.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("saved", out)
