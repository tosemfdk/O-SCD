# Image-level montage: oracle-5 vs uniform-5 for a scene (2 rows x 5 cols),
# with the frame's GT change region overlaid in red and its pixel share in
# the tag — makes "which frames actually see the change" directly visible.
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

BLUE, YELLOW, INK, INK2 = "#2a78d6", "#eda100", "#1a1a19", "#5f5e58"
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

    fig, axes = plt.subplots(2, 5, figsize=(15, 6.1))
    fig.patch.set_facecolor("white")
    for row, (combo, miou, color, name) in enumerate([
            (oc, om, BLUE, "oracle-5 (found by search)"),
            (uc, um, YELLOW, "uniform-5 (best stride-5 offset)")]):
        for col, idx in enumerate(combo):
            ax = axes[row, col]
            im = Image.open(os.path.join(img_dir, files[idx])).convert("RGB")
            im.thumbnail((760, 428))
            mask_path = os.path.join(REPO, "data/PASLCD/Instance_1", scene,
                                     "gt_mask", os.path.splitext(files[idx])[0] + ".png")
            m = np.array(Image.open(mask_path).convert("L").resize(im.size)) > 127
            a = np.asarray(im, dtype=np.float32)
            a[m] = 0.4 * a[m] + 0.6 * np.array([228, 32, 60])
            ax.imshow(a.astype(np.uint8))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(color); sp.set_linewidth(3.5)
            tag = f"frame {idx} · {100*m.mean():.0f}% chg" \
                + ("  (both)" if idx in shared else "")
            ax.set_title(tag, fontsize=10.5,
                         color=INK if idx in shared else INK2,
                         fontweight="bold" if idx in shared else "normal")
        short = "oracle-5" if row == 0 else "uniform-5"
        axes[row, 0].set_ylabel(f"{short}\nmIoU {miou:.3f}", fontsize=12,
                                color=color, fontweight="bold", labelpad=12)
    fig.suptitle(f"{scene}: what the winning 5 frames actually look like "
                 f"(+{(om/um-1)*100:.0f}% over this uniform set) — "
                 f"GT change regions in red",
                 fontsize=15, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(OUT, f"montage_{scene.lower()}.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("saved", out)
