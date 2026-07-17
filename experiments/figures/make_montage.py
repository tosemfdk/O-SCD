# Image-level montage: oracle-5 vs uniform-5 for a scene (2 rows x 5 cols).
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

BLUE, YELLOW, INK, INK2 = "#2a78d6", "#eda100", "#1a1a19", "#5f5e58"
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.dirname(os.path.abspath(__file__))

CASES = [
    ("Cantina", (12, 15, 17, 20, 24), 0.5900, (0, 5, 10, 15, 20), 0.4589),
    ("Garden", (1, 5, 7, 10, 12), 0.5498, (0, 5, 10, 15, 20), 0.5112),
]

for scene, oc, om, uc, um in CASES:
    img_dir = os.path.join(REPO, "data/PASLCD/Instance_1", scene, "inference_scene/images")
    files = sorted(os.listdir(img_dir))
    shared = set(oc) & set(uc)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6.1))
    fig.patch.set_facecolor("white")
    for row, (combo, miou, color, name) in enumerate([
            (oc, om, BLUE, "oracle-5 (found by search)"),
            (uc, um, YELLOW, "uniform-5 (stride 5, offset 0)")]):
        for col, idx in enumerate(combo):
            ax = axes[row, col]
            im = Image.open(os.path.join(img_dir, files[idx]))
            im.thumbnail((760, 428))
            ax.imshow(im)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(color); sp.set_linewidth(3.5)
            tag = f"frame {idx}" + ("  (in both sets)" if idx in shared else "")
            ax.set_title(tag, fontsize=10.5,
                         color=INK if idx in shared else INK2,
                         fontweight="bold" if idx in shared else "normal")
        short = "oracle-5" if row == 0 else "uniform-5"
        axes[row, 0].set_ylabel(f"{short}\nmIoU {miou:.3f}", fontsize=12,
                                color=color, fontweight="bold", labelpad=12)
    fig.suptitle(f"{scene}: what the winning 5 frames actually look like "
                 f"(+{(om/um-1)*100:.0f}% over this uniform set)",
                 fontsize=15, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(OUT, f"montage_{scene.lower()}.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("saved", out)
