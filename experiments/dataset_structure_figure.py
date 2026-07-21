# A single figure describing the current O-SCD (PASLCD) dataset structure:
# a schematic of the per-scene folder layout + task flow, over a concrete
# Garden sample strip (reference views -> 3DGS -> inference query -> GT mask).
#
#   python experiments/dataset_structure_figure.py
from __future__ import annotations

import glob
import os

import numpy as np
import torch
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data", "PASLCD")
CTX = os.path.join(REPO, "outputs", "change_nbv", "diagnostics",
                   "rchange_importance", "Instance_1", "Garden",
                   "frame_context.pt")
OUT = os.path.join(REPO, "experiments", "figures", "oscd_dataset_structure.png")

for _p in ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf",):
    if os.path.exists(_p):
        fm.fontManager.addfont(_p)
        plt.rcParams["font.family"] = fm.FontProperties(fname=_p).get_name()
plt.rcParams["axes.unicode_minus"] = False

INK, INK2, MUTED = "#1a1a19", "#5f5e58", "#b5b4ac"
BLUE, GREEN, YELLOW, RED, AQUA = ("#2a78d6", "#008300", "#eda100", "#e34948",
                                  "#1baf7a")


def box(ax, x, y, w, h, title, lines, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.008,rounding_size=0.02",
                                fc=fc, ec=ec, lw=1.8, zorder=2))
    ax.text(x + w / 2, y + h - 0.03, title, ha="center", va="top",
            fontsize=11, fontweight="bold", color=INK, zorder=3)
    ax.text(x + w / 2, y + h - 0.085, "\n".join(lines), ha="center", va="top",
            fontsize=8.4, color=INK2, zorder=3, linespacing=1.35)


def arrow(ax, x0, y0, x1, y1, color=INK2, label=None, lx=0, ly=0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=16, lw=2.0, color=color,
                                 zorder=1))
    if label:
        ax.text((x0 + x1) / 2 + lx, (y0 + y1) / 2 + ly, label, ha="center",
                va="center", fontsize=8.5, color=color, fontweight="bold",
                zorder=4)


def thumb(ax, img, title, sub, ec):
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(ec); s.set_linewidth(2.4)
    ax.set_title(title, fontsize=10, fontweight="bold", color=INK, pad=3)
    ax.text(0.5, -0.10, sub, transform=ax.transAxes, ha="center", va="top",
            fontsize=8.4, color=INK2)


def load_samples():
    ctx = torch.load(CTX, weights_only=False)
    ref_render = ctx["reference_render"][0].float().numpy().transpose(1, 2, 0) / 255
    inf = ctx["original"][15].float().numpy().transpose(1, 2, 0) / 255
    gt = ctx["gt"][15].numpy().astype(float)
    refimg_path = sorted(glob.glob(os.path.join(
        DATA, "Instance_1", "Garden", "reference_scene", "images", "*")))[0]
    ref = np.asarray(Image.open(refimg_path).convert("RGB").resize((400, 225))) / 255
    return ref, ref_render, inf, gt


def main():
    ref, ref_render, inf, gt = load_samples()
    fig = plt.figure(figsize=(14.5, 9.4))
    fig.suptitle("O-SCD (PASLCD) 데이터셋 구조", fontsize=18, fontweight="bold",
                 y=0.985)
    fig.text(0.5, 0.945,
             "2 instances × 10 scenes = 20 scene-instances  ·  "
             "장면당: reference 다중뷰 → 사전학습 3DGS(‘변화 전’), inference 25뷰(‘변화 후’), "
             "프레임별 GT 변화 마스크",
             ha="center", fontsize=10.5, color=INK2)

    # ---- top schematic -----------------------------------------------------
    ax = fig.add_axes([0.02, 0.42, 0.96, 0.48])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    box(ax, 0.005, 0.60, 0.15, 0.34, "PASLCD/",
        ["Instance_1/", "Instance_2/", "", "각 instance =", "동일 10개 장면"],
        "#eef4fc", BLUE)
    box(ax, 0.185, 0.60, 0.20, 0.34, "<scene>/  (×10)",
        ["Cantina · Garden", "Lounge · Lunch_room", "Meeting_room · Porch",
         "Playground · Pots", "Printing_area · Zen"], "#f4f4f0", MUTED)
    arrow(ax, 0.157, 0.77, 0.183, 0.77)

    # the 4 per-scene folders
    fx = 0.42
    box(ax, fx, 0.70, 0.265, 0.24, "reference_scene/",
        ["posed RGB 다중뷰  (50–109장)",
         "images/  +  sparse/0 (COLMAP:",
         "cameras·images·points3D.bin)"], "#eafaf2", GREEN)
    box(ax, fx, 0.38, 0.265, 0.26, "reference_reconstruction/",
        ["사전학습 3DGS  ‘변화 전’ 장면",
         "point_cloud/iteration_30000/",
         "point_cloud.ply  (~10만–33만 gaussians;",
         "Garden 207,832)  +  cameras.json"], "#eef4fc", BLUE)
    box(ax, fx, 0.10, 0.265, 0.245, "inference_scene/images/",
        ["query-view RGB  25장  ‘변화 후’",
         "고해상도 (예: 4028×2265)",
         "장면에 변화가 있을 수 있음"], "#fdf3e0", YELLOW)
    box(ax, fx + 0.30, 0.10, 0.245, 0.245, "gt_mask/",
        ["프레임별 GT 변화 마스크 25장",
         "binary PNG (0 / 255)",
         "inference 파일명과 1:1 정렬"], "#fdecec", RED)

    arrow(ax, 0.385, 0.77, fx - 0.002, 0.80)
    arrow(ax, 0.385, 0.74, fx - 0.002, 0.50)
    arrow(ax, 0.385, 0.71, fx - 0.002, 0.21)
    # reference_scene --COLMAP+3DGS train--> reconstruction
    arrow(ax, fx + 0.132, 0.688, fx + 0.132, 0.652, color=GREEN)
    ax.text(fx + 0.156, 0.670, "COLMAP → 3DGS 학습", ha="left", va="center",
            fontsize=8.5, color=GREEN, fontweight="bold", zorder=4)
    # inference + gt pairing
    arrow(ax, fx + 0.265, 0.215, fx + 0.298, 0.215, color=RED)

    ax.text(fx + 0.42, 0.045, "→ 두 마스크가 프레임 단위로 짝",
            ha="center", fontsize=8.2, color=INK2)

    # ---- bottom concrete Garden strip -------------------------------------
    labels = [
        (ref, "reference view (1 / 60)", "실제 촬영 다중뷰", GREEN),
        (ref_render, "3DGS render ‘변화 전’", "사전학습 가우시안 렌더", BLUE),
        (inf, "inference query (1 / 25)", "변화 후 시점", YELLOW),
        (gt, "GT change mask", "이 프레임의 정답", RED),
    ]
    for c, (img, t, s, ec) in enumerate(labels):
        a = fig.add_axes([0.045 + c * 0.238, 0.075, 0.20, 0.26])
        thumb(a, img if c != 3 else img, t, s, ec)
        if c == 3:
            a.images[0].set_cmap("gray")
    fig.text(0.5, 0.365, "장면 예시 — Garden (Instance_1)", ha="center",
             fontsize=12, fontweight="bold", color=INK)
    fig.text(0.5, 0.028,
             "과제: 각 inference 시점에서 ‘변화 전’ 3DGS 대비 무엇이 달라졌는지 검출 → "
             "gt_mask로 평가 (online mIoU / F1)",
             ha="center", fontsize=10, color=INK2, fontweight="bold")

    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved", os.path.relpath(OUT, REPO))


if __name__ == "__main__":
    main()
