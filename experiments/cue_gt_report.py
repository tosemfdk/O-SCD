# Per-image cue-vs-GT report for all Instance_1 inference frames (user request
# 2026-07-20). For every frame: dump the exact fusion cue (generate_candidate_
# map — same call the fusion loss consumes), save a heatmap panel, and score it
# against the GT change mask.
#
# The cue is a continuous per-image-normalized change map (~[0,2]); it is
# binarized at --tau only to COUNT errors (fusion itself uses the continuous
# value). Two quantities the user asked for, per image:
#   FP fraction  = FP / (TP+FP) = 1 - precision  ("변화라 예측했는데 실제 아닌 비율")
#   miss (FN) fr = FN / (TP+FN) = 1 - recall      ("아니라 예측했는데 실제 변화인 비율")
#
#   python experiments/cue_gt_report.py            # dump (if needed) + analyze
#   python experiments/cue_gt_report.py --tau 0.5
from __future__ import annotations

import argparse
import csv
import os
import queue
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle_search import INSTANCE, SCENES  # noqa: E402

DUMP_ROOT = os.path.join(REPO, "output_subset", "cue_gt")
HEAT_DIR = os.path.join(REPO, "experiments", "figures", "cue_maps_instance1")
CSV_PATH = os.path.join(REPO, "experiments", "cue_gt_comparison_instance1.csv")
_t0 = time.time()
_lock = threading.Lock()


def log(m):
    with _lock:
        print(f"[{time.time()-_t0:5.0f}s] {m}", flush=True)


def dump_scene(scene: str, gpu: int):
    out = os.path.join(DUMP_ROOT, scene)
    if os.path.exists(os.path.join(out, "cues.pt")):
        return
    env = {**os.environ, "PYTHONPATH": "", "TORCHDYNAMO_DISABLE": "1",
           "CUDA_VISIBLE_DEVICES": str(gpu)}
    r = subprocess.run(
        [sys.executable, os.path.join(REPO, "subset_oscd.py"),
         "-s", os.path.join(REPO, "data", "PASLCD", INSTANCE, scene) + "/",
         "-m", out + "/", "--resolution", "4", "--test_hold", "5",
         "--dump_cues"], capture_output=True, text=True, env=env, cwd=REPO)
    if r.returncode != 0:
        log(f"DUMP FAILED {scene}: {r.stderr[-300:]}")
    else:
        log(f"dumped {scene}")


def gt_for(scene, name, hw):
    p = os.path.join(REPO, "data", "PASLCD", INSTANCE, scene, "gt_mask",
                     name + ".png")
    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    m = cv2.resize(m, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
    return m > 127


def to_hwc(t):
    a = t.numpy()
    if a.ndim == 3 and a.shape[0] in (1, 3):
        a = np.transpose(a, (1, 2, 0))
    return np.clip(a, 0, 1)


def error_rgb(cue_bin, gt):
    """White bg; TP green, FP red (predicted-change-but-not),
    FN blue (predicted-none-but-real-change)."""
    h, w = gt.shape
    img = np.ones((h, w, 3), np.float32)
    tp = cue_bin & gt
    fp = cue_bin & ~gt
    fn = ~cue_bin & gt
    img[tp] = (0.15, 0.65, 0.15)
    img[fp] = (0.85, 0.15, 0.15)
    img[fn] = (0.15, 0.30, 0.85)
    return img


def analyze(scene, tau, cue_vmax, save_panels=True):
    blob = torch.load(os.path.join(DUMP_ROOT, scene, "cues.pt"),
                      map_location="cpu", weights_only=False)
    cues, originals, names = blob["cues"], blob["originals"], blob["image_names"]
    if cues.ndim == 4:
        cues = cues[:, 0]
    sdir = os.path.join(HEAT_DIR, scene)
    if save_panels:
        os.makedirs(sdir, exist_ok=True)
    rows = []
    for i, name in enumerate(names):
        cue = cues[i].numpy()
        gt = gt_for(scene, name, cue.shape)
        if gt is None:
            continue
        cb = cue > tau
        tp = int((cb & gt).sum()); fp = int((cb & ~gt).sum())
        fn = int((~cb & gt).sum()); tn = int((~cb & ~gt).sum())
        npix = cue.size
        pred_pos = tp + fp
        gt_pos = tp + fn
        precision = tp / pred_pos if pred_pos else float("nan")
        recall = tp / gt_pos if gt_pos else float("nan")
        iou = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")
        fp_frac = fp / pred_pos if pred_pos else float("nan")   # 1-precision
        miss_frac = fn / gt_pos if gt_pos else float("nan")     # 1-recall
        rows.append({
            "scene": scene, "frame": i, "image": name,
            "gt_area_pct": 100 * gt_pos / npix,
            "cue_area_pct": 100 * pred_pos / npix,
            "precision": precision, "recall": recall, "iou": iou,
            "fp_frac_of_pred": fp_frac,      # predicted change that is NOT real
            "miss_frac_of_gt": miss_frac,    # real change the cue missed
            "fp_area_pct": 100 * fp / npix,  # false-alarm area over the frame
            "fn_area_pct": 100 * fn / npix,
        })
        if save_panels:
            fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.1))
            ax[0].imshow(to_hwc(originals[i])); ax[0].set_title("original RGB", fontsize=10)
            ax[1].imshow(cue, cmap="inferno", vmin=0, vmax=cue_vmax)
            ax[1].set_title("cue (SSIM+L1+SAM2)", fontsize=10)
            ax[2].imshow(gt, cmap="gray"); ax[2].set_title("GT change mask", fontsize=10)
            for a in ax:
                a.set_xticks([]); a.set_yticks([])
            fig.suptitle(f"{scene}/{i} ({name})  "
                         f"prec {precision:.2f}  recall {recall:.2f}  IoU {iou:.2f}  "
                         f"|  FP {fp_frac*100:.0f}% of pred  ·  miss {miss_frac*100:.0f}% of GT",
                         fontsize=11, y=1.02)
            fig.tight_layout()
            fig.savefig(os.path.join(sdir, f"frame_{i:02d}_{name}.png"),
                        dpi=95, bbox_inches="tight")
            plt.close(fig)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--cue_vmax", type=float, default=1.5)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--no-panels", action="store_true")
    args = ap.parse_args()

    pool = queue.Queue()
    for g in range(args.gpus):
        pool.put(g)

    def worker(sc):
        g = pool.get()
        try:
            dump_scene(sc, g)
        finally:
            pool.put(g)
    log(f"dumping cues for {len(SCENES)} scenes...")
    with ThreadPoolExecutor(max_workers=args.gpus) as ex:
        list(ex.map(worker, SCENES))

    all_rows = []
    for sc in SCENES:
        r = analyze(sc, args.tau, args.cue_vmax, save_panels=not args.no_panels)
        all_rows += r
        log(f"analyzed {sc} ({len(r)} frames)")

    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        for r in all_rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in r.items()})

    # ---- report ------------------------------------------------------------
    print(f"\n=== Instance_1 cue-vs-GT (τ={args.tau}, n={len(all_rows)} frames) ===")
    def col(rows, k): return np.array([r[k] for r in rows], float)
    ov = all_rows
    print(f"overall  precision {np.nanmean(col(ov,'precision')):.3f}  "
          f"recall {np.nanmean(col(ov,'recall')):.3f}  "
          f"IoU {np.nanmean(col(ov,'iou')):.3f}  |  "
          f"FP {np.nanmean(col(ov,'fp_frac_of_pred'))*100:.0f}% of pred  ·  "
          f"miss {np.nanmean(col(ov,'miss_frac_of_gt'))*100:.0f}% of GT")
    print(f"\n{'scene':14s} {'gtA%':>5s} {'cueA%':>6s} {'prec':>5s} {'rec':>5s} "
          f"{'IoU':>5s} {'FP%pred':>8s} {'miss%GT':>8s}  worst-FP  worst-miss")
    by = defaultdict(list)
    for r in ov:
        by[r["scene"]].append(r)
    for sc in SCENES:
        rs = by[sc]
        wfp = max(rs, key=lambda r: r["fp_frac_of_pred"] if r["fp_frac_of_pred"]==r["fp_frac_of_pred"] else -1)
        wms = max(rs, key=lambda r: r["miss_frac_of_gt"] if r["miss_frac_of_gt"]==r["miss_frac_of_gt"] else -1)
        print(f"{sc:14s} {np.nanmean(col(rs,'gt_area_pct')):5.1f} "
              f"{np.nanmean(col(rs,'cue_area_pct')):6.1f} "
              f"{np.nanmean(col(rs,'precision')):5.2f} "
              f"{np.nanmean(col(rs,'recall')):5.2f} "
              f"{np.nanmean(col(rs,'iou')):5.2f} "
              f"{np.nanmean(col(rs,'fp_frac_of_pred'))*100:7.0f}% "
              f"{np.nanmean(col(rs,'miss_frac_of_gt'))*100:7.0f}%  "
              f"f{wfp['frame']}:{wfp['fp_frac_of_pred']*100:.0f}%  "
              f"f{wms['frame']}:{wms['miss_frac_of_gt']*100:.0f}%")
    print(f"\nheatmap panels: {os.path.relpath(HEAT_DIR, REPO)}/<scene>/frame_XX.png")
    print(f"per-image CSV : {os.path.relpath(CSV_PATH, REPO)}")


if __name__ == "__main__":
    main()
