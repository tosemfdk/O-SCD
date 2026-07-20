# Part 3.1 analysis: aggregate the per-frame metrics the scene runner wrote,
# attribute them to Gaussian-space TP purity, and draw the §15 montages and
# §16 plots.
#
# Reads only what visualize_rchange_importance.py produced — no GPU, no model.
#
#   python experiments/analyze_rchange_importance.py --scenes all
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTANCE = "Instance_1"
ROOT = os.path.join(REPO, "outputs", "change_nbv", "diagnostics",
                    "rchange_importance", INSTANCE)
SUMMARY = os.path.join(REPO, "experiments", "rchange_importance_summary.csv")
SCENES = ["Cantina", "Garden", "Lounge", "Lunch_room", "Meeting_room",
          "Playground", "Porch", "Pots", "Printing_area", "Zen"]
HIGHLIGHT = {"Zen": [3, 4, 6], "Porch": [0, 7], "Printing_area": [4],
             "Garden": [15, 23]}
MONTAGE_SCENES = ["Zen", "Porch", "Printing_area", "Garden"]

BLUE, GREEN, YELLOW, AQUA, RED = ("#2a78d6", "#008300", "#eda100", "#1baf7a",
                                  "#e34948")
INK, INK2, GRID, MUTED = "#1a1a19", "#5f5e58", "#e6e5e0", "#b5b4ac"
STATE_PANELS = [("I1_beta_mean", "beta mean m"), ("S0_support", "support S"),
                ("A0_agreement", "LOO agreement A"),
                ("O0_observability", "observability O"),
                ("U0_uncertainty", "uncertainty U"),
                ("C0_confidence", "confidence Conf")]
IMP_PANELS = ["I0_raw_c", "I1_beta_mean", "I2_mean_support",
              "I3_mean_agreement", "I4_mean_support_agreement",
              "I5_full_confirmed", "I6_rawc_full_confidence",
              "V0_verification"]

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "bold",
})


def scene_dir(scene):
    return os.path.join(ROOT, scene)


def have(scene):
    return os.path.exists(os.path.join(scene_dir(scene), "frame_metrics.csv"))


def load_rows(scene):
    with open(os.path.join(scene_dir(scene), "frame_metrics.csv")) as f:
        return list(csv.DictReader(f))


def fnum(r, k):
    v = r.get(k, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def aggregate(scene, rows):
    """Per (variant, norm, task) mean over frames — frames are equally weighted,
    the same convention utils/evaluate.py uses for mIoU."""
    keys = [k for k in rows[0] if k not in
            ("scene", "variant", "norm", "frame", "image", "task")]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["variant"], r["norm"], r["task"])].append(r)
    out = []
    for (variant, norm, task), rs in groups.items():
        rec = {"scene": scene, "variant": variant, "norm": norm, "task": task,
               "n_frames": len(rs)}
        for k in keys:
            vals = [fnum(r, k) for r in rs]
            vals = [v for v in vals if v == v]
            if vals:
                rec[k] = float(np.mean(vals))
        out.append(rec)
    return out


def write_csv(path, rows):
    if not rows:
        return
    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x[ok])).astype(float)
    ry = np.argsort(np.argsort(y[ok])).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    return float((rx * ry).sum() / (np.linalg.norm(rx) * np.linalg.norm(ry)))


def auroc(score, label):
    """Rank-based AUROC of `score` separating label==1 from label==0."""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    n_pos, n_neg = int(label.sum()), int((~label).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(score)).astype(float) + 1
    return float((ranks[label].sum() - n_pos * (n_pos + 1) / 2)
                 / (n_pos * n_neg))


# ---- §14 Gaussian-space attribution ---------------------------------------

def gaussian_attribution(scene, mass_min_q=0.10):
    """Spearman / AUROC of each importance against oracle TP purity.

    This is alpha-compositing attribution, NOT a 3D GT label: a Gaussian is
    called TP-supported when the pixels it is responsible for are mostly true
    positives of the all-25 prediction."""
    d = scene_dir(scene)
    st = torch.load(os.path.join(d, "gaussian_state.pt"), weights_only=False)
    at = torch.load(os.path.join(d, "gt_attribution.pt"), weights_only=False)
    total = at["total_mass"].double()
    valid = total >= torch.quantile(total[total > 0].double(), mass_min_q)
    purity = at["tp_purity"].double()[valid].numpy()
    supported = purity >= 0.5
    w = total[valid].numpy()

    sc = st["scalars"]
    rows = []
    for name, vec in sc.items():
        v = vec.double()[valid].numpy()
        rows.append({
            "scene": scene, "scalar": name,
            "spearman_tp_purity": spearman(v, purity),
            "weighted_spearman": spearman(v * 0 + _rank(v) * w, purity),
            "auroc_tp_supported": auroc(v, supported),
            "n_valid": int(valid.sum()),
            "frac_tp_supported": float(supported.mean()),
        })
    return rows, (sc, purity, supported, valid)


def _rank(x):
    return np.argsort(np.argsort(x)).astype(float)


# ---- §15 montages ----------------------------------------------------------

def _heat(path):
    return torch.load(path, weights_only=False).float().numpy()


def _norm_pair(scene, variant):
    """Scene-wide Q01-Q99 for one variant, shared by every frame (§11)."""
    d = os.path.join(scene_dir(scene), "renders", variant)
    files = sorted(f for f in os.listdir(d) if f.endswith(".pt"))
    allv = np.concatenate([_heat(os.path.join(d, f)).ravel() for f in files])
    return float(np.quantile(allv, 0.01)), float(np.quantile(allv, 0.99))


def error_rgb(pred, gt):
    h, w = gt.shape
    img = np.ones((h, w, 3), np.float32)
    img[pred & gt] = (0.15, 0.65, 0.15)
    img[pred & ~gt] = (0.85, 0.15, 0.15)
    img[~pred & gt] = (0.15, 0.30, 0.85)
    return img


def montage(scene, frame, kind, ctx, scales):
    """kind='state' → §15 A, kind='importance' → §15 B."""
    panels = STATE_PANELS if kind == "state" else [(v, v) for v in IMP_PANELS]
    head = 6 if kind == "state" else 2
    n = head + len(panels)
    cols = min(6, n)
    rowsn = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(3.1 * cols, 2.5 * rowsn))
    gs = GridSpec(rowsn, cols, figure=fig, hspace=0.30, wspace=0.06)
    axes = [fig.add_subplot(gs[i // cols, i % cols]) for i in range(n)]

    rgb = ctx["original"][frame].permute(1, 2, 0).numpy()
    ref = ctx["reference_render"][frame].permute(1, 2, 0).numpy()
    cue = ctx["cue"][frame].float().numpy()
    pred = ctx["pred"][frame].numpy()
    gt = ctx["gt"][frame].numpy()

    if kind == "state":
        first = [(rgb, "original RGB", None), (ref, "3DGS reference render", None),
                 (cue, "combined cue C_v", "inferno"),
                 (_heat(os.path.join(scene_dir(scene), "renders",
                                     "I0_raw_c", f"frame_{frame:02d}.pt")),
                  "all-25 raw change", "inferno"),
                 (gt, "GT change", "gray"),
                 (error_rgb(pred, gt), "all-25 TP/FP/FN", None)]
    else:
        first = [(rgb, "original RGB", None),
                 (error_rgb(pred, gt), "all-25 TP/FP/FN", None)]
    for ax, (img, title, cmap) in zip(axes, first):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=9)
    for ax, (variant, title) in zip(axes[head:], panels):
        lo, hi = scales[variant]
        p = os.path.join(scene_dir(scene), "renders", variant,
                         f"frame_{frame:02d}.pt")
        ax.imshow(_heat(p), cmap="inferno", vmin=lo, vmax=hi)
        ax.set_title(f"{title}\n[{lo:.3g}, {hi:.3g}]", fontsize=8)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{scene} / frame {frame} ({ctx['image_names'][frame]}) — "
                 f"{'state' if kind=='state' else 'importance'} montage",
                 fontsize=13, y=0.995)
    out = os.path.join(scene_dir(scene), "montages",
                       f"{kind}_frame_{frame:02d}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def contact_sheet(scene, variant, ctx, scales):
    lo, hi = scales[variant]
    d = os.path.join(scene_dir(scene), "renders", variant)
    files = sorted(f for f in os.listdir(d) if f.endswith(".pt"))
    cols = 5
    rowsn = int(np.ceil(len(files) / cols))
    fig, axes = plt.subplots(rowsn, cols, figsize=(2.6 * cols, 1.9 * rowsn))
    for ax, f in zip(axes.ravel(), files):
        ax.imshow(_heat(os.path.join(d, f)), cmap="inferno", vmin=lo, vmax=hi)
        ax.set_title(f[:-3].replace("frame_", "f"), fontsize=7)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{scene} — {variant} (shared scale [{lo:.3g}, {hi:.3g}])",
                 fontsize=12)
    out = os.path.join(scene_dir(scene), "montages", f"contact_{variant}.png")
    fig.savefig(out, dpi=95, bbox_inches="tight")
    plt.close(fig)


# ---- §16 plots -------------------------------------------------------------

def plot_variant_bars(agg, scene, metric, title, fname, baseline="I0_raw_c"):
    rs = [r for r in agg if r["task"] == "tp_vs_fp" and r["norm"] == "raw"
          and metric in r]
    if not rs:
        return
    rs.sort(key=lambda r: r[metric])
    names = [r["variant"] for r in rs]
    vals = [r[metric] for r in rs]
    base = next((r[metric] for r in rs if r["variant"] == baseline), None)
    colors = [GREEN if base is not None and v > base else MUTED for v in vals]
    fig, ax = plt.subplots(figsize=(7.6, 0.30 * len(rs) + 1.6))
    ax.barh(names, vals, color=colors, height=0.62)
    if base is not None:
        ax.axvline(base, color=RED, lw=1.4, ls="--",
                   label=f"raw c baseline = {base:.3f}")
        ax.legend(frameon=False, loc="lower right")
    for y, v in enumerate(vals):
        ax.text(v, y, f" {v:.3f}", va="center", fontsize=8, color=INK2)
    ax.set_xlabel(metric)
    ax.set_title(f"{scene} — {title}", loc="left")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.savefig(os.path.join(scene_dir(scene), "plots", fname), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def plot_tp_fp_means(agg, scene):
    rs = [r for r in agg if r["task"] == "tp_vs_fp" and r["norm"] == "raw"
          and "mean_tp" in r]
    if not rs:
        return
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    lim = max(max(r["mean_tp"] for r in rs), max(r["mean_fp"] for r in rs))
    ax.plot([0, lim], [0, lim], color=MUTED, lw=1, ls="--", zorder=1)
    ax.scatter([r["mean_fp"] for r in rs], [r["mean_tp"] for r in rs],
               s=60, color=BLUE, edgecolor="white", linewidth=1.2, zorder=3)
    for r in rs:
        ax.annotate(r["variant"], (r["mean_fp"], r["mean_tp"]), fontsize=7,
                    textcoords="offset points", xytext=(5, 3), color=INK2)
    ax.set_xlabel("mean importance on FP pixels")
    ax.set_ylabel("mean importance on TP pixels")
    ax.set_title(f"{scene} — above the diagonal = importance prefers real "
                 f"change", loc="left")
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.savefig(os.path.join(scene_dir(scene), "plots",
                             "tp_fp_separation.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def plot_topk_curves(agg, scene, percents=(1, 5, 10, 20, 30, 50)):
    rs = [r for r in agg if r["task"] == "tp_vs_fp" and r["norm"] == "raw"]
    show = [v for v in ("I0_raw_c", "I1_beta_mean", "I2_mean_support",
                        "I4_mean_support_agreement", "I5_full_confirmed",
                        "C0_confidence")]
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ramp = [MUTED, "#a8c8ee", "#6ba3e0", BLUE, "#1a4e8f", GREEN]
    for c, name in zip(ramp, show):
        r = next((x for x in rs if x["variant"] == name), None)
        if r is None:
            continue
        fp = [r.get(f"top{p}_fp_retained", np.nan) for p in percents]
        tp = [r.get(f"top{p}_tp_retained", np.nan) for p in percents]
        ax.plot(fp, tp, marker="o", ms=5, lw=2, color=c, label=name)
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1)
    ax.set_xlabel("FP mass retained")
    ax.set_ylabel("TP mass retained")
    ax.set_title(f"{scene} — top-percentile filtering of the all-25 positive",
                 loc="left")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.savefig(os.path.join(scene_dir(scene), "plots", "topk_retention.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_purity_scatters(scene, sc, purity, supported, valid, sub=20000):
    pairs = [("c_raw", "raw c"), ("m", "beta mean m"), ("support", "support S"),
             ("agreement", "agreement A"), ("observability", "observability O"),
             ("uncertainty", "uncertainty U"), ("confidence", "confidence"),
             ("verification", "verification")]
    idx = np.random.default_rng(0).choice(len(purity),
                                          size=min(sub, len(purity)),
                                          replace=False)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, (key, label) in zip(axes.ravel(), pairs):
        v = sc[key].double()[valid].numpy()
        ax.scatter(v[idx], purity[idx], s=3, alpha=0.10, color=BLUE,
                   edgecolors="none")
        ax.set_xlabel(label)
        ax.set_ylabel("TP purity")
        ax.set_title(f"rho={spearman(v, purity):+.3f}  "
                     f"AUROC={auroc(v, supported):.3f}", fontsize=10)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    fig.suptitle(f"{scene} — Gaussian state vs oracle TP purity "
                 f"(responsibility attribution, not a 3D GT label)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(scene_dir(scene), "plots",
                             "importance_vs_tp_purity.png"), dpi=130)
    plt.close(fig)


def plot_state_scatter(scene, sc, sub=30000):
    idx = np.random.default_rng(1).choice(sc["support"].numel(),
                                          size=min(sub, sc["support"].numel()),
                                          replace=False)
    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    ax.scatter(sc["support"].numpy()[idx], sc["agreement"].numpy()[idx],
               s=4, alpha=0.12, c=sc["m"].numpy()[idx], cmap="inferno")
    ax.set_xlabel("support S")
    ax.set_ylabel("LOO agreement A")
    ax.set_title(f"{scene} — support vs agreement (color = belief m)",
                 loc="left")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.savefig(os.path.join(scene_dir(scene), "plots", "state_scatter.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_toxic_vs_oracle(scene, rows, oracle_frames):
    """§16.11: do the flagged-toxic frames look different in state space?"""
    tox = set(HIGHLIGHT.get(scene, []))
    orc = set(oracle_frames)
    if not tox or not orc:
        return
    keys = ["auprc", "precision_at_recall0.95", "mean_tp", "mean_fp"]
    rs = [r for r in rows if r["task"] == "tp_vs_fp" and r["norm"] == "raw"
          and r["variant"] == "I2_mean_support"]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.4 * len(keys), 3.6))
    for ax, k in zip(axes, keys):
        a = [fnum(r, k) for r in rs if int(r["frame"]) in tox]
        b = [fnum(r, k) for r in rs if int(r["frame"]) in orc]
        ax.bar(["toxic", "oracle-5"], [np.nanmean(a), np.nanmean(b)],
               color=[RED, GREEN], width=0.55)
        ax.set_title(k, fontsize=10)
        ax.grid(axis="y", color=GRID, lw=0.7)
        ax.set_axisbelow(True)
    fig.suptitle(f"{scene} — I2_mean_support on toxic vs oracle-best-5 frames",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(scene_dir(scene), "plots",
                             "toxic_vs_oracle.png"), dpi=140)
    plt.close(fig)


def oracle_best5(scene):
    path = os.path.join(REPO, "experiments", "oracle_search_results.csv")
    best, best_m = [], -1.0
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["scene"] != scene:
                continue
            combo = [int(x) for x in r["combo"].split("-")]
            if len(combo) == 5 and float(r["miou_query"]) > best_m:
                best, best_m = combo, float(r["miou_query"])
    return best


# ---------------------------------------------------------------- main -----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["all"])
    ap.add_argument("--montage_scenes", nargs="+", default=MONTAGE_SCENES)
    ap.add_argument("--no_montage", action="store_true")
    args = ap.parse_args()
    scenes = SCENES if args.scenes == ["all"] else args.scenes
    scenes = [s for s in scenes if have(s)]

    summary, purity_rows = [], []
    for scene in scenes:
        d = scene_dir(scene)
        os.makedirs(os.path.join(d, "plots"), exist_ok=True)
        rows = load_rows(scene)
        agg = aggregate(scene, rows)
        req = [r for r in agg if not r["variant"].startswith("sweep_")]
        swp = [r for r in agg if r["variant"].startswith("sweep_")]
        write_csv(os.path.join(d, "variant_metrics.csv"), req)
        write_csv(os.path.join(d, "sweep_metrics.csv"), swp)
        summary.extend(agg)

        pr, (sc, purity, supported, valid) = gaussian_attribution(scene)
        purity_rows.extend(pr)
        write_csv(os.path.join(d, "gaussian_attribution.csv"), pr)

        plot_variant_bars(agg, scene, "auprc", "TP-vs-FP AUPRC by variant",
                          "tp_fp_auprc.png")
        plot_variant_bars(agg, scene, "precision_at_recall0.95",
                          "precision at TP recall >= 0.95",
                          "precision_at_recall.png")
        plot_variant_bars(agg, scene, "fp_rejection_at_recall0.95",
                          "FP rejection at TP recall >= 0.95",
                          "fp_rejection_at_recall.png")
        plot_tp_fp_means(agg, scene)
        plot_topk_curves(agg, scene)
        plot_purity_scatters(scene, sc, purity, supported, valid)
        plot_state_scatter(scene, sc)
        plot_toxic_vs_oracle(scene, rows, oracle_best5(scene))
        print(f"{scene}: plots + {len(agg)} aggregate rows")

        if not args.no_montage and scene in args.montage_scenes:
            ctx = torch.load(os.path.join(d, "frame_context.pt"),
                             weights_only=False)
            variants = sorted(os.listdir(os.path.join(d, "renders")))
            scales = {v: _norm_pair(scene, v) for v in variants}
            frames = sorted(set(HIGHLIGHT.get(scene, []))
                            | set(oracle_best5(scene)))
            for fr in frames:
                montage(scene, fr, "state", ctx, scales)
                montage(scene, fr, "importance", ctx, scales)
            for v in ("I0_raw_c", "I2_mean_support", "I5_full_confirmed",
                      "C0_confidence"):
                if v in scales:
                    contact_sheet(scene, v, ctx, scales)
            print(f"{scene}: montages for frames {frames}")

    write_csv(SUMMARY, summary)
    write_csv(os.path.join(REPO, "experiments",
                           "rchange_importance_gaussian_attribution.csv"),
              purity_rows)
    _print_overall(summary)


def _print_overall(summary):
    """10-scene mean ranking (§10) of the required variants and the sweep."""
    by = defaultdict(list)
    for r in summary:
        if r["task"] == "tp_vs_fp" and r["norm"] == "raw" and "auprc" in r:
            by[r["variant"]].append(r)
    rows = []
    for v, rs in by.items():
        rows.append((float(np.mean([r["auprc"] for r in rs])),
                     float(np.mean([r.get("precision_at_recall0.95", np.nan)
                                    for r in rs])),
                     float(np.mean([r.get("fp_rejection_at_recall0.95",
                                          np.nan) for r in rs])),
                     v, len(rs)))
    rows.sort(reverse=True)
    print(f"\n=== {len(by)} variants, mean over scenes (TP-vs-FP, raw) ===")
    print(f"{'variant':32s}{'AUPRC':>8}{'p@r.95':>9}{'FPrej@.95':>11}{'n':>4}")
    for a, p, f, v, n in rows[:15]:
        print(f"{v:32s}{a:8.4f}{p:9.4f}{f:11.4f}{n:4d}")
    if len(rows) > 15:
        print("  ...")
        for a, p, f, v, n in rows[-2:]:
            print(f"{v:32s}{a:8.4f}{p:9.4f}{f:11.4f}{n:4d}")


if __name__ == "__main__":
    main()
