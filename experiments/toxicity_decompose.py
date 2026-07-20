# Quality-vs-geometry decomposition of frame toxicity (Part 2 cycle 3 prep).
#
# The oracle map's per-frame "marginal" mixes two effects (user question,
# 2026-07-20): intrinsic frame/cue QUALITY (set-independent: a bad cue hurts
# in any company) and geometric NBV VALUE (set-conditional: new parallax on
# the change region given the current set). They have different signatures:
#
#   QUALITY (intrinsic) : toxic on average AND caps even its best combo —
#       good company cannot rescue it. Ceiling of combos-containing-v is far
#       below the scene's global best.
#   GEOMETRY (conditional): toxic on average but appears in near-top combos —
#       the right complementary company rescues it. High ceiling, negative
#       mean.
#
# Two independent lenses, both GT-based (understanding only, no selector):
#   1. Ceiling test  — scene_best - best(combos containing v), vs scene spread.
#   2. Additive model — mIoU ~ Sigma a_v [v in combo] (ridge). R^2 = share of
#      the map explained by set-independent main effects (=quality-ish); the
#      residual is the non-additive interaction (=geometry). a_v is v's
#      additive drag.
#
# Cross-check: does the cue-vs-consensus toxicity score (cue_toxicity_scores
# .csv) track the additive main effect a_v better than the raw marginal? If
# so, that signal was being validated against the wrong (mixed) target.
#
#   python experiments/toxicity_decompose.py
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORACLE_CSV = os.path.join(REPO, "experiments", "oracle_search_results.csv")
CUE_CSV = os.path.join(REPO, "experiments", "cue_toxicity_scores.csv")
OUT_CSV = os.path.join(REPO, "experiments", "toxicity_decompose_scores.csv")
N_FRAMES = 25

# findings §5 rev3-stable worst frames (the ones we called toxic)
KNOWN_TOXIC = {"Porch": [0, 7], "Zen": [3, 4, 6], "Printing_area": [4]}


def load_combos() -> dict[str, dict[tuple, float]]:
    data: dict[str, dict[tuple, float]] = defaultdict(dict)
    for r in csv.DictReader(open(ORACLE_CSV)):
        parts = r["combo"].split("-")
        if len(parts) != 5:
            continue
        combo = tuple(sorted(int(x) for x in parts))
        miou = float(r["miou_query"])
        d = data[r["scene"]]
        if miou > d.get(combo, -1.0):  # best observed per combo (denoise)
            d[combo] = miou
    return data


def additive_fit(combos: dict[tuple, float], alpha: float = 1.0):
    """Ridge additive main effects. Returns (a[25], r2, residuals dict)."""
    items = list(combos.items())
    n = len(items)
    X = np.zeros((n, N_FRAMES))
    y = np.array([m for _, m in items])
    for i, (c, _) in enumerate(items):
        for v in c:
            X[i, v] = 1.0
    ybar = y.mean()
    yc = y - ybar
    # ridge on the dummies (columns collinear: every row sums to 5)
    A = X.T @ X + alpha * np.eye(N_FRAMES)
    a = np.linalg.solve(A, X.T @ yc)
    pred = ybar + X @ a
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - ybar) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    resid = {c: float(y[i] - pred[i]) for i, (c, _) in enumerate(items)}
    return a, r2, resid


def spearman(x, y) -> float:
    def rank(a):
        a = np.asarray(a, float); o = a.argsort(); r = np.empty_like(a)
        r[o] = np.arange(len(a), dtype=float)
        for v in np.unique(a):
            m = a == v
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r
    rx, ry = rank(x), rank(y); rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / d) if d > 0 else 0.0


def main():
    data = load_combos()
    cue = {}
    if os.path.exists(CUE_CSV):
        for r in csv.DictReader(open(CUE_CSV)):
            cue[(r["scene"], int(r["frame"]))] = float(r["tox_mass"])

    rows = []
    print(f"scene-level: additive R^2 = share explained by set-independent "
          f"main effects (quality-ish); 1-R^2 = interaction (geometry)\n")
    print(f"{'scene':14s} {'addR2':>6s} {'best5':>6s} {'spread':>6s}")
    scene_a = {}
    scene_resid = {}
    for scene in sorted(data):
        combos = data[scene]
        a, r2, resid = additive_fit(combos)
        scene_a[scene] = a
        scene_resid[scene] = resid
        vals = np.array(list(combos.values()))
        spread = float(vals.std())
        best5 = float(vals.max())
        print(f"{scene:14s} {r2:6.3f} {best5:6.3f} {spread:6.3f}")

        with_v = defaultdict(list)
        for c, m in combos.items():
            for v in c:
                with_v[v].append(m)
        gmean = float(vals.mean())
        for v in range(N_FRAMES):
            wv = np.array(with_v.get(v, []))
            if wv.size == 0:
                continue
            without = np.array([m for c, m in combos.items() if v not in c])
            marginal = float(wv.mean() - without.mean())
            ceiling_gap = best5 - float(wv.max())      # small => rescuable
            resid_std = float(np.std([resid[c] for c in combos if v in c]))
            rows.append({
                "scene": scene, "frame": v, "n_with": int(wv.size),
                "marginal": marginal, "a_v": float(a[v]),
                "ceiling_gap": ceiling_gap, "ceil_gap_z": ceiling_gap / spread,
                "resid_std": resid_std,
                "best_with": float(wv.max()), "mean_with": float(wv.mean()),
            })

    # ---- classify the frames WE flagged toxic --------------------------------
    print("\nknown-toxic frames — is the toxicity QUALITY or GEOMETRY?")
    print(f"{'scene/frame':16s} {'marginal':>9s} {'a_v':>8s} "
          f"{'ceil_gapz':>9s} {'verdict':>22s}")
    idx = {(r["scene"], r["frame"]): r for r in rows}
    for scene, frames in KNOWN_TOXIC.items():
        for fr in frames:
            r = idx.get((scene, fr))
            if r is None:
                continue
            # rescuable ceiling (gap < 0.5 sigma) + mild a_v => geometry;
            # capped ceiling (gap > 1 sigma) => quality-intrinsic
            if r["ceil_gap_z"] < 0.5:
                verdict = "GEOMETRY (rescuable)"
            elif r["ceil_gap_z"] > 1.0:
                verdict = "QUALITY (capped)"
            else:
                verdict = "mixed"
            print(f"{scene+'/'+str(fr):16s} {r['marginal']:+9.4f} "
                  f"{r['a_v']:+8.4f} {r['ceil_gap_z']:9.2f} {verdict:>22s}")

    # ---- anchors that are "toxic on average" (the geometry tell) --------------
    print("\noracle-best-combo frames whose AVERAGE marginal is negative "
          "(valuable only in the right company = pure geometry):")
    best_combo = {sc: max(d, key=d.get) for sc, d in data.items()}
    for scene in sorted(data):
        flips = [v for v in best_combo[scene]
                 if idx.get((scene, v)) and idx[(scene, v)]["marginal"] < 0]
        if flips:
            print(f"  {scene:14s} {flips}")

    # ---- cross-check: cue-tox vs which target? -------------------------------
    if cue:
        common = [(r["scene"], r["frame"]) for r in rows
                  if (r["scene"], r["frame"]) in cue]
        cvals = [cue[k] for k in common]
        mvals = [idx[k]["marginal"] for k in common]
        avals = [idx[k]["a_v"] for k in common]
        print(f"\ncue-tox_mass vs targets (pooled, n={len(common)}):")
        print(f"  vs raw marginal : spearman {spearman(cvals, mvals):+.3f}")
        print(f"  vs additive a_v : spearman {spearman(cvals, avals):+.3f}"
              f"   (main-effect / quality component)")

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"\nsaved {OUT_CSV}")


if __name__ == "__main__":
    main()
