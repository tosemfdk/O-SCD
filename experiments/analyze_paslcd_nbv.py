# Aggregate experiments/paslcd_nbv_results.csv (20-instance sweep) into the
# method comparison: mean / min / per-scene wins on all-query-view mIoU & F1.
#
#   python experiments/analyze_paslcd_nbv.py [csv]

from __future__ import annotations

import csv
import sys
from collections import defaultdict

import numpy as np


def main(path="experiments/paslcd_nbv_results.csv"):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["budget"] = int(r["budget"])
        r["miou"] = float(r["miou_query"])
        r["f1"] = float(r["f1_query"])

    scenes = sorted({r["scene"] for r in rows})
    print(f"{len(scenes)} scenes, {len(rows)} rows\n")

    # per (scene, method, budget): mean over seeds
    cell = defaultdict(list)
    for r in rows:
        cell[(r["scene"], r["method"], r["budget"])].append((r["miou"], r["f1"]))

    def get(scene, method, budget):
        v = cell.get((scene, method, budget))
        if not v:
            return None
        return tuple(np.mean([x[i] for x in v]) for i in (0, 1))

    ref = {s: get(s, "all", 25) for s in scenes}

    configs = [("uniform", 3), ("random", 3), ("nbv", 3), ("nbv_dopt", 3),
               ("uniform", 5), ("random", 5), ("nbv", 5), ("nbv_dopt", 5)]
    print(f"{'config':>12s} {'mIoU mean':>10s} {'mIoU min':>9s} {'F1 mean':>8s} "
          f"{'vs all-25':>9s} {'scenes>=all':>11s}")
    m_all = np.mean([ref[s][0] for s in scenes if ref[s]])
    f_all = np.mean([ref[s][1] for s in scenes if ref[s]])
    print(f"{'all-25':>12s} {m_all:10.4f} "
          f"{min(ref[s][0] for s in scenes if ref[s]):9.4f} {f_all:8.4f}")
    for method, K in configs:
        vals = [(s, get(s, method, K)) for s in scenes]
        vals = [(s, v) for s, v in vals if v is not None]
        if not vals:
            continue
        mious = np.array([v[0] for _, v in vals])
        f1s = np.array([v[1] for _, v in vals])
        wins = sum(1 for s, v in vals if ref.get(s) and v[0] >= ref[s][0] - 0.005)
        print(f"{method + '@' + str(K):>12s} {mious.mean():10.4f} {mious.min():9.4f} "
              f"{f1s.mean():8.4f} {mious.mean() - m_all:+9.4f} {wins:6d}/{len(vals)}")

    # head-to-head vs uniform per budget
    for challenger in ("nbv", "nbv_dopt"):
        for K in (3, 5):
            diffs = []
            for s in scenes:
                n, u = get(s, challenger, K), get(s, "uniform", K)
                if n and u:
                    diffs.append((s, n[0] - u[0]))
            if not diffs:
                continue
            d = np.array([x[1] for x in diffs])
            wins = int((d > 0.005).sum())
            ties = int((np.abs(d) <= 0.005).sum())
            losses = int((d < -0.005).sum())
            print(f"\n{challenger} vs uniform @K={K}: mean {d.mean():+.4f} mIoU, "
                  f"W/T/L = {wins}/{ties}/{losses} (0.005 noise band)")
            worst = sorted(diffs, key=lambda x: x[1])[:3]
            best = sorted(diffs, key=lambda x: -x[1])[:3]
            print("  best :", ", ".join(f"{s} {v:+.3f}" for s, v in best))
            print("  worst:", ", ".join(f"{s} {v:+.3f}" for s, v in worst))


if __name__ == "__main__":
    main(*sys.argv[1:])
