# Aggregate a target_nbv_pool_eval.py results.csv into the method-comparison
# tables used in experiments/pool_eval_garden/report.md.
#
#   python experiments/analyze_pool_eval.py experiments/pool_eval_garden/results.csv

from __future__ import annotations

import csv
import sys
from collections import defaultdict

import numpy as np


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["k"] = int(r["k"])
        r["logdet"] = float(r["logdet"])
        r["lmax"] = float(r["lmax_sigma"])
        r["valid"] = r["valid"] == "True"
        r["t"] = int(r["target_row"])
    return rows


def main(path):
    rows = load(path)
    targets = sorted({r["t"] for r in rows})
    methods = ["random", "uniform", "max_resp", "proxy", "exact"]
    budget = max(r["k"] for r in rows)
    base = {t: next(r["logdet"] for r in rows if r["t"] == t and r["k"] == 0)
            for t in targets}

    def dlogdet(method, t, k):
        """Mean over seeds of logdet gain at the largest k' <= k (early stops
        keep their last value: an exhausted method holds its information)."""
        per_seed = {}
        for r in rows:
            if r["method"] == method and r["t"] == t and r["k"] <= k:
                s = r["seed"]
                if s not in per_seed or r["k"] > per_seed[s][0]:
                    per_seed[s] = (r["k"], r["logdet"] - base[t])
        return np.mean([v for _, v in per_seed.values()])

    print("mean dlogdet(H) by budget k (nats; higher = more target information):")
    print("k    " + "".join(f"{m:>10s}" for m in methods))
    for k in range(budget + 1):
        print(f"{k}    " + "".join(
            f"{np.mean([dlogdet(m, t, k) for t in targets]):10.3f}" for m in methods))

    print("\nwasted picks (chosen view where target invisible), k>=1:")
    for m in methods:
        d = [not r["valid"] for r in rows if r["method"] == m and r["k"] > 0]
        print(f"  {m:9s}: {100 * np.mean(d):.1f}%")

    print(f"\nviews to reach uniform@{budget} information (mean over targets):")
    goal = {t: dlogdet("uniform", t, budget) for t in targets}
    for m in methods:
        per = []
        for t in targets:
            got = next((k for k in range(budget + 1)
                        if dlogdet(m, t, k) >= goal[t] - 1e-9), np.nan)
            per.append(got)
        a = np.array(per, dtype=float)
        print(f"  {m:9s}: {np.nanmean(a):.2f}  (reached {int(np.isfinite(a).sum())}/{len(a)} targets)")

    print(f"\nlmax(Sigma) remaining at k={budget} vs k=0 (lower = better, %):")
    for m in methods:
        ratio = []
        for t in targets:
            per_seed = {}
            for r in rows:
                if r["method"] == m and r["t"] == t:
                    s = r["seed"]
                    if s not in per_seed or r["k"] > per_seed[s][0]:
                        per_seed[s] = (r["k"], r["lmax"])
            l0 = next(r["lmax"] for r in rows
                      if r["method"] == m and r["t"] == t and r["k"] == 0)
            ratio.append(np.mean([v for _, v in per_seed.values()]) / l0)
        print(f"  {m:9s}: {100 * np.mean(ratio):6.2f}%")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "experiments/pool_eval_garden/results.csv")
