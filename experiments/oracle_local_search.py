# Oracle-5 map via local search (user-directed design, 2026-07-16).
#
# Per Instance_1 scene: seed = the BEST stride-5 uniform offset combo (coarse
# solution from oracle_search.py Phase A), then hill-climb by swapping 1-3
# frames per step. Stop the scene early on a HIT:
#     mIoU >= 1.10 * max(best uniform offset, all-25)
# (simultaneously >=10% over both references), else after --per-scene evals.
# Everything is appended to experiments/oracle_search_results.csv (phase
# "local"), giving the oracle-5 map for later "what makes a good set" analysis.
#
#   python experiments/oracle_local_search.py --per-scene 100

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle_search import (CSV_PATH, N_FRAMES, K, REPO, SCENES,  # noqa: E402
                           load_done, record, run_combo)

PASLCD_CSV = os.path.join(REPO, "experiments", "paslcd_nbv_results.csv")


def load_references():
    """(uniform_best[scene], uniform_best_combo[scene], all25[scene])."""
    done = load_done()
    uni_best, uni_combo = {}, {}
    offsets = {"-".join(map(str, range(o, N_FRAMES, 5))) for o in range(5)}
    for (scene, combo), (miou, _) in done.items():
        if combo in offsets and (scene not in uni_best or miou > uni_best[scene]):
            uni_best[scene] = miou
            uni_combo[scene] = tuple(int(x) for x in combo.split("-"))
    all25 = {}
    for r in csv.DictReader(open(PASLCD_CSV)):
        if r["method"] == "all" and r["scene"].startswith("Instance_1/"):
            all25[r["scene"].split("/", 1)[1]] = float(r["miou_query"])
    missing = [s for s in SCENES if s not in uni_best or s not in all25]
    if missing:
        raise RuntimeError(f"missing references for {missing}; run oracle_search.py Phase A first")
    return uni_best, uni_combo, all25


def mutate(combo: tuple[int, ...], m: int, rng) -> tuple[int, ...]:
    keep = list(combo)
    out_idx = rng.choice(K, size=m, replace=False)
    pool = [i for i in range(N_FRAMES) if i not in combo]
    new = rng.choice(pool, size=m, replace=False)
    for j, o in enumerate(out_idx):
        keep[o] = int(new[j])
    return tuple(sorted(keep))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-scene", type=int, default=100)
    ap.add_argument("--hit-margin-rel", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    uni_best, uni_combo, all25 = load_references()
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    summary = {}

    for scene in SCENES:
        done = load_done()
        target = (1.0 + args.hit_margin_rel) * max(uni_best[scene], all25[scene])
        # scene-best over everything already evaluated (seed included)
        best_combo, best = uni_combo[scene], uni_best[scene]
        for (sc, c), (miou, _) in done.items():
            if sc == scene and miou > best:
                best, best_combo = miou, tuple(int(x) for x in c.split("-"))
        print(f"\n=== {scene}: seed {best_combo} {best:.4f} | refs uniform_best "
              f"{uni_best[scene]:.4f}, all25 {all25[scene]:.4f} -> target {target:.4f}",
              flush=True)

        evals, hit = 0, False
        while evals < args.per_scene:
            m = int(rng.choice([1, 2, 3], p=[0.5, 0.3, 0.2]))
            combo = mutate(best_combo, m, rng)
            key = (scene, "-".join(map(str, combo)))
            if key in done:
                continue
            res = run_combo(scene, combo)
            evals += 1
            if res is None:
                continue
            record(scene, "local", combo, *res)
            done[key] = res
            marker = ""
            if res[0] > best:
                best, best_combo = res[0], combo
                marker = " <- new best"
            print(f"[{scene} {evals:3d}/{args.per_scene} {time.time()-t0:6.0f}s] "
                  f"{combo}: {res[0]:.4f} (best {best:.4f}){marker}", flush=True)
            if res[0] >= target:
                print(f"HIT {scene}: combo {combo} mIoU {res[0]:.4f} >= "
                      f"{target:.4f} (+10% over uniform_best AND all-25)", flush=True)
                hit = True
                break
        summary[scene] = (best_combo, best, hit, evals)
        print(f"SCENE DONE {scene}: best {best_combo} {best:.4f} "
              f"({best/all25[scene]-1:+.1%} vs all25, {best/uni_best[scene]-1:+.1%} "
              f"vs uniform_best), hit={hit}, evals={evals}", flush=True)

    print("\n=== ORACLE-5 MAP (Instance_1) ===", flush=True)
    for scene in SCENES:
        bc, b, hit, ev = summary[scene]
        print(f"  {scene:15s}: best5 {b:.4f} vs all25 {all25[scene]:.4f} "
              f"({b/all25[scene]-1:+.1%}) vs uniform_best {uni_best[scene]:.4f} "
              f"({b/uni_best[scene]-1:+.1%})  combo {bc}  hit={hit} evals={ev}", flush=True)
    print("ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
