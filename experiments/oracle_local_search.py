# Oracle-5 map via local search (user-directed design, 2026-07-16, rev 2).
#
# Phase 0: all-25 is itself nondeterministic (cuda benchmark + atomics), so
#   run it 5x per scene and use the MEAN as the reference.
# Search, per Instance_1 scene: seed = the BEST stride-5 uniform offset combo
#   (from oracle_search.py Phase A), hill-climb by swapping 1-3 frames per
#   step; any improvement becomes the new mutation center. Scene stops on HIT:
#     mIoU >= 1.10 * uniform_best  AND  mIoU >= mean(all-25 x5)
#   or after --per-scene evals (default 30).
# Everything is appended to experiments/oracle_search_results.csv (phase
# "local"), giving the oracle-5 map for later "what makes a good set" analysis.
#
#   python experiments/oracle_local_search.py --per-scene 30

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

ALL25_CSV = os.path.join(REPO, "experiments", "all25_repeats.csv")


def run_all25(scene: str, rep: int) -> tuple[float, float] | None:
    """One all-25 run (frames_method=all), evaluated on query_mask."""
    import re
    import shutil
    import subprocess
    out_dir = os.path.join(REPO, "output_subset", "oracle", scene, f"all25_rep{rep}")
    src = os.path.join(REPO, "data", "PASLCD", "Instance_1", scene)
    env = {**os.environ, "PYTHONPATH": ""}
    r = subprocess.run(
        [sys.executable, os.path.join(REPO, "subset_oscd.py"),
         "-s", src + "/", "-m", out_dir + "/",
         "--resolution", "4", "--test_hold", "5", "--frames_method", "all"],
        capture_output=True, text=True, env=env, cwd=REPO)
    if r.returncode != 0:
        print(f"ALL25 FAILED {scene} rep{rep}: {r.stderr[-300:]}", flush=True)
        return None
    ev = subprocess.run(
        [sys.executable, os.path.join(REPO, "utils", "evaluate.py"),
         "--gt", os.path.join(src, "gt_mask") + "/",
         "--pred_binary", os.path.join(out_dir, "renders", "query_mask") + "/"],
        capture_output=True, text=True, env=env, cwd=REPO)
    m = re.search(r"Mean IoU: ([0-9.]+).*Mean F1: ([0-9.]+)", ev.stdout, re.S)
    shutil.rmtree(out_dir, ignore_errors=True)
    if not m:
        print(f"ALL25 EVAL FAILED {scene} rep{rep}", flush=True)
        return None
    return float(m.group(1)), float(m.group(2))


def all25_means(reps: int, t0: float) -> dict[str, float]:
    """Mean all-25 mIoU per scene over `reps` runs, cached in ALL25_CSV."""
    done: dict[str, list[float]] = {s: [] for s in SCENES}
    if os.path.exists(ALL25_CSV):
        for r in csv.DictReader(open(ALL25_CSV)):
            done[r["scene"]].append(float(r["miou_query"]))
    for scene in SCENES:
        while len(done[scene]) < reps:
            rep = len(done[scene])
            res = run_all25(scene, rep)
            if res is None:
                continue
            new = not os.path.exists(ALL25_CSV)
            with open(ALL25_CSV, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["scene", "rep", "miou_query", "f1_query"])
                w.writerow([scene, rep, f"{res[0]:.6f}", f"{res[1]:.6f}"])
            done[scene].append(res[0])
            print(f"[0 {time.time()-t0:5.0f}s] all25 {scene} rep{rep}: {res[0]:.4f}", flush=True)
    out = {}
    for scene in SCENES:
        v = np.array(done[scene][:reps])
        out[scene] = float(v.mean())
        print(f"  all25 {scene:15s}: {v.mean():.4f} +- {v.std():.4f}  {np.round(v,4).tolist()}",
              flush=True)
    return out


def load_references():
    """(uniform_best[scene], uniform_best_combo[scene]) from Phase A results."""
    done = load_done()
    uni_best, uni_combo = {}, {}
    offsets = {"-".join(map(str, range(o, N_FRAMES, 5))) for o in range(5)}
    for (scene, combo), (miou, _) in done.items():
        if combo in offsets and (scene not in uni_best or miou > uni_best[scene]):
            uni_best[scene] = miou
            uni_combo[scene] = tuple(int(x) for x in combo.split("-"))
    missing = [s for s in SCENES if s not in uni_best]
    if missing:
        raise RuntimeError(f"missing uniform references for {missing}; run oracle_search.py Phase A first")
    return uni_best, uni_combo


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
    ap.add_argument("--per-scene", type=int, default=30)
    ap.add_argument("--hit-margin-rel", type=float, default=0.10,
                    help="relative margin over uniform_best (all-25 mean must merely be reached)")
    ap.add_argument("--all25-reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    uni_best, uni_combo = load_references()
    all25 = all25_means(args.all25_reps, t0)
    summary = {}

    for scene in SCENES:
        done = load_done()
        # HIT: >=10% over uniform_best AND at least the (mean) all-25 level
        target = max((1.0 + args.hit_margin_rel) * uni_best[scene], all25[scene])
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
                      f"{target:.4f} (+10% over uniform_best AND >= all-25 mean)", flush=True)
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
