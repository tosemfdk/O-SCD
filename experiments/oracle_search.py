# Oracle-5 random search on Instance_1 (user-directed experiment, 2026-07-16).
#
# Phase A: the proper uniform@5 reference — all 5 stride-5 offset variants
#   ([o, o+5, o+10, o+15, o+20], o = 0..4) per scene, averaged.
# Phase B: random 5-subsets, scenes round-robin, one combo at a time. STOP as
#   soon as a combo beats its scene's uniform mean by >= --hit-margin-rel
#   (default 10% RELATIVE) on all-query-view mIoU, and report it.
#
# Every run goes through subset_oscd.py --frames_method manual (index list),
# query_mask evaluated at ALL 25 poses vs all 25 GT masks. Resumable: results
# CSV is the state.
#
#   python experiments/oracle_search.py --max-combos 300

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENES = ["Cantina", "Garden", "Lounge", "Lunch_room", "Meeting_room",
          "Playground", "Porch", "Pots", "Printing_area", "Zen"]
CSV_PATH = os.path.join(REPO, "experiments", "oracle_search_results.csv")
N_FRAMES, K = 25, 5


def run_combo(scene: str, combo: tuple[int, ...]) -> tuple[float, float] | None:
    tag = "c" + "-".join(map(str, combo))
    out_dir = os.path.join(REPO, "output_subset", "oracle", scene, tag)
    src = os.path.join(REPO, "data", "PASLCD", "Instance_1", scene)
    cmd = [sys.executable, os.path.join(REPO, "subset_oscd.py"),
           "-s", src + "/", "-m", out_dir + "/",
           "--resolution", "4", "--test_hold", "5",
           "--frames_method", "manual",
           "--frames_list", ",".join(map(str, combo))]
    env = {**os.environ, "PYTHONPATH": ""}
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=REPO)
    if r.returncode != 0:
        print(f"RUN FAILED {scene} {tag}: {r.stderr[-300:]}", flush=True)
        return None
    ev = subprocess.run(
        [sys.executable, os.path.join(REPO, "utils", "evaluate.py"),
         "--gt", os.path.join(src, "gt_mask") + "/",
         "--pred_binary", os.path.join(out_dir, "renders", "query_mask") + "/"],
        capture_output=True, text=True, env=env, cwd=REPO)
    m = re.search(r"Mean IoU: ([0-9.]+).*Mean F1: ([0-9.]+)", ev.stdout, re.S)
    shutil.rmtree(out_dir, ignore_errors=True)
    if not m:
        print(f"EVAL FAILED {scene} {tag}", flush=True)
        return None
    return float(m.group(1)), float(m.group(2))


def load_done() -> dict:
    done = {}
    if os.path.exists(CSV_PATH):
        for r in csv.DictReader(open(CSV_PATH)):
            done[(r["scene"], r["combo"])] = (float(r["miou_query"]), float(r["f1_query"]))
    return done


def record(scene: str, phase: str, combo: tuple[int, ...], miou: float, f1: float):
    new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["scene", "phase", "combo", "miou_query", "f1_query"])
        w.writerow([scene, phase, "-".join(map(str, combo)), f"{miou:.6f}", f"{f1:.6f}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-combos", type=int, default=300)
    ap.add_argument("--hit-margin-rel", type=float, default=0.10,
                    help="report when combo mIoU >= (1 + margin) * scene uniform mean")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    done = load_done()
    t0 = time.time()

    # ---- Phase A: stride-5 uniform offsets ---------------------------------
    for scene in SCENES:
        for off in range(5):
            combo = tuple(range(off, N_FRAMES, 5))
            key = (scene, "-".join(map(str, combo)))
            if key in done:
                continue
            res = run_combo(scene, combo)
            if res:
                record(scene, "uniform_off", combo, *res)
                done[key] = res
                print(f"[A {time.time()-t0:5.0f}s] {scene} off{off} {combo}: mIoU {res[0]:.4f}", flush=True)

    done = load_done()
    uni = {s: [v[0] for (sc, c), v in done.items() if sc == s
               and c in {"-".join(map(str, range(o, N_FRAMES, 5))) for o in range(5)}]
           for s in SCENES}
    print("\nuniform@5 stride-5 reference per scene (mean +- std over 5 offsets):", flush=True)
    for s in SCENES:
        print(f"  {s:15s}: {np.mean(uni[s]):.4f} +- {np.std(uni[s]):.4f}", flush=True)
    print(f"  Instance_1 mean: {np.mean([np.mean(uni[s]) for s in SCENES]):.4f}\n", flush=True)

    # ---- Phase B: random combos, round-robin, stop on hit -------------------
    rng = np.random.default_rng(args.seed)
    tried = 0
    while tried < args.max_combos:
        scene = SCENES[tried % len(SCENES)]
        while True:
            combo = tuple(sorted(rng.choice(N_FRAMES, K, replace=False).tolist()))
            if (scene, "-".join(map(str, combo))) not in done:
                break
        res = run_combo(scene, combo)
        tried += 1
        if res is None:
            continue
        record(scene, "random", combo, *res)
        done[(scene, "-".join(map(str, combo)))] = res
        ref = float(np.mean(uni[scene]))
        rel = res[0] / ref - 1.0
        print(f"[B {time.time()-t0:5.0f}s #{tried}] {scene} {combo}: "
              f"mIoU {res[0]:.4f} (uniform ref {ref:.4f}, {rel:+.1%})", flush=True)
        if rel >= args.hit_margin_rel:
            print(f"\nHIT: {scene} combo {combo} mIoU {res[0]:.4f} beats uniform "
                  f"mean {ref:.4f} by {rel:+.1%} (margin {args.hit_margin_rel:.0%})", flush=True)
            return 0
    print(f"\nNO HIT after {tried} random combos (margin {args.hit_margin_rel:.0%})", flush=True)
    return 2


if __name__ == "__main__":
    sys.exit(main())
