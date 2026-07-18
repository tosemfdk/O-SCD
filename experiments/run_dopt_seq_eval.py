# User-requested experiment (2026-07-19): seed the change scene with the first
# frame, then sequentially D-opt-pick the 4 frames that most constrain the
# current change scene (dopt_seq), K=5, all 10 Instance_1 scenes.
# Scored two ways per scene, then placed on the existing oracle-5 map:
#   online  - the selection pass itself (frames fused in greedy pick order)
#   replay  - clean chronological replay of the chosen 5 (same protocol as the
#             oracle map -> directly comparable to its numbers)
#
#   python experiments/run_dopt_seq_eval.py
from __future__ import annotations

import csv
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from oracle_search import SCENES, load_done, run_combo  # noqa: E402

import argparse

_ap = argparse.ArgumentParser()
_ap.add_argument("--criterion", default="dopt",
                 choices=["dopt", "trace_reduction", "fisher_ratio",
                          "candidate_only"])
_ap.add_argument("--weight", default="pose", choices=["pose", "current_map"])
ARGS = _ap.parse_args()
TAG = f"{ARGS.criterion}_{ARGS.weight}"
OUT_CSV = os.path.join(REPO, "experiments", f"dopt_seq_results_{TAG}.csv")


def eval_masks(scene: str, pred_dir: str):
    ev = subprocess.run(
        [sys.executable, os.path.join(REPO, "utils", "evaluate.py"),
         "--gt", os.path.join(REPO, "data/PASLCD/Instance_1", scene, "gt_mask") + "/",
         "--pred_binary", pred_dir + "/"],
        capture_output=True, text=True, env={**os.environ, "PYTHONPATH": ""},
        cwd=REPO)
    m = re.search(r"Mean IoU: ([0-9.]+).*Mean F1: ([0-9.]+)", ev.stdout, re.S)
    return (float(m.group(1)), float(m.group(2))) if m else None


def one_scene(scene: str, gpu: int, t0: float):
    out_dir = os.path.join(REPO, "output_subset", "dopt_seq", TAG, scene)
    env = {**os.environ, "PYTHONPATH": "", "TORCHDYNAMO_DISABLE": "1",
           "CUDA_VISIBLE_DEVICES": str(gpu)}
    r = subprocess.run(
        [sys.executable, os.path.join(REPO, "subset_oscd.py"),
         "-s", os.path.join(REPO, "data/PASLCD/Instance_1", scene) + "/",
         "-m", out_dir + "/", "--resolution", "4", "--test_hold", "5",
         "--frames_method", "dopt_seq", "--budget", "5",
         "--info_criterion", ARGS.criterion, "--info_weight", ARGS.weight],
        capture_output=True, text=True, cwd=REPO, env=env)
    if r.returncode != 0:
        print(f"[{scene}] SELECTION FAILED: {r.stderr[-300:]}", flush=True)
        return None
    sel = json.load(open(os.path.join(out_dir, "selection.json")))
    combo = tuple(sel["selected_indices"])
    order = sel["processing_order"]
    online = eval_masks(scene, os.path.join(out_dir, "renders", "query_mask"))
    shutil.rmtree(out_dir, ignore_errors=True)
    replay = run_combo(scene, combo, gpu=gpu)
    print(f"[{time.time()-t0:5.0f}s] {scene}: picks {order} | "
          f"online {online[0]:.4f} | replay {replay[0]:.4f}", flush=True)
    return scene, combo, order, online, replay


def main():
    gpus = list(range(8))
    pool: queue.Queue = queue.Queue()
    for g in gpus:
        pool.put(g)
    t0 = time.time()

    def worker(scene):
        g = pool.get()
        try:
            return one_scene(scene, g, t0)
        finally:
            pool.put(g)

    with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        results = [r for r in ex.map(worker, SCENES) if r]

    # ---- place on the oracle map -------------------------------------------
    done = load_done()
    vals = defaultdict(dict)
    for (sc, c), (miou, _f1) in done.items():
        if len(c.split("-")) == 5 and miou > vals[sc].get(c, -1):
            vals[sc][c] = miou
    all25 = defaultdict(list)
    for row in csv.DictReader(open(os.path.join(REPO, "experiments/all25_repeats.csv"))):
        all25[row["scene"]].append(float(row["miou_query"]))
    offs = {"-".join(map(str, range(o, 25, 5))) for o in range(5)}

    rows = []
    print(f"\n{'scene':14s} {'picks(greedy order)':22s} {'online':>7s} {'replay':>7s} "
          f"{'uni_mean':>8s} {'all25':>7s} {'oracle':>7s} {'pctile':>7s}")
    for scene, combo, order, online, replay in results:
        uni = [vals[scene][c] for c in offs]
        o5 = max(vals[scene].values())
        a25 = float(np.mean(all25[scene]))
        dist = np.array(sorted(vals[scene].values()))
        pct = float((dist < replay[0]).mean()) * 100
        print(f"{scene:14s} {str(order):22s} {online[0]:7.4f} {replay[0]:7.4f} "
              f"{np.mean(uni):8.4f} {a25:7.4f} {o5:7.4f} {pct:6.1f}%")
        rows.append({"scene": scene, "combo": "-".join(map(str, combo)),
                     "greedy_order": "-".join(map(str, order)),
                     "miou_online": online[0], "f1_online": online[1],
                     "miou_replay": replay[0], "f1_replay": replay[1],
                     "uniform_mean": float(np.mean(uni)), "uniform_best": max(uni),
                     "all25_mean": a25, "oracle5": o5, "map_percentile": pct})
    m = lambda k: float(np.mean([r[k] for r in rows]))
    print(f"{'MEAN':14s} {'':22s} {m('miou_online'):7.4f} {m('miou_replay'):7.4f} "
          f"{m('uniform_mean'):8.4f} {m('all25_mean'):7.4f} {m('oracle5'):7.4f} "
          f"{np.mean([r['map_percentile'] for r in rows]):6.1f}%")
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nsaved {OUT_CSV}")


if __name__ == "__main__":
    main()
