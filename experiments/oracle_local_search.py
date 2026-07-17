# Oracle-5 map via local search (user-directed design, 2026-07-17, rev 3).
#
# Per Instance_1 scene:
#   - prev_best = the scene's best mIoU over EVERYTHING already in the CSV
#     (frozen at start). HIT target = (1 + --hit-margin-rel) * prev_best,
#     i.e. find a 5-combo 10% better than the previous experiments' best.
#   - hill-climb: mutate the current center by 1-3 frame swaps; a combo that
#     beats the current pool's local best becomes the new center.
#   - stall restart: if the scene-global best has not improved for
#     --stall-restarts consecutive evals (default 50), jump to a NEW pool —
#     an unseen random combo overlapping the global best in <= 2 frames —
#     and hill-climb from there (stall counter resets).
#   - stop on HIT or after --per-scene evals (default 300).
# Rows are appended to experiments/oracle_search_results.csv with phase
# "local3" (hill-climb) / "restart3" (pool-jump combos). uniform / all-25
# references are loaded for reporting only, not for the HIT rule.
#
# Scenes run in PARALLEL, one worker thread per GPU (--gpus, default: all
# visible GPUs); each scene's search stays sequential on its own GPU.
#
#   python experiments/oracle_local_search.py --per-scene 300

from __future__ import annotations

import argparse
import csv
import os
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle_search import (CSV_PATH, INSTANCE, INSTANCE_SUFFIX,  # noqa: E402
                           N_FRAMES, K, REPO, SCENES,
                           load_done, record, run_combo)

ALL25_CSV = os.path.join(REPO, "experiments",
                         f"all25_repeats{INSTANCE_SUFFIX}.csv")
CSV_LOCK = threading.Lock()


def detect_gpus() -> list[int]:
    try:
        r = subprocess.run(["nvidia-smi", "--list-gpus"],
                           capture_output=True, text=True)
        n = len([ln for ln in r.stdout.splitlines() if ln.strip()])
    except FileNotFoundError:
        n = 0
    return list(range(n)) if n else [0]


def run_all25(scene: str, rep: int,
              gpu: int | None = None) -> tuple[float, float] | None:
    """One all-25 run (frames_method=all), evaluated on query_mask."""
    import re
    import shutil
    out_dir = os.path.join(REPO, "output_subset", "oracle", INSTANCE, scene,
                           f"all25_rep{rep}")
    src = os.path.join(REPO, "data", "PASLCD", INSTANCE, scene)
    env = {**os.environ, "PYTHONPATH": "", "TORCHDYNAMO_DISABLE": "1"}
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
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


def restart_center(best_combo: tuple[int, ...], scene_done: dict, rng,
                   max_overlap: int = 2) -> tuple[int, ...]:
    """Unseen random combo sharing <= max_overlap frames with the scene best."""
    while True:
        combo = tuple(sorted(rng.choice(N_FRAMES, K, replace=False).tolist()))
        if len(set(combo) & set(best_combo)) > max_overlap:
            continue
        if "-".join(map(str, combo)) not in scene_done:
            return combo


def ensure_scene_refs(scene: str, scene_done: dict, gpu: int,
                      all25_reps: int) -> None:
    """Make sure the scene has its references IN THIS CSV/instance: the 5
    stride-5 uniform offsets (search seeds) and `all25_reps` all-25 runs.
    No-op when they already exist (Instance_1); runs them on the scene's
    own GPU otherwise (fresh instance)."""
    for off in range(5):
        combo = tuple(range(off, N_FRAMES, 5))
        key = "-".join(map(str, combo))
        if key in scene_done:
            continue
        res = run_combo(scene, combo, gpu=gpu)
        if res is None:
            print(f"[{scene} gpu{gpu}] uniform off{off} FAILED", flush=True)
            continue
        with CSV_LOCK:
            record(scene, "uniform_off", combo, *res)
        scene_done[key] = res
        print(f"[{scene} gpu{gpu}] ref uniform off{off}: {res[0]:.4f}", flush=True)
    have = 0
    with CSV_LOCK:
        if os.path.exists(ALL25_CSV):
            have = sum(1 for r in csv.DictReader(open(ALL25_CSV))
                       if r["scene"] == scene)
    attempts = 0
    while have < all25_reps and attempts < all25_reps + 3:
        attempts += 1
        res = run_all25(scene, have, gpu=gpu)
        if res is None:
            continue
        with CSV_LOCK:
            new = not os.path.exists(ALL25_CSV)
            with open(ALL25_CSV, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["scene", "rep", "miou_query", "f1_query"])
                w.writerow([scene, have, f"{res[0]:.6f}", f"{res[1]:.6f}"])
        have += 1
        print(f"[{scene} gpu{gpu}] ref all25 rep{have-1}: {res[0]:.4f}", flush=True)


def search_scene(scene: str, scene_idx: int, args, gpu_pool: queue.Queue,
                 t0: float):
    """Sequential rev-3 search for one scene, pinned to one GPU."""
    rng = np.random.default_rng(args.seed * 1000 + scene_idx)
    gpu = gpu_pool.get()
    try:
        with CSV_LOCK:
            done = load_done()
        scene_done = {c: v for (sc, c), v in done.items() if sc == scene}
        ensure_scene_refs(scene, scene_done, gpu, args.all25_reps)
        if not scene_done:
            raise RuntimeError(f"{scene}: no results to seed from (refs failed?)")
        best_key = max(scene_done, key=lambda c: scene_done[c][0])
        prev_best = scene_done[best_key][0]
        best_combo = tuple(int(x) for x in best_key.split("-"))
        target = (1.0 + args.hit_margin_rel) * prev_best
        best = prev_best
        center, local_best = best_combo, best
        print(f"=== {scene} [gpu{gpu}]: prev_best {best_combo} {prev_best:.4f} "
              f"-> target {target:.4f} (+{args.hit_margin_rel:.0%})", flush=True)

        # Calibration: the CSV's prev_best may come from another machine/env.
        # Re-run it once here (logged as recheck3, not counted, not a target
        # change) so the machine shift is quantifiable in the report.
        if args.recheck_prev_best:
            res = run_combo(scene, best_combo, gpu=gpu)
            if res is not None:
                with CSV_LOCK:
                    record(scene, "recheck3", best_combo, *res)
                print(f"[{scene} gpu{gpu}] recheck prev_best {best_combo}: "
                      f"{res[0]:.4f} (CSV said {prev_best:.4f}, shift "
                      f"{res[0]-prev_best:+.4f})", flush=True)

        evals, stall, restarts, hit = 0, 0, 0, False
        while evals < args.per_scene:
            if stall >= args.stall_restarts:
                combo = restart_center(best_combo, scene_done, rng)
                phase = "restart3"
                restarts += 1
                stall = 0
                center, local_best = combo, -1.0
                print(f"[{scene} gpu{gpu}] RESTART #{restarts}: new pool {combo} "
                      f"(overlap<=2 with best {best_combo})", flush=True)
            else:
                combo, phase = None, "local3"
                for _ in range(200):  # dup-proposal guard
                    m = int(rng.choice([1, 2, 3], p=[0.5, 0.3, 0.2]))
                    cand = mutate(center, m, rng)
                    if "-".join(map(str, cand)) not in scene_done:
                        combo = cand
                        break
                if combo is None:  # neighbourhood exhausted -> force a jump
                    stall = args.stall_restarts
                    continue
            res = run_combo(scene, combo, gpu=gpu)
            evals += 1
            if res is None:
                stall += 1
                continue
            with CSV_LOCK:
                record(scene, phase, combo, *res)
            scene_done["-".join(map(str, combo))] = res
            if res[0] > local_best:
                local_best, center = res[0], combo
            marker = ""
            if res[0] > best:
                best, best_combo = res[0], combo
                stall = 0
                marker = " <- new best"
            else:
                stall += 1
            print(f"[{scene} gpu{gpu} {evals:3d}/{args.per_scene} "
                  f"{time.time()-t0:6.0f}s] {combo}: {res[0]:.4f} "
                  f"(best {best:.4f}, stall {stall}){marker}", flush=True)
            if res[0] >= target:
                print(f"HIT {scene}: combo {combo} mIoU {res[0]:.4f} >= "
                      f"{target:.4f} (prev_best {prev_best:.4f} +"
                      f"{args.hit_margin_rel:.0%})", flush=True)
                hit = True
                break
        print(f"SCENE DONE {scene}: best {best_combo} {best:.4f} "
              f"({best/prev_best-1:+.1%} vs prev_best), hit={hit}, "
              f"evals={evals}, restarts={restarts}", flush=True)
        return scene, (best_combo, best, prev_best, hit, evals, restarts)
    finally:
        gpu_pool.put(gpu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-scene", type=int, default=300)
    ap.add_argument("--hit-margin-rel", type=float, default=0.10,
                    help="relative margin over the scene's previous best (HIT rule)")
    ap.add_argument("--stall-restarts", type=int, default=50,
                    help="consecutive evals without a new scene best before a pool jump")
    ap.add_argument("--all25-reps", type=int, default=5)
    ap.add_argument("--recheck-prev-best", type=int, default=1,
                    help="1: re-run each scene's prev_best combo once on this "
                         "machine first (phase recheck3, not counted)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--gpus", type=str, default="",
                    help="comma-separated GPU ids (default: all visible)")
    ap.add_argument("--scenes", type=str, default="",
                    help="comma-separated scene subset (default: all)")
    args = ap.parse_args()

    gpus = ([int(g) for g in args.gpus.split(",") if g != ""]
            if args.gpus else detect_gpus())
    scenes = ([s for s in args.scenes.split(",") if s]
              if args.scenes else list(SCENES))
    t0 = time.time()
    print(f"\nrev3 search [{INSTANCE}]: {len(scenes)} scenes on {len(gpus)} "
          f"GPUs {gpus}, {args.per_scene} evals/scene, HIT = prev_best +"
          f"{args.hit_margin_rel:.0%}, pool jump after {args.stall_restarts} "
          f"stalled evals\n", flush=True)

    gpu_pool: queue.Queue = queue.Queue()
    for g in gpus:
        gpu_pool.put(g)
    summary = {}
    with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        futs = [ex.submit(search_scene, s, i, args, gpu_pool, t0)
                for i, s in enumerate(scenes)]
        for f in futs:
            scene, res = f.result()
            summary[scene] = res

    # references for the summary, straight from the CSVs (workers ensured them)
    done = load_done()
    offs = {"-".join(map(str, range(o, N_FRAMES, 5))) for o in range(5)}
    all25 = {}
    if os.path.exists(ALL25_CSV):
        acc = {}
        for r in csv.DictReader(open(ALL25_CSV)):
            acc.setdefault(r["scene"], []).append(float(r["miou_query"]))
        all25 = {s: float(np.mean(v)) for s, v in acc.items()}

    print(f"\n=== ORACLE-5 MAP rev3 ({INSTANCE}) ===", flush=True)
    for scene in scenes:
        bc, b, prev, hit, ev, rs = summary[scene]
        uni = [v[0] for (sc, c), v in done.items() if sc == scene and c in offs]
        extra = ""
        if scene in all25:
            extra += f", {b/all25[scene]-1:+.1%} vs all25"
        if uni:
            extra += f", {b/max(uni)-1:+.1%} vs uniform_best"
        print(f"  {scene:15s}: best5 {b:.4f} ({b/prev-1:+.1%} vs prev_best "
              f"{prev:.4f}{extra})  combo {bc}  "
              f"hit={hit} evals={ev} restarts={rs}", flush=True)
    print("ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
