# Part 2 cycle 2 driver — global-local consensus keyframe selection.
#
# Stage 1: build + cache R_global per (scene, seed) via the EXACT recorded
#   all-25 path (subset_oscd.py --frames_method all --resolution 4
#   --test_hold 5, TORCHDYNAMO_DISABLE=1) with --save_change_model, then
#   validate the checkpoint's all-25 query mIoU against all25_repeats.csv
#   (spec test 3; GT touches the cache VALIDATION only, never selection).
# Stage 2: per (method, scene, seed) run the selection (--selection_only),
#   then score the chosen 5 with the standard clean replay through
#   oracle_search.run_combo — the same manual path as the oracle map, which
#   makes spec test 10 (selector replay == manual replay) hold by construction.
#
# Baselines rerun in the SAME batch (findings §1 noise rules): uniform-5 as
# all 5 stride-5 offsets per seed, and the frozen 4th selector dopt_seq
# (dopt_dir/pose) per seed.
#
# Pilot default: failure scenes Cantina/Porch/Meeting_room + Zen + Playground,
# seeds 0/1/2. Resumable — the results CSV is the state. This driver never
# starts the 10-scene full sweep on its own.
#
#   python experiments/run_keyframe_gl_eval.py                 # pilot
#   python experiments/run_keyframe_gl_eval.py --stage 1       # caches only
from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from oracle_search import INSTANCE, run_combo  # noqa: E402

PILOT_SCENES = ["Cantina", "Porch", "Meeting_room", "Zen", "Playground"]
KF_METHODS = ["kf_g_dir", "kf_l_dir_gseed", "kf_gu_dir", "kf_gl_dir",
              "kf_glu_dir", "kf_glu_nodir"]
ALL_METHODS = ["uniform", "dopt_seq_dir"] + KF_METHODS

GCTX_ROOT = os.path.join(REPO, "outputs", "change_nbv", "global_context")
RESULTS_CSV = os.path.join(REPO, "experiments", "keyframe_gl_results.csv")
ALL25_CSV = os.path.join(REPO, "experiments", "keyframe_gl_all25.csv")
ALL25_REF_CSV = os.path.join(REPO, "experiments", "all25_repeats.csv")
ALL25_TOLERANCE = 0.02  # same-machine warn threshold for the seed-0 build

CSV_FIELDS = ["scene", "method", "seed", "combo", "greedy_order",
              "miou_replay", "f1_replay", "first_frame_id", "frame0_selected",
              "frame0_global_rank", "selection_seconds", "replay_seconds",
              "gctx_hash", "gctx_build_seconds", "manifest_path"]

_csv_lock = threading.Lock()
_print_lock = threading.Lock()
T0 = time.time()


def log(msg: str):
    with _print_lock:
        print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def _env(gpu: int | None):
    env = {**os.environ, "PYTHONPATH": "", "TORCHDYNAMO_DISABLE": "1"}
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def _eval_masks(scene: str, pred_dir: str, gpu: int | None):
    src = os.path.join(REPO, "data", "PASLCD", INSTANCE, scene)
    ev = subprocess.run(
        [sys.executable, os.path.join(REPO, "utils", "evaluate.py"),
         "--gt", os.path.join(src, "gt_mask") + "/",
         "--pred_binary", pred_dir + "/"],
        capture_output=True, text=True, env=_env(gpu), cwd=REPO)
    m = re.search(r"Mean IoU: ([0-9.]+).*Mean F1: ([0-9.]+)", ev.stdout, re.S)
    return (float(m.group(1)), float(m.group(2))) if m else None


def _frame_names(scene: str) -> list[str]:
    """Ordered inference frame names exactly as subset_oscd builds all_views
    (ImageDataset: sorted filenames, name = basename before the first dot)."""
    d = os.path.join(REPO, "data", "PASLCD", INSTANCE, scene,
                     "inference_scene", "images")
    return [f.split(".")[0] for f in sorted(os.listdir(d))
            if not f.startswith(".")]


def _ref_all25_mean(scene: str) -> float | None:
    if not os.path.exists(ALL25_REF_CSV):
        return None
    vals = [float(r["miou_query"]) for r in csv.DictReader(open(ALL25_REF_CSV))
            if r["scene"] == scene]
    return sum(vals) / len(vals) if vals else None


# --------------------------------------------------------------- Stage 1

def ensure_global_context(scene: str, seed: int, gpu: int) -> str:
    """Build (or reuse) the cached R_global for (scene, seed). Returns the
    cache dir. The build IS a standard all-25 run; its query evaluation is
    recorded as the same-batch all-25 baseline for this seed."""
    from view_selection.global_context import (file_sha256,
                                               find_cached_context,
                                               global_context_key,
                                               write_metadata)

    src = os.path.join(REPO, "data", "PASLCD", INSTANCE, scene)
    ref_ply = os.path.join(src, "reference_reconstruction", "point_cloud",
                           "iteration_30000", "point_cloud.ply")
    key = global_context_key(scene, INSTANCE, file_sha256(ref_ply),
                             _frame_names(scene), seed, 4, 0.5, REPO)
    hit = find_cached_context(GCTX_ROOT, scene, INSTANCE, key)
    if hit is not None:
        log(f"gctx {scene} seed{seed}: cache hit {os.path.basename(hit)}")
        return hit

    out_dir = os.path.join(REPO, "output_subset", "keyframe_gl", "gctx",
                           f"{scene}_s{seed}")
    shutil.rmtree(out_dir, ignore_errors=True)
    cmd = [sys.executable, os.path.join(REPO, "subset_oscd.py"),
           "-s", src + "/", "-m", out_dir + "/",
           "--resolution", "4", "--test_hold", "5",
           "--frames_method", "all", "--save_change_model"]
    if seed:
        cmd += ["--train_seed", str(seed)]
    t_build = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True,
                       env=_env(gpu), cwd=REPO)
    if r.returncode != 0:
        raise RuntimeError(f"gctx build failed {scene} seed{seed}: "
                           f"{r.stderr[-500:]}")
    build_seconds = time.time() - t_build

    # spec test 3: same checkpoint, all-25 query eval, compare with the
    # recorded all-25 numbers (strict check only for seed 0 — other seeds are
    # NEW same-batch replicates and are recorded as such)
    res = _eval_masks(scene, os.path.join(out_dir, "renders", "query_mask"),
                      gpu)
    if res is None:
        raise RuntimeError(f"gctx eval failed {scene} seed{seed}")
    miou, f1 = res
    ref = _ref_all25_mean(scene)
    regression_ok = True
    if seed == 0 and ref is not None and abs(miou - ref) > ALL25_TOLERANCE:
        regression_ok = False
        log(f"WARNING gctx {scene} seed0 all-25 mIoU {miou:.4f} deviates from "
            f"recorded {ref:.4f} by more than {ALL25_TOLERANCE}")

    from view_selection.global_context import global_context_dir
    cache_dir = global_context_dir(GCTX_ROOT, scene, INSTANCE, key)
    os.makedirs(cache_dir, exist_ok=True)
    shutil.move(os.path.join(out_dir, "r_change.ply"),
                os.path.join(cache_dir, "r_global.ply"))
    for pid in ("r_change.ply.pid.pt",):
        p = os.path.join(out_dir, pid)
        if os.path.exists(p):
            shutil.move(p, os.path.join(cache_dir, "r_global.ply.pid.pt"))
    shutil.move(os.path.join(out_dir, "all25_rendered_soft_masks.pt"),
                os.path.join(cache_dir, "all25_rendered_soft_masks.pt"))
    shutil.move(os.path.join(out_dir, "alpha_masks.pt"),
                os.path.join(cache_dir, "alpha_masks.pt"))
    write_metadata(cache_dir, key, {
        "build_seconds": round(build_seconds, 1),
        "all25_query_miou": miou, "all25_query_f1": f1,
        "all25_recorded_mean": ref, "regression_ok": regression_ok,
        "builder": "subset_oscd.py --frames_method all --resolution 4 "
                   "--test_hold 5 --save_change_model",
        "train_seed": seed,
    })
    shutil.rmtree(out_dir, ignore_errors=True)

    with _csv_lock:
        new = not os.path.exists(ALL25_CSV)
        with open(ALL25_CSV, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["scene", "seed", "miou_query", "f1_query",
                            "recorded_mean", "regression_ok"])
            w.writerow([scene, seed, f"{miou:.6f}", f"{f1:.6f}",
                        f"{ref:.6f}" if ref is not None else "",
                        regression_ok])
    log(f"gctx {scene} seed{seed}: built in {build_seconds:.0f}s, "
        f"all-25 mIoU {miou:.4f} (recorded {ref})")
    return cache_dir


# --------------------------------------------------------------- Stage 2

def _append_row(row: dict):
    with _csv_lock:
        new = not os.path.exists(RESULTS_CSV)
        with open(RESULTS_CSV, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)


def _done_keys() -> set[tuple]:
    if not os.path.exists(RESULTS_CSV):
        return set()
    return {(r["scene"], r["method"], int(r["seed"]), r["combo"])
            for r in csv.DictReader(open(RESULTS_CSV))}


def run_uniform(scene: str, seed: int, gpu: int, done: set):
    for offset in range(5):
        combo = tuple(range(offset, 25, 5))
        combo_s = "-".join(map(str, combo))
        if (scene, "uniform", seed, combo_s) in done:
            continue
        t = time.time()
        res = run_combo(scene, combo, gpu=gpu, train_seed=seed,
                        tag_suffix="uniform")
        if res is None:
            log(f"uniform FAILED {scene} seed{seed} o{offset}")
            continue
        _append_row({"scene": scene, "method": "uniform", "seed": seed,
                     "combo": combo_s, "greedy_order": combo_s,
                     "miou_replay": f"{res[0]:.6f}",
                     "f1_replay": f"{res[1]:.6f}",
                     "first_frame_id": combo[0], "frame0_selected": 0 in combo,
                     "frame0_global_rank": "", "selection_seconds": 0,
                     "replay_seconds": round(time.time() - t, 1),
                     "gctx_hash": "", "gctx_build_seconds": "",
                     "manifest_path": ""})
        log(f"uniform {scene} seed{seed} o{offset}: {res[0]:.4f}")


def run_dopt_seq(scene: str, seed: int, gpu: int, done: set):
    if any(k[:3] == (scene, "dopt_seq_dir", seed) for k in done):
        return
    src = os.path.join(REPO, "data", "PASLCD", INSTANCE, scene)
    out_dir = os.path.join(REPO, "output_subset", "keyframe_gl",
                           "dopt_seq_dir", f"{scene}_s{seed}")
    shutil.rmtree(out_dir, ignore_errors=True)
    cmd = [sys.executable, os.path.join(REPO, "subset_oscd.py"),
           "-s", src + "/", "-m", out_dir + "/", "--resolution", "4",
           "--test_hold", "5", "--frames_method", "dopt_seq", "--budget", "5",
           "--info_criterion", "dopt_dir", "--info_weight", "pose"]
    if seed:
        cmd += ["--train_seed", str(seed)]
    t_sel = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, env=_env(gpu),
                       cwd=REPO)
    if r.returncode != 0:
        log(f"dopt_seq FAILED {scene} seed{seed}: {r.stderr[-300:]}")
        return
    sel = json.load(open(os.path.join(out_dir, "selection.json")))
    combo = tuple(sel["selected_indices"])
    order = "-".join(map(str, sel["processing_order"]))
    sel_seconds = time.time() - t_sel
    shutil.rmtree(out_dir, ignore_errors=True)
    t_rep = time.time()
    res = run_combo(scene, combo, gpu=gpu, train_seed=seed,
                    tag_suffix="dopt_seq_dir")
    if res is None:
        log(f"dopt_seq replay FAILED {scene} seed{seed}")
        return
    _append_row({"scene": scene, "method": "dopt_seq_dir", "seed": seed,
                 "combo": "-".join(map(str, combo)), "greedy_order": order,
                 "miou_replay": f"{res[0]:.6f}", "f1_replay": f"{res[1]:.6f}",
                 "first_frame_id": sel["processing_order"][0],
                 "frame0_selected": 0 in combo, "frame0_global_rank": "",
                 "selection_seconds": round(sel_seconds, 1),
                 "replay_seconds": round(time.time() - t_rep, 1),
                 "gctx_hash": "", "gctx_build_seconds": "",
                 "manifest_path": ""})
    log(f"dopt_seq_dir {scene} seed{seed}: picks {order} -> {res[0]:.4f}")


def run_kf(method: str, scene: str, seed: int, gpu: int, done: set):
    if any(k[:3] == (scene, method, seed) for k in done):
        return
    src = os.path.join(REPO, "data", "PASLCD", INSTANCE, scene)
    out_dir = os.path.join(REPO, "output_subset", "keyframe_gl", method,
                           f"{scene}_s{seed}")
    shutil.rmtree(out_dir, ignore_errors=True)
    cmd = [sys.executable, os.path.join(REPO, "subset_oscd.py"),
           "-s", src + "/", "-m", out_dir + "/", "--resolution", "4",
           "--test_hold", "5", "--frames_method", method, "--budget", "5",
           "--selection_only",
           "--global_context_root", GCTX_ROOT]
    if seed:
        cmd += ["--train_seed", str(seed)]
    t_sel = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, env=_env(gpu),
                       cwd=REPO)
    if r.returncode != 0:
        log(f"{method} FAILED {scene} seed{seed}: {r.stderr[-400:]}")
        return
    manifest = json.load(open(os.path.join(out_dir,
                                           "selection_manifest.json")))
    combo = tuple(manifest["selected_frame_ids"])
    sel_seconds = time.time() - t_sel
    manifest_path = os.path.join(manifest["selection_output_root"],
                                 "selection_manifest.json")
    shutil.rmtree(out_dir, ignore_errors=True)
    t_rep = time.time()
    res = run_combo(scene, combo, gpu=gpu, train_seed=seed,
                    tag_suffix=method)
    if res is None:
        log(f"{method} replay FAILED {scene} seed{seed}")
        return
    gctx_meta_dir = os.path.dirname(manifest["global_context_checkpoint"])
    gctx_build = ""
    meta_path = os.path.join(gctx_meta_dir, "metadata.json")
    if os.path.exists(meta_path):
        gctx_build = json.load(open(meta_path)).get("build_seconds", "")
    _append_row({"scene": scene, "method": method, "seed": seed,
                 "combo": "-".join(map(str, combo)),
                 "greedy_order": "-".join(map(str, manifest["greedy_order"])),
                 "miou_replay": f"{res[0]:.6f}", "f1_replay": f"{res[1]:.6f}",
                 "first_frame_id": manifest["first_frame_id"],
                 "frame0_selected": manifest["frame0_selected"],
                 "frame0_global_rank": manifest["frame0_global_rank"],
                 "selection_seconds": round(sel_seconds, 1),
                 "replay_seconds": round(time.time() - t_rep, 1),
                 "gctx_hash": manifest["global_context_checkpoint_hash"],
                 "gctx_build_seconds": gctx_build,
                 "manifest_path": os.path.relpath(manifest_path, REPO)})
    log(f"{method} {scene} seed{seed}: "
        f"picks {manifest['greedy_order']} -> {res[0]:.4f} "
        f"(first={manifest['first_frame_id']}, "
        f"sel {sel_seconds:.0f}s)")


# --------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=PILOT_SCENES)
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS,
                    choices=ALL_METHODS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--stage", choices=["all", "1", "2"], default="all")
    args = ap.parse_args()

    pool: queue.Queue = queue.Queue()
    for g in range(args.gpus):
        pool.put(g)

    def with_gpu(fn, *a):
        g = pool.get()
        try:
            return fn(*a, g)
        except Exception as e:  # noqa: BLE001 — keep the sweep alive
            log(f"ERROR {fn.__name__}{a}: {e}")
            return None
        finally:
            pool.put(g)

    from concurrent.futures import ThreadPoolExecutor

    needs_gctx = any(m in KF_METHODS for m in args.methods)
    if args.stage in ("all", "1") and needs_gctx:
        log(f"Stage 1: R_global caches for {len(args.scenes)} scenes x "
            f"{len(args.seeds)} seeds")
        with ThreadPoolExecutor(max_workers=args.gpus) as ex:
            futs = [ex.submit(with_gpu, ensure_global_context, sc, sd)
                    for sc in args.scenes for sd in args.seeds]
            for f in futs:
                f.result()

    if args.stage in ("all", "2"):
        done = _done_keys()
        jobs = []  # every job fn has the (scene, seed, gpu) signature
        for sc in args.scenes:
            for sd in args.seeds:
                for m in args.methods:
                    if m == "uniform":
                        fn = lambda s_, d_, g_: run_uniform(s_, d_, g_, done)  # noqa: E731
                    elif m == "dopt_seq_dir":
                        fn = lambda s_, d_, g_: run_dopt_seq(s_, d_, g_, done)  # noqa: E731
                    else:
                        fn = (lambda s_, d_, g_, _m=m:
                              run_kf(_m, s_, d_, g_, done))
                    jobs.append((fn, sc, sd))
        log(f"Stage 2: {len(jobs)} jobs on {args.gpus} GPUs")
        with ThreadPoolExecutor(max_workers=args.gpus) as ex:
            futs = [ex.submit(with_gpu, fn, sc, sd) for fn, sc, sd in jobs]
            for f in futs:
                f.result()
        log(f"done — results in {os.path.relpath(RESULTS_CSV, REPO)}")


if __name__ == "__main__":
    main()
