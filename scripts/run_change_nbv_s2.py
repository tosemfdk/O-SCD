# Gate S2 — score validity against conditional oracle marginals (spec §20).
#
# Scenes: Zen (mandatory watchlist), Porch, Garden (360). Instance_1.
# Contexts (deterministic uniform prefixes, fixed): S0 = {}, S1 = {12},
# S2 = {6, 18}.
#
# Phases (resumable, run in order):
#   info    - one `subset_oscd dopt_pose --budget 1` per scene: populates the
#             persistent b_v cache (all 25 frames) + alpha coverage metadata.
#   oracle  - conditional oracle marginals: mIoU_25(replay(S u {v})) for every
#             remaining candidate v, plus the context baselines. Replay =
#             oracle_search.run_combo (manual frames, all-query-view eval),
#             all within one batch, scene-parallel on all GPUs.
#   report  - offline: candidate scores for every method x context from the
#             cached diagonals -> Spearman rho / top-1 regret / top-3 capture
#             per (scene, context) -> outputs/change_nbv/s2/ + verdict print.
#
#   python scripts/run_change_nbv_s2.py --phase all
#
# This script STOPS after the report (spec: no auto-sweep past Gate S2).
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments"))

from oracle_search import run_combo  # noqa: E402  (Instance_1 default)

SCENES = ["Zen", "Porch", "Garden"]
CONTEXTS = {"S0": (), "S1": (12,), "S2": (6, 18)}
OUT = os.path.join(REPO, "outputs", "change_nbv", "s2")
CACHE_ROOT = os.path.join(REPO, "outputs", "change_nbv", "cache")
MARGINALS_CSV = os.path.join(OUT, "oracle_marginals.csv")
CSV_LOCK = threading.Lock()
N_FRAMES = 25


# ---------------------------------------------------------------- phase: info
def phase_info(gpus):
    def one(scene, gpu):
        out_dir = os.path.join(REPO, "output_subset", "s2_info", scene)
        env = {**os.environ, "PYTHONPATH": "", "TORCHDYNAMO_DISABLE": "1",
               "CUDA_VISIBLE_DEVICES": str(gpu)}
        r = subprocess.run(
            [sys.executable, os.path.join(REPO, "subset_oscd.py"),
             "-s", os.path.join(REPO, "data/PASLCD/Instance_1", scene) + "/",
             "-m", out_dir + "/", "--resolution", "4", "--test_hold", "5",
             "--frames_method", "dopt_pose", "--budget", "1",
             "--info_cache_root", CACHE_ROOT],
            capture_output=True, text=True, cwd=REPO, env=env)
        if r.returncode != 0:
            raise RuntimeError(f"info run failed for {scene}: {r.stderr[-400:]}")
        print(f"[info] {scene}: cache populated", flush=True)

    with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        futs = [ex.submit(one, s, gpus[i % len(gpus)])
                for i, s in enumerate(SCENES)]
        for f in futs:
            f.result()


def load_diagonals(scene):
    """frame_id -> (diagonal tensor, metadata) from the persistent cache."""
    paths = sorted(glob.glob(os.path.join(CACHE_ROOT, scene, "*", "frame_*.pt")))
    if len(paths) < N_FRAMES:
        raise RuntimeError(f"{scene}: cache incomplete ({len(paths)} frames); "
                           f"run --phase info first")
    out = {}
    for p in paths[-N_FRAMES:]:
        blob = torch.load(p, map_location="cpu", weights_only=False)
        out[int(blob["frame_id"])] = (blob["diagonal"], blob.get("metadata", {}))
    assert sorted(out) == list(range(N_FRAMES)), sorted(out)
    return out


# -------------------------------------------------------------- phase: oracle
def _record(scene, context, candidate, combo, miou, f1):
    with CSV_LOCK:
        new = not os.path.exists(MARGINALS_CSV)
        os.makedirs(OUT, exist_ok=True)
        with open(MARGINALS_CSV, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["scene", "context", "candidate", "combo",
                            "miou_query", "f1_query"])
            w.writerow([scene, context, candidate,
                        "-".join(map(str, combo)), f"{miou:.6f}", f"{f1:.6f}"])


def _done_keys():
    done = set()
    if os.path.exists(MARGINALS_CSV):
        for r in csv.DictReader(open(MARGINALS_CSV)):
            done.add((r["scene"], r["context"], int(r["candidate"])))
    return done


def phase_oracle(gpus):
    jobs = []  # (scene, ctx_name, candidate_id, combo) ; candidate -1 = baseline
    for scene in SCENES:
        for cname, ctx in CONTEXTS.items():
            if ctx:
                jobs.append((scene, cname, -1, tuple(sorted(ctx))))
            for v in range(N_FRAMES):
                if v in ctx:
                    continue
                jobs.append((scene, cname, v, tuple(sorted((*ctx, v)))))
    done = _done_keys()
    jobs = [j for j in jobs if (j[0], j[1], j[2]) not in done]
    print(f"[oracle] {len(jobs)} replay runs to do", flush=True)

    gpu_pool: queue.Queue = queue.Queue()
    for g in gpus:
        gpu_pool.put(g)
    t0 = time.time()

    def one(job):
        scene, cname, cand, combo = job
        gpu = gpu_pool.get()
        try:
            res = run_combo(scene, combo, gpu=gpu)
            if res is None:
                print(f"[oracle] FAILED {scene} {cname} v={cand}", flush=True)
                return
            _record(scene, cname, cand, combo, *res)
            print(f"[oracle {time.time()-t0:6.0f}s] {scene} {cname} "
                  f"v={cand:3d} {combo}: {res[0]:.4f}", flush=True)
        finally:
            gpu_pool.put(gpu)

    with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        list(ex.map(one, jobs))


# -------------------------------------------------------------- phase: report
def _load_cue_area():
    path = os.path.join(REPO, "experiments", "frame_features.csv")
    out = {}
    for r in csv.DictReader(open(path)):
        out[(r["scene"], int(r["frame"]))] = float(r["cue_area"])
    return out


def phase_report():
    from scipy.stats import spearmanr

    from view_selection.criteria import score_candidate
    from view_selection.information import derive_relative_lambda

    cue_area = _load_cue_area()
    marg = {}   # (scene, ctx) -> {cand: miou}
    base = {}   # (scene, ctx) -> baseline miou (0.0 for S0)
    for r in csv.DictReader(open(MARGINALS_CSV)):
        key = (r["scene"], r["context"])
        v = int(r["candidate"])
        if v == -1:
            base[key] = float(r["miou_query"])
        else:
            marg.setdefault(key, {})[v] = float(r["miou_query"])

    rows = []
    for scene in SCENES:
        diags = load_diagonals(scene)
        infos = {i: d for i, (d, _m) in diags.items()}
        alpha = {i: m.get("alpha_coverage", 0.0) for i, (_d, m) in diags.items()}
        lam = derive_relative_lambda(infos.values(), 1e-3, 1e-8)
        for cname, ctx in CONTEXTS.items():
            key = (scene, cname)
            if key not in marg:
                continue
            cands = sorted(marg[key])
            baseline = base.get(key, 0.0)
            oracle = {v: marg[key][v] - baseline for v in cands}
            prior = torch.full_like(infos[0], float(lam))
            for s in ctx:
                prior = prior + infos[s]

            methods = {
                "max_cue_area": {v: cue_area[(scene, v)] for v in cands},
                "max_alpha": {v: alpha[v] for v in cands},
                "candidate_only": {v: score_candidate("candidate_only", prior, infos[v]) for v in cands},
                "fisher_ratio": {v: score_candidate("fisher_ratio", prior, infos[v]) for v in cands},
                "trace_reduction": {v: score_candidate("trace_reduction", prior, infos[v]) for v in cands},
                "dopt_pose": {v: score_candidate("dopt", prior, infos[v]) for v in cands},
            }
            o = np.array([oracle[v] for v in cands])
            best_gain = o.max()
            for name, scores in methods.items():
                s = np.array([scores[v] for v in cands])
                rho = float(spearmanr(s, o).statistic)
                pick = cands[int(np.argmax(s))]
                top1_regret = float(best_gain - oracle[pick])
                top3 = set(np.array(cands)[np.argsort(-s)[:3]].tolist())
                otop3 = set(np.array(cands)[np.argsort(-o)[:3]].tolist())
                rows.append({
                    "scene": scene, "context": cname, "method": name,
                    "spearman": round(rho, 4),
                    "top1_regret": round(top1_regret, 4),
                    "top3_capture": len(top3 & otop3),
                    "pred_best": pick,
                    "oracle_best": cands[int(np.argmax(o))],
                })

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "score_validity.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ---- verdict summary ----------------------------------------------------
    print("\n=== Gate S2: Spearman rho (score vs conditional oracle marginal) ===")
    methods = sorted({r["method"] for r in rows})
    print(f"{'method':16s}", *(f"{s}/{c:2s}" for s in SCENES for c in CONTEXTS), "| median")
    summary = {}
    for m in methods:
        vals = [r["spearman"] for r in rows if r["method"] == m]
        cells = {(r["scene"], r["context"]): r["spearman"] for r in rows if r["method"] == m}
        med = float(np.median(vals))
        zen = float(np.median([r["spearman"] for r in rows
                               if r["method"] == m and r["scene"] == "Zen"]))
        reg = float(np.median([r["top1_regret"] for r in rows if r["method"] == m]))
        summary[m] = {"median_rho": med, "zen_rho": zen, "median_regret": reg}
        print(f"{m:16s}", *(f"{cells.get((s, c), float('nan')):+.2f}"
                            for s in SCENES for c in CONTEXTS), f"| {med:+.3f}")
    print("\nmedian top-1 regret / Zen median rho:")
    for m in methods:
        d = summary[m]
        print(f"  {m:16s} regret {d['median_regret']:.4f}  zen_rho {d['zen_rho']:+.3f}")

    d = summary["dopt_pose"]
    go = (d["median_rho"] > 0 and d["zen_rho"] > 0
          and d["median_regret"] <= summary["candidate_only"]["median_regret"] + 1e-9)
    print(f"\nVERDICT: {'GO_RECOMMENDED' if go else 'NO_GO'} "
          f"(dopt_pose median rho {d['median_rho']:+.3f}, zen {d['zen_rho']:+.3f}, "
          f"regret {d['median_regret']:.4f} vs candidate_only "
          f"{summary['candidate_only']['median_regret']:.4f})", flush=True)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump({"summary": summary, "go_recommended": bool(go)}, f, indent=2)
    print("Gate S2 report written; STOPPING here per spec (no auto sweep).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["info", "oracle", "report", "all"],
                    default="all")
    ap.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    args = ap.parse_args()
    gpus = [int(g) for g in args.gpus.split(",") if g != ""]
    if args.phase in ("info", "all"):
        phase_info(gpus)
    if args.phase in ("oracle", "all"):
        phase_oracle(gpus)
    if args.phase in ("report", "all"):
        phase_report()


if __name__ == "__main__":
    main()
