# Cue-vs-consensus toxicity validation (Part 2 cycle 3 prep, 2026-07-20).
#
# Question: can a frame's SAM2 cue, compared against the cached all-25 global
# context R_global rendered at the SAME pose, separate toxic frames (large
# cue mass the all-view consensus does not support) from benign ones —
# WITHOUT GT? The fusion loss is up-only, so unsupported cue mass predicts
# irreversible damage.
#
# Scores per frame v (V = reference alpha visibility, M = cue, g = consensus):
#   tox_mass  = sum V * relu(M - g)          unsupported cue mass
#   tox_frac  = tox_mass / (sum V * M)       fraction of the cue unsupported
#   support   = sum V * M * g / (sum V * M)  cue precision vs consensus
#   cue_area  = sum V * M                    (Part 1 feature, for reference)
#
# Validated against the oracle map's per-frame marginal contribution
# (mean mIoU of 5-combos containing v minus those without — GT is used only
# to VALIDATE the signal here, never inside any selector) and against the
# known toxic frames from budgeted_view_findings §5.
#
#   python experiments/cue_toxicity_check.py
from __future__ import annotations

import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from oracle_search import INSTANCE, SCENES, load_done  # noqa: E402

GCTX_ROOT = os.path.join(REPO, "outputs", "change_nbv", "global_context")
OUT_CSV = os.path.join(REPO, "experiments", "cue_toxicity_scores.csv")
SIG05 = float(torch.sigmoid(torch.tensor(0.5)))  # cached masks store sigmoid(y); pipeline "change" is y > 0.5

# findings §5 known-toxic list (rev3-stable worst frames)
KNOWN_TOXIC = {"Porch": [0, 7], "Zen": [3, 4, 6], "Printing_area": [4]}


def find_gctx(scene: str, seed: int = 0) -> str:
    """Newest cached R_global for (scene, seed) regardless of code_commit —
    fine for post-hoc analysis (metadata records the regression check)."""
    best, best_t = None, -1.0
    for meta_path in glob.glob(os.path.join(GCTX_ROOT, scene, INSTANCE, "*",
                                            "metadata.json")):
        meta = json.load(open(meta_path))
        if meta.get("train_seed") != seed or not meta.get("regression_ok",
                                                          True):
            continue
        t = os.path.getmtime(meta_path)
        if t > best_t:
            best, best_t = os.path.dirname(meta_path), t
    if best is None:
        raise FileNotFoundError(f"no cached R_global for {scene} seed {seed}")
    return best


def dump_cues(scene: str, gpu: int = 0) -> dict:
    out_dir = os.path.join(REPO, "output_subset", "cue_dump", scene)
    cues_path = os.path.join(out_dir, "cues.pt")
    if not os.path.exists(cues_path):
        env = {**os.environ, "PYTHONPATH": "", "TORCHDYNAMO_DISABLE": "1",
               "CUDA_VISIBLE_DEVICES": str(gpu)}
        r = subprocess.run(
            [sys.executable, os.path.join(REPO, "subset_oscd.py"),
             "-s", os.path.join(REPO, "data", "PASLCD", INSTANCE, scene) + "/",
             "-m", out_dir + "/", "--resolution", "4", "--test_hold", "5",
             "--dump_cues"],
            capture_output=True, text=True, env=env, cwd=REPO)
        if r.returncode != 0:
            raise RuntimeError(f"cue dump failed {scene}: {r.stderr[-300:]}")
    return torch.load(cues_path, map_location="cpu", weights_only=False)


def marginals(scene: str) -> dict[int, float]:
    done = load_done()
    combos = {}
    for (sc, c), (miou, _f1) in done.items():
        if sc == scene and len(c.split("-")) == 5:
            combos[c] = max(miou, combos.get(c, -1))
    out = {}
    for f in range(25):
        w, wo = [], []
        for c, v in combos.items():
            (w if str(f) in c.split("-") else wo).append(v)
        if w and wo:
            out[f] = float(np.mean(w) - np.mean(wo))
    return out


def spearman(x, y) -> float:
    def rank(a):
        a = np.asarray(a, dtype=np.float64)
        order = a.argsort()
        r = np.empty_like(a)
        r[order] = np.arange(len(a), dtype=np.float64)
        for v in np.unique(a):
            m = a == v
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r
    rx, ry = rank(x), rank(y)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else 0.0


def main():
    rows = []
    pooled = defaultdict(list)  # score name -> list of (z_score, z_marginal)
    t0 = time.time()
    print(f"{'scene':14s} {'ρ(tox_mass)':>11s} {'ρ(tox_frac)':>11s} "
          f"{'ρ(support)':>10s} {'ρ(area)':>8s}  known-toxic ranks (1=worst)")
    for scene in SCENES:
        blob = dump_cues(scene)
        cues = blob["cues"].float()            # (25, 1, H, W) soft cue
        cues = cues.reshape(cues.shape[0], *cues.shape[-2:])
        gdir = find_gctx(scene)
        soft = torch.load(os.path.join(gdir, "all25_rendered_soft_masks.pt"),
                          map_location="cpu", weights_only=False)
        alpha = torch.load(os.path.join(gdir, "alpha_masks.pt"),
                           map_location="cpu", weights_only=False)
        assert blob["image_names"] == soft["image_names"], scene
        g_bin = (soft["soft_masks"] >= SIG05).float()   # pipeline's y>0.5
        V = (alpha["alpha_masks"] >= 0.5).float()

        n = cues.shape[0]
        scores = {"tox_mass": [], "tox_frac": [], "support": [],
                  "cue_area": []}
        for v in range(n):
            M, g, vis = cues[v], g_bin[v], V[v]
            cue_mass = float((vis * M).sum())
            tox = float((vis * torch.relu(M - g)).sum())
            scores["tox_mass"].append(tox)
            scores["tox_frac"].append(tox / max(cue_mass, 1e-6))
            scores["support"].append(
                float((vis * M * g).sum()) / max(cue_mass, 1e-6))
            scores["cue_area"].append(cue_mass)

        marg = marginals(scene)
        ids = sorted(marg)
        mvals = [marg[i] for i in ids]
        rho = {}
        for name, vals in scores.items():
            sel = [vals[i] for i in ids]
            rho[name] = spearman(sel, mvals)
            zs = (np.array(sel) - np.mean(sel)) / (np.std(sel) + 1e-12)
            zm = (np.array(mvals) - np.mean(mvals)) / (np.std(mvals) + 1e-12)
            pooled[name].extend(zip(zs, zm))

        # rank of the known toxic frames by tox_mass (1 = most toxic)
        order = np.argsort(scores["tox_mass"])[::-1]
        rank_of = {int(f): int(np.where(order == f)[0][0]) + 1
                   for f in range(n)}
        known = KNOWN_TOXIC.get(scene, [])
        known_str = ", ".join(f"{f}:{rank_of[f]}" for f in known) or "—"

        print(f"{scene:14s} {rho['tox_mass']:11.3f} {rho['tox_frac']:11.3f} "
              f"{rho['support']:10.3f} {rho['cue_area']:8.3f}  {known_str}")

        for v in range(n):
            rows.append({
                "scene": scene, "frame": v,
                "tox_mass": f"{scores['tox_mass'][v]:.2f}",
                "tox_frac": f"{scores['tox_frac'][v]:.4f}",
                "support": f"{scores['support'][v]:.4f}",
                "cue_area": f"{scores['cue_area'][v]:.2f}",
                "marginal": f"{marg.get(v, float('nan')):.5f}",
                "tox_rank": rank_of[v],
                "known_toxic": v in known,
            })

    print(f"\npooled (z-scored across scenes, n={len(pooled['tox_mass'])}):")
    for name, pairs in pooled.items():
        a = np.array(pairs)
        r_p = float(np.corrcoef(a[:, 0], a[:, 1])[0, 1])
        r_s = spearman(a[:, 0], a[:, 1])
        print(f"  {name:9s} pearson {r_p:+.3f}  spearman {r_s:+.3f}"
              f"   (Part 1 best GT-free signal: +0.173)")

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nsaved {OUT_CSV}  ({time.time() - t0:.0f}s)")
    shutil.rmtree(os.path.join(REPO, "output_subset", "cue_dump"),
                  ignore_errors=True)


if __name__ == "__main__":
    main()
