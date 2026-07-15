# Stage-14 real-scene validation: budget-K view selection from a REAL camera
# pool (reference_reconstruction/cameras.json) on a real PASLCD reconstruction.
#
# For each sampled target Gaussian, every method starts from the same seed view
# and greedily spends K more views from the pool; after each pick the TRUE
# information of the picked view (FD Jacobian, cached) is committed, so the
# uncertainty trajectory logdet(H) / tr(Sigma) / lmax(Sigma) measures what the
# selection actually bought, independent of how it was chosen.
#
# Methods: uniform (index-spaced), random (seeded), max_resp (greedy
# responsibility), proxy (greedy geometry-FIM proxy), exact (greedy D-optimal,
# pure d_gain). Also logs proxy-vs-exact ranking agreement per round.
#
#   python experiments/target_nbv_pool_eval.py \
#       --scene data/PASLCD/Instance_1/Garden --pool-size 25 \
#       --num-targets 8 --budget 5 --out experiments/pool_eval_garden

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy.stats import kendalltau

from scene import GaussianModel
from scene.cameras import MiniCam
from utils.graphics_utils import focal2fov, getProjectionMatrix

from target_nbv.candidates import rotation_matrix_to_quaternion
from target_nbv.config import TargetNBVConfig
from target_nbv.info_builder import view_information
from target_nbv.io_utils import default_pipe
from target_nbv.jacobian import compute_target_jacobian
from target_nbv.scorers.exact import TargetPOptimalScorer
from target_nbv.scorers.numerics import (logdet_via_slogdet, max_eig_of_inverse,
                                         trace_of_inverse)
from target_nbv.scorers.proxy import GeometryProxyScorer
from target_nbv.types import (TargetInformationState, TargetParameterSpec,
                              CandidateCamera)
from target_nbv.visibility import ColorProbeVisibilityBackend, project_gaussian_ellipse

SPEC = TargetParameterSpec()


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="data/PASLCD/Instance_1/Garden")
    p.add_argument("--pool-size", type=int, default=25)
    p.add_argument("--num-targets", type=int, default=8)
    p.add_argument("--budget", type=int, default=5)
    p.add_argument("--render-divisor", type=float, default=8.0,
                   help="image downscale vs original capture resolution")
    p.add_argument("--random-seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--target-seed", type=int, default=7)
    p.add_argument("--min-opacity", type=float, default=0.9)
    p.add_argument("--out", default="experiments/pool_eval_garden")
    return p.parse_args(argv)


# --- pool loading -------------------------------------------------------------

def load_pool(scene_dir: str, pool_size: int, divisor: float) -> list[CandidateCamera]:
    with open(os.path.join(scene_dir, "reference_reconstruction", "cameras.json")) as f:
        entries = json.load(f)
    entries.sort(key=lambda e: e["img_name"])
    idx = sorted(set(np.linspace(0, len(entries) - 1, pool_size).round().astype(int)))
    pool = []
    for i, j in enumerate(idx):
        e = entries[j]
        W, H = int(round(e["width"] / divisor)), int(round(e["height"] / divisor))
        fovx = focal2fov(e["fx"], e["width"])   # FoV is resolution-invariant
        fovy = focal2fov(e["fy"], e["height"])
        # cameras.json: position = camera center, rotation = COLMAP c2w
        # (utils/camera_utils.py:camera_to_JSON)
        R_c2w = np.array(e["rotation"], dtype=np.float64)
        pos = np.array(e["position"], dtype=np.float64)
        W2C = np.eye(4, dtype=np.float32)
        W2C[:3, :3] = R_c2w.T.astype(np.float32)
        W2C[:3, 3] = (-R_c2w.T @ pos).astype(np.float32)
        world_view = torch.tensor(W2C).transpose(0, 1).cuda()
        proj = getProjectionMatrix(znear=0.01, zfar=100.0,
                                   fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
        full = world_view.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
        cand = CandidateCamera(
            cand_id=i, position=pos,
            wxyz=rotation_matrix_to_quaternion(R_c2w @ np.diag([1.0, -1.0, -1.0])),
            fovx=fovx, fovy=fovy, width=W, height=H,
            meta={"img_name": e["img_name"], "json_index": int(j)})
        cand.minicam = MiniCam(W, H, fovy, fovx, 0.01, 100.0, world_view, full)
        pool.append(cand)
    return pool


# --- target sampling (analytic only, cheap) ------------------------------------

def sample_targets(model, pool, n_targets: int, rng: np.random.Generator,
                   min_opacity: float) -> list[int]:
    opac = model.get_opacity.squeeze(1)
    rows = torch.nonzero(opac > min_opacity).squeeze(1).cpu().numpy()
    rng.shuffle(rows)
    render_w = pool[0].width
    chosen = []
    for row in rows[:3000]:
        in_view, good_radius = 0, 0
        for cand in pool:
            center, radii, _, bbox = project_gaussian_ellipse(model, int(row), cand.minicam)
            if center is None or bbox is None:
                continue
            in_view += 1
            if 3.0 <= float(radii[1]) <= 0.3 * render_w:
                good_radius += 1
        if in_view >= 0.7 * len(pool) and good_radius >= 0.5 * len(pool):
            chosen.append(int(row))
            if len(chosen) == n_targets:
                break
    return chosen  # analytic pre-filter only; caller re-checks real visibility


# --- cached per-(target, view) quantities --------------------------------------

class ViewCache:
    """Visibility and view-information matrices are selection-independent;
    compute each (target, view) pair once and share across all methods."""

    def __init__(self, model, pool, cfg, pipe, background):
        self.model, self.pool, self.cfg = model, pool, cfg
        self.pipe, self.background = pipe, background
        self.backend = ColorProbeVisibilityBackend(cfg)
        self.vis, self.H, self.timing = {}, {}, {"vis": [], "jac": []}

    def visibility(self, row: int, i: int):
        key = (row, i)
        if key not in self.vis:
            t0 = time.perf_counter()
            self.vis[key] = self.backend.evaluate(
                self.model, self.pool[i].minicam, row, self.pipe, self.background)
            self.timing["vis"].append(time.perf_counter() - t0)
        return self.vis[key]

    def view_H(self, row: int, i: int):
        """(6,6) float64 information of view i about target row, or None."""
        key = (row, i)
        if key not in self.H:
            if not self.visibility(row, i).valid:
                self.H[key] = None
            else:
                t0 = time.perf_counter()
                res = compute_target_jacobian(self.model, self.pool[i].minicam,
                                              row, SPEC, self.pipe,
                                              self.background, self.cfg)
                self.timing["jac"].append(time.perf_counter() - t0)
                self.H[key] = view_information(res.J, res.w) if res.valid else None
        return self.H[key]


# --- information-state bookkeeping ---------------------------------------------

def fresh_state(row: int, cfg) -> TargetInformationState:
    return TargetInformationState(
        target_pid=row, spec=SPEC, H_data=np.zeros((6, 6)),
        absolute_damping=cfg.information.absolute_damping,
        relative_damping=cfg.information.relative_damping)


def commit(state: TargetInformationState, H_view, view_id: str) -> None:
    if H_view is not None:
        state.H_data = 0.5 * ((state.H_data + H_view) + (state.H_data + H_view).T)
    state.observed_view_ids.append(view_id)
    state.version += 1


def metrics(state: TargetInformationState) -> dict:
    H = state.H_prior()
    ld, _ = logdet_via_slogdet(H, 1e-9, 1e-2)
    return {"logdet": ld,
            "trace_sigma": trace_of_inverse(H, 1e-9, 1e-2),
            "lmax_sigma": max_eig_of_inverse(H)}


# --- selection methods ----------------------------------------------------------

def spaced_indices(rest: list[int], k: int) -> list[int]:
    """k index-spaced picks from the ordered remaining pool (capture order)."""
    picks, used = [], set()
    for f in np.linspace(0, len(rest) - 1, k):
        j = int(round(f))
        while j in used:  # rounding collision -> next unused
            j += 1
        used.add(j)
        picks.append(rest[j])
    return picks


def run_method(method: str, row: int, seed_view: int, pool, cache: ViewCache,
               cfg, budget: int, rng: np.random.Generator | None,
               scorer: TargetPOptimalScorer, agreement: list) -> list[dict]:
    state = fresh_state(row, cfg)
    commit(state, cache.view_H(row, seed_view), f"seed{seed_view}")
    rows_out = [{"k": 0, "view": seed_view, "valid": True,
                 "select_s": 0.0, **metrics(state)}]

    remaining = [c.cand_id for c in pool if c.cand_id != seed_view]
    mu = cache.model._xyz[row].detach().cpu().numpy()

    static_order = None
    if method == "uniform":
        static_order = spaced_indices(remaining, budget)
    elif method == "random":
        static_order = list(rng.permutation(remaining)[:budget])
    elif method == "max_resp":
        static_order = sorted(
            remaining,
            key=lambda i: -cache.visibility(row, i).responsibility_sum)[:budget]

    for k in range(1, budget + 1):
        t0 = time.perf_counter()
        if static_order is not None:
            pick = int(static_order[k - 1])
        else:
            cands = [pool[i] for i in remaining]
            viss = [cache.visibility(row, i) for i in remaining]
            proxy_ranked = GeometryProxyScorer(cfg).rank(state, cands, viss, mu)
            valid_proxy = [s for s in proxy_ranked if s.valid]
            if method == "proxy":
                if not valid_proxy:
                    break
                pick = valid_proxy[0].candidate.cand_id
            else:  # exact greedy D-opt
                pq = scorer.prior_quantities(state.H_prior())
                exact_gain = {}
                for i in remaining:
                    H_v = cache.view_H(row, i)
                    if H_v is None:
                        continue
                    gains, reason, _ = scorer.score_delta(state.H_prior(), pq, H_v, 0.0)
                    if gains is not None:
                        exact_gain[i] = gains["d_gain"]
                if not exact_gain:
                    break
                pick = max(exact_gain, key=exact_gain.get)
                both = [i for i in exact_gain
                        if any(s.candidate.cand_id == i for s in valid_proxy)]
                if len(both) >= 3:
                    p_scores = {s.candidate.cand_id: s.proxy_score for s in valid_proxy}
                    tau = kendalltau([p_scores[i] for i in both],
                                     [exact_gain[i] for i in both]).statistic
                    top1_proxy = max(both, key=lambda i: p_scores[i])
                    agreement.append({"k": k, "tau": float(tau),
                                      "top1_match": bool(top1_proxy == pick)})
        select_s = time.perf_counter() - t0

        H_v = cache.view_H(row, pick)
        commit(state, H_v, f"v{pick}")
        remaining.remove(pick)
        rows_out.append({"k": k, "view": pick, "valid": H_v is not None,
                         "select_s": select_s, **metrics(state)})
    return rows_out


# --- main -----------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    cfg = TargetNBVConfig()
    cfg.scoring.movement_weight = 0.0   # pool selection: no robot motion here
    cfg.scoring.e_weight = 0.0          # exact greedy = pure D-optimality
    cfg.validate()

    ply = os.path.join(args.scene, "reference_reconstruction", "point_cloud",
                       "iteration_30000", "point_cloud.ply")
    model = GaussianModel(3)
    model.load_ply(ply)
    print(f"model: {model.get_xyz.shape[0]} gaussians")
    pool = load_pool(args.scene, args.pool_size, args.render_divisor)
    print(f"pool: {len(pool)} cameras at {pool[0].width}x{pool[0].height}")

    pipe = default_pipe()
    background = torch.zeros(3, device="cuda")
    rng = np.random.default_rng(args.target_seed)
    candidates = sample_targets(model, pool, args.num_targets + 8, rng, args.min_opacity)
    print(f"target candidates (rows): {candidates}")

    cache = ViewCache(model, pool, cfg, pipe, background)
    scorer = TargetPOptimalScorer(cfg)
    methods = ([("uniform", None), ("max_resp", None), ("proxy", None), ("exact", None)]
               + [("random", s) for s in args.random_seeds])

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "results.csv")
    agreement: list[dict] = []
    t_start = time.perf_counter()
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target_row", "method", "seed", "k", "view", "img_name",
                    "valid", "logdet", "trace_sigma", "lmax_sigma", "select_s"])
        n_done = 0
        for row in candidates:
            if n_done == args.num_targets:
                break
            # seed = first pool view where the target is ACTUALLY visible;
            # analytic pre-filter can pass fully occluded gaussians — skip those
            seed_view = next((i for i in range(len(pool))
                              if cache.visibility(row, i).valid), None)
            if seed_view is None:
                print(f"target {row}: occluded in every pool view, skipped")
                continue
            n_done += 1
            for method, mseed in methods:
                mrng = np.random.default_rng(mseed) if mseed is not None else None
                out = run_method(method, row, seed_view, pool, cache, cfg,
                                 args.budget, mrng, scorer, agreement)
                for r in out:
                    w.writerow([row, method, mseed if mseed is not None else "",
                                r["k"], r["view"],
                                pool[r["view"]].meta["img_name"], r["valid"],
                                f"{r['logdet']:.6f}", f"{r['trace_sigma']:.6e}",
                                f"{r['lmax_sigma']:.6e}", f"{r['select_s']:.4f}"])
            print(f"target {row} done ({time.perf_counter() - t_start:.0f}s)")
        if n_done < args.num_targets:
            raise RuntimeError(f"only {n_done}/{args.num_targets} targets usable; "
                               "loosen sampling filters or raise --min-opacity pool")

    summary = {
        "n_targets": n_done, "pool_size": len(pool), "budget": args.budget,
        "mean_visibility_s": float(np.mean(cache.timing["vis"])),
        "mean_jacobian_s": float(np.mean(cache.timing["jac"])),
        "n_jacobians": len(cache.timing["jac"]),
        "proxy_exact_kendall_tau": float(np.mean([a["tau"] for a in agreement]))
        if agreement else None,
        "proxy_exact_top1_agreement": float(np.mean([a["top1_match"] for a in agreement]))
        if agreement else None,
        "config": cfg.to_dict(),
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2))
    print(f"results -> {csv_path}")


if __name__ == "__main__":
    main()
