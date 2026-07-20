# Part 3.1 scene runner: per-Gaussian importance / confidence state on the
# FROZEN all-25 R_change, rendered to every query pose (spec §2-§13).
#
# Diagnostic only. The checkpoint is loaded read-only, no densify/prune, no
# optimizer — see tests/test_importance_diagnostics.py for the checksum guard.
#
# GT ordering (§1.7, §21.4) is enforced at runtime, not by convention: the GT
# loader lives behind `GT_GATE`, which refuses to open anything until the
# Gaussian state has been written to disk. view_selection/importance_diagnostics
# never imports GT at all.
#
#   python experiments/visualize_rchange_importance.py --scenes Garden
#   python experiments/visualize_rchange_importance.py --scenes all --gpus 8
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
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments"))

INSTANCE = "Instance_1"
SCENES = ["Cantina", "Garden", "Lounge", "Lunch_room", "Meeting_room",
          "Playground", "Porch", "Pots", "Printing_area", "Zen"]
GCTX_ROOT = os.path.join(REPO, "outputs", "change_nbv", "global_context")
CUE_ROOT = os.path.join(REPO, "output_subset", "cue_gt")
OUT_ROOT = os.path.join(REPO, "outputs", "change_nbv", "diagnostics",
                        "rchange_importance")
SIG05 = float(torch.sigmoid(torch.tensor(0.5)))  # stored masks are sigmoid(z)
_t0 = time.time()
_lock = threading.Lock()


def log(m):
    with _lock:
        print(f"[{time.time()-_t0:6.0f}s] {m}", flush=True)


# ---------------------------------------------------------------- GT gate ---

class _GtGate:
    """Refuses to hand out ground truth until the importance state is final.

    This is the runtime half of the §1.9 manifest claim `gt_used_for_importance:
    false` — the state is written and hashed before anything can read a mask."""

    def __init__(self):
        self._open = False

    def unlock(self, state_path: str):
        if not os.path.exists(state_path):
            raise RuntimeError("cannot unlock GT before the state is written")
        self._open = True

    def mask(self, scene: str, image_name: str, hw) -> np.ndarray | None:
        if not self._open:
            raise RuntimeError(
                "GT was requested before the Gaussian state was finalized — "
                "this would invalidate gt_used_for_importance:false (§1.7)")
        import cv2
        p = os.path.join(REPO, "data", "PASLCD", INSTANCE, scene, "gt_mask",
                         image_name + ".png")
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m is None:
            return None
        return cv2.resize(m, (hw[1], hw[0]),
                          interpolation=cv2.INTER_NEAREST) > 127


GT_GATE = _GtGate()


# ------------------------------------------------------------- artifacts ----

def find_gctx(scene: str, seed: int = 0) -> str:
    """Newest cached all-25 R_global for (scene, seed). Same rule as
    experiments/cue_toxicity_check.py:50 — metadata records the regression."""
    best, best_t = None, -1.0
    for meta_path in glob.glob(os.path.join(GCTX_ROOT, scene, INSTANCE, "*",
                                            "metadata.json")):
        meta = json.load(open(meta_path))
        if meta.get("train_seed") != seed or not meta.get("regression_ok", True):
            continue
        t = os.path.getmtime(meta_path)
        if t > best_t:
            best, best_t = os.path.dirname(meta_path), t
    if best is None:
        raise FileNotFoundError(f"no cached all-25 R_change for {scene} "
                                f"seed {seed}")
    return best


def scene_args(scene: str) -> Namespace:
    src = os.path.join(REPO, "data", "PASLCD", INSTANCE, scene)
    return Namespace(
        source_path=src, resolution=4, masks_dir="", num_loader_threads=8,
        test_hold=5, use_colmap_poses=False, eval_poses=False, pyr_levels=1,
        num_kpts=int(4096 * 1.5), match_max_error=2e-3, fundmat_samples=2000,
        min_num_inliers=100, fix_focal=False, init_focal=-1.0, init_fov=-1.0,
        num_keyframes_for_triangulation=8, num_reference_keyframes=4,
        pnpransac_samples=2000, num_pts_miniba_incr=2000,
        iters_miniba_incr=20)


def build_cameras(scene: str):
    """Phase A pose estimation, verbatim from the pipeline (the same path
    experiments/frame_feature_analysis.py:118-152 reproduces). Returns the 25
    inference `Camera`s in dataset order."""
    from dataloaders.image_dataset import ImageDataset
    from poses.feature_detector import Detector
    from poses.matcher import Matcher
    from poses.pose_initializer import (PoseInitializer,
                                        get_reference_keyframes)
    from poses.triangulator import Triangulator
    from scene.cameras import Camera
    from scene.dense_extractor import DenseExtractor
    from scene.keyframe import Keyframe

    args = scene_args(scene)
    dataset = ImageDataset(args, instance="inf")
    height, width = dataset.get_image_size()
    reference_dataset = ImageDataset(args, instance="ref")

    max_error = max(args.match_max_error * width, 1.5)
    matcher = Matcher(args.fundmat_samples, max_error)
    triangulator = Triangulator(args.num_kpts,
                                args.num_keyframes_for_triangulation, max_error)
    pose_initializer = PoseInitializer(width, height, triangulator, matcher,
                                       max_error, args)
    dense_extractor = DenseExtractor(width, height)
    detector = Detector(args.num_kpts, width, height)

    reference_keyframes = []
    for frame_id in range(len(reference_dataset)):
        image, info = reference_dataset.getnext()
        kf = Keyframe(image, info, detector(image), info["Rt"], frame_id,
                      info["focal"], dense_extractor, triangulator, args)
        reference_keyframes.append(kf)
        fov_x, fov_y, focal = info["FovX"], info["FovY"], info["focal"]
    pose_initializer.init_focal(focal)
    n_ref = len(reference_keyframes)
    for i in range(n_ref):
        for j in range(i + 1, n_ref):
            matcher(reference_keyframes[i].desc_kpts,
                    reference_keyframes[j].desc_kpts, remove_outliers=True,
                    update_kpts_flag="inliers", kID=i, kID_other=j)
    for i in range(n_ref):
        reference_keyframes[i].update_3dpts(reference_keyframes)

    cams = []
    for frame_id in range(len(dataset)):
        image, info = dataset.getnext()
        desc_kpts = detector(image)
        prev = get_reference_keyframes(n=args.num_reference_keyframes,
                                       keyframes=reference_keyframes,
                                       matcher=matcher, desc_kpts=desc_kpts)
        Rt, _ = pose_initializer.init_inference_pose(prev, desc_kpts,
                                                     frame_id, False, image)
        R = Rt[:3, :3].cpu().numpy()
        t = Rt[:3, 3].cpu().numpy()
        cams.append(Camera(colmap_id=f"{frame_id:04d}", R=np.transpose(R), T=t,
                           FoVx=fov_x, FoVy=fov_y, image=image[:3, ...],
                           gt_alpha_mask=None,
                           image_name=info["name"].split(".")[0],
                           uid=f"{frame_id:04d}"))
    return cams


def load_cues(scene: str, cams) -> torch.Tensor:
    """The combined cue C_v the pipeline actually fed to L_SSF.

    subset_oscd.py:339 (--dump_cues) and subset_oscd.py:394-397 (process_view)
    build it with the same two lines against the never-updated RGB model, so
    the dump is the training cue, not a re-derivation. No new normalization
    (§3) — the raw values go straight in."""
    blob = torch.load(os.path.join(CUE_ROOT, scene, "cues.pt"),
                      map_location="cpu", weights_only=False)
    cues = blob["cues"]
    if cues.ndim == 4:
        cues = cues[:, 0]
    by_name = {n: i for i, n in enumerate(blob["image_names"])}
    missing = [c.image_name for c in cams if c.image_name not in by_name]
    if missing:
        raise KeyError(f"{scene}: cue dump is missing {missing[:3]}")
    return torch.stack([cues[by_name[c.image_name]] for c in cams])


# ------------------------------------------------------------ stage 1 -------

def scene_state(scene: str, model, cams, pipe, num_probes: int = 128,
                cfg=None, cues: torch.Tensor | None = None):
    """§3-§8 on one scene. Returns (state, aux) with aux carrying the raw
    per-frame q/s/tau, the accumulated blocks and the split-half observability
    used to report whether O is signal or Hutchinson noise."""
    from target_nbv.change.counts import soft_counts
    from view_selection.importance_diagnostics import (DiagnosticConfig,
                                                       gaussian_state,
                                                       observability,
                                                       robust_unit, saturate,
                                                       saturation_scale)
    from view_selection.information import hutchinson_block_information
    from view_selection.types import InformationConfig
    from view_selection.weights import build_pixel_weight

    cfg = cfg or DiagnosticConfig()
    n = model.get_xyz.shape[0]
    device = model.get_xyz.device
    background = torch.zeros(3, device=device)
    v = len(cams)
    if cues is None:
        cues = load_cues(scene, cams)

    e1_all = torch.zeros((v, n), dtype=torch.float32, device=device)
    tau_all = torch.zeros((v, n), dtype=torch.float32, device=device)
    cue_stats = []
    # Two independent probe halves so the report can state whether O_i is a
    # measurement or a random number (the block estimator is Hutchinson, not
    # exact, despite the "exact block" naming in the selector).
    half = max(1, num_probes // 2)
    icfg_a = InformationConfig(weight_mode="pose", num_probes=half)
    icfg_b = InformationConfig(weight_mode="pose", num_probes=num_probes - half)
    H_a = torch.zeros((n, 3, 3), dtype=torch.float32, device=device)
    H_b = torch.zeros_like(H_a)

    for i, cam in enumerate(cams):
        c_raw_map = cues[i].to(device)
        clamped = c_raw_map.clamp(0.0, 1.0)
        cue_stats.append({
            "frame": i, "image": cam.image_name,
            "cue_min": float(c_raw_map.min()), "cue_max": float(c_raw_map.max()),
            "cue_mean": float(c_raw_map.mean()),
            "cue_q99": float(torch.quantile(c_raw_map.flatten().float(), 0.99)),
            "clipped_frac": float((c_raw_map > 1.0).double().mean()),
        })
        e1, _, tau = soft_counts(model, cam, clamped, pipe)
        e1_all[i], tau_all[i] = e1.float(), tau.float()

        w = build_pixel_weight("pose", model, cam, pipe, background, icfg_a)
        H_a += hutchinson_block_information(model, cam, w, icfg_a,
                                            (scene, "diagA"), pipe, background,
                                            frame_id=i)
        H_b += hutchinson_block_information(model, cam, w, icfg_b,
                                            (scene, "diagB"), pipe, background,
                                            frame_id=i)
        if (i + 1) % 5 == 0:
            log(f"  {scene}: state {i+1}/{v}")

    tau0 = saturation_scale(tau_all, cfg)
    s_all = saturate(tau_all, tau0)
    q_all = (e1_all / (tau_all + 1e-8)).clamp(0.0, 1.0)

    st = gaussian_state(q_all, s_all, cfg)
    obs = observability(0.5 * (H_a + H_b), cfg)
    st.g_raw, st.observability = obs["g_raw"], obs["O"]
    st.eig_min, st.eig_max, st.condition = (obs["eig_min"], obs["eig_max"],
                                            obs["condition"])
    st.n_dir_obs = (tau_all > 0).sum(dim=0).double()
    with torch.no_grad():
        c = model._features_dc.detach()
        st.c_raw = torch.sigmoid(c.reshape(c.shape[0], -1).mean(dim=1)).double()

    # Most Gaussians carry NO position information in the change channel (the
    # xyz adjoint of the change render vanishes where c is flat), so logdet
    # sits exactly on the damping floor for them. Those ties would inflate a
    # whole-population split-half correlation to ~1.0 no matter how noisy the
    # estimator is — the honest number is measured on the informative subset.
    ga = observability(H_a, cfg)["g_raw"]
    gb = observability(H_b, cfg)["g_raw"]
    floor = 3.0 * float(np.log(cfg.lam))
    info = (st.g_raw > floor + 1e-3)
    o_a, o_b = robust_unit(ga, 0.05, 0.95), robust_unit(gb, 0.05, 0.95)
    aux = {"q": q_all, "s": s_all, "tau": tau_all, "tau0": tau0,
           "cue_stats": cue_stats,
           "o_floor_frac": float((~info).double().mean()),
           "o_informative_count": int(info.sum()),
           "o_split_half_spearman_all": spearman(o_a, o_b),
           "o_split_half_spearman": (spearman(ga[info], gb[info])
                                     if int(info.sum()) > 10 else float("nan")),
           "o_split_half_pearson": (float(np.corrcoef(
               ga[info].cpu().numpy(), gb[info].cpu().numpy())[0, 1])
               if int(info.sum()) > 10 else float("nan"))}
    return st, aux


def spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    def rank(t):
        order = torch.argsort(t)
        r = torch.empty_like(order, dtype=torch.float64)
        r[order] = torch.arange(t.numel(), dtype=torch.float64,
                                device=t.device)
        return r
    rx, ry = rank(x.double()), rank(y.double())
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    return float((rx * ry).sum() / (rx.norm() * ry.norm() + 1e-12))


# ------------------------------------------------------------ metrics -------

def curve_stats(score: torch.Tensor, target: torch.Tensor,
                recalls=(0.95, 0.99), percents=(1, 5, 10, 20, 30, 50)) -> dict:
    """One sort gives the whole PR/ROC story (§13 A, C, D).

    score/target: 1-D, target in {0,1}. Everything is computed on the ranking
    of `score`, high first."""
    n_pos = float(target.sum())
    n_neg = float(target.numel() - n_pos)
    if n_pos == 0 or n_neg == 0:
        return {}
    order = torch.argsort(score, descending=True)
    t = target[order].double()
    tp = torch.cumsum(t, 0)
    fp = torch.cumsum(1.0 - t, 0)
    prec = tp / (tp + fp)
    rec = tp / n_pos

    # average precision: sum over the recall steps the positives create
    ap = float((prec * t).sum() / n_pos)
    # rank-based AUROC (Mann-Whitney)
    ranks = torch.arange(1, t.numel() + 1, dtype=torch.float64,
                         device=t.device)
    auroc = float(((ranks * t).sum() - n_pos * (n_pos + 1) / 2)
                  / (n_pos * n_neg))
    auroc = 1.0 - auroc if auroc < 0 else auroc

    out = {"auprc": ap, "auroc": auroc, "n_pos": n_pos, "n_neg": n_neg,
           "prevalence": n_pos / (n_pos + n_neg)}
    for r in recalls:
        idx = torch.nonzero(rec >= r, as_tuple=False)
        if idx.numel():
            k = int(idx[0])
            out[f"precision_at_recall{r:g}"] = float(prec[k])
            out[f"fp_rejection_at_recall{r:g}"] = float(1.0 - fp[k] / n_neg)
            out[f"kept_frac_at_recall{r:g}"] = float((k + 1) / t.numel())
    for p in percents:
        k = max(1, int(round(t.numel() * p / 100.0))) - 1
        tp_k, fp_k = float(tp[k]), float(fp[k])
        out[f"top{p}_tp_retained"] = tp_k / n_pos
        out[f"top{p}_fp_retained"] = fp_k / n_neg
        out[f"top{p}_precision"] = tp_k / (tp_k + fp_k)
        out[f"top{p}_recall"] = tp_k / n_pos
        out[f"top{p}_iou"] = tp_k / (tp_k + fp_k + (n_pos - tp_k))
    return out


def separation_stats(score: torch.Tensor, target: torch.Tensor) -> dict:
    pos, neg = score[target], score[~target]
    if pos.numel() == 0 or neg.numel() == 0:
        return {}
    mp, mn = float(pos.mean()), float(neg.mean())
    sd = float(torch.sqrt(0.5 * (pos.var() + neg.var())).clamp_min(1e-12))
    return {"mean_tp": mp, "mean_fp": mn,
            "median_tp": float(pos.median()), "median_fp": float(neg.median()),
            "mean_separation": mp - mn, "effect_size": (mp - mn) / sd}


# ------------------------------------------------------------ stage 2/3 -----

def run_scene(scene: str, args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from arguments import PipelineParams
    from target_nbv.change.counts import responsibility_probe_render
    from view_selection.global_context import load_frozen_change_model
    from view_selection.importance_diagnostics import (DiagnosticConfig,
                                                       required_variants,
                                                       sweep_grid, sweep_name,
                                                       sweep_variant)
    from view_selection.weights import alpha_map

    out_dir = os.path.join(OUT_ROOT, INSTANCE, scene)
    os.makedirs(out_dir, exist_ok=True)
    pipe_parser = argparse.ArgumentParser()
    pipe = PipelineParams(pipe_parser).extract(pipe_parser.parse_args([]))
    cfg = DiagnosticConfig(tau0_mode=args.tau0_mode,
                           tau0_value=args.tau0_value,
                           kappa0_sweep=tuple(args.kappa0),
                           kappa0_default=args.kappa0_default)

    gdir = find_gctx(scene, args.seed)
    model = load_frozen_change_model(os.path.join(gdir, "r_global.ply"))
    cams = build_cameras(scene)
    log(f"{scene}: {len(cams)} poses, N={model.get_xyz.shape[0]}")

    cues = load_cues(scene, cams)
    st, aux = scene_state(scene, model, cams, pipe, args.num_probes, cfg, cues)

    # ---- persist the state BEFORE any GT can be touched (§1.7) -------------
    state_path = os.path.join(out_dir, "gaussian_state.pt")
    scalars = {k: getattr(st, k).cpu() for k in
               ("m", "a", "b", "kappa", "e_pos", "e_neg", "support", "n_view",
                "agreement", "uncertainty", "cue_variance", "pos_neg_ratio",
                "loo_min", "loo_median", "loo_max", "n_agree", "n_conflict",
                "n_loo_valid", "observability", "g_raw", "eig_min", "eig_max",
                "condition", "n_dir_obs", "c_raw")}
    scalars["confidence"] = st.confidence().cpu()
    scalars["verification"] = st.verification().cpu()
    torch.save({"scalars": scalars,
                "support_by_kappa0": {k: v.cpu() for k, v in
                                      st.support_by_kappa0.items()},
                "tau0": aux["tau0"], "config": cfg.key_dict()}, state_path)
    with open(os.path.join(out_dir, "gaussian_state.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["scalar", "mean", "std", "q05", "q50", "q95", "min", "max"])
        for k, v in scalars.items():
            d = v.double()
            w.writerow([k] + [f"{float(x):.6g}" for x in
                              (d.mean(), d.std(), d.quantile(0.05),
                               d.quantile(0.5), d.quantile(0.95), d.min(),
                               d.max())])

    manifest = {
        "claim_scope": "gt_diagnostic_only",
        "source_model": "all25_rchange",
        "gt_used_for_importance": False,
        "gt_used_for_diagnostic_evaluation": True,
        "model_modified": False,
        "scene": scene, "instance": INSTANCE,
        "gctx_dir": os.path.relpath(gdir, REPO),
        "gctx_key": json.load(open(os.path.join(gdir, "metadata.json")))["key"],
        "n_gaussians": int(model.get_xyz.shape[0]),
        "n_frames": len(cams),
        "cue_threshold": args.cue_threshold,
        "cue_source": os.path.relpath(os.path.join(CUE_ROOT, scene, "cues.pt"),
                                      REPO),
        "cue_stats": aux["cue_stats"],
        "tau0": aux["tau0"], "num_probes": args.num_probes,
        "observability_estimator": "hutchinson_position_block (not exact)",
        "o_split_half_spearman": aux["o_split_half_spearman"],
        "o_split_half_spearman_all_gaussians": aux["o_split_half_spearman_all"],
        "o_split_half_pearson": aux["o_split_half_pearson"],
        "o_floor_frac": aux["o_floor_frac"],
        "o_informative_count": aux["o_informative_count"],
        "diagnostic_config": cfg.key_dict(),
        "agreement": "loo", "densify_or_prune": False,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    log(f"{scene}: state written, O split-half rho="
        f"{aux['o_split_half_spearman']:.3f} on "
        f"{aux['o_informative_count']} informative Gaussians "
        f"({100*aux['o_floor_frac']:.1f}% at the damping floor)")

    # ---- GT unlocked from here (§1.7) -------------------------------------
    GT_GATE.unlock(state_path)
    stored = torch.load(os.path.join(gdir, "all25_rendered_soft_masks.pt"),
                        map_location="cpu", weights_only=False)
    by_name = {n: i for i, n in enumerate(stored["image_names"])}
    device = model.get_xyz.device

    variants = required_variants(st)
    required = [k for k in variants]
    sweeps = [sweep_name(*c) for c in sweep_grid()] if args.importance_sweep \
        else []

    alpha = {}
    P, G = {}, {}
    for i, cam in enumerate(cams):
        a = alpha_map(model, cam, pipe, torch.zeros(3, device=device))
        alpha[i] = a
        soft = stored["soft_masks"][by_name[cam.image_name]].to(device)
        P[i] = soft >= SIG05
        gt = GT_GATE.mask(scene, cam.image_name, tuple(a.shape))
        G[i] = torch.from_numpy(gt).to(device) if gt is not None else None

    # §14 GT attribution: backproject the all-25 prediction's TP/FP pixels onto
    # the SAME topology through the compositing responsibility. Kept in its own
    # file so gaussian_state.pt stays provably GT-free.
    from target_nbv.change.counts import soft_counts
    tp_mass = torch.zeros(model.get_xyz.shape[0], dtype=torch.float64,
                          device=device)
    fp_mass = torch.zeros_like(tp_mass)
    ctx = {"reference_render": [], "cue": [], "pred": [], "gt": [],
           "original": [], "image_names": [c.image_name for c in cams]}
    from gaussian_renderer import render
    from scene import GaussianModel
    rgb_model = GaussianModel(3, 3)
    rgb_model.load_ply(os.path.join(REPO, "data", "PASLCD", INSTANCE, scene,
                                    "reference_reconstruction", "point_cloud",
                                    "iteration_30000", "point_cloud.ply"))
    for i, cam in enumerate(cams):
        if G[i] is not None:
            e1_tp, _, _ = soft_counts(model, cam, (P[i] & G[i]).float(), pipe)
            e1_fp, _, _ = soft_counts(model, cam, (P[i] & ~G[i]).float(), pipe)
            tp_mass += e1_tp
            fp_mass += e1_fp
        with torch.no_grad():
            ref = render(cam, rgb_model, pipe,
                         torch.zeros(3, device=device))["render"]
        ctx["reference_render"].append((ref.clamp(0, 1) * 255).to(torch.uint8).cpu())
        ctx["original"].append(
            (cam.original_image[:3].clamp(0, 1) * 255).to(torch.uint8).cpu())
        ctx["cue"].append(cues[i].half().cpu())
        ctx["pred"].append(P[i].cpu())
        ctx["gt"].append(G[i].cpu() if G[i] is not None
                         else torch.zeros_like(P[i]).cpu())
    del rgb_model
    torch.cuda.empty_cache()
    torch.save({"tp_mass": tp_mass.cpu(), "fp_mass": fp_mass.cpu(),
                "tp_purity": (tp_mass / (tp_mass + fp_mass + 1e-12)).cpu(),
                "total_mass": (tp_mass + fp_mass).cpu()},
               os.path.join(out_dir, "gt_attribution.pt"))
    torch.save({k: (torch.stack(v) if isinstance(v, list) and
                    torch.is_tensor(v[0]) else v) for k, v in ctx.items()},
               os.path.join(out_dir, "frame_context.pt"))
    log(f"{scene}: GT attribution + frame context saved")

    rows = []
    render_dir = os.path.join(out_dir, "renders")

    def score_variant(name: str, weights: torch.Tensor, save: bool):
        vals = weights.to(device).float()
        for i, cam in enumerate(cams):
            heat = responsibility_probe_render(model, cam, vals, pipe)
            if save:
                d = os.path.join(render_dir, name)
                os.makedirs(d, exist_ok=True)
                torch.save(heat.half().cpu(),
                           os.path.join(d, f"frame_{i:02d}.pt"))
            for norm in (("raw", "alpha") if save else ("raw",)):
                h = heat if norm == "raw" else heat / alpha[i].clamp_min(1e-3)
                rows.extend(_frame_rows(scene, name, norm, i, cam, h, P[i],
                                        G[i]))

    for name in required:
        score_variant(name, variants[name], save=True)
    log(f"{scene}: {len(required)} required variants rendered")

    if args.importance_sweep:
        for combo in sweep_grid():
            score_variant(sweep_name(*combo), sweep_variant(st, *combo, cfg),
                          save=False)
        log(f"{scene}: {len(sweeps)} sweep variants scored")

    with open(os.path.join(out_dir, "frame_metrics.csv"), "w", newline="") as f:
        keys = sorted({k for r in rows for k in r})
        head = ["scene", "variant", "norm", "frame", "image", "task"]
        keys = head + [k for k in keys if k not in head]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    log(f"{scene}: wrote {len(rows)} metric rows")

    # required-variant contact previews (montages proper live in analyze_*)
    _save_previews(out_dir, scene, cams, required, render_dir, plt)


def _frame_rows(scene, variant, norm, i, cam, heat, P_v, G_v):
    """§13 A (TP vs FP inside the all-25 positive) and B (GT over all pixels)."""
    rows = []
    flat = heat.flatten()
    if G_v is not None and P_v.any():
        sel = P_v.flatten()
        score = flat[sel]
        tgt = (P_v & G_v).flatten()[sel]
        if tgt.any() and not tgt.all():
            r = {"scene": scene, "variant": variant, "norm": norm, "frame": i,
                 "image": cam.image_name, "task": "tp_vs_fp"}
            r.update(curve_stats(score, tgt))
            r.update(separation_stats(score, tgt))
            rows.append(r)
    if G_v is not None:
        r = {"scene": scene, "variant": variant, "norm": norm, "frame": i,
             "image": cam.image_name, "task": "gt_all_pixels"}
        r.update(curve_stats(flat, G_v.flatten()))
        rows.append(r)
    return rows


def _save_previews(out_dir, scene, cams, required, render_dir, plt):
    """Scene-wide fixed color scale (§11): one Q01-Q99 per variant, shared by
    every frame — never per-frame auto-normalization."""
    png_dir = os.path.join(out_dir, "renders")
    for name in required:
        files = sorted(glob.glob(os.path.join(render_dir, name, "*.pt")))
        if not files:
            continue
        maps = [torch.load(f, weights_only=False).float() for f in files]
        allv = torch.cat([m.flatten() for m in maps]).double()
        lo, hi = float(allv.quantile(0.01)), float(allv.quantile(0.99))
        for f, m in zip(files, maps):
            fig, ax = plt.subplots(figsize=(5.2, 3.0))
            im = ax.imshow(m.numpy(), cmap="inferno", vmin=lo, vmax=hi)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{scene} {name}\n"
                         f"{os.path.basename(f)[:-3]}  (Q01={lo:.3g}, "
                         f"Q99={hi:.3g})", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.033)
            fig.savefig(f[:-3] + ".png", dpi=110, bbox_inches="tight")
            plt.close(fig)


# ---------------------------------------------------------------- driver ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default=INSTANCE)
    ap.add_argument("--scenes", nargs="+", default=["Garden"])
    ap.add_argument("--source", default="all25", choices=["all25"])
    ap.add_argument("--query_frames", default="all", choices=["all"])
    ap.add_argument("--cue_threshold", type=float, default=0.5)
    ap.add_argument("--tau0_mode", default="median",
                    choices=["median", "q25", "q75", "fixed"])
    ap.add_argument("--tau0_value", type=float, default=None)
    ap.add_argument("--kappa0", type=float, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--kappa0_default", type=float, default=4.0)
    ap.add_argument("--num_probes", type=int, default=128)
    ap.add_argument("--agreement", default="loo", choices=["loo"])
    ap.add_argument("--observability", default="exact_position_block")
    ap.add_argument("--importance_variants", default="all")
    ap.add_argument("--importance_sweep", action="store_true")
    ap.add_argument("--fixed_recall", type=float, nargs="+",
                    default=[0.95, 0.99])
    ap.add_argument("--top_percent", type=int, nargs="+",
                    default=[1, 5, 10, 20, 30, 50])
    ap.add_argument("--output", default=OUT_ROOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--worker", default=None,
                    help="internal: run this one scene in-process")
    args = ap.parse_args()

    scenes = SCENES if args.scenes == ["all"] else args.scenes
    if args.worker:
        run_scene(args.worker, args)
        return

    # XFeat/DenseExtractor capture CUDA graphs that conflict across scenes —
    # one subprocess per scene, same rule as experiments/cue_gt_report.py.
    pool = queue.Queue()
    for g in range(args.gpus):
        pool.put(g)
    failures = []

    def work(scene):
        gpu = pool.get()
        try:
            cmd = [sys.executable, os.path.abspath(__file__), "--worker", scene,
                   "--num_probes", str(args.num_probes),
                   "--tau0_mode", args.tau0_mode,
                   "--kappa0_default", str(args.kappa0_default),
                   "--seed", str(args.seed)]
            if args.importance_sweep:
                cmd.append("--importance_sweep")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
                   "TORCHDYNAMO_DISABLE": "1", "PYTHONPATH": REPO}
            r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True,
                               text=True)
            if r.returncode != 0:
                failures.append(scene)
                log(f"FAILED {scene}\n{r.stdout[-1500:]}\n{r.stderr[-2500:]}")
            else:
                log(f"done {scene}")
        finally:
            pool.put(gpu)

    with ThreadPoolExecutor(max_workers=args.gpus) as ex:
        list(ex.map(work, scenes))
    if failures:
        raise SystemExit(f"scenes failed: {failures}")


if __name__ == "__main__":
    main()
