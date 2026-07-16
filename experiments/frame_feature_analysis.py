# GT-free frame-quality features vs oracle marginal value (2026-07-16).
#
# For every (Instance_1 scene, frame) compute features available AT SELECTION
# TIME (no GT masks):
#   pose_pnp_inliers, pose_ba_inliers, pose_ba_residual   (pose quality)
#   cue_mass, cue_area, cue_max                           (change-cue strength)
#   cue_consistency = cos(e1_f, sum of e1_others)         (3D-lifted cue
#       agreement with the other frames; e1_g = per-Gaussian soft counts of
#       the frame's cue via the rasterizer adjoint)
#   dir_isolation = mean angle to other view dirs, cam_dist_to_centroid
# and correlate them with the frame's marginal mIoU from the oracle-5 map
# (mean mIoU of evaluated sets containing the frame minus those without).
#
#   python experiments/frame_feature_analysis.py --out experiments/frame_features.csv

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

SCENES = ["Cantina", "Garden", "Lounge", "Lunch_room", "Meeting_room",
          "Playground", "Porch", "Pots", "Printing_area", "Zen"]
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORACLE_CSV = os.path.join(REPO, "experiments", "oracle_search_results.csv")


def marginal_values() -> dict[tuple[str, int], float]:
    per = defaultdict(list)
    for r in csv.DictReader(open(ORACLE_CSV)):
        combo = tuple(int(x) for x in r["combo"].split("-"))
        if len(combo) == 5:
            per[r["scene"]].append((combo, float(r["miou_query"])))
    out = {}
    for s, data in per.items():
        for f in range(25):
            inc = [m for c, m in data if f in c]
            exc = [m for c, m in data if f not in c]
            if len(inc) >= 3 and len(exc) >= 3:
                out[(s, f)] = float(np.mean(inc) - np.mean(exc))
    return out


def analyze_scene(scene: str, sam_model, pipe, shared: dict):
    """Returns list of per-frame feature dicts. Detector/DenseExtractor carry
    CUDA-graph captures that break when re-instantiated while old graphs are
    alive — reuse them across scenes via `shared`, keyed by image size."""
    from argparse import Namespace

    from dataloaders.image_dataset import ImageDataset
    from oscd import generate_candidate_map
    from poses.feature_detector import Detector
    from poses.matcher import Matcher
    from poses.pose_initializer import PoseInitializer, get_reference_keyframes
    from poses.triangulator import Triangulator
    from scene import GaussianModel
    from scene.cameras import Camera
    from scene.dense_extractor import DenseExtractor
    from scene.keyframe import Keyframe
    from gaussian_renderer import render
    from target_nbv.change.counts import soft_counts

    src = os.path.join(REPO, "data", "PASLCD", "Instance_1", scene)
    args = Namespace(source_path=src, resolution=4, masks_dir="",
                     num_loader_threads=8, test_hold=5, use_colmap_poses=False,
                     eval_poses=False, pyr_levels=1, num_kpts=int(4096 * 1.5),
                     match_max_error=2e-3, fundmat_samples=2000,
                     min_num_inliers=100, fix_focal=False, init_focal=-1.0,
                     init_fov=-1.0, num_keyframes_for_triangulation=8,
                     num_reference_keyframes=4, pnpransac_samples=2000,
                     num_pts_miniba_incr=2000, iters_miniba_incr=20)

    gaussians_rgb = GaussianModel(3, 3)
    gaussians_rgb.load_ply(os.path.join(src, "reference_reconstruction",
                                        "point_cloud", "iteration_30000", "point_cloud.ply"))
    gaussians_change = GaussianModel(3, 0)
    gaussians_change.load_ply_change(os.path.join(src, "reference_reconstruction",
                                                  "point_cloud", "iteration_30000", "point_cloud.ply"))

    dataset = ImageDataset(args, instance="inf")
    height, width = dataset.get_image_size()
    reference_dataset = ImageDataset(args, instance="ref")

    max_error = max(args.match_max_error * width, 1.5)
    matcher = Matcher(args.fundmat_samples, max_error)
    triangulator = Triangulator(args.num_kpts, args.num_keyframes_for_triangulation, max_error)
    pose_initializer = PoseInitializer(width, height, triangulator, matcher, max_error, args)
    key = (width, height)
    if key not in shared:
        shared[key] = (DenseExtractor(width, height), Detector(args.num_kpts, width, height))
    dense_extractor, detector = shared[key]

    # --- instrument pose internals (PnP inliers, miniBA residual/inliers) ----
    stats = {}
    orig_pnp, orig_ba = pose_initializer.PnPRANSAC, pose_initializer.miniBA

    def pnp_wrap(*a, **k):
        Rt, inl = orig_pnp(*a, **k)
        stats["pnp_inliers"] = int(inl.sum())
        return Rt, inl

    def ba_wrap(*a, **k):
        out = orig_ba(*a, **k)
        stats["ba_residual"] = float(torch.as_tensor(out[4], dtype=torch.float64).mean())
        stats["ba_inliers"] = int(out[6].sum())
        return out

    pose_initializer.PnPRANSAC = pnp_wrap
    pose_initializer.miniBA = ba_wrap

    # --- reference keyframes (verbatim subset_oscd Phase pre-A) --------------
    reference_keyframes = []
    for frameID in range(len(reference_dataset)):
        image, info = reference_dataset.getnext()
        desc_kpts = detector(image)
        keyframe = Keyframe(image, info, desc_kpts, info["Rt"], frameID,
                            info["focal"], dense_extractor, triangulator, args)
        reference_keyframes.append(keyframe)
        Fovx, Fovy, f = info["FovX"], info["FovY"], info["focal"]
    pose_initializer.init_focal(f)
    n_ref = len(reference_keyframes)
    for i in range(n_ref):
        for j in range(i + 1, n_ref):
            matcher(reference_keyframes[i].desc_kpts, reference_keyframes[j].desc_kpts,
                    remove_outliers=True, update_kpts_flag="inliers", kID=i, kID_other=j)
    for frameID in range(n_ref):
        reference_keyframes[frameID].update_3dpts(reference_keyframes)

    background = torch.zeros(3, device="cuda")
    patch_size = 14
    rows, e1_vecs, centers, dirs = [], [], [], []

    for frameID in range(len(dataset)):
        image, info = dataset.getnext()
        desc_kpts = detector(image)
        prev = get_reference_keyframes(n=args.num_reference_keyframes,
                                       keyframes=reference_keyframes,
                                       matcher=matcher, desc_kpts=desc_kpts)
        stats.clear()
        Rt, _ = pose_initializer.init_inference_pose(prev, desc_kpts, frameID, False, image)
        R = Rt[:3, :3].cpu().numpy()
        t = Rt[:3, 3].cpu().numpy()
        view = Camera(colmap_id=f"{frameID:04d}", R=np.transpose(R), T=t,
                      FoVx=Fovx, FoVy=Fovy, image=image[:3, ...],
                      gt_alpha_mask=None, image_name=info["name"].split(".")[0],
                      uid=f"{frameID:04d}")

        with torch.no_grad():
            rgb = render(view, gaussians_rgb, pipe, background)["render"]
            cue = generate_candidate_map(view.original_image[:3, ...], rgb,
                                         sam_model, patch_size, height, width)
        cue2d = cue.squeeze(0).clamp(0.0, 1.0)
        e1, _, _ = soft_counts(gaussians_change, view, cue2d, pipe)
        e1_vecs.append(e1.cpu())

        centers.append(view.camera_center.detach().cpu().numpy().astype(np.float64))
        # COLMAP forward in world = third row of R_w2c = third col of R_c2w
        dirs.append((np.transpose(R)[:, 2] / np.linalg.norm(np.transpose(R)[:, 2])).astype(np.float64))

        rows.append({
            "scene": scene, "frame": frameID, "name": view.image_name,
            "pnp_inliers": stats.get("pnp_inliers", -1),
            "ba_inliers": stats.get("ba_inliers", -1),
            "ba_residual": stats.get("ba_residual", -1.0),
            "cue_mass": float(cue2d.sum()),
            "cue_area": float((cue2d > 0.5).float().mean()),
            "cue_max": float(cue.max()),
        })

    # cross-frame features
    E = torch.stack(e1_vecs)                      # (25, N) float64
    En = E / E.norm(dim=1, keepdim=True).clamp_min(1e-12)
    total = En.sum(dim=0)
    centers = np.array(centers)
    dirs = np.array(dirs)
    centroid = centers.mean(axis=0)
    for i, r in enumerate(rows):
        others = total - En[i]
        others = others / others.norm().clamp_min(1e-12)
        r["cue_consistency"] = float((En[i] * others).sum())
        cosang = np.clip(dirs[i] @ dirs[np.arange(25) != i].T, -1, 1)
        r["dir_isolation"] = float(np.degrees(np.arccos(cosang)).mean())
        r["cam_dist_centroid"] = float(np.linalg.norm(centers[i] - centroid))
    return rows


def run_one_scene(scene: str, out: str):
    """Child-process entry: clean CUDA context per scene (the XFeat detector's
    CUDA-graph capture fails on a context that already ran renders/SAM)."""
    from transformers import Sam2Model
    from target_nbv.io_utils import default_pipe

    torch.manual_seed(0)
    np.random.seed(0)
    sam = Sam2Model.from_pretrained("facebook/sam2.1-hiera-tiny").half().to("cuda")
    sam.get_image_embeddings = torch.compile(sam.get_image_embeddings, mode="max-autotune")
    pipe = default_pipe()

    marg = marginal_values()
    rows = analyze_scene(scene, sam, pipe, {})
    for r in rows:
        r["marginal_miou"] = marg.get((scene, r["frame"]), np.nan)
    new_file = not os.path.exists(out)
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new_file:
            w.writeheader()
        w.writerows(rows)
    print(f"{scene} done ({len(rows)} frames)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/frame_features.csv")
    ap.add_argument("--scene", default=None, help="internal: process one scene")
    args = ap.parse_args()

    if args.scene:
        run_one_scene(args.scene, args.out)
        return

    import subprocess
    done_scenes = set()
    if os.path.exists(args.out):
        done_scenes = {r["scene"] for r in csv.DictReader(open(args.out))}
        print(f"resume: {sorted(done_scenes)} already in {args.out}", flush=True)
    for scene in SCENES:
        if scene in done_scenes:
            continue
        r = subprocess.run([sys.executable, os.path.abspath(__file__),
                            "--out", args.out, "--scene", scene],
                           env={**os.environ, "PYTHONPATH": ""}, cwd=REPO)
        if r.returncode != 0:
            print(f"SCENE FAILED: {scene}", flush=True)

    all_rows = []
    for r in csv.DictReader(open(args.out)):
        r["frame"] = int(r["frame"])
        for k in r:
            if k not in ("scene", "name", "frame"):
                r[k] = float(r[k])
        all_rows.append(r)

    # pooled correlation (z-scored per scene), Pearson + Spearman
    from scipy.stats import spearmanr
    feats = ["pnp_inliers", "ba_inliers", "ba_residual", "cue_mass", "cue_area",
             "cue_max", "cue_consistency", "dir_isolation", "cam_dist_centroid"]
    print("\npooled correlation with marginal mIoU (z-scored within scene):")
    by_scene = defaultdict(list)
    for r in all_rows:
        by_scene[r["scene"]].append(r)
    for feat in feats:
        xs, ys = [], []
        for s, rows in by_scene.items():
            x = np.array([r[feat] for r in rows], dtype=np.float64)
            y = np.array([r["marginal_miou"] for r in rows], dtype=np.float64)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 5 or x[ok].std() == 0:
                continue
            xs.extend(((x[ok] - x[ok].mean()) / x[ok].std()).tolist())
            ys.extend(((y[ok] - y[ok].mean()) / (y[ok].std() + 1e-12)).tolist())
        xs, ys = np.array(xs), np.array(ys)
        print(f"  {feat:18s}: pearson {np.corrcoef(xs, ys)[0,1]:+.3f}   "
              f"spearman {spearmanr(xs, ys).statistic:+.3f}   (n={len(xs)})")
    print(f"\nfeatures -> {args.out}")


if __name__ == "__main__":
    main()
