# Phase 1 subset-evaluation runner (see OSCD_Budgeted_View_Project_Context.md §9).
# Mirror of oscd.py with budgeted frame selection; oscd.py itself stays frozen.
#
# Differences vs oscd.py, and nothing else:
#   * --frames_method {all,random,uniform,nbv} --budget K --select_seed S select
#     which inference frames update R_change. Selection uses its own RandomState
#     so the global RNG stream is untouched.
#   * Two-phase since Gate H: Phase A estimates poses for ALL frames, Phase B
#     processes the selected frames (static methods in chronological order; nbv
#     adaptively — each round a Beta-EIG-weighted probe render scores every
#     remaining pose and the frame seeing the most undecided change mass is
#     processed next; selection uses only poses + current change state, never a
#     candidate frame's RGB/cue). NOTE: the two-phase split reorders global RNG
#     consumption vs the pre-Gate-H interleaved loop, so masks are not bitwise
#     comparable with the old garden_sweep_results.csv rows — rerun baselines
#     within the same protocol when comparing selectors.
#   * Outputs: renders/change_mask/ holds online masks for selected frames only
#     (selected-view eval); renders/query_mask/ holds masks rendered from the
#     final R_change at all inference poses (all-query-view eval). Held-out RGB
#     never touches R_change. selection.json records the chosen frames.
#   * The --refine stage is not implemented here (online setting only).

import os
import cv2
import numpy as np
import torch
torch.backends.cuda.benchmark = True
import random
from random import randint
from gaussian_renderer import render, render_change
from scene import GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
import sys
from arguments import ModelParams, PipelineParams, OptimizationParams
from transformers import Sam2Model
from dataloaders.image_dataset import ImageDataset
from poses.feature_detector import Detector
from poses.matcher import Matcher
from poses.pose_initializer import PoseInitializer, get_reference_keyframes
from poses.triangulator import Triangulator
from scene.dense_extractor import DenseExtractor
from scene.keyframe import Keyframe
from scene.cameras import Camera
import json
from utils.camera_utils import camera_to_JSON
from oscd import generate_candidate_map
import warnings
warnings.filterwarnings("ignore")


def parse_args_subset():
    # Same parser as arguments/config_args.py:parse_args, plus selection args.
    parser = ArgumentParser(description="Budgeted subset OSCD runner")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6029)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument('--masks_dir', type=str, default="")
    parser.add_argument('--num_loader_threads', type=int, default=8)
    parser.add_argument('--test_hold', type=int, default=-1)
    parser.add_argument('--use_colmap_poses', action='store_true')
    parser.add_argument('--eval_poses', action='store_true')
    parser.add_argument('--refine', action='store_true')
    parser.add_argument('--pyr_levels', type=int, default=1)
    parser.add_argument('--num_kpts', type=int, default=int(4096*1.5))
    parser.add_argument('--match_max_error', type=float, default=2e-3)
    parser.add_argument('--fundmat_samples', type=int, default=2000)
    parser.add_argument('--min_num_inliers', type=int, default=100)
    parser.add_argument('--fix_focal', action='store_true')
    parser.add_argument('--init_focal', type=float, default=-1.0)
    parser.add_argument('--init_fov', type=float, default=-1.0)
    parser.add_argument('--num_keyframes_for_triangulation', type=int, default=8)
    parser.add_argument('--num_reference_keyframes', type=int, default=4)
    parser.add_argument('--pnpransac_samples', type=int, default=2000)
    parser.add_argument('--num_pts_miniba_incr', type=int, default=2000)
    parser.add_argument('--iters_miniba_incr', type=int, default=20)
    # Selection args
    parser.add_argument('--frames_method', type=str, default='all',
                        choices=['all', 'random', 'uniform', 'nbv', 'nbv_dopt', 'manual'],
                        help="How to select which inference frames update R_change")
    parser.add_argument('--frames_list', type=str, default='',
                        help="manual: comma-separated image names (no extension) or indices")
    parser.add_argument('--nbv_dopt_damping', type=float, default=1.0,
                        help="nbv_dopt: prior damping of the per-Gaussian 3x3 H_g")
    parser.add_argument('--budget', type=int, default=-1,
                        help="Number of update frames K (ignored for method=all)")
    parser.add_argument('--select_seed', type=int, default=0,
                        help="Seed for the random selector (independent of global RNG)")
    parser.add_argument('--nbv_count_scale', type=float, default=0.05,
                        help="nbv: Beta pseudo-count scale for responsibility counts")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    return args, lp, op, pp


def select_frames(method: str, n_frames: int, budget: int, seed: int) -> list[int]:
    if method == 'all':
        return list(range(n_frames))
    assert 0 < budget <= n_frames, f"budget must be in [1, {n_frames}] for method={method}"
    if method == 'uniform':
        idx = np.linspace(0, n_frames - 1, budget).round().astype(int)
        selected = sorted(set(idx.tolist()))
        assert len(selected) == budget, "uniform selection collapsed duplicate indices"
        return selected
    if method == 'random':
        rng = np.random.RandomState(seed)
        return sorted(rng.choice(n_frames, budget, replace=False).tolist())
    raise ValueError(method)


def main(dataset: Namespace, opt: Namespace, pipe: Namespace, args: Namespace):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)

    gaussians_rgb = GaussianModel(dataset.sh_degree, dataset.sh_degree)
    gaussians_rgb.load_ply(os.path.join(args.source_path, "reference_reconstruction", "point_cloud", "iteration_30000", "point_cloud.ply"))
    gaussians_rgb.training_setup(opt)

    gaussians_change = GaussianModel(dataset.sh_degree, 0)
    gaussians_change.load_ply_change(os.path.join(args.source_path, "reference_reconstruction", "point_cloud", "iteration_30000", "point_cloud.ply"))
    gaussians_change.training_setup_change(opt)

    dataset = ImageDataset(args, instance='inf')

    height, width = dataset.get_image_size()

    reference_dataset = ImageDataset(args, instance='ref')

    if args.frames_method in ("nbv", "nbv_dopt", "manual"):
        selected = []  # nbv*: chosen adaptively; manual: resolved after Phase A
    else:
        selected = select_frames(args.frames_method, len(dataset), args.budget, args.select_seed)

    max_error = max(args.match_max_error * width, 1.5)
    matcher = Matcher(args.fundmat_samples, max_error)
    triangulator = Triangulator(
        args.num_kpts, args.num_keyframes_for_triangulation, max_error
    )
    pose_initializer = PoseInitializer(
        width, height, triangulator, matcher, max_error, args
    )
    dense_extractor = DenseExtractor(width, height)
    detector = Detector(args.num_kpts, width, height)

    renders_path = os.path.join(args.model_path, "renders")
    os.makedirs(renders_path, exist_ok=True)
    os.makedirs(os.path.join(renders_path, "change_mask"), exist_ok=True)
    os.makedirs(os.path.join(renders_path, "query_mask"), exist_ok=True)

    reference_keyframes = []

    pbar_ref = tqdm(range(0, len(reference_dataset)), desc="Loading reference keyframes", disable=True)
    for frameID in pbar_ref:
        image, info = reference_dataset.getnext()
        desc_kpts = detector(image)

        Rt = info["Rt"]
        f = info["focal"]
        Fovx = info["FovX"]
        Fovy = info["FovY"]

        keyframe = Keyframe(
            image,
            info,
            desc_kpts,
            Rt,
            frameID,
            f,
            dense_extractor,
            triangulator,
            args,
        )
        reference_keyframes.append(keyframe)

    n_ref_cams = len(reference_keyframes)
    pose_initializer.init_focal(f)
    for i in range(n_ref_cams):
            for j in range(i + 1, n_ref_cams):
                _ = matcher(
                    reference_keyframes[i].desc_kpts,
                    reference_keyframes[j].desc_kpts,
                    remove_outliers= True,
                    update_kpts_flag="inliers", kID=i, kID_other=j)

    for frameID in pbar_ref:
        keyframe = reference_keyframes[frameID]
        keyframe.update_3dpts(reference_keyframes)

    cam_centers = [v.get_centre() for v in reference_keyframes]
    pbar_ref.close()

    viewpoints = []
    all_views = []
    change_masks = {}

    model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-tiny").half().to("cuda")
    model.get_image_embeddings = torch.compile(model.get_image_embeddings, mode='max-autotune')

    patch_size = 14

    background = torch.tensor([0,0,0], dtype=torch.float32, device="cuda")
    total_iterations = 0
    ema_loss_for_log = 0.0

    dummy_image = torch.zeros((3, height, width), device="cuda").half()
    for dummy_id in range(min(10, len(dataset))):
        with torch.no_grad():
            candidate_map = generate_candidate_map(dummy_image, dummy_image, model, patch_size, height, width)

    # ---- Phase A: pose estimation for EVERY frame (needed for query-view
    # evaluation and for candidate scoring; the frame's change cue is NOT
    # computed here, so unprocessed frames contribute pose only).
    pbar_pose = tqdm(range(0, len(dataset)), desc="Estimating poses")
    for frameID in pbar_pose:
        image, info = dataset.getnext()
        desc_kpts = detector(image)
        prev_keyframes = get_reference_keyframes(
            n=args.num_reference_keyframes, keyframes=reference_keyframes, matcher=matcher, desc_kpts=desc_kpts
        )
        Rt, _ = pose_initializer.init_inference_pose(
            prev_keyframes, desc_kpts, frameID, False, image
        )

        R = Rt[:3, :3].cpu().numpy()
        t = Rt[:3, 3].cpu().numpy()
        uid = f"{frameID:04d}"
        image_name = info["name"].split(".")[0]

        view = Camera(colmap_id=uid, R=np.transpose(R), T=t, FoVx=Fovx, FoVy=Fovy,
                     image=image[:3, ...], gt_alpha_mask=None,
                     image_name=image_name, uid=uid)

        cam_centers.append(view.camera_center)
        all_views.append(view)
    pbar_pose.close()

    if args.frames_method == "manual":
        req = [s.strip() for s in args.frames_list.split(",") if s.strip()]
        assert req, "manual needs --frames_list (image names without extension, or indices)"
        name_to_idx = {v.image_name: i for i, v in enumerate(all_views)}
        selected = []
        for r in req:
            if r in name_to_idx:
                selected.append(name_to_idx[r])
            elif r.isdigit() and int(r) < len(all_views):
                selected.append(int(r))
            else:
                raise ValueError(f"unknown frame {r!r}; known: {sorted(name_to_idx)}")
        selected = sorted(set(selected))

    # ---- Phase B: process the selected frames (cue + 16 fusion iterations).
    # Static methods (uniform/random/all/manual) process their precomputed set
    # in chronological order; nbv picks the next frame adaptively.
    def process_view(view, pbar_inf):
        nonlocal total_iterations, ema_loss_for_log
        with torch.no_grad():
            render_pkg = render(view, gaussians_rgb, pipe, background)
            image_rgb = render_pkg["render"]
            original_image = view.original_image[:3, ...]
            candidate_map = generate_candidate_map(original_image, image_rgb, model, patch_size, height, width)
            view.candidate_map = candidate_map.detach().clone()
            viewpoints.append(view)

        for iteration in range(16):
            total_iterations += 1
            if np.random.rand() > 0.33:
                keyframe_idx = randint(0, len(viewpoints)-1)
            else:
                keyframe_idx = -1

            viewpoint = viewpoints[keyframe_idx]
            gaussians_change.update_learning_rate(total_iterations)

            render_pkg_change = render_change(viewpoint, gaussians_change, pipe, background)
            change_mask, viewspace_point_tensor, visibility_filter, radii = render_pkg_change["render"], render_pkg_change["viewspace_points"], render_pkg_change["visibility_filter"], render_pkg_change["radii"]

            gt_change = viewpoint.candidate_map
            change_mask = torch.sigmoid(change_mask.mean(dim=0, keepdim=True))

            d_loss = (gt_change*(1.0-change_mask)).mean()
            d_reg = torch.log(change_mask.mean()**2 + 1.0)

            loss = d_loss + d_reg
            loss.backward()

            gaussians_change.optimizer.step()
            gaussians_change.optimizer.zero_grad(set_to_none = True)

            with torch.no_grad():
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if total_iterations % 5 == 0:
                    pbar_inf.set_postfix({
                        "it": total_iterations,
                        "loss": f"{ema_loss_for_log:.4f}",
                    })

                gaussians_change.max_radii2D[visibility_filter] = torch.max(gaussians_change.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians_change.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if iteration in [4]:
                    grads = gaussians_change.xyz_gradient_accum / gaussians_change.denom
                    grads[grads.isnan()] = 0.0

                    gaussians_change.tmp_radii = radii
                    if iteration == 4:
                        scene_center = torch.stack(cam_centers, dim=0).mean(dim=0)
                        extent = torch.max(torch.linalg.norm(torch.stack([v for v in cam_centers], dim=0) - scene_center.unsqueeze(0), dim=-1)).item() * 1.1

                    gaussians_change.densify_and_clone(grads, opt.densify_grad_threshold*5, extent)
                    gaussians_change.densify_and_split(grads, opt.densify_grad_threshold*5, extent)

                    gaussians_change.tmp_radii = None
                    torch.cuda.empty_cache()


        with torch.no_grad():
            render_pkg_change = render_change(view, gaussians_change, pipe, background)
            change_mask = render_pkg_change["render"]
            change_mask = change_mask.mean(dim=0)
            change_mask = (change_mask > 0.5).float()
            change_masks[view.image_name] = change_mask

    if args.frames_method in ("nbv", "nbv_dopt"):
        # Adaptive selection (uses ONLY poses + current change state, never a
        # candidate frame's RGB/cue).
        #   nbv:      direction-blind — one Beta-EIG-weighted probe render per
        #             remaining frame, pick the most undecided change mass.
        #   nbv_dopt: direction-aware — per-Gaussian 3x3 mean-block D-opt gain
        #             with the (I - r r^T)/d^2 viewing-geometry FIM, Beta
        #             ambiguity as weights (joint geometry x change mode).
        # After processing, exact responsibility-weighted soft counts of the
        # OBSERVED cue update the Beta state (and H_g for nbv_dopt).
        from target_nbv.change.beta_state import BetaChangeState
        from target_nbv.change.counts import responsibility_probe_render, soft_counts
        from target_nbv.change.dopt_scorer import DoptFrameState

        assert 0 < args.budget <= len(all_views), "nbv needs --budget in [1, n_frames]"
        beta = BetaChangeState(gaussians_change.get_xyz.shape[0],
                               pseudo_count_scale=args.nbv_count_scale)
        dopt = (DoptFrameState(gaussians_change.get_xyz.shape[0],
                               damping=args.nbv_dopt_damping)
                if args.frames_method == "nbv_dopt" else None)
        remaining = list(range(len(all_views)))
        selected = []
        pbar_inf = tqdm(range(args.budget),
                        desc=f"Running OSCD subset ({args.frames_method}, K={args.budget})")
        for _ in pbar_inf:
            with torch.no_grad():
                w = beta.unit_eig().float()
                w = w / max(float(w.max()), 1e-12)
                if dopt is None:
                    scores = {i: float(responsibility_probe_render(
                                  gaussians_change, all_views[i], w, pipe).sum())
                              for i in remaining}
            if dopt is not None:  # needs autograd for the tau adjoint
                scores = {i: dopt.score_frame(gaussians_change, all_views[i], w, pipe)
                          for i in remaining}
            pick = max(scores, key=scores.get)
            remaining.remove(pick)
            selected.append(pick)
            process_view(all_views[pick], pbar_inf)

            grown = gaussians_change.get_xyz.shape[0] - len(beta)
            if grown > 0:  # densify clone/split during fusion
                beta.append(grown)
                if dopt is not None:
                    dopt.append(grown)
            M = all_views[pick].candidate_map.squeeze(0).clamp(0.0, 1.0)
            e1, e0, tau = soft_counts(gaussians_change, all_views[pick], M, pipe)
            beta.update(e1, e0)
            if dopt is not None:
                dopt.commit(gaussians_change, all_views[pick], tau)
        pbar_inf.close()
        processing_order = list(selected)
        selected = sorted(selected)
    else:
        pbar_inf = tqdm(selected, desc=f"Running OSCD subset ({args.frames_method}, K={len(selected)})")
        for frameID in pbar_inf:
            process_view(all_views[frameID], pbar_inf)
        pbar_inf.close()
        processing_order = list(selected)

    for key in change_masks:
        cv2.imwrite(os.path.join(renders_path, "change_mask", f"{key}.png"), (change_masks[key].cpu().numpy() * 255).astype(np.uint8))

    # All-query-view evaluation: render the final R_change at every inference pose.
    # Held-out frames contribute only their pose here; their RGB/cues never updated R_change.
    with torch.no_grad():
        for view in all_views:
            render_pkg_change = render_change(view, gaussians_change, pipe, background)
            change_mask = render_pkg_change["render"].mean(dim=0)
            change_mask = (change_mask > 0.5).float()
            cv2.imwrite(os.path.join(renders_path, "query_mask", f"{view.image_name}.png"), (change_mask.cpu().numpy() * 255).astype(np.uint8))

    with open(os.path.join(args.model_path, "selection.json"), 'w') as f:
        json.dump({
            "frames_method": args.frames_method,
            "budget": len(selected),
            "select_seed": args.select_seed,
            "processing_order": processing_order,
            "selected_indices": selected,
            "selected_names": [all_views[i].image_name for i in selected],
        }, f, indent=4)

    cameras_json = []
    for idx, view in enumerate(all_views):
        cameras_json.append(camera_to_JSON(idx, view))

    with open(os.path.join(args.model_path, "cameras.json"), 'w') as f:
        json.dump(cameras_json, f, indent=4)


if __name__ == "__main__":
    args, lp, op, pp = parse_args_subset()
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    main(lp.extract(args), op.extract(args), pp.extract(args), args)
