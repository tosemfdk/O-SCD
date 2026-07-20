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
                        choices=['all', 'random', 'uniform', 'nbv', 'nbv_dopt',
                                 'manual', 'dopt_pose', 'dopt_seq',
                                 'kf_g_dir', 'kf_l_dir_gseed', 'kf_gu_dir',
                                 'kf_gl_dir', 'kf_glu_dir', 'kf_glu_nodir'],
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
    # ChangeNBV (view_selection) args — dopt_pose, batch mode (M1)
    parser.add_argument('--select_mode', type=str, default='batch',
                        choices=['batch'],
                        help="dopt_pose: selection protocol (M1 batch only for now)")
    parser.add_argument('--info_output', type=str, default='raw',
                        choices=['raw', 'sigmoid'])
    parser.add_argument('--info_probes', type=int, default=4)
    parser.add_argument('--info_criterion', type=str, default='dopt',
                        choices=['dopt', 'trace_reduction', 'fisher_ratio',
                                 'candidate_only', 'dopt_dir'])
    parser.add_argument('--info_weight', type=str, default='pose',
                        choices=['pose', 'current_map'],
                        help="dopt_seq: pixel weight for candidate scoring")
    parser.add_argument('--info_lambda_rel', type=float, default=1e-3)
    parser.add_argument('--info_lambda_abs', type=float, default=1e-8)
    parser.add_argument('--info_alpha_threshold', type=float, default=0.5)
    parser.add_argument('--info_cache_root', type=str,
                        default='outputs/change_nbv/cache')
    # Global-local keyframe selection args (Part 2 cycle 2, kf_* methods).
    # Claim scope is offline_pool_keyframe_selection: R_global consumed all 25
    # inference images before selection, so these are NOT active-NBV methods.
    parser.add_argument('--keyframe_budget', type=int, default=-1,
                        help="kf_*: number of keyframes (falls back to --budget)")
    parser.add_argument('--global_context_source', type=str,
                        default='all25_clean', choices=['all25_clean'],
                        help="kf_*: recipe that produced R_global (only the "
                             "standard all-25 pipeline is allowed)")
    parser.add_argument('--global_context_checkpoint', type=str, default='',
                        help="kf_*: explicit r_global.ply path (bypasses cache "
                             "lookup)")
    parser.add_argument('--global_context_seed', type=int, default=-1,
                        help="kf_*: train_seed of the R_global build "
                             "(-1 = same as --train_seed)")
    parser.add_argument('--global_context_root', type=str,
                        default='outputs/change_nbv/global_context')
    parser.add_argument('--local_rebuild_each_round', action='store_true',
                        default=True,
                        help="kf_*: rebuild R_local(S) from the reference "
                             "checkpoint every round (always on; incremental "
                             "updates are a future speed ablation)")
    parser.add_argument('--keyframe_rank_fusion', type=str,
                        default='percentile_mean', choices=['percentile_mean'])
    parser.add_argument('--save_round_models', action='store_true',
                        help="kf_*: persist per-round R_local checkpoints and "
                             "component masks")
    parser.add_argument('--selection_only', action='store_true',
                        help="kf_*: stop after writing selection_manifest.json "
                             "(final metrics come from the driver's clean "
                             "replay through the standard manual path)")
    # Replicate / checkpoint args (Part 2 cycle 2)
    parser.add_argument('--train_seed', type=int, default=0,
                        help="0 = legacy RNG stream (bitwise-identical to "
                             "historical runs); nonzero reseeds torch/np/random "
                             "right after Phase A so poses stay fixed across "
                             "seeds and only fusion/selection randomness varies")
    parser.add_argument('--save_change_model', action='store_true',
                        help="save the final change model (r_change.ply, c "
                             "preserved) plus per-pose soft change masks and "
                             "reference alpha masks under model_path — used by "
                             "the all-25 global-context builder")

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

    if (args.frames_method in ("nbv", "nbv_dopt", "manual", "dopt_pose",
                               "dopt_seq")
            or args.frames_method.startswith("kf_")):
        selected = []  # nbv*/dopt_seq/kf_*: adaptive; manual/dopt_pose: post-Phase A
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

    # Replicate seeding (Part 2 cycle 2): reseed AFTER Phase A so poses are
    # identical across train seeds and only Phase B fusion/selection randomness
    # varies. train_seed 0 keeps the legacy stream untouched (bitwise-identical
    # to all historical runs).
    if args.train_seed != 0:
        torch.manual_seed(args.train_seed)
        torch.cuda.manual_seed(args.train_seed)
        torch.cuda.manual_seed_all(args.train_seed)
        np.random.seed(args.train_seed)
        random.seed(args.train_seed)

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

    if args.frames_method == "dopt_pose":
        # ChangeNBV M1 (batch-static): score all poses on the REFERENCE scoring
        # state (R_change is still c = 0 here — no fusion has run), greedy
        # select, then fall through to the standard chronological replay below.
        # FrameAccessGuard inside makes any candidate-image access fail loudly.
        from view_selection.m1 import select_dopt_pose
        from view_selection.types import InformationConfig
        assert 0 < args.budget <= len(all_views), "dopt_pose needs --budget"
        cfg = InformationConfig(
            output_space=args.info_output, weight_mode="pose",
            num_probes=args.info_probes,
            alpha_threshold=args.info_alpha_threshold,
            lambda_rel=args.info_lambda_rel, lambda_abs=args.info_lambda_abs)
        selected, _manifest = select_dopt_pose(
            gaussians_change, gaussians_rgb, all_views, pipe, background,
            cfg, args.budget, args.info_criterion,
            scene_id=os.path.basename(os.path.normpath(args.source_path)),
            checkpoint_path=os.path.join(
                args.source_path, "reference_reconstruction", "point_cloud",
                "iteration_30000", "point_cloud.ply"),
            cache_root=args.info_cache_root,
            manifest_path=os.path.join(args.model_path, "selection_manifest.json"))

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

    if args.frames_method == "dopt_seq":
        # User-requested recipe: learn the change scene from the FIRST frame,
        # then repeatedly pick the pose-known candidate whose observation most
        # constrains the CURRENT change scene (D-opt on the change channel,
        # recomputed on the evolving model), fuse it, repeat until K frames.
        from view_selection.criteria import score_candidate
        from view_selection.information import (derive_relative_lambda,
                                                hutchinson_information)
        from view_selection.types import InformationConfig
        from view_selection.weights import build_pixel_weight

        assert 0 < args.budget <= len(all_views), "dopt_seq needs --budget"
        cfg = InformationConfig(
            output_space=args.info_output, weight_mode=args.info_weight,
            num_probes=args.info_probes,
            alpha_threshold=args.info_alpha_threshold,
            lambda_rel=args.info_lambda_rel, lambda_abs=args.info_lambda_abs)
        scene_tag = os.path.basename(os.path.normpath(args.source_path))
        pbar_inf = tqdm(range(args.budget),
                        desc=f"Running OSCD subset (dopt_seq, K={args.budget})")
        selected = [0]                      # seed: chronologically first frame
        process_view(all_views[0], pbar_inf)
        pbar_inf.update(1)
        remaining = list(range(1, len(all_views)))
        step_log = []
        while len(selected) < args.budget:
            if args.info_criterion == "dopt_dir":
                # Direction-aware: per-Gaussian 3x3 POSITION Fisher blocks
                # (direction lives in the xyz Jacobian), D-opt logdet gain
                # weighted by per-Gaussian suspicion = current change mass c.
                # "Revisit what looks changed — from a NEW angle."
                from view_selection.information import (
                    block_dopt_gain, hutchinson_block_information)
                blocks = {}
                for i in selected + remaining:
                    w = build_pixel_weight(cfg.weight_mode, gaussians_rgb,
                                           all_views[i], pipe, background,
                                           cfg, model_change=gaussians_change)
                    blocks[i] = hutchinson_block_information(
                        gaussians_change, all_views[i], w, cfg,
                        (scene_tag, "seq", len(selected)), pipe, background,
                        frame_id=i)
                with torch.no_grad():
                    c = gaussians_change._features_dc.detach()
                    susp = c.reshape(c.shape[0], -1).mean(dim=1).clamp_min(0.0)
                    susp = susp / susp.sum().clamp_min(1e-12)
                tr = torch.stack([torch.diagonal(b, dim1=1, dim2=2).sum(dim=1)
                                  for b in blocks.values()])
                pos = tr[tr > 0]
                lam = max(cfg.lambda_abs,
                          cfg.lambda_rel * float(pos.median()) / 3.0
                          if pos.numel() else cfg.lambda_abs)
                H = torch.zeros_like(blocks[selected[0]])
                for s in selected:
                    H = H + blocks[s]
                scores = {i: block_dopt_gain(H, blocks[i], susp, lam)
                          for i in remaining}
                del blocks
                torch.cuda.empty_cache()
            else:
                infos = {}
                for i in selected + remaining:  # b on the CURRENT change model
                    w = build_pixel_weight(cfg.weight_mode, gaussians_rgb,
                                           all_views[i], pipe, background, cfg,
                                           model_change=gaussians_change)
                    infos[i] = hutchinson_information(
                        gaussians_change, all_views[i], w, cfg,
                        (scene_tag, "seq", len(selected)), pipe, background,
                        frame_id=i, strict=False).diagonal.cuda()
                lam = derive_relative_lambda(infos.values(), cfg.lambda_rel,
                                             cfg.lambda_abs)
                h = torch.full_like(infos[selected[0]], float(lam))
                for s in selected:
                    h = h + infos[s]
                scores = {i: score_candidate(args.info_criterion, h, infos[i])
                          for i in remaining}
            pick = min((-v, i) for i, v in scores.items())[1]
            step_log.append({"step": len(selected), "pick": pick,
                             "score": scores[pick], "lambda": lam})
            remaining.remove(pick)
            selected.append(pick)
            process_view(all_views[pick], pbar_inf)
            pbar_inf.update(1)
        pbar_inf.close()
        with open(os.path.join(args.model_path, "dopt_seq_steps.json"), "w") as f:
            json.dump(step_log, f, indent=2)
        processing_order = list(selected)
        selected = sorted(selected)
    elif args.frames_method in ("nbv", "nbv_dopt"):
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
    elif args.frames_method.startswith("kf_"):
        # Global-local consensus keyframe selection (Part 2 cycle 2).
        # Offline pool setting: R_global was built from ALL 25 inference
        # frames by the standard all-25 pipeline (driver Stage 1) and is only
        # looked up here — never built silently, so its cost stays visible.
        # Selection combines R_global's rendered soft change masks with the
        # clean-rebuilt subset state R_local(S); no GT, no forced first frame,
        # no Gaussian-index comparison across models.
        import csv as _csv
        import time as _time

        from view_selection.global_context import (file_sha256,
                                                   find_cached_context,
                                                   global_context_key,
                                                   load_frozen_change_model)
        from view_selection.global_local_keyframe import select_keyframes_gl
        from view_selection.types import InformationConfig

        budget = args.keyframe_budget if args.keyframe_budget > 0 else args.budget
        assert 0 < budget <= len(all_views), \
            "kf_* needs --keyframe_budget (or --budget)"
        cfg = InformationConfig(
            output_space=args.info_output, weight_mode="pose",
            num_probes=args.info_probes,
            alpha_threshold=args.info_alpha_threshold,
            lambda_rel=args.info_lambda_rel, lambda_abs=args.info_lambda_abs)
        scene_tag = os.path.basename(os.path.normpath(args.source_path))
        instance_tag = os.path.basename(
            os.path.dirname(os.path.normpath(args.source_path)))
        ref_ply = os.path.join(args.source_path, "reference_reconstruction",
                               "point_cloud", "iteration_30000",
                               "point_cloud.ply")

        t_gctx = _time.time()
        gseed = (args.global_context_seed if args.global_context_seed >= 0
                 else args.train_seed)
        if args.global_context_checkpoint:
            gctx_ply = args.global_context_checkpoint
            gctx_cache = "explicit_path"
        else:
            key = global_context_key(
                scene_tag, instance_tag, file_sha256(ref_ply),
                [v.image_name for v in all_views], gseed,
                int(args.resolution), cfg.alpha_threshold,
                repo_root=os.path.dirname(os.path.abspath(__file__)))
            hit = find_cached_context(args.global_context_root, scene_tag,
                                      instance_tag, key)
            if hit is None:
                raise FileNotFoundError(
                    f"no cached R_global for {scene_tag}/{instance_tag} "
                    f"seed {gseed} under {args.global_context_root}; build it "
                    f"first (experiments/run_keyframe_gl_eval.py Stage 1)")
            gctx_ply = os.path.join(hit, "r_global.ply")
            gctx_cache = "hit"
        r_global = load_frozen_change_model(gctx_ply,
                                            gaussians_change.max_sh_degree)
        gctx_hash = file_sha256(gctx_ply)
        gctx_seconds = _time.time() - t_gctx

        def _candidate_map_for(view):
            # Cue computed once per frame, the first time it enters S. Only
            # SELECTED frames ever reach here (scoring runs under
            # FrameAccessGuard, so a candidate's content access would raise).
            if getattr(view, "candidate_map", None) is None:
                with torch.no_grad():
                    image_rgb = render(view, gaussians_rgb, pipe,
                                       background)["render"]
                    view.candidate_map = generate_candidate_map(
                        view.original_image[:3, ...], image_rgb, model,
                        patch_size, height, width).detach().clone()
            return view.candidate_map

        def fuse_frames_clean(model_change, views_sorted, rng):
            # Mirror of process_view's 16-iteration fusion on an explicit
            # model + RNG: fresh optimizer state, chronological order, own lr
            # counter — so R_local depends on the SET S only (spec §2B), not
            # on greedy order or on how often this function ran before.
            viewpoints_local = []
            titer = 0
            extent = None
            for view in views_sorted:
                viewpoints_local.append(view)
                for iteration in range(16):
                    titer += 1
                    if rng.rand() > 0.33:
                        keyframe_idx = int(rng.randint(0, len(viewpoints_local)))
                    else:
                        keyframe_idx = len(viewpoints_local) - 1
                    viewpoint = viewpoints_local[keyframe_idx]
                    model_change.update_learning_rate(titer)
                    pkg = render_change(viewpoint, model_change, pipe, background)
                    change_mask, viewspace_point_tensor = pkg["render"], pkg["viewspace_points"]
                    visibility_filter, radii = pkg["visibility_filter"], pkg["radii"]
                    gt_change = viewpoint.candidate_map
                    change_mask = torch.sigmoid(change_mask.mean(dim=0, keepdim=True))
                    d_loss = (gt_change * (1.0 - change_mask)).mean()
                    d_reg = torch.log(change_mask.mean() ** 2 + 1.0)
                    (d_loss + d_reg).backward()
                    model_change.optimizer.step()
                    model_change.optimizer.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        model_change.max_radii2D[visibility_filter] = torch.max(
                            model_change.max_radii2D[visibility_filter],
                            radii[visibility_filter])
                        model_change.add_densification_stats(
                            viewspace_point_tensor, visibility_filter)
                        if iteration == 4:
                            grads = (model_change.xyz_gradient_accum
                                     / model_change.denom)
                            grads[grads.isnan()] = 0.0
                            model_change.tmp_radii = radii
                            if extent is None:
                                scene_center = torch.stack(cam_centers, dim=0).mean(dim=0)
                                extent = torch.max(torch.linalg.norm(
                                    torch.stack([v for v in cam_centers], dim=0)
                                    - scene_center.unsqueeze(0), dim=-1)).item() * 1.1
                            model_change.densify_and_clone(
                                grads, opt.densify_grad_threshold * 5, extent)
                            model_change.densify_and_split(
                                grads, opt.densify_grad_threshold * 5, extent)
                            model_change.tmp_radii = None
                            torch.cuda.empty_cache()
            return model_change

        run_id = f"ts{args.train_seed}_{_time.strftime('%Y%m%d_%H%M%S')}"
        kf_out_root = os.path.join("outputs", "change_nbv", "keyframe",
                                   scene_tag, args.frames_method, run_id)
        os.makedirs(kf_out_root, exist_ok=True)

        def rebuild_local_fn(S_sorted):
            from view_selection.global_local_keyframe import local_rebuild_seed
            for i in S_sorted:
                _candidate_map_for(all_views[i])
            rng = np.random.RandomState(
                local_rebuild_seed(args.train_seed, S_sorted))
            model_local = GaussianModel(gaussians_change.max_sh_degree, 0)
            model_local.load_ply_change(ref_ply)
            model_local.training_setup_change(opt)
            fuse_frames_clean(model_local,
                              [all_views[i] for i in S_sorted], rng)
            if args.save_round_models:
                rd = os.path.join(kf_out_root,
                                  f"round_{len(S_sorted) + 1:02d}")
                os.makedirs(rd, exist_ok=True)
                model_local.save_ply_change(
                    os.path.join(rd, "local_checkpoint.ply"))
            return model_local

        round_artifact_fn = None
        if args.save_round_models:
            def round_artifact_fn(round_idx, masks):
                rd = os.path.join(kf_out_root, f"round_{round_idx:02d}",
                                  "component_masks")
                os.makedirs(rd, exist_ok=True)
                torch.save({str(i): {k: v.cpu() for k, v in masks[i].items()}
                            for i in masks},
                           os.path.join(rd, "masks.pt"))

        manifest = select_keyframes_gl(
            args.frames_method, all_views, gaussians_rgb, r_global, budget,
            cfg, pipe, background, rebuild_local_fn, scene_tag,
            args.train_seed, round_artifact_fn=round_artifact_fn)
        manifest.update({
            "scene": scene_tag, "instance": instance_tag,
            "global_context_source": args.global_context_source,
            "global_context_checkpoint": os.path.abspath(gctx_ply),
            "global_context_checkpoint_hash": gctx_hash,
            "global_context_seed": gseed,
            "global_context_cache": gctx_cache,
            "global_context_load_seconds": round(gctx_seconds, 2),
            "local_rebuild_each_round": True,
            "selection_output_root": os.path.abspath(kf_out_root),
        })
        for dst in (os.path.join(args.model_path, "selection_manifest.json"),
                    os.path.join(kf_out_root, "selection_manifest.json")):
            with open(dst, "w") as f:
                json.dump(manifest, f, indent=2)
        mass_names = ("global", "local", "unresolved", "overlap", "local_only")
        for rnd in manifest["rounds"]:
            rd = os.path.join(kf_out_root, f"round_{rnd['round']:02d}")
            os.makedirs(rd, exist_ok=True)
            with open(os.path.join(rd, "candidate_scores.csv"), "w",
                      newline="") as f:
                wcsv = _csv.writer(f)
                comps = rnd["components"]
                wcsv.writerow(["candidate"]
                              + [f"raw_{c}" for c in comps]
                              + [f"rank_{c}" for c in comps] + ["total"]
                              + [f"mass_{m_}" for m_ in mass_names])
                for i in rnd["candidates"]:
                    si = str(i)
                    wcsv.writerow(
                        [i] + [rnd["raw_scores"][c][si] for c in comps]
                        + [rnd["percentile_ranks"][c][si] for c in comps]
                        + [rnd["total_score"][si]]
                        + [rnd["mask_mass"][m_][si] for m_ in mass_names])

        processing_order = list(manifest["greedy_order"])
        selected = sorted(processing_order)
        del r_global
        torch.cuda.empty_cache()
        if not args.selection_only:
            # Final evaluation model (spec §2C): all selection working models
            # are discarded; gaussians_change is still the untouched reference
            # state, so a chronological process_view pass here is exactly the
            # standard clean replay.
            pbar_inf = tqdm(selected,
                            desc=f"Running OSCD subset ({args.frames_method} "
                                 f"replay, K={len(selected)})")
            for frameID in pbar_inf:
                process_view(all_views[frameID], pbar_inf)
            pbar_inf.close()
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
    # --selection_only (kf_*): no fusion ran in this process, so query masks
    # would be meaningless — the driver's clean replay produces the metrics.
    if not args.selection_only:
        with torch.no_grad():
            for view in all_views:
                render_pkg_change = render_change(view, gaussians_change, pipe, background)
                change_mask = render_pkg_change["render"].mean(dim=0)
                change_mask = (change_mask > 0.5).float()
                cv2.imwrite(os.path.join(renders_path, "query_mask", f"{view.image_name}.png"), (change_mask.cpu().numpy() * 255).astype(np.uint8))

    if args.save_change_model:
        # Global-context artifacts (Part 2 cycle 2): the final change model with
        # c preserved, plus per-pose soft change masks (sigmoid of the raw
        # channel-mean logit) and reference alpha masks for every inference pose.
        from view_selection.weights import alpha_map
        gaussians_change.save_ply_change(
            os.path.join(args.model_path, "r_change.ply"))
        soft_masks, alpha_masks, names = [], [], []
        with torch.no_grad():
            for view in all_views:
                z = render_change(view, gaussians_change, pipe,
                                  background)["render"].mean(dim=0)
                soft_masks.append(torch.sigmoid(z).cpu())
                alpha_masks.append(
                    alpha_map(gaussians_rgb, view, pipe, background).cpu())
                names.append(view.image_name)
        torch.save({"image_names": names,
                    "soft_masks": torch.stack(soft_masks)},
                   os.path.join(args.model_path,
                                "all25_rendered_soft_masks.pt"))
        torch.save({"image_names": names,
                    "alpha_masks": torch.stack(alpha_masks)},
                   os.path.join(args.model_path, "alpha_masks.pt"))

    with open(os.path.join(args.model_path, "selection.json"), 'w') as f:
        json.dump({
            "frames_method": args.frames_method,
            "budget": len(selected),
            "select_seed": args.select_seed,
            "train_seed": args.train_seed,
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
