# Phase 1 subset-evaluation runner (see OSCD_Budgeted_View_Project_Context.md §9).
# Mirror of oscd.py with budgeted frame selection; oscd.py itself stays frozen.
#
# Differences vs oscd.py, and nothing else:
#   * --frames_method {all,random,uniform} --budget K --select_seed S select which
#     inference frames update R_change. Selection uses its own RandomState so the
#     global RNG stream is untouched; with --frames_method all the update path is
#     call-for-call identical to oscd.py (reproduces the frozen baseline).
#   * Frames are always processed in chronological (sorted-filename) order.
#     Pose estimation runs for every frame (needed for query-view evaluation);
#     change cues and fusion updates run only for selected frames.
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
                        choices=['all', 'random', 'uniform'],
                        help="How to select which inference frames update R_change")
    parser.add_argument('--budget', type=int, default=-1,
                        help="Number of update frames K (ignored for method=all)")
    parser.add_argument('--select_seed', type=int, default=0,
                        help="Seed for the random selector (independent of global RNG)")

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

    selected = select_frames(args.frames_method, len(dataset), args.budget, args.select_seed)
    selected_set = set(selected)

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

    pbar_inf = tqdm(range(0, len(dataset)), desc=f"Running OSCD subset ({args.frames_method}, K={len(selected)})")

    for frameID in pbar_inf:

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

        if frameID not in selected_set:
            continue

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

    pbar_inf.close()

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
            "processing_order": "chronological",
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
