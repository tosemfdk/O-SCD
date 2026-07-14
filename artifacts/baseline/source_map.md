# O-SCD Phase 0 Repository Audit — Source Map

Audit date: 2026-07-14
Repository commit: `3abfeaaaef112ba9666b03993edf7d5a91fcaf74` (branch `main`)
Submodules: none registered in git (`git submodule status` empty); `submodules/` contains vendored source trees:
`diff-gaussian-rasterization`, `diff-gaussian-rasterization_fastgs`, `fused-ssim`, `simple-knn`
(requirements.txt builds `diff-gaussian-rasterization_fastgs`, `fused-ssim`, `simple-knn`; the non-fastgs rasterizer is present but not installed).

## Pipeline stage → code mapping (Section 5 of project context)

| Stage | Location | Notes |
|---|---|---|
| Entry point / main loop | `oscd.py:69` `main()`; per-frame loop `oscd.py:175` | seeds fixed to 0 at `oscd.py:70-74` |
| Reference 3DGS loading (`R_ref`) | `oscd.py:76-78`; `scene/gaussian_model.py:309` `load_ply()` | loads prebuilt `<scene>/reference_reconstruction/point_cloud/iteration_30000/point_cloud.ply` — **assets are provided in PASLCD_online, no rebuild needed** |
| `R_change` initialization | `oscd.py:80-82`; `scene/gaussian_model.py:353` `load_ply_change()`, `:202` `training_setup_change()` | same PLY, SH degree 0, learnable change values |
| Inference dataset loading | `dataloaders/image_dataset.py:31` `ImageDataset` (`instance='inf'`) | frame order = **sorted filenames** (`image_dataset.py:40`); threaded preloading; `test_hold` flag marks every 5th frame `is_test` (`:70`) but oscd.py processes all 25 frames |
| Reference dataset + keyframes | `ImageDataset(instance='ref')` + `scene/keyframe.py` `Keyframe`; ref keyframe setup `oscd.py:109-149` | 60 reference images with COLMAP poses from `reference_scene/sparse/0` |
| Descriptor extraction | `poses/feature_detector.py:90` `Detector`, `__call__` at `:182` | called per inference frame at `oscd.py:178` |
| Reference image retrieval | `poses/pose_initializer.py:128` `get_reference_keyframes()` | called at `oscd.py:179`, top `args.num_reference_keyframes` |
| PnP/RANSAC pose estimation + refinement | `poses/pose_initializer.py:55` `init_inference_pose()`; PnP RANSAC at `:90` (`poses/ransac.py` `RANSACEstimator`, P4P) | called at `oscd.py:182` |
| Aligned reference rendering | `gaussian_renderer/__init__.py:18` `render()` | called at `oscd.py:199` on `gaussians_rgb` |
| Pixel + feature change cues | `oscd.py:30` `generate_candidate_map()` | pixel: `0.8*L1 + 0.2*(1-SSIM)` (`oscd.py:34`, matches paper λ=0.2); feature: SAM2 `facebook/sam2.1-hiera-tiny` embedding abs-diff (`oscd.py:47-52`), model loaded `oscd.py:154`, `torch.compile(mode='max-autotune')` at `:155`; cues summed at `:65` |
| Self-supervised fusion update | `oscd.py:207-255` | **16 iterations/frame** (`:207`); newest-frame bias: P(newest)≈0.33 else uniform history (`:209-212`); loss = `d_loss + d_reg` = `(cue*(1-sigmoid(mask))).mean() + log(mask.mean()^2+1)` (`:223-226`); densify clone/split at iteration 4 (`:242-252`) |
| Change-mask rendering | `gaussian_renderer/__init__.py:114` `render_change()`; per-frame mask at `oscd.py:258-263` | threshold: `mean(dim=0) > 0.5` |
| Offline post-refinement (`--refine`) | `oscd.py:267-307` (up to 3000 total iters); refined masks re-rendered `oscd.py:312-319` | outputs to `renders/change_mask_refined/` |
| Mask output | `oscd.py:309-310` → `<model_path>/renders/change_mask/<image_name>.png` | |
| Evaluation (mIoU/F1) | `utils/evaluate.py:36` `evaluate_segmentation()`; torchmetrics `JaccardIndex`/`F1Score` (binary), per-mask then averaged; binarize threshold 127/255 | invoked by `run_oscd.sh:34,40` |
| Config parsing | `arguments/config_args.py` `parse_args()`; group defs `arguments/__init__.py` | |
| Seed handling | `oscd.py:70-74` (all seeds = 0) + `utils/general_utils.py` `safe_state()` | |
| Runtime/FPS measurement | **not found in oscd.py** — no per-module timing instrumentation in the released code | paper Table 2 timing not directly reproducible from released scripts |

## Official commands

- Full benchmark (paper online setting + offline refine): `bash run_oscd.sh` — iterates `Instance_{1,2} × {Cantina,Garden,Lounge,Lunch_room,Meeting_room,Playground,Porch,Pots,Printing_area,Zen}` = 20 instances.
- Single scene equivalent:
  ```
  python oscd.py -s data/PASLCD/Instance_1/Garden/ -m output/Instance_1/Garden/ --resolution 4 --test_hold 5 --refine
  python utils/evaluate.py --gt data/PASLCD/Instance_1/Garden/gt_mask/ --pred_binary output/Instance_1/Garden/renders/change_mask/
  ```
- `renders/change_mask/` = online masks (rendered immediately after each frame's 16 fusion iterations) → compare to paper "Online" row. `renders/change_mask_refined/` = after 3000-iteration offline refinement → compare to paper "Offline" row.

## Dataset layout (verified, `data/PASLCD/`, 15 GB, untracked in git)

```
Instance_{1,2}/<Scene>/
  reference_scene/images/          # 60 pre-change images + sparse/0 COLMAP model
  reference_reconstruction/        # prebuilt 3DGS: point_cloud/iteration_30000/point_cloud.ply, cameras.json, cfg_args
  inference_scene/images/          # 25 post-change frames (candidate pool)
  gt_mask/                         # 25 GT change masks (PNG, matched by basename)
```

## Answers to open questions (context doc §17)

1. Both: online mask saved per frame during the stream; refined mask evaluated separately when `--refine`.
2. Yes — exactly 16 fusion iterations (`oscd.py:207`); newest-frame sampling bias present (~33% newest).
3. All 25 inference poses estimated online via PnP-RANSAC against reference keyframes; not precomputed.
4. Reference 3DGS assets are provided (`reference_reconstruction/`); no rebuild required.
5. Pose failures: 0/500 frames in the full baseline run (2026-07-14).
6. Frame order deterministic: sorted image filenames.
7. Yes — `render_change()` needs only a `Camera` (pose+intrinsics), not RGB content.
8. Rasterizer is `diff-gaussian-rasterization_fastgs`; differentiable params per standard 3DGS (xyz, scaling, rotation, opacity, features). Detail TBD in Phase 2.
9. No FisherRF/POp-GS-style Jacobian path found in the codebase.
10. `utils/evaluate.py` with torchmetrics binary Jaccard/F1, mask binarization at 127, per-mask average over 25 GT masks per instance, then average over 20 instances.

## Environment notes / deviations

- Paper hardware: RTX 4090. This machine: **RTX A6000 48GB**, driver 580.142 → FPS not directly comparable (accuracy should be).
- README notes the released code incorporates FastGS: "slightly better and faster performance than reported."
- No system CUDA toolkit (`nvcc` missing) → installed via conda env (see environment capture).
- System gcc 11.4, Ubuntu 22.04, 62 GB RAM.
