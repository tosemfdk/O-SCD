# O-SCD Phase 0 Baseline — Reproduction & Discrepancy Report

Date: 2026-07-14
Commit: `3abfeaaaef112ba9666b03993edf7d5a91fcaf74` | Command: `bash run_oscd.sh` (unmodified) | Hardware: RTX A6000 48GB (paper: RTX 4090)

## Headline result — H0 confirmed

20-instance PASLCD averages (per-mask mIoU/F1 via `utils/evaluate.py`, official protocol):

| Setting | Repro mIoU | Paper mIoU | Δ | Repro F1 | Paper F1 | Δ |
|---|---:|---:|---:|---:|---:|---:|
| Online (`change_mask/`) | **0.4887** | 0.486 | +0.0027 | **0.6423** | 0.638 | +0.0043 |
| Refined (`change_mask_refined/`) vs paper Offline | **0.5573** | 0.552 | +0.0053 | **0.7009** | 0.694 | +0.0069 |

The small positive gap is consistent with the README statement that the released code
integrates FastGS and gives "slightly better and faster performance than reported."
No settings were changed to force agreement.

Per-scene numbers: `metrics_per_scene.csv`. Summary: `metrics_summary.json`.

## Runtime

- Full benchmark (20 instances, incl. per-scene process startup, SAM2 torch.compile warm-cache, refinement, and evaluation): **10 min 12 s** wall.
- Online loop rate from tqdm: **~4.8–8.4 frames/s per scene (mean ≈ 6.8)** vs paper 11.2 FPS on RTX 4090 — expected hardware gap (A6000 ≈ 40–60% of 4090 throughput); accuracy is the comparable quantity.
- The released code has **no per-module timing instrumentation**; paper Table 2 breakdown cannot be reproduced without adding instrumentation (deferred — prohibited in Phase 0).
- First-run cost not in the numbers above: XFeat + SAM2.1-hiera-tiny checkpoint downloads and cold `torch.compile` (~2.5 min once; inductor cache reused afterwards).

## Nondeterminism quantification

Instance_1/Garden online mIoU across 4 runs, identical command and seeds (seeds hard-coded to 0 in `oscd.py:70-74`):

| Run | mIoU |
|---|---:|
| Smoke run (isolated process) | 0.491831 |
| Full-benchmark run (in 20-scene sequence) | 0.487470 |
| Repeat 2 (isolated) | 0.491821 |
| Repeat 3 (isolated) | 0.491825 |

Isolated runs are reproducible to ~1e-5. The in-sequence run differs by ~0.004 mIoU;
most likely cause is `torch.compile(mode='max-autotune')` kernel selection, which is
timing-benchmark-based and can pick different kernels under different GPU load, plus
nondeterministic atomics in the rasterizer backward. **Budget for ±0.005 mIoU
run-to-run noise at scene level when comparing selectors** (aggregate 20-instance
noise will be smaller).

## Environment deviations from README (none algorithmic)

1. No system CUDA toolkit → CUDA 12.8.93 installed inside the `oscd` conda env (conda-forge), `CUDA_HOME=$CONDA_PREFIX`, gcc 13.4 (conda) for extension builds, `TORCH_CUDA_ARCH_LIST=8.6`.
2. CUDA extensions built with `pip install --no-build-isolation` (setup.py imports torch at build time; README's plain `pip install -r requirements.txt` fails under PEP-517 build isolation).
3. `verlab_accelerated_features` added to `~/.cache/torch/hub/trusted_list` to bypass the interactive torch.hub trust prompt (XFeat download in `scene/dense_extractor.py:29`).
4. Resolved package versions recorded in `pip_freeze.txt` (torch 2.11.0+cu128, transformers 4.56.1 as pinned, opencv 5.0.0, torchmetrics 1.9.0). README does not pin torch; drift here is a standing reproduction risk.

## Checkpoints (downloaded at first run, not in repo)

- XFeat via `torch.hub` `verlab/accelerated_features` (descriptor/dense features)
- SAM2 `facebook/sam2.1-hiera-tiny` via HuggingFace transformers (feature cue)

## Observations relevant to later phases (no action taken)

- Pose failures: **0 / 500 frames** (no failure path triggered in any scene).
- Playground is the weakest scene in both instances (online mIoU 0.36–0.39) and is the only scene where refinement *hurts* (e.g. Inst_2: 0.364 → 0.315). Worth stratifying in budget experiments.
- `--test_hold 5` sets `is_test` flags but `oscd.py` updates `R_change` with **all 25 frames**; the flag does not hold out frames in this pipeline. The Phase 1 all-query-view protocol must implement its own holdout.
- Online masks are rendered immediately after each frame's 16 fusion iterations (order- and history-dependent), confirming the statefulness constraint in the project context.

## Freeze

Baseline outputs live in `output/<Instance>/<Scene>/` (renders + evaluation txt).
Artifacts in `artifacts/baseline/`: source_map.md, commit_and_environment.txt,
pip_freeze.txt, command.txt, metrics_per_scene.csv, metrics_summary.json,
pose_failures.csv, logs/, this report. Freeze tag: `baseline-freeze-phase0`.
