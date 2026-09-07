# PASLCD compact/joint checkpoint evidence

Small copies of the completed 2026-09-07 two-scene experiment records. Contents
are unchanged except that `comparison.csv` line endings are normalized to LF.
These remain available in Git even though `outputs/` is intentionally ignored.

- Original artifact root: `outputs/paslcd_2scene_3d2d_compact_joint_20260907/`.
- [Korean report](../../paslcd-two-scene-3d2d-compact-joint-comparison-20260907-ko.md).
- `comparison.json` / `comparison.csv`: mean-frame accuracy and timing.
- `*_timing.json`: counters, per-frame cue/replay hashes, snapshot checks, summaries.
- `*_process_time.json`: measured process wall time and GPU process context.
- `plan.json`: preselected scenes, conditions, order, scope.
- `runtime_source_sha256.json`: original frozen-source manifest; paths are relative
  to the original artifact's `runtime_source/`, which is not duplicated here.
- `main_source_match.json`: 151 implementation files matched the main worktree;
  two other entries were experiment runner snapshots.
- `verification.json`: original experiment validation, not the later commit audit.
- `artifact_sha256.json`: hashes of the copied records.

`run_benchmark.py` is the exact measured runner snapshot, retained as experiment
provenance rather than a new supported production entry point. From repository
root, with the original PASLCD, model, and prepared cache paths available:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 PYTHONPATH=. \
/home/rvl/miniforge3/envs/oscd/bin/python \
  docs/checkpoints/20260907-paslcd-compact-joint/run_benchmark.py \
  --scene Cantina --cache off --render-mode split \
  --updates 120 --audit --no-profile-components --port 8383 \
  --output-root outputs/paslcd_checkpoint_repeat_cantina_baseline
```

Use `--cache snapshot --render-mode joint_channels` for combined, and `--scene
Garden` for Garden. Each run requires a fresh output directory. The actual order
was Cantina baseline, Cantina combined, Garden combined, Garden baseline.
Both conditions enforce 3D+2D root occupancy, first-OPEN BF10, other BF30,
Gaussian-aggregate binary evidence, and soft-Q u120 training.

Datasets, pretrained weights, cue/DA3 caches, prediction masks, viewer PNGs,
MP4s, and the repeated frozen runtime source are **not** included in this small
archive. A fresh clone alone is not sufficient to run the original GPU experiment.
Local absolute paths and source hashes are intentionally preserved for provenance.
