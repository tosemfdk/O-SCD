# GT-Free Frame-Quality Features vs Oracle Marginal Value (2026-07-16)

**Question**: can anchor/toxic frames (from the oracle-5 map) be identified
WITHOUT ground truth, at selection time?
**Answer: only partially.** Pose quality is ruled out; the strongest signal is
a "large, idiosyncratic cue" toxic type; much of frame value is
set-context-dependent and invisible to per-frame features by construction.

Setup: `experiments/frame_feature_analysis.py` → `frame_features.csv`
(untracked). Per (scene, frame): PnP inliers, miniBA residual/inliers
(instrumented via method wrapping), cue mass/area/max (SAM2+render-diff
candidate_map), cue 3D-consistency (per-Gaussian adjoint lift e1_f, cosine to
the sum of the other 24 frames), viewing-direction isolation, camera-centroid
distance. Target: per-frame marginal mIoU from all evaluated oracle-map sets.
Correlations pooled over 10 scenes with per-scene z-scoring (n=200).

## Results

| feature | pearson | spearman |
|---|---|---|
| cue_area | **−0.185** | −0.148 |
| cue_mass | **−0.169** | −0.139 |
| cue_consistency | +0.104 | +0.069 |
| ba_residual | +0.083 | +0.109 |
| pnp_inliers | −0.080 | −0.062 |
| others | ~0 | ~0 |

1. **Pose quality does not explain toxicity** (r ≈ 0 for inliers/residual, and
   the sign is even slightly wrong). Consistent with the frozen baseline's
   0/500 pose failures — the pose front-end is not the weak link.
2. **The one identifiable toxic type: big, idiosyncratic cues.** Frames whose
   candidate_map is large (area/mass at the 88–96th percentile of their scene)
   while its 3D-lifted consensus agreement is low (4–32nd percentile):
   Zen/3 (−0.109), Zen/4 (−0.102), Cantina/11 (−0.052), Playground/19
   (−0.071) all fit. Mechanism: the fusion loss only pushes change mass UP
   where cue exists, so a large false-positive cue injects damage that the
   global regularizer cannot fully remove.
3. But other toxic frames (Porch/0, Porch/7, Zen/8, Cantina/2) fit no feature
   profile, and a 2-feature score (z(cue_area) − z(cue_consistency)) retrieves
   only 6/30 of the actual bottom-3 frames (chance ≈ 3.6/30). Not usable as a
   standalone filter.
4. Marginal-mIoU targets are themselves noisy (≈35 sets/scene, scene-chaos
   noise), which attenuates all correlations — treat magnitudes as lower
   bounds.

## Implications

- A cheap online guard is still justified: **downweight/defer frames whose
  cue is large but unsupported by the consensus of already-integrated
  frames** (cue_consistency is computable online from committed frames only).
  It would catch the worst identified type (Zen/3/4 class), not everything.
- The unexplained remainder supports the set-level view: a frame's value
  depends on what the SET already contains (parallax on the change region,
  redundancy) — per-frame scoring cannot capture this, which is exactly the
  direction-aware set criterion (nbv_dopt line) argument.
- Next candidates if pursued: set-conditional features (marginal parallax on
  consensus-change Gaussians), or learning-to-rank on the oracle map data.
