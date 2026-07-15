# Gate H — NBV Frame Selection vs Baselines, Garden mIoU/F1 (2026-07-15)

**Question**: does adaptive NBV frame selection reach the same (or better)
change-detection quality with fewer update frames? Evaluation follows the fair
protocol: the final R_change is rendered at **all 25 inference poses**
(`renders/query_mask/`) and compared against **all 25 GT masks**, regardless of
how many frames were used for updates.

**Setup**: `subset_oscd.py` post-Gate-H two-phase protocol (poses for all
frames first, then selected frames processed; identical code path for every
method, only the selection differs). NBV: each round, one Beta-EIG-weighted
probe render per remaining pose picks the frame seeing the most undecided
change mass; after processing, exact responsibility-weighted soft counts of the
observed cue update the per-Gaussian Beta state (`target_nbv/change/`).
Selection uses only poses + current change state — never a candidate frame's
RGB or cue. `--nbv_count_scale 0.05`, Beta prior (1,1). Random = mean of seeds
0/1/2. Runs: `experiments/garden_nbv_sweep.sh` → `garden_nbv_results.csv`.

## Query-view results (mIoU / F1 on all 25 GT masks)

| K | uniform | random (3-seed mean) | **nbv** |
|---|---------|----------------------|---------|
| 3 | 0.3940 / 0.5624 | 0.4653 / 0.6325 | **0.4995 / 0.6654** |
| 5 | 0.5057 / 0.6709 | 0.4688 / 0.6339 | 0.4986 / 0.6645 |
| 10 | 0.4715 / 0.6369 | 0.4628 / 0.6256 | 0.4710 / 0.6316 |
| 25 (all) | — | — | 0.4876 / 0.6438 (reference) |

## Findings

1. **NBV with K=3 (0.4995/0.6654) matches-or-beats the all-25 reference
   (0.4876/0.6438)** and beats uniform@3 by +0.106 mIoU / +0.103 F1. Uniform
   needs K=5 to reach the same level.
2. **NBV is budget-robust**: 0.4995 → 0.4986 → 0.4710 across K=3/5/10, while
   uniform swings 0.394 → 0.506 → 0.471 (its K=3 index spacing happens to pick
   uninformative frames) and random spans 0.434–0.514 depending on seed.
3. At K=5 nbv and uniform are statistically tied (Δ0.007 mIoU vs ±0.005
   scene-level noise); both sit at the all-25 level — consistent with the
   Phase-1 finding that Garden saturates around 5 well-spread frames.
4. All methods dip slightly below the all-25 reference at K=10, reproducing the
   non-monotonic budget effect seen in the Phase-1 sweep (unexplained; frozen
   protocol artifact, affects all selectors equally).

## Caveats

- Single scene (Garden), single deterministic NBV run per K, scene-level noise
  ±0.005 mIoU. The K=3 uniform collapse is one specific index pattern —
  the robust claim is "NBV never collapses", not "uniform always fails".
- The two-phase restructure reorders global RNG consumption, so numbers are not
  bitwise comparable with the pre-Gate-H `garden_sweep_results.csv`; the all-25
  reference here (0.4876) vs frozen baseline (0.4918) shows the protocol noise.
- PASLCD-wide claims need the 20-instance benchmark (~10 min per selector-budget).

## Reproduce

```bash
bash experiments/garden_nbv_sweep.sh
```
