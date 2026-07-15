# Instance_1 10-Scene Sweep + Garden Low-Budget Frontier (2026-07-15)

Sweep stopped after Instance_1 by design (user request); Instance_2 pending.
Protocol identical to `garden_nbv_report.md`: two-phase subset runner, query_mask
rendered at ALL 25 poses vs all 25 GT masks. Random = mean of seeds 0,1.
Run: `experiments/paslcd_nbv_sweep.sh` (resumable) → `paslcd_nbv_results.csv`.

## 1. Budget headroom exists beyond Garden

| config | mIoU mean | mIoU min | F1 mean | vs all-25 | scenes ≥ all-25 |
|---|---|---|---|---|---|
| all-25 | 0.5230 | 0.3592 | 0.6706 | — | — |
| uniform@5 | 0.4931 | 0.3599 | 0.6478 | −0.030 | 4/10 |
| random@5 | 0.4752 | 0.3466 | 0.6301 | −0.048 | 1/10 |
| **nbv@5** | 0.4538 | 0.2482 | 0.6033 | −0.069 | 4/10 |
| uniform@3 | 0.4579 | 0.2565 | 0.6106 | −0.065 | 3/10 |
| random@3 | 0.4404 | 0.2363 | 0.5910 | −0.083 | 2/10 |
| **nbv@3** | 0.4408 | 0.2127 | 0.5929 | −0.082 | 2/10 |

The Garden-era worry ("uniform@5 ≈ all-25, so budget is free") does NOT
generalize: across 10 scenes uniform@5 loses 0.030 mIoU to all-25 and matches it
on only 4/10 scenes. There is real headroom for a smart selector at K≤5.

## 2. Negative result: the current NBV frame scorer loses to uniform

- nbv vs uniform @K=3: mean −0.017 mIoU, per-scene W/T/L = 2/1/7.
- nbv vs uniform @K=5: mean −0.039 mIoU, W/T/L = 2/2/6.
- Wins: Garden (+0.106), Meeting_room (+0.068), Printing_area@5 (+0.084).
  Losses: Porch (−0.11/−0.13), Zen (−0.05/−0.13), Lunch_room (−0.11@5).
- The Garden result was real but unrepresentative; the budget-robustness claim
  also dies (nbv min 0.213 < uniform min 0.257 at K=3).

Garden low-budget frontier (nbv still wins on its home turf at every K):

| K | uniform | random (3-seed) | nbv |
|---|---|---|---|
| 1 | 0.2930 / 0.4461 | 0.3832 / 0.5471 | **0.3947 / 0.5600** |
| 2 | 0.3482 / 0.5111 | 0.4185 / 0.5861 | **0.4776 / 0.6449** |

## 3. Diagnosis: the Beta count is direction-blind

The frame scorer maximizes undecided responsibility mass and its Beta update
treats "seen once" as "resolved", so after each pick it actively avoids
re-observing what it just saw — pure novelty/coverage seeking. But R_change
quality at held-out query poses depends on localizing the changed region in 3D,
which needs the SAME region observed from several parallax-diverse views.
Uniform spacing provides that for free on central objects; coverage-seeking
dilutes it. Consistent with the scene pattern: nbv wins where coverage is the
bottleneck (Garden's wide outdoor ring; K=1/2/3) and loses on indoor scenes
with localized changes (Porch, Zen, Lunch_room).

The stage-14 pool experiment already validated the right criterion at target
level: D-optimal gain with the viewing-geometry FIM rho/d^2 (I - r r^T) —
direction-AWARE, so a second view of the same target from an orthogonal
baseline scores high while a same-direction revisit scores low. The frame-level
Beta scorer discarded exactly that directionality.

## 4. Proposed next iteration (direction-aware frame scorer)

Reuse existing verified pieces, no new machinery:
1. Per-gaussian tau_g(frame) for ALL gaussians via the adjoint (one render +
   backward per candidate per round — `target_nbv/change/counts.py`).
2. Maintain per-gaussian 3x3 mean-block H_g (N x 3 x 3 tensor) for
   change-candidate gaussians (Beta-ambiguous or cue-positive).
3. Frame score = sum_g w_g * [logdet(H_g + tau-weighted (I - r r^T)/d^2)
   - logdet H_g], w_g = Beta ambiguity weight.
"Look again, from a different angle, at what looks changed but unconfirmed."

## 5. nbv_dopt iteration (direction-aware, 2026-07-15 later)

| config | mIoU mean | mIoU min | vs uniform | W/T/L vs uniform |
|---|---|---|---|---|
| nbv_dopt@3 | 0.4273 | 0.2207 | −0.031 | 4/0/6 |
| nbv_dopt@5 | 0.4688 | 0.2858 | −0.024 | 3/0/7 |

- **The diagnosis was confirmed where it was made**: Porch — nbv's worst scene
  (−0.11/−0.13 vs uniform) — flipped to nbv_dopt's best (+0.121/+0.080;
  0.2645→0.4976 at K=3). K=5 mean improved +0.015 over nbv, min 0.248→0.286.
- **But a new failure appeared**: Zen collapsed (0.2207 at K=3, −0.288 vs
  uniform; Cantina −0.131). Suspected proximity bias: the 1/d² factor makes
  close-up views dominate the score while contributing narrow cues.
- **Benchmark noise discovery**: scene-level variance is far larger than the
  ±0.005 Garden estimate. Zen uniform@5 (0.4252) < uniform@3 (0.5091);
  random seeds span ±0.09 within one scene-budget. Runs are deterministic
  (global seed 0), so this is chaotic sensitivity of the fusion/densify path
  to the selected set — single-run scene comparisons carry ~±0.05-0.1 noise,
  and only multi-scene aggregates are meaningful. Instance_1-only means each
  mean is 10 samples of that noise.

## Caveats

- Instance_1 only (10 scenes); Instance_2 not run (resumable). nbv/nbv_dopt
  deterministic (1 run/K).
- selection.json audit shows nbv picks are spatially spread, not clustered —
  the failure is what it optimizes, not a degenerate pick pattern.
