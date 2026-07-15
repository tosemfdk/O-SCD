# Stage-14 Real-Scene Pool Evaluation — Garden (2026-07-15)

**Question**: does target-conditioned NBV selection buy the same target information
with fewer views than blind selection, on a real reconstruction with real camera poses?

**Setup** (`experiments/target_nbv_pool_eval.py`, seed 7):

- Scene: `data/PASLCD/Instance_1/Garden` reference reconstruction (207,832 Gaussians, iter 30000).
- Candidate pool: 25 of the 60 real capture poses from `reference_reconstruction/cameras.json`
  (uniform subsample, capture order), rendered at 504x283 (1/8 of capture resolution).
- Targets: 10 Gaussians sampled with opacity > 0.9 + analytic in-view/size filters
  (1 candidate skipped as fully occluded in every pool view — real occlusion, caught
  by the color-probe backend, not the analytic filter).
- Protocol: every method starts from the same seed view (first pool view where the
  target is actually visible) and greedily spends K=5 more picks. After each pick the
  TRUE FD-Jacobian information of that view is committed, so the uncertainty
  trajectory measures what the pick actually bought regardless of how it was chosen.
  Exact greedy = pure D-optimality (e/trace/movement weights 0).
- Methods: random (3 seeds), uniform (index-spaced), max_resp (greedy visibility
  responsibility), proxy (greedy geometry-FIM proxy), exact (greedy target D-opt).

## Results

Mean Δlogdet(H) over 10 targets (nats; higher = more information about the target):

| k | random | uniform | max_resp | proxy | exact |
|---|--------|---------|----------|-------|-------|
| 1 | 6.79 | 7.86 | 16.96 | 16.35 | **20.50** |
| 2 | 10.23 | 13.66 | 21.50 | 19.37 | **25.71** |
| 3 | 14.67 | 22.70 | 23.96 | 23.61 | **27.71** |
| 5 | 18.28 | 24.88 | 26.66 | 26.88 | **29.41** |

- **Views to reach uniform@5 information: exact = 2.0 (10/10 targets), uniform = 4.2.**
  Exact D-opt needs ~2 chosen views to match the target information uniform selection
  gets from 5 — and its k=2 level (25.7) already exceeds uniform's k=5 (24.9).
- Wasted picks (view chosen where the target is invisible): random 28%, uniform 26%,
  max_resp 10%, **proxy/exact 0%**. Blind selection burns a quarter of a small budget
  on views that contribute nothing to the target.
- Worst-axis uncertainty λmax(Σ) remaining at k=5: exact 13.6% < max_resp 16.1%
  ≈ uniform 16.6% < proxy 18.3% « random 29.2%.
- Proxy vs exact ranking agreement: Kendall τ 0.38, top-1 agreement 29%. The proxy is
  a usable coarse pre-filter (its greedy trajectory tracks max_resp/uniform@5 level and
  never picks invisible views) but not a substitute for exact scoring — consistent with
  its intended role (docs §6: pre-rank, then exact on top-k).
- Cost: visibility 2.3 ms/view, FD Jacobian 18.5 ms/view (504x283, 207k Gaussians,
  cached once per (target, view) pair; 182 Jacobians total). Full 10-target run ≈ 5 s.

## Synthetic tests (spec stage 14 Test A–E)

Automated in pytest since Gates E/F: Test A (anisotropic prior → orthogonal-baseline
candidate wins) = `test_baseline_perpendicular_to_uncertain_axis_wins`,
`test_e_gain_prefers_weakest_axis`; Test B (repeated view → marginal gain decreases)
= `test_repeated_view_marginal_gain_decreases`, `test_marginal_gain_decreases_after_commit`;
Test C (occlusion → lower score/invalid) = `test_occluded_candidate_gains_less`,
`test_fully_occluded_candidate_invalid_and_last`. Test D (Schur neighbor ambiguity)
and Test E (change Beta EIG) require stages 10/11, not yet implemented.

## Caveats

- Metric is target-block Fisher information (logdet/λmax), the selector's direct
  objective — NOT change-detection mIoU. The mIoU question additionally needs the
  change-mode scorer (stage 11) and pipeline integration (stage 13); the fair protocol
  there is already in place (`subset_oscd.py` evaluates `query_mask/` at all 25
  inference poses regardless of budget).
- Single scene (Garden), 10 targets, one pool subsample. Trends are large and
  uniform across targets (10/10 for the headline number), but PASLCD-wide claims
  need the 20-instance run.
- Greedy exact here scores ALL remaining pool views per round (pool is small);
  the production path (proxy top-k gate) is what the latency numbers above feed.

## Reproduce

```bash
python experiments/target_nbv_pool_eval.py --pool-size 25 --num-targets 10 \
    --budget 5 --random-seeds 0 1 2 --out experiments/pool_eval_garden
python experiments/analyze_pool_eval.py experiments/pool_eval_garden/results.csv
```
