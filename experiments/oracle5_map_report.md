# Oracle-5 Map — Instance_1 (2026-07-16)

**Question** (user-directed): does a well-chosen 5-frame set beat all 25 frames?
**Answer: yes, on 10/10 scenes — mean +10.3% mIoU over the all-25 mean (range +2.4% … +25.3%).**

Protocol: hill-climbing from the best stride-5 uniform offset per scene, 1–3
frame swaps per step, ≤30 evals/scene (Cantina reused 70 evals from an earlier
partial run); references = all-25 × 5 repeats (mean) and 5 stride-5 uniform
offsets, all evaluated on `query_mask` at ALL 25 poses vs all 25 GT masks
within one batch (cuDNN-benchmark nondeterminism shifts numbers BETWEEN
batches — Garden all-25 was 0.444/0.452/0.488 across sweeps — but is ~exactly
reproducible within one; all comparisons here are within-batch).
Drivers: `oracle_search.py` (Phase A refs) + `oracle_local_search.py` (search),
data in `oracle_search_results.csv`, `all25_repeats.csv`.

## Map

| scene | best-5 combo | best-5 | all-25 | Δ vs all-25 | Δ vs best uniform |
|---|---|---|---|---|---|
| Cantina | (12,15,17,20,24) | 0.5900 | 0.5655 | +4.3% | **+24.2%** |
| Garden | (1,5,7,10,12) | 0.5498 | 0.4520 | **+21.6%** | +7.5% |
| Lounge | (0,1,3,11,20) | 0.5868 | 0.5345 | +9.8% | +2.6% |
| Lunch_room | (1,3,5,9,22) | 0.4177 | 0.3714 | +12.5% | +9.2% |
| Meeting_room | (10,12,19,20,24) | 0.5661 | 0.5151 | +9.9% | +6.5% |
| Playground | (1,10,16,17,24) | 0.4499 | 0.3590 | **+25.3%** | +5.7% |
| Porch | (4,9,15,19,24) | 0.6292 | 0.6040 | +4.2% | +0.8% |
| Pots | (4,6,11,13,21) | 0.6520 | 0.6116 | +6.6% | +1.4% |
| Printing_area | (2,8,13,20,23) | 0.7031 | 0.6867 | +2.4% | +11.9% |
| Zen | (2,12,15,16,23) | 0.5694 | 0.5368 | +6.1% | +4.0% |

Instance_1 means: best-5 0.573 vs all-25 0.524 vs best-uniform 0.534 vs
uniform-offset-mean 0.482. (Search budget was tiny — 30 evals of C(25,5)=53k —
so these are LOWER bounds on the true oracle.)

## What makes the winning sets good

1. **Not uniform spread.** Mean pairwise index gap ranges 5.4–10.8 (uniform =
   10.0). Cantina's oracle lives entirely in the second half (12–24, 0/5
   overlap with its best uniform offset); Garden's entirely in the first half
   (1–12). Blind spreading is the wrong prior on ~half the scenes.
2. **Anchor frames exist.** Per-frame marginal value (mean mIoU with frame in
   the set minus without, over all evaluated sets): every scene has 2–3 anchor
   frames worth +0.03…+0.08 each, and the oracle set contains 2–3 of the top-3
   anchors in 10/10 scenes.
3. **Toxic frames exist — and they are as important.** Worst frames carry
   −0.03…−0.11 marginal mIoU (Porch frame 0: −0.101; Zen frame 3: −0.109;
   Playground frame 8: −0.073). Part of why all-25 loses: it is FORCED to
   ingest every toxic frame. This quantifies the earlier "more frames can
   hurt" observation: selection is noise rejection, not just budget saving.
4. uniform-offset variance (selection sensitivity) did NOT predict the oracle
   margin over best-uniform (r ≈ +0.01) — likely confounded by the fixed
   30-eval budget.

## Implications

- The right claim for this project is **"5 well-chosen frames beat 25"**
  (10/10), not "beat uniform at K=5" (noise-dominated, humans don't beat it
  either). The selector's real job: find anchors, avoid toxic frames.
- Next analysis: what distinguishes anchor vs toxic frames GT-free? Candidates:
  pose-estimation inlier count, cue (candidate_map) mass/precision, view
  overlap with the changed region, parallax w.r.t. the rest of the set. If
  toxic ≈ bad pose / bad cue, an online filter is straightforward and is the
  practical payoff of this study.

## Reproduce

```bash
python experiments/oracle_search.py --max-combos 0   # Phase A refs only
python experiments/oracle_local_search.py --per-scene 30
```
