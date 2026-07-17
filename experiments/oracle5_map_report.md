# Oracle-5 Map — Instance_1 (2026-07-16; rev3 update 2026-07-17, see bottom)

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

---

# rev3 — 20x deeper search (2026-07-17)

**Question** (user-directed): the 30-eval map covered 0.06% of C(25,5)=53k —
does a much deeper search change the picture?
**Answer: mostly no — and that is the finding.** Mean best-5 improved only
+2.8% over the 30-eval map despite ~550 extra evals/scene. The one exception
is Playground (+15.5%), whose old best was a genuine underestimate.

Protocol (rev3, user-specified): per scene, HIT = frozen previous best x 1.10;
budget 300 evals (round 1) + 250 evals (round 2, targets re-raised on the
updated bests — compounding); hill-climb with 1-3 swaps; if the scene best
stalls for 50 consecutive evals, jump to a NEW pool (unseen random combo,
<= 2 frames overlap with the best) and climb there. Runs executed scene-
parallel on 8x V100 (one worker thread per GPU), 5,315 new runs in ~11.4h
wall. Driver: `oracle_local_search.py` rev3 (phases `local3`/`restart3`).

**Machine caveat**: rev3 ran on the V100 server (torch 2.5.1, dynamo
disabled), the pre-rev3 CSV came from the A6000 machine. Per-scene
`recheck3` rows re-measure each old best here: shift −0.007 … +0.017 mIoU —
an order of magnitude below the +10% HIT margin, so cross-machine
comparisons hold.

## rev3 map (best over ALL ~590 evaluated sets/scene)

| scene | best-5 combo | best5 | vs all-25 | vs uniform_best | vs 30-eval map |
|---|---|---|---|---|---|
| Playground | (0,4,19,20,21) | 0.5199 | **+44.8%** | +22.1% | **+15.5%** (HIT) |
| Garden | (1,4,14,16,19) | 0.5515 | +22.0% | +7.9% | +0.3% |
| Lunch_room | (1,5,19,21,24) | 0.4251 | +14.5% | +11.2% | +1.8% |
| Lounge | (0,1,3,11,20) | 0.6037 | +12.9% | +5.5% | +2.9% |
| Meeting_room | (10,12,19,23,24) | 0.5678 | +10.2% | +6.8% | +0.3% |
| Zen | (2,11,15,18,22) | 0.5850 | +9.0% | +6.8% | +2.7% |
| Pots | (1,7,9,11,21) | 0.6582 | +7.6% | +2.4% | +0.9% |
| Porch | (1,5,14,20,23) | 0.6385 | +5.7% | +2.3% | +1.5% |
| Cantina | (12,15,17,20,21) | 0.5955 | +5.3% | +25.3% | +0.9% |
| Printing_area | (2,8,11,12,14) | 0.7115 | +3.6% | +13.2% | +1.2% |

Instance_1 mean: best-5 0.5857 vs all-25 0.5237 (**+11.8%**, was +10.3%).

## What rev3 adds to the picture

1. **The oracle plateau is shallow.** 9/10 scenes gained <3% from a 20x
   larger search with 4-5 pool restarts each — the 30-eval map was already
   near the practical ceiling. The "+10% over your own best" HIT bar was
   reached only once (Playground, twice in sequence: 0.4499 -> 0.4971 ->
   0.5199). The oracle-vs-all-25 gap is real but its size was measured
   about right the first time.
2. **Anchor structure survives 15x more data.** With ~590 sets/scene the
   marginal-value estimates are far less noisy; the best set still contains
   1-3 of the top-3 anchors in 10/10 scenes (>=2 in 8/10).
3. **Old per-frame marginals had sign errors.** Playground/19 flipped from
   "toxic" (−0.071, n≈35) to anchor (+0.018, n≈376) — and frame 19 is in the
   new Playground best. Magnitudes also shrank overall (Zen/3 −0.109 ->
   −0.072). Treat the pre-rev3 anchor/toxic lists as superseded.
4. **Set structure is still not uniform**: mean pairwise index gap of the
   best sets ranges 4.6 (Cantina, clustered in 12-21) to 12.4 (Lunch_room)
   vs 10.0 for uniform.
5. Caveat: rev3 marginals are estimated from hill-climb-biased samples
   (sets concentrate near good regions), not uniform random sets.

## Reproduce (rev3)

```bash
python experiments/oracle_local_search.py --per-scene 300            # round 1
python experiments/oracle_local_search.py --per-scene 250 --seed 8   # round 2
# logs: experiments/oracle_local_search_r3{,b}.log
```

---

# Round 3 (2026-07-18): +450 evals/scene — the plateau holds a third time

Same rules (HIT = prev_best x 1.10, 50-stall pool jump, seed 9), 4,500 runs
in ~9.5h. **Zero HITs; per-scene gains +0.0% … +1.0%, Playground included
(+0.3%).** Cumulative ~1,018 sets/scene (~2% of C(25,5)) with 8 pool
restarts/scene this round. Updated map (only the movers):

- Lunch_room 0.4251 -> 0.4293 (+1.0%), Printing_area 0.7115 -> 0.7165,
  Pots 0.6582 -> 0.6620, Garden 0.5515 -> 0.5536, Playground 0.5199 ->
  0.5213, Lounge 0.6037 -> 0.6051. Instance_1 mean best-5 0.5875 vs all-25
  0.5237 = **+12.2%**.
- Same-machine rechecks reproduce to +-0.0000 — within-batch noise is nil;
  the plateau is a property of the search space, not measurement noise.

What the doubled sample buys — the marginal-value estimates converge:

- Feature correlations are now stable across rounds: cue_consistency
  **+0.177** (strongest), cue_area −0.153, cue_mass −0.147, pose ≈ 0.
- The 2-feature toxic filter reaches **10/30** bottom-3 retrieval
  (chance 3.6; 6/30 at n≈35, 9/30 at n≈590).
- Anchor structure final: best set contains ≥1 of the top-3 anchors in
  10/10 scenes, ≥2 in 9/10 (Porch is the exception with 1).
- Worst toxic frames, final ranking: Zen/4 (−0.111), Zen/6 (−0.087),
  Porch/0 (−0.077), Printing_area/4 (−0.076), Porch/7 (−0.065).

# Instance_2 replication (2026-07-17)

**Question** (user): does the ~+11% mean gain reproduce on Instance_2?
**Answer: yes — +10.2% (ratio of means) with only 100 evals/scene, and the
per-scene gap structure transfers (r = 0.89 with Instance_1).**

Protocol: same rev3 driver with `OSCD_INSTANCE=Instance_2` (per-instance CSVs
`oracle_search_results_instance_2.csv` / `all25_repeats_instance_2.csv`);
references (5 stride-5 offsets + all-25 x5) generated per scene inside the
scene worker, then 100 hill-climb evals, no early stop, 50-stall pool jump.
Everything within one machine/batch (8x V100, ~2.3h, 1,051 sets). NOTE: the
search here is 6x shallower than Instance_1's ~590 evals/scene — these best-5
are lower bounds by a wider margin.

| scene | best-5 combo | best5 | all-25 | Δ | (Instance_1 Δ) |
|---|---|---|---|---|---|
| Playground | (0,2,11,15,18) | 0.4482 | 0.3335 | **+34.4%** | +44.8% |
| Garden | (10,12,14,16,19) | 0.5359 | 0.4194 | +27.8% | +22.0% |
| Lounge | (6,11,17,18,19) | 0.5408 | 0.4574 | +18.2% | +12.9% |
| Lunch_room | (1,3,5,6,11) | 0.4116 | 0.3633 | +13.3% | +14.5% |
| Meeting_room | (2,4,9,15,22) | 0.5373 | 0.4776 | +12.5% | +10.2% |
| Porch | (5,11,17,20,23) | 0.6033 | 0.5618 | +7.4% | +5.7% |
| Pots | (1,9,17,19,24) | 0.6078 | 0.5865 | +3.6% | +7.6% |
| Printing_area | (1,12,13,20,24) | 0.5875 | 0.5752 | +2.1% | +3.6% |
| Cantina | (3,10,13,14,18) | 0.5246 | 0.5281 | −0.7% | +5.3% |
| Zen | (11,12,16,18,20) | 0.5169 | 0.5213 | −0.9% | +9.0% |

Instance_2 means: best-5 0.5314 vs all-25 0.4824 (**+10.2%**; per-scene-delta
mean +11.8%) vs uniform-offset mean 0.4543. best-5 > all-25 on 8/10 scenes;
Cantina/Zen are ties within noise at this shallow budget (both were positive
on Instance_1 after 6x more search).

Key finding: **the oracle gap is a scene property, not noise.** Per-scene Δ
correlates across the two change instances at Pearson r = 0.887 (p < 0.001,
Spearman 0.84) — Playground/Garden lead and Printing_area/Cantina trail in
BOTH instances, despite different changes, different frames, and different
search depths. all-25 also loses to a single good uniform offset on several
Instance_2 scenes (e.g. Garden 0.419 vs 0.507): the toxic-ingestion failure
mode is not an Instance_1 quirk.

Reproduce:

```bash
OSCD_INSTANCE=Instance_2 python experiments/oracle_local_search.py \
  --per-scene 100 --all25-reps 5 --hit-margin-rel 999 --recheck-prev-best 0
# log: experiments/oracle_local_search_inst2.log
```
