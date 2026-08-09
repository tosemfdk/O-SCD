# Oracle-boundary fixed-capacity MCMC implementation plan

## Scope and stop condition

Implement a fixed-`N` current-state `R_change` representation for the manual
intervals `[0,95)`, `[95,199)`, and `[199,inf)`. Completed states are immutable
CPU/disk archives; only the current state is mutable on GPU. The experiment
stops after A0--A4 are implemented and verified, a three-state engineering
smoke is complete, the full-`N` `oracle_stream` seed-0 pilot is evaluated, and
`docs/oracle-boundary-mcmc-adaptation-report.md` records the result.

Explicit exclusions are BOCD, automatic boundaries/lifespans, GT-assisted
training or relocation, residual-guided proposals, MH/MALA, Gaussian
append/delete, densification/pruning, opacity reset, RGB scene reconstruction,
and object trajectory tracking.

## Phase 0 -- map and lock the baseline

1. Maintain `docs/manual-boundary-mcmc-code-map.md` against the actual checkout.
2. Record the dirty worktree rather than deleting existing artifacts.
3. Run the existing temporal CPU and CUDA renderer tests before changes.
4. Treat the corrected cue-trained DC lifespan result (overall mean-frame
   mIoU `0.6245`, F1 `0.7460`) as the matched-exact A0 reproduction target.

Baseline evidence collected on 2026-08-08:

```text
tests/temporal, non-CUDA: 66 passed
tests/temporal, CUDA:     6 passed
GPU: NVIDIA RTX A6000, 46 GiB
```

## Phase 1 -- representation and renderer

1. Add `FixedCapacityChangeState` with mutable `xyz`, `change_dc`, raw change
   opacity, raw scaling, and raw quaternion rotation.
2. Keep reference opacity separate; initialize unsupported change opacity to
   `0.001` and cue-supported slots to `0.1`.
3. Keep cue support as metadata, never as the MCMC renderer hard gate.
4. Add append-only `StateArchive` and oracle boundary manager with half-open
   interval semantics, warm-start or base-zero initialization, checksums, and
   optimizer reset at target switches.
5. Preserve the existing renderer when overrides are absent and pass gradients
   through all five override tensors.
6. Implement A0 `dc_only`, A1 `dc_opacity`, and A2 `geo_adam` without topology
   mutation.

## Phase 2 -- official-style MCMC components

1. Reuse the existing SSF loss and add explicit opacity/scale reduction
   (`mean` or `sum`) plus optional unsupported-geometry anchor.
2. Add covariance- and opacity-gated xyz-only SGLD with an independent seeded
   generator and full noise diagnostics.
3. Port fixed-`N` live-target relocation: select dead rows by activated change
   opacity, sample live rows proportional to activated change opacity, group all
   assignments before mutation, apply Eq. 9 opacity/scale correction in-place,
   and reset only target Adam moments.
4. Audit pure relocation before any optimizer step or SGLD perturbation. Call
   it *approximate rendering-preserving 3DGS-MCMC-style relocation*, not an
   exact invariant transition.
5. Implement A3 `geo_sgld` and A4 `geo_mcmc`; keep A5 anchor optional.

## Phase 3 -- test and engineering smoke

Required gates:

- all `tests/temporal` and `tests/mcmc` pass;
- renderer equivalence/gradient tests pass on CUDA;
- fixed Gaussian count and parameter identity hold;
- no densify/prune/reset path is called;
- GT training access remains zero;
- archive parameter drift is exactly zero;
- all tensors, noise, relocation terms, renders, and energies are finite;
- synthetic Eq. 9 audit is compared with naive cloning;
- real relocation p95 relative energy jump is at most 5% before pilot.

Request a deterministic three-state smoke with resolution 8, four updates per
arriving frame, the first ten frames of each state, and a 50k deterministic
Gaussian subset. The immutable cached cue artifact is resolution 4, so the
validator rejected resolution 8 rather than silently resampling supervision;
the executed smoke used resolution 4 and records this deviation in
`smoke/validation.json`. This output is engineering evidence only.

## Phase 4 -- full-N oracle-stream pilot

Run A0 through A4 sequentially with full initial `N`, seed 0, previous-state
warm start, 16 updates on each newly arriving frame, and the fixed boundaries.
Every arrival may use only the observed cue and past/current state data. Save a
GT-free arrival-time prediction for later adaptation evaluation. Use the same
input hash manifest for every ablation and abort on mismatch.

Evaluate GT only in the separate evaluator, using the repository's arithmetic
mean of per-frame IoU/F1 convention. Report both arrival-time (`seen_prefix`)
and final archived-state (`full_state_posthoc`) results.

## Phase 5 -- report and deferred work

Generate the requested comparison, adaptation, energy, coverage, capacity,
relocation, displacement, confusion, and runtime plots. The report must expose
negative results and explicitly test whether official live-target relocation
can seed disconnected new geometry. Matched-exact three-seed A0--A4 sweeps are
deferred until the smoke and seed-0 oracle-stream pilot gates pass; they are not
part of this implementation stop condition.
