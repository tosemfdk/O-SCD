# Manual-boundary fixed-capacity MCMC code map

Verified against the working tree on 2026-08-08. The checkout already contained
uncommitted temporal experiments and generated artifacts before this task; this
map follows the code on disk rather than assuming a clean branch.

## Scope and boundary contract

The implemented experiment is oracle-boundary only:

```text
S0 = [0, 95)
S1 = [95, 199)
S2 = [199, end)
```

`temporal.mcmc_state.OracleBoundaryStateManager` uses the same half-open rule.
It archives a completed current state before resetting the mutable state at 95
or 199. It does not discover boundaries and does not contain BOCD.

## 1. Tensor shapes and ownership

### A0 lifespan baseline

`temporal.change_model.TemporalChangeModel` owns only the state DC parameter and
lifespan buffers:

| value | owner | shape |
|---|---|---|
| `state_change_dc` | `TemporalChangeModel` parameter | `[N,S,1,3]` |
| `state_start` | buffer | `[N,S]` |
| `state_end` | buffer | `[N,S]` |
| `state_valid` | bool buffer | `[N,S]` |
| xyz/base DC/rest/opacity/scaling/rotation | external frozen `GaussianModel` | base shapes |

`state_valid` is the corrected DC-only hard support gate. A0 keeps this behavior
for baseline reproducibility.

### A1--A5 current MCMC state

`temporal.mcmc_state.FixedCapacityChangeState` owns exactly five current-state
parameters on the GPU:

| value | shape | representation |
|---|---:|---|
| `current_xyz` | `[N,3]` | world-space mean |
| `current_features_dc` | `[N,1,3]` | SH-degree-0 change DC |
| `current_raw_change_opacity` | `[N,1]` | raw logit, activated with sigmoid |
| `current_scaling` | `[N,3]` | raw log-scale, activated with exp |
| `current_rotation` | `[N,4]` | raw quaternion, normalized before render |

`slot_ids [N]` and `cue_support_mask [N]` are buffers. Cue support is metadata,
not a renderer gate. The reference model remains frozen and its RGB opacity is
never used for dead/live classification. Higher-order SH features stay on the
reference model and are not duplicated into every archive because change
rendering fixes SH degree to zero.

Only the mutable current state is resident as trainable GPU parameters.
`StateArchive` stores detached CPU tensor copies plus state ID, half-open times,
metadata, tensor hashes, and a payload checksum. Archive reads return clones.
Optimizer moments are not archived.

## 2. Renderer attribute path

`gaussian_renderer.render_change` accepts these optional overrides:

```text
override_xyz
override_dc
override_opacity
override_scaling
override_rotation
```

The exact path is:

1. `means3D = pc.get_xyz` or `override_xyz`.
2. `opacity = pc.get_opacity` or `override_opacity`.
3. activated scale/normalized rotation from the state are passed as rasterizer
   scale and rotation, or into `pc.covariance_activation` on the Python
   covariance path.
4. `override_dc` is passed to the fast change rasterizer's `dc` input.
5. `render_change_temporal` preserves the legacy lifespan path and forwards all
   five attributes for models exposing `get_active_render_attributes`.

All overrides are shape/dtype/device checked and are used without detach on the
training path. With no overrides, the original path is unchanged. CUDA tests
cover base-clone equivalence and finite gradients for DC, xyz, opacity, scale,
and rotation. Change rendering remains SH degree zero.

## 3. `state_valid` and change-opacity gating

- A0: `TemporalChangeModel.get_active_change` applies the half-open lifespan and
  `state_valid`; `render_change_temporal` multiplies reference opacity by the
  active mask.
- A1--A5: `FixedCapacityChangeState.get_active_render_attributes` returns an
  all-true active vector. `cue_support_mask` is exposed only as metadata.
  Unsupported slots start at activated change opacity `0.001`; newly observed
  cue-supported slots are promoted to at least `0.1` and may later be optimized
  alive or dead.
- Dead/live relocation uses activated **change opacity** at the CLI threshold
  (default `0.005`), never `GaussianModel.get_opacity`.

For support projection only, the runner temporarily renders the frozen reference
geometry with reference opacity to determine which fixed slot projects into a
positive cached cue pixel. This projection does not become the MCMC render
opacity and is detached.

## 4. Frame schedules

`experiments.train_oracle_boundary_mcmc_rchange` provides two schedules:

- `matched_exact`: reuses `make_training_schedule`; it is state-major and
  epoch-major, visits all 304 frames exactly 120 times, and asserts exactly
  36,480 updates.
- `oracle_stream`: processes timestamps in increasing arrival order and applies
  `updates_per_frame` updates to only the arriving frame (default 16). Cue
  support and relocation audit views are updated only from the observed prefix.

`--frames-per-state K` keeps the first K frames of each oracle interval while
retaining their original global timestamps; it is engineering-smoke only.

## 5. Optimizer and boundary resets

- A0 creates Adam over `state_change_dc` and recreates it whenever the scheduled
  state changes.
- A1 creates groups for DC and raw change opacity.
- A2--A5 create groups for DC, raw change opacity, xyz, raw scale, and rotation.
- `OracleBoundaryStateManager.ensure_state` archives the old state and either
  warm-starts parameters from it (`previous`) or restores reference geometry,
  zero DC, and dead opacity (`base_zero`).
- Adam is recreated after every target switch. There is no scheduler; the
  manifest explicitly records `constant_lr_no_scheduler`.
- During relocation, only original live-target Adam `exp_avg` and `exp_avg_sq`
  rows are zeroed. Dead source moments remain attached and unchanged. Parameter
  objects are never replaced.

S0 always starts from reference geometry, zero change DC, and change opacity
0.001. `previous` affects only S0->S1 and S1->S2 warm starts.

## 6. Densification, pruning, and relocation paths

The original O-SCD implementation still contains topology-changing calls:

- `oscd.py`: `densify_and_clone`, `densify_and_split`,
  `densify_and_prune`, and the legacy opacity reset path.
- `scene.gaussian_model.GaussianModel`: append/prune optimizer helpers and
  densification implementations.

The new runner does not call any of them. It never appends, deletes, or replaces
a Gaussian parameter tensor. `temporal.mcmc_dynamics.relocate_dead_gaussians_`
updates existing rows in place and asserts fixed leading dimension/parameter
identity every iteration.

The MCMC port was cross-checked against the official upstream implementation at
commit `7b4fc9f76a1c7b775f69603cb96e70f80c7e6d13`:

- dead threshold `opacity <= 0.005` (this runner uses the specified strict
  `< 0.005` contract);
- opacity-weighted live `torch.multinomial` sampling with replacement;
- grouped Eq. 9 opacity/scale correction;
- reset live target Adam moments, retain source moments;
- position noise `covariance @ Normal * sigmoid(100*(0.005-opacity)) *
  noise_lr * xyz_lr`, with defaults `noise_lr=5e5`, opacity/scale weights 0.01.

The runner omits upstream `add_new_gs`; N is fixed from initialization.
Relocation is audited before any optimizer or SGLD update and is described only
as approximate rendering-preserving 3DGS-MCMC-style relocation.

## 7. GT mask paths

GT can enter the repository through these evaluation/oracle-only paths:

- `experiments/train_real_temporal_rchange.py`: older explicit oracle-GT pilot;
- `experiments/render_temporal_confusion_maps.py`: evaluation/export;
- `experiments/evaluate_oracle_boundary_mcmc_rchange.py`: the new separate
  evaluator, where `gt_mask` images are loaded;
- `/home/rvl/workspace/github/O-SCD/utils/evaluate.py`: authoritative external
  binary-mask metric.

The new training runner uses `build_no_gt_frame_records`, sets `mask_path=""`,
constructs views only from RGB, fixed cameras, and cached pixel+SAM cues, and
never imports GT masks into a view. Summaries record
`gt_used_for_training=false` and `gt_mask_pixels_loaded=0`.

## 8. Checkpoint and archive paths

Each run writes:

```text
summary.json
config.json
input_hashes.json
manifest.json
temporal_rchange_checkpoint.pt
training_energy.csv
gaussian_counts.csv
relocation_events.jsonl
relocation_audit.csv
state_archives/                 # A1--A5
online_predictions/             # oracle_stream
```

The MCMC checkpoint contains the current snapshot and complete archive state
dict. Individual immutable states are also written as
`state_<id>_<start>_<end>.pt`. Artifact, parameter, input, and checkpoint hashes
are recorded. Existing non-MCMC lifespan artifacts are never overwritten.

## 9. Corrected evaluator path

`experiments/evaluate_oracle_boundary_mcmc_rchange.py` is evaluation-only. It:

- replays the archive selected by half-open timestamp;
- supports both full-N and deterministic-capacity A0 checkpoints;
- writes `pred_binary/<GT stem>.png` for the existing O-SCD evaluator;
- reports arithmetic mean of per-frame IoU/F1 as headline mIoU/mF1 and keeps
  count-aggregated metrics as `micro_iou`/`micro_f1`;
- separates arrival-time `seen_prefix` predictions from final-archive
  `full_state_posthoc` metrics;
- writes per-frame/state/scene CSVs, confusion panels, and contact sheets.

The authoritative compatibility path remains
`/home/rvl/workspace/github/O-SCD/utils/evaluate.py`.

## 10. Files changed and deliberately unchanged

Implemented/extended:

- `gaussian_renderer/__init__.py`
- `temporal/mcmc_state.py`
- `temporal/mcmc_energy.py`
- `temporal/mcmc_dynamics.py`
- `experiments/train_oracle_boundary_mcmc_rchange.py`
- `experiments/evaluate_oracle_boundary_mcmc_rchange.py`
- `tests/mcmc/*`
- relevant `tests/temporal/*` renderer/dynamics tests
- `pytest.ini`
- this code map, implementation plan, and final report

Deliberately unchanged:

- `oscd.py`
- `scene/gaussian_model.py`
- original densification/pruning behavior
- camera-pose estimation and fixed-camera artifacts
- cached cue generation
- BOCD/automatic boundary code (none added)
- existing lifespan checkpoints, evaluations, and documentation artifacts

## Differences from pre-implementation notes

Earlier notes described MCMC support as a hard render mask, duplicated
`features_rest` in every state, used a prefix-replay schedule, and said the
runner did not call relocation. The actual implementation instead renders all N
slots, keeps support as metadata, reuses immutable base rest features, updates
only the arriving oracle-stream frame, and executes/audits fixed-N relocation
in A4/A5.
