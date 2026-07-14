# Target-Conditioned Gaussian NBV — Design Document

Status: Stage 1 design (MVP = stages 1–9: `geometry_proxy` + `geometry_exact`).
Normative spec: the user-provided 14-stage brief; repo facts verified against
commit `29acd34` of this repository.

## 1. Goal

Select the next camera view that most reduces the parameter uncertainty of **one
chosen Gaussian** (identified by a persistent ID), not the scene-global
reconstruction uncertainty.

Non-goals / constraints:

- Scoring a candidate must require only the current 3DGS model and the candidate
  pose+intrinsics. **No candidate RGB image is ever needed for scoring.**
- Scoring must never permanently mutate the Gaussian model.
- The frozen O-SCD baseline (`oscd.py`) must be unaffected while the feature is
  unused (only `scene/gaussian_model.py` gains an inert persistent-ID buffer).

Provenance note (spec rule 14): the D/T/E-optimality criteria follow POp-GS and
FisherRF; the viewing-geometry proxy FIM follows GS-DIFF; the Beta-EIG design
follows B³-Seg. **Target-conditioning (single-Gaussian parameter block), the
Schur neighbor marginalization, and the metric-map visibility backend are
project extensions, not claims from those papers.**

## 2. Implementation modes

| Mode | What it scores | Stage |
|---|---|---|
| `geometry_proxy` | Closed-form viewing-geometry FIM on the mean block: `ΔH_μ = ρ/d²·(I−rrᵀ)` | 8 (MVP) |
| `geometry_exact` | FD render Jacobian `ΔH = JᵀWJ` on the 6-dim target block, D/trace/E gains | 9 (MVP) |
| `geometry_schur` | `geometry_exact` with local neighbor nuisance marginalization (Schur) | 10 (off by default) |
| `change_fisher` | Scalar Fisher information of the target's change parameter | 11 |
| `change_beta` | Beta-Bernoulli expected information gain from responsibility pseudo-counts | 11 |
| `joint` | Weighted normalized combination of geometry and change gains | 11 |

## 3. Target parameter block (MVP)

```
theta_t = [mu_x, mu_y, mu_z, log_s_x, log_s_y, log_s_z]  ∈ R^6
```

- `_xyz` is stored raw in world coordinates → slice `[0:3]`.
- `_scaling` is **already stored in log-space** (`scene/gaussian_model.py:33-34`,
  `scaling_activation = torch.exp`) → the adapter reads/writes `_scaling[row]`
  directly; slice `[3:6]`. No extra log/exp anywhere.
- **Raw quaternions are forbidden as a parameter group** (normalization gauge
  makes the FIM rank-deficient). Future rotation support uses the SO(3) tangent
  space: 3-dim `delta_phi`, applied as `R_new = Exp(delta_phi) @ R`, converted
  back to the repo's wxyz quaternion (`utils/general_utils.py build_rotation`
  convention, real part first).
- Opacity/change-logit groups may be added later as 1-dim slices.

## 4. Data contracts

Defined in `target_nbv/types.py` (plain dataclasses; information-matrix algebra
in **float64 numpy**, images/renders stay float32 torch).

| Type | Key fields | Notes |
|---|---|---|
| `TargetHandle` | `persistent_id:int`, `current_indices:LongTensor[K]`, `mode:"single"\|"cluster"`, `frozen_during_episode:bool` | MVP uses `single` (K=1) |
| `TargetParameterSpec` | `parameter_names:list[str]`, `dimension:int`, `slices:dict[str,slice]`, `description` | default `["mean","log_scale"]`, dim 6 |
| `CandidateCamera` | `cand_id:int`, `position:np(3)`, `wxyz:np(4)` (OpenGL c2w quat, real-first), `fovx,fovy,width,height`, `shell_index`, `movement_cost:float`, `meta:dict`, `minicam` (lazy, not serialized) | serialization = JSON of all but `minicam` |
| `TargetVisibility` | `valid:bool`, `bbox_xyxy:tuple[int,4]`, `projected_radius_px:float`, `responsibility_sum:float`, `responsibility_mean:float`, `visible_pixel_count:int`, `occlusion_ratio:float`, `invalid_reason:str\|None`, `responsibility_map:optional` | occlusion_ratio ∈ [0,1] |
| `TargetInformationState` | `target_pid:int`, `spec`, `H_data:np.f64(6,6)`, `damping_applied:float`, `observed_view_ids:list[str]`, `per_view_cache:dict[str,np.f64(6,6)]`, `version:int`, `model_version:str` | `H_prior()` = finalize view (read-only) |
| `CandidateScore` | `candidate`, `proxy_score`, `exact_score`, `d_gain`, `trace_gain`, `e_gain`, `change_eig`, `movement_cost`, `visibility`, `valid:bool`, `invalid_reason` | every failure carries a reason, never bare NaN |
| `SelectionResult` | `best:CandidateScore`, `scores:list`, `H_before`, `predicted_H_after`, `config_snapshot:dict`, `runtime:dict[str,float]` | dry-run safe: no state mutation |

## 5. Coordinate conventions (repo-verified, normative)

- COLMAP-style **w2c** pose. `Camera.world_view_transform =
  getWorld2View2(R,T).transpose(0,1)` — i.e. stored **transposed** (row-vector
  layout, points right-multiply). `full_proj_transform = wvt @ proj.T`.
  `camera_center = wvt.inverse()[3, :3]` (`scene/cameras.py:54-57`).
- `Camera.R` holds the **c2w rotation**; `getWorld2View2` transposes it
  internally (`utils/graphics_utils.py:38-49`).
- Candidate synthesis follows `viewer.py:125-165 create_mini_cam` verbatim:
  OpenGL c2w quaternion → COLMAP flip `R_c2w @ diag(1,-1,-1)` → `R_w2c = R.T`,
  `t = -R_w2c @ position` → transposed matrices → `MiniCam`.
- Viewing ray for target `μ_t` and camera center `o_c`:
  `r_c = (μ_t − o_c)/‖μ_t − o_c‖`, `d_c = ‖μ_t − o_c‖`.
- Intrinsics are FoV-only; `f_px = fov2focal(FoV, pixels)`
  (`utils/graphics_utils.py:73-76`). `znear=0.01`, `zfar=100` hardcoded.
- **Effective resolution**: `render(..., mult=0.5)` affects tile coverage only,
  but any pixel-metric quantity (f_px, ellipse area, desired projected radius,
  FD epsilon calibration) must use the actual `image_width/height` of the
  camera being rendered. One helper `effective_intrinsics(cam)` centralizes this.

## 6. Information matrices and scores

Prior over observed views `V_obs` (damping added **exactly once**, in
`TargetInformationBuilder.finalize()` — scorers never add it):

```
H_data = Σ_{v∈V_obs} J_v,tᵀ W_v J_v,t          (each term symmetrized)
λ      = max(absolute_damping, relative_damping · mean(diag(H_data)))
H⁻     = H_data + λ I
```

Candidate increment and posterior:

```
ΔH(c) = J_c,tᵀ W_c J_c,t     (symmetrized; PSD by construction)
H⁺(c) = H⁻ + ΔH(c)           (no extra damping)
```

Gains (all via `slogdet` / Cholesky solves; **no explicit inverses**):

```
S_D(c)     = ½ [logdet H⁺(c) − logdet H⁻]
S_trace(c) = tr((H⁻)⁻¹) − tr((H⁺)⁻¹)            (cho_solve)
S_E(c)     = λ_max((H⁻)⁻¹) − λ_max((H⁺)⁻¹)      (eigvalsh, 6×6)
```

Final geometry score (defaults: D primary, E small auxiliary, trace off):

```
S(c) = w_D·S_D + w_tr·S_trace_norm + w_E·S_E/(λ_max((H⁻)⁻¹)+ε) − λ_move·C_move(c)
C_move = ‖Δt‖/translation_scale + w_rot·Δangle/π   (0 if no current camera)
```

Fast proxy (GS-DIFF-style viewing-geometry FIM, mean 3×3 block only):

```
ΔH_μ^proxy(c) = ρ_norm(c)/max(d_c², ε) · (I₃ − r_c r_cᵀ)
S_proxy(c)    = ½[logdet(H_μ⁻ + ΔH_μ^proxy) − logdet H_μ⁻] − λ_move·C_move
```

`ρ_norm` = robust-normalized responsibility, `clamp(ρ/percentile90(valid ρ),0,1)`;
render-free fallback `ρ = sigmoid(opacity_t)` when `proxy_uses_visibility=false`.

Schur (stage 10, design only): local block `H_L = [[H_tt,H_tn],[H_nt,H_nn]]`,
effective target information `Λ_t = H_tt − H_tn (H_nn+λ_n I)⁻¹ H_nt`,
`S_Schur = ½[logdet Λ⁺ − logdet Λ⁻]`. Fallback to target-only on severe
indefiniteness (recorded).

Change Beta-EIG (stage 11, design only): `p_t ~ Beta(a,b)`,
`τ_t(c) = Σ_p α_{t,p}T_{t,p}`, `m=a/(a+b)`, expected counts `ẽ₁=mτ, ẽ₀=(1−m)τ`,
`S_change = H[Beta(a,b)] − H[Beta(a+ẽ₁, b+ẽ₀)]` (gammaln/digamma). After a real
mask: `e₁=Σ r_p M_p`, `e₀=Σ r_p(1−M_p)`. Observable chain caution: the repo's
change mask is `sigmoid(mean over 3 channels of rendered _features_dc)`
(`oscd.py:221`, `render_change` forces `sh_degree=0`) — change_fisher must
chain-rule through `σ'(z)/3` on **rendered** values.

## 7. Jacobian backends

- **`finite_difference` (MVP, correctness oracle — stays forever).**
  Central differences per parameter: 1 unperturbed + 12 perturbed **full-image**
  renders (the rasterizer has no ROI mode), pixels sliced to a crop **frozen
  from the unperturbed view** (bbox padded by 3σ_px + eps-induced motion).
  `J: [n_valid_px·3, 6]`, float64. All renders under `torch.no_grad()`.
  Default epsilons (config-exposed): `eps_mean = 0.01·r_world` (r_world =
  `get_scaling[row].max()`), `eps_log_scale = 0.02`, future `eps_rot = 1e-3 rad`,
  `eps_logit = 1e-3`.
  Diagnostics: non-finite count, second-order residual `‖I₊+I₋−2I₀‖`,
  crop-escape warning, eps-halving stability (<5% change in ‖J‖).
- **`autograd` (optional, later).** The fastgs CUDA backward returns dense
  per-row grads for means3D/scales/rotations/opacities/dc — one backward per
  observation row is too slow for full J, but scalar contractions (e.g.
  change_fisher) can use it. No forward-mode support (custom autograd Function).
- Any optimized backend must match the finite-difference oracle on the toy
  scenes (D-gain relative error, top-1 agreement, Kendall τ) before use.

## 8. Visibility backends

- **Backend A″ (MVP): color probe — exact responsibility, no CUDA change.**
  Render a duck-typed **proxy** of the model (tensors shared read-only, only
  `_features_dc` replaced) with the target's DC color set to render as 1 and
  all others as 0 at `sh_degree=0` (color = `SH_C0·dc + 0.5`). The rendered
  image then equals the per-pixel compositing responsibility
  `r_t,p = α_t,p·T_t,p` exactly. A second single-row render (target only)
  gives the unoccluded self-responsibility; `occlusion_ratio = 1 −
  Σr_occ/Σr_unocc`. Two renders per evaluation; the real model is never
  mutated. The analytic 3σ ellipse projection (Python EWA, single Gaussian)
  still provides the bbox/crop and projected radius.
  Rejected alternative (A′): `metric_map`+`accum_metric_counts` counting — the
  count condition in `forward.cu:387-406` is `α ≥ 1/255 ∧ T ≥ 1e-4` and α is
  clamped at 0.99, so counts ignore the α·T magnitude and a strong occluder
  barely reduces them (verified by test).
  Never consume `visibility_filter` (it is **indices**, not a bool mask); use
  `radii[row] > 0`.
- **Backend B (single-pass exact α·T, deferred).** `forward.cu` computes
  per-pixel transmittance (`final_T`, lines 422-423) but discards it; a CUDA
  export would fuse A″'s two passes into one. ABC stub `ExactAlphaTBackend`
  raises `NotImplementedError`; only worth doing for speed, since A″ is
  already exact.
- Exact Jacobians already include alpha/compositing effects — **never multiply
  responsibility into exact FIMs** (double weighting). Responsibility is for
  gating, crop selection, and the proxy score only.

## 9. Target identity

- New `_persistent_id` int64 buffer + `_next_persistent_id` counter on
  `GaussianModel`; initialized `arange(P)` in `create_from_pcd`/`load_ply`/
  `load_ply_change`.
- Compaction: pruned alongside aux tensors in `prune_points` (434-448) and
  `prune_points_fastgs`; appended with fresh IDs in `densification_postfix`
  (477-495) and the fastgs twin. Split children get fresh IDs; split originals
  disappear via the prune path automatically. Unified through `_append_aux` /
  `_compact_aux` helpers so no surgery site can be missed.
- Checkpoints: `capture`/`restore` carry the ID tensor (length-guarded for old
  tuples); PLY save/load uses a sidecar `<ply>.pid.pt` (int64-lossless, PLY
  schema untouched).
- **Episode freeze policy (default):** during an active selection episode the
  target is excluded from split/clone selection and protected from pruning
  (`protect_ids`). `begin_target_episode`/`end_target_episode` (context
  manager) in `target_nbv/target_registry.py`; unknown IDs raise. Cluster mode
  (children inherit a group ID) is designed but not implemented in MVP.

## 10. Numerical policy

- Symmetrize every H after construction: `H ← ½(H+Hᵀ)`.
- Damping added once (see §6); jitter for failed Cholesky is applied to a
  **local copy** only, escalating `base·10^k, k=0..3`, then the candidate is
  invalid with reason `"cholesky_failed"`.
- `slogdet` sign ≤ 0 → same jitter escalation.
- Small negative gains within `tol=1e-9` are clamped to 0; large negative gains
  are logged as implementation errors (never silently clamped).
- NaN/Inf anywhere → `valid=False` + `invalid_reason`; no NaN scores in output.
- All 6×6 algebra float64; `torch.linalg`/`numpy.linalg` solve/cholesky/slogdet;
  explicit `inv()` is banned.
- `H_prior` handed to scorers is `writeable=False`; `score_candidates` is pure,
  `commit_observed_view` is the only mutator.

## 11. Test plan

- `pytest.ini` markers: `gpu` (needs CUDA + built fastgs rasterizer). CPU suite
  runs anywhere via `pytest -m "not gpu"`.
- Toy fixtures (`tests/conftest.py`, programmatic `GaussianModel`, 128×128
  MiniCams, <1 ms renders): `single_pancake` (anisotropic target),
  `target_plus_occluder`, `two_overlapping` (stage 10), `grid_100` (ID surgery).
  Construction detail: set `_xyz/_scaling(log!)/_rotation([1,0,0,0])/_opacity
  (inverse_sigmoid)/_features_dc/_features_rest` as cuda `nn.Parameter`s,
  `active_sh_degree=0`, zero aux tensors.
- Math unit tests (CPU): config validation, info-builder algebra (incremental ==
  batch, damping-once, symmetry, PSD), proxy preferences, Beta entropy (stage 11).
- Scene tests (GPU): NDC-center candidate test, occlusion ordering,
  FD ray-vs-inplane sensitivity, model-restoration bitwise check,
  diminishing repeated-view gain, proxy↔exact rank correlation.
- Determinism: same seed → identical candidates and scores; candidate order
  shuffle → identical ranking.
- Model-mutation test: scoring twice on the same state gives identical results
  and leaves model tensors and `H_prior` bytes unchanged.
- Baseline regression (after the `gaussian_model.py` edit): `subset_oscd.py
  --frames_method all` on Instance_1/Garden must reproduce online mIoU 0.4918.
