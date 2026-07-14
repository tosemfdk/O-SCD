# Budgeted Informative-View Selection for O-SCD

## Project context, research plan, and Codex operating brief

**Status:** planning and baseline-reproduction stage  
**Primary baseline:** O-SCD, *Changes in Real Time: Online Scene Change Detection with Multi-View Fusion*  
**Initial dataset:** PASLCD  
**Initial candidate pool:** approximately 25 post-change inference frames per scene  
**Current claim level:** feasibility hypothesis only; no performance or speedup claim has been validated yet

---

## 1. Purpose of this document

This file is the single source of project context for a coding agent. It explains:

1. the research idea;
2. the O-SCD components that must remain unchanged during baseline reproduction;
3. the view-selection concepts that may be imported from POp-GS, B3-Seg, MV3DCD, and GS-DIFF;
4. the phased experimental plan;
5. fairness, leakage, and evaluation rules;
6. the immediate baseline task.

The agent must read this file before inspecting or modifying the repository.

---

## 2. One-paragraph project summary

O-SCD processes incoming post-change frames online. For each frame, it estimates the camera pose with respect to a fixed pre-change 3D Gaussian Splatting representation, renders the aligned reference view, computes pixel- and feature-level change cues, and updates a persistent multi-view change representation. The proposed research direction is to avoid running the expensive change-cue and multi-view fusion stages on every available inference frame. Instead, under a limited view budget `K`, select a small informative subset of candidate views and update O-SCD only with those views. The first selector will be a generic D-optimal selector derived from 3DGS information gain. A later selector may become change-aware by using uncertainty or disagreement in the persistent change representation. The central empirical question is whether a selected subset can retain most of full-view O-SCD accuracy while reducing the number of expensive updates and the total runtime.

---

## 3. Research question and working hypotheses

### 3.1 Main research question

> Can O-SCD reach full-view or near-full-view scene-change-detection accuracy using substantially fewer post-change update views, chosen by an information-aware selection policy?

### 3.2 Working hypotheses

These are hypotheses, not established results.

- **H0 — Baseline reproducibility.** The unmodified official O-SCD implementation can reproduce the authors' reported behavior on PASLCD within a documented tolerance.
- **H1 — View redundancy.** The 25 inference frames contain substantial geometric and observational redundancy, so O-SCD performance will saturate before all 25 frames are used.
- **H2 — Generic information-aware selection.** D-optimal selection based on the fixed reference 3DGS and candidate poses will outperform random or simple trajectory-uniform selection at the same budget.
- **H3 — Change-aware selection.** After a small bootstrap set of observed frames, a selector targeting uncertainty or disagreement in `R_change` will outperform generic reconstruction-oriented D-optimality.
- **H4 — Adaptive stopping.** The system can stop when marginal information gain or change uncertainty falls below a threshold, rather than using a fixed number of views for every scene.

### 3.3 Desired outcome

A strong result would have the following form:

> With only `K` selected update views, the method reaches at least 90%, 95%, or 99% of full-view O-SCD mIoU/F1 while reducing change-cue generation, multi-view fusion updates, and total end-to-end inference time.

The result must be reported as a performance-versus-budget curve, not as a single cherry-picked `K`.

---

## 4. Scope and terminology

### 4.1 Current scope

The immediate scope is deliberately narrow:

- reproduce the original O-SCD baseline;
- use PASLCD;
- keep the reference 3DGS fixed;
- treat the approximately 25 inference frames as a fixed candidate pool;
- initially perform **batch subset selection**, not true robotic next-best-view acquisition;
- do not modify O-SCD's self-supervised fusion loss or change-cue definitions during the baseline stage.

### 4.2 Important terminology

- **Candidate view:** an available post-change frame that may be selected.
- **Selected update view:** a candidate whose RGB image and pose are used to compute change cues and update `R_change`.
- **Query/evaluation view:** a camera pose at which a final change mask is rendered for evaluation. A query view may be held out from updating `R_change`.
- **Budget `K`:** the number of selected update views.
- **Full-view baseline:** original O-SCD using all valid inference frames in the original order.
- **Batch subset selection:** all candidate poses are known before selecting a subset.
- **Online active selection / NBV:** the system sequentially decides which new view to acquire without access to future observations. This is a later phase and must not be claimed during batch experiments.

---

## 5. O-SCD technical context

### 5.1 Core state

O-SCD maintains two distinct representations:

- `R_ref`: a fixed, high-fidelity pre-change 3DGS representation.
- `R_change`: a persistent change representation initialized from `R_ref`, with learnable per-primitive change values instead of RGB appearance parameters.

`R_change` is not rebuilt independently for every inference frame. It acts as memory: each selected observation updates the same persistent representation, which aggregates evidence across views.

### 5.2 Per-frame O-SCD pipeline

For each incoming post-change frame `I_inf^k`:

1. Extract local descriptors from the incoming frame.
2. Retrieve a small set of matching reference images.
3. Estimate the incoming camera pose `P_inf^k` by PnP with RANSAC, then refine it.
4. Render the aligned pre-change image `I_ren^k` from `R_ref`.
5. Compute a pixel-level cue from photometric discrepancy:

   `C_pixel^k = (1 - lambda) * L1 + lambda * D-SSIM`, with the paper using `lambda = 0.2`.

6. Compute a feature-level cue from the dense difference between SAM2-Tiny feature maps.
7. Combine the cues:

   `C^k = C_pixel^k + C_feature^k`.

8. Update `R_change` for a small number of optimization iterations with O-SCD's self-supervised fusion objective. The paper uses 16 iterations per incoming frame. Historical observations are sampled during these iterations, with an explicit bias toward the newest frame.
9. Render the current change mask `M^k` from `R_change` at `P_inf^k`.

The repository implementation, not a retyped equation in this document, is the source of truth for the exact self-supervised fusion loss and numerical details.

### 5.3 Why O-SCD is a suitable first target

O-SCD exposes an identifiable per-frame computational bottleneck. The paper reports the following average per-frame runtime on PASLCD at 1008 x 560 resolution on an RTX 4090:

| Module | Time per frame | Share |
|---|---:|---:|
| Descriptor extraction | 1.28 ms | 1.4% |
| Reference image retrieval | 11.50 ms | 12.8% |
| Pose estimation | 16.47 ms | 18.4% |
| Change-cue generation | 1.69 ms | 1.9% |
| Multi-view change-cue fusion | 58.17 ms | 64.9% |
| Change-mask inference | 0.49 ms | 0.6% |
| **Total** | **89.60 ms** | **100%** |

Therefore, skipping the cue/fusion stages for redundant views can reduce meaningful computation, even if candidate pose estimation is still performed.

### 5.4 Published reference metrics

The O-SCD paper reports, averaged over 20 PASLCD instances:

- **Online O-SCD:** mIoU 0.486, F1 0.638, 11.2 FPS.
- **Offline O-SCD:** mIoU 0.552, F1 0.694, total runtime 156 s under the paper's offline protocol.

These are reference targets, not guaranteed outputs on different hardware, dependency versions, dataset preprocessing, or repository revisions.

### 5.5 Critical statefulness constraint

O-SCD is order-dependent because `R_change` persists and is optimized sequentially. Consequently:

- the selected set and the processing order are different experimental variables;
- every run must fully reset `R_change`, optimizer state, random generators, and cached history;
- selected views may be processed in the selector's greedy order or restored to chronological dataset order, and both settings should eventually be measured;
- results from different subsets cannot share a partially updated `R_change`.

---

## 6. Proposed budgeted-view formulation

Let the candidate inference views be `V = {v_1, ..., v_N}`, where `N` is approximately 25. Let `S` be a selected subset with `|S| = K`.

The practical objective is:

`maximize  SCD_quality(S)`  subject to  `|S| <= K`,

or, in an adaptive formulation:

`minimize |S|`  subject to  `SCD_quality(S) >= target_quality`.

The first implementation will not directly optimize mIoU because ground-truth masks cannot be used for selection. Instead it will optimize a proxy information objective and evaluate the resulting subset with ground truth only after selection.

---

## 7. Generic D-optimal selection

### 7.1 Statistical interpretation

Let `theta` denote the selected 3DGS parameters and let `J_i` be the rendering Jacobian for candidate view `i`. The approximate information contribution of the candidate is:

`H_i = J_i^T J_i`.

For selected set `S`:

`H_S = H_prior + sum_{j in S} H_j`.

D-optimal design minimizes the determinant of the parameter covariance or, equivalently, maximizes the log-determinant of the information matrix:

`S* = argmax_{|S|=K} log det(H_S)`.

The greedy marginal gain is:

`gain(i | S) = log det(H_S + H_i) - log det(H_S)`.

At each step, select the remaining candidate with the largest marginal gain and update `H_S`.

### 7.2 Why it may reduce redundancy

Two visually or geometrically similar views contribute similar Jacobian directions. Once one is selected, the second should provide a smaller marginal log-determinant gain. This makes the greedy update sensitive to redundancy, unlike independently ranking each frame by a fixed score.

### 7.3 Initial approximation

A full 3DGS Hessian is intractable. The first practical variant should use a validated diagonal or block-diagonal approximation.

Preferred progression:

1. reuse an existing, license-compatible POp-GS/FisherRF information implementation if present;
2. otherwise implement a documented diagonal approximation;
3. if necessary, estimate `diag(J^T J)` with Hutchinson/Rademacher probes;
4. only later consider block-diagonal per-Gaussian parameter groups.

An important correctness rule:

> The gradient of the sum of all output pixels is not equal to `diag(J^T J)` and must not be used as a shortcut.

### 7.4 Parameter subset

The first D-optimal selector should prioritize geometric parameters:

- Gaussian position;
- scale;
- rotation;
- optionally opacity.

Spherical-harmonic color parameters should be excluded by default in the first experiment. POp-GS reports that geometric parameters were important under sparse-view conditions, while excluding spherical harmonics substantially reduced memory and increased speed with little loss in its tested setting.

### 7.5 Limitation of generic D-optimality

Generic D-optimality values views by their ability to constrain the 3DGS representation, not by their ability to expose scene changes. A view of a poorly reconstructed unchanged wall may receive high generic information gain, whereas a view of a small color change on a well-reconstructed table may receive low gain.

Therefore generic D-optimality is:

- a defensible first selector;
- a reconstruction-information baseline;
- not yet the final change-aware contribution.

---

## 8. Concepts imported from related and prior work

### 8.1 POp-GS: optimal experimental design for 3DGS

Relevant concepts:

- formulate 3DGS uncertainty through an approximate Hessian/information matrix;
- compare P-optimality criteria;
- use D-optimality or T-optimality for informative-view selection;
- update the information matrix after each greedy choice so redundancy is reflected;
- use diagonal or block-diagonal approximations for tractability;
- perform batch/keyframe selection from a fixed candidate pool.

What to borrow now:

- the D-optimal marginal-gain formulation;
- information-vector caching;
- greedy batch selection;
- simple diagonal first, block-diagonal later;
- geometric-parameter-first ablation.

What not to assume:

- POp-GS evaluates reconstruction quality, not scene-change mIoU/F1;
- its success does not prove that D-optimality will select the best views for SCD;
- its candidate pools and datasets differ from PASLCD.

### 8.2 B3-Seg: analytic EIG, Bayesian state, and adaptive stopping

B3-Seg reformulates per-Gaussian binary segmentation evidence with Beta-Bernoulli posteriors and selects views by analytic expected information gain (EIG). It provides three ideas useful for a later change-aware O-SCD extension:

1. **Persistent probabilistic state.** Maintain uncertainty per Gaussian rather than only a deterministic change score.
2. **Analytic candidate scoring.** Approximate the expected posterior update using the current posterior mean and per-view visibility/responsibility, avoiding an expensive segmentation call for every candidate.
3. **Diminishing returns and early stopping.** New views provide decreasing marginal information as posterior concentration increases. A system may stop when entropy or marginal EIG falls below a threshold.

Potential O-SCD adaptation:

- use a Beta posterior per Gaussian for `changed` versus `unchanged`, or another calibrated distribution over change probability;
- derive candidate pseudo-counts from projected Gaussian responsibility and current change belief;
- score candidate views by expected reduction of change entropy rather than reference reconstruction entropy;
- run expensive SAM2/pixel cue extraction only for the selected candidate;
- stop when average predictive entropy or maximum marginal EIG is sufficiently low.

What not to copy blindly:

- B3-Seg solves object segmentation on pre-reconstructed 3DGS assets, not bitemporal scene-change detection;
- its binary mask observations and camera sampling assumptions differ from O-SCD;
- a theoretical adaptive-submodularity guarantee would need a fresh proof under O-SCD's cue model and optimization dynamics.

### 8.3 MV3DCD: evidence that limited views are meaningful

MV3DCD already evaluates limited post-change observations. In its PASLCD indoor-scene experiment, it randomly samples 5, 10, and 15 views from 25 and compares seen and unseen query views across three trials. It reports useful change localization even with five views, and performance improves as more views are added.

Implication:

- a 25-view candidate pool is sufficient for an initial proof-of-concept;
- simply showing that five views work is not novel by itself;
- the necessary contribution is that an informed selector reaches a target quality with fewer views than random or uniform selection.

### 8.4 GS-DIFF: observability as a per-primitive concept

GS-DIFF uses a per-primitive Fisher-information observability term derived from camera geometry. A Gaussian seen from many close and diverse directions is better constrained than one seen from a narrow baseline.

Potential future use:

- weight change uncertainty by observability;
- prefer views that improve poorly observed change hypotheses;
- suppress confidence from Gaussians whose geometry is weakly constrained;
- construct a change-aware information matrix over only Gaussians currently suspected to have changed.

What not to do in the first O-SCD baseline:

- do not require a post-change 3DGS reconstruction;
- do not replace O-SCD with direct primitive-space comparison;
- do not mix GS-DIFF scoring into the baseline before the original O-SCD result is frozen.

---

## 9. Phased research plan

## Phase 0 — Reproduce the original O-SCD baseline

**Goal:** establish a trustworthy, immutable reference before any selector is added.

Tasks:

1. Identify the official repository revision and all submodules.
2. Record environment requirements, CUDA/PyTorch versions, custom rasterizers, and model checkpoints.
3. Verify PASLCD directory structure and preprocessing.
4. Verify reference-scene 3DGS assets or reproduce them exactly as required.
5. Run one scene/instance as a smoke test.
6. Run all 20 PASLCD instances with the original configuration.
7. Record mIoU, F1, FPS/runtime, pose failures, and per-module timing if available.
8. Compare results with the published table and document every discrepancy.
9. Freeze the baseline configuration, command, commit hash, and outputs.

No production algorithm change is allowed in Phase 0.

### Phase 0 acceptance criteria

The baseline is considered usable when:

- the official evaluation command runs end to end;
- masks are produced for the expected frames;
- aggregate metrics are stable across repeated deterministic runs or their nondeterminism is quantified;
- any gap from published metrics is explained as far as practical;
- a single command or script can reproduce the baseline on one scene and on the full benchmark;
- baseline outputs are archived before modifications begin.

Hardware-dependent FPS is not expected to exactly match the paper. Accuracy should be compared under the same dataset, checkpoint, resolution, and threshold protocol.

## Phase 1 — Build a neutral subset-evaluation framework

**Goal:** evaluate arbitrary selected subsets without implementing D-optimality yet.

Selectors:

- all frames;
- deterministic random subset;
- chronological uniform sampling;
- pose-space farthest-point sampling.

Requirements:

- `method=all` must reproduce Phase 0;
- fully reset `R_change` and optimizer state for each run;
- support budgets such as `K = 1, 2, 3, 5, 7, 10, 15, 20, 25` where valid;
- support multiple random seeds;
- record selected frame IDs and order;
- separate selected-update metrics from all-query-view metrics.

## Phase 2 — Add generic D-optimal selection

**Goal:** test whether reference-3DGS information gain is a useful SCD subset proxy.

Initial selector inputs:

- fixed `R_ref`;
- candidate pose and intrinsics;
- no ground-truth masks;
- no post-change change cues;
- preferably no candidate RGB content beyond what is already required to estimate the pose.

Compare:

- random;
- uniform chronological;
- pose farthest-point;
- generic D-optimality;
- optional T-optimality;
- oracle subset only as an analysis upper bound, never as a deployable method.

## Phase 3 — Develop change-aware information gain

**Goal:** use a few bootstrap frames to identify uncertain change regions, then select views that reduce change uncertainty.

Possible state representations:

- uncertainty on each O-SCD change parameter;
- Beta-Bernoulli posterior per Gaussian;
- variance or entropy of rendered change masks;
- disagreement between recent per-view cues and `R_change` predictions;
- visibility-weighted uncertainty over suspected changed Gaussians.

Possible candidate objective:

`gain_change(v | S) = expected reduction in change entropy or disagreement after observing v`.

A hybrid objective may be needed:

`gain(v) = alpha * generic_coverage_gain(v) + beta * change_uncertainty_gain(v)`.

## Phase 4 — True online active selection and early stopping

**Goal:** remove the assumption that all 25 future views are already known.

Later requirements:

- generate feasible candidate poses from robot motion constraints;
- score candidates before acquiring their RGB images;
- include travel cost and collision constraints;
- stop when marginal gain is below a threshold;
- report sensing cost, traveled distance, and total time.

Only Phase 4 should be called true online NBV or active acquisition.

---

## 10. Baseline reproduction procedure for the coding agent

The first coding task is inspection and reproduction, not modification.

### 10.1 Repository audit

Locate and document exact paths and symbol names for:

- reference 3DGS construction/loading;
- inference dataset loading and frame order;
- descriptor extraction;
- reference image retrieval;
- PnP/RANSAC pose estimation and refinement;
- aligned reference rendering;
- pixel cue generation;
- SAM2 feature extraction and feature-difference cue;
- `R_change` initialization;
- self-supervised fusion update;
- change-mask rendering;
- post-refinement, if included in evaluation;
- mIoU/F1 computation;
- runtime/FPS measurement;
- configuration parsing;
- seed handling;
- checkpoint and dataset path handling.

Do not infer function names from the paper. Report only symbols found in the repository.

### 10.2 Environment capture

Record:

- OS;
- GPU model and driver;
- CUDA toolkit/runtime;
- Python version;
- PyTorch version;
- installed package lock or environment export;
- git commit hash and submodule hashes;
- compiler versions for CUDA extensions;
- checkpoint hashes;
- dataset file counts and any preprocessing outputs.

### 10.3 Smoke test

Run the smallest official or near-official command that:

- loads one pre-change reference representation;
- processes at least one post-change frame;
- estimates a valid pose;
- produces a change cue;
- updates `R_change`;
- renders a change mask;
- computes or saves an evaluable output.

### 10.4 Full baseline

Run all PASLCD instances under the paper's online setting. Save:

- per-frame predictions;
- per-scene metrics;
- aggregate metrics;
- total and per-module runtime;
- pose failure counts;
- logs and complete configuration.

### 10.5 Baseline freeze

After reproduction, create a tag or commit and archive:

- the exact command;
- the config;
- the environment;
- the output metrics;
- representative masks;
- a written discrepancy report.

No selection code should be merged before this freeze exists.

---

## 11. Evaluation protocol

### 11.1 Accuracy metrics

Primary:

- mIoU on changed pixels;
- F1 on changed pixels.

Additional recommended metrics:

- change-instance recall;
- false-negative rate for small changes;
- performance split by structural versus surface-level change if labels are available;
- performance split by similar versus different lighting;
- per-scene variance.

### 11.2 Budget metrics

For each selector, report:

- quality versus `K`;
- area under the quality-versus-budget curve;
- `K@90%`, `K@95%`, and `K@99%` of full-view mIoU and F1;
- number of selected views required to exceed the original online O-SCD target;
- marginal gain at each selection step.

### 11.3 Runtime metrics

Separate:

- candidate pose-preparation time;
- selector scoring time;
- cue-generation time;
- fusion-update time;
- mask-rendering time;
- total wall-clock time;
- peak GPU memory.

Report both:

1. **backend savings**, excluding sensing/travel and optionally treating cached poses as given;
2. **true end-to-end savings**, including all candidate preparation and selection overhead.

### 11.4 Two evaluation views

Do not mix these metrics:

- **Selected-view evaluation:** score only masks for the selected update frames.
- **All-query-view evaluation:** after `K` selected updates, render `R_change` at all valid inference poses and score those masks. Held-out RGB images and change cues must not update `R_change`.

All-query-view evaluation measures whether a sparse set produces a scene-level change representation that generalizes to unselected viewpoints.

### 11.5 Repetition and statistics

- use at least five random seeds for stochastic selectors when practical;
- report mean and standard deviation;
- use the same valid candidate set across methods;
- record pose failures and do not silently replace failed candidates;
- perform paired comparisons per scene and budget.

---

## 12. Fairness and information-leakage rules

These rules are mandatory.

1. Ground-truth change masks may be used only for final evaluation and oracle analysis.
2. Generic D-optimal selection may not use ground-truth masks, predicted masks, SAM2 features, pixel residuals, or post-change change cues.
3. A change-aware selector may use only information obtained from already selected/observed frames and the current persistent state.
4. Held-out query RGB images must not be used to update `R_change`.
5. If all 25 candidate RGB images are analyzed before selection, the experiment is computational subset selection, not sensing-efficient NBV.
6. `method=all` must take the original frame order and reproduce the unmodified baseline.
7. All methods must use the same pose-valid candidate pool unless a method explicitly includes pose robustness as part of its objective.
8. Selection overhead must be included in end-to-end runtime.
9. Do not tune thresholds separately for every scene unless the setting is explicitly labeled oracle.
10. Do not describe generic reconstruction D-optimality as change-aware.

---

## 13. Expected baseline artifacts

After Phase 0, the repository should contain or produce equivalents of:

```text
artifacts/
  baseline/
    commit_and_environment.txt
    command.txt
    config.yaml
    metrics_per_frame.csv
    metrics_per_scene.csv
    metrics_summary.json
    runtime_breakdown.csv
    pose_failures.csv
    logs/
    predictions/
    qualitative/
    discrepancy_report.md
```

The exact directories may follow repository conventions, but the information must be preserved.

---

## 14. Risks and failure modes

### 14.1 Baseline reproduction risk

The official code, checkpoints, or prebuilt reference 3DGS assets may not exactly match the paper revision. Dependency changes may affect custom CUDA extensions, feature extraction, pose estimation, or metrics.

Mitigation:

- freeze every version;
- reproduce one scene first;
- compare intermediate outputs, not just final mIoU;
- document differences rather than silently patching the baseline.

### 14.2 Small candidate pool

Twenty-five candidates are enough for a proof-of-concept, but method differences may be noisy or saturated.

Mitigation:

- evaluate many budgets and all 20 instances;
- use multiple seeds;
- include more candidate-rich sequences later;
- avoid claiming general active exploration from PASLCD alone.

### 14.3 Pose cost remains

If all candidate poses are estimated before selection, pose-front-end time is still paid for every candidate. Therefore total speedup will be smaller than the ratio `N/K`.

Mitigation:

- report backend and full end-to-end runtime separately;
- later test selectors using known trajectory poses or a cheaper pose-only screening stage;
- eventually move toward candidate pose generation without processing all RGB frames.

### 14.4 Order sensitivity

Greedy D-optimal order, chronological order, and random order can produce different `R_change` states.

Mitigation:

- treat order as an explicit factor;
- first compare subsets under chronological processing for controlled fairness;
- later test greedy selector order as a separate online-like condition.

### 14.5 Generic D-optimality may select the wrong content

Reconstruction information is not identical to change information.

Mitigation:

- retain random, uniform, and pose-diversity baselines;
- analyze selected camera coverage and visible ground-truth change only after selection;
- proceed to change-aware EIG if generic D-optimality is weak.

### 14.6 Small or surface-only changes

Foundation-model features may be insensitive to subtle color changes, and sparse selection may miss small changes entirely.

Mitigation:

- report change-instance recall and size-stratified performance;
- maintain pixel and feature cues;
- develop a hybrid selector that balances global coverage and current change uncertainty.

---

## 15. Immediate Codex task

The immediate task is **only Phase 0 baseline audit and reproduction**.

The coding agent must:

1. read this file and all repository `AGENTS.md` files;
2. inspect the repository without modifying the algorithm;
3. map the O-SCD implementation to the pipeline in Section 5;
4. identify the official command for one PASLCD scene and the full benchmark;
5. set up the environment and run a smoke test;
6. run the original baseline if dependencies and data are available;
7. save reproducibility artifacts;
8. report blockers precisely;
9. avoid implementing D-optimality or selection until the baseline is frozen.

### Prohibited during the immediate task

- changing the O-SCD loss;
- changing the number of fusion iterations;
- changing image resolution;
- adding new thresholds;
- skipping frames;
- reordering frames;
- changing evaluation code;
- adding D-optimality;
- reporting unverified speedup or accuracy.

---

## 16. Suggested Codex prompt for Phase 0

```text
Read OSCD_Budgeted_View_Project_Context.md and every applicable AGENTS.md.

This task is baseline audit and reproduction only. Do not implement view
selection and do not modify the O-SCD algorithm.

1. Locate the exact repository paths and symbols for each O-SCD stage:
   reference 3DGS loading/building, inference loader, descriptor extraction,
   reference retrieval, PnP/RANSAC, pose refinement, aligned rendering,
   pixel cue, SAM2 feature cue, R_change initialization, self-supervised
   fusion, mask rendering, evaluation, and runtime measurement.
2. Write a source map with file paths and line numbers.
3. Record the repository commit, submodules, environment, checkpoints, and
   PASLCD layout.
4. Find the official command for one scene and for the full PASLCD online
   benchmark.
5. Run one-scene smoke test without changing algorithmic settings.
6. If successful, run the full baseline and save per-frame, per-scene, and
   aggregate metrics plus runtime and pose failures.
7. Compare against the paper's online reference values, but do not force the
   code to match them by changing settings. Document discrepancies.
8. Create a baseline-freeze report and list every blocker.

Do not guess missing paths or APIs. Do not add selection code in this task.
```

---

## 17. Decision log and open questions

These questions should be answered after baseline reproduction, not before it.

1. Does the official repository evaluate the online mask immediately at each frame, after a final post-refinement, or both?
2. Does the released code use exactly 16 fusion iterations and the newest-frame sampling bias described in the paper?
3. Are all 25 PASLCD inference poses estimated online, precomputed, or registered offline in the released evaluation?
4. Are reference 3DGS assets provided, or must they be rebuilt?
5. How many frames fail pose estimation per scene?
6. Is the evaluation frame order deterministic and encoded in filenames, metadata, or the dataloader?
7. Can `R_change` be rendered at held-out poses without loading held-out RGB images?
8. Which Gaussian parameters are differentiable and exposed by the rasterizer for an information-matrix implementation?
9. Is there an existing FisherRF/POp-GS-compatible Jacobian or visibility path in the codebase?
10. Which metric implementation and threshold produce the paper's Table 1 values?

---

## 18. Source map

The following papers define the context used in this document.

1. **O-SCD** — *Changes in Real Time: Online Scene Change Detection with Multi-View Fusion*, arXiv:2511.12370v3. Relevant: Sections 3.1–3.5, Table 1, Table 2, Figure 4.
2. **POp-GS** — *Next Best View in 3D-Gaussian Splatting with P-Optimality*, arXiv:2503.07819v2. Relevant: Sections 3.2–3.4, Tables 1–7, keyframe and batch-selection experiments.
3. **MV3DCD** — *Multi-View Pose-Agnostic Change Localization with Zero Labels*, arXiv:2412.03911v2. Relevant: Section 5.3 and Figure 4 on 5/10/15/25 inference views.
4. **GS-DIFF** — *From Pixels to Primitives: Scene Change Detection in 3D Gaussian Splatting*, arXiv:2605.07203v2. Relevant: per-primitive observability/Fisher information and limitations of image-space aggregation.
5. **B3-Seg** — *Camera-Free, Training-Free 3DGS Segmentation via Analytic EIG and Beta–Bernoulli Bayesian Updates*, arXiv:2602.17134v1. Relevant: Sections 3.2–3.5, theoretical EIG discussion, runtime analysis, and entropy-based early stopping direction.

---

## 19. Final project principle

The first result to trust is not a new selector. It is a reproduced, frozen, and understood O-SCD baseline. Only after that baseline is stable should the project introduce subset selection, D-optimality, change-aware EIG, or adaptive stopping.
