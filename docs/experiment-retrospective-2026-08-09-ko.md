# 2026-08-09 실험 회고 및 다음 연구 방향

> 결론: lifespan state separation은 검증되었다. 반면 geometry 최적화, SGLD, 공식 3DGS-MCMC live-target relocation은 현재 R_change 목적에서 DC-only lifespan보다 좋아지지 않았다. 다음 단계는 relocation을 더 세게 돌리는 것이 아니라, (1) cue/SSF/state_valid의 영향을 완전히 분리하고 (2) immutable reference occlusion layer와 movable residual layer를 분리한 뒤 (3) GT oracle destination으로 표현 상한을 먼저 검증하는 것이다.

## 1. 정리 범위와 데이터 근거

- 대상: `outputs/` 아래 `summary.json` 67개 전수 확인
- 분류: 주요 학습/control 18개, 평가·시각화 20개, smoke/dev 29개
- authoritative metric: 기존 O-SCD `utils/evaluate.py`의 per-frame IoU/F1 산술 평균
- 데이터: Instance 1 `scene_change1_2_3`, 304 frames
- manual boundaries: `S0=[0,95)`, `S1=[95,199)`, `S2=[199,inf)`
- BOCD/automatic boundary는 아직 사용하지 않음
- smoke/dev 결과는 성능 주장에 사용하지 않고 fixed N, rollback, no-GT, archive immutability 검증에만 사용

## 2. 실험 계보

### E0. 최소 lifespan 표현 검증

- state별 `state_change_dc`와 `state_start/state_end/state_valid`를 sidecar tensor로 추가
- base xyz/opacity/scale/rotation은 고정
- timestamp에 따라 half-open interval state를 선택
- 결론: 하나의 checkpoint에서 S0/S1/S2를 분리하고 과거 state를 재현할 수 있음

### E1. GT oracle pilot / all-frame fitting

- 초기 3-view/state oracle: micro IoU 0.4067, F1 0.5783
- 모든 304 frames x 120 updates: micro IoU 0.6211, F1 0.7662
- 해석: view coverage가 중요하지만 이 결과는 oracle upper bound이며 O-SCD 비교에 사용할 수 없음
- 추가 발견: GT도 일반 BCE가 아니라 SSF positive cue와 hard `state_valid` support로 사용했기 때문에 높은 precision/낮은 recall이 발생. cue-vs-GT strict ablation이 아님

### E2. Corrected O-SCD cue lifespan 및 naive controls

공통 조건: O-SCD pixel + SAM2.1 cue, fixed canonical pose, 304 frames, 120 updates/image, total 36,480 updates.

- lifespan DC-only: mIoU 0.62447, F1 0.74598
- schedule-matched persistent O-SCD: 0.38348 / 0.49904
- 기존 joint-random O-SCD 120ep: 0.48774 / 0.61883
- 결론: lifespan은 matched persistent 대비 +24.10pp mIoU, joint-random 대비 +13.67pp. 개선의 핵심은 state separation이며 relocation이 아님

### E3. State-specific geometry optimization

- xyz/DC를 state별로 최적화하고 S0 geometry anchor 적용
- geometry: 0.61907 / 0.73636
- DC-only: 0.62447 / 0.74598
- geometry가 SSF loss는 소폭 낮췄지만 mIoU는 -0.54pp
- 결론: SSF mask loss만으로 depth/geometry가 under-constrained되어 task metric과 objective가 어긋남

### E4. Oracle-stream fixed-capacity A0-A4

공통: full `N=1,283,501`, 304 frames, 16 updates/arriving frame, seed 0, manual boundary, no GT training, no append/delete/densify/prune.

- A0 `dc_only`: 0.65714 / 0.77173
- A1 `dc_opacity`: 0.62567 / 0.74820
- A2 `geo_adam`: 0.61363 / 0.73702
- A3 `geo_sgld`: 0.62066 / 0.74111
- A4 `geo_mcmc` official live-target relocation: 0.22960 / 0.31296
- 모든 mode에서 Gaussian count invariant와 archive drift zero 확인
- A4 relocation 전후 p95 relative energy perturbation은 약 0.032%로 작았지만 posthoc representation은 붕괴
- A4 seen-prefix mIoU 0.5017 대 posthoc 0.2296, S0 replay mIoU 0.0
- 결론: relocation 한 번은 approximate rendering-preserving이었지만 current-frame-only 후속 optimization이 과거 view 표현을 보존하지 못함. 또한 live-target relocation은 disconnected new object destination을 제공하지 않음

### E5. Conservative residual-guided relocation

변경 사항:

- source: observed + zero positive cue support + carryover/removal/tentative protection 통과
- opacity를 source selection의 주 기준에서 제외
- destination: 최근 causal multi-view unexplained positive residual ray triangulation
- source deactivation safety, negative-region mass gate, held-out validation, exact rollback
- relocated rows만 row-local Adam burn-in

결과:

- no relocation: 0.627839 / 0.747288
- burn-in 0: 0.626926 / 0.746521
- burn-in 32: 0.627578 / 0.747081
- burn-in 64: 0.626871 / 0.746463
- burn-in 120: 0.627115 / 0.746787
- burn-in 32는 burn-in 0보다 +0.065pp이지만 no-relocation보다 -0.026pp
- burn-in 32에서 32회 시도 중 4회 승인, 1,024 slots 이동(전체의 0.08%)
- 승인 relocation 평균 cue coverage gain은 4.7e-6 수준으로 시각적 영향이 거의 없음
- 결론: 더 오래 학습하는 것이 병목이 아님. destination quality, usable source capacity, activation strength가 핵심

## 3. 핵심 결과 표

| 실험 | mIoU | F1 |
|---|---:|---:|
| Persistent matched | 0.38348 | 0.49904 |
| Joint-random O-SCD | 0.48774 | 0.61883 |
| Lifespan DC-only | 0.62447 | 0.74598 |
| Geometry lifespan | 0.61907 | 0.73636 |
| Oracle-stream A0 DC | 0.65714 | 0.77173 |
| Oracle-stream A1 DC+opacity | 0.62567 | 0.74820 |
| Oracle-stream A2 geometry Adam | 0.61363 | 0.73702 |
| Oracle-stream A3 +SGLD | 0.62066 | 0.74111 |
| Oracle-stream A4 +official relocation | 0.22960 | 0.31296 |

## 4. 추가된 구현 모듈

- `temporal/lifespan.py`: half-open temporal gate와 active-state lookup
- `temporal/change_model.py`: state-specific DC sidecar baseline
- `temporal/geometry_change_model.py`: state별 geometry delta/anchor 실험
- `temporal/mcmc_state.py`: `FixedCapacityChangeState`, state-specific change opacity, immutable `StateArchive`, `OracleBoundaryStateManager`
- `temporal/mcmc_energy.py`: SSF + opacity + scale + optional anchor energy
- `temporal/mcmc_dynamics.py`: anisotropic positional SGLD, Eq.9 relocation, dead/live selection, Adam target moment reset, synthetic invariance audit
- `temporal/conservative_relocation.py`: conservative source selection, multi-view residual candidate triangulation, deactivation safety, exact rollback, row-local Adam
- `gaussian_renderer/__init__.py`: xyz/DC/opacity/scaling/rotation override 및 gradient path
- `experiments/train_oracle_boundary_mcmc_rchange.py`: A0-A5/conservative, matched_exact/oracle_stream, artifacts/audits
- `experiments/evaluate_oracle_boundary_mcmc_rchange.py`: archive replay, authoritative evaluator, seen-prefix/posthoc 분리
- `tests/mcmc` + `tests/temporal`: 현재 169 tests pass

중요: `scene/gaussian_model.py` 자체는 바꾸지 않았고 temporal attributes는 slot-indexed sidecar/wrapper로 유지했다.

## 5. 지금까지 확정된 사실

1. Lifespan state separation은 유효하다. Naive persistent 누적의 S0/S1 forgetting을 크게 줄인다.
2. 현재 성능 향상은 MCMC가 아니라 state separation에서 온다.
3. 별도 `change_opacity` 학습은 DC-only보다 좋지 않았다.
4. SSF만으로 geometry를 움직이면 objective는 내려가도 GT mIoU는 좋아지지 않는다.
5. 공식 live-target relocation은 기존 live support 내부 capacity redistribution에는 맞지만 disconnected residual geometry 탐색에는 맞지 않는다.
6. Current-frame-only optimization은 state 내부 multi-view catastrophic forgetting을 일으킨다.
7. DC≈0 Gaussian도 behind-change를 가리는 alpha occluder로 유효할 수 있으므로 positive cue support가 없다는 이유만으로 이동시키면 안 된다.
8. GT-mask 실험도 아직 strict upper bound가 아니다. GT가 좁아서 SSF positive gradient와 `state_valid` capacity가 함께 감소했고, resolution/pose/LR도 cue run과 달랐다.
9. Gaussian slot lineage는 object trajectory가 아니다. 과거 state는 archive replay로 보존해야 한다.
10. MH가 없으므로 exact MCMC 또는 stationary-distribution preservation을 주장하지 않는다.

## 6. 새 연구 방향 결정

### 결정

공식 live-target MCMC relocation 튜닝은 중단한다. 다음 핵심 질문은 “fixed reference occlusion representation을 손상시키지 않고, 별도의 fixed-capacity residual pool이 unseen new geometry를 표현할 수 있는가?”로 재정의한다.

### Phase N0 — 완전 matched supervision/gating audit

모든 조건을 res4, O-SCD fixed pose, 동일 schedule/seed/LR로 고정하고 다음을 분리한다.

1. Cue target + cue-derived hard gate (현재 baseline)
2. GT-as-SSF target + 동일 cue-derived gate (target만 교체)
3. GT balanced BCE/Dice target + 동일 gate (loss 영향)
4. GT balanced BCE/Dice + all-reference slots retained/no hard gate (gating 영향)

판단:

- 2가 1보다 좋으면 cue quality가 병목
- 3이 2보다 좋으면 SSF objective가 병목
- 4가 3보다 좋으면 `state_valid` hard gate/occlusion capacity가 병목

### Phase N1 — Reference layer와 residual layer 분리

- `G_ref`: 전부 immutable. DC=0 occluder 포함, relocation 금지
- `state_ref_gate`: removal/reversion lifespan만 담당
- `G_residual`: fixed-capacity movable pool. addition/moved-to/new geometry만 담당
- 과거 state는 CPU/disk immutable archive
- 총 residual capacity는 고정하고 append/delete 없음

이 구조에서 reference Gaussian을 새 물체로 옮기지 않는다. “unused change cue”와 “unused representation capacity”를 구분한다.

### Phase N2 — GT oracle residual relocation upper bound

먼저 GT mask + GT/depth 또는 multi-view oracle destination으로 residual pool을 이동한다.

- 동일 fixed audit view에 pre/post masks 저장
- state-local replay buffer로 seen frames를 반복 학습
- current-frame-only update 금지
- no-relocation residual control과 비교

Stop gate:

- fixed N, archive drift 0, NaN 0
- no-relocation 대비 posthoc mIoU 최소 +1.0pp 또는 FN 10% 이상 감소
- seen-prefix와 posthoc gap 2pp 이하
- accepted relocation의 fixed-view positive coverage gain이 측정 가능해야 함

GT upper bound도 실패하면 cue proposal을 진행하지 않고 representation/loss를 재설계한다.

### Phase N3 — Cue-guided proposal

N2가 통과한 뒤에만 GT destination을 제거하고 causal cue로 대체한다.

- multi-view ray/epipolar consensus
- depth prior와 uncertainty
- destination별 held-out view validation
- source는 zero positive support뿐 아니라 alpha responsibility, negative-region occlusion responsibility, historical support를 함께 검사
- 명칭은 MCMC-inspired/residual-guided fixed-capacity relocation으로 제한

### Phase N4 — BOCD

Manual-boundary representation이 위 stop gate를 통과한 뒤 BOCD/tentative lifecycle을 추가한다. 그 전에는 boundary discovery를 섞지 않는다.

## 7. 다음 즉시 실행할 최소 실험

우선 N0의 4개 matched run을 seed 0으로 수행한다. 이 실험은 새 geometry proposal을 추가하지 않고, 현재 실패가 cue, loss, hard gating 중 어디에 있는지 가장 낮은 비용으로 분리한다. N0 결과가 나온 뒤 N1 residual pool의 용량과 초기화를 결정한다.

## 8. 주요 artifact 경로

- `outputs/instance1_scene_change1_2_3_temporal_confusion_oscd_cues_allframes_120/comparison.json`
- `outputs/instance1_scene_change1_2_3_temporal_geometry_confusion_oscd_cues_allframes_120/comparison.json`
- `outputs/oracle_boundary_mcmc/comparison.json`
- `outputs/oracle_boundary_mcmc/oracle_stream/conservative_burnin_comparison.json`
- `outputs/oracle_boundary_mcmc/oracle_stream/geo_conservative_occlusion_burnin32/online_relocation_confusion/`
- `docs/lifespan-scd-implementation-experiment-log-ko.md`
- `docs/oracle-boundary-mcmc-adaptation-report.md`
- `docs/conservative-residual-relocation-ko.md`

## 9. 재현성과 제한

- 이 checkpoint 이전 temporal/experiments/tests/docs 작업은 Git에서 대부분 untracked 상태여서 commit 단위 lineage가 없었다.
- 대용량 데이터, model checkpoint, state archive, raw event log는 Git에 포함하지 않고 로컬 `data/`와 `outputs/`에 유지한다.
- full matched MCMC seeds 0/1/2 sweep은 수행하지 않았다. Pilot 실패로 중단한 것이 올바른 stop-gate 동작이다.
- 현재 데이터셋 1개 결과이므로 일반화 성능은 주장하지 않는다.
