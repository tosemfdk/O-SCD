# Bayesian + DA3 viewer pipeline checkpoint — 2026-09-04

이 문서는 2026-09-04에 저장한 1--9번 viewer의 **이전 checkpoint**다. 2026-09-06
감사에서 historical camera/cue `k`를 최신 시점 `t`의 lifecycle population과 함께
렌더하던 timestamp mismatch가 확인되어 canonical 지위를 종료했다. 수정된 계약과
304-frame 결과는
[`bayesian-da3-historical-lifespan-replay-20260906-ko.md`](bayesian-da3-historical-lifespan-replay-20260906-ko.md)에
있다. 아래 내용은 당시 비교 기준을 보존하기 위한 기록이며 현재 launcher의 완전한
계약으로 사용하지 않는다.

## 재현 진입점

```bash
./run_bayesian_detector_viewer.sh
```

- Viewer: `http://localhost:8090`
- 실행 코드: `experiments/view_bayesian_detector_steps.py`
- Launcher: `run_bayesian_detector_viewer.sh`
- Machine-readable manifest:
  `docs/bayesian-da3-viewer-pipeline-checkpoint-20260904.json`
- Causal learned-sigmoid artifact:
  `outputs/stage2_l1_power_sigmoid_boundary_causal_prequential/learned_boundaries_causal.json`
- Causal DA3 proposal artifact:
  `outputs/causal_da3metric_scene123_panel7pos010_depthpos003_dynamiccoverage_20260904/da3_seed_replay.pt`
- Signed SAM/PCA trace:
  `outputs/e4_online_xfeat_new_seed_scene123_120_final_corrected_20260814/`

재현성 확인용 SHA-256:

- DA3 proposal artifact: `1aa135bae68b19590edeb2528963f6eea9272bf9b1302add86de0e1b69231a5f`
- Learned boundary artifact: `5031d760d54e8cacc146d40ada2a17de99db29fb093082135ebf47bd60bb6d85`
- 당시 launcher preset: `be93780423c43edef09044539d93ef85aa02b1d3148146ab2daa87fcdd908f41`
- 현재 launcher hash와 계약: 2026-09-06 replacement checkpoint 참조

## 고정된 핵심 조건

```text
depth scale fit:
  reference alpha >= 0.5
  AND Q < 0.2

panel 9 visualization:
  |panel 8| > 0.1
  AND |rendered GS depth - scaled DA3 depth| > 0.03

actual DA3 seed proposal:
  reference alpha >= 0.5
  AND
  panel 8 > +0.1
  AND rendered GS depth - scaled DA3 depth > +0.03
```

- 9번 시각화에는 양수와 음수 residual을 모두 남긴다.
- 실제 seed는 8번도 양수이고 depth residual도 양수인 교집합에만 생성한다.
- Seed birth에 별도의 `Q > 0.5` gate를 두지 않는다. Q는 이미 8번 값에 곱해져 있다.
- SAM magnitude top-40% gate를 사용하지 않는다.
- 고정 2 cm voxel occupancy를 사용하지 않는다.
- 최종 change mask threshold만 기존처럼 raw prediction `>= 0.5`를 사용한다.

## 1--9번 패널 계약

### 1. Main Gaussian layer

기본 선택은 committed lifecycle이다.

- Base Gaussian과 현재까지 태어난 DA3 seed를 같은 패널에 함께 렌더한다.
- 5번에 포함되는 DA3 sidecar가 1번 lifecycle에서 누락되지 않는다.

- 검정: `NEVER_OPEN`, 한 번도 OPEN되지 않은 reference-consistent Gaussian
- 초록: 현재 committed `OPEN`
- 빨강: 과거 OPEN된 뒤 현재 committed `CLOSED`
- Candidate는 lifecycle 색을 덮어쓰지 않는다.

Dropdown으로 다음 진단 layer를 선택할 수 있다.

- `Current cue projected to Gaussians`: 현재 frame Q의 alpha-transmittance positive
  pseudo-count `delta_a`, 범위는 0..1로 clamp
- `Accumulated Bayesian instability`: live candidate에서만
  `clamp(log BF / log 30, 0, 1)`
- `Learned current R_change prediction`: 5번과 같은 current-valid learned prediction

Scalar 진단 layer의 색은 `0=검정 -> 초록 -> 노랑 -> 1=빨강` 연속 gradient다.

### 2. Online RGB + DA3 seeds

- 배경: 현재 online RGB
- 초록 center: 이전 frame까지 실제 viewer에 accepted된 DA3 seed
- 흰 테두리 빨강 center: 현재 frame에서 새로 accepted된 seed
- 아직 도착하지 않은 미래 proposal은 표시하거나 model에 넣지 않는다.
- Overlay 위치는 birth xyz가 아니라 현재 학습된 seed xyz를 투영한다.

### 3. Learned sigmoid cue Q

현재 preset의 raw cue는 다음 순서로 만든다.

```text
P_l1pow = norm(0.8 * L1^0.3 + 0.2 * (1 - SSIM))
q         = P_l1pow * SAM
Q_t       = sigmoid(logit(0.95) * (q - tau_t) / width_t)
```

- 첫 frame: 고정 `tau=0.25`, `width=0.10`
- Frame t: frame `t-1`까지만 학습된 boundary network로 `tau_t,width_t`를 먼저 예측
- 현재 frame teacher update는 `Q_t` 고정 이후 수행되고 frame `t+1`부터 반영
- 표시: Q의 0..1 값을 검정→초록→노랑→빨강 heatmap으로 변환

### 4. BF (Bayes Factor)

- Live candidate가 없으면 0, 즉 검정
- Live candidate이면 `clamp(log BF / log 30, 0, 1)`
- 검정→초록→노랑→빨강 gradient
- 빨강 끝은 BF commit threshold 30에 도달했음을 뜻한다.
- 1번 lifecycle과 달리 candidate 진행도만 보여주는 진단 패널이다.

### 5. Current-valid learned R_change

- 현재 valid base Gaussian과 현재 materialize된 DA3 sidecar를 함께 렌더한다.
- Base `NEVER_OPEN`: geometry/opacity occlusion은 유지하고 change color는 정확한 RGB
  black으로 override
- Base/seed `CLOSED`: 렌더에서 제외
- Base/seed `OPEN`: 현재 학습된 change DC와 유효 geometry/opacity 사용
- 화면은 raw prediction 0..1의 연속 heatmap이다.
- 다음 6번 binary mask와 동일한 원본이며, 이 패널 자체는 threshold하지 않는다.

### 6. Predicted change mask

- 5번의 동일한 raw R_change tensor에 `>= 0.5`를 정확히 적용한다.
- 변화 pixel은 흰색, 나머지는 검정으로 표시한다.
- 별도 render나 후처리 threshold를 사용하지 않는다.

### 7. GT Change

- `GT ADD union GT REMOVE`의 흰색 union mask
- 평가와 육안 확인 전용
- Cue, detector, seed birth, representation 학습에는 절대 사용하지 않는다.

### 8. Upsampled signed SAM diff x learned Q

```text
signed SAM = PC1(SAM(reference render) - SAM(online RGB))
panel 8    = bilinear_upsample(signed SAM / max_abs_64x64) * Q
```

- SAM native grid는 64x64
- 양수: 빨강
- 음수: 파랑
- 0: 검정
- Panel 자체는 연속값을 표시하며 top-40% cut이나 hard Q gate가 없다.
- Seed용 threshold만 `panel 8 > +0.1`이다.

### 9. Scale-aligned GS - DA3 depth difference

1. 현재 DA3Metric-Large depth를 읽는다.
2. Immutable reference GS camera-z depth를 렌더한다.
3. `reference alpha >= 0.5 AND Q < 0.2`인 unchanged 영역에서 positive scale-only
   factor `s_t`를 robust log-ratio median으로 매 frame 추정한다.
4. Signed residual을 계산한다.

```text
d_t = rendered GS depth - s_t * DA3Metric depth
```

5. `|panel 8| > 0.1 AND |d_t| > 0.03`인 교집합만 남긴다.
6. 그 교집합의 `|d_t|` q95로 매 frame 정규화한다.

- 빨강 `d_t > +0.03`: online/DA3 표면이 reference GS보다 앞
- 파랑 `d_t < -0.03`: online/DA3 표면이 reference GS보다 뒤
- 그 외: 검정

중요: 9번의 빨강 전체가 seed가 되는 것은 아니다. 실제 seed는 여기에 다시
`panel 8 > +0.1`을 적용한 양수-SAM 교집합만 사용한다.

## 한 번의 `Next cue ->`가 수행하는 pipeline

1. 다음 online frame 한 장과 저장된 raw O-SCD cue를 읽는다.
2. 현재보다 과거의 boundary state만 사용해 learned sigmoid Q를 만든다.
3. Representation optimization 전에 immutable reference Gaussian에 Q의
   alpha-transmittance evidence를 정확히 한 번 계산한다.
4. Base Gaussian별 BF30 lifecycle detector를 갱신한다.
5. Commit된 event만 `OPEN`, `KEEP`, `CLOSE`, `REOPEN`으로 반영한다.
6. 현재 frame보다 먼저 생성된 seed에 대해서만 seed detector를 갱신한다. Detector는
   learned Q가 아니라 untouched cached `P+S > 0.5` binary cue의 pre-optimization
   alpha-T evidence를 한 번 사용한다.
7. 현재 timestamp의 offline DA3 proposal만 공개한다.
8. 이미 학습된 seed Gaussian coverage로 중복 proposal을 제거한다.
9. 현재 frame의 accepted seed는 `start=inf`, `end=inf`인 `NEVER_OPEN` 상태와
   stable `Beta(1,10)`으로 representation sidecar와 immutable detector probe에
   append한다. Birth frame은 geometry proposal에만 사용하며 BF evidence로 재사용하지
   않는다. 해당 seed의 detector evidence는 다음 신규 frame부터 시작한다.
10. 현재 frame의 representation target을 저장한다.
11. 이미 관측한 frame만 사용해 representation update를 120회 수행한다.
12. Current-valid base와 seed sidecar를 합쳐 5번 raw prediction을 한 번 렌더한다.
13. 동일 raw tensor에 `>= 0.5`를 적용한 6번 binary prediction mask를 만든다.
14. 마지막으로 evaluation-only GT union을 7번에 표시한다.

## Bayesian lifecycle 고정값

- 최초 committed stable prior: `Beta(flip=1, keep=10)`
- Fresh reset candidate prior: `Beta(flip=1, keep=1)`
- Commit threshold: `BF >= 30`
- CLOSED에서 cue-positive는 FLIP, OPEN에서 cue-negative는 FLIP
- Candidate가 살아 있는 동안 committed stable posterior는 동결
- Commit 뒤 initial `Beta(1,10)`을 재주입하지 않는다.
- Reset posterior 자체를 새 상태 좌표로 swap한다. 예를 들어 FLIP 3, KEEP 0이면
  이전 좌표 `Beta(4,1)`, 새 OPEN 좌표 `Beta(1,4)`다.
- Representation update, learned DC/geometry, replay render는 detector evidence로
  역류하지 않는다.
- DA3 seed birth frame은 proposal-only다. 같은 관측으로 proposal을 만들고 곧바로
  검증하는 selection bias를 막기 위해 seed BF evidence는 다음 신규 frame부터 받는다.

## Representation 학습 계약

### View sampling

- Frame당 120 updates
- 각 update에서 가장 최근 frame을 고르는 명시적 branch 확률: 0.33
- 나머지 0.67: 이미 관측한 `[0,t]` frame을 균등 sampling
- 미래 frame 접근 금지

### DC objective: joint original O-SCD SSF

Base와 DA3에 서로 다른 DC loss를 두지 않는다. Sampled view에서 lifecycle-valid base와
DA3를 panel 5와 같은 population으로 합쳐 한 번 렌더하고, 전체 learned cue `Q`에 원본
O-SCD SSF를 한 번 적용한다.

```text
R_joint = Render(
    base OPEN learned DC
  + base NEVER_OPEN black occluder
  + DA3 OPEN learned DC
  + DA3 NEVER_OPEN black occluder
)

P(p) = sigmoid(meanRGB(R_joint(p)))

L_detect = mean_p[Q(p) * (1 - P(p))]
L_sparse = log(mean_p[P(p)]^2 + 1)
L_DC     = L_detect + L_sparse
```

- Base OPEN DC와 DA3 OPEN DC가 동일한 `L_DC`에서 함께 gradient를 받는다.
- Base/DA3 CLOSED는 joint render에서 제외한다.
- NEW/REMOVE sign이나 `new_target`을 DC loss에 사용하지 않는다.
- 9-point footprint BCE를 사용하지 않는다.
- Part19 seed-only projected coverage DC loss를 사용하지 않는다.
- 최종 panel 5는 raw render를 0..1로 clamp하지만, 학습 loss 안에서는 원본 O-SCD와
  동일하게 raw render에 sigmoid를 적용한다.

### Base R_change

- 현재 OPEN이고 sampled view에서 visible한 row만 joint original O-SCD loss로 DC 학습
- NEVER_OPEN은 frozen black occluder
- CLOSED는 숨김
- Immutable reference geometry/opacity는 detector와 분리되어 보존

### DA3 sidecar

- Seed birth target: `reference alpha >= 0.5 AND panel 8 > +0.1 AND depth residual > +0.03`
- Birth lifecycle: `start=inf`, `end=inf`, `NEVER_OPEN`, stable `Beta(1,10)`
- `NEVER_OPEN`: 전체 unsigned `Q` coverage로 xyz, scale, rotation만 학습; DC와
  opacity 고정
- `OPEN`: DC는 base와 동일한 joint original O-SCD SSF로 학습하고, xyz, opacity,
  scale, rotation은 전체 unsigned `Q` coverage로 학습
- `CLOSED`: 숨기고 학습하지 않음
- Detector probe는 birth geometry로 고정되어 representation 학습과 분리
- Geometry trust region:
  - xyz displacement: birth scale의 최대 4배
  - scale: birth scale의 0.25..4배
  - opacity: 0.01..0.99

현재 representation loss에서는 NEW/REMOVE를 분리하지 않는다. Signed SAM과 positive
depth residual은 DA3 geometry proposal 위치를 정할 때만 사용한다. Loss-level semantic
분리는 후속 실험 범위다.

### 학습된 Gaussian coverage

고정 2 cm voxel 대신 현재 학습된 seed geometry로 새 proposal 중복을 제거한다.

- Coverage eligibility: 실제 geometry optimizer update를 최소 4회 받은 row
- Center: 현재 학습된 xyz
- Radius: 현재 세 축 scale 최댓값의 2배
- `NEVER_OPEN`과 OPEN은 coverage에 참여
- CLOSED는 coverage에서 제외
- 같은 frame의 아직 학습되지 않은 proposal끼리는 서로 막지 않음

## 현재 proposal artifact와 audit

Offline artifact는 dynamic learned-Gaussian coverage 적용 전 proposal bank다.

- SC1: 16,842
- SC2: 24,206
- SC3: 24,631
- Total: 65,679
- `future_view_accesses = 0`
- `ground_truth_birth_accesses = 0`
- `reference_geometry_mutated = false`

실제 viewer materialization 수는 학습된 geometry coverage rejection 때문에 이보다
작으며, sidebar에서 frame별 `proposed / accepted / coverage-rejected`를 확인한다.

## 검증 기준

최종 joint-loss/u120 checkpoint 생성 직후 다음을 통과했다.

- 관련 unit/regression tests: 67 passed
- Python compile: passed
- `git diff --check`: passed
- Joint original O-SCD GPU smoke: 14 frames passed
  - frame 10부터 DA3 seed OPEN 및 joint backward 확인
  - frame 13 기준 active DA3 seed 56
- Runtime smoke:
  - SC1 frame 8: proposal 91, accepted 91
  - SC1 frame 9: proposal 104, accepted 70, learned coverage reject 34
  - Panel 9: red 3,356 px, blue 2,208 px
  - 첫 seed birth frame: accepted seed 전부 `NEVER_OPEN`, OPEN/CLOSE event 0
  - 새 seed posterior: stable `Beta(1,10)`, candidate 없음; 다음 frame부터 evidence
