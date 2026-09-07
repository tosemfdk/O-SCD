# O-SCD-evolving 전체 파이프라인과 현재 Bayesian + DA3 viewer

> 기준일: 2026-09-06
>
> 현재 실행 코드: `experiments/view_bayesian_detector_steps.py`
>
> 실행 진입점: `run_bayesian_detector_viewer.sh`
>
> 최신 기본 launcher: `--training-partition panel10_new`. Panel10의 NEW를
> DA3 seed birth와 seed DC+geometry SSF에 사용하고, 나머지 Q는 base DC SSF로
> 학습한다. 모든 NEVER_OPEN은 frozen black occluder이며 최종 mask는 joint render다.
> 자세한 최신 계약은 아래 **28절**을 우선한다. 앞선 coverage loss 설명과 정량
> 결과는 `--training-partition legacy` 경로의 기록이며 새 방법의 성능 결과가 아니다.
>
> 후속 수정: 현재 frame의 DA3 seed를 먼저 생성한 뒤 **base+seed 공동 alpha-T
> evidence 1회 / 단일 BF30 tracker**로 전체 learned-sigmoid soft `Q_t`를 처리한다.
> 기존의 base-only detector + 다음 frame부터 seed 별도 판단은 legacy 기록이다. 아래 기존 304-frame/14-frame 결과는 수정 전 binary seed detector의
> 기록이며, 새 detector 설정의 측정값으로 재해석하지 않는다.

이 문서는 다음 두 층을 한 번에 설명한다.

1. **O-SCD-evolving 전체 시스템이 해결하려는 문제와 설계 계약**
2. **그 설계를 현재 Bayesian + DA3 step viewer가 실제로 어떻게 구현하는지**

연구 목표와 현재 구현 범위를 섞지 않기 위해, 이상적인 최종 목표와 현재 viewer의
제약을 명시적으로 구분한다. 문서의 수식은 실제 코드 경로를 기준으로 적었다.

문서 우선순위도 구분한다.

- 이 문서: 현재 실행 중인 viewer variant와 O-SCD-evolving 전체 구조
- `bayesian-da3-historical-lifespan-replay-20260906-ko.md`: 직전 304-frame 정량 실험
- `current_pipeline_0901.md`: 다른 이전 runner를 기록한 2026-09-01 문서이며 현재
  viewer의 source of truth가 아님

---

## 1. 궁극적 연구 목표

### 1.1 한 문장 research vision

> **시간에 따라 계속 진화하는 장면을 장기간 관측하면서 연속적인 변화들을 검출하고,
> one-step scene change detection보다 우수한 성능을 내며, 변화 종류와 변화 사이의
> 관계를 질의할 수 있는 compact 3D change history를 구축하는 최초의 long-term
> scene change detection module을 목표로 한다.**

영문 contribution target은 다음처럼 정리한다.

> The first long-term scene change detection module for continuously evolving
> scenes, designed to outperform one-step scene change detection methods while
> maintaining a compact 3D history of change types and their relations.

여기서 `first`는 현재 문서가 이미 입증된 사실로 선언하는 표현이 아니라, 논문에서
문헌 조사와 비교 범위 정의를 통해 입증해야 하는 **novelty claim target**이다.

### 1.2 세 가지 최종 기여 목표

#### Goal A — Continuous long-term scene change detection

한 번의 `reference -> changed scene`만 처리하는 것이 아니라, 동일 공간이 여러 번
변하는 긴 stream을 끊지 않고 처리한다.

```text
reference -> SC1 -> SC2 -> SC3 -> ... -> SCn
```

각 시점에는 current change mask를 출력하고, 전체 stream에 대해서는 변화의 시작,
지속, 종료, 재발생을 기록해야 한다.

#### Goal B — One-step change detection보다 우수한 성능

연속 stream을 독립적인 one-step 문제로 잘라 푸는 방법이나 기존 persistent O-SCD보다
단순히 더 많은 history를 저장하는 것이 목적이 아니다. 동일한 causal input, pose,
update budget, evaluation protocol 아래에서 long-term temporal reasoning이 실제로
다음 성능을 개선해야 한다.

```text
현재 변화 mask 정확도
장기 누적 뒤의 contamination 억제
변화 시작/종료 시점 정확도
반복 변화 구간의 안정성
```

따라서 “superior performance”는 설계 의도가 아니라 최종 실험에서 검증해야 할
성공 조건이다.

#### Goal C — Change type과 relation을 담는 compact 3D history

모든 frame, 모든 시점의 전체 Gaussian checkpoint를 그대로 저장하는 대신, 3D 공간의
변화 support와 상태/event metadata를 압축해 보존하는 history를 목표로 한다.

개념적인 최종 history는 다음과 같이 표현할 수 있다.

```text
H_t = (V_t, E_t)

change node v in V_t = {
    3D support 또는 Gaussian subset,
    change type,
    start timestamp,
    end timestamp,
    confidence,
    필요한 compact appearance/geometry residual
}

relation edge e in E_t = {
    source change node,
    target change node,
    temporal/spatial relation,
    confidence
}
```

최종적으로 표현하려는 change type의 예시는 `NEW`, `REMOVED`, `MOVED`, `REVERTED`,
`CHANGED_AGAIN` 등이다. Relation의 예시는 다음과 같다.

```text
같은 3D 영역의 이전/다음 상태
한 변화가 지속되거나 종료됨
reference 상태로 복귀함
기존 상태가 새 상태로 대체됨
이동 전 support와 이동 후 support의 대응
시간적 before/after 및 공간적 overlap/correspondence
```

정확한 type taxonomy와 relation label set은 최종 task definition에서 고정해야 한다.
현재 viewer는 아직 이 typed relational graph 전체를 구현하지 않았다.

### 1.3 최종 목표와 현재 방법의 계층

Lifespan과 Bayesian detector 자체가 최종 contribution은 아니다. 이들은 위 세 목표를
달성하기 위한 현재 방법론이다.

```text
궁극적 연구 기여
  1. continuous long-term SCD
  2. one-step SCD 대비 우수한 성능
  3. compact typed-and-relational 3D change history

이를 위한 현재 방법
  - Gaussian lifespan interval
  - Bayesian flip/keep changepoint detector
  - timestamp-aligned historical replay
  - shared persistent Gaussian parameter
  - current-valid rendering
  - DA3 depth-prior NEW geometry birth
```

현재-valid mask는 중요한 online output이지만 최종 연구 기여의 전부가 아니다. Lifespan은
장기 history를 압축하고 현재 population을 복원하기 위한 temporal indexing mechanism이며,
Bayesian detector는 그 interval을 언제 열고 닫을지 결정하는 mechanism이다.

### 1.4 현재-valid output이 필요한 이유

기존 persistent O-SCD는 여러 view의 change cue를 하나의 `R_change`에 누적하여
멀티뷰 일관성을 얻는다. 그러나 장면이 촬영 중 다시 변하면 과거 상태의 cue가 계속
남을 수 있다.

```text
reference 상태
  -> SC1: 물체 이동
  -> SC2: 다시 이동
  -> SC3: 제거 또는 새로운 물체 등장
```

O-SCD-evolving의 목표는 과거 변화의 합집합을 그리는 것이 아니다.

```text
원하지 않는 출력 = 지금까지 한 번이라도 변했던 모든 영역
원하는 출력       = timestamp t에서 실제로 유효한 변화 영역
```

현재 방법에서는 이를 위해 Gaussian별로 변화가 유효했던 half-open interval을
보존한다.

```text
L_i = { [start_i,0, end_i,0), [start_i,1, end_i,1), ... }
```

---

## 2. 시스템 성공 조건과 현재 detector 범위

### 2.1 End-to-end 기능 성공 조건

시간 `t`에서 미래 frame 없이 다음을 수행할 수 있어야 한다.

1. `I_t`가 immutable reference `R_ref`와 다른 위치를 검출한다.
2. Gaussian별 현재 evidence가 기존 상태와 충돌하는지 판단한다.
3. 변화의 시작과 종료를 causal하게 검출한다.
4. 종료된 변화가 current output을 오염시키지 않도록 lifespan을 닫는다.
5. 다시 변화하면 lifespan을 다시 연다.
6. 모든 과거 관측을 활용하되, 각 replay image에는 그 image timestamp의 Gaussian
   population을 복원한다.
7. 최종 mask에는 `t`에서 유효한 변화만 남긴다.
8. 장기 stream이 끝난 뒤에는 변화 type과 relation을 포함하는 compact 3D history를
   질의할 수 있어야 한다.
9. 동일 조건의 one-step change detection baseline보다 정량적으로 우수해야 한다.

### 2.2 논문 수준 평가 축

세 최종 목표를 평가하려면 current mask mIoU/F1 하나만으로는 부족하다.

| 목표 | 필요한 평가 예시 |
|---|---|
| Continuous detection | framewise mIoU/F1, state별 성능, 긴 stream 후 성능 저하, repeated-change 구간 성능 |
| Temporal event | OPEN/CLOSE/change-point precision·recall, boundary delay, false transition/chattering |
| One-step 대비 우수성 | 동일 causal protocol과 compute budget의 one-step/persistent baseline 비교 |
| Compact history | 저장 byte 수, Gaussian/residual/event 수, frame 수 증가에 따른 memory scaling |
| Type 표현 | NEW/REMOVED/MOVED/REVERTED 등 type 정확도 |
| Relation 표현 | successor/revert/replacement/correspondence edge 정확도 및 history query 정확도 |

특히 compactness는 단순히 파일 크기가 작다는 뜻이 아니다.

```text
full frame/checkpoint history를 저장하지 않으면서도
현재 상태 복원 + 과거 상태 질의 + type/relation 질의를 보존
```

이 세 기능을 유지하는 데 필요한 memory와 accuracy의 trade-off를 함께 보고해야 한다.

### 2.3 현재 구현된 detector의 확정 범위

현재 detector가 Gaussian별로 추적하는 상태는 **binary reference-change validity**다.

```text
0 = reference-consistent / inactive
1 = reference-different / active
```

따라서 현재 구현이 직접 표현하는 전이는 다음과 같다.

```text
inactive -> active   : OPEN 또는 개념적 REOPEN
active   -> active   : KEEP
active   -> inactive : CLOSE
inactive -> inactive : NONE
```

중요한 범위 제한은 다음과 같다.

```text
SC1과 SC2가 둘 다 reference와 다르면
active(SC1) -> active(SC2) = KEEP
```

즉, 두 changed appearance가 의미적으로 다르더라도 둘 다 reference-different이면
현재 detector는 새 lifespan을 만들지 않는다. **active-A -> active-B 의미 상태 분리**는
전체 연구 목표에는 관련되지만 현재 binary detector의 구현 범위는 아니다.

---

## 3. 입력, 출력, 핵심 기호

### 3.1 입력

| 기호/항목 | 의미 |
|---|---|
| `R_ref` | 고정된 reference 3D Gaussian Splatting 표현 |
| `I_t` | timestamp `t`에 도착한 online RGB frame |
| `C_cache,t` | 기존 fixed-pose cache에 저장된 O-SCD `P + S` cue |
| `Q_t` | 현재 viewer가 detector와 representation에 쓰는 learned-sigmoid cue |
| `D_DA3,t` | DA3Metric이 예측한 online camera-z depth |
| camera `K_t, W2C_t` | 고정 pose protocol의 intrinsics/extrinsics |

현재 viewer의 기본 data 경로는 다음과 같다.

```text
source stream:
  data/Instance_1/scene_change1_2_3

fixed cameras:
  /home/rvl/workspace/github/O-SCD/output/
  ESCD_fixedpose_protocols_res4/scene_change1_2_3/cameras_fixed.json

cached P+S cue:
  /home/rvl/workspace/github/O-SCD/artifacts/
  escd_396ref/fixed_pose_cues_res4_v1
```

### 3.2 현재 viewer 출력과 궁극적 출력

현재 viewer가 매 시점 `t`에 직접 만드는 핵심 online 출력은 다음 두 값이다.

```text
S_t(p)    = current-valid learned change score
Mask_t(p) = 1[S_t(p) >= 0.5]
```

평가용 `GT ADD union REMOVE`는 cue, detector, seed birth, optimizer에 사용하지 않고
step이 끝난 뒤 비교·표시에만 사용한다.

궁극적인 O-SCD-evolving module은 여기에 compact 3D history를 추가로 출력해야 한다.

```text
Online output at t:
  current change score S_t
  current change mask Mask_t
  current 3D active change state

Long-term history output up to t:
  typed 3D change nodes
  temporal/spatial relation edges
  compact state/residual representation
```

### 3.3 두 종류의 Gaussian bank

현재 viewer는 다음 population을 함께 사용한다.

1. **Base `R_change` rows**
   - `R_ref`의 topology와 geometry를 복사한 row
   - Gaussian별 learnable change DC를 가짐
   - 현재 variant에서는 base geometry가 고정됨
2. **DA3 NEW seed sidecar rows**
   - reference에 없는 앞쪽 표면을 보충하기 위한 dynamic row
   - causal birth timestamp 이후에만 존재함
   - OPEN이면 DC, xyz, opacity, scale, rotation을 학습함

Detector용 immutable probe와 representation용 learnable parameter는 분리한다.

---

## 4. 전체 시스템 구조

```text
                         immutable R_ref
                              |
             +----------------+----------------+
             |                                 |
             v                                 v
     reference RGB/depth                  alpha-T probe
             |                                 |
I_t ---------+--> pixel/SAM cue --> Q_t -------+--> base BF30 detector
 |                                               --> base lifespan history
 |
 +--> signed SAM/PCA + aligned DA3 depth
 |         |
 |         +--> causal NEW proposals
 |                  |
 |                  +--> learned-coverage rejection
 |                  +--> NEVER_OPEN seed materialization
 |                  +--> fixed birth-geometry BF30 seed detector
 |
 +--> replay bank {I_k, Q_k, camera_k}, k <= t
                 |
                 +--> lifespan query at k
                 |      base/seed OPEN, NEVER_OPEN, CLOSED, future-born
                 |
                 +--> joint DC SSF loss with 2Q_k
                 +--> OPEN DA3 geometry coverage loss with 2Q_k
                 +--> masked Adam on k-valid visible rows
                              |
                              v
                 current-valid render at t
                              |
                              v
                   score and binary mask
```

핵심은 detector와 representation을 같은 것으로 취급하지 않는 것이다.

| 축 | 질문 | 현재 입력 |
|---|---|---|
| Detector evidence | 이 reference Gaussian footprint가 cue 안/밖에 얼마나 놓였는가? | 신규 frame의 pre-optimization raw cue |
| Lifespan | 이 row가 timestamp `k`에 살아 있었는가? | BF30 commit으로 만든 interval history |
| Representation | 살아 있는 row의 DC/geometry가 여러 view cue를 얼마나 설명하는가? | causal replay loss |
| Output | 지금 `t`에서 유효한 change render는 무엇인가? | `t`-valid population |

---

## 5. 지켜야 하는 불변 조건

### 5.1 Online causality

```text
timestamp t의 detector/optimizer가 접근 가능한 frame = {0, 1, ..., t}
```

미래 image, 미래 cue, 미래 seed birth, 미래 GT는 접근할 수 없다.

### 5.2 Detector는 representation 학습과 독립

Base detector는 매 신규 frame에서 representation optimization **전에** immutable
reference geometry/opacity로 evidence를 한 번 계산한다.

다음 값은 detector 입력으로 다시 들어가지 않는다.

```text
learned change DC
optimized representation geometry/opacity
DC와 Q의 차이
optimizer가 parameter를 움직인 양
같은 frame의 post-optimization render
historical replay view
GT mask
```

DA3 seed detector도 학습된 seed geometry가 아니라 별도의 fixed birth-geometry probe를
사용한다.

### 5.3 Current-state validity

현재 output은 `t`에서 OPEN 또는 필요한 black occluder인 row만 렌더한다.

```text
OPEN       : learned change appearance로 렌더
NEVER_OPEN : black occluder로 렌더
CLOSED     : output render에서 제외
future-born: 존재하지 않는 row로 처리
```

### 5.4 Timestamp-aligned historical replay

과거 frame `k`를 뽑으면 camera, cue, lifespan을 모두 `k`에 맞춘다.

```text
camera     = I_k의 camera
target     = Q_k 또는 2Q_k
population = L_base(k), L_seed(k)
```

과거 camera/cue `k`에 최신 `t`의 lifespan을 붙이지 않는다.

### 5.5 한 row의 parameter는 replay view 사이에서 공유

Lifespan interval마다 DC/geometry parameter snapshot을 따로 만들지 않는다.

```text
하나의 Gaussian row
  -> 하나의 persistent DC/geometry/Adam state
  -> 여러 causal replay view가 공동 최적화
  -> lifespan은 timestamp별 참여 여부만 결정
```

이 parameter sharing이 O-SCD의 멀티뷰 fusion을 유지하는 방법이다. CLOSE 뒤 REOPEN해도
새 interval metadata는 만들지만 parameter와 Adam moment는 보존한다.

---

## 6. Stage 0 — 초기화

### 6.1 Immutable reference probe

Reference PLY의 다음 값은 detector probe용으로 고정한다.

```text
x_ref, opacity_ref, scale_ref, rotation_ref
```

모든 raw reference parameter는 `requires_grad=False`다. Base detector의 alpha-T
responsibility는 항상 이 immutable field에서 계산한다.

### 6.2 Base `R_change`

Base representation은 reference row별 change DC를 별도로 둔다.

```text
change_dc_i = 0                  # raw SH DC initialization
xyz_i       = xyz_ref,i          # current variant: frozen
scale_i     = scale_ref,i        # current variant: frozen
rotation_i  = rotation_ref,i     # current variant: frozen
opacity_i   = opacity_ref,i      # current variant: frozen
```

Degree-zero SH의 RGB 관계는 다음과 같다.

```text
RGB_i = 0.5 + C0 * DC_i
C0    ~= 0.282095
```

따라서 raw `DC=0`은 rendered RGB `0.5`다. 정확한 black은 다음 coefficient다.

```text
DC_black = RGB2SH(0) ~= -1.77245
```

Viewer는 `NEVER_OPEN`을 단순 raw zero로 두지 않고 `DC_black`으로 override한다.

### 6.3 Lifespan 초기화

모든 base row는 reference 초기화 시점부터 materialize되어 있지만 아직 lifespan이
열리지 않은 `NEVER_OPEN`이다.

DA3 sidecar는 비어 있다. DA3 proposal은 해당 birth frame이 실제로 도착할 때만
materialize한다.

### 6.4 Bayesian 초기화

각 row의 committed binary 상태는 inactive이며 stable prior는 다음과 같다.

```text
Stable_i = Beta(flip=1, keep=10)
Fresh reset candidate prior = Beta(flip=1, keep=1)
BF commit threshold = 30
```

---

## 7. Stage 1 — RGB와 cached O-SCD cue 읽기

한 번의 `Next cue ->`에서 먼저 신규 frame `I_t`와 cached sum cue를 읽는다.

Cached artifact에는 원래 O-SCD pixel cue `P_0,t`와 SAM cue `S_t`의 합만 있다.

```text
C_cache,t = P_0,t + S_t
```

Viewer는 SAM cue를 다시 추론하지 않고 reference RGB render로 pixel cue를 재계산하여
cached sum에서 `S_t`를 복원한다.

### 7.1 원래 pixel cue

Pixel `p`에서 photometric L1과 structural difference를 계산한다.

```text
L_t(p) = mean_c |R_ref,t,c(p) - I_t,c(p)|
D_t(p) = 1 - mean_c SSIM_map_c(R_ref,t, I_t)(p)

U_0,t(p) = 0.8 * L_t(p) + 0.2 * D_t(p)

P_0,t(p) = [U_0,t(p) - min_p U_0,t]
           / [max_p U_0,t - min_p U_0,t + 1e-8]
```

SAM cue 복원은 다음과 같다.

```text
S_t(p) = clamp(C_cache,t(p) - clamp(P_0,t(p), 0, 1), 0, 1)
```

---

## 8. Stage 2 — 현재 L1-power product cue와 learned sigmoid `Q_t`

현재 preset은 전체 pixel cue에 power를 주는 것이 아니라 **L1 항에만** exponent
`0.3`을 적용한다.

```text
U_alpha,t(p) = 0.8 * L_t(p)^0.3 + 0.2 * D_t(p)

P_alpha,t(p) = minmax_p(U_alpha,t(p))
```

그 뒤 pixel과 SAM cue를 곱하고 원래 O-SCD cue 범위 `0..2`에 맞춘다.

```text
C_pre,t(p) = clamp(2 * P_alpha,t(p) * S_t(p), 0, 2)
q_t(p)     = clamp(C_pre,t(p) / 2, 0, 1)
           = P_alpha,t(p) * S_t(p)
```

주의: 현재 preset은 `P_alpha^0.3 * S`가 아니다. `0.3` power는 `P`를 만든 뒤가
아니라 min-max normalization 전의 L1 항에 이미 적용됐다.

### 8.1 Causal learned-sigmoid remap

Frame별 boundary artifact의 `tau_t`, `width_t`를 이용한다.

```text
ell = log((1 - 0.05) / 0.05) = logit(0.95)

Q_t(p) = sigmoid(ell * [q_t(p) - tau_t] / width_t)
```

`width_t`는 transition half-width다.

```text
q = tau_t - width_t  -> Q ~= 0.05
q = tau_t + width_t  -> Q ~= 0.95
```

첫 frame의 기준값은 `tau=0.25`, `width=0.10`이다. Artifact 생성 시 frame `t`의
boundary는 `t-1`까지의 학습 state로 먼저 예측하고, 현재 teacher update는 다음
frame부터 반영하는 prequential 순서를 사용했다.

```text
predict: theta_(t-1), histogram_t -> tau_t, width_t -> Q_t
update : theta_(t-1), teacher_t   -> theta_t
```

Viewer 자체는 이 network를 다시 학습하지 않고 저장된 causal per-frame
`tau_t,width_t` artifact를 읽는다.

Viewer 내부에는 다시 `0..2` 범위로 저장한다.

```text
view.candidate_map = 2Q_t
normalized cue     = clamp(candidate_map / 2, 0, 1) = Q_t
```

---

## 9. Stage 3 — Base Gaussian alpha-transmittance evidence

### 9.1 Immutable differentiable probe

Gaussian `i`가 pixel `p`에 주는 alpha-transmittance responsibility를 다음처럼 둔다.

```text
rho_i,t(p) = alpha_i,t(p) * T_i,t(p)
```

여기서 `T_i,t(p)`는 Gaussian `i` 앞까지 남아 있는 transmittance다.

Viewer는 immutable reference geometry/opacity에 differentiable dummy RGB color를
붙여 한 번 VJP를 계산한다.

```text
VJP channel 0 weight = Q_t(p)
VJP channel 1 weight = 1 - Q_t(p)
VJP channel 2 weight = 0
```

그 결과 Gaussian별 raw evidence mass는 다음과 같다.

```text
e_i,t(+) = sum_p rho_i,t(p) * Q_t(p)
e_i,t(-) = sum_p rho_i,t(p) * [1 - Q_t(p)]
M_i,t    = e_i,t(+) + e_i,t(-)
```

이 값은 learned DC와 cue의 agreement가 아니다. Reference Gaussian footprint가 현재
cue 안과 밖에 각각 얼마나 투영되는지를 뜻한다.

### 9.2 Capped fractional pseudo-count

현재 viewer는 `count_mode=capped`, `mass_saturation=1`을 사용한다.

```text
q_i,t = e_i,t(+) / [M_i,t + eps]
w_i,t = clamp(M_i,t / 1, 0, 1)

delta_i,t(+) = w_i,t * q_i,t
delta_i,t(-) = w_i,t * [1 - q_i,t]
```

따라서 한 frame이 한 Gaussian에 주는 총 pseudo-count는 최대 1이다.

```text
delta_i,t(+) + delta_i,t(-) = w_i,t <= 1
```

`M_i,t < 1e-6`이면 해당 row는 이번 frame에서 unobserved다.

---

## 10. Stage 4 — Flip/keep single-candidate Beta BF30

Detector의 Beta 좌표는 단순 change/non-change가 아니라 현재 committed bit에 대한
`FLIP/KEEP`이다.

```text
현재 CLOSED 또는 NEVER_OPEN:
  FLIP = delta(+)
  KEEP = delta(-)

현재 OPEN:
  FLIP = delta(-)
  KEEP = delta(+)
```

Stable posterior를 `Beta(a_s,b_s)`, 현재 candidate block의 누적 flip/keep count를
`(A,B)`라고 하자.

Beta-binomial block predictive는 다음과 같다.

```text
log p(A,B | a,b)
  = log BetaFn(a + A, b + B) - log BetaFn(a,b)
```

Fresh reset과 기존 stable 가설을 비교한다.

```text
logBF
  = log p(A,B | Beta(1,1))
  - log p(A,B | Beta(a_s,b_s))
```

현재 threshold는 다음과 같다.

```text
BF >= 30
logBF >= log(30) ~= 3.4012
```

Candidate transition은 다음과 같다.

```text
logBF <= 0
  -> reject
  -> live block 전체를 stable posterior에 merge

0 < logBF < log(30)
  -> candidate start 또는 continue
  -> stable posterior freeze
  -> 다음 신규 observed view의 evidence를 block에 추가

logBF >= log(30)
  -> reset commit
  -> committed binary bit toggle
```

Commit 뒤 winning reset posterior의 FLIP/KEEP 좌표를 새 bit 관점으로 swap한다. 초기
stable `Beta(1,10)`을 다시 주입하지 않는다.

---

## 11. Stage 5 — Lifespan mutation과 timestamp query

### 11.1 Half-open interval

각 row는 최대 16개 interval metadata를 갖는다.

```text
active_i(k)
  = exists slot s:
      state_valid_i,s
      AND start_i,s <= k < end_i,s
      AND materialized_i <= k
```

`[start,end)`이므로 CLOSE가 발생한 `end` timestamp에서는 이미 inactive다.

### 11.2 상태 정의

```text
OPEN at k
  = active_i(k)

NEVER_OPEN at k
  = materialized_i <= k
    AND k까지 시작된 valid interval이 없음

CLOSED at k
  = materialized_i <= k
    AND k까지 적어도 한 번 OPEN
    AND not active_i(k)

future-born at k
  = materialized_i > k
```

### 11.3 Transition 결과

| 이전 상태 | BF30 commit | 결과 |
|---|---|---|
| inactive | 없음 | `NONE` |
| inactive | 있음 | 새 interval을 여는 `OPEN`; 과거 interval이 있으면 개념적으로 `REOPEN` |
| active | 없음 | `KEEP` |
| active | 있음 | 현재 interval을 닫는 `CLOSE` |

구현 action enum은 첫 OPEN과 REOPEN을 모두 `OPEN`으로 기록하지만 interval history를
보면 재개인지 구분할 수 있다.

### 11.4 Parameter sharing

Interval slot에는 시간 metadata만 저장된다.

```text
저장하는 것:
  start, end, valid, current slot, materialized timestamp

interval마다 따로 저장하지 않는 것:
  DC, xyz, opacity, scale, rotation, Adam moment
```

과거 `k`를 replay할 때 parameter 값을 과거 snapshot으로 되돌리는 것이 아니다.
현재 shared parameter를 사용하되 `k`에서 살아 있었던 row만 선택하여 그 parameter가
과거와 현재 causal view를 함께 설명하도록 최적화한다.

---

## 12. Stage 6 — DA3 NEW proposal 생성

Reference topology만으로 표현하기 어려운 새 앞쪽 표면을 위해 DA3 sidecar proposal을
사용한다. 현재 viewer는 매 frame unrestricted birth를 새로 계산하지 않고, 미리 만든
**causal proposal artifact**를 birth timestamp 순서대로 공개한다.

```text
outputs/
  causal_da3metric_scene123_panel7pos010_depthpos003_dynamiccoverage_20260904/
  da3_seed_replay.pt
```

Artifact에는 총 `65,679`개 proposal이 있으며 future-view/GT birth access audit는 0이다.

### 12.1 Signed SAM/PCA support

Reference render와 online RGB의 SAM feature 차이를 causal PC1 축에 투영한다.

```text
Delta_f_t = SAM(R_ref,t) - SAM(I_t)
s_64,t    = reshape(Delta_f_t * v_t, 64, 64)

s_norm,t  = s_64,t / max_abs(s_64,t)
H_t       = bilinear_upsample(s_norm,t) * Q_t
```

`H_t`의 양수는 viewer panel 8의 red, 음수는 blue다. 실제 NEW birth는 causal하게
고정된 NEW sign의 positive support만 사용한다.

```text
H_t(p) > +0.1
```

별도의 `Q >= 0.5` gate는 없다. `Q`는 이미 `H_t`에 연속값으로 곱해져 있다.

### 12.2 DA3 depth scale alignment

Reference alpha가 충분하고 change cue가 낮은 영역을 scale anchor로 사용한다.

```text
A_scale,t(p)
  = 1[alpha_ref,t(p) >= 0.5 AND Q_t(p) < 0.2]
```

Positive scale-only factor를 robust log-ratio로 맞춘다.

```text
r_p = log D_ref,t(p) - log D_DA3,t(p)
s_t = exp(median_inlier(r_p))
```

초기 median에서 MAD 기반 inlier를 고른 뒤 inlier median으로 `s_t`를 계산한다. Affine
depth shift는 사용하지 않는다.

Signed depth residual은 다음과 같다.

```text
d_t(p) = D_ref,t(p) - s_t * D_DA3,t(p)
```

`d_t > 0`이면 online DA3 surface가 reference surface보다 camera 쪽에 있다.

### 12.3 실제 proposal gate

```text
proposal_t(p)
  = [alpha_ref,t(p) >= 0.5]
    AND [H_t(p) > +0.1]
    AND [d_t(p) > +0.03]
```

4-pixel stride cell마다 우선순위가 가장 높은 한 pixel을 선택하며 frame당 최대
2,048개를 만든다. 선택 pixel의 aligned depth를 world coordinate로 unproject한다.

```text
x_cam   = s_t * D_DA3,t(p) * inverse(K_t) * [u, v, 1]^T
x_world = inverse(W2C_t) * [x_cam, 1]^T
```

초기 isotropic Gaussian radius는 대략 다음과 같다.

```text
r = clamp(z * footprint_pixels / focal, min_scale, max_scale)
log_scale = [log r, log r, log r]
```

현재 artifact 기본값은 `footprint_pixels=2`, `min_scale=1e-4`,
`max_scale=0.1`이다.

---

## 13. Stage 7 — Viewer runtime coverage rejection과 materialization

Offline artifact의 proposal을 모두 즉시 sidecar에 넣지 않는다. `frame_global == t`인
proposal만 현재 frame에 공개하고, 이미 학습된 Gaussian support로 중복을 제거한다.

기존 seed `j`가 새 proposal을 막을 수 있는 조건은 다음과 같다.

```text
eligible_j(t)
  = OPEN_j(t) OR NEVER_OPEN_j(t)

mature_j
  = eligible_j(t) AND geometry_update_count_j >= 4

radius_j
  = 2 * max(scale_j,x, scale_j,y, scale_j,z)
```

새 proposal `x`가 어떤 mature row의 support 안에 있으면 reject한다.

```text
covered(x) = exists mature j: ||x - xyz_j|| <= radius_j
```

현재 viewer variant에서는 `NEVER_OPEN` geometry가 frozen이므로 그 row의 geometry
update count는 0에 머문다. 따라서 현재 설정에서 실제로 mature coverage를 만드는
것은 최소 4회 geometry update를 받은 OPEN row다.

Accepted proposal은 다음 두 객체에 동시에 append된다.

1. Learnable DA3 representation sidecar
2. Fixed birth-geometry seed detector probe

초기 상태는 다음과 같다.

```text
materialized timestamp = t
lifespan               = NEVER_OPEN
stable detector prior  = Beta(flip=1, keep=10)
opacity initialization = 0.10
```

같은 frame에서 proposal을 만들고 바로 BF evidence로 검증하지 않는다.

```text
seed detector eligible at t
  <=> birth_timestamp < t
```

즉 birth frame은 proposal-only이고 detector evidence는 다음 신규 frame부터 받는다.

---

## 14. Stage 8 — DA3 seed용 별도 BF30 detector

Seed detector는 base detector와 동일한 post-sigmoid soft `Q_t`를 사용한다.
`view.candidate_map=2Q_t`를 base와 같은 `cue_mode=soft`, `cue_scale=2`로
정규화하므로 detector에 들어가는 evidence target은 `Q_t`이며 `2Q_t`가 아니다.

| 항목 | Base detector | DA3 seed detector |
|---|---|---|
| Geometry | immutable reference | fixed seed birth geometry |
| Cue | learned sigmoid soft `Q_t` | 동일한 learned sigmoid soft `Q_t` |
| 첫 evidence | 모든 base row | birth 다음 신규 frame부터 |
| Learned DC/geometry feedback | 없음 | 없음 |
| BF rule | flip/keep BF30 | 같은 flip/keep BF30 |

기본 `--da3-detector-cue-source shared`는 base의 cue tensor, mode, threshold,
scale을 그대로 공유한다. 기존 Part19/304-frame DC ablation 재현에만 명시적으로
`--da3-detector-cue-source part19_binary`를 사용한다. 기존 DC ablation evaluator는
비교 조건을 바꾸지 않도록 이 legacy 옵션을 지정한다.

Seed detector probe는 학습된 sidecar xyz/scale/opacity를 사용하지 않는다. 따라서
representation geometry가 움직여도 이미 처리한 detector evidence나 다음 seed BF
responsibility가 그 학습 결과로 오염되지 않는다.

Seed가 BF30으로 OPEN/CLOSE되면 representation sidecar lifespan과 별도 interval history를
동시에 갱신한다.

---

## 15. Stage 9 — Causal replay bank와 view sampling

Detector와 seed birth가 끝난 뒤 현재 frame을 replay bank에 추가한다.

```text
ReplayFrame_t = {
  timestamp=t,
  camera/view=I_t,
  cue_target=Q_t,
  signed_NEW_target=Q_t * NEW_support_t
}
```

현재 active DA3 geometry loss는 signed NEW target이 아니라 전체 unsigned `Q_t`를
사용한다. `signed_NEW_target`은 현재 실행 variant의 base-frozen/DA3 path에서는 직접
쓰이지 않는다.

Frame마다 120회 representation update를 수행한다. Update `u`의 sampled timestamp를
`k`라고 하면:

```text
with probability 0.33:
  k = t                       # explicit latest branch

with probability 0.67:
  k ~ UniformInteger(0, t)    # latest가 다시 뽑힐 수도 있음
```

따라서 실제 latest 선택 확률은 다음과 같다.

```text
P(k=t) = 0.33 + 0.67 / (t + 1)
```

항상 `0 <= k <= t`이며 미래 frame은 replay bank에 존재하지 않는다.

---

## 16. Stage 10 — Historical lifespan population 복원

Sampled frame `k`마다 base와 seed lifespan을 `k`에서 다시 query한다.

| `k`에서의 row 상태 | DC/geometry replay render | optimizer |
|---|---|---|
| Base OPEN | learned DC, frozen reference geometry | sampled view에 visible한 DC만 |
| Base NEVER_OPEN | exact black DC, reference opacity/geometry occluder | 없음 |
| Base CLOSED | 제외 | 없음 |
| Seed OPEN | learned DC/xyz/opacity/scale/rotation | visible DC와 geometry |
| Seed NEVER_OPEN | black DC, birth geometry/opacity occluder | 현재 variant에서는 없음 |
| Seed CLOSED | 제외 | 없음 |
| Seed born after `k` | 존재하지 않음 | 없음 |

이것이 2026-09-06에 수정한 historical lifespan replay의 핵심이다.

```text
잘못된 과거 구현:
  camera/cue = k
  population = latest t

현재 구현:
  camera/cue = k
  population = k
```

과거 SC1/SC2 view를 SC3 시점에 replay하더라도 그 과거 timestamp에 살아 있던
Gaussian만 렌더된다. 다만 parameter 값 자체는 과거 snapshot이 아니라 여러 view가
공유하는 현재 persistent parameter다.

---

## 17. Stage 11 — Joint base + DA3 DC optimization

### 17.1 Joint render population

Sampled timestamp `k`에서 다음 population을 한 번 합쳐 렌더한다.

```text
R_joint,k = Render(
    base OPEN learned DC
  + base NEVER_OPEN black occluder
  + seed OPEN learned DC/geometry/opacity
  + seed NEVER_OPEN black occluder
)
```

`k`에서 CLOSED이거나 future-born인 row는 들어가지 않는다.

### 17.2 SSF change probability

학습 loss 안의 change probability는 raw rendered RGB channel mean에 sigmoid를
적용한다.

```text
P_k(p) = sigmoid(mean_c R_joint,k,c(p))
```

### 17.3 현재 DC target `2Q_k`

현재 viewer는 normalized cue `Q_k`에 amplitude 2를 곱하되 clamp하지 않는다.

```text
T_DC,k(p) = 2Q_k(p)
```

원본 O-SCD SSF loss는 다음과 같다.

```text
L_detect
  = mean_p [2Q_k(p) * (1 - P_k(p))]

L_sparse
  = log(mean_p[P_k(p)]^2 + 1)

L_DC
  = L_detect + L_sparse
```

이 objective에는 명시적인 pixelwise negative BCE 항
`(1-Q_k) * P_k`가 없다. Cue 밖의 false positive는 global sparsity term으로만
간접 억제한다.

`2Q`는 detection term의 positive cue gradient를 `Q`보다 두 배로 만들지만 sparsity
term은 그대로 둔다.

### 17.4 DC optimizer selection

현재 `seed_dc_supervision=joint`다.

```text
base DC update row
  = OPEN at k AND raster radius > 0 in camera k

seed DC update row
  = OPEN at k AND raster radius > 0 in camera k
    AND actual DC gradient != 0
```

Base와 seed OPEN DC는 같은 `L_DC` gradient를 받는다. Seed-only SSF, projected BCE,
NEW/REMOVE sign-separated DC loss는 현재 variant에서 사용하지 않는다.

---

## 18. Stage 12 — 현재 DA3 OPEN geometry optimization

현재 variant의 base geometry는 완전히 frozen이다. Geometry loss는 `k`에서 OPEN인
DA3 seed에만 적용한다.

### 18.1 Coverage render

OPEN DA3 row를 모두 white override color로 렌더하고 RGB mean을 clamp한다.

```text
A_k(p) = clamp(mean_c CoverageRender_k,c(p), 0, 1)
```

### 18.2 현재 geometry target `2Q_k`

```text
G_k(p) = 2Q_k(p)               # unclipped
```

현재 coverage loss 구현은 다음과 같다.

```text
Z_pos = max(sum_p G_k(p), 1)
Z_neg = max(sum_p [1 - G_k(p)], 1)

L_inside
  = sum_p G_k(p) * [1 - A_k(p)] / Z_pos

L_outside
  = sum_p [1 - G_k(p)] * A_k(p) / Z_neg

L_geometry
  = 1.0 * L_inside + 0.1 * L_outside
```

### 18.3 `2Q` geometry target의 정확한 해석

이 loss는 원래 `0..1` target을 가정했지만 현재 실험은 요청대로 unclipped `2Q`를
전달한다. 따라서 단순히 geometry gradient 전체가 두 배가 되는 것이 아니다.

```text
Q_k(p) > 0.5
  -> 1 - 2Q_k(p) < 0
```

즉 high-Q pixel은 `L_outside`에서 음의 weight를 갖는다. 또한 `L_inside`는 분자와
정규화 분모가 함께 두 배가 되므로 `sum G > 1`인 일반적인 경우 단순 amplitude
증폭이 대부분 상쇄된다. 현재 코드는 이 값을 clamp하지 않는다.

따라서 현재 조건은 다음과 같이 기록해야 한다.

> “확률 target을 두 배로 강화한 표준 coverage loss”가 아니라,
> **기존 normalized coverage 식에 unclipped `G=2Q`를 넣은 viewer-only 실험**이다.

### 18.4 Trainable geometry row

```text
DA3 OPEN and visible at k:
  xyz, opacity, scale, rotation 학습

DA3 NEVER_OPEN at k:
  현재 variant에서는 xyz, opacity, scale, rotation 모두 frozen

Base rows:
  geometry/opacity 모두 frozen
```

DC와 geometry는 같은 sampled timestamp `k`를 사용하지만 별도 render/loss로 계산한다.

---

## 19. Stage 13 — Masked Adam과 geometry trust region

Optimizer는 row mask를 받아 선택된 parameter row만 갱신한다.

```text
k에서 CLOSED
k 이후에 태어남
k에서 off-view
현재 scope에서 frozen
```

위 row의 parameter와 Adam moment는 해당 update에서 보존된다.

현재 DA3 learning rate는 다음과 같다.

| parameter | learning rate |
|---|---:|
| xyz | 0.00016 |
| DC | 0.0025 |
| opacity | 0.025 |
| scale | 0.005 |
| rotation | 0.001 |

Geometry optimizer step 뒤 birth anchor 기준 trust region을 적용한다.

```text
||xyz - xyz_birth|| <= 4 * max(scale_birth)

0.25 * max(scale_birth)
  <= each learned scale axis
  <= 4.0 * max(scale_birth)

0.01 <= opacity <= 0.99

rotation quaternion -> unit norm
```

Constraint에 걸린 row는 해당 parameter의 Adam state도 reset한다.

---

## 20. Stage 14 — 현재 시점 output render

120 update가 끝나면 sampled `k`가 아니라 현재 timestamp `t`의 lifespan을 query한다.

```text
R_t = RenderCurrent(
    base OPEN learned DC
  + base NEVER_OPEN black occluder
  + seed OPEN learned DC/geometry/opacity
  + seed NEVER_OPEN black occluder
)
```

Base/seed CLOSED는 제외한다.

Raw output score와 binary mask는 다음과 같다.

```text
S_t(p)    = clamp(mean_c R_t,c(p), 0, 1)
Mask_t(p) = 1[S_t(p) >= 0.5]
```

중요하게도 학습 loss는 `sigmoid(mean RGB)`를 사용하지만 final score는 sigmoid 없이
`clamp(mean RGB)`를 사용한다.

```text
training probability = sigmoid(raw mean RGB)
output score         = clamp(raw mean RGB, 0, 1)
```

이는 현재 코드의 정확한 계약이며 두 값을 같은 것으로 해석하면 안 된다.

---

## 21. `Next cue ->` 한 번의 정확한 실행 순서

Viewer의 `step()` 순서는 다음과 같다.

1. 다음 causal frame `I_t`와 cached `P+S`를 읽는다.
2. Immutable reference RGB로 `P_0,t`를 재계산하고 `S_t`를 복원한다.
3. L1-only power product와 causal learned sigmoid로 `Q_t`를 만든다.
4. **Representation optimization 전에** immutable base probe로 soft `Q_t` alpha-T
   evidence를 정확히 한 번 계산한다.
5. Base row별 flip/keep BF30 filter를 갱신한다.
6. Commit된 row만 base lifespan에서 OPEN/CLOSE한다.
7. `birth < t`인 기존 DA3 seed만 fixed birth-geometry probe와 동일한 soft `Q_t`로
   한 번 관측한다.
8. Commit된 seed만 seed lifespan에서 OPEN/CLOSE한다.
9. Offline artifact 중 `birth == t`인 DA3 proposal을 공개한다.
10. Mature learned-Gaussian coverage 안의 proposal을 reject한다.
11. Accepted proposal을 `NEVER_OPEN`으로 representation과 detector probe에
    materialize한다. 현재 birth frame은 BF evidence로 재사용하지 않는다.
12. `{I_t, Q_t, timestamp=t}`를 causal replay bank에 추가한다.
13. 120번 반복하여 sampled `k <= t`를 뽑는다.
14. 각 update에서 `k`의 base/seed lifespan population을 복원한다.
15. `2Q_k` joint SSF로 visible OPEN base/seed DC를 학습한다.
16. 별도 `2Q_k` coverage loss로 visible OPEN DA3 geometry를 학습한다.
17. Masked Adam step과 trust-region projection을 수행한다.
18. `t`에서 current-valid한 population만 다시 렌더한다.
19. Raw score `S_t`와 threshold `0.5` mask를 만든다.
20. 마지막으로 evaluation-only GT를 표시한다.

Detector는 4--8단계에서만 신규 observation을 소비한다. 13--17단계의 historical
replay나 18단계의 post-optimization render는 detector에 재입력되지 않는다.

---

## 22. Viewer 1--10 패널의 의미

### 22.1 Panel 1 — Main Gaussian layer

Dropdown으로 네 진단 layer를 바꾼다.

1. Committed lifecycle
   - black: `NEVER_OPEN`
   - green: `OPEN`
   - red: `CLOSED`
2. Current cue projected to Gaussians
   - 현재 frame의 base reference row positive pseudo-count `delta(+)`
3. Accumulated Bayesian instability
   - base detector의 live candidate에만 `clamp(logBF/log30,0,1)`
4. Learned current `R_change`
   - Panel 5와 같은 current-valid prediction

Candidate는 lifecycle 색을 덮어쓰지 않는다. Candidate 진행도는 BF layer/Panel 4에서
본다.

### 22.2 Panel 2 — Online RGB + accepted DA3 centers

- 초록: 이전 frame까지 accepted된 seed center
- 흰 테두리 빨강: 현재 frame에 accepted된 seed center
- 현재 학습된 xyz를 camera에 투영
- 미래 birth는 표시하지 않음

### 22.3 Panel 3 — Learned sigmoid cue

`Q_t`를 black -> green -> yellow -> red heatmap으로 표시한다.

### 22.4 Panel 4 — Bayes factor

Base detector의 live candidate만 다음 진행도를 표시한다. DA3 seed detector의 BF는
이 image panel에 합성하지 않고 sidebar의 seed lifecycle count로 확인한다.

```text
progress_i = clamp(logBF_i / log30, 0, 1)
```

Candidate가 없으면 black이다.

### 22.5 Panel 5 — Current-valid learned `R_change`

현재 `t`의 base + DA3 population을 합친 연속 raw score heatmap이다. Threshold하지 않은
Panel 6의 원본이다.

### 22.6 Panel 6 — Predicted change mask

```text
Panel6 = 1[Panel5 raw score >= 0.5]
```

### 22.7 Panel 7 — GT change

`GT ADD union GT REMOVE`다. 평가·관찰 전용이다.

### 22.8 Panel 8 — Signed SAM difference times `Q_t`

```text
H_t = normalized upsampled signed SAM/PCA score * Q_t
```

- red: positive sign
- blue: negative sign
- black: zero 또는 표시 quantization 아래

### 22.9 Panel 9 — Scale-aligned GS minus DA3 depth

```text
d_t = D_ref,t - s_t * D_DA3,t
```

`|Panel8| > 0.1 AND |d_t| > 0.03`인 영역만 표시한다.

- red: online surface가 reference보다 앞
- blue: online surface가 reference보다 뒤
- intensity: valid residual 절댓값의 frame별 q95로 정규화

실제 seed는 이 양·음 시각화 전체가 아니라 `Panel8 > +0.1 AND d_t > +0.03`인 양수
교집합에만 생성된다.

### 22.10 Panel 10 — Learned-Q NEW / REMOVE / APPEARANCE 구분

Panel 3의 soft `Q_t`를 Panel 8의 signed SAM×Q와 Panel 9의 scale-aligned
GS−DA3 depth residual로 구분한다. **표시 전용 heuristic**이며 detector, birth,
DC/geometry loss, optimizer에는 다시 입력하지 않는다.

| 표시 | 조건 |
|---|---|
| 빨강 NEW | `SAM×Q > +0.1` AND valid `depth residual > +0.03` |
| 파랑 REMOVE | `SAM×Q < -0.1` AND valid `depth residual < -0.03` |
| 노랑 APPEARANCE / uncertain | 부호 충돌, 한쪽/양쪽 신호가 약함, threshold와 같음, depth/SAM 무효 |

신호 분류는 RGB heatmap의 uint8 색이나 q95 대비가 아니라 원래 float 신호로
수행한다. Depth 부호는 native depth grid에서 threshold한 뒤 nearest-neighbor로
cue grid에 옮겨 보간으로 새로운 부호가 생기지 않게 한다. Panel 9와 마찬가지로
reference-alpha와 strong-SAM support가 없는 depth는 확정 부호로 사용하지 않는다.

밝기는 세 종류 모두 `round(255×Q_t)`이며 별도의 `Q>=0.5` gate는 없다.
따라서 Panel 3의 약한 cue도 그대로 약하게 표시하고, 표시 밝기가 0이면 검정이다.
노랑은 appearance의 확정 판정이 아니라 **두 부호로 NEW/REMOVE를 확정하지 못한
잔여 cue**다. Sidebar에 색상별 pixel 수를 표시하며 capture 시
`<frame>_cue_types.png`와 전체 dashboard에 함께 저장한다.

Panel 11--14는 현재 reserved black panel이다.

---

## 23. 현재 브라우저에서 실행 중인 정확한 variant

현재 process는 다음 command와 같다.

```bash
./run_bayesian_detector_viewer.sh \
  --port 8090 \
  --no-train-never-open-geometry \
  --geometry-cue-amplitude 2
```

Launcher가 앞에서 `--train-never-open-geometry`를 넣지만 `"$@"`가 마지막에 오고
`BooleanOptionalAction`을 사용하므로 뒤의 negative override가 최종적으로 이긴다.

최종 effective 설정은 다음과 같다.

| 축 | 현재 값 |
|---|---|
| cue fusion | `l1_power_product`, L1 exponent `0.3` |
| cue remap | causal artifact의 `learned_sigmoid` |
| base detector cue | soft `Q_t` |
| seed detector cue | 동일한 soft `Q_t` (`da3-detector-cue-source=shared`) |
| BF prior/threshold | stable `Beta(1,10)`, reset `Beta(1,1)`, BF `30` |
| updates/frame | `120` |
| replay | `sampled`, historical lifespan at sampled `k` |
| explicit latest branch | `0.33` |
| DC target | unclipped `2Q_k` |
| geometry target | unclipped `2Q_k` |
| base geometry | frozen |
| base OPEN DC | trainable when visible at `k` |
| DA3 OPEN DC | joint SSF에서 trainable |
| DA3 OPEN geometry | xyz/opacity/scale/rotation trainable |
| DA3 NEVER_OPEN geometry | frozen |
| CLOSED rows | render와 optimizer에서 제외 |
| output threshold | raw current score `>= 0.5` |

Viewer URL:

```text
http://localhost:8090
```

---

## 24. 현재 결과를 해석할 때의 경계

### 24.1 Full 304-frame 결과가 있는 조건

Historical lifespan replay 수정 후 측정된 full 결과는 다음 조건이었다.

```text
DC target                  = 2Q
geometry target            = Q
DA3 NEVER_OPEN geometry    = trainable
base geometry              = frozen
seed detector              = 수정 전 cached P+S > 0.5 binary
```

그 결과는 다음과 같다.

| metric | 값 |
|---|---:|
| mean-frame mIoU | 0.5249 |
| mean-frame F1 | 0.6588 |
| precision | 0.6401 |
| recall | 0.8567 |
| future-view accesses | 0 |
| lifespan/future-row violations | 0 |

결과 artifact:

```text
outputs/bayesian_da3_historical_replay_20260906/
  B_h1_amplitude2/summary.json
```

### 24.2 현재 viewer-only variant의 상태

지금 보고 있는 조건은 위 full run과 다르다.

```text
현재 viewer:
  DC target               = 2Q
  geometry target         = 2Q
  DA3 NEVER_OPEN geometry = frozen
  seed detector           = base와 동일한 learned soft Q
```

Seed detector를 soft Q로 통일하기 전의 `2Q geometry + frozen NEVER_OPEN`
조건은 14-frame CUDA smoke에서 다음 invariant를 통과했다. 이 smoke의 seed
detector는 cached `P+S > 0.5` binary였다.

```text
frames                         = 14
active seed                    = 56
NEVER_OPEN geometry updates    = 0
total geometry updates         = 99
lifespan render violation      = 0
```

그러나 **현재 `shared soft-Q seed detector + 2Q geometry + frozen NEVER_OPEN`
variant의 CUDA smoke와 304-frame mIoU/F1은 아직 측정하지 않았다.** 따라서 위
smoke count나 `0.5249/0.6588`을 현재 viewer variant의 측정값으로 인용하면 안 된다.

---

## 25. 현재 구현이 보장하는 것과 아직 보장하지 않는 것

### 25.1 현재 보장하는 것

- 신규 detector observation은 pre-optimization raw cue를 한 번만 사용한다.
- Base detector는 immutable reference geometry/opacity만 사용한다.
- DA3 seed detector는 fixed birth geometry만 사용한다.
- Future view와 future-born row는 historical replay에 들어가지 않는다.
- Replay frame `k`에는 `k`의 lifespan population을 사용한다.
- CLOSED row는 current output에서 사라진다.
- CLOSE 뒤 REOPEN interval을 다시 만들 수 있다.
- 하나의 row parameter를 여러 causal view가 공유하여 multiview optimization한다.
- GT는 inference/training path에 들어가지 않는다.

이 보장들은 궁극적 목표를 위한 기반 메커니즘의 correctness에 해당한다. 그 자체로
one-step baseline 대비 우수성이나 typed relational history의 완성을 의미하지 않는다.

### 25.2 현재 보장하지 않는 것

- `active-A -> active-B` 의미 변화의 별도 lifespan 분리
- CLOSE/REOPEN interval마다 독립 parameter snapshot
- NEW/REMOVED/MOVED/REVERTED 등 명시적인 change type 추론
- 변화 node 사이 successor/revert/replacement/correspondence relation graph
- Full checkpoint history 대비 compactness와 history-query fidelity의 정량 검증
- 동일 causal protocol의 one-step change detection baseline 대비 최종 우수성
- Long-term SCD 분야의 `first` novelty claim에 대한 체계적인 문헌 검증
- 현재 viewer-only `2Q geometry` variant의 full-sequence 성능 개선
- Live camera/online deployment에서 boundary network와 DA3 proposal을 모두 즉석 생성하는
  완전한 production pipeline
- Geometry `2Q`가 확률적으로 정규화된 표준 coverage target이라는 보장
- Pixelwise explicit negative BCE가 있는 DC objective
- Training sigmoid score와 final raw threshold score의 동일성

---

## 26. 코드와 artifact 대응표

| 역할 | 파일 |
|---|---|
| Viewer orchestration, step 순서, replay, output | `experiments/view_bayesian_detector_steps.py` |
| 실행 preset | `run_bayesian_detector_viewer.sh` |
| Pixel/SAM cue fusion, sigmoid remap | `temporal/change_cue_fusion.py` |
| Immutable alpha-T evidence와 capped count | `temporal/change_evidence.py` |
| Single-candidate Beta block BF | `temporal/single_candidate_beta.py` |
| Flip/keep 의미와 bit toggle | `temporal/lifespan_gate_beta.py` |
| Masked row Adam | `temporal/masked_optimizer.py` |
| DA3 seed 생성·depth scale·coverage rejection | `temporal/depth_prior_new_seeding.py` |
| DA3 active geometry view | `temporal/active_new_gaussians.py` |
| Coverage render/loss | `temporal/active_new_density.py` |
| Base + seed lifespan-aware concatenated render | `temporal/new_seed_gaussians.py` |
| Causal boundary artifact | `outputs/stage2_l1_power_sigmoid_boundary_causal_prequential/learned_boundaries_causal.json` |
| Causal DA3 proposal artifact | `outputs/causal_da3metric_scene123_panel7pos010_depthpos003_dynamiccoverage_20260904/da3_seed_replay.pt` |

관련 checkpoint 문서:

- `docs/bayesian-detector-step-viewer-ko.md`
- `docs/bayesian-da3-historical-lifespan-replay-20260906-ko.md`
- `docs/bayesian-da3-historical-lifespan-replay-20260906.json`
- `docs/part17-da3-depth-prior-new-seeding-ko.md`
- `docs/part19-da3-never-open-rchange-occupancy-ko.md`

---

## 27. 핵심 해석 요약

O-SCD-evolving의 궁극적인 연구 주장은 다음 세 가지다.

```text
1. continuously evolving scene을 위한 long-term change detection
2. 동일 조건의 one-step change detection보다 우수한 성능
3. change type과 relation을 보존하는 compact 3D history
```

현재 lifespan/Bayesian/DA3 viewer는 이 주장의 최종 산출물 자체가 아니라, 먼저
causal current-state validity와 long-term 3D memory를 성립시키기 위한 구현 단계다.

현재 O-SCD-evolving viewer는 다음 세 문제를 분리해 푼다.

```text
어디가 reference와 다른가?
  -> Q와 immutable alpha-T evidence

그 차이가 지금도 유효한가?
  -> flip/keep BF30과 lifespan interval

여러 view에서 그 변화를 어떻게 3D로 표현할 것인가?
  -> timestamp-aligned historical replay와 shared Gaussian parameter
```

현재 viewer가 기존 persistent O-SCD와 다른 핵심은 replay를 없앤 것이 아니라,
**replay timestamp에 맞는 Gaussian population을 복원한 것**이다.

```text
SC3 시점에 SC1 image를 replay
  -> SC1 timestamp에 OPEN/NEVER_OPEN이었던 row만 렌더
  -> SC1 뒤에 태어난 seed와 SC1 당시 CLOSED였던 row는 제외
  -> 선택된 row의 shared parameter를 SC1 view에도 맞도록 학습
```

이 방식으로 multi-view optimization은 유지하면서, 최신 lifespan과 과거 cue를 잘못
섞는 timestamp mismatch를 막는다. 최종 출력은 다시 현재 `t`의 lifespan으로 렌더하므로
과거 변화의 합집합이 아니라 current-valid `R_change`를 목표로 한다.

다음 연구 단계에서는 이 interval history 위에 change type과 relation을 명시적으로
올리고, 저장 비용과 history query 정확도를 측정하며, 동일 causal 조건의 one-step
baseline보다 우수한지를 입증해야 궁극적 목표가 완성된다.

## 28. Panel10 NEW / 나머지 영역의 분리 학습 (2026-09-06 후속)

### 28.1 범위와 target

기본 launcher는 `--training-partition panel10_new`를 사용한다. 이전 실험은
`--training-partition legacy`로 남기되, 새 경로에서는 흰색 coverage 렌더와
`(1-2Q)` outside loss를 사용하지 않는다. Base geometry는 기존처럼 고정하고
base DC만 학습한다. NEW seed는 DC/xyz/opacity/scaling/rotation을 함께 학습한다.

Panel10의 **RGB 이미지가 아니라 동일한 raw sign mask**를 공유한다.

```text
M_NEW = (SAM×Q > 0.1) AND (GS_depth − aligned_DA3_depth > 0.03)
        AND valid_depth_support
Q_NEW  = M_NEW × Q
Q_BASE = Q − Q_NEW
Q_NEW + Q_BASE = Q
```

REMOVE(−/−), APPEARANCE/불확실(부호 충돌·약한 신호·무효 depth)은 모두 base
경로로 간다. Q≥0.5 같은 추가 hard cutoff는 없다. NEW 분류는 휴리스틱이며
의미적 GT나 보정된 NEW posterior가 아니다. 각 frame의 target은 optimization
전에 한 번 만들어 replay item에 저장한다. 학습된 GS가 target을 재분류하지 않는다.

### 28.2 독립 loss, 공동 최종 render

각 branch는 자기 OPEN row와 **두 bank의 모든 NEVER_OPEN**을 함께 렌더한다.
다른 bank의 OPEN은 branch 학습 렌더에 포함하지 않는다.

```text
R_NEW  = render(seed_OPEN + all_NEVER_OPEN_black)
R_BASE = render(base_OPEN + all_NEVER_OPEN_black)
P_j    = sigmoid(mean_RGB(R_j))

L_NEW  = mean[2Q_NEW (1 − P_NEW)]
         + log(1 + mean(P_NEW)²)
L_BASE = mean[2Q_BASE (1 − P_BASE)]
         + log(1 + mean(P_BASE)²)
L      = L_NEW + L_BASE
```

`2`는 기존 `--representation-cue-amplitude` 기본 launcher 값이다. 이는 전체
이미지에서 평가하는 두 SSF objective이며, target 밖에서는 detection 항이 0이고
기존 global sparsity 항은 남는다. **영역 바깥을 loss 계산에서 전부 무시하는
masked average가 아니다.** 두 branch의 global sparsity도 독립이므로, 기존 joint
loss를 단순히 수학적으로 둘로 분해한 것과 같지는 않다.

- `L_NEW`만 seed DC와 geometry를 갱신한다. 같은 loss의 screen-space gradient를
  seed densification에 사용한다. White coverage loss / projected BCE는 없다.
- `L_BASE`만 base DC를 갱신한다. Immutable reference는 변경하지 않는다.
- 두 경로 모두 현재 OPEN이면서 sampled timestamp에도 OPEN이고 현재 학습 뷰에서
  radius>0인 row만 optimizer가 선택한다. 현재 CLOSED row는 historical render에
  필요하면 frozen 값으로 등장하지만 학습되지는 않는다.
- NEVER_OPEN의 change color는 정확한 RGB black이고 geometry/opacity는 frozen이다.
  과거 NEVER_OPEN seed는 birth probe 속성으로 렌더하여 이후 learned geometry를
  과거의 고정 occluder로 잘못 사용하지 않는다.
- 최종 mask는 `render(base_OPEN + seed_OPEN + all_NEVER_OPEN_black)`의 **한 번의
  alpha compositing**으로 만든다. 두 이미지의 합집합/합산이 아니다. 기존 raw-score
  0.5 threshold는 유지한다.

### 28.3 현재 NEW 영역에서 DA3 birth

기존 checkpoint의 accepted xyz 목록을 재생하지 않는다. Checkpoint는 DA3Metric
model/cache 설정을 얻는 데만 사용하며, frame마다 Panel9와 같은 aligned current
DA3 depth를 NEW pixel에서 unproject한다. DA3 intrinsics는 cue 해상도에 맞춰
조정한다. Signed SAM/reference-depth 분석은 immutable reference만 사용한다.

- frame당 최대 1,024 proposals, 4-pixel cell당 최대 한 점을 Q 순으로 선택한다.
- mature OPEN seed와 고정 NEVER_OPEN seed의 Gaussian support로 중복을 억제한다.
  학습하지 않는 pending seed도 중복 birth를 막는다.
- generation-zero seed는 NEVER_OPEN으로 생성한 뒤 **같은 frame의 공동 detector**에
  포함한다. Birth 자체가 OPEN을 강제하지 않으며, 해당 frame Q를 BF30에 한 번만 넣는다.
- Detector는 계속 shared pre-optimization raw soft Q 전체를 사용한다. NEW/non-NEW
  target 분리는 representation용이며 detector를 semantic NEW classifier로 바꾸지 않는다.

### 28.4 Seed-only densification과 pruning

- local update 4에서 현재 OPEN/visible seed의 NEW-loss gradient로 clone/split한다.
- 기본 threshold는 signed gradient `2e-4`, absolute gradient `1.2e-3`이다.
  Root 초기 scale 대비 작으면 clone, 크면 2-child split을 선택한다.
- event당 최대 128 children, 최대 generation 2, root당 최대 32 descendants,
  parent당 한 번의 density event, 전체 archive 20,000 rows로 growth를 제한한다.
- Child는 parent BF buffer를 한 번 복사하고 이후 신규 frame에서 독립적으로 갱신한다.
  과거 parent interval을 복사하지 않으며, child의 렌더 lifespan은 생성 시점부터다.
  Child를 생성한 frame의 cue를 detector에 재입력하지 않는다.
- Split parent는 이전 frame부터 OPEN이었던 경우에만 종료한다. 방금 OPEN된 row나
  same-event child를 바로 닫아서 0-length interval을 만들지 않는다.
- Learned opacity≤0.02, OPEN age≥3 frames, geometry updates≥4인 현재 OPEN/visible
  seed를 pruning한다. NEVER_OPEN/CLOSED/off-view/fresh row는 보호한다.
- Pruning과 split replacement는 **lifespan을 닫는 영구 retirement**다. Retired row는
  다시 OPEN하지 않고, parameter/Adam state는 frozen으로 보존한다. Historical replay
  보존을 위해 행을 물리적으로 삭제하지 않으므로 **GPU memory compaction은 아니다.**
  전체 row budget은 archive까지 센다. Detector CLOSE와 density retirement는 따로 기록한다.

CLI의 `--da3-birth-*`, `--da3-density-*`, `--da3-max-*`, `--da3-prune-*`로 명시적인
ablation을 만들 수 있다. Historical shared-parameter replay 자체는 유지되며 완전한
과거 parameter snapshot을 저장하는 방법으로 바뀐 것은 아니다.

### 28.5 검증 범위

별도 tests에서 target 분리/NEW-only unprojection, branch 간 gradient 격리,
NEVER_OPEN/CLOSED/Adam 동결, 과거 birth visibility, child tracker 독립 복사,
root/event/archive cap과 retirement를 검증한다. Viewer capture에는 매 frame
`*_summary.json`으로 base/NEW loss와 seed birth/density/prune 수를 기록한다.
CUDA smoke와 시각 검증 결과는 해당 run의 outputs에 별도로 보존한다.
이 변경만으로 mIoU/F1 개선을 주장하지 않는다.


### 28.6 Birth-first 단일 detector (DC 수정과 함께 적용)

현재 `panel10_new`의 정확한 순서는 다음과 같다.

```text
현재 RGB / raw learned-sigmoid Q 계산
  → signed SAM + aligned DA3 depth로 Q_NEW, Q_BASE 확정
  → 현재 NEW 위치에 DA3 root seed를 NEVER_OPEN으로 추가
  → immutable base + 고정 birth seed probe를 함께 렌더
  → 전체 Q의 alpha-T evidence를 한 번 계산
  → 하나의 BF30 tracker에서 base와 seed row를 동일 규칙으로 갱신
  → OPEN/CLOSE commit을 각 representation lifespan에 반영
  → 분리 SSF 학습, seed density/prune
  → learned base+seed 공동 output render
```

- Tracker는 `[base prefix | seed archive suffix]` 하나이며, current variant에서
  `seed_tracker`는 만들지 않는다. Chunk는 계산량 제한일 뿐 detector 분리가 아니다.
- Base도 seed와 같은 transmittance를 공유하므로, 앞쪽 seed의 alpha가 배경 base의
  evidence mass에 영향을 준다. NEW/non-NEW 분류는 detector Q를 나누지 않는다.
- Seed root는 생성 frame부터 관측 가능하다. 생성에 사용한 Q를 첫 evidence로 쓰는
  요청된 변경이며, 독립적인 다음-view confirmation과는 다르다. BF30은 유지한다.
- Detector probe geometry/opacity는 여전히 reference/base와 각 seed 생성 시 값이다.
  Learned DC/seed geometry/opacity나 post-optimization/replay render가 detector로
  역류하지 않는다. 따라서 learned seed가 커진 만큼 detector occlusion이 커지는 것은 아니다.
- CLOSED probe는 REOPEN 판단을 위해 유지한다. Prune/split으로 **영구 retired**된
  probe만 공동 detector render와 update에서 제거하며, suffix row identity는 보존한다.
- Optimization 중 태어난 density child는 부모 posterior를 한 번 상속한다. 같은 frame의
  raw Q를 재입력하지 않으며 다음 신규 observation에서 처음 독립 update한다.
- State/loss 분리가 사라진 것이 아니라 **evidence와 BF 추론이 통합된 것**이다.
  NEW seed loss, 나머지 base loss, 각 lifecycle history/optimizer ownership은 유지한다.
- `--training-partition legacy`는 과거 실험 재현을 위해 별도 detector와 다음-frame
  seed 관측 순서를 그대로 보존한다. 최신 launcher는 `panel10_new`다.

공동 alpha-T가 배경의 변화 evidence를 모두 제거한다고 보장하지 않는다. 투명도,
footprint, cue 경계 오차와 capped evidence normalization 때문에 배경 OPEN은 남을 수 있다.

### 28.7 DC adapter 버그 수정과 검증 경계

분리-render adapter가 `SH-rest=(N,0,3)`을 넘기면 FastGS CUDA backward의
`if (shs)`가 false가 되어 DC gradient도 건너뛰었다. Geometry/opacity gradient와
CPU mock-render 테스트가 통과해도 실제 base/seed DC는 초기값에 머물 수 있었다.

현재 adapter는 degree-0 forward에 쓰이지 않는 zero dummy SH-rest `(N,1,3)`을
보유하여 DC backward가 실행되도록 한다. Public feature는 DC 한 coefficient이며,
DC parameterization, sigmoid SSF, target amplitude, final threshold는 변경하지 않는다.
CUDA 소스 수정이나 extension rebuild는 필요하지 않다.

- 수정 전 실제 CUDA `train_partition_update` regression에서 base DC gradient=0으로 실패.
- 수정 후 실제 rasterizer를 거친 **base와 seed DC gradient 및 optimizer 갱신**을 검증한다.
- 단일 detector 테스트는 birth→공동 evidence→학습 순서, row당 1회 update,
  chunk 경계, OPEN→CLOSE→REOPEN, retired mapping과 고정 probe 독립성을 확인한다.
- CUDA occlusion 테스트에서 현재-frame foreground seed가 공동 render의 base evidence
  mass를 실제 감소시키는지 확인한다. BF30의 의미적 정확도 보장은 아니다.
- 초기 분리학습 검증은 실제 CUDA DC 갱신을 놓쳤으므로, 그때의 테스트 통과나
  frame 50의 seed OPEN 수를 정상 학습의 증거로 사용하지 않는다.


#### 수정 후 실제 viewer 검증

`outputs/viewer_panel10_joint_detector_dc_fixed_20260906/`에서 canonical launcher와
동일 설정으로 timestamp 0..50, frame당 120 updates를 처음부터 재실행했다.
Frame 0에서 새 seed 180개를 생성하고 같은 frame에 180개 모두 detector evidence를
받았다. 모든 51 frame의 공동 evidence 호출은 정확히 1회였다.

Frame 50의 120-update 구간에서 base DC 21,454 rows, seed DC 3,368 rows가 실제
변경되었다. 최종 joint mask 양성은 **49,690 pixels**이며, 이전 adapter 오류 run의
0-pixel 상태와 달리 learned prediction이 출력된다. 이는 오류 복구 확인이지 두 변경의
효과를 분리한 mIoU 개선 실험이 아니다.

- Reference / frozen parameter / frozen Adam drift: 51 frame 모두 0.
- Future-view access / lifespan render violation: 모두 0.
- Targeted CPU+CUDA tests: **108 passed**, 실제 rasterizer DC forward parity와
  backward/optimizer update 포함. 별도 코드 리뷰 blocking issue 0.
- AST syntax, launcher `bash -n`, `git diff --check` 통과. 전용 lint/typechecker는
  설치되어 있지 않아 실행하지 않았다.
- 상세: `verification.json`, `test-report.json`, `captures/000050_summary.json`,
  `captures/000050_dashboard.png`. Viewer는 수정된 상태로 `http://localhost:8090`에서 실행한다.
- 전체 304-frame accuracy는 이 수정 검증 범위에 포함하지 않았다.

### 28.8 후속 전체304-frame 평가와 MP4

위51-frame smoke 이후 같은 현재 preset(seed0,u120)을 별도 프로세스에서 처음부터
전체304 frames 연속 실행했다. Mean-frame foreground IoU/F1은 **0.6116/0.7260**,
pixel 통합 IoU/F1은 **0.6594/0.7947**이었다. SC1/SC2/SC3 mean-frame IoU는
**0.5743/0.7063/0.5515**다. 각 scene 사이 state reset은 하지 않았다.

모든 saved mask를 GT와 다시 비교하여 confusion count/aggregate score가 일치함을
확인했고, viewer dashboard MP4는 **304frames,10fps,30.4초,1920×1120**로 완전
디코딩 검증했다. 원래 interactive viewer는 변경하지 않았다.

주의: t106(SC2 frame12)에서 20,000-row archive cap에 도달하여 **SC3의 신규
DA3 root/density child는0**이었다. 현재 측정값은 이 제한을 포함한다.

- [전체 평가 보고서와 비교 조건](panel10-joint-detector-full304-results-20260906-ko.md)
- [전체304-frame viewer MP4](../outputs/panel10_joint_full304_20260906/viewer_dashboard_all304.mp4)
- [Saved-mask 및 video 후처리 검증](../outputs/panel10_joint_full304_20260906/posthoc_verification.json)

### 28.9 후속 archive 무제한 비교 (2026-09-07)

`--da3-max-rows 0`으로 총 archive 상한만 해제하고 같은 u120/seed0/304frames를
재실행했다. 공동 BF tracker는 필요 시 suffix storage를 확장하며 기존 상태를 보존한다.
프레임당1,024 birth, coverage, density/root cap, BF30, loss는 변경하지 않았다.
CLI의 과거 default20,000은 유지하고 별도 평가 launcher에서0을 명시한다.

Mean-frame IoU/F1은 **0.6116/0.7260 → 0.6473/0.7535**,
SC3 IoU는 **0.5515 → 0.6406**이었다. 최종 archive69,033개이며,
Frame243의 텀블러 위 seed와 최종 mask가 다시 나타났다.
304장 saved mask 독립 재평가 및304-frame MP4 전체 디코딩 검증을 통과했다.

- [상세 결과·한계·Frame243 비교](panel10-joint-uncapped-full304-results-20260907-ko.md)
- [무제한 전체304-frame MP4](../outputs/panel10_joint_uncapped_full304_20260907/viewer_dashboard_all304.mp4)

### 28.10 PASLCD 전체 평가 (2026-09-07)

같은 무제한 Panel10+joint BF30/u120을 PASLCD20scenes/500frames에 적용했다.
Scene별로 새 causal SAM/DA3/Stage2 입력을 준비하고 상태를 초기화했다.
Mean-frame IoU/F1은 **0.4288/0.5576**, 동일 GT로 재평가한 원본 O-SCD online은
**0.4906/0.6440**이었다. ESCD에서의 개선이 PASLCD 전체에 일반화되지는 않았다.

초반5-frame mIoU는0.1497, 마지막5-frame은0.5378로 초기 누락이 크다.
모든 scene archive가20k미만이므로 이 데이터에서는 총량 상한 자체가 병목이 아니었다.
Saved prediction/두 baseline mask와500-frame MP4를 독립 검증했다.

- [PASLCD 상세 결과·조건·한계](paslcd-panel10-joint-uncapped-u120-results-20260907-ko.md)
- [전체500-frame MP4](../outputs/paslcd_panel10_uncapped_u120_20260907/viewer_dashboard_all_frames.mp4)

### 28.11 PASLCD 첫 OPEN BF / pixel-cue binary ablation (2026-09-07)

같은 20 scenes/500 frames/u120에서 첫 `NEVER_OPEN → OPEN`만 BF10으로 낮추고
CLOSE/REOPEN은 BF30을 유지했다. 별도로 detector 입력 pixel Q만 `1[Q>0.5]`로
이진화했다. Loss/Panel10/seed birth는 soft Q를 유지하고, 합산 후 Gaussian evidence는
이진화하지 않는다. Seed birth 뒤 공동 evidence 1회와 capped mass≤1 계약도 유지한다.

| 조건 | mean-frame mIoU | mean-frame F1 |
|---|---:|---:|
| BF30 + soft Q (기준) | 0.4288 | 0.5576 |
| 첫 OPEN BF10 + soft Q | 0.5019 | 0.6420 |
| BF30 + binary Q | 0.4358 | 0.5650 |
| 첫 OPEN BF10 + binary Q | 0.5048 | 0.6450 |

BF10 두 조건 모두 20/20 scene에서 기존보다 mIoU가 높았고, 모든 scene에서
두 번째 frame부터 nonempty mask를 출력했다. 기존은 첫 두 frame 모두 비어 있었다.
첫 5-frame mIoU는 0.1497→0.3145(BF10 soft)→0.3202(BF10 binary)다.
Binary의 추가 +0.29%p mIoU와 함께 REOPEN도 266→878로 늘어 안정성 개선으로
해석하지 않는다. 기본 설정은 유지하며 ESCD 연속 전환 재평가는 아직 하지 않았다.
161 tests, 새 1,500-frame mask와 세 MP4의 독립 검증을 통과했다.

- [상세 수식·전체 결과·영상·한계](paslcd-first-open-bf-pixel-binary-ablation-20260907-ko.md)

### 28.12 사용자 의도 정정: Gaussian 합산 후 binary (2026-09-07)

28.11의 pixel 이진화는 사용자의 의도와 달랐다. 수정 방식은 soft pixel Q를 먼저
Gaussian별 alpha-T로 합산한 뒤 `z=1[E⁺>E⁻]`로 판정한다. 이번 관측의 총량
`w=min((E⁺+E⁻)/saturation,1)`을 `(wz,w(1-z))`로 한쪽에만 입력한다.
과거 Beta 누적값을 이진화하는 것은 아니며 loss/Panel10/birth는 soft Q를 유지한다.

PASLCD20scenes/500frames/u120에서 Gaussian binary의 mean-frame mIoU/F1은
**BF30 0.4494/0.5817**, **첫 OPEN BF10 0.5114/0.6535**였다. 첫 OPEN BF10의
이전 pixel binary0.5048/0.6450보다 +0.66%p/+0.85%p 높았지만, soft 대비
REOPEN은266→2,915, base CLOSE는3,225→15,748로 늘었다. 평균 mask 점수 개선을
안정성 개선으로 해석하지 않는다. 새1,000frames의 mixed 입력 row는 모두0이었다.
178 tests와 saved mask/MP4 독립 검증을 통과했다. 기본 설정은 유지한다.

- [수정 수식·결과·한계·영상](paslcd-gaussian-aggregate-binary-ablation-20260907-ko.md)
