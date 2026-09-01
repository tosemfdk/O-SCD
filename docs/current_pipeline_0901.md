# 현재 BF30 evolving SCD 전체 파이프라인

> 상태: 검토용 초안
> 기준 구현: `experiments/run_online_dynamic_active_oscd_density.py`
> 기준 설정: raw-cue `BF=30`, black fixed `NEVER_OPEN`, all-`OPEN` optimization

이 문서는 현재 실험 파이프라인이 실제 코드에서 어떤 순서로 동작하는지를
정리한다. 이상적인 목표 모델이 아니라 **현재 구현되어 실행된 동작**을
기준으로 한다.

특히 다음 세 신호를 서로 구분한다.

1. **Detector evidence:** Gaussian이 image-space change cue에 얼마나 기여하는가
2. **Lifespan state:** Gaussian이 현재 `OPEN`, `NEVER_OPEN`, `CLOSED` 중 무엇인가
3. **Learned change representation:** Gaussian의 DC/geometry/opacity가 최종 change render를 어떻게 만드는가

현재 구현에서는 이 세 신호가 완전히 같은 값이 아니다. 예를 들어 Gaussian이
lifespan상 `OPEN`이면서 detector상 change일 수 있지만, 학습된 DC는 검은 방향일
수 있다.

현재 확정 설계에서 detector가 추적하는 것은 Gaussian별 binary
reference-change validity다. 동일 Gaussian이 `SC1`과 `SC2`에서 모두 change이면
`active -> active KEEP`이며, 두 changed appearance를 별도 lifespan으로 나누는 것은
의도적으로 수행하지 않는다.

---

## 1. 해결하려는 문제

입력은 다음과 같다.

- reference reconstruction으로부터 얻은 3D Gaussian PLY
- 고정 pose를 가진 online image stream
- 각 online image에 대해 미리 계산된 O-SCD pixel + SAM2.1 feature cue

시간 `t`에서 미래 frame을 사용하지 않고 다음을 수행한다.

1. 현재 view의 change cue를 Gaussian별 evidence로 변환한다.
2. Gaussian별 Beta state가 기존 분포에서 reset되었는지 검사한다.
3. reset된 새 분포가 active인지 inactive인지 판단한다.
4. `OPEN`, `KEEP`, `CLOSE`, `REOPEN` lifespan event를 적용한다.
5. 현재 lifespan에 따라 `R_change`를 렌더링하고 학습한다.
6. 필요하면 `OPEN` Gaussian만 clone/split한다.
7. 현재 시점에 렌더되는 change mask를 출력한다.

---

## 2. 전체 구조 요약

```text
reference PLY
    │
    ├─ xyz / opacity / scale / rotation 복사
    └─ change DC = 0으로 초기화
            │
            ▼
  하나의 persistent mutable R_change Gaussian bank
            │
            ├───────────────────────────────────────────────┐
            │                                               │
            ▼                                               ▼
 current frame cue                               lifespan-gated rendering
            │                                               │
            ▼                                               ▼
 alpha×T evidence probe                         SSF representation loss
 (DC 색은 사용하지 않음)                                  │
            │                                               ▼
            ▼                                     masked Adam update
 stable Beta + one candidate                            │
            │                                           ▼
            ▼                                     ACTIVE-only density
 BF30 reset 판단                                          │
            │                                           ▼
            ▼                                    current change render/mask
 OPEN / KEEP / CLOSE / REOPEN
```

한 frame의 실제 실행 순서는 다음과 같다.

```text
I_t와 cue M_t 로드
  → current mutable geometry/opacity로 alpha×T evidence 계산
  → single-candidate Beta update
  → lifecycle controller update
  → first OPEN DC 초기화 옵션 적용
  → optimization 전 current-view render 및 mask 기록
  → 16회의 causal replay optimization
      └─ local update 4에서 ACTIVE-only clone/split
  → optimization 후 current-view render 및 mask 기록
  → 다음 frame
```

중요한 순서 관계는 다음과 같다.

```text
현재 frame detector evidence
    먼저 계산

first-OPEN DC overwrite 및 representation optimization
    그 다음 수행
```

따라서 frame `t`에서 바꾼 DC는 같은 frame의 detector 판단을 직접 바꾸지
않는다. 다만 학습된 geometry/opacity가 frame `t+1` 이후 detector evidence에
간접 feedback된다.

---

## 3. 기호 정의

### 3.1 시간과 영상

| 기호 | 의미 |
|---|---|
| `t` | 현재 online image timestamp |
| `I_t` | 현재 입력 RGB image |
| `M_t(p)` | pixel `p`의 cached O-SCD pixel + SAM2.1 feature cue |
| `B_t(p)` | detector용 binary cue, `1[M_t(p) > 0.5]` |
| `V_0:t` | 지금까지 처리된 causal training view bank |

`M_t`는 학습 loss에는 연속값 그대로 사용되고, detector에서는 threshold
`0.5`를 넘는지에 따라 binary cue `B_t`로 변환된다.

### 3.2 Gaussian과 persistent parameter

Gaussian `g_i`의 persistent raw parameter를 다음처럼 둔다.

```text
Θ_i = {
    x_i,          # xyz
    c_i,          # raw SH DC
    h_i,          # SH-rest
    o_i,          # raw opacity
    s_i,          # raw scaling
    r_i           # raw rotation
}
```

현재 모델은 lifespan slot별 parameter를 따로 갖지 않는다.

```text
g_i의 lifespan 0, 1, 2, ...
    모두 동일한 Θ_i를 공유
```

즉 다음이 성립한다.

```text
CLOSE:
    Θ_i를 삭제하거나 0으로 만들지 않음

REOPEN:
    과거에 학습된 Θ_i와 Adam moment에서 그대로 재개
```

lifespan slot에는 다음 interval metadata만 저장된다.

```text
L_i^k = [τ_start(i,k), τ_end(i,k))
```

각 Gaussian은 최대 16개 lifespan interval을 가질 수 있다.

### 3.3 Lifecycle state

| 상태 | 조건 | 의미 |
|---|---|---|
| `NEVER_OPEN` | `num_states_i = 0` | 한 번도 lifespan이 열린 적 없음 |
| `OPEN` | `current_state_index_i >= 0` | 현재 유효한 lifespan이 있음 |
| `CLOSED` | 과거 interval은 있으나 current slot 없음 | 과거에는 OPEN이었지만 지금은 닫힘 |

### 3.4 Detector Beta state

Gaussian별로 현재 확정 run의 Beta posterior를 저장한다.

```text
Stable_i = Beta(a_i, b_i)
```

- `a_i`: change cue 쪽 pseudo-count
- `b_i`: non-change cue 쪽 pseudo-count
- `p_i = a_i / (a_i + b_i)`: 현재 stable run의 change probability

reset 가능성을 검사하는 동안에는 하나의 candidate block만 추가로 유지한다.

```text
Candidate_i = (A_i, B_i, τ_candidate)
```

full BOCD처럼 모든 run length를 유지하지 않는다. Gaussian당 stable 하나와
candidate 하나만 가지므로 detector state complexity는 `O(1)`이다.

---

## 4. 초기 3D change bank 구성

reference reconstruction PLY에서 다음 값을 읽는다.

```text
reference xyz       → mutable R_change xyz
reference opacity   → mutable R_change opacity
reference scale     → mutable R_change scale
reference rotation  → mutable R_change rotation
reference SH-rest   → mutable R_change SH-rest
```

change DC는 reference RGB DC를 복사하지 않고 raw zero로 초기화한다.

```text
c_i = 0
```

degree-zero SH 변환은 다음 관계를 가진다.

```text
RGB(c_i) = C0 × c_i + 0.5
```

따라서 raw DC `0`의 intrinsic RGB 값은 검정 `0`이 아니라 중립값 `0.5`다.

```text
raw c_i = 0       → intrinsic RGB = 0.5
raw c_i < 0       → intrinsic RGB < 0.5
raw c_i > 0       → intrinsic RGB > 0.5
RGB = 1           → raw c_i = RGB2SH(1) ≈ 1.77245
```

모든 lifespan metadata는 실행 시작 시 `NEVER_OPEN`으로 reset한다. Gaussian
parameter 값은 이 reset으로 변경하지 않는다.

---

## 5. 입력 cue

각 frame에는 다음 cached cue가 붙는다.

```text
M_t = O-SCD pixel cue + SAM2.1 Hiera Tiny feature cue
```

동일한 cue tensor가 두 용도로 사용된다.

```text
view.candidate_map  = M_t    # detector 입력
view.training_target = M_t  # representation 학습 target
```

그러나 사용 방식은 다르다.

### Detector

```text
B_t(p) = 1, if M_t(p) > 0.5
         0, otherwise
```

### Representation optimizer

```text
M_t(p)의 연속값을 그대로 SSF loss에 사용
```

GT mask는 causal inference 또는 optimization에 사용하지 않는다. GT는 전체
online loop가 끝난 뒤 평가 metric을 계산할 때만 로드한다.

---

## 6. Gaussian별 alpha-transmittance evidence

### 6.1 Detector probe render

현재 detector는 학습된 Gaussian DC 색과 cue 색을 비교하지 않는다.

Gaussian마다 differentiable dummy color를 둔다.

```text
u_i = (0, 0, 0), requires_grad = True
```

그리고 현재 mutable bank의 다음 값을 detach하여 probe renderer에 전달한다.

```text
x_i, opacity_i, scale_i, rotation_i
```

probe scaling은 현재 설정에서 native anisotropic scale을 그대로 사용한다.

```text
probe_scaling = native
```

lifespan opacity gating은 detector probe에 사용하지 않는다. 따라서
`NEVER_OPEN`, `OPEN`, `CLOSED` row가 모두 detector에 관측될 수 있다.

현재 runner에서 detector가 보는 geometry/opacity는 별도 immutable reference
bank가 아니라 **현재 mutable bank의 raw geometry/opacity**다. 따라서
representation optimization은 이후 frame의 detector responsibility에 feedback될
수 있다.

### 6.2 Pixel responsibility

Gaussian `g_i`가 pixel `p`에 주는 alpha-transmittance 책임을 다음처럼 둔다.

```text
ρ_i,t(p) = α_i,t(p) × T_i,t(p)
```

- `α_i,t(p)`: Gaussian `i`의 pixel alpha
- `T_i,t(p)`: Gaussian `i` 앞까지의 accumulated transmittance

VJP의 두 channel에 각각 binary change cue와 그 여집합을 넣는다.

```text
e_i,t(+) = Σ_p ρ_i,t(p) × B_t(p)
e_i,t(-) = Σ_p ρ_i,t(p) × (1 - B_t(p))
```

따라서 detector가 측정하는 것은 다음이다.

> 이 Gaussian의 image-space footprint와 visibility responsibility가 현재
> change cue 안과 밖에 각각 얼마나 놓였는가?

다음은 측정하지 않는다.

```text
학습된 DC가 흰색인가?
학습된 DC와 cue 색이 같은가?
현재 rendered R_change mask가 cue와 같은가?
```

### 6.3 Capped pseudo-count

먼저 change 비율과 evidence strength를 구한다.

```text
m_i,t = e_i,t(+) + e_i,t(-)

q_i,t = e_i,t(+) / (m_i,t + ε)

w_i,t = clamp(m_i,t / m_sat, 0, 1)
```

현재 설정은 다음과 같다.

```text
m_sat = 1
minimum evidence mass = 1e-6
```

Beta pseudo-count increment는 다음이다.

```text
Δa_i,t = w_i,t × q_i,t
Δb_i,t = w_i,t × (1 - q_i,t)
```

한 frame에서 들어가는 총 pseudo-count는 최대 1이다.

```text
Δa_i,t + Δb_i,t = w_i,t ≤ 1
```

현재 view에서 evidence mass가 부족한 row는 unobserved로 처리하며 filter state를
업데이트하지 않는다.

---

## 7. Single-candidate Beta reset detector

### 7.1 첫 관측

Gaussian이 처음 reliable evidence를 받으면 changepoint를 검사하지 않고 stable
Beta를 바로 초기화한다.

```text
a_i = 1 + Δa_i,t
b_i = 1 + Δb_i,t
```

이 첫 run은 `changepoint = false`다. 하지만 lifecycle controller는 아직
committed run이 없는 Gaussian에 대해 이 첫 stable Beta만으로 `OPEN`을 선언할
수 있다.

따라서 현재 구조에는 다음 비대칭이 있다.

```text
첫 OPEN:
    BF30 reset 없이도 가능

이미 OPEN된 Gaussian의 CLOSE:
    일반적으로 새로운 BF30 reset 필요
```

### 7.2 Candidate block score

현재 stable Beta가 다음과 같다고 하자.

```text
Stable_i = Beta(a_i, b_i)
```

candidate block에 누적된 pseudo-count를 `(A_i, B_i)`라고 하면 두 가설을
비교한다.

```text
H_KEEP:
    candidate block도 기존 Beta(a_i, b_i)에서 발생

H_RESET:
    candidate block이 fresh Beta(1,1)에서 발생
```

Beta function을 `BetaFn(·,·)`로 표기하면 score는 다음과 같다.

```text
log p(block | RESET)
    = log BetaFn(1 + A_i, 1 + B_i) - log BetaFn(1, 1)

log p(block | KEEP)
    = log BetaFn(a_i + A_i, b_i + B_i) - log BetaFn(a_i, b_i)

S_i
    = log p(block | RESET) - log p(block | KEEP)
```

현재 threshold는 다음과 같다.

```text
BF_threshold = 30
h = ln(30) ≈ 3.4012
```

### 7.3 Candidate state transition

```text
S_i ≤ 0
    → candidate reject
    → candidate block 전체를 stable Beta에 merge

0 < S_i < ln(30)
    → candidate live
    → stable Beta freeze
    → 다음 observed frame의 evidence를 candidate에 추가

S_i ≥ ln(30)
    → reset commit
    → candidate Beta를 새 stable Beta로 승격
    → changepoint_probability = 1
```

candidate가 reject되면 다음처럼 기존 분포가 candidate evidence를 흡수한다.

```text
a_i ← a_i + A_i
b_i ← b_i + B_i
```

이 때문에 non-change evidence가 점진적으로 들어오더라도 BF30 reset으로
분리되지 않고 기존 OPEN stable Beta 안으로 흡수될 수 있다.

---

## 8. Lifecycle controller

### 8.1 Binary label threshold

새 stable run의 change probability는 다음이다.

```text
p_i = a_i / (a_i + b_i)
```

controller는 다음 threshold를 사용한다.

```text
p_i ≥ 0.6
    → confident ACTIVE label

p_i ≤ 0.4
    → confident INACTIVE label

0.4 < p_i < 0.6
    → UNCERTAIN
```

추가 support 조건은 다음이다.

```text
observed = true
visible observations ≥ 1
(a_i + b_i) - (prior_a + prior_b) ≥ 1
```

prior는 `Beta(1,1)`이므로 마지막 조건은 새 run에 pseudo-count mass가 최소
1 이상 있어야 한다는 뜻이다.

### 8.2 Reset gate

이미 committed run이 있는 Gaussian은 다음 중 하나일 때만 새 binary label을
lifespan에 반영한다.

```text
현재 frame에서 BF30 reset commit
또는
이전에 reset은 commit됐지만 p_i가 애매하여 pending으로 latch된 상태
```

reset 없이 stable Beta의 label만 기존 lifecycle과 반대로 바뀌면 lifecycle을
변경하지 않고 `UNCERTAIN`으로 둔다.

```text
OPEN인데 p_i ≤ 0.4
하지만 reset/pending 없음
    → CLOSE하지 않음
    → UNCERTAIN
```

### 8.3 Transition table

reset gate와 confidence를 통과한 뒤의 transition은 다음과 같다.

| 이전 lifespan | 새 binary label | action | parameter/lifespan 결과 |
|---|---:|---|---|
| inactive | 0 | `NONE` | 계속 닫힘 |
| inactive | 1 | `OPEN` | 새 interval slot 할당 |
| active | 1 | `KEEP` | 현재 slot 유지 |
| active | 0 | `CLOSE` | 현재 interval 종료 |

`active -> active` reset은 새로운 lifespan을 만들지 않는다.

```text
active state A → active state B
    Beta reset은 가능
    representation action은 KEEP
    같은 lifespan slot과 같은 persistent Θ_i 유지
```

현재 파이프라인은 `SC1 change`와 `SC2 change`를 둘 다 binary active로 보면
의도적으로 lifespan을 분리하지 않는다. Gaussian 단위에서 reference와 다른 상태가
계속 유효하므로 동일 OPEN interval을 유지하는 것이 현재 설계 계약이다.

### 8.4 CLOSE의 현재 정확한 조건

현재 OPEN Gaussian이 닫히려면 다음 조건을 모두 만족해야 한다.

```text
1. detector에서 Gaussian이 observed됨
2. candidate RESET-vs-KEEP BF가 30 이상으로 commit됨
   또는 이미 해당 reset이 pending으로 latch됨
3. 새 stable Beta의 p_i ≤ 0.4
4. 새 run evidence mass ≥ 1
5. 새 run visible observation count ≥ 1
6. 현재 lifespan이 OPEN
```

학습된 DC 값이 음수이거나 rendered color가 검다는 조건은 CLOSE 판단에 없다.

---

## 9. Lifespan mutation

### 9.1 First OPEN

첫 OPEN은 아직 사용하지 않은 slot 0을 할당한다.

```text
state_start[i,0] = t
state_end[i,0]   = +∞
state_status     = OPEN
current_slot     = 0
```

최근 `render_one` ablation에서는 lifecycle OPEN event가 적용된 직후 다음을
추가로 수행한다.

```text
if action == OPEN and new_slot == 0:
    c_i ← RGB2SH(1) ≈ 1.77245
    DC Adam step/exp_avg/exp_avg_sq ← 0
```

xyz, opacity, scale, rotation 및 그 Adam state는 reset하지 않는다.

### 9.2 CLOSE

```text
state_end[i,k] = t
state_status   = CLOSED
current_slot   = -1
```

parameter와 Adam moment는 변경하지 않는다.

### 9.3 REOPEN

다음 빈 slot `k+1`을 새 interval로 연다.

```text
state_start[i,k+1] = t
state_end[i,k+1]   = +∞
```

하지만 parameter bank는 새로 만들지 않는다.

```text
Θ_i(new interval) = Θ_i(previous interval에서 보존된 값)
```

first-OPEN `C_i=1` 초기화는 REOPEN에는 적용하지 않는다.

---

## 10. 현재 렌더링 support

현재 확정 설정은 다음 mode를 사용한다.

```text
render_support_mode = open_or_never_open_black
```

### 10.1 OPEN

- 현재 persistent DC/geometry/opacity로 렌더링
- 모든 parameter가 trainable path에 연결

### 10.2 NEVER_OPEN

- reference geometry와 opacity로 compositor에 참여
- change color는 RGB 0에 대응하는 degree-zero SH 값으로 override
- 모든 parameter는 detach/freeze

따라서 NEVER_OPEN은 reference geometry/opacity로 뒤의 OPEN Gaussian을 가리지만,
자기 자신은 change render에 회색 또는 흰색 신호를 추가하지 않는 black occluder다.
저장된 parameter와 Adam state는 변경하지 않는다.

### 10.3 CLOSED

- render support에서 완전히 제외
- opacity gate가 0과 같은 효과
- 모든 parameter와 Adam state freeze

### 10.4 Detector와의 차이

representation renderer는 lifespan gating을 적용하지만 detector probe는 적용하지
않는다.

| row 상태 | representation render | detector alpha×T probe |
|---|---|---|
| `NEVER_OPEN` | fixed black occluder | 관측 가능 |
| `OPEN` | learned full attributes | 관측 가능 |
| `CLOSED` | 숨김 | 관측 가능 |

---

## 11. Causal replay optimization

### 11.1 Update schedule

각 image timestamp마다 16번 optimization한다.

```text
updates_per_frame = 16
```

각 local update에서 training view를 다음처럼 선택한다.

```text
확률 0.33:
    현재 view V_t

확률 0.67:
    지금까지 처리된 V_0:t 중 uniform random view
```

미래 view는 sample bank에 들어가지 않는다.

### 11.2 중요한 timestamp 동작

과거 camera/view를 replay하더라도 temporal render에는 현재 timestamp `t`를
명시적으로 전달한다.

```text
camera/target:
    과거 view V_j, j ≤ t

lifespan state:
    현재 시간 t의 OPEN/NEVER_OPEN/CLOSED 상태
```

따라서 현재 OPEN representation을 과거 camera의 cue에 맞춰 학습할 수 있다.
continuous `SC1 -> SC2 -> SC3`에서는 과거 state cue가 현재 state parameter에
다시 gradient를 줄 수 있으므로 cross-state contamination 경로가 존재한다.

### 11.3 SSF loss

현재 training view에서 렌더된 change RGB를 `R_t^j(p)`라고 하자.

```text
P_t^j(p) = sigmoid(mean_RGB(R_t^j(p)))
```

positive detection term은 다음이다.

```text
L_det = mean_p [ M_j(p) × (1 - P_t^j(p)) ]
```

global sparsity regularizer는 다음이다.

```text
L_reg = log(mean_p[P_t^j(p)]² + 1)
```

최종 loss는 다음이다.

```text
L_SSF = L_det + L_reg
```

현재 기본 loss에는 pixel별 explicit negative BCE term
`(1-M) × P`가 없다. non-change 영역은 global sparsity regularizer를 통해
간접적으로만 낮아진다.

### 11.4 Optimizer selection

최근 실험은 다음 옵션을 사용한다.

```text
optimizer_selection = all_open
```

모든 local update에서 모든 `OPEN` row를 모든 parameter group에 대해 선택한다.

```text
selected_i = 1[current lifespan of g_i is OPEN]
```

현재 sampled camera에서 raster radius가 0이어도 optimizer selection에서는
제외하지 않는다.

```text
OPEN + current train view에서 visible
    → 실제 image gradient와 Adam update

OPEN + current train view에서 off-view
    → 현재 gradient는 0일 수 있음
    → 과거 Adam momentum이 있으면 decay된 momentum으로 parameter 이동 가능

NEVER_OPEN 또는 CLOSED
    → optimizer에서 선택하지 않음
    → parameter와 moment 보존
```

학습 parameter와 learning rate는 다음과 같다.

| parameter | learning rate |
|---|---:|
| DC | 0.0025 |
| xyz | 0.00016 |
| SH-rest | 0.000125 |
| opacity | 0.025 |
| scaling | 0.005 |
| rotation | 0.001 |

현재 active SH degree는 0이므로 SH-rest는 optimizer group에 존재하지만 실제
gradient는 0이다.

---

## 12. ACTIVE-only O-SCD density control

### 12.1 Gradient accumulation

각 local update의 screen-space mean gradient를 다음 row에만 누적한다.

```text
OPEN and raster radius > 0
```

optimizer는 all-OPEN selection이지만 density gradient는 여전히 current sampled
view에서 visible한 OPEN row만 갖는다.

### 12.2 실행 시점

16번 update 중 zero-based local update index 4에서 density control을 실행한다.

```text
densify_update_index = 4
```

### 12.3 Clone과 split

평균 screen-space gradient norm이 threshold 이상이어야 한다.

```text
||grad_i|| ≥ 0.001
```

그다음 world scale로 clone과 split을 나눈다.

```text
max_scale_i ≤ 0.01 × scene_extent
    → clone

max_scale_i > 0.01 × scene_extent
    → split
```

split은 source당 두 child를 만들고 source를 제거한다.

### 12.4 Child state 상속

새 child는 생성 순간 source의 다음 값을 한 번 복사한다.

```text
persistent Gaussian parameter
lifespan interval metadata
single-candidate Beta state
lifecycle controller state
```

따라서 source가 OPEN이면 child도 별도의 OPEN event 없이 이미 OPEN인 상태로
태어난다.

생성 후에는 독립 row가 되어 별도 evidence와 posterior를 누적한다. Adam state는
source moment를 복사하지 않고 새 row에 대해 zero로 시작한다.

### 12.5 Pruning

최근 BF30 paired run에서는 다음처럼 실행했다.

```text
min_opacity = 0
max_screen_size = 0
```

따라서 opacity/size pruning은 비활성화됐다. 다만 split source는 child로
대체되므로 항상 제거된다.

FastGS VCD/VCP와 K-view importance는 이 runner에서 사용하지 않는다.

### 12.6 Cue-mixture split ablation

현재 frame의 pre-optimization raw cue evidence에서 change/non-change 책임도가 모두
큰 large ACTIVE Gaussian을 gradient와 무관하게 split하는
`active_oscd_cue_mixture` 정책도 비교 구현으로 보존한다.

Mixture 신호는 large ACTIVE row의 split 조건에만 추가된다. Mixture 기반 clone,
pruning, lifecycle 전이는 없고, 기존 gradient clone/split은 그대로 함께 동작한다.

Threshold 0.5 continuous u16의 3-way 비교 결과는 다음과 같았다.

| Density 정책 | mean-frame mIoU | mean-frame F1 | split source |
|---|---:|---:|---:|
| off | **0.4903** | **0.6271** | 0 |
| gradient-only | 0.4874 | 0.6248 | 1,144 |
| gradient + cue-mixture | 0.4839 | 0.6209 | 24,272 |
| mixture + child-only black prune | 0.4852 | 0.6224 | 16,685 |

Cue-mixture의 24,272 split 중 23,478개는 gradient 조건 없이 mixture만으로 발생했다.
Final black ACTIVE 비율도 `0.6441 / 0.6451 / 0.6625`로 mixture가 가장 높았다.
따라서 cue-mixture OR split은 채택하지 않으며, 이 BF30 black continuous 조건에서는
gradient densify도 density-off보다 낫지 않았다. 상세 결과는
[`bf30-black-cue-mixture-density-ablation-ko.md`](bf30-black-cue-mixture-density-ablation-ko.md)에
기록한다.

Child-only black prune 정책은 generation-zero와 same-event child를 보호하고, 1 frame
이상 지난 OPEN child 중 intrinsic DC가 0.5 미만인 row만 hard-prune한다. 직접 부모와
sibling support가 모두 없으면 가장 덜 검정인 child 하나를 보존한다. Continuous에서는
8,539개를 삭제해 final black ACTIVE 비율을 `0.6625 -> 0.6446`으로 회복했지만
density-off보다 mIoU/F1이 `-0.0051/-0.0047` 낮았다.

PASLCD 20 scenes/500 frames에서도 density-off / gradient / mixture / child-prune
mIoU/F1은 `0.4566/0.6125`, `0.4561/0.6120`, `0.4515/0.6077`,
`0.4523/0.6084`였다. Child pruning은 62,678개 split child를 삭제했지만 O-SCD online
`0.4887/0.6423`보다 `-0.0365/-0.0339` 낮았다. Mixture 대비 `+0.0008` mIoU point
estimate는 Cantina CUDA 반복 변동과 같은 규모여서 개선 증거로 보지 않는다. 상세 결과는
[`bf30-black-child-prune-paslcd-ablation-ko.md`](bf30-black-child-prune-paslcd-ablation-ko.md)에
기록한다.

---

## 13. Current mask 출력

현재 view를 현재 lifespan support로 렌더링한다.

```text
R_t(p) = rendered change RGB
```

평가용 scalar score는 sigmoid가 아니라 raw render의 channel mean을 사용한다.

```text
S_t(p) = clamp(mean_RGB(R_t(p)), 0, 1)
```

최종 binary mask는 다음이다.

```text
Mask_t(p) = 1[S_t(p) ≥ 0.5]
```

학습 loss는 `sigmoid(mean_RGB(R))`를 사용하지만 평가 mask는
`clamp(mean_RGB(R)) ≥ 0.5`를 사용한다는 차이가 있다.

각 frame에서 다음 두 mask를 저장한다.

```text
pre-opt mask:
    lifecycle event 적용 직후, 16-step optimization 전

post-opt mask:
    16-step optimization 및 density control 후
```

---

## 14. Detector와 DC optimizer의 분리

현재 파이프라인에서 가장 중요한 구조적 차이는 다음이다.

### Detector가 보는 것

```text
현재 view의 binary cue
현재 mutable geometry/opacity에 의한 alpha×T responsibility
과거 stable/candidate Beta evidence
```

### Representation optimizer가 보는 것

```text
현재 또는 과거 replay view의 continuous cue
현재 lifespan-gated composite R_change render
positive detection + global sparsity loss
```

### Detector가 직접 보지 않는 것

```text
학습된 raw DC c_i
Gaussian의 intrinsic RGB가 흰색인지 검은색인지
최종 thresholded R_change mask에서 Gaussian이 positive인지
SSF loss가 해당 Gaussian을 아래 방향으로 밀었는지
```

따라서 다음 상태가 가능하다.

```text
lifespan = OPEN
detector stable label = active
learned DC = negative/black direction
```

가능한 과정은 다음과 같다.

```text
1. alpha×T footprint가 change cue와 겹쳐 OPEN
2. replay/global sparsity/다른 Gaussian과의 중복 때문에 해당 DC가 아래로 학습
3. DC 값은 detector input이 아니므로 lifecycle은 그대로 OPEN
4. BF30 reset + inactive posterior 조건이 없으면 CLOSE되지 않음
```

---

## 15. Raw-evidence learned-DC control 설정값 요약

아래 표는 raw single-candidate Beta와 learned DC를 사용하던 control 설정이다.
현재 experiment branch의 실행 기본값과 혼동하면 안 된다.

| 항목 | 값 |
|---|---|
| stream | continuous `ref -> SC1 -> SC2 -> SC3` |
| frames | 304 |
| seed | 0 |
| detector | single-candidate Beta |
| Beta prior | `Beta(1,1)` |
| reset threshold | `BF=30`, `ln BF≈3.4012` |
| binary cue threshold | `> 0.5` |
| evidence mode | capped, 최대 1 pseudo-count/view |
| open threshold | `p ≥ 0.6` |
| close threshold | `p ≤ 0.4` |
| minimum new-run evidence | 1.0 |
| max lifespan slots | 16 |
| detector scaling | native anisotropic |
| render support | `OPEN ∪ NEVER_OPEN`, `CLOSED` hidden |
| NEVER_OPEN color/parameter | RGB black override, 전부 freeze |
| optimizer selection | all `OPEN`, visibility restriction 없음 |
| first OPEN DC | preserve 또는 `C_i=1` ablation |
| REOPEN | persistent parameter/moment 재사용 |
| updates/image | 16 |
| current-view sampling probability | 0.33 |
| past-view replay probability | 0.67 |
| densify index | 4 |
| density target | ACTIVE and gradient-observed |
| opacity/size pruning | off |
| evaluation threshold | raw render mean `≥ 0.5` |

First-OPEN DC는 별도 representation ablation으로 남기며 기본값은 preserve다.

### 15.1 현재 experiment branch 기본 계약

`experiment/bf30-soft-transition-opacity-gate`의 현재 runner 기본값은 다음과 같다.

| 항목 | 값 |
|---|---|
| detector | `lifespan_gate_beta`, BF30 |
| change color | `lifespan_gate` |
| candidate output gate | `close_only_log_bf_progress` |
| render support | `open_or_never_open` |
| optimizer | `all_open` |
| loss | `local_growth_replay`, weight 7.5 |
| density | `active_oscd` |
| min opacity | 0 |

CLOSE-only gate는 detector observation, BF30 commit, training render, optimizer,
density를 바꾸지 않는다. Committed CLOSED의 OPEN candidate는 계속 숨기고,
committed OPEN의 CLOSE candidate만 누적 log BF 진행도에 따라 output opacity를
줄인다.

```text
hard gate:          candidate_render_gate = none
bidirectional gate: candidate_render_gate = log_bf_progress
CLOSE-only default: candidate_render_gate = close_only_log_bf_progress
```

연속 304 frames에서 hard 대비 mIoU/F1은 `0.4261/0.5696 ->
0.4403/0.5833`, PASLCD 20 scenes에서는 `0.3711/0.5125 -> 0.3840/0.5257`로
증가했다. PASLCD에서는 20개 scene 중 19개가 개선됐지만, CLOSE-only 절대 성능은
O-SCD online `0.4887/0.6423`보다 낮다. 상세 내용은
[`docs/bf30-close-only-transition-opacity-gate-ablation-ko.md`](bf30-close-only-transition-opacity-gate-ablation-ko.md)에
기록한다.

### 15.2 Raw-cue learned-DC CLOSE-candidate 출력 감쇠

Section 15의 raw single-candidate Beta + learned-DC control에도 CLOSE-only 출력 감쇠를
적용할 수 있다. Raw detector candidate는 단순 분포 reset 후보이므로, 현재 OPEN이라는
이유만으로 전부 CLOSE 방향으로 간주하지 않는다. Fresh candidate Beta posterior를
직접 계산해 non-change 방향인 row만 선택한다.

```text
candidate_a = 1 + candidate_delta_a
candidate_b = 1 + candidate_delta_b
p_candidate_change = candidate_a / (candidate_a + candidate_b)

close_candidate =
    committed_OPEN
    AND candidate_live
    AND p_candidate_change <= 0.4

progress = clamp(candidate_log_BF / log(30), 0, 1)
output_opacity_multiplier = 1 - progress
```

이때 learned DC는 detector 입력이 아니다. Candidate 방향과 progress는 모두 신규
frame optimization 전 raw cue pseudo-count로 계산한다. 감쇠는 pre/post metric 및
visualization output에만 적용하고 다음 경로에는 넣지 않는다.

```text
detector probe
stable/candidate Beta update
hard OPEN/CLOSE commit
training render와 SSF loss
G_prev 계산
density control
stored Gaussian parameter와 Adam state
```

동일 run에서 동일 parameter/lifecycle의 hard render를 control로 함께 평가한 결과는
다음과 같다.

| 데이터 | hard learned DC | CLOSE-candidate 감쇠 | mIoU/F1 차이 |
|---|---:|---:|---:|
| continuous 304 frames | 0.4896 / 0.6263 | 0.4938 / 0.6304 | +0.0042 / +0.0041 |
| PASLCD 20 scenes/500 frames | 0.4566 / 0.6124 | 0.4604 / 0.6159 | +0.0038 / +0.0035 |

PASLCD scene 승/패는 `19/1`이었다. OPEN-candidate preview, detector learned-DC input,
CLOSED drift, wrong gradient, future-view access는 모두 0이었다. 이 결과는 raw detector의
판단 개선이 아니라 hard CLOSE 전 candidate uncertainty를 output에 반영한 precision
보정이다. 상세 내용은
[`docs/bf30-raw-cue-learned-dc-close-candidate-output-ablation-ko.md`](bf30-raw-cue-learned-dc-close-candidate-output-ablation-ko.md)에
기록한다.

---

## 16. Causality 및 invariant

### Online causality

- detector는 frame `t`의 current view만 사용한다.
- optimizer replay는 `V_0:t`만 사용한다.
- 미래 view access는 runtime audit로 검사한다.
- GT와 manual SC boundary는 causal loop에 들어가지 않는다.
- SC boundary `(95, 199)`는 실험 종료 후 diagnostics에만 사용한다.

### Lifecycle invariant

- `active -> active`는 항상 같은 slot의 `KEEP`이다.
- `CLOSE`는 parameter와 Adam state를 변경하지 않는다.
- CLOSED row는 이후 unrelated update/density event 동안 bitwise 보존돼야 한다.
- REOPEN 전 CLOSED snapshot을 다시 검사한다.

### Gradient invariant

- optimizer mask 밖의 nonzero gradient를 frame마다 검사한다.
- 최근 fixed-NEVER_OPEN/all-OPEN run에서 NEVER_OPEN 선택 수와 outside-mask
  gradient violation은 0이었다.

### Topology invariant

- parameter, optimizer state, lifecycle, detector, controller row 수가 항상 같다.
- 모든 dynamic Gaussian은 unique stable ID를 가진다.
- clone/split lineage는 `topology_events.jsonl`에 기록한다.

---

## 17. 주요 출력 파일

각 run은 다음을 생성한다.

```text
summary.json
    전체 설정, metric, event count, invariant audit

frame_metrics.csv
    frame별 posterior/action/optimization/mask metric

lifecycle_events.jsonl
    stable Gaussian ID별 OPEN/KEEP/CLOSE event

density_events.csv
    frame별 clone/split/prune 및 replay sample 기록

topology_events.jsonl
    dynamic Gaussian CREATE/DELETE lineage

checkpoint.pt
    --skip-checkpoint가 아닐 때만 저장

causal_visuals/raw_render/*.png
causal_visuals/thresholded_render/*.png
causal_visuals/event_render/*.png
```

최근 paired BF30 first-OPEN ablation의 상세 결과는 다음 문서에 있다.

- [`bf30-first-open-render-one-all-open-ablation-ko.md`](bf30-first-open-render-one-all-open-ablation-ko.md)
- [`single-candidate-beta-threshold-sweep-ko.md`](single-candidate-beta-threshold-sweep-ko.md)

---

## 18. 현재 구조에서 검토가 필요한 지점

아래 항목은 문서를 읽은 뒤 의도와 맞는지 확인할 지점이다.

### 18.1 Detector와 learned DC의 분리: 확정

현재 detector label은 신규 frame의 pre-optimization raw alpha×T cue evidence만
뜻한다. 학습된 DC, DC-cue gap, optimizer DC 이동량은 detector evidence가 아니다.
동일 cue로 representation을 학습한 뒤 그 결과를 다시 detector 입력으로 쓰면
관측을 이중 사용하고 transition evidence를 self-erasure하므로 금지한다.

### 18.2 CLOSE에 BF30 reset이 반드시 필요한가?

현재 stable posterior가 inactive 쪽으로 내려가도 reset이 없으면
`UNCERTAIN`이며 CLOSE하지 않는다.

검토 질문:

```text
OPEN row의 p_i가 충분히 낮아진 경우 reset 없이 CLOSE를 허용해야 하는가?
```

### 18.3 First OPEN과 CLOSE가 비대칭인 것이 맞는가?

first reliable run은 BF30 없이 OPEN될 수 있지만 이후 CLOSE는 BF30 reset을
요구한다.

검토 질문:

```text
noise OPEN을 막기 위해 first OPEN도 reset/candidate confirmation을 거쳐야 하는가?
```

### 18.4 Past-view replay가 현재 state를 학습하는 것이 맞는가?

현재 SC3 parameter가 SC1/SC2 camera와 cue로 다시 학습될 수 있다.

검토 질문:

```text
replay bank를 current lifespan/state-local view로 제한해야 하는가?
```

### 18.5 Off-view OPEN momentum update가 맞는가?

`all_open`에서는 현재 sampled view에 보이지 않아도 Adam step이 진행된다.

검토 질문:

```text
visibility restriction을 제거한다는 것이 zero-gradient Adam momentum 이동까지
허용한다는 뜻인가?
```

### 18.6 NEVER_OPEN black occluder: 확정

NEVER_OPEN은 reference geometry/opacity의 alpha occlusion만 제공하고 change color는
RGB 0이어야 한다. 저장 DC를 학습하거나 덮어쓰지 않고 renderer override로 black을
구현한다.

### 18.7 State별 parameter가 없는 것이 맞는가?

lifespan interval은 여러 개지만 Gaussian parameter는 하나다.

검토 질문:

```text
REOPEN 시 과거 parameter를 재사용할 것인가,
새 lifespan 전용 parameter를 할당할 것인가?
```

### 18.8 Dynamic child가 OPEN state를 그대로 상속하는 것이 맞는가?

clone/split child는 별도의 detector 확인 없이 source의 active state와 Beta를
복사한다.

검토 질문:

```text
child를 독립적으로 재검증해야 하는가,
아니면 source evidence를 그대로 상속하는 것이 맞는가?
```

---

## 19. 한 문장 요약

> 현재 파이프라인은 각 신규 frame의 optimization 전 raw cue와 Gaussian의 alpha×T
> overlap만으로 BF30 Beta reset과 binary lifespan을 판단하고, 그 뒤 별도의 SSF
> replay loss로 representation을 학습한다. Learned DC와 replay는 detector evidence로
> 역류하지 않으며 `active -> active`는 의도적으로 동일 lifespan을 유지한다.
