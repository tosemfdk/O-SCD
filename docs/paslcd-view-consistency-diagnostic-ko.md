# PASLCD view-consistency diagnostic (D1)

이 문서는 PASLCD에서 direct binary lifespan detector가 왜 같은 scene 안에서도 `OPEN → CLOSE → REOPEN`을 반복하는지 분해하기 위한 D1 기준선이다.
핵심 목적은 representation을 바꾸는 것이 아니라, **view/cue 조건부 detector chattering**이 실제 원인인지 확인하는 것이다.

## 1. 실행한 것

### causal contract

```
fixed camera + cached O-SCD/SAM cue
        ↓
immutable-reference alpha-T evidence
        ↓
direct binary state filter
        ↓
OPEN / KEEP / CLOSE / REOPEN
        ↓
current slot only
```

이 진단에서는 다음을 하지 않는다.

- GT mask를 detector에 사용하지 않음
- temporal slot DC/xyz/opacity/scale/rotation을 evidence에 사용하지 않음
- geometry optimization을 수행하지 않음
- future frame을 보지 않음

### detector math

관측된 Gaussian `i`에 대해

```
q_i = Δa_i / (Δa_i + Δb_i + ε)
w_i = Δa_i + Δb_i
```

그리고 `η=0.9`의 고정된 sensor reliability로

```
log L1 = w * [q log η + (1-q) log(1-η)]
log L0 = w * [q log(1-η) + (1-q) log η]
```

를 사용했다.

transition posterior는

```
w00 = (1-b)(1-p01)L0
w01 = (1-b)p01L1
w10 = bp10L0
w11 = b(1-p10)L1

P(z_t=1) = normalize(w01 + w11)
P(z_t=0) = normalize(w00 + w10)
P(flip) = normalize(w01 + w10)
```

으로 계산했다.

기본값은

- `p01 = 0.01`
- `p10 = 0.01`
- `η = 0.9`

이다.

### 중요한 해석

`p01=p10=0.01`이고 evidence strength가 `1` 이하로 캡되면 한 프레임의
transition branch odds 상한은 `(0.01 / 0.99) * 9 = 0.090909...`이다. 그런데
기존 controller는 이 transition odds가 아니라 누적된 marginal `p_active`가
`0.6/0.4`를 넘었는지만 보고 lifespan을 바꾼다. D1은 이 두 의미가 실제
PASLCD에서 얼마나 어긋나는지 측정한다.

---

## 2. 재현 명령

### direct replay

```bash
PYTHONPATH=. conda run -n oscd python -m experiments.run_paslcd_view_consistency_diagnostic \
  --output-root outputs/paslcd_d1_view_consistency_diagnostic
```

### aggregate analysis

```bash
PYTHONPATH=. python3 -m experiments.analyze_paslcd_view_consistency \
  outputs/paslcd_d1_view_consistency_diagnostic \
  --output-dir outputs/paslcd_d1_view_consistency_diagnostic/view_consistency_analysis
```

현재 `oscd` conda 환경에는 matplotlib이 없으므로 수치/테스트 환경과 plot 실행
환경을 분리했다. 분석 helper는 matplotlib 없이 import 가능하며, plot 생성 시에만
matplotlib을 요구한다.

---

## 3. 검증된 결과

### 전체 PASLCD replay

| 항목 | 값 |
|---|---:|
| scene 수 | 20 |
| frame 수 | 500 |
| event 수 | 1,369,336 |
| OPEN | 922,714 |
| CLOSE | 446,622 |
| REOPEN (OPEN의 부분집합) | 109,847 |
| KEEP | 4,010,200 |
| UNCERTAIN | 2,535,979 |
| same-scene repeated transition event | 556,469 |
| same-scene repeated Gaussian | 381,587 |
| base tensor bitwise equal | true |
| base max drift | 0.0 |
| baseline lifecycle structure mismatch | 0 |
| detector replay runtime | 350.14 s |
| aggregate analysis runtime (실행 로그 기준) | 101.84 s |
| aggregate analysis peak RSS (실행 로그 기준) | 3.82 GiB |

### 원인 분해 요약

| 진단 | 결과 |
|---|---:|
| CLOSE raw mass / 이전 KEEP median, q50 | 1.2822 |
| CLOSE capped strength / 이전 KEEP median, q50 | 1.0000 |
| CLOSE 직전 1-frame 이상 미관측 비율 | 24.62% |
| event camera delta / 이전 KEEP median, q50 | 1.0763 |
| lifecycle event의 selected branch odds, q50 / q75 / max | 0.08083 / 0.090909 / 0.090909 |
| event에서 이전 KEEP 대비 `q` sign flip 비율 | 11.89% |
| Gaussian `q` variance와 repeated cohort의 point-biserial 상관 | 0.2949 |
| Gaussian mean `|Δq|`와 repeated cohort의 point-biserial 상관 | 0.2213 |

---

## 4. 해석

### H1. low evidence mass가 주원인인가

CLOSE 시 raw mass는 이전 KEEP median보다 오히려 중앙값 기준 `1.282×`이고,
capped strength 중앙값 비율은 `1.0`이다. 따라서 **단순한 low-mass만으로 전체
chattering을 설명할 수 없다.** 다만 CLOSE의 `24.62%`, REOPEN의 `22.15%`는
직전 관측과 사이에 적어도 한 frame의 visibility gap이 있어 보조 원인이다.

### H2. camera viewpoint 변화가 transition을 자극한다

event의 camera delta는 같은 Gaussian의 이전 KEEP 중앙값 대비 q50 `1.076×`였다.
큰 camera jump에만 transition이 몰렸다고 보기는 어렵다. 즉 고정된 하나의
view-delta threshold만 넣는 것은 첫 수정으로 충분하지 않다.

### H3. `p_active` threshold만으로는 부족하다

현재 controller는 `p_active >= 0.6` / `<= 0.4`를 바로 lifecycle action으로
연결한다. 그러나 lifecycle event에서 선택된 `P01/P00` 또는 `P10/P11`은 q75와
최댓값이 모두 약 `0.090909`였고 `1`을 넘은 사례가 없다. 즉 현재 observation의
transition branch는 stay branch보다 항상 약한데도, 과거까지 누적된 marginal
`p_active`가 threshold를 넘었다는 이유로 OPEN/CLOSE가 실행된다. **state belief와
transition decision의 분리가 필요한 직접 증거**다.

### H4. `q` 자체가 안정적이지 않다

event row의 `11.89%`가 이전 KEEP 중앙값과 반대 `q` sign을 보였다. Gaussian별
`q` variance와 repeated cohort의 상관은 `0.2949`, mean `|Δq|` 상관은
`0.2213`이었다. 네 가설 중 가장 강한 신호는 **view-conditioned cue/emission
변동성**이다.

---

## 5. 결론

이 D1의 결론은 단순하다.

> PASLCD chattering의 주된 구조는 representation drift가 아니라, 변동하는
> cue를 누적 state belief에 넣은 뒤 그 belief를 곧바로 transition으로 해석한
> detector/controller coupling이다.

따라서 다음 수정은 geometry가 아니라 detector/controller 쪽이어야 한다.

### 다음 단계 제안: D2

1. `p_active`는 현재-state diagnostic으로만 유지하고, representation transition은
   `P01/P00` 또는 `P10/P11`의 observation evidence로 별도 결정한다.
2. 첫 ablation은 consecutive observed-view confirmation만 넣고, unobserved row는
   counter를 reset하지 않는 `HOLD`로 둔다.
3. `K=1,2,3`을 비교해 PASLCD false transition 감소와 ESCD 실제 transition delay를
   함께 본다.
4. 그 뒤에도 chattering이 남을 때만 low-mass gate, view diversity, `η` calibration을
   순서대로 추가한다.

---

## 6. 남은 제한

- 이 문서는 detector 진단이다. representation 개선 결과가 아니다.
- `OPEN/CLOSE/REOPEN`이 줄어들어도 mIoU 향상은 자동으로 보장되지 않는다.
- D1만으로는 “어떤 transition gate가 최적”인지는 결정하지 않는다.

---

## 7. 생성된 산출물

- `outputs/paslcd_d1_view_consistency_diagnostic/comparison.json`
- `outputs/paslcd_d1_view_consistency_diagnostic/Instance_*/<scene>/summary.json`
- `outputs/paslcd_d1_view_consistency_diagnostic/Instance_*/<scene>/frame_metrics.csv`
- `outputs/paslcd_d1_view_consistency_diagnostic/Instance_*/<scene>/lifecycle_events.jsonl`
- `outputs/paslcd_d1_view_consistency_diagnostic/view_consistency_analysis/summary.json`
- `outputs/paslcd_d1_view_consistency_diagnostic/view_consistency_analysis/cohort_stats.csv`
- `outputs/paslcd_d1_view_consistency_diagnostic/view_consistency_analysis/event_windows.csv`
- `outputs/paslcd_d1_view_consistency_diagnostic/view_consistency_analysis/*.png`

`outputs/`의 NPZ/CSV/PNG는 Git에 커밋하지 않는다.

---

## 8. D2/D3 offline confirmation replay

D1 산출물의 sparse NPZ만 다시 읽어서 detector policy를 offline으로 replay했다. 이 replay는
GT mask, temporal DC/geometry, rendered mask를 읽지 않는다. 각 Gaussian의 committed
representation state만 내부적으로 다시 추적하며, 관측되지 않은 row는 `HOLD`한다.

### Replay rule

현재 committed state가 inactive이면 OPEN support를:

```text
BF_open = (P01 / P00) / (p01 / (1 - p01))
```

현재 committed state가 active이면 CLOSE support를:

```text
BF_close = (P10 / P11) / (p10 / (1 - p10))
```

로 정의했다. 즉 Markov transition prior odds를 제거하고, 현재 observation이 stay보다
flip likelihood를 얼마나 더 지지하는지만 본다. `evidence_strength < 1e-6`이면 HOLD,
관측됐지만 support가 아니면 consecutive counter를 reset한다. Counter가 `K`번 연속
support에 도달할 때만 OPEN/CLOSE를 commit한다.

### PASLCD 20-scene replay 결과

기존 D1 posterior-hysteresis baseline은 OPEN/CLOSE/REOPEN `922,714/446,622/109,847`,
repeated transition extras `556,469`였다.

| condition | OPEN | CLOSE | REOPEN | repeated extras | final active | unique opened |
|---|---:|---:|---:|---:|---:|---:|
| K=1, BF>=1 | 3,906,023 | 3,347,980 | 1,827,332 | 5,175,312 | 558,043 | 2,078,691 |
| K=1, BF>=3 | 1,813,888 | 1,290,570 | 533,161 | 1,823,731 | 523,318 | 1,280,727 |
| K=2, BF>=1 | 1,131,043 | 726,616 | 216,745 | 943,361 | 404,427 | 914,298 |
| K=2, BF>=3 | 493,138 | 176,997 | 44,311 | 221,308 | 316,141 | 448,827 |
| K=3, BF>=1 | 511,019 | 210,728 | 40,849 | 251,577 | 300,291 | 470,170 |
| K=3, BF>=3 | 234,891 | 34,898 | 8,255 | 43,153 | 199,993 | 226,636 |

가장 안정적인 설정은 `K=3, BF>=3`이다. repeated extras가 `556,469 -> 43,153`으로
약 `92.2%` 감소했고, REOPEN도 `109,847 -> 8,255`로 줄었다. 다만 final active와
unique opened도 크게 줄었으므로, 이 설정은 PASLCD false transition 억제에는 강하지만
ESCD 실제 transition delay/recall 검증 없이는 최종 detector로 확정하면 안 된다.

### 구현 반영

- `temporal.view_consistent_binary_lifespan_controller`를 추가해 production runner에서도
  `--lifecycle-controller view_consistent`를 선택할 수 있게 했다.
- `p_active`는 diagnostic으로만 저장하고, lifecycle mutation은 committed state별
  `BF_open`/`BF_close` consecutive support로만 결정한다.
- 기본값은 conservative하게 `--transition-confirmation-views 2`,
  `--min-transition-bayes-factor 3.0`, `--min-transition-evidence-strength 1e-6`이다.

후속 full rendering에서는 `K=3/BF=3`을 선택했다. PASLCD all-geometry는 O-SCD
online보다 mean-frame mIoU가 `+0.0086` 높으면서 repeated extras를 `12.9x`
줄였다. ESCD `ref -> sc1 -> sc2 -> sc3`에서도 CLOSE/REOPEN과 boundary 인접 반응이
유지됐다. 자세한 수치와 detector/representation 분해는
[`view-consistent-binary-lifespan-confirmation-ko.md`](view-consistent-binary-lifespan-confirmation-ko.md)에
기록한다.

### D2 산출물

- `outputs/paslcd_d2_transition_confirmation_replay/summary.json`
- `outputs/paslcd_d2_transition_confirmation_replay/condition_metrics.csv`

이 산출물도 Git에 커밋하지 않는다.
