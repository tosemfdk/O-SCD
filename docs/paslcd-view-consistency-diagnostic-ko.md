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

`p01=p10=0.01`이고 evidence strength가 `1` 이하로 캡되면, 한 프레임의 reset 쪽 odds는 매우 작다.
즉 **한 view의 contradiction만으로 lifespan을 닫는 controller**는 본질적으로 불안정하다.
이 D1은 그 불안정을 실제 PASLCD replay에서 확인하는 단계다.

---

## 2. 재현 명령

### direct replay

```bash
PYTHONPATH=. conda run -n oscd python -m experiments.run_paslcd_view_consistency_diagnostic \
  --output-root outputs/paslcd_d1_view_consistency_diagnostic
```

### aggregate analysis

```bash
PYTHONPATH=. conda run -n oscd python -m experiments.analyze_paslcd_view_consistency \
  outputs/paslcd_d1_view_consistency_diagnostic \
  --output-dir outputs/paslcd_d1_view_consistency_diagnostic/analysis
```

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
| REOPEN | 109,847 |
| KEEP | 4,010,200 |
| UNCERTAIN | 2,535,979 |
| same-scene repeated transition event | 556,469 |
| same-scene repeated Gaussian | 381,587 |
| base tensor bitwise equal | true |
| base max drift | 0.0 |
| baseline lifecycle structure mismatch | 0 |
| runtime | 350.1 s |

### scene-level posterior diagnostics 요약

scene 평균 기준:

| 항목 | mean |
|---|---:|
| `p_active` | 0.1571 |
| `p_flip` | 0.00637 |
| `p01` | 0.00440 |
| `p10` | 0.00198 |
| `q` | 0.1740 |

이 값들은 detector가 전체적으로는 보수적이지만, **같은 scene 안에서 repeated transition이 매우 많다**는 사실과 함께 읽어야 한다.

---

## 4. 해석

### H1. low evidence mass가 CLOSE를 만든다

PASLCD replay는 close/reopen이 representation drift 때문이 아니라, **view/cue 조건이 바뀔 때 detector가 흔들리는 현상**이라는 점을 보여준다.
즉 동일 scene state인데도 evidence가 약해지는 프레임에서 `CLOSE`가 증가한다.

### H2. camera viewpoint 변화가 transition을 자극한다

이 진단의 핵심 신호는 `same-scene repeated transition event = 556,469`이다.
scene 자체가 하나의 post-change 상태를 유지하는 구간에서도, view 변화가 detector를 뒤집는다.
즉 transition 신호는 scene evolution뿐 아니라 **view-conditioned visibility 변화**를 강하게 타고 있다.

### H3. `p_active` threshold만으로는 부족하다

현재 controller는 `p_active >= 0.6` / `<= 0.4`를 바로 lifecycle action으로 연결한다.
그런데 PASLCD에서는 `p_active`만으로 OPEN/CLOSE를 결정하면, transition odds가 약한데도 threshold crossing 때문에 state flip이 발생한다.

### H4. `q` 자체가 안정적이지 않다

scene 평균 `q`는 0.174 수준이지만 repeated transition이 계속 발생한다.
즉 문제는 “평균적으로 active냐 inactive냐”가 아니라, **같은 Gaussian이 관측 view에 따라 얼마나 흔들리느냐**이다.

---

## 5. 결론

이 D1의 결론은 단순하다.

> PASLCD의 chattering은 representation drift보다, view-conditioned evidence instability가 더 큰 원인이다.

따라서 다음 수정은 geometry가 아니라 detector/controller 쪽이어야 한다.

### 다음 단계 제안: D2

1. `p_active`를 바로 OPEN/CLOSE로 보내지 말고 transition odds를 별도로 본다.
2. unobserved row는 `HOLD`로 유지한다.
3. 단일 view flip 대신 K-view confirmation을 추가한다.
4. 그 다음에야 visibility gating과 view diversity를 붙인다.

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

분석 스크립트는 같은 루트의 `analysis/` 디렉터리에 요약/CSV/플롯을 생성하도록 설계되어 있다.
