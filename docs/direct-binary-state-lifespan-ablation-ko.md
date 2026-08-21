# Direct binary Bayesian lifespan ablation

## 1. 목적과 결론

이 문서는 `MAPResetBernoulliFilter`/BOCD 계열을 제거하지 않고 그대로 보존한 상태에서,
별도 ablation으로 추가한 direct binary Bayesian state filter 결과를 정리한다. 질문은 하나다.

> BOCD run-length/reset posterior 없이 `P(z_t=active | D_{1:t})`를 직접 갱신해도
> causal `OPEN/CLOSE/REOPEN` lifespan을 만들 수 있는가?

결론은 세 층으로 나누어야 한다.

1. **Lifecycle correctness:** 성공했다. `CLOSE=178,355`, `REOPEN=78,731`이 발생했고,
   active-to-active false split, closed slot drift, slot reuse, zero-init violation은 모두 0이었다.
2. **Representation performance:** B0 DC-only는 mean-frame mIoU `0.6063232380`, F1
   `0.7282668362`를 기록했다. MAP-reset DC-only 연속 run의 `0.4264/0.5472`보다 높고,
   beam-2 DC-only `0.60466448/0.72814450`와 거의 같다.
3. **Detector quality:** 아직 실패가 크다. 이벤트 수가 `OPEN=299,673`, `CLOSE=178,355`,
   `UNCERTAIN=9,814,881`로 매우 많아 전환부만 깨끗하게 잡는 detector가 아니라 heavy
   chattering detector로 해석해야 한다.

따라서 direct binary filter는 “lifespan mechanism이 닫고 다시 열 수 있음”을 증명하는
ablation이다. 그러나 이것만으로 최종 detector가 안정적이라고 주장하지 않는다.

## 2. BOCD 보존 범위

이번 ablation은 기존 BOCD 라인을 대체하거나 삭제하지 않는다.

- 기존 `exact`, `map_reset`, `beam2` BOCD 실험은 그대로 비교 기준으로 남긴다.
- 새 runner는 `experiments/run_online_binary_state_lifespan_thaw.py`이다.
- 새 filter/controller는 `temporal/binary_state_filter.py`와
  `temporal/binary_state_lifespan_controller.py`에 분리되어 있다.
- Output `summary.json`의 algorithm은 `direct_binary_state_filter`로 기록된다.

즉 이번 결과는 “BOCD 대신 이것을 최종 선택한다”가 아니라,
`BOCD reset inference`와 `binary lifecycle/controller/representation`을 분해하기 위한
독립 ablation이다.

## 3. Direct binary filter 수식

각 Gaussian `i`의 latent label은 reference와의 현재 차이만 의미한다.

- `z_{i,t}=1`: 현재 reference와 다름
- `z_{i,t}=0`: 현재 reference와 다르지 않음

한 update에서 기존 active posterior를

\[
b=P(z_{t-1}=1\mid D_{1:t-1})
\]

라고 둔다. Immutable reference evidence에서 positive/negative pseudo-count
`(\Delta a, \Delta b)`를 얻고,

\[
w=\Delta a+\Delta b,
\qquad
q=\frac{\Delta a}{\Delta a+\Delta b+\epsilon}
\]

로 둔다. Sensor reliability는 `\eta`, transition prior는
`p_{01}=P(0\rightarrow1)`, `p_{10}=P(1\rightarrow0)`이다.

Emission likelihood는 다음 power likelihood 형태다.

\[
\log L_1
= w\left(q\log\eta+(1-q)\log(1-\eta)\right)
\]

\[
\log L_0
= w\left(q\log(1-\eta)+(1-q)\log\eta\right)
\]

현재 observation에 대한 two-state transition joint는 네 항을 log-space에서 정규화한다.

\[
\tilde P_{00}=(1-b)(1-p_{01})L_0
\]

\[
\tilde P_{01}=(1-b)p_{01}L_1
\]

\[
\tilde P_{10}=bp_{10}L_0
\]

\[
\tilde P_{11}=b(1-p_{10})L_1
\]

\[
(P_{00},P_{01},P_{10},P_{11})
=\operatorname{normalize}(\tilde P_{00},\tilde P_{01},\tilde P_{10},\tilde P_{11})
\]

최종 posterior와 diagnostic flip probability는 다음과 같다.

\[
P(z_t=1\mid D_{1:t})=P_{01}+P_{11}
\]

\[
P(\text{flip}_t\mid D_{1:t})=P_{01}+P_{10}
\]

`total_mass < min_evidence_mass`이거나 `w=0`인 row는 unobserved로 처리한다. 이 경우
`p_active`, visible count, timestamp, lifecycle state를 bitwise 변경하지 않는다.

## 4. Causal immutable-reference evidence separation

Evidence path는 기존 Bayesian lifespan runner의 핵심 제약을 유지한다.

1. Alpha-T evidence는 immutable reference Gaussian에서 계산한다.
2. Active lifespan mask, closed slot, state-local geometry delta는 detector evidence에 들어가지 않는다.
3. GT mask와 manual boundary `[95,199]`는 online loop 이후 평가/진단에만 사용한다.
4. 매 frame은 global timestamp 순서로 처리하며 미래 frame을 사용하지 않는다.

따라서 direct binary detector는 representation이 현재 render를 어떻게 바꾸는지와 독립적으로
`ref -> current observation` evidence를 받는다. Closed Gaussian도 reference footprint가 보이면
다시 evidence를 받아 `REOPEN`될 수 있다.

## 5. Lifecycle controller와 optimizer contract

Controller는 `p_active` hysteresis만으로 action을 결정한다.

| 이전 상태 | 조건 | Action | Slot 변화 |
|---|---|---|---|
| inactive | `p_active >= 0.6` | `OPEN` | 다음 unused slot 할당 |
| inactive | `p_active <= 0.4` | `NONE` | closed 유지 |
| active | `p_active >= 0.6` | `KEEP` | 같은 slot 유지 |
| active | `p_active <= 0.4` | `CLOSE` | 현재 slot 종료 |
| any | `0.4 < p_active < 0.6` | `UNCERTAIN` | 변화 없음 |

중요한 invariant는 다음과 같다.

- `active -> active`는 항상 `KEEP`이며 같은 slot을 유지한다.
- `inactive -> active`는 이전 closed slot을 덮어쓰지 않고 새 slot을 연다.
- 새 `OPEN`/`REOPEN` slot의 모든 state-local parameter는 강제로 0으로 초기화한다.
- Optimizer가 있으면 해당 row-slot의 Adam moment와 step도 같이 reset한다.
- Optimizer step은 현재 active `(row, slot)` pair에만 적용한다.
- Closed pair는 snapshot 후 최종 시점에 다시 비교해 future update drift를 검증한다.

이번 B0에서는 `max_states=8` 실행이 한 Gaussian에서 capacity overflow를 냈다. 최종 run은
`max_states=16`으로 수행했다. 최대 사용 slot 수는 15였다.

## 6. B0: direct binary DC-only 결과

Output:

```text
outputs/escd_direct_binary_state_b0_dc_u120_max16_20260821
```

조건:

| 항목 | 값 |
|---|---:|
| frames | 304 |
| cue mode | binary, threshold 0.5 |
| evidence count | capped, mass saturation 1.0 |
| emission reliability `eta` | 0.9 |
| transition prior `p01/p10` | 0.01 / 0.01 |
| initial active probability | 0.5 |
| open/close threshold | 0.6 / 0.4 |
| thaw parameters | `dc` |
| updates per frame | 120 |
| max states | 16 |
| seed | 0 |

### 6.1 Mask metric

| Metric | 값 |
|---|---:|
| mean-frame mIoU | 0.6063232380 |
| mean-frame F1 | 0.7282668362 |
| precision | 0.7162350284 |
| recall | 0.8816540018 |
| aggregate IoU | 0.6534147462 |
| aggregate F1 | 0.7903821443 |

Segment별 mean-frame 결과:

| Segment | mIoU | F1 |
|---|---:|---:|
| scene_change1 | 0.5906133988 | 0.6846129846 |
| scene_change2 | 0.6408789989 | 0.7686377095 |
| scene_change3 | 0.5863102437 | 0.7277767893 |

Open frame pre/post optimization 평균:

| 상태 | mIoU | F1 |
|---|---:|---:|
| pre-optimization | 0.5887763065 | 0.7137778974 |
| post-optimization | 0.6063232380 | 0.7282668362 |
| delta | +0.0175469316 | +0.0144889388 |

### 6.2 Lifecycle event와 integrity

| 항목 | 값 |
|---|---:|
| OPEN | 299,673 |
| CLOSE | 178,355 |
| REOPEN | 78,731 |
| KEEP | 8,322,471 |
| UNCERTAIN | 9,814,881 |
| final active Gaussian | 121,318 |
| event count | 478,028 |
| same-scene repeated transition extra events | 170,574 |

Integrity audit:

| 항목 | 값 |
|---|---:|
| base tensor max drift | 0.0 |
| closed slot max drift | 0.0 |
| frame 첫 optimizer step inactive gradient violations | 0 |
| active-to-active false split | 0 |
| reused slot violations | 0 |
| zero-init violations | 0 |
| reopen allocation contract violations | 0 |

B0/B1 실행 직후 reporting schema만 한 번 명확화했다. 실행 당시 각 frame CSV의 아직
검증 전 closed-slot audit가 임시로 `passed=true, max_abs=0`으로 적혔지만, 실제 판정에는
사용하지 않았고 모든 future update가 끝난 뒤 final summary에서 전체 178,355 closed pair의
parameter와 optimizer state를 다시 비교했다. 현재 runner는 오해를 막기 위해 frame 중간값을
`passed=null, max_abs=null`로 기록하고 final audit에서만 0을 확정한다. 수치 연산 경로에는
변경이 없다.

Detector-only direct binary run과 B0 DC run의 lifecycle event 구조는 동일했다.
Detector-only는 optimization을 끄기 때문에 mask metric 해석 대상이 아니고, lifecycle 구조
검증용으로만 사용한다.

### 6.3 Runtime

| 항목 | 값 |
|---|---:|
| runtime | 498.664 s |
| peak CUDA memory | 3,704,640,000 bytes |

## 7. Beam-2 BOCD와의 비교

Beam-2는 BOCD reset-branch 보존 가설을 검증하기 위한 별도 라인이다. Direct binary B0와
입력 cue/evidence/update schedule은 맞추되 detector 구조는 다르다.

| 방법 | Thaw | mean-frame mIoU | mean-frame F1 | aggregate IoU | aggregate F1 |
|---|---|---:|---:|---:|---:|
| direct binary B0 | DC | 0.6063232380 | 0.7282668362 | 0.6534147462 | 0.7903821443 |
| beam-2 | DC | 0.6046644784 | 0.7281445024 | 0.6500859038 | 0.7879418912 |
| beam-2 | all geometry | 0.6139883607 | 0.7322514135 | 0.6517917954 | 0.7891936468 |

해석:

- Direct binary B0 DC는 beam-2 DC와 거의 같은 2D current-mask 성능을 냈다.
- Beam-2 all-geometry는 mean-frame mIoU/F1을 조금 더 높였지만, geometry 자유도가
  lifecycle detector의 정확성을 직접 증명하지는 않는다.
- 두 계열 모두 MAP-reset `CLOSE=0` failure와 달리 `CLOSE/REOPEN`을 만들 수 있음을 보였다.

## 8. Detector failure: heavy chattering

B0의 가장 큰 문제는 lifecycle이 너무 많이 흔들린다는 점이다.

```text
OPEN 299,673 / CLOSE 178,355 / REOPEN 78,731 / UNCERTAIN 9,814,881
```

이 수치는 “전환부에서 필요한 Gaussian만 닫고 다시 열었다”가 아니라, 많은 Gaussian이
관측/비관측, cue noise, threshold 주변 posterior 변화에 민감하게 반응했음을 뜻한다.

따라서 다음 문장을 구분해야 한다.

- 맞는 말: direct binary controller는 slot을 닫고 새 slot을 열 수 있다.
- 맞는 말: closed slot 보존, zero-init, active-only optimizer invariant는 유지됐다.
- 틀린 말: direct binary detector가 scene transition을 깨끗하게 검출했다.

## 9. Representation performance 해석

B0 DC-only의 mIoU `0.6063`은 MAP-reset DC-only `0.4264`보다 크게 높다. 직접 원인은
`CLOSE/REOPEN`이 실제로 발생해 과거 active state가 current output에 계속 남는 문제를
줄였기 때문이다.

하지만 performance만으로 detector correctness를 판단하면 안 된다.

1. 2D mask metric은 현재 render 결과를 평가한다.
2. Heavy chattering이 있어도 우연히 현재 GT mask와 맞는 방향으로 active set이 정리될 수 있다.
3. Geometry를 thaw하면 하나의 lifecycle failure도 morphing으로 가려질 수 있다.
4. 따라서 lifecycle event quality와 representation metric은 분리해서 봐야 한다.

이번 ablation의 성과는 representation score보다 integrity가 깨지지 않는 상태에서
`CLOSE/REOPEN`이 causal loop 안에서 발생했다는 점이다.

## 10. B1: direct binary all-geometry 결과

Output:

```text
outputs/escd_direct_binary_state_b1_all_geometry_u120_max16_20260821
```

B0와 detector, cue, evidence, transition prior, hysteresis, pose, seed, update 수는 같고
thaw parameter만 `dc,xyz,opacity,scaling,rotation`으로 바꿨다.

| Metric | B0 DC-only | B1 all-geometry | B1 - B0 |
|---|---:|---:|---:|
| mean-frame mIoU | 0.6063232380 | 0.6057476608 | -0.0005755772 |
| mean-frame F1 | 0.7282668362 | 0.7258112154 | -0.0024556207 |
| precision | 0.7162350284 | 0.6787958640 | -0.0374391644 |
| recall | 0.8816540018 | 0.9283325317 | +0.0466785299 |
| aggregate IoU | 0.6534147462 | 0.6449960185 | -0.0084187277 |
| aggregate F1 | 0.7903821443 | 0.7841915862 | -0.0061905580 |

Segment mean-frame mIoU는 B0/B1 순서로 scene change 1에서 `0.5906/0.5728`,
scene change 2에서 `0.6409/0.6194`, scene change 3에서 `0.5863/0.6221`이었다.
즉 all-geometry는 마지막 segment에서는 유리했지만 전체 평균에서는 DC-only를 넘지 못했다.

B0와 B1의 478,028개 lifecycle event를 구조적으로 비교한 결과 Gaussian index, timestamp,
action, old/new label, old/new slot이 전부 같았다. Frame별 observed/OPEN/CLOSE/KEEP/NONE/
UNCERTAIN/REOPEN/active count도 모두 같았다. Posterior float의 최대 차이는 CUDA alpha-T
atomic 연산의 실행 순서 차이 범위인 `1.73e-6`이었다. 따라서 B0/B1 성능 차이는 detector
event 차이가 아니라 state-local representation 자유도 차이로 해석한다.

| 항목 | B1 값 |
|---|---:|
| OPEN / CLOSE / REOPEN | 299,673 / 178,355 / 78,731 |
| KEEP / UNCERTAIN | 8,322,471 / 9,814,881 |
| pre/post mean-frame mIoU | 0.5815594332 / 0.6057476608 |
| OPEN-frame mean IoU delta | +0.0241882276 |
| runtime | 869.415 s |
| peak CUDA memory | 7,213,326,848 bytes |
| base/closed-slot max drift | 0.0 / 0.0 |
| false split/reused slot/zero-init violation | 0 / 0 / 0 |

B1의 valid slot에서 `xyz_delta` mean absolute 값은 `0.01205`, opacity delta는 `3.11384`,
scaling delta는 `0.51045`, rotation delta는 `0.08090`이었다. Beam-2 all-geometry의 같은
통계는 각각 `0.01591`, `4.11275`, `0.67428`, `0.10342`였다. Direct binary B1은 더 많은
짧은 slot을 만들었기 때문에 slot당 geometry 변화량이 전반적으로 작았다.

전환 timestamp 95의 첫 frame에서 B0는 pre/post IoU가 `0.0000/0.0017`이었지만 B1은
`0.0000/0.1151`이었다. Geometry는 zero-init 새 slot의 초기 표현 지연을 줄일 수 있었다.
그러나 전체적으로 precision이 하락하고 mean mIoU/F1도 소폭 하락했으므로, 더 큰 capacity가
항상 더 좋은 current mask를 만든다고 결론낼 수 없다.

## 11. Soft S0/S1

### 11.1 실행 전 detector-only capacity 확인

Soft cue detector-only run은 `max_states=16`에서 완료됐고 실제 최대 사용 state 수는 9였다.
이 run은 `OPEN=204,651`, `CLOSE=80,828`, `REOPEN=27,614`로 binary cue보다 event 수는
줄었지만 여전히 transition boundary에만 국한된 detector는 아니었다.

### 11.2 S0/S1 결과

Output:

```text
outputs/escd_direct_binary_state_s0_soft_dc_u120_max16_20260821
outputs/escd_direct_binary_state_s1_soft_all_geometry_u120_max16_20260821
```

| 항목 | S0 soft DC-only | S1 soft all-geometry |
|---|---:|---:|
| mean-frame mIoU | 0.5975868344 | 0.6091270941 |
| mean-frame F1 | 0.7220999164 | 0.7287675917 |
| precision | 0.7102170147 | 0.6834630966 |
| recall | 0.8718604713 | 0.9294019395 |
| aggregate IoU | 0.6430897719 | 0.6497314048 |
| aggregate F1 | 0.7827810542 | 0.7876814406 |
| pre/post mean-frame mIoU | 0.5802627388 / 0.5975868344 | 0.5825083195 / 0.6091270941 |
| OPEN-frame mean IoU delta | +0.0173240956 | +0.0266187746 |
| OPEN / CLOSE / REOPEN | 204,651 / 80,828 / 27,614 | 204,651 / 80,828 / 27,614 |
| KEEP / UNCERTAIN | 8,171,456 / 13,628,597 | 8,171,456 / 13,628,596 |
| final active GS | 123,823 | 123,823 |
| same-scene repeated transition extra events | 58,426 | 58,426 |
| runtime | 474.910 s | 868.431 s |
| peak CUDA memory | 3,701,874,688 | 7,208,166,912 |
| base/closed drift | 0 / 0 | 0 / 0 |
| false split/reuse/zero-init violation | 0 / 0 / 0 | 0 / 0 / 0 |

S0와 S1의 285,479개 OPEN/CLOSE event는 Gaussian, timestamp, action, slot 기준으로
전부 같았다. Frame action count에는 CUDA atomic 순서로 threshold 근처 한 row가
`NONE/UNCERTAIN` 중 다르게 분류된 1건이 있었지만 lifespan event 차이는 없었다.
따라서 S0 대비 S1 mean-frame mIoU `+0.01154`, F1 `+0.00667`은 representation 효과다.

Soft S1은 같은 soft cue의 Beam-2 all-geometry(`mIoU=0.60575`, `F1=0.72431`)보다
높았지만, same-scene repeated transition extra event는 Beam-2의 `1,441`보다 훨씬 많은
`58,426`이었다. 즉 mask score 향상이 더 안정적인 lifecycle detector를 뜻하지 않는다.

## 12. Posterior와 전체 비교표

아래 posterior 수치는 각 frame observed-row quantile을 구한 뒤 304 frame에서 median을
취한 diagnostic이다.

| Condition | p_active q05/median/q95 | p_flip q05/median/q95 | p01 mean/q95 | p10 mean/q95 |
|---|---|---|---|---|
| B0/B1 binary | 0.001261 / 0.074408 / 0.979107 | 0.001134 / 0.008436 / 0.010215 | 0.005322 / 0.009414 | 0.002157 / 0.006991 |
| S0/S1 soft | 0.003809 / 0.144639 / 0.969096 | 0.002341 / 0.009377 / 0.010050 | 0.005660 / 0.009181 | 0.002463 / 0.007018 |

| Condition | mIoU | F1 | OPEN | CLOSE | REOPEN | same-scene repeat | runtime(s) | peak CUDA(GB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 binary DC | 0.606323 | 0.728267 | 299,673 | 178,355 | 78,731 | 170,574 | 498.7 | 3.70 |
| B1 binary all-geometry | 0.605748 | 0.725811 | 299,673 | 178,355 | 78,731 | 170,574 | 869.4 | 7.21 |
| S0 soft DC | 0.597587 | 0.722100 | 204,651 | 80,828 | 27,614 | 58,426 | 474.9 | 3.70 |
| S1 soft all-geometry | 0.609127 | 0.728768 | 204,651 | 80,828 | 27,614 | 58,426 | 868.4 | 7.21 |
| Beam-2 binary all-geometry | 0.613988 | 0.732251 | 153,458 | 39,852 | 10,041 | 19,305 | 715.4 | 4.45 |
| Beam-2 soft all-geometry | 0.605748 | 0.724312 | 89,434 | 9,746 | 783 | 1,441 | 671.4 | 4.44 |

## 13. 테스트와 검증 범위

관련 테스트 파일:

```text
tests/temporal/test_binary_state_filter.py
tests/temporal/test_binary_state_lifespan_controller.py
tests/experiments/test_online_binary_state_lifespan_thaw.py
tests/experiments/test_plot_binary_state_lifespan_results.py
tests/temporal/test_change_evidence_cuda.py
```

주요 검증 항목:

- log-domain two-state joint가 수동 계산과 일치한다.
- vector chunk update가 지정 row만 바꾸고 duplicate index를 거부한다.
- unobserved row는 filter state와 lifecycle state를 bitwise 보존한다.
- temporal DC/xyz/opacity/scaling/rotation과 lifecycle을 크게 바꿔도 immutable-reference
  alpha-T evidence 네 tensor가 CUDA에서 bitwise 동일하다.
- synthetic inactive/active/inactive/active sequence가 `OPEN -> CLOSE -> OPEN`을 만든다.
- stable active sequence는 한 번만 `OPEN`하고 이후 같은 slot에서 `KEEP`한다.
- ambiguous posterior는 불필요한 flip 없이 `UNCERTAIN`으로 남는다.
- reopen은 closed slot을 재사용하지 않고 새 slot을 zero-init한다.
- real masked optimizer factory는 `MaskedRowSlotAdam`을 사용한다.
- output writer는 `summary.json`, `frame_metrics.csv`, `lifecycle_events.jsonl`,
  `per_frame_binary_state_stats.npz`, `checkpoint.pt`를 생성한다.
- final full run summary에서 base drift와 모든 closed parameter/optimizer state drift가 0이고,
  false split, slot reuse, zero-init violation도 0이다.
- inactive gradient는 각 frame의 첫 optimizer step에서 전 pair를 검사해 0임을 확인했다.
  동일 frame의 나머지 step은 active pair 집합과 render graph가 같으며, 별도 masked-Adam
  regression test가 momentum을 포함한 inactive pair의 exact zero drift를 검증한다. 따라서 이
  값은 all-step gradient scan으로 과장하지 않고 `first_optimizer_step_per_frame`로 기록한다.

최종 검증 결과:

```text
compileall: PASS
pytest tests: 302 passed in 3.92s
```

## 14. 실행 명령과 출력

B0/B1은 `--bayes-cue-mode binary`, S0/S1은 `soft`를 사용했다. 공통 옵션은 다음과 같다.

```bash
COMMON="--source-path data/Instance_1/scene_change1_2_3 \
  --evidence-count-mode capped --evidence-mass-saturation 1.0 \
  --state-emission-reliability 0.9 \
  --inactive-to-active-prior 0.01 --active-to-inactive-prior 0.01 \
  --initial-active-probability 0.5 \
  --open-probability 0.6 --close-probability 0.4 \
  --max-states 16 --updates-per-frame 120 --seed 0"

PYTHONPATH=. conda run -n oscd python -m \
  experiments.run_online_binary_state_lifespan_thaw \
  $COMMON --bayes-cue-mode binary --thaw-parameters dc \
  --output-dir outputs/escd_direct_binary_state_b0_dc_u120_max16_20260821

PYTHONPATH=. conda run -n oscd python -m \
  experiments.run_online_binary_state_lifespan_thaw \
  $COMMON --bayes-cue-mode binary \
  --thaw-parameters dc,xyz,opacity,scaling,rotation \
  --output-dir outputs/escd_direct_binary_state_b1_all_geometry_u120_max16_20260821

PYTHONPATH=. conda run -n oscd python -m \
  experiments.run_online_binary_state_lifespan_thaw \
  $COMMON --bayes-cue-mode soft --bayes-cue-scale 1.0 \
  --thaw-parameters dc \
  --output-dir outputs/escd_direct_binary_state_s0_soft_dc_u120_max16_20260821

PYTHONPATH=. conda run -n oscd python -m \
  experiments.run_online_binary_state_lifespan_thaw \
  $COMMON --bayes-cue-mode soft --bayes-cue-scale 1.0 \
  --thaw-parameters dc,xyz,opacity,scaling,rotation \
  --output-dir outputs/escd_direct_binary_state_s1_soft_all_geometry_u120_max16_20260821
```

각 run은 다음 machine-readable output을 저장한다.

```text
summary.json
frame_metrics.csv
lifecycle_events.jsonl
per_frame_binary_state_stats.npz
checkpoint.pt
```

Plot utility는 run별로 다음 다섯 파일을 만든다.

```text
binary_state_filter_metrics.png
binary_state_lifecycle_counts.png
binary_state_belief_histogram.png
binary_state_transition_probability_timeline.png
pre_vs_post_open_render.png
```

B0/B1 또는 S0/S1 비교 시 `dc_only_vs_all_geometry.png`와 `comparison.json/.md`도
생성한다. Manual boundary `[95,199]`는 plot vertical line과 post-inference segment
diagnostic에만 사용했다.

검증 명령:

```bash
PYTHONPATH=. conda run -n oscd python -m compileall -q \
  experiments poses temporal tests

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. \
  conda run -n oscd pytest -q tests
```

## 15. 최종 판단과 다음 단계

Direct binary filtering은 run-start aggregation 없이도 repeated `OPEN/CLOSE/REOPEN`을
직접 만들었고, lifecycle/slot/optimizer 계약도 지켰다. 그러나 binary cue에서는
same-scene extra transition이 170,574건으로 너무 많다. Soft cue는 이를 58,426건으로
줄였고 all-geometry S1은 S0보다 성능이 높았지만, Beam-2 soft의 1,441건보다 여전히 훨씬
불안정하다.

따라서 다음 단계는 representation capacity를 더 키우는 것이 아니라 direct filter의
emission/transition calibration과 spatial/visibility consistency를 검증하는 것이다. 그 실험은
이번 strict ablation과 섞지 말고 별도 branch에서 수행해야 한다. 현재 결과로는 direct binary
filter가 BOCD를 대체한다고 결론내리지 않는다.
