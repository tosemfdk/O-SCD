# ESCD Bayesian lifespan + active geometry 실험 결과

## 1. 목적

이 문서는 causal Bayesian lifespan 구현 이후 수행한 ESCD 실험을 정리한다. 핵심 질문은 다음 두 가지다.

1. `ref -> scene_change_k` 독립 스트림에서 자동 lifespan과 state-local geometry가 기존 표현을 대체할 수 있는가?
2. `ref -> SC1 -> SC2 -> SC3` 연속 스트림에서 Gaussian이 실제로 `OPEN/CLOSE/REOPEN`되는가?

모든 실험은 fixed pose, cached O-SCD pixel+SAM2.1 cue, binary cue threshold `0.5`, capped alpha-T evidence, `MAPResetBernoulliFilter`, seed `0`, 이미지당 120 updates를 사용했다. GT와 수동 경계 `[95, 199]`는 inference 이후 평가/진단에만 사용했다.

## 2. Locked binary semantics

Gaussian `i`의 상태는 reference와의 차이만 나타낸다.

- `z_i,t=1`: 현재 reference와 다름
- `z_i,t=0`: 현재 reference와 다르지 않음

따라서 `change A -> change B`가 모두 reference와 다르면 `active -> active = KEEP`이다. BOCD가 내부 분포 reset을 검출해도 representation slot은 나누지 않는다.

## 3. 독립 `ref -> SC` 결과

각 장면마다 model, BOCD, optimizer를 새로 초기화했다.

### 프레임 평균 IoU/F1

| 장면 | DC-only IoU/F1 | All-geometry IoU/F1 | Geometry minus DC |
| --- | --- | --- | --- |
| SC1 | 0.5914 / 0.6853 | 0.5735 / 0.6688 | -0.0179 / -0.0165 |
| SC2 | 0.6591 / 0.7857 | 0.6206 / 0.7517 | -0.0384 / -0.0340 |
| SC3 | 0.6148 / 0.7525 | 0.6216 / 0.7531 | +0.0068 / +0.0007 |

All-geometry는 세 장면 모두 recall을 높였지만 precision을 낮췄다. SC1/SC2에서는 추가 FP로 성능이 하락했고 SC3에서만 미세하게 개선됐다. 따라서 독립 스트림에서는 DC-only가 더 안정적이며 geometry improvement를 일반화할 수 없다.

독립 실행의 OPEN/final-active 수는 SC1 `58,545`, SC2 `91,759`, SC3 `86,654`였다. Static changed segment이므로 CLOSE/REOPEN은 없었다.

## 4. 연속 `ref -> SC1 -> SC2 -> SC3` 결과

304프레임을 하나의 global timestamp로 처리했다. SC1/SC2/SC3 전환에서도 model, BOCD, temporal slot, optimizer를 초기화하지 않았다.

### 프레임 평균 IoU/F1

| 방법 | 전체 | SC1 | SC2 | SC3 |
| --- | --- | --- | --- | --- |
| DC-only IoU/F1 | 0.4264 / 0.5472 | 0.5914 / 0.6853 | 0.4500 / 0.5933 | 0.2538 / 0.3766 |
| All-geometry IoU/F1 | 0.6208 / 0.7372 | 0.5730 / 0.6684 | 0.6337 / 0.7606 | 0.6513 / 0.7764 |

Aggregate IoU/F1은 DC-only `0.4460/0.6169`, all-geometry `0.6624/0.7969`였다.

### Lifecycle 결과

- 전체 OPEN: `81,268`
- CLOSE: `0`
- REOPEN: `0`
- active-to-active false split: `0`
- 실제 사용된 slot: 모든 열린 Gaussian에서 slot 0 하나뿐

구간별 새 OPEN 수는 다음과 같다.

- SC1: `58,545`
- SC2에서 추가: `17,127`
- SC3에서 추가: `5,596`

전환 첫 프레임에서는 SC1->SC2에 `1,290`, SC2->SC3에 `21`개가 새로 OPEN됐다. 이 수에는 실제 새 변화와 새로운 viewpoint에서 처음 관측된 reference Gaussian이 함께 포함될 수 있다.

## 5. 왜 CLOSE 없이 all-geometry mIoU가 유지됐는가

mIoU는 lifecycle 정확도가 아니라 최종 2D rendered mask를 평가한다.

DC-only는 OPEN slot의 DC만 계속 학습한다. 이전 상태를 논리적으로 닫지 못한 채 같은 slot을 적응시키므로 SC2와 특히 SC3에서 성능이 무너졌다.

All-geometry는 동일한 OPEN slot의 state-local `xyz/DC/opacity/scaling/rotation`을 계속 갱신한다. 논리적으로 OPEN인 Gaussian도 opacity를 낮추거나 scale을 줄이고, 위치/회전을 바꾸고, DC를 약화해 현재 render에서 기여를 줄일 수 있다. 따라서 하나의 slot이 장면에 맞춰 morphing하면서 lifecycle 실패를 시각적으로 가렸다.

이 결과는 lifespan separation 성공이 아니다. Base Gaussian은 bitwise frozen이고, active slot delta가 현재 장면에 맞게 변형된 것이다.

## 6. CLOSE=0의 원인: BOCD 원리가 아니라 현재 근사/제어 결합

정통 BOCD에서는 기존 run이 설명하기 어려운 observation이 들어오면 growth predictive likelihood가 낮아지고 `P(r_t=0)`이 증가한다. 오래된 evidence가 존재한다는 사실 자체가 reset을 막는 것은 아니다.

이번 full run은 exact BOCD가 아니라 `MAPResetBernoulliFilter`를 사용했다. 현재 구현은 다음 두 점수를 비교한다.

\[
\log p_{continue}=\log p(s,f\mid a,b)+\log(1-H)
\]

\[
\log p_{reset}=\log p(s,f\mid a_0,b_0)+\log H
\]

설정은 `H=1/100=0.01`, reset threshold `0.5`였다. 따라서 reset posterior가 0.5를 넘으려면 새 observation이 prior에서 현재 run보다 대략 99배 더 그럴듯해야 한다. Capped evidence는 한 프레임의 footprint mass를 최대 약 1 pseudo-observation으로 정규화하므로 이 조건을 만족시키기 어렵다.

더 중요한 차이는 MAP 근사에서 reset 후보 branch를 보존하지 않는다는 점이다. 첫 contradictory cue에서 threshold를 넘지 못하면 해당 cue를 기존 `(a,b)`에 합친다. 이후 같은 cue는 기존 run에서 덜 이상해져 changepoint 기회가 더 약해질 수 있다. Exact BOCD는 낮은 확률의 reset/run-length branch를 유지하므로 반복 관측 후 그 branch가 커질 가능성이 있지만, MAP-reset은 이를 버린다.

Controller도 established label 변경에 BOCD reset을 요구한다. Reset 없이 posterior label만 active에서 inactive로 반전되면 `CLOSE`하지 않고 `UNCERTAIN`으로 둔다. 따라서 `MAP branch 폐기 + capped evidence + reset-gated controller` 조합이 CLOSE=0 failure mode를 만들었다.

Exact BOCD가 반드시 문제를 해결한다고 아직 주장할 수 없다. 동일 evidence에서 exact run-length posterior가 전환부에 어떻게 반응하는지 별도로 검증해야 한다.

## 7. 시각화

생성 output은 Git에 commit하지 않는다.

독립 비교:

```text
outputs/escd_ref_to_scn_bayesian_u120_20260819/
```

연속 실행:

```text
outputs/escd_ref_sc1_sc2_sc3_continuous_bayesian_u120_20260820/
```

All-geometry causal raw-render GIF:

```text
all_geometry_rawgif_visuals/scene_change1/scene_change1_confusion.gif
all_geometry_rawgif_visuals/scene_change2/scene_change2_confusion.gif
all_geometry_rawgif_visuals/scene_change3/scene_change3_confusion.gif
```

패널 순서는 `RGB | GT | Raw R_change | Rendered mask (>0.5) | Confusion`이다. Confusion 색은 TP=green, FP=pink, FN=blue, TN=black이다. Raw render는 각 causal optimizer step이 끝난 직후 저장하므로 final checkpoint로 과거를 재렌더한 결과가 아니다.

Raw GIF export 재실행은 같은 seed/config에서도 CUDA rasterization의 작은 비결정성으로 mean-frame IoU `0.6192`를 기록했다. 주 비교 run의 값은 `0.6208`이다.

## 8. 실행 비용과 검증

연속 304-frame 실행의 요약:

| 조건 | runtime | peak CUDA memory |
| --- | ---: | ---: |
| detector-only | 166.6 s | 1.87 GiB |
| DC-only | 715.1 s | 2.35 GiB |
| all-geometry | 1041.9 s | 4.05 GiB |
| all-geometry raw-GIF 단독 재실행 | 726.7 s | 4.05 GiB |

DC와 첫 all-geometry 수치는 GPU 병렬 실행의 contention을 포함하므로 단독 throughput benchmark로 해석하지 않는다.

최종 검증:

```text
compileall: PASS
pytest: 228 passed
git diff --check: PASS
base tensor drift: 0
inactive temporal parameter drift: 0
```

## 9. 다음 연구 과제

1. 작은 Gaussian subset 또는 메모리 허용 범위에서 exact BOCD와 MAP-reset을 동일 evidence sequence로 비교한다.
2. 전환부에서 `log p_continue`, `log p_reset`, Bayes factor, `P(r_t=0)`, candidate label, UNCERTAIN 수를 저장한다.
3. MAP 대안이 필요하면 reset branch를 즉시 폐기하지 않는 two-hypothesis/beam 형태를 검토한다.
4. Capped evidence 한 프레임의 effective sample size와 hazard/threshold의 일관성을 보정한다.
5. Per-Gaussian binary GT 또는 proxy를 정의해 false OPEN/CLOSE와 delay를 평가한다.
6. Geometry가 lifecycle 실패를 가리는 경우와 실제 current-state representation을 개선하는 경우를 분리해 평가한다.
