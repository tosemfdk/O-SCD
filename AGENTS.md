# Persistent Project Context

이 저장소에서 작업하는 모든 Codex 에이전트는 사용자가 달리 명시하지 않는 한 아래 내용을 프로젝트의 기본 연구 방향과 문제 정의로 간주한다.

## 우리가 푸는 문제

기존 online Scene Change Detection(SCD)은 고정된 기준 장면 `R_ref`와 순차적으로 들어오는 이미지 사이의 변화를 검출한다. O-SCD는 이전 프레임에서 얻은 변화 단서를 `R_change`에 누적하여 멀티뷰 일관성을 확보하지만, 하나의 관측 구간 동안 post-change scene이 고정되어 있다고 가정한다.

이 프로젝트가 다루는 **SCD in an on-the-fly evolving scene**은 그 가정을 제거한다. 스트림을 촬영하는 동안 동일한 영역의 상태가 다음처럼 여러 번 바뀔 수 있다.

```text
의자 원래 위치 -> 의자 이동 -> 의자 제거 또는 재이동
```

기존 O-SCD의 단순 누적 방식을 그대로 사용하면 과거 상태를 설명하는 change cue와 현재 상태를 설명하는 cue가 동일한 `R_change`에 함께 남는다. 그 결과:

- 이미 사라진 변화가 계속 검출될 수 있다.
- 서로 충돌하는 증거가 평균화될 수 있다.
- 현재 장면 상태를 정확히 표현하지 못할 수 있다.

## 목표 시스템

입력은 고정된 reference 3D Gaussian Splatting 표현 `R_ref`와 온라인 이미지 스트림 `I_1, I_2, ..., I_t`이다. 시스템은 미래 프레임을 사용하지 않고 매 시점 `t`마다 다음을 수행해야 한다.

1. 현재 관측이 reference와 어디에서 다른지 검출한다.
2. 동일한 Gaussian 영역에서 과거 change evidence와 현재 evidence가 충돌하는지 판단한다.
3. 실제 장면의 변화 상태가 다시 바뀌었다면 change point를 검출한다.
4. 이전 상태를 지지하던 Gaussian의 lifespan을 종료한다.
5. 새로운 상태를 위한 lifespan을 시작한다.
6. 과거 변화의 합집합이 아니라 **현재 시점에 유효한 변화만** change mask로 출력한다.

## 시간적 변화 표현

Gaussian마다 하나의 누적 change score만 유지하지 않는다. 각 Gaussian `g_i`는 개념적으로 다음과 같은 시간적 상태 또는 상태 이력을 가진다.

```text
g_i: {(change state, start time, end time, confidence)}
```

기존 `R_change`의 persistent memory와 멀티뷰 융합 장점은 보존하되, 오래되었거나 현재 관측과 충돌하는 evidence를 자동으로 닫거나 교체할 수 있어야 한다.

## 핵심 목적과 제약

- **Online causality:** 시점 `t`의 판단에는 `I_1 ... I_t`만 사용하며 미래 프레임을 사용하지 않는다.
- **Current-state validity:** 출력 mask는 지금 유효한 변화만 나타내야 한다.
- **Temporal consistency:** 3D 변화 표현은 멀티뷰 일관성을 유지하면서 상태의 시작과 종료를 추적해야 한다.
- **Outdated evidence handling:** 과거 변화 evidence와 현재 evidence를 구분하고, 오래된 evidence가 현재 결과를 오염시키지 않도록 해야 한다.
- **Repeated evolution:** 같은 Gaussian 영역이 changed, reverted, removed, 또는 changed-again 상태로 반복 전이할 수 있음을 전제로 한다.

## 한 문장 요약

> Detect not only where the scene differs from the reference, but also when that difference itself changes.

## 현재 구현 및 연구 단계

현재 저장소 checkpoint는 기존 oracle-boundary lifespan/E3 control과 causal Bayesian lifespan + active-only state geometry 구현을 함께 포함한다.

- Lifespan separation: state별 DC와 half-open interval을 구현하고 naive persistent O-SCD보다 과거 state 보존이 개선되는 것을 확인했다.
- Corrected cue experiment: O-SCD pixel + SAM2.1 cue, fixed pose, 304 frames, 이미지당 120 updates로 DC-only lifespan을 검증했다.
- E3 geometry: Gaussian index와 topology는 고정한 채 state별 xyz/DC/opacity/scale/rotation delta와 S0 soft anchor를 학습한다.
- E3 결과: geometry는 SSF loss를 낮췄지만 DC-only보다 overall mIoU/F1이 소폭 낮았다. 상세 내용은 [`docs/instance1-state-geometry-lifespan-comparison-ko.md`](docs/instance1-state-geometry-lifespan-comparison-ko.md)에 기록한다.
- Bayesian lifespan: immutable reference의 lifespan-agnostic alpha-T evidence, bounded Beta-Bernoulli BOCD와 명시적 MAP-reset 근사, binary OPEN/KEEP/CLOSE/REOPEN lifecycle, row-slot masked Adam, causal frame-major runner를 구현한다. `active -> active`는 항상 같은 slot의 `KEEP`이다.
- Bayesian ESCD 평가: 독립 `ref -> SC1/SC2/SC3`와 상태를 유지한 연속 `ref -> SC1 -> SC2 -> SC3` 304-frame 실험을 완료했다. 연속 MAP-reset run은 OPEN `81,268`, CLOSE/REOPEN `0`으로 lifespan separation에 실패했다. DC-only mean-frame mIoU는 `0.4264`, all-geometry는 `0.6208`이었지만 geometry가 동일 OPEN slot을 morphing해 failure를 가린 결과이므로 lifespan 개선으로 해석하지 않는다. 상세 내용은 [`docs/escd-bayesian-lifespan-experiment-results-ko.md`](docs/escd-bayesian-lifespan-experiment-results-ko.md)에 기록한다.
- BOCD 진단 확장: full posterior/Adams--MacKay lineage diagnostic과 protected reset-candidate Beam-2 ablation을 보존한다. Beam-2 binary all-geometry는 mean-frame mIoU/F1 `0.6140/0.7323`, OPEN/CLOSE/REOPEN `153,458/39,852/10,041`을 기록했다.
- Direct binary-state ablation: BOCD run length 없이 `P(z_t=active)`를 직접 추론하는 two-state Bayesian filter/controller, causal B0/B1/S0/S1 runner, pre/post-OPEN metric, exact closed-pair audit와 diagnostic plot을 구현했다. Immutable-reference alpha-T detector는 temporal slot parameter/lifecycle과 bitwise 독립이다.
- PASLCD view-consistency diagnostic: 20 scenes/500 frames에서 direct binary PASLCD replay를 실행해 baseline lifecycle structure가 전 scene에서 정확히 일치하고, base tensor drift가 0이며, same-scene repeated transition event가 `556,469`임을 확인했다. Event의 transition branch odds는 최대 `0.090909`였지만 marginal `p_active` threshold만으로 lifecycle이 flip했고, Gaussian `q` variance와 repeated cohort의 상관은 `0.2949`였다. 따라서 다음 단계는 state belief와 transition decision을 분리한 consecutive-view confirmation D2다. 상세 내용은 [`docs/paslcd-view-consistency-diagnostic-ko.md`](docs/paslcd-view-consistency-diagnostic-ko.md)에 기록한다.
- Direct binary ESCD 결과: binary B0 DC-only/B1 all-geometry mean-frame mIoU는 `0.6063/0.6057`, soft S0/S1은 `0.5976/0.6091`이었다. Lifespan/optimizer invariant는 유지됐지만 binary/soft same-scene repeated extra transition이 `170,574/58,426`으로 많다. 현재 핵심 failure는 lifecycle 구현이 아니라 direct detector의 chattering이며, 다음 단계는 별도 실험에서 emission/transition calibration과 spatial/visibility consistency를 검증하는 것이다. 상세 내용은 [`docs/direct-binary-state-lifespan-ablation-ko.md`](docs/direct-binary-state-lifespan-ablation-ko.md)에 기록한다.
- PASLCD 비교: 20 scenes/500 frames에서 direct binary all-geometry는 O-SCD online 대비 u16에서 mIoU/F1 `+0.0190/+0.0175`, u120에서 `+0.0260/+0.0150`이었지만 O-SCD refined보다는 낮았다. Single-state PASLCD에서도 OPEN/CLOSE/REOPEN `922,714/446,622/109,847`이 발생해 detector chattering이 재확인됐다. 상세 내용은 [`docs/paslcd-direct-binary-state-comparison-ko.md`](docs/paslcd-direct-binary-state-comparison-ko.md)에 기록한다.
- 이 checkpoint에는 MCMC, SGLD, relocation, densification, pruning을 포함하지 않는다.
