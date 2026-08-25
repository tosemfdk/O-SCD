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
- View-consistent direct binary lifespan: PASLCD D1에서 marginal `p_active`와 lifecycle mutation의 결합이 same-scene chattering의 핵심임을 확인한 뒤, committed-state `BF_open/BF_close`를 연속 observed view에서 확인하는 controller를 추가했다. `K=3, BF>=3`은 PASLCD repeated extras를 `556,469 -> 43,153`, REOPEN을 `109,847 -> 8,255`로 줄였고, all-geometry mIoU `0.4973`으로 O-SCD online `0.4887`을 유지했다. ESCD `ref -> SC1 -> SC2 -> SC3`에서는 B0/B1 mIoU `0.5981/0.5996`, OPEN/CLOSE/REOPEN `123,476/64,619/22,212`를 기록해 실제 전환 반응을 유지했지만 Beam-2 all-geometry `0.6140`보다 낮다. 상세 내용은 [`docs/view-consistent-binary-lifespan-confirmation-ko.md`](docs/view-consistent-binary-lifespan-confirmation-ko.md)에 기록한다.
- Active-visible optimizer isolation: temporal optimizer selection을 전체 OPEN pair가 아니라 `OPEN AND current temporal render radius > 0`인 row-slot으로 제한하고 매 update마다 visibility를 다시 계산한다. Off-view OPEN pair는 parameter와 Adam moment가 정확히 보존된다. 동일 K3/BF3 lifecycle에서 ESCD B0 DC-only mIoU는 `0.5981 -> 0.5922`, B1 all-geometry는 `0.5996 -> 0.6021`이었다. 동일 Beam-2 lifecycle에서는 DC-only `0.6047 -> 0.5991`, all-geometry `0.6140 -> 0.6082`였으며, 수정 후에도 all-geometry가 DC-only보다 `+0.0091` 높았다. 상세 내용은 [`docs/active-visible-temporal-optimizer-ablation-ko.md`](docs/active-visible-temporal-optimizer-ablation-ko.md)에 기록한다.
- Direct binary ESCD 결과: binary B0 DC-only/B1 all-geometry mean-frame mIoU는 `0.6063/0.6057`, soft S0/S1은 `0.5976/0.6091`이었다. Lifespan/optimizer invariant는 유지됐지만 binary/soft same-scene repeated extra transition이 `170,574/58,426`으로 많다. 현재 핵심 failure는 lifecycle 구현이 아니라 direct detector의 chattering이며, 다음 단계는 별도 실험에서 emission/transition calibration과 spatial/visibility consistency를 검증하는 것이다. 상세 내용은 [`docs/direct-binary-state-lifespan-ablation-ko.md`](docs/direct-binary-state-lifespan-ablation-ko.md)에 기록한다.
- PASLCD 비교: 20 scenes/500 frames에서 direct binary all-geometry는 O-SCD online 대비 u16에서 mIoU/F1 `+0.0190/+0.0175`, u120에서 `+0.0260/+0.0150`이었지만 O-SCD refined보다는 낮았다. Single-state PASLCD에서도 OPEN/CLOSE/REOPEN `922,714/446,622/109,847`이 발생해 detector chattering이 재확인됐다. 상세 내용은 [`docs/paslcd-direct-binary-state-comparison-ko.md`](docs/paslcd-direct-binary-state-comparison-ko.md)에 기록한다.
- FastGS-style change-cue density ablation: mutable `R_change` bank에 raw O-SCD pixel+SAM cue의 soft alpha-T VJP와 causal K-view FastGS gradient/importance 교집합을 연결했다. `ref -> SC1` 3-seed 결과에서 K=10은 matched FastGS gradient-only 대비 split을 `3151.7 -> 2078.3`(`-34.1%`)로 줄이고 mean-frame mIoU를 `0.6063 -> 0.6143`으로 높였다. Temporal lifespan sidecar는 여전히 fixed topology이며 cue-based pruning은 지원하지 않는다. 상세 내용은 [`docs/ref-sc1-fastgs-soft-alpha-t-cue-density-ko.md`](docs/ref-sc1-fastgs-soft-alpha-t-cue-density-ko.md)에 기록한다.
- K=10 ESCD scope extension: 독립 `ref -> SC2/SC3` mIoU는 `0.6843/0.6303`이었다. Oracle boundary로 density view bank의 cross-state sample을 0으로 만든 연속 `ref -> SC1 -> SC2 -> SC3`에서도 state별 mIoU가 `0.6124/0.5593/0.3805`로 하락했다. 따라서 density view sampling만 state-local하게 만드는 것으로는 persistent mutable `R_change`의 state contamination을 해결할 수 없으며, current lifespan 전용 residual bank가 필요하다. 상세 내용은 [`docs/escd-k10-oracle-state-local-density-ko.md`](docs/escd-k10-oracle-state-local-density-ko.md)에 기록한다.
- Active-lifespan density integration: immutable reference prefix와 dynamic residual child topology를 분리하고, child를 root Gaussian의 현재 lifespan slot에 고정했다. 현재 OPEN/current-view-visible row-slot만 all-geometry를 학습하며, K=10 soft alpha-T cue VCD는 active row만 split/clone하고 VCP는 현재 OPEN residual만 제거한다. 독립 SC1/SC2/SC3 seed-0 u120에서 frame-weighted mIoU/F1은 density-off `0.5882/0.7091`에서 `0.5891/0.7096`으로 소폭 변했다. SC1은 `+0.0068`, SC2는 `-0.0003`, SC3는 `-0.0033`으로 일관된 개선은 아니었다. Reference/prefix/closed-slot drift, inactive gradient violation, future-view access는 모두 0이었다. 상세 내용은 [`docs/active-lifespan-fastgs-cue-density-ko.md`](docs/active-lifespan-fastgs-cue-density-ko.md)에 기록한다.
- Persistent direct R_change ablation: lifespan slot에서 DC/geometry delta를 제거하고 Gaussian마다 하나의 mutable change DC/xyz/SH-rest/opacity/scaling/rotation을 직접 유지한다. CLOSE는 opacity gate만 숨기며 값과 Adam moment를 보존하고, REOPEN은 새 interval slot을 할당하되 같은 값에서 학습을 재개한다. 동일 K3/BF3 detector의 독립 SC1/SC2/SC3 u120에서 lifecycle event는 state-local baseline과 정확히 같았고 frame-weighted mIoU/F1은 `0.5882/0.7091 -> 0.5897/0.7102`였다. Peak CUDA memory는 약 `62%`, runtime은 `35--39%` 감소했지만 chattering은 그대로이므로 detector blocker를 해결하지는 않는다. 상세 내용은 [`docs/persistent-direct-rchange-lifespan-ablation-ko.md`](docs/persistent-direct-rchange-lifespan-ablation-ko.md)에 기록한다.
- Equal-status dynamic ACTIVE O-SCD density ablation: reference-prefix/residual class를 없애고 모든 mutable `R_change` row를 동등하게 취급한다. ACTIVE-visible row만 16-step O-SCD loss로 직접 DC/geometry를 학습하고 local update 4에서 ACTIVE-only gradient clone/split을 수행한다. 새 row는 source의 Bayesian/controller/lifespan state를 한 번 복사한 뒤 독립적으로 갱신된다. 독립 SC1/SC2/SC3에서 densify-only weighted mIoU/F1은 fixed topology `0.5779/0.7044` 대비 `0.5768/0.7033`으로 거의 중립이었다. ACTIVE `opacity<0.4` prune을 추가하면 18k--41k row를 제거하며 `0.5715/0.6992`로 하락했다. 상세 내용은 [`docs/dynamic-active-oscd-density-ko.md`](docs/dynamic-active-oscd-density-ko.md)에 기록한다.
- Never-open occlusion/appearance ablation: OPEN-only renderer가 unchanged foreground occluder를 제거하는 문제를 분리했다. NEVER_OPEN row를 고정 zero-DC occluder로만 복원하면 weighted mIoU가 `0.5779 -> 0.4557`로 하락했지만, 해당 row의 geometry는 고정하고 DC/opacity만 visible-view에서 학습하면 `0.6351/0.7539`까지 회복했다. 여기에 원본 O-SCD식 update-4 gradient clone/split을 켜면 `0.6449/0.7618`로 올라 원본 독립 O-SCD online `0.6471/0.7619`와 `-0.0023/-0.0002` 차이였다. CLOSED row는 계속 숨기고 모든 parameter/Adam state drift는 0이었다. 상세 내용은 [`docs/open-never-open-appearance-ablation-ko.md`](docs/open-never-open-appearance-ablation-ko.md)에 기록한다.
- 이 checkpoint에는 MCMC, SGLD, relocation, unrestricted image-space Gaussian birth는 포함하지 않는다. Production temporal model의 기본 topology는 fixed이며, dynamic topology는 active residual density와 equal-status mutable-bank ablation runner에서만 명시적으로 사용한다.
