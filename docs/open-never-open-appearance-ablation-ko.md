# Never-open occlusion 및 appearance plasticity ablation

## 질문

기존 equal-status persistent `R_change` runner는 현재 lifespan이 OPEN인
Gaussian만 렌더링했다. 이 경우 unchanged foreground Gaussian까지 opacity가
0이 되어 뒤쪽 active Gaussian이 잘못 노출될 수 있다. 다음 세 조건을
독립 `ref -> SC1`, `ref -> SC2`, `ref -> SC3`에서 비교했다.

1. `open_only`: OPEN row만 렌더링하고 전 속성을 학습한다.
2. `fixed_occluder`: OPEN과 NEVER_OPEN을 렌더링하되 NEVER_OPEN은 zero-DC,
   고정 geometry/opacity occluder로 둔다.
3. `train_dc_opacity`: OPEN은 전 속성을 학습하고, NEVER_OPEN은 DC와
   opacity만 학습한다. NEVER_OPEN의 xyz/SH-rest/scaling/rotation은 고정하며
   CLOSED row는 렌더링하거나 학습하지 않는다.

모든 조건은 binary direct filter, view-consistent K=3/BF=3 controller,
fixed pose/cached O-SCD pixel+SAM cue, capped alpha-T evidence, seed 0,
16 updates/frame, density control OFF를 공유한다.

## 선택 마스크

```text
render support = OPEN union NEVER_OPEN

DC/opacity optimizer mask = visible AND (OPEN union NEVER_OPEN)
geometry optimizer mask   = visible AND OPEN

CLOSED = hidden and exactly frozen
```

NEVER_OPEN row는 초기에는 reference geometry/opacity와 zero change DC를
가진다. DC와 opacity가 cue loss에 맞게 적응하므로 occlusion을 유지하면서도
고정된 검은 occluder가 change 영역을 영구적으로 차단하지 않는다.

## 결과

| 조건 | SC1 mIoU | SC2 mIoU | SC3 mIoU | 가중 mIoU | 가중 F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| OPEN only | 0.5330 | 0.6176 | 0.5792 | 0.5779 | 0.7044 | 0.7202 | 0.8619 |
| 고정 NEVER_OPEN occluder | 0.4710 | 0.4440 | 0.4535 | 0.4557 | 0.5872 | 0.7065 | 0.6893 |
| NEVER_OPEN DC+opacity | **0.6157** | **0.6615** | **0.6266** | **0.6351** | **0.7539** | **0.7695** | **0.8654** |
| NEVER_OPEN DC+opacity + O-SCD densify | **0.6232** | **0.6713** | **0.6383** | **0.6449** | **0.7618** | **0.7680** | **0.8823** |
| 원본 O-SCD online | 0.5915 | 0.6783 | 0.6666 | 0.6471 | 0.7619 | - | - |

DC+opacity plasticity는 OPEN-only 대비 weighted mIoU를 `+0.0572`, F1을
`+0.0496` 높였다. 원본 O-SCD online과의 차이는 mIoU `-0.0120`, F1
`-0.0080`까지 줄었다. SC1은 원본보다 높았지만 SC2/SC3은 각각
`-0.0167/-0.0401` 낮았다.

원본 O-SCD 수치는 동일한 independent 16-update protocol이지만 online
gradient densification을 포함한다. 따라서 마지막 `0.0120` 차이를 오직
lifespan으로 귀속할 수는 없다.

동일하게 local update 4에서 원본 O-SCD gradient clone/split을 켜고 추가
opacity/size pruning은 끈 matched run에서는 weighted mIoU/F1이
`0.6449/0.7618`로 상승했다. Density-off 대비 `+0.0098/+0.0078`이며,
원본 O-SCD와는 `-0.0023/-0.0002` 차이다. SC1은 원본보다 `+0.0316`
높았고 SC2/SC3은 `-0.0070/-0.0283` 낮았다.

| Scope | 초기 GS | 최종 GS | 순증가 | Clone | Split source | Split child |
|---|---:|---:|---:|---:|---:|---:|
| SC1 | 1,283,501 | 1,283,570 | 69 | 0 | 69 | 138 |
| SC2 | 1,283,501 | 1,283,816 | 315 | 38 | 277 | 554 |
| SC3 | 1,283,501 | 1,283,655 | 154 | 2 | 152 | 304 |

세 run의 순증가는 538 GS이며 원본 O-SCD의 832 GS보다 작다. 표의 removed는
추가 pruning이 아니라 split source를 두 child로 교체하면서 발생한 제거다.

## 연속 `ref -> SC1 -> SC2 -> SC3`

독립 실행과 동일한 설정을 하나의 304-frame stream에서 state를 유지한 채
실행했다. Training replay는 원본 O-SCD처럼 현재 frame 또는 임의의 이미
처리된 causal view를 사용하며, manual boundary는 inference에 사용하지
않았다.

| 조건 | SC1 mIoU | SC2 mIoU | SC3 mIoU | 전체 mIoU | 전체 F1 |
|---|---:|---:|---:|---:|---:|
| NEVER_OPEN DC+opacity + densify, 연속 | **0.6228** | 0.5232 | 0.3175 | 0.4833 | 0.6028 |
| 원본 O-SCD online, 연속 | 0.5961 | **0.5468** | **0.4183** | **0.5178** | **0.6401** |

SC1은 원본보다 `+0.0268` 높지만 SC2/SC3은 `-0.0236/-0.1008`
낮아지고, 전체 mIoU/F1은 원본보다 `-0.0345/-0.0373` 낮다. 따라서 독립
single-state 성능은 거의 복구했지만 evolving stream의 누적 오염은 해결하지
못했다.

연속 run lifecycle은 OPEN/CLOSE/REOPEN `55,886/13,954/3,295`, post-hoc
same-state repeated transition `9,007`, final ACTIVE `42,701`이었다. Density는
clone 210, split source 559, split child 1,118로 GS가 `1,283,501 ->
1,284,270` 증가했다. CLOSED drift, 허용 마스크 밖 gradient, active-to-active
false split, reused-slot violation은 모두 0이었다.

이 ablation에서 NEVER_OPEN DC/opacity는 lifecycle OPEN 이전에도 학습되고
렌더링된다. 따라서 아직 OPEN되지 않은 appearance에는 닫을 lifespan 자체가
없어 state가 바뀌어도 stale cue가 남을 수 있다. 또한 causal이지만 모든 과거
state를 섞는 O-SCD replay가 하나의 persistent parameter bank를 이전 cue로
계속 학습시킨다. 이 두 coupling이 연속 SC2/SC3 하락의 유력 원인이며, 현재
결과만으로 각각의 기여도를 분리할 수는 없다.

## Lifecycle 변화

| 조건 | OPEN | CLOSE | REOPEN |
|---|---:|---:|---:|
| OPEN only | 175,135 | 22,241 | 4,465 |
| 고정 occluder | 105,453 | 16,583 | 893 |
| DC+opacity plasticity | 117,789 | 12,035 | 1,177 |

DC+opacity 조건은 OPEN-only보다 CLOSE/REOPEN도 크게 줄였다. 단 detector가
현재 mutable bank의 opacity를 alpha-T evidence에 사용하므로, 이 결과는
순수 renderer-only 변화가 아니다. NEVER_OPEN opacity 학습이 다음 frame의
responsibility와 posterior에도 feedback되어 lifecycle event 자체가 달라졌다.

## 해석

- 고정 occluder 실패는 occlusion의 존재만으로 충분하지 않음을 보여준다.
  fixed high-opacity zero-DC surface가 실제 change mask를 과도하게 가려 recall이
  `0.8619 -> 0.6893`으로 감소했다.
- DC/opacity만 허용하면 occluder가 cue에 맞춰 change intensity와 투과도를
  조절하면서 geometry drift 없이 표현 용량을 회복한다.
- geometry plasticity를 OPEN row에만 제한해 detector가 확정하지 않은
  Gaussian의 공간 구조는 바뀌지 않는다.
- 이 방식은 NEVER_OPEN row가 lifecycle OPEN 이전에도 change DC를 출력할 수
  있으므로, binary detector가 current mask의 유일한 gate라는 이전 의미는
  약해진다. 이를 appearance pre-activation ablation으로 해석해야 한다.

## 무결성 및 비용

- Gaussian topology: 고정, 모든 run `1,283,501` rows
- CLOSED parameter/Adam 최대 drift: `0`
- 허용 마스크 밖 gradient violation: `0`
- active-to-active false split: `0`
- DC+opacity 세 run 총 runtime: `219.76 s`
- peak CUDA memory: `4.385 GiB`
- DC+opacity + densify 총 runtime: `226.53 s`
- DC+opacity + densify peak CUDA memory: `5.109 GiB`
- 연속 DC+opacity + densify runtime: `196.11 s`
- 연속 DC+opacity + densify peak CUDA memory: `6.605 GiB`
- 전체 테스트: `416 passed`

출력은 다음 위치에 생성했으며 repository에 commit하지 않는다.

```text
outputs/escd_dynamic_open_or_never_open_u16_seed0_20260825/
outputs/escd_dynamic_open_or_never_open_dc_opacity_u16_seed0_20260825/
outputs/escd_open_never_dc_opacity_oscd_densify_u16_seed0_20260825/
outputs/escd_open_never_dc_opacity_oscd_densify_continuous_u16_seed0_20260825/
```

실행 예:

```bash
PYTHONPATH=. conda run -n oscd python -m \
  experiments.run_online_dynamic_active_oscd_density \
  --scope scene_change1 \
  --density-policy none \
  --render-support-mode open_or_never_open_dc_opacity \
  --output-dir outputs/<run>/scene_change1
```
