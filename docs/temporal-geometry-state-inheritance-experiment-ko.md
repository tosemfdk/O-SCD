# Temporal geometry predecessor-state inheritance 실험

## 질문

하나의 고정 base Gaussian identity 집합에서 State `k`를 학습한 뒤, 완료된
DC/xyz/opacity/scale/rotation을 State `k+1` slot의 초기값으로 그대로 복사하고
계속 학습하면 state 간 Gaussian identity 연속성이 높아지는가?

## 설정

- Gaussian topology: 고정, `N=1,283,501`
- state boundary: `[95, 199]`
- supervision: O-SCD pixel cue + SAM2.1 feature cue
- frame: 304개, frame당 120 update
- state별 `state_valid` gate 유지
- densification/pruning 없음
- State 0 xyz anchor 없음 (`weight=0`)
- boundary 동작:

```text
train S0
copy all S0 attributes -> S1 slot
train S1 while S0 is frozen
copy all S1 attributes -> S2 slot
train S2 while S0/S1 are frozen
```

`state_valid`, `state_start`, `state_end`는 복사하지 않고 target state의 기존
support/lifespan metadata를 유지한다.

## Inheritance 검증

S0->S1 및 S1->S2 복사 직후의 최대 절대 오차는 DC/xyz/opacity/scale/rotation
모두 `0.0`이다. 완료된 S0/S1 parameter drift도 `0.0`이다.

## Mask 성능

값은 304개 frame의 per-frame IoU/F1 산술 평균이다.

| Scope | Independent geometry IoU | Inherited geometry IoU | Independent F1 | Inherited F1 |
|---|---:|---:|---:|---:|
| S0 | 0.5840 | 0.5833 | 0.6769 | 0.6761 |
| S1 | 0.6392 | 0.6342 | 0.7665 | 0.7627 |
| S2 | 0.6302 | 0.6310 | 0.7602 | 0.7611 |
| Overall | 0.6188 | 0.6172 | 0.7363 | 0.7351 |

Inheritance는 mask 성능을 개선하지 않았고 overall IoU가 약 `0.0016`
감소했다.

## Gaussian identity 연속성

각 state에서 `state_valid=True`이고 mean raw DC가 `0`보다 큰 GS를 bright
identity로 정의했다.

| Transition | Independent: 동일 identity가 bright 유지 | Inherited: 동일 identity가 bright 유지 |
|---|---:|---:|
| S0 -> S1 | 80.81% | 92.04% |
| S1 -> S2 | 53.44% | 95.08% |

즉 최종 mask 정확도는 거의 같지만, predecessor-state inheritance는 사용자가
의도한 **같은 GS가 다음 state까지 이어지는 표현**을 훨씬 강하게 만들었다.

## 산출물

학습 checkpoint와 state switch:

```text
outputs/instance1_scene_change1_2_3_temporal_geometry_inherited_oscd_cues_allframes_120/
  temporal_rchange_checkpoint.pt
  summary.json
  inherited_state_switch_manifest.json
```

GT confusion GIF:

```text
outputs/instance1_scene_change1_2_3_temporal_geometry_inherited_confusion_oscd_cues_allframes_120/
```

Independent geometry와 직접 비교 GIF:

```text
outputs/instance1_scene_change1_2_3_temporal_geometry_inherited_vs_independent_confusion/
```

## 제한

이 실험은 manual boundary와 state 내부 전체-view 120 epoch replay를 사용한
offline representation 실험이다. 실제 online에서 frame을 한 번 처리하고 버리는
조건이나 automatic changepoint는 검증하지 않았다. 또한 independent baseline은
State 0 xyz anchor를 사용했기 때문에 inheritance 하나만을 분리한 strict ablation은
아니다.
