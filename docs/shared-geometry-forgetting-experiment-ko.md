# Shared geometry forgetting 실험

## 질문

State `0`에서 학습한 **동일한 Gaussian geometry**를 State `1`, State `2`에서도
계속 수정하면, 나중에 State `0` DC를 다시 선택해 렌더하더라도 과거 mask 성능이
떨어지는가?

## 실험 설정

- Gaussian topology: 고정, `N=1,283,501`
- manual state boundary: `[95, 199]`
- supervision: O-SCD pixel cue + SAM2.1 feature cue
- frame: 304개, frame당 120 update, 총 36,480 update
- 학습 순서: `S0 -> S1 -> S2`
- state 간 replay: 없음
- densification/pruning: 없음
- base tensor: freeze

Parameter 소유권은 다음과 같다.

```text
state별로 보존:
  state_change_dc[N,S,1,3]
  state_start/end/valid[N,S]

모든 state가 공유하고 계속 덮어씀:
  shared_xyz_delta[N,3]
  shared_opacity_delta[N,1]
  shared_scaling_delta[N,3]
  shared_rotation_delta[N,4]
```

따라서 S0 checkpoint에서는 S0 DC와 S0까지 학습된 geometry를 사용하고, 최종
checkpoint에서 S0를 다시 렌더하면 **S0 DC는 그대로지만 geometry는 S2까지 학습된
최신 값**을 사용한다.

## 결과: forgetting이 명확하게 발생함

각 행은 해당 state 학습이 끝난 직후 저장한 checkpoint이고, 각 열은 그 checkpoint로
다시 렌더한 state다. 값은 GT에 대한 mean per-frame IoU다. 아직 DC를 학습하지 않은
upper triangle은 판정에서 제외한다.

| checkpoint | S0 re-render | S1 re-render | S2 re-render |
|---|---:|---:|---:|
| after S0 | **0.5829** | - | - |
| after S1 | **0.4561** | **0.6404** | - |
| after S2 | **0.2094** | **0.3718** | **0.6296** |

Forgetting delta:

- S0, `after S1 - after S0`: `-0.1268`
- S0, `after S2 - after S0`: `-0.3735`
- S1, `after S2 - after S1`: `-0.2686`

Mean per-frame F1도 같은 경향이다.

| checkpoint | S0 | S1 | S2 |
|---|---:|---:|---:|
| after S0 | 0.6763 | - | - |
| after S1 | 0.5794 | 0.7674 | - |
| after S2 | 0.3255 | 0.5169 | 0.7604 |

아래 예시에서 S0 학습 직후에는 `IoU=0.930`이지만, 동일한 S0 DC를 최종 shared
geometry로 렌더하면 `IoU=0.210`으로 감소한다. 배경의 넓은 FP 영역은 나중 state가
공유 opacity/scale/position을 덮어쓴 결과다.

## geometry가 실제로 이어서 수정됐는지 확인

State 경계를 지날 때 shared xyz의 추가 이동량은 다음과 같다.

| 학습 완료 state | 비교 기준 | mean | median | p95 | max |
|---|---|---:|---:|---:|---:|
| S0 | base | 0.02371 | 0.02094 | 0.05141 | 0.31752 |
| S1 | after S0 | 0.01558 | 0.00899 | 0.05299 | 0.40871 |
| S2 | after S1 | 0.01053 | 0.00000 | 0.04743 | 0.46871 |

즉 State `k+1`이 별도 geometry slot을 학습한 것이 아니라, State `k`가 남긴 동일한
shared geometry parameter를 실제로 추가 수정했다.

경계 checkpoint와 최종 checkpoint를 직접 비교해도 S0 DC drift는 `0.0`인 반면,
S0-valid Gaussian의 shared xyz는 최종 시점까지 평균 `0.01644`, p95 `0.06681`
만큼 변했다. S1 DC drift도 `0.0`이고 S1-valid Gaussian의 S1 이후 shared xyz
이동은 평균 `0.00858`이다.

## 격리 검증

- 완료된 state DC 최대 drift: `0.0`
- inactive DC slot gradient 위반: `0`
- 현재 state에서 invalid인 shared-geometry row gradient 위반: `0`
- frozen base parameter checksum 변화: 없음
- optimizer는 state boundary마다 reset

따라서 과거 성능 저하는 과거 DC가 바뀌어서가 아니라, **DC가 참조하는 공용
geometry가 이후 state에 의해 바뀌었기 때문**이라고 해석할 수 있다.

## 산출물

학습 결과와 경계 checkpoint:

```text
outputs/instance1_scene_change1_2_3_temporal_shared_geometry_forgetting_oscd_cues_allframes_120/
  state0_complete_checkpoint.pt
  state1_complete_checkpoint.pt
  state2_complete_checkpoint.pt
  temporal_rchange_checkpoint.pt -> state2_complete_checkpoint.pt
  summary.json
  retention_matrix.json
```

전체 frame confusion 결과:

```text
..._after_s0_confusion/
..._after_s1_confusion/
..._after_s2_confusion/
```

직접 비교 GIF:

```text
```

## 재현

```bash
conda activate oscd

PYTHONPATH=. python -m experiments.train_shared_geometry_temporal_rchange

RUN=outputs/instance1_scene_change1_2_3_temporal_shared_geometry_forgetting_oscd_cues_allframes_120
for S in 0 1 2; do
  PYTHONPATH=. python -m experiments.render_temporal_confusion_maps \
    --run-dir "$RUN" \
    --checkpoint-path "$RUN/state${S}_complete_checkpoint.pt" \
    --output-dir "${RUN}_after_s${S}_confusion" \
    --overwrite
done

PYTHONPATH=. python -m experiments.summarize_shared_geometry_forgetting
```

## 제한

이 실험은 forgetting mechanism을 분리해 보기 위한 offline ablation이다. Manual
boundary와 state별 전체-view 120 epoch를 사용하고, `state_valid` support도 전체
해당-state view에서 미리 계산한다. 따라서 automatic transition, causal online
support update, BOCD, replay/fusion 정책의 성능을 증명하지 않는다.
