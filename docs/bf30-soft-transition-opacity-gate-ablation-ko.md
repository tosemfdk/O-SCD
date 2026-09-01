# BF30 candidate soft-transition opacity gate ablation

## 1. 질문

BF30 detector는 reset candidate를 관측하는 동안에도 hard lifecycle bit를 유지한다.
따라서 실제 전환이 시작됐더라도 Bayes factor가 commit threshold에 도달하기 전까지
출력 mask는 즉시 반응하지 않는다.

이번 ablation은 hard OPEN/CLOSE 판정은 보존하면서, live candidate의 진행도만
현재-frame 출력 renderer의 opacity에 반영하면 이 지연을 줄일 수 있는지 검증한다.

## 2. 구현한 soft gate

`candidate_active`인 Gaussian에 대해서만 BF30 commit threshold로 정규화한
진행도 `w`를 만든다.

```text
candidate가 없으면: w = 0
candidate가 있으면: w = clamp(log(BF) / log(30), 0, 1)
```

현재 committed lifespan bit를 `z`라고 할 때 출력 change weight는 다음과 같다.

```text
m = (1 - w) * z + w * (1 - z)
```

| committed state | candidate | 출력 weight `m` |
|---|---|---:|
| CLOSED | 없음 | 0 |
| CLOSED | OPEN candidate | `w` |
| OPEN | 없음 | 1 |
| OPEN | CLOSE candidate | `1-w` |

최종 output-only opacity는 다음처럼 구성한다.

```text
alpha_output = alpha_base * m
```

BF30 commit이 발생하면 candidate가 종료되고 hard bit가 토글되므로 `m`도 새
state의 정확한 0 또는 1로 돌아간다.

## 3. 분리한 실행 경로

새 CLI 옵션은 다음과 같다.

```text
--candidate-render-gate log_bf_progress
```

이 옵션은 `lifespan_gate_beta + lifespan_gate` 조합에서만 허용한다. 최초 양방향
ablation 당시 기본값은 `none`이었다. 이후 CLOSE-only 실험의 개선을 확인한 뒤,
해당 experiment branch의 기본값은 `close_only_log_bf_progress`로 승격했다.

Soft gate가 적용되는 경로:

- 현재 frame pre-optimization 평가 render
- 현재 frame post-optimization 최종 mask/시각화 render

Soft gate가 적용되지 않는 경로:

- detector alpha-T evidence
- BF30 candidate update와 hard commit
- representation training render와 loss
- optimizer row selection
- densify/prune
- local growth replay용 pre/post probability

`w`와 `m`은 모두 detach하며, 저장된 opacity parameter와 Adam state는 수정하지
않는다.

## 4. 구현 위치

- `experiments/run_online_dynamic_active_oscd_density.py`
  - `_candidate_transition_progress`
  - `_soft_lifespan_change_weight`
  - `_soft_lifespan_gate_overrides`
  - `_render_dynamic_change(..., soft_change_weight=...)`
  - `--candidate-render-gate`
- `tests/experiments/test_online_dynamic_active_oscd_density.py`
  - BF 정규화, state cross-fade, opacity detach/non-mutation, CLI contract

## 5. 검증

Targeted CPU tests:

```text
70 passed
```

포함 범위:

- BF30 single-candidate filter
- lifespan-gate controller
- dynamic topology state copy
- persistent lifespan model
- dynamic runner helpers와 argument validation

CUDA 3-frame paired smoke에서는 gate on/off의 다음 산출물이 byte-identical했다.

```text
lifecycle_events.jsonl
topology_events.jsonl
density_events.csv
```

동시에 soft run은 frame별 `37,920 -> 47,827 -> 53,978`개의 live OPEN
candidate를 실제 연속 opacity로 렌더하여 새 분기를 실행했다.

## 6. 304-frame full-run 결과

공통 조건:

| 항목 | 값 |
|---|---|
| stream | 연속 `ref -> SC1 -> SC2 -> SC3`, 304 frames |
| seed / updates | seed 0 / frame당 16 updates |
| detector | lifespan-gate Beta, BF30 |
| renderer support | OPEN + NEVER_OPEN black occluder, CLOSED hidden |
| optimizer | all OPEN |
| loss | local cue support + growth replay |
| density | ACTIVE-only O-SCD clone/split, prune off |

비교 결과:

| 지표 | hard gate | candidate soft opacity | 차이 |
|---|---:|---:|---:|
| mean-frame mIoU | 0.4261 | 0.3135 | -0.1126 |
| mean-frame F1 | 0.5696 | 0.4513 | -0.1183 |
| precision | 0.4512 | 0.2983 | -0.1530 |
| recall | 0.9569 | 0.9895 | +0.0326 |
| predicted-positive fraction | 0.1196 | 0.1871 | +0.0675 |
| false-positive pixels | 9,911,560 | 19,827,152 | +9,915,592 |

State별 mean-frame mIoU:

| segment | hard gate | candidate soft opacity | 차이 |
|---|---:|---:|---:|
| SC1 | 0.4412 | 0.3633 | -0.0779 |
| SC2 | 0.4374 | 0.2826 | -0.1548 |
| SC3 | 0.4012 | 0.2991 | -0.1021 |

Soft run의 candidate row-frame은 다음과 같았다.

```text
OPEN candidate row-frames:  63,536,469
CLOSE candidate row-frames: 12,812,160
frame-mean normalized progress: 0.2058
```

Runtime은 `349.0s -> 455.7s`, peak CUDA memory는 `6.26 GiB -> 7.07 GiB`로
증가했다. Live candidate가 존재하는 frame에서 hard render와 soft output render를
분리해 수행한 비용이다.

모든 run에서 다음 invariant는 유지됐다.

```text
CLOSED parameter/Adam drift: 0
inactive gradient violation: 0
future-view access: 0
topology integrity: pass
```

## 7. 해석

즉각 반응 자체는 발생했다. Recall은 높아지고 commit 전 candidate가 출력에 바로
나타났다. 그러나 BF30 이전의 candidate는 아직 state transition이 아니라
view-local mismatch 가설이다. 이를 수십만 Gaussian에 곧바로 change opacity로
해석하자 transient OPEN candidate가 false-positive mask로 누적됐다.

또한 NEVER_OPEN row는 안정 시 black occluder이지만, positive `m`이 생기면 white
partial-opacity change row로 바뀐다. 따라서 자신의 change contribution뿐 아니라
뒤쪽 Gaussian에 대한 transmittance도 동시에 바뀐다. 이 직접 alpha gate는
불확실성 표현과 occlusion 변화를 분리하지 못한다.

## 8. 결론

이번 구현은 ablation 옵션으로 보존하되 기본값은 `none`으로 유지한다.

```text
candidate BF progress -> 양방향 alpha cross-fade
```

는 전환 지연을 줄이지만, 현재 detector의 대규모 transient candidate를 그대로
시각화하여 precision을 크게 악화시켰다. 다음 우선순위는 다음 둘이다.

1. OPEN candidate는 hard commit까지 숨기고, 이미 보이던 OPEN의 CLOSE candidate만
   즉시 fade하는 asymmetric close-only gate
2. alpha/transmittance는 유지하고 semantic change contribution만 줄여 occlusion과
   uncertainty를 분리하는 gate

Full-run artifacts:

```text
/tmp/escd_bf30_soft_transition_opacity_gate_20260901_v1/
  hard_current/
  continuous/
```
