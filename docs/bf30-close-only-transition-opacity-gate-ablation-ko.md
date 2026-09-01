# BF30 CLOSE-only transition opacity gate ablation

## 1. 질문

양방향 candidate opacity gate는 BF30 commit 전의 OPEN candidate까지 change로
렌더하여 precision과 mIoU를 크게 낮췄다. 반대로 이미 OPEN인 Gaussian의 CLOSE
candidate만 즉시 흐리면 stale change를 일찍 제거하면서 OPEN false positive는
만들지 않을 수 있다.

## 2. CLOSE-only gate

Live candidate의 정규화 진행도는 기존 ablation과 같다.

```text
w = clamp(log(BF) / log(30), 0, 1)
```

Committed active bit를 `z`라고 할 때 CLOSE-only output weight는 다음과 같다.

```text
m = z * (1 - w)
alpha_output = alpha_base * m
```

| committed state | candidate | 출력 weight `m` |
|---|---|---:|
| CLOSED | 없음 | 0 |
| CLOSED | OPEN candidate | 0 |
| OPEN | 없음 | 1 |
| OPEN | CLOSE candidate | `1-w` |

즉 OPEN candidate는 BF30 hard commit 전까지 완전히 숨긴다. CLOSE candidate만
evidence가 쌓이는 즉시 기존 change contribution을 약화한다.

CLI:

```text
--candidate-render-gate close_only_log_bf_progress
```

Detector evidence, BF30 commit, training render, optimizer selection,
densify/prune, growth replay는 계속 hard committed lifespan만 사용한다.

## 3. 검증

관련 unit/integration tests:

```text
73 passed
```

검증 내용:

- CLOSED의 OPEN candidate weight는 항상 0
- OPEN의 CLOSE candidate weight만 `1-w`
- candidate progress와 opacity override는 detached
- underlying opacity parameter를 수정하지 않음
- CLI는 `lifespan_gate_beta + lifespan_gate` 조합에서만 허용

## 4. 304-frame 비교

세 run은 연속 `ref -> SC1 -> SC2 -> SC3`, seed 0, frame당 16 updates,
BF30, all-OPEN optimizer, local growth replay, ACTIVE-only density 조건이다.

| 지표 | hard gate | 양방향 soft | CLOSE-only soft |
|---|---:|---:|---:|
| mean-frame mIoU | 0.4261 | 0.3135 | **0.4403** |
| mean-frame F1 | 0.5696 | 0.4513 | **0.5833** |
| precision | 0.4512 | 0.2983 | **0.4689** |
| recall | **0.9569** | 0.9895 | 0.9566 |
| predicted-positive fraction | 0.1196 | 0.1871 | **0.1150** |
| false-positive pixels | 9,911,560 | 19,827,152 | **9,226,618** |

CLOSE-only와 hard gate의 차이:

```text
mean-frame mIoU: +0.0142
mean-frame F1:   +0.0137
precision:       +0.0177
recall:          -0.0003
false positives: -684,942
false negatives: +2,922
```

Recall은 사실상 유지하면서 false positive를 줄인 결과다.

## 5. Scene별 결과

| segment | hard gate mIoU | CLOSE-only mIoU | 차이 |
|---|---:|---:|---:|
| SC1 | 0.4412 | 0.4464 | +0.0053 |
| SC2 | 0.4374 | 0.4526 | +0.0152 |
| SC3 | 0.4012 | 0.4226 | +0.0214 |

304 frames 중:

```text
CLOSE-only 개선: 285 frames
동일:              17 frames
하락:               2 frames
```

Frame별 CLOSE candidate 수와 mIoU 개선량의 Pearson correlation은 `0.502`였다.
오래된 state가 더 누적된 SC2/SC3에서 개선 폭도 더 컸다.

## 6. 실제 soft-gated candidate

```text
OPEN candidate row-frames:          0
CLOSE candidate row-frames: 12,809,703
최대 CLOSE candidate/frame:   100,521
frame-mean transition progress: 0.3175
frame-mean retained weight:      0.6825
```

양방향 gate에서 문제였던 `63.5M` OPEN candidate row-frame이 모두 제거됐다.

## 7. Lifecycle과 invariant

Hard와 CLOSE-only full run의 lifecycle count 차이는 `0.4%` 이하였다. 현재 CUDA
rasterization/density path의 run-to-run 비결정성 범위 안에서 detector trajectory는
거의 유지됐고, mask 차이는 주로 output gate에서 발생했다.

```text
CLOSED parameter/Adam drift: 0
inactive gradient violation: 0
future-view access: 0
topology integrity: pass
```

CLOSE-only run은 `241.2s`, peak CUDA memory `7.05 GiB`였다. 양방향 run의
`455.7s`보다 훨씬 작았는데, OPEN candidate만 있는 frame에서는 추가 soft render를
건너뛰기 때문이다. 순차 실행의 시스템 부하 차이가 있어 hard run과의 wall-time
비교는 참고값으로만 본다.

## 8. 결론

CLOSE-only 비대칭 gate는 이번 조건에서 유효했다.

```text
불확실한 OPEN을 미리 보이지 않음
+ 기존 OPEN의 CLOSE 징후만 즉시 약화
= recall 유지 + precision/mIoU 개선
```

따라서 양방향 gate는 기본 후보에서 제외하고, CLOSE-only를 후속 detector-optimizer
결합 ablation의 우선 후보로 사용한다. 다만 opacity 조절은 transmittance도 바꾸므로,
다음 control에서는 opacity를 유지한 semantic-color attenuation과 비교해 occlusion
효과를 분리해야 한다.

## 9. Experiment branch 기본 계약

검증 결과를 반영해 `experiment/bf30-soft-transition-opacity-gate` 브랜치의
runner 기본값을 CLOSE-only 조건으로 승격했다.

```text
detector_mode: lifespan_gate_beta
change_color_mode: lifespan_gate
candidate_render_gate: close_only_log_bf_progress
render_support_mode: open_or_never_open
optimizer_selection: all_open
loss_regularization_mode: local_growth_replay
min_opacity: 0
```

따라서 runner에는 `--scope`와 `--output-dir`만 주어도 CLOSE-only 계약이 적용된다.
기존 hard gate와 양방향 gate는 각각 다음 override로 계속 재현할 수 있다.

```text
hard:          --candidate-render-gate none
bidirectional: --candidate-render-gate log_bf_progress
```

Artifacts:

```text
/tmp/escd_bf30_soft_transition_opacity_gate_20260901_v1/
  hard_current/
  continuous/   # bidirectional
  close_only/
```

## 10. PASLCD 20-scene 검증

연속 ESCD에서 얻은 개선이 evolving sequence에만 특화된 현상인지 확인하기 위해
PASLCD 20 scenes, 500 frames에서도 동일한 3-way 비교를 수행했다. 세 조건에서 바뀐
인자는 `candidate_render_gate` 하나뿐이다.

```text
detector: lifespan_gate_beta, BF30
change color: lifespan_gate
density: active_oscd
render support: open_or_never_open
optimizer: all_open
loss: local_growth_replay, weight 7.5
updates/frame: 16
min opacity: 0
```

| gate | mean-frame mIoU | mean-frame F1 | precision | recall |
|---|---:|---:|---:|---:|
| hard | 0.3711 | 0.5125 | 0.4443 | **0.7874** |
| bidirectional | 0.2094 | 0.3191 | 0.1934 | 0.9802 |
| **CLOSE-only** | **0.3840** | **0.5257** | **0.4615** | 0.7816 |
| O-SCD online | 0.4887 | 0.6423 | - | - |

CLOSE-only와 matched hard gate의 차이는 다음과 같다.

```text
mean-frame mIoU: +0.0129  (+3.47% relative)
mean-frame F1:   +0.0133  (+2.59% relative)
precision:       +0.0173
recall:          -0.0058
false positives: -674,922 (-7.86%)
false negatives:  +53,571 (+2.69%)
predicted-positive fraction: 0.05767 -> 0.05510
```

Scene 단위로는 20개 중 19개에서 mIoU가 증가했다. 유일한 하락은
`-0.000076`으로 CUDA 반복 변동보다 작은 수준이었다. Frame 단위
승/동률/패는 `422/57/21`이었다. Scene별 mIoU 차이의 평균은 `+0.0129`,
표준편차는 `0.0060`이었고, paired normal approximation의 95% 구간은
대략 `[+0.0101, +0.0157]`이었다.

실제 output gate 적용량은 다음과 같다.

```text
CLOSE-only OPEN candidate row-frames:           0
CLOSE-only CLOSE candidate row-frames:  1,271,154
bidirectional OPEN candidate row-frames: 22,358,056
bidirectional CLOSE candidate row-frames: 1,271,221
```

Hard와 CLOSE-only의 aggregate lifecycle은 각각 OPEN/CLOSE/REOPEN
`299,037/53,266/13,934`, `299,035/53,271/13,935`였다. Clone child와 split
source 합계는 두 조건 모두 `13/146`, final Gaussian 합계도 `3,616,390`으로
같았다. 모든 scene에서 다음 검증을 통과했다.

```text
matched arguments and PLY/camera hashes: pass
CLOSED parameter/Adam drift: 0
inactive gradient violation: 0
future-view access: 0
topology integrity: pass
```

따라서 CLOSE-only의 이득은 single evolving sequence에만 국한되지 않고
single-state PASLCD에서도 재현된다. 다만 PASLCD 절대 성능은 O-SCD online보다
mIoU/F1 `-0.1047/-0.1166` 낮다. 즉 비대칭 output gate는 유효한 precision
보정이지만 detector/representation 전체의 성능 문제를 해결한 것은 아니다.

Artifacts:

```text
/tmp/escd_bf30_close_only_gate_paslcd_20260901_v1/
  hard/
  close_only/
  bidirectional/
  comparison.json
  scene_metrics.csv
  comparison.md
```
