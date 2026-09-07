# Panel10 공동 detector + DC 수정: 연속 304-frame 평가

기준일: 2026-09-06. `ref → SC1 → SC2 → SC3`의 상태를 유지한 연속 run이며,
scene 사이 reset이나 미래 view replay 없이 현재 viewer 기본 설정을 실행했다.

## 결과

| 구간 | Frames | 평균 foreground IoU | 평균 F1 |
|---|---:|---:|---:|
| scene_change1 | 95 | 0.5743 | 0.6563 |
| scene_change2 | 104 | 0.7063 | 0.8186 |
| scene_change3 | 105 | 0.5515 | 0.6975 |
| **전체** | **304** | **0.6116** | **0.7260** |

전체 pixel confusion count를 합친 IoU/F1은 **0.6594/0.7947**이며,
precision/recall은 **0.7815/0.8084**다. 프레임 평균과 pixel 통합 지표는 다르다.

## 기존 결과와의 비교

| 조건 | Updates/frame | 평균 IoU | 평균 F1 |
|---|---:|---:|---:|
| 원본 O-SCD online | 16 | 0.5166 | 0.6409 |
| 이전 viewer B_h1_amplitude2 (historical replay) | 120 | 0.5249 | 0.6588 |
| 현재 Panel10 + 공동 BF30 + DC 수정 | 120 | **0.6116** | **0.7260** |

모든 비교는 동일한 304 frame name, 해상도, ADD∪REMOVE GT-positive pixel count를 확인했다.
원본 O-SCD의 저장된 binary mask는 같은 GT loader로 재평가했다. 원본 authoritative
요약과 1e-6 수준 차이가 있으며, 여기서는 재평가한 값을 사용한다.

- O-SCD는 u16, 현재 run은 u120이므로 compute-budget matched 비교가 아니다.
- 이전 B_h1은 seed용 binary detector, trainable NEVER_OPEN geometry, joint DC loss,
  unit-Q coverage geometry를 사용했다. 따라서 새 loss 분리, birth, detector 통합, DC
  수정 중 어느 하나의 효과로 차이를 귀속할 수 없다. 세 조건을 동일 방법의 한-factor
  ablation으로 해석하지 않는다.
- Seed0 단일 run이다. 다중 seed 유의성/재현 변동은 평가하지 않았다.

## 현재 run의 중요한 제한: seed archive cap

**t=106, SC2의 12번째 frame에서 20,000-row archive cap에 도달했다.**
CLOSED/retired 행도 archive budget에 포함되고 물리적으로 삭제하지 않기 때문에
**SC3에서는 새 DA3 root와 density child가 각각 0개 생성되었다.** 이는 현재 preset
자체의 제약이며, 이번 평가는 이 상한을 변경하지 않았다. SC3 결과를 자유롭게 새로운
seed가 공급되는 방법의 성능으로 해석하면 안 된다. 낮은 SC3 점수의 원인이 이 제한
하나라고 단정하지는 않으며, 상한/retirement 정책 변경은 별도 비교가 필요하다.

| 구간 | DA3 root birth | Density child |
|---|---:|---:|
| scene_change1 | 17,491 | 538 |
| scene_change2 | 1,950 | 21 |
| scene_change3 | 0 | 0 |

## 실행 및 metric 계약

- `training_partition=panel10_new`, seed0, frame당120 updates, latest-view p=0.33.
- Raw learned-sigmoid Q 전체를 사용하며, 현재 DA3 root birth 후 고정 base+seed
  alpha-T evidence를 1회 계산하여 단일 BF30 tracker를 갱신한다.
- Q_NEW는 seed DC/geometry SSF, Q−Q_NEW는 base DC SSF에 사용한다.
- NEVER_OPEN은 frozen black occluder, CLOSED는 output에서 제외, 최종 mask는
  learned base+seed joint render의 raw mean RGB≥0.5이다.
- GT는 object NEW∪REMOVED를 native viewer 528×941로 nearest resize한 union.
  GT를 detector/loss/birth에 공급하지 않고 causal step 완료 후 metric을 계산한다.
- IoU/F1은 foreground binary 지표이며 background class와 평균하지 않는다.
  Empty union / empty F1 denominator는 기존 evaluator와 같이0으로 처리한다.
- Reference, fixed probe, frozen parameter, frozen Adam drift는 모든304 frame에서0.
  미래 view 접근, lifespan render violation도0. Detector evidence는 frame당1회.
- 전체 실행은 1529.9초였으나 audit·GT 평가·PNG 렌더를 포함하고
  GPU가 interactive viewer와 공유되었으므로 모델-only throughput으로 사용하지 않는다.

## 시각화 및 원본 자료

[전체304-frame viewer MP4](../outputs/panel10_joint_full304_20260906/viewer_dashboard_all304.mp4)
— 1920×1120,10fps,30.4초,H.264/yuv420p. Frame 누락이나 subsampling 없이
기존 viewer 전체 dashboard를 렌더하고 frame/scene/IoU/F1 header를 추가했다.
원래 viewer의 reserved panel과 긴 title clipping은 그대로다.

- [전체/scene별 summary](../outputs/panel10_joint_full304_20260906/summary.json)
- [Frame별 confusion count와 score CSV](../outputs/panel10_joint_full304_20260906/frame_metrics.csv)
- [Baseline 비교와 provenance](../outputs/panel10_joint_full304_20260906/baseline_comparison.json)
- [후처리 검증](../outputs/panel10_joint_full304_20260906/posthoc_verification.json)
- [재실행 설명 및 artifact 목록](../outputs/panel10_joint_full304_20260906/README.md)
