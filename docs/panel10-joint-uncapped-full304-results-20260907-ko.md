# Panel10 seed archive 20,000 제한 해제: 연속304-frame 비교

2026-09-07. 사용자 요청에 따라 **seed archive 총량 제한만 해제**하고
`ref → SC1 → SC2 → SC3`, seed0, frame당120 updates를 처음부터 재실행했다.

## 결과

프레임 평균 foreground IoU/F1:

| 구간 | Frames | 20,000 제한 IoU/F1 | 무제한 IoU/F1 | IoU 차이 |
|---|---:|---:|---:|---:|
| SC1 | 95 | 0.5743 / 0.6563 | 0.5744 / 0.6563 | +0.0001 |
| SC2 | 104 | 0.7063 / 0.8186 | 0.7208 / 0.8282 | +0.0145 |
| SC3 | 105 | 0.5515 / 0.6975 | 0.6406 / 0.7675 | +0.0892 |
| **전체** | **304** | **0.6116 / 0.7260** | **0.6473 / 0.7535** | **+0.0358** |

전체 pixel 통합 IoU/F1은 **0.6594/0.7947 → 0.7030/0.8256**,
precision/recall은 **0.7815/0.8084 → 0.7882/0.8668**이다.
TP는496,630 증가, FN은496,630 감소했고 FP는58,390 증가했다.
즉 이 run에서는 주로 누락된 변화 회복으로 점수가 높아졌지만 FP가 사라진 것은 아니다.

## 243번째 프레임의 텀블러

영상의 **Frame243 = timestamp242 = SC3 frame44**를 비교했다.

- 제한 run: archive20,000, 현재-frame 신규 root0, OPEN seed298.
- 무제한 run: archive55,203, 현재-frame 후보1,234 중 coverage570 제외,
  신규 root664, budget 탈락0, OPEN seed5,933.
- 해당 프레임 **전체 이미지** IoU/F1: **0.5048/0.6710 → 0.6807/0.8100**.
- 저장 dashboard에서 텀블러 위 seed projection과 최종 positive mask가 복원됨을
  직접 확인했다. 위664개는 프레임 전체 birth 수이지 텀블러만의 수는 아니며,
  위 지표도 텀블러 전용 IoU/F1이 아니다.

[제한 전 Frame243](../outputs/panel10_joint_full304_20260906/captures/000242_dashboard.png)
· [제한 해제 Frame243](../outputs/panel10_joint_uncapped_full304_20260907/captures/000242_dashboard.png)

## 정확히 바뀐 것과 유지한 것

`--da3-max-rows 0`을 **총 archive 무제한**으로 정의했다. 다른 유한 상한으로
바꾼 것이 아니다. 기존 CLI default20,000은 과거 실험 재현용으로 유지하며 이번
run의 `start.sh`가0을 명시한다. 이미 실행 중인 interactive8090은 변경하지 않았다.

공동 BF tracker는 base prefix만으로 시작하고 필요한 seed suffix를4096-row 단위로
확장한다. 기존 BF tensor row는 bitwise 보존하고 신규 tail은 동일 prior로 초기화한다.
이 여유 storage capacity는 seed budget이 아니다. Root/child append 전에 저장 공간을
확보하며, child BF는 기존과 같이 parent에서 한 번 복사한다.

다음은 변경하지 않았다.

- Panel10 NEW support, learned-sigmoid rawQ, NEW/remainder SSF loss.
- 현재-frame seed birth → 고정 base+seed 공동 alpha-T evidence1회 → 단일 BF30
  판단 → 분리학습 순서. Base와 seed detector를 다시 분리하지 않는다.
- 프레임당 최대1,024 root,4×4 sampling, coverage 중복 방지.
- Density event당128 children, root당32 descendants, generation/one-shot 제한.
- Geometry 제약 및 OPEN pruning; CLOSED/retired archive와 Adam history 보존.
- u120, seed0, current-view p=0.33, frozen black NEVER_OPEN, final joint render.

최종 archive는 **69,033 rows = DA3 root66,807 + density child2,226**.
최종 seed OPEN11,648, retired34,273이다. 무제한은 메모리가 무한하다는 뜻이 아니며
장기 스트림의 archive 증가 문제는 남는다. 이번 run에서는 메모리 오류가 없었다.

## 검증과 해석 범위

- 두 configuration JSON은 `da3_max_rows`와 artifact 경로 `capture_dir`만 다르다.
  동일304 frame name, 해상도, GT positive pixel count를 검사했다.
- Saved mask304장을 GT와 독립 재비교하여 confusion count와 aggregate가 일치했다.
- 공동 evidence는 모든 frame에서1회. Reference/fixed probe/frozen parameter/frozen
  Adam drift, future-view access, lifespan render violation은 모두0.
- Targeted CPU+CUDA tests120개 통과:20k 초과 append, tracker 확장, cap 유지 회귀,
  density child inheritance, reset, 실제 DC backward/optimizer 및 alpha-T 포함.
  AST, shell syntax, diff whitespace 검사 통과. LSP/전용 lint/typechecker는 없었다.
- 독립 scoped 코드 리뷰에서 구체적 결함0. 시각 확인 기록도 저장했다.
- MP4는 전체304 frames,10fps,30.4초,1920×1120,H.264/yuv420p이며 전체 디코딩과
  마지막 frame 일치 검증을 통과했다.
- 단일 seed0 비교이며 통계적 유의성/다중 seed 변동은 평가하지 않았다. Cap 도달 전인
  SC1에서도 약0.000108 IoU 차이가 있어 CUDA 반복 변동을0이라고 가정하지 않는다.
  이번 결과는 이 설정에서의 cap 해제 효과를 지지하지만 모든 변화 누락의 해법은 아니다.
- 실행시간1,609.7초, peak allocated CUDA11,611,783,168 bytes. Audit/캡처를 포함하고
  interactive viewer와 GPU를 공유했으므로 모델-only throughput으로 해석하지 않는다.

## 자료

- [전체304-frame MP4](../outputs/panel10_joint_uncapped_full304_20260907/viewer_dashboard_all304.mp4)
- [전체/scene별 summary](../outputs/panel10_joint_uncapped_full304_20260907/summary.json)
- [Frame별 CSV](../outputs/panel10_joint_uncapped_full304_20260907/frame_metrics.csv)
- [제한 run과 matched 비교 JSON](../outputs/panel10_joint_uncapped_full304_20260907/capped_comparison.json)
- [Saved-mask/MP4 독립 검증](../outputs/panel10_joint_uncapped_full304_20260907/posthoc_verification.json)
- [재실행 방법](../outputs/panel10_joint_uncapped_full304_20260907/README.md)
