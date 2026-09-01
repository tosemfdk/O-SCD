# BF30 raw-cue detector + learned-DC CLOSE-candidate 출력 감쇠

## 1. 질문

기존 learned-DC 출력은 lifecycle이 hard CLOSE를 확정하기 전까지 OPEN Gaussian을
그대로 렌더링한다. 이번 실험은 다른 구성은 유지하고, **CLOSE 방향의 reset
candidate만** Bayes factor 진행도에 따라 점진적으로 약하게 출력하면 mask가
개선되는지 확인한다.

중요하게도 detector의 입력은 바꾸지 않는다. Detector는 학습된 `C_i`와 cue의
차이를 사용하지 않고, 신규 프레임 optimization 전에 계산한 Gaussian별 raw cue
alpha-T pseudo-count 분포만 관측한다.

## 2. Matched 조건

- Detector: `single_candidate_beta`, BF30
- Detector evidence: 현재 신규 프레임의 pre-optimization raw cue alpha-T만 사용
- Change representation: learned persistent DC
- Render support: OPEN + frozen black NEVER_OPEN occluder
- CLOSED: 렌더링 제외, parameter와 Adam state 보존
- Optimizer: 모든 OPEN row, view visibility 제약 없음
- Density: off
- Loss: local support + previous positive-growth replay, weight 7.5
- Updates: frame당 16회

## 3. CLOSE 후보 판정

Raw single-candidate detector의 live candidate는 단순히 “분포가 바뀌는 중”이라는
뜻이므로, committed OPEN row라고 해서 모두 CLOSE 후보로 취급하면 안 된다.
Candidate block의 fresh Beta posterior가 non-change 방향인지 별도로 판정한다.

```text
candidate_a = 1 + candidate_delta_a
candidate_b = 1 + candidate_delta_b

p_candidate_change = candidate_a / (candidate_a + candidate_b)

close_candidate =
    committed_OPEN
    AND candidate_live
    AND p_candidate_change <= 0.4
```

여기서 `candidate_delta_a/b`는 학습된 DC가 아니라 raw cue가 Gaussian에 배분한
change/non-change alpha-T pseudo-count 누적값이다.

## 4. 출력 감쇠

```text
progress = clamp(candidate_log_BF / log(30), 0, 1)

OPEN CLOSE-candidate opacity multiplier = 1 - progress
그 외 OPEN opacity multiplier            = 1
NEVER_OPEN black occluder multiplier      = 1
CLOSED opacity                            = 0
```

Learned DC, geometry, stored opacity, optimizer state는 수정하지 않는다. 이 multiplier는
현재 프레임의 pre/post metric 및 visualization render에만 적용한다. Detector probe,
training render, growth replay, density control, hard BF commit은 모두 기존 hard lifespan
경로를 그대로 사용한다.

## 5. 정확한 비교 방법

별도 CUDA run끼리 비교하면 rasterization/optimization의 작은 비결정성이 누적되어
lifecycle도 조금 달라질 수 있다. 따라서 한 run 안에서 동일한 파라미터와 동일한
lifecycle로 다음 두 render를 모두 계산하고 GT는 causal loop 종료 뒤 함께 평가했다.

1. Hard learned-DC render
2. 동일 hard render에 CLOSE-candidate opacity multiplier만 적용한 render

이 비교에서는 detector, 학습 결과, OPEN/CLOSE/REOPEN event가 두 출력 사이에 완전히
공유된다.

## 6. Continuous ref -> SC1 -> SC2 -> SC3

### 6.1 전체 결과

| 출력 | mean-frame mIoU | mean-frame F1 | precision | recall |
|---|---:|---:|---:|---:|
| Same-run hard learned DC | 0.4896 | 0.6263 | 0.5811 | 0.8454 |
| CLOSE-candidate 감쇠 | **0.4938** | **0.6304** | **0.5866** | **0.8469** |
| 차이 | **+0.0042** | **+0.0041** | **+0.0055** | **+0.0016** |

Pixel confusion 차이:

```text
TP: +13,264
FP: -107,813
FN: -13,264
```

감쇠인데도 TP가 증가할 수 있는 이유는 alpha compositing 때문이다. Foreground의
CLOSE-candidate opacity를 낮추면 뒤쪽의 더 밝은 OPEN change Gaussian이 드러날 수
있어 최종 pixel score는 반드시 단조 감소하지 않는다.

### 6.2 State별 결과

| 구간 | hard mIoU | soft mIoU | 차이 | hard F1 | soft F1 | 차이 |
|---|---:|---:|---:|---:|---:|---:|
| SC1 | 0.4892 | 0.4926 | +0.0034 | 0.5972 | 0.6007 | +0.0035 |
| SC2 | 0.5473 | 0.5505 | +0.0032 | 0.6891 | 0.6918 | +0.0027 |
| SC3 | 0.4328 | 0.4387 | +0.0059 | 0.5905 | 0.5966 | +0.0061 |

- Frame mIoU 승/동률/패: `216/21/67`
- CLOSE-candidate row-frame: `5,545,050`
- OPEN-candidate preview row-frame: `0`
- Shared OPEN/CLOSE/REOPEN: `141,912 / 10,859 / 2,793`

Artifacts:

```text
/tmp/escd_bf30_learned_dc_close_only_metric_mask_visual_20260901_v2/
  run/summary.json
  comparison.json
  comparison.md
  visualization/ref_sc1_sc2_sc3_lifespan_events.mp4
  visualization/ref_sc1_sc2_sc3_lifespan_events.gif
```

영상 패널은 기존 형식을 유지한다.

1. Inference RGB
2. Combined change cue
3. Causal raw R_change
4. 실제 평가에 사용한 threshold mask
5. 현재 timestamp의 hard OPEN/CLOSE event
6. Timestamp별 hard OPEN/CLOSE Gaussian 개수

## 7. PASLCD 20 scenes / 500 frames

| 출력 | mean-frame mIoU | mean-frame F1 | mean-scene precision | mean-scene recall |
|---|---:|---:|---:|---:|
| Same-run hard learned DC | 0.4566 | 0.6124 | 0.5538 | **0.7531** |
| CLOSE-candidate 감쇠 | **0.4604** | **0.6159** | **0.5594** | 0.7520 |
| 차이 | **+0.0038** | **+0.0035** | **+0.0056** | -0.0011 |

```text
Scene mIoU 승/동률/패: 19/0/1
Frame mIoU 승/동률/패: 419/26/55
TP: -7,689
FP: -126,604
FN: +7,689
CLOSE-candidate row-frame: 2,737,417
OPEN-candidate preview row-frame: 0
```

PASLCD에서는 예상대로 주효과가 false-positive 억제이고 recall이 아주 조금
감소했다. 20 scenes 중 19개에서 mIoU가 개선됐지만 절대 성능은 저장된 O-SCD online
`0.4887/0.6423`보다 여전히 낮다.

Artifacts:

```text
/tmp/escd_bf30_raw_cue_learned_dc_close_only_paslcd_20260901_v2/
  close_only/Instance_*/<scene>/summary.json
  comparison.json
  scene_metrics.csv
  comparison.md
```

## 8. 검증

- Continuous와 PASLCD 모두 same-run hard control 사용
- Detector learned-DC input: false
- Replay/post-optimization render의 detector evidence 사용: false
- OPEN-candidate preview: 0
- CLOSED parameter/Adam drift: 0
- Inactive/wrong gradient violation: 0
- Future-view access: 0
- Topology integrity: pass
- 영상 threshold mask 304 frames와 metric CSV positive fraction 오차: 0

## 9. 결론

CLOSE 방향만 점진적으로 감쇠하는 비대칭 출력은 learned-DC representation의 학습이나
raw-cue detector를 섞지 않고도 continuous와 PASLCD 모두에서 약 `+0.004 mIoU`를
얻었다. 따라서 **output precision 보정**으로는 유효하다.

다만 이는 detector 자체를 개선한 결과가 아니다. Hard CLOSE 판정, candidate 생성량,
same-scene transition은 그대로이며 PASLCD의 O-SCD online 격차도 남아 있다. 현재
실험은 raw-cue detector와 learned-DC representation을 분리한 상태에서 candidate
불확실성을 출력에만 반영하는 ablation으로 해석해야 한다.
