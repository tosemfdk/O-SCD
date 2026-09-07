# Part 20: DA3 NEW seed + learned Stage-2 soft cue u120

## 1. 질문

Part 19의 DA3 fixed-geometry NEW sidecar에 학습된 Stage-2 sigmoid cue를 연결하면
성능이 좋아지는지 확인한다. 최종 change prediction의 threshold는 기존처럼 `0.5`로
고정하고, cue를 representation 학습 전에 이산화하지 않는다.

이번 실험의 learned cue는 다음과 같다.

```text
q = norm(0.8 * L1^0.3 + 0.2 * (1 - SSIM)) * SAM
Q = sigmoid(logit(0.95) * (q - tau) / width)
```

Frame별 `tau`, `width`는 다음 artifact에서 읽었다.

`outputs/stage2_l1_power_sigmoid_boundary/learned_boundaries.json`

## 2. Soft cue 연결 계약

처음 시도한 `Q >= 0.5` hard-gated run은 soft cue의 목적과 맞지 않아 중단하고
결과에서 제외했다. 최종 실험에서는 다음 계약을 사용했다.

1. Base signed DC와 NEW seed DC 학습 target은 연속값 `Q`를 그대로 사용한다.
2. BF30 detector는 binary cue가 아니라 `Q`와 `1-Q`의 soft alpha-transmittance
   fractional evidence를 한 번 누적한다.
3. SAM/PCA가 positive/negative 방향을 정하고, 해당 signed target의 픽셀별
   confidence는 `Q`가 담당한다. 기존 causal PC1 axis, epsilon, NEW sign trace는
   고정하여 sign identity 자체는 비교 중 바꾸지 않았다.
4. Gaussian birth만 이산적인 결정을 요구하므로 `Q >= 0.05`를 넓은 support로 쓴다.
   sigmoid 정의상 이는 대략 `q >= tau - width`에 해당한다.
5. 같은 stride cell 안에서는 `Q`가 큰 pixel을 먼저 선택하고, DA3 confidence를
   stable tie-break로 사용한다.
6. 최종 rendered change mask에만 `prediction >= 0.5`를 적용한다.

## 3. 고정한 조건

- independent `ref -> SC3`, 105 frames
- image당 120 current-view updates
- DA3-SMALL, causal past-only 최대 8-view window
- fixed DA3 xyz/scale/rotation/opacity, DC-only optimization
- frame당 최대 2,048 birth, total cap 12,000
- `0.02 m` voxel, R_change + accepted seed occupancy
- accepted seed는 `NEVER_OPEN`, BF30이 OPEN/CLOSE
- final output threshold `0.5`

비교 대상은 Part 19의 raw binary cue run이다.

## 4. 결과

### 4.1 전체와 NEW 성능

| Scope | Metric | Part19 raw binary | Learned soft Q | Delta |
|---|---|---:|---:|---:|
| Overall | Precision | 0.7309 | **0.8875** | **+0.1567** |
| Overall | Recall | **0.7819** | 0.6116 | -0.1703 |
| Overall | IoU | **0.6071** | 0.5676 | -0.0394 |
| Overall | F1 | **0.7555** | 0.7242 | -0.0313 |
| Overall | Mean-frame IoU | **0.5808** | 0.5725 | -0.0084 |
| NEW | Precision | 0.6634 | **0.8012** | **+0.1378** |
| NEW | Recall | **0.3816** | 0.3441 | -0.0375 |
| NEW | Pixel-weighted IoU | **0.3197** | 0.3170 | -0.0027 |
| NEW | Mean-frame IoU | 0.2810 | **0.2930** | **+0.0120** |
| REMOVE | IoU | **0.5261** | 0.4870 | -0.0391 |

Overall confusion count 변화는 다음과 같다.

- TP: `-544,522`
- FP: `-673,021`
- FN: `+544,522`

즉 learned soft cue는 false positive를 강하게 억제했지만 true positive도 더 많이
제거했다. Precision 증가는 분명하지만 recall 손실이 더 커서 overall IoU/F1은
하락했다.

NEW sidecar만 보면 aggregate IoU는 거의 중립인 `-0.0027`이었고 mean-frame IoU는
`+0.0120` 증가했다. 105 frames 중 NEW per-frame IoU가 증가한 frame은 46개,
감소한 frame은 39개, 동일한 frame은 20개였다.

### 4.2 Object004

| Metric | Part19 raw binary | Learned soft Q | Delta |
|---|---:|---:|---:|
| Active-frame mean IoU | 0.3632 | **0.3774** | **+0.0142** |
| Mean precision | 0.4606 | **0.5247** | **+0.0641** |
| Mean recall | **0.6241** | 0.5469 | -0.0772 |

Object004도 같은 precision-recall tradeoff를 보였다. Object010 REMOVE 영역 내부의
NEW-sidecar false-positive pixel은 계속 `0`이었다.

## 5. Birth와 lifecycle

| 항목 | Part19 raw binary | Learned soft Q |
|---|---:|---:|
| Proposed seed | 12,000 | 12,000 |
| Total candidates | 13,808 | 14,368 |
| Final OPEN | 9,658 | 7,760 |
| Final NEVER_OPEN | 1,325 | 2,807 |
| Final CLOSED | 1,017 | 1,433 |
| OPEN events | 11,470 | 9,916 |
| CLOSE events | 1,812 | 2,156 |

`Q >= 0.05` support는 충분히 넓어 12k cap에 도달했지만, soft BF30은 raw binary
BF30보다 최종 OPEN을 1,898개 적게 유지했다. 전체 recall 하락의 한 원인은 이
더 보수적인 lifecycle이다.

## 6. 해석

이번 integrated learned-soft-cue 조건은 Part 19의 기본 방법으로 채택하지 않는다.
Overall IoU/F1이 각각 `-0.0394/-0.0313` 하락했기 때문이다.

다만 결과는 learned cue가 쓸모없다는 뜻은 아니다.

- NEW mean-frame IoU와 Object004 mean IoU는 증가했다.
- FP 억제와 precision 개선은 매우 컸다.
- 가장 큰 실패는 base REMOVE recall과 BF30 OPEN recall이 함께 줄어든 것이다.

따라서 다음 검증은 cue 전체를 한 번에 교체하지 않고 다음 축을 분리해야 한다.

1. raw detector/birth를 유지하고 DC target만 soft `Q`로 변경
2. raw training/birth를 유지하고 BF30 evidence만 soft `Q`로 변경
3. soft BF30의 calibration 또는 raw/soft evidence mixture로 OPEN recall 복구

## 7. Causality 및 한계

- 현재 run 내부의 DA3 view access는 과거와 현재 frame만 사용했다.
- GT는 birth, detector, representation update 이후 evaluation에만 사용했다.
- reference 1,283,501 rows와 topology는 bitwise unchanged였다.
- 모든 seed xyz/SH-rest/opacity/scaling/rotation은 bitwise unchanged였다.
- learned seed DC는 detector input으로 사용하지 않았다.
- 최종 output threshold는 `0.5` 그대로다.

중요한 한계가 있다. 사용한 BoundaryNet artifact는 동일한 전체 304-frame stream에서
post-hoc으로 학습되었다. 따라서 이 결과는 learned cue 연결 feasibility ablation이며,
held-out deployment-causal 성능으로 해석하면 안 된다.

이 문제를 수정한 과거/현재-only prequential boundary 학습과 cue-only diagnostic은
[`part21-causal-prequential-cue-boundary-ko.md`](part21-causal-prequential-cue-boundary-ko.md)에
기록했다. Part 20의 u120 수치는 causal artifact로 재실행하기 전까지 대체되지 않는다.

또한 이번 비교는 fusion/remap, BF30 evidence, birth support/priority, DC target을 함께
바꾼 integrated 비교다. `tau/width` 하나의 독립 효과를 측정한 결과는 아니다.

별도의 CUDA 실험과 GPU를 공유했으므로 wall-clock runtime은 비교하지 않는다.

## 8. 산출물

`outputs/ref_sc3_da3_neveropen_rchange_occupancy_bf30_learned_stage2_sigmoid_soft_u120_20260903/`

- `summary.json`
- `frame_metrics.csv`
- `checkpoint.pt`
- `da3_new_seeds_dc_only.ply`
- `frame_000048/52/64/80/84_dc_only.png`
- `comparison_to_part19_raw.json`
- `comparison_to_part19_raw.png`
- `run.log`

초기에 잘못 실행한 hard-gated partial run은 다음 위치로 격리했고 metric으로 사용하지
않는다.

`outputs/ref_sc3_da3_neveropen_rchange_occupancy_bf30_learned_stage2_sigmoid_soft_u120_invalid_hardgated_20260903/`
