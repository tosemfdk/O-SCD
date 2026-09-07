# Part 18. DA3 fixed geometry + DC-only u120

## 1. 목적

Part 17에서 Depth Anything 3 depth prior가 object004의 원통 표면을 XFeat보다
조밀하게 생성하는 것을 확인했다. 이번 실험은 그 point cloud를 실제 change
representation에 연결하되, 이전 E4d/E4e failure를 피하기 위해 다음 하나만 학습한다.

```text
DA3 seed: xyz / scale / rotation / opacity 고정
          DC만 학습
```

Reference GS는 고정하며, 기존 두 signed base DC는 기존 surface change와 REMOVE를
담당한다. DA3 sidecar DC는 base-free projected NEW cue objective로 학습한다.

## 2. 실행 계약

- 데이터: independent `ref -> SC3`, 105 frames
- representation update: frame당 120회
- DA3: `depth-anything/DA3-SMALL`
- DA3 input: current frame까지의 최대 8-view fixed-pose window
- scale alignment: immutable reference depth와 positive scale-only robust fit
- birth gate:
  - confirmed causal NEW sign
  - current NEW mask
  - DA3 confidence 상위 60%
  - aligned depth가 reference surface보다 앞
- spatial sampling: DA3 output 4×4 cell당 최대 1 seed
- causal deduplication: world-space 2 cm voxel의 최초 seed 보존
- seed opacity: fixed `0.1`
- seed optimizer membership: `seed_dc` 하나
- GT 사용: birth와 update 완료 후 evaluation에서만 사용

구현:

- [`experiments/run_online_da3_new_seed_dc_only.py`](../experiments/run_online_da3_new_seed_dc_only.py)
- [`temporal/depth_prior_new_seeding.py`](../temporal/depth_prior_new_seeding.py)
- [`tests/experiments/test_run_online_da3_new_seed_dc_only.py`](../tests/experiments/test_run_online_da3_new_seed_dc_only.py)

## 3. 12k seed-cap 결과

### 3.1 전체 결과

| Method | Precision | Recall | IoU | F1 | Mean-frame IoU | Mean-frame F1 |
|---|---:|---:|---:|---:|---:|---:|
| Base signed DC-only | 0.7518 | 0.6540 | 0.5379 | 0.6995 | 0.5182 | 0.6700 |
| Base + fixed DA3 seed DC | 0.6958 | 0.7989 | **0.5921** | **0.7438** | **0.5639** | **0.7130** |
| Difference | -0.0560 | +0.1449 | **+0.0543** | **+0.0443** | **+0.0457** | **+0.0430** |

DA3 seed가 recall을 크게 높였다. Precision은 감소했지만 IoU/F1은 모두 상승했다.

### 3.2 NEW sidecar와 REMOVE 분리

| Scope | Precision | Recall | IoU | F1 | Mean-frame IoU |
|---|---:|---:|---:|---:|---:|
| DA3 learned-DC NEW sidecar | 0.5622 | 0.4430 | **0.3294** | **0.4956** | 0.2787 |
| Opposite-sign base REMOVE | 0.6828 | 0.6963 | 0.5261 | 0.6895 | 0.4777 |

REMOVE 결과는 baseline과 정확히 같다. DA3 branch가 REMOVE DC나 reference bank를
변경하지 않았기 때문이다.

### 3.3 object004 / object010

Object004가 실제로 보이는 45 frames, 즉 frame 40--86의 결과는 다음과 같다.

- active-frame mean IoU: `0.3530`
- active-frame mean full-image precision: `0.4126`
- active-frame mean recall: `0.7129`
- object010 REMOVE mask 내부의 NEW-sidecar false-positive pixel: `0`

대표 시점:

| Frame | Seeds | Overall IoU | NEW IoU | NEW P/R | object004 IoU | object004 P/R |
|---:|---:|---:|---:|---:|---:|---:|
| 48 | 5,290 | 0.5356 | 0.5212 | 0.7458 / 0.6338 | 0.4903 | 0.6658 / 0.6504 |
| 52 | 6,584 | 0.6539 | 0.4028 | 0.7148 / 0.4799 | **0.6056** | 0.7042 / 0.8122 |
| 64 | 8,865 | 0.6150 | **0.6187** | 0.7296 / 0.8028 | 0.4884 | 0.5303 / 0.8608 |
| 80 | 12,000 | 0.5615 | 0.4887 | 0.5666 / 0.7804 | 0.3981 | 0.4074 / 0.9459 |
| 84 | 12,000 | 0.6021 | 0.4964 | 0.5576 / 0.8190 | 0.4344 | 0.4417 / 0.9632 |

Frame52 visualization에서는 object004의 원통과 top rim이 learned DC에 직접 나타난다.
이는 기존 gradient density가 기존 reference GS를 이동시켜 만든 coarse proxy와 다르다.

## 4. 12k cap 대 20k cap

12k cap이 frame80에서 포화되어, late-view birth 차단이 결과를 제한했는지 확인했다.
동일 조건에서 cap만 20k로 늘렸고 실제로는 14,271 seed에서 종료됐다.

| Cap | Final seeds | Overall IoU | Overall F1 | NEW IoU | NEW F1 | NEW P/R |
|---:|---:|---:|---:|---:|---:|---:|
| 12k | 12,000 | **0.5921** | **0.7438** | **0.3294** | **0.4956** | 0.5622 / 0.4430 |
| 20k | 14,271 | 0.5904 | 0.7424 | 0.3286 | 0.4947 | 0.5560 / 0.4456 |

추가 2,271 late seed는 TP를 조금 늘렸지만 FP 증가가 더 컸다. 따라서 **more seed is
not always better**이며 12k run을 현재 채택 결과로 둔다. 다음 개선은 cap 확대가 아니라
multi-view support/confidence 기반 birth quality다.

## 5. 기존 XFeat fixed seed와의 문맥 비교

저장된 E4d D0 artifact에서 XFeat fixed seed의 overall/NEW IoU는 `0.5382/0.2518`이었다.
이번 DA3 fixed seed는 `0.5921/0.3294`였다. 다만 D0는 joint base+seed SSF를 사용했고
이번 run은 base-free projected seed cue를 사용하므로 순수 geometry 한 축만의 controlled
comparison은 아니다. 결론은 다음 정도로 제한한다.

- XFeat sparse geometry + 기존 DC objective보다 현재 DA3 fixed geometry + seed-only DC가
  실제 representation 결과에서 높다.
- geometry prior와 DC supervision을 분리한 추가 2×2 comparison이 없으므로 개선량 전부를
  DA3 depth 하나에 귀속하지 않는다.

## 6. 불변성 및 causal audit

- reference 1,283,501-row 모든 field bitwise unchanged
- reference topology unchanged
- DA3 seed xyz bitwise unchanged
- DA3 seed scale/rotation/opacity/SH-rest/start/end bitwise unchanged
- seed optimizer parameter group: `seed_dc` only
- nonzero learned seed DC rows: `12,000 / 12,000`
- future-view access: `0`
- GT birth access: `0`
- object010 NEW-sidecar false-positive pixels: `0`

## 7. 해석

이번 결과는 앞선 실패 원인과 일치한다.

1. XFeat의 sparse point만으로는 surface coverage가 부족했다.
2. Geometry gradient는 reference/background explanation 때문에 NEW surface로 잘 가지 않았다.
3. 자유 xyz/scale은 정확한 depth anchor를 보존하지 못했다.
4. DA3가 초기 surface를 직접 제공하면 geometry를 움직이지 않아도 DC만으로 object shape가
   나타난다.

남은 약점은 causal NEW mask의 false-positive가 그대로 seed/DC false-positive로 남는다는
점이다. Overall precision 감소 `-0.0560`이 이를 보여준다. 다음 우선순위는 geometry
optimization이 아니라 다음 birth filter다.

- 동일 voxel의 서로 다른 causal view support count
- DA3 depth cross-view residual
- NEW posterior confidence와 DA3 confidence의 joint calibration
- low-support seed의 opacity/DC output gate 또는 causal prune

## 8. 실행 시간 분석

채택한 12k run은 105 frames에 `336.63 s`, 즉 `5 min 36.6 s`가 걸렸다.
프레임 평균은 `3.196 s`이고, frame loop 밖의 checkpoint/summary overhead는
`1.01 s`뿐이다. 따라서 아래 차이는 거의 전부 frame 처리 비용이다.

| 구간 | Frames | End seeds | Total | Mean/frame | P90/frame |
|---|---:|---:|---:|---:|---:|
| birth 이전 | 1--12 | 0 | 20.94 s | 1.745 s | 1.765 s |
| 초기 seed 성장 | 13--39 | 2,578 | 80.73 s | 2.990 s | 3.251 s |
| object004 및 seed 성장 | 40--79 | 11,947 | 149.21 s | 3.730 s | 4.801 s |
| 12k cap 도달 | 80 | 12,000 | 4.93 s | 4.926 s | 4.926 s |
| cap 이후 | 81--105 | 12,000 | 79.82 s | 3.193 s | 4.742 s |

DA3 log가 직접 측정한 66회 preprocess + forward + conversion 합은 `4.83 s`로,
전체 실행 시간의 `1.44%`뿐이다. 이 값은 image load, reference-depth render,
scale fit, unprojection/voxel filtering을 포함하지 않지만, 적어도 DA3 network forward가
현재 병목은 아님을 보여준다.

프레임 시간과 resident seed 수의 Pearson correlation은 `0.551`, 같은 프레임에서
새로 받아들인 birth 수와의 correlation은 `0.471`이었다. Birth가 실행된 frame에 대한
설명용 선형 적합은 다음과 같다.

```text
frame seconds ≈ 2.512
              + 0.132 × resident seed 1,000개
              + 0.129 × current-frame birth 100개
```

이는 seed 수, frame content, birth 수가 서로 공변하므로 인과적 profile로 해석하지 않는다.
그러나 birth 전 `1.745 s/frame`에서 seed 성장 구간 `3.730 s/frame`으로 증가하고,
12k cap 이후에도 `3.193 s/frame`이 유지되는 점은 **fixed NEW seed의 projected DC u120
학습/렌더링 비용**이 주된 가변 비용임을 지지한다.

20k audit은 `364.64 s`, 즉 12k보다 `28.01 s` 또는 `8.32%` 더 걸렸다. 두 run의
정책이 갈라지는 frame80--105에서만 `26.69 s`가 추가됐고, frame81--105의 paired
증가는 평균 `1.068 s/frame`이었다. 추가 2,271 seed가 overall/NEW IoU를 각각
`-0.00176/-0.00077` 낮췄으므로, 현재는 cap 확대가 속도와 정확도 양쪽에서 불리하다.

세부 수치와 시각화:

- `runtime_analysis.json`
- `runtime_phase_summary.csv`
- `runtime_analysis.png`

측정상 주의점은 `frame_runtime_seconds`가 explicit per-stage CUDA synchronization을
두지 않은 wall-clock timer라는 것이다. Scalar/CPU transfer와 evaluation이 실질적인
synchronization을 만들기 때문에 run/cap 비교에는 사용할 수 있지만, base DC, seed DC,
reference-depth render 각각의 정확한 비율은 별도 CUDA-event instrumentation이 필요하다.

## 9. 산출물

채택 run:

`outputs/ref_sc3_da3_new_seed_dc_only_u120_20260902/`

- `summary.json`
- `frame_metrics.csv`
- `checkpoint.pt`
- `da3_new_seeds_dc_only.ply`
- `frame_000048/52/64/80/84_dc_only.png`
- `runtime_analysis.json`
- `runtime_phase_summary.csv`
- `runtime_analysis.png`
- `da3_cache/`

cap audit:

`outputs/ref_sc3_da3_new_seed_dc_only_u120_max20k_20260902/`
