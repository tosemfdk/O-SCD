# Stage 2 histogram-MLP sigmoid cue boundary

## 목적

고정 휴리스틱 `τ=0.25`, `width=0.10`을 그대로 쓰는 대신, 현재 frame의 cue
분포로부터 sigmoid 경계를 예측한다. Gaussian representation은 Stage 2에서 완전히
고정하며 GT는 학습에 사용하지 않는다.

입력 cue는 다음 L1-only power 조합이다.

```text
q = norm(0.8·L1^0.3 + 0.2·(1−SSIM)) · SAM
```

Sigmoid의 width는 transition half-width로 정의한다.

```text
Q = sigmoid(logit(0.95) · (q−τ) / width)

q=τ−width -> Q=0.05
q=τ       -> Q=0.50
q=τ+width -> Q=0.95
```

## Teacher와 causality 범위

- Teacher: 원본 O-SCD online run이 각 frame의 16 updates 직후 저장한
  `online_at_arrival` binary mask
- 각 teacher mask 자체는 해당 시점까지의 frame만 사용한다.
- 하지만 BoundaryNet은 304-frame stream을 모은 뒤 post-hoc으로 학습했다.
- 따라서 이번 결과는 calibration feasibility ablation이며 production online 결과가
  아니다.
- GT는 학습이 끝난 뒤 평가에만 사용했다.

## MLP와 warm start

```text
64-bin normalized histogram + 64-bin CDF
  -> Linear(128,32) -> SiLU
  -> Linear(32,16)  -> SiLU
  -> delta_tau, delta_width
```

- 마지막 head의 weight와 bias를 0으로 초기화한다.
- 최초 모든 frame 출력은 정확히 `τ=0.25`, `width=0.10`이다.
- 최초 40 epoch는 trunk를 고정하고 head만 learning-rate warm-up한다.
- 초기 경계 anchor는 100 epoch 동안 cosine으로 0까지 감쇠한다.
- train/validation은 순서상 매 5번째 view를 validation으로 떼어 `244/60`으로 나눴다.

Stage 2 loss는 다음과 같다.

```text
L = balanced_soft_BCE(Q, stopgrad(T))
  + (1 − softIoU(Q, stopgrad(T)))
  + warm_start_anchor
  + temporal_smoothness
```

## 304-frame exact GT 평가

Hard mask 평가는 sigmoid의 `Q>0.5`, 즉 `q>τ`로 수행했다. 아래 값은 512-bin
근사가 아니라 원본 해상도 cue를 다시 계산한 exact pixel 결과다.

| 범위 | 고정 τ=0.25 mIoU | 학습 τ_t mIoU | 차이 |
|---|---:|---:|---:|
| Overall 304 | 0.5871 | 0.6219 | +0.0349 |
| Train views 244 | 0.5852 | 0.6199 | +0.0347 |
| Validation views 60 | 0.5945 | 0.6300 | +0.0355 |
| SC1 | 0.5702 | 0.5789 | +0.0087 |
| SC2 | 0.5815 | 0.6389 | +0.0574 |
| SC3 | 0.6078 | 0.6440 | +0.0362 |

Overall 세부 변화:

- mean-frame F1: `0.7094 -> 0.7361`
- aggregate precision: `0.6263 -> 0.6810`
- aggregate recall: `0.9637 -> 0.9534`
- FP: `4,894,095 -> 3,800,876`, 약 `22.3%` 감소
- FN: `309,097 -> 396,415`

학습된 parameter 분포:

- `τ`: mean `0.2763`, range `0.2173 .. 0.4501`
- `width`: mean `0.1238`, range `0.0769 .. 0.2353`

즉 학습된 경계는 고정 0.25보다 평균적으로 더 엄격해져 FP를 줄였으며, recall을 약
1.03 percentage point 내주는 대신 precision과 mIoU를 높였다. 다만 width는 평균
0.10보다 넓어졌다. Hard IoU 개선은 τ의 효과이고, 현재 Stage-2 objective가 더 sharp한
soft evidence를 자동으로 선택했다는 증거는 아니다.

## 실행과 산출물

```bash
/home/rvl/miniforge3/envs/oscd/bin/python \
  -m experiments.train_stage2_cue_boundary
```

- 결과: `outputs/stage2_l1_power_sigmoid_boundary/summary.json`
- frame별 경계: `outputs/stage2_l1_power_sigmoid_boundary/learned_boundaries.json`
- MLP checkpoint: `outputs/stage2_l1_power_sigmoid_boundary/boundary_mlp.pt`
- 학습 그래프: `outputs/stage2_l1_power_sigmoid_boundary/training_and_boundaries.png`
- GT 비교 그래프: `outputs/stage2_l1_power_sigmoid_boundary/exact_gt_comparison.png`

Viewer는 이제 `./run_bayesian_detector_viewer.sh`로 이 learned boundary artifact를
자동으로 읽는다.

## 해석 제한

1. Validation view는 teacher loss에서 제외했지만 같은 세 장면/동일 stream에 속한다.
   새로운 scene에 대한 generalization은 아직 검증하지 않았다.
2. 평가는 2D cue hard mask 대 GT다. Bayesian alpha-T projection과 lifecycle을 모두
   통과한 최종 detector mIoU 평가는 별도다.
3. 더 sharp한 soft evidence가 목적이면 `width<=0.10` 제약 또는 width penalty를 별도
   ablation해야 한다. 현재 결과에 그 제약을 섞지 않았다.
