# Part 11. Global vs. local regularization: sum cue와 power-product cue의 2x2 검증

> 작성일: 2026-08-13
> 최근 업데이트: 2026-08-14
> 범위: `R_ref -> scene_change1/2/3` 독립 online 실행
> 핵심 결론: **regularizer를 local로 바꾸면 recall이 증가하며, 그 효과는 power-product cue와 큰 change 영역에서 더 강하다. 여기에 직전 update에서 실제로 증가한 pixel을 다시 제약하는 `G_prev` 항을 추가하면 online-at-arrival 결과가 B local보다 개선된다. `lambda_g=7.5`가 이번 단일-seed sweep의 최고점이지만, Original global 대비 전체 mIoU는 아직 `-0.0046`이고 SC2가 주된 실패 원인이다.**

## 1. 질문

Part 10의 product 실험은 cue fusion뿐 아니라 다음 global regularizer의 영향을 함께 받았다.

```text
R_global = log(1 + mean(m)^2)
```

이 항은 현재 view에서 mask가 크게 보일수록 모든 pixel에 더 강한 공통
브레이크를 건다. 따라서 product cue의 recall 저하가 cue의 AND-gate 성질만이
아니라 global regularization과의 결합에서 생겼는지 분리해야 한다.

## 2. 2x2 실험

두 cue를 비교한다.

```text
Sum cue:
C = P + S

Power-product cue:
C = 2 * P^0.3 * S
```

공통 detection 항은 그대로 유지한다.

```text
L_detection = mean(C * (1 - m))
m = sigmoid(mean(render_change, channel))
```

### A. Original global sparsity

```text
L_A = L_detection + log(1 + mean(m)^2)
```

### B. Local sparsity

```text
C_hat = clamp(C / 2, 0, 1)

L_B = L_detection + mean((1 - C_hat) * m)
```

`P+S`와 `2*P^0.3*S`의 이론적 범위가 모두 `0..2`이므로 두 fusion에 같은
고정 변환 `C/2`를 적용했다. 프레임별 재정규화나 GT 기반 보정은 사용하지
않았다. 두 loss 모두 regularization weight는 `1`이다.

## 3. 통제 조건

| 항목 | 설정 |
|---|---|
| scene | `R_ref -> scene_changeX` 독립 실행 |
| pose / cue | 동일 fixed pose와 cached cue |
| seed | 0 |
| resolution | 4 (`941 x 528`) |
| online budget | 이미지당 16 updates |
| replay | 최신 view 약 33%, 나머지는 지금까지 본 view |
| trainable | xyz, DC, SH-rest, opacity, scale, rotation |
| topology | online clone/split densification |
| binary output | raw rendered score `> 0.5` |
| primary phase | `online_at_arrival` |

A도 새 ablation runner로 다시 실행했다. 이전 Part 10 A 산출물과 fresh A
사이에는 condition에 따라 전체 pixel의 약 `0.3%..1.1%`가 달랐다. CUDA
rasterization과 densification 경로의 작은 비결정성이 누적된 것으로 보이므로,
아래 표는 **같은 시점에 다시 실행한 fresh A와 B**만 비교한다.

## 4. Online-at-arrival 결과

### 4.1 Scene별 결과

| Scene | Loss | Cue | mIoU | F1 | Precision | Recall | Pred. fraction |
|---|---|---|---:|---:|---:|---:|---:|
| SC1 | A Global | Sum | 0.600 | 0.693 | 0.675 | 0.758 | 0.048 |
| SC1 | A Global | Product | 0.511 | 0.614 | 0.682 | 0.634 | 0.035 |
| SC1 | B Local | Sum | 0.484 | 0.594 | 0.526 | 0.783 | 0.064 |
| SC1 | B Local | Product | 0.572 | 0.669 | 0.632 | 0.771 | 0.053 |
| SC2 | A Global | Sum | 0.697 | 0.814 | 0.726 | 0.945 | 0.080 |
| SC2 | A Global | Product | 0.709 | 0.823 | 0.771 | 0.899 | 0.071 |
| SC2 | B Local | Sum | 0.559 | 0.705 | 0.567 | 0.979 | 0.104 |
| SC2 | B Local | Product | 0.668 | 0.794 | 0.684 | 0.966 | 0.087 |
| SC3 | A Global | Sum | 0.622 | 0.758 | 0.729 | 0.809 | 0.066 |
| SC3 | A Global | Product | 0.652 | 0.781 | 0.797 | 0.791 | 0.060 |
| SC3 | B Local | Sum | 0.560 | 0.704 | 0.574 | 0.934 | 0.099 |
| SC3 | B Local | Product | 0.641 | 0.770 | 0.671 | 0.925 | 0.083 |

### 4.2 전체 304 frame 평균

| Loss | Cue | mIoU | F1 | Precision | Recall | Pred. fraction |
|---|---|---:|---:|---:|---:|---:|
| A Global | Sum | **0.641** | **0.757** | 0.711 | 0.839 | 0.065 |
| A Global | Product | 0.628 | 0.743 | **0.752** | 0.779 | 0.056 |
| B Local | Sum | 0.536 | 0.670 | 0.557 | **0.902** | 0.090 |
| B Local | Product | 0.629 | 0.747 | 0.663 | 0.891 | 0.075 |

## 5. 해석

### 5.1 Local regularization은 실제로 growth suppression을 완화했다

Global에서 local로 바꾸면 recall이 다음과 같이 증가했다.

```text
Sum:     0.839 -> 0.902  (+6.3 percentage points)
Product: 0.779 -> 0.891  (+11.2 percentage points)
```

Product가 두 배 가까운 recall 증가를 보였다는 것은 이전 product 결과가 cue의
AND-gate 성질만으로 결정된 것이 아니라, 작은 product cue가 global brake를
넘지 못했던 영향도 포함했음을 지지한다.

### 5.2 하지만 단순 local sparsity는 FP를 충분히 제어하지 못한다

Local sum은 recall을 높였지만 precision이 `0.711 -> 0.557`로 떨어지고 mIoU도
`0.641 -> 0.536`으로 감소했다. `P+S`의 diffuse false cue 위치에서는
`C_hat`이 regularizer를 함께 약화시키므로 background activation이 증가한다.

Product는 cue 자체의 AND gate가 diffuse false cue를 줄이기 때문에 local loss와
더 잘 맞았다.

```text
Global loss에서: Product - Sum mIoU = -0.013
Local loss에서:  Product - Sum mIoU = +0.093
```

즉 cue fusion의 순위가 regularizer에 따라 뒤집혔다. **Cue와 regularizer는 독립된
선택이 아니며, 기존 global objective 아래의 product 결과만으로 product cue를
기각할 수 없다.**

### 5.3 큰 change 영역에서 product-local의 이점이 커졌다

GT mask의 화면 점유율로 frame을 세 구간으로 나누고 product에서 `B - A` mIoU를
계산했다.

| GT area group | Mean GT fraction | Product local - global mIoU |
|---|---:|---:|
| Small | 0.021 | -0.057 |
| Medium | 0.053 | -0.012 |
| Large | 0.096 | **+0.073** |

GT fraction과 product의 local 이득 사이 Pearson correlation은 `+0.369`였다.
특히 SC1에서는 large tertile의 mIoU가 평균 `+0.240` 증가했다. 이는 mask가 크게
투영되는 view에서 global brake가 growth를 방해한다는 가설과 일치한다.

다만 SC2에서는 세 면적 구간 모두 product-local mIoU가 소폭 낮았다. 따라서 이
결과는 global coupling의 존재를 지지하지만, 단순 local loss가 모든 scene에서
더 낫다는 의미는 아니다.

## 6. Final rerender 진단

전체 sequence를 본 뒤 과거 frame을 다시 렌더하면 다음 결과가 나온다.

| Loss | Cue | mIoU | Precision | Recall |
|---|---|---:|---:|---:|
| A Global | Sum | 0.672 | 0.704 | 0.893 |
| A Global | Product | **0.698** | **0.764** | 0.854 |
| B Local | Sum | 0.552 | 0.556 | **0.939** |
| B Local | Product | 0.650 | 0.662 | 0.928 |

Local loss는 장기적으로도 recall은 높지만 activation이 더 넓게 남아 final
rerender precision과 mIoU가 낮다. Online SCD의 primary metric은 arrival이지만,
이 진단은 B가 아직 최종 loss가 아님을 보여준다.

## 7. 현재 결론

1. **Global-local coupling 가설은 지지된다.** Local loss가 두 cue의 recall을 모두 높였고 product에서 효과가 더 컸다.
2. **이전 product 실험은 confounded되어 있었다.** Product의 상대 순위가 global과 local loss에서 반전됐다.
3. **단순 B는 최종 대체안이 아니다.** Raw cue가 틀린 위치에서도 regularizer가 약해져 FP가 증가한다.
4. **큰 change 영역에 대한 문제 제기는 데이터에서도 나타났다.** Product-local의 이득은 큰 GT area frame에서 집중됐다.
5. 다음 단계에서는 A/B를 더 넓히기 전에 B의 local support 정의와 weight를 calibration하거나, 계획한 C growth-only regularization으로 넘어가야 한다.

## 8. 구현 및 재현 자료

- Loss 구현: `temporal/fusion.py`
- 2x2 runner: `experiments/run_oscd_loss_locality_ablation.py`
- 집계: `experiments/summarize_loss_locality_ablation.py`
- 생성 결과: `outputs/part11_loss_locality_ablation_20260813/`
- 집계 JSON: `outputs/part11_loss_locality_ablation_20260813/summary.json`
- frame별 지표: `outputs/part11_loss_locality_ablation_20260813/per_frame_metrics.csv`

생성 결과는 용량이 크고 재생성 가능하므로 git에는 포함하지 않는다.

## 9. 추가 실험: product 앞의 2를 제거한 local loss

사용자 제안에 따라 product의 detection cue에서 계수 `2`를 제거하되, local
support는 원래 product 값으로 유지했다.

```text
X = P^0.3 * S

L_raw_local
= mean(
    X * (1-m)
    + (1-X) * m
  )
```

이 조건은 다음과 같이 통제했다.

```text
training cue = X
local support = X
lambda = 1
```

따라서 support까지 `X/2`로 낮추는 confound는 넣지 않았다. Pixel별 성장 문턱은
기존 scaled local product의 `X > 1/3`에서 `X > 1/2`로 올라간다. 상대적인
detection/regularization balance는 기존 `2X` 식에서 `lambda=2`를 적용한 것과
같다.

### 9.1 Online-at-arrival 결과

| Scene | mIoU | F1 | Precision | Recall | Pred. fraction |
|---|---:|---:|---:|---:|---:|
| SC1 | 0.396 | 0.505 | 0.741 | 0.457 | 0.020 |
| SC2 | 0.643 | 0.763 | 0.812 | 0.752 | 0.055 |
| SC3 | 0.556 | 0.694 | 0.830 | 0.617 | 0.044 |
| Overall | **0.536** | **0.659** | **0.796** | **0.613** | **0.040** |

기존 scaled local product와 직접 비교하면 다음과 같다.

| Condition | mIoU | Precision | Recall | Pred. fraction |
|---|---:|---:|---:|---:|
| Local scaled product: `2X`, support `X` | **0.629** | 0.663 | **0.891** | 0.075 |
| Local raw product: `X`, support `X` | 0.536 | **0.796** | 0.613 | 0.040 |

`2`를 제거한 결과:

```text
precision:          +13.3 percentage points
recall:             -27.8 percentage points
mIoU:               -0.093
predicted fraction: -0.035
```

예상대로 FP는 감소했지만 true change growth까지 지나치게 억제했다. 특히 SC1
recall이 `0.771 -> 0.457`로 크게 떨어졌다. 기존 A global product와 비교해도
overall precision은 `+4.4 pp` 높지만 recall은 `-16.6 pp`, mIoU는 `-0.092`로
낮다. 따라서 raw local product는 A와 B 사이의 좋은 절충점이 아니다.

### 9.2 Change 면적별 영향

Raw local에서 scaled local을 뺀 arrival mIoU 차이는 다음과 같다.

| GT area group | Mean GT fraction | Raw - scaled local mIoU |
|---|---:|---:|
| Small | 0.021 | -0.002 |
| Medium | 0.053 | -0.086 |
| Large | 0.096 | **-0.193** |

계수 `2` 제거의 손실은 큰 change 영역에서 가장 컸다. 이는 local loss에서도
성장 문턱을 `1/2`까지 높이면 약한 경계와 넓은 change 내부를 충분히 채우지
못한다는 뜻이다.

### 9.3 Final rerender

```text
Raw local product final:
mIoU     = 0.587
precision = 0.815
recall    = 0.658

Scaled local product final:
mIoU     = 0.650
precision = 0.662
recall    = 0.928
```

후속 replay로 raw condition의 recall이 일부 회복되지만 scaled local보다 여전히
`26.9 pp` 낮다.

### 9.4 결론

```text
lambda=1 / threshold X>1/3:
너무 permissive해서 FP 증가

lambda=2와 동등 / threshold X>1/2:
너무 conservative해서 recall 붕괴
```

따라서 다음 calibration 후보는 두 조건 사이인 `lambda=1.25` 또는 `1.5`다.
다만 이 추가 sweep은 아직 실행하지 않았다.

## 10. 제한 사항

- 현재 결과는 seed 0 단일 실행이다.
- CUDA rasterization/densification 누적 경로에 작은 run-to-run 비결정성이 있다.
- `C_hat = clamp(C/2, 0, 1)`과 `lambda=1`만 검사했으며 local loss의 calibration sweep은 아직 수행하지 않았다.
- GT area fraction은 실제 regularizer가 사용하는 soft prediction mean의 proxy이지 동일한 값은 아니다.

## 11. Previous-update-aware element-wise growth replay

### 11.1 동기

B local은 현재 pixel의 cue가 약한 곳에서 현재 mask mass를 억제한다.

```text
X = P^0.3 * S

B(v)
= mean(2X_v * (1-m_v))
  + mean((1-X_v) * m_v)
```

하지만 B만으로는 **직전 frame update가 실제로 어디를 새로 키웠는지** 구분하지
않는다. 이를 반영하기 위해 frame `t-1`을 학습하기 직전과 직후의 rendered mask
probability 차이에서 양의 증가량만 저장한다.

```text
G_prev
= relu(m_(t-1, after) - m_(t-1, before))
```

`G_prev`는 `t-1` camera의 pixel 좌표에 놓인 `0..1` 연속값 map이며 다음 frame의
loss에서는 detach된 evidence로 사용한다. 현재 구현의 raw `G_prev`는 이론적으로
`0..1`이지만 실제 관측 global maximum은 약 `0.231`이었다.

### 11.2 정확한 online update 규칙

기존 O-SCD replay sampler는 유지했다. 일반 과거 view가 뽑히면 해당 view에 B만
적용한다. 최신 view `t`가 뽑히고 `t > 0`이면 같은 optimizer step에서 `t-1`도
렌더링하고 다음 loss를 사용한다.

```text
L_latest
= 0.5 * (B(t) + B(t-1))
  + lambda_g * mean(
      G_prev
      * (1-X_(t-1))
      * m_(t-1, current)
    )
```

첫 frame은 비교할 이전 update가 없으므로 B만 사용한다. 추가 항은 다음 세 조건을
동시에 만족하는 pixel만 강하게 억제한다.

1. 직전 update에서 실제로 mask가 증가했다: `G_prev > 0`
2. 그 위치의 직전 local cue가 약하다: `1-X_(t-1)`이 크다
3. 현재 다시 렌더해도 그 activation이 남아 있다: `m_(t-1, current)`가 크다

즉 scalar global brake가 아니라, **직전 update가 만든 unsupported growth를 그
pixel에서 다시 검사하는 element-wise regularizer**다. `t-1`은 이 joint step에서
추가로 최적화되지만, 원래 replay sampler에서도 과거 frame이 다시 선택될 수
있었으므로 새로운 기능의 핵심은 재방문 자체보다 `G_prev` 기반 spatial weighting에
있다.

## 12. `G_prev` 크기

`lambda_g=1` 실행의 304 frame audit 결과는 다음과 같다.

| 항목 | 값 |
|---|---:|
| `G_prev` 전체 pixel 평균 | 0.000754 |
| `G_prev > 0` pixel 비율 평균 | 0.1464 |
| frame별 `G_prev` maximum의 global maximum | 0.2311 |
| joint step에서 raw growth-replay loss 평균 | 0.000229 |
| 최신-view joint step 수 | 1,745 |

값이 작기 때문에 `lambda_g=1`에서는 추가 항의 영향이 약했다.

## 13. Weight 1 결과와 weight sweep

### 13.1 `lambda_g=1`

전체 304 frame 평균은 다음과 같다. B local은 같은 `2X` product-local loss이고,
G-prev 조건만 two-view joint loss와 추가 growth replay를 사용한다.

| Condition | mIoU | F1 | Precision | Recall | Pred. fraction |
|---|---:|---:|---:|---:|---:|
| Original global | **0.640891** | **0.756752** | **0.711369** | 0.839314 | 0.065214 |
| B local product | 0.628693 | 0.746648 | 0.663285 | **0.891142** | 0.075028 |
| G-prev `w=1` | 0.626635 | 0.745049 | 0.659280 | 0.888188 | 0.074794 |

`w=1`은 B 대비 arrival mIoU `-0.0021`, Original 대비 `-0.0143`이었다. Raw
growth-replay term의 평균이 작아 weight 1만으로는 유의미한 억제력을 만들지 못했다.

### 13.2 `lambda_g` sweep

SC1/2/3, seed 0, frame당 16 updates의 동일 조건에서 다음 weight를 검사했다.
선택 기준은 primary metric인 **online-at-arrival frame 평균 mIoU**다.

| `lambda_g` | mIoU | Precision | Recall | Pred. fraction |
|---:|---:|---:|---:|---:|
| 0 | 0.629910 | 0.660417 | 0.893215 | 0.075047 |
| 1 | 0.626635 | 0.659280 | 0.888188 | 0.074794 |
| 3 | 0.626178 | 0.657036 | 0.891510 | 0.075021 |
| 5 | 0.629345 | 0.659505 | 0.894129 | 0.074883 |
| 6 | 0.628324 | 0.660419 | 0.889168 | 0.074879 |
| 7 | 0.635627 | 0.668135 | 0.891272 | 0.074187 |
| **7.5** | **0.636325** | **0.668535** | 0.891386 | 0.074047 |
| 8 | 0.632972 | 0.668058 | 0.888879 | 0.073696 |
| 9 | 0.635261 | 0.667687 | 0.891369 | 0.073897 |
| 10 | 0.635604 | 0.667586 | **0.891856** | 0.073965 |
| 15 | 0.632190 | 0.666910 | 0.886955 | 0.073396 |
| 20 | 0.623529 | 0.677538 | 0.862326 | 0.070447 |
| 30 | 0.625882 | 0.684089 | 0.854803 | 0.070017 |
| 100 | 0.613438 | 0.725436 | 0.791811 | 0.060866 |

이번 sweep에서는 `7..10`이 안전한 plateau였고 `7.5`가 최고 mIoU였다. Recall은
`15`에서 감소하기 시작하고 `20` 이상에서 명확하게 무너졌다. `w=10`의 recall이
`w=7.5`보다 `+0.00047` 높은데도 mIoU가 `-0.00072` 낮은 이유는 precision이
`-0.00095` 낮기 때문이다.

이 결과를 exact optimum으로 해석해서는 안 된다. 동일 세 scene, seed 0 한 번의
sweep이므로 현재 선택은 **provisional default `lambda_g=7.5`**, 안정 구간은
`7..10`이다.

## 14. 선택된 `lambda_g=7.5` 결과

### 14.1 전체 결과

| Phase | mIoU | F1 | Precision | Recall | Pred. fraction |
|---|---:|---:|---:|---:|---:|
| Online-at-arrival | 0.636325 | 0.752690 | 0.668535 | 0.891386 | 0.074047 |
| Final rerender | 0.660280 | 0.770979 | 0.674950 | 0.923567 | 0.076688 |

Arrival 기준 비교:

```text
vs B local product:
  mIoU      +0.007632
  precision +0.005250
  recall    +0.000244

vs Original global:
  mIoU      -0.004565
  precision -0.042834
  recall    +0.052071
```

`w=0`은 같은 two-view optimization을 사용하되 `G_prev` 항만 끈 대조군이다.
`w=7.5`는 `w=0`보다 arrival mIoU가 `+0.006416` 높지만 final rerender mIoU는
사실상 같다(`0.660280` vs `0.660310`). 따라서 `G_prev`의 주된 이점은 미래
replay가 들어오기 전 **즉시 causal output**을 개선하는 데 있다고 해석할 수 있다.

### 14.2 Scene별 Original 비교

| Scene | Original mIoU | G-prev `w=1` | G-prev `w=7.5` | `w=7.5 - Original` |
|---|---:|---:|---:|---:|
| SC1 | 0.599752 | **0.605613** | 0.605416 | +0.005663 |
| SC2 | **0.697250** | 0.639478 | 0.661513 | -0.035737 |
| SC3 | 0.622288 | 0.632936 | **0.639344** | +0.017056 |
| Overall | **0.640891** | 0.626635 | 0.636325 | -0.004565 |

SC1과 SC3는 Original보다 개선됐지만 SC2에서 `-0.0357` 떨어져 전체 평균을
끌어내렸다. 따라서 `w=7.5`는 B local의 FP를 줄이는 데 의미가 있지만 아직
Original global을 안정적으로 대체하지 못한다. 다음 분석의 우선순위는 SC2에서
`G_prev`, local support, visibility/occlusion error가 어떻게 겹치는지 확인하는 것이다.

## 16. 구현, 검증, 현재 판단

- Element-wise loss helper: `temporal/fusion.py`의 `compute_growth_replay_regularization`
- Runner: `experiments/run_oscd_two_view_growth_replay_ablation.py`
- Unit test: `tests/temporal/test_cue_temporal_runner.py`
- `w=1` 실행: `/tmp/oscd_two_view_growth_replay_scene123_20260814_v1/`
- Weight sweep: `/tmp/oscd_growth_replay_weight_sweep_20260814_v1/`
- Sweep 표: `/tmp/oscd_growth_replay_weight_sweep_20260814_v1/coarse_summary.csv`
- 선택 결과: `/tmp/oscd_growth_replay_weight_sweep_20260814_v1/selected_w7p5.json`
- 전체 test suite 재검증: `169 passed` (2026-08-14, `oscd` environment)

현재 판단은 다음과 같다.

1. `G_prev` element-wise replay는 B local 대비 arrival mIoU와 precision을 함께 개선했다.
2. 이 효과는 `w=0` 대조군과 비교해도 나타나므로 단순 two-view 재최적화만의 효과는 아니다.
3. `lambda_g=7.5`는 이번 sweep의 실용적 선택이지만 아직 held-out scene이나 multi-seed 검증을 거치지 않았다.
4. Original 대비 recall은 높지만 precision이 낮고, 특히 SC2 실패 때문에 overall mIoU는 아직 Original보다 낮다.
5. 따라서 다음 단계는 weight를 더 미세하게 튜닝하기보다 SC2의 unsupported growth를 visibility/multi-view support로 구분하는 것이다.

## 17. Sum cue에 동일한 `G_prev` regularization을 적용한 통제 비교

Product에만 특화된 효과인지 확인하기 위해 동일한 two-view update와
`lambda_g=7.5`를 Sum cue에도 적용했다. 세 조건을 fresh run으로 다시 실행했다.

```text
Sum G0:
  C = P+S
  support = C/2
  lambda_g = 0

Sum G7.5:
  C = P+S
  support = C/2
  lambda_g = 7.5

Product G7.5:
  C = 2*P^0.3*S
  support = P^0.3*S
  lambda_g = 7.5
```

세 조건 모두 최신 view가 선택되면 `0.5*(B(t)+B(t-1))`를 사용하므로, Sum의
`G0 -> G7.5`는 growth-replay 항 자체의 효과를 분리한다.

### 17.1 Overall online-at-arrival

| Condition | mIoU | F1 | Precision | Recall | Pred. fraction |
|---|---:|---:|---:|---:|---:|
| Sum + G0 | 0.543425 | 0.677788 | 0.558285 | **0.915924** | 0.091221 |
| Sum + G7.5 | 0.559660 | 0.692306 | 0.576328 | 0.910303 | 0.088026 |
| Product + G7.5 | **0.636383** | **0.752451** | **0.667219** | 0.894009 | **0.074272** |

Sum에서 `G_prev`를 켠 순수 효과는 다음과 같다.

```text
Sum G7.5 - Sum G0:
  mIoU              +0.016235
  precision         +0.018043
  recall            -0.005621
  predicted fraction -0.003195
```

즉 `G_prev`는 Sum에서도 unsupported growth를 실제로 억제하며 precision과 mIoU를
개선했다. 하지만 Product와 비교하면 다음 차이가 남는다.

```text
Product G7.5 - Sum G7.5:
  mIoU              +0.076724
  precision         +0.090891
  recall            -0.016294
  predicted fraction -0.013754
```

Sum은 P 또는 S 중 하나만 높아도 local support가 높아진다. 따라서 cue가 지지하는
FP에서는 `1-support`가 작아져 `G_prev` 항도 약해진다. `G_prev`는 cue 밖으로 번진
growth는 줄이지만, Sum cue 자체가 만든 diffuse FP까지 제거하지는 못한다.

### 17.2 Scene별 arrival mIoU

| Scene | Sum G0 | Sum G7.5 | Product G7.5 | Sum의 G 효과 |
|---|---:|---:|---:|---:|
| SC1 | 0.501145 | 0.505799 | **0.607651** | +0.004654 |
| SC2 | 0.573607 | 0.599657 | **0.660397** | +0.026050 |
| SC3 | 0.551783 | 0.568775 | **0.638594** | +0.016992 |

Sum의 `G_prev` 이득은 세 scene 모두 양수였고 SC2에서 가장 컸다. 그러나 Product는
모든 scene에서 Sum G7.5보다 `0.061..0.102` 높은 mIoU를 보였다.

### 17.3 Final rerender

| Condition | mIoU | Precision | Recall |
|---|---:|---:|---:|
| Sum + G0 | 0.554238 | 0.557664 | 0.939049 |
| Sum + G7.5 | 0.564736 | 0.568841 | **0.940245** |
| Product + G7.5 | **0.656933** | **0.671736** | 0.921448 |

Final에서도 Sum G7.5가 G0보다 mIoU `+0.010498` 높았지만 Product G7.5보다
`-0.092197` 낮았다.

### 17.4 결론과 산출물

1. `G_prev` regularization은 Product에만 유효한 기법이 아니며 Sum에서도 작동한다.
2. 그러나 최종 성능 차이의 더 큰 부분은 cue selectivity에서 나온다.
3. Sum은 높은 recall을 유지하지만 cue-supported FP 때문에 precision이 낮다.
4. 따라서 현재 선택은 계속 **Product + B local + G-prev 7.5**다.
5. Sum에 별도 weight sweep을 하면 현재 수치보다 개선될 수 있지만, `7.5`에서도 Product와의 mIoU 차이가 `0.0767`이므로 우선순위는 낮다.

```text
/tmp/oscd_sum_vs_product_gprev_comparison_20260814_v1/
  summary.json
  per_frame_metrics.csv
  sum_g0/scene_change1..3/
  sum_g7p5/scene_change1..3/
  product_g7p5/scene_change1..3/
```
