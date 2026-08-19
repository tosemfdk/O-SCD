# Part 10. Change cue fusion experiment: sum vs. product

> 작성일: 2026-08-13
> 범위: `R_ref -> scene_change1`, `R_ref -> scene_change2`, `R_ref -> scene_change3` 독립 실행
> 핵심 결론: **product fusion은 false positive를 줄이는 대신 false negative를 만들기 쉬운 precision-biased AND gate다. 현재 online 기준 기본값은 `P + S`를 유지한다.**

## 1. 질문

기존 O-SCD change cue는 다음 두 신호를 더한다.

- `P`: 현재 RGB와 reference render의 픽셀 차이
- `S`: SAM2.1 feature 차이

`S`는 `64 x 64` feature map을 bilinear upsampling하므로 물체의 미세 경계가 흐리다. 그래서 처음 세운 가설은 다음과 같았다.

> `P`와 `S`를 더하지 않고 곱하면, pixel error가 거의 없는 영역을 제거하여 false positive를 줄일 수 있지 않을까?

이 실험에서는 그 가설이 cue 이미지뿐 아니라 실제로 학습된 3D `R_change` mask에서도 성립하는지 확인했다.

## 2. Cue 정의

픽셀 cue는 프레임별 min-max normalization을 적용한다.

$$
P = \operatorname{norm}\left(0.8\lVert I-R_{ref}\rVert_1 + 0.2(1-\operatorname{SSIM}(I,R_{ref}))\right)
$$

SAM2.1 cue는 두 영상의 Hiera Tiny embedding 차이를 채널 평균하고 정규화한 뒤 원래 처리 해상도로 upsampling한다.

$$
S = \operatorname{upsample}\left(\operatorname{norm}\left(\operatorname{mean}_{c}\lvert F_I-F_{ref}\rvert\right)\right)
$$

비교한 fusion은 다음과 같다.

$$
C_{sum}=P+S
$$

$$
C_{avg}=\frac{P+S}{2}
$$

$$
C_{prod}=2P^{0.3}S
$$

`P^0.3`은 작은 양의 pixel cue를 증폭하여 단순 `P*S`보다 recall을 보존하려는 변환이다. 앞의 계수 `2`는 product cue의 크기를 기존 `P+S`와 비슷한 범위로 맞춘다.

### 2.1 원래 cue의 구성

`P`는 경계가 세밀하지만 조명, texture, reference rendering error에도 반응한다. `S`는 물체 단위 응답이 강하지만 upsampling된 저해상도 feature라 경계가 흐리다.

### 2.2 중간에 검토한 sigmoid gate

다음 gate도 cue-only probe로 확인했다.

$$
g(P)=\sigma(a(P-k)),\qquad C=g(P)S
$$

`k=0.1`은 지나치게 permissive하고, `k=0.2`는 작은 pixel evidence를 빠르게 제거한다. 한 프레임에서 thresholded cue가 좋아 보여도 이후의 3D SSF 학습 문제를 해결하지는 않으므로 최종 비교에서는 제외했다. 이 probe의 수치는 **cue-only threshold 결과**이지 학습된 `R_change` 성능이 아니다.

## 3. 중요한 점: SSF는 cue reconstruction loss가 아니다

학습되는 change render의 채널 평균을 `z`, sigmoid 출력을 `q`, 전체 평균을 `mu`라고 두면 현재 목적함수는 다음과 같다.

$$
q_i=\sigma(z_i),\qquad \mu=\frac{1}{N}\sum_i q_i
$$

$$
L=\frac{1}{N}\sum_i C_i(1-q_i)
+\lambda\log(\mu^2+1.000000001)
$$

따라서 이 학습은 `z`가 cue `C`를 픽셀별로 그대로 복원하도록 만드는 BCE/L1 loss가 아니다. 한 픽셀에서의 gradient 부호는 다음처럼 결정된다.

$$
\frac{\partial L}{\partial z_i}
=\frac{q_i(1-q_i)}{N}
\left[-C_i+\lambda\frac{2\mu}{\mu^2+1.000000001}\right]
$$

즉 해당 픽셀의 score가 올라가려면 대략 다음 조건이 필요하다.

$$
C_i>\lambda\frac{2\mu}{\mu^2+1.000000001}
$$

초기 `mu`가 약 `0.5`이고 `lambda=1`이면 balance threshold는 약 `0.8`이다. 그래서 시각적으로 비슷해 보이는 두 cue도 `0.8` 위의 면적이 조금 달라지면 gradient의 방향과 누적량이 크게 달라질 수 있다.

### 3.1 Average는 사실상 별도 방법이 아니다

`C_avg=(P+S)/2`, `lambda=0.5`이면

$$
L(C_{avg},0.5)=\frac{1}{2}L(C_{sum},1)
$$

이다. 최적점과 gradient 방향은 같고 전체 scale만 절반이다. Adam의 epsilon과 수치 순서 때문에 실행값이 조금 달라질 수 있지만, fusion 자체의 새로운 가설로 볼 이유는 없다. 이후 비교의 중심은 `P+S`와 `2P^0.3S`다.

### 3.2 왜 product 앞에 `2`가 필요한가

검출항은 cue에 선형이므로

$$
D(2C)+R=2\left[D(C)+0.5R\right]
$$

이다. 따라서 `2P^0.3S + R`은 scale 관점에서 `P^0.3S + 0.5R`과 같은 balance를 갖는다. 반대로 raw product `P^0.3S + R`은 regularization이 상대적으로 두 배 강한 조건이다.

## 4. 비교 조건

| 항목 | 조건 |
|---|---|
| sequence | 각 scene을 `R_ref -> scene_changeX`로 독립 실행; `1 -> 2 -> 3` 연속 학습 아님 |
| pose/cue | 동일 fixed pose와 cached cue |
| seed | 0 |
| resolution | 4 (`941 x 528`) |
| online budget | 이미지당 16 updates |
| sampling | 최신 view 약 33%, 나머지는 지금까지 본 view replay |
| trainable parameters | xyz, DC, SH-rest, opacity, scale, rotation |
| topology update | online clone/split densification 사용, pruning 없음 |
| binary output | raw rendered score `z > 0.5` |
| primary phase | `online_at_arrival`: 해당 프레임이 도착한 직후의 causal 결과 |
| diagnostic phase | `online_final_rerender`: 전체 sequence 학습 후 모든 과거 view 재렌더 |

Online SCD의 주평가는 `online_at_arrival`이다. `online_final_rerender`는 이후 프레임의 학습까지 반영하므로 과거 시점의 causal output이 아니라, 해당 cue가 장기적으로 3D field에 학습될 수 있었는지를 보는 진단값이다.

## 5. 정량 결과

### 5.1 Online at arrival

| Scene | Fusion | mIoU | F1 | Precision | Recall |
|---|---|---:|---:|---:|---:|
| SC1 | `P+S`, `lambda=1` | **0.591** | **0.686** | 0.676 | **0.748** |
| SC1 | `2P^0.3S`, `lambda=1` | 0.550 | 0.657 | **0.712** | 0.668 |
| SC2 | `P+S`, `lambda=1` | 0.682 | 0.798 | 0.710 | **0.929** |
| SC2 | `2P^0.3S`, `lambda=1` | **0.712** | **0.825** | **0.771** | 0.903 |
| SC3 | `P+S`, `lambda=1` | **0.660** | **0.788** | 0.739 | **0.860** |
| SC3 | `2P^0.3S`, `lambda=1` | 0.651 | 0.776 | **0.784** | 0.787 |

Arrival 기준 product의 일관된 효과는 다음과 같다.

- SC1: precision `+3.6 pp`, recall `-7.9 pp`, mIoU `-0.041`
- SC2: precision `+6.1 pp`, recall `-2.6 pp`, mIoU `+0.029`
- SC3: precision `+4.5 pp`, recall `-7.4 pp`, mIoU `-0.009`

즉 세 scene 모두 FP 쪽은 개선되지만 FN 쪽은 악화된다. SC2에서는 precision 이득이 더 커서 최종 mIoU도 올랐지만, SC1과 SC3에서는 recall 손실을 상쇄하지 못했다.

### 5.2 Final rerender

| Scene | Fusion | mIoU | F1 | Precision | Recall |
|---|---|---:|---:|---:|---:|
| SC1 | `P+S`, `lambda=1` | **0.664** | **0.740** | 0.689 | **0.811** |
| SC1 | `2P^0.3S`, `lambda=1` | 0.642 | 0.730 | **0.717** | 0.760 |
| SC2 | `P+S`, `lambda=1` | 0.707 | 0.823 | 0.722 | **0.970** |
| SC2 | `2P^0.3S`, `lambda=1` | **0.739** | **0.845** | **0.780** | 0.929 |
| SC3 | `P+S`, `lambda=1` | 0.667 | 0.792 | 0.708 | **0.922** |
| SC3 | `2P^0.3S`, `lambda=1` | **0.708** | **0.822** | **0.794** | 0.867 |

SC2와 SC3에서는 이후 frame의 multiview evidence가 더 쌓인 뒤 product가 높은 precision으로 이득을 본다. 반면 SC1에서는 끝까지 sum이 더 좋다. 따라서 product는 “항상 나쁜 cue”도 “항상 더 좋은 cue”도 아니다. **scene-dependent precision/recall trade-off**다.

## 6. Scene change 1, frame 56 심층 분석

이 프레임은 product input cue가 물체를 분명히 강조하는데도 learned `R_change`가 일부 물체를 놓친 대표 사례다.

| 값 | Sum | Product |
|---|---:|---:|
| GT 내부 cue 평균 | 1.076 | 0.995 |
| GT 내부 learned score 평균 | 0.789 | 0.581 |
| frame IoU | 0.750 | 0.574 |
| frame F1 | 0.857 | 0.729 |
| GT에서 `C > 0.8` | 91.52% | 83.05% |
| BG에서 `C > 0.8` | 4.94% | 2.72% |

두 input cue의 전체 pixel Pearson correlation은 `0.988`로 매우 높았다. 그러나 product가 없앤 GT support의 `8.47 pp`는 FN으로 이어졌고, learned score 차이는 input cue 평균 차이보다 훨씬 커졌다.

이 현상은 다음 세 가지가 함께 만든다.

1. **AND gate:** `P` 또는 `S` 중 하나만 낮아도 product가 작아진다. `P^0.3`은 작은 `P`를 키울 수 있지만 낮거나 0인 `S`를 복구하지 못한다.
2. **Shared 3D field:** 입력 cue는 현재 2D mask가 아니라 여러 view가 공유하는 Gaussian parameter에 대한 supervision이다. 한 view에서 cue가 높아도 다른 view에서 같은 Gaussian을 지지하지 않으면 render score가 충분히 올라가지 않는다.
3. **Sparse latest-view updates:** frame 56 도착 시 seed 0의 16회 update 중 frame 56 자체는 2회만 선택됐다. 나머지는 이전 view replay였다. 전체 sequence 뒤에는 frame 56이 7회 더 replay되어 product의 누락 일부가 회복됐지만, sum보다 약했다.

이 때문에 “현재 cue에서 하이라이트됨”과 “현재 `R_change` render에서 mask로 나옴”은 같은 사건이 아니다.

## 7. Raw product를 두 배 하지 않으면 왜 실패하는가

`C=P^0.3S`, `lambda=1`을 scene change 1의 95장에 대해 offline으로 정확히 이미지당 120회, 총 11,400 updates 학습했다.

| 항목 | 결과 |
|---|---:|
| mean IoU / F1 | 0 / 0 |
| non-empty prediction frames | 0 / 95 |
| global rendered-score max | 0.122 |
| frame 56 GT cue mean | 0.498 |
| frame 56 cue max | 0.876 |
| frame 56 GT에서 `C > 0.8` | 0.179% |

학습 횟수를 늘려도 cue 대부분이 regularization balance threshold 아래에 있으므로 score를 올리는 gradient가 부족했다. 즉 raw product의 실패는 단순히 online update 수가 적어서만 생긴 것이 아니라 **현재 loss에 대한 cue scale mismatch**다.

이 offline run은 xyz/DC/SH/opacity/scale/rotation과 densification/pruning을 포함했다. 다만 stock opacity reset 뒤 `min_opacity=0.4` pruning이 약 38만 Gaussian을 582개까지 붕괴시켜 CUDA failure를 만들었으므로, 성공한 exact-120 run에서는 opacity reset만 끄고 densification/pruning은 유지했다. 따라서 이 결과는 opacity-reset caveat를 포함한 stress test다.

## 8. Regularization을 0.5배 하면 되는가

`2P^0.3S`의 regularization만 `1 -> 0.5`로 낮추면 recall은 거의 포화하지만 background도 대량 활성화된다.

### Online at arrival: `2P^0.3S`, `lambda=0.5`

| Scene | mIoU | Precision | Recall | Predicted fraction |
|---|---:|---:|---:|---:|
| SC1 | 0.439 | 0.446 | 0.833 | 0.083 |
| SC2 | 0.454 | 0.457 | 0.991 | 0.134 |
| SC3 | 0.467 | 0.472 | 0.965 | 0.126 |

`lambda=0.5`는 FN을 줄이지만 FP를 너무 크게 늘려 세 scene 모두 mIoU를 낮췄다. 따라서 문제는 “regularization이 무조건 너무 세다”가 아니라, **현재 one-sided detection term과 global sparsity regularizer 사이에서 FP와 FN을 동시에 제어하기 어렵다**는 것이다.

## 9. 구현상 추가 주의점

- Renderer는 loss 전에 `z`를 `[0,1]`로 clamp한다. pre-clamp 값이 음수로 내려간 영역은 gradient가 막혀 다시 살리기 어려울 가능성이 있다. 이것은 product의 low-score pixel이 더 많은 현상을 악화할 수 있지만, 이번 실험만으로 primary cause라고 단정하지 않는다.
- Loss는 `q=sigmoid(z)`를 사용하지만 최종 mask는 raw `z>0.5`로 만든다. 학습 공간과 평가 threshold 공간이 정확히 일치하지 않는다.
- Fusion에 따라 xyz, opacity, scale, rotation과 densification trajectory도 달라진다. 비슷한 현재 cue가 반드시 비슷한 Gaussian field로 수렴하지 않는다.

## 10. 결론과 현재 결정

1. **기본 online cue는 `C=P+S`, `lambda=1`을 유지한다.** Arrival에서 더 안정적으로 recall을 보존하며 세 scene 중 두 scene에서 product보다 mIoU가 높다.
2. **`2P^0.3S`, `lambda=1`은 precision-biased alternative로 보관한다.** SC2 arrival과 SC2/SC3 final rerender에서는 이득이 있으므로 완전히 폐기할 방법은 아니다.
3. **raw `P^0.3S + R`은 사용하지 않는다.** 현재 SSF scale에서 거의 모든 cue가 regularization balance threshold 아래로 내려가 empty prediction으로 붕괴했다.
4. **product의 regularization을 단순히 0.5배 하지 않는다.** FN은 회복하지만 FP가 폭증한다.
5. **Average control은 이후 표에서 제외해도 된다.** `(P+S)/2 + 0.5R`은 `P+S+R`의 전체 scale만 바꾼 조건이다.
6. Product의 핵심 한계는 분명하다. **FP를 낮추려면 한 cue가 약한 위치를 지워야 하지만, 그 위치가 실제 object boundary 또는 한 view에서만 약한 true change이면 바로 FN이 된다.**

다음 fusion 개선은 hard AND가 아니라 `P`의 fine boundary를 보존하는 residual/soft gate와, cue fitting 및 sparsity를 분리한 loss를 검토하는 편이 타당하다.

## 12. 재현 자료

문서화 시점의 원본 수치 분석 산출물은 다음 경로에 있다. `/tmp`는 비영속 경로다.

```text
/tmp/oscd_scene13_standalone_fusion_comparison/
/tmp/oscd_scene123_2xpower_reg0p5_comparison/summary.json
/tmp/oscd_scene123_cue_to_continuous_score_videos/
/tmp/oscd_scene1_product1x_reg1_offline_exact120_noreset/
```

지정 Notion 기록:

- [SCD & NBV Part 10. change cue experiment](https://app.notion.com/p/SCD-NBV-Part-10-change-cue-experiment-3bbcbb7d793780d5a5d3e8f461b80ca8)
