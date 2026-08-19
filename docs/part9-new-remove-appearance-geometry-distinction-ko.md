# Part 9. New / Remove / Appearance 변화 구분과 geometry 초기화 방향

> 작성일: 2026-08-12
> 상태: 탐색 실험. 자동 change-event detection, 자동 new-object seeding, 최종
> 온라인 알고리즘은 아직 구현하지 않았다.
> Notion mirror: [SCD & NBV Part 9](https://app.notion.com/p/SCD-NBV-Part-9-New-Remove-and-appearance-geometry-change-distinguish-3bacbb7d793780c9bfdbdff5a0577d28)

## 1. 연구 질문

현재 프로젝트의 다음 질문은 단순히 change cue가 높은 위치를 찾는 것이 아니다.

> Reference 3DGS와 현재 observation 사이의 변화가 **new object**, **removed
> object**, 또는 **동일 표면의 appearance-only change** 중 무엇인지 구분할 수
> 있는가?

이 구분은 geometry를 언제 생성하거나 수정해야 하는지를 결정한다.

| 변화 종류 | Reference surface | Current observation surface | 필요한 처리 |
|---|---:|---:|---|
| Added / new | 없음 | 있음 | 새로운 geometry seed 후보 |
| Removed | 있음 | 없음 | 새 geometry를 만들지 않고 removal 상태 기록 |
| Appearance-only | 있음 | 있음 | 기존 표면을 유지하고 appearance/change feature만 갱신 |
| False positive / uncertainty | 불명확 | 불명확 | geometry update 보류 |

이 문서의 목적은 최종 분류기를 제안하는 것이 아니라, SAM feature difference에
이 세 변화 형태를 분리할 수 있는 신호가 실제로 존재하는지 확인하는 것이다.

---

## 2. 출발점: state별 geometry를 직접 학습했을 때의 문제

E3 실험에서는 고정 Gaussian index 위에 state별 `xyz`, DC, opacity, scale,
rotation delta를 두고 O-SCD cue로 학습했다. 정량 결과는 다음과 같았다.

| 모델 | Overall mIoU | F1 |
|---|---:|---:|
| DC-only lifespan | **0.6245** | **0.7460** |
| Geometry + S0 anchor | 0.6191 | 0.7364 |

Geometry는 cue SSF loss를 조금 낮추고 recall을 높였지만 false positive를
`+441,005` pixel 증가시켰다. 자세한 결과는
[`instance1-state-geometry-lifespan-comparison-ko.md`](instance1-state-geometry-lifespan-comparison-ko.md)에
기록되어 있다.

정량 성능뿐 아니라 depth render를 확인했을 때 state-specific `R_change`의
geometry가 실제 표면을 안정적으로 표현하지 못하고 크게 변형된 모습이
관찰되었다.

이 결과는 모든 change cue pixel을 설명하기 위해 기존 Gaussian의 xyz, opacity,
scale, rotation을 자유롭게 움직이는 방식이 안전하지 않다는 것을 보여준다.
특히 다음 조건이 문제를 키운다.

1. 2D cue가 실제 물체보다 넓고 false positive를 포함한다.
2. 고정 topology에는 새 물체를 설명할 Gaussian support가 충분하지 않을 수 있다.
3. 새 물체가 없던 위치를 기존 reference Gaussian 이동만으로 설명하려 하면
   주변 geometry가 끌려간다.
4. opacity까지 자유롭게 최적화하면 cue loss를 낮추기 위해 잘못된 surface가
   나타나거나 사라질 수 있다.
5. 낮은 2D cue loss는 올바른 3D geometry를 보장하지 않는다.

따라서 현재 기본 표현은 **DC-only fixed-topology lifespan**으로 유지하고,
geometry 생성은 모든 change에 적용하지 않고 **new object로 판정된 영역에만
제한적으로 seed를 넣는 방향**이 더 타당하다.

---

## 3. Geometry seeding 전에 해결해야 하는 병목

New-object seed를 선택적으로 추가하려면 cue 내부의 변화 방향을 먼저 알아야
한다.

```text
O-SCD cue
    |
    +-- reference에만 surface 존재  -> removed, seed 불필요
    +-- inference에만 surface 존재  -> new/added, seed 필요
    +-- 양쪽에 동일 surface 존재    -> appearance-only, seed 불필요
    +-- 어느 쪽도 일관되지 않음     -> false positive / 보류
```

초기에는 다음 방법들을 검토했다.

- `R_ref`와 `R_change` rendered depth 비교
- DA3 depth와 3DGS depth의 scale/shift alignment 후 signed residual 비교
- SLIC로 cue와 reference/inference object boundary의 일치도 비교
- cue 내부 saliency/foreground-background separation

그러나 learned geometry의 depth 자체가 불안정했고, monocular depth의 scale 및
surface error가 추가되었다. SLIC는 cue refinement에는 유용했지만 add/remove
방향을 직접 주지 못했다. 이후 SCaR-3D의 **signed feature direction**을 O-SCD에
적용하는 방향으로 전환했다.

---

## 4. Signed SAM2 feature difference

원 SCaR-3D는 EfficientSAM을 사용하지만, 이번 probe는 기존 O-SCD와 같은
**SAM2.1 Hiera Tiny** image embedding을 사용했다.

같은 camera pose에서:

```text
Reference 3DGS render -> F_ref   [256, 64, 64]
Inference RGB         -> F_inf   [256, 64, 64]
```

각 위치의 signed feature difference는 다음과 같다.

\[
\Delta f_v(p)=f_{ref,v}(p)-f_{inf,v}(p).
\]

Absolute feature norm만 사용하면 변화 크기는 알 수 있지만 방향은 사라진다.
따라서 여러 view의 모든 `Delta f`를 모아 PCA를 수행했다.

\[
v_k
=
\arg\max_{\lVert v\rVert=1,\;v\perp v_1\ldots v_{k-1}}
\operatorname{Var}(\Delta f^\top v).
\]

각 view와 pixel의 signed projection은:

\[
D_{v,k}(p)=\Delta f_v(p)^\top v_k
\]

이다. 코드 수준에서는 다음 연산이다.

```python
# delta: [N, 256, H, W]
# pcs:   [256, K]
scores = torch.einsum("nchw,ck->nkhw", delta, pcs)
```

`PC1`은 기존 SAM channel 하나가 아니라 256 channel의 선형 결합이며, `scores`가
실제 `256 -> K` 차원 축소 결과다.

### 4.1 전체-feature PCA에서 Delta-PCA로 변경

첫 probe는 `F_ref`와 `F_inf` 전체 feature의 covariance에서 PC1을 구한 뒤
`Delta f`를 투영했다. 이후 연구 질문에 더 직접적으로 맞도록 PCA 입력 자체를
`Delta f`로 바꿨다.

Scene-change 3의 23-view 비교에서:

| 항목 | 기존 feature PCA | Delta-PCA |
|---|---:|---:|
| PC1 explained variance | 14.18% | **16.47%** |
| 두 PC1의 cosine similarity | \- | 0.933 |
| 두 signed map의 correlation | \- | 0.993 |

즉 formulation은 Delta-PCA가 더 정확하지만, 이 데이터에서는 두 방향이 이미
매우 유사했다.

위 비교는 동일 frame에서 PCA basis를 전체 feature로 구한 경우와 feature
difference 자체로 구한 경우의 signed projection 및 directional mask를 함께
보여준다. 수치상 높은 correlation과 마찬가지로 주요 old/new lobe가 시각적으로도
유지된다.

---

## 5. Structural add/remove 방향 probe

Scene-change 3의 sampled 23 views에 공통 Delta-PC1을 적용하고 기존 O-SCD cue
`>= 0.5` 내부를 negative/positive 방향으로 나눴다.

각 directional mask를 reference Gaussian footprint에 투표하고 Gaussian center가
여러 view에서 해당 mask 안에 남는지 pruning했다.

```text
2D directional mask
    -> reference Gaussian footprint voting
    -> selected reference GS
    -> multi-view projected-center consistency pruning
    -> retention rate
```

Frame 75의 시각적 old/new lobe와 retention 순위가 일치했다.

| Direction | Selected GS | Kept GS | Retention | 해석 |
|---|---:|---:|---:|---|
| Negative / blue | 69,261 | 462 | 0.00667 | post/current-side, added 후보 |
| Positive / red | 106,687 | 2,037 | **0.01909** | pre/reference-side, removed 후보 |

PCA eigenvector의 부호는 임의이므로 `positive=removed`를 고정 규칙으로 사용할 수
없다. 각 실행에서 reference-GS retention으로 어느 부호가 pre-state인지 다시
정해야 한다.

상세 분석은 한 view의 signed Delta-PC1, cue로 제한한 positive/negative
mask, 각 mask가 선택한 reference Gaussian, multi-view pruning 후 살아남은
Gaussian을 같은 화면에 대응시킨 것이다. 따라서 retention 수치가 어느 2D lobe와
3D reference support에서 나온 것인지 직접 확인할 수 있다.

또한 retention 절대값은 매우 낮다. 현재 O-SCD cue가 넓고 false positive를 많이
포함하며, local FastGS hit-count가 SCaR-3D의 정확한 alpha-transmittance weight는
아니기 때문이다. 이 결과는 **signed 2D direction이 존재한다**는 probe이지,
완성된 add/remove detector가 아니다.

---

## 6. Appearance 변화가 다른 Delta-PC에 존재하는가

PC1은 가장 큰 feature-difference 분산을 설명하므로 넓은 structural movement와
old/new lobe가 지배할 수 있다. 동일 surface의 appearance-only 변화는 더 작은
분산 방향인 PC2 이후에 분리될 수 있다는 가설을 검증했다.

Scene-change 2의 sampled views에서 Delta-PC1~PC10을 구하고 수동 polygon으로
appearance ROI를 지정했다. Stable 영역의 magnitude로 정규화하여 PC 사이 scale
차이를 보정했다.

\[
R_{A,k}
=
\frac{\operatorname{mean}_{p\in A}|D_k(p)|}
{\operatorname{mean}_{p\in S}|D_k(p)|},
\qquad
R_{G,k}
=
\frac{\operatorname{mean}_{p\in G}|D_k(p)|}
{\operatorname{mean}_{p\in S}|D_k(p)|}.
\]

Appearance-vs-structure 선택성은:

\[
Q_k=\frac{R_{A,k}}{R_{G,k}}
\]

로 측정했다. Mask boundary에서 SAM cell과 resize가 섞이는 것을 줄이기 위해
각 영역을 독립적으로 erosion한 내부 pixel만 사용했다.

### 6.1 Frame 56: GT 밖의 appearance change

뒤쪽 monitor의 밝은 반사 영역은 inference에만 나타나지만 binary GT에는 포함되지
않았다. 수동 ROI와 binary structural GT는 겹치지 않았다.

동일한 `F_ref-F_inf`를 Delta-PC1~PC10 각각에 투영해 영역별 응답을 집계했다.
PC1의 structural response와 달리 뒤쪽 monitor appearance ROI는 후속 PC,
특히 PC7에서 상대적으로 강했다.

| PC | Appearance / stable | Structural GT / stable | Appearance / structural |
|---:|---:|---:|---:|
| PC1 | 6.67 | **20.23** | 0.33 |
| PC3 | 3.86 | 3.51 | 1.10 |
| PC5 | 3.24 | 2.90 | 1.12 |
| **PC7** | **8.55** | 4.84 | **1.77** |
| PC9 | 4.60 | 3.71 | 1.24 |

PC1은 structural GT에 훨씬 강했고 PC7은 appearance ROI에 가장 선택적이었다.
PC7 appearance projection의 dominant sign consistency는 약 **95.1%**였지만,
structural GT는 약 **65.5%**였다.

### 6.2 Frame 47: GT 내부 appearance change

Frame 47에서는 binary GT 대부분이 laptop screen의 appearance 변화였다. 수동
polygon으로 GT를 다음처럼 분리했다.

```text
Appearance GT = GT intersect manual appearance ROI
Structural GT = GT minus manual appearance ROI
```

Erosion 후 비교 pixel 수는 appearance `64,043`, structural `2,977`, stable
`391,030`이었다. Structural subset이 작다는 점은 이 probe의 중요한 한계다.

Frame 47에서도 screen appearance GT와 나머지 structural GT를 분리한 뒤 같은
10개 축을 정량 비교했다. PC1은 작은 structural subset에 강하지만 PC3/5/6/7/9는
screen 내부에서 더 coherent한 response를 보인다.

| PC | Appearance GT / stable | Structural GT / stable | Appearance / structural |
|---:|---:|---:|---:|
| PC1 | 3.27 | **11.89** | 0.28 |
| PC3 | 5.09 | 3.02 | 1.68 |
| PC5 | 4.01 | 2.99 | 1.34 |
| PC6 | 3.61 | 2.11 | 1.71 |
| **PC7** | **5.72** | 2.71 | **2.11** |
| PC9 | 3.98 | 2.46 | 1.62 |

PC7의 appearance sign consistency는 **89.5%**, structural sign consistency는
**51.9%**였다. PC3/5/9에서도 appearance 영역의 sign coherence가 structural
영역보다 높았다.

### 6.3 두 frame 사이 PC 축 안정성

Frame 47 분석은 target을 추가한 22-view 집합, frame 56 분석은 기본 stride의
21-view 집합으로 별도 Delta-PCA를 계산했다. 대응 PC의 absolute cosine은:

```text
PC1..PC10 diagonal cosine:
1.000, 0.999, 0.992, 0.977, 0.971,
0.954, 0.931, 0.987, 0.968, 0.914
```

로 높았다. 따라서 두 frame에서 PC7 appearance selectivity가 반복된 것이 단순한
PC permutation 때문일 가능성은 낮다. 그러나 다른 scene이나 online window에서
항상 `PC7`이라는 보장은 없다.

---

## 7. 현재까지의 결론

이번 probe는 다음 현상을 지지한다.

1. **Dominant Delta-PC1은 structural change에 강하다.**
   - Frame 56: structural/appearance ratio 약 `3.03`.
   - Frame 47: structural/appearance ratio 약 `3.63`.
2. **Appearance signal은 PC1에서 완전히 사라진 것이 아니라 여러 후속 PC에
   분산된다.**
3. **PC7은 두 frame에서 appearance에 가장 선택적이었다.**
   - Frame 56: appearance/structural `1.77`.
   - Frame 47: appearance/structural `2.11`.
4. **Appearance 영역은 일부 PC에서 한 부호로 coherent하다.** Structural
   old/new 영역은 서로 반대 lobe를 포함하므로 부호가 더 혼합되는 경향이 있다.
5. Binary GT는 appearance를 일관되게 포함하지 않는다. Frame 47에서는 screen
   appearance가 GT 대부분이지만 frame 56의 monitor reflection은 GT 밖이었다.

따라서 SAM feature difference에는 structural direction과 appearance subspace를
분리할 가능성이 있다. 하지만 현재 증거는 두 수동 ROI와 한 instance에 대한
탐색 실험이며 일반화된 자동 분류기는 아니다.

### 7.1 Scene-change 2 전체 시퀀스의 고정 축 비교

후속 실험에서는 한 프레임의 heatmap만 비교하지 않고, 서로 다른 방식으로 학습한
세 축을 Scene-change 2의 전체 `104` frame에 **고정 적용**했다.

| 축 | 학습 데이터 | Rank-1 통계 | 관찰 |
|---|---|---:|---|
| Global Delta-PC1 | 모든 view의 모든 `64×64` 위치 | explained variance **14.04%** | 큰 geometry/structural difference가 지배 |
| Frame-47 appearance direction | frame 47 appearance ROI의 `F_ref-F_inf` | object-level holdout AUROC **0.929**, AP **0.890** | laptop-screen appearance를 다른 view에서도 반복 강조 |
| Frame-56 geometry direction | frame 56의 모든 GEOMETRY annotation, 131 SAM cells | uncentered rank-1 energy **42.89%** | NEW/REMOVED geometry lobe를 강하게 강조 |

Global PC1은 `104 × 64 × 64 = 425,984`개의 delta vector 전체를 사용했다. 이전
21-view sampled Delta-PC1과의 absolute cosine은 `0.9994`였으므로, sampled
결과와 전체-frame 결과는 사실상 같은 dominant direction을 찾았다.

세 축 자체의 absolute cosine은 다음과 같았다.

| 축 쌍 | Absolute cosine |
|---|---:|
| Global PC1 ↔ frame-56 geometry direction | **0.9010** |
| Global PC1 ↔ frame-47 appearance direction | **0.0190** |
| Appearance direction ↔ geometry direction | **0.0095** |

즉 전체 영역 최대분산축은 geometry 전용 축과 거의 정렬되어 있지만 appearance
전용 축과는 거의 직교한다. 이는 정적 frame에서 PC1이 structural change에 더
강했던 결과를 시퀀스 수준에서도 설명한다.

이 비교가 지지하는 핵심은 다음과 같다.

> SCaR-3D식 global PC1은 add/remove를 포함한 dominant structural direction을
> 얻는 데 적합하지만, appearance 변화 검출을 단독으로 담당하기에는 부족하다.

다만 이는 `global PC1` 하나의 한계이지 SAM feature가 appearance를 표현하지
못한다는 뜻은 아니다. Frame-47 ROI direction의 leave-train-out 32 visible views에서
target appearance object는 stable 대비 `11.16×`, 다른 annotated object 대비
`4.79×` 강했고, sign consistency는 `96.9%`, target-object top-1 rate는 `100%`였다.
따라서 appearance 정보는 feature difference에 존재하지만 전체 분산 PCA에서는 큰
geometry change에 의해 후순위 또는 별도 축으로 밀린다고 해석하는 것이 맞다.

---

## 8. 최종 시스템으로 연결할 때의 역할 분리

현재 가장 자연스러운 online 처리 구조는 다음과 같다.

```text
O-SCD change cue component
        |
        +-- Global Delta-PC1 signed direction + multi-view reference-GS retention
        |       +-- pre/reference-side  -> removed
        |       +-- post/current-side   -> added/new
        |
        +-- same reference surface support on both sides
        |   + component/region-conditioned appearance direction or subspace
        |   + energy/sign coherence and cross-view persistence
        |       -> appearance-only
        |
        +-- low consistency / wide background trace
                -> uncertain or false positive; no geometry update
```

Geometry policy는 다음처럼 제한한다.

| 판정 | Geometry 처리 |
|---|---|
| Added/new, multi-view consistent | 해당 위치에만 new Gaussian seed 생성 및 제한적 최적화 |
| Removed | 새 Gaussian 생성 금지; 이전 state/support lifespan 종료 |
| Appearance-only | 기존 reference geometry 유지; DC/appearance state만 갱신 |
| Uncertain/FP | geometry와 lifespan 변경 보류 |

이렇게 하면 change cue가 높다는 이유만으로 기존 `R_change` geometry 전체를
움직이는 문제를 피할 수 있다.

---

## 9. 자동화 전에 남은 문제

### 9.1 PC 번호를 고정하면 안 된다

이번 scene에서는 PC7이 반복해서 appearance-selective했지만 PCA 축은 view window와
scene에 따라 회전하거나 순서가 바뀔 수 있다. 최종 방법은 `PC7`을 hard-code하지
말고 다음 속성으로 축 또는 subspace를 선택해야 한다.

- stable 대비 response energy
- dominant structural PC 대비 상대 선택성
- component 내부 sign coherence
- multi-view 반복성
- 같은 reference-GS surface에 대한 양쪽 support

더 나아가 이번 전체-sequence 비교는 **global PC1만으로 appearance를 검출하지
말아야 한다**는 점을 명확히 한다. 구조 방향은 global PCA로 얻되, appearance는
cue component 또는 candidate object region 내부의 Delta-SVD/PCA 축을 별도로
추정하거나 global PC1을 제거한 residual subspace에서 찾아야 한다.

### 9.2 수동 ROI 없이 appearance를 정의해야 한다

현재 appearance/structural label은 GT와 수동 polygon으로 만든 oracle probe다.
Online 알고리즘에서는 cue component별로 reference/inference 양쪽에 동일 surface가
존재하는지를 자동 판정해야 한다.

가능한 다음 지표:

1. positive/negative directional reference-GS 집합의 spatial overlap
2. 두 방향이 같은 reference surface에 anchor되는지 여부
3. PC1 old/new lobe의 공간 분리도
4. 후속 PC subspace energy와 sign coherence
5. 여러 view에서 같은 Gaussian region에 반복되는지 여부

### 9.3 Add/remove retention 구현을 정교화해야 한다

현재 voting은 FastGS mask-hit count 근사이고 center-consistency pruning도 occlusion과
alpha-transmittance를 완전히 반영하지 않는다. 정확한 renderer contribution weight,
visibility, view count threshold를 적용해야 한다.

### 9.4 Pose 오차와 render residual을 분리해야 한다

이번 실험은 fixed canonical pose를 사용했다. 실제 online pose retrieval에서는 작은
pose error가 edge 전체에 signed feature response를 만들 수 있다. PnP inlier 수와
MiniBA reprojection error로 update를 gate해야 한다.

---

## 10. 다음 실험

1. Scene/instance를 늘려 PC subspace 분리 재현성을 확인한다.
2. Cue connected component 단위의 multi-view track을 만든다.
3. Oracle ROI 없이 appearance score를 정의한다.
4. Added 후보에만 seed를 생성하는 최소 geometry experiment를 수행한다.
5. Removed와 appearance에는 geometry optimization을 금지한 대조군을 둔다.
6. Seed initialization 후에는 opacity/scale/xyz 자유도를 동시에 열지 않고 단계별
   ablation을 수행한다.
7. 최종 평가는 binary SCD뿐 아니라 added/removed/appearance class별 precision,
   recall과 geometry quality를 따로 측정한다.

---

## 11. 관련 artifact와 구현 상태

탐색용 분석은 `/tmp`에서 수행했다. 이 checkpoint의 repository implementation에는
다음이 아직 포함되지 않는다.

- automatic add/remove/appearance classifier
- automatic new-object seed initialization
- appearance-subspace online selection
- BOCD/change event detection
- densification, pruning, relocation

따라서 이 문서의 결과는 다음 연구 확장을 위한 근거이며 현재 O-SCD training
pipeline의 완성 기능을 의미하지 않는다.
