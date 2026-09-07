# Part 16. Learned NEW DC Isolation + XFeat Root-Anchor Radius (E4e)

> 작성일: 2026-09-02\
> 범위: 독립 `R_ref -> SC1/SC2/SC3`, 총 304 frame, frame당 120 update\
> 핵심 결과: NEW-only learned-DC는 자유 geometry에서 sidecar-vs-NEW IoU를
> `0.0752 -> 0.1182`로 높였지만 overall IoU는 `0.3616 -> 0.2589`로 낮췄다. XFeat xyz
> 고정과 soft radius penalty를 함께 쓴 C3의 NEW IoU는 `0.0636`이었다.

## 1. 정정된 연구 질문

이번 실험의 질문은 다음 두 가지다.

1. reference/background change bank와 함께 render해서 NEW DC를 학습하면, 이미 학습된
   base Gaussian이 cue를 설명하거나 가려 NEW Gaussian의 DC gradient가 약해지는가?
2. generation-0 XFeat anchor의 xyz를 고정하고 child가 root anchor 반경을 벗어날 때
   penalty를 주면 sparse anchor의 localization을 보존하면서 coverage를 넓힐 수 있는가?

모든 variant의 NEW Gaussian은 **learned DC**를 가진다. NEW Gaussian을 무조건 흰색으로
출력하지 않는다.

초기에 실행했던 fixed-white output C1/C3는 이 질문을 잘못 해석한 실험이므로 주 결과에서
제외한다. 해당 artifact는 재현 기록으로만 남아 있다.

```text
제외된 mis-scoped artifact:
outputs/e4e_semantic_new_anchor_radius_20260902_135818/

object-type metric 계약이 잘못되어 superseded된 artifact:
outputs/e4e_learned_dc_separate_anchor_radius_20260902_144614/
```

## 2. 공통으로 고정한 부분

다음 causal frontend와 XFeat birth 조건은 E4a/E4d에서 변경하지 않았다.

```text
R_ref / I_t
 -> SAM2.1 feature delta
 -> causal prefix PCA
 -> previous-PC1 sign alignment
 -> +/- signed cue
 -> global + component-balanced Beta posterior
 -> 두 posterior가 probability 0.8에서 동의
 -> NEW sign 결정
 -> XFeat-512 matching / known-pose triangulation
 -> 3 unique views에서 promotion
```

- 미래 frame, GT mask, reference depth는 optimization에 사용하지 않는다.
- reference 1,283,501 Gaussian은 optimizer에 들어가지 않는다.
- D-plus/D-minus base change DC 학습은 동일하다.
- E4d D3의 geometry LR, density threshold, pruning threshold를 공통으로 사용한다.
- `new-max-gaussians=5000`은 목표 밀도가 아니라 안전 상한이다.
- geometry loss는 active NEW sidecar만 render하므로 base가 geometry gradient를
  explanation-away하는 경로는 원래부터 없다.

## 3. C0-C3 실험 계약

| Variant | NEW DC 학습 | XFeat anchor xyz | Child xyz 제약 | Density/prune |
|---|---|---|---|---|
| C0 | joint base+NEW SSF | trainable | 없음 | E4d D3 |
| C1 | NEW-only projected cue mixture | trainable | 없음 | E4d D3 |
| C2 | joint base+NEW SSF | exact fixed | root-radius soft hinge | E4d D3 |
| C3 | NEW-only projected cue mixture | exact fixed | root-radius soft hinge | E4d D3 |

즉 factorial axis는 다음이다.

```text
DC supervision: joint(C0,C2) vs NEW-only(C1,C3)
geometry:       free(C0,C1)  vs fixed-anchor + radius(C2,C3)
```

## 4. learned DC를 따로 학습하는 방법

### 4.1 공통 causal NEW target

현재 frame에서 confirmed NEW signed cue와 O-SCD candidate의 교집합만 target으로 쓴다.

```text
M_new(p) = confirmed_NEW_sign(p) AND O-SCD_candidate(p)
```

GT NEW mask는 metric 계산에만 사용한다.

### 4.2 C0/C2: 기존 joint DC 학습

현재 view에서 frozen base change bank와 active NEW sidecar를 함께 render한다. NEW geometry는
detach하고 NEW DC만 gradient를 받는다.

```text
joint render = frozen base change rows + active NEW rows
DC loss      = original SSF(M_new, joint render)
```

base DC tensor 자체는 이 branch에서 detach되어 있지만, base의 rendered opacity/color는
NEW row 앞에서 cue를 이미 설명하거나 NEW row를 가릴 수 있다.

### 4.3 C1/C3: NEW-only learned DC 학습

각 active NEW Gaussian을 현재 view에 projection하고, 중심과 footprint 내부 8개 offset에서
`M_new`를 bilinear sample한다.

```text
q_i = Gaussian i의 9개 projected sample에서 얻은 평균 NEW cue
c_i = Gaussian i의 학습 가능한 DC 세 채널 평균
L_new_dc = mean BCE-with-logits(c_i, q_i)
```

- reference/base row는 이 loss graph에 존재하지 않는다.
- xyz, scale, rotation, opacity는 이 DC loss에서 gradient를 받지 않는다.
- DC는 optimizer로 계속 학습되며 final evaluation도 learned DC render를 사용한다.
- 흰색 alpha는 final mask가 아니라 evaluation-only footprint audit에만 사용한다.

로컬 FastGS의 작은 dynamic-bank DC backward가 128 rows에서 CUDA launch 오류를 냈기 때문에,
NEW-only branch는 custom color backward 대신 위의 vectorized projected-cue surrogate를 썼다.
따라서 C1-C0 gradient 크기를 순수하게 “occlusion 하나만 제거한 결과”로 해석하면 안 된다.
학습 context와 loss parameterization이 함께 달라진 controlled implementation ablation이다.

## 5. XFeat anchor 및 child 반경 제약

C2/C3에서 generation-0 XFeat anchor는 매 optimizer step 다음 순서로 고정한다.

```text
anchor xyz gradient = 0
Adam step
anchor xyz = triangulation 당시 root xyz로 exact restore
anchor xyz Adam moment = 0
```

anchor는 pruning하지 않고 직접 densify는 최대 한 번만 한다. 모든 descendant는 root XFeat
xyz와 root birth scale을 상속한다.

```text
root radius = 4 * root birth scale
distance ratio = distance(child xyz, root xyz) / root radius
anchor loss = mean(max(distance ratio - 1, 0)^2)
weight = 1
```

이번 제약은 xyz 중심에만 적용된다. scale, rotation, opacity는 계속 학습되며 hard projection은
하지 않는다.

## 6. 실행

```bash
PYTHONPATH=. conda run --no-capture-output -n oscd \
  python -m experiments.run_online_xfeat_new_seed \
  --e4e-variants C0 C1 C2 C3 \
  --scenes 1 2 3 \
  --updates-per-frame 120 \
  --xfeat-top-k 512 \
  --feature-cache \
    outputs/e4_online_xfeat_new_seed_scene123_120_final_corrected_20260814/xfeat_features.pt \
  --skip-gif \
  --output-dir \
    outputs/e4e_learned_dc_object_matched_metrics_20260902_164614
```

총 runtime은 `1544.3 sec`, 약 `25.7 min`이었다.

## 7. 공식 full-resolution 결과

metric의 예측과 GT를 다음처럼 맞췄다.

```text
overall = 전체 최종 change mask vs 전체 GT change mask
NEW     = learned-DC NEW sidecar mask vs GT NEW mask
REMOVE  = causal opposite-sign base mask vs GT REMOVED mask
```

GT NEW 영역 안에 있는 Gaussian을 미리 고르지 않는다. 모든 active NEW sidecar Gaussian을
렌더하고 GT NEW 밖의 sidecar prediction은 false positive로 계산한다.

### 7.1 전체 change mask

| Variant | Precision | Recall | IoU | F1 |
|---|---:|---:|---:|---:|
| C0 | 0.6672 | 0.4412 | 0.3616 | 0.5312 |
| C1 | 0.3084 | 0.6171 | 0.2589 | 0.4113 |
| C2 | 0.6603 | 0.4346 | 0.3552 | 0.5242 |
| C3 | 0.2523 | 0.5826 | 0.2137 | 0.3521 |

### 7.2 NEW object

| Variant | Precision | Recall | NEW IoU | NEW F1 |
|---|---:|---:|---:|---:|
| C0 | 0.5604 | 0.0799 | 0.0752 | 0.1398 |
| C1 | 0.1300 | 0.5665 | 0.1182 | 0.2114 |
| C2 | 0.3448 | 0.0461 | 0.0424 | 0.0813 |
| C3 | 0.0699 | 0.4130 | 0.0636 | 0.1196 |

### 7.3 scene별 IoU

| Scene | C0 overall/NEW | C1 overall/NEW | C2 overall/NEW | C3 overall/NEW |
|---|---:|---:|---:|---:|
| SC1 | 0.3237 / 0.0125 | 0.3236 / 0.0121 | 0.3233 / 0.0099 | 0.3233 / 0.0101 |
| SC2 | 0.3199 / 0.1294 | 0.3242 / 0.2036 | 0.3064 / 0.0430 | 0.2144 / 0.0458 |
| SC3 | 0.4324 / 0.0649 | 0.2154 / 0.1006 | 0.4273 / 0.0522 | 0.1914 / 0.0755 |

## 8. 질문별 해석

### 8.1 NEW-only learned DC는 실제 NEW 신호를 더 살렸는가?

자유 geometry에서 C1-C0 차이는 다음과 같다.

```text
NEW recall:  +0.4867
NEW IoU:     +0.0430
NEW F1:      +0.0716
overall IoU: -0.1028
overall F1:  -0.1199
```

따라서 base와 분리한 learned-DC supervision은 **NEW recall/IoU에는 이득**이 있었다.
특히 SC2 NEW IoU가 `0.1294 -> 0.2036`, SC3가 `0.0649 -> 0.1006`으로 증가했다.

반면 전체 mask에서는 precision이 `0.6672 -> 0.3084`로 낮아졌다. NEW-only loss가 cue가
있는 anchor를 강하게 양수 DC로 만들면서 잘못 퍼진 footprint도 change로 살아났기 때문이다.

### 8.2 joint branch에서 DC gradient가 약했는가?

frame 마지막 update의 mean DC gradient norm은 다음 경향을 보였다.

| Scene | C0 joint | C1 NEW-only | C2 joint | C3 NEW-only |
|---|---:|---:|---:|---:|
| SC1 | 1.46e-5 | 2.85e-2 | 2.47e-5 | 2.17e-2 |
| SC2 | 1.11e-4 | 9.68e-3 | 1.48e-4 | 1.24e-2 |
| SC3 | 2.29e-4 | 1.89e-2 | 1.28e-4 | 1.47e-2 |

joint branch의 NEW DC gradient가 매우 작았다는 관찰은 맞다. 다만 NEW-only branch는
BCE projected-cue loss를 사용하므로 norm의 절대 배율을 같은 objective의 순수 occlusion
효과로 간주할 수는 없다. metric의 NEW recall/IoU 상승이 더 직접적인 근거다.

### 8.3 DC 부호는 어떻게 바뀌었는가?

최종 row 중 DC 평균이 양수인 비율:

| Scene | C0 | C1 | C2 | C3 |
|---|---:|---:|---:|---:|
| SC1 | 0.661 | 1.000 | 0.413 | 0.877 |
| SC2 | 0.002 | 0.060 | 0.004 | 0.312 |
| SC3 | 0.012 | 0.118 | 0.014 | 0.228 |

C1/C3는 특히 XFeat anchor DC를 양수로 유지했다. 그러나 densified child 다수는 projected
cue 밖으로 벗어나 음수가 됐다. 따라서 separate DC만으로 잘못된 child geometry를 고칠 수
없다.

### 8.4 root-radius penalty는 성공했는가?

아니다. generation-0 XFeat xyz 자체는 bitwise 고정됐지만 child는 계속 radius 밖으로
나갔다.

| Scene | Variant | child ratio median | radius 밖 child | max scale |
|---|---|---:|---:|---:|
| SC1 | C2 | 0.91 | 61 / 171 | 0.29 |
| SC1 | C3 | 0.95 | 70 / 179 | 0.29 |
| SC2 | C2 | 1.82 | 1769 / 1942 | 74.82 |
| SC2 | C3 | 9.67 | 2107 / 2219 | 47.97 |
| SC3 | C2 | 44.46 | 3034 / 3257 | 71.52 |
| SC3 | C3 | 33.16 | 3812 / 4034 | 76.50 |

원인은 세 가지다.

1. hinge는 soft penalty라 optimizer step 뒤 위치를 반경 안으로 강제하지 않는다.
2. 재귀 densification이 이미 커진 child covariance에서 더 먼 descendant를 만든다.
3. xyz만 제한하고 scale/opacity를 자유롭게 두어 center가 맞아도 footprint가 크게 퍼질 수 있다.

결과적으로 C3는 C1보다 NEW IoU가 `0.1182 -> 0.0636`으로 낮아졌다. 현재 형태의 soft
xyz-only constraint는 채택하면 안 된다.

## 9. Density/footprint 진단

최종 NEW Gaussian 수:

| Scene | C0 | C1 | C2 | C3 |
|---|---:|---:|---:|---:|
| SC1 | 407 | 431 | 179 | 187 |
| SC2 | 5000 | 5000 | 2143 | 2420 |
| SC3 | 5000 | 5000 | 3477 | 4254 |

C0/C1은 SC2와 SC3에서 safety cap 5000에 도달했다. 이는 적정 NEW 밀도를 찾았다는 뜻이
아니라 density controller가 계속 birth를 요구했다는 뜻이다.

evaluation-only white-alpha sidecar precision은 overall 기준 C0/C1/C2/C3
`0.0674/0.0724/0.0425/0.0416`에 불과했다. 즉 learned DC와 별개로 geometry footprint
자체가 대부분 실제 NEW mask 밖에 있다.

## 10. 감사와 검증

세 scene 모두 다음을 통과했다.

- reference 1,283,501 rows의 xyz/DC/SH-rest/opacity/scale/rotation bitwise 동일
- reference topology 동일
- GT used in PCA/posterior: false
- GT used in XFeat birth: false
- GT used in geometry optimization/density/pruning: false
- future view used in optimization/pruning: false
- C0-C3 NEW DC가 실제로 nonzero로 학습됨
- C2/C3 generation-0 XFeat xyz exact 고정
- C2/C3 anchor 직접 densification 최대 1회

검증 artifact:

```text
outputs/e4e_learned_dc_object_matched_metrics_20260902_164614/
  artifact_validation.json
  C0_C1_C2_C3_comparison.csv
  C0_C1_C2_C3_comparison.md
  C0_C1_C2_C3_comparison.png
```

테스트:

```text
targeted active-NEW tests: 29 passed
full repository tests:     516 passed
```

## 11. 결론

이번 실험은 두 현상을 분리했다.

1. **NEW DC를 base와 분리해 학습하는 방향은 유효하다.** 자유 geometry에서 NEW IoU/F1과
   recall이 크게 증가했다.
2. **현재 adaptive geometry는 아직 신뢰할 수 없다.** separate DC가 살린 신호가 잘못된
   footprint까지 활성화해 전체 precision을 낮췄고, soft xyz hinge는 재귀 split과
   scale/opacity 폭주를 막지 못했다.

따라서 다음 최소 실험은 topology/geometry 변수를 먼저 제거하는 것이다.

```text
XFeat-512 parent xyz/scale/rotation/opacity fixed
density off, pruning off
joint learned DC vs NEW-only projected-cue learned DC만 비교
```

이 실험에서 NEW IoU 이득이 유지되면 DC gradient isolation을 채택하고, 그 다음 단계에서만
child xyz hard projection, root-relative scale cap, opacity cap을 하나씩 추가하는 것이 맞다.
