# Part 15. XFeat-Anchored Active NEW Geometry (E4d)

> **2026-09-02 metric audit:** 아래 11.2의 legacy `NEW` 표는 전체 최종 change mask를
> GT NEW mask와 비교한 cross-scope diagnostic이었다. learned-DC NEW sidecar만 GT NEW와
> 비교한 object-matched IoU가 아니므로 공식 NEW 성능으로 사용하거나 Part 16의 NEW IoU와
> 직접 비교하면 안 된다. 11.1 overall mask 결과와 reference/causal/geometry audit는 유효하다.

> 작성일: 2026-09-02\
> 범위: `R_ref -> SceneChange1/2/3` 독립 causal replay, 총 304 frame, frame당 120 optimizer step\
> 상태: controlled ablation. production algorithm으로 해석하지 않는다.

## 1. 연구 질문

이번 실험은 다음 질문 하나만 검증한다.

> 정확하지만 sparse한 XFeat triangulated NEW anchor가 있을 때, reference Gaussian을
> 완전히 고정하고 NEW sidecar에만 geometry optimization, adaptive densification,
> causal pruning을 허용하면 sparse seed보다 NEW object의 3D support와 change-mask
> coverage를 의미 있게 확장할 수 있는가?

독립변수는 **XFeat anchor가 승격된 이후 NEW bank를 갱신하는 방법**뿐이다.
SAM/PCA/posterior/XFeat triangulation front-end는 E4a와 동일하며, E4b의 parent-depth
propagation은 사용하지 않는다.

## 2. E4a/E4b/E4c와 E4d의 차이

| 단계 | Anchor birth | NEW geometry | Topology growth | Pruning |
|---|---|---|---|---|
| E4a | XFeat-512 triangulation | fixed | 없음 | 없음 |
| E4b | XFeat-512 triangulation | fixed | nearest-parent camera depth를 다른 pixel로 복사 | 없음 |
| E4c | XFeat-4096 triangulation | fixed | 모든 row가 direct XFeat triangulation | 없음 |
| **E4d** | **E4a XFeat-512 그대로** | **xyz/scale/rotation/opacity trainable** | **3D covariance clone/split** | **causal NEW support** |

E4b child는 image pixel에 가까운 parent의 depth를 복사했지만, E4d child는 parent의
3D covariance 안에서 sampling한다. 따라서 E4d에는 다른 pixel로 depth를 전파하는
경로가 없다.

## 3. 변경하지 않은 causal front-end

frame `t`의 순서는 다음과 같다.

1. `R_ref`와 `I_t`에서 SAM2.1 feature delta를 계산한다.
2. frame `1...t`만 사용해 causal prefix PCA를 갱신한다.
3. 이전 PC1과 sign을 정렬하고 `+/-` signed cue를 만든다.
4. optimization 전 D-plus/D-minus memory를 render한다.
5. Global pooled posterior와 Component-balanced posterior를 갱신한다.
6. 두 posterior가 probability `0.8`에서 같은 NEW sign에 합의한 경우에만 birth를 연다.
7. NEW signed mask 내부 XFeat-512를 inference-to-inference로 matching한다.
8. mutual cosine, baseline, Sampson, ray angle, reprojection gate를 통과한 track을 triangulate한다.
9. 2 unique view에서 candidate, 3 unique view에서 anchor로 승격한다.
10. 그 뒤에만 D0-D3 representation update가 갈라진다.
11. GT는 모든 online 결정과 update가 끝난 뒤 평가에만 읽는다.

다음은 사용하지 않는다.

- reference XFeat 또는 reference depth
- GT mask를 이용한 birth/optimization/density/pruning
- future frame
- monocular depth, optical flow, 새 matcher
- unrestricted image-space Gaussian birth

## 4. E4d NEW sidecar 표현

기존 `NewSeedGaussianModel`은 E4a-c 재현을 위해 변경하지 않았다. E4d는 별도
`ActiveNewGaussianModel`을 사용한다.

### 4.1 Trainable parameter

NEW row `j`는 다음 parameter를 가진다.

```text
x_j       : 3D center
s_j       : log scale
q_j       : raw quaternion
alpha_j   : opacity logit
C_j       : NEW change DC
```

lifespan과 lineage metadata는 non-trainable이다.

```text
[start_j, end_j)
stable_id
parent_stable_id
parent_row
generation
birth_frame
initial_xyz
birth_kind
```

XFeat anchor 초기값은 E4a와 같다.

```text
xyz       = known-pose multi-view XFeat triangulation
scale     = support view에서 약 1.5 px footprint
rotation  = identity quaternion
opacity   = 0.1
DC        = 0
lifespan  = [promotion_time, inf)
generation = 0
birth_kind = xfeat_anchor
```

Reference의 `_xyz`, `_features_dc`, `_features_rest`, `_opacity`, `_scaling`,
`_rotation`은 optimizer group에 들어가지 않는다.

### 4.2 Optimizer group과 기본 LR

| Parameter | 기본 LR |
|---|---:|
| xyz | `1.6e-4 * camera_extent` |
| DC | `2.5e-3` |
| opacity | `2.5e-2` |
| log scale | `5.0e-3` |
| rotation | `1.0e-3` |

각 tensor는 독립 Adam group을 사용한다. append/prune 때 기존 row의 Adam moment는 exact
보존하고 새 child moment만 0으로 시작한다.

## 5. Supervision과 gradient 격리

현재 confirmed NEW target은 다음 두 mask의 교집합이다.

```text
M_new(t) = confirmed signed NEW cue(t) AND O-SCD candidate(t)
```

### 5.1 Geometry branch

Active NEW sidecar만 흰색으로 render한 alpha coverage를 `A_new`라고 한다.

```text
L_inside  = NEW mask 내부의 평균 (1 - A_new)
L_outside = NEW mask 외부의 평균 A_new
L_geometry = lambda_inside * L_inside
           + lambda_outside * L_outside
```

기본값은 `lambda_inside=1.0`, `lambda_outside=0.1`이다. 이 branch에는 base Gaussian이
없으므로 reference가 NEW coverage를 explanation-away할 수 없다. 흰색 override를 쓰므로
geometry branch는 xyz/scale/rotation/opacity에 gradient를 주고 learned DC에는 주지 않는다.

### 5.2 DC branch: E4a objective 유지

Controlled-variable audit에서 DC objective를 E4b seed-only projected loss로 바꾸면 E4a와
공정한 비교가 아니므로 사용하지 않았다. E4d DC는 E4a와 같은 current-view joint render와
원본 O-SCD SSF objective를 사용한다.

```text
P = sigmoid(mean_rgb(render(R_ref change memory + active NEW DC)))
L_detection     = mean(M_new * (1 - P))
L_regularization = log(mean(P)^2 + 1)
L_DC = L_detection + L_regularization
```

DC branch에서는 base DC와 NEW xyz/scale/rotation/opacity를 detach한다. 따라서:

```text
geometry gradient -> NEW xyz/scale/rotation/opacity only
DC gradient       -> NEW C_j only
reference gradient -> always none
```

한 optimizer step의 최종 loss는 `L_geometry + L_DC`다. 두 rasterizer graph를 하나의 `backward()`에 동시에 넣으면 FastGS CUDA launch가 실패할 수 있으므로, parameter 집합이 서로 분리된 성질을 이용해 geometry backward와 DC backward를 순차 실행한 뒤 optimizer를 한 번만 step한다. 합산 gradient와 수학적으로 동일하다. replay view에 visible NEW row가 0개면 유효 geometry responsibility가 없으므로 해당 geometry backward만 skip하고 DC update는 유지하며 skip 횟수를 기록한다.

## 6. Causal multi-view replay

frame당 총 optimizer step은 120으로 고정한다.

- geometry view: current 1/3, 최근 prior NEW-support view 2/3
- replay buffer: 32 frame
- 사용할 prior view: 최대 8개
- DC view: 120 step 모두 current view. E4a supervision schedule을 유지한다.
- neutral posterior: 기존 active row는 render하지만 optimization/density/pruning을 pause한다.
- opposite NEW sign consensus: 이전 active lifespan을 `[start,t)`로 닫고 replay를 비운다.

모든 replay timestamp는 decision timestamp 이하이며 future view API는 timestamp check로
거부한다.

## 7. NEW-only adaptive densification

각 geometry update에서 active NEW row만 다음 통계를 누적한다.

```text
signed positional gradient = norm(dL / d viewspace_xy)
absolute FastGS gradient    = norm(dL / d viewspace_abs_channel)
visibility count
max projected radius
```

기본 density rule은 다음과 같다.

```text
small AND signed_gradient >= 2e-4 -> clone
large AND absolute_gradient >= 1.2e-3 -> split
small/large boundary = 0.01 * scene_extent
interval = 100 optimizer steps
start frame = 5
adaptive-density bank cap = 5000
```

split은 FastGS/3DGS 방식으로 parent scale과 rotation이 만드는 3D covariance에서 child
center를 sampling한다. 2-child split은 parent를 두 child로 교체한다. clone/split source는
항상 NEW sidecar stable ID로만 address되며 reference row index를 받을 수 없다.

D2와 D3는 같은 frame-wise random seed를 사용한다. GPU rasterizer atomic backward의 미세
비결정성은 남지만 variant별 난수 차이는 없다.

## 8. NEW-only causal pruning

D3에서만 수행한다. 순서는 **train -> prune -> densify**다. densify가 projected-radius
통계를 reset하기 전에 pruning이 누적 통계를 소비한다.

현재 eligible view에서 base+active NEW를 함께 render하고 alpha-T VJP로 NEW row의 실제
visible responsibility를 구한다. 중심점을 projection한 뒤:

```text
eroded NEW mask 내부 + visible       -> positive support
clear stable 영역 + visible          -> contradiction
FOV 밖 / occluded / boundary ambiguous -> no update
```

기본 pruning 조건:

- grace 10 frame 뒤 child opacity `< 0.01`
- 관측 8회 이상이고 support ratio `< 0.25`
- projected radius `> 100 px`
- world scale `> 0.10 * scene_extent`

XFeat anchor도 strong causal contradiction 또는 scale/radius sanity violation이면 제거할 수
있다. low-opacity pruning은 densified child에만 적용한다. CLOSED lifespan row는 history이므로
pruning하지 않는다.

## 9. Controlled ablation

| Variant | XFeat | Geometry thaw | Gradient density | Causal prune |
|---|---:|---:|---:|---:|
| D0 | 512 | 아니오 | 아니오 | 아니오 |
| D1 | 512 | 예 | 아니오 | 아니오 |
| D2 | 512 | 예 | 예 | 아니오 |
| D3 | 512 | 예 | 예 | 예 |

모든 branch는 동일 frame, SAM/PCA/posterior, XFeat cache, seed=0, cue threshold, 120 update를
사용한다. D0는 기존 E4a `NewSeedGaussianModel` path를 그대로 호출한다.

## 10. 검증

### 10.1 Unit/CUDA regression

- NEW xyz/scaling/rotation/opacity finite nonzero gradient
- E4a-compatible DC adapter와 legacy joint render state 일치
- reference 모든 parameter gradient 없음 및 exact unchanged
- append 후 기존 Adam state 보존
- prune 후 parameter/Adam row alignment
- split/clone source가 NEW row만 address
- future pruning observation reject
- half-open lifespan 및 CLOSED pruning 보존
- opacity probability/logit saturation
- full checkpoint round-trip
- E4a-c 기존 test regression

최종 전체 결과는 `507 passed`이다.

### 10.2 SC2 smoke

최종-contract 12-frame smoke에서:

- 9 XFeat anchor promotion
- D2/D3 covariance child birth 54개
- reference 1,283,501 rows의 6개 tensor bitwise 동일
- GT/future causal audit 전부 false
- D0-D3 checkpoint/PLY/log/CSV/PNG 생성

공유 seed smoke에서 D2/D3는 동일 lineage와 child count를 만들었고 pruning event가 없던
구간의 full IoU 차이는 약 `3.6e-6`이었다.

## 11. 304-frame 결과

### 11.1 Official full-resolution overall mask

| Variant | Precision | Recall | IoU | F1 | IoU delta vs D0 |
|---|---:|---:|---:|---:|---:|
| D0 | 0.690398 | 0.546309 | **0.438807** | **0.609960** | - |
| D1 | 0.660883 | 0.427793 | 0.350790 | 0.519385 | -0.088017 |
| D2 | 0.668379 | 0.418155 | 0.346307 | 0.514455 | -0.092500 |
| D3 | 0.658687 | 0.436590 | 0.356043 | 0.525120 | -0.082765 |

D0는 기존 E4a full 결과 `IoU/F1 = 0.438807/0.609960`을 정확히 재현했다. 따라서
front-end나 D0 reproduction path가 바뀌어서 생긴 차이가 아니다. D1부터 성능이 크게
하락했고, density를 추가한 D2가 더 낮았다. D3는 D2보다 IoU `+0.009736`, F1
`+0.010665`를 회복했지만 D0보다 여전히 크게 낮다.

### 11.2 Legacy cross-scope `full prediction vs NEW GT` (공식 NEW metric 아님)

| Variant | Precision | Recall | IoU | F1 | IoU delta vs D0 |
|---|---:|---:|---:|---:|---:|
| D0 | 0.272759 | 0.577937 | **0.227452** | **0.370608** | - |
| D1 | 0.150102 | 0.260170 | 0.105199 | 0.190371 | -0.122253 |
| D2 | 0.153698 | 0.257481 | 0.106496 | 0.192492 | -0.120956 |
| D3 | 0.171878 | 0.305055 | 0.123515 | 0.219872 | -0.103937 |

이 표는 당시 이름이 `NEW full-resolution mask`였지만 prediction이 NEW sidecar로 제한되지
않았다. 따라서 D0-D3의 object-specific NEW 성능 결론에는 사용할 수 없다. 다만 전체 최종
mask 안에서 NEW GT와 겹친 정도를 보존한 legacy diagnostic으로만 남긴다.

### 11.3 Scene별 full IoU/F1

| Scene | D0 | D1 | D2 | D3 |
|---|---:|---:|---:|---:|
| SC1 (95) | 0.322573 / 0.487796 | **0.323933 / 0.489350** | 0.323729 / 0.489117 | 0.323729 / 0.489117 |
| SC2 (104) | **0.418223 / 0.589784** | 0.316490 / 0.480808 | 0.306682 / 0.469406 | 0.318785 / 0.483453 |
| SC3 (105) | **0.538170 / 0.699754** | 0.405062 / 0.576575 | 0.404443 / 0.575948 | 0.419138 / 0.590694 |

SC1은 gate가 frame 79에 늦게 열려 anchor가 8개뿐이므로 차이가 작다. gate가 일찍 열리고
anchor가 201/220개 생긴 SC2/SC3에서 geometry branch의 하락이 명확하다.

### 11.4 REMOVED diagnostic 주의

REMOVED-only IoU는 D0 `0.320386`에서 D1/D2/D3 `0.382571/0.393660/0.374249`로
올랐다. 이것은 REMOVED geometry를 새로 잘 표현했다는 뜻이 아니다. NEW sidecar의
고-opacity/낮은 DC splat이 base change render를 가려 NEW 쪽 false positive와 true positive를
동시에 줄였고, 그 부산물로 REMOVED-only precision이 올라간 결과다. NEW recall과 overall
recall이 크게 하락했으므로 성공으로 해석하지 않는다.

## 12. Geometry와 topology diagnostic

### 12.1 Topology 수

| Scene | XFeat parent | D2 child birth / final total | D3 child birth / causal prune / final total |
|---|---:|---:|---:|
| SC1 | 8 | 696 / 431 | 687 / 0 / 429 |
| SC2 | 201 | 8,599 / 5,152 | 17,495 / 4,888 / 5,000 |
| SC3 | 220 | 497 / 508 | 16,450 / 5,012 / 5,000 |
| **합계** | **429** | **9,792 / 6,091** | **34,632 / 9,900 / 10,429** |

`--new-max-gaussians=5000`은 adaptive density call의 bank cap이다. D2 SC2는 cap에 도달한
뒤에도 causal XFeat anchor promotion을 거부하지 않았기 때문에 최종 5,152 rows가 되었다.
D3는 pruning이 공간을 계속 만들면서 최종 5,000 rows였다. 즉 hard birth rejection을 넣지
않고 front-end를 보존한 대가로 total count는 density cap보다 mandatory anchor 수만큼 조금
넘을 수 있다.

D3의 최대 generation은 SC2 `48`, SC3 `40`이었다. 같은 silhouette residual을 따라
split-prune-refill이 반복되면서 매우 깊은 descendant chain이 생겼다.

### 12.2 Evaluation-only center reprojection

| Scene | Variant/kind | image 안 표본 | NEW precision | REMOVED leakage |
|---|---|---:|---:|---:|
| SC2 | D1 XFeat parent | 5,246 | 0.794 | 0.030 |
| SC2 | D2 child | 110,610 | **0.076** | 0.059 |
| SC2 | D3 child | 130,126 | **0.053** | 0.040 |
| SC3 | D1 XFeat parent | 3,652 | 0.850 | 0.006 |
| SC3 | D2 child | 2,561 | **0.342** | 0.000 |
| SC3 | D3 child | 71,232 | **0.100** | 0.033 |

XFeat parent 중심은 geometry optimization 뒤에도 상대적으로 높은 precision을 보였지만,
densified child 중심은 실제 NEW surface에서 크게 벗어났다. 특히 D3는 bad child를 제거한
뒤 다시 density를 채우면서 center precision이 D2보다 더 낮아졌다.

### 12.3 Parameter pathology

- D1 max scale은 SC2 `467.87`, SC3 `643.25`까지 증가했다.
- D1 max xyz displacement는 SC2 `0.97`, SC3 `1.33`이었다.
- D2 max scale은 SC2 `136.83`, SC3 `443.59`였다.
- D3의 sanity pruning 뒤에도 max scale은 SC2 `190.81`, SC3 `5.99`였다.
- SC2 final DC median은 D2 `-1.848`, D3 `-1.872`였고, positive DC row 비율은 각각
  `0.58%`, `0.16%`뿐이었다.
- SC3 D3 final DC median도 `-1.870`이었다.

즉 geometry coverage branch는 흰색 alpha coverage를 늘렸지만, joint SSF DC branch는
cue 밖으로 새는 footprint에 global sparsity gradient를 주어 많은 child DC를 강한 음수로
학습했다. geometry와 semantic change color가 서로 다른 방향으로 진행됐다.

SC1에서는 camera path 후반 replay view에서 active child가 모두 화면 밖으로 나가
zero-visible geometry update가 D1 `152`, D2/D3 각각 `308`회 발생했다. 이 update는 geometry
backward만 skip했고 DC update는 유지했다.

## 13. Causal/reference audit와 runtime

### 13.1 Reference bitwise audit

SC1/SC2/SC3 모두 reference 1,283,501 rows에 대해 다음 tensor가 전후 bitwise 동일했다.

```text
_xyz
_features_dc
_features_rest
_opacity
_scaling
_rotation
```

각 field의 max absolute difference는 0이고 topology/count도 동일하다. Reference는 어떤
optimizer group에도 들어가지 않았다.

### 13.2 Causal audit

세 scene의 `causal_audit.json`에서 모두 다음 값이 false였다.

```text
GT used in PCA/posterior
GT used in XFeat birth
GT used in geometry optimization
GT used in densification
GT used in pruning
future views used in optimization
future views used in pruning
reference topology edited
```

D0-D3는 scene별로 동일한 SAM/PCA/posterior/XFeat promotion trace를 공유했다. D2/D3
covariance sampling도 동일 frame-wise seed를 사용한다.

### 13.3 Runtime

| Scene | seconds | sec/frame | peak CUDA bytes |
|---|---:|---:|---:|
| SC1 | 192.0 | 2.02 | 2,777,388,032 |
| SC2 | 636.7 | 6.12 | 2,987,294,208 |
| SC3 | 572.5 | 5.45 | 2,953,676,288 |
| **전체** | **1,401.3** | **4.61 평균** | **2,987,294,208 max** |

branch별 geometry optimization 누적 시간은 D1 `178.1 s`, D2 `186.8 s`, D3 `206.9 s`였다.
D2/D3 density 자체는 `0.87/2.90 s`, D3 visibility-support-pruning은 `15.70 s`였다.

D3 pruning reason 총계는 SC2/SC3에서 다음과 같았다.

- excessive screen radius: 4,309
- low child opacity: 2,638
- causal contradiction: 1,460
- excessive world scale: 2,515

한 row가 여러 reason을 동시에 가질 수 있어 reason 합은 pruned row 수보다 클 수 있다.

## 14. 해석과 failure mode

이번 controlled ablation은 성공 기준을 만족하지 못했다.

1. **Sparse silhouette geometry는 depth를 충분히 제약하지 못했다.**\
   RGB/depth supervision 없이 `M_new` coverage만으로 xyz/scale/rotation/opacity를 동시에
   풀면 anchor가 mask를 국소적으로 채우기보다 이동하거나 크게 팽창하는 해가 쉽다.

2. **White coverage와 learned DC의 semantic mismatch가 핵심이다.**\
   Geometry branch는 어떤 DC를 가질지와 무관하게 alpha coverage를 늘린다. Joint SSF는
   cue 밖으로 샌 footprint에 sparsity gradient를 주어 그 row의 DC를 음수로 만든다.
   결과적으로 high-opacity/large-geometry인데 change color는 검정인 row가 대량 생성됐다.

3. **NEW sidecar가 base change를 occlude했다.**\
   최종 render에서 이 negative-DC child가 base D-plus/D-minus Gaussian 앞을 가리면서 change
   score를 낮췄다. 그래서 D1-D3의 NEW/overall recall이 무너졌고, REMOVED-only precision이
   겉보기에는 올라갔다.

4. **Gradient split은 surface growth가 아니라 residual chasing이 됐다.**\
   XFeat parent는 비교적 정확했지만 child center precision은 SC2 D2/D3에서 7.6%/5.3%였다.
   parent covariance sampling 자체는 3D이지만, trigger를 만드는 loss가 silhouette residual이므로
   actual NEW surface보다 boundary/background/occlusion residual을 따라갔다.

5. **현재 causal pruning은 나쁜 birth보다 느리다.**\
   D3는 D2보다 mask metric을 일부 회복했지만 SC2/SC3에서 3.46만 child를 만들고 약 9.9천
   row를 제거한 뒤 다시 cap을 채웠다. center support, opacity, radius만으로는 high-opacity
   negative-DC child를 충분히 빨리 막지 못했다.

따라서 training loss 감소, row 수 증가, REMOVED IoU 증가를 성공으로 해석하지 않는다.
질문에 대한 답은 **현재 E4d objective와 자유도 조합으로는 sparse XFeat precision을 유지하며
NEW coverage를 확장하지 못한다**이다.

## 15. Artifact와 재현

최종 artifact:

```text
outputs/e4d_xfeat512_active_new_geometry_20260902_120837/
  configuration.json
  summary.json
  artifact_validation.json
  run.log
  D0_D1_D2_D3_comparison.{csv,md,png}
  scene_change{1,2,3}/
    frame_metrics.csv
    gaussian_count_over_time.csv
    base_bitwise_audit.json
    causal_audit.json
    D0|D1|D2|D3/
      checkpoint.pt
      final_new_gaussians.ply
      geometry_gradient_stats.csv
      densification_events.jsonl
      pruning_events.jsonl
      new_reprojection_metrics.csv
```

Full run:

```bash
PYTHONPATH=. conda run --no-capture-output -n oscd \
  python -m experiments.run_online_xfeat_new_seed \
  --e4d-variants D0 D1 D2 D3 \
  --scenes 1 2 3 \
  --updates-per-frame 120 \
  --xfeat-top-k 512 \
  --feature-cache outputs/e4_online_xfeat_new_seed_scene123_120_final_corrected_20260814/xfeat_features.pt \
  --skip-gif \
  --output-dir outputs/e4d_xfeat512_active_new_geometry_20260902_120837
```

Test:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. \
  conda run --no-capture-output -n oscd pytest -q tests
```


## 16. 결론과 다음 최소 실험

E4d의 결론은 다음과 같다.

> XFeat-512 anchor 자체는 좋은 3D starting point지만, NEW silhouette coverage만으로 모든
> geometry와 opacity를 풀고 gradient density를 허용하면 anchor precision을 보존한 surface
> expansion이 아니라 scale/drift/negative-DC occluder가 만들어진다.

D1이 이미 크게 악화됐으므로 현재 bottleneck은 sparse topology만이 아니다. D2 density는
이를 고치지 못했고, D3 pruning도 일부 회복에 그쳤다.

다음 최소 실험은 하나만 권한다.

> **E4d-local-footprint:** XFeat xyz와 rotation은 다시 고정하고, scale과 opacity만 bounded
> trust region 안에서 학습하며 densification/pruning은 끈다.

예를 들어 initial scale의 0.5--2배, opacity 0.05--0.5 안에서만 최적화한다. 이것은 새 loss나
matcher를 추가하지 않고, “정확한 anchor 위치를 보존한 국소 footprint 확장만으로 E4a NEW
recall을 높일 수 있는가”를 한 축으로 분리한다. 이 실험도 실패하면 geometry freedom보다
signed cue/object support 자체가 다음 bottleneck이라는 결론을 더 강하게 내릴 수 있다.
