# Part 12. Online PCA sign posterior와 XFeat triangulation을 이용한 NEW Gaussian seed

> 작성일: 2026-08-14
> 범위: `R_ref -> scene_change1/2/3` 독립 causal replay, 이미지당 120 DC update
> 핵심 결론: **NEW mask 내부 XFeat만 inference-to-inference로 다시 매칭하여 3D seed를 만드는 경로는 정상 동작했고, seed 위치의 약 90%가 이후 관측의 NEW GT 안에 재투영되었다. 그러나 현재 sparse keypoint seed와 고정 opacity/scale 설정만으로는 fixed-topology DC-only 대비 성능 향상이 매우 작다.**

## 1. 질문

기존 fixed-topology DC-only 변화장은 reference에 존재하는 Gaussian에만 change DC를
학습한다. 따라서 reference에 없던 NEW object는 그 물체 자체의 3D support가 없다.
이번 E4a의 질문은 다음과 같다.

> Online posterior가 어느 PCA 부호가 NEW인지 확신한 뒤, 그 signed cue 내부 XFeat를
> inference view 사이에서 triangulation하여 fixed-geometry Gaussian row를 추가하고
> DC만 학습하면 기존 fixed-topology DC-only보다 좋아지는가?

## 2. 구현한 causal pipeline

Frame `t`에서 다음 순서를 고정했다.

1. `R_ref` render와 `I_t`의 SAM2.1 feature delta를 만든다.
2. `Delta_1 ... Delta_t` prefix만으로 PC1을 갱신하고 이전 PC1과 부호를 정렬한다.
3. stable-token MAD epsilon으로 강한 `+/-` cue mask를 만든다.
4. **현재 frame 학습 전** `D_plus`, `D_minus`를 render한다.
5. same-sign follow evidence를 Global pooled와 Component-balanced 두 방식으로 계산한다.
6. 각 방식의 fractional Beta posterior를 갱신한다.
7. 두 posterior가 `0.8`에서 같은 NEW sign에 합의한 경우에만 seed birth gate를 연다.
8. 현재 signed cue로 fixed reference topology의 `D_plus`, `D_minus` DC를 학습한다.
9. Gate가 열렸다면 NEW mask 내부 XFeat끼리만 matching/triangulation하고 candidate 및
   active seed를 갱신한다.
10. 생성된 seed의 geometry는 고정하고 `seed_dc`만 현재 NEW mask로 학습한다.
11. GT는 위 결정과 학습이 끝난 뒤 평가에만 읽는다.

Primary gate는 다음과 같다.

```python
if p_global >= 0.8 and p_balanced >= 0.8:
    new_sign = "+"
elif p_global <= 0.2 and p_balanced <= 0.2:
    new_sign = "-"
else:
    new_sign = None
```

중립 구간은 새 candidate birth와 seed DC 학습만 pause한다. 이미 승격된 seed의
lifespan은 유지되어 마지막으로 합의한 NEW sign 쪽에서 계속 render된다. 반대 sign에
합의하면 기존 seed의 lifespan을 flip timestamp에서 `[start,end)`로 닫는다.

## 3. XFeat seed geometry 계약

Pose 단계에서 한 번 얻은 raw XFeat를 CPU bounded buffer에 보관하고, 다음 hard gate를
모두 통과한 correspondence만 사용했다.

| 항목 | Primary 설정 |
|---|---:|
| XFeat | top-k 512 |
| Signed mask boundary | 64x64 mask 1-cell erosion |
| Descriptor | mutual nearest, cosine `> 0.82` |
| Pair translation | `>= 0.10` |
| Known-pose Sampson error | `<= 2 px` |
| Ray angle | `>= 1.5 deg` |
| Reprojection RMSE | `<= 3 px` |
| Endpoint/reprojection | 양쪽 모두 같은 NEW mask 내부 |
| Candidate | 2 unique inference views |
| Active promotion | 3 unique inference views |
| Buffer / prior views | 32 frames / 최대 8 prior views |
| Candidate TTL | 20 frames |

Reference XFeat, reference depth, reference point correspondence는 seed xyz 계산에 쓰지
않았다. Gate가 처음 열릴 때는 이미 관측한 최근 32 frame까지만 backfill하며,
과거 출력은 다시 쓰지 않는다.

Seed row 초기값은 다음과 같다.

```text
xyz       = known-pose multi-view DLT point
scale     = support view에서 1.5 px footprint인 world radius의 median
rotation  = identity quaternion
opacity   = fixed 0.1
SH-rest   = zero
DC        = zero에서 시작, NEW mask로만 학습
lifespan  = [promotion time, inf)
```

Reference와 seed의 `xyz/opacity/scale/rotation`은 optimizer에 들어가지 않는다.
Trainable tensor는 `D_plus`, `D_minus`, `seed_dc`뿐이며 generic densification/pruning은
호출하지 않았다.

## 4. 공정 비교 설정

| 항목 | 값 |
|---|---|
| 데이터 | Instance_1 SceneChange1/2/3, 총 304 frame |
| 실행 | scene별 PCA/posterior/DC memory 독립 reset |
| pose | 동일 fixed camera |
| cue | 동일 O-SCD candidate map + SAM2.1 delta |
| updates | 이미지당 120 |
| base Gaussian | 1,283,501개, topology 고정 |
| baseline | 동일 `D_plus`, `D_minus` fixed-topology DC-only |
| seed branch 차이 | active NEW seed row와 `seed_dc`만 추가 |
| official metric | renderer resolution `941 x 528`, threshold 0.5 |
| object-type diagnostic | `64 x 64` NEW/REMOVED/GEOMETRY mask |

두 branch는 동일한 base sign DC memory를 공유하므로 base 학습 stochasticity 차이가
없다. 아래 결과는 seed sidecar를 넣었을 때의 순수한 추가 효과다. 기존 oracle
lifespan 실험의 `IoU 0.6655 / F1 0.7991`은 boundary와 representation이 달라 이번
baseline과 직접 비교하지 않는다.

## 5. 결과

### 5.1 Official full-resolution 결과

| Scene | Frames | Active seed rows | Baseline IoU | Seed IoU | Delta IoU | Baseline F1 | Seed F1 | Delta F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SceneChange1 | 95 | 8 | 0.322572 | 0.322573 | +0.000001 | 0.487795 | 0.487796 | +0.000001 |
| SceneChange2 | 104 | 201 | 0.417882 | 0.418223 | +0.000341 | 0.589445 | 0.589784 | +0.000339 |
| SceneChange3 | 105 | 220 | 0.537861 | 0.538170 | +0.000309 | 0.699492 | 0.699754 | +0.000262 |
| **Overall** | **304** | **429** | **0.438561** | **0.438807** | **+0.000246** | **0.609722** | **0.609960** | **+0.000238** |

Overall precision은 `0.690321 -> 0.690398`, recall은 `0.545975 -> 0.546309`였다.
즉 변화는 양수지만 IoU `+0.000246`, F1 `+0.000238`로 매우 작다.

### 5.2 NEW/REMOVED diagnostic

Object-type metric은 annotation을 `64 x 64`로 내린 진단값이며 official full metric과
해상도가 다르다.

| Scope | Baseline IoU | Seed IoU | Delta IoU | Baseline F1 | Seed F1 | Delta F1 |
|---|---:|---:|---:|---:|---:|---:|
| GEOMETRY | 0.475277 | 0.475608 | +0.000330 | 0.644323 | 0.644626 | +0.000304 |
| NEW | 0.240943 | 0.241330 | **+0.000387** | 0.388322 | 0.388825 | **+0.000503** |
| REMOVED | 0.347833 | 0.347687 | -0.000146 | 0.516137 | 0.515975 | -0.000161 |

NEW recall은 `0.471723 -> 0.472481`로 증가했다. 반면 seed가 전체 change union에
추가되므로 REMOVED-only 평가에서는 동일한 recall에 소량의 FP가 더해져 precision과
IoU가 아주 조금 내려갔다. 이는 seed branch가 NEW 전용이라는 설계와 일치한다.

### 5.3 Gate와 3D seed 품질

| Scene | First gate | First active seed | Pair edges | Final seeds | 미래-view image 내부 표본 | 그중 NEW GT | NEW precision | REMOVED leakage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SC1 | 79 | 79 | 39 | 8 | 5 | 5 | 1.000 | 0.000 |
| SC2 | 7 | 10 | 1,483 | 201 | 5,345 | 4,854 | 0.908 | 0.000 |
| SC3 | 13 | 15 | 1,173 | 220 | 3,720 | 3,354 | 0.902 | 0.006 |

여기서 미래-view 평가는 promotion 이후 frame에 seed **중심점**을 재투영한 값이다.
화면 밖 표본은 precision 분모에서 제외했다. 이 값은 Gaussian footprint mask IoU가
아니므로 seed geometry의 point-location diagnostic으로 해석해야 한다.

SC2/SC3에서 image 안으로 들어온 seed 중심점의 약 90%가 NEW GT 안에 있었고
REMOVED leakage는 `0%`, `0.56%`였다. 따라서 “NEW mask XFeat track으로 reference에
없는 3D 위치를 찾는다”는 핵심 geometry 가정은 지지된다.

## 6. 왜 위치 품질에 비해 성능 향상은 작은가

1. **Sparse point coverage**: XFeat seed는 textured keypoint 중심만 표현한다. 넓거나
   textureless한 NEW surface를 채우지 않는다.
2. **보수적 footprint**: scale은 1.5 px, opacity는 0.1로 고정했다. 201/220 seed도
   full-resolution object 면적에 비하면 영향이 작다.
3. **DC-only 제한**: seed 위치가 맞아도 opacity/scale을 학습하지 않으므로 mask
   footprint를 supervision에 맞게 확장할 수 없다.
4. **Late gate**: SC1은 frame 79에서야 gate가 열려 8개 seed만 생겼다.
5. **Threshold crossing**: 일부 seed score 변화는 0.5 binary threshold를 넘기지
   못한다. 그래서 point precision은 높지만 mask metric은 거의 그대로다.

따라서 이번 결과는 “triangulation이 실패했다”가 아니라 **geometry birth는 맞지만
현재 sparse fixed-footprint representation이 mask coverage를 크게 바꾸지 못했다**로
해석해야 한다.

## 7. 검증과 artifact

- 304 frame 전체에서 base `_xyz/_features_dc/_features_rest/_opacity/_scaling/_rotation`
  bitwise audit 통과
- 모든 promotion이 3 unique views, causal support, ray-angle, reprojection gate 통과
- GT가 seed birth/training에 들어가지 않았음을 per-scene audit로 확인
- checkpoint sidecar tensor/metadata round-trip bitwise 일치
- unit/CUDA/runner 전체 test suite 통과
- 독립 code review: blocker 0, 승인

최종 artifact:

```text
outputs/e4_online_xfeat_new_seed_scene123_120_final_corrected_20260814/
  summary.json
  baseline_vs_xfeat_new_seed_comparison.{csv,md,png}
  global_vs_component_balanced_comparison.{csv,md,png}
  xfeat_features.pt
  scene_change{1,2,3}/
    frame_metrics.csv
    xfeat_seed_matches.jsonl
    seed_candidates.jsonl
    active_seed_reprojections.csv
    active_new_seeds.ply
    new_seed_checkpoint.pt
    final_sign_dc_memories.pt
    causal_pca_posterior_arrays.npz
    summary.json
```

주요 구현 위치:

- `temporal/sign_mapping.py`
- `temporal/new_seed_observation.py`
- `poses/new_seed_triangulation.py`
- `temporal/new_seed_manager.py`
- `temporal/new_seed_gaussians.py`
- `experiments/run_online_xfeat_new_seed.py`

## 8. 결론과 다음 단계

E4a의 결론은 다음과 같다.

> Online Beta posterior가 NEW sign을 결정하고, 그 signed cue 내부 XFeat의
> inference-to-inference correspondence가 3D 위치를 결정하며, 검증된 track만
> fixed-geometry Gaussian으로 승격되는 전체 경로는 구현·검증되었다.

그러나 현재 성능 차이는 너무 작아서 이 설정 그대로 production path에 넣을 근거는
부족하다. 다음 E4b에서는 이번 seed birth를 유지한 채, 별도 계획으로 다음 한 축씩
검증해야 한다.

1. seed scale/opacity만 제한적으로 학습하는 ablation
2. 가까운 duplicate track의 voxel consolidation
3. textureless NEW 영역을 위한 dense/region correspondence
4. point reprojection이 아니라 rendered seed footprint의 NEW precision/recall 평가

이번 E4a에는 geometry optimization, MCMC/SGLD, relocation, generic densification,
pruning을 추가하지 않았다.
