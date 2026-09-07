# Part 17. Depth Anything 3 depth-prior NEW seed feasibility

## 1. 질문

E4a--E4e에서 XFeat triangulation은 정확한 generation-zero anchor를 주었지만 너무
sparse했다. 반대로 sparse anchor만 geometry optimization/densification하면 다음 문제가
반복됐다.

- base/reference Gaussian이 먼저 change cue를 설명해 NEW 쪽 gradient가 약해졌다.
- parent depth 복사는 ray 또는 chain 모양의 잘못된 geometry를 만들었다.
- 자유 xyz/scale optimization은 depth와 footprint를 폭주시켰다.
- gradient densification은 object004의 원통 표면을 직접 만들지 않고 기존 ACTIVE
  reference Gaussian을 거친 proxy로 morphing했다.

따라서 이 실험은 geometry gradient 전에 현재 RGB에서 얻은 dense depth prior를 NEW
영역의 초기 3D point cloud로 사용할 수 있는지를 먼저 분리 검증한다.

## 2. 구현

### 2.1 모델과 causal input

- 공식 Depth Anything 3 repository revision:
  `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`
- checkpoint: `depth-anything/DA3-SMALL`
- pose-conditioned inference: fixed OpenCV world-to-camera extrinsic과 intrinsic을 입력한다.
- current frame `t`마다 `[t-7, ..., t]`의 최대 8-view window만 사용한다.
- DA3 output resolution은 `504×280`이다.
- 과거 frame의 depth를 새 window로 다시 수정하지 않고 current-frame depth만 한 번
  소비한다.

관련 코드:

- [`temporal/depth_prior_new_seeding.py`](../temporal/depth_prior_new_seeding.py)
- [`experiments/analyze_da3_depth_prior_new_seeds.py`](../experiments/analyze_da3_depth_prior_new_seeds.py)
- [`tests/temporal/test_depth_prior_new_seeding.py`](../tests/temporal/test_depth_prior_new_seeding.py)

### 2.2 reference scale alignment

Immutable reference GS를 같은 fixed camera에서 두 번 렌더한다.

1. Gaussian camera-z를 color로 넣은 alpha-weighted depth numerator
2. white color를 넣은 accumulated alpha

두 render의 비율을 reference camera-z depth로 사용한다. DA3 scale anchor는 다음을
동시에 만족하는 pixel이다.

- reference alpha `>= 0.5`
- raw O-SCD candidate cue `<= 0.2`
- DA3/reference depth가 finite positive

Depth alignment는 positive scale-only다.

```text
log s = median(log d_ref - log d_DA3)
d_aligned = s · d_DA3
```

log-ratio MAD로 outlier를 제거한다. Affine shift는 ray마다 비강체 depth 이동을 만들기
때문에 사용하지 않는다.

### 2.3 NEW seed gate

Ground truth 없이 다음 causal 조건만 사용한다.

```text
confirmed signed NEW mask
AND DA3 confidence >= current NEW-region 40% quantile
AND d_aligned <= d_ref - max(0.03, 0.015 · d_ref)
```

- NEW mask는 1 pixel erosion한다.
- output 4×4 cell마다 confidence가 가장 큰 point 하나만 선택한다.
- frame당 최대 2,048개다.
- initial isotropic scale은 2-pixel footprint에 해당하는
  `d_aligned · 2 / sqrt(fx·fy)`다.
- 여러 frame point는 world-space 2 cm voxel에서 가장 먼저 도착한 causal point를
  보존한다.

Object004와 object010 mask는 seed를 모두 만든 뒤 evaluation에만 사용한다.

## 3. SC3 5-view 결과

평가 frame은 `48, 52, 64, 80, 84`다.

| Frame | Scale | Alignment MedRel | Seeds | object004 seeds | object004 precision | any-NEW precision | object010 leakage | object004 causal-mask recall | object004 seed coverage |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 48 | 0.9416 | 0.0420 | 383 | 267 | 0.6971 | 0.7728 | 0.0000 | 0.9985 | 0.5459 |
| 52 | 0.9210 | 0.0434 | 518 | 274 | 0.5290 | 0.7683 | 0.0039 | 0.9173 | 0.4133 |
| 64 | 0.9389 | 0.0276 | 278 | 219 | 0.7878 | 0.8777 | 0.0000 | 0.9735 | 0.5707 |
| 80 | 0.9491 | 0.0349 | 146 | 119 | 0.8151 | 0.9932 | 0.0000 | 1.0000 | 0.3492 |
| 84 | 0.9578 | 0.0526 | 186 | 134 | 0.7204 | 0.9194 | 0.0000 | 0.9970 | 0.3553 |
| Mean | -- | 0.0401 | -- | -- | **0.7099** | **0.8663** | **0.00077** | **0.9773** | **0.4469** |

Raw 1,511 point 중 2 cm causal voxel merge 후 1,392개가 남았고, evaluation-only
object004 subset은 926개였다. Object004에서 `d_ref - d_aligned` median은 frame별
`0.672, 0.993, 0.528, 0.497, 0.925`로 모두 명확한 front separation이었다.

### 3.1 causal XFeat anchor와 density 비교

아래 수치는 각 current frame까지 실제로 태어난 point만 사용한 reprojection audit다.

| Frame | DA3 causal points | DA3 object004 hits | XFeat causal anchors | XFeat visible | XFeat object004 hits |
|---:|---:|---:|---:|---:|---:|
| 48 | 363 | 252 | 124 | 16 | 11 |
| 52 | 821 | 476 | 128 | 14 | 11 |
| 64 | 1,078 | 353 | 164 | 54 | 3 |
| 80 | 1,213 | 369 | 176 | 79 | 11 |
| 84 | 1,392 | 732 | 183 | 94 | 24 |

DA3 seed는 XFeat보다 단순히 수가 많은 것뿐 아니라 object004 표면을 조밀하게
덮었다. `object004_depth_seed_geometry.png`의 XY/XZ/YZ projection에서도 원통의 둥근
단면과 수직 측면이 직접 나타난다. 이는 앞선 gradient density가 두 번의 source
selection만 하고 기존 reference GS를 proxy로 이동시킨 것과 질적으로 다르다.

Cross-view object004 point의 symmetric nearest-neighbor median은 frame pair
`48→52`, `52→64`, `64→80`, `80→84`에서 각각 `0.054, 0.077, 0.212, 0.040`이었다.
`64→80`은 관측 표면이 크게 달라지는 구간이라 높지만, 인접 overlap view에서는
동일한 scene-scale 표면으로 모였다.

## 4. 결론

**Depth prior를 generation-zero geometry로 쓰는 가설은 feasibility를 통과했다.**

특히 object004에서 다음 세 가지가 동시에 확인됐다.

1. reference scene scale에 안정적으로 정렬된다.
2. removed object010으로 거의 누출되지 않는다.
3. XFeat보다 훨씬 dense한 원통 표면 point cloud를 geometry optimization 전에 만든다.

따라서 다음 full representation 실험의 기본 방향은 다음과 같다.

- XFeat-only birth를 DA3 depth-prior NEW birth로 교체하거나 보강한다.
- DA3 seed xyz는 우선 fixed generation-zero anchor로 유지한다.
- 처음부터 unrestricted xyz/scale gradient를 다시 켜지 않는다.
- causal voxel merge와 multi-view support count로 중복/floaters를 줄인다.
- fixed-xyz DC/opacity부터 학습한 뒤 geometry freedom을 한 축씩 추가한다.

## 5. 아직 증명하지 않은 것

- 이 실험은 saved causal PCA/sign trace를 재사용한 seed feasibility audit다.
- 아직 105-frame u120 representation optimization에 DA3 seed bank를 연결하지 않았다.
- 2 cm voxel merge는 occupancy deduplication일 뿐 learned fusion이 아니다.
- 전체 NEW precision 0.8663이므로 causal signed mask의 false-positive object에도 seed가
  생긴다. GT object004 precision을 birth gate로 사용할 수는 없다.
- DA3 package가 현재 `oscd` environment에 editable install되어 있지만 repository는
  외부 sibling path `/home/rvl/workspace/github/Depth-Anything-3`에 두었다.

## 6. 산출물

Root:

`outputs/ref_sc3_da3_depth_prior_new_seed_feasibility_20260902/`

- `summary.json`
- `per_frame_metrics.csv`
- `da3_vs_xfeat_reprojection.csv`
- `da3_depth_prior_new_seeds.ply`
- `da3_depth_prior_new_seeds_object004_evaluation_only.ply`
- `object004_depth_seed_geometry.png`
- `frame_0000XX_depth_seed_audit.png`
- `da3_cache/*.npz`

이 feasibility를 실제 fixed-geometry DC-only u120 representation에 연결한 결과는
[`part18-da3-fixed-geometry-dc-only-u120-ko.md`](part18-da3-fixed-geometry-dc-only-u120-ko.md)에 기록한다.
