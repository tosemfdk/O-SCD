# Part 25: reference XFeat scale에서 DA3Metric-Large와 DA3-Small 비교

## 1. 질문

Immutable reference image와 그 image에서 얻은 XFeat--COLMAP 3D correspondence를
사용해 depth scale을 정하면, `depth-anything/DA3METRIC-LARGE`가 기존
`depth-anything/DA3-SMALL`보다 reference GS의 rendered depth와 더 잘 맞는가?

또 reference 구간에서 구한 하나의 scale을 online 구간까지 고정해서 사용할 수 있을
정도로 view 간 scale이 안정적인가?

## 2. 비교 계약

- Instance 1의 396개 reference image에서 균일하게 선택한 16 view를 사용했다.
- 두 모델 모두 같은 single reference image를 입력받았다.
- 두 모델 모두 같은 XFeat keypoint와 그 keypoint에 연결된 COLMAP 3D point를 scale
  anchor로 사용했다.
- Fixed COLMAP pose로 reprojection error가 8 px 이하인 anchor만 사용했다.
- Depth에는 translation을 더하지 않고 positive scale만 fitting했다.
- DA3Metric canonical output은 공식 모델 계약에 따라 다음처럼 metre depth로 변환했다.

```text
metric_depth = raw_depth * mean(fx, fy) / 300
```

- 평가 target은 immutable reference GS의 alpha-weighted rendered camera-z다.
- Render alpha가 0.5 이상인 pixel만 평가했다.
- Scale anchor 자체에만 잘 맞은 효과를 줄이기 위해, 각 XFeat anchor 주변 원본 image
  기준 8 px를 제거한 dense holdout metric을 primary로 사용했다.
- Change GT mask는 사용하지 않았다.

비교한 scale 정책은 세 가지다.

1. `per_view_xfeat`: 각 view마다 XFeat scale을 다시 fitting
2. `first_view_locked`: 첫 reference view의 scale을 모든 view에 고정
3. `reference_bank_median_locked`: 16개 reference view fitted scale의 median 하나를
   모든 view에 고정

## 3. XFeat scale 자체의 안정성

| 모델 | fitted scale 범위 | median | 표준편차 | 변동계수 |
|---|---:|---:|---:|---:|
| DA3Metric-Large | 5.3922--6.2848 | 5.8135 | 0.2244 | 3.84% |
| DA3-Small | 4.8187--12.0865 | 6.5203 | 2.2661 | 30.42% |

두 모델의 raw depth 단위가 다르므로 scale 절댓값끼리는 비교하지 않는다. 비교 대상은
view가 바뀔 때 같은 모델의 scale이 얼마나 흔들리는가이다. Metric-Large의 변동계수는
Small의 약 1/8이었다.

두 모델은 frame마다 동일한 anchor를 사용했고, 유효 anchor 수는 frame당
115--316개, median 249.5개였다. Anchor 위치 자체에서 per-view fit 후 median AbsRel은
Metric-Large 2.25%, Small 4.75%였다.

## 4. Dense rendered-depth holdout 결과

Primary 값은 16개 frame의 holdout median AbsRel을 다시 median한 값이다.

| Scale 정책 | DA3Metric-Large | DA3-Small | view별 승리 |
|---|---:|---:|---:|
| per-view XFeat | 5.14% | 8.02% | Metric 14 / Small 2 |
| 첫 view 고정 | 7.23% | 19.35% | Metric 15 / Small 1 |
| reference-bank median 고정 | **5.50%** | **19.41%** | **Metric 16 / Small 0** |

Reference-bank median 고정 조건의 mean-frame holdout median AbsRel도
Metric-Large 6.30%, Small 20.95%였다. 같은 조건에서 delta<1.25의 frame 평균은
Metric-Large 0.8275, Small 0.5781이었다.

![reference depth error comparison](static/images/reference_da3metric_large_vs_small_xfeat_depth_error.png)

## 5. 해석

이번 reference-view 비교에서는 Metric-Large가 명확히 더 적합했다.

1. Per-view로 매번 scale을 다시 맞춰도 Metric-Large의 dense local depth shape가 더
   정확했다.
2. 더 중요한 결과는 reference-bank median scale 하나를 고정했을 때다.
   Metric-Large는 per-view fit 대비 holdout median AbsRel이 5.14%에서 5.50%로만
   변했지만, Small은 8.02%에서 19.41%로 악화됐다.
3. 따라서 reference image bank에서 Metric-Large의 scene-units-per-metre를 robust하게
   한 번 초기화하고 online frame에 계속 적용한다는 가설은 reference 구간에서
   지지된다.

앞서 changed online frame 하나에서 scale을 처음 잡고 Metric-Large depth를 바로 사용한
실험이 실패한 것은 Metric-Large 자체의 reference-depth 품질이 낮아서라고 보기 어렵다.
변경 pixel이 포함된 online frame에서 최초 scale과 NEW/REMOVE sign을 동시에 잠근 실험
설계가 더 큰 confound였다. 다음 online 실험에서는 reference bank에서 미리 계산한
`scene_units_per_metre = 5.813452`를 고정한 뒤, sign detector와 seed gate를 별도로
검증해야 한다.

## 6. 한계

- Reference GS와 COLMAP XFeat anchor는 같은 reference reconstruction에서 유래하므로
  완전히 독립적인 metric-depth GT는 아니다.
- 다만 평가 pixel은 XFeat anchor 주변을 제거한 dense holdout이며, 두 모델에 동일한
  target과 anchor를 적용했다. 따라서 모델 간 상대 비교에는 사용할 수 있다.
- 이번 Small 조건은 공정한 image-prior 비교를 위해 single-image inference를 사용했다.
  현재 online pipeline의 pose-conditioned past-8-view Small과 직접 같은 조건은 아니다.
- 이 결과만으로 changed online image의 NEW surface depth 정확도까지 증명되지는 않는다.

## 7. 재현 경로

Runner:

```text
experiments/compare_reference_da3_metric_small_depth.py
```

Output:

```text
outputs/reference_da3metric_large_vs_small_xfeat_20260904/
  summary.json
  per_view_metrics.csv
  depth_error_comparison.png
  da3metric_cache/
  da3small_cache/
```

검증:

- reference views: 16
- XFeat anchor median/min/max: 249.5 / 115 / 316
- change GT access: 0
- dense anchor-neighbourhood holdout 적용

## 8. Runtime 및 비용

RTX A6000, input `941 x 528`, DA3 `process_res=504`, batch 1에서 CUDA synchronize를
포함해 50회 steady-state API inference를 측정했다. 이미지 preprocessing과 prediction
conversion을 포함한 시간이다.

| 모델 | median / frame | 평균 / frame | 단독 FPS | 최초 inference | peak allocated VRAM |
|---|---:|---:|---:|---:|---:|
| DA3Metric-Large | 34.69 ms | 34.87 ms | 28.7 | 108.12 ms | 1.43 GiB |
| DA3-Small | 19.59 ms | 19.65 ms | 50.9 | 70.82 ms | 0.23 GiB |

Metric-Large는 Small보다 약 1.77배 느리고 frame당 약 15.2 ms가 추가된다. 304 frame을
depth inference만 순차 처리하면 Metric-Large 약 10.6초, Small 약 6.0초에 해당한다.
실제 representation training에서는 Gaussian render와 16/120회 optimizer update가 함께
실행되므로 전체 runtime 배수는 이 pure-depth 배수보다 작다.

XFeat 관련 측정:

| 작업 | 시간 | 반복 여부 |
|---|---:|---|
| Detector cold initialization/trace | 1.22 s | process당 1회 |
| XFeat feature extraction | 2.07 ms median | image당 |
| 16-view reference bank load/feature/3D association | 3.35 s | 사전 초기화 1회 |
| 준비된 약 253 anchor의 geometry filter + scale fit | 2.95 ms | scale 계산당 |
| 현재 구현의 query-to-16-reference matching | 164.94 ms | 호출할 때마다 |

권장 실행은 reference 16 view에서 Metric-Large scale median을 **offline/initialization에서
한 번만 계산**하고 online frame에는 `5.813452`를 그대로 사용하는 것이다. 이 경우
online XFeat scale fitting 비용은 0이다. Pose localization을 위해 이미 계산한 XFeat를
재사용하더라도 scale fit 자체는 약 3 ms로 작지만, 현재 Python 구현으로 16 reference와
매번 descriptor matching을 다시 하면 약 165 ms가 추가되어 Metric-Large inference보다
훨씬 비싸다.

Runtime raw 결과:

```text
outputs/reference_da3metric_large_vs_small_xfeat_20260904/runtime_benchmark.json
```
