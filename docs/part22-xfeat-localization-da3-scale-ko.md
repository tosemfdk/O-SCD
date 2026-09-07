# Part 22: XFeat localization anchor 기반 DA3 depth scale

## 1. 목적

기존 DA3 seed replay는 현재 프레임의 monocular depth를 immutable reference GS의
rendered depth에 맞추는 단일 positive scale을 사용했다. 이 방식은 dense anchor를 얻기
쉽지만 reference render의 local depth 오류와 변경 영역 누출에 영향을 받을 수 있다.

이번 변경은 NEW로 승격된 seed가 아니라, **현재 프레임 pose localization에 사용 가능한
XFeat 2D↔COLMAP 3D correspondence 전체**를 metric depth anchor로 사용한다.

## 2. Scale 계산

각 현재 프레임에서 다음 순서로 계산한다.

1. XFeat query descriptor를 16개 reference frame의 descriptor와 mutual-NN matching한다.
2. reference keypoint와 연결된 COLMAP 3D point를 가져온다.
3. replay가 실제 사용하는 fixed pose로 3D point를 현재 이미지에 reprojection한다.
4. reprojection error가 8 px 이하인 geometric inlier만 유지한다.
5. 같은 query keypoint 또는 같은 3D landmark가 여러 reference에서 반복되면 가장 낮은
   reprojection error 하나만 남긴다.
6. 현재 이미지 keypoint를 image K에서 DA3 output K로 직접 옮긴다.
7. DA3 native depth grid에서 bilinear sampling하고, metric camera-z와의 log ratio를
   median + MAD로 robust fitting한다.

핵심 식은 다음과 같다.

```text
r_j = log(z_metric,j) - log(d_DA3,j)
scale = exp(median(r_j among MAD inliers))
```

전체 DA3 depth를 upsampling하지 않는다. XFeat anchor만 native grid에서 bilinear
sampling한다. 최소 anchor는 16개이며 부족한 프레임은 기존 rendered-reference scale로
fallback한다. Q, SAM/PCA sign, GT mask, 승격된 NEW seed는 scale fitting에 사용하지 않는다.

## 3. 전체 SC1→SC2→SC3 replay 결과

설정은 기존 viewer replay와 동일하게 Q≥0.8, 2 cm center-voxel occupancy,
current/past-only DA3 8-view window를 사용했다. 기존 DA3 cache를 재사용해 depth inference
자체는 바꾸지 않았다.

- DA3 처리 프레임: 297
- XFeat scale 사용: 295
- rendered-reference fallback: 2
- 프레임당 geometric anchor 중앙값: 159
- geometric anchor P90: 216
- reprojection error 중앙값: 2.45 px
- reprojection error P90: 3.15 px

동일한 XFeat metric anchor에서 두 scale을 평가한 median absolute relative camera-z error:

| 범위 | 기존 reference scale | XFeat scale | 상대 감소 | XFeat가 낮은 프레임 |
|---|---:|---:|---:|---:|
| 전체 | 3.662% | 3.167% | 13.52% | 213/295 |
| SC1 | 3.545% | 2.998% | 15.43% | 63/88 |
| SC2 | 3.535% | 3.168% | 10.39% | 81/102 |
| SC3 | 3.887% | 3.454% | 11.13% | 69/105 |

따라서 사용자가 제안한 localization-anchor scale은 같은 metric correspondence 기준으로
전 구간 중앙 오차를 줄였다. 다만 82/295 프레임에서는 기존 scale보다 높았으므로 local
DA3 distortion까지 해결한 것은 아니다.

## 4. Seed replay 변화

Accepted seed 수:

| Scene | 기존 rendered-reference scale | XFeat scale |
|---|---:|---:|
| SC1 | 4,299 | 4,741 |
| SC2 | 7,676 | 6,361 |
| SC3 | 8,568 | 9,696 |
| 전체 | 20,543 | 20,798 |

Seed 수의 증감은 metric 정확도 개선 자체가 아니라, 작은 scale 차이가 front-of-reference
gate와 2 cm occupancy 경계를 통과시키는 방식이 달라진 결과다. 별도의 GT metric 없이
seed 수가 많거나 적다는 이유만으로 개선을 주장하지 않는다.

문제가 관찰됐던 SC1 frame 58에서는:

- 기존 scale: 0.96969, candidate/accepted 233/179
- XFeat scale: 0.97613, candidate/accepted 222/181
- XFeat anchor error: 2.488%
- 같은 anchor의 기존 scale error: 2.942%

metric anchor 오차는 줄었지만 accepted seed는 179→181로 거의 같았다. 따라서 frame 58의
잘못 쌓이는 point 문제는 **global scale 하나만의 문제는 아니다**. DA3 local depth shape,
single-view birth, footprint를 무시한 center-voxel occupancy가 여전히 남는다.

## 5. 구현 및 산출물

구현:

- `temporal/depth_prior_new_seeding.py`
  - sparse native-grid bilinear sampling
  - robust metric-anchor scale fit
- `experiments/build_causal_da3_seed_replay.py`
  - cached query XFeat + reference COLMAP 3D matching
  - fixed-pose geometric inlier 및 중복 제거
  - anchor 부족 시 rendered-reference fallback
  - per-frame scale source/error audit
- `run_bayesian_detector_viewer.sh`
  - 새 XFeat-scale replay를 기본으로 표시

산출물:

`outputs/causal_da3_seed_replay_scene123_xfeat_localization_scale_q080_20260903/`

- `da3_seed_replay.pt`
- `frame_rows.json`
- `summary.json`
- `depth_scale_comparison.json`

Audit:

- future-view access: 0
- GT birth/scale access: 0
- cue/sign을 scale input으로 사용: false
- sign mapping reset: 0

## 6. 후속: NEW-sign strong SAM 상위 40% birth gate

Viewer에서 약한 SAM diff를 숨긴 것과 실제 seed birth 조건이 불일치하던 부분을
수정했다. 부호 판정을 위한 누적 front-depth evidence는 기존 strong sign 전체를
유지하지만, 실제 birth는 다음 교집합에서만 허용한다.

```text
Q >= 0.8
AND locked NEW sign
AND 해당 NEW sign의 |SAM Delta-PCA diff| 상위 40%
AND DA3 confidence/front-of-reference
AND 2 cm voxel unoccupied
```

이 gate를 추가한 causal replay의 accepted seed는 다음처럼 감소했다.

| Scene | XFeat scale만 | + SAM top 40% | 변화 |
|---|---:|---:|---:|
| SC1 | 4,741 | 1,651 | -65.2% |
| SC2 | 6,361 | 2,862 | -55.0% |
| SC3 | 9,696 | 4,115 | -57.6% |
| 전체 | 20,798 | 8,628 | -58.5% |

SC1 frame 58은 candidate/accepted가 `222/181 -> 77/58`로 감소했다. 이는 약한
signed response의 반복 voxel birth를 직접 줄인 결과다. 아직 GT 기반 representation
metric을 계산한 것은 아니므로, seed 감소 자체를 mIoU 개선으로 해석하지 않는다.

산출물:

`outputs/causal_da3_seed_replay_scene123_xfeat_scale_q080_samtop40_20260903/`
