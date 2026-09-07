# Part 19: R_change free-space DA3 birth + NEVER_OPEN BF30 ablation

## 0. 현재 요약과 평가 지표

이 문서에서 주 성능 지표는 이전 실험들과 동일하게 다음 두 값이다.

- **mIoU**: 각 frame의 IoU를 먼저 계산한 뒤 105 frames에서 평균한
  `mean_frame_iou`
- **F1**: 각 frame의 F1을 먼저 계산한 뒤 평균한 `mean_frame_f1`

`Overall IoU/F1`은 모든 frame의 TP/FP/FN을 합친 뒤 한 번 계산하는 micro aggregate다.
오류 구성과 precision/recall tradeoff를 분석하기 위한 보조 지표이며, 이 문서에서
말하는 mIoU/F1이 아니다.

지금까지의 핵심 결과는 다음과 같다.

| 설정 | Updates / sampling | Seed cap | mIoU | Mean-frame F1 |
|---|---|---:|---:|---:|
| Base R_change, DA3 없음 | u120 current-exact | - | 0.5182 | 0.6700 |
| DA3 direct ACTIVE + free-space occupancy | u120 current-exact | 12k | 0.5638 | 0.7129 |
| DA3 NEVER_OPEN + BF30 | u120 current-exact | 12k | **0.5808** | **0.7276** |
| DA3 NEVER_OPEN + BF30 | u120 current-exact | non-binding | 0.5799 | 0.7269 |
| Base R_change, DA3 없음 | u16 O-SCD replay | - | 0.4980 | 0.6503 |
| DA3 NEVER_OPEN + BF30 | u16 O-SCD replay | non-binding | 0.5594 | 0.7081 |

따라서 현재 해석은 다음과 같다.

1. Base R_change center-voxel occupancy만 추가한 효과는 사실상 중립이다.
2. DA3 seed를 즉시 ACTIVE로 출력하지 않고 `NEVER_OPEN -> BF30 OPEN`으로 확인하면
   u120에서 mIoU/F1이 direct ACTIVE 대비 `+0.0170/+0.0147` 증가한다.
3. 12k cap을 제거해도 u120 mIoU/F1은 `-0.0009/-0.0007`만 변하지만, seed는
   마지막 frame까지 계속 증가한다. 정확도는 거의 유지되지만 자연 포화된 것은 아니다.
4. 원본 O-SCD 방식의 u16 causal replay로 바꾸면 non-binding u120 대비
   mIoU/F1이 `-0.0206/-0.0188` 낮아진다. 다만 matched base 대비 DA3의 mIoU 이득은
   u120 `+0.0618`, u16 `+0.0613`으로 거의 같다.
5. 모든 비교에서 DA3 seed의 xyz/scale/rotation/opacity는 고정하고 DC만 학습했다.
   Densification과 pruning은 사용하지 않았다.

## 1. 질문

기존 DA3 fixed-geometry u120 run은 causal NEW sign mask에서 만든 seed를 즉시
ACTIVE로 열고 DC를 학습했다. 이번 ablation은 DA3를 geometry proposal로만 사용하고
다음 구조가 성능을 높이는지 확인한다.

```text
causal SAM/PCA NEW mask
  -> DA3 depth-prior candidate
  -> reject voxels already occupied by R_change
  -> append as NEVER_OPEN
  -> pre-optimization raw-cue BF30 detector
  -> OPEN rows only: DC-only u120 training and learned output
```

기본 평가는 `ref -> SC3` 105 frames, fixed pose, image당 120 current-frame updates로
수행했다. 이후 Section 10에서 12k cap을 제거하고, Section 11에서 원본 O-SCD의
u16 causal replay schedule로 바꾼 결과까지 확장한다.

## 2. 구현 계약

### 2.1 Free-space birth

- DA3 inference: 현재 frame까지의 past-only pose-conditioned window, 최대 8 views
- DA3 depth는 immutable reference GS depth에 positive scale-only로 robust alignment
- signed NEW intersection:
  - raw change cue `>= 0.5`
  - PCA signed score가 설정된 NEW 방향 threshold를 통과
  - 교집합 mask를 1 pixel erosion
- aligned DA3 depth/reference depth/confidence가 모두 valid
- valid NEW candidate 내부 confidence 40th percentile 이상, 즉 상위 약 60%
- reference보다 충분히 앞에 있는 점만 유지:

```text
aligned_DA3_depth <= reference_depth - max(0.03 m, 0.015 * reference_depth)
```

- 4x4 image cell마다 confidence가 가장 높은 후보를 선택하고 frame당 최대 2,048개
- 선택된 pixel을 current camera pose로 3D backprojection
- voxel size: `0.02 m`
- 초기 occupancy: reference/R_change base Gaussian 1,283,501 rows
- unique occupied base voxels: `514,810`
- 이후 승인된 DA3 seed voxel을 즉시 occupancy에 추가
- backprojected 2 cm voxel이 base R_change center voxel이나 과거 승인 seed voxel과
  겹치지 않을 때만 최종 birth
- 기본 controlled run의 안전 cap: 12,000 rows

기존 run의 occupancy set은 빈 상태에서 시작하여 DA3 seed끼리만 중복을 막았다.
이번 run은 base R_change center voxel까지 birth 전에 점유한다.

### 2.2 NEVER_OPEN detector

- accepted seed의 초기 interval: `start=end=+inf`
- 초기 DC: RGB zero
- detector: `lifespan_gate_beta`, Bayes factor `30`
- stable CLOSED prior: `Beta(1,10)`
- reset candidate prior: `Beta(1,1)`
- detector input: base + 모든 DA3 seed의 fixed geometry/opacity에 대한 raw binary
  SAM cue alpha-transmittance evidence
- detector update는 current frame representation optimization 전에 한 번만 수행
- learned seed DC, post-optimization render, replay view, GT는 detector input으로 사용하지 않음

`NEVER_OPEN` seed는 기존 representation contract와 동일하게 fixed black occluder로
렌더링하고, `OPEN` seed만 learned DC를 사용한다. `CLOSED` seed는 렌더링하지 않는다.

## 3. Controlled comparison

세 조건을 비교했다.

| 조건 | Occupancy | Seed activation | mIoU | Mean-frame F1 |
|---|---|---|---:|---:|
| 기존 DA3 direct | prior DA3 seed only | immediate ACTIVE | 0.5639 | 0.7130 |
| Occupancy control | base R_change + seed | immediate ACTIVE | 0.5638 | 0.7129 |
| 제안 방식 | base R_change + seed | NEVER_OPEN + BF30 | **0.5808** | **0.7276** |

주 지표상 occupancy control과 제안 방식의 차이는 mIoU `+0.0170`, mean-frame F1
`+0.0147`이다. 아래 micro aggregate와 NEW-only 수치는 왜 성능이 변했는지를 보기
위한 보조 분석이다.

Occupancy control과 제안 방식의 직접 차이는 다음과 같다.

| Scope | Metric | Direct ACTIVE | NEVER_OPEN BF30 | Delta |
|---|---|---:|---:|---:|
| Overall | Precision | 0.6958 | **0.7309** | **+0.0350** |
| Overall | Recall | **0.7988** | 0.7819 | -0.0170 |
| Overall | IoU | 0.5921 | **0.6071** | **+0.0150** |
| Overall | F1 | 0.7438 | **0.7555** | **+0.0117** |
| NEW | Precision | 0.5623 | **0.6634** | **+0.1011** |
| NEW | Recall | **0.4424** | 0.3816 | -0.0608 |
| NEW | Pixel-weighted IoU | **0.3291** | 0.3197 | -0.0094 |
| NEW | Mean-frame IoU | 0.2784 | **0.2810** | +0.0026 |

BF30 gating은 false-positive seed 출력을 크게 줄여 micro precision/IoU를 높였지만,
아직 OPEN하지 않았거나 CLOSE된 true NEW seed도 제거하여 aggregate NEW recall을 낮췄다.
따라서 전체 SCD 성능은 개선됐으나 NEW-only pixel-weighted IoU는 개선되지 않았다.

## 4. Object004 / Object010

Object004가 보이는 frame 40--86의 결과는 다음과 같다.

| Metric | Direct ACTIVE | NEVER_OPEN BF30 | Delta |
|---|---:|---:|---:|
| mean IoU | 0.3529 | **0.3632** | **+0.0103** |
| mean precision | 0.4127 | **0.4606** | **+0.0479** |
| mean recall | **0.7124** | 0.6241 | -0.0883 |

Object004 역시 detector가 외부 spill을 줄여 mean IoU를 높였지만 recall은 낮아졌다.
Object010 REMOVE mask 내부의 NEW-sidecar false-positive pixel은 두 조건 모두 `0`이다.

대표 frame에서는 view-dependent tradeoff가 보인다.

| Frame | Direct NEW IoU | BF30 NEW IoU | Direct object004 IoU | BF30 object004 IoU |
|---:|---:|---:|---:|---:|
| 48 | **0.5210** | 0.4251 | **0.4901** | 0.4286 |
| 52 | **0.4029** | 0.3568 | **0.6062** | 0.5651 |
| 56 | **0.7217** | 0.6932 | 0.6073 | **0.6377** |
| 64 | **0.6187** | 0.6179 | 0.4884 | **0.5279** |
| 80 | 0.4888 | **0.6026** | 0.3986 | **0.5152** |
| 84 | 0.4965 | **0.5868** | 0.4345 | **0.5288** |

초기 view에서는 confirmation delay로 recall 손실이 크고, 충분한 detector history가 쌓인
후반 view에서는 precision gate가 큰 이득을 준다.

## 5. Lifecycle 결과

- first OPEN: local frame `15`
- total proposed rows: `12,000`
- final OPEN: `9,658`
- final NEVER_OPEN: `1,325`
- final CLOSED: `1,017`
- OPEN events: `11,470`
- CLOSE events: `1,812`
- inferred REOPEN events: `795`
- final live candidates: `2,162`

같은 SC3 상태에서도 CLOSE/REOPEN이 발생하므로 detector chattering은 완전히 해결되지
않았다. 이 ablation의 sidecar는 current-state rendering을 위해 REOPEN 시 current interval을
교체하며 전체 historical interval list는 저장하지 않는다.

## 6. Occupancy 가설 결과

Base R_change occupancy를 추가해도 direct-active overall IoU 변화는 `-0.00004`뿐이었다.
Frame79까지 기존 방식은 11,947 seed, R_change occupancy 방식은 11,901 seed로 base
occupancy가 초기 46개 birth를 지연시켰지만, frame80에 다른 free voxel을 채워 둘 다
12,000 cap에 도달했다.

따라서 center voxel occupancy만으로는 seed 증가가 자연스럽게 멈추지 않았다. 원인은
다음과 같다.

1. reference-front-depth 조건 때문에 대부분의 genuine NEW candidate는 원래 base surface와
   다른 voxel에 있다.
2. DA3 depth/view jitter가 동일 surface 주변의 이웃 voxel을 계속 만든다.
3. center voxel만 비교하므로 Gaussian footprint가 이미 덮은 공간도 새 center로 승인된다.

다음 occupancy 실험은 단순 center equality가 아니라 seed scale을 반영한 radius occupancy,
neighbor-voxel dilation, 또는 rendered coverage + multi-view novelty 정지를 사용해야 한다.
12k cap은 현재도 safety bound로 필요하다.

## 7. Runtime 및 audit

| 조건 | Runtime | Final trainable OPEN rows |
|---|---:|---:|
| Direct ACTIVE + occupancy | 325.60 s | 12,000 |
| NEVER_OPEN BF30 + occupancy | **288.41 s** | 9,658 |

Detector probe가 추가됐는데도 runtime이 `37.18 s` 감소했다. 모든 proposed row를 u120으로
학습하지 않고 현재 OPEN row만 projected DC training에 넣은 효과가 더 컸다.

Audit:

- reference 1,283,501 rows와 topology bitwise unchanged
- DA3 xyz/SH-rest/opacity/scaling/rotation bitwise unchanged
- seed optimizer parameter: DC only
- future-view access: `0`
- GT birth access: `0`
- learned seed DC detector input: `0`
- REMOVE metric: direct control과 bitwise-equivalent aggregate counts

## 8. 결론

`R_change free-space birth + NEVER_OPEN BF30`은 direct activation보다 주 지표인
mIoU/mean-frame F1을 `+0.0170/+0.0147` 높였고 object004 mean IoU도 `+0.0103`
높였다. 개선은 occupancy가 아니라 detector의 precision gating에서 왔다. Micro
aggregate Overall IoU/F1의 변화 `+0.0150/+0.0117`은 이 해석을 보조한다.

그러나 NEW-only aggregate IoU/F1은 `-0.0094/-0.0107` 낮아졌다. 현재 method를 최종
NEW representation으로 채택하기 전에 confirmation delay와 false CLOSE를 줄이는
view-consistent calibration이 필요하다.

후속 DA3 scale calibration 실험은
[`part22-xfeat-localization-da3-scale-ko.md`](part22-xfeat-localization-da3-scale-ko.md)에
정리한다. Pose-localization XFeat 2D↔3D inlier를 metric anchor로 사용하면 동일 anchor
기준 depth 상대오차 중앙값은 감소하지만, local DA3 distortion과 single-view birth 문제는
남는다.

## 9. 산출물

제안 방식:

`outputs/ref_sc3_da3_neveropen_rchange_occupancy_bf30_dc_only_u120_20260903/`

- `summary.json`
- `frame_metrics.csv`
- `checkpoint.pt`
- `da3_new_seeds_dc_only.ply`
- `frame_000048/52/64/80/84_dc_only.png`
- `comparison_to_direct_active.json`
- `comparison_to_direct_active.png`

Occupancy control:

`outputs/ref_sc3_da3_direct_active_rchange_occupancy_dc_only_u120_20260903/`

## 10. 12k cap 제거 audit

`NEVER_OPEN + BF30` 조건에서 12k cap만 제거한 효과를 확인했다. 구현의 고정 크기
detector buffer는 유지하되, 105 frames와 frame당 최대 2,048 birth로 가능한 이론적
상한 `215,040`을 `--max-total-seeds`로 주어 실행 중에는 cap이 절대 바인딩되지 않게
했다. 그 외 조건은 12k run과 같다.

| Metric | 12k cap | Non-binding cap | Delta |
|---|---:|---:|---:|
| Final seed | 12,000 | 14,217 | +2,217 |
| Final OPEN | 9,658 | 10,978 | +1,320 |
| Final NEVER_OPEN | 1,325 | 2,069 | +744 |
| Final CLOSED | 1,017 | 1,170 | +153 |
| **mIoU** | **0.5808** | 0.5799 | -0.0009 |
| **Mean-frame F1** | **0.7276** | 0.7269 | -0.0007 |
| Overall IoU | **0.6071** | 0.6063 | -0.0008 |
| Overall F1 | **0.7555** | 0.7549 | -0.0006 |
| NEW IoU | 0.3197 | **0.3201** | +0.0004 |
| NEW mean-frame IoU | 0.2810 | **0.2813** | +0.0003 |
| Object004 active-frame mean IoU | **0.3632** | 0.3631 | -0.0002 |

Frame 1--79의 seed lifecycle count와 모든 evaluation confusion count는 두 run에서
정확히 같았다. Cap 차이는 frame80부터 발생했으며, frame81--105에서 no-cap run은
frame당 평균 88.32개 seed를 계속 받았다. 마지막 frame105에도 68개가 추가됐으므로
14,217은 자연 포화점이 아니라 스트림 종료 시점의 개수다. 더 긴 스트림에서는 현재의
center-voxel occupancy만으로 seed 증가가 계속될 가능성이 높다.

추가 seed는 NEW aggregate recall을 `+0.0024` 높였지만 precision을 `-0.0054` 낮췄다.
그 결과 NEW IoU는 사실상 중립인 `+0.0004`였고 주 지표 mIoU/F1은
`-0.0009/-0.0007` 낮아졌다. 따라서 12k cap 제거는 정확도 개선으로 채택하지 않는다.

전체 wall-clock은 `288.41 s -> 397.01 s`였으나 no-cap run의 cap 도달 전
frame50--57에 외부 실행 변동으로 보이는 8.6--16.1초 spike가 있었다. 정책이 실제로
갈라진 frame80--105만 합하면 `72.07 s -> 93.08 s`였으며, 이 값도 CUDA-event 기반
stage profile이 아니므로 추가 seed의 순수 인과 비용으로 단정하지 않는다.

No-cap audit 산출물:

`outputs/ref_sc3_da3_neveropen_rchange_occupancy_bf30_dc_only_u120_nocap_20260903/`

- `summary.json`
- `frame_metrics.csv`
- `checkpoint.pt`
- `da3_new_seeds_dc_only.ply`
- `comparison_to_12k.json`
- `run.log`

동일 no-cap 조건의 105-frame post-update confusion visualization은 다음 rerun에
저장했다. 각 panel의 confusion count와 `frame_metrics.csv`를 frame마다 exact-compare했고,
전체 metric과 최종 seed/lifecycle은 위 no-cap audit과 동일했다.

`outputs/ref_sc3_da3_neveropen_rchange_occupancy_bf30_dc_only_u120_nocap_confusion_20260903/`

- `temporal_confusion/scene_change3_confusion.gif` (`105 frames`, `720×373`, `160 ms/frame`)
- `temporal_confusion/panels/`
- `temporal_confusion/raw_confusion/`
- `temporal_confusion/pred_binary/`

## 11. 원본 O-SCD replay u16 비교

기존 u120은 신규 current frame 하나를 120회 반복 학습하는 `current_exact`이었다.
원본 online O-SCD 조건에 맞춘 u16은 매 신규 frame을 causal view pool에 추가한 뒤,
16 update마다 다음 방식으로 training view를 다시 뽑는다.

```text
확률 0.33: current view
확률 0.67: 지금까지 관측한 causal pool에서 uniform sample
```

Uniform branch에서도 current view가 뽑힐 수 있다. Seed detector는 이전과 동일하게 신규
current frame의 pre-optimization raw cue만 한 번 사용하며 replay view는 detector로
들어가지 않는다. Representation replay에서는 현재 committed OPEN seed를 sampled 과거
camera에 투영하여 학습한다. Random seed는 0으로 고정했다.

| Metric | u120 current-exact | u16 O-SCD replay | Delta |
|---|---:|---:|---:|
| **mIoU** | **0.5799** | 0.5594 | -0.0206 |
| **Mean-frame F1** | **0.7269** | 0.7081 | -0.0188 |
| Overall precision | 0.7293 | **0.7890** | +0.0597 |
| Overall recall | **0.7823** | 0.6817 | -0.1006 |
| Overall IoU | **0.6063** | 0.5766 | -0.0297 |
| Overall F1 | **0.7549** | 0.7314 | -0.0234 |
| NEW precision | 0.6580 | **0.7206** | +0.0626 |
| NEW recall | **0.3840** | 0.1696 | -0.2145 |
| NEW IoU | **0.3201** | 0.1591 | -0.1610 |
| NEW mean-frame IoU | **0.2813** | 0.1512 | -0.1301 |
| Object004 active-frame mean IoU | **0.3631** | 0.1881 | -0.1749 |

105 frames에서 총 1,680 update 중 current view가 실제 선택된 횟수는 599회, 과거
view는 1,081회였다. Future training view access는 0이었다. Birth, final seed 14,217개,
OPEN/NEVER_OPEN/CLOSED `10,978/2,069/1,170`, 전체 OPEN/CLOSE event는 u120과 정확히
같았다. Detector/lifecycle은 representation update 횟수와 replay로부터 독립이라는
계약도 유지됐다.

u16은 u120보다 보수적으로 학습되어 precision은 올라갔지만 recall, 특히 NEW recall이
크게 낮아졌다. 다만 동일 run 내부의 base 대비 DA3 추가 이득은 mIoU 기준 u120
`+0.0618`, u16 `+0.0613`, mean-frame F1 기준 u120 `+0.0569`, u16 `+0.0578`로
거의 같다. 즉 DA3 seed의 상대적 이득이 사라진 것보다는 전체 representation 학습량
감소가 절대 성능을 낮춘 결과에 가깝다.

이 비교는 사용자가 요청한 실제 online protocol이지만 update budget과 sampling schedule을
동시에 바꾸므로 순수한 update-count ablation으로 해석하지 않는다. Replay projected-coverage
구현은 sampled camera마다 Python projection을 다시 만들기 때문에 runtime 최적화 대상이며,
현재 속도를 원본 CUDA renderer의 u16 비용으로 해석해서도 안 된다.

산출물:

`outputs/ref_sc3_da3_neveropen_rchange_occupancy_bf30_dc_only_u16_oscdreplay_nocap_20260903/`

- `summary.json`
- `frame_metrics.csv`
- `checkpoint.pt`
- `da3_new_seeds_dc_only.ply`
- `comparison_to_u120_current_exact.json`
- `run.log`
