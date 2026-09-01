# BF30 first-OPEN `C_i=1` + all-OPEN optimizer ablation

## 질문

single-candidate Beta detector의 threshold를 `BF=30`으로 고정했을 때,
Gaussian이 **처음 OPEN되는 순간** change DC를 실제 렌더값 `C_i=1`로
강제 초기화하면 어떻게 되는지 확인했다.

사용자가 수정한 학습 제약도 함께 고정했다.

- `NEVER_OPEN`: 고정 zero-DC occluder로 렌더링하지만 모든 parameter를 freeze
- `OPEN`: sampled training view의 visibility와 무관하게 모든 OPEN row를 optimizer가 선택
- `CLOSED`: 렌더링에서 제외하고 parameter/Adam state를 정확히 보존
- `REOPEN`: 과거에 학습한 persistent DC와 Adam state를 그대로 재사용
- first `OPEN`만 `C_i=1`로 초기화하고 DC Adam row state만 reset

비교가 섞이지 않도록 같은 corrected policy에서 다음 두 run을 새로 실행했다.

1. `preserve`: first OPEN 때 raw DC를 그대로 유지한다. 초기 raw DC가 0이므로 intrinsic render 값은 `0.5`이다.
2. `render_one`: first OPEN 때 `RGB2SH(1) = 1.7724538`로 덮어써 intrinsic render 값을 정확히 `1`로 만든다.

두 run 모두 continuous `ref -> SC1 -> SC2 -> SC3`, 304 frames, seed 0,
16 updates/frame, ACTIVE-only O-SCD clone/split, opacity pruning off 조건이다.

## 구현 계약

`--optimizer-selection all_open`은 각 update에서 visibility mask를 optimizer
selection에 적용하지 않는다. 따라서 OPEN row는 현재 sampled view 밖에 있더라도
선택된다. 실제 image-space gradient가 0인 off-view row도 과거 Adam momentum이
있으면 decay된 momentum에 의해 이동할 수 있다. 이것은 이번 요청의
"현재 view에 보이는 Gaussian만 학습" 제약을 제거한 정확한 의미이다.

`--first-open-dc-initialization render_one`은 lifecycle event 중
`OPEN && new_current_slot == 0`인 row에만 적용한다. DC 이외의 parameter 및
optimizer state는 건드리지 않는다. REOPEN은 초기화하지 않는다.

Detector의 current-frame alpha-T evidence는 first-OPEN DC 초기화보다 먼저
계산된다. 또한 detector evidence는 DC를 직접 사용하지 않는다. 따라서 같은
frame에서 이 DC jump가 곧바로 CLOSE evidence가 될 수는 없다. 다만 이후
DC-driven loss가 geometry/opacity/density를 바꾸고, 이 mutable representation이
다음 frame의 alpha-T evidence에 간접적으로 feedback할 수 있다.

## 결과

| 조건 | SC1 mIoU | SC2 mIoU | SC3 mIoU | 전체 mIoU | 전체 F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| corrected `preserve` | 0.2581 | 0.1408 | 0.0438 | 0.1439 | 0.1860 | 0.7386 | 0.2413 |
| corrected first-OPEN `C_i=1` | 0.2845 | 0.3327 | 0.3455 | 0.3221 | 0.4539 | 0.3968 | 0.6810 |
| 차이 | +0.0265 | +0.1918 | +0.3018 | **+0.1781** | **+0.2680** | -0.3417 | +0.4397 |

first-OPEN `C_i=1`은 corrected preserve보다 recall과 후반 state 보존을 크게
높였다. 그러나 precision이 `0.7386 -> 0.3968`로 붕괴했다. 즉 개선은
정교한 mask 학습보다 강한 positive initialization에 의한 coverage 증가가 크다.

기존 BF30 run은 NEVER_OPEN DC/opacity를 학습하고 OPEN-visible row만 학습한
서로 다른 정책이었다. 그 run의 전체 mIoU/F1 `0.4360/0.5466`과 비교하면,
이번 corrected `C_i=1`은 각각 `-0.1139/-0.0927` 낮다. 이 비교는 first-OPEN
초기화만의 차이가 아니므로 참고 control로만 사용한다.

## DC jump와 한 frame 내 수렴

`render_one`에서 first OPEN된 96,443개 source row의 intrinsic DC render
통계는 다음과 같다.

| 시점 | mean | q05 | q50 | q95 |
|---|---:|---:|---:|---:|
| 덮어쓰기 전 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| 덮어쓰기 직후 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 첫 optimizer update 후 | 0.9999 | 0.9993 | 1.0000 | 1.0007 |
| density event 직전 | 0.9994 | 0.9966 | 0.9992 | 1.0030 |
| frame 종료, surviving source | 0.9980 | 0.9901 | 0.9955 | 1.0078 |

급격한 overwrite가 numerical divergence나 oscillation을 만든 증거는 없다.
반대로 더 중요한 문제는 **16 updates가 이 강한 초기화를 거의 수정하지
못했다는 것**이다. 잘못 OPEN된 row도 한 frame 동안 거의 흰색으로 남으므로,
threshold mask에서 즉시 강한 false positive가 된다. 전체 pre-opt mIoU
`0.3016`에서 post-opt `0.3221`로의 개선은 `+0.0205`에 그쳤다.

## OPEN 직후 CLOSE 위험

`render_one`의 explicit first OPEN 96,443개에 대해 첫 CLOSE latency를
추적했다.

- same-frame CLOSE: `0`
- 1 frame 이내: `0`
- 3 frames 이내: `0`
- 5 frames 이내: `17` (`0.0176%`)
- 최소 CLOSE latency: `4` frames
- CLOSE된 row의 median latency: `158` frames

따라서 우려했던 "색을 1로 바꾸자마자 detector가 surprise해서 즉시 닫는"
현상은 나타나지 않았다. 오히려 반대 방향으로 OPEN이
`36,376 -> 97,111`, final ACTIVE가 `35,982 -> 93,354`로 증가했다.
같은-scene repeated transition도 `442 -> 1,343`으로 늘었다.

## 해석

이번 실험은 두 가지를 분리해서 보여준다.

1. **초기화 안정성:** raw DC를 `1.77245`만큼 갑자기 이동해도 optimizer가
   발산하지는 않는다.
2. **통계적/표현적 안정성:** 하지만 hard positive initialization이 false OPEN을
   거의 되돌리지 못해 mask precision과 lifecycle 안정성을 해친다.

즉 병목은 "gradient가 급격한 jump 때문에 수렴하지 못한다"기보다,
`C_i=1`이라는 강한 결론을 detector가 OPEN을 선언한 즉시 representation에
주입하고, 그 결론을 16-step 저학습률 DC optimization이 교정하기 어렵다는
점이다. 또한 mutable geometry/opacity feedback 때문에 이후 detector trajectory
자체가 달라져 더 많은 OPEN을 만든다.

이 결과만 보면 first OPEN을 무조건 1로 덮는 방식은 최종 방법으로 적합하지
않다. `preserve`보다는 훨씬 낫지만, false-positive 증가가 너무 크다. 후속
실험은 hard `1` 대신 detector confidence에 따른 bounded initialization 또는
DC 초기화와 detector evidence 사이의 feedback을 끊은 control이 더 적절하다.

## 불변식 및 산출물

두 paired run 모두 다음을 통과했다.

- NEVER_OPEN optimizer-selected row 최대값: `0`
- inactive/outside-mask nonzero gradient: `0`
- CLOSED parameter/Adam drift: `0`
- future-view replay access: `0`
- topology integrity: pass
- GT causal-loop access: 없음

산출물 루트:

```text
/tmp/escd_bf30_fixed_never_open_all_open_first_open_init_20260827/
```

주요 파일:

```text
preserve/summary.json
render_one/summary.json
comparison/comparison.json
comparison/comparison.csv
comparison/comparison.png
preserve/video/ref_sc1_sc2_sc3_open_or_never_open_lifespan_events_threshold_timeline.mp4
render_one/video/ref_sc1_sc2_sc3_open_or_never_open_lifespan_events_threshold_timeline.mp4
```
