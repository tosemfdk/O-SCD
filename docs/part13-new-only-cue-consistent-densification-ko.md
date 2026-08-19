# Part 13. Confirmed NEW bank만 cue-consistent densification

> 작성일: 2026-08-14
> 비교: fixed-topology two-sign DC-only vs XFeat seed + NEW-only child densification
> 핵심 결론: **reference Gaussian은 그대로 둔 채 confirmed NEW sidecar만 조밀하게 만들 수 있다. 기존 background GS가 이미 cue를 설명하여 joint gradient가 약해지는 문제는 seed-only projected coverage loss로 분리했다. SceneChange1/2/3 전체 replay에서 overall IoU는 +0.002686, NEW IoU는 +0.003447 증가했다.**

## 1. 문제와 설계

기존 XFeat triangulation seed는 3D 위치 precision은 높았지만 `SC1/2/3 = 8/201/220`
개뿐이고 투영 footprint도 약 `1.4--1.8 px`여서 NEW mask를 거의 덮지 못했다.

또 seed와 reference change memory를 한 번에 render하여 같은 loss를 주면 이미 배경
reference GS에 학습된 DC가 target을 설명할 수 있다. 그러면 새 seed에 필요한 gradient가
약해질 수 있다. 이번 구현은 두 문제를 분리한다.

1. posterior `0.8` 합의 후 confirmed NEW sign에서만 densification을 연다.
2. 현재 signed cue의 아직 덮이지 않은 64x64 cell을 찾는다.
3. 가장 가까운 visible XFeat seed의 camera depth를 child의 depth anchor로 사용한다.
4. 최근 causal inference view들에 재투영하여 같은 signed cue 내부 support를 검사한다.
5. 3-view 이상, support ratio `>=0.65`, baseline `>=0.1`, view angle `>=1.5 deg`인
   point만 fixed child seed로 추가한다.
6. child는 parent seed의 DC를 복사하지만 xyz/scale/opacity/rotation은 optimizer에
   넣지 않는다.
7. DC 학습은 reference bank를 제외한 seed-only projected footprint에 직접 적용한다.

따라서 이것은 unrestricted Gaussian split이 아니라 **XFeat depth anchor로 제한된
online signed-cue visual-hull expansion**이다. GT, future view, reference depth는 birth와
학습에 쓰지 않는다.

## 2. NEW-only 안전 계약

```text
R_ref bank
  topology/xyz/DC/opacity/scale/rotation: densifier가 접근하지 않음

NEW sidecar bank
  XFeat triangulated parent
      -> under-covered signed cue cell
      -> parent camera depth로 unproject
      -> causal multi-view signed-cue carving
      -> fixed child row
```

`append_densified_children`은 parent row가 기존 NEW sidecar 범위를 벗어나면 실패한다.
실험 종료 audit에서 reference Gaussian 1,283,501개의 모든 base tensor가 bitwise 동일함을
확인했다. Generic densification/pruning 호출은 여전히 0이다.

## 3. Gradient starvation 처리

최종 branch는 `base+seed` joint render로 평가하지만 seed DC supervision은 별도로 준다.

```text
target = confirmed NEW signed cue ∩ O-SCD candidate
seed_score = differentiable projection of active seed footprints only
L_seed = mean_NEW(1-sigmoid(seed_score))
       + 0.1 mean_nonNEW(sigmoid(seed_score))
```

즉 background reference GS가 target을 이미 설명하더라도 `seed_score`에는 들어오지 않기
때문에 seed gradient는 사라지지 않는다. Geometry는 고정되어 있으므로 현재 camera로
xyz와 scale을 투영한 isotropic kernel의 weight는 detach하고 `seed_dc`만 학습한다.

짧은 SceneChange2 gradient audit에서는 기존 joint loss의 seed gradient norm median이
`2.08e-8`, base-independent seed-only loss는 `3.12e-5`였다. 약 1,500배 큰 직접 gradient로
사용자가 우려한 설명-away 경로가 실제로 존재함과 이를 우회할 수 있음을 확인했다.

## 4. 전체 304-frame 결과

### 4.1 Seed 수와 위치 품질

| Scene | XFeat parent | Densified child | Total | Future-view NEW precision | REMOVED leakage |
|---|---:|---:|---:|---:|---:|
| SC1 | 8 | 42 | 50 | 1.000 | 0.000 |
| SC2 | 201 | 152 | 353 | 0.823 | 0.000 |
| SC3 | 220 | 439 | 659 | 0.796 | 0.018 |

Precision은 seed 중심점이 promotion 이후 화면 안에 들어온 표본만 분모로 한 point-location
진단값이다. Densification으로 coverage가 커진 대신 SC2/3의 중심 precision은 sparse
XFeat-only 결과의 약 0.90보다 낮아졌다. 이는 parent depth를 component 전체에 확장하는
현재 근사가 silhouette boundary와 depth discontinuity에서 leakage를 만든다는 뜻이다.

파랑은 XFeat parent, 자홍은 NEW-only densified child, 주황은 평가용 NEW GT 밖에 투영된
점이다. 초록 tint는 평가에만 사용했다.

### 4.2 Mask metric

| Scene | Full ΔIoU | Full ΔF1 | NEW ΔIoU | NEW ΔF1 | REMOVED ΔIoU |
|---|---:|---:|---:|---:|---:|
| SC1 | +0.000009 | +0.000010 | -0.000076 | -0.000090 | +0.000013 |
| SC2 | +0.001149 | +0.001142 | +0.002441 | +0.003422 | -0.001105 |
| SC3 | +0.005999 | +0.005053 | +0.005636 | +0.007101 | -0.002678 |
| **Overall** | **+0.002686** | **+0.002591** | **+0.003447** | **+0.004464** | **-0.001296** |

Overall full metric은 `IoU 0.438561 -> 0.441246`, `F1 0.609722 -> 0.612312`였다.
NEW recall은 `0.471723 -> 0.478727`로 증가했다. 하지만 최종 change union에는 NEW seed도
포함되므로 REMOVED-only 진단에서는 NEW 예측이 FP가 되어 IoU가 작게 내려갔다.

## 5. 해석

1. **가능성 확인:** reference topology를 전혀 바꾸지 않고 NEW bank만 densify할 수 있다.
2. **gradient 우려 확인:** joint branch만 학습시키면 seed gradient가 매우 작아질 수 있다.
3. **해결 확인:** seed-only coverage objective는 background GS의 기존 DC와 독립적으로
   seed에 gradient를 전달한다.
4. **효과:** XFeat-only E4a의 overall `ΔIoU +0.000246`보다 NEW-only densification의
   `+0.002686`이 약 10.9배 크다.
5. **남은 한계:** SC3 leakage와 3D overview의 ray/surface 방향 줄무늬는 nearest-parent
   depth 복사의 한계다. 다음 단계는 같은 cue component 안에서도 depth discontinuity를
   분리하고, 3D voxel/track consolidation 후 front surface만 남기는 것이다.

## 6. 구현과 artifact

- `temporal/new_seed_densification.py`
- `temporal/new_seed_gaussians.py`
- `experiments/run_online_xfeat_new_seed.py`

최종 결과:

```text
outputs/e4b_new_only_densify_scene123_120_20260814_013657/
```
