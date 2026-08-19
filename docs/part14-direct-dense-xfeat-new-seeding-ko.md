# Part 14. Densification 없이 NEW mask 안에 direct dense XFeat seed 뿌리기

> 작성일: 2026-08-14
> 비교: E4a sparse XFeat vs E4b parent-depth densification vs E4c direct dense XFeat
> 결론: **parent depth를 복사해 child를 확장하지 않고, NEW mask 내부 XFeat 검출 밀도를 512에서 4096로 올려 모든 seed를 실제 inference-view correspondence로 삼각측량했다. E4b의 ray/chain geometry artifact는 사라졌고 NEW IoU는 E4b보다 조금 높았지만, false signed cue까지 조밀하게 복원되어 overall IoU는 E4b보다 조금 낮았다.**

## 1. 변경한 가설

E4b는 XFeat parent가 드문 문제를 다음과 같이 풀었다.

```text
sparse XFeat parent
  -> 현재 NEW cue의 비어 있는 cell 선택
  -> 가까운 parent의 camera depth 복사
  -> causal signed-mask carving
  -> densified child
```

이 과정은 정확한 correspondence 없이 depth를 옆 cell로 복사한다. 실제 checkpoint audit에서
SC2는 child lineage 5세대, SC3는 7세대까지 만들어졌고, SC2 future-view NEW precision은
1세대 `0.827`, 3세대 `0.223`, 4세대 `0.099`로 무너졌다. 이는 recursive
parent-depth propagation의 품질 저하를 정량적으로 보여준다.

E4c는 이 경로를 완전히 제거했다.

```text
P(sign=NEW) >= 0.8
  -> 해당 signed NEW mask 내부에서 XFeat 4096 검출
  -> 최근 inference view 최대 8개와 mutual cosine matching
  -> known-pose epipolar / baseline / ray-angle / cheirality gate
  -> 3-view track DLT triangulation
  -> reprojection RMSE <= 3 px + 모든 support NEW mask 내부 확인
  -> direct NEW sidecar seed promotion
```

- `--new-only-densify`는 사용하지 않았다.
- copied/densified child는 0개다.
- 모든 seed xyz는 실제 inference-to-inference correspondence의 3-view 이상 triangulation 결과다.
- reference depth, GT, future frame은 seed birth에 사용하지 않았다.
- reference `R_ref` 1,283,501 rows의 모든 tensor는 bitwise 고정이다.
- seed geometry도 이번 대조에서는 고정하고 seed DC만 seed-only projected coverage loss로 학습했다.

## 2. Seed 위치 결과

| Method | SC1 | SC2 | SC3 | copied child |
|---|---:|---:|---:|---:|
| E4a sparse XFeat-512 | 8 | 201 | 220 | 0 |
| E4b parent-depth densify | 50 | 353 | 659 | 633 |
| **E4c direct XFeat-4096** | **75** | **2034** | **2325** | **0** |

Promotion 이후 화면 안에 들어온 seed 중심의 evaluation-only NEW precision:

| Method | SC1 | SC2 | SC3 | REMOVE leakage SC3 |
|---|---:|---:|---:|---:|
| E4a sparse XFeat | 1.000 | 0.908 | 0.902 | 0.006 |
| E4b parent-depth densify | 1.000 | 0.823 | 0.796 | 0.018 |
| **E4c direct XFeat-4096** | **0.632** | **0.877** | **0.871** | **0.004** |

SC2/3에서는 E4b보다 위치 precision과 REMOVE leakage가 모두 좋아졌다. 3D overview에서도
E4b child의 ray/chain pattern이 없어지고 object surface 주변 cluster로 바뀌었다.
SC1 precision이 낮아진 것은 gate가 frame 79에 늦게 열려 promotion 이후 future-view 분모가
68개뿐인 작은 표본이고, 4096 검출에서 object boundary/background 반복 texture가 추가로
들어왔기 때문이다.

파랑은 direct triangulated XFeat seed이고, 주황은 evaluation-only NEW GT 밖에 투영된 중심이다.
E4c에는 자홍색 densified child가 없다.

## 3. 304-frame mask metric

| Method | Overall ΔFull IoU | ΔFull F1 | ΔNEW IoU | ΔREMOVE IoU |
|---|---:|---:|---:|---:|
| E4a sparse XFeat | +0.000246 | +0.000238 | +0.000387 | -0.000146 |
| E4b parent-depth densify | **+0.002686** | **+0.002591** | +0.003447 | -0.001296 |
| **E4c direct XFeat-4096** | +0.002519 | +0.002430 | **+0.003719** | -0.001422 |

E4c의 overall IoU는 `0.438561 -> 0.441080`, NEW IoU는 `0.240943 -> 0.244662`였다.

해석은 다음과 같다.

1. **NEW 표현:** direct dense triangulation이 E4b보다 NEW IoU가 `+0.000272` 더 높다.
2. **geometry:** parent-depth 복제 artifact 없이 실제 correspondence 기반 cluster를 얻었다.
3. **overall:** E4b보다 overall ΔIoU가 `0.000167` 낮다. sparse footprint와 false NEW cue가 함께
   증가했기 때문에 seed 수 증가가 곧바로 union metric 증가로 이어지지는 않는다.
4. **다음 bottleneck:** 이제 seed 부족보다 signed NEW cue의 boundary/background contamination과
   중복 3D seed consolidation이 더 큰 문제다. geometry optimization보다 먼저 voxel merge,
   track confidence, future causal reprojection survival로 direct seed를 정리하는 편이 타당하다.

## 4. Artifact와 재현

```text
outputs/e4c_direct_dense_xfeat4096_scene123_120_20260814_023418/
outputs/e4c_xfeat4096_features.pt
```

핵심 실행 차이는 다음 두 옵션이다.

```text
--xfeat-top-k 4096
--seed-coverage-loss-weight 1.0
```

`--new-only-densify`는 주지 않았다. 전체 회귀 테스트는 ROS pytest plugin auto-load를 끈
환경에서 `169 passed`였다.
