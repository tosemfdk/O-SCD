# BF30 black NEVER_OPEN density / cue-mixture / child-prune ablation

## 1. 목적

OPEN Gaussian이 change와 non-change cue 영역을 동시에 크게 덮으면 하나의 learned
DC가 두 신호 사이에서 타협하여 검게 내려갈 수 있다. 기존 O-SCD density는 화면공간
xyz gradient만 사용하므로, 상반된 gradient가 상쇄된 큰 Gaussian을 놓칠 수 있다.

이번 실험은 다음 네 정책을 matched 조건에서 비교한다.

1. densify 완전 비활성화
2. 기존 O-SCD식 gradient clone/split
3. 기존 gradient clone/split에 raw cue-mixture large-row split을 추가
4. Cue-mixture split 뒤, 이전에 생성된 검정 ACTIVE child 일부만 hard prune

목적은 현재 frame의 optimization 전 raw alpha-transmittance evidence에서 cue
mixture를 계산하고, mixture가 큰 ACTIVE Gaussian을 기존 gradient 조건과 무관하게
split했을 때 성능이 개선되는지, 그리고 기존 gradient densify 자체가 density-off보다
나은지 확인하는 것이다.

## 2. Cue-mixture score

Gaussian별 capped pseudo-count를 다음처럼 사용한다.

```text
delta_change     = capped change-cue responsibility
delta_nonchange  = capped non-change-cue responsibility
total            = delta_change + delta_nonchange
change_ratio     = delta_change / total

mixture_score
    = min(total, 1) × 2 × min(change_ratio, 1 - change_ratio)
```

Score는 0부터 1까지다.

- 한쪽 cue만 덮으면 0
- 양쪽을 75%/25%로 덮고 mass가 충분하면 0.5
- 양쪽을 50%/50%로 덮고 mass가 충분하면 1
- 비율이 반반이어도 total mass가 작으면 낮은 score

이번 threshold는 `0.5`다.

## 3. Density 정책

Cue-mixture 정책에서는 기존 gradient-only 정책을 유지하고 large-Gaussian split
후보만 추가했다.

```text
small ACTIVE Gaussian:
    기존 xyz-gradient 조건을 만족할 때만 clone

large ACTIVE Gaussian:
    기존 xyz-gradient 조건을 만족하거나
    cue mixture score가 0.5 이상이면 split
```

Mixture는 lifecycle detector에 사용하지 않는다. Detector는 계속 신규 frame의
pre-optimization raw cue evidence만 사용한다. Split child는 부모의 parameter,
lifespan, Beta/controller state를 한 번 복사하고 Adam state는 0에서 시작한다.

정확히 말하면 mixture는 **split에만** 사용된다.

- mixture 기반 small-row clone: 없음
- mixture 기반 pruning: 없음
- mixture 기반 OPEN/CLOSE: 없음
- 기존 gradient small-row clone: 그대로 유지
- 기존 gradient large-row split: 그대로 유지
- large ACTIVE row의 최종 split 조건: `gradient 조건 OR mixture 조건`

### 3.1 Child-only black hard pruning

네 번째 정책은 다음 조건을 모두 만족하는 row만 hard-prune 후보로 만든다.

```text
현재 OPEN
AND generation > 0
AND 생성 후 최소 1 frame 경과
AND intrinsic learned DC < 0.5
AND 같은 density event의 split/opacity/size 제거 대상이 아님
```

따라서 초기 Gaussian과 방금 생성된 child는 삭제하지 않는다. 또한 split source인
직접 부모가 이미 사라졌고 살아남을 sibling도 없다면, 후보 중 DC가 가장 높은 child
하나는 남긴다. 두 split child가 모두 검정일 때 계보 전체를 한 번에 없애 detector
support를 잃지 않기 위한 guard다.

이 pruning은 learned DC를 lifecycle detector에 넣는 것이 아니다. Detector는 계속
신규 frame의 optimization 전 raw cue evidence만 사용한다. Learned DC는 이미 생성된
추가 topology 중 일부를 제거하는 density 진단으로만 사용한다.

## 4. Matched 실험 조건

- Continuous `ref -> SC1 -> SC2 -> SC3`, 304 frames
- Seed 0, 16 updates/frame
- Raw `single_candidate_beta`, BF30
- Binary cue threshold 0.5, capped evidence
- Black frozen NEVER_OPEN occluder
- 모든 OPEN row optimizer 선택, NEVER_OPEN/CLOSED freeze
- Local support + previous-growth replay, weight 7.5
- Density update index 4
- Opacity/size pruning off
- 비교 간 차이: density policy 하나뿐

산출물:

```text
/tmp/escd_bf30_black_cue_mixture_density_ablation_20260901_v1/
    density_off/
    gradient_only/
    cue_mixture/
    cue_mixture_black_child_prune/
    comparison.json
    comparison.csv
    comparison.md
```

## 5. 결과

### 5.1 Mask

| Metric | Density off | Gradient-only | Cue-mixture | Mixture + child prune |
|---|---:|---:|---:|---:|
| mean-frame mIoU | **0.4903** | 0.4874 | 0.4839 | 0.4852 |
| mean-frame F1 | **0.6271** | 0.6248 | 0.6209 | 0.6224 |
| precision | **0.5824** | 0.5797 | 0.5727 | 0.5723 |
| recall | 0.8456 | 0.8442 | 0.8436 | **0.8484** |
| SC1 mIoU | 0.4891 | 0.4889 | 0.4880 | **0.4894** |
| SC2 mIoU | 0.5459 | 0.5464 | 0.5477 | **0.5478** |
| SC3 mIoU | **0.4363** | 0.4276 | 0.4171 | 0.4195 |

Gradient-only는 density-off보다 전체 mIoU/F1이 `-0.0029/-0.0023` 낮았다.
Cue-mixture는 density-off보다 `-0.0063/-0.0061`, gradient-only보다
`-0.0035/-0.0038` 낮았다. SC2만 densify가 최대 `+0.0017` 높았지만, SC3에서
gradient-only가 `-0.0087`, cue-mixture가 `-0.0191` 하락하면서 전체 손해가 됐다.

Child pruning은 cue-mixture 대비 mIoU/F1을 `+0.0013/+0.0014` 회복했고 SC3도
`+0.0024` 높였다. 그러나 density-off보다는 여전히 `-0.0051/-0.0047` 낮았다.

### 5.2 Topology와 lifecycle

| 진단 | Density off | Gradient-only | Cue-mixture | Mixture + child prune |
|---|---:|---:|---:|---:|
| clone child | 0 | 650 | 518 | 466 |
| split source | 0 | 1,144 | 24,272 | 16,685 |
| mixture-only split source | 0 | 0 | 23,478 | 15,933 |
| black child hard-pruned | 0 | 0 | 0 | 8,539 |
| final Gaussian | 1,283,501 | 1,285,295 | 1,308,291 | 1,292,113 |
| final ACTIVE | 131,391 | 133,815 | 157,912 | 148,184 |
| OPEN | 142,304 | 143,876 | 148,521 | 153,971 |
| CLOSE | 10,913 | 11,855 | 15,399 | 14,399 |
| REOPEN | 2,818 | 3,324 | 4,525 | 4,653 |
| same-scene repeated transition | 7,838 | 8,365 | 8,545 | 9,532 |

Gradient-only도 density-off보다 lifecycle event와 ACTIVE row를 조금 늘렸다.
Threshold 0.5 cue-mixture는 기존 split의 약 20배가 넘는 split을 만들었다. 이 중
23,478개는 gradient 조건 없이 mixture 조건만으로 발생했다. Split child가 부모의
OPEN lifecycle을 그대로 상속하므로 topology 증가가 ACTIVE row 증가로 거의 직접
이어졌다.

Child-prune run은 8,539개 child를 실제 삭제했고, 이후 topology/evidence가 달라지면서
split source 자체도 16,685개로 줄었다. 후보/guard 누적 횟수는
`828,562/820,023`이다. 이는 unique child 수가 아니라, 살아남은 검정 support child가
후속 frame에서 다시 후보와 guard로 집계된 횟수를 포함한다.

Mask는 일부 회복했지만 lifecycle은 좋아지지 않았다. OPEN과 same-scene repeated
transition이 mixture-only보다 각각 5,450개와 987개 증가했다. Hard pruning으로 alpha-T
책임도가 재분배되면서 detector가 새 row를 더 여는 feedback이 생긴 것으로 해석한다.

### 5.3 Black OPEN Gaussian

| 진단 | Density off | Gradient-only | Cue-mixture | Mixture + child prune |
|---|---:|---:|---:|---:|
| final DC below 0.5 count | 84,632 | 86,324 | 104,609 | 95,517 |
| final DC below 0.5 fraction | **0.6441** | 0.6451 | 0.6625 | 0.6446 |
| frame-mean below-0.5 fraction | 0.5040 | 0.5026 | 0.5138 | **0.4985** |

Mixture의 목표와 반대로 final black 방향 ACTIVE Gaussian 수와 비율이 모두
증가했다. Child pruning은 final black 비율을 `0.6625 -> 0.6446`으로 낮춰
density-off `0.6441` 수준까지 되돌렸다. 즉 topology cleanup 목표 자체는 달성했지만
mask와 lifecycle 전체 성능을 density-off 이상으로 만들지는 못했다.

### 5.4 실행 비용

| 진단 | Density off | Gradient-only | Cue-mixture | Mixture + child prune |
|---|---:|---:|---:|---:|
| runtime | **244.0 s** | 251.7 s | 261.8 s | 268.0 s |
| peak CUDA memory | **7.28 GB** | 7.91 GB | 8.02 GB | 7.95 GB |

Density-off가 가장 빠르고 peak memory도 가장 작았다.

## 6. 해석

Raw cue mixture는 하나의 Gaussian이 상반된 cue를 받는다는 사실은 찾지만, 그것만으로
그 Gaussian이 반드시 공간적으로 잘못 분해됐다는 뜻은 아니다. 정상적인 depth edge,
occlusion boundary, cue noise에서도 높은 mixture가 발생할 수 있다.

또한 현재 split은 source covariance에서 child 위치를 샘플링할 뿐 change 쪽 child와
non-change 쪽 child를 명시적으로 분리하지 않는다. 두 child는 같은 DC와 OPEN state를
상속한다. 따라서 자유도와 ACTIVE row는 크게 증가했지만 양쪽 cue로의 specialization은
보장되지 않았고 false-positive가 늘어 precision과 SC3가 하락했다.

## 7. 결론

`mixture_score >= 0.5`를 large-Gaussian의 독립적인 강제 split 조건으로 추가하는
정책은 기존 gradient-only density보다 낫지 않았다. 더 나아가 이 BF30 black
continuous seed-0 조건에서는 gradient-only도 density-off보다 낫지 않았다.

따라서 이 조건에서의 실험적 순위는 다음과 같다.

```text
density off > gradient-only > mixture + child prune > gradient + cue-mixture
```

현재 결과만으로 cue-mixture OR split은 채택하지 않는다. Gradient densify도 이
BF30 branch에서는 필수 구성으로 간주하지 않고 density-off를 우선 baseline으로 둔다.
단, 이번 비교는 continuous seed-0 한 번이므로 모든 데이터셋에서 densify가 해롭다는
일반 결론은 아니다.

Child-only hard pruning은 무제한 mixture split의 일부 손해와 black ACTIVE 비율을
회복했으므로 구현 가능한 cleanup control임은 확인했다. 하지만 density-off를 넘지
못했고 same-scene transition도 늘었으므로 현재 production 기본값으로 채택하지 않는다.

다음에 mixture를 다시 사용한다면 무제한 OR 조건이 아니라 다음처럼 제한해야 한다.

- 기존 split 수와 동일한 budget 안에서 mixture로 우선순위만 재정렬
- mixture와 큰 projected footprint뿐 아니라 black learned DC 또는 persistent
  multi-view mixture까지 함께 요구
- cue-positive/negative spatial centroid를 이용해 child를 서로 다른 방향으로 배치

## 8. 무결성

- 비교 인자 차이: density policy만 존재
- Source PLY/camera checksum 동일
- Future-view access: 0
- Inactive/wrong gradient violation: 0
- CLOSED persistence audit: pass
- Topology integrity: pass

## 9. PASLCD static-scene 확장

동일 u16 조건을 PASLCD 20 scenes/500 frames로 확장했다. Mean-frame mIoU/F1은
density-off `0.4566/0.6125`, gradient-only `0.4561/0.6120`, cue-mixture
`0.4515/0.6077`, mixture+child-prune `0.4523/0.6084`였다. Child pruning은
62,678개 split child를 제거하고 final black ACTIVE 비율을 `0.5495 -> 0.4936`으로
낮췄지만 mask point estimate 회복 `+0.0008/+0.0007`은 Cantina CUDA 반복 변동과
같은 규모여서 유의한 개선으로 보지 않는다. O-SCD online
`0.4887/0.6423`에도 미달했다.

상세 결과는
[`bf30-black-child-prune-paslcd-ablation-ko.md`](bf30-black-child-prune-paslcd-ablation-ko.md)에
기록한다.
