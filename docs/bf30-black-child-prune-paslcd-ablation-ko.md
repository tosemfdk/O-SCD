# BF30 black child-prune PASLCD single-state ablation

## 1. 목적

Continuous ESCD에서 cue-mixture split 뒤 검정으로 남는 densified child 일부를
hard-prune하면 topology 오염을 줄일 수 있었다. 그러나 evolving scene 전용 개선은
단일 post-change state를 유지하는 PASLCD에서도 mask 품질을 보존해야 한다.

이번 실험은 PASLCD 20개 scene, 총 500 frame에서 다음 네 density 정책을 비교한다.

1. Density off
2. 기존 O-SCD식 ACTIVE gradient clone/split
3. Gradient clone/split + raw cue-mixture split
4. Cue-mixture split + child-only black hard pruning

## 2. Matched 조건

- PASLCD `Instance_1/2 × 10 scenes`
- Scene당 25개 inference frame, 총 500 frame
- Seed 0, resolution 4, 16 updates/frame
- 저장된 O-SCD fixed camera와 raw pixel+SAM2.1 cue cache 재사용
- Raw single-candidate Beta detector, BF30
- Frozen black NEVER_OPEN occluder
- 모든 OPEN row optimizer 선택
- Local support + previous-growth replay, weight 7.5
- Cue-mixture threshold 0.5
- Opacity/size pruning 비활성화
- GT mask는 25-frame causal loop가 끝난 뒤에만 평가에 사용

PASLCD scene은 하나의 static post-change state만 포함한다. 따라서 같은 scene 안의
CLOSE/REOPEN은 실제 evolution이 아니라 detector/topology 불안정성이다.

산출물:

```text
/tmp/escd_bf30_black_paslcd_density_ablation_20260901_v1/
    density_off/
    gradient_only/
    cue_mixture/
    cue_mixture_black_child_prune/
    scene_metrics.csv
    comparison.json
    comparison.md
```

## 3. Child-only hard pruning 규칙

Hard-prune 후보는 다음 조건을 모두 만족해야 한다.

```text
OPEN
AND generation > 0
AND 생성 후 최소 1 frame 경과
AND intrinsic learned DC < 0.5
```

초기 Gaussian과 같은 density event에서 방금 생성된 child는 보호한다. 직접 부모가
이미 제거됐고 살아남을 sibling도 없다면 후보 중 가장 덜 검정인 child 하나를 남긴다.
따라서 detector support 계보 전체를 hard-prune 한 번으로 삭제하지 않는다.

## 4. 결과

### 4.1 Mask 품질

| 조건 | mIoU | F1 | precision | recall | O-SCD online 대비 mIoU/F1 | scene 승리 |
|---|---:|---:|---:|---:|---:|---:|
| Density off | **0.4566** | **0.6125** | **0.5538** | 0.7532 | -0.0321/-0.0299 | 8/20 |
| Gradient-only | 0.4561 | 0.6120 | 0.5531 | **0.7537** | -0.0326/-0.0303 | 8/20 |
| Cue-mixture | 0.4515 | 0.6077 | 0.5492 | 0.7468 | -0.0372/-0.0346 | 6/20 |
| Mixture + child prune | 0.4523 | 0.6084 | 0.5505 | 0.7452 | -0.0365/-0.0339 | 7/20 |

저장된 PASLCD 기준선은 다음과 같다.

```text
O-SCD online:  mIoU/F1 = 0.4887/0.6423
O-SCD refined: mIoU/F1 = 0.5573/0.7009
```

Child pruning은 cue-mixture보다 mIoU/F1을 `+0.0008/+0.0007`만 회복했다.
Density-off보다는 `-0.0044/-0.0041`, O-SCD online보다는
`-0.0365/-0.0339` 낮았다.

이 `+0.0008`은 유의한 개선으로 해석하지 않는다. `Instance_1/Cantina`를 현재 코드로
두 번 더 반복했을 때 child-prune mIoU가 `0.48658--0.48715` 범위로 변했고 split 수도
달라졌다. CUDA rasterization과 gradient-threshold density 경로가 bitwise
deterministic하지 않아, 관측된 평균 회복량이 단일-scene 반복 변동과 같은 규모다.

### 4.2 Density와 black child

| 진단 | Density off | Gradient-only | Cue-mixture | Mixture + child prune |
|---|---:|---:|---:|---:|
| split source | 0 | 120 | 112,214 | 104,990 |
| mixture-only split source | 0 | 0 | 112,078 | 104,847 |
| black child hard-pruned | 0 | 0 | 0 | 62,678 |
| final Gaussian 합 | 3,616,231 | 3,616,360 | 3,728,453 | 3,658,566 |
| final ACTIVE 합 | 476,636 | 476,778 | 587,935 | 518,495 |
| scene별 final black ACTIVE 비율 평균 | 0.5216 | 0.5214 | 0.5495 | **0.4936** |

Child pruning은 의도한 topology cleanup은 수행했다.

- 62,678개 hard deletion은 모두 densify의 split child였다.
- 생성 기록이 없는 row deletion: 0
- 생성 후 1 frame 미만 deletion: 0
- Cue-mixture 대비 final Gaussian 합: `-69,887`
- Cue-mixture 대비 final ACTIVE 합: `-69,440`
- Final black ACTIVE 비율: `0.5495 -> 0.4936`

후보/guard 누적 횟수 `825,276/762,598`은 unique row 수가 아니다. Guard로 남은 검정
support child가 이후 frame에서 다시 후보가 된 횟수를 포함한다.

### 4.3 Lifecycle

| 조건 | OPEN | CLOSE | REOPEN | same-scene repeated |
|---|---:|---:|---:|---:|
| Density off | 478,601 | 1,965 | 245 | 2,210 |
| Gradient-only | 478,620 | 1,971 | 265 | 2,231 |
| Cue-mixture | 478,392 | 2,679 | 325 | 1,934 |
| Mixture + child prune | 478,666 | 2,506 | 345 | 1,960 |

Child pruning은 cue-mixture 대비 CLOSE를 조금 줄였지만 REOPEN과 repeated transition을
줄이지 못했다. Dynamic child가 부모의 OPEN/Beta state를 상속하고 topology deletion이
다음 frame alpha-T 책임도를 재분배하므로, deletion 자체가 lifecycle 안정화로 직접
이어지지는 않는다.

## 5. 해석

이번 결과는 두 층을 구분해서 해석해야 한다.

### 5.1 Child pruning은 기계적으로 동작했다

검정 densified child 일부를 실제로 제거했고, 원본 row 및 방금 생성된 child 보호
invariant도 지켰다. Cue-mixture가 늘린 black ACTIVE 비율과 topology 크기를 크게
되돌렸다.

### 5.2 그러나 mixture split의 mask 손해를 회복하지 못했다

높은 raw cue mixture는 semantic하게 분리해야 할 change/non-change 물체 경계만
가리키지 않는다. 일반 depth/occlusion boundary와 cue noise에서도 높다. 현재 split은
positive/negative cue centroid 방향으로 child를 배치하지 않고 두 child에 같은 DC와
OPEN/Beta state를 복사한다. 이후 검정 child를 일부 제거해도 최초의 잘못된 자유도와
alpha-T 재분배를 되돌리지는 못한다.

### 5.3 더 큰 PASLCD blocker는 density 이전에 있다

Density-off도 O-SCD online보다 mIoU/F1이 `-0.0321/-0.0299` 낮다. 따라서 현재
BF30 + frozen-black NEVER_OPEN + local-growth-replay pipeline의 PASLCD 격차는
densification이나 black-child cleanup만으로 설명되지 않는다. Static-scene mask 품질을
회복하려면 detector OPEN coverage, black occlusion composition, replay/regularization을
O-SCD online과 다시 분해 비교해야 한다.

## 6. 결론

```text
PASLCD mask 순위:
density off > gradient-only > mixture + child prune > cue-mixture
```

Child-only hard pruning은 topology cleanup control로는 유효하지만, mask 개선은
명확하게 확인되지 않았다. Point estimate는 `+0.0008 mIoU`지만 반복 변동과 같은
규모다. PASLCD에서 density-off와 O-SCD online을 모두 넘지 못했으므로 cue-mixture
split과 child pruning은 현재 기본 방법으로 채택하지 않는다.

## 7. 검증

- 전체 run: 80개, 총 2,000 processed frame
- Source/camera/cue 경로 scene별 일치
- Future-view access: 0
- GT causal-loop access: 0
- Inactive/wrong gradient violation: 0
- CLOSED persistence audit: 전 run pass
- Topology integrity: 전 run pass
- Black deletion 62,678개 전부 prior CREATE record 존재
- Same-event 또는 generation-zero black hard deletion: 0
- 중간에 추가된 비활성 `candidate_render_gate` 누락값은 `none`으로 정규화했으며 나머지
  scene별 run argument와 PLY/camera hash는 일치
- CUDA 비결정성 확인: Cantina child-prune 반복 mIoU `0.48658--0.48715`
