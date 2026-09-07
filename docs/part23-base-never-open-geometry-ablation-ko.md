# Part 23: DA3/SAM-sign 없는 base NEVER_OPEN geometry ablation

## 1. 질문

DA3 depth seed와 signed SAM/PCA의 ADD/NEW 분기를 모두 제거하고, 기존
`R_change` Gaussian만으로 학습할 때 `NEVER_OPEN` row의 xyz/scale/rotation까지
학습하면 change-mask 성능이 좋아지는지 확인한다.

기본 learned cue의 SAM2.1 semantic scalar는 `Q`의 구성 요소이므로 유지한다. 제거한
것은 별도 NEW sidecar를 만드는 DA3 birth와 signed SAM/PCA NEW 방향 판정이다.

## 2. 실험 계약

- continuous `ref -> SC1 -> SC2 -> SC3`, 304 frames
- causal prequential learned sigmoid `Q`
- soft-Q alpha-transmittance BF30 detector
- detector geometry: immutable reference
- representation: 기존 1,283,501-row base `R_change`만 사용
- DA3 seed: 없음
- signed SAM ADD/NEW trace/model: 없음
- topology growth, densification, pruning: 없음
- image당 16 causal replay updates
  - latest branch probability `0.33`
  - 그 외에는 관측된 `[0,t]` view에서 uniform sampling
- 최종 mask: learned raw prediction `>= 0.5`
- GT: evaluation-only `ADD union REMOVE`

Geometry branch는 선택된 row를 white coverage로 렌더하여 learned `Q`에 맞추고,
DC branch와 분리해 학습한다. NEVER_OPEN의 DC와 opacity는 계속 고정한다.

```text
frozen:
    OPEN DC만 학습

open:
    OPEN DC + OPEN xyz/scale/rotation 학습

open_and_never_open:
    OPEN DC
    + (OPEN union NEVER_OPEN) xyz/scale/rotation 학습
```

Geometry에는 birth/reference anchor 기준 trust region을 적용했다.

- xyz displacement: reference Gaussian scale의 최대 4배
- scale: reference 대비 `[0.25, 4.0]`
- rotation: quaternion normalization

## 3. 결과

| Geometry scope | mIoU | Mean F1 | Precision | Recall | Aggregate IoU |
|---|---:|---:|---:|---:|---:|
| frozen | **0.3977** | **0.5207** | **0.8936** | **0.4411** | **0.4191** |
| OPEN | 0.2158 | 0.2965 | 0.8295 | 0.2971 | 0.2800 |
| OPEN + NEVER_OPEN | 0.0975 | 0.1286 | 0.8538 | 0.1715 | 0.1667 |

변화량은 다음과 같다.

- frozen -> OPEN geometry: mIoU `-0.1818`
- OPEN -> OPEN+NEVER_OPEN geometry: mIoU `-0.1183`
- frozen -> OPEN+NEVER_OPEN geometry: mIoU `-0.3002`

Scene별 mean-frame IoU:

| Geometry scope | SC1 | SC2 | SC3 |
|---|---:|---:|---:|
| frozen | **0.4352** | **0.4582** | **0.3038** |
| OPEN | 0.3751 | 0.2072 | 0.0803 |
| OPEN + NEVER_OPEN | 0.1686 | 0.1236 | 0.0073 |

## 4. 진단

세 조건의 frame별 OPEN/NEVER_OPEN/CLOSED count는 완전히 동일했다. Detector는
immutable reference geometry와 동일한 Q를 사용했으므로 성능 차이는 lifecycle 변화가
아니라 representation geometry에서 발생했다.

OPEN+NEVER_OPEN 조건에서는 다음 현상이 나타났다.

- prediction이 완전히 빈 frame: frozen `19` -> geometry `159`
- IoU가 정확히 0인 frame: `165`
- 마지막 frame predicted-positive pixel: frozen `4,251` -> geometry `0`
- final NEVER_OPEN 1,207,026 rows 중 xyz/scale/rotation 변경 row:
  `735,733 / 735,699 / 758,437`
- scale raw delta max: `1.386295`, 즉 설정한 `log(4)` cap에 도달

즉 optimizer가 실패한 detector를 고친 것이 아니라, sparse 2D Q coverage를 맞추기
위해 대규모 reference-derived geometry를 이동·축소·확장하면서 OPEN change support까지
사라지게 했다. OPEN geometry만 허용해도 mIoU가 크게 하락했고, NEVER_OPEN까지 풀면
recall이 `0.2971 -> 0.1715`로 더 감소했다.

## 5. 결론

이 방식은 채택하지 않는다. 기존 base Gaussian 전체에 2D mask coverage만으로
xyz/scale/rotation 자유도를 주는 것은 depth 또는 multi-view surface constraint가 아니며,
특히 120만 개가 넘는 NEVER_OPEN row를 동시에 움직이면 표현이 붕괴한다.

NEVER_OPEN geometry를 다시 검증하려면 모든 NEVER_OPEN을 푸는 대신 다음처럼 좁혀야 한다.

1. 현재 cue에 실제 alpha-T responsibility가 있는 candidate row만 선택
2. 서로 다른 causal view 2--3개에서 support된 row만 geometry update
3. depth/reprojection residual을 직접 loss에 추가
4. xyz-only부터 시작한 뒤 scale/rotation을 한 축씩 추가

## 6. 산출물

```text
outputs/base_only_learned_q_geometry_frozen_u16_20260904/
outputs/base_only_learned_q_open_geometry_u16_20260904/
outputs/base_only_learned_q_open_never_geometry_u16_20260904/
```

각 directory에 `summary.json`, `frame_metrics.csv`가 있다. 재현 runner는
`experiments/evaluate_base_never_open_geometry.py`다.

후속 signed SAM/PCA NEW target 실험은
[`part24-base-sam-pca-new-never-open-geometry-ko.md`](part24-base-sam-pca-new-never-open-geometry-ko.md)에 기록한다.
