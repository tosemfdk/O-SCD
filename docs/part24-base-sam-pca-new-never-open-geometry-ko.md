# Part 24: SAM/PCA signed NEW base geometry ablation

## 1. 질문

Part 23의 base-only 실험에서 DA3 seed는 계속 사용하지 않되, SAM/PCA가 분리한
ADD/NEW 방향을 `OPEN + NEVER_OPEN` geometry 학습 target으로 쓰면 붕괴가 줄어드는지
확인한다.

핵심 분리는 다음과 같다.

- BF30 detector evidence: 전체 learned soft `Q`
- base DC target: 전체 learned soft `Q`
- base geometry target: SAM/PCA의 signed NEW support 안에 남은 learned soft `Q`
- 최종 mask와 평가 GT: 여전히 `ADD union REMOVE`

즉 REMOVE를 최종 출력에서 버린 실험이 아니다. Reference에 이미 geometry가 있는
REMOVE는 DC가 담당하고, geometry 이동이 필요한 후보 영역만 NEW로 제한하는
실험이다.

## 2. 실험 계약

- continuous `ref -> SC1 -> SC2 -> SC3`, 304 frames
- causal prequential learned sigmoid `Q`
- soft-Q alpha-transmittance BF30 detector
- detector geometry: immutable reference
- DA3 seed와 NEW sidecar: 없음
- topology growth, densification, pruning: 없음
- geometry scope: `OPEN union NEVER_OPEN`
- geometry parameter: xyz/scale/rotation
- image당 16 causal replay updates
  - latest branch probability `0.33`
  - 나머지는 관측된 `[0,t]` view에서 uniform sampling
- signed SAM/PCA:
  - causal PC1 trace
  - 과거 depth 기반 sign 판정과 동일한 global lock `+ = NEW`
  - scene마다 NEW 부호를 다시 선택하지 않음
  - learned `Q >= 0.8` support 안에서 signed magnitude 상위 40% 유지
- 최종 mask: learned raw prediction `>= 0.5`
- GT: evaluation-only `ADD union REMOVE`

## 3. 결과

| 조건 | mIoU | Mean F1 | Precision | Recall | Aggregate IoU |
|---|---:|---:|---:|---:|---:|
| geometry frozen | **0.3977** | **0.5207** | 0.8936 | **0.4411** | **0.4191** |
| OPEN+NEVER, 전체 Q geometry target | 0.0975 | 0.1286 | 0.8538 | 0.1715 | 0.1667 |
| OPEN+NEVER, signed NEW geometry target | 0.1883 | 0.2759 | **0.8744** | 0.2394 | 0.2315 |

변화량:

- 전체 Q target -> signed NEW target: mIoU `+0.0908`
- geometry frozen -> signed NEW geometry: mIoU `-0.2094`

Scene별 mean-frame IoU:

| 조건 | SC1 | SC2 | SC3 |
|---|---:|---:|---:|
| geometry frozen | **0.4352** | **0.4582** | **0.3038** |
| OPEN+NEVER, 전체 Q | 0.1686 | 0.1236 | 0.0073 |
| OPEN+NEVER, signed NEW | 0.2550 | 0.2274 | 0.0892 |

## 4. 진단

### 4.1 signed NEW는 전체-Q 붕괴를 완화한다

- 완전히 빈 prediction frame: 전체 Q `159` -> signed NEW `17`
- 마지막 frame predicted-positive pixel: 전체 Q `0` -> signed NEW `1,643`
- signed NEW support는 304/304 frame에서 non-empty
- frame당 signed NEW target pixel: 평균 `5,011.7`, 최소 `120`, 최대 `17,509`

전체 change 영역을 geometry target으로 쓰는 것보다 NEW 방향만 쓰는 것이 reference
geometry를 무차별적으로 이동·축소하는 압력을 줄였다.

### 4.2 그러나 frozen 기준에는 크게 못 미친다

signed target으로 좁혀도 최종 NEVER_OPEN 1,207,026 row 중 다음 수가 변경됐다.

- xyz: 744,778
- scale: 744,763
- rotation: 758,437

scale raw delta max는 `1.386295`, 즉 `log(4)` cap에 도달했다. 특히 SC3 recall은
`0.0645`, mean-frame IoU는 `0.0892`에 그쳤다. 한 프레임의 2D signed NEW mask만으로
현재 view에 보이는 대규모 `NEVER_OPEN` row를 모두 갱신하므로, NEW pixel과 실제
alpha-T responsibility가 약한 row도 geometry gradient를 받는 문제가 남는다.

### 4.3 detector 차이는 아니다

전체-Q geometry 조건과 signed-NEW geometry 조건의 frame별
OPEN/NEVER_OPEN/CLOSED count는 완전히 동일했다. 따라서 `+0.0908`과 남은 성능
하락은 모두 representation geometry target 차이에서 발생했다.

## 5. 결론

SAM/PCA ADD/NEW 분리는 **방향은 맞지만 단독 해법은 아니다**.

- 채택 가능한 관찰: REMOVE까지 포함한 전체 Q보다 signed NEW만 geometry target으로
  쓰는 것이 훨씬 덜 파괴적이다.
- 채택하지 않는 구성: 모든 visible `OPEN + NEVER_OPEN` row의 xyz/scale/rotation을
  signed NEW 2D coverage만으로 학습하는 방식.
- 다음 최소 실험: signed NEW pixel에 실제 alpha-T responsibility가 있고, 서로 다른
  causal view에서 반복 support된 row만 geometry optimizer에 넣는다.

## 6. NEVER_OPEN opacity 추가 학습

직전 signed NEW geometry 조건을 그대로 유지하고 NEVER_OPEN의 opacity만 추가로
학습했다. DC는 계속 exact black으로 고정했으며, detector는 immutable reference를
사용하므로 lifecycle은 opacity 학습 전과 frame별로 완전히 동일하다. Raw opacity는
원본 O-SCD와 같이 별도 cap 없이 sigmoid activation만 적용했다.

| 조건 | mIoU | Mean F1 | Precision | Recall | Aggregate IoU |
|---|---:|---:|---:|---:|---:|
| signed NEW geometry, opacity 고정 | **0.188301** | 0.275937 | **0.874357** | **0.239423** | **0.231460** |
| signed NEW geometry, NEVER_OPEN opacity 학습 | 0.187715 | **0.276589** | 0.868835 | 0.237517 | 0.229295 |

차이는 mIoU `-0.000585`, F1 `+0.000651`로 사실상 중립이다. Scene별 mIoU는
SC1 `0.2550 -> 0.2497`, SC2 `0.2274 -> 0.2225`, SC3
`0.0892 -> 0.0971`이었다. 마지막 frame predicted-positive pixel은
`1,643 -> 2,975`로 늘었지만 전체 precision과 recall은 모두 소폭 감소했다.

따라서 이 구성의 병목은 고정 opacity 하나가 아니다. NEVER_OPEN DC가 black인 채
white coverage geometry loss로 opacity를 학습하면 opacity는 실제 change appearance를
직접 표현하지 못하고, occlusion/coverage만 재배치한다. 원본 O-SCD의 이점은 opacity
단독이 아니라 DC, opacity, geometry를 실제 change-render loss로 공동 최적화하는 데
있다는 해석과 일치한다.

초기 구현에서 opacity를 `[0.01,0.99]`로 cap한 run은 reference opacity까지 강제로
변경하는 confound가 있어 폐기했으며 본 표에 포함하지 않는다.

## 7. 산출물

```text
outputs/base_only_learned_q_sam_pca_new_open_never_geometry_u16_20260904/
  summary.json
  frame_metrics.csv

outputs/base_only_learned_q_sam_pca_new_open_never_opacity_uncapped_u16_20260904/
  summary.json
  frame_metrics.csv
```

재현 runner:

```text
experiments/evaluate_base_never_open_geometry.py
```

주요 option:

```text
--base-geometry-scope open_and_never_open
--base-geometry-target signed_new
--sam-new-sign +
--train-never-open-opacity
```
