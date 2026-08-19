# Instance_1 독립 `Ref -> SCn` DC-only 대 DC+geometry 비교

## 1. 질문

기존 temporal geometry 실험은 하나의 304-frame sequence 안에 S0/S1/S2 slot을
두고 학습했다. Slot 자체는 state별이었지만 S1/S2의 xyz에는 S0 soft anchor가
들어갔다. 이번 실험은 그 temporal coupling을 완전히 제거하고 다음 여섯 조건을
실제로 별도 모델로 학습했다.

```text
Ref -> scene_change1: DC-only / DC+geometry
Ref -> scene_change2: DC-only / DC+geometry
Ref -> scene_change3: DC-only / DC+geometry
```

질문은 다음과 같다.

> 다른 state의 학습이나 anchor가 전혀 없는 독립 one-step change에서도
> geometry optimization이 DC-only보다 불리한가?

## 2. 통제 조건

각 scene/condition마다 동일한 reference PLY에서 새 모델과 새 Adam optimizer를
만들었다.

| 항목 | 설정 |
|---|---|
| scene frame 수 | SC1 95, SC2 104, SC3 105 |
| 모델 state 수 | 각 run마다 1 |
| cross-scene parameter 공유 | 없음 |
| 이전 state inheritance | 없음 |
| S0 anchor | 없음 |
| topology | 고정, 1,283,501 GS |
| densification/pruning | 없음 |
| pose | 동일 O-SCD fixed canonical pose |
| training supervision | O-SCD pixel cue + SAM2.1 feature cue |
| GT | 학습 후 평가에서만 로드 |
| updates | 이미지당 정확히 120회 |
| condition당 총 updates | 36,480 |
| 두 condition 총 updates | 72,960 |
| seed | 0 |
| binary 평가 | rendered score `> 0.5` |

학습률은 기존 geometry lifespan 실험과 동일하다.

| Parameter | LR |
|---|---:|
| DC | 0.0025 |
| xyz delta | 0.00016 |
| opacity delta | 0.025 |
| scaling delta | 0.005 |
| rotation delta | 0.001 |

입력 checksum도 기존 비교와 동일하다.

```text
base PLY:
ed35bb594b5e4dd2b72624034ab1ee4d2cbfd68d3bf588a96de01816a2df6059

fixed cameras:
3f57d31573e1913101489ebbd08494d88735dd671f7fec61b74a40b66ee25deb

cue metadata:
0d399396d1627f1d6c7699b3d4804913b22e60bfac5cc79245ef76864bc88c90
```

DC-only에서는 reference geometry를 그대로 유지하고 `state_change_dc`만
학습했다. DC+geometry에서는 동일한 고정 index에 대해 DC와
`xyz/opacity/scale/rotation` delta를 함께 학습했다. 따라서 이번 실험도
**새 Gaussian birth를 평가한 것이 아니라 reference scaffold deformation을
평가한 것**이다.

## 3. 결과

### 3.1 Scene별 mean-frame mIoU/F1

| Scene | DC-only mIoU | DC+geometry mIoU | 변화 | DC-only F1 | DC+geometry F1 | 변화 |
|---|---:|---:|---:|---:|---:|---:|
| SC1 | 0.6137 | 0.5830 | -0.0308 | 0.7071 | 0.6761 | -0.0310 |
| SC2 | 0.6464 | 0.6373 | -0.0091 | 0.7761 | 0.7651 | -0.0111 |
| SC3 | 0.6121 | 0.6264 | **+0.0143** | 0.7513 | 0.7573 | **+0.0060** |
| 전체 304-frame 평균 | **0.6243** | 0.6166 | -0.0078 | **0.7460** | 0.7346 | -0.0114 |

SC3에서는 geometry가 실제로 이득이었다. 따라서 이 결과는
`geometry optimization은 항상 나쁘다`를 지지하지 않는다. 다만 SC1/SC2의
하락이 더 커서 전체 평균에서는 DC-only가 우세했다.

### 3.2 Precision/recall과 FP/FN

| Scope | Precision 변화 | Recall 변화 | FP 변화 | FN 변화 |
|---|---:|---:|---:|---:|
| SC1 | -0.0314 | +0.0097 | +114,481 | -21,022 |
| SC2 | -0.0169 | +0.0083 | +128,112 | -26,229 |
| SC3 | -0.0259 | +0.0659 | +244,398 | -210,785 |
| 전체 | -0.0234 | +0.0303 | +486,991 | -258,036 |

Geometry는 세 scene 모두에서 recall을 높이고 FN을 줄였다. 동시에 precision을
낮추고 FP를 늘렸다. SC3에서는 큰 FN 감소가 FP 비용을 상쇄했지만 SC1/SC2에서는
그러지 못했다.

### 3.3 Cue-fitting loss

| Scene | DC-only post SSF | DC+geometry post SSF | Geometry - DC |
|---|---:|---:|---:|
| SC1 | 0.364631 | 0.364201 | -0.000429 |
| SC2 | 0.400089 | 0.399117 | -0.000972 |
| SC3 | 0.384583 | 0.383257 | -0.001326 |
| frame-weighted 전체 | 0.383652 | 0.382728 | -0.000925 |

모든 scene에서 geometry가 O-SCD cue training loss를 더 낮췄다. 그러나 SC1/SC2의
GT mIoU/F1은 하락했다. 즉 이전 temporal 실험에서 관찰한
`SSF loss 감소 != binary GT 성능 증가` 패턴은 temporal coupling을 제거해도
남았다.

## 4. 기존 temporal 실험과의 관계

독립 DC-only의 304개 binary prediction은 기존 3-slot temporal DC-only 결과와
pixel 단위로 완전히 같았다.

```text
SC1 different pixels: 0 / 47,200,560
SC2 different pixels: 0 / 51,672,192
SC3 different pixels: 0 / 52,169,040
```

따라서 기존 DC-only 결과는 sequence 안에서 실행됐지만 각 state slot의 최종
prediction 관점에서는 이미 독립 `Ref -> SCn` 학습과 같았다.

반면 기존 geometry 조건의 S1/S2에는 S0 xyz anchor가 있었다. 이번 실험은 이를
제거했는데도 전체 geometry 성능은 DC-only보다 낮았다. 따라서 geometry 하락을
S0 anchor나 continual forgetting만으로 설명할 수 없다. SC1은 원래부터 anchor가
적용되지 않는 state인데도 geometry가 하락했다.

이번 결과가 분리한 두 현상은 다음과 같다.

1. **Shared mutable geometry forgetting:** 같은 geometry를 다음 state가 덮어써서
   과거 mask가 무너지는 continual-memory 문제.
2. **Independent geometry over-coverage:** 다른 state와 공유하지 않아도 2D cue
   objective 아래에서 geometry가 recall을 위해 support를 넓히고 FP를 만드는
   optimization/generalization 문제.

이번 실험은 2번을 측정한다. 1번은 발생할 수 없는 설정이다.

## 5. 해석

현재 근거에 맞는 결론은 다음과 같다.

> Geometry optimization 자체는 금지할 대상이 아니다. 그러나 모든 change에
> 대해 reference scaffold의 geometry freedom을 일괄적으로 여는 것은 scene에
> 따라 이득과 손해가 갈리며, 현재 SSF cue objective에서는 평균적으로 FP가 더
> 증가했다.

SC3의 개선은 geometry가 필요한 변화가 실제로 존재할 수 있음을 보여준다.
동시에 SC1/SC2의 하락은 geometry를 **변화 유형과 무관하게 전역적으로 허용하면
안정적인 기본값이 되기 어렵다**는 것을 보여준다.

이 결과는 다음 설계를 지지한다.

```text
immutable reference geometry
  + existing-surface change: DC/lifespan 중심
  + confirmed structural novelty: 별도 new geometry birth 및 제한적 refinement
```

특히 이번 DC+geometry는 fixed topology라서 new object를 새 surface로 생성하지
못하고 기존 reference GS를 이동·확대해서 설명해야 했다. 따라서 confirmed NEW
bank를 별도로 생성하는 후속 실험과 모순되지 않는다.

## 7. 재현

```bash
PYTHONPATH=. conda run --no-capture-output -n oscd python \
  experiments/run_independent_ref_geometry_ablation.py
```

Runner는 각 scene/condition 완료 후 즉시 checkpoint와 summary를 저장한다. 동일한
설정의 완료 run이 있으면 이를 검증하고 재사용한다. 다시 학습하려면
`--overwrite`를 사용한다.

## 8. 산출물

```text
outputs/instance1_independent_ref_scene_dc_geometry_oscd_cues_allframes_120/
├── summary.json
├── condition_metrics.csv
├── overall_condition_metrics.csv
├── condition_deltas.csv
├── scene_change1/{dc_only,dc_geometry}/
├── scene_change2/{dc_only,dc_geometry}/
└── scene_change3/{dc_only,dc_geometry}/
```

각 condition 디렉터리에는 다음이 들어 있다.

- one-state checkpoint
- run summary와 schedule/gradient audit
- frame별 metric CSV
- continuous score PNG
- binary prediction PNG
- TP/FP/FN confusion PNG

모든 6개 run에서 exact per-frame schedule과 audited gradient isolation을
통과했다.
