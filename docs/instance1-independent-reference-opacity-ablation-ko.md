# Instance_1 독립 `Ref -> SCn` DC-only 대 DC+opacity 비교

## 1. 질문

DC-only와 완전히 같은 reference 초기화와 state별 독립 학습 조건에서,
geometry는 고정하고 opacity만 추가로 학습하면 change segmentation이 개선되는지
확인했다.

```text
Ref -> scene_change1: DC-only / DC+opacity
Ref -> scene_change2: DC-only / DC+opacity
Ref -> scene_change3: DC-only / DC+opacity
```

## 2. 통제 조건

각 scene마다 동일한 immutable reference PLY에서 새 one-state model과 Adam
optimizer를 만들었다. 다른 scene의 parameter나 optimizer state는 상속하지 않았다.

| 항목 | 설정 |
|---|---|
| scene frame 수 | SC1 95, SC2 104, SC3 105 |
| initialization | `state_change_dc = reference DC`, `opacity_delta = 0` |
| 학습 parameter | state별 DC + opacity delta |
| 고정 parameter | xyz, scale, rotation, topology |
| DC LR | 0.0025 |
| opacity LR | 0.025 |
| updates | 이미지당 정확히 120회 |
| pose/cue | O-SCD fixed pose, O-SCD pixel + SAM2.1 cue |
| support cue gate | cue map `> 0.5` |
| Gaussian support gate | 한 view 이상에서 support되면 valid (`count >= 1`) |
| binary 평가 | rendered score `> 0.5` |
| GT | 학습 종료 후 평가에서만 사용 |

DC-only도 alpha blending을 제거한 모델은 아니다. Reference Gaussian의 기존
opacity로 동일하게 alpha blending하고, 그 opacity를 고정한다. DC+opacity는
`opacity_delta = 0`에서 시작하므로 첫 render가 DC-only와 같고, 학습 중에만
`sigmoid(reference opacity logit + state opacity delta)`가 달라진다.

세 scene 모두 DC-only와 DC+opacity의 base PLY hash, support-count checksum,
valid Gaussian 수가 일치했다. DC+opacity checkpoint의 xyz/scale/rotation delta
최대 절댓값도 모두 정확히 0이었다.

## 3. 결과

### 3.1 Mean-frame mIoU/F1

| Scene | DC-only mIoU | DC+opacity mIoU | 변화 | DC-only F1 | DC+opacity F1 | 변화 |
|---|---:|---:|---:|---:|---:|---:|
| SC1 | 0.6137 | 0.6083 | -0.0054 | 0.7071 | 0.7024 | -0.0047 |
| SC2 | 0.6464 | 0.6382 | -0.0082 | 0.7761 | 0.7685 | -0.0076 |
| SC3 | 0.6121 | 0.6096 | -0.0025 | 0.7513 | 0.7485 | -0.0028 |
| 전체 304-frame 평균 | **0.6243** | 0.6190 | -0.0054 | **0.7460** | 0.7410 | -0.0050 |

Opacity를 추가하면 세 scene 모두 mIoU와 F1이 소폭 하락했다. 따라서 이
설정에서는 DC-only가 더 좋은 기본값이다.

### 3.2 Precision/recall과 FP/FN

| Scope | Precision 변화 | Recall 변화 | FP 변화 | FN 변화 |
|---|---:|---:|---:|---:|
| SC1 | -0.0062 | +0.0032 | +22,583 | -6,858 |
| SC2 | -0.0073 | -0.0021 | +45,116 | +6,586 |
| SC3 | -0.0107 | +0.0165 | +81,807 | -52,773 |
| 전체 | -0.0082 | +0.0062 | +149,506 | -53,045 |

전체적으로 recall은 조금 증가했지만 precision 하락과 FP 증가가 더 커서 mIoU와
F1이 떨어졌다. 특히 SC2는 FP와 FN이 모두 증가했다.

### 3.3 Cue-fitting loss

| Scene | DC-only post SSF | DC+opacity post SSF | 변화 |
|---|---:|---:|---:|
| SC1 | 0.364631 | 0.364482 | -0.000149 |
| SC2 | 0.400089 | 0.399774 | -0.000315 |
| SC3 | 0.384583 | 0.384253 | -0.000330 |
| frame-weighted 전체 | 0.383652 | 0.383384 | -0.000268 |

Opacity freedom은 모든 scene에서 cue loss를 아주 조금 낮췄지만 GT segmentation은
나빠졌다. Geometry 실험과 마찬가지로 현재 cue objective에 더 잘 맞는 것이 GT
mask generalization 개선을 보장하지 않는다.

## 4. Opacity가 포화된 양상

Valid Gaussian만 집계하면 학습 후 opacity는 강하게 양극화됐다.

| Scene | 초기 opacity 평균 | 최종 평균 | 최종 `< 0.01` | 최종 `> 0.99` |
|---|---:|---:|---:|---:|
| SC1 | 0.663 | 0.346 | 64.1% | 33.8% |
| SC2 | 0.654 | 0.351 | 63.7% | 34.2% |
| SC3 | 0.660 | 0.358 | 62.9% | 35.0% |

즉 opacity는 작은 보정값으로 머무르지 않고 다수 Gaussian을 거의 끄고 일부를
거의 불투명하게 만드는 visibility selector처럼 동작했다. 이 자유도가 noisy한
2D cue에 맞춰 occlusion과 projected support를 재배치하면서 SSF loss는 낮췄지만,
GT 기준에서는 false positive가 늘었다. 현재 결과만으로 원인을 단정할 수는
없지만, 관찰된 포화와 precision/recall 변화는 이 설명과 일치한다.

## 5. 결론

> 동일 초기화, 동일 support gate, 독립 state, 고정 geometry 조건에서도
> opacity를 자유롭게 학습하면 DC-only보다 overall mIoU가 0.0054, F1이 0.0050
> 낮아졌다.

따라서 현 checkpoint에서는 DC-only를 기본값으로 유지하는 것이 타당하다.
Opacity를 다시 사용할 경우에는 현 LR 0.025를 그대로 여는 것보다 opacity
delta regularization, 범위 제한, 더 낮은 LR을 각각 분리해서 검증해야 한다.

## 6. 검증 및 산출물

- DC+opacity exact updates: SC1 11,400 / SC2 12,480 / SC3 12,600
- 모든 frame update count: 정확히 120
- gradient isolation audit: 세 run 모두 통과, violation 0
- frozen xyz/scale/rotation delta max abs: 세 run 모두 0
- 회귀 테스트: 23 passed

```text
outputs/instance1_independent_ref_scene_dc_geometry_oscd_cues_allframes_120/
├── summary.json
├── condition_metrics.csv
├── overall_condition_metrics.csv
├── condition_deltas.csv
├── scene_change1/{dc_only,dc_opacity,dc_geometry}/
├── scene_change2/{dc_only,dc_opacity,dc_geometry}/
└── scene_change3/{dc_only,dc_opacity,dc_geometry}/
```

재현 명령은 다음과 같다.

```bash
PYTHONPATH=. conda run --no-capture-output -n oscd python \
  experiments/run_independent_ref_geometry_ablation.py
```
