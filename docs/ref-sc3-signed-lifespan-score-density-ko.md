# ref → SC3 signed lifespan score densification 1차 실험

## 1. 질문

기존 화면공간 xyz gradient 대신, 현재 OPEN Gaussian마다 다음 signed score를 만들고
NEW 쪽만 densify하면 sparse XFeat seed의 gradient starvation을 우회할 수 있는지
확인했다.

```text
sign_supportᵢ = (2 p₊→NEW − 1) (mass₊,ᵢ − mass₋,ᵢ)
ageᵢ          = t − current_lifespan_startᵢ + 1
scoreᵢ        = activeᵢ sign_supportᵢ / ageᵢ
```

- `score > 0`: 작은 Gaussian은 clone, 큰 Gaussian은 split
- `score < 0`: densification 억제
- optional hard prune: `generation > 0`인 OPEN descendant에만 허용
- generation-zero row는 negative score로 삭제하지 않음

`p₊→NEW`는 기존 causal SAM-diff PC1/sign-posterior artifact의 값을 사용했다. 매
신규 frame에서 SAM delta와 plus/minus mask는 다시 계산하고, 현재 mutable topology에
pre-optimization alpha-transmittance mass를 한 번 투영했다.

## 2. Causal contract

기존 lifespan detector 경로는 바꾸지 않았다.

1. 신규 frame의 raw O-SCD cue alpha-T evidence를 optimization 전에 detector가 한 번
   소비한다.
2. signed SAM evidence는 representation density control에만 사용한다.
3. signed score, learned DC, post-optimization render는 detector posterior로 들어가지
   않는다.
4. object GT mask는 run 종료 후 별도 analyzer에서만 읽는다.
5. saved sign artifact는 현재 frame과 같은 causal row만 사용하며, smoke run에서는
   exact prefix만 잘라 쓴다.

전체 run에서 future-view access, inactive-gradient violation, causal-loop GT access는
모두 0이었고 topology integrity audit를 통과했다.

## 3. 대상 object

| Object | GT state | nonzero frames | 주요 구간 | 최대 면적 |
|---|---|---:|---|---:|
| `inference_base__object_004` | NEW | 45/105 | 040–058, 061–086 | 362,329 px |
| `reference_render_base__object_010` | REMOVED | 16/105 | 044–058, 071 | 473,103 px |

Object 004는 texture가 약한 큰 녹색 원통형 NEW object이고, object 010은 reference에만
있는 병 모양 REMOVED object다.

## 4. Matched 설정

- Stream: independent `ref → scene_change3`, 105 frames
- Seed: 0
- Representation update: 120/frame
- Density event: local update 4
- Detector: raw `single_candidate_beta`, BF30
- Renderer: OPEN learned DC + frozen black NEVER_OPEN occluder
- Optimizer: 모든 OPEN row의 DC/geometry/opacity
- Loss: local support + previous positive-growth replay, weight 7.5
- Opacity/size prune: off
- Score threshold: 0
- Score budget: 최대 2,048 source/frame
- Negative hard prune: primary comparison에서는 off

비교한 유일한 density selection 차이는 다음이다.

1. `active_oscd`: 기존 screen-space gradient clone/split
2. `active_signed_lifespan_score`: signed SAM alpha-T score + inverse lifespan age

## 5. 전체 결과

| Metric | Gradient u120 | Signed score u120 | Score − gradient |
|---|---:|---:|---:|
| mean-frame mIoU | **0.5901** | 0.5822 | -0.0079 |
| mean-frame F1 | **0.7291** | 0.7232 | -0.0059 |
| Precision | **0.6152** | 0.6098 | -0.0054 |
| Recall | **0.9848** | 0.9804 | -0.0044 |
| Final Gaussian | 1,283,651 | 1,494,445 | +210,794 |
| Density source | 150 | 210,944 | +210,794 |
| Runtime | 364.6 s | 595.6 s | +63.4% |
| Peak CUDA allocated | 7.80 GB | 9.83 GB | +26.0% |

Score run의 210,944 source는 처음 두 frame 이후 103개 frame에서 매번 2,048 budget을
전부 소진한 값이다. 선택된 positive event의 88.9%가 이미 densified descendant였다.
즉 부호 분리는 되었지만 recursive topology growth가 과했다.

## 6. Target-object 동작

Event 통계는 density source Gaussian의 projected center가 object mask 안에 있는지를
posthoc으로 센 값이다. Splat 전체 footprint overlap이 아니라 center-based diagnostic이다.

| Target | 기대 동작 | Correct signed events | 반대 부호 events | 해석 |
|---|---|---:|---:|---|
| NEW object 004 | positive clone/split | 28,783 / 29,901, **96.26%** | 1,118, 3.74% | NEW 위치에 growth가 집중됨 |
| REMOVED object 010 | negative suppress | 10,157 / 10,523, **96.52%** | 366, 3.48% | REMOVE 위치의 growth를 대부분 차단함 |

Thresholded output이 각 object GT support 안을 덮은 pixel fraction은 다음과 같다. 이는
object-only recall 성격의 값이며 object precision이나 object IoU가 아니다.

| Target | Gradient | Signed score | 차이 |
|---|---:|---:|---:|
| NEW object 004 support coverage | 0.9894 | 0.9882 | -0.0012 |
| REMOVED object 010 support coverage | 0.99964 | 0.99967 | +0.00003 |

따라서 **NEW/REMOVE에 맞춰 topology authority를 라우팅하는 것 자체는 성공**했지만,
이번 공격적인 fixed top-2,048 정책은 target coverage나 전체 mask metric을 높이지
못했다.

Frame 52 overlay에서 NEW 004의 녹색 GT 영역에는 positive clone/split source가,
REMOVE 010의 붉은 GT 영역에는 negative suppression source가 집중되는 것을 확인했다.

```text
outputs/ref_sc3_signed_lifespan_score_u120_f60_20260902_a/
  object_004_010_frame52_score_overlay.png
```

## 7. 결론

이번 결과는 두 부분을 분리해서 해석해야 한다.

1. **Signed score의 공간적 방향성은 유효하다.** 어려운 NEW object 004와 REMOVED
   object 010에서 각각 약 96%의 올바른 growth/suppression routing을 보였다.
2. **현재 density allocation은 유효하지 않다.** raw alpha-T mass에 threshold 0과
   fixed top-2,048를 적용하자 매 frame budget이 포화됐고, Gaussian이 210k 증가하면서
   mIoU/F1이 gradient baseline보다 0.0079/0.0059 낮아졌다.

따라서 다음 실험은 sign score를 버리는 것이 아니라 growth rate를 분리해서 보정해야
한다.

권장 순서:

1. `max_sources = 64/128/256` budget sweep
2. `(mass₊ − mass₋) / (mass₊ + mass₋ + ε)` 형태의 view-scale normalization
3. root 또는 lifespan별 cumulative child budget과 generation cap
4. positive growth calibration이 끝난 뒤에만 negative child-prune를 별도 ablation

Negative hard prune는 이미 optional로 구현했지만, generation-zero detector scaffold를
보호하기 위해 primary run에서는 사용하지 않았다.

## 8. Artifacts

```text
outputs/ref_sc3_gradient_active_oscd_u120_full_20260902_a/
  summary.json
  density_source_events.csv
  object_004_010_analysis.json
  visualization/thresholded_render/

outputs/ref_sc3_signed_lifespan_score_u120_full_20260902_a/
  summary.json
  density_events.csv
  density_source_events.csv
  object_004_010_analysis.json
  object_004_010_per_frame.csv
  visualization/thresholded_render/

outputs/ref_sc3_signed_lifespan_score_vs_gradient_u120_20260902/
  comparison.json
  comparison.csv
```
